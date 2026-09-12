"""Account-level circuit breakers: one account, five bots, one halt.

Chapter 26 (``04_circuit_breakers.py``) builds its breakers around a portfolio value. On an
MT5 CFD account that is not the whole story: several bots share one login, one equity, one
margin pool and one stop-out level, so the account is a risk object none of them owns. This
module reads that object through the shared adapter (``bots/_shared/mt5_broker.MT5Broker``:
``account_summary()`` for ``account_info`` and ``all_positions()`` for the positions of
*every* magic) and runs five breakers on it, all built on the shared
``CLOSED -> OPEN -> HALF_OPEN`` state machine of ``bots/_shared/monitor/base.py``:

========================== ==================================================================
``account_drawdown``       equity below the persisted high-water mark by more than the
                           declared fraction (``DrawdownBreaker``, 26/04:251-281, with the
                           peak persisted so it survives a restart)
``account_daily_loss``     equity below the day's opening equity by more than the declared
                           fraction (``DailyLossBreaker``, 26/04:290-329)
``account_margin``         ``margin_level = equity / margin x 100`` below the declared minimum
``account_stop_out``       margin level within the declared multiple of the broker's own
                           stop-out level (``account_info().margin_so_so``); the breaker
                           ``mt5-exness-broker.md`` section 4 asks for, tripping before the
                           broker does
``account_exposure``       too many open positions or pending orders, or too much gross
                           notional, counted across every magic on the account
========================== ==================================================================

The thresholds are the **strictest** any bot on the account declares - see the header of
``account_limits.yaml`` for why and for the table they were taken from. A per-strategy
breaker never lives here and an account breaker is never re-implemented inside a bot.

When any of them trips, :class:`AccountGuard` writes a **halt file**. Every bot's deployment
loop calls :meth:`AccountGuard.is_halted` before it stages anything, so one trip stops all of
them, and only a person clears it (:meth:`AccountGuard.clear_halt` takes a reason and a name).
The file is JSON under ``ML4T_DATA_PATH/mt5/`` because that is the one directory all the bots
already share.

Every threshold comes from ``bots/_shared/monitor/account_limits.yaml``. This module reads the
account; it does not decide what is safe.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from bots._shared.monitor.base import BreakerManager, BreakerState, BreakerTier, CircuitBreaker

logger = logging.getLogger(__name__)

LIMITS_PATH = Path(__file__).with_name("account_limits.yaml")


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AccountLimits:
    """The YAML, parsed. Every field is declared in ``account_limits.yaml``."""

    expected_login: int | None
    currency: str
    reference_equity: float
    max_drawdown_pct: float
    max_daily_equity_loss_pct: float
    min_equity: float
    min_margin_level_pct: float
    stop_out_buffer_multiple: float
    fallback_stop_out_pct: float
    max_open_positions: int
    max_open_orders: int
    max_gross_notional: float
    halt_path: str
    high_water_mark_path: str
    recovery_timeout_hours: float
    source: str = str(LIMITS_PATH)

    @property
    def recovery_timeout(self) -> timedelta:
        return timedelta(hours=self.recovery_timeout_hours)


def load_account_limits(path: Path | str | None = None) -> AccountLimits:
    """Parse ``account_limits.yaml`` (or a test's copy of it)."""
    path = Path(path) if path is not None else LIMITS_PATH
    raw = yaml.safe_load(path.read_text())
    account, equity = raw["account"], raw["equity"]
    margin, exposure, halt = raw["margin"], raw["exposure"], raw["halt"]
    return AccountLimits(
        expected_login=account.get("expected_login"),
        currency=str(account["currency"]),
        reference_equity=float(account["reference_equity"]),
        max_drawdown_pct=float(equity["max_drawdown_pct"]),
        max_daily_equity_loss_pct=float(equity["max_daily_equity_loss_pct"]),
        min_equity=float(equity["min_equity"]),
        min_margin_level_pct=float(margin["min_margin_level_pct"]),
        stop_out_buffer_multiple=float(margin["stop_out_buffer_multiple"]),
        fallback_stop_out_pct=float(margin["fallback_stop_out_pct"]),
        max_open_positions=int(exposure["max_open_positions"]),
        max_open_orders=int(exposure["max_open_orders"]),
        max_gross_notional=float(exposure["max_gross_notional"]),
        halt_path=str(halt["path"]),
        high_water_mark_path=str(halt["high_water_mark_path"]),
        recovery_timeout_hours=float(halt["recovery_timeout_hours"]),
        source=str(path),
    )


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------
@dataclass
class AccountSnapshot:
    """What the terminal says about the account right now, for every magic."""

    read_at: datetime
    login: int | None
    server: str | None
    trade_mode: int | None
    currency: str | None
    balance: float | None
    equity: float | None
    margin: float | None
    margin_free: float | None
    margin_so_call: float | None
    margin_so_so: float | None
    n_positions: int
    n_pending_orders: int
    gross_notional: float
    positions_by_magic: dict[str, int] = field(default_factory=dict)

    @property
    def margin_level_pct(self) -> float | None:
        """``equity / margin x 100``; ``None`` when nothing is open (margin 0)."""
        if self.equity is None or not self.margin:
            return None
        return 100.0 * float(self.equity) / float(self.margin)

    def as_dict(self) -> dict[str, Any]:
        return {
            "read_at": self.read_at.isoformat(),
            "login": self.login,
            "server": self.server,
            "trade_mode": self.trade_mode,
            "currency": self.currency,
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "margin_free": self.margin_free,
            "margin_level_pct": self.margin_level_pct,
            "margin_so_call": self.margin_so_call,
            "margin_so_so": self.margin_so_so,
            "n_positions": self.n_positions,
            "n_pending_orders": self.n_pending_orders,
            "gross_notional": self.gross_notional,
            "positions_by_magic": dict(self.positions_by_magic),
        }


def read_account_snapshot(broker: Any) -> AccountSnapshot:
    """Read the account through the shared adapter. Read-only: no order is ever sent.

    ``broker`` is a :class:`bots._shared.mt5_broker.MT5Broker` (or anything exposing
    ``account_summary()``, ``all_positions()`` and ``mt5``). The gross notional is summed
    from ``price_current x volume x trade_contract_size`` per position, which is the notional
    a CFD position carries; a symbol whose ``symbol_info`` the terminal does not return is
    counted at ``volume x price_current`` so no leg is silently dropped.
    """
    summary = broker.account_summary() or {}
    mt5 = broker.mt5
    account = mt5.account_info()
    positions = broker.all_positions()
    orders = list(mt5.orders_get() or ())

    by_magic: dict[str, int] = {}
    gross = 0.0
    for position in positions:
        key = str(position.get("magic"))
        by_magic[key] = by_magic.get(key, 0) + 1
        info = mt5.symbol_info(position.get("market_watch") or position.get("symbol"))
        size = float(getattr(info, "trade_contract_size", 1.0)) if info is not None else 1.0
        gross += (
            abs(float(position.get("volume", 0.0)))
            * size
            * float(position.get("price_current", 0.0) or 0.0)
        )

    return AccountSnapshot(
        read_at=datetime.now(UTC),
        login=summary.get("login"),
        server=summary.get("server"),
        trade_mode=summary.get("trade_mode"),
        currency=summary.get("currency"),
        balance=_as_float(summary.get("balance")),
        equity=_as_float(summary.get("equity")),
        margin=_as_float(summary.get("margin")),
        margin_free=_as_float(summary.get("margin_free")),
        margin_so_call=_as_float(getattr(account, "margin_so_call", None)),
        margin_so_so=_as_float(getattr(account, "margin_so_so", None)),
        n_positions=len(positions),
        n_pending_orders=len(orders),
        gross_notional=gross,
        positions_by_magic=by_magic,
    )


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# The breakers
# ---------------------------------------------------------------------------
class AccountDrawdownBreaker(CircuitBreaker):
    """``DrawdownBreaker`` of 26/04, with the peak persisted and an absolute floor.

    The notebook's peak lives in the object and resets when the process does; on an account
    several bots restart against, that would silently forgive every drawdown. The peak is
    therefore read from and written to ``high_water_mark_path``.
    """

    def __init__(
        self,
        name: str,
        *,
        max_drawdown_pct: float,
        min_equity: float,
        hwm_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, **kwargs)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_equity = float(min_equity)
        self.hwm_path = Path(hwm_path) if hwm_path is not None else None
        self.peak_equity: float = self._load_peak()

    def _load_peak(self) -> float:
        if self.hwm_path is None or not self.hwm_path.exists():
            return 0.0
        try:
            return float(json.loads(self.hwm_path.read_text())["peak_equity"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            logger.warning("unreadable high-water mark at %s; starting from 0", self.hwm_path)
            return 0.0

    def _save_peak(self) -> None:
        if self.hwm_path is None:
            return
        self.hwm_path.parent.mkdir(parents=True, exist_ok=True)
        self.hwm_path.write_text(
            json.dumps(
                {"peak_equity": self.peak_equity, "updated_at": datetime.now(UTC).isoformat()},
                indent=2,
            )
        )

    def check_condition(self, snapshot: AccountSnapshot | None = None, **_: Any):
        if snapshot is None or snapshot.equity is None:
            return False, "", None, None
        equity = float(snapshot.equity)
        if equity < self.min_equity:
            return (
                True,
                f"equity {equity:,.2f} below the absolute floor {self.min_equity:,.2f}",
                equity,
                self.min_equity,
            )
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._save_peak()
        if self.peak_equity <= 0:
            return False, "", None, self.max_drawdown_pct
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown >= self.max_drawdown_pct:
            return (
                True,
                f"account drawdown {drawdown:.2%} from peak {self.peak_equity:,.2f} "
                f"exceeds {self.max_drawdown_pct:.0%}",
                drawdown,
                self.max_drawdown_pct,
            )
        return False, "", drawdown, self.max_drawdown_pct


class DailyEquityLossBreaker(CircuitBreaker):
    """``DailyLossBreaker`` of 26/04 on account equity.

    ``reset_day(equity)`` is called once per trading day by the loop; before the first call
    the day's opening equity is the first reading seen, exactly as the notebook does.
    """

    def __init__(self, name: str, *, max_daily_loss_pct: float, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, **kwargs)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.start_of_day_equity: float | None = None

    def reset_day(self, equity: float) -> None:
        self.start_of_day_equity = float(equity)

    def check_condition(self, snapshot: AccountSnapshot | None = None, **_: Any):
        if snapshot is None or snapshot.equity is None:
            return False, "", None, None
        equity = float(snapshot.equity)
        if self.start_of_day_equity is None:
            self.start_of_day_equity = equity
        opening = self.start_of_day_equity
        if opening <= 0:
            return False, "", None, self.max_daily_loss_pct
        loss = (opening - equity) / opening
        if loss >= self.max_daily_loss_pct:
            return (
                True,
                f"account down {loss:.2%} on the day (open {opening:,.2f}) "
                f"exceeds {self.max_daily_loss_pct:.0%}",
                loss,
                self.max_daily_loss_pct,
            )
        return False, "", loss, self.max_daily_loss_pct


class MarginLevelBreaker(CircuitBreaker):
    """``margin_level`` below the declared minimum. No open position means no level."""

    def __init__(self, name: str, *, min_margin_level_pct: float, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, **kwargs)
        self.min_margin_level_pct = float(min_margin_level_pct)

    def check_condition(self, snapshot: AccountSnapshot | None = None, **_: Any):
        level = None if snapshot is None else snapshot.margin_level_pct
        if level is None:
            return False, "", None, self.min_margin_level_pct
        if level < self.min_margin_level_pct:
            return (
                True,
                f"margin level {level:.0f}% below minimum {self.min_margin_level_pct:.0f}%",
                level,
                self.min_margin_level_pct,
            )
        return False, "", level, self.min_margin_level_pct


class StopOutDistanceBreaker(CircuitBreaker):
    """Trips while there is still room: at ``buffer x`` the broker's own stop-out level."""

    def __init__(
        self, name: str, *, buffer_multiple: float, fallback_stop_out_pct: float, **kwargs: Any
    ) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, **kwargs)
        self.buffer_multiple = float(buffer_multiple)
        self.fallback_stop_out_pct = float(fallback_stop_out_pct)

    def check_condition(self, snapshot: AccountSnapshot | None = None, **_: Any):
        if snapshot is None:
            return False, "", None, None
        level = snapshot.margin_level_pct
        stop_out = snapshot.margin_so_so
        if stop_out is None or stop_out <= 0:
            stop_out = self.fallback_stop_out_pct
        threshold = self.buffer_multiple * float(stop_out)
        if level is None:
            return False, "", None, threshold
        if level < threshold:
            return (
                True,
                f"margin level {level:.0f}% within {self.buffer_multiple:g}x the "
                f"{stop_out:.0f}% stop-out level",
                level,
                threshold,
            )
        return False, "", level, threshold


class OpenExposureBreaker(CircuitBreaker):
    """Too many open positions or orders, or too much gross notional - across every magic."""

    def __init__(
        self,
        name: str,
        *,
        max_open_positions: int,
        max_open_orders: int,
        max_gross_notional: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, **kwargs)
        self.max_open_positions = int(max_open_positions)
        self.max_open_orders = int(max_open_orders)
        self.max_gross_notional = float(max_gross_notional)

    def check_condition(self, snapshot: AccountSnapshot | None = None, **_: Any):
        if snapshot is None:
            return False, "", None, None
        if snapshot.n_positions > self.max_open_positions:
            return (
                True,
                f"{snapshot.n_positions} open positions on the account "
                f"(magics {sorted(snapshot.positions_by_magic)}) above {self.max_open_positions}",
                float(snapshot.n_positions),
                float(self.max_open_positions),
            )
        if snapshot.n_pending_orders > self.max_open_orders:
            return (
                True,
                f"{snapshot.n_pending_orders} pending orders above {self.max_open_orders}",
                float(snapshot.n_pending_orders),
                float(self.max_open_orders),
            )
        if snapshot.gross_notional > self.max_gross_notional:
            return (
                True,
                f"gross notional {snapshot.gross_notional:,.0f} above {self.max_gross_notional:,.0f}",
                snapshot.gross_notional,
                self.max_gross_notional,
            )
        return False, "", float(snapshot.n_positions), float(self.max_open_positions)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
@dataclass
class AccountBreakerSet:
    """The five account breakers, wired into one :class:`BreakerManager`."""

    manager: BreakerManager
    drawdown: AccountDrawdownBreaker
    daily_loss: DailyEquityLossBreaker
    margin: MarginLevelBreaker
    stop_out: StopOutDistanceBreaker
    exposure: OpenExposureBreaker


class AccountGuard:
    """Runs the account breakers and owns the halt file every bot reads.

    Args:
        limits: Parsed ``account_limits.yaml``.
        halt_file: Override the path (tests). Otherwise ``limits.halt_path`` resolved under
            ``ML4T_DATA_PATH/mt5/`` when relative.
        bot_id: Written into the halt record so the operator sees which bot's cycle noticed.
    """

    def __init__(
        self,
        limits: AccountLimits | None = None,
        *,
        halt_file: Path | str | None = None,
        hwm_file: Path | str | None = None,
        bot_id: str = "unknown",
    ) -> None:
        self.limits = limits or load_account_limits()
        self.bot_id = bot_id
        self.halt_file = (
            Path(halt_file) if halt_file is not None else self._resolve(self.limits.halt_path)
        )
        self.hwm_file = (
            Path(hwm_file)
            if hwm_file is not None
            else (
                self.halt_file.with_name(Path(self.limits.high_water_mark_path).name)
                if halt_file is not None
                else self._resolve(self.limits.high_water_mark_path)
            )
        )
        self.breakers = self._build()
        self.manager = self.breakers.manager

    def _resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            return candidate
        root = os.environ.get("ML4T_DATA_PATH")
        base = Path(root) / "mt5" if root else Path(__file__).resolve().parents[3] / "data" / "mt5"
        return base / candidate

    def _build(self) -> AccountBreakerSet:
        limits, timeout = self.limits, self.limits.recovery_timeout
        manager = BreakerManager()
        drawdown = AccountDrawdownBreaker(
            "account_drawdown",
            max_drawdown_pct=limits.max_drawdown_pct,
            min_equity=limits.min_equity,
            hwm_path=self.hwm_file,
            recovery_timeout=timeout,
        )
        daily_loss = DailyEquityLossBreaker(
            "account_daily_loss",
            max_daily_loss_pct=limits.max_daily_equity_loss_pct,
            recovery_timeout=timeout,
        )
        margin = MarginLevelBreaker(
            "account_margin",
            min_margin_level_pct=limits.min_margin_level_pct,
            recovery_timeout=timeout,
        )
        stop_out = StopOutDistanceBreaker(
            "account_stop_out",
            buffer_multiple=limits.stop_out_buffer_multiple,
            fallback_stop_out_pct=limits.fallback_stop_out_pct,
            recovery_timeout=timeout,
        )
        exposure = OpenExposureBreaker(
            "account_exposure",
            max_open_positions=limits.max_open_positions,
            max_open_orders=limits.max_open_orders,
            max_gross_notional=limits.max_gross_notional,
            recovery_timeout=timeout,
        )
        for breaker in (drawdown, daily_loss, margin, stop_out, exposure):
            manager.add_breaker(breaker)
        return AccountBreakerSet(manager, drawdown, daily_loss, margin, stop_out, exposure)

    # -- account identity -------------------------------------------------------------
    def verify_account(self, snapshot: AccountSnapshot) -> None:
        """Raise when the login is not the one the limits were written for."""
        expected = self.limits.expected_login
        if (
            expected is not None
            and snapshot.login is not None
            and int(snapshot.login) != int(expected)
        ):
            raise ValueError(
                f"account_limits.yaml declares login {expected}, terminal reports "
                f"{snapshot.login}; the account-level limits do not describe this account"
            )

    # -- the check --------------------------------------------------------------------
    def start_day(self, snapshot: AccountSnapshot) -> None:
        """Pin the day's opening equity for the daily-loss breaker."""
        if snapshot.equity is not None:
            self.breakers.daily_loss.reset_day(float(snapshot.equity))

    def check(self, snapshot: AccountSnapshot, *, event_time: datetime | None = None) -> bool:
        """Update the breakers; write the halt file on a trip. Returns "trading allowed"."""
        self.verify_account(snapshot)
        allowed = self.manager.check_all(event_time=event_time, snapshot=snapshot)
        if not allowed:
            reason = "; ".join(
                e.reason
                for e in self.manager.event_log
                if e.new_state is BreakerState.OPEN and e.reason
            )
            self.write_halt(reason or "account breaker open", snapshot=snapshot)
        return allowed

    # -- the halt file ----------------------------------------------------------------
    def is_halted(self) -> bool:
        return self.halt_file.exists()

    def halt_record(self) -> dict[str, Any] | None:
        if not self.halt_file.exists():
            return None
        try:
            return json.loads(self.halt_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:  # a corrupt halt file still halts
            logger.error(
                "halt file %s unreadable (%s); treating the account as halted", self.halt_file, exc
            )
            return {"reason": "halt file unreadable", "error": repr(exc)}

    def write_halt(self, reason: str, snapshot: AccountSnapshot | None = None) -> Path:
        """Write the halt file. Every bot on this account stops on its next cycle."""
        self.halt_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "halted_at": datetime.now(UTC).isoformat(),
            "raised_by_bot": self.bot_id,
            "reason": reason,
            "open_breakers": self.manager.open_breakers(),
            "breakers": self.manager.get_status(),
            "limits_source": self.limits.source,
            "snapshot": snapshot.as_dict() if snapshot is not None else None,
        }
        self.halt_file.write_text(json.dumps(record, indent=2, default=str))
        logger.error("ACCOUNT HALT written to %s: %s", self.halt_file, reason)
        return self.halt_file

    def clear_halt(self, *, reason: str, cleared_by: str) -> None:
        """Manual clear only. ``reason`` and ``cleared_by`` are required and are archived."""
        if not reason or not cleared_by:
            raise ValueError("clearing an account halt needs a reason and a name")
        if self.halt_file.exists():
            record = self.halt_record() or {}
            record.update(
                {
                    "cleared_at": datetime.now(UTC).isoformat(),
                    "cleared_by": cleared_by,
                    "cleared_reason": reason,
                }
            )
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            self.halt_file.with_name(f"{self.halt_file.stem}.cleared-{stamp}.json").write_text(
                json.dumps(record, indent=2, default=str)
            )
            self.halt_file.unlink()
        self.manager.reset_all()
        logger.warning("account halt cleared by %s: %s", cleared_by, reason)

    def to_record(self) -> dict[str, Any]:
        return {
            "halt_file": str(self.halt_file),
            "halted": self.is_halted(),
            "limits_source": self.limits.source,
            "peak_equity": self.breakers.drawdown.peak_equity,
            **self.manager.to_records(),
        }


__all__ = [
    "AccountBreakerSet",
    "AccountDrawdownBreaker",
    "AccountGuard",
    "AccountLimits",
    "AccountSnapshot",
    "DailyEquityLossBreaker",
    "MarginLevelBreaker",
    "OpenExposureBreaker",
    "StopOutDistanceBreaker",
    "load_account_limits",
    "read_account_snapshot",
]
