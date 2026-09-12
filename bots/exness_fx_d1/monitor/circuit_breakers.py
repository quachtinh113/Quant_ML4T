"""Strategy-tier circuit breakers for exness_fx_d1 — ``26_mlops_governance/04``.

The state machine (``CLOSED -> OPEN -> HALF_OPEN``, ``BreakerEvent``, ``BreakerManager``) is
the shared one in ``bots/_shared/monitor/base.py``; this module only supplies the conditions.
The **account** tier is not here - it lives in ``bots/_shared/monitor`` and stops every bot on
the login at once. A bot never re-implements an account breaker.

Every breaker maps to a written rule of ``bots/exness_fx_d1/BOT.md``, and every threshold comes
from ``deploy/risk_config.yaml::breakers``:

===== ====================================== ===============================================
Rule  BOT.md kill criterion                  Implementation
===== ====================================== ===============================================
(a)   drawdown from peak > 8 % of allocated  :class:`DrawdownBreaker`, portfolio tier
      capital
 --   daily loss (derived, 2 % of allocated) :class:`DailyLossBreaker`, portfolio tier
(b)   rolling 63-session hit rate below 0.5  :class:`RollingHitRateBreaker`, strategy tier.
      for 21 consecutive sessions            A NEW subclass: ``ConsecutiveLossBreaker`` counts
                                             losing trades, which is a different rule, so the
                                             sliding window is written out here.
(c)   realised round-trip cost above 1.5x    :class:`RealisedCostBreaker`, strategy tier, fed
      the assumed 2.6 bp for a month         from the MT5 deal history through
                                             ``bots/_shared/costs_mt5.py``
(d)   reconciliation finds an unexplained    :class:`ReconciliationBreaker`, system tier.
      position                               Trips **immediately**, no HALF_OPEN probe, and
                                             latches the persistent kill switch
(e)   decision bar missing within tolerance  NOT a breaker. It is a no-trade day; the
                                             deployment loop logs it and only
                                             :class:`MissingDecisionBarBreaker` escalates,
                                             after ``escalate_after`` misses in
                                             ``window_sessions``
(f)   realised swap above the declared value :class:`SwapBreaker`, strategy tier, fed from the
      for a week                             rollover history against ``setup.yaml::costs.swap``
 --   five consecutive losing round trips    :class:`ConsecutiveLossBreaker`, trade tier
 --   order round-trip latency               :class:`LatencyBreaker`, system tier
===== ====================================== ===============================================

Retirement is not a breaker either: a breaker pauses and probes, a retire rule ends the bot.
See ``retire.py``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bots._shared.monitor.base import (
    BreakerEvent,
    BreakerManager,
    BreakerState,
    BreakerTier,
    CircuitBreaker,
    transition_breaker,
)

logger = logging.getLogger("exness_fx_d1.monitor")

BOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RISK_CONFIG = BOT_DIR / "deploy" / "risk_config.yaml"


def load_breaker_config(path: Path | str | None = None) -> dict[str, Any]:
    """The ``breakers`` and ``missing_decision_bar`` blocks of ``deploy/risk_config.yaml``."""
    raw = yaml.safe_load(Path(path or DEFAULT_RISK_CONFIG).read_text())
    return {
        "breakers": raw["breakers"],
        "missing_decision_bar": raw["missing_decision_bar"],
        "capital": raw["capital"],
        "source": str(path or DEFAULT_RISK_CONFIG),
    }


# ---------------------------------------------------------------------------------------
# (a) and the derived daily loss — portfolio tier
# ---------------------------------------------------------------------------------------
class DrawdownBreaker(CircuitBreaker):
    """(a) Drawdown from the running peak of this bot's own equity (26/04:251-281)."""

    def __init__(self, name: str, *, max_drawdown_pct: float, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.PORTFOLIO, kill_criterion="a", **kwargs)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.peak_value = 0.0

    def check_condition(self, portfolio_value: float | None = None, **_: Any):
        if portfolio_value is None:
            return False, "", None, self.max_drawdown_pct
        self.peak_value = max(self.peak_value, float(portfolio_value))
        if self.peak_value <= 0:
            return False, "", None, self.max_drawdown_pct
        drawdown = (self.peak_value - float(portfolio_value)) / self.peak_value
        if drawdown >= self.max_drawdown_pct:
            return (
                True,
                f"drawdown {drawdown:.2%} from peak {self.peak_value:,.2f} exceeds "
                f"{self.max_drawdown_pct:.0%} of allocated capital",
                drawdown,
                self.max_drawdown_pct,
            )
        return False, "", drawdown, self.max_drawdown_pct


class DailyLossBreaker(CircuitBreaker):
    """The derived daily-loss rule on this bot's own equity (26/04:290-329)."""

    def __init__(self, name: str, *, max_daily_loss_pct: float, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.PORTFOLIO, kill_criterion="derived:daily_loss", **kwargs)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.start_of_day_value: float | None = None

    def reset_day(self, current_value: float) -> None:
        self.start_of_day_value = float(current_value)

    def check_condition(self, portfolio_value: float | None = None, **_: Any):
        if portfolio_value is None:
            return False, "", None, self.max_daily_loss_pct
        if self.start_of_day_value is None:
            self.start_of_day_value = float(portfolio_value)
            return False, "", None, self.max_daily_loss_pct
        opening = self.start_of_day_value
        if opening <= 0:
            return False, "", None, self.max_daily_loss_pct
        loss = (opening - float(portfolio_value)) / opening
        if loss >= self.max_daily_loss_pct:
            return (
                True,
                f"down {loss:.2%} on the day (open {opening:,.2f}) exceeds "
                f"{self.max_daily_loss_pct:.0%}",
                loss,
                self.max_daily_loss_pct,
            )
        return False, "", loss, self.max_daily_loss_pct


# ---------------------------------------------------------------------------------------
# (b) — strategy tier. A new subclass: the rule is a sliding window, not a streak of trades.
# ---------------------------------------------------------------------------------------
class RollingHitRateBreaker(CircuitBreaker):
    """(b) The rolling ``window_sessions`` hit rate below ``min_hit_rate`` for
    ``consecutive_sessions`` sessions in a row.

    Why this is not ``ConsecutiveLossBreaker`` (26/04:338): that class counts consecutive
    *losing trades* and resets on the first win. This rule counts consecutive *sessions on
    which a 63-session rolling statistic sits below a level* - a run of 21 such sessions can
    contain winning trades throughout. BOT.md chose the hit rate over an IC deliberately: "an
    IC over five names is not a readable statistic".

    Feed it one session at a time with :meth:`record_session`, whose argument is that session's
    hit rate (the share of the book's legs whose sign was right).
    """

    def __init__(
        self,
        name: str,
        *,
        window_sessions: int,
        min_hit_rate: float,
        consecutive_sessions: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="b", **kwargs)
        self.window_sessions = int(window_sessions)
        self.min_hit_rate = float(min_hit_rate)
        self.consecutive_sessions = int(consecutive_sessions)
        self.window: deque[float] = deque(maxlen=self.window_sessions)
        self.sessions_below = 0
        self.last_rolling: float | None = None

    def record_session(self, hit_rate: float) -> float | None:
        """Append one session's hit rate and update the run length. Returns the rolling mean."""
        self.window.append(float(hit_rate))
        if len(self.window) < self.window_sessions:
            self.last_rolling = None
            return None  # the rolling statistic does not exist yet, so nothing is "below"
        self.last_rolling = float(np.mean(self.window))
        if self.last_rolling < self.min_hit_rate:
            self.sessions_below += 1
        else:
            self.sessions_below = 0
        return self.last_rolling

    def check_condition(self, **_: Any):
        if self.sessions_below >= self.consecutive_sessions:
            return (
                True,
                f"rolling {self.window_sessions}-session hit rate "
                f"{self.last_rolling:.3f} below {self.min_hit_rate:.2f} for "
                f"{self.sessions_below} consecutive sessions",
                float(self.sessions_below),
                float(self.consecutive_sessions),
            )
        return False, "", float(self.sessions_below), float(self.consecutive_sessions)


# ---------------------------------------------------------------------------------------
# (c) — strategy tier, fed from the deal history
# ---------------------------------------------------------------------------------------
@dataclass
class RoundTrip:
    """One closed round trip as the cost breaker needs it."""

    closed_at: datetime
    symbol: str
    notional: float
    cost_quote: float  # spread + commission + swap actually paid, in the quote currency

    @property
    def cost_bps(self) -> float:
        return 0.0 if self.notional <= 0 else 10_000.0 * self.cost_quote / self.notional


class RealisedCostBreaker(CircuitBreaker):
    """(c) Realised round-trip cost above ``multiple x assumed_round_trip_bps`` for a month.

    The realised figure is the notional-weighted mean cost of the round trips that closed
    inside ``window_days``, read from the MT5 deal history (``bots/_shared/costs_mt5.py`` turns
    deals and swaps into the cost of a holding period) - never from the backtest's ledger,
    which is what the cost was *assumed* to be. Below ``min_round_trips`` the breaker abstains:
    a monthly mean over three trades is not a monthly mean.
    """

    def __init__(
        self,
        name: str,
        *,
        assumed_round_trip_bps: float,
        multiple: float,
        window_days: int,
        min_round_trips: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="c", **kwargs)
        self.assumed_round_trip_bps = float(assumed_round_trip_bps)
        self.multiple = float(multiple)
        self.window = timedelta(days=int(window_days))
        self.min_round_trips = int(min_round_trips)
        self.round_trips: list[RoundTrip] = []

    @property
    def threshold_bps(self) -> float:
        return self.assumed_round_trip_bps * self.multiple

    def record_round_trip(self, trip: RoundTrip) -> None:
        self.round_trips.append(trip)

    def realised_bps(self, now: datetime) -> tuple[float | None, int]:
        recent = [t for t in self.round_trips if now - t.closed_at <= self.window]
        notional = sum(t.notional for t in recent)
        if len(recent) < self.min_round_trips or notional <= 0:
            return None, len(recent)
        return 10_000.0 * sum(t.cost_quote for t in recent) / notional, len(recent)

    def check_condition(self, now: datetime | None = None, **_: Any):
        realised, n = self.realised_bps(now or datetime.now(UTC))
        if realised is None:
            return False, "", None, self.threshold_bps
        if realised > self.threshold_bps:
            return (
                True,
                f"realised round-trip cost {realised:.2f} bps over {n} trips in "
                f"{self.window.days} days exceeds {self.multiple:g}x the assumed "
                f"{self.assumed_round_trip_bps:.2f} bps",
                realised,
                self.threshold_bps,
            )
        return False, "", realised, self.threshold_bps


# ---------------------------------------------------------------------------------------
# (f) — strategy tier, fed from the rollover history
# ---------------------------------------------------------------------------------------
class SwapBreaker(CircuitBreaker):
    """(f) Realised swap per night above the value ``setup.yaml::costs.swap`` declares.

    The declaration is 0.0 points per lot per night on every pair, read from ``symbol_info`` on
    the **demo** account; a real Pro account may charge. ``record_night`` takes the realised
    charge in points per lot for one rollover, and the breaker trips after
    ``consecutive_nights`` nights above ``declared + tolerance``.
    """

    def __init__(
        self,
        name: str,
        *,
        declared_points_per_lot_per_night: float,
        tolerance_points: float,
        consecutive_nights: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="f", **kwargs)
        self.declared = float(declared_points_per_lot_per_night)
        self.tolerance = float(tolerance_points)
        self.consecutive_nights = int(consecutive_nights)
        self.nights_above = 0
        self.last_points: float | None = None

    @property
    def threshold_points(self) -> float:
        return self.declared + self.tolerance

    def record_night(self, points_per_lot: float) -> None:
        """One rollover. Sign convention: a POSITIVE number is a cost paid."""
        self.last_points = float(points_per_lot)
        if self.last_points > self.threshold_points:
            self.nights_above += 1
        else:
            self.nights_above = 0

    def check_condition(self, **_: Any):
        if self.nights_above >= self.consecutive_nights:
            return (
                True,
                f"realised swap {self.last_points:.2f} points/lot/night above the declared "
                f"{self.declared:.2f} (+{self.tolerance:.2f} tolerance) for "
                f"{self.nights_above} nights",
                float(self.nights_above),
                float(self.consecutive_nights),
            )
        return False, "", float(self.nights_above), float(self.consecutive_nights)


# ---------------------------------------------------------------------------------------
# (d) — system tier, immediate and latching
# ---------------------------------------------------------------------------------------
class ReconciliationBreaker(CircuitBreaker):
    """(d) An unexplained position. Trips on the first occurrence and does not self-heal.

    ``recovery_timeout`` is set to a century by :func:`build_manager`: a reconciliation
    mismatch is not a market condition that passes, it is a state the operator has to explain.
    :meth:`arm_kill_switch` latches ``SafeBroker``'s persistent kill switch so a restart cannot
    resume trading (``25_live_trading/10:632-653``).
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        kwargs.setdefault("recovery_timeout", timedelta(days=36_500))
        super().__init__(name, tier=BreakerTier.SYSTEM, kill_criterion="d", **kwargs)
        self.last_report: dict[str, Any] | None = None

    def check_condition(self, reconciliation: dict[str, Any] | None = None, **_: Any):
        if reconciliation is None:
            return False, "", None, None
        self.last_report = reconciliation
        if reconciliation.get("clean", True):
            return False, "", 0.0, 1.0
        detail = {
            k: reconciliation.get(k)
            for k in (
                "missing_positions",
                "unexpected_positions",
                "quantity_mismatches",
                "unknown_magics",
                "error",
            )
            if reconciliation.get(k)
        }
        return True, f"reconciliation mismatch: {detail}", 1.0, 1.0

    def arm_kill_switch(self, safe_broker: Any) -> None:
        """Latch the persistent kill switch; a restart must not resume trading."""
        safe_broker.enable_kill_switch(f"{self.name}: {self.history[-1].reason if self.history else 'mismatch'}")


# ---------------------------------------------------------------------------------------
# (e) — NOT a breaker until it repeats
# ---------------------------------------------------------------------------------------
class MissingDecisionBarBreaker(CircuitBreaker):
    """(e) escalated. One missing decision bar is a no-trade day, not a pause.

    The deployment loop already refuses to trade on a session whose decision bar is missing or
    stale (the 2018-01-31 rule, BOT.md open question 9) and logs it to
    ``deploy/state/no_trade_days.json``. Pausing the bot for a day because the broker skipped
    one bar would be an over-reaction; ``escalate_after`` misses inside ``window_sessions`` is
    a broken feed, and that is what trips.
    """

    def __init__(
        self, name: str, *, escalate_after: int, window_sessions: int, **kwargs: Any
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="e", **kwargs)
        self.escalate_after = int(escalate_after)
        self.window_sessions = int(window_sessions)
        self.recent: deque[bool] = deque(maxlen=self.window_sessions)

    def record_session(self, *, decision_bar_ok: bool) -> None:
        self.recent.append(not decision_bar_ok)

    @property
    def misses_in_window(self) -> int:
        return sum(self.recent)

    def check_condition(self, **_: Any):
        misses = self.misses_in_window
        if misses >= self.escalate_after:
            return (
                True,
                f"{misses} sessions without a usable decision bar in the last "
                f"{len(self.recent)} (escalate_after {self.escalate_after})",
                float(misses),
                float(self.escalate_after),
            )
        return False, "", float(misses), float(self.escalate_after)


# ---------------------------------------------------------------------------------------
# Trade tier and infrastructure
# ---------------------------------------------------------------------------------------
class ConsecutiveLossBreaker(CircuitBreaker):
    """N consecutive losing round trips (26/04:338-371). A declared pause, not a criterion."""

    def __init__(self, name: str, *, max_consecutive: int, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.TRADE, kill_criterion="derived:consecutive_loss", **kwargs)
        self.max_consecutive = int(max_consecutive)
        self.consecutive_losses = 0

    def record_trade(self, pnl: float) -> None:
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0

    def check_condition(self, **_: Any):
        if self.consecutive_losses >= self.max_consecutive:
            return (
                True,
                f"{self.consecutive_losses} consecutive losing round trips",
                float(self.consecutive_losses),
                float(self.max_consecutive),
            )
        return False, "", float(self.consecutive_losses), float(self.max_consecutive)


class LatencyBreaker(CircuitBreaker):
    """Mean order round-trip latency over the last ``n_recent`` sends (26/04:381-418)."""

    def __init__(self, name: str, *, max_latency_ms: float, n_recent: int, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, kill_criterion=None, **kwargs)
        self.max_latency_ms = float(max_latency_ms)
        self.n_recent = int(n_recent)
        self.latency_history: list[float] = []

    def record_latency(self, latency_ms: float) -> None:
        self.latency_history.append(float(latency_ms))
        self.latency_history = self.latency_history[-100:]

    def check_condition(self, **_: Any):
        if not self.latency_history:
            return False, "", None, self.max_latency_ms
        mean = float(np.mean(self.latency_history[-self.n_recent :]))
        if mean > self.max_latency_ms:
            return True, f"order latency {mean:.0f} ms exceeds {self.max_latency_ms:.0f} ms", mean, self.max_latency_ms
        return False, "", mean, self.max_latency_ms


# ---------------------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------------------
@dataclass
class BotBreakers:
    """Every strategy-tier breaker of this bot, plus the manager that answers "may I trade?"."""

    manager: BreakerManager
    drawdown: DrawdownBreaker
    daily_loss: DailyLossBreaker
    hit_rate: RollingHitRateBreaker
    cost: RealisedCostBreaker
    swap: SwapBreaker
    reconciliation: ReconciliationBreaker
    missing_bar: MissingDecisionBarBreaker
    consecutive_loss: ConsecutiveLossBreaker
    latency: LatencyBreaker
    config_source: str

    def allows_trading(self) -> bool:
        return self.manager.allows_trading()

    def to_record(self) -> dict[str, Any]:
        return {"config_source": self.config_source, **self.manager.to_records()}


def build_manager(
    config_path: Path | str | None = None, *, on_any_trip: Any = None
) -> BotBreakers:
    """Build every breaker from ``deploy/risk_config.yaml``. No threshold is written here."""
    config = load_breaker_config(config_path)
    block = config["breakers"]
    missing = config["missing_decision_bar"]
    timeout = timedelta(hours=float(block["recovery_timeout_hours"]))
    manager = BreakerManager(on_any_trip=on_any_trip)

    drawdown = DrawdownBreaker(
        "strategy_drawdown", max_drawdown_pct=float(block["drawdown"]["max_drawdown_pct"]), recovery_timeout=timeout
    )
    daily_loss = DailyLossBreaker(
        "strategy_daily_loss",
        max_daily_loss_pct=float(block["daily_loss"]["max_daily_loss_pct"]),
        recovery_timeout=timeout,
    )
    hit_rate = RollingHitRateBreaker(
        "strategy_hit_rate",
        window_sessions=int(block["hit_rate"]["window_sessions"]),
        min_hit_rate=float(block["hit_rate"]["min_hit_rate"]),
        consecutive_sessions=int(block["hit_rate"]["consecutive_sessions"]),
        recovery_timeout=timeout,
    )
    cost = RealisedCostBreaker(
        "strategy_realised_cost",
        assumed_round_trip_bps=float(block["cost"]["assumed_round_trip_bps"]),
        multiple=float(block["cost"]["multiple"]),
        window_days=int(block["cost"]["window_days"]),
        min_round_trips=int(block["cost"]["min_round_trips"]),
        recovery_timeout=timeout,
    )
    swap = SwapBreaker(
        "strategy_swap",
        declared_points_per_lot_per_night=float(block["swap"]["declared_points_per_lot_per_night"]),
        tolerance_points=float(block["swap"]["tolerance_points"]),
        consecutive_nights=int(block["swap"]["consecutive_nights"]),
        recovery_timeout=timeout,
    )
    reconciliation = ReconciliationBreaker("system_reconciliation")
    missing_bar = MissingDecisionBarBreaker(
        "strategy_missing_decision_bar",
        escalate_after=int(missing["escalate_after"]),
        window_sessions=int(missing["window_sessions"]),
        recovery_timeout=timeout,
    )
    consecutive = ConsecutiveLossBreaker(
        "trade_consecutive_loss",
        max_consecutive=int(block["consecutive_loss"]["max_consecutive"]),
        recovery_timeout=timeout,
    )
    latency = LatencyBreaker(
        "system_latency",
        max_latency_ms=float(block["latency"]["max_latency_ms"]),
        n_recent=int(block["latency"]["n_recent"]),
        recovery_timeout=timeout,
    )
    for breaker in (
        drawdown,
        daily_loss,
        hit_rate,
        cost,
        swap,
        reconciliation,
        missing_bar,
        consecutive,
        latency,
    ):
        manager.add_breaker(breaker)
    return BotBreakers(
        manager=manager,
        drawdown=drawdown,
        daily_loss=daily_loss,
        hit_rate=hit_rate,
        cost=cost,
        swap=swap,
        reconciliation=reconciliation,
        missing_bar=missing_bar,
        consecutive_loss=consecutive,
        latency=latency,
        config_source=config["source"],
    )


def kill_switch_on_trip(safe_broker: Any):
    """A ``BreakerManager.on_any_trip`` callback that latches the persistent kill switch."""

    def handler(event: BreakerEvent) -> None:
        logger.error("latching the kill switch after %s: %s", event.breaker_name, event.reason)
        safe_broker.enable_kill_switch(f"{event.breaker_name}: {event.reason}")

    return handler


__all__ = [
    "BotBreakers",
    "BreakerEvent",
    "BreakerState",
    "ConsecutiveLossBreaker",
    "DailyLossBreaker",
    "DrawdownBreaker",
    "LatencyBreaker",
    "MissingDecisionBarBreaker",
    "RealisedCostBreaker",
    "ReconciliationBreaker",
    "RollingHitRateBreaker",
    "RoundTrip",
    "SwapBreaker",
    "build_manager",
    "kill_switch_on_trip",
    "load_breaker_config",
    "transition_breaker",
]
