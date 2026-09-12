"""MetaTrader 5 broker adapter shared by every Exness bot.

One adapter, four bots: each bot instantiates :class:`MT5Broker` with its **own magic
number** and wraps it in ``SafeBroker(broker, LiveRiskConfig(...))`` (Chapter 25,
``25_live_trading/10_safety_risk_demo.py``). The class implements the contract of that
notebook's ``MockBroker`` - exactly the members ``SafeBroker`` calls - on top of the
``MetaTrader5`` Python package. Four copies of ``order_send`` would drift; this is the only
one. Per-bot differences (magic, ``LiveRiskConfig`` limits, deployment schedule) live in
``bots/<bot_id>/deploy/``.

Ported from the legacy connector (reference copy in
``bots/<bot_id>/legacy_v9_continuum/src/mt5_connector.py`` and
``v9_continuum/layers/execution.py``), with these changes:

* ``initialize()`` **without a password**: the terminal is already logged in. The adapter
  verifies ``account_info().login`` / ``server`` / ``trade_mode`` against what the bot
  expects instead of holding credentials.
* ``order_send`` with ``TRADE_ACTION_DEAL`` / ``ORDER_FILLING_IOC`` / ``deviation`` /
  ``magic``; retcodes mapped to ``OrderStatus``: ``DONE`` -> ``FILLED``, ``DONE_PARTIAL``
  -> ``FILLED`` with ``filled_quantity < quantity`` (the IOC remainder is gone), ``PLACED``
  -> ``PENDING``, ``REQUOTE`` -> one resend at the fresh price then ``REJECTED``, everything
  else ``REJECTED`` with a stable ``rejection_code``.
* ``positions_get`` / ``orders_get`` filtered by magic, aggregated per bare symbol.
* Close by an opposite market deal carrying the ``position`` ticket (one deal per ticket
  on a hedging account); cancel by ``TRADE_ACTION_REMOVE``.
* :func:`normalize_lot` (``volume_step`` / ``volume_min`` / ``volume_max``): an order that
  rounds **below** ``volume_min`` is rejected (``quantity_rounds_to_zero``); the legacy
  bump-up-to-minimum is opt-in (``min_lot_policy="clamp"``). Rounding is always down.
* ``order_calc_margin`` guard: required margin must not exceed ``max_margin_fraction`` x
  ``margin_free``; an unknown margin **fails closed**.
* Quote-to-account conversion for JPY/CAD/CHF-quoted pairs
  (:meth:`MT5Broker.unit_value_account`) read from a live tick of the USD cross, never from
  a table.
* Symbol names go through :func:`bots._shared.mt5_loader.to_market_watch` / ``to_bare``:
  research and ``SafeBroker`` see bare names (``EURUSD``), the terminal sees Market Watch
  names (``EURUSDm``). Contract sizes, volume limits and swaps are read from
  ``symbol_info`` on every call, never assumed.

Quantities are **units of the base asset** (``quantity = lots x trade_contract_size``),
the convention of ``execution.share_type: integer`` in ``setup.yaml``; lots are an
implementation detail of the terminal. Prices in ``Order`` / ``Position`` are in the
instrument's quote currency; ``unit_value_account`` gives the account-currency value of one
unit so the deployment loop can prime ``SafeBroker.record_market_snapshot`` and set
``LiveRiskConfig`` limits in account currency (``mt5-exness-broker.md`` section 3).

Execution modes: ``"paper"`` (default) requires a **demo** account -
:meth:`MT5Broker.assert_paper_trading` raises otherwise, at ``connect`` and before every
order; ``"live"`` requires ``armed_live=True`` passed by a person for the session and
refuses a demo account. Dry runs are ``SafeBroker(shadow_mode=True)``: the adapter is never
called for an order.

The ``ml4t`` types are imported when installed (the research environment); without them
(the Windows download environment, the only place ``MetaTrader5`` runs) stand-ins with the
same names and fields are used, so this module and its tests import anywhere.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from bots._shared.mt5_loader import to_bare, to_market_watch

logger = logging.getLogger(__name__)

try:  # research environment: the real types SafeBroker and the backtest engine use
    from ml4t.backtest.types import Order, OrderSide, OrderStatus, OrderType, Position
    from ml4t.live.protocols import ExecutionCapability

    ML4T_TYPES_AVAILABLE = True
except ImportError:  # Windows download environment: same names, same fields
    ML4T_TYPES_AVAILABLE = False

    class OrderSide(Enum):  # type: ignore[no-redef]
        BUY = "buy"
        SELL = "sell"

    class OrderStatus(Enum):  # type: ignore[no-redef]
        PENDING = "pending"
        FILLED = "filled"
        CANCELLED = "cancelled"
        REJECTED = "rejected"

    class OrderType(Enum):  # type: ignore[no-redef]
        MARKET = "market"
        MOC = "moc"
        LIMIT = "limit"
        STOP = "stop"
        STOP_LIMIT = "stop_limit"
        TRAILING_STOP = "trailing_stop"

    class ExecutionCapability(str, Enum):  # type: ignore[no-redef]
        LIMIT = "limit"
        STOP = "stop"
        STOP_LIMIT = "stop_limit"
        TRAILING_STOP = "trailing_stop"
        OPENING_AUCTION = "opening_auction"
        CLOSE_AUCTION = "close_auction"
        PARTIAL_FILL = "partial_fill"
        CONTINGENT = "contingent"

    @dataclass
    class Order:  # type: ignore[no-redef]
        asset: str
        side: OrderSide
        quantity: float
        order_type: OrderType = OrderType.MARKET
        limit_price: float | None = None
        stop_price: float | None = None
        order_id: str = ""
        status: OrderStatus = OrderStatus.PENDING
        created_at: datetime | None = None
        filled_at: datetime | None = None
        filled_price: float | None = None
        filled_quantity: float = 0.0
        rejection_reason: str | None = None
        requested_quantity: float | None = None
        _rejection_code: str | None = None

        def __post_init__(self) -> None:
            if self.requested_quantity is None:
                self.requested_quantity = self.quantity

        @property
        def rejection_code(self) -> str | None:
            if self.status is not OrderStatus.REJECTED:
                return None
            return self._rejection_code or "order_validation_failed"

        def reject(self, reason: str, code: str) -> None:
            self.status = OrderStatus.REJECTED
            self.rejection_reason = reason
            self._rejection_code = code

    @dataclass
    class Position:  # type: ignore[no-redef]
        asset: str
        quantity: float
        entry_price: float
        entry_time: datetime
        current_price: float | None = None
        bars_held: int = 0
        context: dict = field(default_factory=dict)
        multiplier: float = 1.0

        @property
        def side(self) -> str:
            return "long" if self.quantity > 0 else "short"


class MT5BrokerError(RuntimeError):
    """Connection or account verification failure."""


class PaperTradingViolation(MT5BrokerError):
    """``execution_mode="paper"`` on an account that is not a demo account."""


class NotArmedError(MT5BrokerError):
    """``execution_mode="live"`` without ``armed_live=True`` for this session."""


# MetaTrader5 constants, read from the module when present (a fake may omit some)
MT5_DEFAULTS: dict[str, int] = {
    "TRADE_ACTION_DEAL": 1,
    "TRADE_ACTION_PENDING": 5,
    "TRADE_ACTION_REMOVE": 8,
    "ORDER_TYPE_BUY": 0,
    "ORDER_TYPE_SELL": 1,
    "ORDER_TYPE_BUY_LIMIT": 2,
    "ORDER_TYPE_SELL_LIMIT": 3,
    "ORDER_TYPE_BUY_STOP": 4,
    "ORDER_TYPE_SELL_STOP": 5,
    "ORDER_TIME_GTC": 0,
    "ORDER_FILLING_IOC": 1,
    "POSITION_TYPE_BUY": 0,
    "POSITION_TYPE_SELL": 1,
    "ACCOUNT_TRADE_MODE_DEMO": 0,
    "TRADE_RETCODE_REQUOTE": 10004,
    "TRADE_RETCODE_PLACED": 10008,
    "TRADE_RETCODE_DONE": 10009,
    "TRADE_RETCODE_DONE_PARTIAL": 10010,
}

# retcode -> stable rejection code (subset of the MQL5 trade server return codes)
REJECTION_CODES: dict[int, str] = {
    10004: "requote",
    10006: "broker_reject",
    10007: "connection",
    10011: "request_error",
    10012: "timeout",
    10013: "order_validation_failed",
    10014: "invalid_volume",
    10015: "invalid_price",
    10016: "invalid_stops",
    10017: "trade_disabled",
    10018: "market_closed",
    10019: "insufficient_buying_power",
    10020: "price_changed",
    10021: "price_off",
    10022: "invalid_expiration",
    10023: "order_changed",
    10024: "too_many_requests",
    10025: "no_changes",
    10026: "autotrading_disabled_server",
    10027: "autotrading_disabled_client",
    10028: "locked",
    10029: "frozen",
    10030: "invalid_fill",
    10031: "connection",
    10032: "only_real",
    10033: "limit_orders",
    10034: "limit_volume",
    10035: "invalid_order",
    10036: "position_closed",
    10038: "close_order_exist",
    10039: "limit_positions",
    10040: "reject_cancel",
    10041: "long_only",
    10042: "short_only",
    10043: "close_only",
    10044: "fifo_close",
    10045: "hedge_prohibited",
}

USD_CURRENCY = "USD"


def _const(mt5: Any, name: str) -> int:
    return int(getattr(mt5, name, MT5_DEFAULTS[name]))


def normalize_lot(lots: float, info: Any, *, min_lot_policy: str = "reject") -> float:
    """Round ``lots`` **down** to ``volume_step`` and clamp to ``[volume_min, volume_max]``.

    Returns ``0.0`` when the rounded size is below ``volume_min`` and the policy is
    ``"reject"``; with ``"clamp"`` it returns ``volume_min`` (the legacy behaviour, which
    silently enlarges a small order and is therefore not the default).
    """
    if min_lot_policy not in ("reject", "clamp"):
        raise ValueError(f"min_lot_policy must be 'reject' or 'clamp', got {min_lot_policy!r}")
    step = float(info.volume_step)
    vmin = float(info.volume_min)
    vmax = float(info.volume_max)
    if lots <= 0 or step <= 0:
        return 0.0
    decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    rounded = round(math.floor(lots / step + 1e-9) * step, decimals)
    if rounded < vmin - 1e-12:
        return vmin if min_lot_policy == "clamp" else 0.0
    return float(min(rounded, vmax))


@dataclass
class FillRecord:
    """Execution-quality record of one deal (the parity tape reads these)."""

    timestamp: datetime
    asset: str
    market_watch: str
    side: OrderSide
    lots: float
    filled_lots: float
    request_price: float
    fill_price: float | None
    slippage_points: float | None
    latency_ms: float
    retcode: int
    order_ticket: int
    deal_ticket: int
    requote_attempts: int
    action: str  # "open" | "close"


class MT5Broker:
    """MetaTrader 5 adapter satisfying the ``SafeBroker`` broker contract.

    Args:
        magic: This bot's magic number; every order carries it and every query filters on it.
        execution_mode: ``"paper"`` (demo account required) or ``"live"`` (real account,
            ``armed_live`` required).
        armed_live: Set by a person for one session; ignored in paper mode.
        expected_login / expected_server: Verified against ``account_info`` at connect.
        max_margin_fraction: An order is refused when ``order_calc_margin`` exceeds this
            fraction of ``margin_free``.
        deviation_points / close_deviation_points: Max slippage in points for opening /
            closing deals (the legacy connector used 20 / 30).
        min_lot_policy: ``"reject"`` (default) or ``"clamp"``; see :func:`normalize_lot`.
        requote_retries: Resends at the fresh price after ``TRADE_RETCODE_REQUOTE``.
        reconnect_backoff: Seconds to wait between ``ensure_connected`` attempts.
        mt5: The ``MetaTrader5`` module; defaults to a lazy import. Tests pass a fake.
        initialize_kwargs: Extra ``initialize()`` arguments (``path=`` to pick a terminal);
            never a password.
        sleep: Injectable ``time.sleep`` for tests.
    """

    def __init__(
        self,
        magic: int,
        *,
        execution_mode: str = "paper",
        armed_live: bool = False,
        expected_login: int | None = None,
        expected_server: str | None = None,
        max_margin_fraction: float = 0.2,
        deviation_points: int = 20,
        close_deviation_points: int = 30,
        comment: str = "ml4t",
        min_lot_policy: str = "reject",
        requote_retries: int = 1,
        reconnect_backoff: tuple[float, ...] = (5.0, 10.0, 30.0),
        mt5: Any = None,
        initialize_kwargs: Mapping[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if execution_mode not in ("paper", "live"):
            raise ValueError(f"execution_mode must be 'paper' or 'live', got {execution_mode!r}")
        if not 0 < max_margin_fraction <= 1:
            raise ValueError("max_margin_fraction must be in (0, 1]")
        if initialize_kwargs and "password" in initialize_kwargs:
            raise ValueError("MT5Broker never holds a password; log the terminal in by hand")
        if min_lot_policy not in ("reject", "clamp"):
            raise ValueError(f"min_lot_policy must be 'reject' or 'clamp', got {min_lot_policy!r}")
        self.magic = int(magic)
        self.execution_mode = execution_mode
        self.armed_live = bool(armed_live)
        self.expected_login = int(expected_login) if expected_login is not None else (int(os.environ["MT5_EXPECTED_LOGIN"]) if "MT5_EXPECTED_LOGIN" in os.environ else None)
        self.expected_server = str(expected_server) if expected_server is not None else os.environ.get("MT5_EXPECTED_SERVER")
        self.max_margin_fraction = float(max_margin_fraction)
        self.deviation_points = int(deviation_points)
        self.close_deviation_points = int(close_deviation_points)
        self.comment = comment[:31]  # MT5 comment limit
        self.min_lot_policy = min_lot_policy
        self.requote_retries = int(requote_retries)
        self.reconnect_backoff = tuple(reconnect_backoff)
        self._mt5 = mt5
        self.initialize_kwargs = dict(initialize_kwargs or {})
        self._sleep = sleep
        self._connected = False
        self.account_snapshot: dict[str, Any] = {}
        self.fills: list[FillRecord] = []

    # ------------------------------------------------------------------ module access
    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            try:
                self._mt5 = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise MT5BrokerError(
                    "MetaTrader5 is not installed (Windows only): "
                    "uv run --no-project --python 3.14 --with MetaTrader5 ..."
                ) from exc
        return self._mt5

    # ------------------------------------------------------------------ connection
    async def connect(self) -> None:
        self._connect()

    def _connect(self) -> None:
        mt5 = self.mt5
        if not mt5.initialize(**self.initialize_kwargs):
            raise MT5BrokerError(f"MetaTrader5.initialize() failed: {mt5.last_error()}")
        try:
            account = mt5.account_info()
            if account is None:
                raise MT5BrokerError(f"account_info() is None after initialize: {mt5.last_error()}")
            self._verify_account(account)
            self._connected = True
            self.account_snapshot = self.account_summary()
        except Exception:
            self._connected = False
            mt5.shutdown()
            raise
        logger.info(
            "MT5Broker connected: login=%s server=%s trade_mode=%s mode=%s magic=%s",
            account.login,
            account.server,
            account.trade_mode,
            self.execution_mode,
            self.magic,
        )

    def _verify_account(self, account: Any) -> None:
        if self.expected_login is not None and int(account.login) != int(self.expected_login):
            raise MT5BrokerError(f"terminal is logged in as {account.login}, expected {self.expected_login}")
        if self.expected_server is not None and str(account.server) != str(self.expected_server):
            raise MT5BrokerError(f"terminal is on server {account.server!r}, expected {self.expected_server!r}")
        demo = _const(self.mt5, "ACCOUNT_TRADE_MODE_DEMO")
        if self.execution_mode == "paper":
            self._assert_demo(account, demo)
        elif int(account.trade_mode) == demo:
            raise MT5BrokerError("execution_mode='live' on a demo account; use execution_mode='paper' for demo")

    @staticmethod
    def _assert_demo(account: Any, demo: int) -> None:
        if int(account.trade_mode) != demo:
            raise PaperTradingViolation(
                f"execution_mode='paper' but account {account.login}@{account.server} has "
                f"trade_mode={account.trade_mode} (demo is {demo}); refusing to trade"
            )

    async def disconnect(self) -> None:
        if self._connected:
            self.mt5.shutdown()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def is_connected_async(self) -> bool:
        return self._connected and self.health_check()

    def health_check(self) -> bool:
        """``terminal_info()`` answers and reports a live server connection."""
        try:
            info = self.mt5.terminal_info()
        except Exception:  # noqa: BLE001 - the terminal DLL can raise anything
            return False
        return info is not None and bool(getattr(info, "connected", False))

    def ensure_connected(self) -> bool:
        """Reconnect with backoff when the terminal dropped; True when connected."""
        if self._connected and self.health_check():
            return True
        for attempt, wait in enumerate(self.reconnect_backoff, start=1):
            logger.warning("MT5Broker: connection lost, reconnect %d in %.0fs", attempt, wait)
            self._sleep(wait)
            try:
                self._connect()
            except MT5BrokerError as exc:
                logger.error("MT5Broker: reconnect %d failed: %s", attempt, exc)
                continue
            if self.health_check():
                return True
        self._connected = False
        return False

    # ------------------------------------------------------------------ contract: static
    @property
    def execution_capabilities(self) -> frozenset:
        return frozenset({ExecutionCapability.LIMIT, ExecutionCapability.STOP, ExecutionCapability.PARTIAL_FILL})

    def assert_paper_trading(self) -> None:
        """Raise unless the account is a demo account when ``execution_mode == "paper"``."""
        if self.execution_mode != "paper":
            return
        account = self.mt5.account_info()
        if account is None:
            raise PaperTradingViolation("account_info() is None: cannot prove this is a demo account")
        self._assert_demo(account, _const(self.mt5, "ACCOUNT_TRADE_MODE_DEMO"))

    def _guard_mode(self) -> None:
        if not self._connected:
            raise MT5BrokerError("not connected; call connect() first")
        if self.execution_mode == "paper":
            self.assert_paper_trading()
        elif not self.armed_live:
            raise NotArmedError("execution_mode='live' but armed_live is False for this session")

    # ------------------------------------------------------------------ account
    def account_summary(self) -> dict[str, Any]:
        a = self.mt5.account_info()
        if a is None:
            return {}
        return {
            key: getattr(a, key, None)
            for key in ("login", "server", "trade_mode", "currency", "leverage", "balance", "equity", "margin", "margin_free")
        }

    async def get_account_value_async(self) -> float:
        a = self.mt5.account_info()
        if a is None:
            raise MT5BrokerError("account_info() is None")
        return float(a.equity)

    async def get_cash_async(self) -> float:
        a = self.mt5.account_info()
        if a is None:
            raise MT5BrokerError("account_info() is None")
        return float(a.margin_free)

    # ------------------------------------------------------------------ symbols
    def symbol_info(self, asset: str) -> Any:
        """``symbol_info`` of the Market Watch name of ``asset``; None when unknown."""
        return self.mt5.symbol_info(to_market_watch(asset))

    def tick(self, asset: str) -> Any:
        return self.mt5.symbol_info_tick(to_market_watch(asset))

    def quote_to_account_rate(self, currency: str, account_currency: str = USD_CURRENCY) -> float:
        """Account-currency value of one unit of ``currency``, from a live tick of the USD cross.

        ``JPY`` -> ``1 / mid(USDJPY)``; ``EUR`` -> ``mid(EURUSD)``; ``USD`` -> 1. Raises when
        neither cross is quoted, which is better than a stale table.
        """
        if currency == account_currency:
            return 1.0
        direct = self.mt5.symbol_info_tick(to_market_watch(f"{currency}{account_currency}"))
        if direct is not None and direct.bid > 0:
            return (direct.bid + direct.ask) / 2
        inverse = self.mt5.symbol_info_tick(to_market_watch(f"{account_currency}{currency}"))
        if inverse is not None and inverse.bid > 0:
            return 2 / (inverse.bid + inverse.ask)
        raise MT5BrokerError(f"no {currency}/{account_currency} cross quoted; cannot convert")

    def unit_value_account(self, asset: str, price: float | None = None) -> float:
        """Account-currency value of **one unit** of ``asset`` at ``price`` (mid when None).

        ``EURUSD`` at 1.10 -> 1.10; ``USDJPY`` at 147 -> 1.0 (one USD); ``XAUUSD`` at 2400
        -> 2400 (one ounce); ``EURJPY`` -> price / USDJPY. Multiply by the quantity in units
        for the notional ``SafeBroker`` compares with ``max_order_value``.
        """
        info = self.symbol_info(asset)
        if info is None:
            raise MT5BrokerError(f"{asset}: {to_market_watch(asset)!r} is not in Market Watch")
        if price is None:
            tick = self.tick(asset)
            if tick is None:
                raise MT5BrokerError(f"{asset}: no tick")
            price = (tick.bid + tick.ask) / 2
        account_ccy = self.account_snapshot.get("currency") or USD_CURRENCY
        return float(price) * self.quote_to_account_rate(str(info.currency_profit), account_ccy)

    # ------------------------------------------------------------------ positions
    def _raw_positions(self, asset: str | None = None) -> list[Any]:
        raw = self.mt5.positions_get() if asset is None else self.mt5.positions_get(symbol=to_market_watch(asset))
        return [p for p in (raw or ()) if int(p.magic) == self.magic]

    def all_positions(self) -> list[dict[str, Any]]:
        """Every position on the account, **any magic** (for the account-level breaker)."""
        buy = _const(self.mt5, "POSITION_TYPE_BUY")
        return [
            {
                "ticket": p.ticket,
                "symbol": to_bare(p.symbol),
                "market_watch": p.symbol,
                "magic": p.magic,
                "type": "buy" if int(p.type) == buy else "sell",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "profit": p.profit,
                "swap": getattr(p, "swap", 0.0),
                "comment": getattr(p, "comment", ""),
            }
            for p in (self.mt5.positions_get() or ())
        ]

    def _aggregate(self, raw: list[Any]) -> dict[str, Position]:
        buy = _const(self.mt5, "POSITION_TYPE_BUY")
        by_symbol: dict[str, list[tuple[Any, float]]] = {}
        for p in raw:
            info = self.mt5.symbol_info(p.symbol)
            if info is None:
                logger.error("position %s on %s: symbol_info is None; skipped", p.ticket, p.symbol)
                continue
            sign = 1.0 if int(p.type) == buy else -1.0
            units = sign * float(p.volume) * float(info.trade_contract_size)
            by_symbol.setdefault(to_bare(p.symbol), []).append((p, units))
        out: dict[str, Position] = {}
        for bare, items in by_symbol.items():
            quantity = sum(u for _, u in items)
            gross = sum(abs(u) for _, u in items)
            if gross == 0:
                continue
            if quantity == 0:
                logger.warning("%s: hedged flat (%d tickets), reported as no net position", bare, len(items))
                continue
            entry = sum(abs(u) * float(p.price_open) for p, u in items) / gross
            opened = min(int(p.time) for p, _ in items)
            out[bare] = Position(
                asset=bare,
                quantity=quantity,
                entry_price=entry,
                entry_time=datetime.fromtimestamp(opened, tz=UTC),  # server epoch read as UTC
                current_price=float(items[0][0].price_current),
                context={
                    "tickets": [int(p.ticket) for p, _ in items],
                    "lots": [float(p.volume) for p, _ in items],
                    "market_watch": items[0][0].symbol,
                    "magic": self.magic,
                    "profit": sum(float(p.profit) for p, _ in items),
                    "swap": sum(float(getattr(p, "swap", 0.0)) for p, _ in items),
                },
            )
        return out

    @property
    def positions(self) -> dict[str, Position]:
        return self._aggregate(self._raw_positions())

    def get_position(self, asset: str) -> Position | None:
        return self._aggregate(self._raw_positions(asset)).get(asset)

    async def get_positions_async(self) -> dict[str, Position]:
        return self.positions

    async def get_position_async(self, asset: str) -> Position | None:
        return self.get_position(asset)

    # ------------------------------------------------------------------ pending orders
    def _order_type_from_mt5(self, mt5_type: int) -> tuple[OrderSide, OrderType]:
        m = self.mt5
        table = {
            _const(m, "ORDER_TYPE_BUY"): (OrderSide.BUY, OrderType.MARKET),
            _const(m, "ORDER_TYPE_SELL"): (OrderSide.SELL, OrderType.MARKET),
            _const(m, "ORDER_TYPE_BUY_LIMIT"): (OrderSide.BUY, OrderType.LIMIT),
            _const(m, "ORDER_TYPE_SELL_LIMIT"): (OrderSide.SELL, OrderType.LIMIT),
            _const(m, "ORDER_TYPE_BUY_STOP"): (OrderSide.BUY, OrderType.STOP),
            _const(m, "ORDER_TYPE_SELL_STOP"): (OrderSide.SELL, OrderType.STOP),
        }
        try:
            return table[int(mt5_type)]
        except KeyError as exc:
            raise MT5BrokerError(f"unsupported MT5 order type {mt5_type}") from exc

    @property
    def pending_orders(self) -> list[Order]:
        out: list[Order] = []
        for o in self.mt5.orders_get() or ():
            if int(o.magic) != self.magic:
                continue
            info = self.mt5.symbol_info(o.symbol)
            size = float(info.trade_contract_size) if info is not None else 1.0
            side, otype = self._order_type_from_mt5(o.type)
            out.append(
                Order(
                    asset=to_bare(o.symbol),
                    side=side,
                    quantity=float(o.volume_current) * size,
                    order_type=otype,
                    limit_price=float(o.price_open) if otype is OrderType.LIMIT else None,
                    stop_price=float(o.price_open) if otype is OrderType.STOP else None,
                    order_id=str(o.ticket),
                    status=OrderStatus.PENDING,
                    created_at=datetime.fromtimestamp(int(o.time_setup), tz=UTC),
                )
            )
        return out

    async def get_pending_orders_async(self) -> list[Order]:
        return self.pending_orders

    # ------------------------------------------------------------------ margin
    def margin_check(self, market_watch: str, side: OrderSide, lots: float, price: float) -> tuple[bool, dict[str, Any]]:
        """Refuse when ``order_calc_margin`` exceeds ``max_margin_fraction x margin_free``.

        Unknown margin (``None`` from the terminal, an exception, or no account info)
        **fails closed**.
        """
        m = self.mt5
        otype = _const(m, "ORDER_TYPE_BUY") if side is OrderSide.BUY else _const(m, "ORDER_TYPE_SELL")
        try:
            required = m.order_calc_margin(otype, market_watch, float(lots), float(price))
        except Exception as exc:  # noqa: BLE001
            required = None
            logger.error("order_calc_margin raised for %s: %s", market_watch, exc)
        account = m.account_info()
        free = float(account.margin_free) if account is not None else None
        detail: dict[str, Any] = {
            "lots": lots,
            "price": price,
            "required_margin": required,
            "margin_free": free,
            "max_fraction": self.max_margin_fraction,
            "allowed_margin": None if free is None else free * self.max_margin_fraction,
        }
        if required is None or free is None:
            detail["status"] = "unknown"
            return False, detail
        if float(required) > free * self.max_margin_fraction:
            detail["status"] = "reject"
            return False, detail
        detail["status"] = "ok"
        return True, detail

    # ------------------------------------------------------------------ orders
    def _send(self, request: dict[str, Any]) -> tuple[Any, float]:
        t0 = time.perf_counter()
        result = self.mt5.order_send(request)
        return result, (time.perf_counter() - t0) * 1000.0

    def _send_with_requote(self, request: dict[str, Any], market_watch: str, side: OrderSide) -> tuple[Any, float, int]:
        """Send a market deal; on ``REQUOTE`` refresh the price and resend up to ``requote_retries``."""
        requote = _const(self.mt5, "TRADE_RETCODE_REQUOTE")
        attempts = 0
        result, latency = self._send(request)
        while result is not None and int(result.retcode) == requote and attempts < self.requote_retries:
            attempts += 1
            tick = self.mt5.symbol_info_tick(market_watch)
            if tick is None:
                break
            request["price"] = float(tick.ask if side is OrderSide.BUY else tick.bid)
            result, more = self._send(request)
            latency += more
        return result, latency, attempts

    def _record_fill(
        self, order: Order, request: dict[str, Any], result: Any, info: Any, latency_ms: float, attempts: int, action: str
    ) -> None:
        point = float(getattr(info, "point", 0.0) or 0.0)
        fill_price = float(result.price) if getattr(result, "price", None) else None
        slippage = abs(fill_price - float(request["price"])) / point if fill_price is not None and point > 0 else None
        self.fills.append(
            FillRecord(
                timestamp=datetime.now(UTC),
                asset=order.asset,
                market_watch=request["symbol"],
                side=order.side,
                lots=float(request["volume"]),
                filled_lots=float(getattr(result, "volume", 0.0) or 0.0),
                request_price=float(request["price"]),
                fill_price=fill_price,
                slippage_points=slippage,
                latency_ms=latency_ms,
                retcode=int(result.retcode),
                order_ticket=int(getattr(result, "order", 0) or 0),
                deal_ticket=int(getattr(result, "deal", 0) or 0),
                requote_attempts=attempts,
                action=action,
            )
        )

    def _apply_result(
        self,
        order: Order,
        request: dict[str, Any],
        result: Any,
        info: Any,
        latency_ms: float,
        attempts: int,
        action: str = "open",
    ) -> Order:
        m = self.mt5
        if result is None:
            order.reject(f"order_send returned None: {m.last_error()}", "connection")
            return order
        rc = int(result.retcode)
        size = float(info.trade_contract_size)
        if rc in (_const(m, "TRADE_RETCODE_DONE"), _const(m, "TRADE_RETCODE_DONE_PARTIAL")):
            filled_lots = float(getattr(result, "volume", 0.0) or request["volume"])
            order.status = OrderStatus.FILLED
            order.filled_quantity = filled_lots * size
            order.filled_price = float(result.price) if getattr(result, "price", None) else float(request["price"])
            order.filled_at = datetime.now(UTC)
            order.order_id = str(getattr(result, "order", "") or "")
            self._record_fill(order, request, result, info, latency_ms, attempts, action)
            if order.filled_quantity + 1e-9 < order.quantity:
                logger.warning("%s: partial fill %.2f of %.2f lots (retcode %s)", order.asset, filled_lots, request["volume"], rc)
            return order
        if rc == _const(m, "TRADE_RETCODE_PLACED"):
            order.status = OrderStatus.PENDING
            order.order_id = str(getattr(result, "order", "") or "")
            return order
        code = REJECTION_CODES.get(rc, "broker_reject")
        comment = getattr(result, "comment", "")
        if rc == _const(m, "TRADE_RETCODE_REQUOTE"):
            comment = f"requote after {attempts + 1} attempt(s): {comment}"
        order.reject(f"retcode {rc} ({comment})", code)
        return order

    async def submit_order_async(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        **_: Any,
    ) -> Order:
        """Send one order; ``quantity`` in units of the base asset (signed when ``side`` is None)."""
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
        quantity = abs(float(quantity))
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            created_at=datetime.now(UTC),
        )
        self._guard_mode()
        m = self.mt5
        mw = to_market_watch(asset)
        info = m.symbol_info(mw)
        if info is None:
            order.reject(f"{asset}: {mw!r} is not in Market Watch ({m.last_error()})", "unknown_symbol")
            return order
        size = float(info.trade_contract_size)
        lots = normalize_lot(quantity / size, info, min_lot_policy=self.min_lot_policy)
        if lots <= 0:
            order.reject(
                f"{quantity:g} units = {quantity / size:.4f} lots rounds to zero "
                f"(volume_min {info.volume_min}, step {info.volume_step})",
                "quantity_rounds_to_zero",
            )
            return order
        order.quantity = lots * size  # requested_quantity keeps what the caller asked for

        tick = m.symbol_info_tick(mw)
        if tick is None:
            order.reject(f"{asset}: no price ({m.last_error()})", "price_unavailable")
            return order

        if order_type is OrderType.MARKET:
            action = _const(m, "TRADE_ACTION_DEAL")
            mt5_type = _const(m, "ORDER_TYPE_BUY") if side is OrderSide.BUY else _const(m, "ORDER_TYPE_SELL")
            price = float(tick.ask if side is OrderSide.BUY else tick.bid)
        elif order_type is OrderType.LIMIT:
            if limit_price is None:
                order.reject("limit order without limit_price", "order_validation_failed")
                return order
            action = _const(m, "TRADE_ACTION_PENDING")
            mt5_type = _const(m, "ORDER_TYPE_BUY_LIMIT") if side is OrderSide.BUY else _const(m, "ORDER_TYPE_SELL_LIMIT")
            price = float(limit_price)
        elif order_type is OrderType.STOP:
            if stop_price is None:
                order.reject("stop order without stop_price", "order_validation_failed")
                return order
            action = _const(m, "TRADE_ACTION_PENDING")
            mt5_type = _const(m, "ORDER_TYPE_BUY_STOP") if side is OrderSide.BUY else _const(m, "ORDER_TYPE_SELL_STOP")
            price = float(stop_price)
        else:
            order.reject(f"order type {order_type} not supported by MT5Broker", "order_validation_failed")
            return order

        ok, detail = self.margin_check(mw, side, lots, price)
        if not ok:
            order.reject(
                f"margin guard {detail['status']}: required {detail['required_margin']} > allowed "
                f"{detail['allowed_margin']} ({self.max_margin_fraction:.0%} of free margin {detail['margin_free']})",
                "insufficient_buying_power",
            )
            return order

        request: dict[str, Any] = {
            "action": action,
            "symbol": mw,
            "volume": lots,
            "type": mt5_type,
            "price": price,
            "deviation": self.deviation_points,
            "magic": self.magic,
            "comment": self.comment,
            "type_time": _const(m, "ORDER_TIME_GTC"),
            "type_filling": _const(m, "ORDER_FILLING_IOC"),
        }
        if action == _const(m, "TRADE_ACTION_DEAL"):
            result, latency, attempts = self._send_with_requote(request, mw, side)
        else:
            result, latency = self._send(request)
            attempts = 0
        return self._apply_result(order, request, result, info, latency, attempts)

    async def cancel_order_async(self, order_id: str) -> bool:
        self._guard_mode()
        m = self.mt5
        result, _ = self._send({"action": _const(m, "TRADE_ACTION_REMOVE"), "order": int(order_id)})
        ok = result is not None and int(result.retcode) == _const(m, "TRADE_RETCODE_DONE")
        if not ok:
            logger.error("cancel %s failed: %s", order_id, getattr(result, "comment", m.last_error()))
        return ok

    async def close_position_async(self, asset: str) -> Order | None:
        """Close every ticket of ``asset`` carrying this magic with opposite market deals."""
        self._guard_mode()
        m = self.mt5
        raw = self._raw_positions(asset)
        if not raw:
            return None
        mw = to_market_watch(asset)
        info = m.symbol_info(mw)
        if info is None:
            raise MT5BrokerError(f"{asset}: {mw!r} is not in Market Watch")
        size = float(info.trade_contract_size)
        buy_pos = _const(m, "POSITION_TYPE_BUY")
        net_units = sum((1.0 if int(p.type) == buy_pos else -1.0) * float(p.volume) * size for p in raw)
        total = Order(
            asset=asset,
            side=OrderSide.SELL if net_units >= 0 else OrderSide.BUY,
            quantity=sum(float(p.volume) for p in raw) * size,
            order_type=OrderType.MARKET,
            created_at=datetime.now(UTC),
        )
        filled_units = 0.0
        weighted = 0.0
        failures: list[str] = []
        tickets: list[str] = []
        for p in raw:
            is_long = int(p.type) == buy_pos
            side = OrderSide.SELL if is_long else OrderSide.BUY
            tick = m.symbol_info_tick(mw)
            if tick is None:
                failures.append(f"{p.ticket}: no price")
                continue
            request: dict[str, Any] = {
                "action": _const(m, "TRADE_ACTION_DEAL"),
                "symbol": mw,
                "volume": float(p.volume),
                "type": _const(m, "ORDER_TYPE_SELL") if is_long else _const(m, "ORDER_TYPE_BUY"),
                "position": int(p.ticket),
                "price": float(tick.bid if is_long else tick.ask),
                "deviation": self.close_deviation_points,
                "magic": self.magic,
                "comment": f"{self.comment} close"[:31],
                "type_time": _const(m, "ORDER_TIME_GTC"),
                "type_filling": _const(m, "ORDER_FILLING_IOC"),
            }
            leg = Order(asset=asset, side=side, quantity=float(p.volume) * size, order_type=OrderType.MARKET)
            result, latency, attempts = self._send_with_requote(request, mw, side)
            leg = self._apply_result(leg, request, result, info, latency, attempts, action="close")
            if leg.status is OrderStatus.FILLED:
                filled_units += leg.filled_quantity
                weighted += leg.filled_quantity * float(leg.filled_price or 0.0)
                tickets.append(leg.order_id)
            else:
                failures.append(f"{p.ticket}: {leg.rejection_reason}")
        total.filled_quantity = filled_units
        total.filled_price = weighted / filled_units if filled_units else None
        total.order_id = ",".join(tickets)
        if failures:
            total.reject("; ".join(failures), "close_failed")
        else:
            total.status = OrderStatus.FILLED
        total.filled_at = datetime.now(UTC) if filled_units else None
        return total


__all__ = [
    "ML4T_TYPES_AVAILABLE",
    "REJECTION_CODES",
    "FillRecord",
    "MT5Broker",
    "MT5BrokerError",
    "NotArmedError",
    "PaperTradingViolation",
    "normalize_lot",
]
