"""bots/_shared/mt5_broker.py against the fake MetaTrader5 module (no terminal, no network).

Covers the SafeBroker contract members and the ported connector behaviour: fills, partial
fill, rejection codes, requote resend, lot normalisation, margin refusal, demo-only
assertion, live arming, symbol mapping, positions / pending orders, close and cancel,
quote-to-account conversion and reconnect.
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import bots._shared.mt5_loader as M
from bots._shared.mt5_broker import (
    ML4T_TYPES_AVAILABLE,
    MT5Broker,
    MT5BrokerError,
    NotArmedError,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperTradingViolation,
    normalize_lot,
)
from bots._shared.testing.fake_mt5 import (
    ACCOUNT_TRADE_MODE_REAL,
    TRADE_ACTION_DEAL,
    TRADE_ACTION_PENDING,
    TRADE_ACTION_REMOVE,
    TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_NO_MONEY,
    TRADE_RETCODE_REQUOTE,
    FakeMT5,
    default_symbol_info,
)

MAGIC = 202601
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
SUFFIX = M.MT5_SYMBOL_SUFFIX
EUR = f"EURUSD{SUFFIX}"
JPY = f"USDJPY{SUFFIX}"
XAU = f"XAUUSD{SUFFIX}"


def run(coro):
    return asyncio.run(coro)


def _fake(**kwargs) -> FakeMT5:
    symbols = {
        EUR: default_symbol_info(EUR),
        JPY: default_symbol_info(JPY),
        XAU: default_symbol_info(XAU, trade_contract_size=100.0, digits=2, point=0.01, currency_profit="USD", currency_base="XAU"),
    }
    fake = FakeMT5(symbols=symbols, now_utc=NOW, **kwargs)
    fake.quotes[EUR] = (1.10000, 1.10010)
    fake.quotes[JPY] = (147.000, 147.020)
    fake.quotes[XAU] = (2400.00, 2400.30)
    fake.account["leverage"] = 2000
    return fake


def _broker(fake: FakeMT5, **kwargs) -> MT5Broker:
    kwargs.setdefault("reconnect_backoff", (0.0,))
    kwargs.setdefault("sleep", lambda _s: None)
    broker = MT5Broker(MAGIC, mt5=fake, **kwargs)
    run(broker.connect())
    return broker


# ---------------------------------------------------------------------------
# connection and modes
# ---------------------------------------------------------------------------
def test_module_imports_with_or_without_ml4t() -> None:
    assert ML4T_TYPES_AVAILABLE == (importlib.util.find_spec("ml4t") is not None)


def test_paper_mode_requires_a_demo_account() -> None:
    real = _fake(account_trade_mode=ACCOUNT_TRADE_MODE_REAL)
    broker = MT5Broker(MAGIC, mt5=real)
    with pytest.raises(PaperTradingViolation):
        run(broker.connect())
    assert not broker.is_connected
    assert real.calls[-1][0] == "shutdown"  # never left initialised after a refusal

    demo = _fake()
    broker = _broker(demo)
    assert broker.is_connected and run(broker.is_connected_async())
    assert broker.account_snapshot["trade_mode"] == 0
    broker.assert_paper_trading()  # no raise
    # the account flips to real under a paper broker (wrong terminal logged in later)
    demo.account_trade_mode = ACCOUNT_TRADE_MODE_REAL
    with pytest.raises(PaperTradingViolation):
        run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert not [c for c in demo.calls if c[0] == "order_send"]


def test_live_mode_needs_arming_and_refuses_demo() -> None:
    demo = _fake()
    with pytest.raises(MT5BrokerError):
        run(MT5Broker(MAGIC, mt5=demo, execution_mode="live").connect())
    real = _fake(account_trade_mode=ACCOUNT_TRADE_MODE_REAL)
    broker = MT5Broker(MAGIC, mt5=real, execution_mode="live")
    run(broker.connect())
    with pytest.raises(NotArmedError):
        run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    armed = MT5Broker(MAGIC, mt5=real, execution_mode="live", armed_live=True)
    run(armed.connect())
    order = run(armed.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert order.status is OrderStatus.FILLED


def test_connect_verifies_login_server_and_never_holds_a_password() -> None:
    fake = _fake()
    with pytest.raises(MT5BrokerError):
        run(MT5Broker(MAGIC, mt5=fake, expected_login=1).connect())
    with pytest.raises(MT5BrokerError):
        run(MT5Broker(MAGIC, mt5=fake, expected_server="Other").connect())
    with pytest.raises(ValueError):
        MT5Broker(MAGIC, mt5=fake, initialize_kwargs={"password": "x"})
    with pytest.raises(MT5BrokerError):
        run(MT5Broker(MAGIC, mt5=_fake(fail_initialize=True)).connect())
    broker = _broker(fake, expected_login=12345678, expected_server="Exness-MT5Trial")
    init_call = next(c for c in fake.calls if c[0] == "initialize")
    assert "password" not in init_call[2] and "login" not in init_call[2]
    assert broker.execution_capabilities  # non-empty frozenset


def test_ensure_connected_reconnects_after_a_drop() -> None:
    fake = _fake()
    broker = _broker(fake)
    fake.connected = False
    fake.fail_initialize = True
    assert not run(broker.is_connected_async())
    assert broker.ensure_connected() is False
    fake.connected = True
    fake.fail_initialize = False
    assert broker.ensure_connected() is True
    assert [c[0] for c in fake.calls].count("initialize") >= 2


# ---------------------------------------------------------------------------
# lots and margin
# ---------------------------------------------------------------------------
def test_normalize_lot_rounds_down_and_clamps() -> None:
    info = default_symbol_info(EUR)  # min 0.01, step 0.01, max 200
    assert normalize_lot(1.234567, info) == 1.23
    assert normalize_lot(0.019, info) == 0.01
    assert normalize_lot(0.02, info) == 0.02
    assert normalize_lot(0.004, info) == 0.0
    assert normalize_lot(0.004, info, min_lot_policy="clamp") == 0.01
    assert normalize_lot(500.0, info) == 200.0
    assert normalize_lot(0.0, info) == 0.0
    coarse = default_symbol_info(XAU, volume_min=0.1, volume_step=0.1, volume_max=50.0)
    assert normalize_lot(0.55, coarse) == 0.5
    with pytest.raises(ValueError):
        normalize_lot(1.0, info, min_lot_policy="bump")


def test_quantity_in_units_becomes_lots_and_rounding_is_reported() -> None:
    fake = _fake()
    broker = _broker(fake)
    order = run(broker.submit_order_async("EURUSD", 123_456, OrderSide.BUY))
    request = [c for c in fake.calls if c[0] == "order_send"][-1][1][0]
    assert request["volume"] == 1.23
    assert order.requested_quantity == 123_456 and order.quantity == 123_000
    assert order.status is OrderStatus.FILLED and order.filled_quantity == 123_000

    tiny = run(broker.submit_order_async("EURUSD", 400, OrderSide.BUY))
    assert tiny.status is OrderStatus.REJECTED and tiny.rejection_code == "quantity_rounds_to_zero"

    clamp = _broker(_fake(), min_lot_policy="clamp")
    bumped = run(clamp.submit_order_async("EURUSD", 400, OrderSide.BUY))
    assert bumped.status is OrderStatus.FILLED and bumped.filled_quantity == 1_000


def test_margin_guard_refuses_and_fails_closed() -> None:
    fake = _fake()
    fake.margin_per_lot[EUR] = 6_000.0  # 10_000 free margin, 20 % -> 2_000 allowed
    broker = _broker(fake, max_margin_fraction=0.2)
    order = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert order.status is OrderStatus.REJECTED
    assert order.rejection_code == "insufficient_buying_power"
    assert not [c for c in fake.calls if c[0] == "order_send"]

    loose = _broker(fake, max_margin_fraction=0.7)
    assert run(loose.submit_order_async("EURUSD", 100_000, OrderSide.BUY)).status is OrderStatus.FILLED

    # unknown margin (terminal cannot compute) is a refusal, not a pass
    fake2 = _fake()
    fake2.order_calc_margin = lambda *a, **k: None  # type: ignore[assignment]
    strict = _broker(fake2)
    unknown = run(strict.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert unknown.status is OrderStatus.REJECTED and "unknown" in unknown.rejection_reason


# ---------------------------------------------------------------------------
# order_send outcomes
# ---------------------------------------------------------------------------
def test_market_buy_request_shape_and_fill() -> None:
    fake = _fake()
    broker = _broker(fake, deviation_points=25, comment="exness_fx_d1")
    order = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    request = [c for c in fake.calls if c[0] == "order_send"][-1][1][0]
    assert request["action"] == TRADE_ACTION_DEAL
    assert request["symbol"] == EUR  # Market Watch name, never the bare one
    assert request["type"] == fake.ORDER_TYPE_BUY
    assert request["price"] == 1.10010  # ask
    assert request["deviation"] == 25
    assert request["magic"] == MAGIC
    assert request["comment"] == "exness_fx_d1"
    assert request["type_filling"] == fake.ORDER_FILLING_IOC
    assert request["type_time"] == fake.ORDER_TIME_GTC
    assert order.asset == "EURUSD" and order.status is OrderStatus.FILLED
    assert order.filled_quantity == 100_000 and order.filled_price == 1.10010
    assert order.order_id and order.filled_at is not None
    assert len(broker.fills) == 1 and broker.fills[0].slippage_points == 0.0

    sell = run(broker.submit_order_async("EURUSD", -50_000))  # side from the sign
    assert sell.side is OrderSide.SELL and sell.filled_price == 1.10000  # bid


def test_partial_fill_keeps_filled_below_requested() -> None:
    fake = _fake()
    fake.scripted_results.append({"volume": 0.6, "price": 1.10012})
    broker = _broker(fake)
    order = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == 60_000 < order.quantity == 100_000
    assert order.filled_price == 1.10012
    assert broker.fills[0].filled_lots == 0.6
    assert broker.fills[0].slippage_points == pytest.approx(2.0)  # 0.00002 / point 0.00001
    assert broker.positions["EURUSD"].quantity == 60_000


def test_rejections_map_to_stable_codes() -> None:
    fake = _fake()
    broker = _broker(fake)
    fake.scripted_results.append({"retcode": TRADE_RETCODE_NO_MONEY, "comment": "No money"})
    no_money = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert no_money.status is OrderStatus.REJECTED
    assert no_money.rejection_code == "insufficient_buying_power"
    assert "10019" in no_money.rejection_reason

    fake.scripted_results.append({"retcode": TRADE_RETCODE_MARKET_CLOSED})
    closed = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert closed.rejection_code == "market_closed"

    fake.scripted_results.append({"none": True})
    dead = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert dead.rejection_code == "connection"

    unknown = run(broker.submit_order_async("XPTUSD", 100, OrderSide.BUY))
    assert unknown.rejection_code == "unknown_symbol"
    assert broker.positions == {}


def test_requote_is_resent_once_at_the_fresh_price_then_rejected() -> None:
    fake = _fake()
    fake.scripted_results.append({"retcode": TRADE_RETCODE_REQUOTE})
    broker = _broker(fake, requote_retries=1)
    fake.quotes[EUR] = (1.10020, 1.10030)
    order = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    sends = [c[1][0] for c in fake.calls if c[0] == "order_send"]
    assert len(sends) == 2 and sends[1]["price"] == 1.10030
    assert order.status is OrderStatus.FILLED and broker.fills[-1].requote_attempts == 1

    fake.scripted_results.extend([{"retcode": TRADE_RETCODE_REQUOTE}, {"retcode": TRADE_RETCODE_REQUOTE}])
    twice = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert twice.status is OrderStatus.REJECTED and twice.rejection_code == "requote"
    assert "2 attempt" in twice.rejection_reason


# ---------------------------------------------------------------------------
# positions, pending orders, close, cancel
# ---------------------------------------------------------------------------
def test_positions_aggregate_per_bare_symbol_and_filter_magic() -> None:
    fake = _fake()
    fake.margin_per_lot[JPY] = 50.0  # the fake's default (notional / leverage) ignores the JPY quote
    broker = _broker(fake)
    run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    fake.quotes[EUR] = (1.10100, 1.10110)
    run(broker.submit_order_async("EURUSD", 50_000, OrderSide.BUY))
    jpy = run(broker.submit_order_async("USDJPY", 100_000, OrderSide.SELL))
    assert jpy.status is OrderStatus.FILLED, jpy.rejection_reason
    # a legacy bot's position on the same account, different magic: invisible to this bot
    fake.positions[1] = SimpleNamespace(
        ticket=1, identifier=1, symbol=XAU, type=fake.POSITION_TYPE_BUY, volume=0.5, price_open=2390.0,
        price_current=2400.0, profit=5.0, swap=0.0, magic=999, comment="legacy", time=1_700_000_000, time_msc=0,
    )
    positions = run(broker.get_positions_async())
    assert set(positions) == {"EURUSD", "USDJPY"}
    eur = positions["EURUSD"]
    assert eur.quantity == 150_000
    assert eur.entry_price == pytest.approx((100_000 * 1.10010 + 50_000 * 1.10110) / 150_000)
    assert eur.context["tickets"] and eur.context["market_watch"] == EUR
    assert eur.entry_time.tzinfo is not None
    assert positions["USDJPY"].quantity == -100_000 and positions["USDJPY"].side == "short"
    assert run(broker.get_position_async("USDJPY")).quantity == -100_000
    assert broker.get_position("XAUUSD") is None
    assert {p["magic"] for p in broker.all_positions()} == {MAGIC, 999}  # account-level view
    assert run(broker.get_account_value_async()) == 10_000.0
    assert run(broker.get_cash_async()) == 10_000.0


def test_close_position_sends_opposite_deal_with_the_ticket() -> None:
    fake = _fake()
    broker = _broker(fake, close_deviation_points=40)
    assert run(broker.close_position_async("EURUSD")) is None
    opened = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    run(broker.submit_order_async("EURUSD", 20_000, OrderSide.BUY))
    closed = run(broker.close_position_async("EURUSD"))
    requests = [c[1][0] for c in fake.calls if c[0] == "order_send" and "position" in c[1][0]]
    assert len(requests) == 2
    assert requests[0]["position"] == int(opened.order_id)
    assert requests[0]["type"] == fake.ORDER_TYPE_SELL and requests[0]["price"] == 1.10000  # bid
    assert requests[0]["deviation"] == 40 and requests[0]["magic"] == MAGIC
    assert closed.status is OrderStatus.FILLED and closed.side is OrderSide.SELL
    assert closed.filled_quantity == 120_000 and closed.filled_price == 1.10000
    assert broker.positions == {}
    assert [f.action for f in broker.fills] == ["open", "open", "close", "close"]

    # one leg refused -> the aggregate is REJECTED but the filled part is reported
    run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    fake.scripted_results.extend([{}, {"retcode": TRADE_RETCODE_MARKET_CLOSED}])
    half = run(broker.close_position_async("EURUSD"))
    assert half.status is OrderStatus.REJECTED and half.filled_quantity == 100_000
    assert "10018" in half.rejection_reason and broker.positions["EURUSD"].quantity == 100_000


def test_pending_limit_order_and_cancel() -> None:
    fake = _fake()
    broker = _broker(fake)
    order = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY, order_type=OrderType.LIMIT, limit_price=1.09))
    request = [c[1][0] for c in fake.calls if c[0] == "order_send"][-1]
    assert request["action"] == TRADE_ACTION_PENDING and request["type"] == fake.ORDER_TYPE_BUY_LIMIT
    assert request["price"] == 1.09
    assert order.status is OrderStatus.PENDING and order.order_id
    pending = run(broker.get_pending_orders_async())
    assert len(pending) == 1
    assert pending[0].order_type is OrderType.LIMIT and pending[0].limit_price == 1.09
    assert pending[0].quantity == 100_000 and pending[0].asset == "EURUSD"
    assert pending[0].order_id == order.order_id

    assert run(broker.cancel_order_async(order.order_id)) is True
    remove = [c[1][0] for c in fake.calls if c[0] == "order_send"][-1]
    assert remove == {"action": TRADE_ACTION_REMOVE, "order": int(order.order_id)}
    assert broker.pending_orders == []
    assert run(broker.cancel_order_async("424242")) is False

    missing = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.SELL, order_type=OrderType.LIMIT))
    assert missing.status is OrderStatus.REJECTED
    stop = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.SELL, order_type=OrderType.STOP, stop_price=1.08))
    assert stop.status is OrderStatus.PENDING and broker.pending_orders[0].stop_price == 1.08


# ---------------------------------------------------------------------------
# quote conversion and symbol mapping
# ---------------------------------------------------------------------------
def test_unit_value_in_account_currency() -> None:
    fake = _fake()
    broker = _broker(fake)
    assert broker.unit_value_account("EURUSD") == pytest.approx(1.10005)
    assert broker.unit_value_account("EURUSD", 1.2) == pytest.approx(1.2)
    assert broker.unit_value_account("USDJPY") == pytest.approx(1.0)  # one USD is one USD
    assert broker.unit_value_account("XAUUSD") == pytest.approx(2400.15)
    assert broker.quote_to_account_rate("JPY") == pytest.approx(1 / 147.01)
    with pytest.raises(MT5BrokerError):
        broker.quote_to_account_rate("CHF")  # no USDCHF quoted in this fake
    with pytest.raises(MT5BrokerError):
        broker.unit_value_account("XPTUSD")


def test_symbol_mapping_follows_the_loader_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    original = M.MT5_SYMBOL_SUFFIX
    M.set_symbol_suffix("z")
    try:
        name = "EURUSDz"
        fake = FakeMT5(symbols={name: default_symbol_info(name)}, now_utc=NOW)
        fake.account["leverage"] = 2000
        broker = _broker(fake)
        order = run(broker.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
        assert order.status is OrderStatus.FILLED
        assert [c[1][0]["symbol"] for c in fake.calls if c[0] == "order_send"] == [name]
        assert list(broker.positions) == ["EURUSD"]
    finally:
        M.set_symbol_suffix(original)


@pytest.mark.skipif(not ML4T_TYPES_AVAILABLE, reason="ml4t not installed in this environment")
def test_safebroker_wraps_the_adapter_when_ml4t_is_installed(tmp_path) -> None:
    """The real ``SafeBroker`` accepts the adapter and routes a paper order through it."""
    from ml4t.live import LiveRiskConfig, SafeBroker

    fake = _fake()
    broker = _broker(fake)
    config = LiveRiskConfig(
        execution_mode="paper",
        max_order_shares=200_000.0,
        max_order_value=300_000.0,
        max_position_shares=500_000.0,
        max_position_value=600_000.0,
        max_total_exposure=1_000_000.0,
        state_file=str(tmp_path / "risk_state.json"),
        journal_file=str(tmp_path / "risk_journal.jsonl"),
    )
    safe = SafeBroker(broker, config)
    run(safe.connect())
    safe.record_market_snapshot("EURUSD", broker.unit_value_account("EURUSD"))
    order = run(safe.submit_order_async("EURUSD", 100_000, OrderSide.BUY))
    assert order.status is OrderStatus.FILLED and order.filled_quantity == 100_000
    assert (run(safe.get_positions_async()))["EURUSD"].quantity == 100_000
    assert [c[1][0]["magic"] for c in fake.calls if c[0] == "order_send"] == [MAGIC]
