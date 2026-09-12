"""bots/_shared/monitor: the account tier, against the fake MetaTrader5 module.

Five breakers, one halt file, no terminal and no network. Each breaker gets a test that makes
it trip on a synthetic account reading, which is the pattern of the simulation in
``26_mlops_governance/04_circuit_breakers.py`` section 4.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import yaml

from bots._shared import MAGIC_ALLOCATION, RESERVED_MAGICS, assert_magics_unique, magic_for
from bots._shared.monitor import (
    AccountGuard,
    AccountSnapshot,
    BreakerState,
    load_account_limits,
    read_account_snapshot,
)
from bots._shared.monitor.account import LIMITS_PATH
from bots._shared.mt5_broker import MT5Broker
from bots._shared.mt5_loader import to_market_watch
from bots._shared.testing.fake_mt5 import FakeMT5, default_symbol_info, make_fake

MAGIC = MAGIC_ALLOCATION["exness_fx_d1"]
NOW = datetime(2026, 9, 7, 20, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def snapshot(**overrides) -> AccountSnapshot:
    base = dict(
        read_at=NOW,
        login=12345678,
        server="Exness-MT5Trial7",
        trade_mode=0,
        currency="USD",
        balance=10_000.0,
        equity=10_000.0,
        margin=0.0,
        margin_free=10_000.0,
        margin_so_call=60.0,
        margin_so_so=50.0,
        n_positions=0,
        n_pending_orders=0,
        gross_notional=0.0,
        positions_by_magic={},
    )
    base.update(overrides)
    return AccountSnapshot(**base)


@pytest.fixture
def guard(tmp_path):
    return AccountGuard(
        load_account_limits(),
        halt_file=tmp_path / "account_halt.json",
        hwm_file=tmp_path / "account_hwm.json",
        bot_id="exness_fx_d1",
    )


def only(guard: AccountGuard, name: str) -> AccountGuard:
    """Keep one breaker registered.

    The five account breakers all read the same snapshot, so a reading chosen to trip one of
    them usually trips another (a 5 % intraday fall is also a 5 % drawdown). Isolating the
    breaker under test is what makes the assertion about *that* breaker.
    """
    for other in list(guard.manager.breakers):
        if other != name:
            guard.manager.breakers.pop(other)
    return guard


# ---------------------------------------------------------------------------------------
# the magic allocation table
# ---------------------------------------------------------------------------------------
class TestTheMagicAllocationTable:
    def test_every_bot_has_a_distinct_magic_and_none_is_reserved(self):
        assert_magics_unique()
        assert len(set(MAGIC_ALLOCATION.values())) == len(MAGIC_ALLOCATION)
        assert 202500 in RESERVED_MAGICS  # the retired legacy bot, never reallocated
        assert 202500 not in MAGIC_ALLOCATION.values()

    def test_this_bot_is_260901(self):
        assert magic_for("exness_fx_d1") == 260901

    def test_an_unallocated_bot_raises_instead_of_inventing_a_number(self):
        with pytest.raises(KeyError, match="MAGIC_ALLOCATION"):
            magic_for("not_a_bot")


# ---------------------------------------------------------------------------------------
# the limits file
# ---------------------------------------------------------------------------------------
class TestTheAccountLimitsAreTheStrictestOnTheAccount:
    def test_drawdown_and_daily_loss_match_the_strictest_declared_bot(self):
        limits = load_account_limits()
        # bots/xau_fx_mt5/BOT.md declares 6 % drawdown and 2 % daily loss, the strictest on
        # this login; the account tier must not be looser than any bot it protects.
        assert limits.max_drawdown_pct == pytest.approx(0.06)
        assert limits.max_daily_equity_loss_pct == pytest.approx(0.02)

    def test_the_file_parses_into_every_declared_field(self):
        raw = yaml.safe_load(LIMITS_PATH.read_text())
        limits = load_account_limits()
        assert limits.min_margin_level_pct == float(raw["margin"]["min_margin_level_pct"])
        assert limits.max_open_positions == int(raw["exposure"]["max_open_positions"])
        assert limits.source.endswith("account_limits.yaml")


# ---------------------------------------------------------------------------------------
# reading the account through the adapter
# ---------------------------------------------------------------------------------------
class TestReadingTheAccountThroughTheAdapter:
    def test_snapshot_counts_positions_of_every_magic_and_sums_gross_notional(self):
        fake = make_fake(["EURUSDm"], n_days=2, server_offset_minutes=0)
        fake.initialize()
        fake.positions = {
            1: SimpleNamespace(
                ticket=1, symbol="EURUSDm", magic=MAGIC, type=0, volume=0.10,
                price_open=1.16, price_current=1.16, profit=0.0, swap=0.0, comment="",
                time=0, time_msc=0,
            ),
            2: SimpleNamespace(
                ticket=2, symbol="EURUSDm", magic=260904, type=1, volume=0.20,
                price_open=1.16, price_current=1.16, profit=0.0, swap=0.0, comment="",
                time=0, time_msc=0,
            ),
        }
        broker = MT5Broker(magic=MAGIC, mt5=fake)
        snap = read_account_snapshot(broker)
        assert snap.n_positions == 2
        assert snap.positions_by_magic == {str(MAGIC): 1, "260904": 1}
        # 0.30 lots x 100,000 x 1.16
        assert snap.gross_notional == pytest.approx(0.30 * 100_000 * 1.16, rel=1e-9)

    def test_margin_level_is_none_when_nothing_is_open(self):
        assert snapshot().margin_level_pct is None
        assert snapshot(margin=1_000.0, equity=5_000.0).margin_level_pct == pytest.approx(500.0)


# ---------------------------------------------------------------------------------------
# one test per breaker: make it trip
# ---------------------------------------------------------------------------------------
class TestEachAccountBreakerTrips:
    def test_drawdown_breaker_trips_after_a_six_percent_fall_from_the_peak(self, guard):
        only(guard, "account_drawdown")
        assert guard.check(snapshot(equity=10_000.0)) is True
        assert guard.breakers.drawdown.peak_equity == pytest.approx(10_000.0)
        assert guard.check(snapshot(equity=9_500.0)) is True  # -5 %, still closed
        assert guard.check(snapshot(equity=9_390.0)) is False  # -6.1 %
        assert guard.breakers.drawdown.state is BreakerState.OPEN
        assert guard.is_halted()

    def test_the_peak_survives_a_restart_because_it_is_persisted(self, tmp_path):
        first = AccountGuard(
            load_account_limits(),
            halt_file=tmp_path / "halt.json",
            hwm_file=tmp_path / "hwm.json",
            bot_id="t",
        )
        only(first, "account_drawdown").check(snapshot(equity=12_000.0))
        assert (tmp_path / "hwm.json").exists()
        second = AccountGuard(
            load_account_limits(),
            halt_file=tmp_path / "halt.json",
            hwm_file=tmp_path / "hwm.json",
            bot_id="t",
        )
        assert second.breakers.drawdown.peak_equity == pytest.approx(12_000.0)
        # 11,000 is -8.3 % from the persisted peak even though it is above the reference.
        assert only(second, "account_drawdown").check(snapshot(equity=11_000.0)) is False

    def test_daily_loss_breaker_trips_at_two_percent_of_the_days_opening_equity(self, guard):
        only(guard, "account_daily_loss")
        opening = snapshot(equity=10_000.0)
        guard.start_day(opening)
        assert guard.check(snapshot(equity=9_900.0)) is True  # -1 %
        assert guard.check(snapshot(equity=9_790.0)) is False  # -2.1 %
        assert guard.breakers.daily_loss.state is BreakerState.OPEN

    def test_margin_level_breaker_trips_below_three_hundred_percent(self, guard):
        only(guard, "account_margin")
        assert guard.check(snapshot(equity=10_000.0, margin=1_000.0)) is True  # 1000 %
        assert guard.check(snapshot(equity=10_000.0, margin=4_000.0)) is False  # 250 %
        assert guard.breakers.margin.state is BreakerState.OPEN

    def test_stop_out_breaker_trips_at_twice_the_brokers_stop_out_level(self, guard):
        # margin_so_so 50 % -> threshold 100 %. 120 % is fine, 90 % is not.
        only(guard, "account_stop_out")
        assert guard.check(snapshot(equity=12_000.0, margin=10_000.0)) is True  # 120 %
        assert guard.check(snapshot(equity=9_000.0, margin=10_000.0)) is False  # 90 %
        assert guard.breakers.stop_out.state is BreakerState.OPEN

    def test_an_unreported_stop_out_level_falls_back_instead_of_reading_as_infinite(self, guard):
        only(guard, "account_stop_out")
        # fallback 50 % x 2 = 100 %; a terminal that reports nothing must not look safe.
        assert guard.check(snapshot(equity=9_000.0, margin=10_000.0, margin_so_so=None)) is False
        assert guard.breakers.stop_out.state is BreakerState.OPEN

    def test_exposure_breaker_trips_on_too_many_positions_across_every_magic(self, guard):
        only(guard, "account_exposure")
        assert guard.check(snapshot(n_positions=25, positions_by_magic={"260901": 25})) is False
        assert guard.breakers.exposure.state is BreakerState.OPEN
        assert "25 open positions" in guard.halt_record()["reason"]

    def test_exposure_breaker_trips_on_gross_notional(self, guard):
        only(guard, "account_exposure")
        assert guard.check(snapshot(gross_notional=250_000.0)) is False
        assert guard.breakers.exposure.state is BreakerState.OPEN


# ---------------------------------------------------------------------------------------
# the halt file: one trip stops every bot
# ---------------------------------------------------------------------------------------
class TestTheHaltFileStopsEveryBot:
    def test_a_trip_writes_a_halt_record_another_bot_can_read(self, tmp_path):
        fx = AccountGuard(
            load_account_limits(), halt_file=tmp_path / "halt.json", hwm_file=tmp_path / "hwm.json", bot_id="exness_fx_d1"
        )
        btc = AccountGuard(
            load_account_limits(), halt_file=tmp_path / "halt.json", hwm_file=tmp_path / "hwm.json", bot_id="exness_btc_8h"
        )
        assert not btc.is_halted()
        fx.check(snapshot(n_positions=99))
        assert btc.is_halted()
        record = btc.halt_record()
        assert record["raised_by_bot"] == "exness_fx_d1"
        assert "account_exposure" in record["open_breakers"]

    def test_clearing_needs_a_reason_and_a_name_and_archives_the_record(self, guard, tmp_path):
        guard.check(snapshot(n_positions=99))
        with pytest.raises(ValueError, match="reason and a name"):
            guard.clear_halt(reason="", cleared_by="")
        guard.clear_halt(reason="closed the stray positions by hand", cleared_by="operator")
        assert not guard.is_halted()
        archives = list(tmp_path.glob("account_halt.cleared-*.json"))
        assert len(archives) == 1
        archived = json.loads(archives[0].read_text())
        assert archived["cleared_by"] == "operator"
        assert guard.manager.allows_trading()

    def test_an_unreadable_halt_file_still_halts(self, guard):
        guard.halt_file.parent.mkdir(parents=True, exist_ok=True)
        guard.halt_file.write_text("{not json")
        assert guard.is_halted()
        assert guard.halt_record()["reason"] == "halt file unreadable"


class TestTheAccountIdentityIsVerified:
    def test_a_declared_login_that_does_not_match_raises(self, tmp_path):
        raw = yaml.safe_load(LIMITS_PATH.read_text())
        raw["account"]["expected_login"] = 999
        path = tmp_path / "limits.yaml"
        path.write_text(yaml.safe_dump(raw))
        guard = AccountGuard(
            load_account_limits(path), halt_file=tmp_path / "halt.json", hwm_file=tmp_path / "hwm.json"
        )
        with pytest.raises(ValueError, match="do not describe this account"):
            guard.check(snapshot(login=12345678))


def test_symbol_names_reaching_the_terminal_carry_the_market_watch_suffix():
    """The account tier reads the terminal, so it must speak the terminal's names."""
    fake = FakeMT5({}, symbols={to_market_watch("EURUSD"): default_symbol_info(to_market_watch("EURUSD"))})
    fake.initialize()
    broker = MT5Broker(magic=MAGIC, mt5=fake)
    assert broker.symbol_info("EURUSD") is not None
