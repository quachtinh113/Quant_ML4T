"""bots/_shared/costs_mt5: swap parameters from symbol_info and the holding-cost arithmetic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import bots._shared.mt5_loader as M
from bots._shared.costs_mt5 import (
    holding_cost_points,
    is_crypto_path,
    read_swaps,
    rollover_nights,
    swap_points_to_account,
)
from bots._shared.sessions import ServerClock
from bots._shared.testing.fake_mt5 import FakeMT5, default_symbol_info

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
CLOCK = ServerClock(0, measured_at=NOW)  # the Exness server clock measured on this account: UTC
EUR = M.to_market_watch("EURUSD")
BTC = M.to_market_watch("BTCUSD")
US500 = M.to_market_watch("US500")

# 2026-09-07 is a Monday
MON = datetime(2026, 9, 7, 10, tzinfo=UTC)
TUE = datetime(2026, 9, 8, 10, tzinfo=UTC)
WED = datetime(2026, 9, 9, 10, tzinfo=UTC)
THU = datetime(2026, 9, 10, 10, tzinfo=UTC)
FRI = datetime(2026, 9, 11, 10, tzinfo=UTC)
NEXT_MON = datetime(2026, 9, 14, 10, tzinfo=UTC)


def _fake() -> FakeMT5:
    return FakeMT5(
        symbols={
            EUR: default_symbol_info(EUR, swap_long=-5.2, swap_short=1.1, swap_rollover3days=3),
            BTC: default_symbol_info(
                BTC, path=f"Standard\\Crypto\\{BTC}", swap_long=-40.0, swap_short=-20.0,
                swap_rollover3days=5, trade_contract_size=1.0, point=0.01, digits=2,
                currency_base="BTC", currency_profit="USD",
            ),
            US500: default_symbol_info(
                US500, path=f"Standard\\Indices\\{US500}", swap_long=-3.0, swap_short=-1.0,
                swap_rollover3days=5, trade_contract_size=1.0, point=0.01, digits=2,
                currency_base="USD", currency_profit="USD",
            ),
        },
        now_utc=NOW,
    )


@pytest.fixture(scope="module")
def swaps() -> dict:
    return read_swaps(["EURUSD", "BTCUSD", "US500"], _fake())


def test_read_swaps_uses_market_watch_names_and_returns_bare_keys(swaps: dict) -> None:
    assert set(swaps) == {"EURUSD", "BTCUSD", "US500"}
    eur = swaps["EURUSD"]
    assert eur["market_watch"] == EUR
    assert eur["swap_long"] == -5.2 and eur["swap_short"] == 1.1
    assert eur["swap_rollover3days"] == 3 and eur["swap_mode"] == 1
    assert eur["point"] == 0.00001 and eur["trade_contract_size"] == 100_000.0
    assert eur["charges_weekends"] is False
    assert swaps["BTCUSD"]["charges_weekends"] is True
    assert swaps["US500"]["charges_weekends"] is False
    with pytest.raises(RuntimeError):
        read_swaps(["XPTUSD"], _fake())


def test_is_crypto_path() -> None:
    assert is_crypto_path("Standard\\Crypto\\BTCUSDm")
    assert not is_crypto_path("Standard\\Forex\\EURUSDm")
    assert not is_crypto_path(None)


def test_rollover_nights_skip_weekends_and_count_the_triple_night() -> None:
    # Mon 10:00 -> Tue 10:00: one boundary (Tue 00:00 server), Monday's night
    assert rollover_nights(MON, TUE, clock=CLOCK, triple_dow=3, charges_weekends=False) == (1, 0)
    # Tue -> Thu: Tuesday night + Wednesday night (triple, MT5 dow 3)
    assert rollover_nights(TUE, THU, clock=CLOCK, triple_dow=3, charges_weekends=False) == (2, 1)
    # Fri -> Mon: Friday night charged, Saturday and Sunday nights skipped
    assert rollover_nights(FRI, NEXT_MON, clock=CLOCK, triple_dow=3, charges_weekends=False) == (1, 0)
    # crypto: every night, Friday triple (MT5 dow 5)
    assert rollover_nights(FRI, NEXT_MON, clock=CLOCK, triple_dow=5, charges_weekends=True) == (3, 1)
    # same day, exit before entry, exactly on the boundary
    assert rollover_nights(MON, MON.replace(hour=20), clock=CLOCK, triple_dow=3, charges_weekends=False) == (0, 0)
    assert rollover_nights(TUE, MON, clock=CLOCK, triple_dow=3, charges_weekends=False) == (0, 0)
    midnight = datetime(2026, 9, 8, 0, 0, tzinfo=UTC)
    assert rollover_nights(MON, midnight, clock=CLOCK, triple_dow=3, charges_weekends=False) == (1, 0)
    assert rollover_nights(midnight, TUE, clock=CLOCK, triple_dow=3, charges_weekends=False) == (0, 0)


def test_rollover_uses_the_server_day_boundary_not_utc_midnight() -> None:
    # a +3h server: its midnight is 21:00 UTC; Mon 20:00 -> Mon 22:00 UTC crosses one boundary
    clock = ServerClock(180, measured_at=NOW)
    entry = datetime(2026, 9, 7, 20, tzinfo=UTC)
    exit_ = datetime(2026, 9, 7, 22, tzinfo=UTC)
    assert rollover_nights(entry, exit_, clock=clock, triple_dow=3, charges_weekends=False) == (1, 0)
    assert rollover_nights(entry, exit_, clock=CLOCK, triple_dow=3, charges_weekends=False) == (0, 0)


def test_holding_cost_points_long_short_triple_and_crypto(swaps: dict) -> None:
    assert holding_cost_points("EURUSD", "long", MON, TUE, swaps=swaps, clock=CLOCK) == pytest.approx(-5.2)
    assert holding_cost_points("EURUSD", "short", MON, TUE, swaps=swaps, clock=CLOCK) == pytest.approx(1.1)
    assert holding_cost_points("EURUSD", "buy", TUE, THU, swaps=swaps, clock=CLOCK) == pytest.approx(-5.2 * 4)
    assert holding_cost_points("EURUSD", "sell", FRI, NEXT_MON, swaps=swaps, clock=CLOCK) == pytest.approx(1.1)
    assert holding_cost_points("EURUSD", "long", MON, MON, swaps=swaps, clock=CLOCK) == 0.0
    # crypto: Fri, Sat, Sun nights + Friday triple -> 5 x
    assert holding_cost_points("BTCUSD", "long", FRI, NEXT_MON, swaps=swaps, clock=CLOCK) == pytest.approx(-40.0 * 5)
    # index: Friday night only, and it is the triple night -> 3 x
    assert holding_cost_points("US500", "long", FRI, NEXT_MON, swaps=swaps, clock=CLOCK) == pytest.approx(-3.0 * 3)
    # explicit override of the weekend rule
    assert holding_cost_points("US500", "long", FRI, NEXT_MON, swaps=swaps, clock=CLOCK, charges_weekends=True) == pytest.approx(-3.0 * 5)
    with pytest.raises(ValueError):
        holding_cost_points("EURUSD", "flat", MON, TUE, swaps=swaps, clock=CLOCK)


def test_holding_cost_points_swap_modes(swaps: dict) -> None:
    disabled = {"EURUSD": {**swaps["EURUSD"], "swap_mode": 0}}
    assert holding_cost_points("EURUSD", "long", MON, THU, swaps=disabled, clock=CLOCK) == 0.0
    percent = {"EURUSD": {**swaps["EURUSD"], "swap_mode": 5}}
    with pytest.raises(NotImplementedError):
        holding_cost_points("EURUSD", "long", MON, THU, swaps=percent, clock=CLOCK)


def test_swap_points_to_account(swaps: dict) -> None:
    # -5.2 points x 0.00001 x 100000 x 0.5 lots = -2.6 USD (USD-quoted pair)
    assert swap_points_to_account(-5.2, swaps["EURUSD"], 0.5) == pytest.approx(-2.6)
    # JPY-quoted: multiply by the JPY -> USD rate
    assert swap_points_to_account(-5.2, swaps["EURUSD"], 0.5, quote_to_account=1 / 147.0) == pytest.approx(-2.6 / 147.0)
