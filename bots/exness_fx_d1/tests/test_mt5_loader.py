"""bots/_shared/mt5_loader.py against a fake MetaTrader5 module and an OANDA-shaped fixture.

No terminal, no network: ``uv run pytest bots/exness_fx_d1/tests -q``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import data.fx.loader as fx_loader
from bots._shared import mt5_loader as M
from bots._shared.sessions import ServerClock
from bots._shared.testing.fake_mt5 import TIMEFRAME_D1, TIMEFRAME_H4, make_fake
from data.exceptions import DataNotFoundError, MissingDependencyError
from data.fx.loader import load_fx_pairs

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
START_SERVER = datetime(2024, 1, 1)  # a Monday on the server clock
N_DAYS = 60
ORIGINAL_SUFFIX = M.MT5_SYMBOL_SUFFIX  # "m" on this account; tests never depend on its value
NOW = datetime(2024, 1, 15, 12, tzinfo=UTC)  # winter: the +120 reading anchors the DST rule


@pytest.fixture(scope="module")
def clock() -> ServerClock:
    return ServerClock(120, measured_at=NOW, follows_dst_of="America/New_York")


@pytest.fixture(scope="module")
def mt5_dir(tmp_path_factory: pytest.TempPathFactory, clock: ServerClock) -> Path:
    """Download through the fake terminal once per module."""
    out = tmp_path_factory.mktemp("data") / "mt5"
    fake = make_fake(SYMBOLS, suffix=M.MT5_SYMBOL_SUFFIX, start_server=START_SERVER, n_days=N_DAYS, server_offset_minutes=120, now_utc=NOW)
    written = M.download_mt5_bars(
        SYMBOLS,
        ("daily", "4h", "1h"),
        start=START_SERVER,
        end=START_SERVER + timedelta(days=120),
        clock=clock,
        mt5=fake,
        data_dir=out,
    )
    assert set(written) == {"daily", "4h", "1h", "history_depth"}
    assert fake.calls[-1][0] == "shutdown"
    assert not fake.initialized
    return out


@pytest.fixture(scope="module")
def fx_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An OANDA-shaped ``fx/market/{daily,4h}.parquet`` pair, the shape ``load_fx_pairs`` reads."""
    root = tmp_path_factory.mktemp("oanda")
    market = root / "fx" / "market"
    market.mkdir(parents=True)
    days = [datetime(2024, 1, 1, 22, tzinfo=UTC) + timedelta(days=i) for i in range(80)]
    days = [d for d in days if d.weekday() < 5]
    rows = {
        "timestamp": [d for d in days for _ in ("EUR_USD", "GBP_USD")],
        "symbol": ["EUR_USD", "GBP_USD"] * len(days),
        "open": [1.1] * 2 * len(days),
        "high": [1.11] * 2 * len(days),
        "low": [1.09] * 2 * len(days),
        "close": [1.1] * 2 * len(days),
        "volume": [1000] * 2 * len(days),
    }
    daily = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        pl.col("volume").cast(pl.Int64),
    )
    daily.write_parquet(market / "daily.parquet")
    h4 = daily.with_columns(pl.col("timestamp") - pl.duration(hours=4))
    pl.concat([daily, h4]).sort(["symbol", "timestamp"]).write_parquet(market / "4h.parquet")
    return root


@pytest.fixture
def oanda(monkeypatch: pytest.MonkeyPatch, fx_dir: Path) -> None:
    monkeypatch.setattr(fx_loader, "ML4T_DATA_PATH", fx_dir)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def test_parquet_shape_and_symbol_convention(mt5_dir: Path) -> None:
    daily = pl.read_parquet(mt5_dir / "daily.parquet")
    assert daily.columns == M.PARQUET_COLUMNS
    assert daily.schema["timestamp"] == pl.Datetime("us", "UTC")
    assert daily.schema["server_time"] == pl.Datetime("us")
    assert sorted(daily["symbol"].unique()) == sorted(SYMBOLS)  # bare names, never the suffix
    assert daily.group_by("symbol").len()["len"].to_list() == [N_DAYS] * len(SYMBOLS)
    # server 2024-01-01 00:00 with a +120 clock is 2023-12-31 22:00 UTC
    first = daily.filter(pl.col("symbol") == "EURUSD").sort("timestamp").row(0, named=True)
    assert first["server_time"] == datetime(2024, 1, 1)
    assert first["timestamp"] == datetime(2023, 12, 31, 22, tzinfo=UTC)
    assert daily["server_time"].dt.weekday().max() <= 5  # no weekend server days


def test_history_depth_records_what_the_stages_and_adapter_read(mt5_dir: Path, clock: ServerClock) -> None:
    depth = M.read_history_depth(mt5_dir)
    assert ServerClock.from_dict(depth["server_clock"]) == clock
    assert depth["account"]["trade_mode"] == 0  # demo
    assert set(depth["symbols"]) == set(SYMBOLS)
    info = depth["symbols"]["USDJPY"]
    assert info["market_watch"] == f"USDJPY{M.MT5_SYMBOL_SUFFIX}"
    assert info["trade_contract_size"] == 100_000.0
    assert info["digits"] == 3 and info["swap_rollover3days"] == 3
    for field in ("volume_min", "volume_step", "volume_max", "swap_long", "swap_short", "point"):
        assert field in info
    d1 = depth["timeframes"]["daily"]["EURUSD"]
    assert d1["n_bars"] == N_DAYS
    assert d1["first_server_day"] == "2024-01-01"
    assert depth["timeframes"]["4h"]["EURUSD"]["n_bars"] == N_DAYS * 6
    assert depth["timeframes"]["1h"]["EURUSD"]["n_bars"] == N_DAYS * 24


def test_download_uses_market_watch_names_and_lazy_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: ServerClock) -> None:
    monkeypatch.setattr(M, "MT5_SYMBOL_SUFFIX", "m")
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)  # importing it would fail loudly
    M.set_symbol_suffix("m")
    try:
        assert M.to_market_watch("EURUSD") == "EURUSDm"
        assert M.to_bare("EURUSDm") == "EURUSD"
        assert M.to_bare("XPTUSDm") == "XPTUSD"  # unlisted instrument, suffix stripped
        fake = make_fake(["EURUSD"], suffix="m", n_days=5, now_utc=NOW)
        M.download_mt5_bars(["EURUSD"], ("daily",), start=START_SERVER, end=START_SERVER + timedelta(days=30), clock=clock, mt5=fake, data_dir=tmp_path)
        requested = [c[1][0] for c in fake.calls if c[0] == "copy_rates_range"]
        assert requested == ["EURUSDm"]
        frame = pl.read_parquet(tmp_path / "daily.parquet")
        assert frame["symbol"].unique().to_list() == ["EURUSD"]
        assert json.loads((tmp_path / "history_depth.json").read_text())["symbol_suffix"] == "m"
        with pytest.raises(MissingDependencyError):
            M.download_mt5_bars(["EURUSD"], ("daily",), data_dir=tmp_path)  # no mt5= -> lazy import
    finally:
        M.set_symbol_suffix(ORIGINAL_SUFFIX)


def test_download_refuses_unknown_symbol_and_failed_initialize(tmp_path: Path, clock: ServerClock) -> None:
    fake = make_fake(["EURUSD"], suffix=M.MT5_SYMBOL_SUFFIX, n_days=3, now_utc=NOW)
    with pytest.raises(RuntimeError, match="not in Market Watch"):
        M.download_mt5_bars(["EURUSD", "XXXYYY"], ("daily",), clock=clock, mt5=fake, data_dir=tmp_path)
    assert not fake.initialized  # shutdown ran in finally
    broken = make_fake(["EURUSD"], suffix=M.MT5_SYMBOL_SUFFIX, n_days=3, fail_initialize=True)
    with pytest.raises(RuntimeError, match="initialize"):
        M.download_mt5_bars(["EURUSD"], ("daily",), clock=clock, mt5=broken, data_dir=tmp_path)


def test_download_measures_the_clock_when_none_is_given(tmp_path: Path) -> None:
    fake = make_fake(["EURUSD"], suffix=M.MT5_SYMBOL_SUFFIX, n_days=3, server_offset_minutes=180)  # tick time is "now"
    M.download_mt5_bars(["EURUSD"], ("daily",), start=START_SERVER, end=START_SERVER + timedelta(days=10), mt5=fake, data_dir=tmp_path)
    depth = M.read_history_depth(tmp_path)
    assert depth["server_clock"]["utc_offset_minutes"] == 180


# ---------------------------------------------------------------------------
# Load: same shape as load_fx_pairs
# ---------------------------------------------------------------------------
def test_daily_schema_equals_load_fx_pairs(mt5_dir: Path, oanda: None) -> None:
    mine = M.load_mt5_bars("daily", data_dir=mt5_dir)
    theirs = load_fx_pairs("daily")
    assert mine.schema == theirs.schema
    assert mine.schema["timestamp"] == pl.Date
    assert mine.columns == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    with_spread = M.load_mt5_bars("daily", data_dir=mt5_dir, include_spread=True)
    assert with_spread.columns[-1] == "spread" and with_spread.drop("spread").schema == theirs.schema


def test_intraday_schema_matches_load_fx_pairs_columns(mt5_dir: Path, oanda: None) -> None:
    mine = M.load_mt5_bars("4h", data_dir=mt5_dir)
    theirs = load_fx_pairs("4h")
    assert mine.columns == theirs.columns
    assert isinstance(mine.schema["timestamp"], pl.Datetime) and isinstance(theirs.schema["timestamp"], pl.Datetime)
    assert mine.schema["timestamp"].time_zone == "UTC"
    assert {c: mine.schema[c] for c in mine.columns if c != "timestamp"} == {c: theirs.schema[c] for c in theirs.columns if c != "timestamp"}


def test_date_filters_behave_like_load_fx_pairs(mt5_dir: Path, oanda: None) -> None:
    mine = M.load_mt5_bars("daily", start_date="2024-01-10", end_date="2024-01-20", data_dir=mt5_dir)
    theirs = load_fx_pairs("daily", start_date="2024-01-10", end_date="2024-01-20")
    # both inclusive on both ends, both on a Date column
    assert mine["timestamp"].min() == date(2024, 1, 10) and mine["timestamp"].max() == date(2024, 1, 19)
    assert theirs["timestamp"].min() >= date(2024, 1, 10) and theirs["timestamp"].max() <= date(2024, 1, 20)
    assert mine["timestamp"].dt.weekday().max() <= 5
    # intraday: end_date keeps the whole day, like load_fx_pairs
    h4 = M.load_mt5_bars("4h", start_date="2024-01-10", end_date="2024-01-10", data_dir=mt5_dir)
    assert h4["timestamp"].min() >= datetime(2024, 1, 10, tzinfo=UTC)
    assert h4["timestamp"].max() < datetime(2024, 1, 11, tzinfo=UTC)
    assert h4.filter(pl.col("symbol") == "EURUSD").height == 6
    ref = load_fx_pairs("4h", start_date="2024-01-10", end_date="2024-01-10")
    assert ref["timestamp"].max() < datetime(2024, 1, 11, tzinfo=UTC)


def test_symbol_filters_and_reduction(mt5_dir: Path) -> None:
    two = M.load_mt5_bars("daily", symbols=["EURUSD", "USDJPY"], data_dir=mt5_dir)
    assert two["symbol"].unique().sort().to_list() == ["EURUSD", "USDJPY"]
    reduced = M.load_mt5_bars("daily", max_symbols=2, data_dir=mt5_dir)
    assert reduced["symbol"].n_unique() == 2  # the repo's top_entities rule, ties by name
    assert reduced["symbol"].unique().sort().to_list() == ["AUDUSD", "EURUSD"]
    assert M.list_mt5_symbols("daily", data_dir=mt5_dir) == sorted(SYMBOLS)


def test_missing_parquet_raises_data_not_found(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError):
        M.load_mt5_bars("daily", data_dir=tmp_path)
    with pytest.raises(ValueError):
        M.load_mt5_bars("weekly", data_dir=tmp_path)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Server day -> UTC for the session-alignment checks
# ---------------------------------------------------------------------------
def test_server_day_open_and_close(mt5_dir: Path, clock: ServerClock) -> None:
    daily = M.load_mt5_bars("daily", data_dir=mt5_dir)
    utc = M.server_day_open_utc(daily, clock)
    assert utc.columns == daily.columns
    assert utc.schema["timestamp"] == pl.Datetime("us")
    row = utc.filter((pl.col("symbol") == "EURUSD")).sort("timestamp").row(0, named=True)
    assert row["timestamp"] == datetime(2023, 12, 31, 22)  # +120 in winter
    assert M.server_day_close_utc(date(2024, 1, 15), clock) == datetime(2024, 1, 15, 22, tzinfo=UTC)
    assert M.server_day_close_utc(date(2024, 3, 20), clock) == datetime(2024, 3, 20, 21, tzinfo=UTC)  # +180 after US DST
    with pytest.raises(TypeError):
        M.server_day_open_utc(utc, clock)
