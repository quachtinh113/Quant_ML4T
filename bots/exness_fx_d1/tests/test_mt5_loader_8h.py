"""bots/_shared/mt5_loader: the derived 8h frequency (H4 bars folded onto 00/08/16 UTC)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from bots._shared import mt5_loader as M
from bots._shared.sessions import ServerClock
from bots._shared.testing.fake_mt5 import TIMEFRAME_H4, make_fake

SYMBOLS = ["BTCUSD", "EURUSD"]
START_SERVER = datetime(2024, 1, 1)  # a Monday
N_DAYS = 20
NOW = datetime(2024, 1, 15, 12, tzinfo=UTC)


def _download(tmp: Path, offset_minutes: int) -> Path:
    clock = ServerClock(offset_minutes, measured_at=NOW)
    fake = make_fake(
        SYMBOLS,
        timeframes=(TIMEFRAME_H4,),
        suffix=M.MT5_SYMBOL_SUFFIX,
        start_server=START_SERVER,
        n_days=N_DAYS,
        server_offset_minutes=offset_minutes,
        now_utc=NOW,
    )
    out = tmp / f"mt5_{offset_minutes}"
    M.download_mt5_bars(
        SYMBOLS, ("4h",), start=START_SERVER, end=START_SERVER + timedelta(days=60), clock=clock, mt5=fake, data_dir=out
    )
    return out


@pytest.fixture(scope="module")
def utc_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _download(tmp_path_factory.mktemp("data"), 0)


@pytest.fixture(scope="module")
def offset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _download(tmp_path_factory.mktemp("data"), 120)


def test_8h_bars_fold_pairs_of_4h_bars_on_the_utc_grid(utc_dir: Path) -> None:
    h4 = M.load_mt5_bars("4h", data_dir=utc_dir, include_spread=True)
    h8 = M.load_mt5_bars("8h", data_dir=utc_dir, include_spread=True)
    assert h8.columns == h4.columns
    assert h8.schema["timestamp"] == h4.schema["timestamp"]
    assert set(h8["timestamp"].dt.hour().unique().to_list()) <= {0, 8, 16}
    assert h8.group_by("symbol").len()["len"].to_list() == [N_DAYS * 3] * len(SYMBOLS)

    first = h8.filter(pl.col("symbol") == "EURUSD").sort("timestamp").row(0, named=True)
    pair = h4.filter((pl.col("symbol") == "EURUSD") & (pl.col("timestamp") < first["timestamp"] + timedelta(hours=8))).sort("timestamp")
    assert pair.height == 2
    assert first["open"] == pair["open"][0]
    assert first["close"] == pair["close"][1]
    assert first["high"] == pair["high"].max()
    assert first["low"] == pair["low"].min()
    assert first["volume"] == pair["volume"].sum()
    assert first["spread"] == pair["spread"][0]


def test_8h_respects_filters_and_symbol_listing(utc_dir: Path) -> None:
    only = M.load_mt5_bars("8h", symbols=["BTCUSD"], start_date="2024-01-08", end_date="2024-01-09", data_dir=utc_dir)
    assert only["symbol"].unique().to_list() == ["BTCUSD"]
    assert only.height == 6  # two days x three 8h bars
    assert only["timestamp"].min() == datetime(2024, 1, 8, 0, tzinfo=UTC)
    assert only["timestamp"].max() == datetime(2024, 1, 9, 16, tzinfo=UTC)
    assert M.list_mt5_symbols("8h", data_dir=utc_dir) == sorted(SYMBOLS)


def test_8h_refuses_a_misaligned_h4_grid(offset_dir: Path) -> None:
    # a +2h server prints H4 bars opening at 22:00, 02:00, 06:00 ... UTC
    with pytest.raises(ValueError, match="divisible by 4"):
        M.load_mt5_bars("8h", data_dir=offset_dir)


def test_resample_partial_buckets_and_require_complete() -> None:
    ts = [datetime(2024, 1, 7, 20, tzinfo=UTC), datetime(2024, 1, 8, 0, tzinfo=UTC), datetime(2024, 1, 8, 4, tzinfo=UTC)]
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series(ts, dtype=pl.Datetime("us", "UTC")),
            "symbol": ["EURUSD"] * 3,
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.02, 1.12, 1.22],
            "volume": [10, 20, 30],
        }
    )
    kept = M.resample_4h_to_8h(frame)
    assert kept.height == 2 and kept["timestamp"][0] == datetime(2024, 1, 7, 16, tzinfo=UTC)
    complete = M.resample_4h_to_8h(frame, require_complete=True)
    assert complete.height == 1
    assert complete.row(0, named=True) == {
        "timestamp": datetime(2024, 1, 8, 0, tzinfo=UTC), "symbol": "EURUSD",
        "open": 1.1, "high": 1.25, "low": 1.05, "close": 1.22, "volume": 50,
    }
    assert M.resample_4h_to_8h(frame.head(0)).height == 0


def test_8h_cannot_be_downloaded(tmp_path: Path) -> None:
    fake = make_fake(["EURUSD"], suffix=M.MT5_SYMBOL_SUFFIX, n_days=2, now_utc=NOW)
    with pytest.raises(ValueError, match="cannot be downloaded"):
        M.download_mt5_bars(["EURUSD"], ("8h",), mt5=fake, data_dir=tmp_path)
    with pytest.raises(ValueError):
        M.load_mt5_bars("2h", data_dir=tmp_path)
