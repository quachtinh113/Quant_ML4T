"""bots/_shared/costs_mt5: spread by session from ticks, against the fake terminal."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import bots._shared.mt5_loader as M
from bots._shared.costs_mt5 import (
    SESSION_BUCKETS,
    measure_spreads,
    minute_buckets,
    summarize_histogram,
    ticks_to_frame,
    write_spreads,
)
from bots._shared.sessions import ServerClock
from bots._shared.testing.fake_mt5 import FakeMT5, default_symbol_info, synthetic_ticks

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)  # a Friday, London and New York both open
CLOCK = ServerClock(0, measured_at=NOW)  # the Exness server clock: UTC+0, no DST
SPREAD = 0.00008
MID = 1.1


def _fake(spread_by_hour: dict[int, float] | None = None, days: int = 2) -> FakeMT5:
    name = M.to_market_watch("EURUSD")
    fake = FakeMT5(symbols={name: default_symbol_info(name)}, server_offset_minutes=0, now_utc=NOW)
    start = (NOW - timedelta(days=days)).replace(tzinfo=None)
    fake.ticks = {
        name: synthetic_ticks(
            start, days * 24 * 60, mid=MID, spread=SPREAD, spread_by_hour=spread_by_hour
        )
    }
    return fake


def test_ticks_to_frame_converts_server_time_and_computes_bps() -> None:
    ticks = synthetic_ticks(datetime(2026, 9, 4, 9, 0), 3, mid=MID, spread=0.00011)
    clock = ServerClock(120, measured_at=NOW)  # a +2h server puts 09:00 server at 07:00 UTC
    frame = ticks_to_frame(ticks, "EURUSD", clock)
    assert frame["timestamp"][0] == datetime(2026, 9, 4, 7, 0)
    assert abs(frame["spread_bps"][0] - 0.00011 / MID * 1e4) < 1e-6
    assert frame["symbol"].unique().to_list() == ["EURUSD"]


def test_minute_buckets_cover_every_bucket_on_a_weekday() -> None:
    buckets = minute_buckets(NOW - timedelta(days=1), NOW, CLOCK)
    seen = set(buckets["session"].unique().to_list())
    assert {"asia", "london", "overlap", "new_york", "rollover"} <= seen
    assert seen <= set(SESSION_BUCKETS)
    # 12:00 UTC on a summer Friday: London (07:00-16:00 UTC) and New York (12:00-21:00 UTC)
    at_noon = buckets.filter(pl.col("minute") == NOW.replace(tzinfo=None))["session"][0]
    assert at_noon == "overlap"
    # 00:05 UTC is within 15 minutes of server midnight, the swap time
    at_rollover = buckets.filter(pl.col("minute") == datetime(2026, 9, 4, 0, 5))["session"][0]
    assert at_rollover == "rollover"


def test_measure_spreads_reads_the_planted_spread_per_session(tmp_path: Path) -> None:
    # 14:00 UTC is overlap; plant a wide spread there and a narrow one everywhere else
    fake = _fake(spread_by_hour={14: 0.00044})
    table = measure_spreads(["EURUSD"], 2, fake, clock=CLOCK, end=NOW, chunk_days=1)
    assert fake.calls[-1][0] == "shutdown"
    rows = {r["session"]: r for r in table.to_dicts()}
    assert rows["overlap"]["spread_bps_p90"] > rows["london"]["spread_bps_p90"]
    assert abs(rows["london"]["spread_bps_p50"] - SPREAD / MID * 1e4) < 0.02
    assert rows["all"]["n_ticks"] == sum(r["n_ticks"] for k, r in rows.items() if k != "all")
    assert rows["all"]["spread_points_p50"] == 8.0  # 0.00008 / point 0.00001
    parquet, js = write_spreads(table, tmp_path)
    assert parquet.exists() and js.exists()
    payload = json.loads(js.read_text())
    assert payload["days"] == 2 and "EURUSD" in payload["spreads"]
    assert set(payload["spreads"]["EURUSD"]) == set(rows)


def test_summarize_histogram_weighted_quantiles() -> None:
    hist = pl.DataFrame(
        {
            "symbol": ["X"] * 3,
            "session": ["london"] * 3,
            "bin": [50, 100, 900],  # 0.50, 1.00, 9.00 bps
            "spread_points": [5, 10, 90],
            "n": [45, 45, 10],
        }
    )
    out = summarize_histogram(hist)
    london = out.filter(pl.col("session") == "london").to_dicts()[0]
    assert london["spread_bps_p50"] in (0.5, 1.0)
    assert london["spread_bps_p90"] == 1.0
    assert london["n_ticks"] == 100
    assert out.filter(pl.col("session") == "all").height == 1


def test_measure_spreads_refuses_unknown_symbol() -> None:
    fake = _fake()
    with pytest.raises(RuntimeError, match="not in Market Watch"):
        measure_spreads(["XXXYYY"], 1, fake, clock=CLOCK, end=NOW)
    assert not fake.initialized
