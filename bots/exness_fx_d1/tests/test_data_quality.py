"""Data-quality checks on the MT5 four-hour history of exness_fx_d1 (roadmap phase 2).

Modelled on ``02_financial_data_universe/13_data_quality_framework.py`` (OHLC invariants, gap
detection, deduplication, outlier flags) and applied to ``ML4T_DATA_PATH/mt5/4h.parquet`` for
the five pairs of ``setup.yaml``, on the development history only (the holdout window is never
read). Hard invariants are asserted; the rest is printed as findings for ``BOT.md`` (run with
``-s`` to see them):

- keys unique, every bar on the four-hour UTC grid;
- ``high >= max(open, close)``, ``low <= min(open, close)``, ``volume >= 0``, no zero-volume bar;
- the bar grid against the trading hours ``sessions_mt5.json`` derived from the terminal:
  whole missing days (holidays), partially missing days (outages such as 2018-01-31 16:00),
  bars printed outside the published hours;
- ``spread`` (points at the bar open) against the p90 the tick measurement recorded in
  ``spreads_by_session.json``: outliers overall and on the decision bars;
- flat bars, four-hour return outliers, and the parquet against ``history_depth.json``.

Skipped when the MT5 history is not downloaded.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_fx_d1/tests/test_data_quality.py -q -s
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta

import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import mt5_data_dir, read_history_depth
from utils.data_quality import check_ohlc_invariants
from utils.paths import REPO_ROOT

CASE_STUDY_ID = "exness_fx_d1"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
BAR_HOURS = 4
SLOT_HOURS = list(range(0, 24, BAR_HOURS))
WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
SPREAD_OUTLIER_MULTIPLE = 5.0  # x the measured in-session p90, in points
RETURN_OUTLIER = 0.02  # a four-hour move larger than this is listed
RETURN_ERROR = 0.10  # ... and larger than this is a data error


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def window(setup: dict) -> tuple[date, date]:
    start = date.fromisoformat(str(setup["universe"]["history_start"]))
    dev_end = date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1)
    return start, dev_end


@pytest.fixture(scope="module")
def bars(setup: dict, window: tuple[date, date]) -> pl.DataFrame:
    path = mt5_data_dir() / "4h.parquet"
    if not path.exists():
        pytest.skip(f"MT5 history not downloaded: {path}")
    start, dev_end = window
    return (
        pl.read_parquet(path)
        .filter(pl.col("symbol").is_in(sorted(setup["universe"]["symbols"])))
        .with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
        .filter(
            (pl.col("timestamp") >= datetime.combine(start, time(0)))
            & (pl.col("timestamp") < datetime.combine(dev_end + timedelta(days=1), time(0)))
        )
        .sort(["symbol", "timestamp"])
    )


@pytest.fixture(scope="module")
def trade_hours() -> dict:
    path = mt5_data_dir() / "sessions_mt5.json"
    if not path.exists():
        pytest.skip(f"trade hours not recorded: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spreads() -> dict:
    path = mt5_data_dir() / "spreads_by_session.json"
    if not path.exists():
        pytest.skip(f"spread table not recorded: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Keys, grid, invariants
# ---------------------------------------------------------------------------
def test_keys_unique_and_on_the_four_hour_grid(bars: pl.DataFrame) -> None:
    dupes = bars.select(["symbol", "timestamp"]).is_duplicated().sum()
    off_grid = bars.filter((pl.col("timestamp").dt.hour() % BAR_HOURS != 0) | (pl.col("timestamp").dt.minute() != 0))
    print(f"\n{bars.height:,} bars, {bars['symbol'].n_unique()} pairs, {bars['timestamp'].min()} .. {bars['timestamp'].max()}")
    print(f"duplicate keys: {dupes}; bars off the 4h UTC grid: {off_grid.height}")
    assert dupes == 0
    assert off_grid.height == 0


def test_ohlc_invariants_hold(bars: pl.DataFrame) -> None:
    table = check_ohlc_invariants(bars)
    print("\n", table)
    assert (table["valid_pct"] == 100.0).all(), table
    assert bars.filter(pl.col("close") <= 0).height == 0


def test_no_zero_volume_bars(bars: pl.DataFrame) -> None:
    zero = bars.filter(pl.col("volume") <= 0)
    print(f"\nzero-volume bars: {zero.height}")
    assert zero.height == 0


# ---------------------------------------------------------------------------
# Gaps against the published trading hours
# ---------------------------------------------------------------------------
def _expected_slots(hours: dict, start: date, end: date) -> set[datetime]:
    """Every 4h slot whose [open, open+4h) overlaps a trading window, on the server clock (UTC+0)."""

    def to_minutes(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    slots: set[datetime] = set()
    day = start
    while day <= end:
        windows = hours.get(WEEKDAY_KEYS[day.weekday()], [])
        for hour in SLOT_HOURS:
            lo, hi = hour * 60, (hour + BAR_HOURS) * 60
            if any(to_minutes(a) < hi and to_minutes(b) > lo for a, b in windows):
                slots.add(datetime.combine(day, time(hour)))
        day += timedelta(days=1)
    return slots


def test_bar_grid_against_trading_hours(bars: pl.DataFrame, trade_hours: dict, window: tuple[date, date]) -> None:
    """Whole missing days are holidays; partially missing days are outages and must be rare."""
    start, end = window
    partial_days: dict[str, list[str]] = {}
    whole_days: dict[str, list[str]] = {}
    extra: dict[str, int] = {}
    for symbol in sorted(bars["symbol"].unique().to_list()):
        hours = trade_hours["symbols"][symbol]["trade_hours_server_time"]
        expected = _expected_slots(hours, start, end)
        actual = set(bars.filter(pl.col("symbol") == symbol)["timestamp"].to_list())
        missing = expected - actual
        by_day: dict[date, int] = {}
        for ts in missing:
            by_day[ts.date()] = by_day.get(ts.date(), 0) + 1
        expected_per_day = {d: sum(1 for ts in expected if ts.date() == d) for d in by_day}
        whole_days[symbol] = sorted(str(d) for d, n in by_day.items() if n == expected_per_day[d])
        partial_days[symbol] = sorted(
            f"{d} ({n} of {expected_per_day[d]} slots)" for d, n in by_day.items() if n < expected_per_day[d]
        )
        extra[symbol] = len(actual - expected)
        print(
            f"\n{symbol}: expected {len(expected):,} slots, present {len(actual & expected):,}, "
            f"whole days missing {len(whole_days[symbol])}, partial days {len(partial_days[symbol])}, "
            f"bars outside published hours {extra[symbol]}"
        )
        print(f"  whole days missing: {whole_days[symbol]}")
        print(f"  partial days: {partial_days[symbol]}")
    n_days = sum(1 for i in range((end - start).days + 1) if (start + timedelta(days=i)).weekday() < 5)
    for symbol, days in partial_days.items():
        assert len(days) <= 0.01 * n_days, f"{symbol}: {len(days)} partially missing days of {n_days}"
    for symbol, n in extra.items():
        assert n <= 0.01 * n_days, f"{symbol}: {n} bars printed outside the published hours"


def test_history_depth_matches_parquet(setup: dict) -> None:
    depth = read_history_depth()
    full = pl.read_parquet(mt5_data_dir() / "4h.parquet")
    for symbol in sorted(setup["universe"]["symbols"]):
        recorded = depth["timeframes"]["4h"][symbol]
        rows = full.filter(pl.col("symbol") == symbol)
        assert rows.height == recorded["n_bars"], (symbol, rows.height, recorded["n_bars"])
        assert rows["timestamp"].min().isoformat() == recorded["first"], symbol
        assert rows["timestamp"].max().isoformat() == recorded["last"], symbol


# ---------------------------------------------------------------------------
# Spreads, flat bars, return outliers
# ---------------------------------------------------------------------------
def test_spread_outliers_against_measured_p90(bars: pl.DataFrame, spreads: dict, setup: dict) -> None:
    """Bars whose opening spread exceeds a multiple of the measured in-session p90, in points."""
    snapshot_hour = int(setup["decision"]["snapshot_utc"].split(":")[0])
    decision_open = snapshot_hour - BAR_HOURS
    rows = []
    for symbol in sorted(bars["symbol"].unique().to_list()):
        table = spreads["spreads"][symbol]
        p90_points = max(
            table[bucket]["spread_points_p90"] for bucket in ("asia", "london", "overlap", "new_york", "rollover")
        )
        sym = bars.filter(pl.col("symbol") == symbol)
        flagged = sym.filter(pl.col("spread") > SPREAD_OUTLIER_MULTIPLE * p90_points)
        on_decision = sym.filter(pl.col("timestamp").dt.hour() == decision_open)
        flagged_decision = on_decision.filter(pl.col("spread") > SPREAD_OUTLIER_MULTIPLE * p90_points)
        by_hour = flagged.group_by(pl.col("timestamp").dt.hour().alias("h")).len().sort("h")
        by_wd = flagged.group_by(pl.col("timestamp").dt.weekday().alias("wd")).len().sort("wd")
        rows.append(
            {
                "symbol": symbol,
                "p90_points": p90_points,
                "median_points": float(sym["spread"].median()),
                "flagged": flagged.height,
                "flagged_share": flagged.height / sym.height,
                "decision_bars": on_decision.height,
                "decision_flagged": flagged_decision.height,
                "by_hour": dict(zip(by_hour["h"].to_list(), by_hour["len"].to_list(), strict=True)),
                "by_weekday": dict(zip(by_wd["wd"].to_list(), by_wd["len"].to_list(), strict=True)),
                "max_points": int(sym["spread"].max()),
                "max_at": sym.sort("spread", descending=True)["timestamp"][0],
            }
        )
    print("\nspread outliers (> %.0fx measured in-session p90 points):" % SPREAD_OUTLIER_MULTIPLE)
    for r in rows:
        print(
            f"  {r['symbol']}: p90 {r['p90_points']:.0f} pts, median {r['median_points']:.0f}, flagged {r['flagged']} "
            f"({r['flagged_share']:.2%}), on decision bars {r['decision_flagged']} of {r['decision_bars']}, "
            f"max {r['max_points']} pts at {r['max_at']}, by hour {r['by_hour']}, by weekday {r['by_weekday']}"
        )
    for r in rows:
        assert r["decision_flagged"] <= 0.01 * r["decision_bars"], r


def test_flat_and_stale_bars_are_rare(bars: pl.DataFrame) -> None:
    flat = bars.filter((pl.col("open") == pl.col("high")) & (pl.col("high") == pl.col("low")) & (pl.col("low") == pl.col("close")))
    stale = bars.filter(pl.col("close") == pl.col("close").shift(1).over("symbol"))
    print(f"\nflat bars (o=h=l=c): {flat.height}; bars whose close repeats the previous close: {stale.height}")
    if flat.height:
        print(flat.select("symbol", "timestamp", "close", "volume").head(20))
    assert flat.height <= 0.001 * bars.height
    assert stale.height <= 0.005 * bars.height


def test_four_hour_return_outliers(bars: pl.DataFrame) -> None:
    rets = bars.with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret")).drop_nulls("ret")
    large = rets.filter(pl.col("ret").abs() > RETURN_OUTLIER).sort("ret")
    print(f"\nfour-hour moves larger than {RETURN_OUTLIER:.0%}: {large.height}")
    if large.height:
        print(large.select("symbol", "timestamp", "ret").head(30))
    assert rets.filter(pl.col("ret").abs() > RETURN_ERROR).height == 0
