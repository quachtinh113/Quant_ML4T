"""Data-quality census for the exness_btc_8h tape (roadmap phase 2 gate).

Every check below runs on the **development window only** - `universe.history_start` to the day
before `evaluation.holdout_start`. A data-quality report that reads the holdout is a look at the
holdout, however innocent the statistic, so the window is a fixture and not a comment.

What this bot has to check that the FX and metals bots do not:

* **The fold.** MetaTrader 5 serves no eight-hour timeframe, so every decision bar is two H4 bars
  folded together (`mt5_loader.resample_4h_to_8h`). A slot built from one bar is a half bar
  published under a full bar's name, and it is invisible in OHLC alone.
* **The grid, on seven days.** There is no session calendar to check against, and no holidays.
  The right test is stronger and simpler: the 8-hour grid must have **no hole at all** on the
  development window, and the slots must be evenly spread over all seven weekdays. That is also
  the form in which a broker maintenance window would show up, so it is checked rather than
  assumed away.
* **The H4 grid alignment.** `bots/exness_fx_d1/BOT.md` open question 24 records a real defect in
  the shared downloader: `_copy_rates_chunked` hands naive datetimes to `copy_rates_range`, which
  MetaTrader 5 converts through the *machine's* local zone. It did not corrupt the download
  because the requested window was far wider than the served history, and the way to see that
  from the data is that every H4 bar opens on a UTC hour divisible by four. That is asserted here
  on this bot's own symbol.
* **The spread field, and its two readings.** The per-bar `spread` is what `16_costs` prices the
  backtest with, so its census matters: no zero-spread bars (gold had 1,081 and needed a fill
  rule), and a measured divergence from today's tick reading that must stay the size
  `setup.yaml::costs` says it is.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/test_data_quality.py -q
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir, read_history_depth
from case_studies.exness_btc_8h._features import SLOT_MINUTES, decision_grid
from utils.paths import REPO_ROOT

CASE_STUDY_ID = "exness_btc_8h"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
H4_HOURS = [0, 4, 8, 12, 16, 20]
DECISION_HOURS = [0, 8, 16]
RETURN_OUTLIER = 0.10   # an 8-hour move larger than this is listed
RETURN_ERROR = 0.60     # ... and larger than this is treated as a data error
SPREAD_OUTLIER_MULTIPLE = 8.0   # x the measured in-sample p90, in points


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def window(setup: dict) -> tuple[str, str]:
    """The development window. Nothing in this file reads a bar outside it."""
    start = str(setup["universe"]["history_start"])
    end = date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1)
    return start, str(end)


@pytest.fixture(scope="module")
def symbols(setup: dict) -> list[str]:
    return sorted(setup["universe"]["symbols"])


@pytest.fixture(scope="module")
def bars_4h(symbols: list[str], window: tuple[str, str]) -> pl.DataFrame:
    if not (mt5_data_dir() / "4h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/4h.parquet)")
    start, end = window
    return load_mt5_bars(
        "4h", symbols=symbols, start_date=start, end_date=end, include_spread=True
    ).with_columns(pl.col("timestamp").dt.replace_time_zone(None))


@pytest.fixture(scope="module")
def panel(symbols: list[str], window: tuple[str, str], setup: dict) -> pl.DataFrame:
    """The decision grid, cut on the DECISION INSTANT rather than on the bar open.

    The two are eight hours apart, and the difference is exactly one slot: the bar opening
    2025-08-31 16:00 closes at 2025-09-01 00:00, which is a HOLDOUT decision. Cutting on the bar
    open would put it in this census; cutting on the close does not. One slot in eight thousand is
    not a material statistic - it is a material *rule*, and it is the same rule 03, 04 and 05
    apply, so it is applied here too.
    """
    start, end = window
    bars = load_mt5_bars(
        "8h", symbols=symbols, start_date=start, end_date=end, include_spread=True
    ).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    holdout_start = datetime.combine(
        date.fromisoformat(str(setup["evaluation"]["holdout_start"])), datetime.min.time()
    )
    return decision_grid(bars, verbose=False).filter(pl.col("timestamp") < holdout_start)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
def test_h4_bars_open_on_the_four_hour_utc_grid(bars_4h: pl.DataFrame) -> None:
    """The source grid is UTC-aligned - which is also what refutes the naive-datetime defect.

    `bots/exness_fx_d1/BOT.md` open question 24: `mt5_loader._copy_rates_chunked` passes naive
    datetimes to `copy_rates_range`, and the MetaTrader5 package converts a naive datetime through
    the machine's local zone (UTC+7 on this host). Had that shifted the served window, the bar
    opens would not sit on a four-hour boundary. They do, on every bar, which is the measurement
    that says this bot's parquet is not affected.
    """
    hours = sorted(bars_4h["timestamp"].dt.hour().unique().to_list())
    minutes = sorted(bars_4h["timestamp"].dt.minute().unique().to_list())
    assert hours == H4_HOURS, f"H4 bars open on UTC hours {hours}, not {H4_HOURS}"
    assert minutes == [0], f"H4 bars open at minutes {minutes}, not on the hour"
    print(f"\n{bars_4h.height:,} H4 bars, opens on UTC hours {hours}, minute 0 on every bar")


def test_every_slot_is_folded_from_two_h4_bars(bars_4h: pl.DataFrame, setup: dict) -> None:
    """No decision bar is a half bar published under a full bar's name.

    This is what `universe.history_start` buys: the account's first served week (2018-02-09 to
    02-15) holds six single-bar slots and ten missing ones, and the declared start is the first
    date after them.
    """
    # Grouped on the bar OPEN, which is the fold key mt5_loader.resample_4h_to_8h uses. Every H4
    # bar loaded here opens before the holdout, so the census reads no holdout bar; the last slot
    # it counts is the one whose DECISION instant is the holdout's first, and that slot is
    # excluded from every other test through the `panel` fixture.
    per_slot = (
        bars_4h.with_columns(pl.col("timestamp").dt.truncate("8h").alias("_slot"))
        .group_by(["symbol", "_slot"])
        .agg(pl.len().alias("n"))
    )
    counts = sorted(per_slot["n"].unique().to_list())
    half = per_slot.filter(pl.col("n") < 2)
    assert counts == [2], (
        f"H4 bars per 8-hour slot are {counts}; {half.height} slot(s) are folded from fewer than "
        f"two, first at {half['_slot'].min() if half.height else None}. Move "
        f"universe.history_start (currently {setup['universe']['history_start']})."
    )
    print(f"\n{per_slot.height:,} decision slots, every one folded from exactly two H4 bars")


def test_keys_unique_and_on_the_eight_hour_grid(panel: pl.DataFrame) -> None:
    hours = sorted(panel["timestamp"].dt.hour().unique().to_list())
    assert hours == DECISION_HOURS
    assert panel.select(["symbol", "timestamp"]).is_duplicated().sum() == 0
    assert panel["timestamp"].dt.minute().max() == 0
    print(f"\n{panel.height:,} decision slots on hours {hours}, no duplicate key")


def test_no_hole_in_the_development_grid(panel: pl.DataFrame) -> None:
    """A 24/7 instrument has no calendar to excuse a gap, so the bar is zero missing slots.

    This is also the shape a broker maintenance window would take. Exness publishes no scheduled
    break on this symbol (`ML4T_DATA_PATH/mt5/history_depth.json` derives mon..sun 00:00-24:00
    from the last eight weeks of H1 bars) and the development window confirms it: nothing is
    missing. A future maintenance stop would fail here, loudly, rather than becoming a silent
    stale-price decision.
    """
    for symbol in panel["symbol"].unique().to_list():
        series = panel.filter(pl.col("symbol") == symbol).sort("timestamp")["timestamp"]
        expected = pl.datetime_range(
            series.min(), series.max(), interval=f"{SLOT_MINUTES}m", eager=True
        )
        missing = sorted(set(expected.to_list()) - set(series.to_list()))
        assert not missing, (
            f"{symbol}: {len(missing)} missing decision slot(s) between {series.min()} and "
            f"{series.max()}, first five {missing[:5]}"
        )
        print(f"\n{symbol}: {len(expected):,} declared slots, 0 missing")


def test_the_week_has_seven_days(panel: pl.DataFrame) -> None:
    """Weekend slots are present and roughly as numerous as weekday ones.

    On every sibling bot this count is zero on Saturday and Sunday. Here it must not be: a
    weekend that vanished would mean the loader had silently applied an FX calendar, and every
    downstream statistic would then be computed on five sevenths of the tape while the swap kept
    charging for seven.
    """
    by_weekday = (
        panel.group_by(pl.col("timestamp").dt.weekday().alias("weekday"))
        .agg(pl.len().alias("slots"))
        .sort("weekday")
    )
    counts = {int(r["weekday"]): int(r["slots"]) for r in by_weekday.iter_rows(named=True)}
    assert set(counts) == set(range(1, 8)), f"weekdays present: {sorted(counts)}"
    spread = max(counts.values()) - min(counts.values())
    assert spread <= 3 * 2, f"weekday slot counts are uneven: {counts}"
    print(f"\nslots per weekday (1 = Monday): {counts}")


def test_history_depth_matches_parquet(setup: dict, symbols: list[str]) -> None:
    """The recorded depth is the depth on disk, and the server clock is the declared one."""
    depth = read_history_depth()
    assert depth["server_clock"]["utc_offset_minutes"] == setup["decision"]["server_clock"][
        "utc_offset_minutes"
    ]
    for symbol in symbols:
        record = depth["timeframes"]["4h"][symbol]
        on_disk = load_mt5_bars("4h", symbols=[symbol])
        assert on_disk.height == record["n_bars"], (
            f"{symbol}: history_depth.json records {record['n_bars']:,} H4 bars, the parquet "
            f"holds {on_disk.height:,}"
        )
        print(
            f"\n{symbol}: {record['n_bars']:,} H4 bars {record['first_server_day']} .. "
            f"{record['last_server_day']}, matching the parquet"
        )


# ---------------------------------------------------------------------------
# The bars themselves
# ---------------------------------------------------------------------------
def test_ohlc_invariants_hold(panel: pl.DataFrame) -> None:
    bad = panel.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("close") <= 0)
    )
    assert bad.is_empty(), f"{bad.height} bars violate the OHLC ordering"
    print(f"\n{panel.height:,} decision bars, 0 OHLC violations")


def test_no_zero_volume_bars(panel: pl.DataFrame) -> None:
    """A zero-volume bar on a 24/7 instrument is a feed outage, not a quiet hour."""
    zero = panel.filter(pl.col("volume") <= 0)
    assert zero.is_empty(), (
        f"{zero.height} decision bars carry zero volume, first at {zero['timestamp'].min()}"
    )
    print(f"\nvolume: min {panel['volume'].min():,}, median {panel['volume'].median():,.0f}")


def test_flat_and_stale_bars_are_rare(panel: pl.DataFrame) -> None:
    """A repeated close is a stale feed; on this instrument it should be essentially absent."""
    stale = panel.sort(["symbol", "timestamp"]).with_columns(
        (pl.col("close") == pl.col("close").shift(1).over("symbol")).alias("_same"),
        (pl.col("high") == pl.col("low")).alias("_flat"),
    )
    same = float(stale["_same"].fill_null(False).mean())
    flat = float(stale["_flat"].mean())
    assert same < 0.001, f"{same:.3%} of decision bars repeat the previous close"
    assert flat < 0.001, f"{flat:.3%} of decision bars have high == low"
    print(f"\nrepeated close {same:.4%}, high == low {flat:.4%}")


def test_eight_hour_return_outliers(panel: pl.DataFrame) -> None:
    """Large moves are listed, not silenced; only an implausible one is an error.

    The threshold is set for THIS instrument and nothing else: an eight-hour 10 % move is an
    ordinary crypto event (the FX bot's equivalent test flags 2 %), while a 60 % move in eight
    hours would be a decimal error.
    """
    moves = (
        panel.sort(["symbol", "timestamp"])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("r"))
        .drop_nulls("r")
    )
    big = moves.filter(pl.col("r").abs() > RETURN_OUTLIER)
    error = moves.filter(pl.col("r").abs() > RETURN_ERROR)
    assert error.is_empty(), (
        f"{error.height} eight-hour moves exceed {RETURN_ERROR:.0%}: "
        f"{error.select('timestamp', 'r').head(5).to_dicts()}"
    )
    print(
        f"\n{big.height} moves exceed {RETURN_OUTLIER:.0%} of "
        f"{moves.height:,} ({big.height / moves.height:.3%}); largest "
        f"{moves['r'].abs().max():.2%} at "
        f"{moves.filter(pl.col('r').abs() == moves['r'].abs().max())['timestamp'][0]}"
    )


# ---------------------------------------------------------------------------
# The spread field: what 16_costs prices with
# ---------------------------------------------------------------------------
def test_spread_field_has_no_free_crossings(panel: pl.DataFrame, setup: dict) -> None:
    """No zero-spread bar, and the count is the one setup.yaml declares.

    `exness_gold_sess` found 1,081 zero-spread H1 bars on gold up to 2023-02-03 and had to write a
    fill rule so that most of 2017-2022 was not priced as a free crossing. This instrument has
    none on the decision grid, which is why no fill rule exists here - and the assertion is what
    stops a re-download from introducing one silently.
    """
    declared = int(setup["costs"]["spread_bps_in_sample"]["zero_spread_slots"])
    zero = panel.filter(pl.col("spread") <= 0)
    assert zero.height == declared, (
        f"{zero.height} decision bars quote spread <= 0; setup.yaml declares {declared}. "
        "A zero spread is a free crossing and needs a declared fill rule before it is priced."
    )
    print(f"\nzero-spread decision bars: {zero.height} (declared {declared})")


def test_in_sample_spread_matches_the_declaration(panel: pl.DataFrame, setup: dict) -> None:
    """The per-bar spread census is the one `costs.spread_bps` was filled from.

    This is the number the engine charges (`_normalize_cfd_costs` takes the top of the declared
    range), so a drift between the file and the tape is a silent mispricing of every backtest.
    """
    declared = setup["costs"]["spread_bps_in_sample"]
    point = float(setup["costs"]["contract"]["point"])
    bps = panel.with_columns((pl.col("spread") * point / pl.col("close") * 1e4).alias("bps"))["bps"]
    p50, p90 = float(bps.median()), float(bps.quantile(0.9))
    assert abs(p50 - float(declared["p50_bps"])) < 0.05, (
        f"in-sample spread p50 is {p50:.3f} bps, setup.yaml declares {declared['p50_bps']}"
    )
    assert abs(p90 - float(declared["p90_bps"])) < 0.10, (
        f"in-sample spread p90 is {p90:.3f} bps, setup.yaml declares {declared['p90_bps']}"
    )
    assert panel.height == int(declared["n_slots"]), (
        f"the development window holds {panel.height:,} slots, setup.yaml declares "
        f"{declared['n_slots']:,}"
    )
    print(f"\nin-sample spread: p50 {p50:.3f} bps, p90 {p90:.3f} bps over {panel.height:,} slots")


def test_the_two_spread_measurements_still_disagree_by_the_declared_factor(
    panel: pl.DataFrame, setup: dict
) -> None:
    """The live tick reading and the in-sample reading are far apart, and that is the point.

    If this test ever fails because the two have converged, the honest response is to re-read the
    declaration - not to widen the tolerance. The whole cost design of this bot rests on charging
    the in-sample number to a backtest and the live number to a live order.
    """
    tick_p90 = float(setup["costs"]["spread_bps_by_session"][sorted(setup["universe"]["symbols"])[0]]["all"][1])
    in_sample_p90 = float(setup["costs"]["spread_bps_in_sample"]["p90_bps"])
    ratio = in_sample_p90 / tick_p90
    assert ratio > 5.0, (
        f"the in-sample p90 is only {ratio:.1f}x the live p90; the two measurements have "
        "converged and costs.spread_bps should be re-derived"
    )
    print(
        f"\nin-sample p90 {in_sample_p90} bps vs live tick p90 {tick_p90} bps = {ratio:.1f}x. "
        "The engine charges the first; a live order pays the second."
    )


def test_spread_outliers_against_the_measured_p90(panel: pl.DataFrame, setup: dict) -> None:
    """Bars quoting far above the in-sample p90 are listed with their dates, not suppressed."""
    point = float(setup["costs"]["contract"]["point"])
    p90_points = float(
        panel.with_columns((pl.col("spread")).alias("_s"))["_s"].quantile(0.9)
    )
    limit = SPREAD_OUTLIER_MULTIPLE * p90_points
    wide = panel.filter(pl.col("spread") > limit)
    share = wide.height / panel.height
    print(
        f"\nper-bar spread p90 {p90_points:,.0f} points; {wide.height} bars ({share:.3%}) quote "
        f"more than {SPREAD_OUTLIER_MULTIPLE:.0f}x that"
    )
    if wide.height:
        worst = wide.sort("spread", descending=True).head(5)
        for row in worst.iter_rows(named=True):
            print(
                f"    {row['timestamp']}  {row['spread']:,.0f} points "
                f"({row['spread'] * point / row['close'] * 1e4:.1f} bps)"
            )
    assert share < 0.01, f"{share:.2%} of bars quote a spread far above the p90"


def test_the_measured_cost_records_are_on_disk(setup: dict) -> None:
    """The JSON records the cost block was filled from exist and name this account.

    A cost block whose provenance file is missing is a number with no vintage, which is exactly
    what `write_artifact`'s digest sidecars exist to prevent for data.
    """
    data_dir = mt5_data_dir()
    for name in ("spreads_by_session.json", "symbol_info_btc_2026-09-08.json"):
        path = data_dir / name
        if not path.exists():
            pytest.skip(f"cost record not on this machine: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record, f"{name} is empty"
    swaps = json.loads((data_dir / "symbol_info_btc_2026-09-08.json").read_text(encoding="utf-8"))
    symbol = sorted(setup["universe"]["symbols"])[0]
    declared = setup["costs"]["swap"]["points_per_lot_per_night"][symbol]
    measured = swaps["swaps"][symbol]
    assert float(measured["swap_long"]) == float(declared["long"])
    assert float(measured["swap_short"]) == float(declared["short"])
    assert bool(measured["charges_weekends"]) is bool(setup["costs"]["swap"]["charges_weekends"])
    assert int(measured["swap_rollover3days"]) == 5, "the triple-swap day is not Friday any more"
    print(
        f"\nswap on record: long {measured['swap_long']} / short {measured['swap_short']} points "
        f"per lot per night, charges_weekends {measured['charges_weekends']}, "
        f"rollover3days {measured['swap_rollover3days']} (Friday)"
    )
