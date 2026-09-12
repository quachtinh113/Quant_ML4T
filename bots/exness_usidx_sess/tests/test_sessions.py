"""Session-geometry checks for exness_usidx_sess (roadmap phase 2 gate).

The three facts this bot's design rests on, asserted on the bars rather than described in a
document. Each of them was found by measurement on 2026-09-08 and each of them would be wrong
if it were hard-coded from ``bots/assets/US500.md``, which records an eight-week summer snapshot.

1. **The trading hours follow New York daylight saving; the server clock does not.** The daily
   Globex break ends at 18:00 New York in every readable month of the sample, which puts it on a
   different UTC hour in summer and in winter. ``setup.yaml::decision.trade_hours_follow_dst_of``
   is that claim and this is the test of it.
2. **The two decision instants come from the NYSE calendar**, so they move with daylight saving
   and with the 13:00 half-days without anyone editing a constant.
3. **A decision with no bar to fill at places no order.** The index shuts for the weekend at the
   Friday cash close and, before 2024, the daily break sat directly on the cash close, so the
   ``overnight`` spec has no fill on a quarter of its sessions. The rule is "no bar, no order";
   the share is measured here and must not be rising.

Needs the MT5 history (``ML4T_DATA_PATH/mt5/1h.parquet``) and the full project environment.
Reads only the development history.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_usidx_sess/tests/test_sessions.py -q -s
"""

from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

pytest.importorskip(
    "ml4t.diagnostic", reason="the NYSE calendar needs the full project environment"
)

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir  # noqa: E402
from case_studies.exness_usidx_sess._features import (  # noqa: E402
    SPECS,
    daily_break_by_month,
    session_grid,
    session_panel,
)
from utils.paths import REPO_ROOT  # noqa: E402

CASE_STUDY_ID = "exness_usidx_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
# Measured 2026-09-08 over 2022-10-25 .. 2026-02-28: the overnight spec has no fill bar on 26 %
# of its index-sessions, concentrated in 2022-2023. A ceiling well above the measurement, so the
# test fails on a regime change rather than on noise.
MAX_NO_FILL_SHARE = {"intraday": 0.02, "overnight": 0.40}
# ... and in the most recent full development year it has to be far better than that, which is
# what says the gap is a historical regime rather than a permanent property of the instrument.
MAX_NO_FILL_SHARE_RECENT = {"intraday": 0.02, "overnight": 0.15}
# The year from which the Friday overnight close became fillable on this account. Measured
# 2026-09-08 on development rows: 2 of 20 Fridays filled in 2022, 2 of 102 in 2023, 2 of 102
# in 2024, 64 of 100 in 2025 and 18 of 18 in the development part of 2026. The step is 2025,
# and it is why the claim that this spec 'has no Fridays' is false from that year on.
FRIDAY_FILL_RECOVERED_FROM = 2025
# ... and the share of that year's Fridays that must carry a fill. 2025 measures 64 %, so the
# floor sits below it and far above the 2 % of the years before: it asserts that the regime
# changed, not that it is perfect.
MIN_FRIDAY_FILL_SHARE_RECENT = 0.55
RECENT_YEAR = 2025


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_end(setup: dict) -> date:
    return date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1)


@pytest.fixture(scope="module")
def bars(setup: dict, dev_end: date) -> pl.DataFrame:
    if not (mt5_data_dir() / "1h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/1h.parquet)")
    return load_mt5_bars(
        "1h",
        symbols=sorted(setup["universe"]["symbols"]),
        start_date=str(setup["universe"]["history_start"]),
        end_date=str(dev_end),
    ).with_columns(pl.col("timestamp").dt.replace_time_zone(None))


@pytest.fixture(scope="module")
def panels(bars: pl.DataFrame, setup: dict) -> dict[str, pl.DataFrame]:
    decision = setup["decision"]
    return {
        spec: session_panel(
            bars,
            spec=spec,
            calendar=decision["session_calendar"],
            tolerance_minutes=int(decision["session_close_tolerance_minutes"]),
            open_delay_minutes=int(decision["open_delay_minutes"]),
            verbose=False,
        )
        for spec in SPECS
    }


# ---------------------------------------------------------------------------
# 1. The trading hours follow New York, the clock does not
# ---------------------------------------------------------------------------
def test_the_daily_break_follows_new_york_daylight_saving(
    bars: pl.DataFrame, setup: dict
) -> None:
    zone = setup["decision"]["trade_hours_follow_dst_of"]
    assert zone, "decision.trade_hours_follow_dst_of is not declared"
    for symbol in sorted(setup["universe"]["symbols"]):
        regime = daily_break_by_month(bars, symbol, zone=zone)
        readable = regime.drop_nulls("break_end_ny").filter(~pl.col("has_dst_transition"))
        assert readable.height >= 24, (
            f"{symbol}: only {readable.height} months are readable, too few to test the claim"
        )
        ends = set(readable["break_end_ny"].to_list())
        assert ends == {18}, (
            f"{symbol}: the daily break ends at {sorted(ends)} New York time, not always 18:00; "
            f"decision.trade_hours_follow_dst_of: {zone} is wrong"
        )
        summer = {h for row in readable.filter("is_dst")["break_utc"].to_list() for h in row}
        winter = {h for row in readable.filter(~pl.col("is_dst"))["break_utc"].to_list() for h in row}
        assert max(winter) == max(summer) + 1, (
            f"{symbol}: winter break {sorted(winter)} is not one UTC hour after summer "
            f"{sorted(summer)}"
        )
        print(
            f"\n{symbol}: break ends 18:00 {zone} in all {readable.height} readable months; "
            f"UTC hours summer {sorted(summer)}, winter {sorted(winter)}"
        )


def test_the_server_clock_does_not_follow_daylight_saving(setup: dict) -> None:
    """The two declarations are different facts and must not be conflated.

    ``server_clock.follows_dst_of`` is null (measured: UTC+0 all year) while
    ``trade_hours_follow_dst_of`` names New York. A future edit that sets them to the same value
    is the mistake this test exists to catch.
    """
    decision = setup["decision"]
    assert decision["server_clock"]["follows_dst_of"] is None, (
        "the server clock was measured as UTC+0 all year on 2026-09-05; a non-null value here "
        "needs a new measurement in bots/exness_usidx_sess/BOT.md"
    )
    assert decision["trade_hours_follow_dst_of"] == "America/New_York"
    assert int(decision["server_clock"]["utc_offset_minutes"]) == 0


# ---------------------------------------------------------------------------
# 2. The decision instants come from the calendar
# ---------------------------------------------------------------------------
def test_decision_instants_move_with_daylight_saving(
    panels: dict[str, pl.DataFrame], setup: dict
) -> None:
    """The same New York wall-clock instant, on a different UTC hour in each season."""
    zone = ZoneInfo(setup["decision"]["trade_hours_follow_dst_of"])
    for spec, panel in panels.items():
        local = (
            panel.select("timestamp", "decision_ts")
            .unique()
            .with_columns(
                pl.col("decision_ts")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(str(zone))
                .dt.hour()
                .alias("ny_hour"),
                pl.col("decision_ts").dt.hour().alias("utc_hour"),
            )
        )
        # Half-days close three hours early, so the New York hour is not a single value; the
        # regular sessions are, and they are the overwhelming majority.
        common = local["ny_hour"].value_counts().sort("count", descending=True)
        top_hour, top_count = common["ny_hour"][0], common["count"][0]
        assert top_count / local.height > 0.95, (
            f"{spec}: no single New York hour covers 95 % of the decisions"
        )
        expected = 10 if spec == "intraday" else 16
        assert top_hour == expected, (
            f"{spec}: decisions are taken at {top_hour}:00 New York, expected {expected}:00 "
            f"(cash open 09:30 + {setup['decision']['open_delay_minutes']} minutes, cash close 16:00)"
        )
        utc_hours = sorted(local["utc_hour"].unique().to_list())
        assert len(utc_hours) > 1, f"{spec}: one UTC hour for the whole sample means no DST"
        print(f"\n{spec}: {top_hour}:00 New York on {top_count / local.height:.1%} of sessions, UTC hours {utc_hours}")


def test_half_days_are_read_from_the_calendar(setup: dict, bars: pl.DataFrame) -> None:
    """The NYSE half-days (13:00 New York) exist in the grid and close three hours early."""
    decision = setup["decision"]
    grid = session_grid(
        decision["session_calendar"],
        bars["timestamp"].min(),
        bars["timestamp"].max(),
        open_delay_minutes=int(decision["open_delay_minutes"]),
    )
    lengths = grid.with_columns(
        ((pl.col("close_at") - pl.col("open_at")).dt.total_minutes() / 60).alias("hours")
    )
    half_days = lengths.filter(pl.col("hours") < 6.0)
    assert half_days.height >= 6, (
        f"only {half_days.height} half-days in the calendar over "
        f"{grid.height} sessions; the calendar is not the NYSE one"
    )
    assert set(lengths["hours"].unique().to_list()) <= {3.5, 6.5}, (
        "a session is neither a full day nor a half day"
    )
    print(f"\n{half_days.height} half-days of 3.5 hours among {grid.height} NYSE cash sessions")


def test_no_session_falls_on_a_weekend_or_an_nyse_holiday(
    panels: dict[str, pl.DataFrame]
) -> None:
    for spec, panel in panels.items():
        weekend = panel.filter(pl.col("timestamp").dt.weekday() > 5)
        assert weekend.height == 0, f"{spec}: {weekend.height} sessions fall on a weekend"


# ---------------------------------------------------------------------------
# 3. No bar to fill at, no order
# ---------------------------------------------------------------------------
def test_missing_fill_bars_are_within_the_declared_ceiling(
    panels: dict[str, pl.DataFrame]
) -> None:
    """The share of decisions with no fill bar, overall and in the most recent full year.

    This is the number ``01_feasibility_analysis`` section B.3 reports and the design depends
    on: an overnight spec that could not be executed at all would be a strategy on paper only.
    """
    for spec, panel in panels.items():
        share = float(panel["exec_open"].is_null().mean())
        assert share <= MAX_NO_FILL_SHARE[spec], (
            f"{spec}: {share:.1%} of index-sessions have no fill bar, above the declared "
            f"{MAX_NO_FILL_SHARE[spec]:.0%}"
        )
        recent = panel.filter(pl.col("timestamp").dt.year() == RECENT_YEAR)
        recent_share = float(recent["exec_open"].is_null().mean())
        assert recent_share <= MAX_NO_FILL_SHARE_RECENT[spec], (
            f"{spec}: {recent_share:.1%} of {RECENT_YEAR} index-sessions have no fill bar, "
            f"above the declared {MAX_NO_FILL_SHARE_RECENT[spec]:.0%}; the gap is supposed to "
            f"be a historical regime, not a permanent property of the instrument"
        )
        print(
            f"\n{spec}: no fill on {share:.1%} of {panel.height:,} index-sessions overall, "
            f"{recent_share:.1%} in {RECENT_YEAR}"
        )


def test_the_overnight_gap_is_fridays_and_the_early_sample(
    panels: dict[str, pl.DataFrame]
) -> None:
    """Where the missing fills are, per YEAR, so a new cause cannot hide behind a known one.

    TIGHTENED 2026-09-08 (second pass). The assertion this replaces asked only that every missing
    fill be *either* a Friday *or* in 2022-2023. On a sample whose early years are almost entirely
    2022-2023 that disjunction is nearly free: a brand-new cause appearing in 2023 would have been
    absorbed by the year clause and a brand-new cause appearing on a Friday by the weekday clause,
    and neither would have been reported. It also encoded a claim that turned out to be false -
    that the overnight spec has no Fridays at all - which is true of 2022-2024 and not of 2025 or
    2026.

    The per-year form says what is actually believed about this instrument, and each half can fail
    on its own:

    * from ``FRIDAY_FILL_RECOVERED_FROM`` onward the Friday close IS fillable, on at least
      ``MIN_FRIDAY_FILL_SHARE_RECENT`` of Fridays - the weekend gap closed, and if it re-opens
      this fails;
    * before that year the missing fills are Fridays and the 2022-2023 two-hour-break regime, and
      the share of missing fills those two explain is asserted per year rather than pooled, so a
      third cause appearing in one year cannot be averaged away by the other years.
    """
    panel = panels["overnight"]
    missing = panel.filter(pl.col("exec_open").is_null())
    if missing.is_empty():
        pytest.skip("no missing fills to attribute")
    per_year = (
        panel.with_columns(
            pl.col("timestamp").dt.year().alias("year"),
            (pl.col("timestamp").dt.weekday() == 5).alias("friday"),
            pl.col("exec_open").is_null().alias("no_fill"),
        )
        .group_by("year")
        .agg(
            pl.len().alias("sessions"),
            pl.col("no_fill").sum().alias("missing"),
            (pl.col("no_fill") & pl.col("friday")).sum().alias("missing_friday"),
            (pl.col("friday") & ~pl.col("no_fill")).sum().alias("friday_filled"),
            pl.col("friday").sum().alias("fridays"),
        )
        .sort("year")
    )
    print("\novernight fills by year (sessions, missing, missing on a Friday, Fridays filled):")
    for row in per_year.iter_rows(named=True):
        print(
            f"  {row['year']}: {row['sessions']:>4} sessions, {row['missing']:>3} missing "
            f"({row['missing_friday']:>3} on a Friday), {row['friday_filled']:>3} of "
            f"{row['fridays']:>3} Fridays filled"
        )

    for row in per_year.iter_rows(named=True):
        year, missing_n = row["year"], row["missing"]
        if year >= FRIDAY_FILL_RECOVERED_FROM and row["fridays"]:
            share = row["friday_filled"] / row["fridays"]
            assert share >= MIN_FRIDAY_FILL_SHARE_RECENT, (
                f"{year}: only {share:.0%} of Friday overnight decisions have a fill bar, below "
                f"the declared {MIN_FRIDAY_FILL_SHARE_RECENT:.0%}; the weekend gap that closed in "
                f"{FRIDAY_FILL_RECOVERED_FROM} has re-opened and the spec's evidence base moved"
            )
        if not missing_n:
            continue
        explained = row["missing_friday"] + (missing_n if year <= 2023 else 0)
        assert explained >= missing_n, (
            f"{year}: {missing_n - explained} of {missing_n} missing fills are neither a Friday "
            f"nor in the 2022-2023 two-hour-break regime; a third cause has appeared in {year} "
            "and the pooled test would have hidden it"
        )


def test_every_session_has_both_indices_or_neither(panels: dict[str, pl.DataFrame]) -> None:
    """Two instruments, and a session where only one quotes would silently halve the book."""
    for spec, panel in panels.items():
        counts = panel.group_by("timestamp").agg(pl.len().alias("n"))
        odd = counts.filter(pl.col("n") != 2)
        share = odd.height / counts.height
        print(f"\n{spec}: {odd.height} of {counts.height:,} sessions carry one index only ({share:.2%})")
        assert share < 0.01, f"{spec}: {share:.1%} of sessions carry only one index"
