"""The session clock of exness_gold_sess (roadmap phase 2).

The decision rule is one sentence and it is the whole design:

    the decision is the close of the first H1 bar closing at or after
    (venue open + ``decision.edge_block_minutes``), and the label ends at the close of the last
    H1 bar closing at or before the venue close.

Everything here is computed through ``bots/_shared/sessions.py`` and the ``zoneinfo`` database.
**No test in this file contains a UTC hour table.** The server clock is UTC+0 all year and does
not follow anybody's daylight saving, while both venues do, so a hard-coded server hour is right
in one season and wrong in the other - the defect the mentor spec section 2.b names. A test that
carried the table would only prove the table agrees with itself.

What it holds:

1. the decision instant is venue open + 60 minutes in both seasons at both venues, and that is a
   *consequence* of the edge block on a UTC-aligned hourly grid, not an independent assumption;
2. the label endpoint is the last bar closing at or before the venue close, eight bars later;
3. ``edge_open`` does not cover the decision instant - the 30-minute edge block holding - and the
   seven other ``sessions.py`` flags that are constant at every decision are constant, which is
   why they are not shipped as features;
4. the DST transition dates specifically, including the four weeks each spring and autumn when
   London and New York are **not** in step, so the two decisions sit four hours apart instead of
   five - and the label overlap that the HAC lag of 2 rests on is re-measured there.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests/test_sessions.py -q -s
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir
from bots._shared.sessions import EDGE_MINUTES, SESSIONS, session_flags
from case_studies.exness_gold_sess._features import (
    BAR_MINUTES,
    CONSTANT_DECISION_FLAGS,
    VENUES,
    _venue_window,
    session_of,
    session_panel,
    session_schedule,
)
from utils.paths import REPO_ROOT

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_window(setup: dict) -> tuple[date, date]:
    return (
        date.fromisoformat(str(setup["universe"]["history_start"])),
        date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1),
    )


@pytest.fixture(scope="module")
def panel(setup: dict, dev_window: tuple[date, date]) -> pl.DataFrame:
    if not (mt5_data_dir() / "1h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/1h.parquet)")
    start, end = dev_window
    bars = load_mt5_bars(
        "1h",
        symbols=sorted(setup["universe"]["symbols"]),
        start_date=str(start),
        end_date=str(end),
    ).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    return session_panel(
        bars,
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
        keep_context=True,
        verbose=False,
    )


def _dst_transitions(zone: str, first: int, last: int) -> list[date]:
    """The dates on which a zone's UTC offset changes, found by scanning, not by a rule."""
    tz = ZoneInfo(zone)
    out, previous = [], None
    day = date(first, 1, 1)
    while day <= date(last, 12, 31):
        offset = datetime(day.year, day.month, day.day, 12, tzinfo=tz).utcoffset()
        if previous is not None and offset != previous:
            out.append(day)
        previous = offset
        day += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# 1. the decision instant
# ---------------------------------------------------------------------------
def test_the_decision_is_open_plus_sixty_minutes_in_both_seasons(
    setup: dict, dev_window: tuple[date, date]
) -> None:
    """Derived from the venue windows, on every declared session of the development span.

    The schedule is built by the same function the panel is, and the *rule* is applied here from
    first principles - "the first hourly close at or after open + the edge block" - so this test
    fails if either the rule or the zone data moves, and passes for a reason rather than by
    construction.
    """
    start, end = dev_window
    schedule = session_schedule(
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.max.time()),
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
    )
    block = int(setup["decision"]["edge_block_minutes"])
    rows = []
    for row in schedule.iter_rows(named=True):
        opened = row["session_open_ts"]
        target = opened + timedelta(minutes=block)
        # the first instant on the UTC hourly grid at or after the target
        hour = target.replace(minute=0, second=0, microsecond=0)
        decision = hour if hour >= target else hour + timedelta(hours=1)
        rows.append(
            {
                "session": row["session"],
                "date": row["session_date"],
                "offset_minutes": int((decision - opened).total_seconds() // 60),
                "utc_hour": decision.hour,
                "summer": bool(
                    datetime(
                        row["session_date"].year,
                        row["session_date"].month,
                        row["session_date"].day,
                        12,
                        tzinfo=ZoneInfo(SESSIONS[VENUES[row["session"]]].zone),
                    ).dst()
                ),
            }
        )
    frame = pl.DataFrame(rows)
    assert frame["offset_minutes"].unique().to_list() == [BAR_MINUTES], (
        frame["offset_minutes"].unique().to_list()
    )
    census = frame.group_by(["session", "summer", "utc_hour"]).len().sort(["session", "summer"])
    print(f"\n{frame.height:,} declared sessions; decision = open + 60 min on every one")
    with pl.Config(tbl_rows=census.height):
        print(census)
    # two UTC hours per venue, and they are the two seasons - not a table, a consequence
    for venue in VENUES:
        hours = set(frame.filter(pl.col("session") == venue)["utc_hour"].to_list())
        assert len(hours) == 2, (venue, hours)
        summer_hours = set(
            frame.filter((pl.col("session") == venue) & pl.col("summer"))["utc_hour"].to_list()
        )
        winter_hours = set(
            frame.filter((pl.col("session") == venue) & ~pl.col("summer"))["utc_hour"].to_list()
        )
        assert len(summer_hours) == len(winter_hours) == 1
        assert summer_hours != winter_hours
        assert min(summer_hours) + 1 == min(winter_hours), (
            f"{venue}: summer decision {summer_hours}, winter {winter_hours} - daylight saving "
            "moves the decision one hour EARLIER in UTC, not later"
        )


def test_session_of_recovers_the_venue_from_the_instant_alone(panel: pl.DataFrame) -> None:
    """13_backtest filters predictions on this, holding only a timestamp."""
    instants = panel.select("timestamp", "session").unique().sort("timestamp")
    recovered = session_of(instants["timestamp"])
    mismatch = (recovered != instants["session"]).sum()
    assert mismatch == 0, f"{mismatch} instants resolve to a different venue"
    assert recovered.null_count() == 0
    print(f"\n{instants.height:,} decision instants; session_of agrees on every one")


# ---------------------------------------------------------------------------
# 2. the label endpoint
# ---------------------------------------------------------------------------
def test_the_endpoint_is_the_last_bar_closing_at_or_before_the_venue_close(
    panel: pl.DataFrame,
) -> None:
    resolved = panel.drop_nulls("label_end_ts")
    late = resolved.filter(pl.col("label_end_ts") > pl.col("session_close_ts"))
    assert late.is_empty(), f"{late.height} endpoints close after the venue close"
    gap = resolved.select(
        (
            (pl.col("session_close_ts").dt.epoch("s") - pl.col("label_end_ts").dt.epoch("s")) // 60
        ).alias("gap")
    )["gap"]
    assert int(gap.max()) <= BAR_MINUTES, int(gap.max())
    spans = resolved.select(
        (
            (pl.col("label_end_ts").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")) // 60
        ).alias("span")
    )["span"].unique().to_list()
    assert sorted(spans) == [8 * BAR_MINUTES]
    print(
        f"\n{resolved.height:,} resolved endpoints: every one closes at or before its venue "
        f"close (max gap {int(gap.max())} min) and exactly 8 bars after its decision; "
        f"{panel['label_end_ts'].null_count()} rows have no endpoint at all"
    )


# ---------------------------------------------------------------------------
# 3. the edge block, and the flags that are constant
# ---------------------------------------------------------------------------
def test_the_edge_block_does_not_cover_the_decision_instant(panel: pl.DataFrame) -> None:
    """``edge_open`` is false at every decision, and seven other flags are constant too.

    ``edge_open`` false is the 30-minute block of ``bots/assets/XAUUSD.md:22`` holding, not an
    accident: the decision is 60 minutes after the open and ``EDGE_MINUTES`` is 30. The other
    seven are recorded because a constant column is not a feature - shipping them would add
    rank-deficient regressors and the register says so instead.
    """
    instants = panel["timestamp"].unique().sort().to_list()
    seen: dict[str, set[bool]] = {name: set() for name in CONSTANT_DECISION_FLAGS}
    for ts in instants:
        flags = session_flags(ts)
        for name in CONSTANT_DECISION_FLAGS:
            seen[name].add(bool(flags[name]))
    print(f"\nsessions.py flags over {len(instants):,} decision instants:")
    for name, expected in CONSTANT_DECISION_FLAGS.items():
        print(f"  {name}: {sorted(seen[name])} (declared constant {expected})")
        assert seen[name] == {expected}, (
            f"{name} is not constant at the decision any more: {sorted(seen[name])}. Either the "
            "grid moved or the flag now carries information and belongs in the register."
        )
    assert EDGE_MINUTES < BAR_MINUTES, (
        "the edge block is at least a whole bar, so the decision would fall inside it"
    )


def test_the_flags_that_do_carry_information_are_the_ones_the_register_declares(
    panel: pl.DataFrame,
) -> None:
    """``new_york`` and ``tokyo`` vary at the decision; both are already covered by state columns.

    ``new_york`` is true at exactly the New York decisions, so it duplicates ``is_ny``; ``tokyo``
    (and ``funding``, which takes the same value) is true at exactly the London decisions in the
    summer season, so it duplicates ``is_ny`` and ``is_venue_dst`` together. Recording that here
    is what makes "the register declares the flags that matter" checkable rather than asserted.
    """
    rows = panel.select("timestamp", "session").unique().sort("timestamp")
    flags = [session_flags(ts) for ts in rows["timestamp"].to_list()]
    is_ny = np.array([s == "ny" for s in rows["session"].to_list()])
    assert (np.array([f["new_york"] for f in flags]) == is_ny).all(), (
        "the new_york flag is no longer exactly the New York decisions"
    )
    tokyo = np.array([f["tokyo"] for f in flags])
    funding = np.array([f["funding"] for f in flags])
    assert (tokyo == funding).all(), "tokyo and funding used to take the same value here"
    assert tokyo[is_ny].sum() == 0, "the tokyo flag now fires on a New York decision"
    print(
        f"\nnew_york == is_ny on all {len(flags):,} instants; tokyo == funding, true on "
        f"{int(tokyo.sum()):,} London decisions (the summer season) and no New York one"
    )


# ---------------------------------------------------------------------------
# 4. daylight saving
# ---------------------------------------------------------------------------
def test_the_decision_hour_moves_on_each_venue_own_transition_date(
    dev_window: tuple[date, date],
) -> None:
    """On the transition weekend the decision hour changes, and only for that venue."""
    start, end = dev_window
    for venue in VENUES:
        zone = SESSIONS[VENUES[venue]].zone
        transitions = [d for d in _dst_transitions(zone, start.year, end.year) if start <= d <= end]
        assert len(transitions) >= 2 * (end.year - start.year), (venue, len(transitions))
        moved = 0
        for day in transitions:
            before = day - timedelta(days=3)
            after = day + timedelta(days=3)
            open_before = _venue_window(before, venue)[0] + timedelta(minutes=BAR_MINUTES)
            open_after = _venue_window(after, venue)[0] + timedelta(minutes=BAR_MINUTES)
            assert open_before.hour != open_after.hour, (venue, day)
            moved += 1
        print(f"\n{venue}: {moved} daylight-saving transitions in the development window")


def test_the_four_weeks_a_year_when_london_and_new_york_are_out_of_step(
    panel: pl.DataFrame, dev_window: tuple[date, date]
) -> None:
    """Europe and the United States switch on different weekends, and the gap narrows to 4 hours.

    The United States moves to daylight saving on the second Sunday in March and back on the
    first Sunday in November; the European Union moves on the last Sunday in March and the last
    Sunday in October. So for about three weeks each spring and one each autumn New York is on
    summer time while London is not, the two decisions sit **four** hours apart instead of five,
    and the London hold to 17:00 UTC contains the New York decision at a different point.

    The consequence that matters is the label overlap, because ``02_labels`` measures it once and
    every information coefficient downstream is Newey-West corrected at the lag it found. This
    test re-measures the overlap **inside the out-of-step weeks alone** and asserts it is still
    one slot, so the HAC lag of 2 is right there too. If a future change to the venue windows
    ever made it two, this fails rather than the standard errors quietly shrinking.
    """
    start, end = dev_window
    rows = (
        panel.filter(pl.col("symbol") == sorted(panel["symbol"].unique().to_list())[0])
        .select("timestamp", "session", "session_open_ts")
        .sort("timestamp")
    )
    day_frame = rows.with_columns(pl.col("timestamp").dt.date().alias("day"))
    gaps = (
        day_frame.pivot(index="day", on="session", values="timestamp")
        .drop_nulls()
        .with_columns(
            (
                (pl.col("ny").dt.epoch("s") - pl.col("london").dt.epoch("s")) // 3600
            ).alias("gap_h")
        )
    )
    census = gaps.group_by("gap_h").len().sort("gap_h")
    print("\nhours between the London and the New York decision, by count of dates:")
    print(census)
    assert set(census["gap_h"].to_list()) == {4, 5}, census
    out_of_step = gaps.filter(pl.col("gap_h") == 4)
    years = end.year - start.year + 1
    assert 10 * years <= out_of_step.height <= 30 * years, (
        f"{out_of_step.height} out-of-step dates over {years} years is not the three spring "
        "weeks plus one autumn week the two calendars imply"
    )
    print(
        f"  {out_of_step.height} dates are 4 hours apart (London and New York out of step), "
        f"{gaps.height - out_of_step.height} are 5; first few: "
        f"{[str(d) for d in out_of_step['day'].to_list()[:6]]}"
    )
    # the overlap, re-measured inside the out-of-step dates only
    horizon = timedelta(hours=8)
    for label, subset in (
        ("out of step (4 h apart)", set(out_of_step["day"].to_list())),
        ("in step (5 h apart)", set(gaps.filter(pl.col("gap_h") == 5)["day"].to_list())),
    ):
        ts = np.array(
            [t for t in rows["timestamp"].to_list() if t.date() in subset],
            dtype="datetime64[us]",
        )
        every = rows["timestamp"].to_numpy().astype("datetime64[us]")
        ends = ts + np.timedelta64(int(horizon.total_seconds()), "s").astype("timedelta64[us]")
        starts = np.searchsorted(every, ts, side="left")
        counts = np.searchsorted(every, ends, side="left") - starts - 1
        print(
            f"  {label}: {len(ts):,} decisions, later decisions starting inside the 8-hour "
            f"window: min {counts.min()}, max {counts.max()}, mean {counts.mean():.3f}"
        )
        assert counts.max() <= 1, (
            f"{label}: the maximum overlap is {counts.max()} slots, so the HAC lag of 2 that "
            "02_labels measured is too short there"
        )
