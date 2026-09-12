"""The declared closure calendar of exness_gold_sess - generation 2, block 1.

``bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md`` section 1, the four tests it names, with
their propositions unchanged. The file under test is
``bots/exness_gold_sess/calendar/closures_metals.parquet``, written only by
``tools/build_closure_calendar.py`` from two sources - the pinned ``exchange_calendars`` package
and the development tape - and read through ``_features.declared_closures``.

Guard: point-in-time (a closure must be knowable in advance, so the package rows are what a
live consumer may read; the tape rows are history) and parity (the same file serves the
backtest's early-close rule and the deployment breaker). Nothing here reads a return.

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess \\
      uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests/test_calendar.py -q -s
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir
from case_studies.exness_gold_sess._features import (
    CLOSURE_CALENDAR_PATH,
    CLOSURE_COLUMNS,
    declared_closures,
    session_panel,
)
from case_studies.exness_gold_sess._hold import assert_no_holdout, development_window
from utils.paths import REPO_ROOT

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
TOOL_PATH = REPO_ROOT / "bots" / CASE_STUDY_ID / "tools" / "build_closure_calendar.py"

#: ``data_census_2026-09-08.md`` section 3 counts 18 Mon-Fri dates with no bar at all, per
#: metal, over the WHOLE parquet (3,477 calendar days, through 2026-08-31). This file reads the
#: development window only (``_hold.development_window``, ending the day before
#: ``evaluation.holdout_start``), and there the same rule finds 17 - measured 2026-09-10. The
#: eighteenth lies inside the holdout and is not read. A count, not a bound.
CENSUS_MON_FRI_HOLES = 18
DEV_WINDOW_MON_FRI_HOLES = 17

#: Endpoint-less panel rows whose date the PACKAGE does not declare, per metal, over the whole
#: development window - measured by the first run of the tool on 2026-09-10 and pinned here with
#: that date, as the declaration asks ("<= the number recorded in BOT.md after the first run").
#: They are printed by name below and kept in the file as ``tape:*`` rows; this pin only stops
#: the set from growing in silence.
TAPE_ONLY_PINNED = {"XAUUSD": 15, "XAGUSD": 20}
TAPE_ONLY_PINNED_ON = "2026-09-10"

#: The package the pinned COUNTS below were measured against (``GEN2_DECLARATION_2026-09-10.md``
#: amendment 1, A2). A different package version or calendar id can legitimately move them (a
#: release that adds Juneteenth to ``CMES`` would), so every pinned count is asserted only when
#: the sidecar says the file was built from this package; otherwise the pinned assertions are
#: skipped with a message that says "re-measure and re-pin", never relaxed.
PINNED_PACKAGE = {"calendar_id": "CMES", "exchange_calendars_version": "4.13.2"}
PINNED_ON = "2026-09-10"
#: Test 1: of the 17 Mon-Fri holes a metal on the development window, the package declares
#: exactly 16 and the seventeenth is the 2018-02-01 feed outage - a tape fact, not a holiday.
HOLES_DECLARED_BY_PACKAGE = 16
HOLES_TAPE_ONLY = ["2018-02-01"]
#: Test 2: endpoint-less panel dates on a package date, a metal, over the development window.
ON_PACKAGE_DATE_PINNED = {"XAUUSD": 62, "XAGUSD": 62}
#: Test 3, amendment 1 (A1): once ``close_utc`` is the last bar AT OR BEFORE the New York close,
#: a ``tape:unresolved_dates`` row whose tape printed right up to that close lands exactly on
#: 17:00 New York - the instant the break begins. Such a row is NOT an early close: the endpoint
#: failed because a bar was missing INSIDE the session (``decision_grid`` "off_horizon"), which
#: ``unresolved_dates`` does not separate from "early_close". Measured 2026-09-10: exactly one,
#: XAGUSD on Memorial Day 2017 (bars 00-18, 20, 22, 23 UTC; no 19:00 bar), in the pre-fold 2017
#: regime no fold reaches. Pinned BY NAME so it cannot grow in silence; a second such row is a
#: finding to record, not a pin to widen.
AT_BREAK_PINNED = {("2017-05-29", "XAGUSD")}

#: Tokens whose presence in the tool's source would mean it read a return, a label or a P&L.
FORBIDDEN_TOKENS = ("fwd_ret", "dir_tb", "label_end_close", "y_true", "y_score", "pct_change",
                    "sharpe", "returns", "labels_df")


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bars(setup: dict) -> pl.DataFrame:
    if not (mt5_data_dir() / "1h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/1h.parquet)")
    lo, hi = development_window(setup)
    frame = (
        load_mt5_bars("1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi)
        .with_columns(pl.col("timestamp").dt.replace_time_zone(None).cast(pl.Datetime("us")))
    )
    assert_no_holdout(setup, frame)
    return frame


@pytest.fixture(scope="module")
def panel(bars: pl.DataFrame, setup: dict) -> pl.DataFrame:
    frame = session_panel(
        bars,
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
        keep_context=True,
        verbose=False,
    )
    assert_no_holdout(setup, frame)
    return frame


@pytest.fixture(scope="module")
def closures() -> pl.DataFrame:
    if not CLOSURE_CALENDAR_PATH.exists():
        pytest.skip(f"closure calendar not built: {CLOSURE_CALENDAR_PATH}")
    return declared_closures()


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("build_closure_calendar", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sidecar() -> dict:
    if not CLOSURE_CALENDAR_PATH.exists():
        pytest.skip(f"closure calendar not built: {CLOSURE_CALENDAR_PATH}")
    return json.loads(CLOSURE_CALENDAR_PATH.with_suffix(".parquet.digest.json").read_text())


def _built_from_the_pinned_package(sidecar: dict) -> bool:
    """True when the on-disk file came from the package the pinned counts were measured on."""
    return all(sidecar.get(k) == v for k, v in PINNED_PACKAGE.items())


def _package_dates(closures: pl.DataFrame, symbol: str, kind: str | None = None) -> set[date]:
    rows = closures.filter(
        (pl.col("symbol") == symbol) & pl.col("source").str.starts_with("exchange_calendars:")
    )
    if kind is not None:
        rows = rows.filter(pl.col("kind") == kind)
    return set(rows["date"].to_list())


def _any_dates(closures: pl.DataFrame, symbol: str, kind: str | None = None) -> set[date]:
    rows = closures.filter(pl.col("symbol") == symbol)
    if kind is not None:
        rows = rows.filter(pl.col("kind") == kind)
    return set(rows["date"].to_list())


# ---------------------------------------------------------------------------
# 1. the 18 Mon-Fri holes of the census are declared closures
# ---------------------------------------------------------------------------
def test_every_mon_fri_hole_of_the_census_is_a_declared_closure(
    bars: pl.DataFrame, setup: dict, closures: pl.DataFrame, sidecar: dict
) -> None:
    """Every Mon-Fri date with no bar is a ``closed`` row of the file; 17 a metal before the holdout.

    The holes are re-derived from the tape by the census rule, not read from the census, so the
    count is re-measured (18 in the census over the whole parquet, 17 on the development window).
    Which SOURCE declares each hole is printed: a hole the package declares is a closure a live
    consumer could have known; a hole only the tape declares is history.

    Pinned content (amendment 1, A2, measured 2026-09-10 on ``CMES`` / ``exchange_calendars
    4.13.2``): the package declares **16** of the 17 and the one it does not is **2018-02-01**,
    the two-day feed outage the declaration itself names. Before the pin this test only asked
    "is every hole declared by SOMEONE", which the ``tape:mon_fri_holes`` source makes true by
    construction - a tautology. The pinned assertions run only when the sidecar says the file was
    built from the pinned package (:data:`PINNED_PACKAGE`); otherwise they skip and say so.
    """
    pinned = _built_from_the_pinned_package(sidecar)
    lo, hi = development_window(setup)
    start, end = date.fromisoformat(lo), date.fromisoformat(hi)
    weekdays = {
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    }
    for symbol in sorted(setup["universe"]["symbols"]):
        present = set(bars.filter(pl.col("symbol") == symbol)["timestamp"].dt.date().to_list())
        holes = sorted(weekdays - present)
        declared = _any_dates(closures, symbol, "closed")
        by_package = _package_dates(closures, symbol, "closed")
        missing = [d for d in holes if d not in declared]
        print(
            f"\n{symbol}: {len(holes)} Mon-Fri holes; {sum(d in by_package for d in holes)} declared "
            f"by the package, {sum(d not in by_package for d in holes)} only by the tape: "
            f"{[str(d) for d in holes if d not in by_package]}"
        )
        assert len(holes) == DEV_WINDOW_MON_FRI_HOLES, (symbol, [str(d) for d in holes])
        assert not missing, f"{symbol}: holes with no closure row: {[str(d) for d in missing]}"
        if not pinned:
            continue
        assert sum(d in by_package for d in holes) == HOLES_DECLARED_BY_PACKAGE, (
            f"{symbol}: the package declares {sum(d in by_package for d in holes)} of the "
            f"{len(holes)} holes, not the {HOLES_DECLARED_BY_PACKAGE} pinned on {PINNED_ON}; "
            "re-measure and re-pin with a date, do not relax"
        )
        assert [str(d) for d in holes if d not in by_package] == HOLES_TAPE_ONLY, (
            f"{symbol}: the tape-only holes are {[str(d) for d in holes if d not in by_package]}, "
            f"not the {HOLES_TAPE_ONLY} pinned on {PINNED_ON}"
        )
    if not pinned:
        pytest.skip(
            f"the file was built from {sidecar.get('calendar_id')} / exchange_calendars "
            f"{sidecar.get('exchange_calendars_version')}, not the pinned {PINNED_PACKAGE}; the "
            "pinned counts are not a test of it - re-measure and re-pin"
        )


# ---------------------------------------------------------------------------
# 2. every endpoint-less panel row is declared, or named as tape-only
# ---------------------------------------------------------------------------
def test_early_close_rows_of_the_panel_are_declared_or_named(
    panel: pl.DataFrame, setup: dict, closures: pl.DataFrame, sidecar: dict
) -> None:
    """A row with ``label_end_ts`` null is on a package date, or it is printed as tape-only.

    Stale-decision sessions never reach the panel (``decision_grid`` drops them), so every
    endpoint-less row here is an early close or an off-horizon session. Each one must have a
    closure row for its (date, symbol); the ones the package does not know are named, and their
    count per metal may not exceed the pin recorded after the first run.

    Pinned content (amendment 1, A2, measured 2026-09-10 on the pinned package): **62** of the
    endpoint-less dates a metal sit on a package date. Without it the ``undeclared`` assertion is
    a tautology (every unresolved date is a ``tape:unresolved_dates`` row by construction) and
    the pin ``<=`` only bounds the tape-only set from above.
    """
    pinned = _built_from_the_pinned_package(sidecar)
    early = panel.filter(pl.col("label_end_ts").is_null()).with_columns(
        pl.col("timestamp").dt.date().alias("date")
    )
    assert early.height > 0, "the panel has no endpoint-less row; the census counted 163"
    for symbol in sorted(setup["universe"]["symbols"]):
        rows = early.filter(pl.col("symbol") == symbol).sort("timestamp")
        dates = sorted(set(rows["date"].to_list()))
        package = _package_dates(closures, symbol)
        anywhere = _any_dates(closures, symbol)
        undeclared = [d for d in dates if d not in anywhere]
        tape_only = [d for d in dates if d not in package]
        print(
            f"\n{symbol}: {rows.height} endpoint-less rows on {len(dates)} dates; "
            f"{len(dates) - len(tape_only)} on a package date; {len(tape_only)} tape-only "
            f"(pinned <= {TAPE_ONLY_PINNED[symbol]} on {TAPE_ONLY_PINNED_ON}):"
        )
        for d in tape_only:
            books = rows.filter(pl.col("date") == d)["session"].to_list()
            print(f"  {d} {d.strftime('%a')} {books}")
        assert not undeclared, f"{symbol}: endpoint-less dates with no closure row at all: {undeclared}"
        assert len(tape_only) <= TAPE_ONLY_PINNED[symbol], (
            f"{symbol}: {len(tape_only)} tape-only dates exceed the pin {TAPE_ONLY_PINNED[symbol]} "
            f"({TAPE_ONLY_PINNED_ON}); re-measure and re-record, do not relax"
        )
        if pinned:
            assert len(dates) - len(tape_only) == ON_PACKAGE_DATE_PINNED[symbol], (
                f"{symbol}: {len(dates) - len(tape_only)} endpoint-less dates on a package date, "
                f"not the {ON_PACKAGE_DATE_PINNED[symbol]} pinned on {PINNED_ON}; re-measure and "
                "re-pin with a date, do not relax"
            )
    if not pinned:
        pytest.skip(
            f"the file was built from {sidecar.get('calendar_id')} / exchange_calendars "
            f"{sidecar.get('exchange_calendars_version')}, not the pinned {PINNED_PACKAGE}; the "
            "pinned counts are not a test of it - re-measure and re-pin"
        )


# ---------------------------------------------------------------------------
# 3. the daily break is not a closure
# ---------------------------------------------------------------------------
def test_the_daily_break_is_not_a_closure(setup: dict, closures: pl.DataFrame) -> None:
    """No row of the file sits at the broker's daily break, 17:00 America/New_York.

    That break is 21:00-22:00 UTC under US DST and 22:00-23:00 UTC otherwise (``BOT.md`` kill
    criterion (g), refinement (i)); it is a declared state of the tape, not a closure, and the
    breaker must stay silent on it. The break is a LOCAL instant, so the assertion is made on the
    New York clock and not on a UTC hour (a 21:00 UTC Friday print in winter is 16:00 New York,
    one hour before the close, and is a tape fact this file may carry). Two assertions: no close
    instant equals the break instant of its date, and no row is a weekend (the file declares
    weekday closures only; the weekend is the schedule, not a closure).

    Amendment 1 (A1): ``close_utc`` of a tape row is now the last bar AT OR BEFORE the New York
    close, so a session whose tape ran right up to that close - an off-horizon session, not an
    early close - lands exactly on 17:00 New York. The one such row is pinned by name in
    :data:`AT_BREAK_PINNED` and printed; any other is a failure. The set is asserted EQUAL, so a
    row leaving the set is as loud as one joining it.
    """
    zone = ZoneInfo("America/New_York")
    with_close = closures.drop_nulls("close_utc")
    rows = with_close.to_dicts()
    local = [r["close_utc"].replace(tzinfo=UTC).astimezone(zone) for r in rows]
    at_break = {
        (str(r["date"]), r["symbol"])
        for r, ts in zip(rows, local, strict=True)
        if ts.time() == time(17, 0)
    }
    by_local_hour: dict[int, int] = {}
    for ts in local:
        by_local_hour[ts.hour] = by_local_hour.get(ts.hour, 0) + 1
    weekend = closures.filter(pl.col("date").dt.weekday() > 5)
    print(
        f"\n{with_close.height} rows carry a close instant; at 17:00 New York: {len(at_break)} "
        f"{sorted(at_break)} (pinned by name: {sorted(AT_BREAK_PINNED)}); "
        f"weekend rows: {weekend.height}; close instants by New York hour: "
        f"{dict(sorted(by_local_hour.items()))}"
    )
    assert at_break == AT_BREAK_PINNED, (
        f"rows at 17:00 New York {sorted(at_break)} != the pinned {sorted(AT_BREAK_PINNED)}: "
        "an unpinned row at the break is a closure the file must not claim; a pinned row that "
        "left is a tape change to record"
    )
    assert weekend.is_empty(), weekend
    assert set(closures["kind"].unique().to_list()) <= {"closed", "early_close"}


# ---------------------------------------------------------------------------
# 4. no closure row reads a price
# ---------------------------------------------------------------------------
def test_no_closure_row_reads_a_price(
    bars: pl.DataFrame, setup: dict, closures: pl.DataFrame, tool
) -> None:
    """The file is a function of the package calendar and of bar TIMESTAMPS, and of nothing else.

    Three checks. (a) The tool's source carries none of the tokens a return, a label or a P&L
    would travel under. (b) Rebuilding the frame from a tape whose open/high/low/close/volume
    have been multiplied, shifted and shuffled gives a frame identical to the one built from the
    real tape - a closure row that read a price would move. (c) The rebuilt frame, under the
    sidecar's own arguments, equals the file on disk row for row, so the artifact is reproducible
    and not edited.
    """
    source = TOOL_PATH.read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]  # the module docstring names the forbidden things on purpose
    hits = [t for t in FORBIDDEN_TOKENS if t in body]
    assert not hits, f"the tool's code mentions {hits}"

    sidecar = json.loads(CLOSURE_CALENDAR_PATH.with_suffix(".parquet.digest.json").read_text())
    reconciled_on = date.fromisoformat(sidecar["reconciled_on"])
    calendar_end = date.fromisoformat(sidecar["calendar_window"][1])
    calendar_id = sidecar["calendar_id"]

    real, _ = tool.build_closures(
        bars, setup, calendar_id=calendar_id, calendar_end=calendar_end, reconciled_on=reconciled_on
    )
    perturbed_bars = bars.with_columns(
        (pl.col("open") * 1.37 + 5.0),
        (pl.col("high") * 1.37 + 9.0),
        (pl.col("low") * 1.37 + 1.0),
        (pl.col("close") * 1.37 + 5.0),
        pl.col("volume").shuffle(seed=7),
    )
    perturbed, _ = tool.build_closures(
        perturbed_bars, setup, calendar_id=calendar_id, calendar_end=calendar_end,
        reconciled_on=reconciled_on,
    )
    assert real.equals(perturbed), "a closure row moved when every price on the tape moved"

    import exchange_calendars as xc

    print(
        f"\n{real.height} rows rebuilt from calendar {calendar_id} and bar timestamps; identical "
        f"under a price perturbation; exchange_calendars {xc.__version__} vs sidecar "
        f"{sidecar['exchange_calendars_version']}"
    )
    if xc.__version__ != sidecar["exchange_calendars_version"]:
        pytest.skip("package version differs from the sidecar's; the on-disk equality is not a test")
    on_disk = closures.select(CLOSURE_COLUMNS).sort(["symbol", "date", "source"])
    assert real.select(CLOSURE_COLUMNS).sort(["symbol", "date", "source"]).equals(on_disk), (
        "the file on disk is not what the tool builds from its two sources today"
    )
