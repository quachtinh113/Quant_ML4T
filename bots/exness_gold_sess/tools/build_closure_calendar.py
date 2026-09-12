"""Build the declared closure calendar of ``exness_gold_sess`` - generation 2, block 1.

``bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md`` section 1. The bot needs a holiday /
early-close calendar **with provenance** for three consumers: kill criterion (g) (a gap across
an UNDECLARED closure), the early-close residual of the New York book that
``NY_EXIT_DECLARATION.md`` section 6.3 deferred "until a calendar file exists", and the
Christmas-2024 carry (``BOT.md``, open questions). None of them may learn a closure from the
last bar never printing: that is the point-in-time defect ``_features.decision_grid`` documents.

**Two sources, and only two. No date in the output was typed by hand.**

1. ``exchange_calendars`` - the calendar package the repository pins (``pyproject.toml``,
   ``exchange-calendars>=4.5``). The calendar id is resolved the way the bot's fold ladder
   resolves it, through ``utils.cv_splits._map_calendar_id`` (``setup.yaml::evaluation.calendar``
   ``FX`` -> ``CME_FX``). That id is a *pandas_market_calendars* name (``_CALENDAR_MAP``'s own
   docstring says so, and ``ml4t.diagnostic.splitters.calendar`` opens it with
   ``pandas_market_calendars.get_calendar``), so ``exchange_calendars`` does not know it. The
   declaration foresees exactly this case: try the ``exchange_calendars`` ids of the same house
   (``CMES`` = CME Globex, ``XNYS``), choose by a **printed, mechanical** criterion, and record
   why. The criterion is written below (:func:`choose_calendar`) and is measured against the tape,
   never against a return.
2. The tape - ``ML4T_DATA_PATH/mt5/1h.parquet`` over the development window only
   (``_hold.assert_no_holdout`` refuses anything later). Two derived sets, both produced by code
   that already exists: the ``unresolved_dates`` that ``_features.decision_grid`` reports per
   symbol (a decision resolved, the endpoint did not: the broker closed early), and the Mon-Fri
   dates with no bar at all - the "Mon-Fri holes" of ``data_census_2026-09-08.md`` section 3.

**Tape wins for history, the calendar declares the future** (``setup.yaml::decision.
closure_calendar.tape_wins_for_history``). Every row says where it came from (``source``), so a
consumer can ask "declared by the package" or "seen on the tape" separately. A date the tape has
and the package does not is printed as **tape-only**, never filtered; a date the package declares
closed on which the tape traded is printed as **calendar-only**, never filtered either.

**What this file never reads**: a return, a label, a P&L. It reads bar *timestamps* (to know when
the last bar of a day closed) and nothing of open/high/low/close/volume;
``tests/test_calendar.py::test_no_closure_row_reads_a_price`` rebuilds the frame from a
price-perturbed tape and asserts it is identical.

Output: ``bots/exness_gold_sess/calendar/closures_metals.parquet`` with columns
``date, symbol, kind in {closed, early_close}, close_utc, source, source_version, reconciled_on``
and a ``.digest.json`` sidecar written by ``case_studies.utils.artifact_digest.write_artifact``.
Read it through ``case_studies.exness_gold_sess._features.declared_closures``.

Run (WSL2, from the repository root)::

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess uv run python \
        bots/exness_gold_sess/tools/build_closure_calendar.py --reconciled-on 2026-09-10

``ML4T_OUTPUT_DIR`` is only used to read the label timeline for the validation window of the
reconciliation print; without it the panel's own dates are used and that is printed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import exchange_calendars as xc  # noqa: E402

from bots._shared.mt5_loader import load_mt5_bars  # noqa: E402
from case_studies.exness_gold_sess._features import (  # noqa: E402
    BAR_MINUTES,
    _venue_window,
    decision_grid,
    session_panel,
)
from case_studies.exness_gold_sess._hold import (  # noqa: E402
    assert_no_holdout,
    development_window,
)
from case_studies.utils.artifact_digest import write_artifact  # noqa: E402
from utils.cv_splits import _map_calendar_id, generate_cv_splits  # noqa: E402
from utils.paths import get_case_study_dir  # noqa: E402

CASE_STUDY_ID = "exness_gold_sess"
BOT_DIR = REPO_ROOT / "bots" / CASE_STUDY_ID
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
CALENDAR_PATH = BOT_DIR / "calendar" / "closures_metals.parquet"

#: ``exchange_calendars`` ids of the same house as the fold ladder's ``CME_FX``, tried in this
#: order when that id is not an ``exchange_calendars`` id. ``CMES`` is CME Globex (the venue the
#: metals' hours follow, ``bots/assets/XAUUSD.md`` section 1); ``XNYS`` is the New York Stock
#: Exchange, the US-holiday calendar most early closes are keyed on.
#: HAND CONSTANT 1 of 2 (``GEN2_DECLARATION_2026-09-10.md``, amendment 1, A3): the ORDER of this
#: tuple is the tie-break of :func:`choose_calendar` ("ties go to the first candidate"). The
#: 2026-09-10 measurement was not a tie (CMES 2 contradicted closures vs XNYS 56), so the order
#: decided nothing; it is named so that a future tie is seen to be decided by a typed order.
CANDIDATE_IDS: tuple[str, ...] = ("CMES", "XNYS")
#: The venue whose session close bounds a date's ``last_close`` (amendment 1, A1). The New York
#: close is the start of the broker's daily break in both seasons (``_hold.py`` module docstring),
#: so the last bar closing at or before it is the last print BEFORE the break and never the
#: Globex re-open after it. The window is the one ``_features.session_of`` derives.
CLOSE_VENUE = "ny"
COLUMNS = ["date", "symbol", "kind", "close_utc", "source", "source_version", "reconciled_on"]
KINDS = ("closed", "early_close")
SOURCE_PACKAGE = "exchange_calendars"
SOURCE_TAPE_UNRESOLVED = "tape:unresolved_dates"
SOURCE_TAPE_HOLES = "tape:mon_fri_holes"
#: The clock whose 17:00 is both the New York session close and the broker's daily break
#: (``BOT.md`` kill criterion (g), refinement (i)). Used only to PRINT that no row sits there.
BREAK_ZONE = ZoneInfo("America/New_York")
BREAK_LOCAL = time(17, 0)

SCHEMA = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "kind": pl.Utf8,
    "close_utc": pl.Datetime("us"),
    "source": pl.Utf8,
    "source_version": pl.Utf8,
    "reconciled_on": pl.Date,
}

TRACKED_TABLES = (
    "backtest_runs",
    "backtest_metrics",
    "backtest_fold_metrics",
    "cohort_metrics",
    "candidate_sets",
    "candidate_set_members",
    "official_populations",
    "official_population_members",
    "prediction_sets",
    "training_runs",
)


# --------------------------------------------------------------------------- registry guard
def registry_counts(tag: str) -> dict[str, int]:
    """Same read-only count as ``tools/presweep_measurements.py::registry_counts``."""
    db = get_case_study_dir(CASE_STUDY_ID, create=False) / "run_log" / "registry.db"
    if not db.exists():
        print(f"[{tag}] no registry.db at {db}")
        return {}
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    out = {
        t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in TRACKED_TABLES
        if t in have
    }
    con.close()
    print(f"[{tag}] registry rows: {out}")
    return out


# --------------------------------------------------------------------------- the tape
def load_development_bars(setup: Mapping) -> pl.DataFrame:
    """The H1 tape of the declared universe over the development window, naive UTC."""
    lo, hi = development_window(setup)
    bars = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    )
    if bars.schema["timestamp"].time_zone is not None:
        bars = bars.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    bars = bars.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    assert_no_holdout(setup, bars)
    return bars


def tape_facts(bars: pl.DataFrame, setup: Mapping) -> dict:
    """Everything the tape contributes, per symbol - timestamps only, no price.

    ``unresolved_dates`` and ``dropped_dates`` come from :func:`decision_grid`'s own report
    (``_features.py``, ``report["per_symbol"]``); ``holes`` are the Mon-Fri dates of the
    development window with no bar at all, the census section-3 rule; ``last_close`` is the close
    instant of the last bar printed on each date **at or before the New York venue close of that
    date** (a timestamp, not a price); ``bars_on`` counts bars per date so a calendar-declared
    closure the tape traded through can be shown with its size.

    **Amendment 1 (2026-09-10), A1.** ``last_close`` used to be ``max(bar_close)`` of the date,
    which on 104 of 159 ``tape:unresolved_dates`` rows was the 23:00 UTC bar printed AFTER Globex
    re-opened (Juneteenth 2024-06-19 -> ``2024-06-20 00:00``). An early close is the last print
    before the break, so the bound is the New York session close of the date - the same window
    ``_features.session_of`` derives through ``_venue_window`` - and a bar after it is not the
    day's close. Only timestamps are read; a date with no bar at or before that close carries a
    null ``last_close`` and is counted in the reconciliation print.
    """
    lo, hi = development_window(setup)
    start, end = date.fromisoformat(lo), date.fromisoformat(hi)
    weekdays = {
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    }
    _, report = decision_grid(
        bars,
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
        verbose=False,
    )
    stamped = bars.select(
        "symbol",
        pl.col("timestamp").dt.date().alias("date"),
        (pl.col("timestamp") + pl.duration(minutes=BAR_MINUTES)).alias("bar_close"),
    )
    # the New York venue close of every date on the tape, read from the venue's own clock
    dates = sorted(stamped["date"].unique().to_list())
    venue_close = pl.DataFrame(
        {"date": dates, "venue_close": [_venue_window(d, CLOSE_VENUE)[1] for d in dates]},
        schema={"date": pl.Date, "venue_close": pl.Datetime("us")},
    )
    counts = stamped.group_by(["symbol", "date"]).agg(pl.len().alias("bars_on"))
    last_before_close = (
        stamped.join(venue_close, on="date", how="left")
        .filter(pl.col("bar_close") <= pl.col("venue_close"))
        .group_by(["symbol", "date"])
        .agg(pl.col("bar_close").max().alias("last_close"))
    )
    by_day = counts.join(last_before_close, on=["symbol", "date"], how="left")
    facts: dict[str, dict] = {}
    for symbol in sorted(bars["symbol"].unique().to_list()):
        rows = by_day.filter(pl.col("symbol") == symbol)
        present = set(rows["date"].to_list())
        per = report["per_symbol"][symbol]
        facts[symbol] = {
            "holes": sorted(weekdays - present),
            "unresolved": sorted(date.fromisoformat(d) for d in per["unresolved_dates"]),
            "dropped": sorted(date.fromisoformat(d) for d in per["dropped_dates"]),
            "last_close": dict(zip(rows["date"].to_list(), rows["last_close"].to_list())),
            "bars_on": dict(zip(rows["date"].to_list(), rows["bars_on"].to_list())),
            "counts": {k: v for k, v in per.items() if isinstance(v, int)},
        }
    tape_version = (
        f"mt5_1h@{bars['timestamp'].min().date()}..{bars['timestamp'].max().date()}"
    )
    return {"per_symbol": facts, "window": (lo, hi), "tape_version": tape_version}


# --------------------------------------------------------------------------- the package
def open_calendar(calendar_id: str, start: date, end: date):
    return xc.get_calendar(calendar_id, start=str(start), end=str(end))


def package_facts(cal, start: date, end: date) -> dict:
    """Weekday non-sessions (``closed``) and early closes with their UTC close, from the package."""
    import pandas as pd

    sessions = set(cal.sessions.normalize())
    days = pd.bdate_range(str(start), str(end))
    closed = sorted(d.date() for d in days if d not in sessions)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    early = [d for d in cal.early_closes if lo <= d <= hi]
    schedule = cal.schedule.loc[early, "close"] if early else pd.Series(dtype="datetime64[ns, UTC]")
    early_close_utc = {
        d.date(): ts.tz_convert("UTC").tz_localize(None).to_pydatetime()
        for d, ts in schedule.items()
    }
    return {"closed": closed, "early_close_utc": early_close_utc}


def choose_calendar(
    fold_ladder_id: str, tape: dict, start: date, end: date, *, symbols: Sequence[str]
) -> tuple[str, object, dict, list[str]]:
    """Resolve the id the fold ladder uses; if ``exchange_calendars`` lacks it, try the same house.

    The choice among candidates is mechanical and printed: the candidate whose ``closed`` days in
    the development window are **least contradicted by the tape** (a declared closure on which
    the broker printed a full session) wins; ties go to the first candidate. The tape is the truth
    for history, so a calendar that declares the metals shut on days they quoted all day is the
    wrong calendar for these instruments, whatever its early-close list looks like. No return is
    read to decide this.
    """
    notes: list[str] = []
    tried: list[str] = []
    try:
        cal = open_calendar(fold_ladder_id, start, end)
        notes.append(f"{fold_ladder_id!r} opened in exchange_calendars as-is")
        return fold_ladder_id, cal, package_facts(cal, start, end), notes
    except xc.errors.InvalidCalendarName as exc:
        notes.append(
            f"{fold_ladder_id!r} is not an exchange_calendars id ({exc}); it is the "
            "pandas_market_calendars name ml4t.diagnostic opens for the fold ladder"
        )
    scored: list[tuple[int, int, str, object, dict]] = []
    dev_hi = min(end, max(date.fromisoformat(tape["window"][1]), start))
    for cid in CANDIDATE_IDS:
        cal = open_calendar(cid, start, end)
        facts = package_facts(cal, start, end)
        closed_dev = [d for d in facts["closed"] if d <= dev_hi]
        contradicted = 0
        for d in closed_dev:
            # a "full session" is a weekday the tape printed at least half its usual bars on.
            # HAND CONSTANT 2 of 2 (amendment 1, A3): 12 = half of the 23-24 H1 bars of a
            # weekday. Measured 2026-09-10: the two contradicted CMES dates (2018-12-05,
            # 2025-01-09) carry a full weekday of bars and the eight other package closures the
            # tape touched carry exactly ONE bar, so every threshold in 2..22 gives the same 2.
            traded = all(tape["per_symbol"][s]["bars_on"].get(d, 0) >= 12 for s in symbols)
            contradicted += int(traded)
        early_dev = [d for d in facts["early_close_utc"] if d <= dev_hi]
        matched = sum(
            1
            for d in early_dev
            if all(d in tape["per_symbol"][s]["unresolved"] for s in symbols)
        )
        tried.append(
            f"  {cid:5s}: closed weekdays in dev {len(closed_dev):3d}, of which the tape traded a "
            f"full session on {contradicted:3d}; early closes in dev {len(early_dev):3d}, of which "
            f"both metals' endpoints are unresolved on the tape {matched:3d}"
        )
        scored.append((contradicted, -matched, cid, cal, facts))
    scored.sort(key=lambda t: (t[0], t[1]))
    contradicted, neg_matched, cid, cal, facts = scored[0]
    notes.append("candidates of the same house, measured against the tape (no return read):")
    notes.extend(tried)
    notes.append(
        f"chosen {cid!r}: fewest tape-contradicted closures ({contradicted}), then most "
        f"early closes the tape confirms ({-neg_matched})"
    )
    return cid, cal, facts, notes


# --------------------------------------------------------------------------- the frame
def build_closures(
    bars: pl.DataFrame,
    setup: Mapping,
    *,
    calendar_id: str | None = None,
    calendar_end: date | None = None,
    reconciled_on: date,
) -> tuple[pl.DataFrame, dict]:
    """The closure frame from the two sources, plus the facts the reconciliation prints.

    Pure of prices: the only thing read off a bar is its timestamp. ``calendar_id`` ``None``
    means "resolve through the fold ladder's map and choose among the same house";
    ``calendar_end`` defaults to the end of the year after ``reconciled_on``, because the file is
    also the declaration a live breaker reads about days that have not happened.
    """
    symbols = sorted(setup["universe"]["symbols"])
    tape = tape_facts(bars, setup)
    start = date.fromisoformat(str(setup["universe"]["history_start"]))
    calendar_end = calendar_end or date(reconciled_on.year + 1, 12, 31)
    fold_ladder_id = _map_calendar_id(setup["evaluation"]["calendar"])
    if calendar_id is None:
        chosen, cal, facts, notes = choose_calendar(
            fold_ladder_id, tape, start, calendar_end, symbols=symbols
        )
    else:
        cal = open_calendar(calendar_id, start, calendar_end)
        chosen, facts, notes = calendar_id, package_facts(cal, start, calendar_end), [
            f"calendar id {calendar_id!r} given explicitly"
        ]
    package_version = f"{SOURCE_PACKAGE}:{chosen}@{xc.__version__}"

    rows: list[dict] = []
    for symbol in symbols:
        for d in facts["closed"]:
            rows.append(
                {
                    "date": d, "symbol": symbol, "kind": "closed", "close_utc": None,
                    "source": package_version, "source_version": xc.__version__,
                    "reconciled_on": reconciled_on,
                }
            )
        for d, close_utc in facts["early_close_utc"].items():
            rows.append(
                {
                    "date": d, "symbol": symbol, "kind": "early_close", "close_utc": close_utc,
                    "source": package_version, "source_version": xc.__version__,
                    "reconciled_on": reconciled_on,
                }
            )
        per = tape["per_symbol"][symbol]
        for d in per["unresolved"]:
            rows.append(
                {
                    "date": d, "symbol": symbol, "kind": "early_close",
                    "close_utc": per["last_close"][d], "source": SOURCE_TAPE_UNRESOLVED,
                    "source_version": tape["tape_version"], "reconciled_on": reconciled_on,
                }
            )
        for d in per["holes"]:
            rows.append(
                {
                    "date": d, "symbol": symbol, "kind": "closed", "close_utc": None,
                    "source": SOURCE_TAPE_HOLES, "source_version": tape["tape_version"],
                    "reconciled_on": reconciled_on,
                }
            )
    frame = (
        pl.DataFrame(rows, schema=SCHEMA)
        .select(COLUMNS)
        .unique(subset=["date", "symbol", "source"], keep="first")
        .sort(["symbol", "date", "source"])
    )
    assert frame["kind"].is_in(KINDS).all()
    assert frame.select(["date", "symbol", "source"]).is_duplicated().sum() == 0
    return frame, {
        "calendar_id": chosen,
        "fold_ladder_id": fold_ladder_id,
        "package_version": package_version,
        "calendar_window": (str(start), str(calendar_end)),
        "notes": notes,
        "tape": tape,
        "package": facts,
    }


# --------------------------------------------------------------------------- reconciliation
def validation_window(setup: Mapping, panel: pl.DataFrame) -> tuple[str, str, str]:
    """``(val_lo, val_hi, timeline_source)`` from the fold ladder, exactly as the sweep saw it.

    The folds are generated from the primary label's date timeline when the experiment holds one
    (the same call ``tools/presweep_measurements.py`` makes; only the ``timestamp`` column is
    scanned, no value), and from the panel's own dates otherwise - equal on the development
    window by ``tests/test_fold_geometry.py::test_label_key_set_equals_the_decision_grid``.
    """
    import pandas as pd

    primary = setup["labels"]["primary"]
    path = get_case_study_dir(CASE_STUDY_ID, create=False) / "labels" / f"{primary}.parquet"
    if path.exists():
        timeline = (
            pl.scan_parquet(path)
            .select(pl.col("timestamp").dt.date().alias("timestamp"))
            .unique()
            .sort("timestamp")
            .collect()
        )
        source = f"labels/{primary}.parquet timeline (dates only)"
    else:
        timeline = (
            panel.select(pl.col("timestamp").dt.date().alias("timestamp")).unique().sort("timestamp")
        )
        source = "session panel dates (no experiment labels found)"
    folds = list(
        generate_cv_splits(
            timeline,
            case_study_id=CASE_STUDY_ID,
            label_buffer=setup["labels"]["buffer"],
            date_col="timestamp",
        )
    )
    val_lo = min(str(pd.Timestamp(f["val_start"]).date()) for f in folds)
    val_hi = max(str(pd.Timestamp(f["val_end"]).date()) for f in folds)
    return val_lo, val_hi, source


def _status(d: date, symbol: str, frame: pl.DataFrame) -> str:
    rows = frame.filter((pl.col("date") == d) & (pl.col("symbol") == symbol))
    pkg = rows.filter(pl.col("source").str.starts_with(SOURCE_PACKAGE))
    if pkg.height:
        return f"calendar:{pkg['kind'][0]}"
    if rows.height:
        return "TAPE-ONLY"
    return "ABSENT"


def reconcile(frame: pl.DataFrame, info: dict, setup: Mapping, panel: pl.DataFrame) -> dict:
    """Print the comparison the declaration requires and return its counts for the sidecar."""
    symbols = sorted(setup["universe"]["symbols"])
    tape = info["tape"]
    out: dict = {"holes": {}, "early_close_validation": {}, "tape_only": {}, "calendar_only": {}}
    print("\n=== 1. The Mon-Fri holes of the census (section 3) against the package calendar ===")
    for symbol in symbols:
        holes = tape["per_symbol"][symbol]["holes"]
        statuses = {d: _status(d, symbol, frame) for d in holes}
        matched = sum(s.startswith("calendar") for s in statuses.values())
        out["holes"][symbol] = {"n": len(holes), "in_package": matched}
        print(f"{symbol}: {len(holes)} holes, {matched} declared by the package")
        for d, s in statuses.items():
            print(f"  {d} {d.strftime('%a')}  {s}")

    print("\n=== 2. Package closures the tape traded through (calendar-only, kept, printed) ===")
    dev_hi = date.fromisoformat(tape["window"][1])
    for symbol in symbols:
        bars_on = tape["per_symbol"][symbol]["bars_on"]
        odd = [
            (d, bars_on[d])
            for d in info["package"]["closed"]
            if d <= dev_hi and bars_on.get(d, 0) > 0
        ]
        out["calendar_only"][symbol] = [str(d) for d, _ in odd]
        print(f"{symbol}: {len(odd)} package-closed weekdays with bars on the tape")
        for d, n in odd:
            print(f"  {d} {d.strftime('%a')}  {n} bars, last close {tape['per_symbol'][symbol]['last_close'][d]}")

    val_lo, val_hi, timeline_source = validation_window(setup, panel)
    print(f"\n=== 3. Early closes: New York endpoint-less decisions in the validation window ===")
    print(f"validation window {val_lo} -> {val_hi} from {timeline_source}")
    early = panel.filter(pl.col("label_end_ts").is_null()).with_columns(
        pl.col("timestamp").dt.date().alias("date")
    )
    for symbol in symbols:
        rows = early.filter(
            (pl.col("symbol") == symbol)
            & (pl.col("date") >= date.fromisoformat(val_lo))
            & (pl.col("date") <= date.fromisoformat(val_hi))
        ).sort("timestamp")
        statuses = [(r["date"], r["session"], _status(r["date"], symbol, frame)) for r in rows.to_dicts()]
        matched = sum(s.startswith("calendar") for _, _, s in statuses)
        named = [(d, v, s) for d, v, s in statuses if not s.startswith("calendar")]
        books = rows.group_by("session").len().sort("session").to_dicts()
        out["early_close_validation"][symbol] = {
            "n": rows.height, "in_package": matched, "tape_only": [str(d) for d, _, _ in named],
        }
        print(f"{symbol}: {rows.height} endpoint-less rows ({books}); {matched} on a package date, "
              f"{len(named)} tape-only")
        for d, v, s in statuses:
            print(f"  {d} {d.strftime('%a')} {v:6s} {s}")

    print("\n=== 4. Tape-only dates over the whole development window (kept, printed) ===")
    for symbol in symbols:
        per = tape["per_symbol"][symbol]
        only = sorted(
            d for d in set(per["unresolved"]) | set(per["holes"])
            if _status(d, symbol, frame) == "TAPE-ONLY"
        )
        out["tape_only"][symbol] = [str(d) for d in only]
        print(f"{symbol}: {len(only)} tape-only dates of {len(per['unresolved'])} unresolved + "
              f"{len(per['holes'])} holes")
        for d in only:
            kind = "closed (hole)" if d in per["holes"] else f"early_close, last bar close {per['last_close'][d]}"
            print(f"  {d} {d.strftime('%a')}  {kind}")
        print(f"  stale-decision dates (dropped by decision_grid, not closures): "
              f"{[str(d) for d in per['dropped']]}")

    print("\n=== 5. The two named outage dates ===")
    for d in (date(2018, 1, 31), date(2018, 2, 1)):
        print(f"  {d}: " + ", ".join(f"{s} {_status(d, s, frame)}" for s in symbols))

    print("\n=== 6. No row at the daily break ===")
    with_close = frame.drop_nulls("close_utc")
    at_break = [
        r for r in with_close.to_dicts()
        if r["close_utc"].replace(tzinfo=UTC).astimezone(BREAK_ZONE).time() == BREAK_LOCAL
    ]
    print(f"  rows with a close instant: {with_close.height}; at 17:00 America/New_York: {len(at_break)}")
    out["rows_at_daily_break"] = len(at_break)
    out["validation_window"] = [val_lo, val_hi]

    print(f"\n=== 7. close_utc of the {SOURCE_TAPE_UNRESOLVED} rows by UTC hour (amendment 1, A1) ===")
    unresolved_rows = frame.filter(pl.col("source") == SOURCE_TAPE_UNRESOLVED)
    by_hour = (
        unresolved_rows.drop_nulls("close_utc")
        .group_by(pl.col("close_utc").dt.hour().alias("utc_hour"))
        .len()
        .sort("utc_hour")
    )
    hours = {int(r["utc_hour"]): int(r["len"]) for r in by_hour.to_dicts()}
    null_close = unresolved_rows.filter(pl.col("close_utc").is_null()).height
    fridays_21 = unresolved_rows.filter(
        (pl.col("close_utc").dt.hour() == 21) & (pl.col("date").dt.weekday() == 5)
    )
    print(f"  {unresolved_rows.height} rows: by UTC hour {hours}; null close_utc {null_close}; "
          f"Friday rows at 21:00 UTC {fridays_21.height} "
          f"({fridays_21['date'].min()} -> {fridays_21['date'].max()})" if fridays_21.height else
          f"  {unresolved_rows.height} rows: by UTC hour {hours}; null close_utc {null_close}; "
          "Friday rows at 21:00 UTC 0")
    out["unresolved_close_utc_by_utc_hour"] = {str(k): v for k, v in hours.items()}
    out["unresolved_null_close_utc"] = null_close
    return out


# --------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reconciled-on", type=date.fromisoformat, default=date.today())
    parser.add_argument("--calendar-id", default=None, help="skip the resolution and use this id")
    parser.add_argument("--calendar-end", type=date.fromisoformat, default=None)
    parser.add_argument("--output", type=Path, default=CALENDAR_PATH)
    args = parser.parse_args(argv)

    setup = yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))
    before = registry_counts("before")
    print(f"exchange_calendars {xc.__version__}; fold-ladder calendar "
          f"{setup['evaluation']['calendar']!r} -> {_map_calendar_id(setup['evaluation']['calendar'])!r}")
    try:
        import pandas_market_calendars as pmc

        print(f"pandas_market_calendars {pmc.__version__} (what ml4t.diagnostic opens the fold "
              "ladder's id with; NOT a source of this file)")
    except ImportError:
        pass

    bars = load_development_bars(setup)
    panel = session_panel(
        bars,
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
        keep_context=True,
        verbose=False,
    )
    assert_no_holdout(setup, panel)
    print(f"tape: {bars.height:,} H1 bars {bars['timestamp'].min()} -> {bars['timestamp'].max()}; "
          f"panel {panel.height:,} decision rows, {panel['label_end_ts'].null_count()} without an endpoint")

    frame, info = build_closures(
        bars, setup, calendar_id=args.calendar_id, calendar_end=args.calendar_end,
        reconciled_on=args.reconciled_on,
    )
    print("\n=== calendar resolution ===")
    for line in info["notes"]:
        print(line)
    print(f"source for package rows: {info['package_version']}; calendar window "
          f"{info['calendar_window'][0]} -> {info['calendar_window'][1]}")
    print(frame.group_by(["source", "kind"]).len().sort(["source", "kind"]))

    counts = reconcile(frame, info, setup, panel)
    record = write_artifact(
        frame,
        args.output,
        keys=["date", "symbol", "source"],
        written_by="bots/exness_gold_sess/tools/build_closure_calendar.py",
        inputs={},
        metadata={
            "calendar_id": info["calendar_id"],
            "fold_ladder_calendar_id": info["fold_ladder_id"],
            "exchange_calendars_version": xc.__version__,
            "calendar_window": list(info["calendar_window"]),
            "tape_window": list(info["tape"]["window"]),
            "tape_version": info["tape"]["tape_version"],
            "reconciled_on": str(args.reconciled_on),
            "resolution_notes": info["notes"],
            "reconciliation": counts,
        },
    )
    print(f"\nwrote {args.output} : {record['n_rows']:,} rows, digest {record['digest']}")
    after = registry_counts("after")
    assert before == after, f"registry moved: {before} != {after}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
