"""Point-in-time test for the exness_gold_sess feature panel (roadmap phase 2 gate).

Every feature must be recomputable at the decision instant from the bars that had closed by
then, and the result must equal the batch panel `03_financial_features` writes. The test drives
the same functions the stages call (`case_studies/exness_gold_sess/_features.py`) - there is no
second implementation here to agree with the first on the day it is written and drift after.

What it holds this bot to, one test each:

a. the session-close endpoint on the panel reproduces the sealed ``fwd_ret_8h``, max diff 0.0;
b. the feature matrix rebuilt from only the H1 and D1 bars that had closed at each of eight
   sampled decision instants - both venues, both daylight-saving seasons - equals the batch row
   on every column, max diff 0.0;
c. withholding every bar after a cut date moves no value before it;
d. the warmup census matches the feature register, on **each grid in its own unit**;
e. the batch build reproduces ``features/financial.parquet`` once ``03`` has written it;
f. every decision bar closes at open + 60 minutes on its own session date, and every resolved
   label endpoint closes at or before the session close, in both seasons;
g. the D1 asof join never reads a D1 bar whose close is after the decision instant, and the age
   of the bar it does read is 8-9 hours at a London decision and 13-14 at a New York one;
h. perturbation is symmetric: a bar printed AFTER the decision moves nothing, and a bar printed
   between the session open and the decision moves ONLY the in-session columns.

Two guards that matter more than they look. ``features_as_of`` is asserted to return a non-empty
frame before anything is compared: a comparison of two empty frames passes and proves nothing,
and before the decision/endpoint split in ``_features.decision_grid`` this function could only
ever return zero rows. And every read stops at the day before ``evaluation.holdout_start``; the
holdout is not loaded by this file.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests/test_lookahead.py -q
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import yaml

pytest.importorskip("ml4t.engineer", reason="the D1 families need the full project environment")

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir  # noqa: E402
from bots._shared.sessions import SESSIONS  # noqa: E402
from case_studies.exness_gold_sess._features import (  # noqa: E402
    BAR_MINUTES,
    VENUES,
    bars_closed_by,
    build_features,
    feature_columns,
    features_as_of,
    session_panel,
    warmup_expectations,
)
from case_studies.utils.feature_engineering import assert_values_agree, warmup_audit  # noqa: E402
from utils.paths import REPO_ROOT, get_case_study_dir  # noqa: E402

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
#: Decision instants sampled per (venue, daylight-saving season) cell. Four cells, so eight
#: instants - above the five the phase-2 gate asks for, and spread across both seasons and both
#: venues on purpose: the UTC hour of a decision moves with the venue's own DST, so a sample
#: taken from one season would never exercise the other.
N_PER_CELL = 2
IN_SESSION_COLUMNS = {"sess_range_1h", "sess_close_pos_1h"}


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_end(setup: dict) -> date:
    """The last calendar day this file may read: the day before the holdout opens."""
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
def d1(setup: dict, dev_end: date) -> pl.DataFrame:
    # From the first D1 bar the account serves, not from universe.history_start: that head start
    # is what pays for a 315-bar chain without moving the H1 panel's first row.
    return load_mt5_bars(
        "daily",
        symbols=sorted(setup["universe"]["symbols"]),
        start_date=str(setup["features"]["grid"]["long_window_history_start"]),
        end_date=str(dev_end),
    )


@pytest.fixture(scope="module")
def d1_reference(setup: dict, dev_end: date) -> pl.DataFrame | None:
    extra = sorted(set(setup["features"]["reference_symbols"]) - set(setup["universe"]["symbols"]))
    if not extra:
        return None
    return load_mt5_bars(
        "daily",
        symbols=extra,
        start_date=str(setup["features"]["grid"]["long_window_history_start"]),
        end_date=str(dev_end),
    )


@pytest.fixture(scope="module")
def panel(bars: pl.DataFrame, setup: dict) -> pl.DataFrame:
    return session_panel(
        bars,
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
        keep_context=True,
        verbose=False,
    )


@pytest.fixture(scope="module")
def batch(
    panel: pl.DataFrame, bars: pl.DataFrame, d1: pl.DataFrame, d1_reference, setup: dict
) -> pl.DataFrame:
    return build_features(
        panel, bars, d1, setup["features"]["windows"], d1_reference=d1_reference
    )


def _is_summer(day: date, venue: str) -> bool:
    """True when the venue is on daylight saving on that date. Read from the zone database."""
    zone = ZoneInfo(SESSIONS[VENUES[venue]].zone)
    return bool(datetime(day.year, day.month, day.day, 12, tzinfo=zone).dst())


def _sampled_instants(batch: pl.DataFrame, setup: dict) -> list[tuple[str, bool, datetime]]:
    """Decision instants spread over both venues and both seasons, evenly through the sample.

    The carrier column is required to be filled, so an instant inside the warmup - where a
    comparison would be a comparison of nulls - cannot be picked.
    """
    carrier = setup["features"]["null_policy_carrier"]
    dense = (
        batch.filter(pl.col(carrier).is_not_null())
        .select("timestamp", "session")
        .unique()
        .sort("timestamp")
    )
    picked: list[tuple[str, bool, datetime]] = []
    for venue in VENUES:
        rows = dense.filter(pl.col("session") == venue)["timestamp"].to_list()
        for summer in (True, False):
            season = [ts for ts in rows if _is_summer(ts.date(), venue) is summer]
            assert season, f"no {venue} decision in the {'summer' if summer else 'winter'} season"
            for index in np.linspace(0, len(season) - 1, N_PER_CELL + 2)[1:-1].round().astype(int):
                picked.append((venue, summer, season[int(index)]))
    return sorted(picked, key=lambda row: row[2])


# ---------------------------------------------------------------------------
# (a) the panel's endpoint is the price the labels were sealed on
# ---------------------------------------------------------------------------
def test_session_close_reproduces_the_sealed_label(panel: pl.DataFrame, setup: dict) -> None:
    """The panel's session-close endpoint reproduces the sealed ``fwd_ret_8h``.

    Generation 2 (2026-09-10): the primary label is ``fwd_ret_sess``, sealed at the last
    TRADABLE bar at or before the session close, so "the session close reproduces the label" is
    a proposition about ``fwd_ret_8h`` by name - the generation-1 parquet that stays on disk,
    unrepublished - and not about ``labels.primary``. The primary's own seal is
    ``tests/test_labels_gen2.py::test_the_new_label_is_sealed_at_its_own_endpoint``.
    """
    label_name = "fwd_ret_8h"
    path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{label_name}.parquet"
    if not path.exists():
        pytest.skip(f"labels not built yet: {path}")
    labels = pl.read_parquet(path)
    implied = panel.drop_nulls("label_end_ts").with_columns(
        (pl.col("label_end_close") / pl.col("close") - 1).alias("_implied")
    )
    joined = implied.join(labels, on=["symbol", "timestamp"], how="inner")
    assert joined.height == implied.height, (
        "a development row with a resolved endpoint has no sealed label"
    )
    # RE-STATED AND RE-MEASURED 2026-09-08 (RULINGS_2026-09-08.md execution step 8). This used to
    # be a SEMI join asserting that an endpoint-less row has no ROW in the label parquet. A-prime
    # (FOLD_GEOMETRY_DECLARATION.md) deliberately changed that: every label is now published on
    # the decision grid's key set with a null where its own rule is undefined, so those rows exist
    # by design - 163 of them, exactly the count the decision/endpoint split added. The
    # proposition that has to hold is about the VALUE, not the row, and it is the one the
    # assertion message always named. Measured today: 163 endpoint-less panel rows, all 163
    # present as rows in labels/fwd_ret_8h.parquet, and 0 carrying a non-null value.
    unlabelled = panel.filter(pl.col("label_end_ts").is_null()).join(
        labels, on=["symbol", "timestamp"], how="inner"
    )
    carries_a_value = unlabelled.filter(pl.col(label_name).is_not_null())
    assert carries_a_value.is_empty(), "a row without an endpoint carries a sealed session return"
    gap = (joined["_implied"] - joined[label_name]).abs().max()
    print(
        f"\n{joined.height:,} development rows compared with the sealed {label_name}, "
        f"max |diff| {gap:.3e}; {panel['label_end_ts'].null_count()} rows have no endpoint, "
        f"{unlabelled.height} of them published as null-valued rows under A-prime and "
        f"{carries_a_value.height} carrying a value"
    )
    assert gap == 0.0


# ---------------------------------------------------------------------------
# (b) rebuilding at the decision instant
# ---------------------------------------------------------------------------
def test_features_recomputed_at_the_decision_instant(
    bars: pl.DataFrame, d1: pl.DataFrame, d1_reference, batch: pl.DataFrame, setup: dict
) -> None:
    """Truncating the raw bars at the instant reproduces the batch row on every column.

    The truncation is on the H1 and D1 bars, so it tests the session aggregation and the
    two-grid join as well as the arithmetic: a panel that absorbed a bar closing after its own
    decision, or a D1 join that reached to the calendar date rather than the bar close, would
    both fail here and nowhere else.
    """
    cols = feature_columns(batch)
    rows = []
    for venue, summer, instant in _sampled_instants(batch, setup):
        as_of = features_as_of(
            instant, bars, d1, setup["features"]["windows"], d1_reference=d1_reference
        )
        # The guard the first version of this module could not pass. An empty frame compares
        # equal to an empty frame and proves nothing.
        assert as_of.height == len(setup["universe"]["symbols"]), (
            f"features_as_of returned {as_of.height} rows at {instant}; the decision row must "
            "exist at its own instant even though the session's endpoint has not printed"
        )
        assert as_of["label_end_ts"].null_count() == as_of.height, (
            "the session endpoint is eight hours in the future of the decision and must be null"
        )
        live = as_of.sort("symbol")
        ref = batch.filter(pl.col("timestamp") == instant).sort("symbol")
        assert ref.height == live.height
        census = assert_values_agree(ref, live, columns=cols, keys=["timestamp", "symbol"])
        rows.append(
            (
                venue,
                "summer" if summer else "winter",
                str(instant),
                bars_closed_by(bars, instant).height,
                float(census["max abs difference"].max()),
            )
        )
    print(f"\nrebuilt at {len(rows)} decision instants x {len(cols)} columns:")
    for row in rows:
        print("  ", *row)
    assert {row[0] for row in rows} == set(VENUES), "a venue was not sampled"
    assert {row[1] for row in rows} == {"summer", "winter"}, "a season was not sampled"


# ---------------------------------------------------------------------------
# (c) withholding later dates
# ---------------------------------------------------------------------------
def test_withholding_later_bars_changes_nothing_before_the_cut(
    bars: pl.DataFrame, d1: pl.DataFrame, d1_reference, batch: pl.DataFrame, setup: dict
) -> None:
    """A transform fitted across the sample would move when the last years are removed."""
    cut = datetime(2022, 1, 1)
    withheld = build_features(
        session_panel(
            bars.filter(pl.col("timestamp") < cut), keep_context=True, verbose=False
        ),
        bars.filter(pl.col("timestamp") < cut),
        d1.filter(pl.col("timestamp") < cut.date()),
        setup["features"]["windows"],
        d1_reference=(
            None if d1_reference is None else d1_reference.filter(pl.col("timestamp") < cut.date())
        ),
    )
    # The last decision of the truncated build has no endpoint and its neighbours' slot returns
    # are unaffected, but the final session's own row is the one the truncation created, so the
    # comparison stops one slot short of the cut on each symbol.
    last = withheld.group_by("symbol").agg(pl.col("timestamp").max().alias("_last"))
    comparable = withheld.join(last, on="symbol").filter(pl.col("timestamp") < pl.col("_last"))
    census = assert_values_agree(
        batch.join(comparable.select("timestamp", "symbol"), on=["timestamp", "symbol"]),
        comparable.drop("_last"),
        columns=feature_columns(batch),
        keys=["timestamp", "symbol"],
    )
    print(
        f"\n{comparable.height:,} rows before {cut.date()} rebuilt without any later bar, "
        f"max |diff| over {len(feature_columns(batch))} columns "
        f"{float(census['max abs difference'].max()):.3e}"
    )


# ---------------------------------------------------------------------------
# (d) the warmup census against the register
# ---------------------------------------------------------------------------
def test_warmup_matches_the_register(
    batch: pl.DataFrame, d1: pl.DataFrame, d1_reference, setup: dict
) -> None:
    """Each grid audited in its OWN unit; auditing the D1 columns on the panel is meaningless.

    A ``d1_*`` column's warmup is counted in D1 bars. The D1 frame starts three years before the
    H1 panel, so every one of them is already dense at panel row 1 - an audit expecting 252 rows
    of panel warmup would report a column "populated from fewer bars than its window spans" and
    raise on a construction that is correct. The split is the register's, not this test's.
    """
    from case_studies.exness_gold_sess._features import d1_state, gold_silver_ratio

    expected = warmup_expectations(setup["features"]["windows"])
    panel_census = warmup_audit(batch, expected["decision slots"], entity="symbol")
    d1_features = gold_silver_ratio(
        d1_state(d1, setup["features"]["windows"]),
        setup["features"]["windows"],
        reference=d1_reference,
    )
    d1_census = warmup_audit(d1_features, expected["D1 bars"], entity="symbol")
    print(
        f"\nwarmup audited on two grids: {panel_census.height} columns in decision slots, "
        f"{d1_census.height} columns in D1 bars"
    )
    covered = set(expected["decision slots"]) | set(expected["D1 bars"])
    missing = sorted(set(feature_columns(batch)) - covered)
    assert not missing, f"columns with no declared warmup: {missing}"


# ---------------------------------------------------------------------------
# (e) the stage artifact
# ---------------------------------------------------------------------------
def test_batch_build_reproduces_the_stage_artifact(batch: pl.DataFrame, setup: dict) -> None:
    """03_financial_features writes what this code path computes, on every development row."""
    path = get_case_study_dir(CASE_STUDY_ID) / "features" / "financial.parquet"
    if not path.exists():
        pytest.skip(f"stage artifact not built yet: {path}")
    holdout = datetime.combine(
        date.fromisoformat(str(setup["evaluation"]["holdout_start"])), datetime.min.time()
    )
    artifact = pl.read_parquet(path).filter(pl.col("timestamp") < holdout)
    cols = [c for c in artifact.columns if c not in {"timestamp", "symbol"}]
    assert set(cols) == set(feature_columns(batch)), "the artifact carries a different column set"
    rebuilt = batch.select("timestamp", "symbol", *cols).join(
        artifact.select("timestamp", "symbol"), on=["timestamp", "symbol"], how="inner"
    )
    assert rebuilt.height == artifact.height, (
        "the artifact holds a development row the batch build does not"
    )
    assert_values_agree(
        artifact.select(rebuilt.columns), rebuilt, columns=cols, keys=["timestamp", "symbol"]
    )
    sidecar = path.with_suffix(".parquet.digest.json")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["written_by"].endswith("03_financial_features.py")
    assert "load_mt5_bars:1h" in record["inputs"] and "load_mt5_bars:daily" in record["inputs"]
    print(
        f"\n{artifact.height:,} artifact rows x {len(cols)} columns reproduced from the shared "
        f"code path; sidecar digest {record['digest']}"
    )


# ---------------------------------------------------------------------------
# (f) the decision bar and the endpoint
# ---------------------------------------------------------------------------
def test_decision_and_endpoint_sit_where_the_rule_puts_them(panel: pl.DataFrame) -> None:
    """open + 60 minutes on the session's own date; the endpoint at or before the close."""
    offsets = panel.select(
        (
            (pl.col("timestamp").dt.epoch("s") - pl.col("session_open_ts").dt.epoch("s")) // 60
        ).alias("offset")
    )["offset"].unique().to_list()
    assert sorted(offsets) == [BAR_MINUTES], offsets
    late = panel.drop_nulls("label_end_ts").filter(
        pl.col("label_end_ts") > pl.col("session_close_ts")
    )
    assert late.is_empty(), f"{late.height} endpoints close after the session close"
    spans = panel.drop_nulls("label_end_ts").select(
        (
            (pl.col("label_end_ts").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")) // 60
        ).alias("span")
    )["span"].unique().to_list()
    assert sorted(spans) == [8 * BAR_MINUTES], spans
    # both seasons, per venue, in UTC hours - and the two hours per venue ARE the two seasons
    hours = (
        panel.group_by("session")
        .agg(pl.col("timestamp").dt.hour().unique().sort().alias("utc_hours"))
        .sort("session")
    )
    print("\ndecision instants by venue (UTC hours; two per venue = the two DST seasons):")
    for row in hours.iter_rows(named=True):
        assert len(row["utc_hours"]) == 2, row
        print(f"  {row['session']}: {row['utc_hours']}")
    for venue in VENUES:
        dates = panel.filter(pl.col("session") == venue)["timestamp"].dt.date().unique().to_list()
        assert any(_is_summer(d, venue) for d in dates)
        assert any(not _is_summer(d, venue) for d in dates)


# ---------------------------------------------------------------------------
# (g) the two-grid join
# ---------------------------------------------------------------------------
def test_the_d1_join_never_reaches_past_the_decision(batch: pl.DataFrame) -> None:
    """``d1_close_ts <= timestamp`` on every row, and the age is what the calendar implies.

    This is the join `PHASE1_SPEC_MENTOR.md` section 2.c' calls out. A D1 bar of server day D
    closes at 00:00 UTC on day D+1. Joining on the bar's calendar DATE instead of its close would
    hand a 09:00 UTC decision a bar that closes fifteen hours later, and no other assertion in
    this repository would see it: the values would be perfectly self-consistent.
    """
    late = batch.filter(pl.col("d1_close_ts") > pl.col("timestamp"))
    assert late.is_empty(), f"{late.height} rows read a D1 bar closing after their decision"
    aged = batch.with_columns(
        (
            (pl.col("timestamp").dt.epoch("s") - pl.col("d1_close_ts").dt.epoch("s")) // 3600
        ).alias("age_h")
    )
    census = (
        aged.group_by(["session", "age_h"]).len().sort(["session", "age_h"])
    )
    print("\nage of the D1 bar each decision reads (hours since its 00:00 UTC close):")
    with pl.Config(tbl_rows=census.height):
        print(census)
    # A weekend or a holiday pushes the newest D1 bar further back, so the age is bounded below
    # rather than fixed; the FLOOR is what the calendar implies and what a date join would break.
    for venue, floor in (("london", 8), ("ny", 13)):
        rows = aged.filter(pl.col("session") == venue)
        assert int(rows["age_h"].min()) == floor, (
            f"{venue}: youngest D1 bar is {rows['age_h'].min()} h old, expected {floor} "
            "(a date join would make this negative or zero)"
        )
        # the modal age is the ordinary weekday case, one bar back
        modal = int(rows.group_by("age_h").len().sort("len", descending=True)["age_h"][0])
        assert modal in (floor, floor + 1), modal


# ---------------------------------------------------------------------------
# (h) perturbation, both directions
# ---------------------------------------------------------------------------
def test_perturbing_a_later_bar_moves_nothing_and_an_earlier_one_moves_only_the_session(
    bars: pl.DataFrame, d1: pl.DataFrame, d1_reference, batch: pl.DataFrame, setup: dict
) -> None:
    """Symmetry, which is what separates "reads no future" from "reads nothing".

    A test that only shows a later bar changing nothing passes for a column that is constant.
    So the second half asserts the converse: perturbing the decision bar - the one bar of the
    session that HAS closed - must move the in-session columns and only those.
    """
    windows = setup["features"]["windows"]
    venue, _summer, instant = _sampled_instants(batch, setup)[3]
    symbol = sorted(setup["universe"]["symbols"])[0]
    cols = feature_columns(batch)
    ref = batch.filter((pl.col("timestamp") == instant) & (pl.col("symbol") == symbol))

    def rebuild(perturbed: pl.DataFrame) -> pl.DataFrame:
        built = build_features(
            session_panel(perturbed, keep_context=True, verbose=False),
            perturbed,
            d1,
            windows,
            d1_reference=d1_reference,
        )
        return built.filter((pl.col("timestamp") == instant) & (pl.col("symbol") == symbol))

    # 1. a bar three hours AFTER the decision: the row must not move at all.
    later = instant + timedelta(hours=2)
    after = bars.with_columns(
        pl.when((pl.col("symbol") == symbol) & (pl.col("timestamp") == later))
        .then(pl.col("close") * 1.05)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert after.filter(
        (pl.col("symbol") == symbol) & (pl.col("timestamp") == later)
    ).height == 1, "the bar chosen for the perturbation does not exist"
    moved = [
        c
        for c in cols
        if not _same(ref[c][0], rebuild(after)[c][0])
    ]
    assert not moved, f"a bar printed after the decision moved {moved}"

    # 2. the decision bar's own HIGH: only the two in-session shape columns may move.
    decision_bar_open = instant - timedelta(minutes=BAR_MINUTES)
    before = bars.with_columns(
        pl.when((pl.col("symbol") == symbol) & (pl.col("timestamp") == decision_bar_open))
        .then(pl.col("high") * 1.02)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    live = rebuild(before)
    moved = {c for c in cols if not _same(ref[c][0], live[c][0])}
    print(
        f"\nperturbation at {instant} ({venue}, {symbol}): a bar 2 h later moved 0 columns; "
        f"the decision bar's high moved {sorted(moved)}"
    )
    assert moved == IN_SESSION_COLUMNS, (
        f"perturbing the decision bar's high moved {sorted(moved)}, expected exactly "
        f"{sorted(IN_SESSION_COLUMNS)} - if it moved nothing, the in-session family is not "
        "reading the bar it claims to read"
    )


def _same(left, right) -> bool:
    if left is None or right is None:
        return left is right
    return float(left) == float(right)
