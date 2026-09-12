"""Generation 2, block 2: the primary label ``fwd_ret_sess`` measures the window each book trades.

``bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md`` section 2, the tests it names, with their
propositions unchanged. The label is sealed by ``case_studies/exness_gold_sess/02_labels.py`` from
the rule in ``_hold.tradable_exit`` (ruling 2.1): the return from the decision bar's close to the
LAST H1 bar close at or before the session close at which a ``next_bar_open`` exit can still
fill. One rule, two constants that fall out of the tape - and nothing here types 7 or 8:

a. on every London row with a sealed endpoint the label EQUALS ``fwd_ret_8h`` (max |diff| 0.0);
b. on every New York row the fold ladder reaches it EQUALS the seven-hour probe of
   ``_report_phase5.py::ny_probe`` (``label_end_ts - 60 min``), max |diff| 0.0; the rows before the
   earliest ``train_start`` - the 2017 regime in which the broker's break did not sit at the New
   York close - are printed separately and their mechanism is asserted;
c. the tradable exit instant is the exit FILL instant ``_hold.derive_hold_bars`` measures on the
   registered price grid, on 100 % of the rows the folds reach, and the bars from decision to
   exit are ``HOLD_BARS[book]`` on 100 % of them;
d. the label is sealed at its own endpoint: rebuilding it from the development tape reproduces
   the parquet, max |diff| 0.0, and a row without an endpoint carries null;
e. the three generation-1 parquets did not move (digests pinned), every label carries the same
   ``market_data`` vintage, and the new parquet sits on the decision grid's 9,810 keys;
f. ``04_model_based_features`` re-run once on the new label reproduced its digest
   ``881f249635f32deb`` byte for byte (the block-2 falsifier of L0.3), with ``session_panel``
   ``526e4afb5f1679cc`` unmoved and ``labels:fwd_ret_sess`` as its label input.

Guard: parity (Chapter 25, the label and the book measure one window), point-in-time (the
endpoint is the venue's calendar close; whether a bar opens there is known at that instant, which
is admissible for a label and not for a feature), evidence boundary (every tape frame here stops
the day before ``evaluation.holdout_start`` and goes through ``assert_no_holdout``).

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess \\
      uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests/test_labels_gen2.py -q -s
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir
from case_studies.exness_gold_sess._features import BAR_MINUTES, session_of, session_panel
from case_studies.exness_gold_sess._hold import (
    assert_no_holdout,
    development_window,
    tradable_exit,
)
from case_studies.utils.artifact_digest import value_digest
from utils.cv_splits import generate_cv_splits
from utils.paths import REPO_ROOT, get_case_study_dir

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"

NEW_LABEL = "fwd_ret_sess"
SESSION_CLOSE_LABEL = "fwd_ret_8h"

#: The generation-1 parquets, sealed 2026-09-08 (A-prime) and NOT republished by generation 2.
GENERATION_ONE_DIGESTS = {
    "fwd_ret_8h": "9fbf73e65699693c",
    "fwd_ret_24h": "1a96d0952eb9d1de",
    "dir_tb_8h": "3b76fd3d690b7fa9",
}
#: The price vintage every label - generation 1 and 2 - is sealed on (``02_labels.py``,
#: ``MARKET_DATA_DIGEST`` over the rows that carry an endpoint).
SEALED_MARKET_DATA_DIGEST = "6d63567e7b83eea2"
#: The decision grid's key count (A-prime: every label is published on the whole grid).
GRID_KEYS = 9_810

#: The stage-04 signature written BEFORE the re-run (GEN2 section 2, "Tests bat buoc", last
#: bullet): digest, rows, feature columns, folds, the 20 fold-geometry fields unchanged, purge
#: 3-5 slots, and the session-panel input unmoved.
STAGE04 = {
    "digest": "881f249635f32deb",
    "n_rows": 24_740,
    "n_feature_columns": 9,
    "n_folds": 5,
    "session_panel": "526e4afb5f1679cc",
    "purge_gap_slots": (3, 5),
}
STAGE04_FOLD_GEOMETRY = [
    {"fold": 0, "train_start": "2021-09-01", "train_end": "2024-08-28", "val_start": "2024-08-30", "val_end": "2025-08-28"},
    {"fold": 1, "train_start": "2020-09-01", "train_end": "2023-08-29", "val_start": "2023-08-31", "val_end": "2024-08-29"},
    {"fold": 2, "train_start": "2019-09-02", "train_end": "2022-08-29", "val_start": "2022-08-31", "val_end": "2023-08-30"},
    {"fold": 3, "train_start": "2018-08-31", "train_end": "2021-08-30", "val_start": "2021-09-01", "val_end": "2022-08-30"},
    {"fold": 4, "train_start": "2018-08-31", "train_end": "2025-08-27", "val_start": "2025-09-01", "val_end": "2026-08-31"},
]


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def labels_dir(setup: dict):
    directory = get_case_study_dir(CASE_STUDY_ID) / "labels"
    if not (directory / f"{NEW_LABEL}.parquet").exists():
        pytest.skip(f"02_labels has not published {NEW_LABEL} in this output directory")
    assert setup["labels"]["primary"] == NEW_LABEL, "generation 2 declares fwd_ret_sess as primary"
    return directory


@pytest.fixture(scope="module")
def holdout_boundary(setup: dict) -> datetime:
    return datetime.fromisoformat(str(setup["evaluation"]["holdout_start"]))


@pytest.fixture(scope="module")
def bars(setup: dict) -> pl.DataFrame:
    if not (mt5_data_dir() / "1h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/1h.parquet)")
    lo, hi = development_window(setup)
    frame = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    ).with_columns(pl.col("timestamp").dt.replace_time_zone(None).cast(pl.Datetime("us")))
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
def exits(panel: pl.DataFrame, bars: pl.DataFrame, setup: dict) -> pl.DataFrame:
    frame = tradable_exit(panel, bars)
    assert_no_holdout(setup, frame)
    assert_no_holdout(setup, frame.drop_nulls("tradable_exit_ts"), column="tradable_exit_ts")
    return frame


@pytest.fixture(scope="module")
def sealed(labels_dir, holdout_boundary: datetime) -> pl.DataFrame:
    """The development rows of the sealed generation-2 parquet; the holdout rows are not read."""
    return (
        pl.scan_parquet(labels_dir / f"{NEW_LABEL}.parquet")
        .filter(pl.col("timestamp") < pl.lit(holdout_boundary))
        .collect()
    )


@pytest.fixture(scope="module")
def fold_reach_start(labels_dir, setup: dict) -> datetime:
    """The earliest ``train_start`` of the fold ladder, derived exactly as ``04`` derives it."""
    timeline = (
        pl.scan_parquet(labels_dir / f"{NEW_LABEL}.parquet")
        .select(pl.col("timestamp").dt.date().alias("timestamp"))
        .unique()
        .sort("timestamp")
        .collect()
    )
    folds = generate_cv_splits(
        timeline,
        case_study_id=CASE_STUDY_ID,
        label_buffer=setup["labels"]["buffer"],
        date_col="timestamp",
    )
    return datetime.fromisoformat(min(str(f["train_start"])[:10] for f in folds))


def _with_session(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(session_of(frame["timestamp"]).alias("session"))


# ---------------------------------------------------------------------------
# (a) London: identical to the session-close label
# ---------------------------------------------------------------------------
def test_fwd_ret_sess_equals_fwd_ret_8h_on_every_london_row(
    labels_dir, sealed: pl.DataFrame, holdout_boundary: datetime
) -> None:
    old = (
        pl.scan_parquet(labels_dir / f"{SESSION_CLOSE_LABEL}.parquet")
        .filter(pl.col("timestamp") < pl.lit(holdout_boundary))
        .collect()
    )
    joined = _with_session(sealed.join(old, on=["timestamp", "symbol"], how="inner"))
    assert joined.height == sealed.height == old.height, "the two parquets differ on their key set"
    london = joined.filter(pl.col("session") == "london")
    with_endpoint = london.drop_nulls(SESSION_CLOSE_LABEL)
    assert with_endpoint.height > 0
    assert with_endpoint[NEW_LABEL].null_count() == 0, "a London endpoint row has no fwd_ret_sess"
    assert london.filter(
        pl.col(SESSION_CLOSE_LABEL).is_null() & pl.col(NEW_LABEL).is_not_null()
    ).is_empty(), "a London row without a session-close label carries a tradable session return"
    gap = float((with_endpoint[NEW_LABEL] - with_endpoint[SESSION_CLOSE_LABEL]).abs().max())
    print(
        f"\n{with_endpoint.height:,} London development rows with an endpoint: "
        f"max |fwd_ret_sess - fwd_ret_8h| = {gap:.3e}"
    )
    assert gap == 0.0


# ---------------------------------------------------------------------------
# (b) New York: identical to the seven-hour probe on the rows the folds reach
# ---------------------------------------------------------------------------
def test_fwd_ret_sess_equals_the_seven_hour_probe_on_every_ny_row_the_folds_reach(
    panel: pl.DataFrame,
    bars: pl.DataFrame,
    exits: pl.DataFrame,
    sealed: pl.DataFrame,
    fold_reach_start: datetime,
) -> None:
    """The probe formula of ``_report_phase5.py:1219-1228``, byte for byte, joined to the parquet."""
    closes = bars.select(
        pl.col("symbol"),
        (pl.col("timestamp") + pl.duration(minutes=BAR_MINUTES)).alias("probe_end_ts"),
        pl.col("close").alias("probe_close"),
    )
    probe = (
        panel.drop_nulls("label_end_ts")
        .with_columns((pl.col("label_end_ts") - pl.duration(minutes=BAR_MINUTES)).alias("probe_end_ts"))
        .join(closes, on=["symbol", "probe_end_ts"], how="inner")
        .with_columns((pl.col("probe_close") / pl.col("close") - 1).alias("fwd_ret_7h_probe"))
        .select("symbol", "session", "timestamp", "label_end_ts", "fwd_ret_7h_probe")
    )
    ny = (
        probe.filter(pl.col("session") == "ny")
        .join(sealed, on=["symbol", "timestamp"], how="inner")
        .join(exits.select("symbol", "timestamp", "tradable_exit_ts"), on=["symbol", "timestamp"], how="left")
    )
    assert ny.height == probe.filter(pl.col("session") == "ny").height, "a probe row has no sealed label"
    assert ny[NEW_LABEL].null_count() == 0
    reach = ny.filter(pl.col("timestamp") >= pl.lit(fold_reach_start))
    before = ny.filter(pl.col("timestamp") < pl.lit(fold_reach_start))
    assert reach.height > 0
    gap = float((reach[NEW_LABEL] - reach["fwd_ret_7h_probe"]).abs().max())
    disagree = before.filter((pl.col(NEW_LABEL) - pl.col("fwd_ret_7h_probe")).abs() > 0)
    print(
        f"\nNew York rows the folds reach (>= {fold_reach_start.date()}): {reach.height:,}, "
        f"max |fwd_ret_sess - fwd_ret_7h_probe| = {gap:.3e}; rows before: {before.height:,}, of "
        f"which {disagree.height} differ from the probe (the 2017 regime in which the break did "
        f"not sit at the New York close): {sorted({str(d) for d in disagree['timestamp'].dt.date().to_list()})[:8]} ..."
    )
    assert gap == 0.0
    # The mechanism of every pre-fold disagreement, asserted rather than waved at: on those rows a
    # bar opened AT the session close, so the tradable exit is the endpoint itself and the label
    # is fwd_ret_8h, not the probe. No fold fits on, scores on or walks any of them.
    at_endpoint = disagree.filter(pl.col("tradable_exit_ts") == pl.col("label_end_ts"))
    assert at_endpoint.height == disagree.height, (
        f"{disagree.height - at_endpoint.height} pre-fold New York rows differ from the probe "
        "without exiting at their own endpoint"
    )


# ---------------------------------------------------------------------------
# (c) the tradable exit is the exit fill instant the hold derivation measures
# ---------------------------------------------------------------------------
def test_the_tradable_exit_reproduces_the_derived_hold(
    setup: dict, exits: pl.DataFrame, labels_dir
) -> None:
    """``_hold.derive_hold_bars`` on the registered grid and ``_hold.tradable_exit`` on the raw
    bars are two code paths to one instant; they must agree on 100 % of the rows any fold reaches."""
    from bots.exness_gold_sess.tests.test_backtest_grid import _derive

    hold, measured = _derive(setup, setup["labels"]["primary"])
    joined = measured.join(
        exits.rename({"timestamp": "decision_ts"}).select("symbol", "decision_ts", "tradable_exit_ts"),
        on=["symbol", "decision_ts"],
        how="left",
    )
    assert joined["tradable_exit_ts"].null_count() == 0, "a measured endpoint has no tradable exit"
    reach = joined.filter(pl.col("fold") != -2)
    assert reach.height > 0
    for book, n_bars in hold.items():
        rows = reach.filter(pl.col("session") == book)
        assert rows.height > 0, book
        mismatch = rows.filter(pl.col("exit_fill_instant") != pl.col("tradable_exit_ts"))
        bars_to_exit = (
            rows.select(
                ((pl.col("tradable_exit_ts").dt.epoch("s") - pl.col("decision_ts").dt.epoch("s")) // 3600)
                .alias("bars")
            )["bars"]
            .unique()
            .sort()
            .to_list()
        )
        print(
            f"\n[{book}] {rows.height:,} endpoints the folds reach: exit fill instant == tradable "
            f"exit on {rows.height - mismatch.height:,}; bars decision -> exit {bars_to_exit}; "
            f"derived HOLD_BARS = {n_bars}"
        )
        assert mismatch.is_empty(), f"{book}: {mismatch.height} rows exit at a different instant"
        assert bars_to_exit == [n_bars], f"{book}: {bars_to_exit} != derived {n_bars}"
    pre_fold = joined.filter(pl.col("fold") == -2)
    pre_mismatch = pre_fold.filter(pl.col("exit_fill_instant") != pl.col("tradable_exit_ts"))
    print(f"pre-fold rows (fold -2): {pre_fold.height:,}, exit instants differing: {pre_mismatch.height}")
    assert hold["ny"] < hold["london"], hold


# ---------------------------------------------------------------------------
# (d) sealed at its own endpoint
# ---------------------------------------------------------------------------
def test_the_new_label_is_sealed_at_its_own_endpoint(
    panel: pl.DataFrame, exits: pl.DataFrame, sealed: pl.DataFrame
) -> None:
    """Extends ``test_lookahead.py::test_session_close_reproduces_the_sealed_label`` to the
    generation-2 primary: the tradable exit on the development tape reproduces the parquet."""
    implied = (
        exits.drop_nulls("tradable_exit_ts")
        .join(panel.select("symbol", "timestamp", "close"), on=["symbol", "timestamp"], how="inner")
        .with_columns((pl.col("tradable_exit_close") / pl.col("close") - 1).alias("_implied"))
    )
    joined = implied.join(sealed, on=["symbol", "timestamp"], how="inner")
    assert joined.height == implied.height, "a development row with a tradable exit has no sealed label"
    assert joined[NEW_LABEL].null_count() == 0
    unlabelled = panel.filter(pl.col("label_end_ts").is_null()).join(
        sealed, on=["symbol", "timestamp"], how="inner"
    )
    carries_a_value = unlabelled.filter(pl.col(NEW_LABEL).is_not_null())
    gap = float((joined["_implied"] - joined[NEW_LABEL]).abs().max())
    print(
        f"\n{joined.height:,} development rows compared with the sealed {NEW_LABEL}, max |diff| "
        f"{gap:.3e}; {panel['label_end_ts'].null_count()} rows have no endpoint, {unlabelled.height} "
        f"of them present as null rows and {carries_a_value.height} carrying a value"
    )
    assert carries_a_value.is_empty(), "a row without an endpoint carries a tradable session return"
    assert gap == 0.0


# ---------------------------------------------------------------------------
# (e) the generation-1 parquets did not move; one vintage; the decision grid's keys
# ---------------------------------------------------------------------------
def test_the_generation_one_label_digests_did_not_move(labels_dir) -> None:
    for name, pinned in GENERATION_ONE_DIGESTS.items():
        sidecar = json.loads((labels_dir / f"{name}.parquet.digest.json").read_text())
        assert sidecar["digest"] == pinned, f"{name}: {sidecar['digest']} != sealed {pinned}"
        assert sidecar["inputs"]["market_data"] == SEALED_MARKET_DATA_DIGEST, name
        assert value_digest(pl.read_parquet(labels_dir / f"{name}.parquet")) == pinned, (
            f"{name}: the parquet on disk no longer matches its own sidecar"
        )
    new = json.loads((labels_dir / f"{NEW_LABEL}.parquet.digest.json").read_text())
    frame = pl.read_parquet(labels_dir / f"{NEW_LABEL}.parquet")
    assert new["inputs"]["market_data"] == SEALED_MARKET_DATA_DIGEST, (
        f"{NEW_LABEL} was sealed on a different price vintage: {new['inputs']}"
    )
    assert new["written_by"] == "02_labels"
    assert value_digest(frame) == new["digest"]
    assert frame.height == new["n_rows"] == GRID_KEYS, frame.height
    assert frame.select("timestamp", "symbol").n_unique() == GRID_KEYS
    assert new["digest"] not in GENERATION_ONE_DIGESTS.values()
    print(
        f"\ngeneration-1 digests reproduced: {GENERATION_ONE_DIGESTS}; {NEW_LABEL} "
        f"{new['digest']} on {frame.height:,} keys, {frame[NEW_LABEL].drop_nulls().len():,} non-null, "
        f"market_data {new['inputs']['market_data']}"
    )


# ---------------------------------------------------------------------------
# (f) stage 04 reproduced its digest on the new label
# ---------------------------------------------------------------------------
def test_stage_04_reproduced_its_digest_on_the_new_label(labels_dir) -> None:
    path = labels_dir.parent / "features" / "model_based.parquet.digest.json"
    if not path.exists():
        pytest.skip("04_model_based_features has not run in this output directory")
    sidecar = json.loads(path.read_text())
    label_sidecar = json.loads((labels_dir / f"{NEW_LABEL}.parquet.digest.json").read_text())
    feature_columns = [c for c in sidecar["columns"] if c not in {"timestamp", "symbol", "fold"}]
    gaps = list(sidecar["purge_gap_slots"].values())
    print(
        f"\nstage 04: digest {sidecar['digest']}, {sidecar['n_rows']:,} rows, "
        f"{len(feature_columns)} feature columns, {len(sidecar['fold_geometry'])} folds, purge "
        f"{min(gaps)}-{max(gaps)} slots, inputs {sidecar['inputs']}"
    )
    assert sidecar["digest"] == STAGE04["digest"], (
        f"04 moved: {sidecar['digest']} != {STAGE04['digest']} - the block-2 falsifier of L0.3"
    )
    assert sidecar["n_rows"] == STAGE04["n_rows"]
    assert len(feature_columns) == STAGE04["n_feature_columns"], feature_columns
    assert len(sidecar["fold_geometry"]) == STAGE04["n_folds"]
    assert sidecar["fold_geometry"] == STAGE04_FOLD_GEOMETRY, "a fold_geometry field moved"
    assert (min(gaps), max(gaps)) == STAGE04["purge_gap_slots"], gaps
    assert sidecar["inputs"]["session_panel"] == STAGE04["session_panel"], sidecar["inputs"]
    assert sidecar["inputs"].get(f"labels:{NEW_LABEL}") == label_sidecar["digest"], sidecar["inputs"]
    assert f"labels:{SESSION_CLOSE_LABEL}" not in sidecar["inputs"], sidecar["inputs"]
    assert np.isfinite(sidecar["n_rows"])
