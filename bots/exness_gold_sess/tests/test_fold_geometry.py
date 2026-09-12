"""One fold ladder, three labels - the geometry a stage-04 artifact can honestly carry.

Written 2026-09-08 for ``bots/exness_gold_sess/FOLD_GEOMETRY_DECLARATION.md`` section 5. Unlike
``test_backtest_grid.py``'s New York test, **this file is not a characterisation to be inverted
later**: two of its three tests are red today, they go green when the declaration's A-prime lands,
and they stay green. The third is green today and is the regression guard that proves A-prime
changed a *timeline* and not a *label*.

The defect they measure
-----------------------
``case_studies/exness_gold_sess/02_labels.py:693`` publishes each label with a bare
``.drop_nulls()``. That makes the definedness of a label decide the label's own timeline:

    fwd_ret_8h   4,817 decision instants
    fwd_ret_24h  3,894   (every Friday is null - two slots ahead is 72 hours, not 24)
    dir_tb_8h    4,793   (the first 24 instants are null - D1 ATR(14) warm-up)

``cv_window._derive_modeling_splits`` builds each label's folds from *that label's own* unique
timestamps, while ``04_model_based_features`` stamps **one** fold set, built on the primary
label's dates, into ``features/model_based.parquet``. A model that trains on a variant label and
reads fold F of that artifact by id therefore reads Kalman / HMM / ARIMA values fitted past its
own ``val_start``: 14 leaking (label, fold) instants, which is why ``06_linear`` and ``07_gbm``
raise on both variants at ``request.resolve()`` before anything is fitted.

Why not ``modeling_fold_boundaries``
------------------------------------
``cv_window.modeling_fold_boundaries`` goes through ``fold_boundary_date``
(``cv_window.py:72-76``), which **raises** when a boundary carries a time of day. This bot's
decision instants are 09:00 and 13:00, so that helper is unusable here. These tests call
``assert_variant_folds_are_out_of_sample`` (which returns a ``gap`` per row) instead - the
declaration's own instruction.

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess \
      uv run --with pytest==9.0.3 python -m pytest \
      bots/exness_gold_sess/tests/test_fold_geometry.py -q -s
"""

from __future__ import annotations

import json
from datetime import timedelta

import polars as pl
import pytest
import yaml

from utils.paths import REPO_ROOT, get_case_study_dir

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"

#: Non-null rows per label, sealed by ``02_labels`` on 2026-09-08 and re-measured here. A-prime
#: adds ``null``-valued rows to each parquet and must not move ONE of these counts: that is the
#: whole difference between changing a timeline and changing a label.
#: Generation 2 (2026-09-10) adds ``fwd_ret_sess``, predicted BEFORE ``02`` ran to carry exactly
#: the non-null count of ``fwd_ret_8h`` (every sealed session-close endpoint has a tradable exit
#: at or before it, and no other row does) - 9,629 - and measured equal on the first run.
SEALED_NON_NULL_ROWS = {
    "fwd_ret_8h": 9_629,
    "fwd_ret_24h": 7_788,
    "dir_tb_8h": 9_581,
    "fwd_ret_sess": 9_629,
}

#: The three generation-1 parquets that stay on disk, unrepublished, beside the generation-2
#: primary. They are no longer in ``labels.primary`` / ``labels.variants``, so every test here
#: takes its label list from :func:`_label_names` - the declared labels plus these, when present -
#: rather than from the declaration alone; otherwise a one-label declaration would make the
#: timeline and key-set tests vacuous.
GENERATION_ONE_SEALED = ("fwd_ret_8h", "fwd_ret_24h", "dir_tb_8h")

#: The price vintage every label was sealed on. Taken over ``panel.drop_nulls("label_end_ts")``
#: (``02_labels.py:179-180``) - exactly the rows A-prime pads around - so it must not move either.
SEALED_MARKET_DATA_DIGEST = "6d63567e7b83eea2"


def _label_names(setup: dict, labels_dir) -> list[str]:
    """Declared labels first, then every generation-1 sealed parquet present on disk."""
    declared = [setup["labels"]["primary"], *setup["labels"]["variants"]]
    extra = [n for n in GENERATION_ONE_SEALED if n not in declared and (labels_dir / f"{n}.parquet").exists()]
    return [*declared, *extra]


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def labels_dir(setup: dict):
    """The labels directory of the configured output dir, or skip."""
    directory = get_case_study_dir(CASE_STUDY_ID) / "labels"
    names = [setup["labels"]["primary"], *setup["labels"]["variants"]]
    missing = [n for n in names if not (directory / f"{n}.parquet").exists()]
    if missing:
        pytest.skip(f"02_labels has not run in this output directory (missing {missing})")
    return directory


def test_three_labels_share_one_fold_ladder(setup: dict, labels_dir) -> None:
    """Every variant fold must open its validation after the primary fold stops training.

    RED until A-prime lands: ``assert_variant_folds_are_out_of_sample`` raises at
    ``case_studies/utils/cv_window.py:382`` naming 14 leaking (label, fold) instants -
    ``fwd_ret_24h`` fold 0 by 2 decision instants and ``dir_tb_8h`` folds 1 / 2 / 3 by 3 / 4 / 5.

    GREEN after it, and green for a stronger reason than "no negative gap": with one shared
    timeline every gap is *the same* gap, namely the primary label's own purge. A test that only
    asserted ``gap > 0`` would pass on a geometry that still let each label drift, which is the
    weakness ``cv_window``'s own docstring records in the ``fx_pairs`` version of this check.
    """
    from case_studies.utils.cv_window import assert_variant_folds_are_out_of_sample

    primary = setup["labels"]["primary"]
    variants = [v for v in setup["labels"]["variants"] if v != primary]
    n_folds = int(setup["evaluation"]["n_splits"])

    rows = assert_variant_folds_are_out_of_sample(CASE_STUDY_ID, primary)

    if not variants:
        # Generation 2 declares no variant (GEN2_DECLARATION_2026-09-10.md section 2), so the
        # function has nothing to compare and returns no row. Said rather than skipped, and
        # asserted: zero rows is the correct answer for zero variants, and the one ladder the
        # stage-04 artifact stamps is the primary's own.
        assert rows == [], f"0 variants declared but {len(rows)} (variant, fold) rows returned"
        print(f"\n0 variants declared beside {primary}: one ladder by construction, 0 rows")
        return

    assert len(rows) == len(variants) * n_folds, (
        f"expected one row per (variant, fold) = {len(variants)} x {n_folds}, got {len(rows)}: "
        f"{[(r['label'], r['fold']) for r in rows]}"
    )
    gaps = {(r["label"], int(r["fold"])): r["gap"] for r in rows}
    negative = {k: v for k, v in gaps.items() if v <= timedelta(0)}
    assert not negative, f"a variant fold validates on sessions its features saw: {negative}"
    assert len(set(gaps.values())) == 1, (
        "the three labels do not sit on one fold ladder: the gap between the primary's "
        f"train_end and a variant's val_start varies by label or fold -> {gaps}"
    )


def test_label_timelines_are_identical(setup: dict, labels_dir) -> None:
    """The three label parquets carry one key set, ``null`` where a label is undefined.

    RED today: 4,817 / 3,894 / 4,793 unique decision instants. GREEN after A-prime, when
    ``02_labels.py:693`` publishes on ``.drop_nulls(["timestamp", "symbol"])`` instead of on
    ``.drop_nulls()``, so the *decision grid* - not the definedness of a label - owns the timeline
    every fold ladder is derived from.
    """
    names = _label_names(setup, labels_dir)
    assert len(names) >= 2, f"nothing to compare: {names}"
    frames = {n: pl.read_parquet(labels_dir / f"{n}.parquet") for n in names}

    instants = {n: set(f["timestamp"].to_list()) for n, f in frames.items()}
    sizes = {n: len(v) for n, v in instants.items()}
    reference = names[0]
    for name in names[1:]:
        assert instants[name] == instants[reference], (
            f"{name} sits on a different timeline from {reference} "
            f"({sizes[name]:,} vs {sizes[reference]:,} decision instants); "
            f"only in {reference}: {len(instants[reference] - instants[name]):,}, "
            f"only in {name}: {len(instants[name] - instants[reference]):,}"
        )

    keys = {n: set(zip(f["timestamp"].to_list(), f["symbol"].to_list())) for n, f in frames.items()}
    for name in names[1:]:
        assert keys[name] == keys[reference], (
            f"{name} and {reference} differ on (timestamp, symbol)"
        )
    heights = {n: f.height for n, f in frames.items()}
    assert len(set(heights.values())) == 1, f"one key set but different row counts: {heights}"


def test_padding_added_no_labelled_row(setup: dict, labels_dir) -> None:
    """A-prime changes a timeline, not a label. Green before AND after; a regression guard.

    Three things it pins, all of them the reason the declaration could authorise three sealed
    digests to move without any evidence being lost:

    1. the non-null row count of each label is exactly what ``02_labels`` sealed on 2026-09-08;
    2. ``dir_tb_8h`` still travels with its continuous evaluation label in the same frame, which
       ``case_studies/research/labels.py:143-159`` requires of any classification label - it must
       keep doing so once that column carries ``null``;
    3. the ``market_data`` input digest of every sidecar is unmoved, because it is taken over
       ``panel.drop_nulls("label_end_ts")`` - precisely the rows the padding surrounds.
    """
    names = _label_names(setup, labels_dir)
    classification = dict(setup["labels"].get("classification_eval_label") or {})
    assert set(names) == set(SEALED_NON_NULL_ROWS), (
        f"the pinned labels {sorted(SEALED_NON_NULL_ROWS)} and the labels on disk {sorted(names)} "
        "differ; a label that appears or disappears is recorded, not ignored"
    )

    measured = {}
    for name in names:
        frame = pl.read_parquet(labels_dir / f"{name}.parquet")
        measured[name] = frame.drop_nulls(name).height
        sidecar = json.loads((labels_dir / f"{name}.parquet.digest.json").read_text())
        assert sidecar["inputs"]["market_data"] == SEALED_MARKET_DATA_DIGEST, (
            f"{name} was rebuilt on a different price vintage: "
            f"{sidecar['inputs']['market_data']} != {SEALED_MARKET_DATA_DIGEST}"
        )
        if name in classification:
            eval_label = classification[name]
            assert eval_label in frame.columns, (
                f"{name} lost its evaluation column {eval_label}; LabelCatalog.publish requires "
                "a classification label to carry it in the same frame"
            )

    assert measured == SEALED_NON_NULL_ROWS, (
        "the label VALUES moved, which no declaration authorised - only the timeline was allowed "
        f"to grow: {measured} != {SEALED_NON_NULL_ROWS}"
    )


# ---------------------------------------------------------------------------
# Generation 2, block 1: the sealed label parquets and the sealed panel agree on their key set
# ---------------------------------------------------------------------------
#: The content digest of the development-window session panel (``BAR_COLUMNS``), measured on
#: 2026-09-10 BEFORE block 1 added ``declared_closures`` to ``_features.py``. Block 1 may not
#: touch ``decision_grid`` / ``session_panel`` / ``build_features``; this pin is the falsifier.
#: The stage-04 input ``session_panel 526e4afb5f1679cc`` is taken over the panel that
#: ``04_model_based_features.py:149-155`` builds through ``holdout_end``; this file does not build
#: a frame past the holdout boundary, so it asserts that recorded input unmoved in the sidecar and
#: reproduces the panel on the development window instead. ``04``'s own re-run (block 2) is where
#: ``526e4afb5f1679cc`` is reproduced in full.
SEALED_DEV_PANEL_DIGEST = "e3ec09d299c6451b"
#: Generation 2, block 2 (GEN2_DECLARATION_2026-09-10.md L0.3 and amendment 1): ``04`` re-ran
#: once on the new primary, so its ``inputs`` carry ``labels:fwd_ret_sess`` instead of
#: ``labels:fwd_ret_8h`` while ``session_panel`` is asserted UNMOVED at ``526e4afb5f1679cc`` - the
#: pin that moved from block 1's signature to block 2's. The label digest is the one ``02`` wrote
#: on 2026-09-10 (``labels/fwd_ret_sess.parquet.digest.json``) and is re-read from that sidecar
#: below rather than typed twice.
SEALED_STAGE04_INPUTS = {"session_panel": "526e4afb5f1679cc"}
SEALED_STAGE04_LABEL_INPUT = "labels:fwd_ret_sess"


def test_label_key_set_equals_the_decision_grid(setup: dict, labels_dir) -> None:
    """``set(keys(labels/*.parquet)) == set(keys(session_panel dev))`` - one line, ``BOT.md``.

    Mechanism: ``02_labels.py`` publishes every label on the decision grid's key set (A-prime), so
    the keys of each parquet on the development window must be exactly the keys of the panel the
    shared code path builds from the development tape. The label parquets extend into the holdout
    by construction of ``02``; only their ``(timestamp, symbol)`` columns are scanned and only the
    development rows are collected, so no holdout value is read.
    """
    from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir
    from case_studies.exness_gold_sess._features import BAR_COLUMNS, session_panel
    from case_studies.exness_gold_sess._hold import assert_no_holdout, development_window
    from case_studies.utils.artifact_digest import value_digest

    if not (mt5_data_dir() / "1h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/1h.parquet)")
    lo, hi = development_window(setup)
    bars = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    ).with_columns(pl.col("timestamp").dt.replace_time_zone(None).cast(pl.Datetime("us")))
    panel = session_panel(
        bars,
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
        keep_context=False,
        verbose=False,
    )
    assert_no_holdout(setup, bars, panel)
    panel_keys = set(zip(panel["timestamp"].to_list(), panel["symbol"].to_list()))
    boundary = pl.lit(str(setup["evaluation"]["holdout_start"])).str.to_datetime()

    names = _label_names(setup, labels_dir)
    for name in names:
        keys_frame = (
            pl.scan_parquet(labels_dir / f"{name}.parquet")
            .select("timestamp", "symbol")
            .filter(pl.col("timestamp") < boundary)
            .collect()
        )
        label_keys = set(zip(keys_frame["timestamp"].to_list(), keys_frame["symbol"].to_list()))
        print(
            f"\n{name}: {len(label_keys):,} development keys vs {len(panel_keys):,} panel keys; "
            f"only in labels {len(label_keys - panel_keys)}, only in panel {len(panel_keys - label_keys)}"
        )
        assert label_keys == panel_keys, (
            f"{name}: the sealed label parquet and the sealed panel disagree on their key set "
            f"(only labels {len(label_keys - panel_keys)}, only panel {len(panel_keys - label_keys)})"
        )

    digest = value_digest(panel.select(BAR_COLUMNS))
    print(f"development panel digest {digest} (pinned {SEALED_DEV_PANEL_DIGEST})")
    assert digest == SEALED_DEV_PANEL_DIGEST, (
        f"session_panel moved on the development window: {digest} != {SEALED_DEV_PANEL_DIGEST}"
    )
    sidecar_path = labels_dir.parent / "features" / "model_based.parquet.digest.json"
    if sidecar_path.exists():
        recorded = json.loads(sidecar_path.read_text())["inputs"]
        assert {k: recorded.get(k) for k in SEALED_STAGE04_INPUTS} == SEALED_STAGE04_INPUTS, (
            f"stage 04 inputs moved: {recorded}"
        )
        primary = setup["labels"]["primary"]
        label_sidecar = json.loads((labels_dir / f"{primary}.parquet.digest.json").read_text())
        assert recorded.get(SEALED_STAGE04_LABEL_INPUT) == label_sidecar["digest"], (
            f"stage 04 was not built on the sealed {primary}: inputs {recorded} vs label sidecar "
            f"{label_sidecar['digest']}"
        )
        print(f"stage-04 sidecar inputs unmoved: {SEALED_STAGE04_INPUTS}; "
              f"{SEALED_STAGE04_LABEL_INPUT} = {label_sidecar['digest']}")
