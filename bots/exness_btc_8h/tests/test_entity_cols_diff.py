"""Contract tests for the mentor-gated `utils/modeling.py` entity-column diff (fleet gate item
#6, 2026-09-10).

`utils/modeling.py::load_modeling_dataset` is SHARED (top-level `utils/`, not
`case_studies/utils/`, but held to the same single-writer policy - "propose, do not apply" -
while `exness_fx_d1`'s builder edits `case_studies/utils/` and `utils/` live). These tests are
written AHEAD of the diff landing, per the mentor's six-item list verbatim.

**The mentor's correction to the first proposal is the point of this whole file.** A fix that
falls back to the constant entity column whenever the FILTERED list is empty would ALSO silently
"fix" `max_symbols=1` PREVIEW reductions on `etfs` / `fx_pairs` - which today correctly raise
(`linear.py:543-544`), because a preview reduction to one symbol is an artefact of the reduction,
not a declaration that the universe IS one symbol. The required fix is DECLARATION-driven: keep
the constant column only when the case study's OWN `setup.yaml::universe.n_assets == 1` (the one
key every registered case study declares - verified 2026-09-10: `etfs` 100, `fx_pairs` 20,
`crypto_perps_funding` 19, `exness_gold_sess` 2, `exness_fx_d1` 5, `exness_btc_8h` 1). Test 2 below
is the one that would have caught the wrong (emptiness-driven) fix.

* Tests 1, 2, 5 exercise CURRENT behaviour on already-registered, multi-asset case studies and
  PASS TODAY - real regression protection.
* Tests 3, 4, 6 need the diff itself (`exness_btc_8h` reaching a fold, R1's NaN-safety, and the
  identity-hash proof). They SKIP only when this bot's data is genuinely not built in the
  running environment (`FileNotFoundError`); if the data IS present and `entity_cols` still
  comes back empty - the diff missing or reverted - they RUN and FAIL, on purpose (fleet mentor
  material #4, 2026-09-10: collapsing "no data" and "diff reverted" into one skip reason would
  let a reverted fix hide behind a green, skipped run).

    uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/test_entity_cols_diff.py -q
"""

from __future__ import annotations

import math

import pytest

CASE_STUDY_ID = "exness_btc_8h"


def _entity_cols_state_for_btc() -> tuple[str, str]:
    """``(state, detail)``, ``state`` in ``{"fixed", "missing_data", "not_fixed"}``.

    Fleet mentor material #4 (2026-09-10, combined re-gate): the earlier version of this probe
    caught ANY exception from `load_modeling_dataset` and treated it the same as "entity_cols
    came back empty" - one skip reason for two different facts. That MASKS a real regression:
    if the diff is reverted (or never lands) while this environment's data IS present,
    `entity_cols` comes back empty with NO exception at all, and the gated tests below must
    FAIL, not skip, because there is nothing stopping them from running except the diff itself.
    Only a genuinely MISSING data file (`FileNotFoundError`, this bot's labels/features not
    built in this sandbox) is an environment fact worth a skip.
    """
    from utils.modeling import load_modeling_dataset

    try:
        mds = load_modeling_dataset(CASE_STUDY_ID, "fwd_ret_8h")
    except FileNotFoundError as exc:
        return "missing_data", f"load_modeling_dataset: {exc!r}"
    if mds.date_col == "timestamp" and mds.entity_cols:
        return "fixed", f"entity_cols={mds.entity_cols!r}"
    return (
        "not_fixed",
        f"date_col={mds.date_col!r} entity_cols={mds.entity_cols!r} (still empty, WITH data "
        "present - the declaration-driven fix is missing or reverted)",
    )


def _multi_asset_case_study_available(case_study: str, label_stem: str) -> str | None:
    """A registered, likely-data-present multi-asset case study, or None to skip on."""
    from utils.paths import get_case_study_dir

    label_path = get_case_study_dir(case_study) / "labels" / f"{label_stem}.parquet"
    return case_study if label_path.exists() else None


class TestEntityColsUnchangedOnMultiAssetItem1:
    """(1) entity_cols identical (unaffected) on fx_pairs and etfs - real data, pinned snapshot."""

    # Captured 2026-09-10 by calling load_modeling_dataset directly, before any diff existed.
    # A declaration-driven fix must not move either of these: both case studies declare
    # universe.n_assets > 1, so the fallback branch the diff adds must never fire for them.
    _SNAPSHOTS = {"fx_pairs": ["symbol"], "etfs": ["symbol"]}

    @pytest.mark.parametrize("case_study,primary_label", [("fx_pairs", "fwd_ret_21d"), ("etfs", "fwd_ret_21d")])
    def test_entity_cols_matches_pinned_snapshot(self, case_study, primary_label):
        from utils.modeling import load_modeling_dataset

        available = _multi_asset_case_study_available(case_study, primary_label)
        if available is None:
            pytest.skip(f"{case_study}/{primary_label}.parquet not built in this environment")
        mds = load_modeling_dataset(case_study, primary_label)
        assert mds.date_col == "timestamp"
        assert mds.entity_cols == self._SNAPSHOTS[case_study]
        # And it is genuinely non-constant - the thing a declaration-driven fix must key off of
        # instead of re-deriving from data every time.
        assert mds.dataset[mds.entity_cols[0]].n_unique() > 1


class TestMaxSymbolsOnePreviewStillRaisesItem2:
    """(2) max_symbols=1 on a MULTI-ASSET case study still raises - the test a wrong
    (emptiness-driven) fix would fail and a correct (declaration-driven) fix must pass.
    """

    @pytest.mark.parametrize("case_study,primary_label", [("fx_pairs", "fwd_ret_21d"), ("etfs", "fwd_ret_21d")])
    def test_max_symbols_1_preview_raises_in_the_linear_runner(self, case_study, primary_label):
        from case_studies.utils.linear import _load_batch_base
        from case_studies.research import open_study

        available = _multi_asset_case_study_available(case_study, primary_label)
        if available is None:
            pytest.skip(f"{case_study}/{primary_label}.parquet not built in this environment")
        study = open_study(case_study, execution_tier="canonical")
        request = {"label": primary_label, "reductions": {"max_symbols": 1}}
        with pytest.raises(ValueError, match="requires timestamp and an entity key"):
            _load_batch_base(study, request)


BTC_ENTITY_STATE, _BTC_ENTITY_DETAIL = _entity_cols_state_for_btc()
BTC_FIXED = BTC_ENTITY_STATE == "fixed"
# Skip ONLY on a missing-data environment fact. If the diff is missing or reverted
# (`BTC_ENTITY_STATE == "not_fixed"`), this is False and the gated classes below RUN and FAIL
# for real - exactly the fleet mentor material #4 distinction.
SKIP_ON_MISSING_DATA = BTC_ENTITY_STATE == "missing_data"
SKIP_REASON = f"exness_btc_8h data not built in this environment: {_BTC_ENTITY_DETAIL}"


@pytest.mark.skipif(SKIP_ON_MISSING_DATA, reason=SKIP_REASON)
class TestBtcEntityColsAndFoldGateItem3:
    """(3) this bot yields entity_cols == ['symbol'] and reaches the first fold fit."""

    def test_entity_cols_is_symbol_and_constant(self):
        from utils.modeling import load_modeling_dataset

        mds = load_modeling_dataset(CASE_STUDY_ID, "fwd_ret_8h")
        assert mds.entity_cols == ["symbol"]
        assert mds.dataset["symbol"].n_unique() == 1, (
            "exness_btc_8h is a genuinely one-instrument universe; entity_cols must be kept "
            "BECAUSE it is constant by declaration, not despite it"
        )

    def test_linear_runner_gate_clears(self):
        """Replicates the exact guard `case_studies/utils/linear.py:543-544` applies - the
        precondition for reaching a fold fit - without running a full walk-forward fit here
        (out of scope for a unit test; `06_linear.py` itself is the end-to-end proof).
        """
        from utils.modeling import load_modeling_dataset

        mds = load_modeling_dataset(CASE_STUDY_ID, "fwd_ret_8h")
        assert not (mds.date_col != "timestamp" or not mds.entity_cols), (
            "the linear runner's own gate would still raise 'requires timestamp and an entity "
            "key' on this dataset"
        )
        entity_col = mds.entity_cols[0]
        assert entity_col in {"product", "symbol"}, (
            f"entity key {entity_col!r} would still fail the runner's supported-key check"
        )


@pytest.mark.skipif(SKIP_ON_MISSING_DATA, reason=SKIP_REASON)
class TestTwoSourcesOfTruthMaterial2:
    """Fleet mentor material #2 (2026-09-10, combined re-gate): when the single-asset branch
    fires, `utils/modeling.py` must assert `int(n_assets) == len(universe.symbols)` and raise
    on mismatch - `setup.yaml::universe.n_assets` and `universe.symbols` are two independently
    typed declarations, and nothing upstream of this fix enforced they agree.
    """

    def test_exness_btc_8h_declaration_is_self_consistent(self):
        """The real declaration this bot ships: n_assets and len(symbols) already agree."""
        from utils.artifact_specs import load_setup_config

        universe = load_setup_config(CASE_STUDY_ID)["universe"]
        assert int(universe["n_assets"]) == len(universe["symbols"]) == 1

    def test_mismatched_declaration_raises(self, monkeypatch):
        """A synthetic n_assets/symbols disagreement must raise, not silently pick one."""
        from utils.artifact_specs import load_setup_config as real_load_setup_config

        def _load_setup_with_mismatch(case_study: str):
            cfg = real_load_setup_config(case_study)
            if case_study == CASE_STUDY_ID:
                cfg = dict(cfg)
                cfg["universe"] = dict(cfg["universe"])
                # n_assets says 1; symbols says 2 - a real drift a maintainer could introduce
                # by editing one list and forgetting the count (or vice versa).
                cfg["universe"]["symbols"] = ["BTCUSD", "ETHUSD"]
            return cfg

        monkeypatch.setattr("utils.artifact_specs.load_setup_config", _load_setup_with_mismatch)
        from utils.modeling import load_modeling_dataset

        with pytest.raises(ValueError, match="two declarations disagree"):
            load_modeling_dataset(CASE_STUDY_ID, "fwd_ret_8h")


class TestConsumersDoNotCrossSectionalNormaliseItem5:
    """(5) consumers of entity_cols[0] apply no per-day cross-sectional normalisation.

    Static check: every cited consumer uses `entity_cols[0]` purely as an identity / supported-key
    gate (`entity_col not in {"product", "symbol"}` or an identity-column list), never as a
    `.over(date_col)` / rank-within-day grouping key. Read from source rather than executed - a
    full DML, sequence or GBM fit is out of scope for a unit test - so this is a real, if narrow,
    regression check: it fails loudly if a future edit adds cross-sectional math near these exact
    call sites without this test being updated too.
    """

    _CITED = [
        ("case_studies/utils/causal.py", (1295, 1305), "entity_cols[0]"),
        ("case_studies/utils/deep_learning.py", (420, 430), "entity_cols[0]"),
        ("case_studies/utils/deep_learning.py", (680, 690), "entity_cols"),
        ("case_studies/utils/gbm.py", (1720, 1730), "entity_cols[0]"),
        ("case_studies/utils/darts_forecasting.py", (660, 670), "entity_col"),
    ]

    def test_no_over_date_col_near_entity_cols_usage(self):
        from utils.paths import REPO_ROOT

        offenders = []
        for rel_path, (lo, hi), _needle in self._CITED:
            path = REPO_ROOT / rel_path
            lines = path.read_text().splitlines()
            window = "\n".join(lines[lo - 1 : hi])
            if ".over(" in window and "entity_col" in window:
                offenders.append(f"{rel_path}:{lo}-{hi}")
        assert not offenders, (
            f"found a `.over(...)` grouping near an entity_cols consumer, which would break on "
            f"a constant (one-instrument) entity column: {offenders}"
        )

    def test_cited_lines_contain_the_expected_gate_shape(self):
        """The citation is a supported-key GATE, not a cross-sectional transform - confirmed by
        the literal `{"product", "symbol"}` (or a subset check) appearing in the same window.
        """
        from utils.paths import REPO_ROOT

        for rel_path, (lo, hi), needle in self._CITED:
            path = REPO_ROOT / rel_path
            lines = path.read_text().splitlines()
            window = "\n".join(lines[lo - 1 : hi])
            assert needle in window, f"{rel_path}:{lo}-{hi} no longer mentions {needle!r}"


@pytest.mark.skipif(SKIP_ON_MISSING_DATA, reason=SKIP_REASON)
class TestRegistryNullAndR1AndGbmNanSafetyItem4:
    """(4) registry ic_mean/ic_std/ic_t/ic_n_days null on this bot AND R1 reads pooled panel IC
    (12_model_analysis.py:262-263) AND GBM checkpoint selection never compares on NaN
    (gbm.py:845-856).
    """

    def test_cross_sectional_ic_is_nan_not_raising_on_one_entity(self):
        """The registry's min_obs=5 cross-sectional IC on a one-entity frame: NaN/None, not an
        exception - the property REGISTRY_IC_IS_NULL_REASON documents and gbm.py:845-856's
        `cross_sectional_ic(..., min_obs=5)` call inherits unchanged.
        """
        import polars as pl
        from ml4t.diagnostic.metrics import cross_sectional_ic

        predictions = pl.DataFrame(
            {
                "timestamp": [f"2026-01-{d:02d}" for d in range(1, 11)],
                "symbol": ["BTCUSD"] * 10,
                "prediction": [float(i) for i in range(10)],
            }
        )
        returns = predictions.rename({"prediction": "forward_return"})
        metric = cross_sectional_ic(
            predictions,
            returns,
            pred_col="prediction",
            ret_col="forward_return",
            date_col="timestamp",
            entity_col="symbol",
            min_obs=5,
        )
        ic_mean = metric.get("ic_mean")
        assert ic_mean is None or (isinstance(ic_mean, float) and math.isnan(ic_mean)), (
            f"expected null/NaN on a one-entity-per-day frame under min_obs=5, got {ic_mean!r}"
        )

    def test_gbm_checkpoint_metrics_do_not_crash_on_nan(self):
        """gbm.py:845-856 wraps the same cross_sectional_ic call per checkpoint; no downstream
        `argmax`/`sorted` on `ic_mean` exists in `case_studies/utils/gbm.py` (checked 2026-09-10:
        `grep -n "best_checkpoint\\|argmax" case_studies/utils/gbm.py` = 0 hits) - every
        checkpoint is published as its own prediction set and selection happens in
        `13_backtest` on backtest Sharpe, never here. This test pins that absence.
        """
        from utils.paths import REPO_ROOT

        source = (REPO_ROOT / "case_studies/utils/gbm.py").read_text()
        assert "argmax" not in source
        assert "best_checkpoint" not in source

    def test_r1_gate_reads_pooled_panel_ic_not_registry_ic_mean(self):
        """12_model_analysis.py's R1 section (~262-263 as forked) must build its p-values from
        `summarise_pooled_ic` / `pooled_panel_ic_series`, never from the catalog's `ic_mean`
        column - which is null by construction on this one-instrument panel.
        """
        from utils.paths import REPO_ROOT

        source = (REPO_ROOT / "case_studies/exness_btc_8h/12_model_analysis.py").read_text()
        r1_section = source.split("## R1 - the early-close gate", 1)[1]
        r1_section = r1_section.split("## Fold stability", 1)[0]
        assert "summarise_pooled_ic" in r1_section or "pooled_panel_ic_series" in r1_section
        assert 'catalog["ic_mean"]' not in r1_section
        assert "catalog.get_column(\"ic_mean\")" not in r1_section


class TestEntityColsNotInTrainingIdentityItem6:
    """(6) entity_cols is not part of the training identity; fx_pairs hashes unchanged."""

    def test_entity_col_string_is_identical_across_case_studies(self):
        """`entity_col` (the COLUMN NAME, e.g. "symbol") is what could theoretically enter a
        spec/hash - not the underlying data. It is "symbol" for exness_btc_8h exactly as it is
        for every other symbol-keyed case study (etfs, fx_pairs, exness_gold_sess, ...), so even
        if a caller persisted it, the persisted value does not change. Confirmed against
        `_PRICE_CONFIG`, the one place every case study's entity column name is declared, rather
        than against `load_modeling_dataset` (would need built data for every case study).
        """
        from case_studies.utils.backtest_loaders import _PRICE_CONFIG

        symbol_keyed = {
            cs: cfg["entity_col"]
            for cs, cfg in _PRICE_CONFIG.items()
            if cfg.get("entity_col") == "symbol"
        }
        assert symbol_keyed, "expected at least one symbol-keyed case study in _PRICE_CONFIG"
        assert all(v == "symbol" for v in symbol_keyed.values())

    def test_no_hash_function_reads_entity_cols_by_name(self):
        """Static check: no `training_hash` / identity-hashing helper in `case_studies/research/`
        reads `entity_cols` (or `mds.entity_cols`) directly. `entity_col` (singular, the string
        name) is used to build the expected-keys frame the fold pipeline reads
        (`case_studies/utils/linear.py:557-560`); the constant-vs-not-constant DECISION the diff
        changes is upstream of that, in `utils/modeling.py`, and never itself serialised.
        """
        from utils.paths import REPO_ROOT

        research_dir = REPO_ROOT / "case_studies" / "research"
        offenders = []
        for path in sorted(research_dir.glob("*.py")):
            text = path.read_text()
            if "entity_cols" in text and ("hash" in text.lower()):
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if "entity_cols" in line and (
                        "hash" in line.lower() or "canonical_json" in line
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
        assert not offenders, (
            f"entity_cols appears near hashing code in case_studies/research/: {offenders}; "
            "read these lines before assuming the training identity is unaffected"
        )
