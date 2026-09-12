"""Static, offline checks on the phase-4 declarations (roadmap phase 4 gate, forked-not-fitted).

Every test here reads `.py` source and `.yaml` config only - no MT5 data, no registry, no
`run_log/`, no fit. They exist because `06_linear.py`, `07_gbm.py`, `12_model_analysis.py` and
`13_backtest.py` were FORKED from `exness_gold_sess` and adapted 2026-09-10
(`bots/exness_btc_8h/BOT.md` Decisions log) but have never been run, so the only thing there is
to check today is what the source and the declarations themselves say.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/test_phase4_declarations.py -q
"""

from __future__ import annotations

import re

import pytest
import yaml

from utils.paths import REPO_ROOT

CASE_STUDY_ID = "exness_btc_8h"
CASE_STUDY_DIR = REPO_ROOT / "case_studies" / CASE_STUDY_ID
SETUP_PATH = CASE_STUDY_DIR / "config" / "setup.yaml"
LGB_CONFIG_DIR = REPO_ROOT / "case_studies" / "config" / "lgb"
FORKED_STAGES = ["06_linear.py", "07_gbm.py", "12_model_analysis.py", "13_backtest.py"]

# Frozen 2026-09-10, corrected from a first-pass 43/86/516 that used |declared configurations|
# in place of |prediction sets| (bots/exness_btc_8h/BOT.md Decisions log, Trials table). A
# change here without the same change in BOT.md and setup.yaml::kill_criteria is a stale test,
# not a passing one - the assertions below compare this constant against a value DERIVED from
# the declarations, not the other way around.
EXPECTED_PREDICTION_SETS_PER_LABEL = 178  # 28 linear + 15 GBM x 10 checkpoints
EXPECTED_TOTAL_PREDICTION_SETS = 356  # x 2 labels
EXPECTED_SESSION_BOOKS = 3  # all, no_swap_night, us_hours
EXPECTED_SIGNAL_SPECS = 2  # fixed_threshold_0, per_symbol_p80
EXPECTED_K5 = 2136  # 356 x 3 x 2


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text())


@pytest.fixture(scope="module")
def training_menus() -> dict[str, dict]:
    menus = {}
    for label in ("fwd_ret_8h", "fwd_ret_24h"):
        path = CASE_STUDY_DIR / "config" / "training" / f"{label}.yaml"
        menus[label] = yaml.safe_load(path.read_text())
    return menus


def _gbm_checkpoints(preset_name: str) -> int:
    """Checkpoints a GBM preset publishes: max_iterations // checkpoint_interval.

    Reads the SHARED preset file directly (`case_studies/config/lgb/<preset_name>.yaml`) rather
    than assuming 10 - the same file `07_gbm.py` resolves through `load_model_configs`, so a
    preset that stops declaring `checkpoint_interval` (one "final" checkpoint) changes this
    computation exactly the way it would change what `07_gbm` actually publishes.
    """
    preset_path = LGB_CONFIG_DIR / f"{preset_name}.yaml"
    preset = yaml.safe_load(preset_path.read_text())
    if "checkpoint_interval" not in preset:
        return 1
    return int(preset["max_iterations"]) // int(preset["checkpoint_interval"])


class TestK5DerivableFromSetupAlone:
    """K5 (and the phase-4 prediction-set count R1 counts against) from declarations only."""

    def test_prediction_sets_per_label_include_gbm_checkpoints(self, training_menus):
        for label, menu in training_menus.items():
            linear_sets = len(menu.get("linear", []))
            gbm_sets = sum(_gbm_checkpoints(name) for name in menu.get("gbm", []))
            total = linear_sets + gbm_sets
            assert total == EXPECTED_PREDICTION_SETS_PER_LABEL, (
                f"{label}: {linear_sets} linear + {gbm_sets} GBM checkpoint-expanded = {total}, "
                f"not the frozen {EXPECTED_PREDICTION_SETS_PER_LABEL}. A configuration is NOT a "
                "prediction set for a checkpointed GBM preset "
                "(case_studies/research/model_planning.py::checkpoint_schedule)."
            )
            # no deep_learning / tabular_dl / causal_dml: declared absent, not commented out
            assert not menu.get("deep_learning"), f"{label}: deep_learning must stay undeclared"
            assert not menu.get("tabular_dl"), f"{label}: tabular_dl must stay undeclared"
            assert not menu.get("causal_dml"), f"{label}: causal_dml must stay undeclared"

    def test_total_prediction_sets_matches_kill_criteria(self, setup, training_menus):
        total = 0
        for menu in training_menus.values():
            total += len(menu.get("linear", [])) + sum(
                _gbm_checkpoints(name) for name in menu.get("gbm", [])
            )
        assert total == EXPECTED_TOTAL_PREDICTION_SETS
        declared = int(setup["kill_criteria"]["r1_expected_prediction_sets"])
        assert declared == total == EXPECTED_TOTAL_PREDICTION_SETS, (
            "setup.yaml::kill_criteria.r1_expected_prediction_sets must equal the count derived "
            "from config/training/*.yaml + the shared GBM presets' checkpoint schedules"
        )

    def test_k5_derivable_from_setup_alone(self, setup, training_menus):
        """The number 12_model_analysis and 13_backtest print as K5, from setup.yaml alone."""
        total_prediction_sets = sum(
            len(menu.get("linear", []))
            + sum(_gbm_checkpoints(name) for name in menu.get("gbm", []))
            for menu in training_menus.values()
        )
        books = setup["backtest"]["sweep"]["session_books"]
        specs = setup["backtest"]["sweep"]["signal_specs"]
        assert len(books) == EXPECTED_SESSION_BOOKS
        assert len(specs) == EXPECTED_SIGNAL_SPECS
        assert "pooled_book" not in setup["backtest"]["sweep"], (
            "this bot declares no arithmetic pooled sleeve (unlike exness_gold_sess): all three "
            "session books are engine-run, so K5 has no extra pooled term to add"
        )
        k5 = total_prediction_sets * len(books) * len(specs)
        assert k5 == EXPECTED_K5

    def test_vol_target_arms_declared_before_first_fit(self, setup):
        """K6's n is listed in setup.yaml (fleet convention), not left implicit."""
        sizing = setup["backtest"]["sweep"]["position_sizing"]
        assert sizing["method"] == "volatility_target"
        arms = sizing["vol_target_arms"]
        assert sizing["n_vol_target_arms"] == len(arms) == 3
        for arm in arms:
            assert {"name", "target_annualized_vol", "vol_lookback_slots"} <= set(arm)


class TestR1DeclarativeNotHardCoded:
    """R1's numbers live in setup.yaml::kill_criteria, and 12_model_analysis reads them from there."""

    def test_kill_criteria_block_present(self, setup):
        kc = setup["kill_criteria"]
        assert kc["r1_early_close"] is True
        assert kc["r1_fdr_alpha"] == pytest.approx(0.05)
        assert kc["r1_required_sign_consistency"] == pytest.approx(1.0)
        assert kc["r1_ic_star_validation"] == pytest.approx(0.1144)

    def test_12_model_analysis_reads_the_flag_not_a_literal(self):
        source = (CASE_STUDY_DIR / "12_model_analysis.py").read_text()
        assert 'KILL_CRITERIA.get("r1_early_close"' in source
        assert 'KILL_CRITERIA.get("r1_fdr_alpha"' in source
        assert 'KILL_CRITERIA.get("r1_required_sign_consistency"' in source
        # the raw thresholds must not additionally be hard-coded as bare literals in the gate cell
        gate_cell = source.split("## R1 - the early-close gate", 1)[1]
        gate_cell = gate_cell.split("## Fold stability", 1)[0]
        assert "0.1144" not in gate_cell
        assert re.search(r"(?<!r1_fdr_)alpha\s*=\s*0\.05", gate_cell) is None

    def test_verdict_artifact_path_declared(self):
        source = (CASE_STUDY_DIR / "12_model_analysis.py").read_text()
        assert '"evaluation" / "r1_early_close_verdict.json"' in source
        thirteen = (CASE_STUDY_DIR / "13_backtest.py").read_text()
        assert '"evaluation" / "r1_early_close_verdict.json"' in thirteen
        assert "generation_closed" in thirteen

    def test_clears_r1_gates_on_bh_alone_not_sign_consistency(self):
        """M2 resolved 2026-09-11, option B: the approved B2 text gates on BH alone; the code

        must not widen it with an extra sign-consistency AND condition. sign_consistency stays
        computed and REPORTED (still a column read for the display table and the verdict JSON's
        n_sign_consistent_4of4_reported), but must not be ANDed into clears_r1 itself.
        """
        source = (CASE_STUDY_DIR / "12_model_analysis.py").read_text()
        gate_cell = source.split("## R1 - the early-close gate", 1)[1]
        gate_cell = gate_cell.split("## Fold stability", 1)[0]
        assert 'pl.col("bh_significant").alias("clears_r1")' in gate_cell, (
            "clears_r1 must be BH significance alone (M2 option B); found something else"
        )
        assert (
            '(pl.col("bh_significant") & (pl.col("sign_consistency")' not in gate_cell
        ), "clears_r1 must not AND sign_consistency into the gate (M2 option B narrowed this away)"
        # sign consistency is still reported, just not gating
        assert "sign_consistency_reported_threshold" in gate_cell
        assert '"gate_condition": "bh_fdr_only"' in gate_cell

    def test_setup_yaml_kill_criteria_comment_is_bh_only(self):
        """The kill_criteria comment block must describe the BH-only predicate (M2 option B,

        2026-09-11), not the retired "BH AND sign_consistency" draft wording. Reads the RAW text
        of setup.yaml between the `kill_criteria:` line and the next top-level key, so this
        catches stale prose even though it never reaches a Python literal or a YAML value.
        """
        raw = SETUP_PATH.read_text()
        start = raw.index("kill_criteria:")
        after = raw[start + len("kill_criteria:") :]
        next_key = re.search(r"\n\S", after)
        block = after[: next_key.start()] if next_key else after
        assert "AND (sign_consistency" not in block, (
            "setup.yaml kill_criteria comment block still states the retired BH-AND-"
            "sign_consistency predicate; reword to the BH-only gate (gate_condition: "
            "bh_fdr_only, M2 resolved 2026-09-11 option B)"
        )


class TestForkedStagesStayInsideTheDevelopmentWindow:
    """The forked stages read only development rows; the holdout is never loaded by them."""

    @pytest.mark.parametrize("stage", FORKED_STAGES)
    def test_stage_guards_against_the_holdout(self, stage):
        source = (CASE_STUDY_DIR / stage).read_text()
        assert "holdout_start" in source, (
            f"{stage} does not reference evaluation.holdout_start at all - a forked stage that "
            "cannot even name the boundary cannot be asserted to respect it"
        )
        # 06/07/12 assert the LATEST VALIDATION row is strictly before the holdout; 13 relies on
        # the registry's split == "validation" scoping (upstream of this file) plus its own R1
        # verdict read, so it is allowed to satisfy the weaker "reads split" form instead.
        explicit_guard = re.search(r">=\s*holdout_start", source) or re.search(
            r"holdout_start\s*<=", source
        )
        split_scoped = '"split"' in source or "split ==" in source or 'SPLIT = "validation"' in source
        assert explicit_guard or split_scoped, (
            f"{stage} names holdout_start but has neither an explicit >=/<= comparison against "
            "it nor a split == 'validation' scoping; the guard the other forked stages carry is "
            "missing here"
        )

    @pytest.mark.parametrize("stage", FORKED_STAGES)
    def test_no_hardcoded_holdout_or_later_date_literal(self, setup, stage):
        """No ISO date literal on or after holdout_start is typed into a forked stage's source.

        The campaign rule for this bot (`reviews.jsonl` 2026-09-10T06:37:55Z, `quant-orchestrator`)
        is that every data call carries `as_of <= 2025-08-31`. A stage cannot honour that rule if
        it carries a hard-coded date on or after it; every date this bot's stages actually need
        (`holdout_start`, `holdout_end`) is read from `setup.yaml`, never typed as a literal in
        the stage source.
        """
        holdout_start = str(setup["evaluation"]["holdout_start"])
        source = (CASE_STUDY_DIR / stage).read_text()
        # Comment / markdown-cell lines (jupytext percent format: everything from a bare '#' to
        # end of line, including whole '# %% [markdown]' cells) carry plenty of legitimate dates
        # - "Decisions log 2026-09-10", timestamps quoted from reviews.jsonl - that are prose, not
        # a data-window bound. Only CODE lines matter here, and only a date that appears as a
        # quoted string literal (what a filter/comparison would actually receive), not a bare
        # mention.
        code_lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            code_lines.append(re.sub(r"#.*$", "", line))
        code_only = "\n".join(code_lines)
        dates = re.findall(r"""['"](20\d{2}-\d{2}-\d{2})""", code_only)
        offending = [d for d in dates if d >= holdout_start]
        assert not offending, (
            f"{stage} contains a hard-coded quoted date literal on or after holdout_start "
            f"({holdout_start}) in a CODE line: {offending}. Every boundary date must be read "
            "from setup.yaml, not typed."
        )

    def test_setup_yaml_holdout_unchanged_by_this_fork(self, setup):
        """The holdout window this task forked stages around is the one sealed 2026-09-08."""
        assert str(setup["evaluation"]["holdout_start"]) == "2025-09-01"
        assert str(setup["evaluation"]["holdout_end"]) == "2026-08-31"
