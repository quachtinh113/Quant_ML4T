# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model analysis - Exness BTC 8h
#
# **Chapter reference**: Volume 1, Chapters 11-15.
# **Docker image**: `ml4t` (default).
# **Prerequisites**: `06_linear` and `07_gbm` must have published their canonical validation
# populations into the workspace this notebook opens.
# **What it writes**: `evaluation/phase4_ic_diagnostics.parquet` (+ `.csv`), the complete
# per-prediction-set diagnostics frame (added 2026-09-10, mentor gate material M1), and
# `evaluation/r1_early_close_verdict.json`, the R1 kill-criterion verdict (section "R1"). It
# fits nothing, registers no training or prediction row.
#
# Forked from `case_studies/exness_gold_sess/12_model_analysis.py`
# (`bots/exness_btc_8h/BOT.md` Decisions log 2026-09-10), adapted for one instrument: no
# cross-sectional IC, no `dir_tb_8h`. **New here**: the R1 early-close gate
# (`setup.yaml::kill_criteria`), approved by the user 2026-09-10 as part of B2 - this stage is
# "the stage that reads the 86 [corrected: 356] prediction sets" the gate is wired into.
#
# This notebook reads the complete registered validation-prediction population of the bot
# `exness_btc_8h` (`bots/exness_btc_8h/BOT.md`) and compares model families, checkpoints, fold
# stability, prediction agreement and uncertainty **without choosing a model for deployment**.
# Every exact model configuration continues to `13_backtest`, UNLESS R1 closes the generation.
#
# **Nothing here reads the holdout.** The catalog is filtered to `split == "validation"`, and the
# maximum prediction timestamp is asserted below `evaluation.holdout_start` as a second,
# independent check. `17_holdout_predictions` / `18_holdout_backtest` are the only stages that
# ever score the sealed window.
#
# ## The one thing that is different from the template
#
# The template's model analysis is built on the **cross-sectional** information coefficient. This
# bot's panel is **one instrument**, and that statistic is not defined on it -
# `case_studies/utils/registry/metrics.py:113` scores it under a hard-coded `min_obs=5`, so the
# registry's `ic_mean`, `ic_std`, `ic_t` and `ic_n_days` are **null on every row**.
#
# So this notebook reads the **pooled panel IC** through the one shared implementation in
# `case_studies/exness_btc_8h/_model_reading.py` - on one name it reduces exactly to that name's
# own Spearman correlation through time. Every IC in this notebook arrives with a **Newey-West
# interval** and with the **per-fold sign consistency** beside it.
#
# ## Learning objectives
#
# - Audit the complete prediction population by label, family, configuration and checkpoint.
# - Read predictive diagnostics on a panel too narrow for a cross-sectional statistic.
# - Report an IC as an interval and a fold pattern, never as a point estimate alone.
# - Evaluate R1 (the early-close gate) declaratively, from `setup.yaml::kill_criteria`.
# - Keep causal estimates separate from predictive-family evidence.

# %%
"""Analyze the complete exness_btc_8h validation-prediction population; evaluate R1."""

import json

import numpy as np
import plotly.express as px
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr

import utils.style  # noqa: F401
from case_studies.exness_btc_8h._features import session_book_of
from case_studies.exness_btc_8h._model_reading import (
    REGISTRY_IC_IS_NULL_REASON,
    hac_lag_for_label,
    outcome_column_for,
    pooled_panel_ic_series,
    prediction_staleness_by_book,
    summarise_pooled_ic,
)
from case_studies.research import CausalResult, Result, open_study, superseded_members
from case_studies.research.results import PredictionResult
from case_studies.utils.conformal import (
    sizing_conformal_lag,
    walk_forward_conformal_coverage,
)
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY = "exness_btc_8h"
N_BUCKETS = 5
MIN_FOLD_SLOTS = 20
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None

# %% [markdown]
# ## Load the canonical population

# %%
case_dir = get_case_study_dir(CASE_STUDY)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
primary_label = setup["labels"]["primary"]
configured_labels = [primary_label, *setup["labels"].get("variants", [])]
LABEL_HORIZONS = {k: int(str(v).rstrip("Hh")) for k, v in setup["labels"]["horizons"].items()}
HOLDOUT_START = str(setup["evaluation"]["holdout_start"])
SESSION_BOOKS = list(setup["decision"]["session_filter_values"])
CLASSIFICATION_EVAL = dict(setup["labels"].get("classification_eval_label") or {})
KILL_CRITERIA = dict(setup.get("kill_criteria") or {})

study = open_study(CASE_STUDY, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
catalog = study.predictions.table().filter(
    (pl.col("identity_status") == "current")
    & (pl.col("execution_tier") == "canonical")
    & (pl.col("split") == "validation")
)
retired = superseded_members(study, member_kind="prediction")
if retired:
    catalog = catalog.filter(~pl.col("prediction_hash").is_in(list(retired)))

if catalog.is_empty():
    raise RuntimeError("no current canonical validation predictions are registered")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("model analysis requires complete prediction sets")
if catalog.filter(~pl.col("artifact_available")).height:
    raise RuntimeError("model analysis requires every prediction artifact")
if "causal_dml" in set(catalog.get_column("family")):
    raise RuntimeError("causal DML must not enter the predictive catalog")

identity_columns = [
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
]
if catalog.select(identity_columns).n_unique() != catalog.height:
    raise RuntimeError("prediction rows do not retain one complete model identity")

print(f"{catalog.height} current canonical validation prediction sets")
print(f"registry rows carrying an ic_mean: {catalog.filter(pl.col('ic_mean').is_not_null()).height}")
print(REGISTRY_IC_IS_NULL_REASON)

# %% [markdown]
# ## Compare the assembled population against the configured menu
#
# `FITTED_FAMILIES` is `{linear, gbm}` only, because that is the whole menu this bot declares:
# `deep_learning`, `tabular_dl` and `causal_dml` are DELIBERATELY ABSENT from
# `config/training/*.yaml` (not commented out), so there is no excluded-but-configured member to
# name - unlike `exness_gold_sess`, whose menus were copied whole from `exness_fx_d1`.

# %% tags=["results"]
FITTED_FAMILIES = {"linear", "gbm"}
configured_members = {
    (label, family, config["config_name"])
    for label in configured_labels
    for family in FITTED_FAMILIES
    for config in load_configs(CASE_STUDY, label, family=family)
}
present_members = set(catalog.select("label", "family", "config_name").unique().iter_rows())
missing_members = sorted(configured_members - present_members)
unexpected_members = sorted(present_members - configured_members)
if missing_members or unexpected_members:
    raise RuntimeError(
        "the assembled population does not match the configured menu; "
        f"missing {missing_members}, unexpected {unexpected_members}"
    )
if set(catalog.get_column("label")) != set(configured_labels):
    raise RuntimeError("the canonical prediction population does not cover every configured label")

# %% tags=["results"]
population_summary = (
    catalog.group_by("label", "family")
    .agg(
        pl.col("config_name").n_unique().alias("configurations"),
        pl.len().alias("prediction_sets"),
        pl.col("checkpoint_value").n_unique().alias("checkpoints"),
        pl.col("complete").all().alias("complete"),
    )
    .sort("label", "family")
)
display(population_summary)
EXPECTED_TOTAL = int(KILL_CRITERIA.get("r1_expected_prediction_sets", 356))
if catalog.height != EXPECTED_TOTAL:
    print(
        f"  NOTE: assembled population is {catalog.height} prediction sets, not the declared "
        f"{EXPECTED_TOTAL} (setup.yaml::kill_criteria.r1_expected_prediction_sets = 28 linear + "
        "15x10 GBM checkpoints, x 2 labels). A different count must be explained in "
        "bots/exness_btc_8h/BOT.md before R1 or K5 are read."
    )
print(
    f"total prediction sets in the analysed population: {catalog.height}; every one of them "
    "advances to 13_backtest UNLESS R1 below closes the generation, and every one of them is a "
    "member of the trial count K recorded in bots/exness_btc_8h/BOT.md"
)

# %% [markdown]
# ## Load every prediction artifact
#
# The `session_book` each slot belongs to is derived from the decision timestamp through
# `_features.session_book_of` - the same rule `05_evaluation` and `06_linear` used.

# %%
frames = []
for row in catalog.iter_rows(named=True):
    result = Result.open(study, row["prediction_hash"])
    if not isinstance(result, PredictionResult):
        raise TypeError(f"{row['prediction_hash']} is not a prediction result")
    frame = result.load()
    required = {"symbol", "timestamp", "fold", "prediction", "actual"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction {result.hash} lacks canonical columns: {sorted(missing)}")
    if frame.select("symbol", "timestamp", "fold").n_unique() != frame.height:
        raise ValueError(f"prediction {result.hash} has duplicate eligible keys")
    frames.append(
        frame.with_columns(
            pl.lit(row["label"]).alias("label"),
            pl.lit(row["family"]).alias("family"),
            pl.lit(row["config_name"]).alias("config_name"),
            pl.lit(row["checkpoint_value"]).alias("checkpoint_value"),
            pl.lit(row["prediction_hash"]).alias("prediction_hash"),
        )
    )
predictions = pl.concat(frames, how="diagonal_relaxed")

slots = predictions.select("timestamp").unique().sort("timestamp")
books = slots.with_columns(session_book_of(slots["timestamp"]).alias("session_book"))
predictions = predictions.join(books, on="timestamp", how="left")
if predictions["session_book"].null_count():
    raise RuntimeError("a prediction timestamp is not a declared decision instant")

latest = predictions["timestamp"].max()
if str(latest)[:10] >= HOLDOUT_START:
    raise RuntimeError(
        f"a validation prediction reaches {latest}, at or past holdout_start {HOLDOUT_START}"
    )
print(
    f"{predictions.height:,} prediction rows over {predictions['timestamp'].n_unique():,} slots "
    f"and {predictions['symbol'].n_unique()} instrument; latest {latest} < holdout {HOLDOUT_START}"
)

# %% [markdown]
# ## Predictive diagnostics: an interval and a fold pattern, never a point estimate
#
# For every prediction set in the population: the pooled panel IC, its Newey-West standard error
# at a lag measured on this label's own decision grid, sign consistency, positive-fold share and
# ICIR. This is the frame R1 below reads directly.

# %% tags=["results"]
rows = []
for row in catalog.iter_rows(named=True):
    frame = predictions.filter(pl.col("prediction_hash") == row["prediction_hash"])
    single = frame.filter(pl.col("symbol") == frame["symbol"][0])["timestamp"]
    overlap, hac_lag = hac_lag_for_label(single, LABEL_HORIZONS[row["label"]])
    outcome_col = outcome_column_for(frame)
    summary = summarise_pooled_ic(
        pooled_panel_ic_series(frame, outcome_col=outcome_col),
        hac_lag=hac_lag,
        min_fold_slots=MIN_FOLD_SLOTS,
    )
    rows.append(
        {
            "label": row["label"],
            "family": row["family"],
            "config_name": row["config_name"],
            "checkpoint_value": row["checkpoint_value"],
            "scored_against": outcome_col,
            "overlap_slots": overlap,
            **{k: v for k, v in summary.items() if k != "fold_ics"},
            "fold_ics": str([round(v, 5) for v in summary.get("fold_ics", [])]),
            "prediction_hash": row["prediction_hash"],
        }
    )
diagnostics = pl.DataFrame(rows).sort(["label", "family", "ic_mean"], descending=[False, False, True])
hac_lags = sorted(set(diagnostics["hac_lags"].to_list()))
overlaps = sorted(set(diagnostics["overlap_slots"].to_list()))
print(
    f"measured overlap on this grid: {overlaps} later decisions start inside a holding window, "
    f"so the requested Newey-West floor is {[o + 1 for o in overlaps]}"
)
print(f"lags actually APPLIED: {hac_lags}")
with pl.Config(tbl_rows=40, tbl_cols=16, tbl_width_chars=240):
    display(
        diagnostics.select(
            "label", "family", "config_name", "checkpoint_value", "n_slots", "ic_mean",
            "hac_se", "hac_t", "hac_p", "hac_lags", "n_folds", "sign_consistency",
            "positive_fold_share", "icir", "fold_ics",
        ).head(40)
    )

# %% tags=["results"]
by_family = (
    diagnostics.group_by("label", "family")
    .agg(
        candidates=pl.len(),
        best_ic=pl.col("ic_mean").max(),
        median_ic=pl.col("ic_mean").median(),
        worst_ic=pl.col("ic_mean").min(),
        max_abs_hac_t=pl.col("hac_t").abs().max(),
        clearing_1p96=(pl.col("hac_t").abs() >= 1.96).sum(),
        all_folds_agree=(pl.col("sign_consistency") >= 1.0).sum(),
        three_of_four_agree=(pl.col("sign_consistency") >= 0.75).sum(),
        icir_above_1=(pl.col("icir").abs() >= 1.0).sum(),
    )
    .with_columns(expected_under_null=(pl.col("candidates") * 0.05).round(1))
    .sort("label", "family")
)
display(by_family)

# %% [markdown]
# ## Persist the complete diagnostics frame (mentor gate material M1, 2026-09-10)
#
# Every column R1 below reads - `ic_mean`, `hac_se`, `hac_t`, `hac_p`, `sign_consistency`,
# `fold_ics`, plus `label`, `family`, `config_name`, `checkpoint_value` and the rest of what
# `summarise_pooled_ic` returns - written for all 356 rows BEFORE the R1 cell runs, so a closed
# generation still leaves a readable record of every candidate's OOF IC and its inference, not
# only the leader. This cell fits nothing and registers nothing (module docstring, "What it
# writes": only this parquet/CSV and, two cells down, the R1 verdict JSON) - it is a read of the
# `diagnostics` frame already built above, not a new computation.

# %% tags=["results"]
diagnostics_path = case_dir / "evaluation" / "phase4_ic_diagnostics.parquet"
diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
diagnostics.write_parquet(diagnostics_path)
diagnostics.write_csv(diagnostics_path.with_suffix(".csv"))
print(f"{diagnostics.height} rows written to {diagnostics_path} (+ .csv)")

# %% [markdown]
# ## R1 - the early-close gate (`setup.yaml::kill_criteria`, approved verbatim 2026-09-10;
# M2 resolved 2026-09-11, option B - code narrowed to the approved B2 text)
#
# **Rule** (`bots/exness_btc_8h/BOT.md` §Kill criteria, B2, as approved): for every prediction set
# in `diagnostics` above, apply Benjamini-Hochberg across the pooled OOF HAC p-values at
# `kill_criteria.r1_fdr_alpha`. A prediction set **clears R1** when its BH-adjusted p-value is
# below `r1_fdr_alpha` alone - BH significance is the ONLY gating condition, exactly as B2's
# approved text says ("có HAC p < 0.05 sau BH"). Sign consistency (4 of 4 folds pointing the
# configuration's own way, `r1_required_sign_consistency`) is computed and REPORTED alongside
# every row for context - it is not a pre-registered condition of B2 and does not enter the
# pass/fail test. `r1_ic_star_validation` is likewise reported for context - how far the leading
# candidate sits from the correlation that would clear the in-sample breakeven cost - and also
# does **not** enter the pass/fail test, which is BH significance alone
# (`kill_criteria.gate_condition: bh_fdr_only`, M2 resolved 2026-09-11).
#
# **If ZERO prediction sets clear R1 and `kill_criteria.r1_early_close` is true, this cell RAISES
# before any row reaches the handoff section below** - the generation closes, the phase-5 backtest
# sweep (K5 = 2,136) is NOT run, and the verdict is written to
# `evaluation/r1_early_close_verdict.json` for BOT.md to cite. This is a report BY DEFAULT
# (`roadmap.md:297`, R1's own description in BOT.md) that the user's 2026-09-10 approval turned
# into a gate for this generation specifically.

# %% tags=["results"]
r1_early_close = bool(KILL_CRITERIA.get("r1_early_close", False))
r1_ic_star = float(KILL_CRITERIA.get("r1_ic_star_validation", float("nan")))
r1_alpha = float(KILL_CRITERIA.get("r1_fdr_alpha", 0.05))
r1_min_sign_consistency = float(KILL_CRITERIA.get("r1_required_sign_consistency", 1.0))

r1_pool = diagnostics.filter(pl.col("hac_p").is_not_null())
if r1_pool.height != diagnostics.height:
    raise RuntimeError(
        f"{diagnostics.height - r1_pool.height} of {diagnostics.height} prediction sets carry a "
        "null hac_p (too few validation slots for summarise_pooled_ic to fit a fold); R1 cannot "
        "be evaluated on an incomplete pool"
    )
p_values = r1_pool.get_column("hac_p").to_list()
fdr = benjamini_hochberg_fdr(p_values, alpha=r1_alpha, return_details=True)
r1_pool = r1_pool.with_columns(
    pl.Series("hac_p_bh", fdr["adjusted_p_values"]),
    pl.Series("bh_significant", fdr["rejected"]),
).with_columns(
    # M2 resolved 2026-09-11, option B: clears_r1 follows the approved B2 text exactly - BH
    # significance is the ONLY gating condition. sign_consistency is still computed above (it is
    # a column of `diagnostics`) and is REPORTED below (n_sign_consistent_4of4, the display
    # table) but does not enter clears_r1.
    pl.col("bh_significant").alias("clears_r1")
)
n_pool = r1_pool.height
n_bh_significant = int(r1_pool.get_column("bh_significant").sum())
n_clears_r1 = int(r1_pool.get_column("clears_r1").sum())
# REPORT ONLY (not gating, M2 resolved 2026-09-11 option B): how many prediction sets have all 4
# validation folds pointing the configuration's own way. Printed and persisted for the reader;
# never combined into clears_r1.
n_sign_consistent_4of4 = int(
    (r1_pool.get_column("sign_consistency") >= r1_min_sign_consistency).sum()
)
verdict = {
    "case_study": CASE_STUDY,
    "n_prediction_sets": n_pool,
    "n_bh_significant": n_bh_significant,
    "n_clears_r1": n_clears_r1,
    "fdr_alpha": r1_alpha,
    "ic_star_validation": r1_ic_star,
    "gate_condition": "bh_fdr_only",
    "sign_consistency_reported_threshold": r1_min_sign_consistency,
    "n_sign_consistent_4of4_reported": n_sign_consistent_4of4,
    "early_close_rule_active": r1_early_close,
    "generation_closed": bool(r1_early_close and n_clears_r1 == 0),
    "best_ic_mean": float(r1_pool.get_column("ic_mean").max()),
    "best_config": r1_pool.sort("ic_mean", descending=True).row(0, named=True)["config_name"],
    "leading_ic_over_ic_star": (
        float(r1_pool.get_column("ic_mean").max() / r1_ic_star) if r1_ic_star else None
    ),
}
verdict_path = case_dir / "evaluation" / "r1_early_close_verdict.json"
verdict_path.parent.mkdir(parents=True, exist_ok=True)
verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True))
print(
    f"R1: {n_clears_r1}/{n_pool} prediction sets clear BH p < {r1_alpha} "
    f"({n_bh_significant}/{n_pool} BH-significant, BH is the only gating condition - M2 option B, "
    f"resolved 2026-09-11). REPORTED, not gating: {n_sign_consistent_4of4}/{n_pool} have "
    f"sign_consistency >= {r1_min_sign_consistency} (4/4 folds). Leading IC "
    f"{verdict['best_ic_mean']:+.4f} ({verdict['best_config']}), which is "
    f"{verdict['leading_ic_over_ic_star']:.2%} of IC* = {r1_ic_star} (validation)."
)
print(f"verdict written to {verdict_path}")
display(
    r1_pool.sort("hac_p_bh")
    .select("label", "family", "config_name", "checkpoint_value", "ic_mean", "hac_p", "hac_p_bh",
            "bh_significant", "sign_consistency", "clears_r1")
    .head(20)
)
if verdict["generation_closed"]:
    raise RuntimeError(
        f"R1 EARLY-CLOSE GATE TRIPPED: 0 of {n_pool} prediction sets clear BH p < {r1_alpha} "
        "(BH is the only gating condition, M2 resolved 2026-09-11 option B; "
        f"{n_sign_consistent_4of4}/{n_pool} are sign-consistent 4/4 folds, reported only). "
        "kill_criteria.r1_early_close is true, so this generation is CLOSED: the phase-5 "
        "backtest sweep (13_backtest, K5 = 2,136) MUST NOT run. Verdict written to "
        f"{verdict_path} for bots/exness_btc_8h/BOT.md to cite."
    )
print(
    f"R1 does not close this generation ({n_clears_r1} prediction set(s) clear it); "
    "13_backtest may proceed. R1 remains a REPORT for phase 5 (roadmap.md:297), not itself a "
    "selection - selection happens on validation backtest Sharpe at the cumulative trial count."
)

# %% [markdown]
# ### Fold stability, drawn

# %% tags=["results"]
fold_figure = px.box(
    diagnostics.to_pandas(),
    x="family",
    y="ic_mean",
    color="family",
    facet_row="label",
    points="all",
    hover_data=["config_name", "checkpoint_value", "hac_t", "sign_consistency"],
    title="Pooled panel IC across the population, by family and label",
    labels={"family": "Model family", "ic_mean": "Pooled panel IC (validation)"},
)
fold_figure.show()

# %% tags=["results"]
consistency_figure = px.scatter(
    diagnostics.to_pandas(),
    x="ic_mean",
    y="sign_consistency",
    color="family",
    facet_row="label",
    hover_data=["config_name", "checkpoint_value", "hac_t", "icir"],
    title="Per-fold sign consistency against pooled panel IC",
    labels={
        "ic_mean": "Pooled panel IC (validation)",
        "sign_consistency": "Share of folds pointing the configuration's own way",
    },
)
consistency_figure.show()

# %% [markdown]
# ## Does a score go stale inside a session book?

# %% tags=["results"]
leaders = (
    diagnostics.sort(["label", "family", "ic_mean"], descending=[False, False, True])
    .group_by("label", "family", maintain_order=True)
    .first()
)
stale_rows = []
for row in leaders.iter_rows(named=True):
    frame = predictions.filter(pl.col("prediction_hash") == row["prediction_hash"])
    for record in prediction_staleness_by_book(frame).iter_rows(named=True):
        stale_rows.append(
            {
                "label": row["label"],
                "family": row["family"],
                "config_name": row["config_name"],
                "checkpoint": row["checkpoint_value"],
                **record,
            }
        )
staleness = pl.DataFrame(stale_rows).sort(["label", "family", "book"])
flagged = staleness.filter(pl.col("staleness") > 0.50)
print(
    f"{flagged.height} of {staleness.height} (label, family, book) triples repeat on more than "
    "50 % of consecutive slots"
)
with pl.Config(tbl_rows=staleness.height):
    display(staleness)

# %% [markdown]
# ## Prediction agreement

# %% tags=["results"]
primary_leaders = leaders.filter(pl.col("label") == primary_label)
primary_rows = predictions.filter(
    (pl.col("label") == primary_label)
    & pl.col("prediction_hash").is_in(primary_leaders["prediction_hash"].to_list())
)
print(f"Score agreement is computed on {primary_label}, the primary label in the menu")
wide = primary_rows.pivot(
    on="prediction_hash", index=["symbol", "timestamp", "fold"], values="prediction"
)
prediction_columns = [c for c in wide.columns if c not in {"symbol", "timestamp", "fold"}]
if any(wide.get_column(column).null_count() for column in prediction_columns):
    raise RuntimeError("a representative is missing scores at keys the others cover")
display(wide.select(prediction_columns).corr())

# %% [markdown]
# ## Ranking shape: buckets taken through time, on the one name
#
# The template buckets the cross-section at each decision instant. **Not available here**: one
# instrument at every slot. This bot is a per-asset signal-timing book
# (`setup.yaml::mapping.class`), so the question is the time-series one - *when this model scored
# BTCUSD in its own top fifth, what did it go on to do?* Buckets are quintiles of the score
# distribution over the validation window.

# %% tags=["results"]
bucket_rows = []
for row in leaders.iter_rows(named=True):
    frame = predictions.filter(pl.col("prediction_hash") == row["prediction_hash"])
    for symbol in sorted(frame["symbol"].unique().to_list()):
        outcome_col = outcome_column_for(frame)
        part = frame.filter(pl.col("symbol") == symbol).drop_nulls(["prediction", outcome_col])
        if part.height < N_BUCKETS * 10:
            continue
        ranked = part.sort("prediction").with_columns(
            ((pl.int_range(pl.len()) * N_BUCKETS) // pl.len())
            .clip(upper_bound=N_BUCKETS - 1)
            .alias("bucket")
        )
        for bucket, values in ranked.group_by("bucket"):
            bucket_rows.append(
                {
                    "label": row["label"],
                    "family": row["family"],
                    "config_name": row["config_name"],
                    "symbol": symbol,
                    "bucket": bucket[0],
                    "slots": values.height,
                    "mean_realized": float(values[outcome_col].mean()),
                }
            )
buckets = pl.DataFrame(bucket_rows).sort(["label", "family", "symbol", "bucket"])
monotone = (
    buckets.group_by("label", "family", "symbol")
    .agg(
        spread=pl.col("mean_realized").sort_by("bucket").last()
        - pl.col("mean_realized").sort_by("bucket").first(),
        buckets=pl.len(),
    )
    .sort("label", "family", "symbol")
)
with pl.Config(tbl_rows=buckets.height):
    display(buckets)
display(monotone)

# %% [markdown]
# ## Coverage of the widths that size positions

# %% tags=["results"]
conformal_rows = []
for row in leaders.iter_rows(named=True):
    frame = predictions.filter(pl.col("prediction_hash") == row["prediction_hash"])
    embargo_steps = sizing_conformal_lag(CASE_STUDY, row["label"])
    for record in walk_forward_conformal_coverage(frame, embargo_steps=embargo_steps):
        conformal_rows.append(
            {
                "label": row["label"],
                "family": row["family"],
                "config_name": row["config_name"],
                "embargo_slots": embargo_steps,
                **record,
            }
        )
conformal = pl.DataFrame(conformal_rows).sort("label", "nominal_level", "family")
with pl.Config(tbl_rows=conformal.height, tbl_cols=conformal.width, tbl_width_chars=200):
    display(conformal)

# %% [markdown]
# ## No classification label this generation
#
# See `06_linear.py` section 5.

# %% tags=["results"]
if CLASSIFICATION_EVAL:
    raise RuntimeError("classification_eval_label is non-empty but this section was not written for it")
print("no classification label is declared")

# %% [markdown]
# ## Causal evidence remains separate
#
# `11_causal_dml` is not copied into this case study (`config/training/*.yaml`: "DELIBERATELY
# ABSENT"), so the registry holds no causal run.

# %% tags=["results"]
causal_rows = []
for label in configured_labels:
    try:
        causal = CausalResult.one(study, label=label)
    except ValueError as exc:
        print(f"{label}: no causal estimate registered ({exc}); 11_causal_dml is not yet copied")
        continue
    causal_rows.append({"label": label, "causal_hash": causal.hash, **causal.metrics})
if causal_rows:
    display(pl.DataFrame(causal_rows).sort("label"))

# %% [markdown]
# ## Handoff to `13_backtest`
#
# Reached only if R1 did not close the generation. Every complete catalog row, including every
# checkpoint, advances. Three session books (`all`, `no_swap_night`, `us_hours`) are declared,
# and unlike `exness_gold_sess` (two ENGINE books plus one ARITHMETIC pooled sleeve, because a
# broker holds one net position per symbol and a London-plus-New-York pooled book would collide
# two live holds), all three of this bot's books are directly engine-run: `all` is the real,
# untouched grid, and `no_swap_night` / `us_hours` are disjoint row-level filters of it
# (`setup.yaml::decision.session_filter_definitions`). No arithmetic pooled book is needed or
# declared.

# %% tags=["results"]
handoff = catalog.select([*identity_columns, "cv_identity", "complete"]).sort(
    "label", "family", "config_name", "checkpoint_value"
)
books = len(SESSION_BOOKS)
specs = len(setup["backtest"]["sweep"]["signal_specs"])
k5 = handoff.height * books * specs
print(
    f"{handoff.height} prediction sets x {books} session books ({SESSION_BOOKS}) x {specs} "
    f"signal specs = {k5} signal-stage trials at the declared sweep size"
)
if k5 != int(KILL_CRITERIA.get("r1_expected_prediction_sets", 356)) * books * specs:
    raise RuntimeError(
        f"K5 computed here ({k5}) does not match the declared arithmetic "
        f"({KILL_CRITERIA.get('r1_expected_prediction_sets')} x {books} x {specs}); "
        "bots/exness_btc_8h/BOT.md must be corrected before 13_backtest runs"
    )
handoff

# %% [markdown]
# ## Key takeaways
#
# - Model identity includes label, family, configuration, checkpoint and validation protocol.
# - **The registry's cross-sectional IC is null on a one-name panel and always will be.**
# - **R1 is evaluated declaratively from `setup.yaml::kill_criteria`**, not hard-coded: if it
#   closes the generation, this notebook raises before the handoff section runs.
# - The per-book staleness screen catches a model that has learnt the session and nothing else.
# - Buckets are taken through time on the one name, because a one-name cross-section has no shape.
# - The equal-weight validation backtest receives the complete prediction population (if R1
#   allows it); selection happens there, on Sharpe, at the cumulative trial count.
# - Causal estimates remain a separate form of evidence.
