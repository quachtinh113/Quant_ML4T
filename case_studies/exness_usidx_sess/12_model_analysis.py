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
# > **Route B fork for the bot `exness_usidx_sess`** (`bots/exness_usidx_sess/BOT.md`): this file
# > began as `case_studies/exness_fx_d1/12_model_analysis.py` with the study id, the loader and the population names
# > changed. **This bot has TWO instruments (`US500`, `USTEC`) and no cross-section**, and one
# > label per experiment workspace. Any sentence below that speaks of five currency pairs, of
# > ranking assets against each other, of `usd_corr_*` / `gold_corr_*` columns or of a dollar
# > regime is template residue describing a different bot; where such prose asserted a NUMBER it
# > has been deleted rather than reworded (see `BOT.md`, Decisions log, 2026-09-08 third pass),
# > and where it survives as a description of the template it says so explicitly. Report any that
# > does neither. This banner follows `case_studies/xau_fx_mt5/13_backtest.py`.

# %% [markdown]
# # Model Analysis - Exness US indices, session bot
#
# (The title said "Exness FX D1" until 2026-09-08, third pass: the wrong bot's name at the top of
# this bot's model analysis.)
#
# This notebook reads the complete registered validation-prediction population of the bot
# `exness_usidx_sess` (`bots/exness_usidx_sess/BOT.md`). It compares model families, checkpoints, fold
# stability, prediction agreement, and uncertainty without choosing a model for deployment. Every
# exact model configuration continues to the equal-weight backtest; validation backtest Sharpe
# performs selection later. Nothing here reads the holdout: the catalog is filtered to
# `split == "validation"` and the holdout stages are the only ones that ever score the sealed
# window.
#
# **Learning objectives**
#
# - Audit the complete prediction population by label, family, configuration, and checkpoint.
# - Compare predictive diagnostics without using them as selection criteria.
# - Evaluate stability and chronological conformal coverage from canonical prediction artifacts.
# - Keep causal estimates separate from predictive-family evidence.
#
# **Book reference**: Chapters 11-15
#
# **Prerequisites**: `06_linear` and `07_gbm`. The template's `08_tabular_dl`, `09_dl_tcn`,
# `10_dl_nlinear`, `10a_dl_lstm` and `11_causal_dml` are not yet copied into this case study
# (phase 4 of the bot fits the linear baseline and the GBM grid); the menu still declares their
# members, and the coverage check below names them as declared exclusions rather than letting
# the population read as complete.

# %%
"""Analyze the complete exness_usidx_sess validation-prediction population."""

import plotly.express as px
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.metrics import cross_sectional_ic

import utils.style  # noqa: F401
from case_studies.research import CausalResult, Result, open_study, superseded_members
from case_studies.research.results import PredictionResult
from case_studies.utils.conformal import (
    sizing_conformal_lag,
    walk_forward_conformal_coverage,
)
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY = "exness_usidx_sess"
# Buckets are cut WITHIN an index over its own scored values, not across the instruments on a
# date, so this is a number of quantiles of one index's own score distribution rather than a
# split of a cross-section. Three, not the five the FX fork uses: a validation fold of this bot
# is about 125 sessions per index, and five buckets of that leaves 25 sessions a bucket before
# any of them is compared with another.
N_BUCKETS = 3
# Sessions one index needs inside one fold before that block's information coefficient is worth
# taking. The FX fork asks for 4 INSTRUMENTS on a date; this bot has two, so `cross_sectional_ic`
# with `min_obs=4` would return nothing on every fold, and `min_obs=2` would return a rank
# correlation over two points, which is +1 or -1 whatever the numbers are. The floor is on rows
# within a block instead, matching 05_evaluation.
MIN_ROWS_PER_BLOCK = 30
# This notebook reads; it fits nothing and registers nothing, so it has no preview form - a
# preview population is not the thing whose assembly it checks. It still takes the pair,
# because WORKSPACE is what lets a run read an isolated registry instead of the published one.
# Study.open(CASE_STUDY) resolved through the repo case directory, which holds a registry only
# where a maintainer worktree has linked one there; anywhere else it read nothing at all.
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None

# %% [markdown]
# ## Load the canonical population
#
# A downstream-selectable row must use the current identity schema, carry exact coverage and fold
# metrics, and have its prediction artifact available. Causal DML is deliberately absent because it
# estimates a treatment effect rather than a cross-sectional score.

# %%
case_dir = get_case_study_dir(CASE_STUDY)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
primary_label = setup["labels"]["primary"]
configured_labels = [primary_label, *setup["labels"].get("variants", [])]
study = open_study(CASE_STUDY, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
catalog = study.predictions.table().filter(
    (pl.col("identity_status") == "current")
    & (pl.col("execution_tier") == "canonical")
    & (pl.col("split") == "validation")
)
# `identity_status` names the schema version a row was written under. It says nothing about
# which generation the row's producer still publishes: a model notebook that refits leaves the
# generation it replaced in the registry, complete and current under that column, so the filter
# above carries retired prediction sets into the analysis. The lineage is what answers it, and
# `superseded_members` reads that - the same exclusion `13_backtest` applies before it freezes
# the baseline population, so the analysed catalog and the backtested one describe one set of
# models rather than two.
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

# %% [markdown]
# ## Compare the assembled population against the configured menu
#
# Each notebook checks that it produced what it requested. Nothing checks that the requests covered
# what the case study configures, and a population that is internally consistent but short a
# configured model reads exactly like a complete one. This compares the assembled population to the
# menu itself.
#
# The menus were copied from `fx_pairs` whole, so they declare `tabular_dl` and `deep_learning`
# members that no notebook in this case study produces yet. Those members are excluded by family
# and reason rather than passing unnoticed, and the exclusions are derived from the menu per label
# rather than listed, so a label that never declares a member does not gain a phantom exclusion
# for it. `FITTED_FAMILIES` is the set of families whose stage exists here; when `08`-`10a` are
# copied, extend it and the check below starts requiring their members.

# %% tags=["results"]
FITTED_FAMILIES = {"linear", "gbm"}
EXCLUSION_REASON = (
    "declared in the menu copied from fx_pairs, but its stage is not yet copied into "
    "exness_usidx_sess (bots/exness_usidx_sess/BOT.md, phase 4 fits linear and gbm)"
)

configured_members = {
    (label, family, config["config_name"])
    for label in configured_labels
    for family in ("linear", "gbm", "tabular_dl", "deep_learning")
    for config in load_configs(CASE_STUDY, label, family=family)
}
excluded_members = {
    (label, family, config_name)
    for label, family, config_name in configured_members
    if family not in FITTED_FAMILIES
}
expected_members = configured_members - excluded_members
present_members = set(catalog.select("label", "family", "config_name").unique().iter_rows())

if excluded_members:
    print(f"Declared exclusions ({EXCLUSION_REASON}):")
    for member in sorted(excluded_members):
        print(f"  {member[0]} / {member[1]} / {member[2]}")

missing_members = sorted(expected_members - present_members)
unexpected_members = sorted(present_members - expected_members)
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
        pl.len().alias("checkpoints"),
        pl.col("complete").all().alias("complete"),
        pl.col("ic_mean").is_not_null().sum().alias("diagnosed_checkpoints"),
    )
    .sort("label", "family")
)
population_summary

# %% [markdown]
# ## Compare predictive diagnostics
#
# The catalog's own `ic_mean` is a cross-sectional statistic and is null on this universe (see
# the section below), so the representatives are chosen on the columns that do exist and the
# reading is done on the time-series coefficient computed further down. The rows shown are
# descriptive representatives for plots: they do not filter the official prediction population
# and do not choose checkpoints for backtesting, which happens on realised net return in
# `13_backtest`.

# %% tags=["results"]
# The FX fork picks one representative per (label, family) by the registry's `ic_mean` and charts
# that column across checkpoints. Both are unavailable here: `ic_mean` is a cross-sectional
# statistic and is null on all 178 prediction sets of this workspace, for the reason set out two
# sections below. Rather than pick a representative by a null column, this notebook reads the
# WHOLE registered population - it is 178 sets, which is small - and computes the statistic this
# universe supports on every one of them.
diagnosed = catalog
representatives = catalog.sort(
    ["label", "family", "config_name", "checkpoint_value", "prediction_hash"], nulls_last=True
)
print(
    f"{representatives.height} prediction sets will be read "
    f"({catalog['ic_mean'].null_count()} of {catalog.height} carry a null cross-sectional IC, "
    "which is all of them)"
)
representatives.group_by("label", "family").agg(
    pl.col("config_name").n_unique().alias("configs"),
    pl.len().alias("prediction_sets"),
    pl.col("checkpoint_value").n_unique().alias("checkpoints"),
).sort("label", "family")

# %% [markdown]
# ## Load canonical prediction columns
#
# Producers use `symbol`, `timestamp`, `fold`, `prediction`, and `actual`. Adding the full catalog
# identity to each frame keeps two checkpoints from collapsing into one plot label or fold count.

# %%
prediction_frames = []
for row in representatives.iter_rows(named=True):
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
    prediction_frames.append(
        frame.with_columns(
            pl.lit(row["label"]).alias("label"),
            pl.lit(row["family"]).alias("family"),
            pl.lit(row["config_name"]).alias("config_name"),
            pl.lit(row["checkpoint_value"]).alias("checkpoint_value"),
            pl.lit(row["prediction_hash"]).alias("prediction_hash"),
        )
    )

representative_predictions = pl.concat(prediction_frames, how="diagonal_relaxed")

# %% [markdown]
# ## The statistic this bot reads, and the one it cannot
#
# The notebook this is forked from measures every model with a **cross-sectional** information
# coefficient: on each decision date, rank the instruments by the model's score, rank them by the
# return they went on to earn, and correlate the two orderings. It calls
# `ml4t.diagnostic.metrics.cross_sectional_ic` with `min_obs=4`, because a rank correlation over
# fewer than four points is noise.
#
# This bot has **two** instruments. `min_obs=4` returns nothing on every fold, and lowering it to
# two would not help: a rank correlation between two points is +1 or -1 whatever the two numbers
# are, so the series would be a coin flip and its average a number with no content. That is also
# why `ic_mean` is null for every row of the registry catalog these predictions came from.
#
# The statistic below is the time-series one, which is what `mapping.class:
# time_series_threshold` says this strategy acts on: **within one index, over the sessions of one
# fold, does a high score precede a high return for that index?** It is built the way
# `05_evaluation` builds it - rank the score and the outcome inside the block, standardise both,
# and average the product, which is the block's Spearman coefficient - and it is reported **per
# index**, because with two of them an average would hide a disagreement rather than summarise
# one. Two indices that point opposite ways have not found a weak signal; they have found nothing,
# or two different things.

# %%
def time_series_ic(frame: pl.DataFrame, *, min_rows: int) -> dict:
    """Per-index Spearman IC of score against outcome inside one block, and its pooled mean."""
    per_symbol: dict[str, float] = {}
    n_rows = 0
    for symbol_block in frame.partition_by("symbol", maintain_order=True):
        valid = symbol_block.select("prediction", "actual").drop_nulls().drop_nans()
        if valid.height < min_rows:
            continue
        rank_p = valid["prediction"].rank("average").to_numpy().astype(float)
        rank_a = valid["actual"].rank("average").to_numpy().astype(float)
        if rank_p.std() < 1e-12 or rank_a.std() < 1e-12:
            continue
        product = ((rank_p - rank_p.mean()) / rank_p.std()) * (
            (rank_a - rank_a.mean()) / rank_a.std()
        )
        per_symbol[symbol_block["symbol"][0]] = float(product.mean())
        n_rows += valid.height
    return {
        "ic_mean": (sum(per_symbol.values()) / len(per_symbol)) if per_symbol else None,
        "ic_by_symbol": per_symbol,
        "ic_spread": (max(per_symbol.values()) - min(per_symbol.values()))
        if len(per_symbol) > 1
        else None,
        "n_periods": n_rows,
        "n_symbols_scored": len(per_symbol),
    }


# %% [markdown]
# ## Fold stability
#
# Fold identifiers are labels, not dates. The table orders validation windows by their earliest
# timestamp and retains checkpoint identity in every row.

# %% tags=["results"]
fold_rows = []
for keys, frame in representative_predictions.group_by(
    "label", "family", "config_name", "checkpoint_value", "prediction_hash", "fold"
):
    stats = time_series_ic(frame, min_rows=MIN_ROWS_PER_BLOCK)
    fold_rows.append(
        {
            "label": keys[0],
            "family": keys[1],
            "config_name": keys[2],
            "checkpoint_value": keys[3],
            "prediction_hash": keys[4],
            "fold": keys[5],
            "validation_start": frame.get_column("timestamp").min(),
            "ic_mean": stats["ic_mean"],
            # The disagreement between the two indices, kept beside the average that hides it.
            "ic_spread": stats["ic_spread"],
            "ic_by_symbol": str(stats["ic_by_symbol"]),
            "n_symbols_scored": stats["n_symbols_scored"],
            "n_decision_times": stats["n_periods"],
        }
    )
fold_metrics = pl.DataFrame(fold_rows).sort("label", "validation_start", "family")
fold_metrics

# %% tags=["results"]
fold_figure = px.box(
    fold_metrics,
    x="family",
    y="ic_mean",
    color="family",
    facet_row="label",
    points="all",
    title="Validation rank correlation varies across chronological folds",
    labels={"family": "Model family", "ic_mean": "Fold mean rank correlation"},
)
fold_figure.show()

# %% [markdown]
# ## Prediction agreement and ranking shape
#
# Correlation between model scores shows whether families score the two indices similarly.
# This one diagnostic is unaffected by the size of the universe - it correlates score columns
# with each other rather than ranking instruments - so it is the FX fork's cell unchanged. One correlation
# matrix covers one label: scores fitted against different targets are not comparable ranks of the
# same quantity, so the pivot takes the primary label the menu declares rather than all three.
# Return buckets and conformal coverage carry no such restriction and run over every label.

# %% tags=["results"]
primary = representative_predictions.filter(pl.col("label") == primary_label)
print(f"Score agreement is computed on {primary_label}, the primary label in the menu")
# The pivot index is the canonical eligibility key alone. `actual` is the same realized return for
# every model at a given key, but the families do not agree on its float representation, so including
# it split the rows into disjoint groups - each score column populated on a different half - and every
# correlation, including the diagonal, came out NaN.
wide = primary.pivot(
    on="prediction_hash",
    index=["symbol", "timestamp", "fold"],
    values="prediction",
)
prediction_columns = [
    column for column in wide.columns if column not in {"symbol", "timestamp", "fold"}
]
if wide.height != primary.height // primary.get_column("prediction_hash").n_unique():
    raise RuntimeError("the agreement pivot did not align every representative on the same keys")
if any(wide.get_column(column).null_count() for column in prediction_columns):
    raise RuntimeError("a representative is missing scores at keys the others cover")
agreement = wide.select(prediction_columns).corr()
agreement

# %% tags=["results"]
# Buckets are cut WITHIN an index over its own scored values, not across the instruments on a
# date. Cutting across a date is what the FX fork does and it needs a cross-section; on two rows
# it would put one index in the bottom bucket and the other in the top on every session, which
# reads as a perfect staircase and means nothing. Cutting within an index answers the question a
# threshold strategy asks: when this model scored this index in its own top third, what followed?
bucket_rows = []
for keys, frame in representative_predictions.group_by(
    "label", "family", "config_name", "checkpoint_value", "prediction_hash", "symbol"
):
    if frame.height < N_BUCKETS * MIN_ROWS_PER_BLOCK:
        continue
    ranked = frame.sort("prediction").with_columns(
        ((pl.int_range(pl.len()) * N_BUCKETS) // pl.len())
        .clip(upper_bound=N_BUCKETS - 1)
        .alias("bucket")
    )
    for bucket, values in ranked.group_by("bucket"):
        bucket_rows.append(
            {
                "label": keys[0],
                "family": keys[1],
                "config_name": keys[2],
                "checkpoint_value": keys[3],
                "prediction_hash": keys[4],
                "symbol": keys[5],
                "bucket": bucket[0],
                "actual": values.get_column("actual").mean(),
            }
        )
bucket_summary = (
    pl.DataFrame(bucket_rows)
    .group_by("label", "family", "config_name", "checkpoint_value", "prediction_hash", "bucket")
    .agg(pl.col("actual").mean().alias("mean_realized_return"))
    .sort("label", "family", "bucket")
)
# Three labels sort as 1d, 21d, 5d, so the default row window hides the middle one entirely and
# a reader would see the same two-thirds coverage this section exists to remove.
with pl.Config(tbl_rows=bucket_summary.height):
    display(bucket_summary)

# %% [markdown]
# ## Coverage of the widths that size positions
#
# The width reported here is the one `conformal_weighted` allocates with: calibrated per symbol on
# every residual known at `t - h`, where `h` is the label's horizon in data steps
# (`HOLDOUT_CONFORMAL_EMBARGO_STEPS` in `case_studies/utils/conformal.py` holds this case
# study's three entries, 1 / 5 / 21 sessions), with a pooled quantile where a symbol has too few
# of its own. A decision is covered when its absolute residual falls inside that half-width.
#
# Read it as a diagnostic of residual dispersion, not as a guarantee. Split conformal's
# finite-sample coverage needs the calibration and evaluation residuals to be exchangeable, and
# currency returns are heteroskedastic and regime-dependent. Nothing in the allocation path reads
# an interval or a coverage level - the width stands in for a volatility estimate, and `n_test`
# counts the decisions a width could be calibrated for.

# %% tags=["results"]
conformal_rows = []
for keys, frame in representative_predictions.group_by(
    "label", "family", "config_name", "checkpoint_value", "prediction_hash"
):
    embargo_steps = sizing_conformal_lag(CASE_STUDY, keys[0])
    for row in walk_forward_conformal_coverage(frame, embargo_steps=embargo_steps):
        conformal_rows.append(
            {
                "label": keys[0],
                "family": keys[1],
                "config_name": keys[2],
                "checkpoint_value": keys[3],
                "prediction_hash": keys[4],
                **row,
            }
        )
conformal = pl.DataFrame(conformal_rows).sort("label", "nominal_level", "family")
with pl.Config(tbl_rows=conformal.height, tbl_cols=conformal.width, tbl_width_chars=200):
    display(conformal)

# %% [markdown]
# ## Causal evidence remains separate
#
# A causal result answers whether the configured momentum treatment has an estimated effect after
# adjustment for its declared confounders. It does not count as predictive-family coverage and does
# not enter the model or backtest population. `11_causal_dml` registers one estimate per label the
# menu declares, so naming a single label here would leave the rest of them unreported. Until that
# stage is copied into this case study the registry holds no causal run, and the frame below says
# so per label instead of failing the analysis.

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
    causal_summary = (
        pl.DataFrame(causal_rows)
        .select(
            "label",
            "causal_hash",
            "n_obs",
            "dml_effect",
            "dml_se_hac",
            "p_value_hac",
            "naive_effect",
            "confounding_bias_pct",
            "refutation_p",
        )
        .sort("label")
    )
    display(causal_summary)

# %% [markdown]
# ## Handoff to the equal-weight backtest
#
# Every complete catalog row, including every checkpoint, advances to `13_backtest`. The backtest
# receives these rows directly and records one result per row. Neither this notebook's descriptive
# representatives nor any IC or causal statistic changes that population.

# %%
handoff = catalog.select([*identity_columns, "cv_identity", "complete"]).sort(
    "label", "family", "config_name", "checkpoint_value"
)
handoff

# %% [markdown]
# ## Key takeaways
#
# - Model identity includes label, family, configuration, checkpoint, and validation protocol.
# - Daily rank correlation, fold stability, bucket shape, and conformal coverage are diagnostics.
# - The equal-weight validation backtest receives the complete prediction population.
# - Causal estimates remain a separate form of evidence.
