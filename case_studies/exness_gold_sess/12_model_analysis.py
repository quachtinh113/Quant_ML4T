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
# # Model analysis - Exness Gold Sessions
#
# **Chapter reference**: Volume 1, Chapters 11-15.
# **Docker image**: `ml4t` (default).
# **Prerequisites**: `06_linear` and `07_gbm` must have published their canonical validation
# populations into the workspace this notebook opens.
# **What it writes**: nothing. It reads the registry and the prediction artifacts, fits nothing and
# registers nothing.
#
# This notebook reads the complete registered validation-prediction population of the bot
# `exness_gold_sess` (`bots/exness_gold_sess/BOT.md`) and compares model families, checkpoints, fold
# stability, prediction agreement and uncertainty **without choosing a model for deployment**. Every
# exact model configuration continues to the equal-weight backtest; validation backtest Sharpe
# performs selection later.
#
# **Nothing here reads the holdout.** The catalog is filtered to `split == "validation"`, and the
# maximum prediction timestamp is asserted below `evaluation.holdout_start` as a second,
# independent check. The holdout stages `17_holdout_predictions` and `18_holdout_backtest` are the
# only ones that ever score the sealed window.
#
# ## The one thing that is different from the template
#
# The template's model analysis is built on the **cross-sectional** information coefficient: rank
# the constituents within each decision instant, correlate with the outcome, average through time.
# This bot's panel is **two metals**, and that statistic is not defined on it -
# `case_studies/utils/registry/metrics.py:113` scores it under a hard-coded `min_obs=5`, so the
# registry's `ic_mean`, `ic_std`, `ic_t` and `ic_n_days` are **null on every row**. Lowering the
# floor would not help: a Spearman over `n = 2` takes only the values +/-1.
#
# So this notebook reads the **pooled panel IC** - the same statistic `05_evaluation` used on the
# features and `06_linear` / `07_gbm` used on the models - through the one shared implementation in
# `case_studies/exness_gold_sess/_model_reading.py`. And it never reports a pooled point estimate
# on its own: every IC in this notebook arrives with a **Newey-West interval** and with the
# **per-fold sign consistency** beside it. An average over four validation years can come from four
# years of the same weak effect or from one year of a strong one, and a pooled mean cannot tell the
# two apart.
#
# The Newey-West bandwidth is built from an overlap **measured on the decision grid** - one later
# decision starts inside every 8-hour hold, so the floor is 2 - but the floor is not the lag that
# gets used. `compute_ic_hac_stats` applies `max(label_horizon - 1, Newey-West auto)`, and on a
# series of a few thousand slots the automatic bandwidth is about 9. The measured overlap
# guarantees the interval cannot be narrower than the dependence the grid actually has; the
# automatic rule usually makes it wider. Both numbers are printed, and what is reported as
# `hac_lags` is what was applied.
#
# ## Learning objectives
#
# - Audit the complete prediction population by label, family, configuration and checkpoint.
# - Read predictive diagnostics on a panel too narrow for a cross-sectional statistic.
# - Report an IC as an interval and a fold pattern, never as a point estimate alone.
# - Keep causal estimates separate from predictive-family evidence.

# %%
"""Analyze the complete exness_gold_sess validation-prediction population."""

import numpy as np
import plotly.express as px
import polars as pl
import yaml
from IPython.display import display

import utils.style  # noqa: F401
from case_studies.exness_gold_sess._features import session_of
from case_studies.exness_gold_sess._model_reading import (
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
CASE_STUDY = "exness_gold_sess"
# Buckets are taken **within a metal through time**, not across the panel at a slot. Five buckets
# of a two-name cross-section is one name in bucket 0 and one in bucket 4, which measures nothing.
N_BUCKETS = 5
MIN_FOLD_SLOTS = 20
# This notebook reads; it fits nothing and registers nothing, so it has no preview form. It still
# takes WORKSPACE, because that is what lets a run read an isolated registry rather than the
# published one.
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None

# %% [markdown]
# ## Load the canonical population
#
# A downstream-selectable row must use the current identity schema, carry exact coverage and fold
# metrics, and have its prediction artifact available. Causal DML is deliberately absent because it
# estimates a treatment effect rather than a score.

# %%
case_dir = get_case_study_dir(CASE_STUDY)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
primary_label = setup["labels"]["primary"]
configured_labels = [primary_label, *setup["labels"].get("variants", [])]
LABEL_HORIZONS = {k: int(str(v).rstrip("Hh")) for k, v in setup["labels"]["horizons"].items()}
HOLDOUT_START = str(setup["evaluation"]["holdout_start"])
SESSION_BOOKS = list(setup["decision"]["session_filter_values"])
CLASSIFICATION_EVAL = dict(setup["labels"].get("classification_eval_label") or {})

study = open_study(CASE_STUDY, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
catalog = study.predictions.table().filter(
    (pl.col("identity_status") == "current")
    & (pl.col("execution_tier") == "canonical")
    & (pl.col("split") == "validation")
)
# `identity_status` names the schema version a row was written under, not which generation its
# producer still publishes: a refit leaves the generation it replaced in the registry, complete and
# current under that column. `superseded_members` reads the lineage - the same exclusion
# `13_backtest` applies before it freezes the baseline population, so the analysed catalog and the
# backtested one describe one set of models rather than two.
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
# Each model notebook checks that it produced what it requested. Nothing checks that the requests
# covered what the case study configures, and a population that is internally consistent but short a
# configured model reads exactly like a complete one.
#
# The menus were copied from `exness_fx_d1` (itself a fork of `fx_pairs`) whole, so they declare
# `tabular_dl` and `deep_learning` members that no notebook in this case study produces yet, and
# the two regression labels declare `causal_dml`. Those members are excluded **by family and with a
# reason** rather than passing unnoticed. `FITTED_FAMILIES` is the set of families whose stage
# exists here; when `08`-`11` are copied, extend it and this check starts requiring their members.

# %% tags=["results"]
FITTED_FAMILIES = {"linear", "gbm"}
EXCLUSION_REASON = (
    "declared in the menu copied whole from exness_fx_d1 (PHASE1_SPEC_MENTOR.md section 2.d "
    "forbids trimming it by taste), but its stage is not yet copied into exness_gold_sess; "
    "phase 4 of bots/exness_gold_sess/BOT.md fits the linear baseline and the GBM grid"
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
        pl.len().alias("prediction_sets"),
        pl.col("checkpoint_value").n_unique().alias("checkpoints"),
        pl.col("complete").all().alias("complete"),
    )
    .sort("label", "family")
)
display(population_summary)
print(
    f"total prediction sets in the analysed population: {catalog.height}; every one of them "
    "advances to 13_backtest, and every one of them is a member of the trial count K recorded "
    "in bots/exness_gold_sess/BOT.md"
)

# %% [markdown]
# ## Load every prediction artifact
#
# Producers use `symbol`, `timestamp`, `fold`, `prediction` and `actual`. The full catalog identity
# is added to each frame so two checkpoints cannot collapse into one plot label or fold count. The
# `session` each slot belongs to is derived from the decision timestamp through
# `_features.session_of` - the same rule `05_evaluation` used - so the per-book screens below run on
# the book a `SESSION_FILTER` would actually trade.

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
books = slots.with_columns(session_of(slots["timestamp"]).alias("session"))
predictions = predictions.join(books, on="timestamp", how="left")
if predictions["session"].null_count():
    raise RuntimeError("a prediction timestamp is not a declared decision instant")

latest = predictions["timestamp"].max()
if str(latest)[:10] >= HOLDOUT_START:
    raise RuntimeError(
        f"a validation prediction reaches {latest}, at or past holdout_start {HOLDOUT_START}"
    )
print(
    f"{predictions.height:,} prediction rows over {predictions['timestamp'].n_unique():,} slots "
    f"and {predictions['symbol'].n_unique()} metals; latest {latest} < holdout {HOLDOUT_START}"
)

# %% [markdown]
# ## Predictive diagnostics: an interval and a fold pattern, never a point estimate
#
# For every prediction set in the population: the **pooled panel IC**, its **Newey-West** standard
# error at a lag measured on this label's own decision grid, the **share of validation folds
# pointing the configuration's own way**, the share pointing up, and the **ICIR** - the mean fold IC
# over its dispersion across folds.
#
# Why all four and not just the mean. `sign_consistency` and `positive_fold_share` differ where a
# configuration is steadily inverted: the first asks "did it repeat", the second "did it repeat
# *upwards*", and a bot that can go short cares about the first while a long-only reading cares
# about the second. `icir` separates a small agreement that repeats from a larger one that came out
# of a single window. `hac_t` is the only one of the four that is an inferential statistic, and it
# is the one that must never be read without its multiplicity context: this population is hundreds
# of candidates, so at the 5 % level dozens of |t| >= 1.96 are expected under a null of no skill.

# %% tags=["results"]
rows = []
for row in catalog.iter_rows(named=True):
    frame = predictions.filter(pl.col("prediction_hash") == row["prediction_hash"])
    one_metal = frame.filter(pl.col("symbol") == frame["symbol"][0])["timestamp"]
    overlap, hac_lag = hac_lag_for_label(one_metal, LABEL_HORIZONS[row["label"]])
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
print(
    "  The two differ, and that is correct rather than a discrepancy. compute_ic_hac_stats "
    "applies max(label_horizon - 1, Newey-West auto) capped at T // 2, where the automatic "
    "bandwidth is floor(4 * (T/100) ** (2/9)) - about 9 on a series of this length. The measured "
    "overlap is the guarantee that the correction can never be NARROWER than the dependence the "
    "grid actually has; the automatic bandwidth is usually wider, which is the conservative "
    "direction. What is reported is what was applied."
)
if min(overlaps) < 1:
    print(
        "  NOTE: a measured overlap below 1 appeared. Every prior measurement on this grid "
        "(02_labels via N_eff, 05_evaluation, tests/test_sessions.py) found exactly 1, so a "
        "smaller value means the decision grid or the label horizon moved."
    )
with pl.Config(tbl_rows=40, tbl_cols=16, tbl_width_chars=240):
    display(
        diagnostics.select(
            "label",
            "family",
            "config_name",
            "checkpoint_value",
            "n_slots",
            "ic_mean",
            "hac_se",
            "hac_t",
            "hac_p",
            "hac_lags",
            "n_folds",
            "sign_consistency",
            "positive_fold_share",
            "icir",
            "fold_ics",
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
    .with_columns(
        expected_under_null=(pl.col("candidates") * 0.05).round(1),
    )
    .sort("label", "family")
)
display(by_family)
print(
    "`expected_under_null` is 5 % of the candidate count: the number of rows expected to clear "
    "|HAC t| >= 1.96 with no skill at all. Read `clearing_1p96` against it, not against zero."
)

# %% [markdown]
# ### Fold stability, drawn
#
# One box per family and label over the population's per-configuration pooled ICs, with the
# individual points shown. A family whose box straddles zero is one whose configurations disagree
# about the sign; a family whose box sits above zero with a tight spread is one where the choice of
# configuration matters little - which is a different and more interesting statement than any single
# leading row.

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

# %% [markdown]
# ### Per-fold sign consistency, drawn
#
# The share of folds pointing each configuration's own way, against its pooled IC. The interesting
# region is the top right and the top left - a configuration whose four folds all agree, in either
# direction - and the interesting *absence* is a population with nothing there. With four folds the
# statistic takes five values (0, 0.25, 0.5, 0.75, 1), and a share of 1.0 has a one-in-eight chance
# under a coin flip, so a handful of them in a population of hundreds is expected.

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
#
# `05_evaluation` ran a staleness screen **per session book** on the features and it stopped four
# columns before they were scored: `d1_max_dd_63d` at 0.51 pooled, and `is_venue_dst`,
# `news_nfp_decision_bar` and `news_nfp_label_window` at 0.93-0.95. The same screen belongs on the
# model's output. A score that repeats inside `london` or `ny` is a constant for the book that
# would trade it, whatever it does on the pooled panel - the failure that stopped `exness_fx_d1`'s
# eight `rank_*` columns, caught here one book early.
#
# Run on the leading configuration of each label and family, which is where a stale score would
# otherwise be most likely to travel unnoticed into `13_backtest`.

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
    "50 % of consecutive slots - the same ceiling 05_evaluation applies to a feature"
)
with pl.Config(tbl_rows=staleness.height):
    display(staleness)

# %% [markdown]
# ## Prediction agreement
#
# Correlation between model scores says whether the families score the same slots the same way. One
# matrix covers one label: scores fitted against different targets are not comparable readings of
# the same quantity, so the pivot takes the primary label.
#
# The pivot index is the canonical eligibility key alone. `actual` is the same realized outcome for
# every model at a given key, but the families do not agree on its float representation, so
# including it splits the rows into disjoint groups and every correlation, including the diagonal,
# comes out NaN.

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
# ## Ranking shape: buckets taken within a metal, through time
#
# The template buckets the cross-section at each decision instant. **That is not available here** -
# five buckets of two names puts one metal in bucket 0 and one in bucket 4 at every slot, which
# measures the sign of a two-point comparison and nothing else. This bot is a per-asset
# signal-timing book (`setup.yaml::mapping.class`), so the question it actually asks is the
# time-series one: *when this model scored this metal in its own top fifth, what did the metal go on
# to do?*
#
# So the buckets are quintiles of each metal's own score distribution over the validation window,
# and the frame reports the mean realized outcome per metal and bucket. A monotone column from
# bucket 0 to bucket 4 is a model whose score orders that metal's own future; a flat one is not.

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
#
# The width reported here is the one `conformal_weighted` allocates with: calibrated per metal on
# every residual known at `t - h`, where `h` is the label's horizon **in decision slots** -
# `HOLDOUT_CONFORMAL_EMBARGO_STEPS` in `case_studies/utils/conformal.py` carries this case study's
# three entries at 1 / 2 / 1 slots, because a step of this panel is half a weekday and not a day. A
# decision is covered when its absolute residual falls inside that half-width.
#
# Read it as a diagnostic of residual dispersion, not as a guarantee. Split conformal's
# finite-sample coverage needs the calibration and evaluation residuals to be exchangeable, and
# metal returns are heteroskedastic and regime-dependent - `04_model_based_features` fitted a
# two-state HMM to each metal precisely because they are. Nothing in the allocation path reads an
# interval or a coverage level: the width stands in for a volatility estimate.

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
# ## The classification label, read on its own terms
#
# `dir_tb_8h` is a three-class label and the registry scores it with a different set of columns from
# the two return labels. `06_linear` section 5 carries the full statement, including why no
# uniqueness sample weights exist for it; this is the population-level reading of the same columns.
# `auc_mean_daily` is null for two independent reasons - the label is not binary, and the
# cross-section is two wide - and both are printed rather than left as an empty cell.

# %% tags=["results"]
if CLASSIFICATION_EVAL:
    cls = catalog.filter(pl.col("label").is_in(sorted(CLASSIFICATION_EVAL)))
    columns = [
        c
        for c in (
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "brier_score",
            "auc_roc",
            "auc_pr",
            "auc_mean_daily",
            "ic_mean",
        )
        if c in cls.columns
    ]
    for column in columns:
        print(f"  {column}: {cls.filter(pl.col(column).is_not_null()).height} of {cls.height} rows")
    print(f"scored against the continuous evaluation label: {CLASSIFICATION_EVAL}")
    summary = (
        cls.select("label", "family", "config_name", "checkpoint_value", *columns)
        .sort("label", "family", "config_name", "checkpoint_value")
    )
    with pl.Config(tbl_rows=30, tbl_cols=14, tbl_width_chars=220):
        display(summary.head(30))
else:
    print("no classification label is declared")

# %% [markdown]
# ## Causal evidence remains separate
#
# A causal result answers whether the configured treatment (`setup.yaml::causal.treatment` =
# `sess_ret_1h`, the one intraday column whose construction window is unambiguous) has an estimated
# effect after adjustment for its declared confounders. It is not predictive-family coverage and it
# does not enter the model or backtest population. `11_causal_dml` is not copied into this case
# study, so the registry holds no causal run and the frame below says so per label rather than
# failing the analysis.

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
# ## Handoff to the equal-weight backtest
#
# Every complete catalog row, including every checkpoint, advances to `13_backtest`. The backtest
# receives these rows directly and records one result per row. Neither this notebook's diagnostics
# nor any IC or causal statistic changes that population, and the Deflated Sharpe Ratio reported
# there is computed against the cumulative trial count in `bots/exness_gold_sess/BOT.md`, which
# counts every one of these rows in every session book and against every signal spec.

# %% tags=["results"]
handoff = catalog.select([*identity_columns, "cv_identity", "complete"]).sort(
    "label", "family", "config_name", "checkpoint_value"
)
# london and ny are engine books; the pooled `both` book is the deterministic 50/50 sleeve sum
# of the two registered return series, built in 19_strategy_analysis rather than run as its own
# engine book (bots/exness_gold_sess/PRICE_GRID_DECLARATION.md section 3.4: the broker holds one
# net position per symbol, so a New York entry lands inside the London hold and the time-exit rule
# would close the whole position at 17:00, killing the New York leg after three hours). It is
# still one spec per (label x prediction set x signal spec), so it counts in K exactly as an
# engine book would - it is only *produced* differently.
books = len(SESSION_BOOKS) + 1
specs = len(setup["backtest"]["sweep"]["signal_specs"])
print(
    f"{handoff.height} prediction sets x {books} session books ({SESSION_BOOKS} run by the engine "
    f"+ the pooled sleeve sum) x {specs} signal specs = {handoff.height * books * specs} "
    "signal-stage trials at the declared sweep size"
)
print(
    "Turnover warning for whoever reads 13_backtest next: the hold is expressed as a broker-level "
    "position rule (time_exit), and a rule-driven exit does not pass through the weight series, so "
    "the registry's avg_turnover column under-reports by roughly half. Honest turnover comes from "
    "fills.parquet / trades.parquet. The sibling bot hit the same trap (bots/exness_fx_d1/BOT.md)."
)
handoff

# %% [markdown]
# ## Key takeaways
#
# - Model identity includes label, family, configuration, checkpoint and validation protocol.
# - **The registry's cross-sectional IC is null on a two-name panel and always will be.** The
#   readable statistic is the pooled panel IC, and it is reported with a Newey-West interval at a
#   measured lag and with the per-fold sign pattern beside it, never as a point estimate alone.
# - The per-book staleness screen catches a model that has learnt the session and nothing else.
# - Buckets are taken within a metal through time, because a two-name cross-section has no shape.
# - The equal-weight validation backtest receives the complete prediction population; selection
#   happens there, on Sharpe, at the cumulative trial count.
# - Causal estimates remain a separate form of evidence.
