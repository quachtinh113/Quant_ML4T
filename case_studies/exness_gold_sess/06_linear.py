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
# # Exness Gold Sessions: the linear baseline on a two-name panel
#
# **Chapter reference**: Volume 1, Chapter 11 (`11_ml_pipeline`), notebooks `02_regularization_paths`
# and `03_logistic_classification`.
# **Docker image**: `ml4t` (default).
# **Prerequisites**: `03_financial_features` and `04_model_based_features` must have written
# `features/financial.parquet` and `features/model_based.parquet`, and `02_labels` the three label
# parquets. Run with `ML4T_OUTPUT_DIR` pointing at a writable experiment; the repository
# `case_studies/exness_gold_sess/run_log/` is never written to.
# **What it writes**: one `training_runs` row and one `prediction_sets` row per label and
# configuration, plus the fitted coefficients and the walk-forward predictions, into the
# experiment's `run_log/`. It selects nothing.
#
# ## What is different here, and why
#
# `06_linear` in the template ranks a cross-section. This bot has **two instruments**, and that is
# not a smaller cross-section - it is a different object. Three consequences run through the whole
# notebook and are stated once, here:
#
# 1. **There are no `rank_*` columns to fit on.** `setup.yaml::features.ranked` is empty by design
#    (`PHASE1_SPEC_MENTOR.md` section 2.d): a percentile over two names takes two values. The
#    eight `rank_*` columns `exness_fx_d1` carried were all STOPped for staleness at five names;
#    at two they would be constant.
# 2. **The registry's `ic_mean` is NULL on every row this notebook produces**, and that is a
#    property of the measurement, not a failure of the fit. `case_studies/_model_reading.py`
#    carries the measurement and the arithmetic; the short version is that
#    `compute_prediction_fold_metrics` scores a cross-sectional Spearman under a hard-coded
#    `min_obs=5` and every decision instant here has 2 rows. Section 4 reads the **pooled panel
#    IC** instead, the same statistic `05_evaluation` used on the features.
# 3. **The comparison that matters is horizon and family, not ranking skill.** Three labels are
#    fitted in one run so that the folds, the features and the metals are held fixed and only the
#    prediction horizon moves.
#
# ## Two declarations this notebook makes before it fits anything
#
# **Nulls.** The feature matrix carries declared nulls on `pre_ret_*h` (12-166 rows) and on
# `overnight_gap` / `prev_sess_ret` (163 rows). scikit-learn raises on a null; LightGBM reads one
# as a branch. `bots/exness_gold_sess/BOT.md` left the choice open for phase 4, and the answer is
# taken from the shared runner rather than invented here: `case_studies/utils/linear.py:402-403`
# fits a `SimpleImputer(strategy="median")` followed by a `StandardScaler` **inside the fold
# pipeline**, on the training rows only, and applies both to the validation rows. So the choice is
# *median imputation fitted on the training window*, it is the same choice every other case study
# in this repository makes, and it is recorded in `setup.yaml::modeling.linear.null_policy` so a
# reader does not have to find it in a library file. The alternative - dropping rows with a null -
# was rejected because it would give each configuration a different sample.
#
# **What `dir_tb_8h` is scored with.** It is a three-class label (-1 / 0 / +1) and the menu fits
# logistic regressions to it. Section 5 sets out what the registry computes for it and what it
# does not, and `setup.yaml::labels.classification_scoring` carries the same declaration.
#
# ## Learning objectives
#
# - Fit a declared penalty grid on three labels with the folds, features and metals held fixed.
# - Read a model population on a panel too narrow for a cross-sectional statistic.
# - Separate a pooled point estimate from the fold-by-fold agreement behind it.
# - Keep every number in this notebook a diagnostic, and selection in `13_backtest`.

# %%
"""Fit the declared exness_gold_sess linear population on the walk-forward validation folds."""

import re

import numpy as np
import plotly.graph_objects as go
import polars as pl
import yaml
from IPython.display import display
from plotly.subplots import make_subplots

from case_studies.exness_gold_sess._features import session_of
from case_studies.exness_gold_sess._model_reading import (
    REGISTRY_IC_IS_NULL_REASON,
    hac_lag_for_label,
    outcome_column_for,
    pooled_panel_ic_series,
    prediction_staleness_by_book,
    summarise_pooled_ic,
)
from case_studies.research import (
    Result,
    declared_labels,
    load_model_configs,
    model_requests,
    narrows_declared_catalog,
    open_study,
    population_supersedes,
    primary_label,
    resolved_model_plan,
    run_model_population,
)
from case_studies.research.results import PredictionResult
from utils.paths import get_case_study_dir
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
CONFIG_NAMES: list[str] = []
POPULATION_NAME = ""
SUPERSEDES_POPULATION: str = ""

# %%
CASE_STUDY = "exness_gold_sess"
study = open_study(CASE_STUDY, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
setup = yaml.safe_load((get_case_study_dir(CASE_STUDY) / "config" / "setup.yaml").read_text())
SESSION_BOOKS = list(setup["decision"]["session_filter_values"])
LABEL_HORIZONS = {k: int(str(v).rstrip("Hh")) for k, v in setup["labels"]["horizons"].items()}
CLASSIFICATION_EVAL = dict(setup["labels"].get("classification_eval_label") or {})

# %% [markdown]
# ## 1. Which labels, and which models
#
# Three labels are declared in `config/setup.yaml`: `fwd_ret_8h`, the return from the decision bar's
# close to the session close and the horizon this bot is designed to trade; `fwd_ret_24h`, the same
# slot one calendar day later; and `dir_tb_8h`, which of an ATR-scaled take-profit / stop-loss pair
# is touched first inside the session.
#
# **This notebook fits every declared label**, because fitting them together is what makes the
# horizon comparable: the same penalty grid on three targets with the features, the folds and the
# two metals held fixed isolates what the horizon does. `LABELS` narrows the run when you want one.
#
# `fwd_ret_24h` carries a property worth holding in mind while reading its numbers, measured on
# 2026-09-08 and recorded in `bots/exness_gold_sess/BOT.md`: **it is a Monday-to-Thursday book**.
# Its null share by weekday is Mon 0.4 %, Tue 0.8 %, Wed 0.6 %, Thu 2.6 %, **Fri 100 %**, identical
# on both metals at 100 % of the 4,903 shared decision instants. The models below are fitted on
# whatever rows the label has; what that does to *annualisation* is a `13_backtest` question, not
# one this notebook can answer.

# %%
declared_labels(study, "linear")

# %% [markdown]
# Each name in a label's menu at `config/training/{label}.yaml` resolves to a preset in
# `case_studies/config/{model_type}/`. The regression menus were copied whole from
# `case_studies/exness_fx_d1/config/training/` and are **not trimmed**: `PHASE1_SPEC_MENTOR.md`
# section 2.d is explicit that a menu cut by taste is a selection made before the backtest, and the
# untouched menu is also what makes this bot's result comparable with `exness_fx_d1`'s.
#
# The two regression labels declare 28 configurations each:
#
# - **OLS**, no penalty, as the reference the penalties are read against.
# - **Ridge** at eleven `alpha` values stepping by powers of ten. Ridge penalizes the sum of
#   squared coefficients: it shrinks correlated coefficients towards each other and keeps every
#   feature. That is the shape of penalty this feature set asks for - `05_evaluation` collapsed 39
#   scored columns into 18 redundancy groups at a 0.70 cut, with `slot_ret_1 / kalman_innovation`
#   at +1.000 and `kalman_innovation / arima_residual` at +0.995.
# - **Lasso** and **ElasticNet** at eight `alpha_frac` values each. `alpha_frac` is a fraction of
#   each fold's own $\alpha_{\max}$ - the smallest penalty that zeros every coefficient - so one
#   declared value means the same thing on every fold, which a fixed raw penalty would not.
#
# `dir_tb_8h` declares 13: `logistic_none` plus six L2 and six L1 values of `C`. There is no
# `causal_dml` entry on that label, and the menu says why in a comment: the DML nuisance models in
# `case_studies/utils/causal.py` are both `HistGradientBoostingRegressor`, so a declared causal
# menu on a class target would fit a regressor to it and bank the result as a canonical causal
# estimate.

# %%
configs = load_model_configs(
    study,
    "linear",
    labels=LABELS or None,
    config_names=CONFIG_NAMES or None,
)
display(configs.group_by("label").agg(pl.len().alias("configurations")).sort("label"))
configs

# %% [markdown]
# `LABELS` and `CONFIG_NAMES` both narrow what is fitted, and a narrowed run declares a different
# set of members than the canonical population does. A population is immutable once written, so
# such a run must publish under its own name. Comparing the loaded rows against the complete
# declared catalog catches either knob here rather than several cells later in a message about
# hashes.

# %%
if narrows_declared_catalog(study, "linear", configs) and not POPULATION_NAME:
    raise ValueError(
        f"this run declares {configs.height} label-configuration pairs, which is not the "
        "complete declared catalog, so it cannot publish the canonical population; pass "
        "POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# A menu entry says which estimator to fit. It does not say which feature columns exist today,
# where the folds fall, or which metal-slot pairs have both a feature row and a label. **Resolving**
# finds all of that: it reads the label and feature artifacts, computes the fold boundaries from
# `config/setup.yaml` through `utils.cv_splits`, works out the exact rows each fit is expected to
# predict, and turns any data-dependent hyperparameter into the number it will use.
#
# Resolving fits nothing, so the plan can be read before any computation starts. Four things to
# check in it, and the fourth is specific to this bot:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row.** A row that
#   differs is a configuration measured on a different sample from its neighbours.
# - **`folds` is 4 everywhere**, the walk-forward count `setup.yaml::evaluation.n_splits` declares
#   and `01_feasibility_analysis` generated.
# - **`validation_start` and `validation_end` bracket the development sample.** The holdout window
#   opens on 2025-09-01; the last validation window ends 2025-08-28, and the assertion below says
#   so rather than leaving it to be read off a printed frame.
# - **`eligible_entities` is 2.** It is the number this whole notebook's reading turns on.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved = tuple(request.resolve() for request in requests)

plan = resolved_model_plan(resolved)
holdout_start = str(setup["evaluation"]["holdout_start"])
latest_validation = max(str(v) for v in plan.get_column("validation_end").to_list())
if latest_validation >= holdout_start:
    raise RuntimeError(
        f"a resolved request validates to {latest_validation}, at or past the holdout boundary "
        f"{holdout_start}; the holdout is scored once, by 17_holdout_predictions"
    )
print(f"latest validation end {latest_validation} < holdout_start {holdout_start}: untouched")
plan.select(
    "config_name",
    "label",
    "feature_count",
    "eligible_entities",
    "eligible_rows",
    "folds",
    "validation_start",
    "validation_end",
)

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` fits every resolved request. For one request it walks the folds, and on
# each one:
#
# 1. takes the rows inside that fold's training window,
# 2. fills missing feature values with **the training window's median** for that column, then
#    standardizes each column to zero mean and unit variance - both fitted on training rows only
#    and then applied to the validation rows, so nothing from the validation window reaches the
#    fit (`case_studies/utils/linear.py:402-403`, preprocessing identity
#    `median-imputer-standard-scaler/v1`). This is the null policy declared in the preamble;
#    reading it out of the library rather than restating it here is deliberate, because a copy
#    would drift,
# 3. fits the estimator with that fold's resolved parameters,
# 4. predicts the fold's validation rows.
#
# The fold predictions are concatenated into one series covering the whole validation period: each
# slot predicted by a model that saw only data before it. A linear model has no intermediate states
# worth scoring, so there is **one checkpoint per configuration** - unlike `07_gbm`, where a
# checkpoint is part of the identity.
#
# The purge between a training window and the validation window that follows it comes from the
# label buffer (`setup.yaml::labels.buffer` = 1 trading day, `variant_buffers.fwd_ret_24h` = 2).
# Measured on the decision grid itself, `04_model_based_features` found that buffer to be **3-5
# decision slots** on every fold and metal, against a label that spans 1 slot.

# %%
population_name = POPULATION_NAME or "exness_gold_sess-linear-validation-v1"
execution, population = run_model_population(
    study,
    resolved,
    population_name=population_name,
    supersedes=population_supersedes(study, name=population_name, declared=SUPERSEDES_POPULATION),
)

fitted = sum(len(item["fitted_folds"]) for item in execution.diagnostics)
reused = sum(len(item["reused_folds"]) for item in execution.diagnostics)
print(f"{len(execution.runs)} configurations: {fitted} folds fitted, {reused} reused")
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# `reused` is not zero on a second run: every identity is re-derived from the inputs and the runner
# returns the stored result rather than fitting again.
#
# ### Running configurations of your own
#
# ```python
# study = open_study("exness_gold_sess", workspace="~/ml4t/experiments/exness_gold_sess")
# configs = load_model_configs(
#     study, "linear", labels=["fwd_ret_8h"], config_names=["ols", "ridge_a1.0"]
# )
# requests = model_requests(study, configs)
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-linear-v1")
# ```
#
# To fit something new, add a preset under `case_studies/config/ridge/` and list it in the label's
# menu. Give the run its own `population_name`: a name refers to one set of members permanently.
# **Every such run is a trial** and belongs in the Trials table of `bots/exness_gold_sess/BOT.md`
# before it is read.

# %% [markdown]
# ## 4. What came out, on a panel two names wide
#
# The registry writes one catalog row per configuration and label. On this case study those rows
# arrive with **`ic_mean` null**, and the reason is the one stated in the preamble and measured in
# `_model_reading.py`. The assertion below is deliberate: it turns an empty column into a stated
# fact at the exact place a reader would otherwise conclude the fits had failed.

# %% tags=["results"]
catalog = (
    execution.catalog_rows.select(
        "config_name",
        "label",
        "complete",
        "ic_mean",
        "ic_n_days",
        "n_folds",
        "training_hash",
        "prediction_hash",
    )
    .sort(["label", "config_name"])
    .join(
        configs.select("config_name", "label", "model_class", "params"),
        on=["config_name", "label"],
        how="left",
    )
)
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("linear execution returned a partial prediction set")

registry_ic_defined = catalog.filter(pl.col("ic_mean").is_not_null()).height
print(f"catalog rows: {catalog.height}; rows carrying a registry ic_mean: {registry_ic_defined}")
print(REGISTRY_IC_IS_NULL_REASON)
catalog.select("label", "config_name", "model_class", "params", "ic_mean", "ic_n_days", "n_folds")

# %% [markdown]
# ### The pooled panel IC, with a Newey-West interval and the folds behind it
#
# For each configuration: rank-normalise the model's score and the realised outcome **within each
# metal** over the validation window, multiply row-wise, and average across the metals at each
# slot. That gives one IC value per decision slot; the mean of that series is the configuration's
# pooled IC. On one metal it reduces to that metal's own Spearman correlation, which
# `05_evaluation` checked against `scipy` to a residual of 3.6e-9.
#
# Consecutive slots overlap - the New York decision at 14:00 UTC starts inside the London hold that
# runs 09:00 to 17:00 - so the interval is **Newey-West corrected**, with the overlap **measured on
# the grid** rather than assumed. The measured overlap is a *floor*, not the lag applied:
# `compute_ic_hac_stats` uses `max(label_horizon - 1, Newey-West auto)`, and on a series of a few
# thousand slots the automatic bandwidth (about 9) is the larger of the two. Both are printed, and
# `hac_lags` reports what was applied.
#
# Beside the pooled estimate sit the four fold means, because they answer a different question. A
# configuration with a pooled IC of +0.02 built from four folds of +0.02 and one built from three
# folds of 0.00 and one of +0.08 are not the same finding, and only the first is a reason to keep
# looking. `sign_consistency` is the share of folds pointing the configuration's own way, `icir`
# the mean fold IC over its dispersion across folds.

# %%
grid_stamps = None
rows = []
for row in catalog.iter_rows(named=True):
    result = Result.open(study, row["prediction_hash"])
    if not isinstance(result, PredictionResult):
        raise TypeError(f"{row['prediction_hash']} is not a prediction result")
    frame = result.load()
    missing = {"symbol", "timestamp", "fold", "prediction", "actual"} - set(frame.columns)
    if missing:
        raise ValueError(f"prediction {result.hash} lacks canonical columns: {sorted(missing)}")
    if grid_stamps is None:
        grid_stamps = frame.get_column("timestamp")
    overlap, hac_lag = hac_lag_for_label(
        frame.filter(pl.col("symbol") == frame["symbol"][0])["timestamp"],
        LABEL_HORIZONS[row["label"]],
    )
    outcome_col = outcome_column_for(frame)
    series = pooled_panel_ic_series(frame, outcome_col=outcome_col)
    summary = summarise_pooled_ic(series, hac_lag=hac_lag)
    rows.append(
        {
            "label": row["label"],
            "config_name": row["config_name"],
            "model_class": row["model_class"],
            "params": row["params"],
            "scored_against": outcome_col,
            "overlap_slots": overlap,
            **{k: v for k, v in summary.items() if k != "fold_ics"},
            "fold_ics": str([round(v, 5) for v in summary.get("fold_ics", [])]),
            "prediction_hash": row["prediction_hash"],
        }
    )
pooled = pl.DataFrame(rows).sort(["label", "ic_mean"], descending=[False, True])
print(
    f"pooled panel IC computed for {pooled.height} configurations; HAC lag "
    f"{sorted(set(pooled['hac_lags'].to_list()))} from a measured overlap of "
    f"{sorted(set(pooled['overlap_slots'].to_list()))} decision slots"
)
with pl.Config(tbl_rows=pooled.height, tbl_cols=16, tbl_width_chars=240):
    display(
        pooled.select(
            "label",
            "config_name",
            "n_slots",
            "ic_mean",
            "hac_t",
            "hac_p",
            "hac_lags",
            "n_folds",
            "sign_consistency",
            "positive_fold_share",
            "icir",
            "fold_ics",
        )
    )

# %% [markdown]
# ### Does the model's own score go stale inside a book?
#
# `05_evaluation` ran a staleness screen per session book on the features and it stopped four
# columns - `is_venue_dst`, two payroll flags and `d1_max_dd_63d` - before they were scored. The
# same screen belongs on the model's output, and for the same reason: a score can move on the
# pooled panel and repeat inside the sleeve that would actually trade it. That is what a
# `SESSION_FILTER` book is, so the screen is run per book here, on the best-scoring configuration
# of each label, before any of these predictions reaches `13_backtest`.
#
# A near-1.0 staleness inside `london` or `ny` on a configuration would mean its score is a
# constant for that book - the model has learnt the session and nothing else - and the row is worth
# nothing there whatever its pooled IC says.

# %% tags=["results"]
stale_rows = []
for label in sorted(set(pooled.get_column("label"))):
    best = pooled.filter(pl.col("label") == label).row(0, named=True)
    frame = Result.open(study, best["prediction_hash"]).load()
    stamps = frame.select("timestamp").unique().sort("timestamp")
    books = stamps.with_columns(session_of(stamps["timestamp"]).alias("session"))
    frame = frame.join(books, on="timestamp", how="left")
    for record in prediction_staleness_by_book(frame).iter_rows(named=True):
        stale_rows.append({"label": label, "config_name": best["config_name"], **record})
staleness = pl.DataFrame(stale_rows)
ceiling = 0.50
flagged = staleness.filter(pl.col("staleness") > ceiling)
print(
    f"{flagged.height} of {staleness.height} (label, book) pairs repeat on more than "
    f"{ceiling:.0%} of consecutive slots"
)
with pl.Config(tbl_rows=staleness.height):
    display(staleness.sort(["label", "book"]))

# %% [markdown]
# ### How the penalty grid ranks, by label
#
# One panel per label, the configurations held in the same order in all three, that order being
# their pooled IC on the primary label. Held in one order, a panel that slopes down from left to
# right is a horizon that ranks the grid the way the primary label does, and one that does not is a
# horizon where the penalty that works at 8 hours does not work at 24. Each panel keeps its own
# vertical scale, because the horizons do not produce ICs of the same size.
#
# The zero line is the reference that matters: a bar below it is a model whose ranking pointed the
# wrong way out of sample. **A bar above it is not evidence of an edge** - it is an in-sample-free
# but multiplicity-uncorrected reading of one of 69 configurations, and `13_backtest` at the
# cumulative trial count is where the question is settled.


# %%
def compact(params: str) -> str:
    """Render declared parameters for a label: `alpha=1000000.0` reads as `alpha=1e+06`."""
    return re.sub(r"\d+\.?\d*(?:[eE][+-]?\d+)?", lambda m: f"{float(m.group()):g}", params)


primary = primary_label(study)
charted = sorted(set(pooled.get_column("label")))
panel_labels = [label for label in [primary] if label in charted] + [
    label for label in charted if label != primary
]
order_label = panel_labels[0]
config_order = (
    pooled.filter(pl.col("label") == order_label)
    .sort("ic_mean", descending=True)
    .get_column("config_name")
    .to_list()
)
leader = pooled.sort("ic_mean", descending=True).row(0, named=True)

fig_ic = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row_index, label in enumerate(panel_labels, start=1):
    panel = pooled.filter(pl.col("label") == label)
    fig_ic.add_trace(
        go.Bar(
            x=panel.get_column("config_name").to_list(),
            y=panel.get_column("ic_mean").to_list(),
            marker_color=[
                COLORS["amber"]
                if (name, label) == (leader["config_name"], leader["label"])
                else COLORS["blue"]
                for name in panel.get_column("config_name")
            ],
        ),
        row=row_index,
        col=1,
    )
    fig_ic.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row_index, col=1
    )
    fig_ic.update_yaxes(title_text="Pooled panel IC (validation)", row=row_index, col=1)
fig_ic.update_xaxes(
    categoryorder="array",
    categoryarray=config_order,
    tickangle=-45,
    title_text=f"Configuration (ordered by pooled panel IC on {order_label})",
    row=len(panel_labels),
    col=1,
)
fig_ic.update_layout(
    title="Validation pooled panel IC across the linear grid, by label",
    height=320 * len(panel_labels),
    width=1100,
    showlegend=False,
    margin=dict(t=90),
)
show_plotly_with_alt(
    fig_ic,
    "Stacked bar charts of the pooled panel information coefficient across the linear grid, one "
    "panel per label, the configurations in the same order in each, that order being their "
    f"ranking on {order_label}. Each panel carries a dashed zero line. The highest bar anywhere "
    f"is {leader['config_name']} ({compact(str(leader['params']))}) on {leader['label']} at IC "
    f"{leader['ic_mean']:+.4f} (HAC t {leader['hac_t']:+.2f}), highlighted in amber.",
)

# %% [markdown]
# ### Whether the ranking transfers between horizons
#
# The rank correlation between two labels' orderings of the configurations they both charted. One
# would mean the horizons agree on the whole grid, zero that the ordering at one horizon says
# nothing about the other. Each pair is built from its own two labels, so a pair that shares its
# whole grid is reported on that whole grid rather than on a three-way intersection.


# %% tags=["results"]
def rank_transfer(label_a: str, label_b: str) -> dict:
    """Rank correlation between two labels' orderings, over what both of them charted."""
    pair = (
        pooled.filter(pl.col("label").is_in([label_a, label_b]))
        .select("label", "config_name", "ic_mean")
        .pivot(on="label", index="config_name", values="ic_mean")
        .drop_nulls()
    )
    return {
        "label_a": label_a,
        "label_b": label_b,
        "configurations": pair.height,
        "rank_correlation": (
            pair.select(pl.corr(pl.col(label_a).rank(), pl.col(label_b).rank())).item()
            if pair.height > 2
            else None
        ),
    }


pl.DataFrame(
    [rank_transfer(a, b) for index, a in enumerate(panel_labels) for b in panel_labels[index + 1 :]]
)

# %% [markdown]
# ### What shrinkage does on its own
#
# Tracing the pooled IC across the Ridge penalty alone isolates the effect of shrinkage, with the
# estimator, the features and the folds held fixed and only `alpha` moving. The alpha is read from
# each configuration's declared parameters rather than parsed out of its name, so the curve plots
# what was fitted. One curve per label.

# %%
ridge = (
    pooled.filter(pl.col("model_class") == "Ridge")
    .with_columns(alpha=pl.col("params").str.extract(r"alpha=([0-9.eE+-]+)").cast(pl.Float64))
    .drop_nulls("alpha")
    .sort("label", "alpha")
)
if ridge.height:
    ridge_labels = [label for label in panel_labels if label in set(ridge.get_column("label"))]
    curve_colors = [COLORS["blue"], COLORS["copper"], COLORS["amber"]]
    fig_alpha = go.Figure()
    peaks = {}
    for index, label in enumerate(ridge_labels):
        series = ridge.filter(pl.col("label") == label)
        log_alpha = np.log10(series.get_column("alpha").to_numpy())
        ridge_ic = series.get_column("ic_mean").to_numpy()
        peak = int(np.argmax(ridge_ic))
        peaks[label] = (float(log_alpha[peak]), float(ridge_ic[peak]))
        fig_alpha.add_trace(
            go.Scatter(
                x=log_alpha,
                y=ridge_ic,
                mode="lines+markers",
                name=label,
                line=dict(color=curve_colors[index % len(curve_colors)], width=2),
                marker=dict(size=7, color=curve_colors[index % len(curve_colors)]),
            )
        )
    fig_alpha.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
    fig_alpha.update_layout(
        title="Ridge pooled panel IC against penalty strength, by label",
        height=520,
        width=900,
        margin=dict(t=70),
        legend=dict(title_text="Label"),
    )
    fig_alpha.update_xaxes(title_text="log10(alpha)  (Ridge penalty strength)", zeroline=False)
    fig_alpha.update_yaxes(title_text="Pooled panel IC (validation)")
    peak_text = ", ".join(
        f"{label} at 1e{int(round(peaks[label][0]))} (IC {peaks[label][1]:+.4f})"
        for label in ridge_labels
    )
    show_plotly_with_alt(
        fig_alpha,
        "Line chart of the pooled panel information coefficient against the base-ten logarithm of "
        f"the Ridge penalty, one line per label, against a dashed zero line. Peaks: {peak_text}.",
    )
else:
    print("No declared label declares Ridge configurations; nothing to trace.")

# %% [markdown]
# ## 5. `dir_tb_8h`: what a three-class label is scored with, and what it is not
#
# The triple-barrier label takes three values - `-1` if the lower barrier is touched first, `+1`
# for the upper, `0` if the session closes with neither touched. `02_labels` measured the classes
# as XAUUSD 1,365 up / 1,380 down / 2,048 timeout and XAGUSD 1,331 / 1,466 / 1,991: the timeout
# class is the largest on both metals, at 43 % and 41 %.
#
# **What the registry computes for it**, from `case_studies/utils/registry/metrics.py`:
#
# | Metric | Computed on | Status here |
# |---|---|---|
# | `ic_mean` | the model score against `labels.classification_eval_label[dir_tb_8h]` = `fwd_ret_8h`, the return over the same window the barriers are evaluated on, carried in the same label frame | **null**, for the two-name reason in section 4; the pooled panel IC above replaces it |
# | `accuracy`, `balanced_accuracy` | the three-class label itself | computed |
# | `log_loss`, `brier_score` | the predicted class probabilities | computed |
# | `auc_pr` | the class probabilities | computed |
# | `auc_roc` | derived by collapsing to "up versus not-up" (`metrics.py:141-153`) | computed, and to be read knowing the collapse puts the timeout class in with the falls |
# | `auc_mean_daily`, `auc_t_hac` | a **binary** label, cross-sectionally within each date | **null on both counts**: the label has three states, and the cross-section is two wide |
#
# **What it is NOT scored with, and why: uniqueness sample weights.**
#
# `LabelingConfig.triple_barrier(..., calculate_uniqueness=True)` is where Chapter 7's uniqueness
# weights come from (`07_defining_the_learning_task/03_label_methods.py:520`): it computes each
# event's uniqueness from the overlap of its holding window with its neighbours' and turns it into
# a per-row `sample_weight`. `02_labels` does **not** produce them, and after checking where they
# would go, the decision recorded here and in `setup.yaml::labels.classification_scoring` is that
# **this bot does not use them**, for three reasons in order of weight:
#
# 1. **Nothing in the case-study fitting path can consume a weight.** `grep -rn sample_weight
#    case_studies/ utils/` returns nothing: `case_studies/utils/folds.py` builds `X_train, y_train`
#    and hands them to `estimator.fit(X, y)`; `case_studies/utils/linear.py` and
#    `case_studies/utils/gbm.py` never pass a third argument; `case_studies/research/models.py` has
#    no weight parameter. Adding the column would produce an artifact no stage reads. The three
#    places `sample_weight` appears in this repository are chapter notebooks
#    (`07_defining_the_learning_task/03_label_methods.py`, `11_ml_pipeline/02` and `/03`), which
#    call scikit-learn directly.
# 2. **Writing it now would move the label identity for no effect.** `02_labels` sealed
#    `dir_tb_8h` at digest `1eb71e01111c53b3` over 9,581 rows, and every phase-2 and phase-3
#    artifact is pinned to it. A new column changes the frame, so it changes the digest, so it
#    re-hashes the feature stages and everything downstream - to feed a parameter no fitting
#    function accepts.
# 3. **The overlap the weights would correct for is already measured and already corrected.**
#    `02_labels` reports `N_eff` = 4,284 against N = 8,567 for `dir_tb_8h`, a ratio of 0.5001,
#    because exactly one later decision starts inside every holding window. That is the same
#    quantity uniqueness weighting is built from, and this case study responds to it the way the
#    mentor specification asks - by correcting **inference** at HAC lag 2 everywhere, in
#    `05_evaluation`, in section 4 above, and in `12_model_analysis` - rather than by reweighting
#    the fit. A weight changes the estimate; the HAC lag changes the interval, and it is the
#    interval that was wrong.
#
# The honest statement of the cost: point 3 is a real substitution and not a free one. Uniqueness
# weights would down-weight the 50 % of each label window that a neighbour also covers, which
# changes what the estimator fits; HAC correction leaves the fit alone and widens the error bar.
# If a later generation of this bot wants the weighted fit, the change is to the shared runner
# (`case_studies/utils/folds.py` and `linear.py` / `gbm.py`), not to this case study, and it must be
# made on an empty run log because it moves every training identity.

# %% tags=["results"]
classification_labels = sorted(CLASSIFICATION_EVAL)
if classification_labels:
    cls = execution.catalog_rows.filter(pl.col("label").is_in(classification_labels))
    available = [
        column
        for column in (
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "brier_score",
            "auc_roc",
            "auc_pr",
            "auc_mean_daily",
            "ic_mean",
        )
        if column in cls.columns
    ]
    summary = cls.select("label", "config_name", *available).sort("label", "config_name")
    for column in available:
        defined = summary.filter(pl.col(column).is_not_null()).height
        print(f"  {column}: {defined} of {summary.height} rows carry a value")
    print(
        f"scored against the continuous evaluation label "
        f"{ {k: v for k, v in CLASSIFICATION_EVAL.items()} }, declared in "
        "setup.yaml::labels.classification_eval_label (a repository-pinned key)"
    )
    with pl.Config(tbl_rows=summary.height, tbl_cols=12, tbl_width_chars=200):
        display(summary)
else:
    print("no classification label is declared")

# %% [markdown]
# ## 6. What to notice
#
# Fill this section in from the frames above after the first canonical run, quoting the numbers as
# they came out and naming the host, the date, the experiment and the population - the way
# `case_studies/exness_fx_d1/06_linear.py` section 5 does. Until then it is deliberately empty
# rather than carrying a placeholder that would read as a result.
#
# Three things a reader should carry into `07_gbm` and `13_backtest` whatever the numbers say:
#
# - **The registry's `ic_mean` column is empty on this case study and always will be.** It is a
#   cross-sectional statistic and there is no cross-section here. Any downstream code that ranks or
#   filters on it is ranking on nulls.
# - **Nothing in this notebook selects.** The complete prediction population - every configuration,
#   every label - advances to `13_backtest`, where selection happens on validation backtest Sharpe
#   at the cumulative trial count, and the Deflated Sharpe Ratio is reported against that count.
# - **The linear baseline exists to be beaten or not beaten.** `07_gbm` changes what can be
#   represented, not what the panel is; reading its result against this one is the point of fitting
#   both on identical folds, features and metals.

# %%
handoff = catalog.select(
    "label", "config_name", "training_hash", "prediction_hash", "complete"
).sort("label", "config_name")
print(f"{handoff.height} prediction sets advance to 13_backtest")
handoff
