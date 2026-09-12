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
# # Exness BTC 8h: the linear baseline on a one-name panel
#
# **Chapter reference**: Volume 1, Chapter 11 (`11_ml_pipeline`), notebooks `02_regularization_paths`
# and `03_logistic_classification`.
# **Docker image**: `ml4t` (default).
# **Prerequisites**: `03_financial_features` and `04_model_based_features` must have written
# `features/financial.parquet` and `features/model_based.parquet`, and `02_labels` the two label
# parquets. Run with `ML4T_OUTPUT_DIR` pointing at a writable experiment; the repository
# `case_studies/exness_btc_8h/run_log/` is never written to.
# **What it writes**: one `training_runs` row and one `prediction_sets` row per label and
# configuration, plus the fitted coefficients and the walk-forward predictions, into the
# experiment's `run_log/`. It selects nothing.
#
# ## What is different here, and why
#
# Forked from `case_studies/exness_gold_sess/06_linear.py` (`bots/exness_btc_8h/BOT.md` Decisions
# log 2026-09-10). `06_linear` in the template ranks a cross-section; the two-metal sibling reads
# a pooled panel IC over two names. **This bot has ONE instrument**, which is not a smaller panel
# than two - it is the case the pooled-panel construction was built to degrade to correctly. Three
# consequences run through the whole notebook and are stated once, here:
#
# 1. **There are no `rank_*` columns to fit on, and `features.ranked` is empty by design**
#    (`setup.yaml::features.ranked`). A percentile over one name is the constant 0.5 at every
#    instant - not stale, degenerate.
# 2. **The registry's `ic_mean` is NULL on every row this notebook produces.**
#    `case_studies/exness_btc_8h/_model_reading.py` carries the measurement: the registry's
#    cross-sectional Spearman needs a hard-coded `min_obs=5` and every decision instant here has
#    ONE row. Section 4 reads the **pooled panel IC** instead - on one name it reduces exactly to
#    that name's own Spearman correlation through time, which `05_evaluation` checked.
# 3. **No classification label is declared this generation** (`setup.yaml::labels.
#    classification_eval_label: {}`), so section 5 below is a stub that prints why rather than a
#    populated table - a `dir_tb_8h`-shaped label is a later generation and its own trial family
#    (`features.families` "on-chain and flow" note carries the same convention).
#
# ## Two declarations this notebook makes before it fits anything
#
# **Nulls.** The feature matrix carries a declared null on the `usidx_*` family (9.3-11.0 % of
# development rows, all before 2019-09-18 - `setup.yaml::features.null_policy`). scikit-learn
# raises on a null; LightGBM reads one as a branch. The answer is taken from the shared runner
# rather than invented here: `case_studies/utils/linear.py:402-403` fits a
# `SimpleImputer(strategy="median")` followed by a `StandardScaler` **inside the fold pipeline**,
# on the training rows only, and applies both to the validation rows - the same choice every other
# case study in this repository makes, recorded in `setup.yaml::modeling.linear.null_policy` so a
# reader does not have to find it in a library file.
#
# **Prediction-set count is not configuration count.** `config/training/{label}.yaml` declares 43
# CONFIGURATIONS (28 linear + 15 GBM); this notebook's linear grid publishes 28 prediction sets a
# label because linear presets carry no `checkpoint_interval` (one "final" checkpoint each).
# `07_gbm`'s grid publishes 150 a label (15 configs x 10 checkpoints) - the arithmetic corrected
# 2026-09-10, before either notebook was forked (`bots/exness_btc_8h/BOT.md` Decisions log,
# Trials table).
#
# ## Learning objectives
#
# - Fit a declared penalty grid on two labels with the folds and features held fixed.
# - Read a model population on a panel too narrow for a cross-sectional statistic.
# - Separate a pooled point estimate from the fold-by-fold agreement behind it.
# - Keep every number in this notebook a diagnostic, and selection in `13_backtest`.

# %%
"""Fit the declared exness_btc_8h linear population on the walk-forward validation folds."""

import re

import numpy as np
import plotly.graph_objects as go
import polars as pl
import yaml
from IPython.display import display
from plotly.subplots import make_subplots

from case_studies.exness_btc_8h._features import session_book_of
from case_studies.exness_btc_8h._model_reading import (
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
CASE_STUDY = "exness_btc_8h"
study = open_study(CASE_STUDY, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
setup = yaml.safe_load((get_case_study_dir(CASE_STUDY) / "config" / "setup.yaml").read_text())
SESSION_BOOKS = list(setup["decision"]["session_filter_values"])
LABEL_HORIZONS = {k: int(str(v).rstrip("Hh")) for k, v in setup["labels"]["horizons"].items()}
CLASSIFICATION_EVAL = dict(setup["labels"].get("classification_eval_label") or {})

# %% [markdown]
# ## 1. Which labels, and which models
#
# Two labels are declared in `config/setup.yaml`: `fwd_ret_8h`, the return from the decision bar's
# close to the next decision close and the horizon this bot is designed to trade, and
# `fwd_ret_24h`, the same slot one calendar day later. **This notebook fits both**, because fitting
# them together is what makes the horizon comparable: the same penalty grid on two targets with
# the features and the folds held fixed isolates what the horizon does. `LABELS` narrows the run
# when you want one.

# %%
declared_labels(study, "linear")

# %% [markdown]
# Each name in a label's menu at `config/training/{label}.yaml` resolves to a preset in
# `case_studies/config/{model_type}/`. The two labels declare 28 configurations each:
#
# - **OLS**, no penalty, as the reference the penalties are read against.
# - **Ridge** at eleven `alpha` values stepping by powers of ten.
# - **Lasso** and **ElasticNet** at eight `alpha_frac` values each - a fraction of each fold's own
#   $\alpha_{\max}$, so one declared value means the same thing on every fold.

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
# such a run must publish under its own name.

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
# Resolving reads the label and feature artifacts, computes the fold boundaries from
# `config/setup.yaml` through `utils.cv_splits`, works out the exact rows each fit is expected to
# predict, and turns any data-dependent hyperparameter into the number it will use. Resolving fits
# nothing, so the plan can be read before any computation starts. Four things to check:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row.**
# - **`folds` is 4 everywhere**, `setup.yaml::evaluation.n_splits`.
# - **`validation_start` and `validation_end` bracket the development sample**; the holdout opens
#   2025-09-01 and the assertion below says the last validation window ends before it.
# - **`eligible_entities` is 1.** It is the number this whole notebook's reading turns on.

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
if set(plan.get_column("eligible_entities").unique().to_list()) != {1}:
    raise RuntimeError(
        f"eligible_entities is not 1 on every row: {sorted(set(plan['eligible_entities']))}; "
        "this bot's universe is one symbol (BTCUSD)"
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
# each one: (1) takes the rows inside that fold's training window, (2) fills missing feature
# values with the training window's median then standardizes, both fitted on training rows only
# (`case_studies/utils/linear.py:402-403`), (3) fits the estimator, (4) predicts the fold's
# validation rows. The fold predictions are concatenated into one series covering the whole
# validation period. The purge between a training window and the validation window that follows
# comes from the label buffer (`setup.yaml::labels.buffer` = 24H, `variant_buffers.fwd_ret_24h` =
# 48H) - measured at exactly 3 decision slots on the primary label.

# %%
population_name = POPULATION_NAME or "exness_btc_8h-linear-validation-v1"
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
# `reused` is not zero on a second run: every identity is re-derived from the inputs and the
# runner returns the stored result rather than fitting again.
#
# **Every such run is a trial** and belongs in the Trials table of `bots/exness_btc_8h/BOT.md`
# before it is read.

# %% [markdown]
# ## 4. What came out, on a panel one name wide
#
# The registry writes one catalog row per configuration and label. On this case study those rows
# arrive with **`ic_mean` null**, for the reason stated in the preamble and measured in
# `_model_reading.py`.

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
# For each configuration: rank-normalise the model's score and the realised outcome (trivially,
# over the one symbol) over the validation window, multiply row-wise. That gives one IC value per
# decision slot, identical to the Spearman correlation of score against outcome; the mean of that
# series is the configuration's pooled IC.
#
# Consecutive slots on `fwd_ret_8h` do NOT overlap (`labels.rebalance_step.fwd_ret_8h: 1`, measured
# overlap 0), so the Newey-West floor is 1; `fwd_ret_24h` spans three decision slots, measured
# overlap 2, floor 3. Both are printed rather than assumed, and `compute_ic_hac_stats` applies
# `max(label_horizon - 1, Newey-West auto)`, which on a series of a few thousand slots is usually
# the larger automatic bandwidth. `hac_lags` reports what was applied.

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
# `05_evaluation` ran a staleness screen per session book on the features. The same screen belongs
# on the model's output: a score can move on the pooled panel and repeat inside a filtered book, so
# it is run here per book on the best-scoring configuration of each label, before any of these
# predictions reaches `13_backtest`. `session_book_of` resolves `no_swap_night` / `us_hours`
# (`__pooled__` -> `all` is added by `prediction_staleness_by_book` itself).

# %% tags=["results"]
stale_rows = []
for label in sorted(set(pooled.get_column("label"))):
    best = pooled.filter(pl.col("label") == label).row(0, named=True)
    frame = Result.open(study, best["prediction_hash"]).load()
    stamps = frame.select("timestamp").unique().sort("timestamp")
    books = stamps.with_columns(session_book_of(stamps["timestamp"]).alias("session_book"))
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
# ### What shrinkage does on its own

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
# ## 5. No classification label this generation
#
# `setup.yaml::labels.classification_eval_label` is `{}`. A `dir_tb_8h`-shaped triple-barrier
# label is a candidate for a later generation (`setup.yaml::features.families` "on-chain and
# flow" carries the same "+1 trial family when first tried" convention) and is not declared here
# so that it cannot be added without the count being noticed.

# %% tags=["results"]
classification_labels = sorted(CLASSIFICATION_EVAL)
if classification_labels:
    raise RuntimeError(
        "classification_eval_label is non-empty but section 5 of this notebook was not written "
        "for it; a classification label changes what this section must compute"
    )
print("no classification label is declared")

# %% [markdown]
# ## 6. What to notice
#
# Fill this section in from the frames above after the first canonical run, quoting the numbers as
# they came out and naming the host, the date, the experiment and the population. Until then it is
# deliberately empty rather than carrying a placeholder that would read as a result.
#
# Three things a reader should carry into `07_gbm` and `13_backtest` whatever the numbers say:
#
# - **The registry's `ic_mean` column is empty on this case study and always will be.** It is a
#   cross-sectional statistic and there is no cross-section here.
# - **Nothing in this notebook selects.** The complete prediction population - every configuration,
#   every label - advances to `13_backtest`, where selection happens on validation backtest Sharpe
#   at the cumulative trial count, and the Deflated Sharpe Ratio is reported against that count.
# - **The linear baseline exists to be beaten or not beaten.** `07_gbm` changes what can be
#   represented, not what the panel is.

# %%
handoff = catalog.select(
    "label", "config_name", "training_hash", "prediction_hash", "complete"
).sort("label", "config_name")
print(f"{handoff.height} prediction sets advance to 07_gbm / 12_model_analysis")
handoff
