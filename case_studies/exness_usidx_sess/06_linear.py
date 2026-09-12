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
# > began as `case_studies/exness_fx_d1/06_linear.py` with the study id, the loader and the population names
# > changed. **This bot has TWO instruments (`US500`, `USTEC`) and no cross-section**, and one
# > label per experiment workspace. Any sentence below that speaks of five currency pairs, of
# > ranking assets against each other, of `usd_corr_*` / `gold_corr_*` columns or of a dollar
# > regime is template residue describing a different bot; where such prose asserted a NUMBER it
# > has been deleted rather than reworded (see `BOT.md`, Decisions log, 2026-09-08 third pass),
# > and where it survives as a description of the template it says so explicitly. Report any that
# > does neither. This banner follows `case_studies/xau_fx_mt5/13_backtest.py`.

# %% [markdown]
# # Exness US indices, session bot: regularizing a two-instrument time series
#
# **DELETED 2026-09-08 (third pass).** The three paragraphs that opened this notebook were the FX
# template's and described a different bot: "the five pairs in this universe", `EURUSD` / `GBPUSD`
# / `AUDUSD` / `USDJPY` / `USDCAD`, "`01_feasibility_analysis` counts 2.4 independent bets among
# the five", the `usd_corr_*` state columns, and "`05_evaluation` collapsed its 52 scored columns
# into eleven redundancy groups". None of it is true here. This bot has **two** instruments, no
# `usd_corr_*` column exists in its feature matrix, `01_feasibility_analysis` computes no
# independent-bet count at all, and `05_evaluation` scored **84** columns on this bot's own folds.
# Because the paragraphs sat under this bot's title, they read as a report on it. They are
# deleted rather than reworded, and named here so a reader who quoted a number from this page
# before today can find out which numbers were fiction.
#
# What is true of **this** universe, and what this notebook is therefore about:
#
# **There is no cross-section to be dependent.** Two instruments cannot be ranked against each
# other in any way this bot acts on: `setup.yaml::mapping.class` is `time_series_threshold`, and
# each index is compared with its own score distribution. The registry's `ic_mean`, which is a
# cross-sectional statistic, is null on every row here for exactly that reason, and section 4
# below says so rather than charting it.
#
# **The feature set is collinear, and that part is real.** The design matrix holds the return,
# volatility, rolling-Sharpe and daily-momentum windows the register in `setup.yaml::features`
# declares, several of them computed over overlapping windows of the same series, so columns
# carry nearly the same information.
#
# **Regularization** - adding a penalty on coefficient size to the fitting objective - addresses
# that collinearity directly. It cannot manufacture a signal the columns do not carry, and
# `05_evaluation` reported that **not one** of the 84 candidate columns cleared the
# Benjamini-Hochberg screen over the set, so the prior for this population is that it finds
# little. This notebook runs the penalty sweep the template
# [`fx_pairs/06_linear`](../fx_pairs/06_linear.ipynb) runs, on this bot's own labels and
# features, and the results are read against that prior. The design matrix is the
# whole feature matrix `03` and `04` wrote, the eight `rank_*` columns the `05` ledger marked
# STOP and the saturated `kalman_smoothness` included: the ledger is a screen, not a gate on a
# model, and the penalty is what handles a column that carries nothing.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read the set of models a case study has declared for a label, and say which estimator and
#   which hyperparameters each declared name resolves to.
# - Bind those declarations to the data on disk and check, before anything is fitted, that every
#   configuration will be measured on the same pairs, the same dates and the same folds.
# - Fit a population of models on walk-forward folds and publish one complete set of validation
#   predictions per configuration.
# - Tell apart the two things a penalty can do to a correlated feature set - shrink correlated
#   coefficients towards each other, or select a few features and zero the rest - and read from
#   the results which one this data rewards.
# - Distinguish collinearity among the features from dependence among the assets being ranked,
#   and say which one a penalty can fix.
# - Run configurations of your own into a private copy of the run log, and have them compared on
#   the same footing as the ones shipped here.
#
# **Book reference**: Chapter 11 (The ML Pipeline). Chapter 6, Section 6.7 (Search accounting and
# run logging) introduces the run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# and [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds.
#
# **What it writes**: one training run and one complete validation prediction set per
# configuration, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population. It runs inside the bot's experiment
# (`ML4T_OUTPUT_DIR=experiments/exness_usidx_sess` and `WORKSPACE` naming the same directory, where
# `02`-`05` wrote the labels and features); the case study's own `run_log/` stays empty.
# [`13_backtest`](13_backtest.ipynb) reads that population, runs every member against the
# equal-weight baseline, and selects on validation backtest Sharpe. **Selection happens there,
# not here.** This notebook ranks configurations by information coefficient to show what
# regularization does to a correlated feature set; that ranking decides nothing.

# %%
"""Fit the declared exness_usidx_sess linear-model population on the walk-forward validation folds."""

import re

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from case_studies.research import (
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
study = open_study("exness_usidx_sess", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. Which label, and which models
#
# A label is the thing being predicted. This case study defines three in `config/setup.yaml`:
# `fwd_ret_1d`, the return over the trading day after the decision date, and `fwd_ret_5d` and
# `fwd_ret_21d` over longer horizons. The one-day label is the primary one - the horizon the
# strategy chapters trade - and the other two are variants, kept so the effect of the prediction
# horizon can be examined separately.
#
# **This notebook fits every declared label.** Each has its own training menu at
# `config/training/{label}.yaml`, listing family by family the named configurations to fit for
# that label, and the notebook fits the union of them. A label with no menu file has nothing
# declared and nothing to fit; the labels below are the ones that declare linear models.
#
# **This bot declares ONE label per workspace** (`setup.yaml::labels.primary`, `variants: []`),
# so the template's paragraph about fitting three labels in one run to make the horizons
# comparable does not apply and has been deleted: there is one horizon here and the other spec
# lives in the other workspace with its own feature matrix. `LABELS` restricts the run to a
# subset when you want one, and defaults to everything the menus declare, which is one.

# %%
declared_labels(study, "linear")


# %% [markdown]
# Each name in the menu resolves to a preset file in the shared directory
# `case_studies/config/{model_type}/`, which holds that configuration's hyperparameters. The
# frame below is the menu for every label above, with each name resolved to the estimator class
# it names and the arguments that class is constructed with. To change what runs, edit the menu
# or the presets rather than this notebook.
#
# The grid covers the two shapes a penalty can take:
#
# - **Ridge** penalizes the sum of squared coefficients. It shrinks correlated coefficients
#   towards each other and keeps every feature, at a strength set by `alpha`. The grid steps
#   `alpha` by powers of ten across ten orders of magnitude, because the useful value depends on
#   the scale and the collinearity of the design matrix and neither is known in advance.
# - **Lasso** penalizes the sum of absolute coefficients, which drives some of them exactly to
#   zero: it selects features rather than shrinking them. **ElasticNet** mixes the two.
#
# Lasso and ElasticNet are parameterized here by `alpha_frac` rather than a raw penalty. For any
# fold there is a threshold penalty $\alpha_{\max}$ - the smallest one that zeros every
# coefficient - which is computed from that fold's own data. `alpha_frac` is the fraction of it
# to apply, so one declared `alpha_frac` means the same thing on every fold, while a fixed raw
# penalty would mean something different on each.

# %%
configs = load_model_configs(
    study,
    "linear",
    labels=LABELS or None,
    config_names=CONFIG_NAMES or None,
)
configs

# %% [markdown]
# `LABELS` and `CONFIG_NAMES` both narrow what is fitted, and a narrowed run declares a different
# set of members than the canonical population does. A population is immutable once written, so
# such a run must publish under its own name: on a fresh workspace it would otherwise register an
# incomplete snapshot under the canonical one, and where the full population already exists the
# registry refuses it. Comparing the loaded rows against the complete declared catalog catches
# either knob, and says so here rather than several cells later in a message about hashes.

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
# where the walk-forward folds fall, or which pair-date pairs have both a feature row and a
# label. **Resolving** a request is the step that goes and finds all of that: it reads the label
# and feature files, computes the fold boundaries from the walk-forward parameters in
# `config/setup.yaml`, works out the exact set of rows each fit is expected to predict, and turns
# any data-dependent hyperparameter into the number it will actually use - each fold's own
# $\alpha_{\max}$ times `alpha_frac`, in the case of Lasso.
#
# Resolving reads the inputs and fits nothing, so the plan below can be inspected before any
# computation starts. The three things to check in it:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row.** They are
#   the width of the design matrix, the number of currency pairs, and the number of pair-date
#   pairs to be predicted. Every configuration here reads the same feature matrix, so a row that
#   differs is a configuration being measured on a different sample from its neighbours, and its
#   results are not comparable with theirs.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   `05_evaluation` established.
# - **`validation_start` and `validation_end` bracket the development sample.** The held-out tail
#   must not appear here: it is scored once, at the end of the case study, and any of it visible
#   in this window would mean it had been used to choose something.
#
# Each row also carries a `training_hash`: the identity of that computation, derived from
# everything that can change its result. [`RUN_LOG.md`](../RUN_LOG.md#identity) sets out what goes
# into one and what follows from it.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved = tuple(request.resolve() for request in requests)

plan = resolved_model_plan(resolved)
plan.select(
    "config_name",
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
# 2. fills missing feature values with the training window's median for that column, then
#    standardizes each column to zero mean and unit variance - both fitted on training rows only
#    and then applied to the validation rows, so nothing from the validation window reaches the
#    fit,
# 3. fits the estimator with that fold's resolved parameters,
# 4. predicts the fold's validation rows.
#
# The fold predictions are concatenated into one series covering the whole validation period,
# which is what a walk-forward prediction set is: each date predicted by a model that saw only
# data before it. The run then writes a `training_runs` row and the fitted coefficients, a
# `prediction_sets` row and the predictions themselves, and the metrics computed from them. It
# does this per configuration rather than once at the end, so an interruption costs the
# configuration in flight and nothing else.
#
# Every case study that fits linear models calls this same runner, which is what makes their
# results comparable. Unlike gradient boosting or a neural network, a linear model has no
# intermediate states worth scoring: there is one fit and therefore one checkpoint per
# configuration, and no learning curve to plot.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# is going to produce. The list is computed from the resolved specifications before the first fit
# and written down, and afterwards every member must exist and be complete. It is why a
# configuration that raises fails the whole call rather than publishing a population one member
# short.
#
# **One population covers every label**, because one run fits every label and the population is
# what that run declares. A population is immutable once written: registering a different set of
# members under a name that already exists is refused unless the caller names the snapshot it
# supersedes. A notebook that fitted one label per run under a single name would therefore
# publish the first label and be refused for the second, which is what happened before this
# notebook fitted them together. Everything that finished stays registered, and re-running fits
# only what is missing.

# %% [markdown]
# `SUPERSEDES_POPULATION` names the population hash this run replaces. A population is the set
# of prediction identities it publishes, so anything that moves a training identity produces a
# different population under the same name and the registry refuses to write it without being
# told which snapshot it retires. The feature artifact is part of that identity: `03` and `04`
# are pinned into every training spec by digest, so a stage-04 artifact that gains a fold moves
# every model fitted on it even where the values those models read are unchanged.
#
# It names the hash the published population superseded rather than being left empty, because
# the declared value is part of what the snapshot is hashed over: a re-run that left it blank
# would compute a different population and be refused against the one on record. Carrying the
# published value is what lets this notebook re-run and resolve to what it published.

# %%
population_name = POPULATION_NAME or "exness_usidx_sess-linear-validation-v1"
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
# `reused` is not zero on a second run. Every identity is re-derived from the inputs, the
# registry already holds the matching rows, and the runner returns the stored result rather than
# fitting again - so re-running this notebook unchanged costs the time it takes to read the data.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which
# holds its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("exness_usidx_sess", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "linear", labels=["fwd_ret_1d"], config_names=["ols", "ridge_a1.0", "ridge_a3.0"]
# )
# requests = model_requests(study, configs)
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-linear-v1")
# ```
#
# `CONFIG_NAMES` fits a subset of what the menu already declares; a name the menu does not
# declare raises rather than quietly fitting fewer models than you asked for. To fit something
# new, add a preset at `case_studies/config/ridge/ridge_a3.0.yaml` and list `ridge_a3.0` under
# `linear:` in the label's menu. Editing an existing preset changes that configuration's
# identity, so its result registers as a new row beside the old one instead of replacing it.
#
# Give the run its own `population_name`: a name refers to one set of members permanently, and
# reusing it for a different set raises. Everything downstream reads the registry rather than the
# notebook, so predictions produced this way are selected and backtested on the same footing as
# the ones shipped here, inside your workspace.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest, including how to
# rehearse on a reduced universe first.

# %% [markdown]
# > **Read `ic_mean` in this notebook's tables as a diagnostic that does not apply here.** The
# > registry computes it the way every case study in this book computes it: on each validation
# > date, rank the instruments by the model's prediction, rank them by the return they went on to
# > earn, and correlate the two rankings. That statistic needs a cross-section. This bot has
# > **two** instruments, so the rank correlation on a date is +1 or -1 whatever the two numbers
# > are, and its average over the validation period is a coin-flip series rather than a
# > measurement. It is left in the tables because it is what the registry stores and hiding a
# > column does not make it not exist, but nothing is selected on it.
# >
# > What this bot selects on is the per-index **time-series** information coefficient computed in
# > [`12_model_analysis`](12_model_analysis.ipynb) - within one index, over time, does a high
# > score precede a high return for that index? - and, after that, the realised net return of
# > [`13_backtest`](13_backtest.ipynb). `mapping.class` is `time_series_threshold`
# > (`config/setup.yaml`), so the time-series statistic is the one the strategy acts on.

# %% [markdown]
# ## 4. What came out
#
# One row per configuration, read back from the registry. `ic_mean` is the registry's
# **cross-sectional information coefficient**: on each validation date it ranks the instruments
# by the model's prediction, ranks them by the return they went on to earn, correlates the two
# rankings, and averages that correlation over the validation period. **On this bot it is null on
# every row**, because a rank correlation over two points is +1 or -1 whatever the numbers are
# and `case_studies/utils/analytics.py` declines to compute it; the cell after next explains what
# is done instead. It is described here because it is the column the registry stores, not because
# it measures anything on this universe.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it is not a
# footnote. A model whose coefficients collapse to one or two features predicts nearly the same
# value for every pair on some dates; a constant has no rank correlation with anything, so those
# dates contribute nothing. Its `ic_mean` would then be an average over fewer dates than its
# neighbours', chosen by where it happened to stay non-degenerate, and comparing it with theirs
# would compare two different samples. `full_coverage` marks the configurations measured on all
# of them - the column is there so you can see whether that happened, not because it always does.
#
# **Coverage is judged within a label, not across them.** The three horizons do not have the same
# number of scorable validation dates to begin with: a 21-day label runs out of forward window
# earlier than a one-day label does, so it has fewer dates before any model is fitted. Comparing
# every configuration against one global maximum would mark the entire 21-day grid incomplete for
# a reason that has nothing to do with the models. The reference is each label's own maximum, and
# `full_coverage` then means what it says: measured on every date that label offers.

# %% tags=["results"]
catalog = (
    execution.catalog_rows.select(
        "config_name",
        "label",
        "complete",
        "ic_mean",
        "ic_std",
        "ic_n_days",
        "n_folds",
        "training_hash",
        "prediction_hash",
    )
    .sort(["label", "ic_mean"], descending=[False, True])
    .join(
        configs.select("config_name", "label", "model_class", "params"),
        on=["config_name", "label"],
        how="left",
    )
)

if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("linear execution returned a partial prediction set")

catalog = catalog.with_columns(
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max().over("label")
)
catalog.select(
    "label",
    "config_name",
    "model_class",
    "params",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "full_coverage",
)

# %% [markdown]
# The configurations left out of the charts below, with the number of dates each was measured on
# against its label's maximum. An empty frame means every configuration scored every date its
# label offered.

# %% tags=["results"]
catalog.with_columns(label_dates=pl.col("ic_n_days").max().over("label")).filter(
    ~pl.col("full_coverage")
).select("label", "config_name", "model_class", "ic_mean", "ic_n_days", "label_dates")

# %% [markdown]
# ### Why the ranking charts are not drawn
#
# Every row of the table above carries a null `ic_mean` and a null `ic_n_days`, and that is the
# honest answer rather than a broken run. The registry scores a configuration with a
# **cross-sectional** information coefficient: on each validation date it ranks the instruments by
# the model's prediction, ranks them again by the return they went on to earn, and correlates the
# two orderings. `case_studies/utils/analytics.py` requires a minimum number of instruments on a
# date before it will compute that correlation, because a rank correlation over a handful of
# points is noise - and over **two** points, which is this universe, it is +1 or -1 by
# construction and carries nothing at all. So the column comes back empty, as it should.
#
# The notebook this is forked from spends its remaining sections charting that column: how the
# penalty grid ranks, whether the ranking transfers between horizons, what shrinkage does on its
# own. On a column of nulls those charts are either an exception or an empty axis that reads like
# a finding, and both are worse than this paragraph. They are removed rather than guarded, so that
# nobody re-enables them by accident on a universe that still cannot support them.
#
# **What was produced is below, and it is the substance:** the declared population was fitted on
# every walk-forward fold and every checkpoint, and each prediction set is in the registry under
# its own hash. **Where the comparison happens** is
# [`12_model_analysis`](12_model_analysis.ipynb), which computes a per-index **time-series**
# information coefficient - within one index, over time, does a high score precede a high return
# for that index? - and then [`13_backtest`](13_backtest.ipynb), which scores the survivors on
# realised net return after the spread this account actually quoted. `mapping.class` is
# `time_series_threshold`, so the time-series statistic is the one the strategy acts on and the
# cross-sectional one was never the criterion here.
#
# Nothing in this notebook selects anything, so nothing in it enters the Deflated Sharpe Ratio
# trial count on its own. The count enters at the point a prediction set is turned into a
# position, which is `13_backtest`, and it counts every prediction set registered here.

# %% tags=["results"]
print(f"Label:                {catalog['label'].unique().to_list()}")
print(f"Configurations:       {catalog.height}")
print(f"Folds per config:     {int(catalog['n_folds'].max())}")
print(f"Prediction sets:      {execution.catalog_rows.height}")
print(f"Cross-sectional IC:   unavailable ({catalog['ic_mean'].null_count()} of {catalog.height} null)")
print(f"Registered under:     {population_name}")
_summary_cols = [
    c
    for c in ("config_name", "model_class", "params", "checkpoint_value", "n_folds",
              "training_hash", "prediction_hash")
    if c in catalog.columns
]
display(catalog.select(_summary_cols).sort("config_name"))

# %% [markdown]
# ## 5. What to notice
#
# **This section deliberately carries no measured result.** The notebook it is forked from ends
# with several pages of them - which penalty ranked best, how the ordering transferred between
# horizons, what shrinkage did - and every one of those numbers is a cross-sectional information
# coefficient over five currency pairs. On two indices that statistic is undefined, the registry
# returns null for it, and a fork that kept the paragraphs would be reporting `exness_fx_d1`'s
# results under this bot's name. That is worse than reporting nothing, so nothing is reported.
#
# What this notebook is for, and what it did:
#
# - **It fitted a declared population and registered it.** The training menu in
#   `config/training/<label>.yaml` was copied whole from `exness_fx_d1` and not pruned: 28 linear
#   configurations, one label per workspace, every walk-forward fold. Pruning a menu after seeing
#   which members do well is how a trial count stops being honest, so the menu is declared before
#   the first fit and every member is fitted whatever it turns out to be worth.
# - **It selected nothing.** No configuration is dropped here, and no number in this notebook is
#   a selection criterion. That matters for the Deflated Sharpe Ratio: what enters K is the count
#   of prediction sets that reach `13_backtest` and are turned into positions, and that count is
#   the whole population, not a shortlist chosen here.
# - **The comparison happens twice, downstream, on statistics that exist on two instruments.**
#   [`12_model_analysis`](12_model_analysis.ipynb) computes a per-index time-series information
#   coefficient over the validation folds. [`13_backtest`](13_backtest.ipynb) scores the
#   population on realised net return after the spread this account actually quoted - which for
#   the development sample is a p90 of 4.05 basis points per crossing, four times what the recent
#   regime would charge, because the sample holds two cost regimes and the engine charges one
#   number (`config/setup.yaml::costs`).
#
# **What the prior says to expect.** `05_evaluation` scored 84 candidate columns on this bot's own
# validation folds and **not one** cleared the Benjamini-Hochberg screen over the set; every
# PROCEED in the triage ledger came through the exploration route. A linear model reads exactly
# those columns, so the honest prior for this population is that it finds little, and a
# configuration that looks strong in `12` should be read against that prior rather than against
# the hope that a model recovers what the features did not carry.
#
# **Known limitations.** Three folds of roughly 125 validation sessions on two instruments is a
# small sample for choosing among 28 configurations; the folds are 18 months of training each,
# which is short by the standards of the case studies this is forked from; and every number
# downstream is measured on validation windows `05_evaluation` has already read once, which is
# why the holdout is scored once and only after phases 5 and 6 are green.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) asks whether gradient boosting represents something a linear
# model cannot - a volatility-regime condition on the session-leg and mean-reversion columns -
# and is read against this population in `12_model_analysis`.
