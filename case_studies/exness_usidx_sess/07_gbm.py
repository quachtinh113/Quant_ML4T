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
# > began as `case_studies/exness_fx_d1/07_gbm.py` with the study id, the loader and the population names
# > changed. **This bot has TWO instruments (`US500`, `USTEC`) and no cross-section**, and one
# > label per experiment workspace. Any sentence below that speaks of five currency pairs, of
# > ranking assets against each other, of `usd_corr_*` / `gold_corr_*` columns or of a dollar
# > regime is template residue describing a different bot; where such prose asserted a NUMBER it
# > has been deleted rather than reworded (see `BOT.md`, Decisions log, 2026-09-08 third pass),
# > and where it survives as a description of the template it says so explicitly. Report any that
# > does neither. This banner follows `case_studies/xau_fx_mt5/13_backtest.py`.

# %% [markdown]
# # Exness US indices (session): what a tree can express that a weighted sum cannot
#
# **DELETED 2026-09-08 (third pass).** The two paragraphs that opened this notebook were the FX
# template's: "the five pairs all have the dollar on one side", "`01` counts 2.4 independent bets
# among the five", "the dollar-regime HMM state and the `usd_corr_*` and `gold_corr_*` columns
# from `03` and `04`", "whether an exchange-rate cross section rewards that". None of those
# columns exists in this bot's feature matrix, `01` computes no independent-bet count, and this
# bot trades two US index CFDs rather than an exchange rate. Sitting under this bot's title they
# read as a report on it, so they are deleted rather than reworded and named here.
#
# What [`06_linear`](06_linear.ipynb) actually established on this universe: there is **no
# cross-section** to be dependent (two instruments cannot be ranked, `mapping.class` is
# `time_series_threshold`, and the registry's cross-sectional `ic_mean` is null on every row),
# the feature set **is** collinear over overlapping windows of the same series, and
# `05_evaluation` found **not one** of 84 candidate columns clearing the Benjamini-Hochberg
# screen. That is the prior this notebook is read against.
#
# Gradient boosting changes what can be represented. A linear model assigns one weight per
# feature and applies it everywhere; a tree splits on one feature inside a region defined by
# another, so it can express "when realized volatility is high, weight the mean-reversion
# z-score; otherwise weight momentum" without anyone constructing that column. The conditions
# available to it here are the ones `03` and `04` actually wrote for these two indices - the
# session and daily realized-volatility families, the regime and trend families, and the
# calendar columns. Whether a two-instrument session panel rewards that is the question, and it
# is a fair one: equity-index returns are widely modelled as volatility-regime dependent, and a
# regime is exactly a condition on one variable that changes how another behaves.
#
# Three dials control how far the model goes, and this notebook varies all three:
#
# - **Capacity**, set by `num_leaves`: how many regions one tree may carve the feature space into,
#   and therefore how fine a set of conditions it can express.
# - **The loss function**, which decides what "got wrong" means. Squared error weights an
#   observation by the square of its error; absolute error and Huber do not. Daily exchange-rate
#   returns are less heavy-tailed than commodity or crypto returns, which makes this case study
#   the useful control on whether the objective matters for the reason the tails suggest.
# - **When to stop**, set by the number of trees. A boosted model has a meaningful state at every
#   iteration, so each configuration is scored at ten points along its own training run.
#
# **A checkpoint is part of a configuration, not a detail of how it was fitted.** Fifteen declared
# configurations at ten checkpoints is 150 candidate models, and reporting the leading row of that
# table as though it were one experiment would be reporting the maximum of 150 numbers.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read a declared gradient boosting grid and say what each configuration varies.
# - Explain why a boosted model produces one result per checkpoint while a linear model produces
#   one result in total, and what that implies for counting candidates.
# - Read a learning curve against tree count and say whether an apparent peak is a turning point
#   or the highest of ten noisy readings.
# - Say what a tree ensemble can represent that a penalized linear model cannot, and judge from
#   the results whether this two-instrument session panel rewards it.
# - Say why the registry's cross-sectional information coefficient is null on a two-instrument
#   universe, and where the statistic this bot does act on is computed instead.
#
# **Book reference**: Chapter 12, Section 12.2 (GBM libraries) and Section 12.3 (how to tune a
# boosted model). Chapter 6, Section 6.7 (Search accounting and run logging) introduces the run
# log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds, and
# [`06_linear`](06_linear.ipynb) fitted the linear population this one is compared against.
#
# **What it writes**: one training run per configuration and one complete validation prediction
# set per configuration and checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population. It runs inside the bot's experiment
# (`ML4T_OUTPUT_DIR=experiments/exness_usidx_sess` and `WORKSPACE` naming the same directory) on the
# CPU (`modeling.gbm.device: cpu` in the experiment's `setup.yaml`).
# [`13_backtest`](13_backtest.ipynb) reads that population and selects on validation backtest
# Sharpe. **Selection happens there, not here.**

# %%
"""Fit the declared exness_usidx_sess gradient boosting population on the walk-forward folds."""

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
# Generation 1 of this population (hash 434954baa5f6) was the snapshot written by the first run
# on the WSL host at LightGBM's default 8 threads, stopped after two folds when the thread
# contention was measured (bots/exness_usidx_sess/BOT.md, Decisions log); `modeling.gbm.num_threads: 2`
# in the experiment's setup.yaml moves every training identity, so the published generation
# names the one it retired, as the template's declaration does.
SUPERSEDES_POPULATION: str = "434954baa5f6"

# %%
study = open_study("exness_usidx_sess", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. Which labels, and which models
#
# The labels are the same ones the linear notebook fitted, and for the same reason: the two
# families are compared on identical targets, folds and features, so what separates their results
# is the family. Fitting all three horizons here rather than one also carries the linear
# notebook's finding forward to be checked - that both the best result a grid reaches and the
# number of configurations above zero rise with the prediction horizon.

# %%
declared_labels(study, "gbm")


# %% [markdown]
# Each label's menu at `config/training/{label}.yaml` lists 15 named configurations under `gbm:`,
# and each resolves to a preset in `case_studies/config/lgb/`. The grid is a product of two axes:
#
# - **Five capacity profiles.** `default` uses the library's own leaf count; the rest fix it at 7,
#   15, 31 and 63. Leaf count is the direct control on how finely one tree may partition the
#   feature space. It is the dial that decides whether a model can express a condition like "when
#   the dollar is trending, rank on momentum; otherwise rank on carry" - a statement about one
#   feature that only holds inside a region defined by another.
# - **Three objectives.** `mse` minimizes squared error, `mae` absolute error, and `huber` behaves
#   like squared error for small residuals and like absolute error beyond a threshold derived from
#   each fold's own label spread.
#
# Every configuration runs the same number of boosting iterations with the same learning rate, so
# the grid isolates capacity and loss rather than confounding them with training length.

# %%
configs = load_model_configs(
    study,
    "gbm",
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
if narrows_declared_catalog(study, "gbm", configs) and not POPULATION_NAME:
    raise ValueError(
        f"this run declares {configs.height} label-configuration pairs, which is not the "
        "complete declared catalog, so it cannot publish the canonical population; pass "
        "POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# Resolving reads the label and feature files, computes the fold boundaries, works out the exact
# rows each fit must predict, and turns any data-dependent parameter into the number it will use.
# Huber's threshold is one of those: it is a fraction of the training labels' standard deviation,
# so it is a different number on every fold and is resolved from that fold's own data.
#
# Nothing is fitted here, so the plan can be inspected first. Four things to check:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row.** A row that
#   differs is a configuration measured on a different sample from its neighbours.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits.
# - **`validation_start` and `validation_end` bracket the development sample**, with none of the
#   held-out tail visible.
# - **`checkpoints` is where this differs from the linear plan.** It is the number of training
#   states each configuration will publish predictions for. Multiply it by the number of rows to
#   get the number of candidate models this notebook is about to create.

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
    "checkpoints",
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
# 2. casts the design matrix to the precision LightGBM works in and leaves missing values in
#    place - a tree routes a missing value down its own branch, so imputing a median here would
#    hand the model an observation nobody made,
# 3. fits the declared number of boosting iterations,
# 4. predicts the fold's validation rows at each checkpoint, using only the trees built up to that
#    iteration.
#
# Step 4 is what makes one fit produce many results. The fold predictions are concatenated into
# one series per checkpoint covering the whole validation period, and each becomes its own
# registered prediction set with its own identity.
#
# Slicing the window and cleaning its rows depends on the data and not on the model, so it is
# work several configurations could share. Whether they do depends on how the requests reach the
# runner. Handed over unresolved, they go to a batch path that walks folds on the outside and
# configurations on the inside, so one prepared fold serves every configuration and only one is
# held at a time. Resolved first - as they are here, so that the plan above can be shown against
# the real data - each configuration prepares its own. On a cross-section of this size that costs
# seconds and buys a plan that can be read before anything is fitted. On a panel large enough
# that one prepared fold set does not comfortably fit in memory the trade runs the other way,
# which is why the call also accepts requests that have not been resolved.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, written down before the first fit. Afterwards every member must exist and be
# complete, which is what makes the downstream comparison well defined.
#
# `SUPERSEDES_POPULATION` names the population hash this run replaces. A population is the set of
# prediction identities, so anything that moves a training identity - a changed estimator
# parameter as much as a changed configuration menu - produces a different population under the
# same name, and the registry refuses to write it without being told which snapshot it supersedes.
# That lineage is the only record of which generation is which.
#
# It defaults to the hash the published population actually superseded, not to empty. The hash is
# part of what the snapshot is hashed over, so a run that left it empty would compute a different
# population and be refused against the one on record. Carrying the value the published run used
# is what lets this notebook re-run and resolve to the population it published rather than to a
# new one.
#
# Whether the declared hash may be offered is `population_supersedes`, which every notebook that
# publishes a population now shares. It is offered when the declaration names the generation in
# force, so a refit publishes the next one, and when it names the generation that one superseded,
# so a re-run resolves to the population it published. It is withheld everywhere else: on a
# reader's clean clone, where `run_log/` is gitignored and `OfficialPopulation.create` refuses a
# first version claiming to supersede something; under a caller's own `POPULATION_NAME`, which has
# no prior generation; and in a reduced-scale run, whose isolated registry holds nothing under this
# name and whose population is thrown away with the workspace anyway.
#
# **One population covers every label**, because one run fits every label. A population is
# immutable once written, so a notebook fitting one label per run under a single name publishes
# the first label and is refused for the second; fitting them together is what lets one name
# describe the whole declared set.

# %%
population_name = POPULATION_NAME or "exness_usidx_sess-gbm-validation-v1"
execution, population = run_model_population(
    study,
    resolved,
    population_name=population_name,
    supersedes=population_supersedes(study, name=population_name, declared=SUPERSEDES_POPULATION),
)

print(f"{len(execution.runs)} configurations fitted")
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# Re-running this notebook unchanged costs the time it takes to read the data. Every identity is
# re-derived from the inputs, the registry already holds the matching rows, and the runner returns
# the stored result rather than fitting again.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which holds
# its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("exness_usidx_sess", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "gbm", labels=["fwd_ret_1d"], config_names=["leaves_15_huber", "leaves_31_huber"]
# )
# requests = model_requests(study, configs)
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-gbm-v1")
# ```
#
# `CONFIG_NAMES` fits a subset of what the menu declares. To fit something new, add a preset at
# `case_studies/config/lgb/leaves_127_huber.yaml` and list `leaves_127_huber` under `gbm:` in the
# label's menu. Editing an existing preset changes that configuration's identity, so its result
# registers as a new row beside the old one rather than replacing it.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

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
# One row per configuration and checkpoint. `ic_mean` is the registry's **cross-sectional
# information coefficient**: on each validation date it ranks the instruments by the model's
# prediction, ranks them by the return they went on to earn, correlates the two rankings, and
# averages that correlation over the validation period. **It is null on every row of this bot**,
# for the reason the cell above gives; it is described because it is what the registry stores.
#
# The table is sorted by label and then by IC, and the top of each label's block is the trap this
# notebook exists to describe. The leading row for a label is the maximum of 150 numbers - fifteen
# configurations at ten checkpoints each. Reading it as the result of one experiment would
# attribute to the model whatever the stopping point contributed, and the section below measures
# how large that contribution is before anything is concluded from the ranking.
#
# Coverage is judged against each label's own maximum number of scorable validation dates. The
# horizons do not offer the same number to begin with, since a longer forward window runs out
# earlier, so one global maximum would mark whole labels incomplete for a reason unrelated to any
# model.

# %% tags=["results"]
catalog = execution.catalog_rows.select(
    "config_name",
    "label",
    "complete",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "n_folds",
    "training_hash",
    "prediction_hash",
).sort(["label", "ic_mean"], descending=[False, True])

if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("gbm execution returned a partial prediction set")

catalog = catalog.with_columns(
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max().over("label")
)

primary = primary_label(study)
present = sorted(set(catalog.get_column("label")))
# The primary label leads when it was fitted. A subset run that leaves it out orders the panels
# by whichever label it did fit rather than by one that is not there.
panel_labels = [label for label in [primary] if label in present] + [
    label for label in present if label != primary
]
order_label = panel_labels[0]
print(f"{catalog.height} candidate models: {catalog.n_unique('config_name')} configurations")
print(f"at {catalog.n_unique('checkpoint_value')} checkpoints each, on {len(panel_labels)} labels")
catalog.select(
    "label",
    "config_name",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "full_coverage",
).head(15)

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
# **This section deliberately carries no measured result**, for the reason
# [`06_linear`](06_linear.ipynb) gives at the same point: the numbers the forked notebook ends
# with are cross-sectional information coefficients over five currency pairs, that statistic does
# not exist on two indices, and reprinting them under this bot's name would be a fabricated
# result rather than a comparison.
#
# What this notebook is for, and what it did:
#
# - **It fitted the declared gradient-boosting menu whole** - 15 LightGBM configurations spanning
#   three loss functions and five leaf counts, one label per workspace, every walk-forward fold -
#   and registered each prediction set under its own hash. Nothing was pruned after the fact.
# - **`modeling.gbm.num_threads` is 2 and `LD_LIBRARY_PATH` points at a local OpenMP runtime**,
#   both recorded in `bots/exness_usidx_sess/BOT.md`. They are environment facts about this
#   machine, not modelling choices, and they are declared in configuration so that a run
#   elsewhere fails loudly rather than silently fitting something else.
# - **It selected nothing**, and its whole population enters the Deflated Sharpe Ratio trial count
#   at `13_backtest` along with the linear one.
#
# **What a tree can and cannot add here.** A linear model reads a weighted sum of the columns; a
# tree can condition one column on another - momentum inside a turbulent regime, the overnight
# leg inside a wide opening range. The register carries exactly the state columns that condition
# would need (`rv_cash_*`, `vol_gk_*`, the `d1_vol_*` regime block, `hmm_regime_prob_high_vol`),
# so the question is a fair one on this feature set. What it cannot do is find structure the
# columns do not carry, and `05_evaluation` found no column clearing its FDR screen in either
# spec. A tree that scores well against that prior is more likely to have fitted the folds than
# to have found an interaction, which is what `12_model_analysis` and the fold-by-fold reading
# there are for.
#
# **Known limitations.** Boosted trees on 1,500 rows and 80 columns overfit readily and the leaf
# counts in the menu run to 63, which on this sample is a large tree; the checkpoint sweep means
# several correlated views of one fit rather than several independent trials, and the Deflated
# Sharpe Ratio's trial count treats them as separate, which is conservative in the right
# direction; and the same three validation folds carry every comparison.
#
# **Next**: [`12_model_analysis`](12_model_analysis.ipynb) reads the linear and the boosted
# populations together, on a per-index time-series information coefficient rather than on the
# cross-sectional one this universe cannot support.
