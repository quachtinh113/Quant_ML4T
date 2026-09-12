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
# # Exness Gold Sessions: what a tree can express that a weighted sum cannot
#
# **Chapter reference**: Volume 1, Chapter 12 (`12_gradient_boosting`).
# **Docker image**: `ml4t` (default).
# **Prerequisites**: `03_financial_features`, `04_model_based_features`, `02_labels` and
# [`06_linear`](06_linear.ipynb) - the linear grid is the baseline this notebook is read against,
# fitted on identical folds, features and metals. Run with `ML4T_OUTPUT_DIR` pointing at a writable
# experiment.
# **What it writes**: one `training_runs` row per label and configuration, and one
# `prediction_sets` row **per checkpoint**, into the experiment's `run_log/`. It selects nothing.
#
# ## What a tree buys on this panel
#
# `06_linear` fits one weight per feature and applies it everywhere. A tree splits on one feature
# inside a region defined by another, so it can express a statement like *"in the New York session
# rank on the gold-silver ratio; in London rank on the overnight gap"* without anyone constructing
# that column. On a bot whose whole hypothesis is that **intraday structure differs between two
# venues** (`bots/exness_gold_sess/BOT.md`, Hypothesis), that is the interaction worth testing, and
# it is why `is_ny` is in the feature set at all despite being constant inside a filtered book.
#
# What a tree does **not** change is that the panel is two names wide. Everything `06_linear`
# section 4 said about the registry's `ic_mean` applies here unchanged, and this notebook reads the
# same pooled panel IC through the same module.
#
# ## The checkpoint is part of the identity
#
# Every `lgb` preset in this menu declares `max_iterations: 500` and `checkpoint_interval: 50`, so
# each configuration publishes predictions at **ten** training states: 50, 100, ... 500 trees.
# **Fifteen declared configurations at ten checkpoints is 150 candidate models per regression
# label**, and reporting the leading row of that as "the GBM result" would be reporting the maximum
# of 150 draws. The checkpoint travels into `13_backtest` as part of what is selected, and into the
# Trials table of `bots/exness_gold_sess/BOT.md` as part of what is counted.
#
# ## Learning objectives
#
# - Fit a declared capacity-and-loss grid on three labels with folds, features and metals fixed.
# - Read a boosted model as a curve over training states rather than as one end-of-training number.
# - Separate "the checkpoint moved the result" from "the configuration moved the result".
# - Keep every number here a diagnostic; selection is validation backtest Sharpe in `13_backtest`.

# %%
"""Fit the declared exness_gold_sess GBM population on the walk-forward validation folds."""

import lightgbm  # noqa: F401  # GBM libraries import before scikit-learn (OpenMP clash)
import numpy as np
import plotly.graph_objects as go
import polars as pl
import yaml
from IPython.display import display

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

# %% [markdown]
# ### Two host settings that are part of the training identity, not of the environment
#
# Both were measured on `exness_fx_d1` (`bots/exness_fx_d1/BOT.md`, Decisions log) and both belong
# in the **experiment's** `config/setup.yaml`, never in the repository copy - they are properties
# of the machine the fit runs on, and putting them in the repository file would move every training
# identity for every reader:
#
# - **`modeling.gbm.num_threads: 2`.** On the WSL2 host, LightGBM's default of one thread per core
#   turned a 1.2-second fit into a 78-second one through OpenMP contention on a small panel. The
#   value enters the training spec, so a run at 8 threads and a run at 2 are *different identities*
#   and the second must name the population the first published (`SUPERSEDES_POPULATION`).
# - **`LD_LIBRARY_PATH` must reach `libgomp.so.1`.** Measured on this host on 2026-09-08: without
#   it, `import lightgbm` raises `OSError: libgomp.so.1: cannot open shared object file` - it fails
#   at *import*, not at fit time, so a run dies in the cell above rather than after an hour. The
#   library is **not** in `/usr/lib/x86_64-linux-gnu` on this WSL2 host (there is no `libgomp1`
#   package); it is at `/home/pro_trader/omp/libgomp.so.1`, and also inside the sudo-free conda
#   toolchain at `/home/pro_trader/cc/lib/`. So the shell that launches this notebook must export
#   `LD_LIBRARY_PATH=/home/pro_trader/omp:$LD_LIBRARY_PATH`, which was verified today (lightgbm
#   4.6.0 imports and `resolve_gbm_execution_config` returns `('cpu', 255, 2)`). The documented
#   permanent fix is `sudo apt install libgomp1`. It is an environment variable and not part of any
#   hash, which is exactly why it is written down here rather than assumed.
#
# `max_bin` stays at LightGBM's own CPU default of 255 (`setup.yaml::modeling.gbm.max_bin`).

# %%
CASE_STUDY = "exness_gold_sess"
study = open_study(CASE_STUDY, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
setup = yaml.safe_load((get_case_study_dir(CASE_STUDY) / "config" / "setup.yaml").read_text())
SESSION_BOOKS = list(setup["decision"]["session_filter_values"])
LABEL_HORIZONS = {k: int(str(v).rstrip("Hh")) for k, v in setup["labels"]["horizons"].items()}
CLASSIFICATION_EVAL = dict(setup["labels"].get("classification_eval_label") or {})
gbm_settings = dict(setup.get("modeling", {}).get("gbm", {}))
print(f"modeling.gbm as this workspace reads it: {gbm_settings}")
if "num_threads" not in gbm_settings:
    print(
        "  NOTE: modeling.gbm.num_threads is not declared in this setup.yaml. On WSL2 that means "
        "LightGBM's default thread count, which was measured at 78 s per fit against 1.2 s at 2 "
        "threads on the sibling bot. Set it in the EXPERIMENT's config/setup.yaml, not the "
        "repository one."
    )

# %% [markdown]
# ## 1. Which labels, and which models
#
# The labels are the ones `06_linear` fitted, and for the same reason: the two families are
# compared on identical targets, folds, features and metals, so what separates their results is the
# family.

# %%
declared_labels(study, "gbm")

# %% [markdown]
# Each regression label's menu lists **15** configurations under `gbm:`, a product of two axes:
#
# - **Five capacity profiles.** `default` uses the library's own leaf count; the rest fix it at 7,
#   15, 31 and 63. Leaf count is the direct control on how finely one tree may partition the
#   feature space - the dial that decides whether the model can express a condition that only holds
#   inside a region defined by another feature.
# - **Three objectives.** `mse` minimizes squared error, `mae` absolute error, and `huber` behaves
#   like squared error for small residuals and like absolute error beyond a threshold derived from
#   each fold's own label spread.
#
# `dir_tb_8h` lists **5** multiclass configurations, one per capacity profile: there is no loss
# axis on a multiclass objective. The presets declare `num_class: 5`, which is a leftover from the
# case study they were written for; `case_studies/utils/gbm.py:1095` overwrites it with
# `len(class_values)` read from the data, so the three classes of this label are what LightGBM is
# actually given. The menu is otherwise untouched.
#
# Every configuration runs the same number of boosting iterations at the same learning rate, so the
# grid isolates capacity and loss rather than confounding either with training length.

# %%
configs = load_model_configs(
    study,
    "gbm",
    labels=LABELS or None,
    config_names=CONFIG_NAMES or None,
)
display(configs.group_by("label").agg(pl.len().alias("configurations")).sort("label"))
configs

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
# Resolving reads the label and feature artifacts, computes the fold boundaries, works out the rows
# each fit must predict, and turns any data-dependent parameter into the number it will use -
# Huber's threshold is a fraction of each fold's own label standard deviation, so it is a different
# number on every fold.
#
# Nothing is fitted here. Five things to check:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row.**
# - **`eligible_entities` is 2.** The reading in section 4 turns on it.
# - **`folds` is 4 everywhere.**
# - **`validation_start` / `validation_end` bracket the development sample**, asserted below against
#   `evaluation.holdout_start` rather than eyeballed.
# - **`checkpoints` is where this differs from the linear plan.** Multiply it by the number of rows
#   to get the number of candidate models this notebook is about to create.

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
candidates = int(plan.select((pl.col("checkpoints")).sum()).item())
print(f"latest validation end {latest_validation} < holdout_start {holdout_start}: untouched")
print(
    f"{plan.height} configurations x their checkpoints = {candidates} candidate models, "
    f"which is {candidates} prediction sets and {plan.height} training runs"
)
plan.select(
    "config_name",
    "label",
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
# 2. hands the design matrix to LightGBM **without imputing** - a null is a branch direction to a
#    tree, which is the one substantive difference from the linear path, where
#    `SimpleImputer(strategy="median")` runs inside the fold pipeline. The nulls are the declared
#    ones on `pre_ret_*h` and `overnight_gap` / `prev_sess_ret`,
# 3. fits to `max_iterations` trees,
# 4. predicts the fold's validation rows **at each checkpoint**, using only the trees built up to
#    that point.
#
# The fold predictions are concatenated per checkpoint into one series covering the whole validation
# period, and each becomes its own prediction set with its own hash. The purge between train and
# validation comes from the label buffer and was measured at 3-5 decision slots per fold and metal
# in `04_model_based_features`.

# %%
population_name = POPULATION_NAME or "exness_gold_sess-gbm-validation-v1"
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
# ## 4. What came out
#
# One catalog row per configuration **and checkpoint**. As in `06_linear`, the registry's `ic_mean`
# is null on every one of them, and the assertion below turns that into a stated fact at the place
# a reader would otherwise conclude the fits had failed.

# %% tags=["results"]
catalog = (
    execution.catalog_rows.select(
        "config_name",
        "label",
        "checkpoint_kind",
        "checkpoint_value",
        "complete",
        "ic_mean",
        "ic_n_days",
        "n_folds",
        "training_hash",
        "prediction_hash",
    )
    .sort(["label", "config_name", "checkpoint_value"])
    .join(
        configs.select("config_name", "label", "model_class", "params"),
        on=["config_name", "label"],
        how="left",
    )
)
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("gbm execution returned a partial prediction set")
registry_ic_defined = catalog.filter(pl.col("ic_mean").is_not_null()).height
print(
    f"{catalog.height} catalog rows over {catalog.n_unique('config_name')} configurations "
    f"at {catalog.n_unique('checkpoint_value')} checkpoints on "
    f"{catalog.n_unique('label')} labels"
)
print(f"rows carrying a registry ic_mean: {registry_ic_defined}")
print(REGISTRY_IC_IS_NULL_REASON)
catalog.select(
    "label", "config_name", "checkpoint_value", "ic_mean", "ic_n_days", "n_folds"
).head(20)

# %% [markdown]
# ### The pooled panel IC at every checkpoint
#
# The same statistic `06_linear` section 4 computed, on every checkpoint of every configuration:
# rank-normalise score and outcome within each metal, multiply row-wise, average across the metals
# at each slot, then take the mean of that series with a **Newey-West interval at a lag measured on
# the decision grid**. The fold breakdown travels beside the pooled estimate, because four folds of
# +0.02 and one fold of +0.08 with three of zero are different findings.
#
# This is the expensive cell of the notebook: it opens every prediction artifact. It reads and
# writes nothing to the registry.

# %%
rows = []
for row in catalog.iter_rows(named=True):
    result = Result.open(study, row["prediction_hash"])
    if not isinstance(result, PredictionResult):
        raise TypeError(f"{row['prediction_hash']} is not a prediction result")
    frame = result.load()
    missing = {"symbol", "timestamp", "fold", "prediction", "actual"} - set(frame.columns)
    if missing:
        raise ValueError(f"prediction {result.hash} lacks canonical columns: {sorted(missing)}")
    overlap, hac_lag = hac_lag_for_label(
        frame.filter(pl.col("symbol") == frame["symbol"][0])["timestamp"],
        LABEL_HORIZONS[row["label"]],
    )
    outcome_col = outcome_column_for(frame)
    summary = summarise_pooled_ic(
        pooled_panel_ic_series(frame, outcome_col=outcome_col), hac_lag=hac_lag
    )
    rows.append(
        {
            "label": row["label"],
            "config_name": row["config_name"],
            "checkpoint_value": row["checkpoint_value"],
            "scored_against": outcome_col,
            "overlap_slots": overlap,
            **{k: v for k, v in summary.items() if k != "fold_ics"},
            "prediction_hash": row["prediction_hash"],
        }
    )
pooled = pl.DataFrame(rows).sort(["label", "config_name", "checkpoint_value"])
print(
    f"pooled panel IC computed for {pooled.height} candidate models; HAC lag "
    f"{sorted(set(pooled['hac_lags'].to_list()))} from a measured overlap of "
    f"{sorted(set(pooled['overlap_slots'].to_list()))} decision slots"
)
display(
    pooled.group_by("label")
    .agg(
        candidates=pl.len(),
        best_ic=pl.col("ic_mean").max(),
        worst_ic=pl.col("ic_mean").min(),
        above_zero=(pl.col("ic_mean") > 0).sum(),
        max_abs_hac_t=pl.col("hac_t").abs().max(),
        clearing_1p96=(pl.col("hac_t").abs() >= 1.96).sum(),
    )
    .sort("label")
)

# %% [markdown]
# `clearing_1p96` is printed without a multiplicity correction attached and must be read that way:
# at 150 candidates a label, about 7 rows are expected to clear |t| >= 1.96 under a null of no
# skill. It is here to be compared against that expectation, not to nominate a winner.

# %% [markdown]
# ### What more trees do
#
# Each line traces one configuration's out-of-sample pooled IC as trees are added to it. This is the
# figure the checkpoint dimension exists to produce, and it separates two things a single
# end-of-training number cannot: a configuration that improves monotonically with capacity, and one
# that peaks early and then decays as the added trees fit the training window's noise.

# %%
primary = primary_label(study)
charted = sorted(set(pooled.get_column("label")))
panel_labels = [label for label in [primary] if label in charted] + [
    label for label in charted if label != primary
]
palette = [COLORS["blue"], COLORS["copper"], COLORS["amber"], COLORS["neutral"]]
for label in panel_labels:
    series_frame = pooled.filter(pl.col("label") == label)
    fig = go.Figure()
    for index, config_name in enumerate(sorted(set(series_frame.get_column("config_name")))):
        line = series_frame.filter(pl.col("config_name") == config_name).sort("checkpoint_value")
        fig.add_trace(
            go.Scatter(
                x=line.get_column("checkpoint_value").to_list(),
                y=line.get_column("ic_mean").to_list(),
                mode="lines+markers",
                name=config_name,
                line=dict(color=palette[index % len(palette)], width=1.5),
            )
        )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_layout(
        title=f"Pooled panel IC against boosting iterations - {label}",
        height=520,
        width=1000,
        margin=dict(t=70),
        legend=dict(title_text="Configuration"),
    )
    fig.update_xaxes(title_text="Trees")
    fig.update_yaxes(title_text="Pooled panel IC (validation)")
    ends = series_frame.filter(
        pl.col("checkpoint_value") == pl.col("checkpoint_value").max()
    )
    show_plotly_with_alt(
        fig,
        f"Line chart of validation pooled panel information coefficient against the number of "
        f"boosting iterations for {label}, one line per configuration, against a dashed zero line. "
        f"At the last checkpoint {int((ends['ic_mean'] > 0).sum())} of {ends.height} "
        f"configurations end above zero.",
    )

# %% [markdown]
# ### Is the checkpoint as consequential as the model?
#
# One number per configuration: the range its pooled IC covers across its own ten checkpoints,
# against the spread of final-checkpoint ICs across configurations. A ratio near one means choosing
# a checkpoint after seeing the folds would be worth as much as choosing a model - which is exactly
# why the checkpoint is part of the identity that travels into `13_backtest` and into the trial
# count, rather than a training detail.

# %% tags=["results"]
spread = (
    pooled.group_by("label", "config_name")
    .agg(
        ic_min=pl.col("ic_mean").min(),
        ic_max=pl.col("ic_mean").max(),
        peak_checkpoint=pl.col("checkpoint_value").sort_by("ic_mean", descending=True).first(),
        first_checkpoint=pl.col("checkpoint_value").min(),
        last_checkpoint=pl.col("checkpoint_value").max(),
        ic_final=pl.col("ic_mean").sort_by("checkpoint_value").last(),
    )
    .with_columns(
        checkpoint_range=pl.col("ic_max") - pl.col("ic_min"),
        interior_peak=pl.col("peak_checkpoint").is_between(
            pl.col("first_checkpoint"), pl.col("last_checkpoint"), closed="none"
        ),
    )
    .sort(["label", "checkpoint_range"], descending=[False, True])
)
comparison = (
    spread.group_by("label")
    .agg(
        across_configurations=pl.col("ic_final").max() - pl.col("ic_final").min(),
        within_one_configuration=pl.col("checkpoint_range").median(),
        interior_peaks=pl.col("interior_peak").sum(),
        configurations=pl.len(),
    )
    .with_columns(ratio=pl.col("within_one_configuration") / pl.col("across_configurations"))
    .sort("label")
)
display(comparison)
with pl.Config(tbl_rows=spread.height):
    display(spread.select("label", "config_name", "ic_min", "ic_max", "checkpoint_range",
                          "peak_checkpoint", "interior_peak"))

# %% [markdown]
# ### Does the score go stale inside a book?
#
# The per-book staleness screen `05_evaluation` runs on features, applied to the model's own output
# at its final checkpoint. A score that repeats inside `london` or `ny` is a constant for the book
# that would trade it, whatever it does on the pooled panel.

# %% tags=["results"]
final_checkpoint = pooled.filter(
    pl.col("checkpoint_value") == pl.col("checkpoint_value").max().over("label")
)
stale_rows = []
for label in panel_labels:
    best = final_checkpoint.filter(pl.col("label") == label).sort("ic_mean", descending=True).row(
        0, named=True
    )
    frame = Result.open(study, best["prediction_hash"]).load()
    stamps = frame.select("timestamp").unique().sort("timestamp")
    books = stamps.with_columns(session_of(stamps["timestamp"]).alias("session"))
    frame = frame.join(books, on="timestamp", how="left")
    for record in prediction_staleness_by_book(frame).iter_rows(named=True):
        stale_rows.append(
            {
                "label": label,
                "config_name": best["config_name"],
                "checkpoint": best["checkpoint_value"],
                **record,
            }
        )
staleness = pl.DataFrame(stale_rows)
print(
    f"{staleness.filter(pl.col('staleness') > 0.50).height} of {staleness.height} (label, book) "
    "pairs repeat on more than 50 % of consecutive slots"
)
with pl.Config(tbl_rows=staleness.height):
    display(staleness.sort(["label", "book"]))

# %% [markdown]
# ## 5. `dir_tb_8h` under a multiclass objective
#
# `06_linear` section 5 sets out in full what the three-class label is scored with, what it is not
# scored with, and why no uniqueness sample weights are produced for it. The short version, so this
# notebook stands on its own: **IC against the continuous evaluation label `fwd_ret_8h`** (null
# here for the two-name reason, replaced by the pooled panel IC above), plus `accuracy`,
# `balanced_accuracy`, `log_loss`, `brier_score` and `auc_pr` on the label itself, plus an
# `auc_roc` the registry derives by collapsing to "up versus not-up". `auc_mean_daily` is null:
# it needs a binary label and a scorable cross-section, and this label has three states on a
# two-name panel.
#
# One thing to read carefully on a boosted multiclass model: the score that reaches
# `pooled_panel_ic_series` is the model's scalar decision score, and the timeout class is the
# largest of the three (43 % on XAUUSD, 41 % on XAGUSD, `02_labels`). A configuration that predicts
# the timeout class everywhere scores well on `accuracy` and produces a near-constant decision
# score - which is what the staleness table above is there to catch.

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
    for column in available:
        defined = cls.filter(pl.col(column).is_not_null()).height
        print(f"  {column}: {defined} of {cls.height} rows carry a value")
    summary = cls.select("label", "config_name", "checkpoint_value", *available).sort(
        "label", "config_name", "checkpoint_value"
    )
    with pl.Config(tbl_rows=30, tbl_cols=14, tbl_width_chars=220):
        display(summary.head(30))
else:
    print("no classification label is declared")

# %% [markdown]
# ## 6. What to notice
#
# Fill this in from the frames above after the first canonical run, quoting the numbers as they came
# out and naming the host, the date, the experiment and the population, the way
# `case_studies/exness_fx_d1/07_gbm.py` section 5 does. It is deliberately empty rather than
# carrying a placeholder that would read as a result.
#
# What to carry forward whatever the numbers are:
#
# - **Every checkpoint of every configuration advances to `13_backtest`.** Nothing here narrows the
#   population, and the trial count in `bots/exness_gold_sess/BOT.md` counts all of them.
# - **The GBM grid is read against `06_linear`, not against zero.** Identical folds, features,
#   metals and labels; the only thing that changed is what the estimator can represent.
# - **A checkpoint chosen after seeing the folds is a selection.** The `comparison` frame above says
#   how large that selection is relative to choosing a model.

# %%
handoff = catalog.select(
    "label", "config_name", "checkpoint_kind", "checkpoint_value", "training_hash",
    "prediction_hash", "complete",
).sort("label", "config_name", "checkpoint_value")
print(f"{handoff.height} prediction sets advance to 13_backtest")
handoff
