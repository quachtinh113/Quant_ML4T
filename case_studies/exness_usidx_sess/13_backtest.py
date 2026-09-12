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
# > began as `case_studies/exness_fx_d1/13_backtest.py` with the study id, the loader and the
# > population names changed. Wherever the prose below still described the FX template - five
# > currency pairs, a cross-sectional ranking, `spread_bps.major_pairs`, a benchmark approved on
# > a date - it was **not** silently reworded into a claim about this bot: it was deleted and
# > replaced with what this bot measured, and the deletions are listed in `BOT.md`, Decisions
# > log, 2026-09-08 (third pass). This banner follows `case_studies/xau_fx_mt5/13_backtest.py`.
# > **This bot has two instruments and no cross-section.** Any sentence anywhere in this file
# > that ranks assets against each other is template residue and a defect; report it.

# %% [markdown]
# # Threshold Backtest - Exness US indices, session bot
#
# **Docker image**: `ml4t`
#
# The bot's `13_backtest` (bot `exness_usidx_sess`, roadmap phase 5). At the decision instant this
# workspace's spec declares - the close of the first H1 bar closing 30 minutes after the NYSE cash
# open, or the close of the last H1 bar closing at or before the cash close - it turns each index's
# own score into a position. Every complete model configuration and checkpoint of the linear and
# GBM populations is evaluated at every setting the sweep declares, so the number of rows this
# stage writes is the trial count the Deflated Sharpe Ratio divides by.
#
# ## The execution mode, and why it is `vectorized`
#
# **CHANGED 2026-09-08 (third pass), after a mentor review.** This stage ran once under the
# default `engine` mode and its 339 rows were deleted, because the engine mode cannot express
# this bot and produced a lookahead rather than a pessimistic approximation:
#
# 1. **The fill happened before the decision.** Prediction sets are keyed on the session **date**
#    (midnight) - the feature matrix and the label file are, and `load_modeling_dataset` carries
#    that key through - while the registered price grid is **hourly**. `next_bar` fills a weight
#    decided at row `t` at the open of row `t+1`, so a weight keyed at 00:00 filled in the
#    00:00-01:00 bar: measured, every fill at 01:00 UTC against decision instants of 14:00/15:00.
# 2. **The holding period was not the label's.** The engine holds continuously from one fill to
#    the next; both specs are flat for part of their cycle, and `_signals_to_equal_weights` emits
#    no row at all for a flat session, so no zero target ever reached the executor. Measured: a
#    median holding period of **2,088 hours** against a six-hour label.
#
# `build_backtest_spec` has always exposed `execution_mode` as a keyword
# (`case_studies/utils/backtest_presets.py`), and `run_backtest` branches on it, so this stage
# passes `execution_mode="vectorized"` and needs **no change to any shared module**. (Adding this
# bot to `backtest_loaders.VECTORIZED_CASE_STUDIES` would not have worked: that constant is read
# only by the `*_risk_management` stages.) The vectorised path is `weight x forward return - cost`
# on the label the prediction set carries, which is exactly this bot's economics: one decision,
# one label, one session. It registers `backtest_runs` normally, which is what the phase-5 gate
# of the roadmap asks for.
#
# **What that route costs, stated rather than discovered later:**
#
# - `_run_vectorized` charges **turnover**, `|dw|` between consecutive rebalances, not a round
#   trip per session. This bot's declared convention is a **full round trip on every session a
#   symbol is held**, because both specs close their own position at the end of their own window;
#   there is no position to carry and no turnover to net. So the registered rows **under-charge**
#   a book that holds the same weight on consecutive sessions. `_report_phase5.py` charges the
#   declared round trip and is the number `BOT.md` reports; the registry is the audit trail.
# - The vectorised path **refuses** `daily_loss` and every position-level rule (stop loss,
#   trailing stop, time exit). That is the fill, margin and risk-overlay machinery phases 6 to 8
#   assume, and it is the real price of this route.
# - `execution.mode` enters the identity hash, so every future backtest identity of this bot is
#   pinned to this route until a generation declares otherwise.
#
# A third route exists and is recorded rather than used: `backtest_runner.py` already supports an
# explicit flat through `_state_transition` -> `flatten_all_positions`, which is exactly the exit
# defect 2 needs - but it is refused unless `execution_mode` is `same_bar`.
#
# ## The two signal families
#
# **There is no `equal_weight_top_k` here, and its absence is the design.** Ranking **two**
# instruments is not a cross-sectional strategy; it is a US500/USTEC spread trade, which is not
# what `bots/README.md` declares this bot to be, and its "information coefficient" would be +1 or
# -1 on every session by construction. `setup.yaml::mapping.class` is therefore
# `time_series_threshold` and both signal families below compare an index with **its own** score
# distribution, so both, one or neither index can be held on a session. The framing is Chapter 16
# `04_single_asset_ml4t_backtest` / `05_stateful_strategies`.
#
# | `SIGNAL_FAMILY` | book | what an index is compared with |
# |---|---|---|
# | `fixed_threshold` | long when the forecast clears `+t`, short when it clears `-t`, flat between | a declared absolute number of basis points |
# | `per_symbol_rolling_percentile` | long when the score is in its own trailing top fifth, equal weight, long only | its own trailing scores |
#
# **The short threshold is `-t`, and that too changed on 2026-09-08 (third pass).** The library's
# `fixed_threshold_signal` puts the short threshold of a long-short book at `1 - t`, the mirror of
# the long threshold about 0.5, which is right for a probability and wrong for a regression score:
# at 1e-3 every non-long observation is below 0.9992 and becomes a short, so the book is invested
# **100 % of the time** and is never flat. Generation 1 of this sweep tested that book - 1,068 of
# its 1,780 trials - not the book the hypothesis describes. `signals.py` now takes a
# `threshold_convention`, `setup.yaml` declares `signed`, and the value is part of the signal dict
# and therefore of the identity. At `t = 0.0` the book is still invested whenever the forecast is
# non-zero, because a zero threshold **is** the sign of the forecast; that is the declared meaning
# of that grid value, not a residue of the defect.
#
# **One workspace, one spec.** This bot decides twice per cash session and the two decisions have
# different feature matrices (`setup.yaml::labels.spec_of`), so each spec has its own experiment
# workspace and its own registry. `SESSION_FILTER` is therefore not a parameter of this notebook:
# the workspace **is** the filter, and `labels.primary` names it. The trial count `K` in
# `bots/exness_usidx_sess/BOT.md` accumulates over the **bot**, so both workspaces' rows count
# toward it and `_report_phase5.py` reads both registries. **No Deflated Sharpe Ratio may be
# printed from inside this stage**: every helper that computes one reads a single registry and
# would report half of K as though it were all of it.
#
# **What the cost model charges.** The percentage cost model reads `setup.yaml::costs` through
# `get_backtest_config`: no commission (Pro account) and a slippage leg of
# `spread_bps.indices[-1]`, charged on every crossing. That number was **corrected on 2026-09-08**
# from 0.66 to **4.05 basis points** after the first figure was found to come from a thirty-day
# tick window that could not see the 2024-10 spread regime break; it is the worst symbol-month p90
# of this account own H1 tape at the six hours this bot crosses in, measured on development rows
# only. The cell below asserts that exactly that number was resolved, because
# `config/backtest/base.yaml` carries an inherited `commission.rate 0.0005` and `slippage.rate
# 0.0002` that `case_studies/utils/backtest_presets.build_resolved_backtest_config` overwrites -
# the assertion is what turns "we traced the code" into "we checked the run". One number is
# charged here for the whole window; the per-session regime charge that
# `setup.yaml::costs.spread_bps_charge_rule` declares is applied in `_report_phase5.py`, which is
# the only place it can be applied, and the conservative rate is the one that stands in the
# registry.
#
# **Swap is a holding cost and this path has no model for it.** `setup.yaml::costs.swap` declares
# -147.4 points a night on `US500` and -592.7 on `USTEC` for a long, zero for a short, tripled on a
# Friday. The `intraday` spec never crosses a server midnight and so never pays it; the `overnight`
# spec pays it on every long night. The FX fork refuses to run at all when a non-zero swap is
# declared, which would stop this bot dead. The rule here is different and narrower: a non-zero
# swap is allowed **only** because `_report_phase5.py` books it outside the backtest with
# `bots/_shared/costs_mt5.holding_cost_points`, per real night, long only, three nights on a
# Friday. This notebook numbers are therefore **gross of financing** for the overnight spec, and
# the assertion below records that rather than letting it be forgotten.
#
# **Learning objectives**
#
# - Freeze the exact prediction population before running a baseline sweep.
# - Send selected catalog rows to the shared backtest entry point, priced with the costs
#   `setup.yaml` declares, on the execution mode this bot decision grid can support.
# - Verify that every declared prediction and signal setting produces one backtest.
#
# **Book reference**: Chapter 16
#
# **Prerequisite**: `12_model_analysis`.

# %%
"""Run the complete threshold validation backtest population of exness_usidx_sess."""

import polars as pl
import yaml

from case_studies.research import (
    OfficialPopulation,
    open_study,
    plan_backtests,
    population_supersedes,
    research_name,
    run_backtests,
    superseded_members,
)
from case_studies.exness_usidx_sess._sweep import declared_settings
from case_studies.utils.backtest_loaders import get_backtest_config
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "exness_usidx_sess"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
# The workspace declares which spec it models (`labels.primary`), so LABEL is a narrowing control
# and not the spec selector. There is no SESSION_FILTER parameter for the same reason: the
# workspace is the filter.
LABEL = ""
SPLIT = "validation"
SEED = 42
RUN_SWEEP = True
FORCE_REBACKTEST = False
TOP_N_PREDICTIONS = None
POPULATION_NAME = ""
# "fixed_threshold" or "per_symbol_rolling_percentile"; each publishes its own population.
SIGNAL_FAMILY = "fixed_threshold"
# The execution model, and part of every identity this stage registers. "vectorized" is the only
# value this bot's decision grid supports; see the header for the two defects of "engine" and the
# three costs of this route. It is a parameter rather than a literal so that a run which changes
# it has to say so in its own record.
EXECUTION_MODE = "vectorized"
SUPERSEDES_FIXED_THRESHOLD_BASELINES: str = ""
SUPERSEDES_TIMESERIES_BASELINES: str = ""
SUPERSEDES_VALIDATION_PREDICTIONS: str = ""

# %% [markdown]
# ## Select the exact prediction population
#
# Canonical production includes every complete validation prediction the model notebooks currently
# publish. Label or population limits are preview controls and cannot define the official baseline.
#
# **"Currently publish" is a question about lineage, and the catalog cannot answer it.** A row's
# `identity_status` is derived from the schema version it was written under, so it says the
# registry still understands the row - not that the row is the one its producer stands behind. The
# two agree until a model notebook refits: it then publishes a second generation under the same
# name, the first generation's prediction sets stay in the registry complete and current, and a
# sweep selecting on the catalog alone runs over both. It would not fail; it would report every
# member complete over twice the population, and freeze the retired half into the baseline the
# rest of the case study is measured against. `superseded_members` asks the registry which
# identities a later generation retired, which is exactly the set to drop. This bot has one
# generation of each population, so the count printed below is expected to be 0, and it is printed
# rather than left implicit.
#
# The three `SUPERSEDES_*` declarations are empty because every population here is generation 1.
# They stop being empty the moment `03_financial_features` rebuilds the matrix - which the VIX and
# FOMC families would do (`03`, Known limitations) - because `06_linear` and `07_gbm` would then
# refit under new identities and a run that backtested them would have to declare which snapshot
# it replaces, or it would claim a lineage it does not have.

# %% tags=["results"]
set_global_seeds(SEED)
if SPLIT != "validation":
    raise ValueError("the baseline sweep uses validation predictions")
if FORCE_REBACKTEST:
    raise ValueError("identical complete backtests are reused by identity")
if not RUN_SWEEP:
    raise ValueError("set RUN_SWEEP=True to execute the visible baseline request")

study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
# The execution tier decides which registry namespace this run reads and writes;
# the reduction knobs decide only how much of it is covered. Inferring the tier
# from the knobs conflated the two, so any reduced run went looking for preview
# predictions - and a reduced run over a canonical upstream, which is what the
# test suite exercises, then resolved no rows at all.
include_preview = EXECUTION_TIER == "preview"
catalog = study.predictions.table(include_preview=include_preview).filter(
    (pl.col("identity_status") == "current") & (pl.col("split") == SPLIT) & pl.col("complete")
)
if include_preview:
    catalog = catalog.filter(pl.col("execution_tier") == "preview")
else:
    catalog = catalog.filter(pl.col("execution_tier") == "canonical")
retired = superseded_members(study, member_kind="prediction")
if retired:
    offered = catalog.height
    catalog = catalog.filter(~pl.col("prediction_hash").is_in(list(retired)))
    print(f"Retired by a later generation, excluded: {offered - catalog.height} of {offered}")
if LABEL:
    catalog = catalog.filter(pl.col("label") == LABEL)
if TOP_N_PREDICTIONS is not None:
    catalog = catalog.sort("label", "family", "config_name", "checkpoint_value").head(
        TOP_N_PREDICTIONS
    )

if catalog.is_empty():
    raise RuntimeError("the baseline request resolved no complete prediction rows")

# TOP_N_PREDICTIONS and LABEL both narrow what is backtested, and a narrowed run declares
# a different set of members than the canonical population does. A population is immutable
# once written, so such a run must publish under its own name rather than register a
# partial snapshot under the canonical one. The tier is a separate question: a canonical
# run may legitimately be narrowed, it just may not claim to be the whole population.
if (
    (TOP_N_PREDICTIONS is not None or LABEL)
    and not include_preview
    and not POPULATION_NAME
):
    raise ValueError(
        "this run narrows the baseline sweep, so it cannot publish the canonical "
        "population; pass POPULATION_NAME to give it its own"
    )
if catalog.get_column("prediction_hash").n_unique() != catalog.height:
    raise RuntimeError("the baseline population contains duplicate prediction identities")

if not include_preview:
    predictions_name = research_name(CASE_STUDY_ID, "validation-predictions", scope=POPULATION_NAME)
    prediction_population = OfficialPopulation.create(
        study,
        name=predictions_name,
        member_kind="prediction",
        members=catalog.get_column("prediction_hash").to_list(),
        supersedes=population_supersedes(
            study, name=predictions_name, declared=SUPERSEDES_VALIDATION_PREDICTIONS
        ),
    )
    prediction_population.require_complete()
    print(f"Frozen prediction population: {prediction_population.hash}")
else:
    print("Preview selection is isolated from the official prediction population.")

catalog.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "prediction_hash",
)

# %% [markdown]
# ## Build the baseline strategy grid
#
# **DELETED 2026-09-08 (third pass).** Everything that stood between this heading and the cost
# assertion below was the FX template's, describing a `cross_sectional_top_k` sweep over five
# currency pairs, a `top_k_grid` this bot does not declare, "the 1/N long book of the five pairs
# (validation Sharpe 0.41)", "the benchmark the user approved on 2026-09-06", "the same 1,068
# trials the cross-sectional generations each added", and a cost of "2.6 bps" taken from
# `spread_bps.major_pairs[-1]`, the p90 of AUDUSD. None of it is true of `exness_usidx_sess`:
# there is no `top_k_grid`, no approval of any benchmark on any date (`BOT.md`, "Awaiting user
# approval", item 3, is still open), no five pairs, and the declared cost is 4.05 bps a crossing
# on `spread_bps.indices`. It is deleted rather than reworded, and it is named here so that
# anyone who read a number out of this file before today can find out which numbers were fiction.
#
# What this stage actually declares is two families and five settings, and they are built by
# `case_studies/exness_usidx_sess/_sweep.py::declared_settings` from `setup.yaml` alone -
# **one function**, imported here and by `_report_phase5.py`, so that the dicts the identity hash
# is taken over and the dicts the report scores cannot drift apart. They were built twice, in two
# files, until 2026-09-08.
#
# ### `fixed_threshold`
#
# `setup.yaml::backtest.sweep.threshold_grid.fixed_threshold` declares `[0.0, 0.0002, 0.0008]`
# on a **regression** score, so the threshold is in the units of the label, a forecast return:
# 0.0 is the sign of the forecast, 0.0002 is a round trip in the post-2024-10 spread regime, and
# 0.0008 is a round trip at the declared p90. The book is long above `+t` and **short below
# `-t`**, flat in the band between, which is `signed` in
# `threshold_grid.fixed_threshold_convention` and what the header explains at length.
#
# ### `per_symbol_rolling_percentile`
#
# The Chapter 16 `08_signal_method_comparison` time-series framing: an index is compared with
# **its own** trailing scores rather than with the other index on the same session, so both, one
# or neither can be held. `signals.py::per_symbol_rolling_percentile_signal` takes the trailing
# `long_q` quantile of each index's own score, lagged one session, over a window of
# `lookback_days * bars_per_day` rows, and `_signals_to_equal_weights` splits the book equally
# over whichever indices cleared it.
#
# - `bars_per_day: 1` - one decision per session per spec. Not the library default of 390, which
#   is a minute-bar number and would make `min_samples` unreachable on this window, leaving every
#   threshold null and every book permanently flat while the sweep completed and registered.
# - `long_q: 0.80` - an index is held when its score is in its own top fifth.
# - `lookback_days: [63, 252]` - a quarter and a year, the two windows the feature register
#   already uses.
# - `direction: long_only`, `long_short: false` - the construction a cash account holds exactly.
#   The pinned `config/backtest/base.yaml` declares `allow_short_selling` without
#   `allow_leverage`, so a short is sized by the cash on hand. A short leg here would double what
#   this family adds to K and has not been approved.
#
# **5 settings x 178 prediction sets x 2 workspaces = 1,780 trials, and that is K.** No setting
# was added for this generation and none was removed; the corrected run asks the same questions.
#
# ### The cost assertion
#
# The check below refuses to run the sweep on a cost block that fell through to the generic
# fallback, and it compares against `setup.yaml` rather than against a number typed here.

# %% tags=["results"]
setup = yaml.safe_load((get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text())
universe_symbols = setup["universe"]["symbols"]
n_assets = len(universe_symbols)
LABEL_IN_WORKSPACE = setup["labels"]["primary"]
SPEC = setup["labels"]["spec_of"][LABEL_IN_WORKSPACE]

case_config = get_backtest_config(CASE_STUDY_ID)
declared_costs = setup["costs"]
declared_spread_bps = float(declared_costs["spread_bps"]["indices"][-1])
if case_config.commission_bps != 0.0 or case_config.slippage_bps != declared_spread_bps:
    raise RuntimeError(
        "the engine cost block does not match setup.yaml::costs: "
        f"commission {case_config.commission_bps} bps, slippage {case_config.slippage_bps} bps "
        f"against commission_per_lot {declared_costs['commission_per_lot']} and spread "
        f"{declared_spread_bps} bps. config/backtest/base.yaml carries an inherited "
        "commission.rate 0.0005 and slippage.rate 0.0002 which build_resolved_backtest_config is "
        "supposed to overwrite from setup.yaml; if this fires, it did not"
    )

# The swap rule, and it is narrower than the FX fork's. That notebook refuses any non-zero swap,
# because its demo account reports zero on every pair and a non-zero value would mean a holding
# cost nobody had priced. Here the swap is non-zero by measurement and material on one of the two
# specs, so refusing it would stop the bot rather than protect it. What is refused instead is
# running the OVERNIGHT spec while pretending the engine charged the financing: the intraday spec
# never crosses a server midnight and owes nothing, the overnight spec owes one night on every
# long (three on a Friday, nothing on a short), and that ledger is kept outside the engine by
# `_report_phase5.py` through `bots/_shared/costs_mt5.holding_cost_points`.
swap_points = declared_costs["swap"]["points_per_lot_per_night"]
any_swap = any(
    float(v["long"]) != 0.0 or float(v["short"]) != 0.0 for v in swap_points.values()
)
if any_swap and SPEC not in ("intraday", "overnight"):
    raise RuntimeError(f"unknown spec {SPEC!r}: cannot decide whether it crosses a server midnight")
if SPEC == "intraday":
    swap_note = (
        "the intraday spec is entered after the cash open and closed at the cash close, so it "
        "never crosses a server midnight and pays no financing; these returns are net of costs"
    )
else:
    swap_note = (
        "THE OVERNIGHT SPEC PAYS FINANCING THE ENGINE CANNOT CHARGE: "
        + ", ".join(
            f"{s} {float(v['long']):.1f} points a night long / {float(v['short']):.1f} short"
            for s, v in sorted(swap_points.items())
        )
        + f", tripled on {declared_costs['swap']['rollover3days']}. These returns are GROSS of "
        "financing; _report_phase5.py subtracts it per real night before any statistic is read"
    )
print(
    f"Workspace spec {SPEC!r} (label {LABEL_IN_WORKSPACE}). "
    f"Engine costs from setup.yaml: commission {case_config.commission_bps:.2f} bps, "
    f"spread {case_config.slippage_bps:.2f} bps per crossing "
    f"({2 * case_config.slippage_bps:.2f} bps round trip), "
    f"fill {case_config.execution_delay}, {case_config.share_type} units"
)
print(f"Swap: {swap_note}")

# One signal dict per job, built from `setup.yaml` alone. The dict is what the identity hash is
# taken over, so it is assembled once here and both `plan_backtests` and `run_backtests` are given
# the same object: a key that differed between the two would plan one identity and run another.
BASELINE_POPULATION_SUFFIX = {
    "fixed_threshold": "fixed-threshold-baselines",
    "per_symbol_rolling_percentile": "timeseries-percentile-baselines",
}
if SIGNAL_FAMILY not in BASELINE_POPULATION_SUFFIX:
    raise ValueError(
        f"unknown SIGNAL_FAMILY {SIGNAL_FAMILY!r}: {sorted(BASELINE_POPULATION_SUFFIX)}"
    )
supersedes_baselines = {
    "fixed_threshold": SUPERSEDES_FIXED_THRESHOLD_BASELINES,
    "per_symbol_rolling_percentile": SUPERSEDES_TIMESERIES_BASELINES,
}[SIGNAL_FAMILY]

# One function, imported by this stage and by `_report_phase5.py`, so the dicts the identity
# hash is taken over and the dicts the report scores are the same objects built the same way.
# It reads `setup.yaml` and nothing else, and it refuses a convention or a bars_per_day the
# library cannot honour.
settings = declared_settings(setup, family=SIGNAL_FAMILY)
jobs = []
for label in sorted(catalog.get_column("label").unique()):
    selected = catalog.filter(pl.col("label") == label)
    for signal in settings:
        jobs.append(
            {
                "label": label,
                "signal": signal,
                "predictions": selected,
                "expected": selected.height,
            }
        )

declared_trials = sum(job["expected"] for job in jobs)
print(
    f"Signal family: {SIGNAL_FAMILY}; {len(jobs)} settings x {catalog.height} prediction sets "
    f"= {declared_trials} declared trials, every one a trial in K"
)

# %% [markdown]
# ## The trials that have no book, and why they are counted but not registered
#
# A member whose weight frame is **empty** - a book that never opened a position anywhere in its
# window - cannot be registered, and that is this repository's rule rather than this bot's
# convenience. `backtest_runner._refuse_an_allocation_that_produced_no_target` refuses to write
# such a run, with the reason spelled out there: an empty weight frame "is not a strategy that
# traded little, it is a strategy the engine was never given anything to trade towards", every
# return-derived metric it would record is the metric of a flat account, and a Sharpe of 0.0
# written into `backtest_runs` "sits above every candidate whose Sharpe is negative" while
# nothing downstream filters it out of the trial count. The vectorised path refuses it a second
# time and earlier, because an empty book produces an empty return series over the canonical
# window.
#
# These members are **still trials**. They were declared before the run, they were asked, and
# they count in K exactly like every other. What they do not have is a backtest to register. So
# the cell below splits them out **before** the population is frozen, prints how many there are
# per setting, and leaves them out of the frozen expected set - a population that promised a
# member the repository refuses to write could never be complete.
#
# The split is mechanical and deterministic: it is the same
# `build_target_weights_from_config` on the same predictions with the same signal dict that the
# run itself will use, and the test is `height == 0`. It is not a filter on a result: nothing
# here reads a return, a Sharpe or a cost. `_report_phase5.py` scores these members anyway - a
# book that never trades earns zero on every session, which is a perfectly well defined return
# series - and reports them with `n_traded = 0`.
#
# **This condition is new to the corrected generation, and it is the correction working.** Under
# the mirrored short threshold that generation 1 ran, every `fixed_threshold` book was invested
# on every session and no book could be empty; under the declared `signed` convention a forecast
# inside the dead band is flat, which is what the hypothesis says the bot does.

# %% tags=["results"]
from case_studies.utils.registry import read_predictions  # noqa: E402
from case_studies.utils.signals import build_target_weights_from_config  # noqa: E402

no_book_rows = []
for job in jobs:
    kept = []
    for prediction_hash in job["predictions"].get_column("prediction_hash").to_list():
        preds = read_predictions(CASE_STUDY_ID, prediction_hash)
        if preds is None or preds.is_empty():
            raise RuntimeError(f"prediction set {prediction_hash} resolved no rows")
        if build_target_weights_from_config(preds, job["signal"]).height:
            kept.append(prediction_hash)
        else:
            no_book_rows.append(
                {
                    "label": job["label"],
                    **{k: str(v) for k, v in job["signal"].items()},
                    "prediction_hash": prediction_hash,
                }
            )
    job["predictions"] = job["predictions"].filter(pl.col("prediction_hash").is_in(kept))
    job["expected"] = job["predictions"].height

no_book = pl.DataFrame(no_book_rows)
registrable = sum(job["expected"] for job in jobs)
if registrable + no_book.height != declared_trials:
    raise RuntimeError("the split into registrable and no-book members lost a trial")
print(
    f"{declared_trials} declared trials: {registrable} have a book and will be registered, "
    f"{no_book.height} never open a position anywhere in their window and are refused by "
    "backtest_runner._refuse_an_allocation_that_produced_no_target. All "
    f"{declared_trials} count in K."
)
jobs = [job for job in jobs if job["expected"]]
if not jobs:
    raise RuntimeError(
        "every declared trial of this signal family produces an empty book: there is nothing to "
        "register and the setting itself has to be re-read before the sweep is run again"
    )
no_book.group_by([c for c in no_book.columns if c != "prediction_hash"]).len() if no_book.height else no_book
pl.DataFrame(
    [
        {
            "label": job["label"],
            **{k: str(v) for k, v in job["signal"].items()},
            "prediction_sets": job["expected"],
        }
        for job in jobs
    ]
)

# %% [markdown]
# ## Freeze the expected baseline population
#
# Planning resolves every backtest identity without running or writing it. Production freezes that
# complete expected set before the first member executes, so a failed member remains visible.

# %% tags=["results"]
planned_hashes = []
for job in jobs:
    plan = plan_backtests(
        study,
        predictions=job["predictions"],
        # The dict built in the grid cell from setup.yaml. `run_backtests` below is given the
        # same object, so plan and run resolve the same identity.
        signal=job["signal"],
        # See the header. `build_backtest_spec` exposes this keyword, `run_backtest` branches on
        # it, and it enters the identity hash - so it has to be passed to BOTH plan and run or
        # the plan would freeze one identity and the run would register another.
        execution_mode=EXECUTION_MODE,
        chapter="16",
    )
    if len(plan.members) != job["expected"]:
        raise RuntimeError("a baseline plan omitted a selected prediction")
    planned_hashes.extend(plan.expected_hashes)

if len(planned_hashes) != len(set(planned_hashes)):
    raise RuntimeError("two planned baseline jobs collapse to the same backtest identity")

baseline_population = None
if not include_preview:
    baselines_name = research_name(
        CASE_STUDY_ID, BASELINE_POPULATION_SUFFIX[SIGNAL_FAMILY], scope=POPULATION_NAME
    )
    baseline_population = OfficialPopulation.create(
        study,
        name=baselines_name,
        member_kind="backtest",
        members=planned_hashes,
        supersedes=population_supersedes(
            study, name=baselines_name, declared=supersedes_baselines
        ),
    )
    print(f"Frozen expected baseline population: {baseline_population.hash}")
else:
    print("Preview backtests remain outside official populations and selection.")

# %% [markdown]
# ## Run every catalog row through the engine, then validate what was frozen
#
# Each selected row produces an independent backtest. The loop has no exception-and-continue path:
# one failed member leaves the predeclared population incomplete and stops publication.
#
# The population is validated in the same cell that fills it, because the two are one act: the
# expected set was written down before the first member ran, and `require_complete` is the only
# thing that turns it from a declaration into a published result. It can pass only when every
# planned model, checkpoint and portfolio size completed.

# %% tags=["results"]
# A sweep that recomputes everything and a sweep that recomputes nothing print the same summary
# unless the two are counted apart. `run_backtests` serves an identity that is already registered
# and complete instead of running it again, which is what makes a re-run affordable and what makes
# a bare member count say nothing about whether this run did any work.
#
# The runner already knows which it did and says so per member in `execution.diagnostics`, as
# `status` "reused" or "completed". Comparing against the registered hashes instead would be
# wrong in both directions: a registered-but-partial backtest is in that set, gets recomputed and
# would report as reused, and a preview re-run reads a table that excludes preview rows by default
# and would report every reused member as computed.
run_status: list[str] = []
backtests = []
for job in jobs:
    execution = run_backtests(
        study,
        predictions=job["predictions"],
        # The same dict the plan above was given (same identity).
        signal=job["signal"],
        execution_mode=EXECUTION_MODE,
        chapter="16",
    )
    if len(execution.results) != job["expected"]:
        raise RuntimeError("a baseline member disappeared during execution")
    backtests.extend(execution.results)
    run_status.extend(entry["status"] for entry in execution.diagnostics)

expected_count = sum(job["expected"] for job in jobs)
if len(backtests) != expected_count:
    raise RuntimeError(f"expected {expected_count} baseline runs, found {len(backtests)}")
if {result.hash for result in backtests} != set(planned_hashes):
    raise RuntimeError("completed baseline identities differ from the frozen plan")
if any(not result.complete for result in backtests):
    raise RuntimeError("the baseline population contains an incomplete backtest")

backtest_rows = pl.DataFrame(
    [
        {
            "backtest_hash": result.hash,
            "prediction_hash": result.registry_record()["prediction_hash"],
            "stage": result.registry_record()["stage"],
            "complete": result.complete,
        }
        for result in backtests
    ]
)
if set(backtest_rows.get_column("stage")) != {"signal"}:
    raise RuntimeError("threshold baseline runs must register with stage='signal'")

served = run_status.count("reused")
print(
    f"{SIGNAL_FAMILY} baselines: {len(backtests) - served} computed, {served} served from the "
    f"registry, {len(backtests)} in the population"
)

if not include_preview:
    if baseline_population is None:
        raise RuntimeError("the canonical baseline population was not frozen before execution")
    baseline_population.require_complete()
    print(f"Official {baselines_name} population: {baseline_population.hash}")

backtest_rows

# %% [markdown]
# ## Key takeaways
#
# - The sweep evaluates every complete model configuration and checkpoint at every declared
#   setting, and the number of members is what the Deflated Sharpe Ratio in `BOT.md` divides by.
#   Nothing was narrowed after seeing a result, and the corrected generation adds no setting.
# - **Two instruments carry no cross-section**, so both families compare an index with its own
#   scores rather than with the other index. There is no `equal_weight_top_k` and no `top_k_grid`.
# - **The execution mode is `vectorized`, declared, and in the identity.** It registers
#   `backtest_runs` and it charges TURNOVER, not the round trip per session this bot declares;
#   `_report_phase5.py` charges the round trip, the per-session regime spread and the swap, and
#   its numbers are the ones `BOT.md` reports. The vectorised path also refuses `daily_loss` and
#   every position-level rule, which is the machinery phases 6 to 8 assume.
# - **One workspace is one spec.** The other spec's backtests live in the other workspace's
#   registry, and `_report_phase5.py` reads both, because `K` accumulates over the bot. **No DSR
#   is printed here**: every helper reads one registry and would report half of K.
# - The cost model charges the corrected spread - 4.05 basis points a crossing, the worst
#   symbol-month p90 of this account's own development tape - and the cell above asserts it did,
#   because the pinned `config/backtest/base.yaml` carries a contradicting inherited default.
# - **The overnight spec's returns here are gross of financing.** There is no holding-cost model
#   on this path; `_report_phase5.py` subtracts the swap per real night, long only, tripled on a
#   Friday, before any statistic is read. A number taken from this notebook alone would overstate
#   that spec by roughly 1.9 to 2.0 basis points for every night it held.
# - Immutable population membership makes missing or failed jobs visible.
# - Selection happens on this validation split only; the holdout is scored once, later, and only
#   after phases 5 and 6 are green - and because this bot has TWO registries, that discipline is
#   enforced by hand across both (`BOT.md`, the two-registry rule).
