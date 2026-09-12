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
# # Equal-Weight Backtest - Exness FX D1
#
# **Docker image**: `ml4t`
#
# The bot's `13_backtest` (bot `exness_fx_d1`, roadmap phase 5): a copy of `fx_pairs/13_backtest`
# with the study id, the population lineage and the prose changed; code paths untouched. At each
# decision time (the close of the last MT5 four-hour bar before the New York rollover, 20:00 UTC
# on this server) it turns the five dollar-pair scores into a book and runs the shared engine on
# the session panel of `_features.py`: the fill is the open of the bar that starts at the
# decision instant (`execution_delay: next_bar_open`), the mark is the decision bar's close (the
# label price). Every complete model configuration and checkpoint of the linear and GBM
# populations is evaluated at every setting the sweep declares, so the number of rows this stage
# writes is the trial count the Deflated Sharpe Ratio divides by. The input is a Polars catalog
# selection, so no prediction hash is copied into orchestration code.
#
# **Two signal families, two populations.** `SIGNAL_FAMILY` in the parameters cell selects which
# book the sweep builds, and each family publishes under its own population name, because the
# two are different strategies rather than two generations of one strategy:
#
# | `SIGNAL_FAMILY` | book | population | framing |
# |---|---|---|---|
# | `cross_sectional_top_k` | `top_k` pairs long and `top_k` short, equal-weighted per sleeve | `exness_fx_d1:equal-weight-baselines` | cross-sectional: a pair is ranked against the other four on the same date |
# | `per_pair_timeseries` | every pair whose own score clears its own trailing quantile, equal-weighted, long only | `exness_fx_d1:timeseries-percentile-baselines` | time series: a pair is ranked against its own history (Chapter 16, `08_signal_method_comparison`) |
#
# Superseding one population with the other would be wrong. Generation 1 of the equal-weight
# baselines was superseded by generation 2 because the only spec change was a single key
# (`long_short`), so the two are one experiment answered twice; a different `method` is a
# different question and gets its own name. The trial count `K` in `bots/exness_fx_d1/BOT.md`
# accumulates over the **bot**, not over a population, so both families count toward it.
#
# **Two generations of the baseline population.** Generation 1 (2026-09-05/06, population
# `b57aa4a93f72`, 1,068 rows) was a **long-only top-k rotation**: the stage was written and
# reviewed as a long-short sleeve book, but the signal dict it sent was
# `{"method": "equal_weight_top_k", "top_k": k}` and
# `case_studies/utils/signals.py::build_target_weights_from_config` defaults `long_short` to
# `False`; `allow_short_selling: true` on the account only permits shorts, it does not create a
# short sleeve. Verified on the registry: `weights.parquet` of the generation-1 winner
# `1279637ca26c` holds exactly one `+1.0` row per session (1,031 rows, five names, no negative
# weight) and `portfolio_state.parquet` has `net_exposure == gross_exposure` on 100 % of
# sessions. The template `fx_pairs/13_backtest.py` lost `"long_short": bt_config.long_short` in
# commit `3f54bf88` (#590), which rewrote the stage onto the research boundary; the copy
# inherited the omission (`bots/exness_fx_d1/BOT.md`, Decisions log 2026-09-06). **Generation 2
# (this source, user decision 2026-09-06)** sends `"long_short": True` and declares
# `SUPERSEDES_EQUAL_WEIGHT_BASELINES = "b57aa4a93f72"`, so the long-short population supersedes
# the long-only one in the registry lineage: generation 1 stays queryable (its rows are not
# deleted, `superseded_members` reports them retired) and the two sweeps are counted together as
# trials, 1,068 + 1,068 = 2,136, for the Deflated Sharpe Ratio in `BOT.md`.
#
# **The time-series family and the account the engine holds it on.** `top_k = 1` long-short
# books were found to be one-legged on a third of their sessions (`BOT.md` open question 23):
# the pinned `config/backtest/base.yaml` declares `allow_short_selling: true` without
# `allow_leverage`, so the engine runs a cash account and a short is sized by the cash on hand.
# That file is pinned and is not edited here, and no experiment is created with leverage. The
# per-pair family is therefore declared **long only** (`direction: long_only`, `long_short:
# false` in `setup.yaml::backtest.sweep.timeseries_percentile_grid`): gross exposure is at most
# one and net equals gross, which is exactly what a cash account can hold. Running the same
# family a second time as `short_only` would double what it adds to `K` and has not been
# approved. The exposure is *measured* rather than assumed: `_report_phase5.py` reads
# `portfolio_state.parquet` for every member and prints the share of invested sessions with
# `|net| / gross > 0.5` in the results table, and a spec above that declared threshold is
# reported as a directional book, never as a dollar-neutral one.
#
# **`bars_per_day` is not optional.** `signals.py::per_symbol_rolling_percentile_signal` defaults
# to `bars_per_day = 390`, the minute bars of a United States equity session, and computes its
# trailing quantile over `lookback_days * bars_per_day` rows with `min_samples = W // 2`. This
# bot decides once a session, so the value must be `1`, and `setup.yaml` declares it. Left at the
# default the window is 24,570 or 98,280 rows against 1,031 validation sessions, every threshold
# is null, every signal is zero, and the sweep completes having registered a thousand books that
# never traded. Measured before this family was opened, on six representative prediction sets and
# with no backtest run (`_diagnose_timeseries_signal.py`): activation rate 0.0000 at
# `bars_per_day = 390` on every setting, 0.1938 to 0.2653 at `bars_per_day = 1`, with 54 % to
# 75 % of sessions invested and a transition rate of 0.06 to 0.20.
#
# **Generation 3 addendum (2026-09-10, `bots/exness_fx_d1/GEN3_DECLARATION_2026-09-10.md`, user
# decision, OQ 32(a)/(b)).** The two paragraphs above describe generations 1-2, run under the
# account `config/backtest/base.yaml` declared at the time (`allow_short_selling: true` alone, a
# cash account). That file is no longer left untouched: it now also declares `allow_leverage:
# true` (Exness leverage cap, `initial_margin: 0.0005` = 1/2000), so the engine runs a MARGIN
# account and a `top_k = 1` long-short book can hold both legs without the one-legged rotations
# OQ 23 measured. `setup.yaml::execution.share_type` moved from `integer` to `mt5_lot`, which
# gates `case_studies/utils/backtest_runner.py::_apply_lot_floor_if_declared`: every leg below
# the Exness 0.01-lot floor (1,000 units) is rounded down and DROPPED, never bumped up. Both
# changes move every backtest_hash of both signal families (cash + share_type are declared
# spec-hash inputs in `setup.yaml`'s own top comment), so this is a new generation of each
# population, not an edit to the registered rows above: `SUPERSEDES_EQUAL_WEIGHT_BASELINES` now
# names generation 2's tip and `SUPERSEDES_TIMESERIES_BASELINES` now names generation 1's tip
# (parameters cell). The per-pair time-series family stays declared long only for the same reason
# as before (a `short_only` pass would double what it adds to `K` and has not been approved); it
# is re-run only because `share_type` moved, not because its account model changed.
#
# **Learning objectives**
#
# - Freeze the exact prediction population before running a baseline sweep.
# - Send selected catalog rows to the shared engine, priced with the costs `setup.yaml` declares.
# - Verify that every declared prediction and portfolio-size combination produces one backtest.
#
# **Book reference**: Chapter 16
#
# **Prerequisite**: `12_model_analysis`.

# %%
"""Run the complete equal-weight validation backtest population of exness_fx_d1."""

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
from case_studies.utils.backtest_loaders import get_backtest_config
from case_studies.utils.sweep_config import get_top_k_values_for
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "exness_fx_d1"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
LABEL = ""
SPLIT = "validation"
TOP_K = 0
SEED = 42
RUN_SWEEP = True
FORCE_REBACKTEST = False
TOP_N_PREDICTIONS = None
POPULATION_NAME = ""
SIGNAL_FAMILY = "per_pair_timeseries"
# Generation 3 (2026-09-10, bots/exness_fx_d1/GEN3_DECLARATION_2026-09-10.md, OQ 32(a)/(b)):
# config/backtest/base.yaml::account.allow_leverage and setup.yaml::execution.share_type both
# moved, so every backtest_hash of both signal families moves too (cash + share_type are
# spec-hash inputs, setup.yaml top comment). Both tips are re-run under the new engine settings:
# SUPERSEDES_EQUAL_WEIGHT_BASELINES now names generation 2's tip ("52365960d8ca", long-short,
# not generation 1's "b57aa4a93f72" -- the long-only generation stays frozen, superseded,
# untouched, and still counted in K); SUPERSEDES_TIMESERIES_BASELINES now names generation 1's
# tip ("1836e5594858"). Both declared before either fit runs.
SUPERSEDES_EQUAL_WEIGHT_BASELINES: str = "52365960d8ca"
SUPERSEDES_TIMESERIES_BASELINES: str = "1836e5594858"
SUPERSEDES_VALIDATION_PREDICTIONS: str = ""

# %% [markdown]
# ## Select the exact prediction population
#
# Canonical production includes every complete validation prediction the model notebooks currently
# publish. Label, configuration, or population limits are preview controls and cannot define the
# official baseline.
#
# **"Currently publish" is a question about lineage, and the catalog cannot answer it.** A row's
# `identity_status` is derived from the schema version it was written under, so it says the registry
# still understands the row - not that the row is the one its producer stands behind. The two agree
# until a model notebook refits. Then it publishes a second generation of its population under the
# same name, the first generation's prediction sets stay in the registry complete and current, and a
# sweep selecting on the catalog alone runs over both. It would not fail; it would report every
# member complete, over twice the population, and freeze the retired half into the baseline the rest
# of the case study is measured against.
#
# `superseded_members` asks the registry which identities a later generation retired and no
# generation in force still lists, which is exactly the set to drop. Here the GBM population has
# a retired generation (`434954baa5f6`, the snapshot of the stopped eight-thread run, superseded
# by `967e92ff4753` fitted at `num_threads: 2`; `bots/exness_fx_d1/BOT.md`, Decisions log), and
# that generation registered no prediction set, so the count printed below is expected to be 0.
# It is printed rather than left implicit.
#
# **A published population can need a second generation too.** The two names this notebook
# publishes are lists of identities, and the exclusion above changes both of them the moment a
# model notebook refits: the prediction population loses the retired members, and every baseline
# backtest resolved from them goes with it. `OfficialPopulation.create` refuses a changed list
# under an existing name without being told which snapshot it replaces, so each name has its own
# declaration and each is offered through `population_supersedes` on the same rule the model
# notebooks use. `SUPERSEDES_VALIDATION_PREDICTIONS` is empty: the prediction population has one
# generation (`930dcbfa4e57`, the same 534 members generation 1 froze), the two signal families
# below read exactly those members, and the same list under the same name resolves to it. It
# stays empty until the feature matrix changes: the **carry family** (user decision 4) rebuilds
# `features/financial.parquet`, so `06_linear` and `07_gbm` refit and publish prediction sets
# under new identities, and the run that backtests them has to declare
# `SUPERSEDES_VALIDATION_PREDICTIONS = "930dcbfa4e57"` and
# `SUPERSEDES_EQUAL_WEIGHT_BASELINES = "52365960d8ca"` (generation 2, the tip) or it would claim
# a lineage it does not have. That run is blocked on the FRED key (`BOT.md` open question 10).
#
# `SUPERSEDES_EQUAL_WEIGHT_BASELINES` names generation 1 of the cross-sectional baseline
# population (`b57aa4a93f72`, long-only). Generation 2 (long-short, `52365960d8ca`) was published
# by declaring exactly that, so a re-run of this source with `SIGNAL_FAMILY =
# "cross_sectional_top_k"` resolves to generation 2 instead of writing a third one: the tip's own
# `supersedes` equals the declaration, which `population_supersedes` reads as the re-run case. On
# a clean clone the hash is withheld and the run publishes generation 1 of its own.
# `SUPERSEDES_TIMESERIES_BASELINES` is empty because the per-pair family publishes generation 1
# of a population of its own; it is not a generation of the equal-weight chain.

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

# TOP_K and TOP_N_PREDICTIONS both narrow what is backtested, and a narrowed run declares
# a different set of members than the canonical population does. A population is immutable
# once written, so such a run must publish under its own name rather than register a
# partial snapshot under the canonical one. The tier is a separate question: a canonical
# run may legitimately be narrowed, it just may not claim to be the whole population.
if (
    (TOP_K or TOP_N_PREDICTIONS is not None or LABEL)
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
# ### `cross_sectional_top_k`
#
# `top_k` is the number of pairs held in each sleeve. **Generation 2 is long-short**: the signal
# dict carries `"long_short": True`, so `build_target_weights_from_config` holds the `top_k`
# highest-ranked pairs at `+1/top_k` each and the `top_k` lowest-ranked pairs at `-1/top_k` each
# (`signals.py::build_target_weights`, sleeves disjoint, each sleeve capped at `n_assets // 2`);
# the *target* book is dollar-neutral (net exposure zero, gross two) and the account's
# `allow_short_selling: true` is what lets the engine fill the short sleeve.
# `setup.yaml::backtest.sweep.top_k_grid` declares `[1, 2]` for every label
# (`bots/exness_fx_d1/BOT.md`, Decisions log):
#
# - At `top_k = 1`, ranking decides which one pair is held long and which one short; three sit out.
# - At `top_k = 2`, two pairs long at 0.5 each and two short at -0.5 each; the median pair sits out.
#
# The sleeve ceiling of `n_assets // 2 = 2` is checked below so that no `top_k` collapses onto a
# duplicate weight series under a distinct identity. What the engine *held* is a separate
# question from what the signal *targeted*, and open question 23 is that gap: a `top_k = 1`
# long-short target is one-legged on about a third of the sessions on this cash account.
#
# ### `per_pair_timeseries`
#
# The Chapter 16 `08_signal_method_comparison` time-series framing, and the second leg of the
# approved hypothesis (user decision 6, 2026-09-06): a pair is compared with **its own** trailing
# scores rather than with the other four on the same date, so two pairs can both clear their
# threshold on the same session, or neither can.
# `signals.py::per_symbol_rolling_percentile_signal` takes the trailing `long_q` quantile of each
# pair's own score, lagged one session, over a window of `lookback_days * bars_per_day` rows, and
# `_signals_to_equal_weights` splits the book equally over whichever pairs cleared it.
# `setup.yaml::backtest.sweep.timeseries_percentile_grid` declares the settings:
#
# - `bars_per_day: 1` - one decision per session. Not the library default of 390; see the header.
# - `long_q: 0.80` - a pair is held when its score is in its own top fifth, which puts about one
#   pair per session in the book, the same concentration the `top_k = 1` cross-sectional book has.
# - `lookback_days: [63, 252]` - a quarter and a year, the two windows the feature register
#   already uses (`execution.allocator_lookback: 63`; `features.windows.zscore`,
#   `dollar_exposure` and `gold_exposure` are all 252). Two settings, so this family adds the
#   same 1,068 trials the cross-sectional generations each added and the three are comparable.
# - `direction: long_only`, `long_short: false` - the construction a cash account holds exactly
#   (see the header, open question 23). The first `lookback_days // 2` sessions of each pair have
#   a null threshold and no position, which is the rule's declared warmup, not a defect.
#
# The statistics in `BOT.md` are reported on the raw return and on the **active return** over
# the 1/N long book of the five pairs (validation Sharpe 0.41, `BOT.md` phase 5), the benchmark
# the user approved on 2026-09-06.
#
# ### Costs the engine charges
#
# The engine's percentage cost model reads `setup.yaml::costs` through `get_backtest_config`: a
# Pro account pays no commission (`commission_per_lot: 0`), and the slippage leg is the top of the
# measured in-session spread range of the dollar pairs (`spread_bps.major_pairs[-1]`, the p90 of
# AUDUSD), charged on every crossing - the same 2.6 bps round trip `01_feasibility_analysis`
# prices. The swap is a per-night holding cost the engine cannot express per crossing; the demo
# account reports `0.0` on every pair, so nothing is missing in this iteration, and the value
# re-read on the real Pro account enters `16_costs`, never this stage. The check below refuses to
# run the sweep on a cost block that fell through to the generic fallback.

# %% tags=["results"]
setup = yaml.safe_load((get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text())
universe_symbols = setup["universe"]["symbols"]
n_assets = len(universe_symbols)

case_config = get_backtest_config(CASE_STUDY_ID)
declared_costs = setup["costs"]
declared_spread_bps = float(declared_costs["spread_bps"]["major_pairs"][-1])
if case_config.commission_bps != 0.0 or case_config.slippage_bps != declared_spread_bps:
    raise RuntimeError(
        "the engine cost block does not match setup.yaml::costs: "
        f"commission {case_config.commission_bps} bps, slippage {case_config.slippage_bps} bps "
        f"against commission_per_lot {declared_costs['commission_per_lot']} and spread "
        f"{declared_spread_bps} bps"
    )
if any(
    float(v["long"]) != 0.0 or float(v["short"]) != 0.0
    for v in declared_costs["swap"]["points_per_lot_per_night"].values()
):
    raise RuntimeError(
        "a non-zero swap is declared; the engine cannot charge it per crossing, price it in "
        "16_costs before reading this sweep as net of costs"
    )
print(
    f"Engine costs from setup.yaml: commission {case_config.commission_bps:.2f} bps, "
    f"spread {case_config.slippage_bps:.2f} bps per crossing "
    f"({2 * case_config.slippage_bps:.2f} bps round trip), swap 0.0 on every pair "
    f"(measured {declared_costs['swap']['measured_on']} on the demo account), "
    f"fill {case_config.execution_delay}, {case_config.share_type} units"
)
max_sleeve = n_assets // 2

# One signal dict per job, built from `setup.yaml` alone. The dict is what the identity hash is
# taken over, so it is assembled once here and both `plan_backtests` and `run_backtests` are given
# the same object: a key that differed between the two would plan one identity and run another.
BASELINE_POPULATION_SUFFIX = {
    "cross_sectional_top_k": "equal-weight-baselines",
    "per_pair_timeseries": "timeseries-percentile-baselines",
}
if SIGNAL_FAMILY not in BASELINE_POPULATION_SUFFIX:
    raise ValueError(f"unknown SIGNAL_FAMILY {SIGNAL_FAMILY!r}: {sorted(BASELINE_POPULATION_SUFFIX)}")
supersedes_baselines = {
    "cross_sectional_top_k": SUPERSEDES_EQUAL_WEIGHT_BASELINES,
    "per_pair_timeseries": SUPERSEDES_TIMESERIES_BASELINES,
}[SIGNAL_FAMILY]

jobs = []
for label in sorted(catalog.get_column("label").unique()):
    selected = catalog.filter(pl.col("label") == label)
    if SIGNAL_FAMILY == "cross_sectional_top_k":
        top_k_values = [TOP_K] if TOP_K else get_top_k_values_for(CASE_STUDY_ID, label, n_assets)
        duplicates = sorted({value for value in top_k_values if value > max_sleeve})
        if duplicates:
            raise RuntimeError(
                f"top_k {duplicates} exceed the {max_sleeve}-pair sleeve ceiling (half of "
                f"{n_assets} pairs, the rule for the long-short book generation 2 runs); the "
                f"engine would clamp a long-short sleeve to {max_sleeve} and register a duplicate "
                "weight series under a distinct identity"
            )
        settings = [
            {"method": "equal_weight_top_k", "top_k": top_k, "long_short": True}
            for top_k in top_k_values
        ]
    else:
        grid = setup["backtest"]["sweep"]["timeseries_percentile_grid"]
        bars_per_day = int(grid["bars_per_day"])
        if bars_per_day != 1:
            raise RuntimeError(
                f"timeseries_percentile_grid.bars_per_day is {bars_per_day}: this bot takes one "
                "decision per session, and any other value makes min_samples unreachable on the "
                "validation window, so every threshold is null and every book never trades"
            )
        if bool(grid["long_short"]) or str(grid["direction"]) != "long_only":
            raise RuntimeError(
                "the per-pair family is declared long only because the engine holds it on a cash "
                "account (BOT.md open question 23); a short leg needs user approval and doubles K"
            )
        settings = [
            {
                "method": "per_symbol_rolling_percentile",
                "long_q": float(grid["long_q"]),
                "lookback_days": int(lookback),
                "bars_per_day": bars_per_day,
                "long_short": bool(grid["long_short"]),
                "direction": str(grid["direction"]),
            }
            for lookback in grid["lookback_days"]
        ]
    for signal in settings:
        jobs.append(
            {
                "label": label,
                "signal": signal,
                "predictions": selected,
                "expected": selected.height,
            }
        )

print(f"Signal family: {SIGNAL_FAMILY}; {len(jobs)} jobs over {catalog.height} prediction sets")
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
    raise RuntimeError("equal-weight baseline runs must register with stage='signal'")

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
# - The baseline evaluates every complete model configuration and checkpoint at every declared
#   setting; the number of members is what the Deflated Sharpe Ratio in `BOT.md` divides by.
# - One stage, two signal families, two populations. `cross_sectional_top_k` publishes
#   `equal-weight-baselines`, whose generation 2 is the **long-short** sleeve book
#   (`"long_short": True`) and whose generation 1 (`b57aa4a93f72`) was a long-only top-k rotation
#   kept as the superseded generation. `per_pair_timeseries` publishes generation 1 of
#   `timeseries-percentile-baselines`, a long-only book read per pair against its own history.
#   Every family counts toward the same cumulative `K` for the bot.
# - The registry column `avg_turnover` of the 1,068 generation-1 equal-weight rows is wrong
#   (shared-runner defect fixed after that sweep; `BOT.md` open question 16) - read the ledger
#   for those; every row registered after the fix carries the corrected metric.
# - Catalog rows pass directly to the shared engine on the MT5 session panel, at the costs
#   `setup.yaml` declares.
# - Immutable population membership makes missing or failed jobs visible.
# - Selection happens on this validation split only; the holdout is scored once, later.
