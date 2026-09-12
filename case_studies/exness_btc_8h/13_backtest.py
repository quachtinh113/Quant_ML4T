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
# # Session-Book Backtest - Exness BTC 8h
#
# **Docker image**: `ml4t`
#
# The bot's `13_backtest` (bot `exness_btc_8h`, roadmap phase 5). Forked from
# `case_studies/exness_gold_sess/13_backtest.py` (`bots/exness_btc_8h/BOT.md` Decisions log
# 2026-09-10) for its frame - freeze the prediction population, plan every backtest identity
# before running one, run them, then `require_complete` - and simplified around the two things
# that are structurally DIFFERENT here, both already declared in `config/setup.yaml` before this
# file existed:
#
# **1. The price grid IS the decision grid, so no hold has to be derived from a finer tape.**
# `setup.yaml::backtest.price_grid_expresses_primary_label: true` - the entry and the exit of a
# `fwd_ret_8h` trade are two CONSECUTIVE rows of the 8-hour grid, and for `fwd_ret_24h` they are
# three rows apart. `exness_gold_sess` derives its hold by counting H1 rows inside an 8-hour
# session window (`_hold.py::derive_hold_bars`) because its price grid is finer than its label
# horizon; that derivation does not apply here; `HOLD_BARS[label]` is simply
# `setup.yaml::labels.rebalance_step[label]` - 1 bar for `fwd_ret_8h`, 3 for `fwd_ret_24h` - and it
# is the SAME for every book, not per-book like the gold sibling's asymmetric London/New-York
# exit.
#
# **2. Three session books, ALL of them engine-run - no arithmetic pooled sleeve.** The broker
# holds one net position per symbol, which is what forces `exness_gold_sess`'s `both` book to be
# an arithmetic 50/50 sum built outside the engine (a London order landing inside a New York hold
# would adjust one broker position rather than open a second). This bot's `all` book is not a
# combination of two other books - it is the untouched, single-venue decision grid itself - so it
# is a REAL engine run like `no_swap_night` and `us_hours`, and no pooled-sleeve arithmetic is
# declared (`setup.yaml::backtest.sweep` carries no `pooled_book` key, unlike the gold sibling).
#
# **`session_filter` has to move rows, not only the hash** (same reasoning as the gold sibling):
# `build_target_weights_from_config` ignores keys it does not know, so a book name in the signal
# dict alone would register three hashes carrying one result. The rows are filtered by
# `backtest_runner::apply_session_filter`, called from `run_backtest` right after
# `apply_universe_filter`.
#
# ## TWO BLOCKING SHARED-LIBRARY DEPENDENCIES, NEITHER APPLIED HERE
#
# This notebook is FORKED AND NOT RUN (`bots/exness_btc_8h/BOT.md` phase status, 2026-09-10). Two
# calls it makes will raise today, in `case_studies/utils/*`, which this bot's writer may not
# edit without the mentor's review (a diff is PROPOSED in `bots/exness_btc_8h/BOT.md` Open
# questions, not applied):
#
# 1. **`backtest_runner.py::apply_session_filter` is hard-coded to `case_study ==
#    "exness_gold_sess"`** and raises `ValueError` for any other case study, by design
#    (`"a new case study must register its own session resolver here rather than inherit this
#    one"`). The proposed diff extends the check to `exness_btc_8h`, treats `book == "all"` as an
#    EXPLICIT no-op (mentor gate #4, 2026-09-10: `session_book_of` never returns `"all"`, so
#    matching it against that value would filter every row out and raise), and dispatches to
#    `case_studies.exness_btc_8h._features.session_book_of` for `"no_swap_night"` / `"us_hours"`.
# 2. **`backtest_loaders.py::_PRICE_CONFIG` has no `"exness_btc_8h"` entry** (BOT.md open question
#    5), so `load_backtest_prices_for` cannot resolve a price loader for this case study yet. The
#    grid to register is the 8-hour decision grid itself, keyed on the bar close - the same
#    parquet `02_labels` already reads - which is a much smaller registration than the gold
#    sibling's dedicated H1 loader.
#
# **Learning objectives**
#
# - Freeze the exact prediction population before running a session-book sweep.
# - Put the hold, the book and the signal spec inside the identity, then prove each one moves rows.
# - Verify that every declared (label x prediction x signal x book) pair produces one backtest.
#
# **Book reference**: Chapter 16
#
# **Prerequisite**: `12_model_analysis` (which itself gates on R1 - see that notebook).

# %%
"""Run the complete validation session-book backtest population of exness_btc_8h."""

import polars as pl
import yaml

from case_studies.exness_btc_8h._features import session_book_of
from case_studies.research import (
    OfficialPopulation,
    PredictionResult,
    Result,
    open_study,
    plan_backtests,
    population_supersedes,
    research_name,
    run_backtests,
    superseded_members,
)
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
)
from case_studies.utils.backtest_runner import apply_session_filter
from case_studies.utils.registry.store import _infer_stage
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "exness_btc_8h"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
LABEL = ""
SPLIT = "validation"
SEED = 42
RUN_SWEEP = True
FORCE_REBACKTEST = False
TOP_N_PREDICTIONS = None
POPULATION_NAME = ""
SESSION_BOOKS: list[str] = []
SUPERSEDES_VALIDATION_PREDICTIONS: str = ""
SUPERSEDES_SESSION_BOOK_BASELINES: str = ""

# %% [markdown]
# ## The grid this stage is allowed to run on

# %% tags=["results"]
set_global_seeds(SEED)
if SPLIT != "validation":
    raise ValueError("the baseline sweep uses validation predictions")
if FORCE_REBACKTEST:
    raise ValueError("identical complete backtests are reused by identity")
if not RUN_SWEEP:
    raise ValueError("set RUN_SWEEP=True to execute the visible baseline request")

setup = yaml.safe_load((get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text())
if not setup["backtest"]["price_grid_expresses_primary_label"]:
    raise RuntimeError(
        "setup.yaml::backtest.price_grid_expresses_primary_label is false: the registered grid "
        "does not express the label horizon and no sweep may run on it"
    )
KILL_CRITERIA = dict(setup.get("kill_criteria") or {})
PRIMARY_LABEL = setup["labels"]["primary"]
LABEL_HORIZON_HOURS = {
    name: int(str(hours).rstrip("Hh")) for name, hours in setup["labels"]["horizons"].items()
}
# HOLD_BARS is the rebalance step, in decision slots - NOT derived from a finer price tape, unlike
# the gold sibling. `fwd_ret_8h` trades one slot (entry and exit are the same grid, 8 hours apart);
# `fwd_ret_24h` trades three (24 hours = 3 x 8h slots). Same for every book: there is one venue.
HOLD_BARS = {
    label: int(step) for label, step in setup["labels"]["rebalance_step"].items()
}
if HOLD_BARS.get(PRIMARY_LABEL) != 1:
    raise RuntimeError(
        f"HOLD_BARS[{PRIMARY_LABEL!r}] = {HOLD_BARS.get(PRIMARY_LABEL)}, not 1; the primary "
        "label's entry and exit are declared as adjacent grid rows and a different value here "
        "means setup.yaml::labels.rebalance_step moved without this stage being reviewed"
    )

# %% [markdown]
# ## R1 must have cleared before this stage may run
#
# `12_model_analysis` writes the R1 verdict to `evaluation/r1_early_close_verdict.json` and
# raises before its own handoff section if the generation is closed. This stage reads that
# artifact independently and refuses to proceed if it is missing (12 was not run) or if it
# records a closed generation - a second gate is cheaper than a shared assumption.

# %% tags=["results"]
r1_path = get_case_study_dir(CASE_STUDY_ID) / "evaluation" / "r1_early_close_verdict.json"
if not r1_path.exists():
    raise RuntimeError(
        f"{r1_path} does not exist; run 12_model_analysis first (it writes the R1 verdict and "
        "raises before its own handoff if R1 closes the generation, which stops it here too)"
    )
import json  # noqa: E402

r1_verdict = json.loads(r1_path.read_text())
if r1_verdict.get("generation_closed"):
    raise RuntimeError(
        f"R1 closed this generation ({r1_verdict}); 13_backtest must not run. See "
        "bots/exness_btc_8h/BOT.md Kill criteria, R1."
    )
print(f"R1 cleared: {r1_verdict}")

# %% [markdown]
# ## BLOCKED: price loader not yet registered
#
# `load_backtest_prices_for` resolves through `backtest_loaders.py::_PRICE_CONFIG`, which has no
# `"exness_btc_8h"` entry (see the module docstring, blocking dependency 2). This cell will raise
# `KeyError` (or equivalent) until that registration exists; it is left in place, uncommented,
# because the point of forking this stage today is to have the exact call that will run once the
# registration lands - not a stub that has to be rewritten later.

# %% tags=["results"]
grid = load_backtest_prices_for(CASE_STUDY_ID, PRIMARY_LABEL, split="validation")
spacing = (
    grid.sort(["symbol", "timestamp"])
    .with_columns(
        (
            (
                pl.col("timestamp").shift(-1).over("symbol").dt.epoch("s")
                - pl.col("timestamp").dt.epoch("s")
            )
            // 60
        ).alias("gap_min")
    )
    .drop_nulls("gap_min")
)
modal_gap = int(spacing.group_by("gap_min").len().sort("len", descending=True)["gap_min"][0])
if modal_gap != 8 * 60:
    raise RuntimeError(
        f"the registered price grid has a modal spacing of {modal_gap} minutes, not 480 (8h): "
        "this stage requires the native 8-hour decision grid"
    )
print(f"price grid modal spacing {modal_gap} min, {grid.height} rows")

# %% [markdown]
# ## Select the exact prediction population

# %% tags=["results"]
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
include_preview = EXECUTION_TIER == "preview"
catalog = study.predictions.table(include_preview=include_preview).filter(
    (pl.col("identity_status") == "current") & (pl.col("split") == SPLIT) & pl.col("complete")
)
catalog = catalog.filter(
    pl.col("execution_tier") == ("preview" if include_preview else "canonical")
)
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
if (
    (TOP_N_PREDICTIONS is not None or LABEL or SESSION_BOOKS)
    and not include_preview
    and not POPULATION_NAME
):
    raise ValueError(
        "this run narrows the declared sweep, so it cannot publish the canonical population; "
        "pass POPULATION_NAME to give it its own"
    )
if catalog.get_column("prediction_hash").n_unique() != catalog.height:
    raise RuntimeError("the baseline population contains duplicate prediction identities")

EXPECTED_PREDICTION_SETS = int(KILL_CRITERIA.get("r1_expected_prediction_sets", 356))
if not (LABEL or TOP_N_PREDICTIONS) and catalog.height != EXPECTED_PREDICTION_SETS:
    print(
        f"  NOTE: catalog resolves {catalog.height} prediction sets, not the declared "
        f"{EXPECTED_PREDICTION_SETS}; K5 below must be read against what actually resolves, and "
        "the difference belongs in bots/exness_btc_8h/BOT.md before any Sharpe is read."
    )

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
    "label", "family", "config_name", "checkpoint_kind", "checkpoint_value", "prediction_hash"
)

# %% [markdown]
# ## The books move rows, and `bars_per_day` is measured per book
#
# `apply_session_filter` is exercised on the decision instants of the sealed primary label. The
# three books must be **non-empty**; `no_swap_night` and `us_hours` must be **disjoint** and
# together must **exhaust** `all` (`setup.yaml::decision.session_filter_definitions`).
#
# `signal_specs[1]` (`per_symbol_p80`) declares `bars_per_day: null` on purpose
# (`setup.yaml::backtest.sweep.signal_specs`): `signals.py:377` computes the rolling-percentile
# window as `int(lookback_days * bars_per_day)` over ROWS OF THE FILTERED PREDICTION FRAME, and a
# single value cannot be right on all three books - `all` resolves 3 rows/symbol/day,
# `no_swap_night` 2, `us_hours` 1 - so it is measured here, per book, and asserted rather than
# typed (the `nasdaq100_microstructure` failure `setup.yaml:632-641` names: "the values did not
# change; what they count did").

# %% tags=["results"]
DECLARED_BOOKS = list(setup["backtest"]["sweep"]["session_books"])
books = SESSION_BOOKS or DECLARED_BOOKS
unknown_books = sorted(set(books) - set(setup["decision"]["session_filter_values"]))
if unknown_books:
    raise ValueError(f"undeclared session books {unknown_books}")
if "pooled_book" in setup["backtest"]["sweep"]:
    raise RuntimeError(
        "setup.yaml declares a pooled_book, but this bot's design has no arithmetic pooled "
        "sleeve (all three books are engine-run); this stage was not written for one"
    )

probe = pl.read_parquet(get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{PRIMARY_LABEL}.parquet")
probe_dev = probe.filter(pl.col("timestamp") < setup["evaluation"]["holdout_start"])
book_frames: dict[str, pl.DataFrame] = {}
for book in books:
    # `session_filter` is PASSED for every book, including "all" - fleet mentor gate #4
    # (2026-09-10): omitting the key for "all" and relying on apply_session_filter's generic
    # "empty session_filter is a no-op" short-circuit never exercises the "all" branch the
    # diff registers, and leaves a spec whose hash cannot be told apart from one that never
    # declared a book at all. The shared diff (proposed, not applied - BOT.md Open question 5)
    # must treat "all" as an EXPLICIT no-op: `case_studies.exness_btc_8h._features.
    # session_book_of` never returns "all" by design (it is a two-way partition,
    # no_swap_night/us_hours), so a resolver that tried to match book == "all" against it would
    # filter every row out and raise at backtest_runner.py:888-893 - the exact bug the gate
    # caught before this stage ever ran.
    book_frames[book] = apply_session_filter(probe_dev, CASE_STUDY_ID, {"session_filter": book})
    if book_frames[book].is_empty():
        raise RuntimeError(f"session book {book!r} selected zero rows of the primary label")

partitioned = [book_frames[b] for b in books if b != "all"]
if len(partitioned) >= 2:
    union_rows = pl.concat([f.select("timestamp", "symbol") for f in partitioned]).unique()
    if union_rows.height != book_frames["all"].select("timestamp", "symbol").unique().height:
        raise RuntimeError(
            "the non-'all' books do not exhaust 'all': "
            f"{union_rows.height} != {book_frames['all'].height}"
        )
    pairwise = partitioned[0].join(partitioned[1], on=["timestamp", "symbol"], how="inner")
    if pairwise.height:
        raise RuntimeError(f"session books overlap on {pairwise.height} rows; they must be disjoint")

BARS_PER_DAY: dict[str, float] = {}
for book, frame in book_frames.items():
    per_day = (
        frame.with_columns(pl.col("timestamp").dt.date().alias("_day"))
        .group_by("symbol", "_day")
        .agg(pl.len().alias("rows"))
        .group_by("rows")
        .len()
        .sort("len", descending=True)
    )
    BARS_PER_DAY[book] = float(per_day["rows"][0])
print(f"Measured bars_per_day per book: {BARS_PER_DAY}")
EXPECTED_BARS_PER_DAY = {"all": 3.0, "no_swap_night": 2.0, "us_hours": 1.0}
mismatched_bpd = {b: BARS_PER_DAY[b] for b in books if BARS_PER_DAY.get(b) != EXPECTED_BARS_PER_DAY.get(b)}
if mismatched_bpd:
    raise RuntimeError(
        f"measured bars_per_day {mismatched_bpd} does not match the declared expectation "
        f"{ {b: EXPECTED_BARS_PER_DAY[b] for b in mismatched_bpd} } "
        "(setup.yaml::backtest.sweep.signal_specs comment); the session grid moved"
    )

# %% [markdown]
# ## Build the session-book strategy grid
#
# One job per (label x session book x signal spec). `long_short: True` because
# `mapping.position_state_space` is `long_short` and `build_target_weights_from_config` defaults
# it to `False`. `session_filter` is set to `book` for EVERY book, including `"all"`
# (fleet mentor gate #4, 2026-09-10 - see BOT.md Open question 5's rewritten diff): the shared
# `apply_session_filter` must treat `book == "all"` as an EXPLICIT no-op, because
# `case_studies.exness_btc_8h._features.session_book_of` never returns `"all"` (it is a
# two-way partition, `no_swap_night` / `us_hours`) and matching `"all"` against it would filter
# every row out and raise at `backtest_runner.py:888-893`. The earlier draft of this stage
# omitted the key for `"all"` and relied on `apply_session_filter`'s generic "empty
# `session_filter` is a no-op" short-circuit - which never exercises the dispatch branch the
# diff adds, and leaves a registered spec that cannot be told apart from one that never
# declared a book at all.
#
# **Cost check first.**

# %% tags=["results"]
case_config = get_backtest_config(CASE_STUDY_ID)
declared_costs = setup["costs"]
declared_spread_bps = float(declared_costs["spread_bps"]["crypto"][-1])
if case_config.commission_bps != 0.0 or case_config.slippage_bps != declared_spread_bps:
    raise RuntimeError(
        "the engine cost block does not match setup.yaml::costs: "
        f"commission {case_config.commission_bps} bps, slippage {case_config.slippage_bps} bps "
        f"against commission_per_lot {declared_costs['commission_per_lot']} and spread "
        f"{declared_spread_bps} bps"
    )
print(
    f"Engine costs from setup.yaml: commission {case_config.commission_bps:.2f} bps, "
    f"spread {case_config.slippage_bps:.2f} bps per crossing (in-sample range; swap is priced in "
    f"16_costs, not here), fill {case_config.execution_delay}, {case_config.share_type} units"
)

jobs = []
for label in sorted(catalog.get_column("label").unique()):
    selected = catalog.filter(pl.col("label") == label)
    hold = HOLD_BARS[label]
    risk = {"position_rules": [{"type": "time_exit", "bars": hold}]}
    for book in books:
        for spec in setup["backtest"]["sweep"]["signal_specs"]:
            signal = {k: v for k, v in spec.items() if k != "name"}
            signal["long_short"] = True
            # ALWAYS set session_filter, including for "all" - fleet mentor gate #4, same
            # reasoning as the probe loop above: "all" is an explicit no-op declared in the
            # spec, not an absent key relying on a generic short-circuit.
            signal["session_filter"] = book
            if signal.get("method") == "per_symbol_rolling_percentile":
                signal["bars_per_day"] = BARS_PER_DAY[book]
            jobs.append(
                {
                    "label": label,
                    "book": book,
                    "spec_name": spec["name"],
                    "signal": signal,
                    "risk": risk,
                    "predictions": selected,
                    "expected": selected.height,
                }
            )

expected_total = sum(job["expected"] for job in jobs)
K_THIS_RUN = expected_total
K_DECLARED = EXPECTED_PREDICTION_SETS * len(DECLARED_BOOKS) * len(setup["backtest"]["sweep"]["signal_specs"])
print(
    f"{len(jobs)} jobs over {catalog.height} prediction sets = {expected_total} DECLARED ENGINE "
    f"backtests ({len(books)} books x {len(setup['backtest']['sweep']['signal_specs'])} signal "
    "specs x the prediction sets of each label). No arithmetic pooled sleeve exists on this bot, "
    f"so K THIS RUN = expected_total = {K_THIS_RUN}. THE GATE DIVIDES BY THE DECLARED "
    f"K5 = {K_DECLARED} ({EXPECTED_PREDICTION_SETS} x {len(DECLARED_BOOKS)} x "
    f"{len(setup['backtest']['sweep']['signal_specs'])}), fixed before the first model was fitted."
)
if K_THIS_RUN != K_DECLARED and not (LABEL or TOP_N_PREDICTIONS):
    print(
        f"  NOTE: this run resolves K = {K_THIS_RUN}, not the declared {K_DECLARED}. The "
        "declared value stands and the difference has to be explained in BOT.md before any "
        "Deflated Sharpe Ratio is quoted."
    )
pl.DataFrame(
    [
        {
            "label": job["label"],
            "book": job["book"],
            "signal_spec": job["spec_name"],
            "hold_bars": job["risk"]["position_rules"][0]["bars"],
            "prediction_sets": job["expected"],
        }
        for job in jobs
    ]
)

# %% [markdown]
# ## Plan-time exclusion of a constant score under a rolling-percentile spec
#
# `rolling_percentile_signal` compares a score to a rolling percentile of itself
# (`case_studies/utils/signals.py:63-120`), so a prediction set whose score is constant inside a
# book can never exceed its own percentile: no weight is emitted, and
# `_refuse_an_allocation_that_produced_no_target` refuses to register a flat account as Sharpe
# 0.0 - the same mechanism and the same reason `exness_gold_sess` declared its own exclusion
# predicate (`RULINGS_2026-09-08.md` addendum, section 4). Unlike that sibling, this bot has not
# yet measured how many of its members are degenerate on which book - IT HAS NOT BEEN RUN - so no
# expected count is declared here. **The predicate below therefore RAISES rather than silently
# excludes**: the first time this stage actually runs, whoever runs it must read the printed list,
# measure the count, and declare it (mirroring the sibling's `UNRUNNABLE_ENGINE_EXPECTED`) before
# it can proceed past a non-zero exclusion. `fixed_threshold_0` is unaffected - a constant score
# is a real, runnable, permanently-short-or-long book under that spec, not a degenerate one.

# %% tags=["results"]
PERCENTILE_METHOD = "per_symbol_rolling_percentile"


def _score_series(frame: pl.DataFrame) -> pl.Series:
    for column in ("y_score", "prediction"):
        if column in frame.columns:
            return frame.get_column(column)
    raise RuntimeError(f"a prediction frame carries no score column: {frame.columns}")


score_frames: dict[str, pl.DataFrame] = {}
degenerate: list[dict] = []
for job in jobs:
    if job["signal"].get("method") != PERCENTILE_METHOD:
        continue
    for row in job["predictions"].iter_rows(named=True):
        prediction_hash = row["prediction_hash"]
        if prediction_hash not in score_frames:
            result = Result.open(study, prediction_hash)
            if not isinstance(result, PredictionResult):
                raise TypeError(f"{prediction_hash} is not a prediction result")
            score_frames[prediction_hash] = result.load()
        in_book = apply_session_filter(score_frames[prediction_hash], CASE_STUDY_ID, job["signal"])
        scores = _score_series(in_book)
        if int(scores.n_unique()) <= 1:
            degenerate.append(
                {
                    "prediction_hash": prediction_hash,
                    "label": job["label"],
                    "config_name": row.get("config_name"),
                    "book": job["book"],
                    "signal_spec": job["spec_name"],
                    "rows_in_book": in_book.height,
                }
            )
if degenerate:
    for entry in degenerate:
        print(f"  DEGENERATE UNDER {PERCENTILE_METHOD!r}: {entry}")
    raise RuntimeError(
        f"{len(degenerate)} prediction set(s) score constant under {PERCENTILE_METHOD!r} on at "
        "least one book; this is the first time the count is known. Declare it explicitly "
        "(UNRUNNABLE_ENGINE_EXPECTED-style, see exness_gold_sess/13_backtest.py) with today's "
        "date and re-run - K does NOT move, the identity is named and counted in K, only removed "
        "from the runnable plan."
    )
print(f"No degenerate {PERCENTILE_METHOD!r} member found over {len(score_frames)} prediction sets read.")

# %% [markdown]
# ## Freeze the expected population

# %% tags=["results"]
planned_hashes = []
for job in jobs:
    plan = plan_backtests(
        study,
        predictions=job["predictions"],
        signal=job["signal"],
        risk=job["risk"],
        chapter="16",
    )
    if len(plan.members) != job["expected"]:
        raise RuntimeError("a plan omitted a selected prediction")
    planned_hashes.extend(plan.expected_hashes)

if len(planned_hashes) != len(set(planned_hashes)):
    raise RuntimeError(
        "two planned jobs collapse to the same backtest identity: the session book or the hold "
        "did not reach the spec"
    )

baseline_population = None
if not include_preview:
    baselines_name = research_name(CASE_STUDY_ID, "session-book-baselines", scope=POPULATION_NAME)
    baseline_population = OfficialPopulation.create(
        study,
        name=baselines_name,
        member_kind="backtest",
        members=planned_hashes,
        supersedes=population_supersedes(
            study, name=baselines_name, declared=SUPERSEDES_SESSION_BOOK_BASELINES
        ),
    )
    print(f"Frozen expected baseline population: {baseline_population.hash}")
else:
    print("Preview backtests remain outside official populations and selection.")

# %% [markdown]
# ## Run every job through the engine, then validate what was frozen
#
# No exception-and-continue path: one failed member leaves the predeclared population incomplete
# and stops publication. The post-run guard reads the CONTENT of every registered spec (the
# derived hold, the session filter), not a stage label copied from a sibling bot.

# %% tags=["results"]
run_status: list[str] = []
backtests = []
job_by_hash: dict[str, dict] = {}
for job in jobs:
    execution = run_backtests(
        study,
        predictions=job["predictions"],
        signal=job["signal"],
        risk=job["risk"],
        chapter="16",
    )
    if len(execution.results) != job["expected"]:
        raise RuntimeError("a member disappeared during execution")
    backtests.extend(execution.results)
    for result in execution.results:
        job_by_hash[result.hash] = job
    run_status.extend(entry["status"] for entry in execution.diagnostics)

if len(backtests) != expected_total:
    raise RuntimeError(f"expected {expected_total} runs, found {len(backtests)}")
if {result.hash for result in backtests} != set(planned_hashes):
    raise RuntimeError("completed identities differ from the frozen plan")
if any(not result.complete for result in backtests):
    raise RuntimeError("the population contains an incomplete backtest")

backtest_row_records: list[dict] = []
stage_labels: set[str] = set()
for result in backtests:
    job = job_by_hash[result.hash]
    record = result.registry_record()
    strategy = result.spec().get("strategy", {})
    expected_risk = {"position_rules": [{"type": "time_exit", "bars": HOLD_BARS[job["label"]]}]}
    if strategy.get("risk") != expected_risk:
        raise RuntimeError(
            f"{result.hash}: the registered risk block {strategy.get('risk')!r} is not the "
            f"derived hold {expected_risk!r} of {job['label']}"
        )
    session_filter = strategy.get("signal", {}).get("session_filter")
    # "all" is registered EXPLICITLY in the spec (fleet mentor gate #4) - not None.
    expected_filter = job["book"]
    if session_filter != expected_filter:
        raise RuntimeError(
            f"{result.hash}: the registered session_filter {session_filter!r} is not "
            f"{expected_filter!r}, the book {job['book']!r} of its job"
        )
    stage_labels.add(record["stage"])
    backtest_row_records.append(
        {
            "backtest_hash": result.hash,
            "prediction_hash": record["prediction_hash"],
            "stage": record["stage"],
            "complete": result.complete,
        }
    )
backtest_rows = pl.DataFrame(backtest_row_records)

if len(stage_labels) != 1:
    raise RuntimeError(
        f"the registry gave the baseline population {len(stage_labels)} stage labels "
        f"{sorted(stage_labels)}; one population must carry exactly one"
    )
registered_stage = next(iter(stage_labels))
inferred_stage = _infer_stage(backtests[0].spec())
if registered_stage != inferred_stage:
    raise RuntimeError(
        f"the registry labelled the baseline rows {registered_stage!r} but the library infers "
        f"{inferred_stage!r} from the same content - read case_studies/utils/registry/store.py "
        "and the gold sibling's D5.2 amendment (PHASE567_DECLARATION.md) before assuming which "
        "one is right; this bot's baseline specs also carry a risk block (the hold), same as the "
        "gold sibling's, so 'risk_overlay' rather than 'signal' is the expected label"
    )
print(f"Registry stage label on all {len(backtests):,} baseline rows: '{registered_stage}'")

served = run_status.count("reused")
print(
    f"Session-book baselines: {len(backtests) - served} computed, {served} served from the "
    f"registry, {len(backtests)} in the population"
)

if not include_preview:
    if baseline_population is None:
        raise RuntimeError("the canonical population was not frozen before execution")
    baseline_population.require_complete()
    print(f"Official {baselines_name} population: {baseline_population.hash}")

backtest_rows

# %% [markdown]
# ## Key takeaways
#
# - The grid IS the decision grid; the hold is `labels.rebalance_step[label]` decision slots, the
#   same for every book, expressed as a `TimeExit` position rule inside every spec.
# - `all`, `no_swap_night` and `us_hours` are all THREE engine books; `no_swap_night` and
#   `us_hours` are disjoint and exhaust `all`. No arithmetic pooled sleeve exists or is declared.
# - `bars_per_day` for `per_symbol_p80` is measured per book (3 / 2 / 1) and asserted, not typed.
# - Swap is NOT in the percentage cost model above; it is priced in `16_costs`
#   (`setup.yaml::costs.swap`).
# - Selection happens on this validation split only; the holdout is scored once, later.
# - **This stage has never been run.** The exclusion predicate above will raise the first time it
#   is, until someone measures the degenerate count and declares it explicitly.
