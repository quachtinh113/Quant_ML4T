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
# # Session-Book Backtest - Exness Gold Sessions
#
# **Docker image**: `ml4t`
#
# The bot's `13_backtest` (bot `exness_gold_sess`, roadmap phase 5). Forked from
# `case_studies/exness_fx_d1/13_backtest.py` for its frame - freeze the prediction population,
# plan every backtest identity before running one, run them, then `require_complete` - and
# rewritten around the three things that are different here, every one of them declared in
# `bots/exness_gold_sess/PRICE_GRID_DECLARATION.md` before this file existed.
#
# **1. The price grid is the raw H1 tape, keyed on the bar close.**
# `_PRICE_CONFIG["exness_gold_sess"]` names the loader `exness_gold_sess_h1`: every hourly bar of
# both metals over the declared window, `timestamp` shifted by +60 minutes so it names the instant
# the bar *closed*. Every decision instant of this bot is a bar close, so the decision grid is a
# **subset** of the price grid and predictions stay at session cadence - no as-of join, no
# invented session-close prediction row. Sessions are **not** cropped out of the grid: the bars
# between entry and exit are what the Ch19 stop and trailing rules read, and not trading outside a
# session is guaranteed by the rebalance schedule, which is built from *prediction* timestamps
# (`backtest_runner.py:1408-1410`), never by deleting price rows.
#
# **2. The hold is a broker position rule, not the grid spacing and not `slot_strategy`.**
# `slot_strategy` expresses an exit by ceasing to emit a weight row, and the engine only acts at
# `timestamp in rebalance_schedule` - a subset of the prediction timestamps. This bot has no
# prediction at 17:00, so that exit would never execute (nasdaq100_microstructure gets away with it
# because its prediction panel is fifteen minutes dense). `TimeExit(max_bars=N)` is set once on the
# first bar (`backtest_runner.py:1473-1476`), evaluated on the position's own `bars_held`, and is
# the only mechanism in this repository that closes a position at an instant that is not a decision
# instant. So **every** spec below - the Chapter 16 baselines too, not only the Chapter 19 risk
# arms - carries `risk={"position_rules": [{"type": "time_exit", "bars": HOLD_BARS[label, book]}]}`,
# and because `risk` is part of the spec the hold is inside `backtest_hash` rather than an implicit
# default.
#
# **The hold is PER BOOK, and both numbers are derived rather than typed**
# (`bots/exness_gold_sess/NY_EXIT_DECLARATION.md` section 1.2, which corrects sections 3.2 and 3.3
# of `PRICE_GRID_DECLARATION.md`). The rule is one rule for both venues -
# `case_studies/exness_gold_sess/_hold.py::derive_hold_bars`, imported by this stage and by
# `bots/exness_gold_sess/tests/test_backtest_grid.py` so the two cannot disagree:
#
#     HOLD_BARS[book] = i1 - i0, where i0 indexes the first grid row keyed AFTER the decision (the
#     row the entry fills at) and i1 indexes the last grid row whose fill instant `key - 60min` is
#     at or before the sealed `label_end_ts`.
#
# It falls out as **london 8** (480 minutes, exit fill exactly on the endpoint) and **ny 7** (420
# minutes, exit fill 60 minutes before it). The asymmetry is in the CONSTANT, not in the rule: the
# New York session close *is* the start of the broker's daily one-hour break in both seasons -
# both are anchored on 17:00 America/New_York (`bots/_shared/sessions.py`, `bots/assets/XAUUSD.md`
# :110,188), summer 21:00-22:00 UTC and winter 22:00-23:00 UTC - so the row that would fill at the
# New York endpoint does not exist, in either season, and an 8-bar hold fills at the first print
# AFTER the break (60 minutes late on a weekday, Sunday 22:00 UTC after a Friday). That is kill
# criterion (g) of this bot firing on 100 % of New York positions by design, which is why 8 is
# rejected there. The price, declared: the New York book trains on an 8-hour label and trades a
# 7-hour window - `setup.yaml::backtest.exit_gap_minutes_by_book`.
#
# **3. Two engine books, and the pooled book is arithmetic.** The broker holds one net position per
# symbol, so on a pooled book a New York order at 14:00 lands inside a London hold, the position is
# adjusted rather than opened, `bars_held` keeps counting from the London entry and `TimeExit`
# closes *everything* at 17:00 - killing the New York leg after three hours. `london` and `ny` are
# therefore the only engine books; `both` is a deterministic 50/50 sum of the two registered return
# series built in `19_strategy_analysis` (`setup.yaml::backtest.sweep.pooled_book`). It is exact,
# conservative on cost, and does not change `K`.
#
# **`session_filter` has to move rows, not only the hash.**
# `build_target_weights_from_config` dispatches on `config["method"]` and ignores keys it does not
# know, so a book name in the signal dict alone would register two hashes carrying one result -
# "a new hash is a cache key, not proof an override reached the model". The rows are filtered by
# `backtest_runner::apply_session_filter`, called from `run_backtest` right after
# `apply_universe_filter` and driven purely by `strategy.signal.session_filter`, so the sweep,
# `16_costs` and the holdout producer all filter identically. This notebook **measures** the split
# before it plans anything and refuses to continue if the two books are not disjoint and non-empty.
#
# **Learning objectives**
#
# - Freeze the exact prediction population before running a session-book sweep.
# - Put the hold, the book and the weekday rule inside the identity, then prove each one moves rows.
# - Verify that every declared (label x prediction x signal x book) pair produces one backtest.
#
# **Book reference**: Chapter 16
#
# **Prerequisite**: `12_model_analysis`.

# %%
"""Run the complete validation session-book backtest population of exness_gold_sess."""

import polars as pl
import yaml

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_gold_sess._features import session_panel
from case_studies.exness_gold_sess._hold import (
    DERIVATION_KEY,
    assert_no_holdout,
    derive_backstop_bars,
    derive_hold_bars,
    development_window,
    exit_gap_minutes_by_book,
)
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
    get_rebalance_step,
    load_backtest_prices,
    load_backtest_prices_for,
    resolve_decision_schedule,
)
from case_studies.utils.backtest_runner import apply_session_filter
from case_studies.utils.registry.store import _infer_stage
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "exness_gold_sess"
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
#
# Read before anything else, because every number below depends on it and because the registered
# grid changed on 2026-09-08. Three facts are asserted rather than assumed:
#
# 1. the price frame is **hourly**, not two rows a weekday - the modal spacing is 60 minutes;
# 2. it is keyed on the **bar close**, which is checked where it is checkable: every decision
#    instant of the sealed primary label is a row of the grid. On a grid keyed on the bar open the
#    decision instants would fall between rows and this check would fail on every row;
# 3. `setup.yaml` says so too (`backtest.price_grid_expresses_primary_label`), so a future reader
#    finds the claim and the check in the same place.

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
        "does not express the label horizon and no sweep may run on it "
        "(bots/exness_gold_sess/PRICE_GRID_DECLARATION.md)"
    )
PRIMARY_LABEL = setup["labels"]["primary"]
LABEL_HORIZON_BARS = {
    name: int(str(hours).rstrip("Hh")) for name, hours in setup["labels"]["horizons"].items()
}
# HOLD_BARS is NOT set here any more, and not from labels.horizons either. It is derived per
# (label, book) from the sealed panel and the registered grid further down this notebook, once the
# panel exists - NY_EXIT_DECLARATION.md section 1.2. `LABEL_HORIZON_BARS` survives only as what the
# derivation is compared against: the London book must come out at the declared horizon, and any
# book that comes out ABOVE it is a bug, never a longer hold.

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
if modal_gap != 60:
    raise RuntimeError(
        f"the registered price grid has a modal spacing of {modal_gap} minutes, not 60: this "
        "stage requires the H1 grid registered by loader 'exness_gold_sess_h1'"
    )
label_frame = pl.read_parquet(
    get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{PRIMARY_LABEL}.parquet"
)
grid_keys = grid.select("timestamp", "symbol").unique()
window_lo, window_hi = grid["timestamp"].min(), grid["timestamp"].max()
decision_rows = label_frame.select("timestamp", "symbol").unique().filter(
    (pl.col("timestamp") >= window_lo) & (pl.col("timestamp") <= window_hi)
)
off_grid = decision_rows.join(grid_keys, on=["timestamp", "symbol"], how="anti")
if off_grid.height:
    raise RuntimeError(
        f"{off_grid.height} of {decision_rows.height} decision instants inside the price window "
        f"are NOT rows of the price grid (first: {off_grid.head(3).to_dicts()}). The grid is not "
        "keyed on the bar close and the fill convention would be an hour late."
    )
print(
    f"Price grid: {grid.height:,} rows, {grid['symbol'].n_unique()} metals, "
    f"{window_lo} -> {window_hi}, modal spacing {modal_gap} min; all "
    f"{decision_rows.height:,} decision instants in the window are grid rows"
)
print(
    f"Declared label horizons in bars: {LABEL_HORIZON_BARS}. HOLD_BARS is derived per "
    "(label, book) below, from the sealed panel and this grid - not from this dict."
)

# %% [markdown]
# ## Select the exact prediction population
#
# Canonical production includes every complete validation prediction the model notebooks currently
# publish. `superseded_members` asks the registry which identities a later generation retired and
# no generation in force still lists, which is exactly the set to drop: `identity_status` says the
# registry still understands a row, not that its producer stands behind it.

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
# ## The three row-level rules, each measured before it is used
#
# ### The session books move rows
#
# `apply_session_filter` is exercised here on the decision instants of the sealed primary label -
# the superset every prediction frame is drawn from - and the two books must be **non-empty**,
# **disjoint** and together **exhaustive**. A book that selected the same rows as the other would
# be a duplicate spec and a wasted trial in the Deflated Sharpe Ratio.
#
# ### The rebalance schedule is not thinned
#
# `labels.rebalance_step` is 1 on all three labels, re-authored on 2026-09-08 while `run_log/` was
# empty. The check below is the one `nasdaq100_microstructure/config/setup.yaml:632-641` calls
# "the values did not change; what they count did": on a book filtered to one venue the schedule
# is one slot a day, so a step of 2 would trade every *other* day, and on a pooled sleeve list
# `gather_every(2)` would keep one venue and silently drop the other.
#
# ### Friday, and only Friday, is filtered out of the 24-hour book
#
# A weekday is knowable in advance for every week of the sample, so filtering on it is a rule.
# Filtering on the label's own null endpoint would be a point-in-time violation: at 14:00 on
# 3 July the bot does not know the tape stops at 18:00. The two are asserted to agree on Friday
# exactly - every Friday decision instant is null for `fwd_ret_24h`, with zero exceptions - and
# the residual Monday-to-Thursday nulls are early closes, which are **not** filtered and simply
# carry no prediction row. Measured 2026-09-08 over 4,817 decision instants: Friday 944/944 null,
# Monday 0.53 %, Tuesday 0.51 %, Wednesday 0.31 %, Thursday 2.47 %.

# %% tags=["results"]
DECLARED_BOOKS = list(setup["backtest"]["sweep"]["session_books"])
books = SESSION_BOOKS or DECLARED_BOOKS
unknown_books = sorted(set(books) - set(setup["decision"]["session_filter_values"]))
if unknown_books:
    raise ValueError(f"undeclared session books {unknown_books}")
pooled = setup["backtest"]["sweep"]["pooled_book"]
print(
    f"Engine books: {books}. The pooled book is NOT an engine run: "
    f"{pooled['method']} {pooled['weights']} produced in {pooled['produced_in']}."
)

probe = label_frame.select("timestamp", "symbol").with_columns(pl.lit(0.0).alias("y_score"))
book_rows = {
    book: apply_session_filter(probe, CASE_STUDY_ID, {"session_filter": book}) for book in books
}
book_keys = {
    book: set(zip(frame["timestamp"].to_list(), frame["symbol"].to_list(), strict=True))
    for book, frame in book_rows.items()
}
for book, keys in book_keys.items():
    if not keys:
        raise RuntimeError(f"session book {book!r} selects no row of the decision grid")
overlap = set.intersection(*book_keys.values()) if len(book_keys) > 1 else set()
if overlap:
    raise RuntimeError(
        f"session books {books} overlap on {len(overlap)} rows; they would register distinct "
        "hashes carrying the same result"
    )
probe_keys = set(zip(probe["timestamp"].to_list(), probe["symbol"].to_list(), strict=True))
uncovered = len(probe_keys - set().union(*book_keys.values()))
print(
    "Rows selected per book: "
    + ", ".join(f"{book} {len(keys):,}" for book, keys in book_keys.items())
    + f"; overlap {len(overlap)}, rows in no book {uncovered}"
)

# ROWS PER SYMBOL PER DAY, inside each book - NY_EXIT_DECLARATION.md section 4.b. This is the
# number `signals.py:377` multiplies by `lookback_days` to size the rolling percentile window, so
# `backtest.sweep.signal_specs[*].bars_per_day` has to EQUAL it rather than merely be near it. It
# is measured on the modal day of each book, and the whole distribution is printed: a book that
# carried two rows a day for one symbol would be the pooled book leaking back in.
declared_bars_per_day = {
    spec["name"]: int(spec["bars_per_day"])
    for spec in setup["backtest"]["sweep"]["signal_specs"]
    if "bars_per_day" in spec
}
for book, frame in book_rows.items():
    per_day = (
        frame.group_by([pl.col("timestamp").dt.date().alias("day"), "symbol"])
        .agg(pl.len().alias("rows"))
        .group_by("rows")
        .agg(pl.len().alias("symbol_days"))
        .sort("symbol_days", descending=True)
    )
    modal_rows = int(per_day["rows"][0])
    print(
        f"  {book}: rows per (symbol, day) {per_day.sort('rows').to_dicts()}; modal {modal_rows}"
    )
    for spec_name, declared in declared_bars_per_day.items():
        if declared != modal_rows:
            raise RuntimeError(
                f"signal spec {spec_name!r} declares bars_per_day={declared} while book {book!r} "
                f"carries {modal_rows} row(s) per symbol on the modal day. signals.py:377 sizes "
                f"the rolling window as lookback_days x bars_per_day ROWS, so the window would be "
                f"{declared / modal_rows:.0f}x the declared "
                f"{setup['backtest']['sweep']['signal_specs'][-1]['lookback_days']} days."
            )
print(
    f"  bars_per_day {declared_bars_per_day} equals the measured rows per symbol per day in every "
    "book: the rolling percentile window is the declared number of DAYS."
)

calendar = get_backtest_config(CASE_STUDY_ID).calendar
for label in sorted(catalog.get_column("label").unique()):
    step = get_rebalance_step(CASE_STUDY_ID, label)
    for book in books:
        instants = book_rows[book]["timestamp"].unique().sort()
        schedule = resolve_decision_schedule(
            instants, setup["decision"]["cadence"], step, calendar
        )
        if schedule.len() != instants.len():
            raise RuntimeError(
                f"label {label} book {book}: the rebalance schedule keeps {schedule.len()} of "
                f"{instants.len()} decision instants at step {step}. A step above 1 thins a "
                "one-venue book to every other day; PRICE_GRID_DECLARATION.md section 2 pins it "
                "to 1 on all three labels."
            )
    print(f"  {label}: step {step}, schedule keeps every decision instant on {books}")

FRIDAY_LABEL = "fwd_ret_24h"
if FRIDAY_LABEL in set(catalog.get_column("label")):
    # RE-STATED 2026-09-08 (RULINGS_2026-09-08.md execution step 8). "Labelled" used to mean
    # "present as a ROW in the parquet", which was equivalent until A-prime
    # (FOLD_GEOMETRY_DECLARATION.md) made every label publish on the decision grid's key set with
    # a NULL where its own rule is undefined. After A-prime every Friday instant is a row, so this
    # guard measured a Friday null share of 0.0 and refused the sweep - on a change in what a row
    # means, not on a disagreement between the weekday rule and the label. The proposition it
    # enforces is about the VALUE: the weekday filter may only remove instants the label could
    # never have scored. Measured today on the A-prime parquet: 964 London and 966 New York Friday
    # rows, and 0 of them carry a non-null fwd_ret_24h - so the rule and the label still agree
    # exactly, and the guard now says so instead of refusing.
    labelled_24h = (
        pl.read_parquet(get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{FRIDAY_LABEL}.parquet")
        .filter(pl.col(FRIDAY_LABEL).is_not_null())
        .get_column("timestamp")
        .unique()
        .to_list()
    )
    weekday_table = (
        pl.DataFrame({"timestamp": label_frame["timestamp"].unique().sort()})
        .with_columns(
            pl.col("timestamp").dt.weekday().alias("dow"),
            pl.col("timestamp").is_in(labelled_24h).alias("has_label"),
        )
        .group_by("dow")
        .agg(pl.len().alias("instants"), (~pl.col("has_label")).sum().alias("null_24h"))
        .with_columns((pl.col("null_24h") / pl.col("instants")).alias("null_share"))
        .sort("dow")
    )
    print(f"\n{FRIDAY_LABEL} null endpoints by weekday (1 = Monday):")
    print(weekday_table)
    friday = weekday_table.filter(pl.col("dow") == 5)
    if friday.height != 1 or float(friday["null_share"][0]) != 1.0:
        raise RuntimeError(
            "the Friday rule and the 24-hour label's null set disagree: the weekday filter may "
            "only remove instants the label could never have scored"
        )
    print(
        f"  Friday: {int(friday['null_24h'][0])} of {int(friday['instants'][0])} instants null, "
        "0 disagreements - the weekday rule removes exactly what the label never defines. "
        "Monday-Thursday nulls are early closes and are NOT filtered."
    )

# %% [markdown]
# ## The early-close sessions: printed, never filtered
#
# 79-93 sessions a metal end before the venue's declared close - US public holidays on which the
# metals quote and then stop three to five hours early. They are **traded normally**. Dropping a
# row because its label endpoint is null would select on the future: at 14:00 on 3 July the bot
# does not know the tape stops at 18:00, and removing exactly the thin-liquidity sessions is a
# bias in one direction. `TimeExit` counts **bars**, so on such a day the eighth bar after entry is
# the next morning's and the position rides through the closure - which is what would happen live.
# The set is printed so the realised-hold test can exempt exactly it when it asserts the book's
# clock hold, while still asserting the book's BAR hold on 100 % of positions. They are also
# excluded from the hold derivation below - a row with no endpoint has nothing to measure against -
# but they are NOT excluded from trading (NY_EXIT_DECLARATION.md section 1.2, which leaves
# PRICE_GRID_DECLARATION.md section 3.3 standing).

# %% tags=["results"]
DEV_LO, DEV_HI = development_window(setup)
bars_1h = load_mt5_bars(
    "1h",
    symbols=sorted(setup["universe"]["symbols"]),
    start_date=DEV_LO,
    end_date=DEV_HI,
)
if bars_1h.schema["timestamp"].time_zone is not None:
    bars_1h = bars_1h.with_columns(
        pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    )
panel = session_panel(bars_1h, keep_context=True, verbose=False)
assert_no_holdout(setup, bars_1h, panel)
early = (
    panel.filter(pl.col("label_end_ts").is_null())
    .filter((pl.col("timestamp") >= window_lo) & (pl.col("timestamp") <= window_hi))
    .select("symbol", "session", "timestamp")
    .sort(["symbol", "timestamp"])
)
print(
    f"Early-close / unresolved-endpoint decision rows inside the backtest window: {early.height} "
    f"({early.group_by('symbol').len().sort('symbol').to_dicts()}). Traded normally, not filtered."
)
with pl.Config(tbl_rows=min(max(early.height, 1), 200)):
    print(early)

# %% [markdown]
# ## The hold, derived per (label, book) - the numbers are printed, never typed
#
# `NY_EXIT_DECLARATION.md` section 1.2. One rule, both venues, and the constants fall out:
#
#     HOLD_BARS[book] = i1 - i0 on the registered grid's bar index, where i0 is the first row
#     keyed strictly after the decision (the row `next_bar` fills the entry at) and i1 is the last
#     row whose fill instant `key - 60min` is at or before the sealed `label_end_ts`.
#
# Measured over the development window only - `universe.history_start` to the day before
# `evaluation.holdout_start`, enforced by `assert_no_holdout` - on the sealed session panel and the
# registered price grid. Rows with a null `label_end_ts` are excluded from the derivation and still
# trade with the same constant.
#
# **Constancy is asserted, not hoped for**: `N` must be one value inside a book across both metals,
# both seasons and every fold, or `derive_hold_bars` raises `HoldIsNotConstant` and this stage
# stops. One measured exception is *declared* rather than filtered: 102 New York endpoints dated
# before the earliest fold `train_start` (2018-08-30) give 8 instead of 7, because the broker's
# daily break did not sit at the New York close in 2017. No fold of this bot fits on, scores on or
# walks those rows; the count is passed in below, so a change in it raises too.
#
# `fwd_ret_24h` is the one label whose `N` cannot be a single number - a whole day of grid contains
# public holidays as well as the recurring break - so its backstop comes from the same rule under
# `derive_backstop_bars`, which requires every deviation to be explained by MORE holes in the tape.
# It measures **23**, not the 24 this stage used to type.

# %% tags=["results"]
import pandas as pd  # noqa: E402

from utils.cv_splits import generate_cv_splits  # noqa: E402

dev_grid = load_backtest_prices(CASE_STUDY_ID, start_date=DEV_LO, end_date=DEV_HI)
assert_no_holdout(setup, dev_grid)
_timeline = (
    label_frame.select(pl.col("timestamp").dt.date().alias("timestamp")).unique().sort("timestamp")
)
FOLDS = [
    {
        "fold": int(split["fold"]),
        **{
            name: str(pd.Timestamp(split[name]).date())
            for name in ("train_start", "train_end", "val_start", "val_end")
        },
    }
    for split in generate_cv_splits(
        _timeline,
        case_study_id=CASE_STUDY_ID,
        label_buffer=setup["labels"]["buffer"],
        date_col="timestamp",
    )
]
print(
    f"Deriving the hold on {dev_grid.height:,} grid rows and {panel.height:,} sealed panel rows, "
    f"{DEV_LO} -> {DEV_HI}; fold ladder {[(f['fold'], f['train_start'], f['val_end']) for f in FOLDS]}"
)

#: MEASURED 2026-09-08 and declared, not filtered. See the markdown above and BOT.md.
PRE_FOLD_BREAK_REGIME = {"ny": 102}

HOLD_BARS: dict[tuple[str, str], int] = {}
EXIT_GAP_MINUTES: dict[str, int] = {}
BACKSTOP_LABELS = {label for label, bars in LABEL_HORIZON_BARS.items() if bars > 12}
for label in sorted(setup["labels"]["horizons"]):
    if label in BACKSTOP_LABELS:
        derived, measured = derive_backstop_bars(
            dev_grid,
            panel,
            label=label,
            horizons=setup["labels"]["horizons"],
            books=books,
            folds=FOLDS,
        )
    else:
        derived, measured = derive_hold_bars(
            dev_grid,
            panel,
            label=label,
            horizons=setup["labels"]["horizons"],
            books=books,
            folds=FOLDS,
            pre_fold_disagreements=PRE_FOLD_BREAK_REGIME,
        )
        gaps = exit_gap_minutes_by_book(measured, books)
        if EXIT_GAP_MINUTES and gaps != EXIT_GAP_MINUTES:
            raise RuntimeError(
                f"the exit gap differs between labels on the same books: {EXIT_GAP_MINUTES} then "
                f"{gaps}. The residual is a property of the tape, not of a label."
            )
        EXIT_GAP_MINUTES = gaps
    for book, bars in derived.items():
        if bars > LABEL_HORIZON_BARS[label]:
            raise RuntimeError(
                f"the derived hold for {label} / {book} is {bars} bars, longer than the declared "
                f"{LABEL_HORIZON_BARS[label]}-bar horizon. A hold may fall inside the label window "
                "and never past it: NY_EXIT_DECLARATION.md section 1.1."
            )
        HOLD_BARS[label, book] = bars

declared_gap = setup["backtest"]["exit_gap_minutes_by_book"]
measured_gap = {book: EXIT_GAP_MINUTES[book] for book in books}
if {k: int(v) for k, v in declared_gap["by_book"].items()} != measured_gap:
    raise RuntimeError(
        f"setup.yaml::backtest.exit_gap_minutes_by_book declares {declared_gap['by_book']} and the "
        f"grid measures {measured_gap}. The declaration and the tape have to agree before a sweep "
        "runs (NY_EXIT_DECLARATION.md section 2)."
    )
if declared_gap["derivation"] != DERIVATION_KEY:
    raise RuntimeError(
        f"exit_gap_minutes_by_book.derivation is {declared_gap['derivation']!r}, not "
        f"{DERIVATION_KEY!r}: the published residual names a rule this stage does not run"
    )
print(
    f"DERIVED HOLD_BARS (label, book) -> bars: {HOLD_BARS}. "
    f"Exit gap label_end_ts - exit_fill_instant, minutes: {measured_gap} "
    f"(declared {declared_gap['by_book']}, derivation {declared_gap['derivation']}, "
    f"measured {declared_gap['measured_on']}). The New York book trains on an 8-hour label and "
    "trades a 7-hour window; the shortfall is inside the label window, never past it."
)

# %% [markdown]
# ## Build the session-book strategy grid
#
# One job per (label x session book x signal spec). The signal dict is what the identity hash is
# taken over, so it is assembled once here and handed to both `plan_backtests` and `run_backtests`:
# a key that differed between the two would plan one identity and run another. It carries
#
# - `method` and its parameters, from `setup.yaml::backtest.sweep.signal_specs`;
# - `session_filter`, the book, which `apply_session_filter` turns into a row filter;
# - `drop_friday`, on the 24-hour label only, the weekday rule measured above;
# - `long_short: True` - `mapping.position_state_space` is `long_short`, and
#   `build_target_weights_from_config` defaults it to `False`, which is how `exness_fx_d1`
#   registered 1,068 long-only books under a long-short design (`BOT.md`, 2026-09-06).
#
# and the `risk` block carries the hold. Both go into the spec, so the record says what the book
# held rather than leaving it to a default.
#
# **Cost check first.** The engine's percentage cost model reads `setup.yaml::costs` through
# `get_backtest_config`; the Pro account pays no commission and the slippage leg is the top of the
# measured metals spread, charged on every crossing. The swap is a per-night holding cost the
# percentage engine cannot express and is priced in `16_costs`, never here.

# %% tags=["results"]
case_config = get_backtest_config(CASE_STUDY_ID)
declared_costs = setup["costs"]
declared_spread_bps = float(declared_costs["spread_bps"]["metals"][-1])
if case_config.commission_bps != 0.0 or case_config.slippage_bps != declared_spread_bps:
    raise RuntimeError(
        "the engine cost block does not match setup.yaml::costs: "
        f"commission {case_config.commission_bps} bps, slippage {case_config.slippage_bps} bps "
        f"against commission_per_lot {declared_costs['commission_per_lot']} and spread "
        f"{declared_spread_bps} bps"
    )
print(
    f"Engine costs from setup.yaml: commission {case_config.commission_bps:.2f} bps, "
    f"spread {case_config.slippage_bps:.2f} bps per crossing "
    f"({2 * case_config.slippage_bps:.2f} bps round trip), fill {case_config.execution_delay}, "
    f"{case_config.share_type} units"
)

jobs = []
for label in sorted(catalog.get_column("label").unique()):
    selected = catalog.filter(pl.col("label") == label)
    for book in books:
        # The hold is PER BOOK - NY_EXIT_DECLARATION.md section 1.1 - so the risk block is built
        # inside this loop and not outside it. London holds 8 bars, New York 7, and both numbers
        # came out of derive_hold_bars above rather than out of a literal. The hold is in EVERY
        # spec, Ch16 baselines included; Ch19 stacks stop and trailing rules on top of this one
        # through RuleChain (backtest_runner.py:2450) and never replaces it.
        hold = HOLD_BARS[label, book]
        risk = {"position_rules": [{"type": "time_exit", "bars": hold}]}
        for spec in setup["backtest"]["sweep"]["signal_specs"]:
            signal = {k: v for k, v in spec.items() if k != "name"}
            signal["long_short"] = True
            signal["session_filter"] = book
            if label == FRIDAY_LABEL:
                signal["drop_friday"] = True
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
# The trial count is NOT `expected_total`. PHASE567_DECLARATION.md D5.1: the engine runs one
# backtest per (label x prediction x book x signal spec) and `19_strategy_analysis` then builds
# ONE pooled series per (label x prediction x signal spec) by the declared 50/50 sleeve sum
# (setup.yaml::backtest.sweep.pooled_book). Those pooled series are members of the record's
# cohort - D5.3 names them `pooled:<london_hash>+<ny_hash>` - so K is the sum of the two, and the
# sentence this stage used to print ("this is the trial count the DSR divides by", on
# `expected_total` alone) under-stated K by exactly the pooled count.
pooled_series_total = catalog.height * len(setup["backtest"]["sweep"]["signal_specs"])
K_THIS_RUN = expected_total + pooled_series_total
# The gate's K is the one declared BEFORE the first fit (D5.1): 419 prediction sets x 2 engine
# books x 2 signal specs = 1,676 registered, plus 419 x 1 pooled book x 2 signal specs = 838, so
# K = 2,514. It is printed beside what this run resolves so the two can never be confused: a run
# that resolves fewer prediction sets does NOT get a smaller K, because K counts the search that
# was declared, not the members that happened to complete.
K_DECLARED_BY_DECLARATION = 2514
print(
    f"{len(jobs)} jobs over {catalog.height} prediction sets = {expected_total} DECLARED ENGINE "
    f"backtests ({len(books)} books x {len(setup['backtest']['sweep']['signal_specs'])} signal "
    "specs x the prediction sets of each label). That is the number of REGISTERED members, NOT "
    "the trial count. The DECLARED search is "
    f"K = {expected_total} + {pooled_series_total} = {K_THIS_RUN}, the second term being the "
    "pooled series built arithmetically in 19_strategy_analysis "
    "(PHASE567_DECLARATION.md D5.1 / D5.3 / D5.4): one per (label x prediction x signal spec), "
    "carrying no `backtest_runs` row and no `cohort_metrics` row, and counted in K because they "
    "were searched. "
    f"THE GATE DIVIDES BY THE DECLARED K = {K_DECLARED_BY_DECLARATION} (1,676 + 838), fixed "
    "before the first model was fitted; a run that resolves fewer members does not earn a "
    "smaller K. The plan-time exclusion predicate below NAMES every identity that cannot be run "
    "instead of discounting it, and prints the resolvable count beside this declared one."
)
if K_THIS_RUN != K_DECLARED_BY_DECLARATION:
    print(
        f"  NOTE: this run resolves K = {K_THIS_RUN}, not the declared "
        f"{K_DECLARED_BY_DECLARATION}. The declared value stands and the difference has to be "
        "explained in BOT.md before any Deflated Sharpe Ratio is quoted."
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
# ## The plan-time exclusion predicate: a constant score cannot exceed its own percentile
#
# Declared in `bots/exness_gold_sess/RULINGS_2026-09-08.md` (addendum of 2026-09-08, section 4)
# after the first sweep stopped at member 122 of 1,676, and executed here: **before**
# `plan_backtests`, before the population is frozen, and on a property of the **score** that is
# knowable without running the engine and without reading one row of P&L.
#
# **The mechanism, stated as narrowly as the mechanism actually is.** `rolling_percentile_signal`
# compares a score to a rolling percentile **of itself** (`case_studies/utils/signals.py:63-120`),
# so a prediction set whose score takes ONE distinct value inside a book can never exceed its own
# percentile: no target weight is emitted at any rebalance, and
# `_refuse_an_allocation_that_produced_no_target` (`case_studies/utils/backtest_runner.py
# :2503-2546`) refuses to register that flat account as a Sharpe of 0.0. That guard is correct and
# is NOT touched here - a 0.0 Sharpe would sit above every candidate with a negative one, and
# "an absence is counted as a trial against every real candidate beside it".
#
# **The exclusion is therefore scoped to the spec family, not to the prediction set.** The same
# constant score under `fixed_threshold_0` is a **runnable** member and stays in the plan:
# `signals.py:53-62` sets `lower_threshold = 1.0 - threshold` on the long-short branch, so at
# `threshold = 0.0` the short threshold is **1.0**; a constant score of 0.0 satisfies `score < 1.0`
# and the member is a permanently short book on both metals, with positions, turnover and a Sharpe.
# It is a real draw from the declared search - it could even be the leader if the metals fell over
# this window - so it is planned, run and registered like any other member.
#
# **K does not move.** This file said so before the first backtest existed - *"a run that resolves
# fewer prediction sets does NOT get a smaller K, because K counts the search that was declared,
# not the members that happened to complete"* - and the addendum holds it to that sentence: K =
# 2,514, unchanged. The identities this predicate removes from the PLAN are named, printed here,
# and carried into the cohort digest by `_report_phase5.py` as literal `unrunnable:` members. They
# are never deleted from the trial count and they contribute no Sharpe. Removing them from the plan
# is what lets the other 1,674 run; it is not a discount.
#
# **Three constraints make this a guard rather than a silencer**, all enforced below:
#
# 1. every excluded identity is printed with its prediction hash, book, spec name, distinct-score
#    count and row count;
# 2. the stage **raises** unless the number of excluded engine identities is exactly
#    `UNRUNNABLE_ENGINE_EXPECTED` (2), so the predicate can never grow in silence;
# 3. it is applied **only** to jobs whose `signal.method` is `per_symbol_rolling_percentile`,
#    because the mechanism is "a constant cannot exceed its own percentile", not "a degenerate set
#    should be dropped".
#
# This is **not** the `except-and-continue` branch D5.1 forbids: it catches no exception, it runs
# at planning time on a property computable before any engine call, and the run loop below is
# unchanged - one failed member there still leaves the frozen population incomplete and stops
# publication.

# %% tags=["results"]
PERCENTILE_METHOD = "per_symbol_rolling_percentile"
# Declared in advance (addendum section 4): exactly two engine identities, and exactly one pooled
# pair, are unrunnable. Any other number is a different fact about the search and stops the stage.
UNRUNNABLE_ENGINE_EXPECTED = 2
UNRUNNABLE_POOLED_EXPECTED = 1
DECLARED_ENGINE_TOTAL = expected_total
DECLARED_POOLED_TOTAL = pooled_series_total


def _score_series(frame: pl.DataFrame) -> pl.Series:
    """Return the column the signal reads.

    `backtest_runner.py:506-508` renames `prediction` to `y_score` on the way into the engine, so
    both names are accepted here and neither is assumed.
    """
    for column in ("y_score", "prediction"):
        if column in frame.columns:
            return frame.get_column(column)
    raise RuntimeError(f"a prediction frame carries no score column: {frame.columns}")


score_frames: dict[str, pl.DataFrame] = {}
unrunnable_engine: list[dict] = []
for job in jobs:
    if job["signal"].get("method") != PERCENTILE_METHOD:
        continue
    survivors: list[str] = []
    for row in job["predictions"].iter_rows(named=True):
        prediction_hash = row["prediction_hash"]
        if prediction_hash not in score_frames:
            result = Result.open(study, prediction_hash)
            if not isinstance(result, PredictionResult):
                raise TypeError(f"{prediction_hash} is not a prediction result")
            score_frames[prediction_hash] = result.load()
        # The signal dict carries `session_filter` and, on fwd_ret_24h, `drop_friday`, so this is
        # the exact row set the engine would build the rolling percentile over
        # (`apply_session_filter` runs at backtest_runner.py:1185, before any weight exists).
        in_book = apply_session_filter(score_frames[prediction_hash], CASE_STUDY_ID, job["signal"])
        scores = _score_series(in_book)
        distinct = int(scores.n_unique())
        if distinct > 1:
            survivors.append(prediction_hash)
            continue
        unrunnable_engine.append(
            {
                "prediction_hash": prediction_hash,
                "label": job["label"],
                "family": row.get("family"),
                "config_name": row.get("config_name"),
                "book": job["book"],
                "signal_spec": job["spec_name"],
                "method": job["signal"]["method"],
                "n_unique_score": distinct,
                "rows_in_book": in_book.height,
                "constant_value": float(scores[0]) if in_book.height else float("nan"),
                "member_name": f"unrunnable:{prediction_hash}/{job['book']}/{job['spec_name']}",
            }
        )
    if len(survivors) != job["expected"]:
        job["predictions"] = job["predictions"].filter(pl.col("prediction_hash").is_in(survivors))
        job["expected"] = job["predictions"].height

print(
    f"Plan-time exclusion predicate, applied to the {PERCENTILE_METHOD!r} family only "
    f"({sum(1 for job in jobs if job['signal'].get('method') == PERCENTILE_METHOD)} of "
    f"{len(jobs)} jobs), over {len(score_frames)} prediction sets read for their SCORE alone:"
)
for entry in unrunnable_engine:
    print(
        f"  EXCLUDED FROM THE PLAN  prediction {entry['prediction_hash']} "
        f"({entry['label']} / {entry['family']} / {entry['config_name']}) "
        f"x book {entry['book']} x spec {entry['signal_spec']} [{entry['method']}]: "
        f"n_unique(score) = {entry['n_unique_score']} over {entry['rows_in_book']:,} rows in this "
        f"book, constant value {entry['constant_value']}. Named "
        f"{entry['member_name']} and COUNTED IN K."
    )
if len(unrunnable_engine) != UNRUNNABLE_ENGINE_EXPECTED:
    raise RuntimeError(
        f"the plan-time exclusion predicate removed {len(unrunnable_engine)} engine identities, "
        f"not the {UNRUNNABLE_ENGINE_EXPECTED} declared in RULINGS_2026-09-08.md (addendum, "
        "section 4). A predicate that grows in silence is a machine for making results appear, so "
        "this stage stops: either a new prediction set is degenerate - which is a fact about the "
        "search that has to be declared with a date before it may be excluded - or the predicate "
        f"is wrong. Removed: {[entry['member_name'] for entry in unrunnable_engine]}"
    )

# The pooled book is arithmetic (D5.4): a pair needs BOTH sleeves. The pair whose two sleeves are
# both unrunnable cannot be built either, which is why the unrunnable count is 3 and not 2.
surviving_books: dict[tuple[str, str, str], set[str]] = {}
for job in jobs:
    for prediction_hash in job["predictions"].get_column("prediction_hash").to_list():
        surviving_books.setdefault((job["label"], job["spec_name"], prediction_hash), set()).add(
            job["book"]
        )
pooled_buildable = 0
unrunnable_pooled: list[dict] = []
for spec in setup["backtest"]["sweep"]["signal_specs"]:
    for row in catalog.iter_rows(named=True):
        sleeves = surviving_books.get((row["label"], spec["name"], row["prediction_hash"]), set())
        if sleeves == set(books):
            pooled_buildable += 1
        elif sleeves:
            raise RuntimeError(
                f"pooled pair {row['prediction_hash']}/{spec['name']} has sleeves "
                f"{sorted(sleeves)} and not {sorted(books)}. D5.4 pairs two sleeves; a pair with "
                "exactly one runnable sleeve is a case this declaration does not cover and it is "
                "not decided here."
            )
        else:
            unrunnable_pooled.append(
                {
                    "prediction_hash": row["prediction_hash"],
                    "label": row["label"],
                    "signal_spec": spec["name"],
                    "member_name": f"unrunnable:pooled:{row['prediction_hash']}/{spec['name']}",
                }
            )
for entry in unrunnable_pooled:
    print(
        f"  UNBUILDABLE POOLED PAIR  {entry['label']} / prediction {entry['prediction_hash']} "
        f"x spec {entry['signal_spec']}: neither sleeve exists, so the 0.5/0.5 sleeve sum of D5.4 "
        f"has nothing to sum. Named {entry['member_name']} and COUNTED IN K."
    )
if len(unrunnable_pooled) != UNRUNNABLE_POOLED_EXPECTED:
    raise RuntimeError(
        f"{len(unrunnable_pooled)} pooled pairs are unbuildable, not the "
        f"{UNRUNNABLE_POOLED_EXPECTED} declared: "
        f"{[entry['member_name'] for entry in unrunnable_pooled]}"
    )

expected_total = sum(job["expected"] for job in jobs)
pooled_series_total = pooled_buildable
scorable_members = expected_total + pooled_series_total
unrunnable_total = len(unrunnable_engine) + len(unrunnable_pooled)
if expected_total + len(unrunnable_engine) != DECLARED_ENGINE_TOTAL:
    raise RuntimeError(
        f"the plan lost members it did not name: {expected_total} planned + "
        f"{len(unrunnable_engine)} excluded != {DECLARED_ENGINE_TOTAL} declared"
    )
if scorable_members + unrunnable_total != K_DECLARED_BY_DECLARATION:
    raise RuntimeError(
        f"{scorable_members} scorable + {unrunnable_total} unrunnable != "
        f"{K_DECLARED_BY_DECLARATION} declared. Every declared member is either scorable or "
        "named unrunnable; there is no third state and none is created here."
    )
print(
    f"K = {K_DECLARED_BY_DECLARATION} DECLARED, and it does not move. "
    f"{expected_total} engine members runnable ({DECLARED_ENGINE_TOTAL} declared - "
    f"{len(unrunnable_engine)} excluded), {pooled_series_total} pooled pairs buildable "
    f"({DECLARED_POOLED_TOTAL} declared - {len(unrunnable_pooled)}), {scorable_members} members "
    f"SCORABLE, {unrunnable_total} identities NAMED AND NEVER DELETED. The three named identities "
    "get no `backtest_runs` row, no Sharpe, no place in the Sharpe distribution and no place in "
    "`variance_trials`; they are counted in K only, because they were searched."
)

# %% [markdown]
# ## Freeze the expected population
#
# Planning resolves every backtest identity without running or writing one. Production freezes that
# complete expected set before the first member executes, so a failed member remains visible. The
# uniqueness check below is what proves the book and the hold really are inside the identity: two
# books collapsing to one hash would be caught here rather than after the whole sweep.

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
# Each selected row produces an independent backtest. The loop has no exception-and-continue path:
# one failed member leaves the predeclared population incomplete and stops publication.
#
# `run_backtests` serves an identity that is already registered and complete instead of running it
# again, so a bare member count says nothing about whether this run did any work; the runner reports
# `reused` or `completed` per member in `execution.diagnostics` and that is what is counted.
#
# **The post-run guard reads the CONTENT of every registered spec, not a stage label (amended
# 2026-09-09, `PHASE567_DECLARATION.md` D5.2 amendment).** The assertion that used to stand here,
# `set(stage) == {"signal"}`, was copied from `exness_fx_d1/13_backtest.py:533`, whose baseline
# specs carry no `risk` block. This bot's baseline specs carry the hold as a `time_exit` position
# rule (`PRICE_GRID_DECLARATION.md` section 2), and the registry does not take a stage from the
# caller - it infers one from the spec's content (`case_studies/utils/registry/store.py:447-449`:
# a `risk` block whose `name` is not `baseline` is `risk_overlay`). So the first complete sweep
# registered all 1,674 members as `risk_overlay` and the old assertion raised AFTER every member
# was registered and the population was complete. It is the FIFTH unwritten convention of the
# shared library this bot has met ("a spec with a risk block is a risk_overlay run" against "the
# hold is a position rule inside the baseline spec"); the library's exit, `risk: {name: baseline,
# ...}`, moves every spec hash and belongs to a generation declared BEFORE its first backtest.
# What is asserted instead is what the D5.2 table lists: per row, the risk block equals the
# derived hold of that row's (label, book), the session filter is that row's book, and the registry
# gave every row ONE stage label that equals what `_infer_stage` derives from that content. The
# predicate reads `spec_json` only - no return, no metric - so it cannot admit or exclude a member.

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
    expected_risk = {
        "position_rules": [
            {"type": "time_exit", "bars": HOLD_BARS[job["label"], job["book"]]}
        ]
    }
    if strategy.get("risk") != expected_risk:
        raise RuntimeError(
            f"{result.hash}: the registered risk block {strategy.get('risk')!r} is not the derived "
            f"hold {expected_risk!r} of ({job['label']}, {job['book']})"
        )
    session_filter = strategy.get("signal", {}).get("session_filter")
    if session_filter not in set(books) or session_filter != job["book"]:
        raise RuntimeError(
            f"{result.hash}: the registered session_filter {session_filter!r} is not the book "
            f"{job['book']!r} of its job (declared books {books})"
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
if registered_stage != inferred_stage or registered_stage != "risk_overlay":
    raise RuntimeError(
        f"the registry labelled the baseline rows {registered_stage!r}, the library infers "
        f"{inferred_stage!r} from the same content, and this generation declared 'risk_overlay' "
        "(PHASE567_DECLARATION.md D5.2, amended 2026-09-09); a stage convention moved"
    )
print(
    f"Registry stage label on all {len(backtests):,} baseline rows: '{registered_stage}'. It is "
    "inferred by store.py:447-449 from the spec content - every baseline spec of this bot carries "
    "the hold as a `time_exit` position rule - so it is NOT 'signal'; the assertion this stage "
    "used to carry was copied from exness_fx_d1, whose baseline specs have no risk block. The "
    "label is the registry's tier name for the rows, not a claim that a risk overlay was swept; "
    "the D5.2 row `stage == \"signal\"` is amended on 2026-09-09 and the content of every row was "
    "asserted instead (derived hold per (label, book), session_filter == book)."
)

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
# - The grid is the H1 tape keyed on the bar close; the hold is a `TimeExit` position rule inside
#   every spec, so the registry records what each book held instead of a default.
# - `london` and `ny` are the engine books and their rows are disjoint, measured before planning.
#   `both` is a deterministic 50/50 sleeve sum built in `19_strategy_analysis`; it adds no trial.
# - `rebalance_step` is 1 everywhere and the schedule is asserted to keep every decision instant:
#   on a one-venue book a step of 2 would trade every other day, and on a pooled schedule it would
#   keep one venue.
# - The Friday rule for `fwd_ret_24h` is a weekday rule, and it agrees with the label's null set on
#   100 % of Fridays; early closes are printed and traded, never filtered.
# - `avg_turnover` in the registry under-reports on this bot, because a rule exit does not pass
#   through the weight series. Read turnover from `fills.parquet` / `trades.parquet`; `16_costs` is
#   unaffected because the engine charges on real fills.
# - Selection happens on this validation split only; the holdout is scored once, later.
