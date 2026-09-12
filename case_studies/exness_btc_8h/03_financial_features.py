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
# # Exness BTC 8h (exness_btc_8h): Feature Engineering
#
# One instrument, three decisions a day, every day of the week. Every column below answers one
# question: at the moment the position is decided, which bars are already on the tape, and what
# does the feature make of them?
#
# The answer is given on **two grids**, and that is the design this notebook exists to check. The
# eight-hour decision bars carry the short windows; anything longer is read from **daily** bars
# and joined backward on the D1 bar's own **close instant** - never on its calendar date, which
# would hand a 16:00 UTC decision a bar that closes eight hours after it. Three of the families
# on the daily grid are **cross-asset**: gold, a broad-dollar proxy built from the five FX majors,
# and a US index. All three are priced and never traded, all three come out of the same MT5
# parquet, and all three keep their own five-day week while this instrument keeps a seven-day one -
# which is why the join, and not the arithmetic, is the part that has to be checked.
#
# What is **not** here is every cross-sectional mechanism of the templates this pipeline descends
# from. One instrument is not a cross-section: a percentile over one name is the constant 0.5 at
# every instant. `setup.yaml::features.ranked` is empty by design and the point-in-time test
# asserts that no `rank_*`, `xs_*` or `vs_median` column reached the matrix.
#
# ## Learning objectives
#
# - State how far back a feature reads, and on which grid, before writing its code
# - Join a slower five-day grid onto a faster seven-day decision grid without reading a bar that
#   had not closed
# - Show that withholding later bars leaves every earlier value unchanged, and that rebuilding the
#   matrix from only the bars that had closed at a decision instant reproduces the batch row
# - Carry a cost that is knowable in advance - the broker's overnight financing - as a feature
#   rather than as a footnote, and check the flags against the broker's own rule
# - Read a feature set for scale, redundancy and decay before any model sees it
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Sections 8.2-8.4 (price and volume features, structural cross-instrument features).
# Route B fork of [`exness_gold_sess/03_financial_features`](../exness_gold_sess/03_financial_features.ipynb)
# for the bot `exness_btc_8h` (`bots/exness_btc_8h/BOT.md`), with three differences: the decision
# grid is a native eight-hour bar grid rather than a venue session; the cross-asset families are
# three partners rather than a two-leg ratio; and two of the state columns describe a **cost**
# rather than a market. It reads MT5 H4 bars folded to eight hours and D1 bars, plus its settings
# from `config/setup.yaml`, and reads nothing another notebook wrote. The construction itself lives
# in `_features.py` beside this file, because the point-in-time test
# (`bots/exness_btc_8h/tests/test_lookahead.py`), the label stage and, later, step 2 of the
# deployment loop all have to compute exactly what this notebook computes. It writes
# `features/financial.parquet` with a `.digest.json` sidecar.

# %%
"""exness_btc_8h: Feature Engineering (Route B fork of exness_gold_sess)."""

import warnings
from datetime import date, datetime, timedelta
from fnmatch import fnmatch

import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_btc_8h._features import (
    build_features,
    cross_asset,
    d1_state,
    decision_grid,
    feature_columns,
    features_as_of,
    load_d1,
    warmup_expectations,
)
from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.feature_engineering import (
    assert_values_agree,
    assign_families,
    families_from_config,
    family_coverage,
    plot_coverage_through_time,
    plot_feature_distributions,
    plot_redundancy_clusters,
    plot_timing_contract,
    register_frame,
    warmup_audit,
)
from utils.paths import display_path, get_case_study_dir
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")
set_global_seeds(42)

CASE_STUDY_ID = "exness_btc_8h"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
FEATURES_DIR = CASE_DIR / "features"

# %% [markdown]
# Two settings are left open. `START_DATE` of `None` starts at `setup.yaml::universe.history_start`
# - the first date the folded eight-hour grid is complete, measured rather than guessed - which is
# what the production run does. `N_RECOMPUTE_INSTANTS` is how many decision instants Section D.4
# rebuilds the whole matrix at from only the bars that had closed by then. There is no cap on
# symbols: the universe is one name.

# %% tags=["parameters"]
START_DATE = None
N_RECOMPUTE_INSTANTS = 4

# %% [markdown]
# ## Configuration
#
# Every window, the grid split, the null policy and the holdout boundary live in
# `config/setup.yaml` and are bound here. A number retyped in this notebook would be a second
# answer to a question that already has one.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
FAMILIES = families_from_config(setup)
WINDOWS = setup["features"]["windows"]
CARRIER = setup["features"]["null_policy_carrier"]
NULL_POLICY = setup["features"]["null_policy"]
MAY_BE_NULL = list(NULL_POLICY["may_be_null"])
MAX_NULL_SHARE = float(NULL_POLICY["max_null_share"])
PERSISTENCE_HORIZON = int(setup["features"]["persistence_horizon"])
REDUNDANCY_CUT = float(setup["features"]["redundancy_cut"])
UNIVERSE = sorted(setup["universe"]["symbols"])
REFERENCE = sorted(set(setup["features"]["reference_symbols"]) - set(UNIVERSE))
HISTORY_START = START_DATE or str(setup["universe"]["history_start"])
D1_HISTORY_START = str(setup["features"]["grid"]["long_window_history_start"])
HOLDOUT_START = date.fromisoformat(str(setup["evaluation"]["holdout_start"]))
HOLDOUT_END = date.fromisoformat(str(setup["evaluation"]["holdout_end"]))
HOLDOUT_TS = datetime.combine(HOLDOUT_START, datetime.min.time())
SLOTS_PER_DAY = len(setup["decision"]["snapshots_utc"])
SLOTS_PER_YEAR = float(setup["decision"]["slots_per_year"])
CYCLE = int(setup["labels"]["rebalance_step"][setup["labels"]["primary"]])
_ROLLOVER_DOW = {"sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
                 "friday": 5, "saturday": 6}[str(setup["costs"]["swap"]["rollover3days"]).lower()]

print(f"{len(FAMILIES)} feature families are declared, and Section A prints what each one claims")
print(
    f"Two grids: {setup['features']['grid']['slot']} for the decision-grid families, "
    f"{setup['features']['grid']['long_window']} for anything longer, joined "
    f"{setup['features']['grid']['long_window_join']}"
)
print(
    f"Traded: {UNIVERSE}. Priced but NOT traded, for the cross-asset families: {REFERENCE} - no "
    "label is built on any of them and no order is ever placed in one."
)
print(
    f"The matrix is written through {HOLDOUT_END} so the holdout stages have a feature vintage; "
    f"every figure below reads only rows before {HOLDOUT_START}."
)

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The bot is not looking for a cross-sectional edge - there is no cross-section. Its hypothesis is
# about **time-series structure at an eight-hour horizon**: whether this instrument's own recent
# path, read at three fixed points of the day, says anything about the next eight hours, and
# whether the state it is read in - its own longer-run position, three cross-asset partners, and
# the cost of the slot the decision falls on - conditions that reading.
#
# That splits the matrix in two, and the register's `role` column is where the split is written
# down. A **signal** column is one a position may be formed on: momentum and mean reversion on the
# decision grid, and the oscillator family. A **state** column describes the environment the signal
# is read in and is never traded on its own: volatility and range, the long-window daily state, the
# three cross-asset families, and the swap-and-calendar flags.
#
# Two things in the register deserve to be read before the numbers arrive.
#
# **The `usidx` family carries a declared hole.** This account serves USTEC daily bars only from
# 2019-07-16, so its 63-day columns are null until 2019-09-18 while the panel opens in December
# 2018. That is written into `features.null_policy.may_be_null` with its measured share, before any
# information coefficient was read, because "one family is null on nine per cent of the
# development window" is a very different sentence when it is discovered afterwards.
#
# **Two state columns describe a cost, not a market.** `pays_swap_night` is true exactly on the
# 16:00 decision, whose hold crosses server midnight and pays the broker's overnight financing;
# `pays_triple_swap` marks the Fridays where that charge is tripled. Both are functions of the
# timestamp alone, so they are knowable weeks in advance, and both are checked against
# `bots/_shared/costs_mt5.holding_cost_points` on the account's own `symbol_info` record rather
# than against the rule they were written from.
#
# One entry is `PLANNED` and carries no column: `on-chain and flow`, the family
# `bots/assets/BTCUSD.md` names as the driver a price series cannot contain. No free, versioned,
# point-in-time source is wired into this repository and every API key in `.env` has an empty
# value, so it is declared rather than silently omitted - and it is **+1 trial** when it is first
# tried.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "driver hypothesis", "inputs", "lookback (bars)", "lag (bars)", "frame"]
)

# %% [markdown]
# ## B. Inputs and their observability
#
# ### B.1 The decision grid
#
# The eight-hour bars are folded from the account's H4 bars and keyed on their **close**, which is
# the decision instant. `decision_grid` asserts the decision hours and reports every gap other than
# one slot; `01_feasibility_analysis` found none on the development window, and
# `bots/exness_btc_8h/tests/test_data_quality.py` re-runs the census on every test run.

# %%
bars = load_mt5_bars(
    "8h", symbols=UNIVERSE, start_date=HISTORY_START, end_date=str(HOLDOUT_END)
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
panel = decision_grid(bars)
print(
    f"panel {panel.height:,} decision slots, {panel['timestamp'].min()} to "
    f"{panel['timestamp'].max()} UTC"
)

# %% [markdown]
# ### B.2 The daily grid, and the partners that are priced but not traded
#
# The daily frame is read from `features.grid.long_window_history_start`, **not** from
# `universe.history_start`. That head start is what a two-grid design buys: a 315-D1-bar chain is
# already dense when the panel opens, so no long window pushes the panel's start forward.
#
# It buys less here than it does on the sibling metals bot, and the difference is worth saying out
# loud. The cross-asset partners serve daily bars from 2014, so their families are dense from the
# first panel row. **This instrument's own** daily bars start on 2018-02-09 - the same week its
# H4 tape starts - so a 315-D1-bar chain of its own is *not* dense until December 2018, and that
# is the longest chain in the matrix. It is the reason `features.null_policy_carrier` is a D1
# column here and a decision-grid column on the metals bot.

# %%
d1 = load_d1(setup, UNIVERSE, end_date=str(HOLDOUT_END))
d1_reference = load_d1(setup, REFERENCE, end_date=str(HOLDOUT_END)) if REFERENCE else None
print(
    f"D1 traded: {d1.height:,} bars {d1['timestamp'].min()} to {d1['timestamp'].max()} "
    f"(requested from {D1_HISTORY_START})"
)
if d1_reference is not None:
    per_symbol = (
        d1_reference.group_by("symbol")
        .agg(pl.len().alias("bars"), pl.col("timestamp").min().alias("first"))
        .sort("first")
    )
    display(per_symbol)
    print(
        "The last row is why `usidx` is in `null_policy.may_be_null`: its history starts inside "
        "this bot's development window, and it is the only family that does."
    )

# %% [markdown]
# ### B.3 The build
#
# One call, the same one every consumer makes. The construction is in `_features.build_features`
# rather than in this notebook, so the point-in-time test and the deployment loop compute the same
# thing rather than a copy of it.

# %%
built = build_features(
    panel,
    d1,
    WINDOWS,
    reference=d1_reference,
    slots_per_year=SLOTS_PER_YEAR,
    rollover3days_dow=_ROLLOVER_DOW,
)
feature_cols = feature_columns(built)
print(f"{len(feature_cols)} feature columns over {built.height:,} decision slots")

# %% [markdown]
# ## C. Feature construction, one subsection per family
#
# ### C.1 Momentum on the decision grid (8h)
#
# Trailing returns over 1, 3, 9, 21, 63, 126 and 252 slots - eight hours to twelve weeks - plus
# the difference between neighbouring horizons, which says whether a move is accelerating, and a
# Sharpe ratio at each horizon of 21 slots or more, which says whether it was earned with
# dispersion or without it. `01_feasibility_analysis` measured a one-day (three-slot)
# autocorrelation of about $-0.08$ against a white-noise band of $0.022$, so the shortest horizons
# are the ones with something to explain.
#
# ### C.2 Mean reversion on the decision grid (8h)
#
# Trailing z-scores of the multi-horizon returns against a 252-slot window, the position of the
# close inside its own trailing channel, and Bollinger %B. The declared failure mode is the one
# this sample is made of: Bitcoin trends for months, and a z-score calls every month of a trend a
# dislocation.
#
# ### C.3 Volatility and range (8h)
#
# Garman-Klass from the bar's own OHLC and close-to-close from the log return, plus the ratio of a
# short window to a long one, the mean normalised bar range, and the distance below the trailing
# peak. `role: state`: none of these says anything about direction, and all of them condition what
# a signal of a given size is worth.
#
# ### C.4 Oscillator and trend (8h)
#
# Wilder-smoothed RSI at 14 and 42 slots, and the price against its own moving average at three
# horizons.
#
# ### C.5 Long-window state (D1, asof-joined on the bar close)
#
# The same statistics on daily bars: momentum, both volatility estimators, drawdown, z-scores, RSI
# and price-to-moving-average. Their windows are counted in **daily bars**, which is what makes
# them affordable: `d1_ret_252d` is a year of history for 252 rows rather than 756 decision slots.
#
# ### C.6 The three cross-asset families (D1, same join)
#
# Each is the partner's own trailing return at two horizons plus this instrument's rolling beta to
# it, estimated on 63 daily log returns. The partner's return is measured on the partner's own
# grid and then carried onto this one by a **forward fill of the price**: a forward-filled price is
# what a trader at that instant could see, a forward-filled return would not be. That fill is doing
# real work on every Saturday and Sunday row, because all three partners keep a five-day week and
# this instrument does not.
#
# - **gold** (`XAUUSD`): the debasement hedge Bitcoin is most often compared with.
# - **dollar** (`usd_*`): a signed proxy built from the five FX majors, negating a pair that quotes
#   the dollar second, then compounded into an index level. `BTCUSD` is quoted in dollars, so part
#   of every move is the numeraire moving rather than the asset.
# - **usidx** (`USTEC`): the risk-appetite state variable `bots/assets/BTCUSD.md` names, and the
#   one whose history starts inside this bot's development window.
#
# ### C.7 Swap and calendar state
#
# `slot_of_day`, `dow`, `is_weekend`, `is_us_hours`, `pays_swap_night`, `pays_triple_swap`. No
# market data at all - every value is a function of the decision timestamp - so `lookback` and
# `lag` are both zero. The cell below is the one that earns the family its place: it shows that
# the financing charge falls on exactly one slot in three, and that the triple charge falls on
# Friday and not on the Wednesday the FX and metals bots on this same account use.

# %%
display(
    built.group_by(pl.col("timestamp").dt.hour().alias("decision hour"))
    .agg(
        pl.len().alias("slots"),
        pl.col("pays_swap_night").mean().alias("share paying swap"),
        pl.col("pays_triple_swap").mean().alias("share paying triple"),
        pl.col("is_us_hours").mean().alias("is_us_hours"),
    )
    .sort("decision hour")
)
_triple_days = (
    built.filter(pl.col("pays_triple_swap") > 0)["timestamp"].dt.weekday().unique().to_list()
)
assert _triple_days == [5], f"the triple swap falls on weekday {_triple_days}, not Friday"
print(
    f"the triple charge falls only on weekday {_triple_days[0]} (Friday), which is "
    f"symbol_info.swap_rollover3days = {_ROLLOVER_DOW} on this symbol. FX and metals on the same "
    "account report 3 (Wednesday); reading their convention onto this one would put the charge on "
    "the wrong night."
)

# %% [markdown]
# ## D. The timing contract
#
# Each feature makes two promises about time: how many bars back it reads, and how long its inputs
# take to become available after the period they describe. Together those fix the earliest decision
# the feature can be used at. This section checks four things - what the constructions read, that
# no column fills before its window could have, that no value depends on a date after it, and that
# the value a decision instant would have seen is the value the batch panel holds.
#
# ### D.1 What each construction reads

# %%
plot_timing_contract(
    FAMILIES,
    bar_unit="bars of the family's own grid",
    title="Nothing waits to publish: every family reads up to the decision",
    subtitle=(
        "Register lookback per family. Read the unit per row: the four decision-grid families "
        "count 8-hour slots, the long-window and cross-asset families count D1 bars, and the "
        "state family reads one instant"
    ),
    alt=(
        "Horizontal bars, one per family, each extending leftward from the decision line by that "
        "family's declared lookback - zero for the swap and calendar state, 252 to 315 for the "
        "momentum, mean-reversion, long-window and cross-asset families. Every bar reaches the "
        "line, so no family shows a publication lag gap."
    ),
)

# %% [markdown]
# ### D.2 Warmup, on each grid in its own unit
#
# A trailing window cannot produce a value until it has enough bars to fill, and the audit checks
# that length rather than describing it. The unit is where a two-grid design goes wrong: a
# `gold_ret_63d` column's warmup is 63 **D1** bars, and because the reference frame starts in 2014
# that column is already dense at panel row 1. Auditing it against 63 rows of the *panel* would
# report a correct construction as "populated from fewer bars than its window spans" and raise. So
# the register returns two dictionaries and each is audited on its own frame.

# %%
expected = warmup_expectations(WINDOWS)
panel_census = warmup_audit(built, expected["decision slots"], entity="symbol")
d1_features = d1_state(d1, WINDOWS)
if d1_reference is not None:
    d1_features = cross_asset(d1_features, d1_reference, WINDOWS)
d1_census = warmup_audit(d1_features, expected["D1 bars"], entity="symbol")
print(f"{panel_census.height} columns audited in decision slots, {d1_census.height} in D1 bars")
display(
    pl.concat(
        [
            panel_census.with_columns(pl.lit("decision slots").alias("unit")),
            d1_census.with_columns(pl.lit("D1 bars").alias("unit")),
        ]
    ).sort("unit", "column")
)

# %% [markdown]
# ### D.3 Withholding the holdout changes nothing
#
# Trailing statistics share a property worth checking directly: recomputed on a panel that stops
# before the holdout, they reproduce the same values on the rows the two panels share. A parameter
# fitted over a whole column - a winsorization bound, a scaler - does not, because truncating the
# column moves the parameter and with it every row it was applied to. Building twice and comparing
# tests the whole construction at once, every emitted column rather than a sample, and does not
# depend on anyone having flagged the transform that fits.
#
# The comparison raises if any column moves, so reaching the next cell is the result.

# %%
_withheld = build_features(
    panel.filter(pl.col("timestamp") < HOLDOUT_TS),
    d1.filter(pl.col("timestamp") < HOLDOUT_START),
    WINDOWS,
    reference=(
        None if d1_reference is None else d1_reference.filter(pl.col("timestamp") < HOLDOUT_START)
    ),
    slots_per_year=SLOTS_PER_YEAR,
    rollover3days_dow=_ROLLOVER_DOW,
)
seal = assert_values_agree(
    built.filter(pl.col("timestamp") < HOLDOUT_TS),
    _withheld,
    columns=feature_cols,
    keys=["timestamp", "symbol"],
)
print(
    f"{seal['rows compared'].max():,} rows x {seal.height} columns unchanged when everything from "
    f"{HOLDOUT_START} is withheld"
)
display(seal.filter(pl.col("column").is_in([CARRIER, "slot_zscore_63", "usd_beta_63d", "slot_rsi_14"])))

# %% [markdown]
# ### D.4 Rebuilding the matrix at the decision instant
#
# D.3 withholds whole dates. The stronger form withholds **bars**: at a decision instant only the
# bars that had closed by then were on the tape, and a feature computed from those alone must equal
# the batch value. That is a claim about the two-grid join as much as about the arithmetic - a D1
# join that reached to the calendar date would pass D.3 and fail here, because at the 16:00
# decision the day's own daily bar has not printed.
#
# `_features.features_as_of` truncates the eight-hour bars, the daily bars and the reference bars
# each on their own bar length, rebuilds the panel and the whole matrix, and the row is compared
# column by column. The row count is asserted: an empty frame compares equal to an empty frame.

# %%
_dense = (
    built.filter(pl.col(CARRIER).is_not_null() & (pl.col("timestamp") < HOLDOUT_TS))["timestamp"]
    .unique()
    .sort()
)
_step = max(1, len(_dense) // (N_RECOMPUTE_INSTANTS + 1))
recompute_rows = []
for instant in _dense.gather_every(_step, offset=_step)[:N_RECOMPUTE_INSTANTS].to_list():
    as_of = features_as_of(
        instant,
        bars,
        d1,
        WINDOWS,
        reference=d1_reference,
        slots_per_year=SLOTS_PER_YEAR,
        rollover3days_dow=_ROLLOVER_DOW,
    )
    assert as_of.height == len(UNIVERSE), f"features_as_of returned {as_of.height} rows at {instant}"
    gap = assert_values_agree(
        built.filter(pl.col("timestamp") == instant).sort("symbol"),
        as_of.sort("symbol"),
        columns=feature_cols,
        keys=["timestamp", "symbol"],
    )
    recompute_rows.append(
        {
            "decision (UTC)": instant,
            "slot": int(instant.hour),
            "8h bars seen": bars.filter(
                pl.col("timestamp") + pl.duration(minutes=8 * 60) <= instant
            ).height,
            "D1 bar read": as_of["d1_close_ts"][0],
            "max abs difference": gap["max abs difference"].max(),
        }
    )
pl.DataFrame(recompute_rows)

# %% [markdown]
# ## E. Matrix assembly and the null policy
#
# The panel key is `symbol` + `timestamp`. Raw OHLCV, `bar_open_ts`, the intermediate log return
# and `d1_close_ts` are excluded: they are the inputs the features are made of, or the evidence
# that the join was backward, and neither is a feature.
#
# **The null policy is a cut plus a declared set of columns that may stay null.** Rows before the
# carrier can fill are dropped - pure warmup on the longest chain in the matrix, which here is a
# *daily* column because this instrument's own daily tape starts the same week its intraday tape
# does. After that, exactly one family may still hold a null: the US index, whose partner series
# starts in July 2019. Cutting those rows instead would throw away nine more months of this
# instrument's own history to accommodate a partner, which is the wrong trade - and the share is
# asserted against the declared cap rather than described.
#
# The matrix is written **through** the holdout, because the holdout stages need a feature vintage
# for the rows they score. Every figure from here on reads `development` only.

# %%
features = (
    built.select(["timestamp", "symbol", *feature_cols])
    .drop_nulls(subset=[CARRIER])
    .sort(["timestamp", "symbol"])
)
assert features.select(["timestamp", "symbol"]).is_duplicated().sum() == 0, "duplicate panel key"
development = features.filter(pl.col("timestamp") < HOLDOUT_TS)
null_counts = {c: development[c].null_count() for c in feature_cols if development[c].null_count()}
for column, count in sorted(null_counts.items()):
    assert any(fnmatch(column, pattern) for pattern in MAY_BE_NULL), (
        f"{column} holds {count} nulls after the carrier cut and is not in "
        f"features.null_policy.may_be_null {MAY_BE_NULL}"
    )
    assert count <= MAX_NULL_SHARE * development.height, (
        f"{column}: {count} nulls is {count / development.height:.2%} of the development window, "
        f"above the declared {MAX_NULL_SHARE:.0%}"
    )
assignment = assign_families(feature_cols, FAMILIES)
print(f"{len(feature_cols)} features, {features.height:,} rows, {features['symbol'].n_unique()} instrument")
print(f"{features['timestamp'].min()} to {features['timestamp'].max()}")
print(f"the null policy dropped {built.height - features.height:,} warmup rows on {CARRIER}")
print(f"{development.height:,} development rows before {HOLDOUT_START}, the rows every figure reads")
print("columns still holding a null, all declared, all inside the cap:")
for column, count in sorted(null_counts.items()):
    print(f"  {column}: {count:,} ({count / development.height:.2%})")
register_frame(FAMILIES, feature_cols).select(["family", "columns", "role", "representation"])

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ### E.1 Coverage through time
#
# Drawn on the panel *before* the null policy, so the figure shows what the policy is for. Each
# family climbs to complete coverage as its longest window on its own grid fills, and the dashed
# boundary marks the first slot the emitted matrix keeps. The gold and dollar families are already
# at one on the first slot - that is the two-grid head start paying for itself, because their
# warmup was spent on daily bars printed four years before this panel opens. The US index family
# is the one that is not, and it is the one the null policy names.

# %%
plot_coverage_through_time(
    family_coverage(
        built.filter(pl.col("timestamp") < HOLDOUT_TS).select(["timestamp", *feature_cols]),
        assignment,
        every="1mo",
    ),
    warmup_boundary=features["timestamp"].min(),
    title="The gold and dollar families start full; the US index arrives in 2019",
    subtitle="Monthly non-null share per family before the policy, with the boundary drawn",
    alt=(
        "Non-null share by feature family on an axis from zero to one. The gold and dollar "
        "families sit at one from the first month because their daily partners begin in 2014. "
        "The decision-grid families reach one within the first three months. The long-window "
        "state family reaches one at the end of 2018, which is where the emitted matrix starts. "
        "The US index family stays at zero until the second half of 2019 and then steps to one."
    ),
)

# %% [markdown]
# ## F. What the features look like
#
# Three properties decide whether this matrix can be used at all: the scale each feature arrives
# on, how much of the set is one ordering under several names, and how long a value lasts.
# `05_evaluation` is where it is tested fold by fold for whether any of it predicts.
#
# ### F.1 Distributions

# %%
plot_feature_distributions(
    development,
    ["slot_ret_1", "slot_channel_pos_63", "slot_zscore_63", "d1_vol_gk_63d", "usd_beta_63d", "slot_rsi_14"],
    title="The slot returns are thin and fat-tailed; the state columns are wide",
    subtitle="One panel per representative column, development window, display tails clipped",
    alt=(
        "Six histograms in two rows. The one-slot return is a narrow single-peaked body centred "
        "on zero with long tails; the channel position is bounded by zero and one with mass "
        "banked against both ends; the return z-score is broad and roughly symmetric; the daily "
        "Garman-Klass volatility is right-skewed; the dollar beta is centred near zero and "
        "symmetric; the oscillator is a wide body between about twenty and eighty."
    ),
)

# %% [markdown]
# ### F.2 Redundancy structure
#
# Two columns are redundant when they carry the same ordering, however different their formulas
# look, so the distance clustered on is $1 - |\rho_s|$ and the absolute value treats a feature and
# its negation as the same thing. The tree is cut at the rank correlation the configuration
# declares.
#
# One cluster is known in advance and is declared in the register rather than discovered here:
# `slot_of_day`, `is_us_hours` and `pays_swap_night` are the *same* column on this grid, because
# the hold that crosses midnight is exactly the 16:00 one. All three are kept because they carry
# different meanings to a reader and because the tree, not the author, is what decides. Read the
# whole figure as a screen for duplication and nothing stronger: which member of a cluster to keep
# needs a criterion measured out of sample, and `05_evaluation` measures one fold by fold.

# %%
clusters = plot_redundancy_clusters(
    development,
    feature_cols,
    cut=REDUNDANCY_CUT,
    title="The neighbouring horizons cluster; the state flags form one group of their own",
    subtitle=(
        r"Average linkage on $1 - |\rho_s|$, cut at a rank correlation of " f"{REDUNDANCY_CUT}"
    ),
    alt=(
        "Dendrogram of every feature in the matrix. The slot returns at neighbouring horizons "
        "join into one cluster with the acceleration columns built from them, the moving-average "
        "and channel columns join into another, and the two volatility estimators into a third. "
        "The swap and calendar flags sit together, and the three cross-asset families sit apart "
        "from everything else."
    ),
)
print(f"{len(set(clusters.values()))} redundancy clusters at the drawn cut, for {len(feature_cols)} features")
_cost_group = {c: clusters[c] for c in ("slot_of_day", "is_us_hours", "pays_swap_night") if c in clusters}
print(f"the three cost-state columns land in clusters {_cost_group} - one cluster if the tree agrees they are one column")

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ### F.3 Persistence
#
# The book is re-decided every slot, three times a day, so the question is how long a value lasts
# against that cadence: a feature whose value has decayed before the next rebalance cannot support
# it, however well it predicts the moment it is computed.
#
# **`case_studies.utils.feature_engineering.plot_persistence` is not used here, and the reason is
# the fourth silent cross-sectional failure this bot has had to route around.** Its right-hand
# panel is a *cross-sectional* rank correlation between consecutive decision dates:
# `group_by(timestamp).agg(pl.corr(rank, rank_prev, method="spearman"))`. With one instrument every
# group has one row, a Spearman over n = 1 is null, every column's stability comes back `nan`, and
# `np.nanmin` over an all-NaN vector then makes the axis limit NaN - so on this panel the helper
# does not return a wrong number, it raises `Axis limits cannot be NaN or Inf`. That is the loudest
# of the four (the other three - `family_coverage`'s `min_cross_section`, the registry's hard-coded
# `min_obs=5` and `exness_fx_d1/05_evaluation`'s `MIN_PERIODS` - all return an empty result
# instead), and it is recorded in `bots/exness_btc_8h/BOT.md` for the maintainer.
#
# What replaces it is the same question asked of one series: the feature's own autocorrelation on
# the decision grid, and the **half-life** read off it - the first lag at which the autocorrelation
# falls below one half. A half-life shorter than the rebalance cycle means the value is gone before
# it can be traded; a half-life of weeks means the column is a state variable and not a signal.

# %%
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt  # noqa: E402

PERSISTENCE_COLUMNS = [
    "slot_ret_1",
    "slot_ret_21",
    "slot_zscore_63",
    "d1_vol_gk_63d",
    "d1_ret_252d",
    "gold_beta_63d",
]
_lags = np.unique(np.linspace(1, PERSISTENCE_HORIZON, 24).astype(int))
_series = development.sort("timestamp")
_curves, _halflife = {}, {}
for column in PERSISTENCE_COLUMNS:
    values = _series[column].to_numpy().astype(float)
    curve = []
    for lag in _lags:
        a, b = values[lag:], values[:-lag]
        ok = ~(np.isnan(a) | np.isnan(b))
        curve.append(float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 else np.nan)
    _curves[column] = curve
    below = [int(lag) for lag, rho in zip(_lags, curve, strict=True) if rho < 0.5]
    _halflife[column] = below[0] if below else None

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for column, colour in zip(PERSISTENCE_COLUMNS, list(COLORS.values()), strict=False):
    ax.plot(_lags, _curves[column], "-", lw=1.5, color=colour, label=column)
ax.axhline(0.5, color=COLORS["slate"], ls=":", lw=1.2)
ax.axhline(0.0, color=COLORS["neutral"], lw=0.8)
ax.axvline(CYCLE, color=COLORS["copper"], ls="--", lw=1.2)
ax.set_xlabel(f"Lag in decision slots (the book is re-decided every {CYCLE})")
ax.set_ylabel("Autocorrelation of the feature")
ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
add_message_title(
    ax,
    "The one-slot return is gone before the next decision; the daily state survives weeks",
    subtitle="Feature autocorrelation on the decision grid, development window, one instrument",
)
show_with_alt(
    fig,
    "Six autocorrelation curves against lag in decision slots out to 126. The daily volatility, "
    "the yearly daily return and the gold beta decay slowly and are still well above one half at "
    "the right edge; the 63-slot return z-score falls through one half within a few dozen slots; "
    "the 21-slot return follows it; the one-slot return is at about zero from the first lag. A "
    "dotted line marks the half level and a dashed vertical line the rebalance cycle.",
)
display(
    pl.DataFrame(
        [
            {
                "column": column,
                "acf at lag 1": round(_curves[column][0], 4),
                f"acf at the cycle ({CYCLE})": round(_curves[column][0], 4),
                "half-life (slots)": _halflife[column],
                "half-life (days)": (
                    None if _halflife[column] is None else round(_halflife[column] / SLOTS_PER_DAY, 1)
                ),
            }
            for column in PERSISTENCE_COLUMNS
        ]
    )
)
print(
    "A half-life of `None` means the autocorrelation never falls below one half inside "
    f"{PERSISTENCE_HORIZON} slots ({PERSISTENCE_HORIZON / SLOTS_PER_DAY:.0f} days): the column is a "
    "state variable, and a model reading it is reading a regime rather than an event."
)

# ## G. Emit
#
# The parquet is written with a sidecar beside it recording a hash of the table's contents, its row
# count, its key columns, and the same hash taken over each price panel it was built from. The hash
# is over values rather than bytes, so re-writing the parquet or changing its compression leaves it
# alone while any change to a single feature value moves it.

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
inputs = {
    "decision_panel": value_digest(panel.select(["symbol", "timestamp", "close"])),
    "load_mt5_bars:8h": value_digest(bars),
    "load_mt5_bars:daily": value_digest(d1),
}
if d1_reference is not None:
    inputs["load_mt5_bars:daily:reference"] = value_digest(d1_reference)
record = write_artifact(
    features,
    FEATURES_DIR / "financial.parquet",
    keys=["symbol", "timestamp"],
    written_by="case_studies/exness_btc_8h/03_financial_features.py",
    inputs=inputs,
)
print(f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')} under digest {record['digest']}")
print(f"input digests: {inputs}")

# %% [markdown]
# ## Key takeaways
#
# - **Declare how far back each feature reads, and on which grid, before writing it.** The
#   configuration holds one lookback and one lag per family, and the warmup audit, the timing
#   figure and the register table all read the declared numbers instead of re-deriving them.
# - **A two-grid design pays only where the coarse grid has more history.** Here the cross-asset
#   partners do and the instrument itself does not, which is why the longest chain in the matrix -
#   and therefore the null-policy carrier - is one of its own daily columns.
# - **A backward asof join on the bar's close is the whole point-in-time content of a mixed-grid
#   feature.** Joining on the calendar date instead passes every value check and fails only the
#   rebuild-at-the-instant test, which is why that test exists.
# - **A cost that is knowable in advance is a feature.** The broker charges financing on one slot
#   in three and triples it on Friday; that is a state column, checked against the broker's own
#   rule, not a footnote in the cost stage.
# - **On one instrument, absence is a design decision.** Every cross-sectional mechanism is
#   removed, `features.ranked` is empty, and a test asserts that no ranked column reached the
#   matrix - because a percentile over one name is a constant, and a constant regressor is worse
#   than a weak one.
#
# ### Known limitations
#
# - The US index family is null on the first nine months of the development window. It is declared,
#   capped and measured, but it still means the oldest fold trains on a matrix one family short.
# - `pays_swap_night` is exactly `slot_of_day == 16` on an unbroken grid, so the two are the same
#   column. They would stop being the same only if the broker moved its rollover or the grid
#   acquired a hole; both are checked elsewhere, and neither is expected.
# - Every cost number the state flags describe comes from the **demo** login. The real Pro
#   account's swap is still unread, and the demo's zero on the short side is unusual enough to
#   treat as unverified.
#
# **Next**: the model-based features, and then the evaluation that measures whether any of this
# predicts.
