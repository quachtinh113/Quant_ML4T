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
# # Exness FX D1 (exness_fx_d1): Feature Engineering
#
# Five dollar pairs, one decision a day, taken at the close of the last MT5 four-hour bar before
# the New York 5PM rollover - the 16:00-20:00 UTC bar on this UTC+0 server. Every column below
# answers one question: at the moment the position is decided, which bars are already on the
# tape, and what does the feature make of them?
#
# ## Learning objectives
#
# - State how far back a feature reads and how long its inputs take to publish, before writing
#   its code
# - Aggregate intraday bars onto the decision calendar so a daily feature ends at the decision
#   bar it is decided on, and no later
# - Show that withholding later dates leaves every value unchanged, which separates a trailing
#   statistic from one fitted over the whole sample, and that rebuilding the panel from the bars
#   that had closed at a decision instant reproduces the batch row for that instant
# - Read a feature set for scale, dispersion, redundancy and decay before any model sees it
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Sections 8.2-8.4 (price and volume features, structural cross-instrument
# features). This is a Route B fork of [`fx_pairs/03_financial_features`](../fx_pairs/03_financial_features.ipynb)
# for the bot `exness_fx_d1` (`bots/exness_fx_d1/BOT.md`): the same register, the MT5 history
# through `bots/_shared/mt5_loader.py` instead of OANDA, and one addition the asset profiles
# name, a gold factor from `XAUUSD` on the same decision bar. It reads four-hour MT5 bars and
# its settings from `config/setup.yaml`, and reads nothing another notebook wrote. The
# construction itself lives in `_features.py` beside this file, because the point-in-time test
# (`bots/exness_fx_d1/tests/test_lookahead.py`), the backtest price loader and the deployment
# loop all have to compute exactly what this notebook computes. It writes
# `features/financial.parquet` with a `.digest.json` sidecar beside it.
# [`05_evaluation`](05_evaluation.ipynb) reads that parquet and tests fold by fold whether any
# column predicts. The modelling notebooks reach it through the shared loader
# `utils.modeling.load_modeling_dataset`, which joins it to the labels and to the fold-aware
# features [`04_model_based_features`](04_model_based_features.ipynb) builds from the same prices.

# %%
"""exness_fx_d1: Feature Engineering (Route B fork of fx_pairs)."""

import warnings
from datetime import date, time, timedelta

import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_fx_d1._features import (
    GOLD_SYMBOL,
    build_features,
    feature_columns,
    features_as_of,
    load_session_panel,
    warmup_expectations,
)
from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.feature_engineering import (
    assert_values_agree,
    assign_families,
    families_from_config,
    family_coverage,
    plot_coverage_through_time,
    plot_cross_sectional_dispersion,
    plot_feature_distributions,
    plot_persistence,
    plot_redundancy_clusters,
    plot_timing_contract,
    register_frame,
    warmup_audit,
)
from utils.artifact_specs import resolve_label_horizon
from utils.paths import display_path, get_case_study_dir

warnings.filterwarnings("ignore")

CASE_STUDY_ID = "exness_fx_d1"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
FEATURES_DIR = CASE_DIR / "features"

# %% [markdown]
# Two settings are left open for whoever runs the notebook. `START_DATE` of `None` starts at
# `setup.yaml::universe.history_start`, the first server day on which every pair carries the
# full six-bar four-hour grid, which is what the production run does; a shorter run overrides
# it through Papermill. `N_RECOMPUTE_DATES` is how many decision instants Section D.4 rebuilds
# the panel at from the bars that had closed by then. There is no cap on how many pairs are
# loaded, because the percentile columns need a cross-section to rank within and this universe
# is only five pairs wide to begin with.

# %% tags=["parameters"]
START_DATE = None
N_RECOMPUTE_DATES = 4

# %% [markdown]
# ## Configuration
#
# Every window, the list of columns that get a percentile, the session calendar and the holdout
# boundary live in `config/setup.yaml` and are bound here. The rest of the pipeline reads the same
# file, so a number retyped in this notebook would be a second answer to a question that already
# has one. What each of them decides is said where it is used; the four printed below set the
# shape of everything that follows.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
FAMILIES = families_from_config(setup)
WINDOWS = setup["features"]["windows"]
RANKED = setup["features"]["ranked"]
CARRIER = setup["features"]["null_policy_carrier"]
PERSISTENCE_HORIZON = setup["features"]["persistence_horizon"]
REDUNDANCY_CUT = setup["features"]["redundancy_cut"]
SESSION_CALENDAR = setup["decision"]["session_calendar"]
SNAPSHOT_UTC = time.fromisoformat(setup["decision"]["snapshot_utc"])
PERIODS_PER_YEAR = setup["evaluation"]["periods_per_year"]
HOLDOUT_START = date.fromisoformat(str(setup["evaluation"]["holdout_start"]))
HOLDOUT_END = date.fromisoformat(str(setup["evaluation"]["holdout_end"]))
DEVELOPMENT_END = HOLDOUT_START - timedelta(days=1)  # the last calendar day any diagnostic may read
UNIVERSE = sorted(setup["universe"]["symbols"])
HISTORY_START = START_DATE or str(setup["universe"]["history_start"])
CYCLE = int(resolve_label_horizon(CASE_STUDY_ID, setup["labels"]["primary"], setup).rstrip("Dd"))

print(f"{len(FAMILIES)} feature families are declared, and Section A prints what each one claims")
print(f"The position is re-ranked every {CYCLE} session(s), the gap a feature has to survive")
print(f"Rows begin where {CARRIER} can first hold a value, the longest warmup in the matrix")
print(f"Dates from {HOLDOUT_START} are the holdout; Section D rebuilds the panel without them")

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The strategy is a long-short rank rebalance over five pairs, so the hypothesis has to be a
# cross-sectional one: a pair stretched against its own recent range comes back relative to the
# rest, and how far it can stretch depends on the state the market is in.
#
# That splits the matrix in two, and the register's `role` column is where the split is written
# down. A **signal** column is one the ranking may be formed on: the trailing z-score of a
# multi-horizon return, and where price sits in the channel it has recently traded through. Which
# horizon the effect lives at is an empirical question, so the matrix carries three of each,
# alongside the returns they are standardized from - mean reversion and momentum are the same
# measurement read with opposite sign. A **state** column describes the environment the ranking is
# formed in and is never ranked on: volatility, distance below a trailing peak, the width of the
# daily range, the broad dollar, and gold - the one cross-instrument input this data set carries
# (`bots/assets/AUDUSD.md` names it; the oil feature `bots/assets/USDCAD.md` names has no
# instrument in the MT5 download, and carry needs policy rates the macro download does not hold,
# see the known limitations). Nothing in a column's values says which of the two it is, so the
# role is declared here rather than inferred later.
#
# How a quantity is represented matters as much as the quantity, so eight of the signal and state
# columns are carried a second time as percentiles within the decision date, which are comparable
# across dates in a way a level is not.
#
# The register is declared in `config/setup.yaml`, one row per family: what it reads, how far
# back, with what delay. Every lag is zero - nothing here waits for a publication - and in this
# market that is a claim about the session calendar and the decision bar rather than about the
# vendor, which is what Section B checks.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "driver hypothesis", "inputs", "lookback (bars)", "lag (bars)", "frame"]
)

# %% [markdown]
# ## B. Inputs and their observability
#
# Spot FX runs continuously from Sunday evening to Friday evening, so a "day" is a convention
# rather than a fact, and the convention has to be the one the decision is taken on.
# `config/setup.yaml` fixes it at the New York 5PM rollover, names the venue calendar that
# implements it - the same calendar `02_labels` aggregates on - and names the decision bar: the
# last four-hour bar that *closes* at or before the session's declared close, the 16:00-20:00 UTC
# bar on this server (`decision.snapshot`). The bar that opens at 20:00 UTC closes after the
# rollover in both seasons, so a rule that took the latest *open* in the session would read a
# price from after the cut-off; `01_feasibility_analysis` measured that on the real server and
# the closing condition is the one every stage applies.
#
# That aggregation is where a daily FX feature leaks if it is going to, so it is one function,
# `_features.session_panel`, and its rule is asserted rather than described. A bar belongs to the
# session whose declared close is the first at or after the bar's close, so every bar sits in
# exactly one session and a session is everything that printed between two consecutive decision
# instants; the session's `close` is the decision bar's close, the one `02_labels` sealed the
# labels on. The 20:00-24:00 UTC bar therefore opens the *next* session, so that session's `open`
# is the price the previous decision was executed at (`decision.execution_delay: next_bar_open`)
# and its high and low cover the thin post-New York hours. A pair-session whose last bar closes
# more than one bar ahead of the declared close had no fresh price at the decision (a server
# outage: 2018-01-31, the 16:00 UTC bar missing on every pair) and leaves the panel, reported
# rather than decided on a stale bar. `decision_ts` is kept through this section because it is
# the instant every feature has to be knowable at, and Section D.4 rebuilds the panel at a
# sample of them.

# %%
prices = load_session_panel(
    setup, symbols=UNIVERSE, start_date=HISTORY_START, end_date=str(HOLDOUT_END), keep_decision_ts=True
)
assert prices.filter(pl.col("decision_ts").dt.date() != pl.col("timestamp")).height == 0, (
    "a session's decision bar closes on a date other than its own"
)
assert prices.filter(pl.col("decision_ts").dt.time() > SNAPSHOT_UTC).height == 0, (
    f"a decision bar closes after the {SNAPSHOT_UTC} UTC snapshot setup.yaml declares"
)
# The Garman-Klass variance proxy is non-negative only for a bar whose high and low bracket its
# open and close. A malformed bar would make the window mean negative and its square root arrive
# as a silent null, so the input is checked here rather than the output guarded downstream.
assert prices.select(
    (
        (pl.col("high") >= pl.max_horizontal("open", "close"))
        & (pl.col("low") <= pl.min_horizontal("open", "close"))
    ).all()
).item(), "an OHLC bar does not bracket its own open and close"
decision_instants = prices.select("timestamp", "decision_ts").unique().sort("timestamp")
prices = prices.drop("decision_ts")
snapshot_hours = decision_instants["decision_ts"].dt.hour().value_counts().sort("decision_ts")
print(f"{len(prices):,} sessions over {prices['symbol'].n_unique()} pairs")
print(f"{prices['timestamp'].min()} to {prices['timestamp'].max()}")
print("decision bar closes at (UTC hour: sessions):")
for row in snapshot_hours.iter_rows(named=True):
    print(f"  {row['decision_ts']:02d}:00  {row['count']:,}")

# %% [markdown]
# Gold comes through the same function on the same calendar. `XAUUSD` keeps its own hours on this
# server - a daily break after the New York close, and it does not trade on Good Friday when the
# currency pairs do - so it has fewer sessions than the pairs, and the gold family carries its
# close onto the pairs' session grid as the last print at or before each decision instant. That
# is what a trader at the decision could see, and it is a forward-filled *price*: the return the
# family reads is measured on the aligned series afterwards, never forward-filled itself.

# %%
gold = load_session_panel(setup, symbols=[GOLD_SYMBOL], start_date=HISTORY_START, end_date=str(HOLDOUT_END))
missing_gold = prices.select("timestamp").unique().join(gold.select("timestamp"), on="timestamp", how="anti")
print(f"{GOLD_SYMBOL}: {len(gold):,} sessions, {gold['timestamp'].min()} to {gold['timestamp'].max()}")
print(f"  {missing_gold.height} pair sessions without a gold session of their own:")
print(f"  {sorted(str(d) for d in missing_gold['timestamp'].to_list())}")

# %% [markdown]
# The five pairs are not five independent markets. Every one of them has the dollar on one leg
# and shares whatever the dollar does, which is the common component Section C.5 measures and the
# reason `01_feasibility_analysis` counted 2.4 independent bets among five names. Which leg the
# dollar is on matters and is not the same across the five: it is the second currency in `EURUSD`
# and the first in `USDJPY`, so the same dollar move pushes those two quotes in opposite
# directions, and C.5 has to sign each of them before it can average them. Every pair has the same
# number of sessions, so the panel is balanced and no feature below has to reason about a pair
# joining or leaving part-way through.
#
# The range column is in basis points, hundredths of a percent, which is already a division by the
# price level - and that division is why every feature below is a return or a ratio rather than a
# price. `USDJPY` is quoted two orders of magnitude above the others, so a move of one unit means
# something different in each of them, and the table shows that even after scaling they do not
# travel the same distance in a session.

# %%
_base, _quote = pl.col("symbol").str.head(3), pl.col("symbol").str.tail(3)
universe = (
    prices.with_columns(
        pl.when(_base == "USD")
        .then(pl.lit("dollar first"))
        .when(_quote == "USD")
        .then(pl.lit("dollar second"))
        .otherwise(pl.lit("cross"))
        .alias("block"),
        (1e4 * (pl.col("high") - pl.col("low")) / pl.col("close")).alias("_range_bp"),
    )
    .group_by(["symbol", "block"])
    .agg(
        pl.len().alias("sessions"),
        pl.col("_range_bp").median().round(0).alias("median daily range (bp)"),
    )
    .sort("symbol")
)
with pl.Config(tbl_rows=universe.height):
    display(universe)

# %% [markdown]
# ## C. Feature construction, one subsection per family
#
# The code for every family is in `_features.py` and is imported above rather than written in
# a cell, and the reason is the one Chapter 25 (Section 25.1) gives: a feature that exists in two
# implementations - one in the research notebook, one in the live loop - agrees on the day the
# second is written and drifts on the first edit, and nothing in either output says so. The
# point-in-time test in `bots/exness_fx_d1/tests/test_lookahead.py`, the backtest price loader
# and the deployment loop's "recompute features" step all import the same functions this
# notebook calls. Each subsection says what its function computes and why; the function is the
# authority on how.
#
# ### C.1 Momentum, volatility and their differences
#
# Three quantities come out of the same trailing window and a shared helper computes them
# together: the return over the window, the annualized standard deviation of the daily log returns
# inside it, and the ratio of the two, which is a trailing Sharpe ratio. Reading the same window
# three ways separates a move that was steady from one that was a single jump.
#
# What stays local is specific to this universe. `mom_skip_recent` is the return over a year that
# stops a month short of the decision, because the most recent month of an FX move is the part
# most likely to reverse, and leaving it out asks whether the slower trend is still there.
# `accel_21_63` and `accel_63_126` are differences between horizons, and say whether a trend is
# speeding up or bleeding away. (`_features.momentum_features`)
#
# ### C.2 Garman-Klass volatility
#
# A close-to-close deviation sees two prices a session apart and discards everything that happened
# between them, so a session that travelled a long way and came back reads as quiet. The
# Garman-Klass estimator uses all four prices in the bar, combining the high-to-low range with the
# open-to-close move:
#
# $$\sigma^2_{GK} = \tfrac{1}{2}\left(\ln\tfrac{H}{L}\right)^2 - (2\ln 2 - 1)\left(\ln\tfrac{C}{O}\right)^2$$
#
# For a session of the same true volatility that is several times less noisy than the
# close-to-close estimate, which is why Chapter 8 uses it here: the range is where most of an FX
# session's information about dispersion sits. The columns carry the square root of the window
# mean of that variance, annualized, so they are on the same scale as the close-to-close deviation
# beside them. The two ratios read the regime rather than the level - a short window against a
# longer one says whether dispersion is rising or falling, which no single window can. On this
# panel the session's high and low cover every bar between two decision instants, the thin
# post-New York hours included, which is what the by-close assignment in Section B is for.
# (`_features.volatility_features`)
#
# ### C.3 Mean reversion: z-scores, channel position and Bollinger %B
#
# These are the columns the ranking is formed on. Each multi-horizon return is standardized
# against its own trailing year, so the z-score says how unusual the move is for that pair rather
# than how large it is in price terms. The universe table above is the reason that matters: the
# same move is unremarkable in a pair that travels the widest daily range here and a large one in
# the pair that travels the narrowest, and standardizing each pair against its own history is what
# makes the two comparable on the day the ranking is formed.
#
# Channel position and Bollinger %B ask the same question of the level instead of the change:
# where does price sit in the range it has recently traded through? Channel position measures that
# against the highest and lowest close of the window, so it reads zero at the bottom of the range
# and one at the top. Bollinger bands replace the range with a moving average of the close plus
# and minus two standard deviations of it, so the band widens when the pair is volatile instead of
# staying pinned by one old extreme for the length of the window. %B is where the close sits
# between the two bands, on the same zero-to-one reading:
#
# $$\%B = \frac{C - (\mu - 2\sigma)}{4\sigma}$$
#
# It leaves that interval whenever price is more than two deviations away from the average, which
# is why the column runs a little below zero and a little above one rather than stopping at the
# ends. All three are trailing statistics within one pair, so nothing here is estimated across
# pairs or across the sample. (`_features.mean_reversion_features`)
#
# ### C.4 Drawdown, range and oscillators
#
# `max_dd_63d` is the share by which price sits below its highest close of the trailing quarter,
# so it is zero at a new high and negative otherwise. That is the *current* drawdown and not the
# worst peak-to-trough decline anywhere inside the window: the two are different statistics, and a
# pair that sold off early in the quarter and then made the whole move back scores nothing on the
# current drawdown and its full decline on the worst one.
#
# The relative strength index, RSI, compares how much price gained on the sessions that closed up
# with how much it lost on the sessions that closed down over a trailing window, and maps the
# comparison onto a nought-to-hundred scale where fifty is as much up move as down. The averaging
# is Wilder's recursive one, which folds each new session into the running average with weight
# `1/period` rather than taking a plain mean of the last `period` sessions; the two give different
# numbers on the same data, and Wilder's is the convention the library implements.
#
# The last two families are plainer. `avg_range_*` is the high-to-low range as a share of the
# close, averaged over the window, which says how far a pair typically travels within a session.
# `price_to_ma_*` is how far price sits above or below its own moving average.
# (`_features.drawdown_range_and_oscillators`)
#
# ### C.5 The dollar factor
#
# All five pairs have the dollar on one leg and share a common component that is not
# pair-specific. The proxy is a signed average of the five returns, and the sign is what Section
# B flagged: a rise in `EURUSD` is a falling dollar and a rise in `USDJPY` is a rising one, so a
# pair quoting the dollar second enters the average with its return negated. The five are found
# from the symbols themselves rather than from a list typed here, so the proxy follows the universe.
#
# It is a transparent average and not an estimated factor: no loading is fitted and nothing is
# regressed. The average is over log returns, so accumulating it over a horizon is a sum and the
# aggregates are the proxy's own cumulative move rather than an approximation of one.
#
# How much each pair moves with the proxy is then a rolling correlation over the trailing year,
# reading the pair's history and the proxy's, both ending at the decision. A correlation and not a
# regression slope, which is what the column name has to say: `usd_corr_63d` runs between minus
# one and one and reports how *reliably* the pair tracks the dollar, where a beta would report by
# how much and would move with the pair's own volatility whether or not the relationship changed.
# With every pair in the average, the proxy is close to the universe's own first principal
# component, and a pair's correlation with it says how much of that common move it carries.
# (`_features.dollar_factor`)
#
# ### C.6 The gold factor
#
# The one cross-instrument input this data set carries (Chapter 8, Section 8.3). `gold_ret_21d`
# and `gold_ret_63d` are gold's trailing returns on the aligned series Section B built, the same
# value for every pair on a session - a market state, not a ranking column - and `gold_corr_*` is
# each pair's rolling correlation with them over the trailing year, the same construction as the
# dollar exposure. The commodity currencies are the reason it is here: `AUDUSD` tracks gold through
# Australia's export mix and `USDCAD` moves against it, and the correlation column says whether
# that link is currently holding. (`_features.gold_factor`)
#
# ### C.7 Cross-sectional position, and the whole construction
#
# The ranked columns become percentiles within their own decision date. The strategy is long-short
# over a ranking, so relative standing is what it can act on, and a percentile is comparable across
# dates in a way a level is not. The partition is the decision timestamp and nothing else, so the
# statistic reads five rows and no other date. `build_features` is the whole construction as one
# function, and every consumer calls that one function.

# %%
built = build_features(
    prices, windows=WINDOWS, ranked=RANKED, periods_per_year=PERIODS_PER_YEAR, gold=gold
)
feature_cols = feature_columns(built)
print(f"{len(built):,} sessions carrying {len(feature_cols)} features")

# %% [markdown]
# ## D. The timing contract
#
# Each feature makes two promises about time: how many bars back it reads, and how long its inputs
# take to become available after the period they describe. Together those fix the earliest decision
# the feature can be used at, and they are what the register declares in Section A. This section
# checks all four - what the constructions actually read, that no column fills before its window
# could have, that no value depends on a date after it, and that the value a decision instant
# would have seen is the value the batch panel holds.
#
# ### D.1 What each construction reads
#
# Three kinds of operation appear above. A **rolling** window ends at its own row and reads a fixed
# number of bars backward within one pair. A **cross-sectional** statistic - the percentiles and
# the dollar proxy - is taken over the decision timestamp, so it reads every pair on that date and
# no other. A **join** broadcasts the dollar proxy and the aligned gold series back to the panel
# and is followed by an explicit sort, so the rolling correlation that reads it sees the order it
# assumes.
#
# The figure below draws the register: each bar runs leftward from the decision by the number of
# sessions that family reads, and stops where its oldest input sits. What to look for is the right
# edge. A family whose input is published with a delay - a survey, a restated fundamental, an
# exchange file that lands the next morning - would end short of the line by the length of that
# delay, and everything between the two would be information the decision cannot have. Every input
# here is a bar that had closed at the 20:00 UTC decision, so every bar reaches the line.

# %%
plot_timing_contract(
    FAMILIES,
    bar_unit="trading sessions",
    title="Nothing waits to publish: every family reads up to the decision",
    subtitle="Register lookback per family; a gap at the right edge would be an information lag",
    alt=(
        "Horizontal bars, one per family, each extending leftward from the decision line by that "
        "family's declared lookback - shortest for range and drawdown, longest for the dollar and "
        "gold factors and mean reversion. Every bar reaches the line, so no family shows a lag gap."
    ),
)

# %% [markdown]
# ### D.2 Warmup
#
# A trailing window cannot produce a value until it has enough bars to fill. The audit checks that
# length rather than describing it: a column carrying a value before its window could have filled
# is reading bars that do not exist, and that is what it raises on. The three longest are chains
# rather than single windows - `zscore_126d` standardizes a return against a further year of them,
# and `usd_corr_63d` and `gold_corr_63d` correlate a horizon return over a year of sessions. The
# census it returns shows, per column, the warmup declared for it and the bar it actually first
# held a value on; a column may fill later than its window if the pair's own history is short, and
# never earlier. The declared numbers come from the window register through
# `_features.warmup_expectations`, the same dictionary the point-in-time test audits.

# %%
warmup_audit(built, warmup_expectations(WINDOWS), entity="symbol")

# %% [markdown]
# ### D.3 Withholding the holdout changes nothing
#
# Trailing and within-date statistics share a property worth checking directly: recomputed on a
# panel that stops before the holdout, they reproduce the same values on the rows the two panels
# share. A parameter fitted over a whole column - a winsorization bound, a scaler, an encoder -
# does not, because truncating the column moves the parameter and with it every row it was applied
# to. Building the panel twice and comparing tests the whole construction at once, every emitted
# column rather than a sample, and does not depend on anyone having flagged the transform that
# fits. A value on one side against a null on the other counts as a difference, which is the form
# of failure a null-skipping comparison hides.
#
# The comparison raises if any column moves, so reaching the next cell is the result. Four rows of
# it are shown - one trailing statistic, one percentile taken within a date, two rolling
# correlations - each reporting how many rows were compared and the largest gap found between the
# two builds.

# %%
seal = assert_values_agree(
    built.filter(pl.col("timestamp") < HOLDOUT_START),
    build_features(
        prices.filter(pl.col("timestamp") < HOLDOUT_START),
        windows=WINDOWS,
        ranked=RANKED,
        periods_per_year=PERIODS_PER_YEAR,
        gold=gold.filter(pl.col("timestamp") < HOLDOUT_START),
    ),
    columns=feature_cols,
    keys=["timestamp", "symbol"],
)
seal.filter(pl.col("column").is_in(["zscore_126d", "rank_ret_126d", "usd_corr_63d", "gold_corr_63d"]))

# %% [markdown]
# ### D.4 Rebuilding the panel at the decision instant
#
# D.3 withholds whole dates. The stronger form of the same check withholds *bars*: at a decision
# instant, only the bars that had closed by then were on the tape, and a feature computed from
# those bars alone must equal the batch value for that session. That is a claim about the
# aggregation as much as about the features - a session that absorbed a bar closing after the
# decision would pass D.3 and fail here - so `_features.features_as_of` truncates the raw
# four-hour bars, both the pairs' and gold's, at the decision bar's close, rebuilds the session
# panel and the whole matrix, and the row for that session is compared column by column with the
# batch row. A handful of instants across the development history is enough for a notebook;
# `bots/exness_fx_d1/tests/test_lookahead.py` runs the same comparison as a test, over more
# instants, on every build. The holdout is never sampled.

# %%
bars_4h = load_mt5_bars(
    "4h", symbols=UNIVERSE, start_date=HISTORY_START, end_date=str(DEVELOPMENT_END)
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
gold_4h = load_mt5_bars(
    "4h", symbols=[GOLD_SYMBOL], start_date=HISTORY_START, end_date=str(DEVELOPMENT_END)
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
dense_dates = (
    built.filter(pl.col(CARRIER).is_not_null() & (pl.col("timestamp") < HOLDOUT_START))["timestamp"]
    .unique()
    .sort()
)
step = max(1, len(dense_dates) // (N_RECOMPUTE_DATES + 1))
sampled = dense_dates.gather_every(step, offset=step)[:N_RECOMPUTE_DATES].to_list()
recompute_rows = []
for session in sampled:
    instant = decision_instants.filter(pl.col("timestamp") == session)["decision_ts"][0]
    as_of = features_as_of(bars_4h, instant, setup, gold_bars=gold_4h)
    assert as_of["timestamp"].max() == session, "the truncated panel ends on a different session"
    live = as_of.filter(pl.col("timestamp") == session).sort("symbol")
    batch = built.filter(pl.col("timestamp") == session).sort("symbol")
    gap = assert_values_agree(batch, live, columns=feature_cols, keys=["timestamp", "symbol"])
    recompute_rows.append(
        {
            "session": session,
            "decision (UTC)": instant,
            "bars seen": bars_4h.filter(pl.col("timestamp") + pl.duration(hours=4) <= instant).height,
            "pairs": live.height,
            "max abs difference": gap["max abs difference"].max(),
        }
    )
pl.DataFrame(recompute_rows)

# %% [markdown]
# ## E. Matrix assembly and coverage
#
# The panel key is `symbol` + `timestamp`. Raw OHLC, volume and the intermediate log return are
# excluded: they are the inputs the features are made of, and the log return dated `t` is the
# primary label dated one session earlier, so a model handed it beside that label would be reading
# an answer it was asked for one row before. The shortest trailing return the matrix ships spans a
# trading week.
#
# One null policy is applied once, and it is a cut rather than a fill: nothing is imputed,
# forward-filled or zero-filled, because a filled warmup value is a number the window never
# produced. A pair's rows start where the column with the longest chain of trailing windows can
# first hold a value - the 126-session z-score, which then needs a further trailing year to
# standardize against - and every shorter window has filled by then. The assertion below makes
# that a fact rather than an intention.
#
# The matrix is written through the holdout, because the holdout stages need a feature vintage
# for the sessions they score. Every figure from here on reads `development`, the rows before
# the holdout opens: a distribution or a redundancy tree drawn over the holdout would be a look
# at it, however innocent.

# %%
features = (
    built.select(["timestamp", "symbol", *feature_cols])
    .drop_nulls(subset=[CARRIER])
    .sort(["timestamp", "symbol"])
)
assert features.select(["timestamp", "symbol"]).is_duplicated().sum() == 0, "duplicate panel key"
assert features.null_count().sum_horizontal().item() == 0, "the null policy left a gap behind"
assignment = assign_families(feature_cols, FAMILIES)
development = features.filter(pl.col("timestamp") < HOLDOUT_START)
print(
    f"{len(feature_cols)} features, {len(features):,} rows, {features['symbol'].n_unique()} pairs"
)
print(f"{features['timestamp'].min()} to {features['timestamp'].max()}")
print(f"null policy dropped {len(built) - len(features):,} warmup rows")
print(f"{len(development):,} development rows before {HOLDOUT_START}, the rows every figure below reads")
register_frame(FAMILIES, feature_cols).select(["family", "columns", "role", "representation"])

# %% [markdown] tags=["results"]
# The matrix carries **57 features** on **10,390 rows** across **5 pairs**, from **2018-08-20** to
# **2026-08-31**. The null policy costs **1,885 rows**, every one of them inside the warmup stretch
# at the start of the sample; **9,095 rows** precede the holdout. The decision bar closes at 20:00
# UTC on 2,408 of the 2,455 sessions, at 16:00 UTC on 42 early-close sessions and at 12:00 UTC on 5.

# %% [markdown]
# ### F1. Coverage through time
#
# Drawn on the panel *before* the null policy, so the figure shows what the policy is for. Each
# family climbs to complete coverage as its longest window fills, and the dashed boundary marks
# the first session the emitted matrix keeps. Read the two halves separately: the ramp to the left
# of the boundary is the stretch the policy removes, and the flat run at one to the right of it is
# the assertion in the cell above, drawn month by month.

# %%
plot_coverage_through_time(
    family_coverage(
        built.filter(pl.col("timestamp") < HOLDOUT_START).select(["timestamp", *feature_cols]),
        assignment,
        every="1mo",
    ),
    warmup_boundary=features["timestamp"].min(),
    title="Each family fills as its longest window does, and never thins again",
    subtitle="Monthly non-null share per family before the policy, with the boundary drawn",
    alt=(
        "Non-null share by feature family, on an axis from zero to one. Every family starts at "
        "zero and climbs to one during the first eighteen months as its longest window fills, "
        "the short-window families first and mean reversion last. A dashed line in late 2018 marks "
        "where the null policy starts, and every line is flat at one from there to the end."
    ),
)

# %% [markdown]
# ## F. What the features look like
#
# Four properties decide whether this matrix can be used at all: the scale each feature arrives on,
# whether the cross-section disagrees enough to rank on, how much of the set is one ordering under
# several names, and how long a value lasts. `05_evaluation` is where the matrix is tested fold by
# fold for whether any of it predicts.
#
# ### F2. Feature distributions

# %%
plot_feature_distributions(
    development,
    ["zscore_21d", "zscore_63d", "zscore_126d"] + [f"channel_pos_{w}d" for w in WINDOWS["channel"]],
    title="Z-scores stay centred; channel position banks against its edges",
    subtitle="The mean-reversion family the ranking is formed on, display tails clipped",
    alt=(
        "Six histograms in two rows. The top row holds the trailing z-scores of the three "
        "horizon returns, each a broad single-peaked body roughly centred on zero. The bottom row "
        "holds channel position at the same windows, bounded by zero and one with mass banked "
        "against both ends."
    ),
)

# %% [markdown]
# ### F3. Cross-sectional dispersion through time
#
# A cross-sectional strategy needs the cross-section to disagree. On a date where the band narrows
# to nothing there is nothing to rank, whatever the average level of the feature. With five pairs
# the tenth and ninetieth percentiles are close to the extremes of the cross-section, so the band
# is what the two most stretched pairs disagree by. It never closes, which is the necessary
# condition rather than a promising one: a band this wide says a ranking exists to be formed, not
# that it predicts anything.

# %%
plot_cross_sectional_dispersion(
    development,
    "zscore_63d",
    every="1mo",
    title="The band never narrows: there is always a ranking to form",
    subtitle="Tenth to ninetieth percentile of the three-month return z-score across the pairs",
    alt=(
        "Shaded band of the tenth to ninetieth percentile of the three-month return z-score by "
        "month, with the median drawn through it. The band is between one and three units wide in "
        "every year of the sample and never approaches zero width. The median wanders between "
        "roughly minus one and plus one and a half, and shows no trend."
    ),
)

# %% [markdown]
# ### F5. Redundancy structure
#
# Two columns are redundant when they carry the same ordering, however different their formulas
# look, so the distance clustered on is $1 - |\rho_s|$: $\rho_s$ is the rank correlation between
# the pair of columns, and taking its absolute value treats a feature and its negation as the same
# thing. The tree is cut at the rank correlation the configuration declares, drawn as the dashed
# line and named in the subtitle. Linkage is average, so a cluster is a group whose members are
# that close to each other *on average*, not one in which every pair clears the threshold. Read it
# as a screen for duplication and nothing stronger: these columns largely agree about which pair is
# high and which is low, which is a reason to suspect they are one measurement under several names.
#
# What the tree groups on is the horizon rather than the family, as it did on the twenty-pair
# panel: the largest clusters are the short, medium and long windows, each mixing the return, the
# risk-adjusted return, the z-score, the channel position and the percentile columns at roughly one
# window. The volatility columns cluster with each other across every window rather than with the
# signals beside them, and the two market-level gold returns take the same value on every pair and
# so carry no cross-sectional ordering at all. Which member of a cluster to keep is a question the
# tree cannot answer, because it needs a criterion measured out of sample; `05_evaluation` measures
# one fold by fold.

# %%
clusters = plot_redundancy_clusters(
    development,
    feature_cols,
    cut=REDUNDANCY_CUT,
    title="Horizon groups the signal families; volatility groups on its own",
    subtitle=(
        r"Average linkage on $1 - |\rho_s|$, cut at a rank correlation of "
        f"{REDUNDANCY_CUT}"
    ),
    alt=(
        "Dendrogram of every feature in the matrix. Large clusters each hold one horizon and mix "
        "every signal family in it: the short-window return, risk-adjusted return, z-score, "
        "channel position, Bollinger %B, oscillator, trend ratio and percentile columns join "
        "together, and the medium and long windows do the same. A further cluster holds the "
        "volatility levels and the average range from every window. The dollar-factor "
        "aggregates, the gold returns, the volatility ratios and the acceleration columns sit "
        "apart from the horizon clusters."
    ),
)
print(f"{len(set(clusters.values()))} redundancy clusters at the drawn cut")

# %% [markdown] tags=["results"]
# Cutting the redundancy tree leaves **18 clusters** for **57 features**, so most of the matrix
# repeats an ordering another column already carries.

# %% [markdown]
# ### F6. Persistence and rank stability
#
# The cadence is daily, so the right panel compares the ordering across consecutive sessions, the
# schedule the strategy rebalances on. At that cadence the panel separates nothing - one session is
# short enough that every feature keeps its ordering - and the left panel is what carries the
# information: the autocorrelation of the feature itself, run out to a quarter. It is estimated per
# pair on sessions exactly one lag apart and summarized by the median over pairs, with a bootstrap
# interval over pairs; a correlation pooled over every pair-date would read high whenever pairs sit
# at different levels, whether or not any one of them persists. With five pairs the bootstrap
# interval over pairs is wide, and it is drawn that way rather than narrowed. Read that way the
# figure says which columns are state and which are signal, and it puts the short z-score at zero
# at a lag equal to its own window - a value that has nothing left to say about the one before it.

# %%
plot_persistence(
    development,
    ["zscore_21d", "zscore_126d", "channel_pos_63d", "vol_gk_63d", "rsi_14d"],
    entity="symbol",
    max_lag=PERSISTENCE_HORIZON,
    decision_dates=development["timestamp"].unique().sort().to_list(),
    title="Every feature survives a session; only the long windows survive a quarter",
    subtitle=f"Median over pairs; rank correlation across consecutive {CYCLE}-session rebalances",
    alt=(
        "Two panels. On the left, autocorrelation against lag out to a quarter. Three-month "
        "Garman-Klass volatility decays slowest and is still near a half at the right edge, the "
        "126-session z-score is around a third, and the 63-session channel position, the "
        "21-session z-score and the 14-day oscillator have all reached zero by then - the short "
        "z-score crossing zero at a lag equal to its own window and staying slightly negative. "
        "The bootstrap ribbons over five pairs are wide. On the right, the cross-sectional rank "
        "correlation between consecutive sessions is close to one for all five features, with "
        "volatility highest and the oscillator lowest."
    ),
)

# %% [markdown]
# ## G. Emit
#
# The parquet is written with a sidecar beside it, a small JSON file. The sidecar records a hash of
# the table's contents, alongside its row count, its key columns and the same hash taken over the
# two price panels the table was built from. The hash is over values rather than over the bytes of
# the file, so re-writing the parquet, reordering its rows or changing how it is compressed leaves
# the hash alone, while any change to a single feature value moves it. That is what makes "the
# artifact on disk is the one this code produces" a question anyone can settle by reading two
# files, instead of re-deriving the table and comparing it column by column.

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
record = write_artifact(
    features,
    FEATURES_DIR / "financial.parquet",
    keys=["symbol", "timestamp"],
    written_by="case_studies/exness_fx_d1/03_financial_features.py",
    inputs={
        "load_mt5_bars:4h": value_digest(prices),
        f"load_mt5_bars:4h:{GOLD_SYMBOL}": value_digest(gold),
    },
)
print(f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')} under digest {record['digest']}")

# %% [markdown]
# ## Key takeaways
#
# - **Declare how far back each feature reads before writing it.** The configuration holds one
#   lookback and one lag per family, and the warmup assertion, the timing figure and the register
#   table all read those declared numbers instead of re-deriving them from the code, so a window
#   that disagrees with what was promised raises rather than passing quietly.
# - **In a market that never closes, the aggregation is what makes a feature knowable.** A daily
#   feature is available at its decision only because every bar in the session had closed by the
#   decision bar's close, which is why Section B asserts that boundary and Section D.4 rebuilds
#   the panel from the bars that had closed rather than describing it.
# - **One construction, every consumer.** The functions this notebook calls are the ones the
#   point-in-time test, the backtest loader and the live loop call, so there is no second
#   implementation to drift.
# - **Check for a fitted transform by rebuilding, not by reading.** Recomputing the whole panel with
#   the later dates removed and comparing value by value catches anything estimated across the
#   sample, including the transforms nobody thought to flag.
# - **Read the matrix before modelling it.** Distribution, dispersion, redundancy and decay each
#   rule out a use: a feature with no cross-sectional spread cannot rank, and one whose ordering
#   decays inside the rebalance cycle cannot be traded at that cadence.
#
# ### Known limitations
#
# - There is no carry feature, and carry is the effect every asset profile in `bots/assets/` names
#   first. It needs the policy-rate differential between the two legs. `FRED_API_KEY` is present in
#   the Windows `.env`, but `data/macro/config.yaml` downloads United States series only (no ECB,
#   BoE, BoJ, RBA or BoC rate) and `data/macro/fred_macro.parquet` has not been downloaded, so
#   the register carries no carry family. TODO (user decision, one DSR trial when it is tried):
#   add the five foreign policy-rate series to the macro download, take each rate as of its
#   *publication* date (Chapter 8, Section 8.4; `04_fundamentals_macro_calendar`), and add a
#   `carry` family to `setup.yaml::features.families` before rebuilding this matrix. Until then
#   every conclusion drawn downstream is a conclusion about spot dynamics alone.
# - The oil feature `bots/assets/USDCAD.md` names has no instrument in the MT5 download (no CL
#   or WTI symbol was fetched), so it is absent; gold is the only cross-instrument input.
# - Session flags (`bots/_shared/sessions.py`) are constant at a fixed 20:00 UTC decision and
#   are not features of a daily bot; they belong to the session bots.
# - The dollar proxy is an equal-weighted signed average of the five pairs, not an estimated
#   factor, and with every pair in it the proxy and the cross-section are nearly one bet.
# - Volume on an MT5 feed is tick count at one broker rather than traded notional, so it is used
#   only through the daily range and not as a participation feature.
# - Every feature here is a rule written in advance. `04_model_based_features` adds the features
#   that are themselves model outputs, where the rule is estimated from the data.
