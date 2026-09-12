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
# > began as `case_studies/exness_fx_d1/03_financial_features.py` with the study id, the loader and the population names
# > changed. **This bot has TWO instruments (`US500`, `USTEC`) and no cross-section**, and one
# > label per experiment workspace. Any sentence below that speaks of five currency pairs, of
# > ranking assets against each other, of `usd_corr_*` / `gold_corr_*` columns or of a dollar
# > regime is template residue describing a different bot; where such prose asserted a NUMBER it
# > has been deleted rather than reworded (see `BOT.md`, Decisions log, 2026-09-08 third pass),
# > and where it survives as a description of the template it says so explicitly. Report any that
# > does neither. This banner follows `case_studies/xau_fx_mt5/13_backtest.py`.

# %% [markdown]
# # Exness US indices, session bot (exness_usidx_sess): Feature Engineering
#
# Two US index CFDs, two decisions per NYSE cash session, and therefore **two feature matrices**.
# This notebook builds one of them - the one belonging to the label its workspace declares - and
# every column in it answers the same question: at the moment the position is decided, which
# bars had closed, and what does the feature make of them?
#
# ## Learning objectives
#
# - State how far back a feature reads and how long its inputs take to publish, before writing
#   its code, and keep the two declarations in configuration where a test can read them
# - Reach a window the sample cannot warm up, by taking it from a coarser price file and paying
#   for it with a declared, asserted staleness rather than with an unstated one
# - Show that withholding later dates leaves every value unchanged, and that rebuilding the
#   matrix from the bars that had closed at a decision instant reproduces the batch row - on
#   **both** price files, because a two-frequency join is where a lookahead hides best
# - Read a feature set for scale, redundancy and decay before any model sees it
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Sections 8.2-8.4. This is a Route B fork of
# [`exness_fx_d1/03_financial_features`](../exness_fx_d1/03_financial_features.ipynb) for the bot
# `exness_usidx_sess` (`bots/exness_usidx_sess/BOT.md`), with three differences that follow from
# the market rather than from taste: the cross-sectional percentile family is gone (two
# instruments cannot be ranked, `setup.yaml::mapping.class`), two families only an index session
# has are added (the overnight / intraday decomposition and the opening range), and the long
# windows come from the **daily** MT5 file because this account's hourly history cannot warm
# them. It reads `config/setup.yaml`, one-hour and daily MT5 bars, and nothing another notebook
# wrote. It writes `features/financial.parquet` with a `.digest.json` sidecar beside it.
#
# **Which spec this run builds** is not a parameter of this notebook. The workspace declares one
# label in `setup.yaml::labels.primary`, and `labels.spec_of` maps it to a session panel. One
# workspace per spec is forced by `utils.modeling.load_modeling_dataset`, which reads exactly one
# `features/financial.parquet` per case study, and by the fact that the overnight panel's columns
# for a session are computed from bars inside the intraday label's own holding period. See
# `setup.yaml::labels` for the whole argument.

# %%
"""exness_usidx_sess: Feature Engineering (Route B fork of exness_fx_d1)."""

import warnings
from datetime import date, timedelta

import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_usidx_sess._features import (
    build_features,
    feature_columns,
    features_as_of,
    load_daily_bars,
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
    plot_feature_distributions,
    plot_persistence,
    plot_redundancy_clusters,
    plot_timing_contract,
    register_frame,
    warmup_audit,
)
from utils.paths import display_path, get_case_study_dir

warnings.filterwarnings("ignore")

CASE_STUDY_ID = "exness_usidx_sess"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
FEATURES_DIR = CASE_DIR / "features"

# %% [markdown]
# Two settings are left open. `START_DATE` of `None` starts at `setup.yaml::universe.history_start`
# - 2022-10-25, the first hourly bar this account serves for either index - which is what the
# production run does. `N_RECOMPUTE_DATES` is how many decision instants Section D.4 rebuilds the
# whole matrix at from the bars that had closed by then. There is no cap on how many indices are
# loaded: there are two, and one of the families is the ratio between them.

# %% tags=["parameters"]
START_DATE = None
N_RECOMPUTE_DATES = 4

# %% [markdown]
# ## Configuration
#
# Every window, the register, the session calendar, the label the workspace models and the
# holdout boundary live in `config/setup.yaml` and are bound here. A number retyped in this
# notebook would be a second answer to a question that already has one.
#
# The two window blocks are read separately and are counted in different units, which is the
# whole point of separating them. `features.windows` is counted in **sessions of this bot's own
# grid**, and nothing in it may exceed the 96 sessions of warmup that sit before the earliest
# fold - `01_feasibility_analysis` asserts that against the *composed* warmup, not against the
# largest single number. `features.daily_windows` is counted in **bars of the daily MT5 file**,
# which starts 2019-07-16 and has 950 bars in hand before the earliest fold, so a 252-bar window
# is warm three years before the panel opens.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
FAMILIES = families_from_config(setup)
WINDOWS = setup["features"]["windows"]
DAILY_WINDOWS = setup["features"]["daily_windows"]
RANKED = setup["features"]["ranked"]
CARRIER = setup["features"]["null_policy_carrier"]
PERSISTENCE_HORIZON = setup["features"]["persistence_horizon"]
REDUNDANCY_CUT = setup["features"]["redundancy_cut"]
PERIODS_PER_YEAR = setup["evaluation"]["periods_per_year"]
HOLDOUT_START = date.fromisoformat(str(setup["evaluation"]["holdout_start"]))
HOLDOUT_END = date.fromisoformat(str(setup["evaluation"]["holdout_end"]))
DEVELOPMENT_END = HOLDOUT_START - timedelta(days=1)  # the last calendar day any diagnostic may read
UNIVERSE = sorted(setup["universe"]["symbols"])
HISTORY_START = START_DATE or str(setup["universe"]["history_start"])
DAILY_HISTORY_START = str(setup["universe"]["daily_history_start"])
LABEL = setup["labels"]["primary"]
SPEC = setup["labels"]["spec_of"][LABEL]

print(f"This workspace models {LABEL}, which lives on the {SPEC!r} session panel")
print(f"{len(FAMILIES)} feature families are declared, and Section A prints what each one claims")
print(f"Rows begin where {CARRIER} can first hold a value, the longest warmup in the matrix")
print(f"Dates from {HOLDOUT_START} are the holdout; Section D rebuilds the matrix without them")

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The strategy is a per-index threshold on each index's own forecast, not a ranking, so the
# hypothesis has to be a time-series one: an index session is two processes - an overnight leg
# that carries the drift and an intraday leg where the opening auction is worked off and the
# closing auction is met - and how far either can travel depends on the state the market is in.
#
# That splits the matrix in two, and the register's `role` column is where the split is written
# down. A **signal** column is one the threshold may be formed on: the two legs of the session,
# the trailing returns and their z-scores, where price sits in the channel it has recently traded
# through. A **state** column describes the environment that signal is read in and is never the
# signal itself: realized volatility inside the session and inside the decision window, the
# distance below a trailing peak, the daily-grid regime columns, and the calendar. Nothing in a
# column's values says which of the two it is, so the role is declared rather than inferred.
#
# Two things in the register deserve reading twice. First, `lag`: it is zero for every family
# built on the hourly grid, because every one of their inputs is a bar that had closed at the
# decision, and **one** for every `d1_` family, because the daily bar covering the decision's own
# day has not finished printing at 14:00 or 20:00 UTC and the join therefore reaches back to the
# previous day. That is a declared staleness and Section B asserts it. Second, `lookback` for the
# `d1_` families is counted in daily bars and for the rest in sessions; the two are close but not
# equal, and the register says which frame each family lives in.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "driver hypothesis", "inputs", "lookback (bars)", "lag (bars)", "frame"]
)

# %% [markdown]
# ## B. Inputs and their observability
#
# ### B.1 The session panel
#
# An index CFD on this server runs almost around the clock, so a "session" is a convention, and
# the convention has to be the one the decision is taken on. `config/setup.yaml` fixes two
# decision instants per NYSE cash session and derives both from the `NYSE` calendar at run time,
# never from a table: `us_cash_open` is the close of the first hourly bar closing at or after the
# cash open plus thirty minutes, and `us_cash_close` the close of the last hourly bar closing at
# or before the cash close. New York daylight saving moves both by an hour and the 13:00 half
# days move the second by three, which is why a calendar and not a table decides.
#
# `_features.session_panel` is the one function that does the aggregation, and its rule is
# asserted rather than described: a bar belongs to the window whose decision instant is the first
# at or after the bar's close, so every bar sits in exactly one window and a window is everything
# that printed between two consecutive decisions. The panel's `close` is the decision bar's
# close - it is **not** the label endpoint, which rides separately as `label_close`, because a
# rolling window that included the current row would otherwise read each row's own answer.

# %%
prices = load_session_panel(
    setup, spec=SPEC, symbols=UNIVERSE, start_date=HISTORY_START, end_date=str(HOLDOUT_END)
)
assert prices.filter(pl.col("decision_ts").dt.date() != pl.col("timestamp")).height == 0, (
    "a session's decision bar closes on a date other than its own"
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
snapshot_hours = decision_instants["decision_ts"].dt.hour().value_counts().sort("decision_ts")
print(f"{len(prices):,} index-sessions over {prices['symbol'].n_unique()} indices, spec {SPEC!r}")
print(f"{prices['timestamp'].min()} to {prices['timestamp'].max()}")
print("decision bar closes at (UTC hour: sessions):")
for row in snapshot_hours.iter_rows(named=True):
    print(f"  {row['decision_ts']:02d}:00  {row['count']:,}")

# %% [markdown]
# ### B.2 The daily file, and the one hour of staleness it costs
#
# Five of the fifteen families cannot be built on the panel above, and the reason is arithmetic
# rather than preference. This account serves hourly bars from 2022-10-25 and the earliest fold
# trains from 2023-03-15, which leaves 96 sessions of warmup; a trailing year is 252. Every
# window on the session grid is therefore capped at 63, and a quarter is the longest trend this
# bot could otherwise see. The daily file goes back to 2019-07-16 and carries 2,191 bars per
# index, so the same year-long window is warm three years before the panel opens.
#
# What it costs is one day of freshness, and that is a fact about the bar rather than a modelling
# choice. A daily bar dated `d` on this server is exactly the 00:00-24:00 UTC aggregate of that
# day's hourly bars - checked below on the sample rather than asserted from the documentation -
# so it has finished printing at 00:00 UTC on `d + 1`. Both of this bot's decisions fall between
# 14:00 and 21:00 UTC, so the freshest daily bar either of them may read is dated the previous
# day. `_features.daily_factor` joins on that availability instant and not on the date, and the
# assertion below is what stops the join from ever reaching a bar dated on or after the decision.
#
# Saturday and Sunday rows are dropped first. A Sunday daily bar here holds only the two evening
# hours 22:00 and 23:00 UTC - the week re-opening, not a trading day - and leaving it in would
# put a two-hour bar into a 252-bar window as though it were a session. Close-to-close returns
# over the Monday-to-Friday subset lose nothing by it, because a return reads two levels and the
# Sunday evening move is already inside Monday's close.

# %%
daily = load_daily_bars(
    setup, symbols=UNIVERSE, start_date=DAILY_HISTORY_START, end_date=str(HOLDOUT_END)
)
weekday_daily = daily.filter(pl.col("timestamp").dt.weekday() <= 5)
print(
    f"{len(daily):,} daily bars, {daily['timestamp'].min()} to {daily['timestamp'].max()}; "
    f"{len(daily) - len(weekday_daily):,} weekend rows dropped, {len(weekday_daily):,} kept"
)

# The claim the join rests on: the daily bar of day d is that day's hourly bars, 00:00 to 24:00.
_probe_start, _probe_end = "2025-06-01", "2025-06-30"
_hourly = load_mt5_bars(
    "1h", symbols=UNIVERSE, start_date=_probe_start, end_date=_probe_end
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
_from_hours = (
    _hourly.with_columns(pl.col("timestamp").dt.date().alias("day"))
    .group_by(["symbol", "day"])
    .agg(
        pl.col("open").first().alias("h_open"),
        pl.col("high").max().alias("h_high"),
        pl.col("low").min().alias("h_low"),
        pl.col("close").last().alias("h_close"),
    )
)
_check = daily.join(_from_hours, left_on=["symbol", "timestamp"], right_on=["symbol", "day"])
_gaps = {
    column: float((_check[column] - _check[f"h_{column}"]).abs().max())
    for column in ("open", "high", "low", "close")
}
print(
    f"daily-vs-hourly agreement over {_check.height} bars in June 2025: "
    + ", ".join(f"{k} {v:.2e}" for k, v in _gaps.items())
)
assert max(_gaps.values()) < 1e-6, (
    "the daily bar is not the UTC-day aggregate of the hourly bars, so the availability instant "
    "the daily join uses is wrong"
)

# %% [markdown]
# ### B.3 What this universe looks like
#
# Two indices, one of them quoted about four times the other, both of them the same economy. The
# range column is in basis points, which is already a division by the price level - and that
# division is why every feature below is a return or a ratio rather than a price.

# %%
universe = (
    prices.with_columns(
        (1e4 * (pl.col("high") - pl.col("low")) / pl.col("close")).alias("_range_bp")
    )
    .group_by("symbol")
    .agg(
        pl.len().alias("sessions"),
        pl.col("close").median().round(0).alias("median level"),
        pl.col("_range_bp").median().round(0).alias("median window range (bp)"),
    )
    .sort("symbol")
)
display(universe)

# %% [markdown]
# ## C. Feature construction, one subsection per family
#
# The code for every family is in `_features.py` and is imported above rather than written in a
# cell, for the reason Chapter 25 (Section 25.1) gives: a feature that exists in two
# implementations - one in the research notebook, one in the live loop - agrees on the day the
# second is written and drifts on the first edit, and nothing in either output says so. The
# point-in-time test in `bots/exness_usidx_sess/tests/test_lookahead.py`, the backtest price
# loader and the deployment loop's "recompute features" step all call the functions this notebook
# calls.
#
# ### C.1 The two legs of the session
#
# The decomposition Chapter 8 Section 8.2 builds and `bots/assets/US500.md` names: the overnight
# leg and the intraday leg of each session's return, at three horizons, plus the gap measured
# against the previous sessions' own dispersion so that a two-point move on a quiet index and on
# a violent one are comparable.
#
# One detail decides whether this family is honest, and it is a knowability detail rather than an
# arithmetic one. `intraday_ret(d)` finishes at the cash close: the **overnight** spec decides
# there and may read it, the **intraday** spec decides six hours earlier and may not, so on the
# intraday panel the column is shifted one session before anything is built on it. Getting that
# wrong is invisible in the batch panel and is exactly what Section D.4 catches.
#
# A second detail is a naming one and is corrected here rather than left to be discovered. The
# cut point between the two legs is `session_open_px`, which is the **open of the hourly bar
# straddling the cash open** - on this grid, the price thirty minutes *before* the auction. So
# `overnight_ret` is a pre-open gap and `intraday_ret` carries the last half hour of pre-open with
# it. The two still sum to the session; the cut simply sits half an hour early, because an hourly
# grid cannot put it anywhere else.
#
# ### C.2 The opening range
#
# `or_ret` and `or_range` are the return and the high-low range of that same straddling bar, so
# they read the half hour before the auction and the first half hour of it. Both have printed at
# both decision instants, so the family rides on both panels unshifted.
#
# ### C.3 Momentum, volatility and their differences on the session grid
#
# Three quantities come out of the same trailing window and a shared helper computes them
# together: the return over the window, the annualized deviation of the session log returns
# inside it, and the ratio of the two. Reading one window three ways separates a move that was
# steady from one that was a single jump. `mom_skip_recent` is the return over a quarter that
# stops a week short of the decision, and the two `accel_*` columns are differences between
# horizons.
#
# ### C.4 Garman-Klass volatility, mean reversion, drawdown and oscillators
#
# The Garman-Klass estimator uses all four prices of the window rather than two,
#
# $$\sigma^2_{GK} = \tfrac{1}{2}\left(\ln\tfrac{H}{L}\right)^2 - (2\ln 2 - 1)\left(\ln\tfrac{C}{O}\right)^2$$
#
# which for a window of the same true volatility is several times less noisy than the
# close-to-close estimate. The mean-reversion block standardizes each horizon return against its
# own trailing quarter and asks where the close sits in the range it has recently traded through,
# both as a channel position and as Bollinger %B. `max_dd_63d` is the *current* distance below
# the trailing quarter's peak and not the worst decline inside it - two different statistics that
# have shipped under one name elsewhere.
#
# ### C.5 Realized volatility, inside the window and inside the cash session
#
# The one family that reads the hourly bars as a series rather than as an aggregate.
# `rv_window_1d` is the root sum of squared hourly log returns over everything that printed
# between the previous decision and this one; `rv_cash_1d` is the same sum taken over the bars
# **inside** the NYSE cash session, which is a different quantity because the cash session is
# where the volume is. `rv_cash` carries the same knowability asymmetry as `intraday_ret` and is
# shifted a session on the intraday panel for the same reason.
#
# ### C.6 The structural spread
#
# With two instruments there is no factor to estimate, only the one relative price the pair
# defines: the `USTEC / US500` ratio at three horizons, and each index's rolling correlation with
# it. Both legs are the decision-window close, so the ratio is knowable at the decision.
#
# ### C.7 The daily-grid families
#
# Momentum at 21, 63, 126 and 252 daily bars and its risk-adjusted form; close-to-close volatility
# at a month, a quarter and a year and the ratio of the ends; the ratio to the fifty and
# two-hundred day moving averages; the current drawdown from the trailing year's peak and the
# position in the trailing year's range. Every one of them is a column this bot could not have on
# its own grid, and every one of them arrives one day stale by construction.
#
# ### C.8 The calendar
#
# Seven columns that are functions of the session date and of nothing else: the weekday and the
# month as sine and cosine pairs, the third-Friday index-option expiry, the position of the day
# inside its calendar month and a turn-of-month flag. They are pure date arithmetic on purpose -
# a "position among this month's sessions" column would have to know how many sessions the month
# ends up holding, which a panel truncated at a decision instant does not, and it would then take
# two different values for one session.

# %%
built = build_features(
    prices,
    spec=SPEC,
    windows=WINDOWS,
    daily_windows=DAILY_WINDOWS,
    ranked=RANKED,
    periods_per_year=PERIODS_PER_YEAR,
    daily=daily,
)
feature_cols = feature_columns(built)
print(f"{len(built):,} index-sessions carrying {len(feature_cols)} features")

# %% [markdown]
# ## D. The timing contract
#
# Each feature makes two promises about time: how many bars back it reads, and how long its
# inputs take to become available after the period they describe. Together those fix the earliest
# decision the feature can be used at, and they are what the register declares in Section A. This
# section checks all five - what the constructions read, that the daily join never reaches its own
# day, that no column fills before its window could have, that no value depends on a date after
# it, and that the value a decision instant would have seen is the value the batch matrix holds.
#
# ### D.1 What each construction reads

# %%
plot_timing_contract(
    FAMILIES,
    bar_unit="trading sessions",
    title="Only the daily families wait to publish, and they wait exactly one day",
    subtitle="Register lookback per family; the gap at the right edge is the declared lag",
    alt=(
        "Horizontal bars, one per family, each extending leftward from the decision line by that "
        "family's declared lookback - shortest for the opening range and the calendar, longest "
        "for the four daily-grid families at 252 bars. The hourly-grid families all reach the "
        "line; the daily families stop one bar short of it, which is the one-day staleness of "
        "the daily file."
    ),
)

# %% [markdown]
# ### D.2 The daily join never reaches its own day
#
# The single assertion that makes the two-frequency join safe. For every row of the matrix, the
# daily bar the join landed on must be dated strictly before the decision's own calendar day - not
# "usually", not "on average", but on every row, because one row that read its own day would be a
# feature computed from the session it is forecasting.

# %%
joined_daily_dates = (
    built.select("symbol", "timestamp", "decision_ts")
    .join(
        prices.select("symbol", "timestamp", "decision_ts"),
        on=["symbol", "timestamp", "decision_ts"],
        how="inner",
    )
    .sort("decision_ts")
    .join_asof(
        weekday_daily.select(
            "symbol", pl.col("timestamp").alias("d1_date"), pl.col("close").alias("_d1_close")
        )
        .with_columns(
            (pl.col("d1_date").cast(pl.Datetime("us")) + pl.duration(days=1)).alias(
                "available_from"
            )
        )
        .sort("available_from"),
        left_on="decision_ts",
        right_on="available_from",
        by="symbol",
        strategy="backward",
    )
)
lag_days = (
    joined_daily_dates.select(
        (pl.col("timestamp") - pl.col("d1_date")).dt.total_days().alias("lag")
    )
)
print(
    f"daily bar age at the decision, in days: min {lag_days['lag'].min()}, "
    f"median {lag_days['lag'].median()}, max {lag_days['lag'].max()} "
    "(3 is a Monday reading Friday, 4 a Tuesday after a Monday holiday)"
)
assert int(lag_days["lag"].min()) >= 1, (
    "the daily join landed on a bar dated on the decision's own day, which had not finished "
    "printing at 14:00 or 20:00 UTC"
)

# %% [markdown]
# ### D.3 Warmup
#
# A trailing window cannot produce a value until it has enough bars to fill. The audit checks
# that length rather than describing it: a column carrying a value before its window could have
# filled is reading bars that do not exist, and that is what it raises on. The declared numbers
# come from `_features.warmup_expectations`, the same dictionary the point-in-time test and
# `01_feasibility_analysis`'s fold assertion read - which is the correction this notebook's second
# pass carries, because the first version compared the available warmup with the largest single
# window rather than with the composed one, and `zscore_21d` composes a 21-session return with a
# 63-session standardization for a true cost of 84.
#
# The `d1_` columns are deliberately absent from the audit and that is not an oversight: their
# warmup is consumed inside the daily file three years before this panel opens, so on this grid
# they are complete on the first row and there is nothing to count. D.2 is their timing check.

# %%
warmup_audit(built, warmup_expectations(WINDOWS, SPEC), entity="symbol")

# %% [markdown]
# ### D.4 Withholding the holdout changes nothing
#
# Trailing statistics share a property worth checking directly: recomputed on a panel that stops
# before the holdout, they reproduce the same values on the rows the two builds share. A parameter
# fitted over a whole column - a winsorization bound, a scaler, an encoder - does not, because
# truncating the column moves the parameter and with it every row it was applied to. **Both**
# price files are truncated here, not only the hourly one: a statistic fitted across the daily
# file would otherwise survive untouched, and the daily families are exactly where such a thing
# would hide.

# %%
seal = assert_values_agree(
    built.filter(pl.col("timestamp") < HOLDOUT_START),
    build_features(
        prices.filter(pl.col("timestamp") < HOLDOUT_START),
        spec=SPEC,
        windows=WINDOWS,
        daily_windows=DAILY_WINDOWS,
        ranked=RANKED,
        periods_per_year=PERIODS_PER_YEAR,
        daily=daily.filter(pl.col("timestamp") < HOLDOUT_START),
    ),
    columns=feature_cols,
    keys=["timestamp", "symbol"],
)
seal.filter(
    pl.col("column").is_in(
        ["zscore_21d", "spread_corr_1d", "rv_cash_21d", "d1_ret_252d", "d1_channel_pos_252d"]
    )
)

# %% [markdown]
# ### D.5 Rebuilding the matrix at the decision instant
#
# D.4 withholds whole dates. The stronger form withholds **bars**: at a decision instant only the
# bars that had closed were on the tape, and a feature computed from those alone must equal the
# batch value for that session. That is a claim about the aggregation as much as about the
# features - a window that absorbed a bar closing after the decision would pass D.4 and fail here
# - and it is the only check that can catch a wrong shift in the overnight / intraday
# decomposition or a daily join that reached a day too far. `_features.features_as_of` truncates
# **both** files at the decision instant, the hourly bars at their close and the daily bars at the
# end of their own UTC day, rebuilds the panel and the whole matrix, and the row for that session
# is compared column by column with the batch row. `bots/exness_usidx_sess/tests/test_lookahead.py`
# runs the same comparison as a test over more instants on every build. The holdout is never
# sampled.

# %%
bars_1h = load_mt5_bars(
    "1h", symbols=UNIVERSE, start_date=HISTORY_START, end_date=str(DEVELOPMENT_END)
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
daily_dev = daily.filter(pl.col("timestamp") <= DEVELOPMENT_END)
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
    as_of = features_as_of(bars_1h, instant, setup, spec=SPEC, daily_bars=daily_dev)
    assert as_of["timestamp"].max() == session, "the truncated panel ends on a different session"
    live = as_of.filter(pl.col("timestamp") == session).sort("symbol")
    batch = built.filter(pl.col("timestamp") == session).sort("symbol")
    gap = assert_values_agree(batch, live, columns=feature_cols, keys=["timestamp", "symbol"])
    recompute_rows.append(
        {
            "session": session,
            "decision (UTC)": instant,
            "hourly bars seen": bars_1h.filter(
                pl.col("timestamp") + pl.duration(minutes=60) <= instant
            ).height,
            "indices": live.height,
            "max abs difference": gap["max abs difference"].max(),
        }
    )
pl.DataFrame(recompute_rows)

# %% [markdown]
# ## E. Matrix assembly and coverage
#
# The panel key is `symbol` + `timestamp`. Raw OHLC, volume, the two realized-volatility inputs,
# the opening-range prices, the execution price and the label endpoint are all excluded: they are
# the inputs the features are made of, and two of them are dated *after* the decision by
# construction, so a model handed them would be reading an answer it was asked for.
#
# One null policy is applied once, and it is a cut rather than a fill: nothing is imputed,
# forward-filled or zero-filled, because a filled warmup value is a number the window never
# produced. Rows start where the column with the longest chain of trailing windows can first hold
# a value, and every shorter window has filled by then. The assertion below makes that a fact
# rather than an intention.
#
# The matrix is written through the holdout, because the holdout stages need a feature vintage for
# the sessions they score. Every figure from here on reads `development`, the rows before the
# holdout opens.

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
    f"{len(feature_cols)} features, {len(features):,} rows, {features['symbol'].n_unique()} indices"
)
print(f"{features['timestamp'].min()} to {features['timestamp'].max()}")
print(f"null policy dropped {len(built) - len(features):,} warmup rows")
print(
    f"{len(development):,} development rows before {HOLDOUT_START}, the rows every figure below "
    "reads"
)
register_frame(FAMILIES, feature_cols).select(["family", "columns", "role", "representation"])

# %% [markdown]
# ### E.1 Coverage through time
#
# Drawn on the matrix *before* the null policy, so the figure shows what the policy is for. Each
# family climbs to complete coverage as its longest window fills, and the dashed boundary marks
# the first session the emitted matrix keeps. The daily families are the exception worth looking
# for: they are flat at one from the very first session, because their warmup was spent in a file
# that starts three years earlier.

# %%
plot_coverage_through_time(
    family_coverage(
        built.filter(pl.col("timestamp") < HOLDOUT_START).select(["timestamp", *feature_cols]),
        assignment,
        every="1mo",
    ),
    warmup_boundary=features["timestamp"].min(),
    title="The hourly families fill over one quarter; the daily families start full",
    subtitle="Monthly non-null share per family before the policy, with the boundary drawn",
    alt=(
        "Non-null share by feature family, on an axis from zero to one. The families built on the "
        "session grid start at zero and climb to one over the first quarter as their longest "
        "window fills, mean reversion last. The four daily-grid families and the calendar are "
        "flat at one from the first month. A dashed line in early 2023 marks where the null "
        "policy starts, and every line is flat at one from there to the end."
    ),
)

# %% [markdown]
# ## F. What the features look like
#
# Three properties decide whether this matrix can be used at all: the scale each feature arrives
# on, how much of the set is one ordering under several names, and how long a value lasts.
# `05_evaluation` is where the matrix is tested fold by fold for whether any of it predicts.
#
# There is deliberately **no cross-sectional dispersion figure** here, and its absence is the
# point: with two instruments the "cross-section" on a date is two numbers, and the tenth and
# ninetieth percentiles of two numbers are the two numbers. `exness_fx_d1` draws that figure
# because it ranks five pairs; this bot compares each index with its own history
# (`mapping.class: time_series_threshold`), so the question the figure answers is not one this
# bot asks.
#
# ### F.1 Feature distributions

# %%
plot_feature_distributions(
    development,
    ["overnight_ret_1d", "intraday_ret_1d", "zscore_21d", "rv_cash_1d", "d1_ret_252d", "or_ret"],
    title="The two session legs are centred and fat-tailed; realized volatility is not",
    subtitle="One column per family the threshold will be formed on, display tails clipped",
    alt=(
        "Six histograms in two rows. The overnight and intraday legs of the session and the "
        "21-session return z-score are broad single-peaked bodies roughly centred on zero. "
        "Realized volatility inside the cash session is bounded below at zero and skewed right. "
        "The 252-bar daily return is centred well above zero, and the opening-range return is "
        "the narrowest of the six."
    ),
)

# %% [markdown]
# ### F.2 Redundancy structure
#
# Two columns are redundant when they carry the same ordering, however different their formulas
# look, so the distance clustered on is $1 - |\rho_s|$: $\rho_s$ is the rank correlation between
# the pair, and taking its absolute value treats a feature and its negation as the same thing. The
# tree is cut at the rank correlation the configuration declares. Read it as a screen for
# duplication and nothing stronger: which member of a cluster to keep needs a criterion measured
# out of sample, and `05_evaluation` measures one fold by fold.

# %%
clusters = plot_redundancy_clusters(
    development,
    feature_cols,
    cut=REDUNDANCY_CUT,
    title="Horizon groups the signal families; the daily block clusters on its own",
    subtitle=(
        r"Average linkage on $1 - |\rho_s|$, cut at a rank correlation of " f"{REDUNDANCY_CUT}"
    ),
    alt=(
        "Dendrogram of every feature in the matrix. Clusters group by horizon rather than by "
        "family: the short-window return, z-score, channel position and oscillator columns join "
        "together, and the medium and long session windows do the same. The daily-grid columns "
        "form their own clusters, as do the realized-volatility and Garman-Klass columns. The "
        "calendar columns and the opening range sit apart from everything."
    ),
)
print(f"{len(set(clusters.values()))} redundancy clusters at the drawn cut")

# %% [markdown]
# ### F.3 Persistence
#
# The cadence is one session, so what carries information is the autocorrelation of the feature
# itself, run out to a month. It is estimated per index on sessions exactly one lag apart and
# summarized by the median over the two, with a bootstrap interval; with two instruments that
# interval is wide and is drawn that way rather than narrowed. Read this way the figure says which
# columns are state and which are signal: a state column that survives a month is describing a
# regime, and a signal column that has decayed to nothing inside one session cannot be traded at
# this cadence.

# %%
plot_persistence(
    development,
    ["overnight_ret_1d", "zscore_21d", "rv_cash_21d", "vol_gk_63d", "d1_price_to_ma_200d"],
    entity="symbol",
    max_lag=PERSISTENCE_HORIZON,
    decision_dates=development["timestamp"].unique().sort().to_list(),
    title="The state columns survive a month; the session legs do not survive a session",
    subtitle="Median over the two indices; rank correlation across consecutive rebalances",
    alt=(
        "Two panels. On the left, autocorrelation against lag out to a month. The 200-day daily "
        "trend ratio and the three-month Garman-Klass volatility decay slowest and are still high "
        "at the right edge, the 21-session realized volatility is in between, and the 21-session "
        "z-score and the overnight leg have reached zero within a few sessions - the overnight "
        "leg crossing below zero at lag one. The bootstrap ribbons over two indices are wide. On "
        "the right, the rank correlation between consecutive sessions is near one for the state "
        "columns and near zero for the session legs."
    ),
)

# %% [markdown]
# ## G. Emit
#
# The parquet is written with a sidecar beside it, a small JSON file recording a hash of the
# table's contents alongside its row count, its key columns and the same hash taken over the two
# price files it was built from. The hash is over values rather than over the bytes of the file,
# so re-writing the parquet or reordering its rows leaves it alone, while any change to a single
# feature value moves it. That is what makes "the artifact on disk is the one this code produces"
# a question anyone can settle by reading two files.
#
# The `spec` is recorded in the inputs, because the same case study writes a different matrix
# under this same path in the other workspace and the digest is the only thing that tells them
# apart afterwards.

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
record = write_artifact(
    features,
    FEATURES_DIR / "financial.parquet",
    keys=["symbol", "timestamp"],
    written_by="case_studies/exness_usidx_sess/03_financial_features.py",
    inputs={
        f"session_panel:1h:{SPEC}": value_digest(prices),
        "load_mt5_bars:daily": value_digest(daily),
    },
)
print(
    f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')} for spec {SPEC!r} "
    f"(label {LABEL}) under digest {record['digest']}"
)

# %% [markdown]
# ## Key takeaways
#
# - **Declare how far back each feature reads, and how stale its input is, before writing it.**
#   The configuration holds one lookback and one lag per family; the warmup assertion, the timing
#   figure and the register table all read those declared numbers instead of re-deriving them, so
#   a window that disagrees with what was promised raises rather than passing quietly.
# - **A window the sample cannot warm up is not a window you may take from the same file.** The
#   long trends here come from the daily bar, and they cost exactly one day of freshness - a cost
#   that is declared as `lag: 1`, checked on every row in D.2, and rebuilt from truncated daily
#   bars in D.5. A join on the date rather than on the availability instant would have been
#   invisible in every figure in this notebook.
# - **The knowability asymmetry between two decisions in one session is a feature-level fact.**
#   `intraday_ret` and `rv_cash` are shifted on one panel and not the other, which no assertion
#   about dates can catch and only a rebuild at the decision instant can.
# - **One construction, every consumer.** The functions this notebook calls are the ones the
#   point-in-time test, the backtest loader and the live loop call, so there is no second
#   implementation to drift.
# - **Read the matrix before modelling it.** Distribution, redundancy and decay each rule out a
#   use, and the figure this notebook does *not* draw - cross-sectional dispersion - is ruled out
#   by the universe rather than by the data.
#
# ### Known limitations
#
# - **No VIX.** `bots/assets/US500.md` section 7 names implied volatility first among the state
#   variables an index bot should carry, and this account quotes no VIX instrument: it is absent
#   from `data/mt5/1h.parquet` and from the Market Watch list read on 2026-09-08. The realized
#   volatility families are a backward-looking substitute and are not the same thing - realized
#   volatility says what happened and implied volatility says what is being paid for. TODO (user
#   decision, one DSR trial when it is tried): if a VIX or VXN feed is added, record its source
#   and its publication lag and add a family to `setup.yaml::features.families` before rebuilding.
# - **No FOMC or macro-release calendar.** `bots/assets/US500.md` section 7 names it and this
#   repository has no news calendar carrying a *publication* date, which is what Chapter 8 Section
#   8.4 requires before an event feature may be used. A calendar of meeting dates typed from
#   memory would be a point-in-time claim nobody measured, so the family is absent rather than
#   approximated. The deployment loop's `avoid_windows` (phase 8) is the place the FOMC hour is
#   handled instead, as an execution rule rather than as a feature.
# - **A 30-minute edge block cannot be resolved on an hourly grid.** The session-bounded
#   volatility family of `nasdaq100_microstructure` splits the session into 30-minute edges; the
#   finest this bot can see is one hour, so `rv_cash` covers the whole cash session and the
#   opening range covers the hour straddling the open. Sub-hourly bars are not served for these
#   symbols by this account.
# - **The cut between the overnight and intraday legs sits 30 minutes before the cash open**, not
#   at it, because that is where the hourly bar boundary falls. The two legs still sum to the
#   session and the split is stable, but neither is exactly the quantity a minute-bar study would
#   call by that name.
# - **Volume on an MT5 feed is tick count at one broker** rather than traded notional, so it is
#   used only through the range and the realized-volatility columns and never as a participation
#   feature.
# - **Session flags** (`bots/_shared/sessions.py`) are constant for a given spec - the intraday
#   decision is always in the London/New York overlap and the overnight decision always in the
#   New York session - so they carry no information within a workspace and are not registered.
# - Every feature here is a rule written in advance. `04_model_based_features` adds the features
#   that are themselves model outputs, where the rule is estimated from the data.
