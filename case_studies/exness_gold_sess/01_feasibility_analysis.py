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
# # Exness Gold Sessions (exness_gold_sess): Feasibility Analysis
#
# Before building a trading strategy it is worth asking whether the data can support one at all.
# This notebook does that and nothing else: it fits no model and makes no forecast.
#
# The strategy being checked is described in `config/setup.yaml`. It trades gold and silver as
# contracts for difference on an Exness MetaTrader 5 account, and it decides **twice on every
# weekday**: one hour after the London session opens, and one hour after the New York session
# opens. Each decision is held to the close of the session it was taken in.
#
# That design puts three questions in front of the data, and all three are settled here rather
# than assumed:
#
# 1. **Does the decision instant exist?** A session snapshot is a rule, not a bar. The rule has
#    to resolve to a bar that actually printed, on both metals, in both daylight-saving seasons,
#    and the sessions where it does not have to be counted and dropped rather than filled with a
#    stale price.
# 2. **Is a session move larger than what it costs to trade?** The spread on this account was
#    measured on 2026-09-07 and is a constant in points, so the answer is arithmetic rather than
#    an assumption - but it is very different for the two metals, and that difference is the
#    single most important number this notebook produces.
# 3. **Does the history hold the evaluation the design declares?** Four folds of three years'
#    training and one year's validation, ahead of a twelve-month holdout that nothing here reads.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Turn a *session* rule ("one hour after the open") into a decision timestamp on a bar grid,
#   for a venue whose session moves with daylight saving while the data server's clock does not
# - Check that rule against the data instead of trusting it, and report the sessions it fails on
# - Compare a session's move with the round trip it would have to pay, per instrument, when two
#   instruments in the same book charge very different spreads
# - Compute the minimum information coefficient a signal needs before it pays for its own
#   spread - the number that can close a question before a single model is fitted
# - Check that a walk-forward split of the history fits the sample available and leaves the
#   holdout unread
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. This notebook reads MT5 H1 and D1 bars and `history_depth.json`,
# all written by `bots/_shared/mt5_loader.py`, plus `config/setup.yaml`, and writes nothing.
# It is a Route B fork of `case_studies/exness_fx_d1/01_feasibility_analysis.py`; what differs
# is the decision grid (a session snapshot on H1 bars instead of one daily bar), the two-asset
# cost comparison, and section B.5, which the FX fork does not have.
#
# ## Prerequisites
#
# None beyond what the sections below define.

# %%
"""exness_gold_sess Case Study - Feasibility Analysis (Route B fork of exness_fx_d1)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars, read_history_depth
from bots._shared.sessions import ServerClock
from case_studies.exness_gold_sess._features import decision_grid, session_panel
from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")
set_global_seeds(42)

# %% tags=["parameters"]
CASE_STUDY_ID = "exness_gold_sess"
# START_DATE = None reads setup.yaml::universe.history_start (2017-02-27, the first week the H1
# grid is dense on both metals, measured in task B0); END_DATE is evaluation.holdout_end.
START_DATE = None
END_DATE = "2026-08-31"

# %% [markdown]
# ## Configuration
#
# Everything the strategy assumes is declared in `config/setup.yaml`, and this notebook reads
# those values rather than repeating them, so the two can never disagree.
#
# **How the history is divided.** The sample runs from the last week of February 2017 - the
# first week this account serves a dense hourly grid on *both* metals - to the end of August
# 2026. The last twelve months are the **holdout**: a stretch of history that is not looked at
# while the strategy is being designed, so that when it is finally scored there the result is
# not a rehearsal of choices already made on the same data. Everything below uses the earlier
# part, the development period.
#
# **When the strategy decides.** Not at a clock time. `decision.snapshots` names two *venue*
# sessions, London and New York, and `decision.edge_block_minutes` says to skip the first half
# hour of each. The venues' opening hours are read from their own time zones, so the UTC hour of
# a decision moves with each venue's daylight saving - while the data server's clock is UTC+0 all
# year and follows nobody's daylight saving. Writing a server hour into this notebook would
# therefore be right in one season and wrong in the other; section B.1 derives the instant and
# then checks what it derived.
#
# **What a trade is assumed to cost.** `costs.spread_bps_by_session` carries a spread measured on
# this account from thirty days of tick data, per metal and per session bucket. It is not a
# marketing figure and not a range taken from a document.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

START_DATE = START_DATE or str(SETUP["universe"]["history_start"])
HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
SYMBOLS = sorted(SETUP["universe"]["symbols"])
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = SETUP["labels"]["buffer"]
LABEL_NAMES = [PRIMARY_LABEL, *SETUP["labels"]["variants"]]
HORIZON_HOURS = {n: int(str(SETUP["labels"]["horizons"][n]).rstrip("Hh")) for n in LABEL_NAMES}
EDGE_BLOCK = int(SETUP["decision"]["edge_block_minutes"])
CLOSE_TOLERANCE = int(SETUP["decision"]["session_close_tolerance_minutes"])
SESSION_VALUES = list(SETUP["decision"]["session_filter_values"])
SERVER_CLOCK = ServerClock.from_dict(SETUP["decision"]["server_clock"])
SPREAD_BY_SESSION = SETUP["costs"]["spread_bps_by_session"]
ROUND_TRIP_P90 = SPREAD_BY_SESSION["round_trip_p90_bps"]
# The number of DECISION SLOTS a year (504 = 2 snapshots x 252). It is not the annualisation
# factor of a return series: that is `evaluation.periods_per_year` (252, a DAILY grid), and this
# line used to read it and so misnamed what it held
# (bots/exness_gold_sess/PRICE_GRID_DECLARATION.md section 5, 2026-09-08). Nothing below uses
# this constant; it is printed nowhere, so the value and every printed number are unchanged.
PERIODS_PER_YEAR = int(SETUP["decision"]["slots_per_year"])
SLOTS_PER_DAY = len(SETUP["decision"]["snapshots"])
BAR_MINUTES = 60

print(
    f"{len(SYMBOLS)} instruments {SYMBOLS} on the {SETUP['decision']['cadence']} grid: "
    f"{SLOTS_PER_DAY} decisions a weekday ({', '.join(SESSION_VALUES)}), each taken "
    f"{EDGE_BLOCK} minutes or more after its venue opens and held to that venue's close."
)
print(
    f"Sample {START_DATE} to {END_DATE}; the holdout {HOLDOUT_START} to {HOLDOUT_END} is not "
    f"read below. Primary label {PRIMARY_LABEL} over {HORIZON_HOURS[PRIMARY_LABEL]} hours."
)

# %% [markdown]
# ## A. Orientation
#
# ### What these two instruments are
#
# `XAUUSD` is the price of one troy ounce of gold in US dollars and `XAGUSD` the price of one
# ounce of silver. On this account they are contracts for difference: no metal is delivered, the
# position is a bet on the price, and the broker quotes a bid and an ask around it. A contract is
# 100 ounces of gold or 5,000 ounces of silver, which matters later - the smallest order the
# broker accepts is a hundredth of a contract, so gold can be traded one ounce at a time and
# silver only fifty ounces at a time.
#
# ### Why this is a session strategy and not a daily one
#
# A daily strategy on gold takes one decision a day and holds it through every session, including
# the hours when the metal barely trades. This design says the interesting hours are the ones
# after London opens and after New York opens - when the physical market, the futures market and
# the dollar are all being priced at once - and that a forecast made *at* those moments, for the
# rest of *that* session, is a different and sharper question than a forecast for the next
# twenty-four hours. Whether it is a better one is what the rest of the pipeline decides.
#
# ### The three questions this notebook asks
#
# Does the decision instant exist on both metals, in both seasons (B.1)? Is a session move large
# enough to pay for itself (B.3, B.5)? And does the declared evaluation fit the history (D)?

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 The decision grid
#
# The H1 bars are stamped at their UTC open on a server whose clock is UTC+0. The decision rule
# turns each venue session into one instant, and the rule is applied by
# `_features.decision_grid`, the same function `02_labels` and the backtest price loader use - so
# the instant this notebook checks is the instant those stages act on, not a copy of it.
#
# The first cell reads the data and the recorded history depth. The depth record is checked for
# two things: that the server clock in `setup.yaml` is the one that was measured, and that the H1
# depth was read with the **count-based** request. Those are not interchangeable: on this account
# a calendar-range request returns gold from 2022-10-25 and a count-based one returns it from
# 2014-01-14 (of which 2017-02-27 onward is a real hourly grid), and a design built on the first
# number would have 2.85 years of one gold bull market where the account serves 8.5 years.

# %%
depth = read_history_depth()
measured = ServerClock.from_dict(depth["server_clock"])
assert measured.utc_offset_minutes == SERVER_CLOCK.utc_offset_minutes, (
    "setup.yaml::decision.server_clock disagrees with the clock measured in history_depth.json"
)
for symbol in SYMBOLS:
    record = depth["timeframes"]["1h"][symbol]
    assert record.get("history_mode") == "deep", (
        f"{symbol}: history_depth.json records the H1 depth as "
        f"{record.get('history_mode', 'range (no history_mode key)')!r}. A range download "
        "under-reports the depth on this account; re-read with "
        "mt5_loader.fetch_mt5_bars_deep before trusting universe.history_start."
    )
    assert record["dense_history_start"] <= START_DATE, (
        f"{symbol}: universe.history_start {START_DATE} is earlier than the measured "
        f"dense_history_start {record['dense_history_start']}"
    )

# The MT5 parquet carries every instrument of the four bots; this study reads its own universe.
bars = load_mt5_bars("1h", symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE).with_columns(
    pl.col("timestamp").dt.replace_time_zone(None)
)
d1 = load_mt5_bars("daily", symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE)

panel, report = session_panel(
    bars,
    edge_block_minutes=EDGE_BLOCK,
    tolerance_minutes=CLOSE_TOLERANCE,
    expected_horizon_bars=HORIZON_HOURS[PRIMARY_LABEL],
    keep_context=True,
    return_report=True,
)
research = panel.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_datetime())

# %% [markdown]
# The checks below are the ones that decide whether this design is expressible at all. The rule
# says the decision is the close of the first bar closing at or after the open plus half an hour;
# on a UTC-aligned hourly grid that must be the open plus exactly sixty minutes, and the label
# must end exactly eight bars later, in **both** venues and **both** seasons. `decision_grid`
# asserts both; what is printed here is the evidence behind the assertion, plus the sessions the
# rule could not resolve and why.

# %%
print(
    f"decision offset from the session open: {report['decision_offset_minutes']} minutes "
    f"(the rule asks for >= {EDGE_BLOCK}; an hourly grid gives exactly {BAR_MINUTES})"
)
print(
    f"label span from decision to session close: {report['label_span_minutes']} minutes "
    f"= {report['label_span_minutes'][0] // 60} bars, in both venues and both seasons"
)
print(f"sessions declared by the venue calendars: {report['sessions_declared']:,}")
for symbol, rec in sorted(report["per_symbol"].items()):
    print(
        f"  {symbol}: decisions kept {rec['kept']:,}, of which labelled {rec['labelled']:,} "
        f"| no bar inside the session {rec['no_bar_in_session']} "
        f"| stale decision (row DROPPED) {rec['stale_decision']} "
        f"| endpoint unresolved (row KEPT, label null) {rec['endpoint_unresolved']} "
        f"= {rec['early_close']} early close + {rec['off_horizon']} off-horizon"
    )
    if rec["dropped_dates"]:
        print(f"    dropped dates (no fresh price at the snapshot): {rec['dropped_dates']}")
    if rec["unresolved_dates"]:
        print(f"    dates kept without an endpoint: {rec['unresolved_dates']}")
print(
    "A decision needs only the bars that had closed at the snapshot; an endpoint needs bars "
    "eight hours later. The two are resolved separately, so an early close costs the LABEL and "
    "not the row. Resolving them together - the first version of _features.decision_grid - both "
    "emptied features_as_of (no endpoint has printed at the decision instant, so the truncated "
    "panel lost exactly the row a live loop asks for) and selected the panel against "
    "thin-liquidity sessions on information the decision could not have had."
)

by_session = (
    research.group_by(["symbol", "session"]).len().sort(["symbol", "session"])
)
display(by_session)
print(
    f"development rows {research.height:,} over "
    f"{research['timestamp'].n_unique():,} distinct decision instants, "
    f"{research['timestamp'].min()} to {research['timestamp'].max()}"
)

# %% [markdown]
# ### B.2 Are both metals quoting when the strategy decides
#
# A book of two instruments has no cross-section to fall back on: if one metal is missing at a
# decision instant, the book is a one-instrument book at that instant, and any statistic computed
# across the two is computed across one. The count below is therefore a floor of two, not a
# ratio.

# %%
breadth = (
    research.group_by("timestamp")
    .agg(pl.col("symbol").n_unique().alias("n"))
    .sort("timestamp")
)
short = breadth.filter(pl.col("n") < len(SYMBOLS))

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.axhline(len(SYMBOLS), color=COLORS["copper"], lw=5, alpha=0.35, zorder=1, label="both metals")
ax.plot(breadth["timestamp"], breadth["n"], color=COLORS["blue"], lw=0.8, zorder=3, label="quoting")
ax.set_ylim(0, len(SYMBOLS) + 1)
ax.set_ylabel("Instruments quoting")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.24))
add_message_title(
    ax,
    "Both metals quote at almost every decision instant",
    subtitle="Instruments with a decision bar at each session snapshot, development period",
)
show_with_alt(
    fig,
    "Count of instruments quoting at each decision instant from 2017 to 2025. The line sits at "
    "two for essentially the whole sample, with a small number of single-instrument instants.",
)
print(
    f"{short.height} of {breadth.height:,} decision instants carry fewer than {len(SYMBOLS)} "
    f"instruments ({short.height / max(breadth.height, 1):.4%})"
)

# %% [markdown]
# ### B.3 What a round trip costs, and what a session move is worth
#
# Entering a position and leaving it crosses the spread twice. On this account the spread was
# measured on 2026-09-07 from thirty days of ticks and is a **constant in points** in every
# session bucket - 260 points on gold, 30 points on silver - so the cost of a round trip is not an
# assumption here, it is a reading. In basis points of price that is 1.2 for gold and 9.4 for
# silver: silver charges nearly eight times what gold charges for the same trade.
#
# Because the two costs differ by that much, one cost line drawn across both series would answer
# the question for neither. Every move is therefore divided by its own instrument's round trip,
# which puts break-even at 1 for both and lets them share an axis.

# %%
cost = pl.DataFrame(
    {"symbol": list(ROUND_TRIP_P90), "cost_bps": [float(v) for v in ROUND_TRIP_P90.values()]}
).sort("symbol")
display(cost)

moves = (
    research.with_columns(
        (pl.col("label_end_close") / pl.col("close") - 1).abs().alias("session_move"),
        (pl.col("close") / pl.col("session_open_px") - 1).abs().alias("opening_hour_move"),
    )
    .join(cost, on="symbol")
    .with_columns(
        (pl.col("session_move") * 1e4 / pl.col("cost_bps")).alias("session_x"),
        (pl.col("opening_hour_move") * 1e4 / pl.col("cost_bps")).alias("hour_x"),
    )
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
styles = {
    ("XAUUSD", "session_x"): (COLORS["amber"], "-"),
    ("XAGUSD", "session_x"): (COLORS["blue"], "-"),
    ("XAUUSD", "hour_x"): (COLORS["amber"], "--"),
    ("XAGUSD", "hour_x"): (COLORS["blue"], "--"),
}
for (symbol, col), (colour, style) in styles.items():
    series = moves.filter(pl.col("symbol") == symbol)[col].drop_nulls().to_numpy()
    if series.size == 0:
        continue
    magnitude, fraction = exceedance_curve(series)
    tag = "session" if col == "session_x" else "opening hour"
    ax.plot(magnitude, fraction, color=colour, ls=style, lw=1.5, label=f"{symbol}, {tag}")
ax.axvline(1, color=COLORS["slate"], ls=":", lw=1.8, label="break-even on the round trip")
ax.set(xscale="log", xlim=(0.01, 500))
ax.set_xlabel("Absolute move as a multiple of that metal's own round trip (log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Gold clears its round trip far more often than silver clears its own",
    subtitle="Absolute moves scaled by each metal's measured round trip, development period",
)
show_with_alt(
    fig,
    "Four exceedance curves on a logarithmic axis giving the fraction of absolute moves at "
    "least a given multiple of each metal's own round-trip cost. The gold curves sit above the "
    "silver ones throughout, and the session curves above the opening-hour curves.",
)

# %% [markdown]
# ### B.4 How much of one session's move carries into the next
#
# A position opened at one decision and closed at the session close earns that session's move.
# Before forecasting it, it is worth asking how much of it the instrument's own recent sessions
# already account for. The autocorrelation is computed **inside each instrument** and then
# averaged: stacking two instruments into one series would correlate the last gold session with
# the first silver one at the join.
#
# The lag unit here is a *decision slot*, and the two venues alternate, so lag 1 compares a London
# session with the New York session that follows it and lag 2 compares London with London.

# %%
slot_returns = research.sort(["symbol", "timestamp"]).with_columns(
    (pl.col("label_end_close") / pl.col("close") - 1).alias("ret")
)
max_lags = 2 * SLOTS_PER_DAY * 5  # two weeks of slots
acf = panel_acf(slot_returns, entity_col="symbol", value_col="ret", max_lags=max_lags).filter(
    pl.col("lag") > 0
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.axhspan(
    -acf["band"][0],
    acf["band"][0],
    color=COLORS["copper"],
    alpha=0.3,
    zorder=1,
    label="range expected from no information",
)
ax.fill_between(
    acf["lag"],
    acf["acf_p10"],
    acf["acf_p90"],
    color=COLORS["blue"],
    alpha=0.15,
    zorder=2,
    label="10th to 90th percentile across the two metals",
)
ax.bar(acf["lag"], acf["acf"], color=COLORS["blue"], width=0.6, zorder=3, label="average")
ax.set_xlabel("Decision slots between the two session returns")
ax.set_ylabel("Autocorrelation of the session return")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
add_message_title(
    ax,
    "A session's own recent history accounts for almost none of its next move",
    subtitle="Computed within each metal on the decision grid, then averaged",
)
show_with_alt(fig, "Within-instrument autocorrelation of the session return against lag in slots.")

# %% [markdown]
# ### B.5 The minimum information coefficient, per metal and per label
#
# This is the cheapest strong test in the whole of phase 1, and it runs before a single model is
# fitted. A signal that ranks a forward return with information coefficient $IC$ earns, per
# decision, roughly $IC \times \sigma$ of that return - so it only pays for itself when
#
# $$IC > IC^{*} = \frac{\text{spread}_{p90}}{\sigma(\text{label})}$$
#
# where the spread is the **one-way** p90 measured on this account (the convention
# `case_studies/xau_fx_mt5_d1/config/setup.yaml:322` uses, so the two bots' numbers are
# comparable). If a metal's $IC^{*}$ is larger than any information coefficient measured anywhere
# in this repository, that metal's question is answered here and no trial needs to be spent on it.
#
# Silver's spread is 7.8 times gold's while its session moves are only about twice as large, so
# the two $IC^{*}$ figures are expected to differ by a factor of about four. What that means for
# the universe is decided at the cost gate in phase 6, not here - but it is declared here.

# %%
ic_star_rows = []
for symbol in SYMBOLS:
    one_way_bps = float(ROUND_TRIP_P90[symbol]) / 2
    g = research.filter(pl.col("symbol") == symbol).sort("timestamp")
    session_ret = (g["label_end_close"] / g["close"] - 1).drop_nulls()
    # fwd_ret_24h EXACTLY as 02_labels defines it, and not the two-slot return that stands in for
    # it. Two slots ahead is twenty-four hours on four weekdays in five; on a Friday it is
    # seventy-two, and holidays add more. 02_labels nulls those rows rather than renaming the
    # label, so measuring sigma on an unfiltered two-slot return mixes weekend moves into the
    # dispersion of a label that never contains one - inflating sigma and reporting IC* too LOW,
    # which is the direction that flatters the strategy. Corrected 2026-09-08.
    step = SLOTS_PER_DAY
    span_minutes = (
        pl.col("timestamp").shift(-step).dt.epoch("s") - pl.col("timestamp").dt.epoch("s")
    ) // 60
    daily = (
        g.with_columns(span_minutes.alias("_span"))
        .with_columns(
            pl.when(pl.col("_span") == 24 * 60)
            .then(pl.col("close").shift(-step) / pl.col("close") - 1)
            .otherwise(None)
            .alias("r24")
        )["r24"]
        .drop_nulls()
    )
    for label, series in ((PRIMARY_LABEL, session_ret), ("fwd_ret_24h", daily)):
        sigma = float(series.std())
        ic_star_rows.append(
            {
                "symbol": symbol,
                "label": label,
                "n_rows": series.len(),
                "one_way_spread_bps": round(one_way_bps, 3),
                "sigma_bps": round(sigma * 1e4, 1),
                "median_abs_move_bps": round(float(series.abs().median()) * 1e4, 1),
                "ic_star": round(one_way_bps * 1e-4 / max(sigma, 1e-12), 4),
            }
        )
ic_star = pl.DataFrame(ic_star_rows).sort(["label", "symbol"])
display(ic_star)
print(
    "IC* is the information coefficient a signal must beat before it pays for the spread alone; "
    "swap, slippage and the cost of being wrong are all on top of it."
)
print(
    "Both series are the labels 02_labels publishes, on the rows it publishes them on: the "
    "session return runs to the resolved endpoint, and the 24-hour return is null wherever the "
    "span is not exactly 1,440 minutes."
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## C. Design decisions
#
# ### C.1 Why the decision is one hour after the open, and not at the open
#
# `bots/assets/XAUUSD.md:22` asks for the first thirty minutes of a session to be blocked: the
# opening auction of a venue is where the spread is widest and the price least representative,
# and `bots/_shared/sessions.py:64` implements that half hour as the `edge_open` flag. On an
# hourly grid the first bar that clears the block is the one that closes sixty minutes after the
# open, which is also - and this saves a trial - after the 13:30 UTC United States data window
# that `bots/assets/XAUUSD.md:24` proposed as a second candidate snapshot. One rule covers both.
#
# ### C.2 What would send this design back
#
# Three results would, and each is measured where its evidence is. The one this notebook could
# have produced is a **grid failure**: if the decision rule did not resolve to a bar on a
# material share of sessions, or resolved to different offsets in the two seasons, the design
# would not be expressible and section B.1 would have said so. The second is a **cost failure**,
# opened in B.3 and B.5 and settled in `16_costs` against the trades a backtest actually places.
# The third belongs to Chapter 7: whether any signal has a relationship to the session return at
# all.
#
# ### C.3 What the strategy does with the forecast
#
# `setup.yaml::mapping.class` is `per_asset_signal_timing`, not a cross-sectional rank. Two names
# do not make a cross-section - a percentile over two values takes two values - so each metal is
# compared with **its own** trailing scores and traded when it clears its own threshold. The two
# may therefore both be held, or neither.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 How much an evaluation has to spend
#
# The unit a strategy spends is the decision it takes. This grid holds two a weekday, so the
# development period is counted in slots below and compared against what a complete panel would
# carry.

# %%
dev_days = research["timestamp"].dt.date().n_unique()
print(
    f"development decision instants {research['timestamp'].n_unique():,} over {dev_days:,} "
    f"calendar dates ({research['timestamp'].n_unique() / max(dev_days, 1):.2f} a date; the "
    f"design declares {SLOTS_PER_DAY}) | instrument-slots {research.height:,}"
)
print("H1 history depth on this account, per symbol (history_depth.json):")
for symbol in SYMBOLS:
    r = depth["timeframes"]["1h"][symbol]
    print(
        f"  {symbol}: {r['n_bars']:,} bars {r['first_server_day']} to {r['last_server_day']} "
        f"read {r['history_mode']}; dense from {r['dense_history_start']} "
        f"({r.get('dense_n_bars', 0):,} bars), sparse prefix {r.get('sparse_prefix_bars', 0):,}"
    )

# %% [markdown]
# ### D.2 The folds
#
# A model is fitted on one stretch of history and evaluated on the stretch that follows it; each
# such pair is a **fold**. The gap between training and validation is the **purge**, and it has
# to be at least as wide as the label reaches forward, or a training row's outcome would be known
# inside the validation window. `labels.buffer` sets it, and `evaluation.calendar` is the
# calendar the splitter counts those windows on.
#
# The folds are laid out on **dates**, not on decision instants: a fold boundary that fell between
# the London and the New York slot of one day would split a day in half for no reason and make the
# two sleeves of the book train on different samples. The splitter is handed the timeline and no
# prices, and it applies `evaluation.holdout_start` itself, so nothing the holdout contains
# reaches a number computed above.

# %%
timeline = (
    panel.select(pl.col("timestamp").dt.date().alias("timestamp")).unique().sort("timestamp")
)
splits = generate_cv_splits(
    timeline, case_study_id=CASE_STUDY_ID, label_buffer=LABEL_BUFFER, date_col="timestamp"
)
grid = timeline["timestamp"].to_numpy()
purge_gaps = {
    int(((grid > np.datetime64(s["train_end"])) & (grid < np.datetime64(s["val_start"]))).sum())
    for s in splits
}
last_val = max(split["val_end"] for split in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert last_val < np.datetime64(HOLDOUT_START), "a fold reaches into the holdout"
assert max(purge_gaps) >= 1, "no purge gap between training and validation"
print(f"{len(splits)} folds | purge gap {sorted(purge_gaps)} trading day(s) at every boundary")
for s in sorted(splits, key=lambda s: s["fold"]):
    print(
        f"  fold {s['fold']}: train {pd.Timestamp(s['train_start']).date()} -> "
        f"{pd.Timestamp(s['train_end']).date()} | validate "
        f"{pd.Timestamp(s['val_start']).date()} -> {pd.Timestamp(s['val_end']).date()}"
    )
print(
    f"last validation ends {pd.Timestamp(last_val).date()}, the holdout opens {HOLDOUT_START}: "
    "holdout untouched"
)

fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
ax.set_xlabel("Trading day")
add_message_title(
    ax,
    "Folds roll forward and stop short of the holdout",
    subtitle="Boundaries as generate_cv_splits returned them; the purge is too narrow to see",
)
show_with_alt(
    fig,
    "One horizontal bar per walk-forward fold, each a three-year training stretch running into "
    "a one-year validation stretch, stepping up and to the right from the oldest fold to the "
    "most recent. A shaded region on the right marks the holdout and no bar reaches into it.",
)

# %% [markdown]
# ## E. What this notebook hands on
#
# Nothing on disk. The universe is declared in `setup.yaml::universe.symbols`, the decision grid
# is rebuilt from `_features.session_panel` by every stage that needs it, and section B.1
# established that both metals resolve on essentially every session, so there is no eligibility
# table for a later notebook to filter on.

# %% [markdown]
# ## F. What the evidence says about each setting
#
# | Setting | Evidence | Choose differently when |
# |---|---|---|
# | `universe.history_start` | D.1, the measured `dense_history_start` of both metals | a count-based re-read moves either metal's dense start |
# | `decision.snapshots`, `edge_block_minutes` | B.1, the resolved offset and label span in both seasons | the bar grid stops being UTC-aligned, or a venue's hours change |
# | `universe.symbols` | B.2 breadth, B.3 and B.5 cost against move per metal | a metal's IC* exceeds any IC this repository has measured |
# | `costs.spread_bps` | B.3, measured from thirty days of ticks on this account | the account type changes, or a fresh window disagrees |
# | `evaluation.n_splits` | D.1 slots available, D.2 fold boundaries | the folds no longer fit the development period |

# %%
print(
    f"universe.symbols {SYMBOLS}, instruments quoting per decision {breadth['n'].min()} to "
    f"{breadth['n'].max()}, short of {len(SYMBOLS)} on {short.height} of {breadth.height:,} "
    f"instants\n"
    f"decision.cadence {SETUP['decision']['cadence']} | snapshot "
    f"{SETUP['decision']['snapshot']} | offset {report['decision_offset_minutes']} min | label "
    f"span {report['label_span_minutes']} min\n"
    f"costs round trip p90 {ROUND_TRIP_P90} bps | IC* "
    f"{dict(zip(ic_star.filter(pl.col('label') == PRIMARY_LABEL)['symbol'].to_list(), ic_star.filter(pl.col('label') == PRIMARY_LABEL)['ic_star'].to_list(), strict=True))}\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, last "
    f"validation ends {pd.Timestamp(last_val).date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## Key takeaways
#
# 1. **A session snapshot is a rule; derive the instant and then check what you derived.** The
#    venue's hours move with its own daylight saving while the server's clock does not, so a
#    hard-coded server hour is correct in one season and wrong in the other.
# 2. **Measure how deep the history is with the request that reaches the server**, not the one
#    that reads the terminal's cache. The two differ by five and a half years on this account.
# 3. **Scale every move by its own instrument's cost before comparing them.** Two instruments
#    whose spreads differ by a factor of eight do not share a break-even line.
# 4. **Compute IC\* before fitting anything.** It is one division, and it can close a question
#    that would otherwise cost hundreds of trials.
# 5. **Lay folds out on dates even when decisions are intraday**, so a boundary never splits a
#    day between two sleeves of the same book.
#
# ### Known limitations
#
# - The spread was measured on the **demo** account. A real Pro account may quote differently,
#   and every cost figure here inherits that caveat.
# - Swap is read as zero on this demo and is not charged anywhere in this notebook; it is priced
#   from `16_costs` onward, after a read on the real account.
# - The sample is one monetary regime for gold as much as it is 8.5 years: the metal trends up
#   through most of it, so every statistic downstream must be reported on the **active** return
#   over a 1/N long book as well as on the raw return.
#
# **Next**: labels at the declared horizons, built on this development period.
