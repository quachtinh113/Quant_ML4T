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
# # Exness BTC 8h (exness_btc_8h): Feasibility Analysis
#
# Before building a trading strategy it is worth asking whether the data can support one at all.
# This notebook does that and nothing else: it fits no model and makes no forecast.
#
# The strategy being checked is described in `config/setup.yaml`. It trades Bitcoin as a contract
# for difference on an Exness MetaTrader 5 account, and it decides **three times a day, every day
# of the week**: at 00:00, 08:00 and 16:00 UTC. Each decision is held to the next one.
#
# That design puts four questions in front of the data, and all four are settled here rather than
# assumed:
#
# 1. **Does the decision grid exist?** MetaTrader 5 serves no eight-hour timeframe, so the grid is
#    folded from the account's four-hour bars. The fold has to land on 00:00 / 08:00 / 16:00 UTC,
#    every slot has to be there, and the slots that are not have to be counted rather than filled.
# 2. **What does a trade cost, and which of the two answers is the right one?** The spread was
#    measured twice - from thirty days of ticks today, and from the broker's own per-bar record
#    over the whole sample - and the two disagree by a factor of five. Section B.3 shows why, and
#    which one a backtest of 2018-2025 has to pay.
# 3. **Is an eight-hour move larger than what it costs to trade?** Not the spread alone: on this
#    instrument the overnight financing charge is larger than the spread on any hold that crosses
#    midnight, it falls on exactly one of the three slots, and it is tripled on a Friday.
# 4. **Does the history hold the evaluation the design declares?** Four folds of two years'
#    training and one year's validation, ahead of a twelve-month holdout that nothing here reads.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Build a decision grid on an instrument that never closes, and say what "the decision instant"
#   means when there is no session, no calendar and no closing bell
# - Tell a *live* cost measurement from an *in-sample* one, and see what happens to a backtest
#   when a spread quoted in fixed points meets a price that rises fifteen-fold
# - Put a per-night financing cost and a per-crossing spread on one axis, so that "does the move
#   pay for the trade" is a question about the total and not about half of it
# - Compute the minimum information coefficient a signal needs before it pays for its own costs -
#   the number that can close a question before a single model is fitted
# - Check that a walk-forward split fits the sample and leaves the holdout unread, on a market
#   whose calendar the splitter does not know
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. This notebook reads MT5 H4 bars (folded to 8h) and D1 bars through
# `bots/_shared/mt5_loader.py`, plus `history_depth.json` and `config/setup.yaml`, and writes
# nothing. It is a Route B fork of `case_studies/exness_gold_sess/01_feasibility_analysis.py`;
# what differs is the decision grid (a native bar grid rather than a venue session), the two-way
# cost comparison in B.3, and the swap arithmetic in B.5, which no other bot on this account needs.
#
# ## Prerequisites
#
# None beyond what the sections below define.

# %%
"""exness_btc_8h Case Study - Feasibility Analysis (Route B fork of exness_gold_sess)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars, read_history_depth
from bots._shared.sessions import ServerClock
from case_studies.exness_btc_8h._features import SLOT_MINUTES, decision_grid
from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")
set_global_seeds(42)

# %% tags=["parameters"]
CASE_STUDY_ID = "exness_btc_8h"
# START_DATE = None reads setup.yaml::universe.history_start (2018-02-16, the first date from
# which the folded 8-hour grid is complete); END_DATE is evaluation.holdout_end.
START_DATE = None
END_DATE = "2026-08-31"

# %% [markdown]
# ## Configuration
#
# Everything the strategy assumes is declared in `config/setup.yaml`, and this notebook reads
# those values rather than repeating them, so the two can never disagree.
#
# **How the history is divided.** The sample runs from 16 February 2018 - the first date this
# account serves a complete eight-hour grid - to the end of August 2026. The last twelve months
# are the **holdout**: a stretch of history that is not looked at while the strategy is being
# designed, so that when it is finally scored there, the result is not a rehearsal of choices
# already made on the same data. Everything below uses the earlier part, the development period.
#
# **When the strategy decides.** At the close of every eight-hour bar. There is no venue and no
# session here: the instrument quotes twenty-four hours a day, seven days a week, so the decision
# instant is a property of the bar grid and of nothing else. The server's clock is UTC+0 all year
# (measured), so the three instants sit at the same UTC hour in every season - which is the one
# thing this bot does not have to derive that the FX and metals bots do.
#
# **What a trade costs.** Two numbers, and they are not interchangeable. `costs.spread_bps` is the
# **in-sample** spread, measured from the broker's own per-bar record over the development window,
# and it is what a backtest pays. `costs.spread_bps_by_session` is the **live** spread, measured
# from thirty days of ticks, and it is what an order pays today. On top of either sits
# `costs.swap`, a financing charge per night that this instrument pays *every* night, weekends
# included, and triple on Friday.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

START_DATE = START_DATE or str(SETUP["universe"]["history_start"])
HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
SYMBOLS = sorted(SETUP["universe"]["symbols"])
SYMBOL = SYMBOLS[0]
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = str(SETUP["labels"]["buffer"])
LABEL_NAMES = [PRIMARY_LABEL, *SETUP["labels"]["variants"]]
HORIZON_HOURS = {n: int(str(SETUP["labels"]["horizons"][n]).rstrip("Hh")) for n in LABEL_NAMES}
SLOTS_PER_YEAR = int(SETUP["decision"]["slots_per_year"])
SLOTS_PER_DAY = len(SETUP["decision"]["snapshots_utc"])
DECISION_HOURS = sorted(int(t.split(":")[0]) for t in SETUP["decision"]["snapshots_utc"])
SERVER_CLOCK = ServerClock.from_dict(SETUP["decision"]["server_clock"])
COSTS = SETUP["costs"]
TICK_SPREAD = COSTS["spread_bps_by_session"]
IN_SAMPLE_SPREAD = COSTS["spread_bps_in_sample"]
SWAP = COSTS["swap"]
SWAP_BPS_NIGHT = float(SWAP["derived_bps_per_night"]["long"])
AVOID = SETUP["decision"]["avoid_windows_utc"]["thin_hours"]

print(
    f"{SYMBOL} on the {SETUP['decision']['cadence']} grid: {SLOTS_PER_DAY} decisions a day at "
    f"{', '.join(SETUP['decision']['snapshots_utc'])} UTC, seven days a week "
    f"= {SLOTS_PER_YEAR} slots a year. Each decision is held to the next one."
)
print(
    f"Sample {START_DATE} to {END_DATE}; the holdout {HOLDOUT_START} to {HOLDOUT_END} is not read "
    f"below. Primary label {PRIMARY_LABEL} over {HORIZON_HOURS[PRIMARY_LABEL]} hours."
)
print(
    f"Cost, live (30 days of ticks, {TICK_SPREAD['measured_on']}): round trip "
    f"{TICK_SPREAD['round_trip_p90_bps'][SYMBOL]} bps at the p90.\n"
    f"Cost, in sample (per-bar spread, {IN_SAMPLE_SPREAD['development_window'][0]} .. "
    f"{IN_SAMPLE_SPREAD['development_window'][1]}): round trip "
    f"{IN_SAMPLE_SPREAD['round_trip_p90_bps']} bps at the p90 - "
    f"{IN_SAMPLE_SPREAD['round_trip_p90_bps'] / TICK_SPREAD['round_trip_p90_bps'][SYMBOL]:.1f}x "
    f"the live figure."
)
print(
    f"Swap, long side: {SWAP['points_per_lot_per_night'][SYMBOL]['long']} points per lot per "
    f"night = {SWAP_BPS_NIGHT} bps of notional, charged EVERY night "
    f"(charges_weekends={SWAP['charges_weekends']}) and tripled on "
    f"{SWAP['rollover3days']}. Short side "
    f"{SWAP['points_per_lot_per_night'][SYMBOL]['short']}."
)

# %% [markdown]
# ## A. Orientation
#
# ### What this instrument is
#
# `BTCUSD` on this account is a contract for difference on the price of one Bitcoin in US
# dollars. No coin is delivered; the position is a bet on the price and the broker quotes a bid
# and an ask around it. A contract is **one** Bitcoin and the smallest order the broker accepts is
# a hundredth of a contract, so the position size moves in steps of 0.01 BTC - about 790 dollars
# at the price this account quoted when the costs were measured.
#
# That single fact is the reason `setup.yaml::execution.share_type` is `fractional` and not the
# repository's usual `integer`. The backtest engine sizes a position in whole *units* of the
# instrument when told to, and one unit here is one Bitcoin - roughly the whole book. Under
# `integer`, every target weight below about 0.79 would round to zero and the strategy would place
# no order at all, while the sweep completed and wrote a full set of registry rows showing a flat
# equity curve. On the FX bot the same rounding is a hundred-thousandth of a percent.
#
# ### Why the market never closing changes the design and not just the calendar
#
# Three consequences, and each one removes a whole section that the sibling bots on this account
# need:
#
# - there is no session to aggregate into, so the decision instant is simply a bar close;
# - there is no gap between a decision and its fill. The next bar opens at the instant the last
#   one closed, so `next_bar_open` execution happens at the decision instant itself. The FX bot
#   waits through a weekend for its Monday fill; this one does not wait at all;
# - and the financing charge does not stop for the weekend either. Every other instrument on this
#   account is charged swap on five nights a week; this one is charged on seven, and the triple
#   charge falls on **Friday** rather than the Wednesday that FX and metals use.
#
# ### The four questions
#
# Does the grid exist (B.1)? What does it cost, live and in sample (B.3)? Does an eight-hour move
# pay for that cost (B.4, B.5)? And does the declared evaluation fit the history (D)?

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 The decision grid
#
# The account serves four-hour bars stamped at their UTC open. `mt5_loader.resample_4h_to_8h`
# folds them in pairs onto the eight-hour grid, refusing outright any bar that does not open on a
# UTC hour divisible by four, and `_features.decision_grid` then keys the panel on the bar's
# **close**, which is the decision instant. The `timestamp` column of every artifact this pipeline
# writes is therefore the moment a decision was taken, and the claim "this bot decides at 00:00,
# 08:00 and 16:00 UTC" is literally true of the data rather than true of a comment.
#
# The first cell checks the recorded history depth before it reads a bar: that the server clock in
# `setup.yaml` is the one that was measured, and that the four-hour depth on file is the depth this
# design was sized against.

# %%
depth = read_history_depth()
measured = ServerClock.from_dict(depth["server_clock"])
assert measured.utc_offset_minutes == SERVER_CLOCK.utc_offset_minutes, (
    "setup.yaml::decision.server_clock disagrees with the clock measured in history_depth.json"
)
h4 = depth["timeframes"]["4h"][SYMBOL]
print(
    f"H4 depth on this account: {h4['n_bars']:,} bars {h4['first_server_day']} to "
    f"{h4['last_server_day']}; the 8-hour grid is folded from these."
)
print(
    f"server clock {measured.utc_offset_minutes:+d} minutes from UTC, follows_dst_of "
    f"{measured.follows_dst_of!r} - so the three decision instants sit at the same UTC hour all "
    "year, in both seasons, with no venue calendar to consult."
)

bars_4h = load_mt5_bars(
    "4h", symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE, include_spread=True
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
bars_8h = load_mt5_bars(
    "8h", symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE, include_spread=True
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
panel, report = decision_grid(bars_8h, return_report=True)
research = panel.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_datetime())

# %% [markdown]
# Four properties of the grid, each checked rather than described. The first two are assertions
# inside `decision_grid` and are re-stated here with the evidence behind them; the last two are
# specific to this bot's declared constraints.

# %%
assert report["decision_hours"] == DECISION_HOURS, (
    f"decisions land on UTC hours {report['decision_hours']}, setup.yaml declares {DECISION_HOURS}"
)
print(f"1. decision hours: {report['decision_hours']} - exactly the declared set")

# The fold has to be two H4 bars per slot. A slot built from one bar is a half bar published under
# a full bar's name, which is why universe.history_start starts after the account's ragged first
# week rather than at the first bar it serves.
per_slot = (
    bars_4h.with_columns(pl.col("timestamp").dt.truncate("8h").alias("_b"))
    .group_by("_b")
    .agg(pl.len().alias("n"))
)
half = per_slot.filter(pl.col("n") < 2)
print(
    f"2. H4 bars per 8-hour slot: {sorted(per_slot['n'].unique().to_list())}; "
    f"{half.height} slot(s) folded from fewer than two bars"
    + (f" at {sorted(str(d) for d in half['_b'].to_list())}" if 0 < half.height <= 10 else "")
)
assert half.height == 0, (
    f"{half.height} slots are folded from a single H4 bar; move universe.history_start"
)

print(
    f"3. gaps other than one slot: {report['n_gaps']}"
    + (f" -> {report['gaps']}" if report["n_gaps"] else "")
)
assert report["n_gaps"] == 0, "the 8-hour grid has a hole; see decision_grid's report"

# The fill instant. `next_bar_open` fills at the open of the bar that STARTS at the decision
# instant, and on a 24/7 grid that bar starts the moment the decision bar closed. The identity is
# measured on the data rather than argued from the definition: for every row, the open of the
# NEXT row's bar is the price at the decision instant itself.
fill_gap = (
    panel.sort("timestamp")
    .with_columns(
        (pl.col("bar_open_ts").shift(-1) - pl.col("timestamp")).alias("_gap")
    )
    .drop_nulls("_gap")
)
gaps_minutes = sorted({int(v.total_seconds() // 60) for v in fill_gap["_gap"].to_list()})
print(
    f"4. minutes between a decision and the bar it is filled on: {gaps_minutes} "
    f"(setup.yaml::decision.execution_gap_minutes = "
    f"{SETUP['decision']['execution_gap_minutes']})"
)
assert gaps_minutes == [int(SETUP["decision"]["execution_gap_minutes"])], (
    "the fill does not happen at the decision instant"
)

# bots/assets/BTCUSD.md section 5 asks for the thin 04:00-07:00 UTC window to be avoided. Nothing
# is filtered to satisfy it: the grid never lands there, and that is checked rather than claimed.
lo, hi = (int(t.split(":")[0]) for t in AVOID)
in_window = panel.filter(
    (pl.col("timestamp").dt.hour() >= lo) & (pl.col("timestamp").dt.hour() < hi)
)
print(
    f"5. decisions inside the thin {AVOID[0]}-{AVOID[1]} UTC window "
    f"bots/assets/BTCUSD.md asks to avoid: {in_window.height} - satisfied by the grid, "
    "not by a filter"
)
assert in_window.is_empty()

print(
    f"\ndevelopment rows {research.height:,} over "
    f"{research['timestamp'].n_unique():,} decision instants, "
    f"{research['timestamp'].min()} to {research['timestamp'].max()}"
)
display(
    research.group_by(pl.col("timestamp").dt.hour().alias("decision_hour"))
    .agg(pl.len().alias("slots"))
    .sort("decision_hour")
)

# %% [markdown]
# ### B.2 Is the instrument quoting at every declared slot
#
# A book of one instrument has no cross-section to fall back on, so the breadth question the
# multi-asset bots ask ("how many names are quoting?") collapses into a simpler and harsher one:
# is there a bar at every slot the design declares? B.1 already answered it for the fold and the
# gaps; this is the same question asked of the calendar, which is the form a re-download would
# break first.

# %%
expected = pl.datetime_range(
    research["timestamp"].min(),
    research["timestamp"].max(),
    interval=f"{SLOT_MINUTES}m",
    eager=True,
)
present = set(research["timestamp"].to_list())
missing = [t for t in expected.to_list() if t not in present]
by_weekday = (
    research.group_by(pl.col("timestamp").dt.weekday().alias("weekday"))
    .agg(pl.len().alias("slots"))
    .sort("weekday")
)
print(
    f"declared slots {len(expected):,}, present {research.height:,}, missing {len(missing)} "
    f"({len(missing) / max(len(expected), 1):.4%})"
)
print("slots by weekday (1 = Monday); a 24/7 instrument should be flat across all seven:")
display(by_weekday)
weekend_share = float(
    research.select((pl.col("timestamp").dt.weekday() >= 6).mean()).item()
)
print(
    f"{weekend_share:.1%} of decision slots fall on a Saturday or a Sunday. On every other bot "
    "on this account that share is zero, and it is the reason this one cannot reuse their "
    "session calendars, their swap rule or their cost buckets."
)

# %% [markdown]
# ### B.3 What a round trip costs - and why there are two answers
#
# Entering a position and leaving it crosses the spread twice. On this account the spread is
# quoted as a near-constant number of **points**, and this is where the two measurements come
# apart:
#
# - measured from ticks over the last thirty days it is a flat 1,000 points, which at today's
#   price is about 1.3 basis points and a round trip of 3.2;
# - measured from the broker's own per-bar record over the whole development window it is a median
#   of 1,640 points - but the price was 6,700 dollars in 2018 and 103,500 in 2025, so the *same*
#   spread in points is 14.6 basis points in the first year and 2.1 in the last.
#
# Neither number is wrong. They answer different questions, and using the live one to price a
# backtest of 2018-2025 would understate its cost by a factor of five - concentrated in the oldest
# years, which is where the training data is. `setup.yaml::costs.spread_bps` therefore carries the
# in-sample range, which is what the engine charges; the tick table is what a live order and the
# go-live gate are read against.
#
# The figure below is that argument, drawn.

# %%
spread_bps = (
    research.with_columns(
        (pl.col("spread") * float(COSTS["contract"]["point"]) / pl.col("close") * 1e4).alias(
            "spread_bps"
        )
    )
    .with_columns(pl.col("timestamp").dt.year().alias("year"))
)
by_year = (
    spread_bps.group_by("year")
    .agg(
        pl.col("spread_bps").median().alias("p50_bps"),
        pl.col("spread_bps").quantile(0.9).alias("p90_bps"),
        pl.col("spread").median().alias("p50_points"),
        pl.col("close").median().alias("median_price"),
        pl.len().alias("slots"),
    )
    .sort("year")
)
display(by_year)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(by_year["year"], by_year["p50_bps"], "o-", color=COLORS["blue"], lw=1.8, label="per-bar spread, median (bps)")
ax.fill_between(
    by_year["year"], by_year["p50_bps"], by_year["p90_bps"], color=COLORS["blue"], alpha=0.15,
    label="median to 90th percentile",
)
ax.axhline(
    float(TICK_SPREAD[SYMBOL]["all"][1]), color=COLORS["copper"], ls="--", lw=1.8,
    label=f"live tick p90, measured {TICK_SPREAD['measured_on']}",
)
ax.set_xlabel("Year")
ax.set_ylabel("Quoted spread (bps of price, per crossing)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
add_message_title(
    ax,
    "The spread is flat in points and therefore falls as the price rises",
    subtitle="Per-bar spread of the decision bar, development period, against today's tick measurement",
)
show_with_alt(
    fig,
    "The median per-bar spread in basis points falls from about 14.6 in 2018 to about 2.1 in "
    "2025 while the quoted spread in points is roughly unchanged; a dashed line marks the much "
    "lower spread measured from ticks today.",
)
print(
    f"development window: per-bar spread p50 {spread_bps['spread_bps'].median():.2f} bps, "
    f"p90 {spread_bps['spread_bps'].quantile(0.9):.2f} bps per crossing "
    f"(setup.yaml declares {IN_SAMPLE_SPREAD['p50_bps']} / {IN_SAMPLE_SPREAD['p90_bps']})"
)
print(
    f"today's ticks: p50 {TICK_SPREAD[SYMBOL]['all'][0]} bps, p90 {TICK_SPREAD[SYMBOL]['all'][1]} "
    f"bps - identical in every session bucket including the swap rollover, and identical at the "
    f"weekend (Saturday p90 {TICK_SPREAD['weekend']['saturday'][1]}, Sunday "
    f"{TICK_SPREAD['weekend']['sunday'][1]} against weekday "
    f"{TICK_SPREAD['weekend']['weekday'][1]}). bots/assets/BTCUSD.md warns of a wider weekend "
    "spread; on this account, today, that is not what the ticks say."
)

# %% [markdown]
# ### B.4 The other half of the cost: financing
#
# The spread is charged per crossing. The **swap** is charged per night, and on this instrument it
# is the larger of the two on any hold that crosses one. Three properties make it a design
# constraint rather than a footnote, and all three are read from `symbol_info` rather than assumed:
#
# - it is **asymmetric**: the long side pays 1,638.6 points a night and the short side pays
#   nothing. A long position pays its entire round trip again every one and a half nights;
# - it is charged on **every** night, weekends included, because the market never closes. The
#   class is derived from the Market Watch path, `Standard\Crypto\BTCUSDm`;
# - the triple charge falls on **Friday**, not the Wednesday that FX and metals on this same
#   account use.
#
# The server clock is UTC+0, so the charge lands at 00:00 UTC - which is one of the three decision
# instants. A position opened at the 16:00 decision and closed at the 00:00 one therefore crosses
# exactly one midnight and pays a full night; positions opened at 00:00 and at 08:00 cross none.
# One slot in three carries a cost the other two do not, it is knowable in advance, and the cell
# below measures it on the grid rather than reasoning about it.

# %%
slot_cost = research.with_columns(
    ((pl.col("timestamp") + pl.duration(minutes=SLOT_MINUTES)).dt.date() > pl.col("timestamp").dt.date())
    .cast(pl.Int8)
    .alias("crosses_midnight"),
    (pl.col("timestamp").dt.weekday() == 5).cast(pl.Int8).alias("friday"),
    pl.col("timestamp").dt.hour().alias("decision_hour"),
)
display(
    slot_cost.group_by("decision_hour")
    .agg(
        pl.col("crosses_midnight").mean().alias("share_paying_swap"),
        (pl.col("crosses_midnight") * pl.col("friday")).mean().alias("share_paying_triple"),
        pl.len().alias("slots"),
    )
    .sort("decision_hour")
)
nights_per_hold = slot_cost["crosses_midnight"].mean()
print(
    f"{nights_per_hold:.1%} of eight-hour holds cross a server midnight - one slot in three, by "
    "construction. On a long position that costs "
    f"{abs(SWAP_BPS_NIGHT):.2f} bps, or {abs(SWAP_BPS_NIGHT) * 3:.2f} bps into a Friday, on top "
    "of the spread."
)
print(
    f"Annualised, a long position held continuously pays "
    f"{abs(float(SWAP['derived_pct_per_year_at_measured_mid']['long'])):.1f} % of notional a year "
    f"at the measured mid of {SWAP['measured_mid']:,.0f} - "
    "365 nights plus 104 extra for the Friday triple."
)

# %% [markdown]
# ### B.5 Move size against total cost
#
# Now the two halves go on one axis. For every decision the total cost of taking a position and
# leaving it at the next decision is
#
# $$c_t = 2 \times \text{spread}_t + \mathbb{1}[\text{the hold crosses midnight}] \times k_t \times \text{swap}$$
#
# where $\text{spread}_t$ is the broker's own per-bar spread on the decision bar - point-in-time,
# and falling through the sample as the price rises - and $k_t$ is 3 into a Friday and 1 otherwise.
# The exceedance curve then reads: *what fraction of eight-hour moves are at least this many times
# their own cost?* Break-even sits at 1 for every row whatever year it is in, which is what makes
# 2018 and 2025 comparable on one chart.
#
# Two curves are drawn, because the answer is not the same for the two directions: the short side
# pays no financing at all on this account, so its cost is the spread alone.

# %%
moves = (
    spread_bps.sort("timestamp")
    .with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1).abs().alias("abs_move"),
        ((pl.col("timestamp") + pl.duration(minutes=SLOT_MINUTES)).dt.date() > pl.col("timestamp").dt.date())
        .cast(pl.Float64)
        .alias("_night"),
        pl.when(pl.col("timestamp").dt.weekday() == 5).then(3.0).otherwise(1.0).alias("_mult"),
    )
    .drop_nulls("abs_move")
    .with_columns(
        (2 * pl.col("spread_bps")).alias("cost_short_bps"),
        (
            2 * pl.col("spread_bps")
            + pl.col("_night") * pl.col("_mult") * abs(SWAP_BPS_NIGHT)
        ).alias("cost_long_bps"),
    )
    .with_columns(
        (pl.col("abs_move") * 1e4 / pl.col("cost_long_bps")).alias("x_long"),
        (pl.col("abs_move") * 1e4 / pl.col("cost_short_bps")).alias("x_short"),
    )
)
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for col, colour, tag in (
    ("x_long", COLORS["amber"], "long: 2 x spread + swap"),
    ("x_short", COLORS["blue"], "short: 2 x spread only"),
):
    magnitude, fraction = exceedance_curve(moves[col].drop_nulls().to_numpy())
    ax.plot(magnitude, fraction, color=colour, lw=1.6, label=tag)
ax.axvline(1, color=COLORS["slate"], ls=":", lw=1.8, label="break-even on the total cost")
ax.set(xscale="log", xlim=(0.01, 500))
ax.set_xlabel("Absolute 8-hour move as a multiple of that decision's own total cost (log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Most eight-hour moves clear the round trip; the swap costs the long side a slice of them",
    subtitle="Point-in-time per-bar spread plus the nights the hold actually crosses, development period",
)
show_with_alt(
    fig,
    "Two exceedance curves on a logarithmic axis giving the fraction of absolute eight-hour moves "
    "at least a given multiple of that decision's own total cost. The short curve, which pays no "
    "financing, sits above the long one throughout.",
)
for col, tag in (("x_long", "long"), ("x_short", "short")):
    series = moves[col].drop_nulls()
    print(
        f"{tag}: {float((series >= 1).mean()):.1%} of moves clear their own cost, "
        f"{float((series >= 2).mean()):.1%} clear twice it; median multiple {series.median():.2f}"
    )
print(
    "Reading it: a move clearing its cost is necessary, not sufficient. The strategy has to pick "
    "the DIRECTION as well, and half of the moves that clear the cost go the wrong way."
)

# %% [markdown]
# ### B.6 How much of one slot's move carries into the next
#
# A position opened at one decision and closed at the next earns that slot's move. Before
# forecasting it, it is worth asking how much of it the instrument's own recent slots already
# account for. With one instrument the "panel" autocorrelation is a single series, so the 10th-to
# -90th percentile band across entities collapses onto the mean; the reference to read is the
# white-noise band, `1.96 / sqrt(T)`.

# %%
slot_returns = research.sort("timestamp").with_columns(
    (pl.col("close").shift(-1) / pl.col("close") - 1).alias("ret")
).drop_nulls("ret")
max_lags = 3 * SLOTS_PER_DAY * 7  # three weeks of slots
acf = panel_acf(
    slot_returns, entity_col="symbol", value_col="ret", max_lags=max_lags
).filter(pl.col("lag") > 0)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.axhspan(
    -acf["band"][0], acf["band"][0], color=COLORS["copper"], alpha=0.3, zorder=1,
    label="range expected from no information",
)
ax.bar(acf["lag"], acf["acf"], color=COLORS["blue"], width=0.7, zorder=3, label="autocorrelation")
for k in (SLOTS_PER_DAY, 7 * SLOTS_PER_DAY):
    ax.axvline(k, color=COLORS["slate"], ls=":", lw=1.0)
ax.set_xlabel("Decision slots between the two returns (3 = one day, 21 = one week)")
ax.set_ylabel("Autocorrelation of the 8-hour return")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
add_message_title(
    ax,
    "An eight-hour move barely predicts the next one from its own history",
    subtitle="Computed on the decision grid, development period",
)
show_with_alt(fig, "Autocorrelation of the eight-hour return against lag in decision slots.")
top = acf.sort(pl.col("acf").abs(), descending=True).head(5)
print("largest |autocorrelation| and the white-noise band:")
for row in top.iter_rows(named=True):
    print(f"  lag {int(row['lag']):3d}: {row['acf']:+.4f}  (band +-{row['band']:.4f})")

# %% [markdown]
# ### B.7 The minimum information coefficient, per label and per direction
#
# This is the cheapest strong test in the whole of phase 1 and it runs before a single model is
# fitted. A signal that ranks a forward return with information coefficient $IC$ earns, per
# decision, roughly $IC \times \sigma$ of that return, so it only pays for itself when
#
# $$IC > IC^{*} = \frac{\text{cost per decision}}{\sigma(\text{label})}$$
#
# The convention every bot on this account uses puts the **one-way** spread in the numerator, so
# the four numbers below are comparable with `exness_fx_d1` and `exness_gold_sess`. What is added
# here, and has no counterpart on those bots, is a second row per label carrying the swap: on the
# long side, on the slot that crosses midnight, the financing is part of the cost the signal has
# to beat.
#
# If an $IC^{*}$ is larger than any information coefficient measured anywhere in this repository,
# that question is answered here and no trial needs to be spent on it.

# %%
one_way_in_sample = float(IN_SAMPLE_SPREAD["p90_bps"])
one_way_live = float(TICK_SPREAD[SYMBOL]["all"][1])
ic_rows = []
grid = research.sort("timestamp")
for label, hours in HORIZON_HOURS.items():
    step = hours * 60 // SLOT_MINUTES
    span = (
        pl.col("timestamp").shift(-step).dt.epoch("s") - pl.col("timestamp").dt.epoch("s")
    ) // 60
    series = (
        grid.with_columns(span.alias("_span"))
        .with_columns(
            pl.when(pl.col("_span") == hours * 60)
            .then(pl.col("close").shift(-step) / pl.col("close") - 1)
            .otherwise(None)
            .alias("_r")
        )["_r"]
        .drop_nulls()
    )
    sigma = float(series.std())
    nights = max(1, hours // 24) if hours >= 24 else 1.0 / SLOTS_PER_DAY
    for scope, cost_bps in (
        ("spread only, in sample", one_way_in_sample),
        ("spread only, live ticks", one_way_live),
        ("spread + swap, long, in sample", one_way_in_sample + nights * abs(SWAP_BPS_NIGHT)),
    ):
        ic_rows.append(
            {
                "label": label,
                "cost basis": scope,
                "n_rows": series.len(),
                "cost_bps": round(cost_bps, 3),
                "sigma_bps": round(sigma * 1e4, 1),
                "median_abs_move_bps": round(float(series.abs().median()) * 1e4, 1),
                "ic_star": round(cost_bps * 1e-4 / max(sigma, 1e-12), 4),
            }
        )
ic_star = pl.DataFrame(ic_rows)
display(ic_star)
print(
    "IC* is the information coefficient a signal must beat before it pays for the cost in that "
    "row alone; slippage, the cost of being wrong and the trials spent finding the signal are all "
    "on top of it."
)
print(
    "The in-sample rows are the honest ones for a backtest of this history. The live rows say "
    "what the same strategy would need if it were traded today, and the gap between them is the "
    "single biggest reason a result on this instrument can look better than it will trade."
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## C. Design decisions
#
# ### C.1 Why all three slots are kept, and what would remove one
#
# The 00:00 UTC decision sits exactly on the swap rollover, and `bots/assets/BTCUSD.md` asks for
# that window to be avoided. Two measurements settle what to do about it and both were taken
# before `setup.yaml` was written. First, the spread does **not** widen at rollover on this
# account: the `rollover` bucket of the tick measurement is 1.29 / 1.58 basis points, the same to
# within a hundredth of a point as every other bucket. Second, the swap does bite, but it bites on
# the **16:00** slot rather than the 00:00 one - it is the hold that crosses midnight that pays,
# not the decision taken at it.
#
# So no slot is dropped. Dropping one would remove a third of the sample to avoid a cost that is
# knowable in advance and can be carried as a feature instead (`pays_swap_night`,
# `pays_triple_swap`). Whether the 16:00 slot survives its own cost is a question for the cost
# stage, and the filters that would answer it are declared in
# `setup.yaml::decision.session_filter_values` - each of them one more trial in the Deflated
# Sharpe Ratio, and none of them run yet.
#
# ### C.2 What would send this design back
#
# Three results would, and each is measured where its evidence is. The one this notebook could
# have produced is a **grid failure**: a fold that did not land on the declared hours, a slot built
# from one bar, or a hole in the series would all have raised in B.1. The second is a **cost
# failure**, opened in B.3 and B.7 and settled in `16_costs` against the trades a backtest actually
# places - and on this instrument the swap makes that a harder gate than on any sibling bot. The
# third belongs to Chapter 7: whether any signal has a relationship to the eight-hour return at all.
#
# ### C.3 What the strategy does with the forecast
#
# `setup.yaml::mapping.class` is `per_asset_signal_timing`, not a cross-sectional rank. **One**
# instrument: a percentile over one name is the constant 0.5 at every instant, and a
# cross-sectional information coefficient over one row is undefined - not weak, undefined. So
# every mechanism the templates use to compare names is removed, and what replaces it is a
# comparison of this instrument with **its own** trailing scores.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 How much an evaluation has to spend
#
# The unit a strategy spends is the decision it takes. This grid holds three a day, every day, so
# the development period is counted in slots below and compared against what a complete panel
# would carry.

# %%
dev_days = research["timestamp"].dt.date().n_unique()
print(
    f"development decision instants {research.height:,} over {dev_days:,} calendar dates "
    f"({research.height / max(dev_days, 1):.2f} a date; the design declares {SLOTS_PER_DAY})"
)
print(
    f"holdout, not read here: {panel.filter(pl.col('timestamp') >= pl.lit(HOLDOUT_START).str.to_datetime()).height:,} slots"
)
for freq in ("4h", "daily"):
    r = depth["timeframes"][freq][SYMBOL]
    print(
        f"  {freq} depth on this account: {r['n_bars']:,} bars "
        f"{r['first_server_day']} to {r['last_server_day']}"
    )

# %% [markdown]
# ### D.2 The folds
#
# A model is fitted on one stretch of history and evaluated on the stretch that follows it; each
# such pair is a **fold**. The gap between training and validation is the **purge**, and it has to
# be at least as wide as the label reaches forward, or a training row's outcome would be known
# inside the validation window. `labels.buffer` sets it - 24 hours, three times the primary
# label's own horizon - and `evaluation.calendar` is the calendar the splitter counts the windows
# on.
#
# That calendar is `crypto`, which `utils/cv_splits.py` maps to **None**: a market that never
# closes has no session calendar to count. It is also the reason `train_size` and `val_size` are
# written in **days** rather than years. With no calendar the library reads the size as a pandas
# offset alias, and `P2Y` normalises to `2YE` - a *year-end* offset - which produces windows of
# about 320 days instead of 365 and walks the oldest training start forward by more than a year.
# Measured on this grid: `P2Y`/`P1Y` gives 2,046-slot training windows and 951-slot validation
# windows starting in April 2020, against 2,187 and 1,095 starting in September 2019 for
# `P730D`/`P365D`. The sibling bots are unaffected because their calendar is `FX`, which maps to
# `CME_FX` and takes the session-counting path.
#
# The folds are laid out on the decision instants themselves, and the splitter is handed the
# timeline and no prices. It applies `evaluation.holdout_start` itself, so nothing the holdout
# contains reaches a number computed above.

# %%
timeline = panel.select("timestamp").unique().sort("timestamp")
splits = generate_cv_splits(
    timeline, case_study_id=CASE_STUDY_ID, label_buffer=LABEL_BUFFER, date_col="timestamp"
)
grid_np = timeline["timestamp"].to_numpy()
purge_gaps = {
    int(((grid_np > np.datetime64(s["train_end"])) & (grid_np < np.datetime64(s["val_start"]))).sum())
    for s in splits
}
last_val = max(split["val_end"] for split in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert last_val < np.datetime64(HOLDOUT_START), "a fold reaches into the holdout"
assert min(purge_gaps) >= 1, "no purge gap between training and validation"
print(
    f"{len(splits)} folds | purge gap {sorted(purge_gaps)} decision slot(s) at every boundary "
    f"(labels.buffer = {LABEL_BUFFER} = {int(pd.Timedelta(LABEL_BUFFER.replace('H', 'h')) / pd.Timedelta(minutes=SLOT_MINUTES))} slots)"
)
for s in sorted(splits, key=lambda s: s["fold"]):
    n_train = int(
        ((grid_np >= np.datetime64(s["train_start"])) & (grid_np <= np.datetime64(s["train_end"]))).sum()
    )
    n_val = int(
        ((grid_np >= np.datetime64(s["val_start"])) & (grid_np <= np.datetime64(s["val_end"]))).sum()
    )
    print(
        f"  fold {s['fold']}: train {pd.Timestamp(s['train_start']).date()} -> "
        f"{pd.Timestamp(s['train_end']).date()} ({n_train:,} slots) | validate "
        f"{pd.Timestamp(s['val_start']).date()} -> {pd.Timestamp(s['val_end']).date()} "
        f"({n_val:,} slots)"
    )
oldest_train = min(pd.Timestamp(s["train_start"]) for s in splits)
warmup = int((grid_np < np.datetime64(oldest_train)).sum())
print(
    f"oldest training window opens {oldest_train.date()}, leaving {warmup:,} decision slots of "
    f"warmup before it - the budget every feature window in setup.yaml::features.windows has to "
    f"fit inside."
)
print(
    f"last validation ends {pd.Timestamp(last_val).date()}, the holdout opens {HOLDOUT_START}: "
    "holdout untouched"
)

fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
ax.set_xlabel("Decision instant")
add_message_title(
    ax,
    "Folds roll forward and stop short of the holdout",
    subtitle="Boundaries as generate_cv_splits returned them; the purge is too narrow to see",
)
show_with_alt(
    fig,
    "One horizontal bar per walk-forward fold, each a two-year training stretch running into a "
    "one-year validation stretch, stepping up and to the right from the oldest fold to the most "
    "recent. A shaded region on the right marks the holdout and no bar reaches into it.",
)

# %% [markdown]
# ## E. What this notebook hands on
#
# Nothing on disk. The universe is one instrument declared in `setup.yaml::universe.symbols`, the
# decision grid is rebuilt from `_features.decision_grid` by every stage that needs it, and
# section B.1 established that it resolves on every slot of the development window, so there is no
# eligibility table for a later notebook to filter on.

# %% [markdown]
# ## F. What the evidence says about each setting
#
# | Setting | Evidence | Choose differently when |
# |---|---|---|
# | `universe.history_start` | B.1, the fold census and the gap report | a re-download changes where the complete grid starts |
# | `decision.cadence`, `snapshots_utc` | B.1, the decision hours and the fill-gap identity | the account's H4 grid moves off the four-hour boundary |
# | `execution.share_type` | A, one unit is one Bitcoin and one Bitcoin is most of the book | the account's contract size changes |
# | `costs.spread_bps` | B.3, the per-bar record over the development window | the sample window moves, or a re-download changes the field |
# | `costs.swap` | B.4, `symbol_info` on this account | the real Pro account reports differently from the demo |
# | `decision.session_filter_values` | B.4, one slot in three pays the swap | the swap goes to zero on the real account |
# | `evaluation.n_splits`, `train_size` | D.1 slots available, D.2 fold boundaries and the warmup left | the folds no longer fit, or a feature window outgrows the warmup |

# %%
print(
    f"universe.symbols {SYMBOLS} | decision.cadence {SETUP['decision']['cadence']} at "
    f"{SETUP['decision']['snapshots_utc']} UTC | fill gap {gaps_minutes} minutes\n"
    f"grid: {research.height:,} development slots, {report['n_gaps']} gaps, {half.height} "
    f"half-folded slots, {len(missing)} missing\n"
    f"costs: in-sample round trip p90 {2 * one_way_in_sample:.1f} bps, live round trip p90 "
    f"{2 * one_way_live:.2f} bps, swap {SWAP_BPS_NIGHT:.2f} bps a night on "
    f"{nights_per_hold:.0%} of holds\n"
    f"IC* ({PRIMARY_LABEL}, spread + swap, in sample) "
    f"{ic_star.filter((pl.col('label') == PRIMARY_LABEL) & (pl.col('cost basis') == 'spread + swap, long, in sample'))['ic_star'][0]}\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, oldest "
    f"train {oldest_train.date()}, warmup {warmup:,} slots, last validation ends "
    f"{pd.Timestamp(last_val).date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## Key takeaways
#
# 1. **On an instrument that never closes, the decision instant is a property of the bar grid -
#    so check the grid.** A fold that lands off the declared hours, a slot built from one bar
#    instead of two, or a hole in the series are all silent failures; each is an assertion here.
# 2. **A spread quoted in points is not a spread quoted in basis points.** When the price rises
#    fifteen-fold, the same quote is five times cheaper, and pricing a nine-year backtest at
#    today's number understates its cost exactly where the training data is.
# 3. **Put financing on the same axis as the spread before deciding anything.** Here the swap is
#    larger than the round trip on any hold that crosses midnight, it falls on one slot in three,
#    and it is asymmetric between the long and the short side.
# 4. **Compute IC\* before fitting anything.** It is one division, and it can close a question
#    that would otherwise cost hundreds of trials.
# 5. **A calendar name is not decoration.** `crypto` maps to *no calendar*, which silently changed
#    what `P2Y` means to the fold splitter; the sizes are written in days because the measurement
#    said so.
#
# ### Known limitations
#
# - Every cost figure here comes from the **demo** login. A real Pro account may quote a different
#   spread and, far more importantly, a different swap - the demo's zero on the short side is
#   unusual enough to be treated as unverified.
# - The per-bar spread field is the broker's own record for a bar; the notebook does not know
#   whether it is the bar's opening, minimum or typical spread, only that it is point-in-time.
# - The sample is one monetary regime as much as it is 8.5 years: Bitcoin rises through most of
#   it, so every statistic downstream must be reported on the **active** return over a buy-and-hold
#   book as well as on the raw return.
#
# **Next**: labels at the declared horizons, built on this development period.
