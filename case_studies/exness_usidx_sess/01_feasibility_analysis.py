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
# > began as `case_studies/exness_fx_d1/01_feasibility_analysis.py` with the study id, the loader and the population names
# > changed. **This bot has TWO instruments (`US500`, `USTEC`) and no cross-section**, and one
# > label per experiment workspace. Any sentence below that speaks of five currency pairs, of
# > ranking assets against each other, of `usd_corr_*` / `gold_corr_*` columns or of a dollar
# > regime is template residue describing a different bot; where such prose asserted a NUMBER it
# > has been deleted rather than reworded (see `BOT.md`, Decisions log, 2026-09-08 third pass),
# > and where it survives as a description of the template it says so explicitly. Report any that
# > does neither. This banner follows `case_studies/xau_fx_mt5/13_backtest.py`.

# %% [markdown]
# # Exness US index sessions (exness_usidx_sess): Feasibility Analysis
#
# Before building a trading strategy it is worth asking whether the data can support one at all.
# This notebook does that and nothing else: it fits no model and makes no forecast.
#
# The strategy being checked is described in `config/setup.yaml`. It trades two US index CFDs -
# US500 and USTEC - on an Exness MetaTrader 5 account, and it decides **twice** in each New York
# cash session: once shortly after the opening auction and once at the closing auction. The first
# decision is held to the cash close (the `intraday` label), the second is held overnight to the
# next session's first decision (the `overnight` label). Chapter 8 section 8.1 is where that
# split comes from: an index session is two different processes stacked on one bar, and most of
# the long-run return of a US index is earned while the cash market is shut.
#
# `setup.yaml` says which indices it trades, which moments of the session it decides at, what a
# trade is assumed to cost, and how the history is divided between designing the strategy and
# testing it. This notebook checks each of those assumptions against the data and reports what it
# finds.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Derive a session's decision instants from an exchange calendar rather than from a table, so
#   that daylight saving and half-days move them without anyone editing a constant
# - Separate two facts a CFD account keeps apart: what the broker's clock does, and what the
#   instrument's trading hours do
# - Check that the bar a decision would be filled at exists, and turn its absence into "no order"
#   rather than into a fill at a price the decision never saw
# - Read off one chart what fraction of session moves is larger than the spread that has to be
#   crossed to trade them
# - Price a financing cost that the backtest engine cannot see, and compare it with the spread
# - Check that a walk-forward split of a short history fits the sample and leaves the test period
#   unread
#
# ## Book reference
#
# Chapter 6, sections 6.2-6.6, and Chapter 8 section 8.1 for the overnight / intraday split. This
# notebook reads MT5 H1 bars and `history_depth.json` (both written by
# `bots/_shared/mt5_loader.py:download_mt5_bars`), the spread table written by
# `bots/exness_usidx_sess/tools/measure_index_costs.py`, and `config/setup.yaml`; it writes
# nothing. It is a Route B fork of `case_studies/exness_fx_d1/01_feasibility_analysis.py`; the
# differences are the loader frequency, the two decision instants per session, the trading-hours
# DST assertion in section B.2, the execution-bar audit in B.3 and the swap arithmetic in B.6.
#
# ## Prerequisites
#
# None beyond what the sections below define.

# %%
"""exness_usidx_sess Case Study - Feasibility Analysis (Route B fork of exness_fx_d1)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars, read_history_depth
from bots._shared.sessions import ServerClock
from case_studies.exness_usidx_sess._features import (
    SPECS,
    daily_break_by_month,
    load_daily_bars,
    session_grid,
    session_panel,
    warmup_expectations,
)
from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

SYMBOL_COLORS = (COLORS["blue"], COLORS["amber"])
SPEC_COLORS = (COLORS["blue"], COLORS["copper"])

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "exness_usidx_sess"
# START_DATE = None reads setup.yaml::universe.history_start (2022-10-25, the first H1 bar this
# account serves for either index); END_DATE is evaluation.holdout_end.
START_DATE = None
END_DATE = "2026-08-31"

# %% [markdown]
# ## Configuration
#
# Everything the strategy assumes is declared in `config/setup.yaml`, and this notebook reads
# those values rather than repeating them, so the two can never disagree.
#
# **How the history is divided.** The sample runs from 25 October 2022, the first one-hour bar
# this account serves for either index, to the end of August 2026. That is 3.9 years, a quarter
# of what the FX bot has, and it is the reason the fold geometry below is small. The last six
# months are the *holdout*: a stretch that is not looked at while the strategy is designed, so
# that when it is finally evaluated there, the result is not a rehearsal of choices already tuned
# on the same data.
#
# **What the strategy trades.** Two indices. Two is not a cross-section: ranking two instruments
# against each other is a spread trade, so `mapping.class` is `time_series_threshold` and each
# index is compared with its own past scores. Nothing in this notebook counts "how many
# instruments are quoting", the question the FX bot asks, because the answer is always two or the
# session does not exist.
#
# **When it decides.** Twice per NYSE cash session, both instants derived from the calendar:
# thirty minutes after the opening auction, and at the closing auction. `decision.open_delay_minutes`
# is where the thirty comes from, and `bots/assets/US500.md` section 5 is why it is there.
#
# **What a trade is assumed to cost.** A spread per crossing, measured on this account's own
# ticks (section B.4), and - for the overnight label only - a financing charge per night that the
# backtest engine has no way to express (section B.6).

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

START_DATE = START_DATE or str(SETUP["universe"]["history_start"])
HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
UNIVERSE = sorted(SETUP["universe"]["symbols"])
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_NAMES = [PRIMARY_LABEL, *SETUP["labels"]["variants"]]
LABEL_BUFFER = SETUP["labels"]["buffer"]
DECISION = SETUP["decision"]
SESSION_CALENDAR = DECISION["session_calendar"]
OPEN_DELAY = int(DECISION["open_delay_minutes"])
CLOSE_TOLERANCE = int(DECISION["session_close_tolerance_minutes"])
TRADE_HOURS_DST_ZONE = DECISION["trade_hours_follow_dst_of"]
SERVER_CLOCK = ServerClock.from_dict(DECISION["server_clock"])
COSTS = SETUP["costs"]
# The cost declaration, corrected on the second pass: the p90 quoted spread at the bars this
# bot fills on, over the DEVELOPMENT window, from the `spread` field of this account's own H1
# tape. The thirty-day tick table is kept under its own key as a record and is no longer the
# declaration - its window lies inside the holdout and it cannot see the 2024-10 regime break.
SPREAD_BPS = COSTS["spread_bps"]
SPREAD_BY_YEAR = COSTS["spread_bps_by_year"]
TICK_RECORD = COSTS["spread_bps_by_session_tick_record"]
MIN_FILL_COVERAGE = float(DECISION["min_fill_coverage"])
SWAP = COSTS["swap"]
PERIODS_PER_YEAR = SETUP["evaluation"]["periods_per_year"]
BAR_MINUTES = 60

print(f"Sample: {START_DATE} to {END_DATE}")
print(f"  Development period, used everywhere below:  {START_DATE} to {HOLDOUT_START}")
print(f"  Holdout, not read by this notebook:         {HOLDOUT_START} to {HOLDOUT_END}")
print(f"Universe: {', '.join(UNIVERSE)} ({SETUP['universe']['n_assets']} indices, no cross-section)")
print(
    f"Decisions: {SESSION_CALENDAR} cash open + {OPEN_DELAY} minutes, and the {SESSION_CALENDAR} "
    f"cash close; both derived from the calendar, both on the H1 grid"
)
print(f"Labels: {', '.join(LABEL_NAMES)} (primary {PRIMARY_LABEL})")
print(
    f"Server clock: {SERVER_CLOCK.utc_offset_minutes:+d} minutes from UTC, measured "
    f"{SERVER_CLOCK.measured_at:%Y-%m-%d}, follows_dst_of "
    f"{DECISION['server_clock']['follows_dst_of']}"
)
print(f"Trading hours declared to follow the DST of: {TRADE_HOURS_DST_ZONE} (asserted in B.2)")

# %% [markdown]
# ## A. Orientation
#
# ### What a US index CFD is
#
# `US500m` and `USTECm` on this account are contracts for difference on the S&P 500 and the
# Nasdaq 100. `trade_contract_size` is 1, so one unit is one index point: about $7,700 of
# exposure per unit of US500 and $29,500 per unit of USTEC at the September 2026 quote. They
# quote around the clock on the CME Globex schedule, with a daily break, and they are not the
# cash indices - they are the broker's price, and the only price this account can trade.
#
# ### Why the session is split in two
#
# The cash market is open for six and a half hours and shut for seventeen and a half. Those two
# stretches are different: the overnight stretch collects the earnings releases, the foreign
# session and the futures-led repricing, and it is where the long-run return of a US index has
# historically been earned; the cash stretch is where the opening auction, the intraday flow and
# the closing auction live. `bots/assets/US500.md` section 10 states the split and Chapter 8
# section 8.1 builds it. A model that predicts the sum of the two predicts an average of two
# processes, so this bot predicts them separately: two labels, two snapshots, two specs.
#
# ### The three questions this notebook asks
#
# 1. Do the two decision instants exist on the bar grid, in both daylight-saving regimes and on
#    half-days, and is there a bar to fill the order at?
# 2. Is the move over a session large enough, often enough, to pay the spread - and for the
#    overnight label, the spread plus the financing?
# 3. Does the walk-forward split fit a 3.9-year history without touching the holdout?

# %% [markdown]
# ## B. Sessions, hours and cost feasibility
#
# ### B.1 The bar grid, and the two panels built on it

# %%
bars = load_mt5_bars("1h", symbols=UNIVERSE, start_date=START_DATE, end_date=END_DATE).with_columns(
    pl.col("timestamp").dt.replace_time_zone(None)
)
depth = read_history_depth()
print(f"{len(bars):,} H1 bars, {bars['timestamp'].min()} to {bars['timestamp'].max()} UTC")
for symbol in UNIVERSE:
    record = depth["timeframes"]["1h"][symbol]
    print(
        f"  {symbol}: {record['n_bars']:,} H1 bars on the server, {record['first_server_day']} to "
        f"{record['last_server_day']}"
    )

panels = {
    spec: session_panel(
        bars,
        spec=spec,
        calendar=SESSION_CALENDAR,
        tolerance_minutes=CLOSE_TOLERANCE,
        open_delay_minutes=OPEN_DELAY,
    )
    for spec in SPECS
}
for spec, panel in panels.items():
    print(
        f"{spec:9s} panel: {panel.height:,} index-sessions, "
        f"{panel['timestamp'].min()} to {panel['timestamp'].max()}"
    )

# %% [markdown]
# The decision instants are read back from the panel rather than asserted from a table. In a
# winter month the cash session runs 14:30-21:00 UTC and the two instants are 15:00 and 21:00; in
# summer both move an hour earlier; on a half-day the close moves three hours earlier again. The
# table below is the evidence, not the declaration.

# %%
instants = (
    pl.concat(
        [
            panel.select(
                pl.lit(spec).alias("spec"),
                pl.col("decision_ts").dt.hour().alias("hour_utc"),
                pl.col("timestamp").dt.month().alias("month"),
            )
            for spec, panel in panels.items()
        ]
    )
    .group_by(["spec", "hour_utc"])
    .agg(pl.len().alias("index_sessions"), pl.col("month").unique().sort().alias("months"))
    .sort(["spec", "hour_utc"])
)
display(instants)

# %% [markdown]
# ### B.2 The server clock and the trading hours are two different facts
#
# `decision.server_clock.follows_dst_of` is `null`: the broker's clock is UTC+0 all year, measured
# on 2026-09-05. That is a fact about timestamps. It is *not* a fact about when the instrument
# trades, and on this account the two disagree: the daily Globex break moves with New York
# daylight saving while the clock does not. A winter month therefore carries a 21:00 UTC bar and
# (almost) no 22:00 bar; a summer month the reverse.
#
# `sessions_mt5.json` records an eight-week summer snapshot, which would hard-code the summer
# regime into a bot that reads it. The declaration `decision.trade_hours_follow_dst_of` is the
# alternative, and the measurement below is what keeps it honest.
#
# The quantity that has to be stable is the break in **New York local time**, not the break in
# UTC. `daily_break_by_month` finds the quiet hours of each month on Monday to Thursday - so the
# Friday close and the Sunday open, which are week boundaries rather than the daily break, cannot
# be mistaken for it - and converts them to New York time. Two things follow, and both are
# asserted:
#
# 1. the break **ends at 18:00 New York in every readable month**, in both daylight-saving
#    regimes and across the change from a two-hour break (2022, 16:00-18:00 New York) to a
#    one-hour one (from 2023, 17:00-18:00);
# 2. the UTC hour it falls on therefore **moves by one** between summer and winter, which is what
#    a hard-coded 21:00 or 22:00 would get wrong for half of every year.
#
# A month in which New York changes offset contains both regimes at once, so its quiet hours are
# a mixture; those months are exempt rather than guessed at, and so is any month whose quiet set
# is empty.

# %%
regime = daily_break_by_month(bars, UNIVERSE[0], zone=TRADE_HOURS_DST_ZONE)
readable = regime.drop_nulls("break_end_ny").filter(~pl.col("has_dst_transition"))
display(
    readable.group_by(["is_dst", "break_utc", "break_ny", "break_end_ny"])
    .agg(pl.len().alias("months"), pl.col("month").min().alias("from"), pl.col("month").max().alias("to"))
    .sort(["from"])
)
ends = set(readable["break_end_ny"].to_list())
print(
    f"{readable.height} of {regime.height} months are readable and free of a daylight-saving "
    f"transition; in those the break ends at {sorted(ends)} New York time"
)
assert ends == {18}, (
    f"the daily break ends at {sorted(ends)} New York time, not always at 18:00: the trading "
    f"hours do not follow {TRADE_HOURS_DST_ZONE} and decision.trade_hours_follow_dst_of is wrong"
)
summer_utc = {h for row in readable.filter("is_dst")["break_utc"].to_list() for h in row}
winter_utc = {h for row in readable.filter(~pl.col("is_dst"))["break_utc"].to_list() for h in row}
print(
    f"break hours in UTC: summer {sorted(summer_utc)}, winter {sorted(winter_utc)} - one hour "
    f"apart, so a hard-coded UTC break hour is wrong for half of every year"
)
assert max(winter_utc) == max(summer_utc) + 1, (
    "the winter break is not one UTC hour later than the summer break"
)

# %% [markdown]
# ### B.3 Is there a bar to fill the order at?
#
# `decision.execution_delay` is `next_bar_open`: the order goes at the open of the bar that opens
# at the decision instant. On a 24-hour instrument that bar is normally there. On this one it
# sometimes is not, and the reason is structural rather than an outage:
#
# - the index shuts for the weekend **at** the Friday cash close, so a decision taken at the
#   Friday close has no next bar until Sunday evening;
# - the daily break sat directly on the cash close in the early sample. Counting the bars: in
#   2022 the break ran two hours (20:00-22:00 UTC in summer, 21:00-23:00 in winter), from 2023 it
#   is one hour and starts an hour after the close.
#
# The rule is **no bar within one bar of the decision, no order**. Filling hours later at a price
# the decision never saw would be a different strategy wearing this one's backtest. The rate is
# reported here, per spec and per year, and asserted in `bots/exness_usidx_sess/tests/test_sessions.py`.

# %%
coverage = pl.concat(
    [
        panel.select(
            pl.lit(spec).alias("spec"),
            "symbol",
            "timestamp",
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.weekday().alias("weekday"),
            pl.col("exec_open").is_null().alias("no_fill"),
            pl.col("label_close").is_null().alias("no_label"),
            (pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date()).alias("development"),
        )
        for spec, panel in panels.items()
    ]
)
development_coverage = coverage.filter("development")
by_year = (
    development_coverage.group_by(["spec", "year"])
    .agg(
        pl.len().alias("index_sessions"),
        pl.col("no_fill").sum().alias("no_fill"),
        pl.col("no_label").sum().alias("no_label"),
    )
    .with_columns((pl.col("no_fill") / pl.col("index_sessions")).alias("no_fill_share"))
    .sort(["spec", "year"])
)
display(by_year)
print(
    f"{development_coverage.height:,} of {coverage.height:,} symbol-sessions are development; "
    f"the {coverage.height - development_coverage.height:,} holdout rows are counted here and "
    "nowhere else in this notebook"
)
for spec, panel in panels.items():
    dev = panel.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date())
    tradable = dev.drop_nulls(["exec_open", "label_close"])
    print(
        f"{spec:9s}: {tradable.height:,} of {dev.height:,} DEVELOPMENT index-sessions are "
        f"tradable ({tradable.height / dev.height:.1%})"
    )

# %% [markdown]
# The Friday question, which the first pass of this notebook got wrong. It reported that the
# overnight label "has no Fridays", and on the 2022-2024 sample that is nearly true - the index
# shut for the weekend at the Friday cash close, so the decision had no next bar. It stopped
# being true in 2025. The table below counts, per year and on development rows only, how many of
# the overnight spec's Friday sessions carry a fill, and what share of the missing fills are
# Fridays at all. Both halves matter: the first says whether the spec can trade the day it most
# wants to (a Friday overnight position pays three nights of swap), and the second says whether
# the missing fills have a single structural cause or several.

# %%
overnight_dev = development_coverage.filter(pl.col("spec") == "overnight")
friday_view = (
    overnight_dev.filter(pl.col("weekday") == 5)
    .group_by("year")
    .agg(
        pl.len().alias("friday_symbol_sessions"),
        (~pl.col("no_fill")).sum().alias("with_a_fill"),
        (100 * (~pl.col("no_fill")).mean()).round(1).alias("with_a_fill_pct"),
    )
    .join(
        overnight_dev.filter(pl.col("no_fill"))
        .group_by("year")
        .agg(
            pl.len().alias("missing_fills"),
            (100 * (pl.col("weekday") == 5).mean()).round(1).alias("friday_share_of_missing"),
        ),
        on="year",
        how="left",
    )
    .sort("year")
)
display(friday_view)
print(
    "The regime moved: a Friday overnight decision was unfillable in 98% of 2023 and 2024 "
    "sessions, in 36% of 2025 and in none of the development part of 2026. The spec is not "
    "Friday-free, and the sentence that said so is corrected in bots/exness_usidx_sess/BOT.md."
)

# %% [markdown]
# ### B.3b Does every fold block carry enough fillable sessions to model?
#
# The year table is a description; what a modelling decision depends on is the block a model is
# actually fitted and selected on. `decision.min_fill_coverage` declares, before any model is
# fitted, the smallest share of a block's symbol-sessions that must carry both a fill and a label
# for that block to be used. It is a design declaration and not a trial: it fixes in advance how
# thin a block this bot will accept, rather than letting the answer be chosen once the results
# are in. The assertion below is the whole of its enforcement.

# %%
splits_for_coverage = generate_cv_splits(
    panels["intraday"].select("timestamp").unique().sort("timestamp"),
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    date_col="timestamp",
)
fill_rows = []
for spec, panel in panels.items():
    tradable = panel.select(
        "timestamp",
        (pl.col("exec_open").is_not_null() & pl.col("label_close").is_not_null()).alias("tradable"),
    )
    for split in sorted(splits_for_coverage, key=lambda s: s["fold"]):
        for block in ("train", "val"):
            part = tradable.filter(
                (pl.col("timestamp") >= pl.lit(split[f"{block}_start"]).cast(pl.Date))
                & (pl.col("timestamp") <= pl.lit(split[f"{block}_end"]).cast(pl.Date))
            )
            fill_rows.append(
                {
                    "spec": spec,
                    "fold": int(split["fold"]),
                    "block": block,
                    "symbol_sessions": part.height,
                    "tradable": int(part["tradable"].sum()),
                    "coverage": round(float(part["tradable"].mean()), 3),
                }
            )
fill_coverage = pl.DataFrame(fill_rows).sort(["spec", "fold", "block"])
display(fill_coverage)
worst = fill_coverage.sort("coverage").row(0, named=True)
assert worst["coverage"] >= MIN_FILL_COVERAGE, (
    f"{worst['spec']} fold {worst['fold']} {worst['block']} carries a fill on "
    f"{worst['coverage']:.1%} of its symbol-sessions, below the declared "
    f"decision.min_fill_coverage of {MIN_FILL_COVERAGE:.0%}"
)
print(
    f"Thinnest block: {worst['spec']} fold {worst['fold']} {worst['block']} at "
    f"{worst['coverage']:.1%}, against a declared floor of {MIN_FILL_COVERAGE:.0%}"
)

# %% [markdown]
# ### B.4 What a round trip costs
#
# **Corrected on the second pass.** The first version of this section read a thirty-day tick
# measurement and declared 0.66 basis points. That number is true of the thirty days it measured
# and wrong for the sample this bot is fitted on, by about four times over half of it - and it
# was read from a window lying inside the declared holdout, which is a boundary crossing on top
# of an error.
#
# The replacement reads the `spread` field of this account's **own H1 bars**, at the bars the
# two specs actually fill on (`exec_ts`), on **development rows only**. The two sources are
# comparable and that is checked rather than assumed: over exactly the tick window the tape
# reproduces the tick measurement for `US500` to the point. The tape then extends it backwards
# over the whole history, which the ticks cannot do.
#
# There is no session bucket in this version, and that is deliberate. The bucket map the first
# version used sent the overnight spec to `edge_close`, and `bots/_shared/sessions.py` tests
# `start <= t < end`, so `edge_close` is 15:30-15:59:59 New York while the overnight order fills
# on the bar **opening** at the cash close, 16:00-17:00 New York. The bucket did not contain the
# minute it was supposed to price. Reading the execution bar itself needs no bucket and cannot
# address the wrong half hour.

# %%
spread_bars = load_mt5_bars(
    "1h", symbols=UNIVERSE, start_date=START_DATE, end_date=END_DATE, include_spread=True
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
tick_window = spread_bars.filter(
    (pl.col("timestamp") >= pl.datetime(2026, 8, 8, 23, 0))
    & (pl.col("timestamp") < pl.datetime(2026, 9, 7, 23, 0))
)
print("H1 `spread` over the 30-day tick window, against the tick measurement it has to reproduce:")
for row in (
    tick_window.group_by("symbol")
    .agg(pl.col("spread").median().alias("p50"), pl.col("spread").quantile(0.9).alias("p90"))
    .sort("symbol")
    .iter_rows(named=True)
):
    recorded = TICK_RECORD[row["symbol"]]["all"]
    print(
        f"  {row['symbol']}: tape p50 {row['p50']:.0f} / p90 {row['p90']:.0f} points; the tick "
        f"record declared {recorded[0]:.2f} / {recorded[1]:.2f} bps"
    )

execution_spreads = pl.concat(
    [
        panel.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date())
        .drop_nulls("exec_ts")
        .join(
            spread_bars.select(
                "symbol", pl.col("timestamp").alias("exec_ts"), "spread", pl.col("close").alias("px")
            ),
            on=["symbol", "exec_ts"],
            how="inner",
        )
        .select(
            pl.lit(spec).alias("spec"),
            "symbol",
            pl.col("timestamp").dt.year().alias("year"),
            # point value 0.01 on both indices, so points -> price is a division by 100
            (1e4 * (pl.col("spread") / 100.0) / pl.col("px")).alias("spread_bps"),
        )
        for spec, panel in panels.items()
    ]
)
by_year_spread = (
    execution_spreads.group_by(["spec", "symbol", "year"])
    .agg(
        pl.len().alias("n"),
        pl.col("spread_bps").median().round(2).alias("p50"),
        pl.col("spread_bps").quantile(0.9).round(2).alias("p90"),
    )
    .sort(["spec", "symbol", "year"])
)
display(by_year_spread)

worst_p90 = float(by_year_spread["p90"].max())
declared_p90 = float(SPREAD_BPS["indices"][-1])
print(
    f"{execution_spreads.height:,} development execution bars; the worst per-year, per-spec p90 "
    f"is {worst_p90:.2f} bp against a declared {declared_p90:.2f} bp"
)
assert declared_p90 >= worst_p90 - 1e-9, (
    f"costs.spread_bps.indices declares {declared_p90:.2f} bp but the execution bars of the "
    f"development window reach {worst_p90:.2f} bp: the backtest would under-price its own trades"
)
round_trip = {
    spec: {symbol: 2 * declared_p90 for symbol in UNIVERSE} for spec in SPECS
}
recent_p90 = SPREAD_BY_YEAR["recent_regime_p90"]
print(
    f"Round trip charged by the engine: {2 * declared_p90:.2f} bp on both indices. In the regime "
    f"since {SPREAD_BY_YEAR['recent_regime_from']} the same round trip is "
    + ", ".join(f"{s} {2 * float(v):.2f} bp" for s, v in sorted(recent_p90.items()))
    + " - so every post-2024-10 trade is charged about four times what it would have cost, which "
    "is the price of one engine cost number over a sample holding two regimes."
)

# %% [markdown]
# ### B.5 Move size against cost
#
# One curve per index and spec: the fraction of session moves whose size is at least the value on
# the horizontal axis. Read against the round-trip cost line it says what fraction of the moves
# the strategy is trying to catch are large enough to pay for being caught. A curve that is
# already below the line at the cost is a strategy that has to be right far more often than it
# can be.

# %%
moves = {}
for spec, panel in panels.items():
    tradable = panel.drop_nulls(["exec_open", "label_close"])
    moves[spec] = tradable.with_columns(
        ((pl.col("label_close") / pl.col("exec_open") - 1) * 1e4).alias("move_bps")
    ).filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date())

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"], sharey=True)
for ax, spec in zip(axes, SPECS, strict=True):
    for symbol, colour in zip(UNIVERSE, SYMBOL_COLORS, strict=False):
        values = moves[spec].filter(pl.col("symbol") == symbol)["move_bps"].abs().to_numpy()
        if values.size == 0:
            continue
        magnitude, share = exceedance_curve(values)
        ax.plot(magnitude, share, color=colour, label=symbol)
        ax.axvline(round_trip[spec][symbol], color=colour, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Move size (basis points, log scale)")
    ax.set_title(spec)
    ax.legend(frameon=False)
axes[0].set_ylabel("Fraction of sessions at least this large")
add_message_title(
    axes[0],
    "Most session moves clear the spread; the spread is not what decides this bot",
    subtitle="Dashed lines: the round-trip p90 spread of each index in its own execution bucket",
)
show_with_alt(
    fig,
    "Two panels, intraday on the left and overnight on the right. Each draws a falling curve "
    "per index showing the fraction of sessions whose absolute move is at least the size on the "
    "horizontal axis, on a logarithmic scale. Vertical dashed lines mark each index's round-trip "
    "spread; both curves are still high where the lines fall.",
)
for spec in SPECS:
    for symbol in UNIVERSE:
        values = moves[spec].filter(pl.col("symbol") == symbol)["move_bps"].abs().to_numpy()
        if values.size == 0:
            continue
        share = float((values >= round_trip[spec][symbol]).mean())
        print(
            f"{spec:9s} {symbol}: median |move| {np.median(values):.1f} bp, "
            f"{share:.1%} of sessions clear the {round_trip[spec][symbol]:.2f} bp round trip"
        )

# %% [markdown]
# ### B.6 The cost the engine cannot see
#
# A CFD position held across the broker's midnight is charged **swap**, quoted in points per lot
# per night. `symbol_info` on 2026-09-08 reports `swap_long` -147.4 points on US500 and -592.7 on
# USTEC, `swap_short` 0.0 on both, and `swap_rollover3days` 5, meaning Friday is charged three
# times. At `point` 0.01 and `trade_contract_size` 1 that is -$1.474 and -$5.927 per lot per
# night.
#
# The `intraday` label opens after the cash open and closes at the cash close, so it never crosses
# a server midnight and never pays swap. The `overnight` label crosses exactly one, so a long
# position pays one night and a short pays nothing. Expressed against the index level, that is a
# larger number than the spread - which is the single most important cost fact about this bot,
# and it is asymmetric: the same strategy costs nothing to hold short.
#
# `ml4t.backtest` has no holding-cost model, so nothing below phase 5 charges this. It is booked
# outside the engine with `bots/_shared/costs_mt5.holding_cost_points`, and the backtest stage
# refuses an `overnight` spec that has no such ledger.

# %%
levels = {
    symbol: float(bars.filter(pl.col("symbol") == symbol)["close"].tail(1)[0]) for symbol in UNIVERSE
}
for symbol in UNIVERSE:
    points = float(SWAP["points_per_lot_per_night"][symbol]["long"])
    per_night_usd = points * 0.01 * float(SWAP["contract_size"])
    per_night_bps = per_night_usd / levels[symbol] * 1e4
    print(
        f"{symbol}: level {levels[symbol]:,.0f} | swap_long {points:+.1f} points = "
        f"{per_night_usd:+.3f} USD/lot/night = {per_night_bps:+.2f} bp/night | "
        f"Friday x3 {3 * per_night_bps:+.2f} bp | Mon-Fri (4 nights + a triple) "
        f"{7 * per_night_bps:+.2f} bp | "
        f"round-trip spread {round_trip['overnight'][symbol]:.2f} bp | short pays "
        f"{float(SWAP['points_per_lot_per_night'][symbol]['short']):.1f}"
    )
print(
    "\nA long overnight position pays roughly one and a half round trips in financing for every "
    "night it is held. The overnight label has to beat the spread AND the swap; the intraday "
    "label only the spread."
)

# %% [markdown]
# ### B.7 How much of one session's return carries into the next
#
# Autocorrelation computed inside each index and then averaged, not over the two stacked into one
# series - stacking measures the join between them. A signal that is worth building needs some
# dependence somewhere in this picture; a flat curve at zero says the label is close to a coin
# flip at every lag and the edge, if there is one, has to come from the features rather than from
# the label's own persistence.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for spec, colour in zip(SPECS, SPEC_COLORS, strict=False):
    frame = moves[spec].select("symbol", "timestamp", pl.col("move_bps").alias("ret"))
    acf = panel_acf(frame, entity_col="symbol", value_col="ret", max_lags=21)
    ax.plot(acf["lag"][1:], acf["acf"][1:], color=colour, marker="o", ms=3, label=spec)
    ax.fill_between(
        acf["lag"][1:],
        (acf["acf"] - 2 * acf["pooled_se"])[1:],
        (acf["acf"] + 2 * acf["pooled_se"])[1:],
        color=colour,
        alpha=0.15,
        lw=0,
    )
    lag1 = acf.filter(pl.col("lag") == 1).to_dicts()[0]
    print(
        f"{spec:9s} lag-1 autocorrelation {lag1['acf']:+.4f} "
        f"(pooled se {lag1['pooled_se']:.4f}, {int(lag1['n_entities'])} indices, "
        f"{lag1['obs_per_entity']:.0f} sessions each)"
    )
ax.axhline(0, color=COLORS["neutral"], lw=1)
ax.set_xlabel("Lag (sessions)")
ax.set_ylabel("Within-index autocorrelation")
ax.legend(frameon=False)
add_message_title(
    ax,
    "Session returns carry little of themselves into the next session",
    subtitle="Mean of the per-index autocorrelation, development period only",
)
show_with_alt(
    fig,
    "Two lines, one per label spec, showing the mean within-index autocorrelation of the session "
    "return at lags one to twenty-one. Both hover close to zero.",
)

# %% [markdown]
# ## C. Design decisions
#
# ### C.1 What would send this design back
#
# - A month whose daily break disagrees with New York daylight saving (B.2 asserts it, so this
#   arrives as an error rather than as a number).
# - The share of sessions with no fill rising in the recent sample rather than falling: the
#   `overnight` spec is only tradable where a bar opens at the cash close, and B.3 says how often.
# - A round-trip cost that eats the median move (B.5).
#
# ### C.2 What the strategy does with the forecast
#
# Two indices cannot be ranked, so the forecast is compared with a threshold on the instrument's
# own scale - `fixed_threshold` on the sign and size of the regression score, or
# `per_symbol_rolling_percentile` against the instrument's own trailing distribution
# (`case_studies/utils/signals.py`). Both, one or neither index can be held on a session. Every
# value in `backtest.sweep.threshold_grid` is one trial in the Deflated Sharpe Ratio, and so is
# every session filter and every label.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 How much an evaluation has to spend

# %%
timeline = (
    panels[PRIMARY_LABEL.replace("fwd_ret_", "")]
    .select(pl.col("timestamp"))
    .unique()
    .sort("timestamp")
)
development = timeline.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date())
grid = session_grid(
    SESSION_CALENDAR,
    pd.Timestamp(START_DATE),
    pd.Timestamp(HOLDOUT_END),
    open_delay_minutes=OPEN_DELAY,
)
calendar_sessions = grid.filter(
    (pl.col("session") >= pl.lit(START_DATE).str.to_date())
    & (pl.col("session") < pl.lit(HOLDOUT_START).str.to_date())
).height
print(
    f"{SESSION_CALENDAR} cash sessions in the development period {calendar_sessions:,} | "
    f"observed on the panel {development.height:,} | index-sessions "
    f"{panels[PRIMARY_LABEL.replace('fwd_ret_', '')].filter(pl.col('timestamp') < pl.lit(HOLDOUT_START).str.to_date()).height:,} "
    f"of {calendar_sessions * len(UNIVERSE):,}"
)

# %% [markdown]
# ### D.2 The folds
#
# A model is fitted on one stretch of history and evaluated on the stretch that follows it, then
# the pair moves forward and the process repeats. Each fit-then-evaluate pair is a **fold**.
#
# Both labels resolve one session ahead, so a training row dated on the last day of its block is
# labelled with a price from after the block ends. The fix is a gap at least as wide as the
# horizon - **purging** - and its width comes from `labels.buffer`.
#
# Three folds of eighteen months' training and six months' validation is what 3.9 years of H1
# history supports. `evaluation.holdout_start` is applied by the splitter itself, and the
# splitter is handed trading days and no prices, so nothing the holdout contains reaches a number
# computed above. The three assertions establish what the figure cannot: that the number of folds
# is what `setup.yaml` declares, that the purge gap is the label horizon, and that no validation
# window reaches into the holdout.

# %%
splits = generate_cv_splits(
    timeline,
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    date_col="timestamp",
)
sessions = timeline["timestamp"].to_numpy()
purge_gaps = {
    int(((sessions > np.datetime64(s["train_end"])) & (sessions < np.datetime64(s["val_start"]))).sum())
    for s in splits
}
last_val = max(split["val_end"] for split in splits)
first_train = min(split["train_start"] for split in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert last_val < np.datetime64(HOLDOUT_START), "a fold reaches into the holdout"
assert purge_gaps == {1}, f"purge gaps {purge_gaps} are not the one-session label horizon"
print(
    f"{len(splits)} folds | purge gap {min(purge_gaps)} session at every boundary, from "
    f"labels.buffer {LABEL_BUFFER} | earliest training starts "
    f"{pd.Timestamp(first_train).date()}, last validation ends {pd.Timestamp(last_val).date()}, "
    f"the holdout opens {HOLDOUT_START}"
)
warmup = int((sessions < np.datetime64(first_train)).sum())
# CORRECTED on the second pass. The first version took the largest single number in
# `features.windows`, which is not the warmup any column actually costs: `zscore_21d`
# standardizes a 21-session return against a 63-session window and so needs 84 rows, and
# `spread_corr_1d` correlates over 63 sessions of a 5-session return and needs 68. The
# composition is already written down, once, in `_features.warmup_expectations` - the same
# dictionary the warmup audit and the point-in-time test hold every column to - so the
# assertion reads that rather than re-deriving a weaker version of it. The two specs differ by
# one row, so both are checked.
session_warmup = {
    spec: max(warmup_expectations(SETUP["features"]["windows"], spec).values()) for spec in SPECS
}
longest_window = max(session_warmup.values())
print(
    f"Warmup available before the earliest training block: {warmup} sessions; the longest "
    f"COMPOSED session-grid warmup is {longest_window} "
    f"({', '.join(f'{s} {w}' for s, w in session_warmup.items())})"
)
assert warmup >= longest_window, (
    f"only {warmup} sessions of warmup before the earliest fold, but a feature column needs "
    f"{longest_window} sessions of it: the oldest fold would train on a sparse matrix"
)

# The D1-sourced families are warmed on a DIFFERENT grid and must not be checked against the
# session warmup - registering a 252-bar daily momentum would otherwise fail an assertion it
# has nothing to do with. `features.daily_windows` is counted in bars of data/mt5/daily.parquet,
# which starts 2019-07-16, and the check is that enough of THOSE have printed before the
# earliest training session.
daily_windows = SETUP["features"]["daily_windows"]
longest_daily = max(max(v) if isinstance(v, list) else int(v) for v in daily_windows.values())
daily_bars = load_daily_bars(SETUP, end_date=str(pd.Timestamp(first_train).date()))
daily_warmup = int(
    daily_bars.filter(pl.col("timestamp").dt.weekday() <= 5)
    .group_by("symbol")
    .len()["len"]
    .min()
)
print(
    f"Daily bars available before the earliest training block: {daily_warmup} (Monday to "
    f"Friday, the thinner of the two indices); the longest declared daily window is "
    f"{longest_daily} bars"
)
assert daily_warmup >= longest_daily, (
    f"only {daily_warmup} daily bars before the earliest fold against a declared daily window "
    f"of {longest_daily}: the d1_ families would start the oldest fold empty"
)
for split in sorted(splits, key=lambda s: s["fold"]):
    print(
        f"  fold {split['fold']}: train {pd.Timestamp(split['train_start']).date()} .. "
        f"{pd.Timestamp(split['train_end']).date()} | val "
        f"{pd.Timestamp(split['val_start']).date()} .. {pd.Timestamp(split['val_end']).date()}"
    )

fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
ax.set_xlabel("Cash session")
add_message_title(
    ax,
    "Three folds roll forward and stop short of the holdout",
    subtitle="Boundaries as generate_cv_splits returned them; the one-session purge is too narrow to see",
)
show_with_alt(
    fig,
    "One horizontal bar per walk-forward fold, each an eighteen-month training stretch running "
    "into a six-month validation stretch. The bars step up and to the right from fold two at the "
    "bottom to fold zero at the top. A shaded region on the right marks the six-month holdout, "
    "and no bar reaches into it.",
)

# %% [markdown]
# ## E. What this notebook hands on
#
# Nothing. The universe is fixed and declared in `setup.yaml::universe.symbols`; the panels are
# rebuilt from `_features.py` by every stage that needs them, from the same function this
# notebook called, so there is no intermediate file for the two to disagree on.

# %% [markdown]
# ## F. What the evidence says about each setting
#
# | Setting | Evidence | Choose differently when |
# |---|---|---|
# | `decision.trade_hours_follow_dst_of` | B.2, the break hour per month against New York DST | a month's break stops following New York, which B.2 raises rather than reports |
# | `decision.open_delay_minutes` | B.1 decision instants, B.5 move sizes | the first thirty minutes stop being the widest part of the session on this account |
# | `decision.execution_delay` | B.3, the share of sessions with a bar at the decision instant | the broker extends the Friday session past the cash close on both indices |
# | `costs.spread_bps` and `costs.spread_bps_by_year` | B.4, the `spread` field of the development execution bars | a re-measurement on the real account lands outside the declared range, or a new regime break appears |
# | `decision.min_fill_coverage` | B.3b, per fold block and spec | a fold block falls below the declared floor |
# | `costs.swap` | B.6, `symbol_info` on 2026-09-08 | the real account's swap differs from the demo's, which is expected |
# | `evaluation.n_splits`, `train_size`, `val_size` | D.1 sessions available, D.2 fold boundaries and the warmup assertion | the folds no longer fit, or a feature window outgrows the warmup |

# %%
print(
    f"universe.symbols {UNIVERSE} | decision.cadence {DECISION['cadence']} | "
    f"labels {LABEL_NAMES}\n"
    f"tradable index-sessions: "
    + ", ".join(
        f"{spec} {panels[spec].drop_nulls(['exec_open', 'label_close']).height:,}/{panels[spec].height:,}"
        for spec in SPECS
    )
    + f"\nevaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, last "
    f"validation ends {pd.Timestamp(last_val).date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# Results are recorded in `bots/exness_usidx_sess/BOT.md` (phase 1) after each run.

# %% [markdown]
# ## Key takeaways
#
# 1. **Derive a session's decision instants from the exchange calendar, not from a table.** New
#    York daylight saving and the 13:00 half-days move both instants, and a hard-coded 15:00 is
#    wrong for seven months of the year and for eight days more.
# 2. **The broker's clock and the instrument's trading hours are two facts, and on this account
#    they disagree.** The clock is UTC+0 all year; the daily break follows New York. Only one of
#    them is what `server_clock.follows_dst_of` describes.
# 3. **Check that the fill bar exists before assuming the fill.** A CFD that shuts at the Friday
#    cash close leaves a quarter of the overnight decisions with no next bar, and filling them
#    two days later at the Sunday open would be a different strategy.
# 4. **Price the cost the engine cannot express.** Overnight financing on a long index position
#    is larger than the spread on this account and zero on the short side; a backtest that only
#    charges the spread is measuring a strategy nobody can trade.
# 5. **A short history is a fold budget, not a reason to shrink the holdout to nothing.** 3.9
#    years buys three folds of eighteen months plus six months of holdout, and the count of
#    validation windows is the honest limit on how many model configurations can be compared.
#
# ### Known limitations
#
# - The spread was measured over thirty days on a **demo** account. The real Pro account is not
#   read yet, and the cost stage is blocked until it is.
# - There is no VIX instrument on this account and no point-in-time FOMC calendar in this
#   repository, so two feature families `bots/assets/US500.md` names cannot be built.
# - H1 history begins 2022-10-25 on this server. Windows longer than about sixty sessions cannot
#   be warmed up on this grid and have to come from the D1 parquet, which reaches back to
#   2019-07-16.
#
# **Next**: the two labels, sealed on their own endpoints, built on this development period.
