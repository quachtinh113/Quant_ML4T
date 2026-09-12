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
# # Exness Gold Sessions (exness_gold_sess): Feature Engineering
#
# Two metals, two decisions a weekday, taken one hour after the London open and one hour after the
# New York open. Every column below answers one question: at the moment the position is decided,
# which bars are already on the tape, and what does the feature make of them?
#
# The answer is given on **two grids**, and that is the design this notebook exists to check
# (`bot-portfolio-exness.md:72-74`). A session bot's feature windows stop at the session boundary,
# so anything longer than a session is read from **daily** bars and joined backward on the D1
# bar's own **close instant** - never on its calendar date, which would hand a 09:00 UTC decision a
# bar that closes fifteen hours after it. The payoff is that a 315-bar chain costs 315 *D1* bars,
# about fifteen months, available on this account from 2014, so no long feature pushes
# `universe.history_start` forward the way it did for `exness_fx_d1` (1,885 warmup rows lost).
#
# ## Learning objectives
#
# - State how far back a feature reads, and on which grid, before writing its code
# - Join a slower grid onto a faster decision grid without reading a bar that had not closed
# - Show that withholding later bars leaves every earlier value unchanged, and that rebuilding
#   the matrix from only the bars that had closed at a decision instant reproduces the batch row
# - Keep a null that the tape earned, instead of cutting the row it sits in - and say which
#   columns may hold one and why
# - Read a feature set for scale, redundancy and decay before any model sees it
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Sections 8.2-8.4 (price and volume features, structural cross-instrument features).
# Route B fork of [`exness_fx_d1/03_financial_features`](../exness_fx_d1/03_financial_features.ipynb)
# for the bot `exness_gold_sess` (`bots/exness_gold_sess/BOT.md`), with three differences: the
# decision grid is a **session** grid on H1 bars rather than a daily one; nothing is ranked
# cross-sectionally, because two names are not a cross-section (`setup.yaml::features.ranked` is
# empty by design); and the long-window families live on a second grid. It reads MT5 H1 and D1
# bars and its settings from `config/setup.yaml`, and reads nothing another notebook wrote. The
# construction itself lives in `_features.py` beside this file, because the point-in-time test
# (`bots/exness_gold_sess/tests/test_lookahead.py`), the label stage and, later, step 2 of the
# deployment loop all have to compute exactly what this notebook computes. It writes
# `features/financial.parquet` with a `.digest.json` sidecar.

# %%
"""exness_gold_sess: Feature Engineering (Route B fork of exness_fx_d1)."""

import warnings
from datetime import date, datetime, timedelta
from fnmatch import fnmatch

import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_gold_sess._features import (
    build_features,
    d1_state,
    feature_columns,
    features_as_of,
    gold_silver_ratio,
    session_panel,
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

CASE_STUDY_ID = "exness_gold_sess"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
FEATURES_DIR = CASE_DIR / "features"

# %% [markdown]
# Two settings are left open. `START_DATE` of `None` starts at `setup.yaml::universe.history_start`
# - the first week both metals print a dense hourly grid, measured in task B0, not guessed - which
# is what the production run does. `N_RECOMPUTE_INSTANTS` is how many decision instants Section D.4
# rebuilds the whole matrix at from only the bars that had closed by then. There is no cap on
# symbols: the universe is two names and the gold-silver ratio needs both.

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
DEVELOPMENT_END = HOLDOUT_START - timedelta(days=1)
HOLDOUT_TS = datetime.combine(HOLDOUT_START, datetime.min.time())
SLOTS_PER_DAY = len(setup["decision"]["snapshots"])
CYCLE = int(setup["labels"]["rebalance_step"][setup["labels"]["primary"]])

print(f"{len(FAMILIES)} feature families are declared, and Section A prints what each one claims")
print(f"Two grids: {setup['features']['grid']['intraday']} inside the session, "
      f"{setup['features']['grid']['long_window']} for anything longer, joined "
      f"{setup['features']['grid']['long_window_join']}")
print(f"H1 bars from {HISTORY_START}; D1 bars from {D1_HISTORY_START}, which is the head start "
      "that keeps a 315-bar chain from moving the panel's first row")
print(f"The position is re-decided every {CYCLE} slot ({SLOTS_PER_DAY} slots a weekday)")
print(f"Rows begin where {CARRIER} can first hold a value; {MAY_BE_NULL} may stay null after that")
print(f"Dates from {HOLDOUT_START} are the holdout; every figure below stops before it")

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The bot is not looking for a cross-sectional edge - two names do not make a cross-section, and a
# percentile over two values is a coin flip. Its hypothesis is about **intraday structure**: flow
# concentrates at the London and the New York open, and a forecast of the move from the opening
# hour to the session close may carry information a daily forecast smears out. Its second leg is
# the **gold-silver ratio**, which mean-reverts when one metal moves first.
#
# That splits the matrix in two, and the register's `role` column is where the split is written
# down. A **signal** column is one a position may be formed on: the opening hour's move, its range
# and where the close sits inside it; the hours before the open and the overnight gap; trailing
# returns on the decision grid; and the ratio family. A **state** column describes the environment
# the signal is read in and is never traded on its own: the metal's own position in its year, and
# the session/calendar flags.
#
# Two entries in the register are `PLANNED` and carry no column. `real yields and the dollar` is
# the driver `bots/assets/XAUUSD.md:128` names first, and `data/macro/config.yaml` does not hold
# the series while `FRED_API_KEY` is absent from this environment. `CPI and FOMC release windows`
# needs an economic-calendar file: unlike payrolls - first Friday, 08:30 New York, a rule this
# notebook computes - neither date is derivable, and a hand-typed table of ninety dates would be a
# data source with no provenance. Both are declared rather than silently omitted, and each is
# **+1 trial** when it is first tried.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "driver hypothesis", "inputs", "lookback (bars)", "lag (bars)", "frame"]
)

# %% [markdown]
# ## B. Inputs and their observability
#
# ### B.1 The decision grid
#
# The metals quote continuously from Sunday evening to Friday evening, so a "session" is a
# convention and the convention has to be the one the decision is taken on. `config/setup.yaml`
# fixes it as a **rule**, not a clock time: the decision is the close of the first H1 bar closing
# at or after the venue's open plus the 30-minute edge block, which on a UTC-aligned hourly grid is
# always the open plus sixty minutes. The venue windows come from `bots/_shared/sessions.py`
# through `ZoneInfo`, so daylight saving moves the UTC hour without a table - which matters,
# because the *server* is UTC+0 all year and follows nobody's DST.
#
# `_features.session_panel` is the one function that does it, and it resolves the **decision** and
# the **endpoint** separately. A decision needs only bars that had closed at the snapshot; an
# endpoint needs bars eight hours later. Sessions the broker closed early - the US public holidays
# - keep their decision row and carry a null endpoint, so `02_labels` writes no eight-hour label
# there and the panel does not select against thin sessions on information the decision could not
# have had.

# %%
# The hourly tape is loaded once and used twice: the panel is aggregated from it, and the
# pre-session probes read it directly. Handing those probes the panel instead would silently give
# them one price every five hours, which is why the two are named apart here.
bars_h1 = load_mt5_bars(
    "1h", symbols=UNIVERSE, start_date=HISTORY_START, end_date=str(HOLDOUT_END)
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
panel = session_panel(
    bars_h1,
    edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
    tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
    max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
    keep_context=True,
    verbose=True,
)
assert panel.select(["symbol", "timestamp"]).is_duplicated().sum() == 0, "duplicate panel key"
# The Garman-Klass variance proxy is non-negative only for a bar whose high and low bracket its
# open and close. A malformed bar would make the window mean negative and its square root arrive
# as a silent null, so the input is checked here rather than the output guarded downstream.
assert panel.select(
    (
        (pl.col("high") >= pl.max_horizontal("open", "close"))
        & (pl.col("low") <= pl.min_horizontal("open", "close"))
    ).all()
).item(), "an OHLC bar does not bracket its own open and close"
print(f"{panel.height:,} rows over {panel['symbol'].n_unique()} metals and "
      f"{panel['timestamp'].n_unique():,} decision instants")
print(f"{panel['timestamp'].min()} to {panel['timestamp'].max()}")
print(f"{panel['label_end_ts'].null_count()} rows have no session endpoint (an early close); "
      "they carry features and no eight-hour label")
display(
    panel.group_by(["session", pl.col("timestamp").dt.hour().alias("utc_hour")])
    .len()
    .sort(["session", "utc_hour"])
)

# %% [markdown]
# ### B.2 The daily grid, and the leg that is priced but not traded
#
# D1 bars are read from the first one the account serves, three years before the H1 panel opens.
# That head start is the whole reason a 252-day z-score costs nothing at the front of this sample.
#
# `features.reference_symbols` is loaded alongside the universe. A ratio needs both legs
# observable; only one of them has to be tradable. Phase 6 may take XAGUSD out of the universe if
# the silver leg cannot pay its own p90 spread (`bots/assets/XAGUSD.md:144`), and without this the
# family both asset profiles name **first** would leave with it.

# %%
d1 = load_mt5_bars("daily", symbols=UNIVERSE, start_date=D1_HISTORY_START, end_date=str(HOLDOUT_END))
d1_reference = (
    load_mt5_bars("daily", symbols=REFERENCE, start_date=D1_HISTORY_START, end_date=str(HOLDOUT_END))
    if REFERENCE
    else None
)
print(f"D1 {d1.height:,} bars, {d1['timestamp'].min()} to {d1['timestamp'].max()}, "
      f"{d1['symbol'].n_unique()} traded symbols"
      + (f" + {len(REFERENCE)} reference {REFERENCE}" if REFERENCE else " (no reference symbol)"))
_lead = (panel["timestamp"].min().date() - d1["timestamp"].min()).days
print(f"the D1 grid starts {_lead:,} calendar days before the first decision, which is what pays "
      f"for the {max(warmup_expectations(WINDOWS)['D1 bars'].values())}-bar longest chain")

# %% [markdown]
# The two metals are not two independent markets. They share a driver - US real yields and the
# dollar - and correlate around 0.8 (`bots/assets/XAGUSD.md:140`), which is exactly why the ratio
# is a feature and why nothing here is ranked. The table below is the size of the session each one
# travels, in basis points, which is the quantity the cost gate in phase 6 will read against the
# measured spread: silver moves about twice as far and pays about eight times as much.

# %%
_universe = (
    panel.with_columns(
        (1e4 * (pl.col("dec_high") - pl.col("dec_low")) / pl.col("close")).alias("_open_hour_bp"),
        (1e4 * (pl.col("label_end_close") / pl.col("close") - 1).abs()).alias("_session_bp"),
    )
    .group_by(["symbol", "session"])
    .agg(
        pl.len().alias("slots"),
        pl.col("_open_hour_bp").median().round(1).alias("median opening-hour range (bp)"),
        pl.col("_session_bp").median().round(1).alias("median |session move| (bp)"),
    )
    .sort(["symbol", "session"])
)
with pl.Config(tbl_rows=_universe.height):
    display(_universe)
print("round trip at the measured p90 spread: "
      f"{setup['costs']['spread_bps_by_session']['round_trip_p90_bps']} bps")

# %% [markdown]
# ## C. Feature construction, one subsection per family
#
# The code for every family is in `_features.py` and imported above rather than written in a cell,
# for the reason Chapter 25 (Section 25.1) gives: a feature that exists in two implementations
# agrees on the day the second is written and drifts on the first edit, and nothing in either
# output says so. Each subsection says what its function computes and why; the function is the
# authority on how.
#
# ### C.1 The session's opening move (H1, inside the session)
#
# Exactly one bar of the session has closed at the decision, by the open+1h rule, and
# `intraday_features` asserts that rather than assuming it. Three columns come out of that one
# bar: its return from the session's opening price, its high-low range as a share of the open, and
# where the close sits inside that range. The first is the direction, the second is how much was
# on offer, the third separates a bar that closed on its high from one that gave it back.
#
# ### C.2 The hours before the open, with a staleness ceiling (H1, crossing the boundary)
#
# `pre_ret_2h` .. `pre_ret_8h` read what the metal did while this venue was shut. The family
# **deliberately crosses the session boundary** - that is its hypothesis - which is why it is
# declared separately rather than smuggled into C.1, and why "blocked at the session edge" stays a
# statement about C.1 rather than a slogan about the matrix.
#
# Every probe carries a ceiling on the age of the bar it matched. A backward asof join always
# returns something: through the nightly 21:00-22:00 UTC break, over a weekend or over a holiday
# it reaches back until it finds a print, and `pre_ret_8h` would quietly become a ten-hour or a
# sixty-hour return under the same name. A match older than one bar is **null**, and Section E
# reports how many.
#
# ### C.3 Overnight against intraday (H1 + the decision grid)
#
# `overnight_gap` is this session's opening price against **the same venue's previous session
# close**, and `prev_sess_ret` is that previous session's own open-to-close move. Same venue, not
# previous row: the panel sorts London before New York on a day, so the row before a New York
# decision is that morning's London row - whose endpoint is 17:00 UTC, three hours **after** the
# New York decision. Reaching for the previous row's endpoint would read the future on every New
# York row, and `_features.overnight_decomposition` asserts it does not.
#
# ### C.4 Momentum on the decision grid
#
# `slot_ret_1`, `slot_ret_2`, `slot_ret_5`: returns over one, two and five previous decisions. One
# slot is one session, so none of these is a sub-session window. The declared failure mode is worth
# repeating: the two venues alternate, so `slot_ret_1` compares a London close with a New York one.
#
# ### C.5 Long-window state (D1, asof-joined on the bar close)
#
# Trailing D1 momentum, Garman-Klass and close-to-close volatility, drawdown, the z-score of a
# multi-horizon return, RSI and price against moving averages. The join is the point-in-time
# content of the family: a D1 bar of server day D closes at 00:00 UTC on day D+1, so a decision at
# 09:00 UTC on day D may read the bar of day D-1 and no later one. `_features.join_d1_asof` carries
# the bar's close instant onto the panel as `d1_close_ts` and asserts it never exceeds the
# decision, and `bots/exness_gold_sess/tests/test_lookahead.py` reads the ages off it: eight or
# nine hours at a London decision, thirteen or fourteen at a New York one.
#
# ### C.6 The gold-silver ratio (D1, cross-asset)
#
# The ratio, its trailing z-score, and each metal's rolling beta to the other. It is a *cross-asset*
# family with two members, which is why it is a ratio and a beta and not a rank.
#
# ### C.7 Session and calendar state
#
# Which venue, whether the venue is on summer time, the weekday, and whether the payroll release
# falls inside the decision bar or inside the hold. All five are knowable weeks in advance - a
# venue's DST calendar and the first Friday of a month are not market data - so the family's lag is
# zero in a stronger sense than the price families'.
#
# Eight `sessions.py` flags are **not** shipped, and that is a measurement rather than an omission:
# over the development window `london` and `market_open` are true at every decision and
# `edge_open`, `edge_close`, `rollover`, `sydney`, `us_cash` and `asia` are false at every one.
# `edge_open` false *is* the 30-minute edge block holding. A constant column is a rank-deficient
# regressor, so the facts live in `tests/test_sessions.py` and the register says so.

# %%
built = build_features(panel, bars_h1, d1, WINDOWS, d1_reference=d1_reference)
feature_cols = feature_columns(built)
assert set(feature_cols) == set(assign_families(feature_cols, FAMILIES)), (
    "a column reached the matrix without a register row"
)
print(f"{built.height:,} rows carrying {len(feature_cols)} features, every one claimed by a "
      f"register family")

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
        "Register lookback per family. Read the unit per row: the intraday families count H1 "
        "bars, the momentum family counts decision slots, the long-window and ratio families "
        "count D1 bars"
    ),
    alt=(
        "Horizontal bars, one per family, each extending leftward from the decision line by that "
        "family's declared lookback - shortest for the opening move and the state flags, longest "
        "for the long-window state at 315 bars. Every bar reaches the line, so no family shows a "
        "publication lag gap."
    ),
)

# %% [markdown]
# ### D.2 Warmup, on each grid in its own unit
#
# A trailing window cannot produce a value until it has enough bars to fill, and the audit checks
# that length rather than describing it. The unit is where a two-grid design goes wrong: a
# `d1_ret_252d` column's warmup is 252 **D1** bars, and because the D1 frame starts three years
# before the panel, that column is already dense at panel row 1. Auditing it against 252 rows of
# the *panel* would report a correct construction as "populated from fewer bars than its window
# spans" and raise. So the register returns two dictionaries and each is audited on its own frame.

# %%
expected = warmup_expectations(WINDOWS)
panel_census = warmup_audit(built, expected["decision slots"], entity="symbol")
d1_features = gold_silver_ratio(d1_state(d1, WINDOWS), WINDOWS, reference=d1_reference)
d1_census = warmup_audit(d1_features, expected["D1 bars"], entity="symbol")
print(f"{panel_census.height} columns audited in decision slots, {d1_census.height} in D1 bars")
display(pl.concat([panel_census.with_columns(pl.lit("decision slots").alias("unit")),
                   d1_census.with_columns(pl.lit("D1 bars").alias("unit"))]).sort("unit", "column"))

# %% [markdown]
# ### D.3 Withholding the holdout changes nothing
#
# Trailing and within-instant statistics share a property worth checking directly: recomputed on a
# panel that stops before the holdout, they reproduce the same values on the rows the two panels
# share. A parameter fitted over a whole column - a winsorization bound, a scaler - does not,
# because truncating the column moves the parameter and with it every row it was applied to.
# Building twice and comparing tests the whole construction at once, every emitted column rather
# than a sample, and does not depend on anyone having flagged the transform that fits.
#
# The comparison raises if any column moves, so reaching the next cell is the result.

# %%
_withheld = build_features(
    panel.filter(pl.col("timestamp") < HOLDOUT_TS),
    bars_h1.filter(pl.col("timestamp") < HOLDOUT_TS),
    d1.filter(pl.col("timestamp") < HOLDOUT_START),
    WINDOWS,
    d1_reference=(
        None if d1_reference is None else d1_reference.filter(pl.col("timestamp") < HOLDOUT_START)
    ),
)
# The truncated build's LAST slot per metal is the one the truncation itself created: its
# `overnight_gap` and `prev_sess_ret` read the previous session of the same venue, which exists,
# but the full build's row at that slot was formed with a later bar available for the block
# aggregate. Comparing it would be comparing two different questions, so the comparison runs on
# the slots both builds hold in full - and the count of what that excludes is printed, because
# "we dropped the awkward rows" is exactly the sentence a reader has to be able to audit.
_shared = (
    _withheld.filter(pl.col("timestamp") < pl.col("timestamp").max().over("symbol"))
    .select("timestamp", "symbol")
)
print(f"{_shared.height:,} of {_withheld.height:,} withheld rows compared; the "
      f"{_withheld.height - _shared.height} excluded are each metal's final slot, which only the "
      "truncation created")
seal = assert_values_agree(
    built.join(_shared, on=["timestamp", "symbol"], how="semi"),
    _withheld.join(_shared, on=["timestamp", "symbol"], how="semi"),
    columns=feature_cols,
    keys=["timestamp", "symbol"],
)
display(seal.filter(pl.col("column").is_in(["zscore_63d", "gsr_z_252d", "slot_ret_5", "pre_ret_8h"])))

# %% [markdown]
# ### D.4 Rebuilding the matrix at the decision instant
#
# D.3 withholds whole dates. The stronger form withholds **bars**: at a decision instant only the
# bars that had closed by then were on the tape, and a feature computed from those alone must equal
# the batch value. That is a claim about the two-grid join as much as about the arithmetic - a D1
# join that reached to the calendar date would pass D.3 and fail here.
#
# `_features.features_as_of` truncates the H1 and D1 bars at the instant, rebuilds the panel and
# the whole matrix, and the row is compared column by column. Note what it returns: **the decision
# row, with a null endpoint**, because the session close is eight hours in the future. Before the
# decision and the endpoint were resolved separately this function could only return an empty
# frame, and an empty frame compares equal to an empty frame. The count is asserted.

# %%
_dense = (
    built.filter(pl.col(CARRIER).is_not_null() & (pl.col("timestamp") < HOLDOUT_TS))["timestamp"]
    .unique()
    .sort()
)
_step = max(1, len(_dense) // (N_RECOMPUTE_INSTANTS + 1))
recompute_rows = []
for instant in _dense.gather_every(_step, offset=_step)[:N_RECOMPUTE_INSTANTS].to_list():
    as_of = features_as_of(instant, bars_h1, d1, WINDOWS, d1_reference=d1_reference)
    assert as_of.height == len(UNIVERSE), f"features_as_of returned {as_of.height} rows at {instant}"
    assert as_of["label_end_ts"].null_count() == as_of.height, "the endpoint is not in the future"
    gap = assert_values_agree(
        built.filter(pl.col("timestamp") == instant).sort("symbol"),
        as_of.sort("symbol"),
        columns=feature_cols,
        keys=["timestamp", "symbol"],
    )
    recompute_rows.append(
        {
            "decision (UTC)": instant,
            "session": as_of["session"][0],
            "H1 bars seen": bars_h1.filter(
                pl.col("timestamp") + pl.duration(minutes=60) <= instant
            ).height,
            "rows": as_of.height,
            "max abs difference": gap["max abs difference"].max(),
        }
    )
pl.DataFrame(recompute_rows)

# %% [markdown]
# ## E. Matrix assembly and the null policy
#
# The panel key is `symbol` + `timestamp`. Raw OHLCV, the session context and `d1_close_ts` are
# excluded: they are the inputs the features are made of, or the evidence that the join was
# backward, and neither is a feature.
#
# **The null policy is a cut plus a declared set of columns that may stay null**, and the second
# half is what this bot needs and `exness_fx_d1` did not. Rows before `slot_ret_5` can fill are
# dropped - pure warmup, five slots. After that, two families can still hold a null the tape
# earned: `pre_ret_*h` probes an hour the market was shut, and `overnight_gap` / `prev_sess_ret`
# read a previous session that was closed early. Cutting those rows instead would drop about three
# per cent of the panel, concentrated on holidays and thin Mondays - reintroducing exactly the
# availability-driven selection the decision/endpoint split was written to remove. A null is
# visible to the coverage screen in `05_evaluation`; a dropped holiday row is invisible everywhere.
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
null_counts = {
    c: development[c].null_count()
    for c in feature_cols
    if development[c].null_count()
}
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
print(f"{len(feature_cols)} features, {features.height:,} rows, {features['symbol'].n_unique()} metals")
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
# boundary marks the first slot the emitted matrix keeps. The long-window families are already at
# one on the first slot, and that is the two-grid design paying for itself: their warmup was spent
# on D1 bars printed three years before this panel opens.

# %%
plot_coverage_through_time(
    family_coverage(
        built.filter(pl.col("timestamp") < HOLDOUT_TS).select(["timestamp", *feature_cols]),
        assignment,
        every="1mo",
    ),
    warmup_boundary=features["timestamp"].min(),
    title="The long-window families start full: their warmup was spent on D1 bars",
    subtitle="Monthly non-null share per family before the policy, with the boundary drawn",
    alt=(
        "Non-null share by feature family on an axis from zero to one. The long-window state and "
        "gold-silver ratio families sit at one from the first month, because their windows are "
        "counted in daily bars that begin three years earlier. The decision-grid momentum family "
        "reaches one within the first week. The pre-session family sits slightly below one "
        "throughout, which is the staleness ceiling nulling probes that fall in a shut market."
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
    ["sess_ret_1h", "sess_close_pos_1h", "overnight_gap", "zscore_63d", "gsr_z_252d", "d1_rsi_14"],
    title="The session columns are centred and thin; the state columns are wide",
    subtitle="One panel per representative column, development window, display tails clipped",
    alt=(
        "Six histograms in two rows. The opening-hour return and the overnight gap are narrow "
        "single-peaked bodies centred on zero; the close-position column is bounded by zero and "
        "one with mass banked against both ends; the return z-score and the ratio z-score are "
        "broad and roughly symmetric; the oscillator is a wide body between about twenty and "
        "eighty."
    ),
)

# %% [markdown]
# ### F.2 Redundancy structure
#
# Two columns are redundant when they carry the same ordering, however different their formulas
# look, so the distance clustered on is $1 - |\rho_s|$ and the absolute value treats a feature and
# its negation as the same thing. The tree is cut at the rank correlation the configuration
# declares. Read it as a screen for duplication and nothing stronger: which member of a cluster to
# keep needs a criterion measured out of sample, and `05_evaluation` measures one fold by fold.

# %%
clusters = plot_redundancy_clusters(
    development,
    feature_cols,
    cut=REDUNDANCY_CUT,
    title="The daily horizons cluster together; the session columns stand apart",
    subtitle=(
        r"Average linkage on $1 - |\rho_s|$, cut at a rank correlation of " f"{REDUNDANCY_CUT}"
    ),
    alt=(
        "Dendrogram of every feature in the matrix. The daily momentum, moving-average and "
        "z-score columns at neighbouring horizons join into one cluster, and the two volatility "
        "estimators join into another. The intraday columns - the opening hour, the pre-session "
        "probes and the decision-grid returns - sit apart from those and largely apart from each "
        "other, and the state flags form their own group."
    ),
)
print(f"{len(set(clusters.values()))} redundancy clusters at the drawn cut, for {len(feature_cols)} features")

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ### F.3 Persistence
#
# The book is re-decided every slot, twice a weekday, so the question is how long a value lasts
# against that cadence. The left panel is the feature's own autocorrelation out to
# `features.persistence_horizon` slots; it is estimated per metal on slots exactly one lag apart
# and summarized by the median over the two, because a correlation pooled over both would read
# high whenever the two sit at different levels. With two names the bootstrap interval is wide, and
# it is drawn that way rather than narrowed.

# %%
plot_persistence(
    development,
    ["sess_ret_1h", "overnight_gap", "slot_ret_5", "d1_vol_gk_63d", "zscore_63d", "gsr_z_252d"],
    entity="symbol",
    max_lag=PERSISTENCE_HORIZON,
    decision_dates=development["timestamp"].unique().sort().to_list(),
    title="The session columns are gone within a slot; the daily state survives a quarter",
    subtitle=f"Median over the two metals; re-decided every {CYCLE} slot",
    alt=(
        "Two panels. On the left, autocorrelation against lag in decision slots out to 126. The "
        "three-month Garman-Klass volatility and the ratio z-score decay slowest and are still "
        "well above zero at the right edge; the 63-day return z-score follows; the opening-hour "
        "return, the overnight gap and the five-slot return fall to about zero within a handful "
        "of slots. The bootstrap ribbons over two metals are wide. On the right, the ordering "
        "between consecutive slots is near one for the daily columns and low for the session "
        "ones."
    ),
)

# %% [markdown]
# ## G. Emit
#
# The parquet is written with a sidecar beside it recording a hash of the table's contents, its row
# count, its key columns, and the same hash taken over each price panel it was built from. The hash
# is over values rather than bytes, so re-writing the parquet or changing its compression leaves it
# alone while any change to a single feature value moves it.

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
inputs = {
    "session_panel": value_digest(panel.select(["symbol", "timestamp", "close", "session_open_px"])),
    "load_mt5_bars:1h": value_digest(bars_h1),
    "load_mt5_bars:daily": value_digest(d1),
}
if d1_reference is not None:
    inputs["load_mt5_bars:daily:reference"] = value_digest(d1_reference)
record = write_artifact(
    features,
    FEATURES_DIR / "financial.parquet",
    keys=["symbol", "timestamp"],
    written_by="case_studies/exness_gold_sess/03_financial_features.py",
    inputs=inputs,
)
print(f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')} under digest {record['digest']}")
print(f"input digests: {inputs}")

# %% [markdown]
# ## Key takeaways
#
# - **Declare how far back each feature reads, and on which grid, before writing it.** The
#   configuration holds one lookback and one lag per family, and the warmup audit, the timing
#   figure and the register table all read the declared numbers instead of re-deriving them, so a
#   window that disagrees with what was promised raises rather than passing quietly.
# - **On two grids, the unit is the trap.** A daily column's warmup is daily bars. Auditing it
#   against panel rows makes a correct construction look like a lookahead and an incorrect one look
#   fine, depending which way the two grids happen to line up.
# - **A slower grid joins on its bar's CLOSE, never on its date.** That single choice is fifteen
#   hours of lookahead on every London row, and it is invisible in the values.
# - **Keep the null the tape earned.** Cutting the rows where a probe found a shut market would
#   have removed the holidays and the thin Mondays - a selection on availability that correlates
#   with everything a cost model cares about.
# - **Check for a fitted transform by rebuilding, not by reading.** Recomputing with the later bars
#   removed and comparing value by value catches anything estimated across the sample, including
#   the transforms nobody thought to flag.
#
# ### Known limitations
#
# - **No real-yield or dollar family**, which `bots/assets/XAUUSD.md:128` names as the primary
#   driver of both metals. `data/macro/config.yaml` does not carry the series and `FRED_API_KEY` is
#   absent from this environment. Declared PLANNED in the register; +1 trial when it is tried.
# - **No CPI or FOMC flag.** Payrolls is a rule and is computed; the other two are announced
#   schedules and need a calendar file. Declared PLANNED rather than approximated.
# - **The per-bar `spread` field is not read.** 1,081 XAUUSD and 1,074 XAGUSD hourly bars up to
#   2023-02-03 quote zero (`bots/exness_gold_sess/data_census_2026-09-08.md` section 1d), so a
#   feature or a cost taken from that field would price most of 2017 as free. Costs come from the
#   30-day tick measurement instead.
# - **Volume on an MT5 feed is tick count at one broker**, not traded notional, so it enters only
#   through the range and never as a participation feature.
# - Every feature here is a rule written in advance. `04_model_based_features` adds the features
#   that are themselves model outputs, where the rule is estimated from the data.
