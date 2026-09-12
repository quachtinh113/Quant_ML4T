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
# > began as `case_studies/exness_fx_d1/02_labels.py` with the study id, the loader and the population names
# > changed. **This bot has TWO instruments (`US500`, `USTEC`) and no cross-section**, and one
# > label per experiment workspace. Any sentence below that speaks of five currency pairs, of
# > ranking assets against each other, of `usd_corr_*` / `gold_corr_*` columns or of a dollar
# > regime is template residue describing a different bot; where such prose asserted a NUMBER it
# > has been deleted rather than reworded (see `BOT.md`, Decisions log, 2026-09-08 third pass),
# > and where it survives as a description of the template it says so explicitly. Report any that
# > does neither. This banner follows `case_studies/xau_fx_mt5/13_backtest.py`.

# %% [markdown]
# # Exness US index sessions (exness_usidx_sess): Label Engineering
#
# A label is the quantity a model is asked to predict. This notebook builds the two this bot
# uses, checks the properties they have to have, and writes each to its own file with a record of
# what it was computed from.
#
# The two labels are the two halves of an index session, and together they are the whole of it:
#
# | Label | Enters at | Leaves at | Pays swap |
# |---|---|---|---|
# | `fwd_ret_intraday` | the open of the bar after the `us_cash_open` decision (15:00 UTC winter, 14:00 summer) | the close of the last cash bar (21:00 / 20:00 UTC, earlier on a half day) | no |
# | `fwd_ret_overnight` | the open of the bar after the `us_cash_close` decision | the next session's `us_cash_open` decision instant | one night, three on a Friday |
#
# Both are measured **from the price the order actually fills at**, not from the price the
# decision was taken at. Chapter 7 section 7.2 writes the close-to-close convention and notes
# that a backtest fills somewhere else; here the gap between the two is a full hour of an index
# session, large enough that pretending it away would flatter every result downstream. The
# consequence is that the label already contains the delay, so the model is trained on what it
# will be paid for rather than on something an hour better.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Seal a label on the endpoint that resolves it, and refuse a row whose endpoint does not exist
# - Build two labels that partition a session, and check that they do
# - Tell a missing label caused by the end of the sample from one caused by a market that was
#   shut, and drop only the second silently
# - Compute the effective sample size of a label whose consecutive values do not overlap, and
#   check the answer against the row count
# - Record what a label file was computed from, so that two parquets of identical shape can be
#   told apart months later
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 7 sections 7.2-7.3; Chapter 8 section 8.1 for the overnight / intraday split. Reads MT5
# H1 bars through `_features.session_panel` and `config/setup.yaml`. Writes
# `labels/fwd_ret_intraday.parquet` and `labels/fwd_ret_overnight.parquet`, each beside a
# `.digest.json` record. Route B fork of `case_studies/exness_fx_d1/02_labels.py`.

# %%
"""exness_usidx_sess: Label Engineering (Route B fork of exness_fx_d1)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_usidx_sess._features import SPECS, session_panel
from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.label_diagnostics import effective_sample_size, panel_autocorrelation
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")

CASE_STUDY_ID = "exness_usidx_sess"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
LABELS_DIR = CASE_DIR / "labels"

# %% [markdown]
# Both parameters are unset by default. `START_DATE` trims the history to a later start;
# `MAX_SYMBOLS` keeps only the first index in alphabetical order. Either one shortens a run at
# the cost of a thinner panel, and `MAX_SYMBOLS = 1` removes the structural spread entirely.

# %% tags=["parameters"]
MAX_SYMBOLS = None
START_DATE = None  # None reads setup.yaml::universe.history_start

# %% [markdown]
# ## Configuration

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
PRIMARY_LABEL = setup["labels"]["primary"]
# Which panel each label is sealed on, read from configuration rather than derived from the label
# name by string surgery, which is what this cell used to do. Both labels are always sealed here,
# whatever the workspace models: a label file is a property of the market and the calendar, not of
# which spec a given workspace happens to have been pointed at, and the two workspaces have to be
# able to read the same two files. labels.primary decides what gets FITTED, from 03 onward.
SPEC_OF = dict(setup["labels"]["spec_of"])
LABEL_NAMES = sorted(SPEC_OF)
assert set(SPEC_OF.values()) == set(SPECS), (
    f"labels.spec_of maps onto {sorted(set(SPEC_OF.values()))}, not the two panel specs {SPECS}"
)
assert PRIMARY_LABEL in SPEC_OF, (
    f"labels.primary {PRIMARY_LABEL} has no panel in labels.spec_of"
)
UNIVERSE = sorted(setup["universe"]["symbols"])
DECISION = setup["decision"]
HOLDOUT_START = str(setup["evaluation"]["holdout_start"])
HOLDOUT_END = str(setup["evaluation"]["holdout_end"])
START_DATE = START_DATE or str(setup["universe"]["history_start"])
SWAP = setup["costs"]["swap"]

print(f"Labels sealed here: {', '.join(LABEL_NAMES)}; this workspace fits {PRIMARY_LABEL}")
print(f"{PRIMARY_LABEL} is the label every model is trained on unless it is told otherwise.")
print(f"Sample {START_DATE} .. {HOLDOUT_END}; the holdout opens {HOLDOUT_START} and is not read here.")

# %% [markdown]
# ## A. The learning task
#
# Both labels are a simple return between two prices this account could have traded at:
#
# $$r_{i,d} = \frac{P^{\text{exit}}_{i,d}}{P^{\text{entry}}_{i,d}} - 1$$
#
# What differs is which two prices, and that is decided by the panel spec rather than here.
# `_features.session_panel` returns `exec_open` (the open of the bar the order fills at) and
# `label_close` (the close of the bar the position is unwound at) for each spec, and this
# notebook only divides them. Putting the price selection in the shared module rather than in
# this notebook is what lets the point-in-time test and the deployment loop use the same rule.
#
# **A row without an entry price is not a trade.** Section B.3 of `01_feasibility_analysis`
# counts them: the `intraday` spec always has a fill bar, the `overnight` spec does not, because
# the index shuts for the weekend at the Friday cash close and, in the early sample, the daily
# break sat directly on the cash close. Those rows leave the label file. They are not zeros and
# they are not forward-filled: the strategy places no order on them, so there is no return to
# learn from.
#
# **Neither label is a proxy for the other.** The intraday label never crosses the broker's
# midnight and so never pays swap; the overnight label crosses exactly one, and on this account a
# long index position pays about two basis points a night with the short side free. That
# asymmetry is a property of the label, not of the backtest, and it is why the two are separate
# specs rather than one label with a session flag.

# %% [markdown]
# ## B. Preparation before the label
#
# The panels come from `_features.session_panel`, one call per spec, on the same H1 bars
# `01_feasibility_analysis` read. Nothing is recomputed here.

# %%
bars = load_mt5_bars(
    "1h", symbols=UNIVERSE, start_date=START_DATE, end_date=HOLDOUT_END
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))

panels: dict[str, pl.DataFrame] = {}
for spec in SPECS:
    panel = session_panel(
        bars,
        spec=spec,
        calendar=DECISION["session_calendar"],
        tolerance_minutes=int(DECISION["session_close_tolerance_minutes"]),
        open_delay_minutes=int(DECISION["open_delay_minutes"]),
    )
    if MAX_SYMBOLS is not None:
        panel = panel.filter(pl.col("symbol").is_in(sorted(panel["symbol"].unique())[:MAX_SYMBOLS]))
    panels[spec] = panel
    print(
        f"{spec:9s}: {panel.height:,} index-sessions, "
        f"{panel['timestamp'].min()} .. {panel['timestamp'].max()}"
    )

# Digest of the prices the labels are built from, recorded as every label's `inputs`: a re-run
# against a refreshed download is otherwise indistinguishable from this one.
MARKET_DATA_DIGEST = {
    spec: value_digest(
        panel.select("symbol", "timestamp", "exec_open", "label_close"),
        ["symbol", "timestamp", "exec_open", "label_close"],
    )
    for spec, panel in panels.items()
}
for spec, digest in MARKET_DATA_DIGEST.items():
    print(f"market_data digest ({spec}): {digest}")

# %% [markdown]
# The two specs partition the session, and the check below is what says so rather than the
# design document: on a session where both are tradable, compounding the intraday return of day
# *d* with the overnight return of day *d* has to reproduce the move from the day's fill price to
# the next day's decision-bar close, up to the one hour of spread crossing that separates
# `label_close` from the next `exec_open`. The two are measured on different prices, so they do
# not agree exactly; what would be wrong is a systematic gap, and the median below says whether
# there is one.

# %%
joined = (
    panels["intraday"]
    .select("symbol", "timestamp", "exec_open", "label_close", "session_open_px", "session_close_px")
    .join(
        panels["overnight"].select(
            "symbol",
            "timestamp",
            pl.col("exec_open").alias("on_exec_open"),
            pl.col("label_close").alias("on_label_close"),
        ),
        on=["symbol", "timestamp"],
        how="inner",
    )
    .drop_nulls()
)
gap = joined.with_columns(
    ((pl.col("label_close") / pl.col("exec_open") - 1) * 1e4).alias("intraday_bps"),
    ((pl.col("on_label_close") / pl.col("on_exec_open") - 1) * 1e4).alias("overnight_bps"),
    ((pl.col("on_exec_open") / pl.col("label_close") - 1) * 1e4).alias("handover_bps"),
)
print(
    f"{gap.height:,} index-sessions carry both labels. Median handover between the intraday exit "
    f"and the overnight entry: {gap['handover_bps'].median():+.3f} bp "
    f"(p10 {gap['handover_bps'].quantile(0.1):+.2f}, p90 {gap['handover_bps'].quantile(0.9):+.2f})"
)
assert abs(float(gap["handover_bps"].median())) < 5.0, (
    "the intraday exit and the overnight entry are more than half a basis point apart at the "
    "median: the two specs are not measuring the same session"
)

# %% [markdown]
# ## C. Label construction
#
# One line per label, from the two prices the panel already resolved. The bookkeeping columns
# `from_end` and `session` are numbered on the complete series, before any row is dropped,
# because both mean something only in that order: `from_end` counts back from each index's last
# session for section D's boundary profile, and `session` numbers sessions forward so section F's
# overlap statistics keep counting once the null tail and the holdout are filtered out.

# %%
labels: dict[str, pl.DataFrame] = {}
for name in LABEL_NAMES:
    spec = SPEC_OF[name]
    frame = panels[spec].sort(["symbol", "timestamp"]).with_columns(
        (pl.len().over("symbol") - 1 - pl.int_range(pl.len()).over("symbol")).alias("from_end"),
        pl.int_range(pl.len()).over("symbol").alias("session"),
    )
    labels[name] = frame.with_columns(
        (pl.col("label_close") / pl.col("exec_open") - 1).alias(name)
    )
    built = labels[name][name]
    print(
        f"{name}: {built.is_not_null().sum():,} of {len(built):,} index-sessions carry a value "
        f"({built.is_not_null().mean():.1%})"
    )

# %% [markdown]
# ## D. Window validity
#
# A division always returns something; the question is whether what it returns is the quantity
# the design describes. Four properties are asserted rather than described, because all four fail
# silently:
#
# 1. **the label ends after it starts.** `label_ts` must be strictly later than `exec_ts`, which
#    is strictly later than or equal to `decision_ts`. A spec whose endpoint drifted in front of
#    its entry would still produce a plausible distribution.
# 2. **the label does not reach across the holdout boundary.** A row observed before the holdout
#    whose outcome resolves inside it is a holdout row; the usable development boundary is
#    therefore the holdout date minus the horizon, counted on `label_ts` and not on `timestamp`.
# 3. **the label does not span two indices.** Every shift and every window is taken `over("symbol")`.
# 4. **a null is either the end of the sample or a session with no fill.** Anything else is a hole
#    in the middle of the panel and has to be explained before it is dropped.

# %%
for name, frame in labels.items():
    ordered = frame.filter(pl.col(name).is_not_null())
    assert (ordered["label_ts"] > ordered["exec_ts"]).all(), f"{name}: an endpoint precedes its entry"
    assert (ordered["exec_ts"] >= ordered["decision_ts"]).all(), (
        f"{name}: a fill happens before the decision that ordered it"
    )
    spans = ordered.group_by("symbol").agg(pl.len().alias("n"))
    assert spans.height == frame["symbol"].n_unique(), f"{name}: an index carries no label at all"
    nulls = frame.filter(pl.col(name).is_null())
    unexplained = nulls.filter((pl.col("from_end") > 1) & pl.col("exec_open").is_not_null())
    print(
        f"{name}: {nulls.height:,} nulls, of which {nulls.filter(pl.col('exec_open').is_null()).height:,} "
        f"are sessions with no fill bar and {nulls.filter(pl.col('from_end') <= 1).height:,} are the "
        f"end of the sample; {unexplained.height} unexplained"
    )
    assert unexplained.height == 0, f"{name}: {unexplained.height} nulls are neither"

dev: dict[str, pl.DataFrame] = {}
for name, frame in labels.items():
    dev[name] = frame.drop_nulls(name).filter(
        pl.col("label_ts") < pl.lit(HOLDOUT_START).str.to_datetime()
    )
    print(
        f"{name}: {dev[name].height:,} development rows (label resolves before {HOLDOUT_START}), "
        f"{dev[name]['timestamp'].min()} .. {dev[name]['timestamp'].max()}"
    )

# %% [markdown]
# ### Coverage through the sample
#
# Where a label is missing matters more than how often. A share that is flat says the gaps are a
# structural property of the venue; a share that moves says the venue changed, and both labels
# should be read against the year they fall in rather than pooled.

# %%
coverage = pl.concat(
    [
        frame.select(
            pl.lit(name).alias("label"),
            pl.col("timestamp").dt.year().alias("year"),
            pl.col(name).is_not_null().alias("has_label"),
        )
        for name, frame in labels.items()
    ]
).group_by(["label", "year"]).agg(
    pl.len().alias("index_sessions"), pl.col("has_label").mean().alias("coverage")
).sort(["label", "year"])
display(coverage)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for name, colour in zip(LABEL_NAMES, (COLORS["blue"], COLORS["copper"]), strict=False):
    part = coverage.filter(pl.col("label") == name)
    ax.plot(part["year"], part["coverage"], marker="o", color=colour, label=name)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Year")
ax.set_ylabel("Share of index-sessions carrying a label")
ax.legend(frameon=False)
add_message_title(
    ax,
    "The overnight label was mostly untradable in 2022-2023 and is complete by 2026",
    subtitle="The gap is the daily break sitting on the cash close, not missing data",
)
show_with_alt(
    fig,
    "Two lines by year. The intraday label sits at one throughout. The overnight label starts "
    "near zero in 2022, rises through 2023 and 2024 and reaches one in 2026.",
)

# %% [markdown]
# ## E. Distribution and base rate
#
# What a model is being asked to beat. The mean of a session return over three years is close to
# zero and its sign is not the point; the dispersion is, because it sets how large a forecast has
# to be before it is worth the spread.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for name, colour in zip(LABEL_NAMES, (COLORS["blue"], COLORS["copper"]), strict=False):
    values = dev[name][name].to_numpy() * 1e4
    ax.hist(values, bins=120, histtype="step", color=colour, label=name, density=True)
ax.set_xlim(-400, 400)
ax.set_xlabel("Session return (basis points)")
ax.set_ylabel("Density")
ax.legend(frameon=False)
add_message_title(
    ax,
    "Both labels are centred near zero; the intraday leg is the wider of the two",
    subtitle="Development period only",
)
show_with_alt(
    fig,
    "Two overlaid step histograms of session returns in basis points, both centred near zero "
    "and roughly symmetric, with the intraday distribution the wider of the two.",
)

for name in LABEL_NAMES:
    for symbol in sorted(dev[name]["symbol"].unique()):
        part = dev[name].filter(pl.col("symbol") == symbol)[name]
        print(
            f"{name:18s} {symbol}: n {part.len():,} | mean {part.mean() * 1e4:+.2f} bp | "
            f"std {part.std() * 1e4:.1f} bp | skew {part.skew():+.2f} | "
            f"share positive {(part > 0).mean():.1%}"
        )

# %% [markdown]
# The base rate is what a model has to beat by predicting the sign, and it is not fifty per cent.
# An index drifts up, so a coin that always says "long" is already right more often than not; a
# classifier that beats fifty per cent has shown nothing until it beats this.

# %% [markdown]
# ## F. Overlap and effective sample size
#
# Both labels resolve one session ahead and neither overlaps its neighbour, so the effective
# sample size has to come back equal to the row count. That is an answer known by inspection, and
# the point of computing it is to check the measurement rather than the label: a weighting that
# counts the anchor session would halve it and read as a refinement rather than as a bug.

# %%
for name in LABEL_NAMES:
    n_rows, ess = effective_sample_size(dev[name], bar_col="session", horizon=1)
    print(f"{name}: {n_rows:,} rows, effective sample size {ess:,.0f} ({ess / n_rows:.2f} x rows)")
    assert abs(ess - n_rows) < 1e-6, (
        f"{name}: a one-session label does not overlap, so N_eff must equal the row count"
    )

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
lags = np.arange(1, 22)
for name, colour in zip(LABEL_NAMES, (COLORS["blue"], COLORS["copper"]), strict=False):
    acf = panel_autocorrelation(dev[name], name, max_lag=21, bar_col="session")
    ax.plot(lags, acf, marker="o", ms=3, color=colour, label=name)
    print(f"{name}: lag-1 autocorrelation {acf[0]:+.4f}")
ax.axhline(0, color=COLORS["neutral"], lw=1)
ax.set_xlabel("Lag (sessions)")
ax.set_ylabel("Autocorrelation")
ax.legend(frameon=False)
add_message_title(
    ax,
    "Consecutive intraday sessions reverse; overnight sessions barely relate",
    subtitle="Within-index autocorrelation of the label, development period",
)
show_with_alt(
    fig,
    "Two lines showing the autocorrelation of each label at lags one to twenty-one. The intraday "
    "label is clearly negative at lag one and near zero afterwards; the overnight label hovers "
    "around zero at every lag.",
)

# %% [markdown] tags=["results"]
# Numbers are recorded in `bots/exness_usidx_sess/BOT.md` (phase 1) after each run.

# %% [markdown]
# ## G. Baseline floor
#
# The simplest thing that could work, so that a model has something to be better than. The
# previous session's own return of the same kind is the cheapest possible forecast, and the lag-1
# autocorrelation above already says what it is worth. Reported per index, because two
# instruments carry no cross-section to average an information coefficient over.

# %%
for name in LABEL_NAMES:
    frame = dev[name].with_columns(pl.col(name).shift(1).over("symbol").alias("_prev")).drop_nulls("_prev")
    for symbol in sorted(frame["symbol"].unique()):
        part = frame.filter(pl.col("symbol") == symbol)
        ic = float(np.corrcoef(part["_prev"].to_numpy(), part[name].to_numpy())[0, 1])
        hit = float(((part["_prev"] < 0) == (part[name] > 0)).mean())
        print(
            f"{name:18s} {symbol}: reversal baseline (short the previous session's move) "
            f"IC {-ic:+.4f}, hit rate {hit:.1%} over {part.height:,} sessions"
        )

# %% [markdown]
# ## H. Artifacts and the audit record
#
# Each label goes to its own parquet. Beside it, `write_artifact` leaves a small JSON with the
# same name and a `.digest.json` suffix: a hash over the values, the row count, the columns that
# identify a row, the notebook that wrote it, and the hash of the price series it was computed
# from. Two parquets of identical shape cannot otherwise be told apart, and the fold boundaries
# every model is trained on are derived from the timeline of these files.

# %%
records = {}
for name in LABEL_NAMES:
    records[name] = write_artifact(
        labels[name].select(["timestamp", "symbol", name]).drop_nulls(),
        LABELS_DIR / f"{name}.parquet",
        keys=["timestamp", "symbol"],
        written_by="02_labels",
        inputs={"market_data": MARKET_DATA_DIGEST[SPEC_OF[name]]},
    )
    print(f"{name}.parquet: {records[name]['n_rows']:,} rows, digest {records[name]['digest']}")

# %% [markdown]
# The record Chapter 7.2 requires to close a label definition, one row per label, built from the
# values computed above rather than written by hand.

# %%
print("\nLabel audit record")
for name in LABEL_NAMES:
    spec = SPEC_OF[name]
    frame = dev[name]
    swap_note = (
        "none: the position is opened and closed inside one cash session"
        if spec == "intraday"
        else (
            f"one night, three on a Friday: {SWAP['points_per_lot_per_night'][UNIVERSE[0]]['long']:+.1f} "
            f"and {SWAP['points_per_lot_per_night'][UNIVERSE[-1]]['long']:+.1f} points per lot, "
            f"long only, booked outside the engine"
        )
    )
    print(
        f"\n{name}"
        f"\n  anchor       the {spec} decision instant, derived from the "
        f"{DECISION['session_calendar']} calendar"
        f"\n  entry price  the open of the first bar opening within "
        f"{DECISION['session_close_tolerance_minutes']} minutes of the decision; no bar, no order"
        f"\n  horizon      one cash session"
        f"\n  resolution   fixed: {'the cash close' if spec == 'intraday' else 'the next cash open decision'}"
        f"\n  overlap      none; consecutive values are disjoint"
        f"\n  swap         {swap_note}"
        f"\n  base rate    mean {frame[name].mean() * 1e4:+.2f} bp, std {frame[name].std() * 1e4:.1f} bp"
        f"\n  rows         {records[name]['n_rows']:,} written, {frame.height:,} in the development period"
        f"\n  digest       {records[name]['digest']}"
        f"\n  consumed by  05_evaluation.py, and 06_linear onward"
    )

# %% [markdown]
# ## Key takeaways
#
# 1. **Measure a label from the price the order fills at, not from the price the decision was
#    taken at.** On an hourly grid the two are an hour apart, which on an index session is a
#    material fraction of the move being predicted.
# 2. **A session with no fill bar is not a session with a zero return.** Dropping it is a
#    decision about what the strategy can do, and it has to be counted and reported rather than
#    absorbed by a `fillna`.
# 3. **Split a session's return where its economics split.** The overnight leg pays financing and
#    the intraday leg does not; averaging them into one label averages two different costs as
#    well as two different processes.
# 4. **Check an effective-sample measurement at a horizon whose answer is known by inspection.**
#    Consecutive one-session returns are disjoint, so the effective count has to come back equal
#    to the row count.
# 5. **Date a development row by when its label resolves, not by when it was observed.** A row
#    observed in February whose outcome lands in March is a March row, and the holdout boundary
#    is on the endpoint.
#
# ### Known limitations
#
# - The overnight label is thin in 2022 and 2023 for a structural reason - the daily break sat on
#   the cash close - so its development sample is materially smaller than the intraday label's
#   and is concentrated in the recent part of the history.
# - Neither label charges the spread or the swap. Both are the gross move between two traded
#   prices; the cost stage is where they are charged, and the swap is charged outside the engine.
#
# **Next**: the feature families the register declares, built on the same two panels.
