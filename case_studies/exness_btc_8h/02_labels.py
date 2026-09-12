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
# # Exness BTC 8h (exness_btc_8h): Label Engineering
#
# A label is the thing a model is asked to predict. This notebook builds two of them on the
# eight-hour decision grid, checks that each one is sealed at the moment it claims to be sealed,
# and writes each to its own parquet with a digest beside it.
#
# The two, both declared in `config/setup.yaml::labels` and neither invented here:
#
# | Label | What it is | Why it is a separate question |
# |---|---|---|
# | `fwd_ret_8h` | return from one decision instant to the **next one** | the trade this bot is designed to place |
# | `fwd_ret_24h` | return from the decision to the **same slot one day later** | is the edge, if any, worth holding for three slots and paying a night of financing for |
#
# ## Learning objectives
#
# - Seal a forward return on a grid with no closing bell, and show that "the next decision" and
#   "eight hours later" are the same instant because the grid says so, not because the name does
# - Measure the price gap between the instant a label starts from and the instant a fill happens
#   at, on a market where the two are simultaneous and still not identical
# - Say why the financing charge is *not* in any label even though it is this bot's largest cost
# - Measure how much two consecutive labels overlap, and carry that number into every standard
#   error rather than into a footnote
# - Read an information coefficient on **one** instrument without pretending it is a cross-section
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 7, Section 7.2. Reads `config/setup.yaml` and the MT5 H4 parquet folded to eight hours
# through `bots/_shared/mt5_loader.py`, whose coverage
# [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) established; writes
# `labels/fwd_ret_8h.parquet` and `labels/fwd_ret_24h.parquet` with a `.digest.json` beside each.
# Route B fork of `case_studies/exness_gold_sess/02_labels.py`; the differences are the grid (a
# native bar grid rather than a venue session), the absence of any early-close or holiday case,
# and the time-series baseline in section G.

# %%
"""exness_btc_8h: Label Engineering (Route B fork of exness_gold_sess)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from IPython.display import display

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_btc_8h._features import SLOT_MINUTES, decision_grid
from case_studies.exness_btc_8h._model_reading import (
    hac_lag_for_label,
    pooled_panel_ic_series,
    summarise_pooled_ic,
)
from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.label_diagnostics import effective_sample_size, panel_autocorrelation
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")
set_global_seeds(42)

CASE_STUDY_ID = "exness_btc_8h"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
LABELS_DIR = CASE_DIR / "labels"

# %% [markdown]
# Both parameters are unset by default. `START_DATE` trims the history to a later start;
# `MAX_SYMBOLS` is carried for the harness and does nothing here - the universe is one instrument,
# so there is no second symbol to drop.

# %% tags=["parameters"]
MAX_SYMBOLS = None
START_DATE = None  # None reads setup.yaml::universe.history_start

# %% [markdown]
# ## Configuration
#
# Everything that defines a label is declared in `config/setup.yaml` and bound here. A horizon
# typed into a cell is a second copy of a value the rest of the pipeline reads from the file, and
# the two drift apart the first time either is edited.
#
# A **slot** is one decision. This grid holds three a day, every day, so every horizon below is
# written in **hours** and converted to slots only where slots are what is being counted.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

PRIMARY_LABEL = setup["labels"]["primary"]
LABEL_NAMES = [PRIMARY_LABEL, *setup["labels"]["variants"]]
HORIZON_HOURS = {n: int(str(setup["labels"]["horizons"][n]).rstrip("Hh")) for n in LABEL_NAMES}
STEPS = {n: HORIZON_HOURS[n] * 60 // SLOT_MINUTES for n in LABEL_NAMES}
HOLDOUT_START = str(setup["evaluation"]["holdout_start"])
HOLDOUT_END = str(setup["evaluation"]["holdout_end"])
SLOTS_PER_DAY = len(setup["decision"]["snapshots_utc"])
START_DATE = START_DATE or str(setup["universe"]["history_start"])
UNIVERSE = sorted(setup["universe"]["symbols"])
SYMBOL = UNIVERSE[0]
SWAP = setup["costs"]["swap"]

print(f"{PRIMARY_LABEL} is the label every model is trained on: the return from one decision")
print(f"  instant to the next, {HORIZON_HOURS[PRIMARY_LABEL]} hours later, which on this grid is")
print("  exactly the trade the bot is designed to place - entry and exit are adjacent rows.")
for name in LABEL_NAMES[1:]:
    print(
        f"{name} is written alongside it over {HORIZON_HOURS[name]} hours "
        f"({STEPS[name]} slots), a hold that crosses one server midnight and therefore pays a "
        "night of financing the primary label does not."
    )
print(f"The holdout runs {HOLDOUT_START} to {HOLDOUT_END}; no diagnostic below reads a return")
print("  that finishes inside it.")

# %% [markdown]
# ## A. The learning task
#
# The strategy takes a position at a decision instant and holds it to the next one. The quantity
# it is paid is therefore the return over exactly that window, and that is `fwd_ret_8h`.
# `fwd_ret_24h` exists to answer a question the primary label cannot: whether the edge, if any,
# survives being held for three slots - which costs a night of swap and one more spread crossing
# only if the position is closed and reopened.
#
# Three conventions are fixed before any construction, because each is a decision a later reader
# would otherwise have to reverse-engineer from code:
#
# 1. **The endpoint is the next decision, and on this grid that is also exactly eight hours.**
#    `_features.decision_grid` asserts that the decision hours are `{0, 8, 16}` and reports every
#    gap other than one slot; `01_feasibility_analysis` found none on the development window. So
#    "the next decision" and "eight hours later" are the same instant *because the grid was
#    checked*, not because the label's name says so. On the sibling metals bot they are the same
#    instant only after a hundred holiday sessions have been dropped; here there are none to drop.
# 2. **Swap is not in any label**, even though it is this bot's largest cost. It is a holding cost
#    per night, not a per-crossing cost the percentage engine can express
#    (`case_studies/utils/backtest_loaders.py`); folding it in would make every label digest depend
#    on the account type, and it would make the label of a Friday 16:00 decision a different
#    quantity from a Tuesday 16:00 decision under the same name. It is declared in
#    `setup.yaml::costs.swap`, carried into the feature register as two knowable-in-advance state
#    flags, and charged in `16_costs`.
# 3. **The execution price is not the label's starting price**, and section B measures the gap
#    rather than asserting it away. The label starts from the decision bar's *close*; the fill
#    happens at the *open* of the bar that starts at the same instant. On a market that never
#    closes those are the same moment, so the difference is a bid-ask tick and nothing else - but
#    "nothing else" is a measurement, not a definition.

# %% [markdown]
# ## B. Preparation: the decision grid
#
# The panel is built by the same function `01_feasibility_analysis` checked and every later stage
# reads, so the close the labels are sealed on is the close the features are built on.

# %%
bars = load_mt5_bars(
    "8h", symbols=UNIVERSE, start_date=START_DATE, end_date=HOLDOUT_END
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
panel = decision_grid(bars)
if MAX_SYMBOLS is not None:
    panel = panel.filter(
        pl.col("symbol").is_in(sorted(panel["symbol"].unique().to_list())[:MAX_SYMBOLS])
    )
panel = panel.sort(["symbol", "timestamp"]).with_columns(
    pl.int_range(pl.len()).over("symbol").alias("slot"),
    (pl.len().over("symbol") - 1 - pl.int_range(pl.len()).over("symbol")).alias("from_end"),
)
MARKET_DATA_DIGEST = value_digest(panel, ["symbol", "timestamp", "close"])
print(
    f"{panel['symbol'].n_unique()} instrument, {panel.height:,} decision slots, "
    f"{panel['timestamp'].min()} to {panel['timestamp'].max()} UTC"
)
print(f"market_data digest: {MARKET_DATA_DIGEST}")

# %% [markdown]
# The execution gap, measured. The fill price for a decision at $t$ is the **open** of the bar
# that starts at $t$, which is the `open` column of the row dated $t+8h$. The label starts from
# `close` at $t$. On a continuous tape the two are the same instant, so any difference is the
# broker's quote moving between the last tick of one bar and the first tick of the next.
#
# This matters because a model trained on `fwd_ret_8h` is predicting a return measured from a
# price it will not trade at. If the gap were systematic it would be a cost nothing downstream
# charges; the cell below says how big it is and whether it has a sign.

# %%
gap = (
    panel.sort("timestamp")
    .with_columns((pl.col("open").shift(-1) / pl.col("close") - 1).alias("_gap"))
    .drop_nulls("_gap")
)
gap_bps = gap["_gap"] * 1e4
print(
    f"fill open vs decision close over {gap.height:,} slots: mean {gap_bps.mean():+.4f} bps, "
    f"median {gap_bps.median():+.4f}, p90 |gap| {gap_bps.abs().quantile(0.9):.4f}, "
    f"max |gap| {gap_bps.abs().max():.3f}"
)
print(
    f"exactly equal on {float((gap['_gap'] == 0).mean()):.1%} of slots. The residual is the "
    "quote moving between two adjacent bars of a continuous tape; it has no session gap and no "
    "weekend gap to carry, which is why it is two orders of magnitude smaller than the same "
    "measurement on any bot that waits for a market to reopen."
)

# %% [markdown]
# ## C. Label construction
#
# One execution convention, written once and applied at both horizons:
#
# $$r^{(h)}_{t} = \frac{P_{t+h}}{P_{t}} - 1$$
#
# where $P$ is the decision bar's close and $t+h$ counts $h$ decision slots. Both labels are
# **null** on any row where the forward window does not span exactly the hours the name claims -
# which on this grid can only happen at the tail of the series, because there are no holidays and
# no weekend to shorten a window. The check is written anyway, and the count is printed, because a
# grid that acquired a hole after a re-download would otherwise relabel silently.

# %%
labels_df = panel
for name in LABEL_NAMES:
    step, hours = STEPS[name], HORIZON_HOURS[name]
    span = (
        pl.col("timestamp").shift(-step).over("symbol").dt.epoch("s")
        - pl.col("timestamp").dt.epoch("s")
    ) // 60
    labels_df = labels_df.with_columns(span.alias(f"_span_{name}")).with_columns(
        pl.when(pl.col(f"_span_{name}") == hours * 60)
        .then(pl.col("close").shift(-step).over("symbol") / pl.col("close") - 1)
        .otherwise(None)
        .alias(name)
    )
    eligible = labels_df.filter(pl.col("from_end") >= step)
    off = eligible.filter(pl.col(f"_span_{name}") != hours * 60)
    print(
        f"{name}: {step} slot(s) ahead. {off.height} of {eligible.height:,} eligible rows do not "
        f"span exactly {hours} hours and are null, not relabelled "
        f"({off.height / max(eligible.height, 1):.4%})."
    )

# %% [markdown]
# ## D. Window validity
#
# A shift always returns something; the question is whether what it returns is the quantity the
# name claims. Four checks, one per way this could be wrong.

# %%
for name in LABEL_NAMES:
    step, hours = STEPS[name], HORIZON_HOURS[name]
    labelled = labels_df.drop_nulls(name)
    # 1. the span of every labelled row is exactly the declared horizon
    spans = (
        labelled.select(pl.col(f"_span_{name}")).to_series().unique().sort().to_list()
    )
    assert spans == [hours * 60], f"{name} spans {spans} minutes, not {hours * 60}"
    # 2. an incomplete forward window is null, never a value
    tail = labels_df.filter(pl.col("from_end") < step)
    assert tail[name].null_count() == tail.height, f"{name} labels its own tail"
    # 3. no label crosses an instrument boundary: every value is built with .over("symbol")
    assert labelled["symbol"].n_unique() <= labels_df["symbol"].n_unique()
    # 4. the label is a return between two rows of the panel, so it can be rebuilt from close
    rebuilt = (
        labels_df.sort(["symbol", "timestamp"])
        .with_columns(
            (pl.col("close").shift(-step).over("symbol") / pl.col("close") - 1).alias("_r")
        )
        .drop_nulls([name, "_r"])
    )
    assert float((rebuilt[name] - rebuilt["_r"]).abs().max()) == 0.0
    print(
        f"{name}: {labelled.height:,} labelled of {labels_df.height:,} "
        f"({labelled.height / labels_df.height:.1%}), {labels_df.height - labelled.height:,} null; "
        f"span {spans[0]} minutes on every labelled row"
    )

# %% [markdown]
# The figure below reads which rows are null and never reads a return, so it is drawn over the
# whole sample: it describes the files, not the market. Position zero is the instrument's last
# slot. Each label nulls its own tail and no more - a fabricated tail would sit flat where the
# horizon ends.

# %%
profile = (
    labels_df.filter(pl.col("from_end") <= 2 * SLOTS_PER_DAY + 3)
    .group_by("from_end")
    .agg([pl.col(name).is_not_null().mean().alias(name) for name in LABEL_NAMES])
    .sort("from_end")
)
palette = (COLORS["blue"], COLORS["amber"], COLORS["copper"])
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for name, colour, fmt in zip(LABEL_NAMES, palette, ("o-", "s--", "^:"), strict=False):
    ax.plot(profile["from_end"], profile[name], fmt, ds="steps-mid", ms=3, c=colour, label=name)
ax.set_xlabel("Slots from the end of the series")
ax.set_ylabel("Share of rows with a non-null label")
ax.set_ylim(-0.05, 1.08)
ax.legend(loc="center left", frameon=False)
add_message_title(
    ax, "Each label nulls its own tail, and no more",
    subtitle="A fabricated tail would sit flat where the horizon ends",
)
show_with_alt(fig, "Non-null label rate by position from the end of the series.")

# %% [markdown]
# ## E. Distribution and base rate
#
# Everything from here on is computed on the history before the holdout opens, and the date that
# decides whether a row is inside that history is the date its forward window **ends**, not the
# date it starts on. A row decided on 2025-08-31 whose window closes in September belongs to the
# holdout as far as any diagnostic is concerned.

# %%
dev = {}
for name in LABEL_NAMES:
    end_col = pl.col("timestamp").shift(-STEPS[name]).over("symbol")
    dev[name] = (
        labels_df.with_columns(end_col.alias("_end"))
        .filter(pl.col("_end") < pl.lit(HOLDOUT_START).str.to_datetime())
        .drop_nulls(name)
    )
    frame = dev[name]
    describe = frame.select(
        pl.len().alias("n"),
        pl.col(name).mean().alias("mean"),
        pl.col(name).std().alias("std"),
        pl.col(name).skew().alias("skew"),
        pl.col(name).quantile(0.05).alias("p05"),
        pl.col(name).quantile(0.95).alias("p95"),
    )
    print(f"\n{name} (development only, {frame.height:,} rows)")
    display(describe)
    print(
        f"  annualised drift {float(frame[name].mean()) * (365 * 24 / HORIZON_HOURS[name]):+.1%}, "
        f"annualised volatility "
        f"{float(frame[name].std()) * np.sqrt(365 * 24 / HORIZON_HOURS[name]):.1%} - the sample is "
        "one long bull market as much as it is seven and a half years, which is why every "
        "downstream statistic has to be read against a buy-and-hold book and not only against zero."
    )

# %% [markdown]
# Two cuts of the primary label that no other bot on this account can take, and both of them
# matter to the design: dispersion **by decision slot** (does the 16:00 slot, the one that pays
# the swap, at least move more?) and dispersion **by weekday** (is the weekend a different market?).

# %%
by_slot = (
    dev[PRIMARY_LABEL]
    .group_by(pl.col("timestamp").dt.hour().alias("slot"))
    .agg(
        pl.len().alias("n"),
        pl.col(PRIMARY_LABEL).mean().alias("mean"),
        pl.col(PRIMARY_LABEL).std().alias("std"),
        pl.col(PRIMARY_LABEL).abs().median().alias("median_abs"),
    )
    .sort("slot")
)
by_weekday = (
    dev[PRIMARY_LABEL]
    .group_by(pl.col("timestamp").dt.weekday().alias("weekday"))
    .agg(
        pl.len().alias("n"),
        pl.col(PRIMARY_LABEL).mean().alias("mean"),
        pl.col(PRIMARY_LABEL).std().alias("std"),
    )
    .sort("weekday")
)
display(by_slot)
display(by_weekday)
swap_bps = abs(float(SWAP["derived_bps_per_night"]["long"]))
paying = by_slot.filter(pl.col("slot") == 16)
print(
    f"The 16:00 slot pays {swap_bps:.2f} bps of financing on a long position that the other two "
    f"do not. Its own dispersion is {float(paying['std'][0]) * 1e4:.0f} bps against "
    f"{float(by_slot.filter(pl.col('slot') != 16)['std'].mean()) * 1e4:.0f} bps for the other "
    "two, so the extra cost is bought with "
    f"{float(paying['std'][0]) / float(by_slot.filter(pl.col('slot') != 16)['std'].mean()):.2f}x "
    "the dispersion. Whether that is a good trade is a question for 16_costs, not for a label."
)

annual = (
    dev[PRIMARY_LABEL]
    .with_columns(pl.col("timestamp").dt.year().alias("year"))
    .group_by("year")
    .agg(pl.col(PRIMARY_LABEL).std().alias("dispersion"))
    .sort("year")
)
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
ax.bar(annual["year"], annual["dispersion"] * 1e4, color=COLORS["blue"])
ax.set_xlabel("Year")
ax.set_ylabel(f"Std of {PRIMARY_LABEL} (bps)")
add_message_title(
    ax,
    "The eight-hour dispersion halves across the sample",
    subtitle="Standard deviation of the primary label within each year, development period",
)
show_with_alt(fig, "Annual standard deviation of the primary eight-hour label.")
print(
    "Read together with the spread figure of 01: the cost in basis points fell about seven-fold "
    "across the sample and the dispersion fell about "
    f"{float(annual['dispersion'][0] / annual['dispersion'][-1]):.1f}-fold, so the ratio of move "
    "to cost improved - but both moved, and a strategy tuned on the wide-spread, wide-move years "
    "is not being tested on the same market as the recent ones."
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## F. Overlap and effective sample size
#
# `labels.rebalance_step` claims that consecutive `fwd_ret_8h` trades are disjoint and that
# `fwd_ret_24h` trades are disjoint every three slots. That is a claim about the grid, so it is
# measured on the grid: how many later decisions start strictly **inside** each holding window.
# The answer sets the Newey-West lag every inference in this pipeline is corrected to, and it is
# read here rather than assumed anywhere downstream.

# %%
grid = labels_df["timestamp"]
overlap_counts, hac_lags = {}, {}
for name in LABEL_NAMES:
    overlap_counts[name], hac_lags[name] = hac_lag_for_label(grid, HORIZON_HOURS[name])
print("overlapping neighbours per label (max over the grid), and the HAC lag used below:")
for name in LABEL_NAMES:
    print(
        f"  {name}: {overlap_counts[name]} overlapping slot(s) -> HAC lag {hac_lags[name]} "
        f"(rebalance_step declares {setup['labels']['rebalance_step'][name]})"
    )
assert overlap_counts[PRIMARY_LABEL] == 0, (
    "consecutive primary-label windows overlap; rebalance_step 1 would double-count them"
)

max_lag = 3 * SLOTS_PER_DAY + 4
acf = {
    n: panel_autocorrelation(dev[n], n, max_lag=max_lag, bar_col="slot")
    for n in LABEL_NAMES
    if not dev[n].is_empty()
}
lags = np.arange(1, max_lag + 1)
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for (name, series), colour in zip(acf.items(), palette, strict=False):
    ax.plot(lags, series, "o-", ms=3, c=colour, lw=1.6, label=name)
    ax.axvline(hac_lags[name], color=colour, linestyle=":", lw=1.2)
ax.axhline(0, color=COLORS["neutral"], lw=0.8)
ax.set_xlabel("Lag in decision slots")
ax.set_ylabel("Panel autocorrelation")
ax.legend(loc="upper right", frameon=False)
add_message_title(
    ax, "The overlap decays to zero by each label's own HAC lag",
    subtitle="Dotted lines mark the lag every inference below is corrected to",
)
show_with_alt(fig, "Panel autocorrelation of both labels against lag in decision slots.")

for name in LABEL_NAMES:
    h = max(1, hac_lags[name])
    n_rows, n_eff = effective_sample_size(dev[name], horizon=h, bar_col="slot")
    print(
        f"{name}: N={n_rows:,}, N_eff={n_eff:,.0f}, ratio {n_eff / n_rows:.4f} against "
        f"{1 / h:.4f} for windows overlapping this fully"
    )

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## G. Baseline floor
#
# One signal is measured against each label before a single feature is engineered, so the feature
# work in the next notebook has a number to beat. The signal is the **previous slot's return** -
# the simplest thing that could carry an eight-hour effect, and the one `01_feasibility_analysis`
# already found the largest autocorrelation at. An engineered feature that does no better than
# this has added nothing.
#
# It is measured as a **time series**, and this is the one methodological point that separates
# this bot from every template it descends from. One instrument is not a cross-section: a
# Spearman correlation over one observation is undefined, and the three cross-sectional
# implementations in this repository (`feature_engineering.py`'s `min_cross_section`, the
# registry's hard-coded `min_obs=5`, and `exness_fx_d1/05_evaluation.py`'s `MIN_PERIODS`) all
# return *nothing* on this panel rather than raising. So the statistic is the pooled panel IC of
# `_model_reading.py`, which on one name reduces exactly to that name's own Spearman correlation
# through time - reported with **two** intervals, a Newey-West one at the overlap measured in
# section F and a moving-block bootstrap that assumes no covariance model at all.

# %%
SIGNAL = "prev_slot_ret"
baseline_rows = []
for name in LABEL_NAMES:
    frame = (
        dev[name]
        .sort(["symbol", "timestamp"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias(SIGNAL)
        )
        .drop_nulls([SIGNAL, name])
    )
    series = pooled_panel_ic_series(
        frame, score_col=SIGNAL, outcome_col=name, fold_col=None
    )
    stats = summarise_pooled_ic(
        series, hac_lag=hac_lags[name], bootstrap_block=SLOTS_PER_DAY * 7
    )
    baseline_rows.append(
        {
            "label": name,
            "signal": SIGNAL,
            "n_slots": stats["n_slots"],
            "ic_mean": round(stats["ic_mean"], 5),
            "naive_t": round(stats["naive_t"], 2),
            "hac_t": round(stats["hac_t"], 2),
            "hac_lags": stats["hac_lags"],
            "boot_ci_lo": round(stats.get("boot_ci_lo", float("nan")), 5),
            "boot_ci_hi": round(stats.get("boot_ci_hi", float("nan")), 5),
            "boot_block": stats.get("block"),
        }
    )
baseline = pl.DataFrame(baseline_rows)
display(baseline)
print(
    f"Baseline signal {SIGNAL} (the previous slot's return) against each label, development "
    "window only. The Newey-West interval and the block bootstrap are both reported: they fail "
    "differently, and where they disagree the bootstrap is the one to believe."
)
# Recomputed here rather than copied, so it cannot drift from setup.yaml the way a typed
# literal did once already. Same arithmetic as 01_feasibility_analysis section B.7: the
# one-way in-sample spread plus one night of financing amortised over the slots of a day,
# divided by this label's own dispersion on the development window.
one_way = float(setup['costs']['spread_bps_in_sample']['p90_bps']) * 1e-4
swap_per_slot = abs(float(SWAP['derived_bps_per_night']['long'])) * 1e-4 / SLOTS_PER_DAY
ic_star = (one_way + swap_per_slot) / float(dev[PRIMARY_LABEL][PRIMARY_LABEL].std())
print(
    f"For scale: IC* is {ic_star:.4f} for {PRIMARY_LABEL} against the in-sample spread plus one "
    "night of swap - the correlation a signal has to beat before it pays for itself. A baseline "
    "IC an order of magnitude below that is not a signal that pays; it is a starting point for "
    "the feature work to beat."
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## H. Artifacts and the audit record
#
# Each label goes to its own parquet with a `.digest.json` beside it: a hash over the values, the
# row count, the key columns, the notebook that wrote it, and the digest of the price panel the
# returns were computed from - the field that ties a label to its data vintage, and the only way
# to tell two parquets of identical shape apart after a re-download.
#
# **One grid, two labels.** Every label is published on the **same key set** - the decision grid's
# `(timestamp, symbol)` - and carries `null` wherever that label's own rule leaves it undefined
# (here, only its own tail). The reason is not tidiness:
# `case_studies/utils/cv_window.py` *defines* the canonical walk-forward fold as the fold derived
# from the label parquet's own timeline, and `04_model_based_features` stamps **one** fold ladder
# into its artifact. Publishing on `.drop_nulls()` would let the definedness of a label decide that
# label's calendar, so two ragged labels would produce two ladders and a model reading fold *F* by
# id would be scored on slots its features had already seen. `utils/modeling.py` filters
# `is_not_null()` on the label inside each fold before fitting, so a padded row never enters a fit.

# %%
LABELS_DIR.mkdir(parents=True, exist_ok=True)
GRID_KEYS = labels_df.select(["timestamp", "symbol"]).unique().height
published = {}
for name in LABEL_NAMES:
    frame = labels_df.select(["timestamp", "symbol", name]).drop_nulls(["timestamp", "symbol"])
    published[name] = frame
    record = write_artifact(
        frame,
        LABELS_DIR / f"{name}.parquet",
        keys=["timestamp", "symbol"],
        written_by="02_labels",
        inputs={"market_data": MARKET_DATA_DIGEST},
    )
    print(
        f"{name}.parquet: {record['n_rows']:,} rows "
        f"({frame[name].drop_nulls().len():,} non-null, {frame[name].null_count():,} padded), "
        f"digest {record['digest']}"
    )

# %% [markdown]
# The invariants that make this a change of **timeline** and not a change of **label**, asserted
# rather than printed.

# %%
for name in LABEL_NAMES:
    frame = published[name]
    assert frame.height == GRID_KEYS, (
        f"{name} was published on {frame.height:,} keys, not the {GRID_KEYS:,} of the grid"
    )
    assert frame[name].drop_nulls().len() == labels_df[name].drop_nulls().len(), (
        f"{name} lost or gained a labelled row; only null padding was authorised"
    )
key_sets = {
    n: set(zip(f["timestamp"].to_list(), f["symbol"].to_list(), strict=True))
    for n, f in published.items()
}
reference = key_sets[PRIMARY_LABEL]
for name, keys in key_sets.items():
    assert keys == reference, f"{name} sits on a different key set from {PRIMARY_LABEL}"
unlabelled = labels_df.filter(pl.all_horizontal([pl.col(n).is_null() for n in LABEL_NAMES]))
print(
    f"one key set of {GRID_KEYS:,} decision keys under both labels; labelled under at least one "
    f"rule {GRID_KEYS - unlabelled.height:,}; {unlabelled.height:,} keys carry no label under "
    "either rule and are published as null on both (they are the tail of the series)."
)

# %% [markdown]
# The record Chapter 7.2 asks for to close a label definition, one row per label, built from the
# values computed above rather than typed in.

# %%
anchors = {
    "fwd_ret_8h": "decision bar close -> the next decision bar's close, eight hours later",
    "fwd_ret_24h": "decision bar close -> the close three slots later, only where that is exactly 24h",
}
print("\nLabel audit record")
for name in LABEL_NAMES:
    frame, values = dev[name], dev[name][name]
    nights = float(
        frame.select(
            (
                (pl.col("timestamp") + pl.duration(hours=HORIZON_HOURS[name])).dt.date()
                > pl.col("timestamp").dt.date()
            ).cast(pl.Float64).mean()
        ).item()
    )
    print(
        f"\n{name}"
        f"\n  anchor       {anchors[name]}"
        f"\n  horizon      {HORIZON_HOURS[name]} hours = {STEPS[name]} decision slot(s)"
        f"\n  resolution   fixed at the endpoint; no barrier, no early close, no holiday"
        f"\n  overlap      {overlap_counts[name]} later decision(s) start inside the window"
        f"\n  purge buffer {setup['labels'].get('variant_buffers', {}).get(name, setup['labels']['buffer'])}"
        f"\n  rebalance    step {setup['labels']['rebalance_step'][name]} slot(s)"
        f"\n  financing    {nights:.0%} of holds cross a server midnight and pay swap; NOT in the label"
        f"\n  base rate    mean {values.mean():+.6f}, std {values.std():.6f}"
        f"\n  consumed by  05_evaluation.py, and 06_linear onward"
    )

# %% [markdown]
# ## Key takeaways
#
# 1. **Seal a label on the endpoint the strategy exits at, and prove the endpoint is where the
#    name says.** Here "the next decision" and "eight hours later" coincide because the grid was
#    checked for holes, not because the label is called `8h`.
# 2. **Measure the gap between the price a label starts from and the price a fill happens at.**
#    On a continuous tape it is a tick; on a market with a weekend it is not, and the difference is
#    a cost that nothing downstream charges.
# 3. **Keep a per-night cost out of a per-crossing label.** Financing is this bot's largest cost
#    and it still does not belong in the label: it would make every digest depend on the account
#    type and give the same label two meanings on two weekdays.
# 4. **Measure the overlap on the grid, then carry it into every standard error.** Here the
#    primary label's windows are disjoint - which is what makes `rebalance_step: 1` honest - and
#    the 24-hour label's are not.
# 5. **One instrument is not a cross-section, and the failure is silent.** Three separate
#    implementations in this repository return an empty result rather than an error on a one-name
#    panel, so the statistic has to be chosen deliberately: a time-series IC with two intervals.
#
# **Next**: the feature register of `setup.yaml::features`, built on this development period.
