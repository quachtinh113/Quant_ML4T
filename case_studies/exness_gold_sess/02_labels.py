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
# # Exness Gold Sessions (exness_gold_sess): Label Engineering
#
# A label is the thing a model is asked to predict. This notebook builds the labels of this bot
# on the session decision grid, checks that each one is sealed at the moment it claims to be
# sealed, and writes the ones `config/setup.yaml::labels` declares to their own parquet with a
# digest beside it.
#
# **Generation 2 (2026-09-10, `bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md` section 2).**
# The primary label is `fwd_ret_sess`: the return from the decision bar's close to the **last
# H1 bar close at or before the session close at which a `next_bar_open` exit can still fill**.
# One rule, two constants that fall out of the tape - 8 bars in London, 7 in New York, where the
# session close is the start of the broker's daily break - so the label measures exactly the
# window each book trades. The three generation-1 labels are still constructed below (the new
# label is defined against `fwd_ret_8h`, and the comparison is printed) but they are **not
# republished**: their sealed parquets stay on disk, byte for byte, and this notebook asserts it.
#
# | Label | What it is | Status in generation 2 |
# |---|---|---|
# | `fwd_ret_sess` | return from the decision bar's close to the **last tradable bar close at or before the session close** | **primary**; the trade each book actually places |
# | `fwd_ret_8h` | return from the decision bar's close to the **session close** | generation 1, sealed; equals `fwd_ret_sess` on London, differs by one bar on New York (its New York window was never traded, ruling 2.2) |
# | `fwd_ret_24h` | return from the decision to the **same slot one day later** | generation 1, sealed; **deferred** (ruling 2.3: its cost cannot be priced before the real account is read) |
# | `dir_tb_8h` | which of an ATR-scaled take-profit / stop-loss pair is touched first inside the session | generation 1, sealed; **closed** (ruling 2.4) |
#
# ## Learning objectives
#
# - Seal a forward return on a *session* endpoint rather than on a bar count, and show that the
#   two coincide only because the grid says so
# - Seal the primary label on the **tradable** endpoint - the last bar an exit can fill at - so
#   the label and the book measure the same window on both venues (parity, Chapter 25)
# - Refuse to name a label `24h` on the rows where twenty-four hours is not what it spans
# - Build a triple-barrier label whose expiry is cut at the session close, not at a bar count
# - Measure how much two consecutive labels overlap when a book runs two sleeves a day, and what
#   that leaves of the nominal sample size
# - Read an information coefficient on two instruments without pretending they are a
#   cross-section
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 7, Sections 7.2-7.3 (`07_defining_the_learning_task/03_label_methods.py` for the
# triple barrier and `measure_n_eff`). Reads `config/setup.yaml` and the MT5 H1 and D1 parquet;
# writes `labels/<name>.parquet` and a `.digest.json` beside each. Route B fork of
# `case_studies/exness_fx_d1/02_labels.py`; the differences are the session grid, the
# twenty-four-hour guard and the triple-barrier section.

# %%
"""exness_gold_sess: Label Engineering (Route B fork of exness_fx_d1)."""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.metrics import compute_ic_hac_stats

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.exness_gold_sess._features import BAR_MINUTES, session_panel
from case_studies.exness_gold_sess._hold import tradable_exit
from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.label_diagnostics import effective_sample_size, panel_autocorrelation
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")
set_global_seeds(42)

CASE_STUDY_ID = "exness_gold_sess"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
LABELS_DIR = CASE_DIR / "labels"

# %% [markdown]
# Both parameters are unset by default. `START_DATE` trims the history to a later start;
# `MAX_SYMBOLS` keeps only the first symbols in alphabetical order. Either shortens a run at the
# cost of a thinner panel - and with two instruments, `MAX_SYMBOLS = 1` removes the gold-silver
# comparison entirely.

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
# A **slot** is one decision. This grid holds two a weekday, so every horizon below is written in
# **hours** and converted to slots only where slots are what is being counted.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

PRIMARY_LABEL = setup["labels"]["primary"]
LABEL_NAMES = [PRIMARY_LABEL, *setup["labels"]["variants"]]
# Every declared horizon, not only the published labels': the generation-1 labels are still
# constructed below for the comparison, and their horizons are read from the file, never typed.
HORIZON_HOURS = {n: int(str(h).rstrip("Hh")) for n, h in setup["labels"]["horizons"].items()}
for _name in LABEL_NAMES:
    assert _name in HORIZON_HOURS, f"{_name} has no entry in labels.horizons"
CLASSIFICATION = dict(setup["labels"].get("classification_eval_label") or {})
TB = setup["labels"]["triple_barrier"]
HOLDOUT_START = str(setup["evaluation"]["holdout_start"])
HOLDOUT_END = str(setup["evaluation"]["holdout_end"])
EDGE_BLOCK = int(setup["decision"]["edge_block_minutes"])
CLOSE_TOLERANCE = int(setup["decision"]["session_close_tolerance_minutes"])
SLOTS_PER_DAY = len(setup["decision"]["snapshots"])
START_DATE = START_DATE or str(setup["universe"]["history_start"])
UNIVERSE = sorted(setup["universe"]["symbols"])

print(f"{PRIMARY_LABEL} is the label every model is trained on: the return from the decision")
print(f"  bar's close to the last tradable bar close at or before the session close, at most")
print(f"  {HORIZON_HOURS[PRIMARY_LABEL]} hours later, which on this grid is the window each book trades.")
for name in LABEL_NAMES[1:]:
    kind = "classification" if name in CLASSIFICATION else "continuous"
    print(f"{name} is written alongside it over {HORIZON_HOURS[name]} hours ({kind}).")
for cls, cont in CLASSIFICATION.items():
    print(f"{cls} is a class label, so {cont} travels in the same frame as its evaluation label")
    print("  (case_studies/research/labels.py:143-159 requires it).")
print(f"The holdout runs {HOLDOUT_START} to {HOLDOUT_END}; no diagnostic below reads a return")
print("  that finishes inside it.")

# %% [markdown]
# ## A. The learning task
#
# The strategy takes a position at the decision instant and holds it to the session close - or,
# where the session close is the start of the broker's daily break and no bar opens there, to the
# last bar before it at which an exit order can still fill. The quantity it is paid is the return
# over exactly that window, and in generation 2 that is `fwd_ret_sess` (section C.2). `fwd_ret_8h`
# is the same return measured to the session close itself; generation 1 trained on it and traded
# a seven-hour window in New York, the parity gap `bots/exness_gold_sess/NY_EXIT_DECLARATION.md`
# section 2 declared and this label closes. The two remaining generation-1 labels answered
# questions the primary cannot: whether holding past the close pays (`fwd_ret_24h`), and whether
# the *path* inside the session matters as much as its endpoint (`dir_tb_8h`).
#
# Three conventions are fixed before any construction, because each of them is a decision that a
# later reader would otherwise have to reverse-engineer from code:
#
# 1. **The endpoint is a session close, not a bar count.** `_features.decision_grid` resolves each
#    session to a decision bar and an endpoint bar and refuses any session where they are not
#    exactly eight bars apart. So on the rows that survive, "eight bars" and "the session close"
#    are the same instant - and the ones where they would not be were already dropped and counted
#    in `01_feasibility_analysis`, rather than silently mislabelled here.
# 2. **Swap is not in any label.** A label that contained the overnight financing cost would
#    change when the account type changes, and every label, feature and prediction hash with it.
#    It is a holding cost per night, not a per-crossing cost the percentage engine can express
#    (`case_studies/utils/backtest_loaders.py:2283-2285`), so it is declared in
#    `setup.yaml::costs.swap` and priced in `16_costs`.
# 3. **The execution price is not the label's starting price.** `decision.execution_delay` puts
#    the fill at the next bar's open, an hour after the close this return starts from. A model
#    trained on this label predicts slightly more than it will be paid for, and the difference is
#    a cost nothing in this notebook measures.

# %% [markdown]
# ## B. Preparation: the decision grid
#
# The panel is built by the same function `01_feasibility_analysis` checked and the backtest
# price loader reads, so the close the labels are sealed on is the close the backtest marks at.

# %%
bars = load_mt5_bars(
    "1h", symbols=UNIVERSE, start_date=START_DATE, end_date=HOLDOUT_END
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
d1 = load_mt5_bars("daily", symbols=UNIVERSE, start_date=START_DATE, end_date=HOLDOUT_END)

panel, report = session_panel(
    bars,
    edge_block_minutes=EDGE_BLOCK,
    tolerance_minutes=CLOSE_TOLERANCE,
    expected_horizon_bars=HORIZON_HOURS[PRIMARY_LABEL],
    keep_context=True,
    return_report=True,
)
if MAX_SYMBOLS is not None:
    panel = panel.filter(pl.col("symbol").is_in(sorted(panel["symbol"].unique().to_list())[:MAX_SYMBOLS]))

panel = panel.sort(["symbol", "timestamp"]).with_columns(
    pl.int_range(pl.len()).over("symbol").alias("slot"),
    (pl.len().over("symbol") - 1 - pl.int_range(pl.len()).over("symbol")).alias("from_end"),
)
MARKET_DATA_DIGEST = value_digest(
    panel.drop_nulls("label_end_ts"),
    ["symbol", "timestamp", "close", "label_end_ts", "label_end_close"],
)
print(
    f"{panel['symbol'].n_unique()} instruments, {panel.height:,} instrument-slots, "
    f"{panel['timestamp'].min()} to {panel['timestamp'].max()}"
)
print(
    f"of those, {panel['label_end_ts'].null_count():,} carry NO endpoint: the broker closed the "
    "session early (a US public holiday), so the decision row exists and the eight-hour labels "
    "do not. The decision is resolved from bars that had closed at the snapshot; the endpoint "
    "needs bars eight hours later. Resolving them together - the first version of "
    "_features.decision_grid - dropped the whole session, which both emptied features_as_of and "
    "removed precisely the thin-liquidity sessions on a criterion the decision could not know."
)
display(panel.group_by(["symbol", "session"]).len().sort(["symbol", "session"]))
# The digest is taken over the rows that CARRY a label, so it names the vintage the labels were
# sealed on and does not move when a row without an endpoint is added to or removed from the
# panel around them.
print(f"market_data digest (labelled rows only): {MARKET_DATA_DIGEST}")

# %% [markdown]
# ## C. Label construction
#
# ### C.1 `fwd_ret_8h` - the session return
#
# $$r_{i,t} = \frac{P^{\text{close of session}}_{i,t}}{P_{i,t}} - 1$$
#
# Both prices are already on the panel, because `decision_grid` resolved them together: the
# label cannot silently take its endpoint from a different session than the decision belongs to.

# %%
labels_df = panel.with_columns(
    (pl.col("label_end_close") / pl.col("close") - 1).alias("fwd_ret_8h")
)

# %% [markdown]
# ### C.2 `fwd_ret_sess` - the tradable session return (generation 2)
#
# $$r^{\text{sess}}_{i,t} = \frac{P_{i,c}}{P_{i,t}} - 1, \qquad
# c = \max\{\, c \le E_{i,t} : \text{a bar closes at } c \text{ and a bar opens at } c \,\}$$
#
# where $E_{i,t}$ is the sealed session-close endpoint `label_end_ts`. A bar that *opens* at $c$
# is the price-grid row keyed $c + 60$, and a `next_bar_open` exit fills at that row's open, which
# is the price at $c$ (`bots/exness_gold_sess/PRICE_GRID_DECLARATION.md` section 1.1). So $c$ is
# the last instant the position can be closed at a price this label reads. The rule is written
# once, in `_hold.tradable_exit`, and applied to both venues; the two constants are its output:
#
# - London: the tape runs through the close, $c = E$, and the label **equals** `fwd_ret_8h`;
# - New York: the close is the start of the daily break, no bar opens at $E$, $c = E - 60$ min,
#   and the label equals the seven-hour probe `_report_phase5.py::ny_probe` measured in
#   generation 1. Nothing new is invented here; a diagnostic is sealed as a label.
#
# Point-in-time: $E$ is the venue's calendar close; whether a bar opens at an instant is known at
# that instant, at or after the endpoint - admissible for a label sealed at its endpoint, never for
# a feature. The backtest's hold constant is still derived separately, on the registered grid, by
# `_hold.derive_hold_bars`; `tests/test_labels_gen2.py` asserts the two derivations agree.

# %%
exit_frame = tradable_exit(panel, bars)
labels_df = labels_df.join(
    exit_frame.select("symbol", "timestamp", "tradable_exit_ts", "tradable_exit_close"),
    on=["symbol", "timestamp"],
    how="left",
).with_columns((pl.col("tradable_exit_close") / pl.col("close") - 1).alias("fwd_ret_sess"))

# The label exists on exactly the rows whose session-close endpoint resolved: a sealed endpoint
# always has a tradable exit at or before it, and a row without an endpoint has none.
assert labels_df.filter(
    pl.col("label_end_ts").is_not_null() & pl.col("tradable_exit_ts").is_null()
).is_empty(), "a sealed endpoint has no tradable exit"
assert labels_df.filter(
    pl.col("label_end_ts").is_null() & pl.col("fwd_ret_sess").is_not_null()
).is_empty(), "a row without an endpoint carries a tradable session return"

_sealed = labels_df.drop_nulls("tradable_exit_ts").with_columns(
    ((pl.col("label_end_ts").dt.epoch("s") - pl.col("tradable_exit_ts").dt.epoch("s")) // 60)
    .alias("label_end_minus_exit_minutes"),
    ((pl.col("tradable_exit_ts").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")) // 3600)
    .alias("hours_to_exit"),
)
exit_geometry = (
    _sealed.group_by(["session", "hours_to_exit", "label_end_minus_exit_minutes"])
    .len()
    .sort(["session", "hours_to_exit"])
)
print("tradable exit geometry, per book (nothing typed; both columns are outputs of the rule):")
display(exit_geometry)
# The exit must never run past the declared horizon, which is an UPPER bound (London's 8 bars).
assert int(_sealed["hours_to_exit"].max()) <= HORIZON_HOURS["fwd_ret_sess"], (
    f"a tradable exit lies beyond the declared {HORIZON_HOURS['fwd_ret_sess']}H horizon"
)

# The comparison the declaration asks to be printed, on the DEVELOPMENT window only (the date
# that decides is the endpoint's, not the decision's). London: identical to fwd_ret_8h on every
# row. New York: the Pearson correlation with fwd_ret_8h, once on every development row (the
# generation-1 probe's own definition, _report_phase5.py::ny_probe) and once on the rows the fold
# ladder reaches (2018-08-30 onward), because before that date the broker's break did not sit
# at the New York close and the tradable exit is the endpoint itself.
_dev_sealed = _sealed.filter(pl.col("label_end_ts") < pl.lit(HOLDOUT_START).str.to_datetime())
# The earliest train_start of the fold ladder, derived by the same call 04_model_based_features
# makes on the same timeline (the decision grid's dates), never typed.
_timeline = labels_df.select(pl.col("timestamp").dt.date().alias("timestamp")).unique().sort("timestamp")
_folds = generate_cv_splits(
    _timeline, case_study_id=CASE_STUDY_ID, label_buffer=setup["labels"]["buffer"], date_col="timestamp"
)
FOLD_REACH_START = min(str(f["train_start"])[:10] for f in _folds)
print(f"fold ladder: {len(_folds)} folds, earliest train_start {FOLD_REACH_START}")
# The generation-1 probe itself (`_report_phase5.py::ny_probe`: the close one bar BEFORE the
# endpoint on every row, including the 2017 rows where that bar is not the tradable exit), so
# the number generation 1 reported can be reproduced next to the new label's.
_probe_closes = bars.select(
    "symbol",
    (pl.col("timestamp") + pl.duration(minutes=BAR_MINUTES)).alias("_probe_end_ts"),
    pl.col("close").alias("_probe_close"),
)
_dev_sealed = (
    _dev_sealed.with_columns((pl.col("label_end_ts") - pl.duration(minutes=BAR_MINUTES)).alias("_probe_end_ts"))
    .join(_probe_closes, on=["symbol", "_probe_end_ts"], how="left")
    .with_columns((pl.col("_probe_close") / pl.col("close") - 1).alias("fwd_ret_7h_probe"))
)


def _pearson(frame: pl.DataFrame, a: str, b: str) -> float:
    rows = frame.drop_nulls([a, b])
    return round(float(np.corrcoef(rows[a].to_numpy(), rows[b].to_numpy())[0, 1]), 4)


comparison_rows = []
for symbol in UNIVERSE:
    ldn = _dev_sealed.filter((pl.col("session") == "london") & (pl.col("symbol") == symbol))
    ny = _dev_sealed.filter((pl.col("session") == "ny") & (pl.col("symbol") == symbol))
    ny_reach = ny.filter(pl.col("timestamp") >= pl.lit(FOLD_REACH_START).str.to_datetime())
    comparison_rows.append(
        {
            "symbol": symbol,
            "london_rows": ldn.height,
            "london_max_abs_diff_vs_8h": float((ldn["fwd_ret_sess"] - ldn["fwd_ret_8h"]).abs().max()),
            "ny_rows": ny.height,
            "ny_pearson_sess_vs_8h_all_dev": _pearson(ny, "fwd_ret_sess", "fwd_ret_8h"),
            "ny_pearson_probe_vs_8h_all_dev": _pearson(ny, "fwd_ret_7h_probe", "fwd_ret_8h"),
            "ny_rows_fold_reach": ny_reach.height,
            "ny_pearson_sess_vs_8h_fold_reach": _pearson(ny_reach, "fwd_ret_sess", "fwd_ret_8h"),
            "ny_max_abs_diff_sess_vs_probe_fold_reach": float(
                (ny_reach["fwd_ret_sess"] - ny_reach["fwd_ret_7h_probe"]).abs().max()
            ),
            "ny_rows_exit_at_endpoint": ny.filter(pl.col("label_end_minus_exit_minutes") == 0).height,
        }
    )
comparison = pl.DataFrame(comparison_rows)
with pl.Config(tbl_cols=12, tbl_width_chars=240):
    display(comparison)
assert (comparison["ny_max_abs_diff_sess_vs_probe_fold_reach"] == 0.0).all(), (
    "fwd_ret_sess is not the seven-hour probe on a New York row the folds reach"
)
assert (comparison["london_max_abs_diff_vs_8h"] == 0.0).all(), (
    "fwd_ret_sess is not fwd_ret_8h on every London row"
)

# %% [markdown]
# ### C.3 `fwd_ret_24h` - and the rows where twenty-four hours is not what it spans
#
# Two slots ahead is the same venue's decision one day later - on four weekdays out of five. On a
# Friday, two slots ahead is Monday, seventy-two hours away, and a Wednesday-to-Thursday overnight
# is not the same trade as a weekend hold: it pays one night of swap instead of three
# (`setup.yaml::costs.swap.rollover3days`), and it crosses no weekend gap risk.
#
# The choice made here, and it is a choice: the label is **null** on any row where the span is
# not exactly twenty-four hours, and the count is reported. The alternative - defining the label
# as "the next same-venue decision, whenever that is" - keeps the Friday rows but makes the name
# a lie on a fifth of the sample, and a label whose name does not describe it is the defect
# `nasdaq100_microstructure/config/setup.yaml:604-607` records in another form.

# %%
step = HORIZON_HOURS["fwd_ret_24h"] * 60 // (24 * 60 // SLOTS_PER_DAY)  # = 2 slots
span_minutes = (
    pl.col("timestamp").shift(-step).over("symbol").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")
) // 60
labels_df = labels_df.with_columns(span_minutes.alias("_span_24h")).with_columns(
    pl.when(pl.col("_span_24h") == HORIZON_HOURS["fwd_ret_24h"] * 60)
    .then(pl.col("close").shift(-step).over("symbol") / pl.col("close") - 1)
    .otherwise(None)
    .alias("fwd_ret_24h")
)
off_span = labels_df.filter(
    pl.col("from_end") >= step, pl.col("_span_24h") != HORIZON_HOURS["fwd_ret_24h"] * 60
)
print(
    f"fwd_ret_24h: {step} slots ahead. {off_span.height:,} of "
    f"{labels_df.filter(pl.col('from_end') >= step).height:,} rows "
    f"({off_span.height / max(labels_df.filter(pl.col('from_end') >= step).height, 1):.1%}) do not "
    f"span exactly {HORIZON_HOURS['fwd_ret_24h']} hours and are null, not relabelled."
)
display(
    off_span.group_by(pl.col("_span_24h").alias("span_minutes"))
    .len()
    .sort("span_minutes")
    .head(10)
)
# What that leaves, stated before any sweep rather than discovered in one. Every Friday row of
# both venues is null, so `fwd_ret_24h` is a MONDAY-TO-THURSDAY book: about 403 decision slots a
# year, not the 504 evaluation.periods_per_year declares, and 13_backtest must hold CASH on the
# Friday slots rather than let Thursday's weights drift through the weekend paying three nights
# of swap. Recorded in bots/exness_gold_sess/BOT.md as a phase-5 prerequisite.
_null_by_weekday = (
    labels_df.filter(pl.col("from_end") >= step)
    .group_by([pl.col("timestamp").dt.weekday().alias("weekday"), "symbol"])
    .agg(pl.col("fwd_ret_24h").is_null().mean().alias("null_share"))
    .sort(["weekday", "symbol"])
)
display(_null_by_weekday)
# The null pattern is a property of the CALENDAR, not of either instrument: the two metals quote
# on the same sessions, so a row null on gold is null on silver. A difference here would mean one
# metal is missing a session the other has, which is a data problem and not a label rule.
_pattern = (
    labels_df.filter(pl.col("from_end") >= step)
    .select("timestamp", "symbol", pl.col("fwd_ret_24h").is_null().alias("_null"))
    .pivot(index="timestamp", on="symbol", values="_null")
    .drop_nulls()
)
_disagree = _pattern.filter(pl.col(UNIVERSE[0]) != pl.col(UNIVERSE[1]))
print(
    f"fwd_ret_24h null pattern: identical on both metals at "
    f"{(_pattern.height - _disagree.height) / max(_pattern.height, 1):.4%} of the "
    f"{_pattern.height:,} shared decision instants ({_disagree.height} disagree)"
)
weekday_shares = {
    int(r["weekday"]): round(float(r["null_share"]), 4)
    for r in _null_by_weekday.filter(pl.col("symbol") == UNIVERSE[0]).iter_rows(named=True)
}
print(f"  null share by weekday (1 = Monday), {UNIVERSE[0]}: {weekday_shares}")

# %% [markdown]
# ### C.4 `dir_tb_8h` - the triple barrier, cut at the session close
#
# A take-profit and a stop-loss are placed either side of the entry, and the label records which
# is touched first inside the session; if neither is, the label is `0`. The barriers are scaled by
# the instrument's own recent volatility rather than fixed at a percentage
# (`07_defining_the_learning_task/03_label_methods.py:690-698`), because gold and silver differ
# enough in volatility that one fixed percentage would be touched constantly on silver and never
# on gold - and because the same percentage means different things in 2017 and in 2026.
#
# Two details that matter more than they look:
#
# - the ATR comes from **D1** bars joined **asof backward on the D1 bar's close instant** (server
#   midnight, 00:00 UTC). At a 09:00 UTC decision the newest usable D1 bar is the previous day's.
#   Joining on the calendar date instead would hand the label a bar that closes after it;
# - the expiry is `min(8 bars, the session close)`, and on this grid those coincide on every
#   surviving row *by construction* - which is the point of having `decision_grid` refuse the rows
#   where they would not.
#
# The library's `triple_barrier_labels` counts a `max_holding_period` in **rows of a continuous
# series**, which on a grid with a nightly break and a weekend would run past the session close;
# the evaluation below is therefore written against the session's own eight bars. It is the same
# rule the chapter describes, applied where the chapter's row counting does not reach.

# %%
atr_period = int(str(TB["atr_source"]).split("_")[-1])
d1_atr = (
    d1.sort(["symbol", "timestamp"])
    .with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1).over("symbol")).abs(),
            (pl.col("low") - pl.col("close").shift(1).over("symbol")).abs(),
        ).alias("_tr")
    )
    .with_columns(
        (pl.col("_tr").rolling_mean(atr_period).over("symbol") / pl.col("close")).alias("atr_pct"),
        (pl.col("timestamp").cast(pl.Datetime("us")) + pl.duration(days=1)).alias("at"),
    )
    .select("symbol", "at", "atr_pct")
    .sort(["symbol", "at"])
)
labels_df = (
    labels_df.sort(["symbol", "timestamp"])
    .with_columns(pl.col("timestamp").alias("at"))
    .join_asof(d1_atr, on="at", by="symbol", strategy="backward")
    .drop("at")
    .sort(["symbol", "timestamp"])
)

h_bars = int(TB["max_holding_bars"])
bar_close = (pl.col("timestamp") + pl.duration(minutes=BAR_MINUTES)).alias("bar_close")
tb_rows: list[pl.DataFrame] = []
tb_report: dict[str, dict] = {}
for symbol in sorted(labels_df["symbol"].unique().to_list()):
    g = bars.filter(pl.col("symbol") == symbol).with_columns(bar_close).sort("bar_close")
    rows = labels_df.filter(pl.col("symbol") == symbol).sort("timestamp")
    closes = g["bar_close"].to_numpy().astype("datetime64[us]")
    highs, lows = g["high"].to_numpy(), g["low"].to_numpy()
    # position of each decision bar in the symbol's own H1 series; the barrier path is the
    # h_bars bars that follow it, which end exactly at the session close on every kept row
    i = np.searchsorted(closes, rows["timestamp"].to_numpy().astype("datetime64[us]"), side="left")
    ok = (i + h_bars) < len(closes)
    idx = np.where(ok, i, 0)
    path = idx[:, None] + np.arange(1, h_bars + 1)[None, :]
    hi, lo = highs[path], lows[path]
    entry = rows["close"].to_numpy()
    atr = rows["atr_pct"].to_numpy()
    upper = entry * (1 + float(TB["upper_atr_multiple"]) * atr)
    lower = entry * (1 - float(TB["lower_atr_multiple"]) * atr)
    hit_up = hi >= upper[:, None]
    hit_dn = lo <= lower[:, None]
    t_up = np.where(hit_up.any(1), hit_up.argmax(1), h_bars + 1)
    t_dn = np.where(hit_dn.any(1), hit_dn.argmax(1), h_bars + 1)
    label = np.where(t_up < t_dn, 1, np.where(t_dn < t_up, -1, int(TB["timeout_label"])))
    # both barriers inside the same bar: the H1 bar does not say which came first, so the row is
    # NOT guessed. It is the timeout class and it is counted.
    ambiguous = (t_up == t_dn) & (t_up <= h_bars)
    # An endpoint that does not exist cannot cut the expiry at the session close: on those rows
    # the next eight H1 bars run PAST the close into the following session, so the barrier path
    # would answer a different question from the one the label's name asks.
    valid = ok & np.isfinite(atr) & rows["label_end_ts"].is_not_null().to_numpy()
    tb_report[symbol] = {
        "rows": int(len(label)),
        "usable": int(valid.sum()),
        "upper_first": int((label[valid] == 1).sum()),
        "lower_first": int((label[valid] == -1).sum()),
        "timeout_or_ambiguous": int((label[valid] == int(TB["timeout_label"])).sum()),
        "ambiguous_same_bar": int((ambiguous & valid).sum()),
    }
    # NaN, not None: np.where(cond, int_array, None) produces an object array polars cannot cast.
    values = np.where(valid, label, np.nan).astype("float64")
    tb_rows.append(
        rows.select("symbol", "timestamp").with_columns(
            pl.Series("dir_tb_8h", values, dtype=pl.Float64).fill_nan(None)
        )
    )
labels_df = labels_df.join(pl.concat(tb_rows), on=["symbol", "timestamp"], how="left")

print(
    f"dir_tb_8h: barriers at +-{TB['upper_atr_multiple']} x D1 ATR({atr_period}) of the entry "
    f"price, expiry min({h_bars} bars, session close), timeout class {TB['timeout_label']}"
)
display(pl.DataFrame([{"symbol": k, **v} for k, v in sorted(tb_report.items())]))

# %% [markdown]
# ## D. Window validity
#
# A shift always returns something; the question is whether what it returns is the quantity the
# name claims. Four checks, one per way this could be wrong.

# %%
for label_name in LABEL_NAMES:
    labelled = labels_df.drop_nulls(label_name)
    if label_name == "fwd_ret_8h":
        # 1. the label exists on exactly the rows whose endpoint resolved, and nowhere else.
        # It is NOT "every row": a session the broker closed early keeps its decision and has no
        # endpoint, so it has no eight-hour return either. Asserting equality of the two SETS is
        # the check; asserting "every row has one" would only be true because the early closes
        # had been silently removed from the panel.
        assert labelled.height == labels_df.drop_nulls("label_end_ts").height, (
            "fwd_ret_8h is labelled on a different set of rows than the endpoints resolved on"
        )
        assert labels_df.filter(
            pl.col("label_end_ts").is_null() & pl.col(label_name).is_not_null()
        ).is_empty(), "a row without an endpoint carries a session return"
        spans = (
            labelled.select(
                (pl.col("label_end_ts").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")) // 3600
            )
            .to_series()
            .unique()
            .to_list()
        )
        assert spans == [HORIZON_HOURS[label_name]], f"fwd_ret_8h spans {spans} hours"
    if label_name == "fwd_ret_sess":
        # 1'. the same set identity as fwd_ret_8h - every sealed endpoint has a tradable exit -
        # and the span is bounded by the declared horizon, never asserted to a typed constant:
        # what each book resolves to is printed in section C.2 and re-derived on the registered
        # grid by tests/test_labels_gen2.py.
        assert labelled.height == labels_df.drop_nulls("label_end_ts").height, (
            "fwd_ret_sess is labelled on a different set of rows than the endpoints resolved on"
        )
        assert labels_df.filter(
            pl.col("label_end_ts").is_null() & pl.col(label_name).is_not_null()
        ).is_empty(), "a row without an endpoint carries a tradable session return"
        by_book = (
            labelled.group_by("session")
            .agg(
                ((pl.col("tradable_exit_ts").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")) // 3600)
                .unique()
                .sort()
                .alias("hours")
            )
            .sort("session")
        )
        print(f"  fwd_ret_sess hours from decision to tradable exit, per book: {by_book.to_dicts()}")
        assert all(max(r["hours"]) <= HORIZON_HOURS[label_name] for r in by_book.to_dicts())
    if label_name == "fwd_ret_24h":
        # 2. an incomplete or off-span forward window is null, never a value
        tail = labels_df.filter(pl.col("from_end") < step)
        assert tail[label_name].null_count() == tail.height, label_name
    # 3. no label crosses an instrument boundary: every value is built with .over("symbol")
    per_symbol = labelled.group_by("symbol").len()
    assert per_symbol.height == labels_df["symbol"].n_unique() or labelled.is_empty(), label_name
    print(
        f"{label_name}: {labelled.height:,} labelled of {labels_df.height:,} "
        f"({labelled.height / labels_df.height:.1%}), "
        f"{labels_df.height - labelled.height:,} null"
    )
# 4. the classification label's evaluation label travels in the same frame (only for a class
#    label this run publishes; the pinned mapping keeps its generation-1 entry as documentation)
for cls, cont in CLASSIFICATION.items():
    if cls in LABEL_NAMES:
        assert cont in labels_df.columns, f"{cls} needs {cont} in the same frame"

# %% [markdown]
# The figure below reads which rows are null and never reads a return, so it is drawn over the
# whole sample: it describes the files, not the market. Position zero is each instrument's last
# slot. `fwd_ret_8h` and `dir_tb_8h` end on the session close, so their tails are short;
# `fwd_ret_24h` nulls its own two-slot tail *and* every Friday.

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
ax.set_xlabel("Slots from the end of each instrument's series")
ax.set_ylabel("Share of instruments with a non-null label")
ax.set_ylim(-0.05, 1.08)
ax.legend(loc="center left", frameon=False)
add_message_title(
    ax,
    "Each label nulls its own tail, and no more",
    subtitle="A fabricated tail would sit flat where the horizon ends",
)
show_with_alt(fig, "Non-null label rate by position from the end of each instrument's series.")

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
    end_col = pl.col("label_end_ts") if name != "fwd_ret_24h" else pl.col("timestamp").shift(
        -step
    ).over("symbol")
    dev[name] = (
        labels_df.with_columns(end_col.alias("_end"))
        .filter(pl.col("_end") < pl.lit(HOLDOUT_START).str.to_datetime())
        .drop_nulls(name)
    )
    frame = dev[name]
    if frame.is_empty():
        continue
    describe = (
        frame.group_by("symbol")
        .agg(
            pl.len().alias("n"),
            pl.col(name).mean().alias("mean"),
            pl.col(name).std().alias("std"),
            pl.col(name).quantile(0.05).alias("p05"),
            pl.col(name).quantile(0.95).alias("p95"),
        )
        .sort("symbol")
    )
    print(f"\n{name} (development only, {frame.height:,} rows)")
    display(describe)

# %% [markdown]
# The base rate a session bot has to watch is not the mean of the label - it is the **dispersion**
# a signal has to work inside, per session and per instrument. Where dispersion narrows, the same
# information coefficient converts into less return, and the cost stays where it was.

# %%
annual = (
    dev[PRIMARY_LABEL]
    .with_columns(pl.col("timestamp").dt.year().alias("year"))
    .group_by(["year", "symbol"])
    .agg(pl.col(PRIMARY_LABEL).std().alias("dispersion"))
    .sort(["symbol", "year"])
)
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
width = 0.4
for offset, (symbol, colour) in enumerate(zip(sorted(UNIVERSE), palette, strict=False)):
    g = annual.filter(pl.col("symbol") == symbol)
    ax.bar(g["year"] + (offset - 0.5) * width, g["dispersion"] * 1e4, width=width, color=colour, label=symbol)
ax.set_xlabel("Year")
ax.set_ylabel(f"Std of {PRIMARY_LABEL} (bps)")
ax.legend(frameon=False, loc="upper left")
add_message_title(
    ax,
    "Silver's session dispersion is roughly twice gold's, every year",
    subtitle=f"Standard deviation of {PRIMARY_LABEL} within each year, development period",
)
show_with_alt(fig, "Annual standard deviation of the primary session label, one bar per metal.")

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## F. Overlap and effective sample size
#
# This is the section the design of this bot makes unavoidable. On the pooled book a London
# decision at 09:00 UTC is held to 17:00 UTC while the New York slot at 14:00 UTC opens **inside**
# that hold: consecutive labels share three of their eight hours. `labels.rebalance_step` is 1 for
# the eight-hour labels and the overlap is not bent away - it is measured here and carried into
# every inference as a HAC lag.
#
# The number of overlapping neighbours is read off the grid itself rather than assumed, because it
# differs between the London slot (whose window contains the next slot) and the New York slot
# (whose window contains no other decision).

# %%
overlap_counts = {}
for name in LABEL_NAMES:
    window_h = HORIZON_HOURS[name]
    g = labels_df.filter(pl.col("symbol") == sorted(UNIVERSE)[0]).sort("timestamp")
    ts = g["timestamp"].to_numpy().astype("datetime64[us]")
    end = ts + np.timedelta64(window_h, "h")
    # how many later decisions start strictly inside each holding window
    counts = np.searchsorted(ts, end, side="left") - np.arange(len(ts)) - 1
    overlap_counts[name] = int(np.max(counts)) if len(counts) else 0
HAC_LAGS = {name: overlap_counts[name] + 1 for name in LABEL_NAMES}
print("overlapping neighbours per label (max over the grid), and the HAC lag used below:")
for name in LABEL_NAMES:
    print(f"  {name}: {overlap_counts[name]} overlapping slot(s) -> HAC lag {HAC_LAGS[name]}")

max_lag = 2 * SLOTS_PER_DAY + 4
acf = {
    n: panel_autocorrelation(dev[n], n, max_lag=max_lag, bar_col="slot")
    for n in LABEL_NAMES
    if not dev[n].is_empty()
}
lags = np.arange(1, max_lag + 1)
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for (name, series), colour in zip(acf.items(), palette, strict=False):
    ax.plot(lags, series, "o-", ms=3, c=colour, lw=1.6, label=name)
    ax.axvline(HAC_LAGS[name], color=colour, linestyle=":", lw=1.2)
ax.axhline(0, color=COLORS["neutral"], lw=0.8)
ax.set_xlabel("Lag in decision slots")
ax.set_ylabel("Panel autocorrelation")
ax.legend(loc="upper right", frameon=False)
add_message_title(
    ax,
    "The overlap decays to zero by each label's own HAC lag",
    subtitle="Dotted lines mark the lag every inference below is corrected to",
)
show_with_alt(fig, "Panel autocorrelation of the three labels against lag in decision slots.")

for name in LABEL_NAMES:
    if dev[name].is_empty():
        continue
    h = max(1, HAC_LAGS[name])
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
# One signal is measured against each continuous label before a single feature is engineered, so
# the feature work in the next notebook has a number to beat. The signal is the **opening hour's
# move** - the return of the very bar the decision is taken on - which is the simplest thing that
# could carry a session effect. An engineered feature that does no better than this has added
# nothing.
#
# It is measured **inside each instrument**, not across them. Two names are not a cross-section: a
# Spearman correlation over two points takes three values, so a cross-sectional information
# coefficient here would measure the coin flip and not the signal. The construction is the
# rank-normalised product of `case_studies/xau_fx_mt5_d1/05_evaluation.py:286-307` - within-symbol
# ranks mapped to zero mean and unit variance, multiplied, then averaged per slot - whose mean is
# the Spearman correlation and whose per-slot series can be given a Newey-West standard error at
# the overlap measured in section F.

# %%
SIGNAL = "sess_ret_1h"


def rank_normalised(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Within-instrument ranks mapped to zero mean and unit variance."""
    return frame.with_columns(
        (
            ((pl.col(c).rank(method="average").over("symbol") - 0.5) / pl.len().over("symbol") - 0.5)
            * np.sqrt(12.0)
        ).alias(f"_z_{c}")
        for c in columns
    )


baseline_rows = []
for name in LABEL_NAMES:
    if name in CLASSIFICATION:
        continue  # a class label needs a different statistic; 05_evaluation is where that lives
    frame = dev[name].with_columns(
        (pl.col("close") / pl.col("session_open_px") - 1).alias(SIGNAL)
    )
    for scope in ("pooled", *sorted(UNIVERSE)):
        rows = frame if scope == "pooled" else frame.filter(pl.col("symbol") == scope)
        rows = rows.drop_nulls([SIGNAL, name])
        if rows.height < 100:
            continue
        z = rank_normalised(rows, [SIGNAL, name]).with_columns(
            (pl.col(f"_z_{SIGNAL}") * pl.col(f"_z_{name}")).alias("_p")
        )
        series = (
            z.group_by("timestamp", maintain_order=True)
            .agg(pl.col("_p").mean().alias("ic"), pl.len().alias("n_obs"))
            .sort("timestamp")
        )
        stats = compute_ic_hac_stats(series, ic_col="ic", label_horizon=HAC_LAGS[name])
        baseline_rows.append(
            {
                "label": name,
                "scope": scope,
                "n_rows": rows.height,
                "n_periods": int(stats["n_periods"]),
                "mean_ic": round(float(stats["mean_ic"]), 5),
                "hac_t": round(float(stats["t_stat"]), 2),
                "hac_lags": int(stats["effective_lags"]),
                "naive_t": round(float(stats["naive_t_stat"]), 2),
                "p_value": float(stats["p_value"]),
            }
        )
baseline = pl.DataFrame(baseline_rows)
display(baseline)
print(
    f"Baseline signal {SIGNAL} (the decision bar's own return) against each continuous label, "
    "development window only, Newey-West corrected at the overlap measured in section F."
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## H. Artifacts and the audit record
#
# Each label goes to its own parquet with a `.digest.json` beside it: a hash over the values, the
# row count, the key columns, the notebook that wrote it, and the digest of the price panel the
# labels were computed from - the field that ties a label to its data vintage, and the only way to
# tell two parquets of identical shape apart after a re-download.
#
# `dir_tb_8h` carries `fwd_ret_8h` in the same file, because
# `case_studies/research/labels.py:143-159` requires a classification label's continuous
# evaluation label to be in the frame it is published from.
#
# **One grid, three labels** (`bots/exness_gold_sess/FOLD_GEOMETRY_DECLARATION.md`, A-prime, and
# the second declaration of 2026-09-08 that accepted the fold ladder moving one day). Every label
# is published on the **same key set** - the decision grid's `(timestamp, symbol)` - and carries
# `null` wherever that label's own rule leaves it undefined. The dropped rows are dropped by
# `drop_nulls(["timestamp", "symbol"])`, which can only remove a key that is not a key, never a
# row whose label happens to be unresolvable.
#
# The reason is not tidiness. `case_studies/utils/cv_window.py:95-179` *defines* the canonical
# walk-forward fold as the fold derived from the label parquet's own timeline, and
# `04_model_based_features` stamps **one** fold ladder into its artifact. Publishing on
# `.drop_nulls()` made the definedness of a label decide that label's calendar, so three ragged
# labels produced three ladders and a model reading fold F of the artifact by id was scored on
# sessions its features had already seen. Publishing on the decision grid makes the library's
# proxy - "the label parquet is the trading calendar" - an equality instead of an assumption.
#
# Nothing is added to or removed from any *labelled* row: `utils/modeling.py:1896-1898` filters
# `is_not_null() & is_not_nan()` on the label inside each fold before fitting, so a padded row
# never enters a fit.
#
# **Generation 2 publishes only what `setup.yaml::labels` declares** (`fwd_ret_sess`). The
# generation-1 parquets already on disk are not rewritten, not deleted and not re-digested: their
# sidecar digests are read before the publish loop and asserted unchanged after it, so a re-run of
# this notebook can never move a sealed generation-1 identity (`GEN2_DECLARATION_2026-09-10.md`
# L0.3).

# %%
LABELS_DIR.mkdir(parents=True, exist_ok=True)
GRID_KEYS = labels_df.select(["timestamp", "symbol"]).unique().height
# the sealed parquets this notebook must NOT touch: every label on disk that is not declared
SEALED_ELSEWHERE = sorted(
    p.stem for p in LABELS_DIR.glob("*.parquet")
    if p.stem not in LABEL_NAMES and p.with_suffix(".parquet.digest.json").exists()
)
_sealed_before = {
    n: json.loads((LABELS_DIR / f"{n}.parquet.digest.json").read_text())["digest"]
    for n in SEALED_ELSEWHERE
}
print(f"sealed parquets left untouched by this run: {_sealed_before}")
published = {}
for label_name in LABEL_NAMES:
    columns = ["timestamp", "symbol", label_name]
    if label_name in CLASSIFICATION:
        columns.append(CLASSIFICATION[label_name])
    frame = labels_df.select(columns).drop_nulls(["timestamp", "symbol"])
    published[label_name] = frame
    record = write_artifact(
        frame,
        LABELS_DIR / f"{label_name}.parquet",
        keys=["timestamp", "symbol"],
        written_by="02_labels",
        inputs={"market_data": MARKET_DATA_DIGEST},
    )
    print(
        f"{label_name}.parquet: {record['n_rows']:,} rows "
        f"({frame[label_name].drop_nulls().len():,} non-null, "
        f"{frame[label_name].null_count():,} padded), digest {record['digest']}"
    )

# %% [markdown]
# The four invariants that make this a change of **timeline** and not a change of **label**, all
# asserted rather than printed. The last one is the count the declaration asks to be shown rather
# than left silent: the decision keys that no rule resolves at all.

# %%
for label_name in LABEL_NAMES:
    frame = published[label_name]
    assert frame.height == GRID_KEYS, (
        f"{label_name} was published on {frame.height:,} keys, not the "
        f"{GRID_KEYS:,} keys of the decision grid"
    )
    assert (
        frame[label_name].drop_nulls().len()
        == labels_df[label_name].drop_nulls().len()
    ), f"{label_name} lost or gained a labelled row; only null padding was authorised"
key_sets = {
    n: set(zip(f["timestamp"].to_list(), f["symbol"].to_list(), strict=True))
    for n, f in published.items()
}
reference = key_sets[PRIMARY_LABEL]
for label_name, keys in key_sets.items():
    assert keys == reference, f"{label_name} sits on a different key set from {PRIMARY_LABEL}"
# `classification_eval_label` is a repository-pinned key and keeps its generation-1 entry
# (`dir_tb_8h: fwd_ret_8h`) as inert documentation; the invariant applies to a class label only
# when it is actually published by this run.
for cls, cont in CLASSIFICATION.items():
    if cls in published:
        assert cont in published[cls].columns, f"{cls} must carry {cont} in the same frame"
unlabelled = labels_df.filter(
    pl.all_horizontal([pl.col(n).is_null() for n in LABEL_NAMES])
)
print(
    f"one key set of {GRID_KEYS:,} decision keys under all {len(LABEL_NAMES)} published "
    f"label(s); labelled under at least one rule "
    f"{GRID_KEYS - unlabelled.height:,}; **{unlabelled.height:,} decision keys carry no label "
    f"under any of the {len(LABEL_NAMES)} rule(s)** and are published as null"
)
_sealed_after = {
    n: json.loads((LABELS_DIR / f"{n}.parquet.digest.json").read_text())["digest"]
    for n in SEALED_ELSEWHERE
}
assert _sealed_after == _sealed_before, (
    f"a sealed parquet this notebook does not publish moved: {_sealed_before} -> {_sealed_after}"
)
print(f"sealed parquets untouched, digests reproduced from their sidecars: {_sealed_after}")

# %% [markdown]
# The record Chapter 7.2 asks for to close a label definition, one row per label, built from the
# values computed above rather than typed in.

# %%
anchors = {
    "fwd_ret_sess": "decision bar close (session open + 1h) -> the last bar close at or before that session's close at which a next-bar-open exit fills (London: the close; New York: one bar before the break)",
    "fwd_ret_8h": "decision bar close (session open + 1h) -> that session's close",
    "fwd_ret_24h": f"decision bar close -> the close {step} slots later, only where that is exactly 24h",
    "dir_tb_8h": f"decision bar close -> first touch of +-{TB['upper_atr_multiple']} x D1 ATR({atr_period}), expiry at the session close",
}
print("\nLabel audit record")
for label_name in LABEL_NAMES:
    frame = dev[label_name]
    if frame.is_empty():
        continue
    values = frame[label_name]
    print(
        f"\n{label_name}"
        f"\n  anchor       {anchors[label_name]}"
        f"\n  horizon      {HORIZON_HOURS[label_name]} hours"
        f"\n  resolution   {'first barrier touched, else the timeout class' if label_name in CLASSIFICATION else 'fixed at the endpoint'}"
        f"\n  overlap      {overlap_counts[label_name]} later decision(s) start inside the window"
        f"\n  purge buffer {setup['labels'].get('variant_buffers', {}).get(label_name, setup['labels']['buffer'])}"
        f"\n  rebalance    step {setup['labels']['rebalance_step'][label_name]} slot(s)"
        f"\n  base rate    mean {values.mean():+.6f}, std {values.std():.6f}"
        f"\n  consumed by  05_evaluation.py, and 06_linear onward"
    )

# %% [markdown]
# ## Key takeaways
#
# 1. **Seal a label on the endpoint the strategy actually exits at.** Here that is a session
#    close; that it also happens to be eight bars is a property of the grid, checked, not assumed.
# 2. **A label's name is a claim about every row of it.** Where twenty-four hours is seventy-two,
#    the row is null and counted, not quietly relabelled.
# 3. **Cut a path-dependent label at the boundary the strategy is cut at**, not at a row count in
#    a series that runs through the night.
# 4. **Measure the overlap on the grid, then carry it into every standard error.** Two sleeves a
#    day means consecutive labels share hours, and a naive t-statistic on overlapping labels is
#    the most common way a session bot flatters itself.
# 5. **Two instruments are not a cross-section.** Measure inside each one and pool the per-slot
#    products; do not compute a rank correlation over two points.
#
# **Next**: the feature register of `setup.yaml::features`, built on this development period.
