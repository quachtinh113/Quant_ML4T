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
# # Exness BTC 8h (exness_btc_8h): Feature Evaluation
#
# Every column the last two notebooks produced is tested here, on its own, against the label it was
# built for - on the validation stretches of the four walk-forward folds and on nothing else.
#
# **This is the stage a one-instrument bot cannot inherit**, and the reason deserves to be stated
# before any number. Every information coefficient this repository computes by default is a
# *cross-sectional* one: within each decision instant it ranks the entities by the score, ranks
# them by the outcome, correlates the two rankings and averages through time. That statistic needs
# a panel wide enough to rank. This panel is **one** instrument, and three separate implementations
# return *nothing* on it rather than raising:
#
# | Where | What it does on one name |
# |---|---|
# | `case_studies/utils/feature_engineering.py::quantile_profile` | filters `len().over(date) >= min_cross_section` before scoring, so every instant is removed and every feature comes back `None` |
# | `case_studies/utils/registry/metrics.py` | a hard-coded `min_obs=5`, so every `ic_mean`, `ic_t` and `pct_positive` the registry ever writes for this case study is NULL by construction |
# | `case_studies/exness_fx_d1/05_evaluation.py` | `MIN_PERIODS = max(3, n_symbols // 4)`, which is 3 on one name and still needs three rows in an instant that has one |
#
# A fourth is louder: `feature_engineering.plot_persistence`'s right-hand panel is a
# cross-sectional rank correlation, and on one name it produces an all-NaN vector and then raises
# `Axis limits cannot be NaN or Inf`. `03_financial_features` routes around it and says so.
#
# Lowering a floor rescues none of them: a Spearman correlation over $n = 1$ is undefined. So the
# statistic here is a **time-series** one - the pooled panel IC of `_model_reading.py`, which on one
# name reduces exactly to that instrument's own Spearman rank correlation through time - and it is
# reported with **two** intervals rather than one: a Newey-West standard error at the overlap
# measured off the grid, and a **moving-block bootstrap** that assumes no covariance model at all.
# Where the two disagree the bootstrap is the one to believe, and that disagreement is exactly what
# a single number would hide.
#
# ## Learning objectives
#
# - Recognise a cross-sectional statistic that returns an empty result instead of an error, and
#   replace it with the time-series statistic the design actually asks for
# - Give one estimate two intervals that fail differently, and read them together
# - Screen a column for coverage and staleness **inside each book that would trade it**, not only
#   on the pooled panel
# - Pay for the size of the search with a false-discovery adjustment declared over the whole set
# - End with one decision per column and the evidence attached, and keep that decision a **screen**
#   rather than a selection
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 20, Sections 20.2-20.4. Route B fork of
# [`exness_gold_sess/05_evaluation`](../exness_gold_sess/05_evaluation.ipynb). Reads
# `features/financial.parquet`, `features/model_based.parquet` and `labels/*.parquet`.
#
# **Outputs**: `evaluation/ic_timeseries.parquet` (the per-slot score for every column, by fold)
# and `evaluation/triage_ledger.parquet` (one decision per column with its evidence).

# %%
"""exness_btc_8h: test each feature column on its own against the label it is built for."""

import warnings
from datetime import date, datetime

import numpy as np
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from scipy.stats import spearmanr

from case_studies.exness_btc_8h._model_reading import (
    REGISTRY_IC_IS_NULL_REASON,
    block_bootstrap_ic,
    fold_ic_summary,
    hac_lag_for_label,
    pooled_panel_ic_series,
)
from utils.artifact_specs import resolve_label_buffer
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.data_quality import validate_modeling_inputs
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore", category=FutureWarning)
set_global_seeds(42)

# %% tags=["parameters"]
MAX_SYMBOLS = 0
MAX_FOLDS = 0
N_BOOTSTRAP = 2000

# %% [markdown]
# ## Configuration
#
# Every label, window and boundary is read from `config/setup.yaml`. Five numbers are judgments
# rather than measurements and are settled once, here, so nothing further down decides anything on
# a literal typed beside it:
#
# - **Coverage floor.** How often a column has to hold a value, counted from the first slot it is
#   ever filled in. Below it, whatever the column measures is absent from most of the sample.
# - **Staleness ceiling.** The share of consecutive observations identical to the one before. A
#   column that mostly repeats cannot tell this slot from the last.
# - **Effect-size floor** and **direction agreement**: the two conditions of the exploration route
#   to `PROCEED`, which promotes on steadiness and size rather than on significance.
# - **Bootstrap block**: one week of decision slots. It has to be at least as long as the
#   dependence the grid really has; a week is generous against a one-slot label.

# %%
CASE_STUDY_ID = "exness_btc_8h"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
EVAL_DIR = CASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
evaluation_config = load_evaluation_config(CASE_STUDY_ID)

JOIN_COLS = ["timestamp", "symbol"]
DATE_COL = "timestamp"
LABEL_COL = setup["labels"]["primary"]
LABEL_NAMES = [LABEL_COL, *setup["labels"]["variants"]]
LABEL_BUFFERS = {LABEL_COL: setup["labels"]["buffer"], **setup["labels"]["variant_buffers"]}
HORIZON_HOURS = {n: int(str(setup["labels"]["horizons"][n]).rstrip("Hh")) for n in LABEL_NAMES}
HOLDOUT_START = datetime.combine(
    date.fromisoformat(str(evaluation_config["holdout_start"])), datetime.min.time()
)
REDUNDANCY_CUT = float(setup["features"]["redundancy_cut"])
SESSION_BOOKS = list(setup["decision"]["session_filter_values"])
UNIVERSE = sorted(setup["universe"]["symbols"])
SLOTS_PER_DAY = len(setup["decision"]["snapshots_utc"])

MIN_FOLD_SLOTS = 60
MIN_SLOTS = 200
COVERAGE_FLOOR = 0.70
STALENESS_CEILING = 0.50
IC_THRESHOLD = 0.005
STABILITY_THRESHOLD = 0.60
FDR_ALPHA = 0.05
BOOTSTRAP_BLOCK = 7 * SLOTS_PER_DAY

print(f"Primary label: {LABEL_COL}, the return from one decision instant to the next.")
print(f"Holdout: {HOLDOUT_START.date()} to {evaluation_config['holdout_end']}, not read here.")
print(f"Two columns agreeing above {REDUNDANCY_CUT:.2f} in absolute rank correlation are one piece "
      "of evidence, not two (setup.yaml::features.redundancy_cut).")
print(f"Coverage floor {COVERAGE_FLOOR:.0%}; staleness ceiling {STALENESS_CEILING:.0%}; "
      f"effect-size floor {IC_THRESHOLD}; direction agreement {STABILITY_THRESHOLD:.0%} of folds.")
print(f"Session books screened separately: {SESSION_BOOKS} (the first is the pooled book).")
print(f"Block bootstrap: {N_BOOTSTRAP:,} resamples of {BOOTSTRAP_BLOCK}-slot blocks (one week).")
print(f"\nWhy the registry's own IC column will be empty for this case study:\n  {REGISTRY_IC_IS_NULL_REASON}")

# %% [markdown]
# ## 1. The panel every statistic runs on
#
# Three files are joined on instrument and slot: the price-derived columns, the model-derived
# columns and the label. The model-derived file needs care - its estimators are refitted once per
# fold, so the same slot appears under every fold whose window reaches it. What is kept is, for
# each fold, only that fold's own **validation** slots. The holdout fold's rows are dropped by
# **id**, and then the maximum timestamp is asserted below the holdout boundary as a second,
# independent check.

# %%
financial = pl.read_parquet(CASE_DIR / "features" / "financial.parquet")
model_based = pl.read_parquet(CASE_DIR / "features" / "model_based.parquet")
labels = pl.read_parquet(CASE_DIR / "labels" / f"{LABEL_COL}.parquet")
missing = {*JOIN_COLS, "fold"}.difference(model_based.columns)
if missing:
    raise ValueError(f"the model-based artifact lacks fold provenance: {sorted(missing)}")

LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, LABEL_COL, setup)
timeline = labels.select(pl.col(DATE_COL).dt.date().alias(DATE_COL)).unique().sort(DATE_COL)


def fold_windows(buffer: str) -> list[dict]:
    """The same call 04 made, so a fold id names the same window on both sides of the join."""
    splits = generate_cv_splits(
        timeline, case_study_id=CASE_STUDY_ID, label_buffer=buffer, date_col=DATE_COL
    )
    return splits[:MAX_FOLDS] if MAX_FOLDS > 0 else splits


def validation_rows(splits: list[dict]) -> pl.DataFrame:
    """Each fold's own validation slots, taken from the fold it was fitted out of."""
    stamped = set(model_based["fold"].unique().to_list())
    frames = []
    for split in splits:
        fold = int(split["fold"])
        if fold not in stamped:
            continue
        rows = model_based.filter(
            (pl.col("fold") == fold)
            & pl.col(DATE_COL).dt.date().is_between(
                pl.lit(split["val_start"]).cast(pl.Date),
                pl.lit(split["val_end"]).cast(pl.Date),
                closed="both",
            )
        )
        if not len(rows):
            span = model_based.filter(pl.col("fold") == fold)
            raise ValueError(
                f"fold {fold} is stamped on rows spanning {span[DATE_COL].min()}.."
                f"{span[DATE_COL].max()} but this configuration validates it over "
                f"{split['val_start']}..{split['val_end']}"
            )
        frames.append(rows)
    if not frames:
        raise ValueError(f"no fold in {sorted(stamped)} appears in the configured splits")
    return pl.concat(frames).sort([DATE_COL, "symbol"])


splits = fold_windows(LABEL_BUFFER)
VALIDATION_FOLD_IDS = {int(s["fold"]) for s in splits}
HOLDOUT_FOLD_IDS = set(model_based["fold"].unique().to_list()) - VALIDATION_FOLD_IDS
print(f"validation folds {sorted(VALIDATION_FOLD_IDS)}; folds excluded by id as holdout "
      f"vintages: {sorted(HOLDOUT_FOLD_IDS)}")
validation_temporal = validation_rows(splits)
if len(validation_temporal.group_by(JOIN_COLS).len().filter(pl.col("len") > 1)):
    raise ValueError("the folds' validation windows overlap on slot and instrument")
for fold in sorted(validation_temporal["fold"].unique().to_list()):
    window = validation_temporal.filter(pl.col("fold") == fold)
    print(f"  Fold {fold}: validation {window[DATE_COL].min()} .. {window[DATE_COL].max()} "
          f"({window.height:,} rows)")

# %%
eval_panel = (
    validation_temporal.join(financial, on=JOIN_COLS, how="inner")
    .join(labels, on=JOIN_COLS, how="inner")
    .drop_nulls(LABEL_COL)
    .sort([DATE_COL, "symbol"])
)
dropped = len(validation_temporal) - len(eval_panel)
if dropped:
    print(
        f"{dropped:,} of {len(validation_temporal):,} validation rows have no matching feature row "
        f"or no {LABEL_COL}. On this bot that should be ZERO in the interior of the sample: the "
        "grid has no hole, no holiday and no early close, so the only rows a join can lose are "
        "the ones the feature matrix cut for warmup."
    )
    assert dropped <= 0.02 * len(validation_temporal), "more rows lost than warmup explains"
if MAX_SYMBOLS > 0:
    selected = sorted(eval_panel["symbol"].unique().to_list())[:MAX_SYMBOLS]
    eval_panel = eval_panel.filter(pl.col("symbol").is_in(selected))

financial_cols = [c for c in financial.columns if c not in JOIN_COLS]
temporal_cols = [c for c in model_based.columns if c not in {*JOIN_COLS, "fold"}]
all_feature_cols = financial_cols + temporal_cols

assert eval_panel[DATE_COL].max() < HOLDOUT_START, "the evaluation panel reaches the holdout"
if eval_panel.select(JOIN_COLS).n_unique() != len(eval_panel):
    raise ValueError("the evaluation panel has duplicate slot-instrument rows")
print(f"\nPanel: {len(eval_panel):,} rows over {eval_panel[DATE_COL].n_unique():,} slots and "
      f"{eval_panel['symbol'].n_unique()} instrument, from the validation stretches of "
      f"{eval_panel['fold'].n_unique()} folds")
print(f"Slots run {eval_panel[DATE_COL].min()} to {eval_panel[DATE_COL].max()}")
print(f"Candidates: {len(financial_cols)} price-derived + {len(temporal_cols)} model-derived "
      f"= {len(all_feature_cols)}")

# %% [markdown]
# ### The books, which are slot filters rather than a partition
#
# `setup.yaml::decision.session_filter_values` declares three books and they **overlap**: `all` is
# every slot, `no_swap_night` is the 00:00 and 08:00 decisions, `us_hours` is the 16:00 one. That
# is different from the metals bot, whose two books partition its slots, and it matters for the
# screen below: a book here is a *mask*, so the same slot is screened more than once and a column
# constant inside one of them is not thereby constant inside another.

# %%
BOOK_HOURS = {"all": {0, 8, 16}, "no_swap_night": {0, 8}, "us_hours": {16}}
assert set(BOOK_HOURS) == set(SESSION_BOOKS), (
    f"the books declared in setup.yaml are {SESSION_BOOKS}; this notebook defines {sorted(BOOK_HOURS)}"
)
book_masks = {
    book: eval_panel.filter(pl.col(DATE_COL).dt.hour().is_in(sorted(hours)))
    for book, hours in BOOK_HOURS.items()
}
display(
    pl.DataFrame(
        [
            {
                "book": book,
                "slots": rows.height,
                "share of pooled": round(rows.height / len(eval_panel), 4),
                "hours": ", ".join(str(h) for h in sorted(BOOK_HOURS[book])),
            }
            for book, rows in book_masks.items()
        ]
    )
)

# %%
validate_modeling_inputs(
    features_df=eval_panel,
    label_df=eval_panel,
    feature_cols=all_feature_cols,
    label_col=LABEL_COL,
    join_cols=JOIN_COLS,
    asset_col="symbol",
    max_abs_return=0.5,
    fail_on_critical=True,
)

# %% [markdown]
# ### What the candidates are
#
# A list of seventy column names says little about what is being tested. Grouping them by the idea
# behind each shows how concentrated the search is, which is why the correlations in part 5 matter
# and why the multiplicity adjustment in part 4 is applied over the whole set rather than family by
# family. A column matching no family stops the run rather than landing in "other".

# %%
def assign_feature_family(name: str) -> str:
    """Map a column to the economic or modelling idea it comes from."""
    family_map = [
        (["garch_"], "conditional volatility"),
        (["hmm_"], "volatility regime"),
        (["slot_zscore", "slot_channel", "slot_bollinger"], "slot mean reversion"),
        (["slot_vol", "slot_avg_range", "slot_max_dd"], "slot volatility and range"),
        (["slot_rsi", "slot_price_to_ma"], "slot oscillator and trend"),
        (["slot_ret", "slot_accel", "slot_sharpe"], "slot momentum"),
        (["gold_"], "gold"),
        (["usd_"], "dollar"),
        (["usidx_"], "US index"),
        (["d1_"], "long-window state"),
        (["slot_of_day", "dow", "is_", "pays_"], "swap and calendar state"),
    ]
    lowered = name.lower()
    for prefixes, family in family_map:
        if any(prefix in lowered for prefix in prefixes):
            return family
    return "other"


families = {feature: assign_feature_family(feature) for feature in all_feature_cols}
unfamiliar = sorted(name for name, family in families.items() if family == "other")
if unfamiliar:
    raise ValueError(f"{len(unfamiliar)} columns match no family: {unfamiliar}")
family_table = (
    pl.DataFrame(
        {
            "family": list(families.values()),
            "source": ["model-derived" if f in temporal_cols else "price-derived" for f in families],
            "feature": list(families),
        }
    )
    .group_by(["source", "family"])
    .agg(pl.len().alias("columns"), pl.col("feature").sort().str.join(", ").alias("names"))
    .sort(["source", "columns", "family"], descending=[False, True, False])
)
with pl.Config(tbl_rows=family_table.height, tbl_width_chars=260, fmt_str_lengths=200):
    display(family_table)

# %% [markdown]
# ## 2. Two screens that come before any inference, run per book
#
# Testing a column that is mostly empty, or mostly the same number repeated, produces a p-value
# that means nothing and looks like the others. Both are cheap to detect and both are settled
# before a single correlation is computed.
#
# **The screens are run separately on each book**, and on this bot that is not decoration: three of
# the state columns are *constants* inside a filtered book by construction. `slot_of_day`,
# `is_us_hours` and `pays_swap_night` all take one value inside `us_hours`, and the last two take
# one value inside `no_swap_night` too. A column that is perfectly informative on the pooled panel
# and constant in the book that would trade it is the failure that stopped `exness_fx_d1`'s eight
# `rank_*` columns, found one book too late.
#
# ### Staleness is measured at the column's own grid, and this bot had to fix that
#
# The template measures staleness as the share of **consecutive decision slots** whose value
# repeats. On a one-grid bot that is the right question. On this one it is not, and running it
# unchanged sent **twenty-eight columns to STOP for a reason that was arithmetic rather than
# empirical**: a D1 column is joined onto a grid with three decisions a day, so it *cannot* change
# more than once in three slots and its slot-resolution staleness is exactly 2/3 = 0.667 by
# construction - above a 0.50 ceiling calibrated for a daily bot. The whole long-window state
# family, all three cross-asset families and the daily calendar flags failed on that, before any of
# them had been measured against anything.
#
# So the screen asks the question at the resolution the column has:
#
# | Grid | How staleness is measured | Ceiling |
# |---|---|---|
# | slot (`slot_*`, `garch_*`, `hmm_*`, and the three per-slot cost flags) | consecutive decision slots | 0.50 |
# | daily (`d1_*`, `gold_*`, `usd_*`, `usidx_*`) | consecutive **days**, sampled one slot a day | 0.50 |
# | calendar (`dow`, `is_weekend`, `pays_triple_swap`) | not applicable - a weekday flag repeats by definition; screened on **cardinality** instead, which is the property a regressor needs | at least 2 distinct values |
#
# The grid a column sits on is read from the prefixes `_features.py` builds it with, so the
# classification is the construction rather than a second opinion about it. This is a change to the
# screen and not to any threshold: 0.50 is unchanged everywhere it applies.

# %%
#: Prefixes of the D1-grid families - the ones asof-joined onto the decision grid in
#: `_features.join_d1_asof`, which by construction change at most once a day.
DAILY_GRID_PREFIXES = ("d1_", "gold_", "usd_", "usidx_")
#: Calendar flags whose cadence is a day or a week. A repeat rate is meaningless on them.
CALENDAR_COLUMNS = {"dow", "is_weekend", "pays_triple_swap"}


def column_grid(feature: str) -> str:
    if feature in CALENDAR_COLUMNS:
        return "calendar"
    return "daily" if feature.startswith(DAILY_GRID_PREFIXES) else "slot"


COLUMN_GRID = {f: column_grid(f) for f in all_feature_cols}
print(
    "columns by the grid their staleness is measured on: "
    + ", ".join(
        f"{g} {sum(v == g for v in COLUMN_GRID.values())}" for g in ("slot", "daily", "calendar")
    )
)


def screen(frame: pl.DataFrame) -> tuple[dict, dict, dict]:
    """Coverage, staleness at the column's own grid, and cardinality."""
    coverage, staleness, cardinality = {}, {}, {}
    # One slot a day for the daily-grid columns: the earliest decision of each date, so the
    # comparison is between consecutive days rather than between slots of one day.
    daily_frame = (
        frame.sort(["symbol", DATE_COL])
        .group_by(["symbol", pl.col(DATE_COL).dt.date().alias("_d")], maintain_order=True)
        .agg(pl.all().first())
        .sort(["symbol", "_d"])
    )
    for feature in all_feature_cols:
        non_null = frame.filter(pl.col(feature).is_not_null())
        if len(non_null) == 0:
            coverage[feature] = 0.0
        else:
            eligible = frame.filter(pl.col(DATE_COL) >= non_null[DATE_COL].min())
            coverage[feature] = len(non_null) / len(eligible)
        cardinality[feature] = int(frame[feature].drop_nulls().n_unique())
        daily = COLUMN_GRID[feature] == "daily"
        source = daily_frame if daily else frame
        order = ["symbol", "_d"] if daily else ["symbol", DATE_COL]
        chronological = source.select([*order, feature]).sort(order)
        unchanged = chronological.select(
            (pl.col(feature) == pl.col(feature).shift(1).over("symbol")).sum()
        ).item()
        comparable = chronological.select(
            pl.col(feature).shift(1).over("symbol").is_not_null().sum()
        ).item()
        staleness[feature] = float(unchanged) / max(comparable, 1)
    return coverage, staleness, cardinality


coverage, staleness, cardinality = screen(eval_panel)
_book_screens = {book: screen(rows) for book, rows in book_masks.items()}
per_book = {book: result[1] for book, result in _book_screens.items()}
per_book_cardinality = {book: result[2] for book, result in _book_screens.items()}
correctness = {
    f: (
        coverage[f] >= COVERAGE_FLOOR
        and (
            cardinality[f] >= 2
            if COLUMN_GRID[f] == "calendar"
            else staleness[f] <= STALENESS_CEILING
        )
    )
    for f in all_feature_cols
}
failed = [f for f, ok in correctness.items() if not ok]
print(f"{len(correctness) - len(failed)} of {len(correctness)} columns are filled on at least "
      f"{COVERAGE_FLOOR:.0%} of their slots and repeat on at most {STALENESS_CEILING:.0%} of them")
if failed:
    print(f"  failing the pooled screen: {failed}")
screen_table = pl.DataFrame(
    {
        "feature": all_feature_cols,
        "family": [families[f] for f in all_feature_cols],
        "grid": [COLUMN_GRID[f] for f in all_feature_cols],
        "cardinality": [cardinality[f] for f in all_feature_cols],
        "coverage": [round(coverage[f], 4) for f in all_feature_cols],
        "staleness_pooled": [round(staleness[f], 4) for f in all_feature_cols],
        **{
            f"staleness_{book}": [round(per_book[book][f], 4) for f in all_feature_cols]
            for book in SESSION_BOOKS
        },
        "passes_pooled": [correctness[f] for f in all_feature_cols],
    }
).with_columns(
    pl.max_horizontal([pl.col(f"staleness_{b}") for b in SESSION_BOOKS]).alias("staleness_worst_book")
).sort("staleness_worst_book", descending=True)
with pl.Config(tbl_rows=screen_table.height, tbl_cols=12, tbl_width_chars=220):
    display(screen_table)
constant_in_a_book = screen_table.filter(
    (pl.col("grid") != "calendar") & (pl.col("staleness_worst_book") > STALENESS_CEILING)
)
print(f"{constant_in_a_book.height} columns exceed the staleness ceiling inside at least one book: "
      f"{constant_in_a_book['feature'].to_list()}")
_one_valued = [
    f for f in all_feature_cols if min(per_book_cardinality[b][f] for b in SESSION_BOOKS) < 2
]
print(f"{len(_one_valued)} columns take ONE value inside at least one book: {_one_valued}")
print(
    "Those are the columns a SESSION_FILTER would turn into constants. They are not dropped here - "
    "they are informative on the pooled book - but a phase-5 spec that filters to one book and "
    "keeps them is fitting on a constant, and the ledger says so."
)

# %% [markdown]
# ## 3. The time-series IC, with two intervals
#
# The measurement is one number per **slot**: the column and the label are replaced by their ranks
# over the panel, mapped to zero mean and unit variance, and multiplied. The mean of that series is
# the statistic, and on one instrument it *is* that instrument's Spearman rank correlation - which
# is checked against `scipy` below rather than asserted.
#
# Two intervals, because they fail differently:
#
# - **Newey-West**, at a lag measured off the grid rather than assumed. The measured overlap for
#   the primary label is **zero** later decisions starting inside a holding window - the next
#   decision begins exactly when the hold ends - which is the property `labels.rebalance_step: 1`
#   claims and `02_labels` also measured. That measured value is a floor, not the lag applied:
#   `compute_ic_hac_stats` takes `max(label_horizon - 1, Newey-West auto)`, and on several thousand
#   slots the automatic bandwidth is what gets used. The applied lag is reported from the result.
# - **A moving-block bootstrap** over one-week blocks, which assumes no covariance structure at
#   all. Where the two disagree, the bootstrap is the honest one.

# %%
_grid = eval_panel.filter(pl.col("symbol") == UNIVERSE[0])[DATE_COL].unique().sort()
_overlap, HAC_LAG = hac_lag_for_label(_grid, HORIZON_HOURS[LABEL_COL])
print(
    f"{HORIZON_HOURS[LABEL_COL]}-hour label on this grid: at most {_overlap} later decision(s) "
    f"start inside a holding window, so the Newey-West floor is lag {HAC_LAG}"
)

# The claim that the pooled panel IC reduces to Spearman, checked rather than asserted. On one
# symbol the mean of the per-slot product is the Spearman correlation up to the (1 - 1/n^2)
# rank-variance factor.
_one = eval_panel.select(DATE_COL, "symbol", "d1_vol_gk_63d", LABEL_COL)
_ref = float(spearmanr(_one["d1_vol_gk_63d"].to_numpy(), _one[LABEL_COL].to_numpy())[0])
_ours = float(pooled_panel_ic_series(
    _one, score_col="d1_vol_gk_63d", outcome_col=LABEL_COL, fold_col=None
)["ic"].mean())
_gap = abs(_ours - _ref * (1 - 1 / len(_one) ** 2))
assert _gap < 1e-7, (_ours, _ref, _gap)
print(
    f"on one instrument the pooled panel IC reproduces scipy's Spearman: {_ours:+.8f} vs "
    f"{_ref:+.8f} x (1 - 1/n^2) over {len(_one):,} slots, residual {_gap:.2e}"
)

# %%
evaluable = [f for f in all_feature_cols if correctness[f]]
computed = []
for feature in evaluable:
    series = pooled_panel_ic_series(eval_panel, score_col=feature, outcome_col=LABEL_COL)
    if len(series) >= MIN_SLOTS:
        computed.append(series.with_columns(pl.lit(feature).alias("feature")))
IC_SERIES_SCHEMA = {
    "feature": pl.String,
    DATE_COL: pl.Datetime("us"),
    "fold": pl.Int64,
    "ic": pl.Float64,
    "n_obs": pl.Int64,
}
ic_series_frame = (
    pl.concat(computed).select(*IC_SERIES_SCHEMA).cast(IC_SERIES_SCHEMA)
    if computed
    else pl.DataFrame(schema=IC_SERIES_SCHEMA)
)
ic_series_frame.write_parquet(EVAL_DIR / "ic_timeseries.parquet")
stored_ic = pl.read_parquet(EVAL_DIR / "ic_timeseries.parquet")
ic_timeseries = {
    part["feature"][0]: part.drop("feature").sort(DATE_COL)
    for part in stored_ic.partition_by("feature")
}
ic_results = {
    feature: compute_ic_hac_stats(series, ic_col="ic", label_horizon=HAC_LAG)
    for feature, series in ic_timeseries.items()
}
boot_results = {
    feature: block_bootstrap_ic(
        series, block=BOOTSTRAP_BLOCK, n_resamples=N_BOOTSTRAP, seed=42
    )
    for feature, series in ic_timeseries.items()
}
fold_stats = {
    feature: fold_ic_summary(series, min_fold_slots=MIN_FOLD_SLOTS)
    for feature, series in ic_timeseries.items()
}
print(f"Wrote and read back evaluation/ic_timeseries.parquet: {len(stored_ic):,} scored slots "
      f"over {len(ic_results)} of {len(evaluable)} eligible columns")
_applied = {int(r["effective_lags"]) for r in ic_results.values()}
print(f"Newey-West lag actually applied: {sorted(_applied)} (the measured floor was {HAC_LAG})")

# %% [markdown]
# ### Do the two intervals agree?
#
# One estimate, two ways of saying how uncertain it is. The table below counts the columns each
# route calls distinguishable from zero and, more usefully, the ones where they **disagree** -
# because a column significant under Newey-West and not under the bootstrap is a column whose
# result depends on the covariance model rather than on the data.

# %%
agreement_rows = []
for feature in ic_results:
    hac_sig = float(ic_results[feature]["p_value"]) < 0.05
    boot = boot_results[feature]
    boot_sig = bool(boot.get("n_resamples")) and not (
        boot["boot_ci_lo"] <= 0.0 <= boot["boot_ci_hi"]
    )
    agreement_rows.append(
        {
            "feature": feature,
            "ic_mean": ic_results[feature]["mean_ic"],
            "hac_p": ic_results[feature]["p_value"],
            "hac_sig": hac_sig,
            "boot_ci_lo": boot.get("boot_ci_lo"),
            "boot_ci_hi": boot.get("boot_ci_hi"),
            "boot_sig": boot_sig,
            "agree": hac_sig == boot_sig,
        }
    )
agreement = pl.DataFrame(agreement_rows).sort(pl.col("ic_mean").abs(), descending=True)
print(
    f"significant at 5% under Newey-West: {int(agreement['hac_sig'].sum())}; "
    f"under the block bootstrap: {int(agreement['boot_sig'].sum())}; "
    f"the two disagree on {int((~agreement['agree']).sum())} of {agreement.height} columns"
)
with pl.Config(tbl_rows=20, tbl_width_chars=200):
    display(agreement.head(20))
_disagree = agreement.filter(~pl.col("agree"))
if _disagree.height:
    print("columns where the two intervals disagree - read the bootstrap:")
    with pl.Config(tbl_rows=_disagree.height, tbl_width_chars=200):
        display(_disagree)

# %% [markdown]
# ### Did the folds agree, or was it one window?
#
# An average over four validation years can come from four years of the same weak effect or from
# one year of a strong one, and only the second is a reason to be sceptical. The same average is
# taken inside each fold and two summaries are kept: the share of folds pointing the **column's
# own** way - so a steadily inverted column is not disqualified for being inverted - and the
# **ICIR**, the mean fold IC divided by its spread across folds.

# %%
fold_table = pl.DataFrame(
    [
        {
            "feature": feature,
            "family": families[feature],
            "ic_mean": ic_results[feature]["mean_ic"],
            "n_folds": stats.get("n_folds"),
            "sign_consistency": stats.get("sign_consistency"),
            "positive_fold_share": stats.get("positive_fold_share"),
            "icir": stats.get("icir"),
            "worst_fold_ic": stats.get("worst_fold_ic"),
            "best_fold_ic": stats.get("best_fold_ic"),
        }
        for feature, stats in fold_stats.items()
    ]
).sort(pl.col("ic_mean").abs(), descending=True)
with pl.Config(tbl_rows=25, tbl_width_chars=200):
    display(fold_table.head(25))
_all_four = fold_table.filter(pl.col("sign_consistency") >= 1.0)
print(
    f"{_all_four.height} of {fold_table.height} columns point the same way in all "
    f"{len(VALIDATION_FOLD_IDS)} validation folds"
)

# %% [markdown]
# ## 4. What was searched, and what that costs
#
# Test seventy columns at the conventional five-percent level against a label none of them predicts
# and about three will still come back significant, because that is what the five-percent level
# means. A p-value is only readable against the set of tests it came out of, so that set is
# declared first and **Benjamini-Hochberg** is applied over the whole of it.
#
# Nothing here was chosen after a score was seen. The price-derived columns come from the window
# register in `setup.yaml`, which fixed every window before any of them was built; the
# model-derived columns are whatever the two estimators of `04` produce.

# %%
searched_set = {
    "columns the feature stages generated": len(all_feature_cols),
    "of those, filled in and moving enough to test": sum(correctness.values()),
    "of those, scored on the validation slots": len(ic_results),
    "label horizons the case study declares": len(LABEL_NAMES),
    "session books the sweep will run": len(SESSION_BOOKS),
}
for description, count in searched_set.items():
    print(f"{description}: {count}")

# %%
feature_names = list(ic_results)
hac_p_values = [
    value if np.isfinite(value := ic_results[f]["p_value"]) else 1.0 for f in feature_names
]
fdr_result = (
    benjamini_hochberg_fdr(hac_p_values, alpha=FDR_ALPHA, return_details=True)
    if feature_names
    else {"adjusted_p_values": [], "rejected": [], "n_rejected": 0}
)
eval_summary = pl.DataFrame(
    {
        "feature": feature_names,
        "family": [families[f] for f in feature_names],
        "source": ["model_based" if f in temporal_cols else "financial" for f in feature_names],
        "ic_mean": [ic_results[f]["mean_ic"] for f in feature_names],
        "naive_t": [ic_results[f]["naive_t_stat"] for f in feature_names],
        "hac_t": [ic_results[f]["t_stat"] for f in feature_names],
        "hac_lags": [int(ic_results[f]["effective_lags"]) for f in feature_names],
        "hac_p": hac_p_values,
        "fdr_p": list(fdr_result["adjusted_p_values"]),
        "fdr_sig": list(fdr_result["rejected"]),
        "boot_ci_lo": [boot_results[f].get("boot_ci_lo") for f in feature_names],
        "boot_ci_hi": [boot_results[f].get("boot_ci_hi") for f in feature_names],
        "icir": [fold_stats.get(f, {}).get("icir") for f in feature_names],
        "sign_consistency": [fold_stats.get(f, {}).get("sign_consistency") for f in feature_names],
    }
).sort(pl.col("ic_mean").abs(), descending=True)
n_naive = sum(abs(ic_results[f]["naive_t_stat"]) > 1.96 for f in feature_names)
n_hac = sum(p < 0.05 for p in hac_p_values)
n_fdr = int(fdr_result["n_rejected"])
print(f"Columns clearing 5% treating each slot as independent: {n_naive}")
print(f"Columns clearing 5% once Newey-West rescales the interval: {n_hac}")
print(f"Columns clearing 5% after Benjamini-Hochberg over the set: {n_fdr}")
with pl.Config(tbl_rows=eval_summary.height, tbl_cols=16, tbl_width_chars=240):
    display(eval_summary)

# %% [markdown]
# **The number that puts all of this in scale.** `01_feasibility_analysis` computed IC\*, the
# information coefficient a signal has to beat before it pays for the spread alone - about 0.092
# against the in-sample spread and 0.095 once one night of financing is added, measured there on
# the whole development window. The cell below recomputes it on the VALIDATION slots this notebook
# actually scored, where the label's dispersion is smaller and the bar is therefore higher.
# Nothing in the table above is within an order of magnitude of either figure. That is not a reason to stop: an IC measured on a
# single column is not the IC of a model that combines seventy, and the backtest in phase 5 charges
# costs on the trades it actually places rather than on every slot. But it is the number every
# result from here on has to be read against, and it is printed here rather than in a footnote.

# %%
IC_STAR_IN_SAMPLE = float(setup["costs"]["spread_bps_in_sample"]["p90_bps"]) * 1e-4
_sigma = float(eval_panel[LABEL_COL].std())
_ic_star_spread = IC_STAR_IN_SAMPLE / _sigma
_swap = abs(float(setup["costs"]["swap"]["derived_bps_per_night"]["long"])) * 1e-4 / SLOTS_PER_DAY
_ic_star_total = (IC_STAR_IN_SAMPLE + _swap) / _sigma
_best = float(eval_summary["ic_mean"].abs().max())
print(
    f"IC* on the validation slots: {_ic_star_spread:.4f} against the one-way in-sample spread, "
    f"{_ic_star_total:.4f} once a night of financing is amortised over the three slots of a day"
)
print(
    f"the largest |IC| in the table above is {_best:.4f} = "
    f"{_best / _ic_star_total:.2f}x IC*, on {eval_summary.filter(pl.col('ic_mean').abs() == _best)['feature'][0]}"
)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ### The same columns against the longer label
#
# Everything above is measured against the next decision. The case study declares one more label,
# and how long a column stays informative decides how long a position built on it can be held.

# %%
horizon_rows = []
for label_name in LABEL_NAMES:
    frame = pl.read_parquet(CASE_DIR / "labels" / f"{label_name}.parquet")
    panel = (
        validation_rows(fold_windows(LABEL_BUFFERS[label_name]))
        .join(financial, on=JOIN_COLS, how="inner")
        .join(frame, on=JOIN_COLS, how="inner")
        .drop_nulls(label_name)
        .sort([DATE_COL, "symbol"])
    )
    assert panel[DATE_COL].max() < HOLDOUT_START, f"{label_name} panel reaches the holdout"
    _, lag = hac_lag_for_label(panel[DATE_COL].unique().sort(), HORIZON_HOURS[label_name])
    for feature in evaluable:
        if feature in temporal_cols:
            continue  # the model-based columns are stamped per fold on the primary ladder
        series = pooled_panel_ic_series(panel, score_col=feature, outcome_col=label_name)
        if len(series) < MIN_SLOTS:
            continue
        stats = fold_ic_summary(series, min_fold_slots=MIN_FOLD_SLOTS)
        horizon_rows.append(
            {
                "feature": feature,
                "label": label_name,
                "n_slots": len(series),
                "hac_lag": lag,
                "ic_mean": float(series["ic"].mean()),
                "icir": stats.get("icir"),
            }
        )
horizon_ic = pl.DataFrame(horizon_rows)
_wide = horizon_ic.pivot(index="feature", on="label", values="ic_mean").sort(
    pl.col(LABEL_COL).abs(), descending=True
)
with pl.Config(tbl_rows=25, tbl_width_chars=200):
    display(_wide.head(25))
print(
    f"Horizon profile computed for {len(horizon_rows)} feature-label pairs. Sample sizes: "
    + ", ".join(
        f"{n}: {horizon_ic.filter(pl.col('label') == n)['n_slots'].max():,} slots"
        for n in LABEL_NAMES
    )
)

# %% [markdown]
# ## 5. How much of this is the same evidence counted twice?
#
# Two columns can agree so closely on how they order the slots that a model gains nothing from
# having both - and the multiplicity adjustment above has already paid for testing both as if they
# were separate questions. Columns linked by a pair above the declared cut form a group, following
# the links transitively, and one column stands for each group by the largest median fold agreement
# and, where those tie, the steadiest across folds.
#
# Standing for a group promotes nothing: that column still has to earn its own decision below. The
# grouping is reported and used to drop nothing, because chaining at one threshold can put a column
# in the same group as something it has almost nothing in common with - `weakest_link` is how far
# that went.

# %%
correlation = eval_panel.select(evaluable).to_pandas().corr(method="spearman")
pairs = [
    {
        "left": str(correlation.columns[i]),
        "right": str(correlation.columns[j]),
        "correlation": float(correlation.iloc[i, j]),
    }
    for i in range(len(correlation))
    for j in range(i + 1, len(correlation))
    if np.isfinite(correlation.iloc[i, j]) and abs(correlation.iloc[i, j]) > REDUNDANCY_CUT
]
pairs.sort(key=lambda row: abs(row["correlation"]), reverse=True)
component_of = {f: f for f in evaluable}


def root(feature: str) -> str:
    while component_of[feature] != feature:
        component_of[feature] = component_of[component_of[feature]]
        feature = component_of[feature]
    return feature


for pair in pairs:
    left, right = root(pair["left"]), root(pair["right"])
    if left != right:
        component_of[left] = right
clusters: dict[str, list[str]] = {}
for feature in evaluable:
    clusters.setdefault(root(feature), []).append(feature)


def cluster_key(feature: str) -> tuple[float, float]:
    stats = fold_stats.get(feature)
    if not stats or not stats.get("fold_ics"):
        return (-1.0, 0.0)
    return (abs(stats["median_fold_ic"]), -float(np.std(stats["fold_ics"], ddof=0)))


representative = {}
for members in clusters.values():
    chosen = max(members, key=cluster_key)
    for member in members:
        representative[member] = chosen
redundant = {name: members for name, members in clusters.items() if len(members) > 1}
print(f"{len(pairs)} column pairs agree above {REDUNDANCY_CUT:.2f} in absolute rank correlation, "
      f"out of {len(evaluable) * (len(evaluable) - 1) // 2} compared")
print(f"{len(evaluable)} scored columns collapse to {len(clusters)} groups, of which "
      f"{len(redundant)} hold more than one column")
group_rows = [
    {
        "stands_for": representative[members[0]],
        "columns_in_group": len(members),
        "members": ", ".join(sorted(members)),
    }
    for members in redundant.values()
]
if group_rows:
    with pl.Config(tbl_rows=len(group_rows), tbl_width_chars=240, fmt_str_lengths=200):
        display(pl.DataFrame(group_rows).sort("columns_in_group", descending=True))
for pair in pairs[:15]:
    print(f"  {pair['left']} / {pair['right']}: {pair['correlation']:+.3f}")

# %% [markdown]
# ## 6. One decision per column
#
# The output of this stage is not a shortlist. It is a decision per column with the evidence
# attached, in the book's three categories - and Chapter 20's warning applies in full: **surviving
# this triage is a filter, not evidence of a strategy.** Nothing here is selection. Selection
# happens on the validation backtest in phase 5, on returns net of the measured spread and swap,
# and a column that reaches phase 5 has been screened rather than confirmed.
#
# - `PROCEED`: worth carrying into multivariate work. It says nothing about whether the column ends
#   up in a trained model.
# - `REVISE`: nothing usable was found by this test, and this test is narrow - a column that
#   matters only through an interaction, or past a threshold, is invisible to a rank correlation.
# - `STOP`: the column failed a screen in part 2. Nothing was measured on it because nothing
#   measured on it would mean anything.
#
# Two routes lead to `PROCEED` and the ledger records which one a column took. The first is a
# confirmation: it cleared the false-discovery adjustment over the declared search **and** the
# block bootstrap agreed. The second is a **search** in the sense of Section 7.4 - it promotes on
# agreeing in most folds and on exceeding the effect-size floor - and exists so the next stage is
# not left with nothing to model. A column promoted that way has been confirmed by nothing, and
# `note` is what lets a reader tell the two apart.

# %%
fdr_significant = set(eval_summary.filter(pl.col("fdr_sig"))["feature"].to_list())
bootstrap_significant = set(agreement.filter(pl.col("boot_sig"))["feature"].to_list())
triage = {}
for feature in all_feature_cols:
    if not correctness[feature]:
        triage[feature] = ("STOP", "correctness_fail")
    elif feature not in ic_results:
        triage[feature] = ("REVISE", "insufficient_validation_data")
    elif feature in fdr_significant and feature in bootstrap_significant:
        triage[feature] = ("PROCEED", "fdr_significant_and_bootstrap_agrees")
    elif feature in fdr_significant:
        triage[feature] = ("REVISE", "fdr_significant_but_bootstrap_disagrees")
    elif (
        fold_stats.get(feature, {}).get("sign_consistency", 0) >= STABILITY_THRESHOLD
        and abs(ic_results[feature]["mean_ic"]) >= IC_THRESHOLD
    ):
        triage[feature] = ("PROCEED", "stable_and_above_threshold")
    else:
        triage[feature] = ("REVISE", "weak_standalone_association")

ledger_rows = []
for feature in all_feature_cols:
    decision, note = triage[feature]
    match = eval_summary.filter(pl.col("feature") == feature)
    boot = boot_results.get(feature, {})
    ledger_rows.append(
        {
            "feature": feature,
            "family": families[feature],
            "source": "model_based" if feature in temporal_cols else "financial",
            "statistic": "time_series_pooled_ic",
            "hac_lag_floor": HAC_LAG,
            "hac_lag_applied": int(ic_results[feature]["effective_lags"])
            if feature in ic_results
            else None,
            "ic_mean": ic_results.get(feature, {}).get("mean_ic"),
            "hac_t": ic_results.get(feature, {}).get("t_stat"),
            "hac_p": ic_results.get(feature, {}).get("p_value"),
            "fdr_p": match["fdr_p"][0] if len(match) else None,
            "fdr_sig": bool(match["fdr_sig"][0]) if len(match) else False,
            "boot_block": boot.get("block"),
            "boot_ci_lo": boot.get("boot_ci_lo"),
            "boot_ci_hi": boot.get("boot_ci_hi"),
            "boot_sig": feature in bootstrap_significant,
            "icir": fold_stats.get(feature, {}).get("icir"),
            "positive_fold_share": fold_stats.get(feature, {}).get("positive_fold_share"),
            "sign_consistency": fold_stats.get(feature, {}).get("sign_consistency"),
            "n_folds": fold_stats.get(feature, {}).get("n_folds"),
            "worst_fold_ic": fold_stats.get(feature, {}).get("worst_fold_ic"),
            "coverage": coverage[feature],
            "staleness": staleness[feature],
            "grid": COLUMN_GRID[feature],
            "cardinality": cardinality[feature],
            "staleness_worst_book": max(per_book[b][feature] for b in SESSION_BOOKS),
            "constant_in_a_book": min(per_book_cardinality[b][feature] for b in SESSION_BOOKS) < 2,
            "stands_for_group": representative.get(feature),
            "decision": decision,
            "note": note,
        }
    )
triage_ledger = pl.DataFrame(ledger_rows).sort(["decision", "feature"])
triage_ledger.write_parquet(EVAL_DIR / "triage_ledger.parquet")
print(f"Wrote evaluation/triage_ledger.parquet: one row for each of {len(triage_ledger)} columns")
decision_table = (
    triage_ledger.group_by(["decision", "note"]).agg(pl.len().alias("columns")).sort(["decision", "note"])
)
with pl.Config(tbl_rows=decision_table.height):
    display(decision_table)
with pl.Config(tbl_rows=triage_ledger.height, tbl_cols=20, tbl_width_chars=260):
    display(
        triage_ledger.select(
            "feature", "family", "ic_mean", "hac_t", "hac_p", "fdr_sig", "boot_sig", "icir",
            "sign_consistency", "constant_in_a_book", "decision", "note"
        )
    )

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## Key takeaways
#
# 1. **Pick the statistic the universe can carry, and notice when the default returns nothing.**
#    Three implementations in this repository return an empty result on a one-name panel and a
#    fourth raises on an axis limit. None of them says "this panel is too narrow"; the first
#    responsibility of this stage was to find that out.
# 2. **Give one estimate two intervals that fail differently.** Newey-West rescales under an
#    assumed covariance structure; a moving-block bootstrap assumes none. Where they disagree the
#    result depends on the model rather than on the data, and the ledger records which columns
#    those are.
# 3. **Screen per book, not only pooled.** Three of the state columns are constants inside a
#    filtered book by construction, and a phase-5 spec that filters and keeps them is fitting on a
#    constant.
# 4. **Print IC\* beside the ICs.** A table of correlations means nothing until it is read against
#    the correlation a signal needs to pay its own spread and financing, and on this instrument
#    that number is an order of magnitude above anything measured here.
# 5. **A triage is a screen, not a selection.** Nothing in this notebook chooses a model, and no
#    column reaches phase 5 confirmed - only unrefuted.
#
# ### Known limitations
#
# - Every IC here is univariate. A column that matters only in interaction with another is
#   invisible to a rank correlation, and `REVISE` says exactly that and nothing more.
# - The model-based columns are scored against the primary label only: they are stamped per fold on
#   the primary label's ladder, so scoring them against the 24-hour label would read a vintage from
#   a fold whose geometry that label does not share.
# - The whole sample is one monetary regime for this instrument as much as it is seven years, so a
#   column correlated with the trend will look informative here and add nothing to a book that is
#   measured against buy-and-hold.
#
# **Next**: phase 4, the models - which is where this bot stops until its Hypothesis and Kill
# criteria are written and approved. No trial may be counted before there is a falsification line
# to count it against.
