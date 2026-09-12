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
# # Exness Gold Sessions (exness_gold_sess): Feature Evaluation
#
# The two feature stages built a matrix of candidate columns for the two metals of the bot
# `exness_gold_sess` (`bots/exness_gold_sess/BOT.md`). This notebook asks the one question that has
# to be answered before any of them reaches a model: taken on its own, does a column say anything
# about where that metal's session goes next?
#
# **The statistic is not the template's.** `exness_fx_d1` and `fx_pairs` rank five and twenty names
# against each other on each decision and take the rank correlation with the next return. Over
# **two** names a Spearman correlation takes three values - `+1`, `-1` and undefined - so it would
# measure a coin flip, and the two `gsr` columns take the *same* value on both metals on every
# slot, so their cross-sectional variance is exactly zero and the correlation is not merely noisy
# but undefined. What is used instead is the **pooled panel IC**: within each metal the column and
# the label are replaced by their ranks over the panel, mapped to zero mean and unit variance; the
# product is averaged across the metals present on each slot; and the mean of that per-slot series
# is the statistic, with a Newey-West standard error at the overlap the grid actually has. On one
# symbol it reduces to that symbol's own Spearman correlation, which is asserted below rather than
# claimed. It is the construction `02_labels` already used for its baseline and the one
# `case_studies/xau_fx_mt5_d1/05_evaluation.py` uses on three names.
#
# Each column is tested alone, on the same walk-forward folds `04_model_based_features` fitted its
# models on, and only on each fold's own **validation** slots - the window its estimator was not
# fitted on. The final twelve months from 2025-09-01 are the holdout: the files loaded below run
# through it, the panel every statistic here is computed on stops before it, and the holdout fold
# `04` wrote is excluded by id rather than by date so a boundary error cannot let it in.
#
# **Learning objectives**
# - Choose a statistic the universe can carry, and show it reduces to the familiar one where the
#   familiar one is defined
# - Screen for coverage and staleness **per session book**, because a column can be informative
#   pooled and constant inside the book that would actually trade it
# - Put an interval around an average of overlapping observations, and adjust for the number of
#   columns tested at once
# - Record one decision per column with the evidence behind it, and remember that surviving a
#   triage is a filter and not a finding (Chapter 20)
#
# **Book reference**: Chapter 7, Section 7.3 (univariate feature-label evaluation) and Section 7.4
# (search accounting and multiple testing)
#
# **Outputs**: `evaluation/ic_timeseries.parquet` (the per-slot score for every column, by fold)
# and `evaluation/triage_ledger.parquet` (one decision per column with its evidence).

# %%
"""exness_gold_sess: test each feature column on its own against the label it is built for."""

import warnings
from datetime import date, datetime

import numpy as np
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from scipy.stats import spearmanr

from utils.artifact_specs import resolve_label_buffer
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.data_quality import validate_modeling_inputs
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore", category=FutureWarning)

# %% tags=["parameters"]
MAX_SYMBOLS = 0
MAX_FOLDS = 0

# %% [markdown]
# ## Configuration
#
# Every label, window and boundary is read from `config/setup.yaml`. Four numbers are judgments
# rather than measurements and are settled once, here, so that nothing further down decides
# anything on a literal typed beside it:
#
# - **Coverage floor.** How often a column has to hold a value, counted from the first slot it is
#   ever filled in. Below it, whatever the column measures is absent from most of the sample.
# - **Staleness ceiling.** The share of a metal's consecutive observations identical to the one
#   before. A column that mostly repeats cannot tell this slot from the last.
# - **Effect-size floor** and **direction agreement**: the two conditions of the exploration route
#   to `PROCEED`, which promotes on steadiness and size rather than on significance.

# %%
CASE_STUDY_ID = "exness_gold_sess"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
EVAL_DIR = CASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
evaluation_config = load_evaluation_config(CASE_STUDY_ID)

JOIN_COLS = ["timestamp", "symbol"]
DATE_COL = "timestamp"
LABEL_COL = setup["labels"]["primary"]
LABEL_NAMES = [LABEL_COL, *setup["labels"]["variants"]]
CLASSIFICATION = dict(setup["labels"].get("classification_eval_label") or {})
LABEL_BUFFERS = {LABEL_COL: setup["labels"]["buffer"], **setup["labels"]["variant_buffers"]}
HOLDOUT_START = datetime.combine(
    date.fromisoformat(str(evaluation_config["holdout_start"])), datetime.min.time()
)
REDUNDANCY_CUT = float(setup["features"]["redundancy_cut"])
SESSION_BOOKS = list(setup["decision"]["session_filter_values"])
UNIVERSE = sorted(setup["universe"]["symbols"])

MIN_FOLD_SLOTS = 20
MIN_SLOTS = 100
MARKET_LEVEL_SHARE = 0.90
COVERAGE_FLOOR = 0.70
STALENESS_CEILING = 0.50
IC_THRESHOLD = 0.005
STABILITY_THRESHOLD = 0.60
FDR_ALPHA = 0.05

print(f"Primary label: {LABEL_COL}, the return from the decision to the session close.")
print(f"Holdout: {HOLDOUT_START.date()} to {evaluation_config['holdout_end']}, not read here.")
print(f"Two columns agreeing above {REDUNDANCY_CUT:.2f} in absolute rank correlation are one "
      "piece of evidence, not two (setup.yaml::features.redundancy_cut).")
print(f"Coverage floor {COVERAGE_FLOOR:.0%}; staleness ceiling {STALENESS_CEILING:.0%}; "
      f"effect-size floor {IC_THRESHOLD}; direction agreement {STABILITY_THRESHOLD:.0%} of folds.")
print(f"Session books screened separately: {SESSION_BOOKS} plus the pooled book.")

# %% [markdown]
# ## 1. The panel every statistic runs on
#
# Three files are joined on metal and slot: the price-derived columns, the model-derived columns,
# and the label. The model-derived file needs care - its estimators are refitted once per fold, so
# the same slot appears under every fold whose window reaches it. What is kept is, for each fold,
# only that fold's own **validation** slots. The holdout fold's rows are dropped by **id**, and
# then the maximum timestamp is asserted below the holdout boundary as a second, independent check.

# %%
financial = pl.read_parquet(CASE_DIR / "features" / "financial.parquet")
model_based = pl.read_parquet(CASE_DIR / "features" / "model_based.parquet")
labels = pl.read_parquet(CASE_DIR / "labels" / f"{LABEL_COL}.parquet")
missing = {*JOIN_COLS, "fold"}.difference(model_based.columns)
if missing:
    raise ValueError(f"the model-based artifact lacks fold provenance: {sorted(missing)}")

LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, LABEL_COL, setup)
timeline = (
    labels.select(pl.col(DATE_COL).dt.date().alias(DATE_COL)).unique().sort(DATE_COL)
)


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
        val_start = pl.lit(split["val_start"]).cast(pl.Date)
        val_end = pl.lit(split["val_end"]).cast(pl.Date)
        rows = model_based.filter(
            (pl.col("fold") == fold)
            & pl.col(DATE_COL).dt.date().is_between(val_start, val_end, closed="both")
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
    raise ValueError("the folds' validation windows overlap on slot and metal")
for fold in sorted(validation_temporal["fold"].unique().to_list()):
    window = validation_temporal.filter(pl.col("fold") == fold)
    print(f"  Fold {fold}: validation {window[DATE_COL].min()} .. {window[DATE_COL].max()} "
          f"({window.height:,} rows)")

# %%
# Since A-prime (`bots/exness_gold_sess/FOLD_GEOMETRY_DECLARATION.md`, 2026-09-08) every label is
# published on the whole decision grid and carries `null` where its own rule leaves it undefined,
# so the join no longer removes an unlabelled session - `drop_nulls` does, explicitly and with a
# count. The rows removed are the same rows as before the change; what moved is where they are
# removed, from an implicit inner join to a named filter.
eval_panel = (
    validation_temporal.join(financial, on=JOIN_COLS, how="inner")
    .join(labels, on=JOIN_COLS, how="inner")
    .drop_nulls(LABEL_COL)
    .sort([DATE_COL, "symbol"])
)
dropped = len(validation_temporal) - len(eval_panel)
if dropped:
    print(f"{dropped:,} of {len(validation_temporal):,} validation rows have no matching feature "
          f"row or no {LABEL_COL}. On this bot that is expected and bounded: a session the broker "
          f"closed early keeps its decision and carries no {LABEL_COL}, so it has features and a "
          "null label.")
    assert dropped <= 0.03 * len(validation_temporal), "more rows lost than early closes explain"
if MAX_SYMBOLS > 0:
    selected = sorted(eval_panel["symbol"].unique().to_list())[:MAX_SYMBOLS]
    eval_panel = eval_panel.filter(pl.col("symbol").is_in(selected))

# the session each slot belongs to, so the screens can be run per book
from case_studies.exness_gold_sess._features import session_of  # noqa: E402

_slots = eval_panel.select(DATE_COL).unique().sort(DATE_COL)
_books = _slots.with_columns(session_of(_slots[DATE_COL]).alias("session"))
eval_panel = eval_panel.join(_books, on=DATE_COL, how="left")
assert eval_panel["session"].null_count() == 0, "a validation slot is not a declared decision"

financial_cols = [c for c in financial.columns if c not in JOIN_COLS]
temporal_cols = [c for c in model_based.columns if c not in {*JOIN_COLS, "fold"}]
all_feature_cols = financial_cols + temporal_cols

assert eval_panel[DATE_COL].max() < HOLDOUT_START, "the evaluation panel reaches the holdout"
if eval_panel.select(JOIN_COLS).n_unique() != len(eval_panel):
    raise ValueError("the evaluation panel has duplicate slot-metal rows")
print(f"\nPanel: {len(eval_panel):,} rows over {eval_panel[DATE_COL].n_unique():,} slots and "
      f"{eval_panel['symbol'].n_unique()} metals, from the validation stretches of "
      f"{eval_panel['fold'].n_unique()} folds")
print(f"Slots run {eval_panel[DATE_COL].min()} to {eval_panel[DATE_COL].max()}")
print(f"Candidates: {len(financial_cols)} price-derived + {len(temporal_cols)} model-derived "
      f"= {len(all_feature_cols)}")
display(eval_panel.group_by(["session", "symbol"]).len().sort(["session", "symbol"]))

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
# A list of forty column names says little about what is being tested. Grouping them by the idea
# behind each shows how concentrated the search is, which is why the correlations in part 6 matter
# and why the multiplicity adjustment in part 4 is applied over the whole set rather than family
# by family. A column matching no family stops the run rather than landing in "other".

# %%
def assign_feature_family(name: str) -> str:
    """Map a column to the economic or modelling idea it comes from."""
    family_map = [
        (["kalman_"], "trend and level"),
        (["hmm_"], "volatility regime"),
        (["arima_"], "return forecast"),
        (["sess_"], "session opening move"),
        (["pre_ret", "overnight_gap", "prev_sess_ret"], "overnight and pre-session"),
        (["slot_ret"], "decision-grid momentum"),
        (["gsr"], "gold-silver ratio"),
        (["is_", "dow", "news_"], "session and calendar state"),
        (["zscore", "d1_"], "long-window state"),
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
            "source": [
                "model-derived" if f in temporal_cols else "price-derived" for f in families
            ],
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
# **The screens are run separately on each session book**, and that is not decoration. A column can
# be perfectly informative on the pooled panel and constant inside the book that would actually
# trade it: `is_ny` is exactly that - it separates the two sleeves and takes one value inside each.
# It is the same failure that stopped `exness_fx_d1`'s eight `rank_*` columns, found one book too
# late. A column is carried forward when it passes on the **pooled** panel; the per-book table
# below records where it would be useless, and the triage ledger carries the flag.

# %%
def screen(frame: pl.DataFrame) -> tuple[dict, dict]:
    """Coverage from a column's first filled slot, and staleness within each metal."""
    coverage, staleness = {}, {}
    for feature in all_feature_cols:
        non_null = frame.filter(pl.col(feature).is_not_null())
        if len(non_null) == 0:
            coverage[feature] = 0.0
        else:
            eligible = frame.filter(pl.col(DATE_COL) >= non_null[DATE_COL].min())
            coverage[feature] = len(non_null) / len(eligible)
        chronological = frame.select([*JOIN_COLS, feature]).sort(["symbol", DATE_COL])
        unchanged = chronological.select(
            (pl.col(feature) == pl.col(feature).shift(1).over("symbol")).sum()
        ).item()
        comparable = chronological.select(
            pl.col(feature).shift(1).over("symbol").is_not_null().sum()
        ).item()
        staleness[feature] = float(unchanged) / max(comparable, 1)
    return coverage, staleness


coverage, staleness = screen(eval_panel)
per_book = {
    book: screen(eval_panel.filter(pl.col("session") == book))[1] for book in SESSION_BOOKS
}
correctness = {
    f: coverage[f] >= COVERAGE_FLOOR and staleness[f] <= STALENESS_CEILING
    for f in all_feature_cols
}
failed = [f for f, ok in correctness.items() if not ok]
print(f"{len(correctness) - len(failed)} of {len(correctness)} columns are filled on at least "
      f"{COVERAGE_FLOOR:.0%} of their slots and repeat on at most {STALENESS_CEILING:.0%} of them")
screen_table = pl.DataFrame(
    {
        "feature": all_feature_cols,
        "family": [families[f] for f in all_feature_cols],
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
with pl.Config(tbl_rows=screen_table.height, tbl_cols=12, tbl_width_chars=200):
    display(screen_table)
constant_in_a_book = screen_table.filter(pl.col("staleness_worst_book") > STALENESS_CEILING)
print(f"{constant_in_a_book.height} columns exceed the staleness ceiling inside at least one "
      f"session book: {constant_in_a_book['feature'].to_list()}")

# %% [markdown]
# ### A second screen beside the first: scale-independent staleness (generation 2, block 3)
#
# The screen above measures the share of consecutive **bit-identical** values, which asks "does
# this column repeat?" and not "does this column vary?". A column whose whole range is a few parts
# per million of its own level passes it with float jitter in the eleventh digit - `kalman_smoothness`
# did, at 0.097 against a ceiling of 0.50, while carrying no information (`bots/exness_gold_sess/BOT.md`,
# phase-4 re-fit, step 3). The screen below was **pre-declared on 2026-09-08 before one number of it
# was computed** (`BOT.md`, "Pre-declaration: a scale-independent staleness screen") and is installed
# by `GEN2_DECLARATION_2026-09-10.md` ruling 3.2: every statistic is a ratio (`n_unique / n`, a
# coefficient of variation per (fold, symbol) cell, the modal share), the thresholds are read from
# `setup.yaml::features.scale_free_screen`, and a zero-mean column skips the CV test by rule.
#
# It **stands beside** the first screen and does not replace it: the first is shared with four sibling
# case studies, and rewriting it after seeing which column it missed would be a threshold moved after
# a result. So the `decision` column of the ledger is still the first screen's, and this one's verdict
# is carried as `scale_free_verdict` with its reasons. It reads feature values, the fold id and the
# symbol - no label, no score - and runs here, before any correlation is computed. One implementation,
# `_screens.scale_free_screen`, serves this stage and `tests/test_scale_free_screen.py`.

# %%
from case_studies.exness_gold_sess._screens import scale_free_screen  # noqa: E402

SCALE_FREE = dict(setup["features"]["scale_free_screen"])
print(f"scale-free screen thresholds, declared {SCALE_FREE.get('declared_on')}: "
      f"n_unique/n < {SCALE_FREE['n_unique_ratio_stop']} STOP; CV median < {SCALE_FREE['cv_stop']:g} STOP, "
      f"< {SCALE_FREE['cv_revise']:g} REVISE; modal share > {SCALE_FREE['modal_share_revise']} REVISE, "
      f"> {SCALE_FREE['modal_share_stop']} STOP; |mean| < std -> CV not applied")
scale_free_table = scale_free_screen(
    eval_panel, all_feature_cols, thresholds=SCALE_FREE, fold_col="fold", symbol_col="symbol"
).with_columns(
    pl.col("feature").replace_strict(families, return_dtype=pl.String).alias("family"),
    pl.col("feature").is_in(temporal_cols).alias("model_based"),
)
with pl.Config(tbl_rows=scale_free_table.height, tbl_cols=14, tbl_width_chars=260, fmt_str_lengths=120):
    display(scale_free_table)
scale_free_verdict = dict(zip(scale_free_table["feature"], scale_free_table["verdict"], strict=True))
scale_free_reason = dict(zip(scale_free_table["feature"], scale_free_table["reasons"], strict=True))
for source, cols in (("model-derived", temporal_cols), ("price-derived", financial_cols)):
    counts = {v: sum(scale_free_verdict[c] == v for c in cols) for v in ("STOP", "REVISE", "PASS")}
    named = {v: [c for c in cols if scale_free_verdict[c] == v] for v in ("STOP", "REVISE")}
    print(f"scale-free screen on the {len(cols)} {source} columns: {counts}; "
          f"STOP {named['STOP']}; REVISE {named['REVISE']}")
disagree = [
    f for f in all_feature_cols
    if (scale_free_verdict[f] == "STOP") != (not correctness[f])
]
print(f"{len(disagree)} columns on which the two screens disagree about STOP: {disagree}")

# %% [markdown]
# One more group is identified before any scoring - the columns that hold **one value for the
# whole market** on a slot, `gsr` and its z-score among them. On a cross-sectional statistic they
# would have to be set aside, because a column constant across names cannot rank them and the
# correlation is undefined. **On the pooled panel IC they are perfectly well defined**: each metal
# carries its own copy of the same series, the ranks are taken within each metal, and the product
# with that metal's own label still varies slot by slot. So they are scored here rather than
# discarded, and the flag is carried into the ledger - a difference from the template that is a
# consequence of the statistic, not a relaxation of it.

# %%
market_level = []
for feature in all_feature_cols:
    if not correctness[feature]:
        continue
    per_slot = eval_panel.group_by(DATE_COL).agg(
        pl.col(feature).drop_nulls().n_unique().alias("n_values")
    )
    if float((per_slot["n_values"] <= 1).mean()) > MARKET_LEVEL_SHARE:
        market_level.append(feature)
print(f"{len(market_level)} columns hold one value for both metals on a slot and are scored "
      f"anyway (the pooled panel IC is defined on them): {market_level}")

# %% [markdown]
# ## 3. The pooled panel IC, and the overlap its interval has to allow for
#
# The measurement is one number per **slot**. Within each metal the column and the label are
# replaced by their ranks over the panel, mapped to zero mean and unit variance; the product of the
# two is averaged across the metals quoting that slot. The mean of that series is the statistic. It
# answers the question this bot actually asks - does the column line up with what this metal's
# session went on to do - rather than the cross-sectional question two names cannot support.
#
# The interval needs care because consecutive labels **overlap**: a London decision at 09:00 UTC is
# held to 17:00 while the New York slot opens inside that hold. The overlap is measured off the
# grid rather than assumed, exactly as `02_labels` measured it, and the Newey-West bandwidth is set
# from what is found.

# %%
_grid = (
    eval_panel.filter(pl.col("symbol") == UNIVERSE[0])[DATE_COL]
    .unique()
    .sort()
    .to_numpy()
    .astype("datetime64[us]")
)
_horizon = int(str(setup["labels"]["horizons"][LABEL_COL]).rstrip("Hh"))
_ends = _grid + np.timedelta64(_horizon, "h")
_overlap = int(np.max(np.searchsorted(_grid, _ends, side="left") - np.arange(len(_grid)) - 1))
HAC_LAG = _overlap + 1
print(f"{_horizon}-hour label on this grid: at most {_overlap} later decision starts inside a "
      f"holding window, so every interval below is Newey-West corrected at lag {HAC_LAG}")


def rank_normalised(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Within-metal ranks of *columns* mapped to zero mean and unit variance."""
    return frame.with_columns(
        (
            ((pl.col(c).rank(method="average").over("symbol") - 0.5) / pl.len().over("symbol") - 0.5)
            * np.sqrt(12.0)
        ).alias(f"_z_{c}")
        for c in columns
    )


def pooled_panel_series(frame: pl.DataFrame, feature: str, label: str) -> pl.DataFrame:
    """One value per slot: the mean over metals of the rank-normalised product."""
    keep = [DATE_COL, "symbol", feature, label]
    if "fold" in frame.columns:
        keep.append("fold")
    rows = frame.select(keep).drop_nulls([feature, label])
    if rows.is_empty():
        return pl.DataFrame()
    z = rank_normalised(rows, [feature, label]).with_columns(
        (pl.col(f"_z_{feature}") * pl.col(f"_z_{label}")).alias("_p")
    )
    aggs = [pl.col("_p").mean().alias("ic"), pl.len().alias("n_obs")]
    if "fold" in rows.columns:
        aggs.append(pl.col("fold").first().cast(pl.Int64))
    out = z.group_by(DATE_COL, maintain_order=True).agg(aggs).sort(DATE_COL)
    if "fold" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("fold"))
    return out.select(
        DATE_COL,
        pl.col("fold").cast(pl.Int64),
        pl.col("ic").cast(pl.Float64),
        pl.col("n_obs").cast(pl.Int64),
    )


# The claim that this reduces to Spearman where Spearman is defined, checked rather than asserted.
# On one symbol the mean of the per-slot product is the Spearman correlation up to the (1 - 1/n^2)
# rank-variance factor.
_one = eval_panel.filter(pl.col("symbol") == UNIVERSE[0]).select(
    DATE_COL, "symbol", "d1_vol_gk_63d", LABEL_COL
)
_ref = float(spearmanr(_one["d1_vol_gk_63d"].to_numpy(), _one[LABEL_COL].to_numpy())[0])
_ours = float(pooled_panel_series(_one, "d1_vol_gk_63d", LABEL_COL)["ic"].mean())
_gap = abs(_ours - _ref * (1 - 1 / len(_one) ** 2))
# 1e-7 rather than 1e-9: the identity is exact only without ties, and both columns carry a few.
# The residual is printed rather than hidden inside the tolerance, and on this panel it is seven
# significant figures below the statistic itself.
assert _gap < 1e-7, (_ours, _ref, _gap)
print(f"on one metal the pooled panel IC reproduces scipy's Spearman: {_ours:+.8f} vs "
      f"{_ref:+.8f} x (1 - 1/n^2) over {len(_one):,} slots, residual {_gap:.2e}")

# %%
evaluable = [f for f in all_feature_cols if correctness[f]]
computed = []
for feature in evaluable:
    series = pooled_panel_series(eval_panel, feature, LABEL_COL)
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
print(f"Wrote and read back evaluation/ic_timeseries.parquet: {len(stored_ic):,} scored slots "
      f"over {len(ic_results)} of {len(evaluable)} eligible columns")

# %% [markdown]
# ### Did the folds agree, or was it one window?
#
# An average over four validation years can come from four years of the same weak effect or from
# one year of a strong one, and only the second is a reason to be sceptical. The same average is
# taken inside each fold and two summaries are kept: the share of folds pointing the **column's
# own** way - so a steadily inverted column is not disqualified for being inverted - and the
# **ICIR**, the mean fold IC divided by its spread across folds, which separates a small agreement
# that repeats from a larger one that came out of a single window.

# %%
fold_stats = {}
for feature, stats in ic_results.items():
    fold_means = []
    for fold in sorted(ic_timeseries[feature]["fold"].drop_nulls().unique().to_list()):
        values = ic_timeseries[feature].filter(pl.col("fold") == fold)["ic"].to_numpy()
        if len(values) >= MIN_FOLD_SLOTS:
            fold_means.append(float(np.mean(values)))
    if not fold_means:
        continue
    direction = 1 if stats["mean_ic"] >= 0 else -1
    dispersion = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else float("nan")
    fold_stats[feature] = {
        "n_folds": len(fold_means),
        "direction": "positive" if direction > 0 else "negative",
        "sign_consistency": sum((v * direction) > 0 for v in fold_means) / len(fold_means),
        "positive_fold_share": sum(v > 0 for v in fold_means) / len(fold_means),
        "icir": float(np.mean(fold_means)) / dispersion if dispersion else float("nan"),
        "worst_fold_ic": min(fold_means),
        "best_fold_ic": max(fold_means),
        "median_fold_ic": float(np.median(fold_means)),
        "fold_ics": fold_means,
    }
print(f"Fold-by-fold summary built for {len(fold_stats)} columns over "
      f"{len(VALIDATION_FOLD_IDS)} validation folds")

# %% [markdown]
# ## 4. What was searched, and what that costs
#
# Test forty columns at the conventional five-percent level against a label none of them predicts
# and about two will still come back significant, because that is what the five-percent level
# means. A p-value is only readable against the set of tests it came out of, so that set is
# declared first and **Benjamini-Hochberg** is applied over the whole of it.
#
# Nothing here was chosen after a score was seen. The price-derived columns come from the window
# register in `setup.yaml`, which fixed every window before any of them was built; the
# model-derived columns are whatever the three estimators of `04` produce.

# %%
searched_set = {
    "columns the feature stages generated": len(all_feature_cols),
    "of those, filled in and moving enough to test": sum(correctness.values()),
    "of those, scored on the validation slots": len(ic_results),
    "label horizons the case study declares": len(LABEL_NAMES),
    "session books the sweep will run": len(SESSION_BOOKS) + 1,
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
        "hac_se": [ic_results[f]["hac_se"] for f in feature_names],
        "hac_t": [ic_results[f]["t_stat"] for f in feature_names],
        "hac_lags": [int(ic_results[f]["effective_lags"]) for f in feature_names],
        "hac_p": hac_p_values,
        "fdr_p": list(fdr_result["adjusted_p_values"]),
        "fdr_sig": list(fdr_result["rejected"]),
        "icir": [fold_stats.get(f, {}).get("icir") for f in feature_names],
        "positive_fold_share": [
            fold_stats.get(f, {}).get("positive_fold_share") for f in feature_names
        ],
        "sign_consistency": [fold_stats.get(f, {}).get("sign_consistency") for f in feature_names],
    },
    schema={
        "feature": pl.String,
        "family": pl.String,
        "source": pl.String,
        "ic_mean": pl.Float64,
        "naive_t": pl.Float64,
        "hac_se": pl.Float64,
        "hac_t": pl.Float64,
        "hac_lags": pl.Int64,
        "hac_p": pl.Float64,
        "fdr_p": pl.Float64,
        "fdr_sig": pl.Boolean,
        "icir": pl.Float64,
        "positive_fold_share": pl.Float64,
        "sign_consistency": pl.Float64,
    },
).sort(pl.col("ic_mean").abs(), descending=True)
n_naive = sum(abs(ic_results[f]["naive_t_stat"]) > 1.96 for f in feature_names)
n_hac = sum(p < 0.05 for p in hac_p_values)
n_fdr = int(fdr_result["n_rejected"])
print(f"Columns clearing 5% treating each slot as independent: {n_naive}")
print(f"Columns clearing 5% once Newey-West rescales the interval at lag {HAC_LAG}: {n_hac}")
print(f"Columns clearing 5% after Benjamini-Hochberg over the set: {n_fdr}")
with pl.Config(tbl_rows=eval_summary.height, tbl_cols=16, tbl_width_chars=240):
    display(eval_summary)

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ### The same columns against the longer labels
#
# Everything above is measured against the session close. The case study declares two more labels,
# and how long a column stays informative decides how long a position built on it can be held. Each
# label is joined on its own rows: `fwd_ret_24h` is null wherever two slots ahead is not exactly
# twenty-four hours - every Friday and every holiday - so it is scored on a Monday-to-Thursday
# sample, which is stated here rather than discovered in a backtest.

# %%
horizon_rows = []
for label_name in LABEL_NAMES:
    frame = pl.read_parquet(CASE_DIR / "labels" / f"{label_name}.parquet")
    panel = (
        validation_rows(fold_windows(LABEL_BUFFERS[label_name]))
        .join(financial, on=JOIN_COLS, how="inner")
        .join(frame, on=JOIN_COLS, how="inner")
        .sort([DATE_COL, "symbol"])
    )
    assert panel[DATE_COL].max() < HOLDOUT_START, f"{label_name} panel reaches the holdout"
    for feature in evaluable:
        series = pooled_panel_series(panel, feature, label_name)
        if len(series) < MIN_SLOTS:
            continue
        fold_means = [
            float(part["ic"].mean())
            for part in series.partition_by("fold")
            if len(part) >= MIN_FOLD_SLOTS
        ]
        dispersion = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else np.nan
        horizon_rows.append(
            {
                "feature": feature,
                "label": label_name,
                "n_slots": len(series),
                "ic_mean": float(series["ic"].mean()),
                "icir": float(np.mean(fold_means)) / dispersion if dispersion else np.nan,
            }
        )
horizon_ic = pl.DataFrame(horizon_rows)
_wide = horizon_ic.pivot(index="feature", on="label", values="ic_mean").sort(
    pl.col(LABEL_COL).abs(), descending=True
)
with pl.Config(tbl_rows=_wide.height, tbl_width_chars=200):
    display(_wide)
print(f"Horizon profile computed for {len(horizon_rows)} feature-label pairs. Sample sizes: "
      + ", ".join(
          f"{n}: {horizon_ic.filter(pl.col('label') == n)['n_slots'].max():,} slots"
          for n in LABEL_NAMES
      ))

# %% [markdown]
# ## 5. How much of this is the same evidence counted twice?
#
# Two columns can agree so closely on how they order a metal's slots that a model gains nothing
# from having both - and the multiplicity adjustment above has already paid for testing both as if
# they were separate questions. Columns linked by a pair above the declared cut form a group,
# following the links transitively, and one column stands for each group by the largest median fold
# agreement and, where those tie, the steadiest across folds.
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
    if not stats:
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
        "weakest_link": min(
            (
                abs(pair["correlation"])
                for pair in pairs
                if representative[pair["left"]] == representative[members[0]]
            ),
            default=float("nan"),
        ),
    }
    for members in redundant.values()
]
if group_rows:
    with pl.Config(tbl_rows=len(group_rows), tbl_width_chars=220, fmt_str_lengths=180):
        display(pl.DataFrame(group_rows).sort("columns_in_group", descending=True))
for pair in pairs[:15]:
    print(f"  {pair['left']} / {pair['right']}: {pair['correlation']:+.3f}")

# %% [markdown]
# ## 6. One decision per column
#
# The output of this stage is not a shortlist. It is a decision per column with the evidence
# attached, in the book's three categories - and Chapter 20's warning applies in full: **surviving
# this triage is a filter, not evidence of a strategy.** Nothing here is selection. Selection
# happens on the validation backtest in phase 5, on returns net of the measured spread, and a
# column that reaches phase 5 has been screened rather than confirmed.
#
# - `PROCEED`: worth carrying into multivariate work. It says nothing about whether the column ends
#   up in a trained model.
# - `REVISE`: nothing usable was found by this test, and this test is narrow - a column that
#   matters only through an interaction, or past a threshold, is invisible to a rank correlation.
# - `STOP`: the column failed a screen in part 2. Nothing was measured on it because nothing
#   measured on it would mean anything.
#
# Two routes lead to `PROCEED` and the ledger records which one a column took. The first is a
# confirmation: it cleared the false-discovery adjustment over the declared search. The second is a
# **search** in the sense of Section 7.4 - it promotes on agreeing in most folds and on exceeding
# the effect-size floor - and exists so that a two-metal panel, where almost nothing clears a
# multiplicity adjustment, does not leave the next stage with nothing to model. A column promoted
# that way has been confirmed by nothing, and `note` is what lets a reader tell the two apart.

# %%
fdr_significant = set(eval_summary.filter(pl.col("fdr_sig"))["feature"].to_list())
triage = {}
for feature in all_feature_cols:
    if not correctness[feature]:
        triage[feature] = ("STOP", "correctness_fail")
    elif feature not in ic_results:
        triage[feature] = ("REVISE", "insufficient_validation_data")
    elif feature in fdr_significant:
        triage[feature] = ("PROCEED", "fdr_significant")
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
    ledger_rows.append(
        {
            "feature": feature,
            "family": families[feature],
            "source": "model_based" if feature in temporal_cols else "financial",
            "statistic": "pooled_panel_ic",
            "hac_lag": HAC_LAG,
            "ic_mean": ic_results.get(feature, {}).get("mean_ic"),
            "hac_se": ic_results.get(feature, {}).get("hac_se"),
            "hac_t": ic_results.get(feature, {}).get("t_stat"),
            "hac_p": ic_results.get(feature, {}).get("p_value"),
            "fdr_p": match["fdr_p"][0] if len(match) else None,
            "fdr_sig": bool(match["fdr_sig"][0]) if len(match) else False,
            "icir": fold_stats.get(feature, {}).get("icir"),
            "positive_fold_share": fold_stats.get(feature, {}).get("positive_fold_share"),
            "sign_consistency": fold_stats.get(feature, {}).get("sign_consistency"),
            "n_folds": fold_stats.get(feature, {}).get("n_folds"),
            "worst_fold_ic": fold_stats.get(feature, {}).get("worst_fold_ic"),
            "coverage": coverage[feature],
            "staleness": staleness[feature],
            "staleness_worst_book": max(per_book[b][feature] for b in SESSION_BOOKS),
            "constant_in_a_book": max(per_book[b][feature] for b in SESSION_BOOKS)
            > STALENESS_CEILING,
            "market_level": feature in market_level,
            "stands_for_group": representative.get(feature),
            # generation 2, block 3: the pre-declared scale-free screen, carried BESIDE the
            # decision (ruling 3.2); it moves no decision and gates nothing here
            "scale_free_verdict": scale_free_verdict[feature],
            "scale_free_reason": scale_free_reason[feature],
            "decision": decision,
            "note": note,
        }
    )
triage_ledger = pl.DataFrame(ledger_rows).sort(["decision", "feature"])
triage_ledger.write_parquet(EVAL_DIR / "triage_ledger.parquet")
print(f"Wrote evaluation/triage_ledger.parquet: one row for each of {len(triage_ledger)} columns")
decision_table = (
    triage_ledger.group_by(["decision", "note"]).agg(pl.len().alias("columns")).sort(
        ["decision", "note"]
    )
)
with pl.Config(tbl_rows=decision_table.height):
    display(decision_table)
with pl.Config(tbl_rows=triage_ledger.height, tbl_cols=20, tbl_width_chars=260):
    display(
        triage_ledger.select(
            "feature", "family", "ic_mean", "hac_t", "hac_p", "fdr_sig", "icir",
            "positive_fold_share", "constant_in_a_book", "market_level", "scale_free_verdict",
            "decision", "note"
        )
    )

# %% [markdown] tags=["results"]
# Results are recorded here after the first run.

# %% [markdown]
# ## Key takeaways
#
# 1. **Pick the statistic the universe can carry.** A cross-sectional rank correlation over two
#    names is a coin flip, and on the two market-level columns it is not even that - it is
#    undefined. The pooled panel IC asks the question this bot asks and reduces to Spearman where
#    Spearman is defined, which is checked here and not asserted.
# 2. **Screen per book, not only pooled.** A column can be informative on the panel and constant
#    inside the book that would trade it. `exness_fx_d1` found that one book too late.
# 3. **Measure the overlap on the grid, then carry it into every standard error.** Two sleeves a
#    day means consecutive labels share hours, and a naive t-statistic on overlapping labels is the
#    most common way a session bot flatters itself.
# 4. **Declare the search before reading a p-value out of it**, and adjust over the whole declared
#    set rather than family by family.
# 5. **Surviving this is a filter, not a finding** (Chapter 20). The ledger records evidence, not a
#    shortlist, and it gates nothing: `06_linear` reads the feature matrix, not these decisions.
#
# **Next**: `06_linear.py` fits the first models that read several columns at once.
