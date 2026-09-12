"""How a ONE-NAME panel is read - the one implementation `05`, and later `06`/`07`/`12`, import.

WHY THIS MODULE EXISTS
----------------------
Every information coefficient this repository computes by default is a **cross-sectional**
one: within each decision instant it ranks the entities by the score, ranks them by the
outcome, correlates the two rankings and averages the per-instant correlation through time.
That statistic needs a panel wide enough to rank. **This bot's panel is one instrument**, and
on one row per instant every one of the three implementations returns nothing - silently, as an
empty result rather than an error:

1. ``case_studies/utils/feature_engineering.py:607`` filters
   ``pl.len().over(date_col) >= max(min_cross_section, n_quantiles)`` before it scores anything,
   so every decision instant here (one row) is removed and the function returns ``None`` for
   every feature.
2. ``case_studies/utils/registry/metrics.py:543-563`` and the registry's own
   ``compute_prediction_fold_metrics`` carry a hard-coded ``min_obs=5``. Every ``ic_mean``,
   ``ic_std``, ``ic_t``, ``ic_n_days`` and ``pct_positive`` the registry writes for this case
   study is therefore **NULL by construction**, on every configuration, at every checkpoint, for
   every label - and it is null for a reason that has nothing to do with the model.
3. ``case_studies/exness_fx_d1/05_evaluation.py:330`` sets ``MIN_PERIODS = max(3, n_symbols //
   4)``, which on one name is 3 and still needs three rows in an instant that has one.

Lowering a floor would not rescue any of them. A Spearman correlation over ``n = 1`` is
undefined; over ``n = 2`` it takes only the values ``+1`` and ``-1``, which is a coin flip
recorded as a correlation.

WHAT IS READ INSTEAD
--------------------
A **time-series** information coefficient, which is the question a per-asset signal-timing bot
(``setup.yaml::mapping.class``) is actually asking: does this score, at this instrument, rank
its own future returns through time? The construction is the pooled panel IC of
``case_studies/exness_gold_sess/_model_reading.py``, and on one name it reduces exactly to that
name's Spearman rank correlation - rank-normalise the score and the outcome within the symbol
over the whole window, multiply row-wise, and average. Keeping the pooled form rather than
calling ``scipy.stats.spearmanr`` buys two things: the per-slot product series, which is what a
Newey-West standard error and a block bootstrap both need, and a statistic that is literally the
same measurement as the one the two sibling bots report.

INFERENCE, TWICE
----------------
Overlapping holds make consecutive slot ICs dependent, so no naive t-statistic is reported
without a corrected one beside it. Two independent corrections are applied, because they fail
differently:

* **Newey-West.** ``hac_lag_for_label`` MEASURES the overlap on the decision grid - how many
  later decisions start inside one holding window - and adds one. On this grid the answer is
  ``(0, 1)`` for the 8-hour label (the next decision starts exactly when the hold ends, so
  nothing overlaps) and ``(2, 3)`` for the 24-hour one. The measured value is a **floor**:
  ``ml4t.diagnostic.metrics.compute_ic_hac_stats`` applies ``max(label_horizon - 1, Newey-West
  auto)`` capped at ``T // 2``, and on a few thousand slots the automatic bandwidth is around 10,
  so 10 is what gets used. ``hac_lags`` is reported from the result rather than from the request,
  because a reader who saw "lag 1" while 10 was applied would mis-state the inference.
* **A moving-block bootstrap.** Newey-West corrects the variance of the mean under an assumed
  covariance structure; the bootstrap resamples contiguous blocks of the IC series and makes no
  such assumption. A block length of one day of slots (3) or one week (21) preserves whatever
  dependence is really there. When the two disagree the bootstrap is the one to believe, and
  that disagreement is exactly what a single number would hide.

Four validation folds is four numbers, so the fold breakdown always travels with the pooled
estimate: an average over four validation years can come from four years of the same weak effect
or from one year of a strong one, and only the second is a reason to be sceptical.

NOTHING HERE SELECTS
--------------------
Every statistic in this module is a diagnostic. Selection happens on validation **backtest**
Sharpe in `13_backtest`, never on an IC (`.claude/skills/ml4t/SKILL.md` rule 6). The complete
prediction population advances to the backtest whatever these numbers say.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = [
    "REGISTRY_IC_IS_NULL_REASON",
    "block_bootstrap_ic",
    "fold_ic_summary",
    "hac_lag_for_label",
    "outcome_column_for",
    "pooled_panel_ic_series",
    "prediction_staleness_by_book",
    "rank_normalised",
    "summarise_pooled_ic",
]


def outcome_column_for(frame: pl.DataFrame) -> str:
    """Which column a prediction set is scored against: ``eval_actual`` when it exists.

    Ported unchanged from ``case_studies/exness_gold_sess/_model_reading.py`` (same rule, same
    reason): a classification prediction set carries ``eval_actual``, the continuous return its
    classes were cut from (``setup.yaml::labels.classification_eval_label``); a regression set
    has none and is scored on ``actual`` directly. **This generation of `exness_btc_8h` declares
    no classification label** (`labels.classification_eval_label: {}`), so every call this
    generation makes returns ``"actual"`` - the function is carried over rather than inlined so
    that a later `dir_tb_8h`-shaped label (declared PLANNED, `+1 trial family` note in
    `setup.yaml::features.families`) does not require touching `06`/`07`/`12` again.
    """
    if "eval_actual" in frame.columns and frame.get_column("eval_actual").null_count() < frame.height:
        return "eval_actual"
    return "actual"

#: Quoted verbatim by every stage that would otherwise leave a reader guessing why the registry's
#: IC column is empty, so they all give the same explanation rather than three paraphrases.
REGISTRY_IC_IS_NULL_REASON = (
    "the registry's ic_mean is NULL on every row of this case study by construction: it is a "
    "cross-sectional Spearman computed within each decision instant under a hard-coded "
    "min_obs=5 (case_studies/utils/registry/metrics.py) and this panel carries ONE instrument "
    "per instant, so no instant is scorable. Lowering the floor would not help - a Spearman "
    "over n=1 is undefined. The readable statistic on a one-name panel is the time-series "
    "pooled IC in case_studies/exness_btc_8h/_model_reading.py, which 05_evaluation uses."
)


def rank_normalised(
    frame: pl.DataFrame, columns: list[str], *, by: str = "symbol"
) -> pl.DataFrame:
    """Within-symbol ranks of *columns* mapped to zero mean and unit variance.

    The same transform ``05_evaluation`` applies to a feature and ``12_model_analysis`` applies
    to a model score, so a feature's IC and a model's IC in this case study are the same
    measurement of the same shape and can be put on one axis.
    """
    return frame.with_columns(
        (
            ((pl.col(c).rank(method="average").over(by) - 0.5) / pl.len().over(by) - 0.5)
            * np.sqrt(12.0)
        ).alias(f"_z_{c}")
        for c in columns
    )


def pooled_panel_ic_series(
    frame: pl.DataFrame,
    *,
    score_col: str = "prediction",
    outcome_col: str = "actual",
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    fold_col: str | None = "fold",
) -> pl.DataFrame:
    """One IC value per decision slot: the mean over symbols of the rank-normalised product.

    On this bot's one-name panel the "mean over symbols" is over a single term, so the series is
    the row-wise product of two rank-normalised series and its average IS the Spearman
    correlation of score against outcome. The pooled form is kept because a per-slot series is
    what both inference paths need, and because it is the identical function the two sibling
    bots on this account report.

    Returns ``date_col, fold, ic, n_obs`` sorted by slot. Rows with a null score or a null
    outcome are dropped **before** ranking, so a configuration that predicts nothing on some rows
    is ranked over the rows it did predict rather than against a null.
    """
    keep = [date_col, symbol_col, score_col, outcome_col]
    if fold_col and fold_col in frame.columns:
        keep.append(fold_col)
    rows = frame.select(keep).drop_nulls([score_col, outcome_col])
    if rows.is_empty():
        return pl.DataFrame(
            schema={
                date_col: frame.schema[date_col],
                "fold": pl.Int64,
                "ic": pl.Float64,
                "n_obs": pl.Int64,
            }
        )
    z = rank_normalised(rows, [score_col, outcome_col], by=symbol_col).with_columns(
        (pl.col(f"_z_{score_col}") * pl.col(f"_z_{outcome_col}")).alias("_p")
    )
    aggs = [pl.col("_p").mean().alias("ic"), pl.len().alias("n_obs")]
    if fold_col and fold_col in rows.columns:
        aggs.append(pl.col(fold_col).first().cast(pl.Int64).alias("fold"))
    out = z.group_by(date_col, maintain_order=True).agg(aggs).sort(date_col)
    if "fold" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("fold"))
    return out.select(
        pl.col(date_col),
        pl.col("fold").cast(pl.Int64),
        pl.col("ic").cast(pl.Float64),
        pl.col("n_obs").cast(pl.Int64),
    )


def hac_lag_for_label(grid: np.ndarray | pl.Series, horizon_hours: int) -> tuple[int, int]:
    """``(overlap, hac_lag)`` MEASURED on the decision grid, never assumed.

    *grid* is the sorted array of decision instants. *overlap* is the largest number of later
    decisions that start strictly inside a holding window of ``horizon_hours``; the Newey-West
    lag is that plus one, so a label whose windows do not overlap still gets a lag of 1 rather
    than 0.

    On this bot's grid the answer is ``(0, 1)`` for ``fwd_ret_8h`` - the next decision starts
    exactly when the hold ends, so consecutive labels are disjoint, which is the property
    ``labels.rebalance_step: 1`` claims - and ``(2, 3)`` for ``fwd_ret_24h``.
    """
    stamps = (grid.to_numpy() if isinstance(grid, pl.Series) else np.asarray(grid)).astype(
        "datetime64[us]"
    )
    stamps = np.sort(np.unique(stamps))
    ends = stamps + np.timedelta64(int(horizon_hours), "h")
    overlap = int(np.max(np.searchsorted(stamps, ends, side="left") - np.arange(len(stamps)) - 1))
    return overlap, overlap + 1


def block_bootstrap_ic(
    series: pl.DataFrame,
    *,
    block: int,
    n_resamples: int = 2000,
    seed: int = 42,
    ic_col: str = "ic",
) -> dict[str, float | int]:
    """Moving-block bootstrap of the mean IC: a confidence interval that assumes no model.

    Newey-West widens an interval under an assumed covariance structure. This resamples
    contiguous blocks of length ``block`` from the IC series with replacement until the resample
    is as long as the original, and reports the distribution of the resampled means. Overlapping
    blocks are drawn (a moving-block bootstrap), so every starting point is available and the
    procedure does not depend on where the series happens to begin.

    ``block`` should be at least the dependence the grid has: one day of slots (3) is the floor,
    one week (21) the conservative choice. Returns the point estimate, the bootstrap standard
    error, the 2.5th and 97.5th percentiles, and the share of resampled means on the opposite
    side of zero from the point estimate - which is the one-sided p-value a reader wants.
    """
    values = series[ic_col].drop_nulls().to_numpy()
    n = len(values)
    if n < 2 * block or block < 1:
        return {"n": int(n), "block": int(block), "n_resamples": 0}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)[:, :n]
    means = values[idx].mean(axis=1)
    point = float(values.mean())
    return {
        "n": int(n),
        "block": int(block),
        "n_resamples": int(n_resamples),
        "boot_mean": float(means.mean()),
        "boot_se": float(means.std(ddof=1)),
        "boot_ci_lo": float(np.percentile(means, 2.5)),
        "boot_ci_hi": float(np.percentile(means, 97.5)),
        # share of resamples on the other side of zero from the point estimate
        "boot_p_one_sided": float(np.mean(means <= 0.0) if point > 0 else np.mean(means >= 0.0)),
    }


def fold_ic_summary(
    series: pl.DataFrame, *, min_fold_slots: int = 60
) -> dict[str, float | int | list]:
    """Fold-by-fold reading of one IC series.

    ``sign_consistency`` is the share of folds pointing the series' **own** way, so a steadily
    inverted configuration is not disqualified for being inverted; ``positive_fold_share`` is the
    share pointing up, which is the one to read when the sign itself is the claim. ``icir`` is
    the mean fold IC over its dispersion across folds - it separates a small agreement that
    repeats from a larger one that came out of a single window.
    """
    if series.is_empty() or "fold" not in series.columns:
        return {"n_folds": 0}
    fold_means: list[float] = []
    for fold in sorted(series["fold"].drop_nulls().unique().to_list()):
        values = series.filter(pl.col("fold") == fold)["ic"].drop_nulls().to_numpy()
        if len(values) >= min_fold_slots:
            fold_means.append(float(np.mean(values)))
    if not fold_means:
        return {"n_folds": 0}
    mean_of_folds = float(np.mean(fold_means))
    direction = 1 if mean_of_folds >= 0 else -1
    dispersion = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else float("nan")
    return {
        "n_folds": len(fold_means),
        "sign_consistency": sum((v * direction) > 0 for v in fold_means) / len(fold_means),
        "positive_fold_share": sum(v > 0 for v in fold_means) / len(fold_means),
        "icir": mean_of_folds / dispersion if dispersion else float("nan"),
        "worst_fold_ic": min(fold_means),
        "best_fold_ic": max(fold_means),
        "median_fold_ic": float(np.median(fold_means)),
        "fold_ics": fold_means,
    }


def summarise_pooled_ic(
    series: pl.DataFrame,
    *,
    hac_lag: int,
    bootstrap_block: int = 21,
    n_resamples: int = 2000,
    seed: int = 42,
    min_fold_slots: int = 60,
) -> dict:
    """The pooled IC with BOTH intervals and the fold breakdown, in one dictionary.

    A point estimate alone is not a reading, and one interval alone hides the case where the two
    disagree - so no caller can print any of the three without the others.
    """
    from ml4t.diagnostic.metrics import compute_ic_hac_stats

    if series.is_empty():
        return {"n_slots": 0}
    stats = compute_ic_hac_stats(series, ic_col="ic", label_horizon=hac_lag)
    out = {
        "n_slots": series.height,
        "ic_mean": float(stats["mean_ic"]),
        "naive_t": float(stats["naive_t_stat"]),
        "hac_se": float(stats["hac_se"]),
        "hac_t": float(stats["t_stat"]),
        "hac_p": float(stats["p_value"]),
        "hac_lags": int(stats["effective_lags"]),
    }
    out |= block_bootstrap_ic(
        series, block=bootstrap_block, n_resamples=n_resamples, seed=seed
    )
    out |= fold_ic_summary(series, min_fold_slots=min_fold_slots)
    return out


def prediction_staleness_by_book(
    frame: pl.DataFrame,
    *,
    score_col: str = "prediction",
    book_col: str = "session_book",
    symbol_col: str = "symbol",
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """Share of consecutive scores identical to the one before, per session book.

    The screen ``05_evaluation`` runs on features, run on a model's own output. A score that
    repeats cannot tell this slot from the last, and a score can be perfectly non-constant on the
    pooled grid and constant inside ONE book - which is what would happen to ``slot_of_day``,
    ``is_us_hours`` and ``pays_swap_night`` inside the ``us_hours`` filter, where all three are
    constants. Run per book because a book is what a ``SESSION_FILTER`` trades.
    """
    rows = []
    books = sorted(frame[book_col].unique().to_list()) if book_col in frame.columns else []
    for book in [*books, "__pooled__"]:
        part = frame if book == "__pooled__" else frame.filter(pl.col(book_col) == book)
        chronological = part.select([date_col, symbol_col, score_col]).sort([symbol_col, date_col])
        unchanged = chronological.select(
            (pl.col(score_col) == pl.col(score_col).shift(1).over(symbol_col)).sum()
        ).item()
        comparable = chronological.select(
            pl.col(score_col).shift(1).over(symbol_col).is_not_null().sum()
        ).item()
        rows.append(
            {
                "book": "pooled" if book == "__pooled__" else book,
                "rows": part.height,
                "staleness": float(unchanged) / max(int(comparable), 1),
            }
        )
    return pl.DataFrame(rows)
