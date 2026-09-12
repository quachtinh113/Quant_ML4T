"""How a two-name panel is read - the one implementation `06`, `07` and `12` all import.

WHY THIS MODULE EXISTS
----------------------
The registry scores every prediction set with a **cross-sectional** information coefficient:
within each decision instant it ranks the entities by the model's score, ranks them by the
outcome, correlates the two rankings, and averages the per-instant correlation through time
(``case_studies/utils/registry/metrics.py`` ->
``ml4t.diagnostic.metrics.cross_sectional_ic``). That statistic is defined for a panel wide
enough to rank. **This bot's panel is two metals**, and two things follow, both measured
rather than argued:

1. ``compute_prediction_fold_metrics`` calls ``cross_sectional_ic(..., min_obs=5)`` -
   ``case_studies/utils/registry/metrics.py:113``, a hard-coded literal, not a setting. Every
   decision instant here carries **2** rows, so no instant clears the floor. Measured on this
   bot's own panel on 2026-09-08, with ``sess_ret_1h`` standing in for a model score against
   ``fwd_ret_8h``: 9,619 rows over 4,812 slots (4,807 slots carry 2 rows, 5 carry 1), and at
   ``min_obs=5`` the result is ``n_periods = 0``, ``ic_mean = nan``. Every ``ic_mean``,
   ``ic_std``, ``ic_t``, ``ic_n_days`` and ``pct_positive`` the registry writes for this case
   study is therefore **NULL by construction**, on every configuration, at every checkpoint,
   for every label.
2. Lowering the floor would not rescue it. A Spearman correlation over ``n = 2`` takes only
   the values ``+1`` and ``-1``. The same measurement at ``min_obs=2`` returns 4,807 scored
   slots with ``ic_std = 1.0000`` - a coin flip recorded as a correlation. Averaging coin
   flips over four years produces a number with a standard error and no content.
3. Some columns are degenerate even at ``min_obs=2``. ``gsr`` and ``gsr_z_252d`` hold **one
   value for the whole market** at a slot, so their cross-section has exactly zero variance
   and the correlation is undefined: measured on the same panel, ``gsr`` yields
   ``n_periods = 0`` at ``min_obs=2`` as well as at 5. ``05_evaluation`` identified this group
   before scoring anything, for the same reason.

So the notebooks in this case study **do not rank on the registry's ``ic_mean``**, and they
say so where a reader would otherwise assume the column is simply missing. What they rank on
is the **pooled panel IC** that ``05_evaluation`` already established for the same reason and
checked against ``scipy.stats.spearmanr`` to a residual of 3.6e-9: rank-normalise the score
and the outcome **within each metal over the whole window**, multiply them row-wise, and
average across the metals at each slot. On one metal it reduces to that metal's Spearman
correlation; on two it is the average of two time-series correlations rather than a
cross-section of width two. It is a **time-series** statistic, which is what a per-asset
signal-timing bot (``setup.yaml::mapping.class: per_asset_signal_timing``) is asking about.

INFERENCE
---------
Overlapping holds make consecutive slot ICs dependent, so every interval here is
Newey-West corrected. The overlap is **measured on the decision grid** rather than assumed:
``hac_lag_for_label`` counts how many later decisions start inside one holding window and
adds one. On this grid the answer is 2 for the 8-hour labels (one later decision starts
inside every window, measured 2026-09-08 over 4,812 slots) - the same number ``02_labels``
measured through ``N_eff`` and ``bots/exness_gold_sess/tests/test_sessions.py`` re-measured
inside the 149 days a year when London and New York are four hours apart instead of five.

**That measured value is a floor, not the lag that is used**, and the difference matters when
reading ``hac_lags`` in the output. ``ml4t.diagnostic.metrics.compute_ic_hac_stats`` takes it
as ``label_horizon`` and applies ``max(label_horizon - 1, Newey-West auto)`` capped at
``T // 2``, where the automatic bandwidth is ``floor(4 * (T/100) ** (2/9))``. On a 4,812-slot
series that automatic value is **9**, so 9 is what gets used and 2 is only the guarantee that
the correction can never be *narrower* than the measured overlap. Reporting ``hac_lags`` from
the result rather than reprinting the requested 2 is deliberate: a wider bandwidth is the
conservative direction, and a reader who saw "lag 2" in the output while 9 was applied would
mis-state the inference.

A pooled point estimate on its own is not enough evidence to read a model by, so
``summarise_pooled_ic`` always returns the fold breakdown beside it: the share of validation
folds pointing the configuration's own way (``sign_consistency``), the share pointing up
(``positive_fold_share``), and the ICIR - the mean fold IC over its dispersion across folds -
which separates a small agreement that repeats from a larger one that came out of a single
window. Four folds is four numbers; the point of printing them is that an average over four
validation years can come from four years of the same weak effect or from one year of a
strong one, and only the second is a reason to be sceptical.

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
    "fold_ic_summary",
    "hac_lag_for_label",
    "outcome_column_for",
    "pooled_panel_ic_series",
    "prediction_staleness_by_book",
    "rank_normalised",
    "summarise_pooled_ic",
]

# Quoted verbatim by 06, 07 and 12 so all three give a reader the same explanation of the
# empty column rather than three paraphrases of it.
REGISTRY_IC_IS_NULL_REASON = (
    "the registry's ic_mean is NULL on every row of this case study by construction: it is a "
    "cross-sectional Spearman computed within each decision instant under a hard-coded "
    "min_obs=5 (case_studies/utils/registry/metrics.py:113) and this panel carries 2 metals "
    "per instant, so no instant is scorable. Lowering the floor would not help - a Spearman "
    "over n=2 takes only the values +/-1. The readable statistic on a two-name panel is the "
    "pooled panel IC below, the same one 05_evaluation used."
)


def rank_normalised(
    frame: pl.DataFrame, columns: list[str], *, by: str = "symbol"
) -> pl.DataFrame:
    """Within-metal ranks of *columns* mapped to zero mean and unit variance.

    The identical transform ``05_evaluation.rank_normalised`` applies, so a feature's IC and a
    model's IC in this case study are the same measurement of the same shape and can be put
    on one axis.
    """
    return frame.with_columns(
        (
            ((pl.col(c).rank(method="average").over(by) - 0.5) / pl.len().over(by) - 0.5)
            * np.sqrt(12.0)
        ).alias(f"_z_{c}")
        for c in columns
    )


def outcome_column_for(frame: pl.DataFrame) -> str:
    """Which column a prediction set is scored against: ``eval_actual`` when it exists.

    A classification prediction set carries two outcome columns. ``actual`` is the class the
    model was fitted on (-1 / 0 / +1 on this bot); ``eval_actual`` is the continuous return the
    class was cut from, joined in by ``utils.modeling.load_modeling_dataset`` from the label
    declared in ``setup.yaml::labels.classification_eval_label`` and written to the prediction
    frame by ``case_studies/utils/linear.py:987-988`` and the GBM runner.

    Scoring on ``actual`` for a classification label computes ``Spearman(predicted class, true
    class)``, which is not the declared ranking statistic and is not comparable with the
    regression labels' IC - it is on a different target with a different variance. Measured
    2026-09-08 on this bot: the same 13 logistic prediction sets score **0.68-0.69** against
    ``actual`` and are ranked against ``fwd_ret_8h`` at a value two orders of magnitude smaller.
    ``setup.yaml`` says which one is the ranking target: ``probability_to_score_actual:
    fwd_ret_8h``, "the same target the primary label is scored on".

    A regression prediction set has no ``eval_actual`` and is scored on ``actual``, which for it
    IS the continuous return. So one rule covers both and no caller chooses.

    The all-null test matters because ``12_model_analysis`` concatenates every prediction set into
    one frame before slicing it: there the column exists on the regression slices too, holding
    nothing. Presence of the column is not presence of the outcome.
    """
    if "eval_actual" in frame.columns and frame.get_column("eval_actual").null_count() < frame.height:
        return "eval_actual"
    return "actual"


def pooled_panel_ic_series(
    frame: pl.DataFrame,
    *,
    score_col: str = "prediction",
    outcome_col: str = "actual",
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    fold_col: str | None = "fold",
) -> pl.DataFrame:
    """One IC value per decision slot: the mean over metals of the rank-normalised product.

    Returns a frame of ``date_col, fold, ic, n_obs`` sorted by slot. Rows with a null score or
    a null outcome are dropped **before** ranking, so a configuration that predicts nothing on
    some rows is ranked over the rows it did predict rather than against a null.
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


def hac_lag_for_label(
    grid: np.ndarray | pl.Series,
    horizon_hours: int,
) -> tuple[int, int]:
    """``(overlap, hac_lag)`` measured on the decision grid, never assumed.

    *grid* is the sorted array of decision instants of ONE metal (they share the grid).
    *overlap* is the largest number of later decisions that start inside a holding window of
    ``horizon_hours``; the Newey-West lag is that plus one. On this bot's grid the answer is
    ``(1, 2)`` for an 8-hour label: the New York decision at 14:00 UTC starts inside the
    London hold that runs 09:00 -> 17:00.
    """
    stamps = (grid.to_numpy() if isinstance(grid, pl.Series) else np.asarray(grid)).astype(
        "datetime64[us]"
    )
    stamps = np.sort(np.unique(stamps))
    ends = stamps + np.timedelta64(int(horizon_hours), "h")
    overlap = int(np.max(np.searchsorted(stamps, ends, side="left") - np.arange(len(stamps)) - 1))
    return overlap, overlap + 1


def fold_ic_summary(
    series: pl.DataFrame, *, min_fold_slots: int = 20
) -> dict[str, float | int | list]:
    """Fold-by-fold reading of one pooled IC series.

    ``sign_consistency`` is the share of folds pointing the series' **own** way, so a steadily
    inverted configuration is not disqualified for being inverted; ``positive_fold_share``
    is the share pointing up, which is the one to read when the sign itself is the claim.
    """
    if series.is_empty() or "fold" not in series.columns:
        return {"n_folds": 0}
    fold_means: list[float] = []
    for fold in sorted(series["fold"].drop_nulls().unique().to_list()):
        values = series.filter(pl.col("fold") == fold)["ic"].to_numpy()
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
    min_fold_slots: int = 20,
) -> dict:
    """Pooled IC with a Newey-West interval **and** the fold breakdown beside it.

    A pooled point estimate alone is not a reading; the two are returned together so no
    caller can print one without the other.
    """
    from ml4t.diagnostic.metrics import compute_ic_hac_stats

    if series.is_empty():
        return {"n_slots": 0}
    stats = compute_ic_hac_stats(series, ic_col="ic", label_horizon=hac_lag)
    return {
        "n_slots": series.height,
        "ic_mean": float(stats["mean_ic"]),
        "naive_t": float(stats["naive_t_stat"]),
        "hac_se": float(stats["hac_se"]),
        "hac_t": float(stats["t_stat"]),
        "hac_p": float(stats["p_value"]),
        "hac_lags": int(stats["effective_lags"]),
        **fold_ic_summary(series, min_fold_slots=min_fold_slots),
    }


def prediction_staleness_by_book(
    frame: pl.DataFrame,
    *,
    score_col: str = "prediction",
    book_col: str = "session",
    symbol_col: str = "symbol",
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """Share of a metal's consecutive scores identical to the one before, per session book.

    The screen ``05_evaluation`` runs on features, run on the model's own output. A score that
    repeats inside the book that would trade it cannot tell this session from the last, and a
    model can be perfectly non-constant on the pooled panel and constant inside one sleeve -
    the failure that stopped ``exness_fx_d1``'s eight ``rank_*`` columns, and the reason
    ``is_ny`` is flagged here. Run per book because a book is what a `SESSION_FILTER` trades.
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
