"""exness_gold_sess: the scale-independent staleness screen (generation 2, block 3).

Pre-declared in ``bots/exness_gold_sess/BOT.md`` ("Pre-declaration: a scale-independent
staleness screen", written 2026-09-08 before one number of it was computed) and installed by
``GEN2_DECLARATION_2026-09-10.md`` ruling 3.2. It stands **beside** the bit-identity screen of
``05_evaluation.py`` (shared with four sibling case studies, not edited) and does not replace it:
``05`` prints this table as a second table and carries its verdict as one more ledger column;
the ledger's ``decision`` column is still the first screen's.

The defect it addresses is a *definition*: the first screen asks "does this column repeat the
same float?", this one asks "does this column vary?". Three statistics, every one a ratio, so a
column measured in units of ``1e10`` and the same column in units of 1 get the same verdict:

==========================================  =========================  ==================
statistic                                   threshold (fixed 2026-09-08)  verdict
==========================================  =========================  ==================
``n_unique / n``, pooled and per            ``< 0.01``                 STOP
(fold, symbol) cell (median over cells)
``CV = std / |mean|`` per (fold, symbol)    ``< 1e-6``                 STOP
cell, median over cells                     ``1e-6 <= CV < 1e-4``      REVISE
share of rows equal to the modal value,     ``> 0.20``                 REVISE
pooled                                      ``> 0.50``                 STOP
==========================================  =========================  ==================

Zero-mean safe: a cell whose ``|mean| < std`` has no meaningful CV (a z-score, a centred
return) and is left out of the CV median; when no cell remains the CV test is not applied and
only the other two run. Declared with the thresholds so it cannot be chosen later.

One aggregation is fixed here, before the first run, because the pre-declaration's table did
not spell it out for the first row: the per-cell ``n_unique / n`` is summarised by its
**median** over cells, the same summary the CV row uses, and the pooled ratio is tested beside
it; the column STOPs if either is below the threshold.

It is a screen, not a gate on results. It reads feature columns, the fold id and the symbol,
and nothing else - no outcome, no score, no statistic of ``05``'s inference - and
``tests/test_scale_free_screen.py`` asserts that from this module's own source (the import ban
pattern of ``_report_phase5.py``). It may be run before any outcome is scored and may never be
consulted after one is known.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

#: The thresholds of the pre-declaration, keyed exactly as ``setup.yaml::features.scale_free_screen``
#: declares them. Read from the configuration by the stage; the defaults here are the same
#: numbers so a test can run the screen without a configuration and get the declared verdicts.
DECLARED_THRESHOLDS: dict[str, float | str] = {
    "n_unique_ratio_stop": 0.01,
    "cv_stop": 1e-6,
    "cv_revise": 1e-4,
    "modal_share_revise": 0.20,
    "modal_share_stop": 0.50,
    "declared_on": "2026-09-08",
}
VERDICTS = ("PASS", "REVISE", "STOP")
_RANK = {v: i for i, v in enumerate(VERDICTS)}


def _worst(*verdicts: str) -> str:
    return max(verdicts, key=_RANK.__getitem__)


def screen_one_column(
    frame: pl.DataFrame,
    column: str,
    *,
    thresholds: Mapping[str, float | str] | None = None,
    fold_col: str = "fold",
    symbol_col: str = "symbol",
) -> dict:
    """The three statistics and the verdict for one column. Nulls are not values and are dropped.

    Returns a flat record; :func:`scale_free_screen` stacks them. A column with fewer than two
    non-null rows cannot vary and STOPs on the unique-ratio row with ``n`` reported.
    """
    t = dict(DECLARED_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    rows = frame.select(fold_col, symbol_col, column).drop_nulls(column)
    n = rows.height
    record: dict = {"feature": column, "n": n, "n_cells": 0}
    if n < 2:
        record.update(
            n_unique_ratio_pooled=float(n),
            n_unique_ratio_cell_median=None,
            cv_cells_applied=0,
            cv_cell_median=None,
            modal_share=1.0 if n else None,
            modal_value=None,
            verdict="STOP",
            reasons="n_unique_ratio<threshold (fewer than two non-null rows)",
        )
        return record

    values = rows[column]
    pooled_ratio = values.n_unique() / n
    cells = (
        rows.group_by([fold_col, symbol_col])
        .agg(
            pl.len().alias("_n"),
            pl.col(column).n_unique().alias("_nu"),
            pl.col(column).mean().alias("_mean"),
            pl.col(column).std(ddof=1).alias("_std"),
        )
        .with_columns((pl.col("_nu") / pl.col("_n")).alias("_ratio"))
    )
    cell_ratio_median = float(cells["_ratio"].median())
    # zero-mean safe: |mean| < std -> the CV of that cell is not meaningful and is not applied
    cv_cells = cells.filter(
        pl.col("_std").is_not_null() & (pl.col("_mean").abs() >= pl.col("_std"))
    ).with_columns((pl.col("_std") / pl.col("_mean").abs()).alias("_cv"))
    cv_median = float(cv_cells["_cv"].median()) if cv_cells.height else None
    counts = values.value_counts().sort(["count", column], descending=[True, False])
    modal_value = counts[column][0]
    modal_share = float(counts["count"][0]) / n

    reasons: list[str] = []
    verdict = "PASS"
    if pooled_ratio < float(t["n_unique_ratio_stop"]) or cell_ratio_median < float(
        t["n_unique_ratio_stop"]
    ):
        verdict = _worst(verdict, "STOP")
        reasons.append(
            f"n_unique_ratio<{t['n_unique_ratio_stop']} (pooled {pooled_ratio:.4g}, cell median "
            f"{cell_ratio_median:.4g})"
        )
    if cv_median is not None:
        if cv_median < float(t["cv_stop"]):
            verdict = _worst(verdict, "STOP")
            reasons.append(f"cv<{t['cv_stop']:g} (cell median {cv_median:.3g})")
        elif cv_median < float(t["cv_revise"]):
            verdict = _worst(verdict, "REVISE")
            reasons.append(f"cv<{t['cv_revise']:g} (cell median {cv_median:.3g})")
    if modal_share > float(t["modal_share_stop"]):
        verdict = _worst(verdict, "STOP")
        reasons.append(f"modal_share>{t['modal_share_stop']} ({modal_share:.4f})")
    elif modal_share > float(t["modal_share_revise"]):
        verdict = _worst(verdict, "REVISE")
        reasons.append(f"modal_share>{t['modal_share_revise']} ({modal_share:.4f})")

    record.update(
        n_cells=cells.height,
        n_unique_ratio_pooled=float(pooled_ratio),
        n_unique_ratio_cell_median=cell_ratio_median,
        cv_cells_applied=int(cv_cells.height),
        cv_cell_median=cv_median,
        modal_share=modal_share,
        modal_value=float(modal_value) if modal_value is not None else None,
        verdict=verdict,
        reasons="; ".join(reasons),
    )
    return record


SCREEN_SCHEMA = {
    "feature": pl.String,
    "n": pl.Int64,
    "n_cells": pl.Int64,
    "n_unique_ratio_pooled": pl.Float64,
    "n_unique_ratio_cell_median": pl.Float64,
    "cv_cells_applied": pl.Int64,
    "cv_cell_median": pl.Float64,
    "modal_share": pl.Float64,
    "modal_value": pl.Float64,
    "verdict": pl.String,
    "reasons": pl.String,
}


def scale_free_screen(
    frame: pl.DataFrame,
    columns: Sequence[str],
    *,
    thresholds: Mapping[str, float | str] | None = None,
    fold_col: str = "fold",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Run :func:`screen_one_column` over ``columns`` and return one row per column.

    ``frame`` must carry ``fold_col`` and ``symbol_col`` - the (fold, symbol) cell is the unit the
    per-cell statistics are taken in, because a column that varies only *between* folds (a
    weak time index, which is what ``kalman_smoothness`` turned out to be) shows no variation
    inside one. Rows are sorted STOP first, then by feature name.
    """
    missing = [c for c in (fold_col, symbol_col) if c not in frame.columns]
    if missing:
        raise KeyError(f"the screen needs the cell keys {missing} on the frame")
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise KeyError(f"columns not on the frame: {absent}")
    records = [
        screen_one_column(
            frame, c, thresholds=thresholds, fold_col=fold_col, symbol_col=symbol_col
        )
        for c in columns
    ]
    table = pl.DataFrame(records, schema=SCREEN_SCHEMA)
    return table.with_columns(
        pl.col("verdict").replace_strict(_RANK, return_dtype=pl.Int64).alias("_rank")
    ).sort(["_rank", "feature"], descending=[True, False]).drop("_rank")
