"""Drift monitoring for exness_fx_d1 — ``26_mlops_governance/01_drift_monitoring.py``.

Three diagnostics, all read-only:

1. **Feature drift**: PSI and a two-sample K-S test per monitored feature, reference against
   current (``compute_psi`` is the notebook's function, bin edges from the reference only with
   outer infinities so a move to new extremes amplifies PSI rather than disappearing).
2. **Prediction drift**: the same two statistics on the score distribution.
3. **Rolling signal quality**: per-session IC, hit rate and MSE on the live stream, and their
   63-session rolling means against the stream's own launch baseline.

THE ONE DIFFERENCE FROM THE NOTEBOOK, AND IT IS THE IMPORTANT ONE
-----------------------------------------------------------------
Notebook 01 takes its reference from the last pre-holdout year and its current slice from
**inside the holdout**, because the case study it monitors had already scored its holdout. This
bot has not: ``2025-09-01 .. 2026-08-31`` is unscored evidence. So the reference window here is
the last validation year (``monitor_config.yaml::drift.reference``, 2024-09-02 .. 2025-08-28)
and :func:`assert_no_holdout` refuses any current slice that reaches into the holdout while
``deploy/state/holdout_scored.json`` is absent. Passing ``allow_holdout=True`` is possible only
after that marker exists.

A second thing worth saying out loud: this bot's phase-4 reading is that **no model shows a
validation IC distinguishable from zero** (BOT.md phase table). Rolling IC here is therefore a
*data-quality* instrument - it tells you the pipeline changed - and not a performance
instrument. Reading a falling IC as "the edge decayed" would presume an edge the evidence does
not show.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from bots.exness_fx_d1.monitor import load_monitor_config

Status = Literal["OK", "WATCH", "ALERT"]


# ---------------------------------------------------------------------------------------
# The evidence boundary
# ---------------------------------------------------------------------------------------
def holdout_bounds(setup: dict[str, Any]) -> tuple[date, date]:
    ev = setup["evaluation"]
    return date.fromisoformat(str(ev["holdout_start"])), date.fromisoformat(str(ev["holdout_end"]))


def assert_no_holdout(
    frame: pl.DataFrame,
    setup: dict[str, Any],
    *,
    allow_holdout: bool = False,
    holdout_scored_marker: Path | None = None,
    column: str = "timestamp",
) -> None:
    """Refuse a frame that reaches into the unscored holdout.

    ``allow_holdout=True`` is honoured only when the holdout has actually been scored, i.e.
    when ``holdout_scored_marker`` exists. A flag alone cannot open the evidence boundary.
    """
    start, _ = holdout_bounds(setup)
    if frame.height == 0:
        return
    latest = frame[column].max()
    latest = latest.date() if hasattr(latest, "date") else latest
    if latest < start:
        return
    scored = bool(holdout_scored_marker and Path(holdout_scored_marker).exists())
    if allow_holdout and scored:
        return
    raise ValueError(
        f"the frame reaches {latest}, inside the declared holdout starting {start}. "
        "The holdout has not been scored, so drift monitoring may not read it; the reference "
        "and current windows come from train/validation (monitor_config.yaml::drift.reference)."
    )


# ---------------------------------------------------------------------------------------
# PSI and K-S
# ---------------------------------------------------------------------------------------
def compute_psi(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, epsilon: float = 1e-6
) -> tuple[float, np.ndarray]:
    """Population Stability Index, bins from the reference only (26/01:317-336).

    Letting the current sample stretch the range concentrates reference mass into fewer bins
    under genuine drift and *reduces* sensitivity; the outer infinite edges capture current
    values outside the reference range so a regime shift to new extremes amplifies PSI.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return float("nan"), np.array([])
    inner = np.linspace(reference.min(), reference.max(), n_bins + 1)[1:-1]
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current) + epsilon
    bin_psi = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(np.sum(bin_psi)), bin_psi


@dataclass
class DriftMetric:
    name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    n_reference: int
    n_current: int
    status: Status

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def to_numpy(frame: pl.DataFrame, column: str) -> np.ndarray:
    return np.asarray(frame.get_column(column).drop_nulls().to_numpy(), dtype=float)


def classify(psi: float, ks_pvalue: float, thresholds: dict[str, Any]) -> Status:
    """The notebook's bands (26/01:384-388), read from ``monitor_config.yaml``."""
    psi_block, ks_block = thresholds["psi"], thresholds["ks"]
    if np.isnan(psi):
        return "WATCH"
    if psi >= psi_block["alert"]:
        return "ALERT"
    if psi >= psi_block["watch"] or (
        psi >= ks_block["psi_floor"] and ks_pvalue < ks_block["watch_pvalue"]
    ):
        return "WATCH"
    return "OK"


def summarize_feature_drift(
    reference_frame: pl.DataFrame,
    current_frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    config: dict[str, Any] | None = None,
) -> list[DriftMetric]:
    """PSI and K-S for each monitored feature, with the alert level."""
    from scipy import stats

    thresholds = (config or load_monitor_config())["drift"]
    metrics: list[DriftMetric] = []
    for feature in feature_columns:
        if feature not in reference_frame.columns or feature not in current_frame.columns:
            continue
        reference, current = to_numpy(reference_frame, feature), to_numpy(current_frame, feature)
        if reference.size == 0 or current.size == 0:
            continue
        psi, _ = compute_psi(reference, current, n_bins=int(thresholds["psi"]["n_bins"]))
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        metrics.append(
            DriftMetric(
                name=feature,
                psi=psi,
                ks_stat=float(ks_stat),
                ks_pvalue=float(ks_pvalue),
                reference_mean=float(reference.mean()),
                current_mean=float(current.mean()),
                n_reference=int(reference.size),
                n_current=int(current.size),
                status=classify(psi, float(ks_pvalue), thresholds),
            )
        )
    return sorted(metrics, key=lambda m: (-m.psi if not np.isnan(m.psi) else 0.0))


def prediction_drift(
    reference_scores: np.ndarray, current_scores: np.ndarray, *, config: dict[str, Any] | None = None
) -> DriftMetric:
    """The same two statistics on the model's output distribution."""
    from scipy import stats

    thresholds = (config or load_monitor_config())["drift"]
    psi, _ = compute_psi(reference_scores, current_scores, n_bins=int(thresholds["psi"]["n_bins"]))
    ks_stat, ks_pvalue = stats.ks_2samp(reference_scores, current_scores)
    return DriftMetric(
        name="prediction_score",
        psi=psi,
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
        reference_mean=float(np.mean(reference_scores)),
        current_mean=float(np.mean(current_scores)),
        n_reference=int(np.size(reference_scores)),
        n_current=int(np.size(current_scores)),
        status=classify(psi, float(ks_pvalue), thresholds),
    )


# ---------------------------------------------------------------------------------------
# Rolling signal quality on the live stream
# ---------------------------------------------------------------------------------------
def session_metrics(stream: pl.DataFrame) -> pl.DataFrame:
    """Per-session IC, hit rate and MSE from a ``timestamp, symbol, score, actual`` stream.

    The IC is the cross-sectional Pearson correlation the notebook uses; on five names it is a
    noisy statistic and BOT.md's kill criterion (b) deliberately reads the **hit rate**
    instead, which is why both are computed here and only the hit rate feeds a breaker.
    """
    required = {"timestamp", "symbol", "score", "actual"}
    missing = required - set(stream.columns)
    if missing:
        raise ValueError(f"live stream is missing {sorted(missing)}")
    return (
        stream.group_by("timestamp")
        .agg(
            pl.corr("score", "actual").alias("ic"),
            ((pl.col("score") * pl.col("actual")) > 0).mean().alias("hit_rate"),
            ((pl.col("actual") - pl.col("score")) ** 2).mean().alias("mse"),
            pl.col("score").mean().alias("score_mean"),
            pl.col("score").std().alias("score_std"),
            pl.len().alias("n_assets"),
        )
        .sort("timestamp")
    )


def rolling_quality(daily: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """63-session rolling means of IC, hit rate and MSE (26/01:465-469)."""
    min_periods = max(1, window // 3)
    return daily.with_columns(
        pl.col("ic").rolling_mean(window, min_samples=min_periods).alias(f"rolling_ic_{window}"),
        pl.col("hit_rate").rolling_mean(window, min_samples=min_periods).alias(f"rolling_hit_rate_{window}"),
        pl.col("mse").rolling_mean(window, min_samples=min_periods).alias(f"rolling_mse_{window}"),
    )


def performance_alerts(daily: pl.DataFrame, *, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Baseline (first window) against current (last window), with the declared bands."""
    thresholds = (config or load_monitor_config())["drift"]["performance"]
    window = int(thresholds["window_sessions"])
    if daily.height < 2:
        return []
    baseline, current = daily.head(window), daily.tail(window)
    rows: list[dict[str, Any]] = []

    def add(metric: str, base: float, cur: float, watch: float, alert: float, higher_is_worse: bool) -> None:
        if higher_is_worse:
            status = "ALERT" if cur > alert else ("WATCH" if cur > watch else "OK")
        else:
            status = "ALERT" if cur < alert else ("WATCH" if cur < watch else "OK")
        rows.append(
            {
                "metric": metric,
                "baseline": base,
                "current": cur,
                "watch_threshold": watch,
                "alert_threshold": alert,
                "status": status,
            }
        )

    base_ic, cur_ic = float(baseline["ic"].mean()), float(current["ic"].mean())
    add(
        "rolling_ic",
        base_ic,
        cur_ic,
        base_ic - float(thresholds["ic_watch_drop"]),
        base_ic - float(thresholds["ic_alert_drop"]),
        higher_is_worse=False,
    )
    base_hr, cur_hr = float(baseline["hit_rate"].mean()), float(current["hit_rate"].mean())
    add(
        "rolling_hit_rate",
        base_hr,
        cur_hr,
        base_hr - float(thresholds["hit_rate_watch_drop"]),
        base_hr - float(thresholds["hit_rate_alert_drop"]),
        higher_is_worse=False,
    )
    base_mse, cur_mse = float(baseline["mse"].mean()), float(current["mse"].mean())
    add(
        "rolling_mse",
        base_mse,
        cur_mse,
        base_mse * float(thresholds["mse_watch_ratio"]),
        base_mse * float(thresholds["mse_alert_ratio"]),
        higher_is_worse=True,
    )
    return rows


# ---------------------------------------------------------------------------------------
# The live stream
# ---------------------------------------------------------------------------------------
def build_live_stream(run_records: list[dict[str, Any]], panel: pl.DataFrame) -> pl.DataFrame:
    """``timestamp, symbol, score, actual`` from the deployment loop's run records.

    ``run_records`` are the JSON documents ``deploy/state/<date>/run_*.json`` holds; each one
    carries the decision date and the latest cross-section the loop scored. ``panel`` is the
    session panel (``_features.load_session_panel``); the realised return is the next session's
    close-to-close move on the same pair, which is exactly the ``fwd_ret_1d`` construction the
    labels use, computed here from prices so a live stream needs no label file.
    """
    rows: list[dict[str, Any]] = []
    for record in run_records:
        cross = (record.get("steps", {}).get("5_predict") or {}).get("latest_cross_section") or []
        for entry in cross:
            stamp = entry.get("timestamp")
            rows.append(
                {
                    "timestamp": date.fromisoformat(str(stamp)[:10]),
                    "symbol": entry["symbol"],
                    "score": float(entry["score"]),
                }
            )
    if not rows:
        return pl.DataFrame(schema={"timestamp": pl.Date, "symbol": pl.Utf8, "score": pl.Float64, "actual": pl.Float64})
    scores = pl.DataFrame(rows).unique(subset=["timestamp", "symbol"], keep="last")
    realised = (
        panel.sort(["symbol", "timestamp"])
        .with_columns(
            (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("actual")
        )
        .select(["timestamp", "symbol", "actual"])
    )
    return scores.join(realised, on=["timestamp", "symbol"], how="left").drop_nulls("actual").sort(
        ["timestamp", "symbol"]
    )


# ---------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------
def drift_report(
    *,
    reference_features: pl.DataFrame,
    current_features: pl.DataFrame,
    setup: dict[str, Any],
    live_stream: pl.DataFrame | None = None,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    allow_holdout: bool = False,
    holdout_scored_marker: Path | None = None,
) -> dict[str, Any]:
    """One dashboard-shaped dict: feature drift, prediction drift, rolling quality, alerts."""
    cfg = config or load_monitor_config()
    block = cfg["drift"]
    for frame in (reference_features, current_features):
        assert_no_holdout(
            frame, setup, allow_holdout=allow_holdout, holdout_scored_marker=holdout_scored_marker
        )
    metrics = summarize_feature_drift(
        reference_features, current_features, list(block["monitored_features"]), config=cfg
    )
    report: dict[str, Any] = {
        "reference_window": dict(block["reference"]),
        "current_window_sessions": int(block["current_window_sessions"]),
        "feature_drift": [m.as_dict() for m in metrics],
        "max_psi": max((m.psi for m in metrics), default=None),
        "n_alert": sum(1 for m in metrics if m.status == "ALERT"),
        "n_watch": sum(1 for m in metrics if m.status == "WATCH"),
        "holdout_read": False,
    }
    if reference_scores is not None and current_scores is not None:
        report["prediction_drift"] = prediction_drift(reference_scores, current_scores, config=cfg).as_dict()
    if live_stream is not None and live_stream.height:
        assert_no_holdout(
            live_stream, setup, allow_holdout=allow_holdout, holdout_scored_marker=holdout_scored_marker
        )
        daily = session_metrics(live_stream)
        window = int(block["performance"]["window_sessions"])
        report["rolling_quality"] = rolling_quality(daily, window=window).to_dicts()
        report["performance_alerts"] = performance_alerts(daily, config=cfg)
        report["n_sessions_in_stream"] = int(daily.height)
    report["status"] = (
        "ALERT"
        if report["n_alert"] or any(a["status"] == "ALERT" for a in report.get("performance_alerts", []))
        else ("WATCH" if report["n_watch"] or any(a["status"] == "WATCH" for a in report.get("performance_alerts", [])) else "OK")
    )
    return report


__all__ = [
    "DriftMetric",
    "assert_no_holdout",
    "build_live_stream",
    "classify",
    "compute_psi",
    "drift_report",
    "holdout_bounds",
    "performance_alerts",
    "prediction_drift",
    "rolling_quality",
    "session_metrics",
    "summarize_feature_drift",
    "to_numpy",
]
