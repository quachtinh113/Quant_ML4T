"""Drift monitoring for ``exness_usidx_sess`` — ``26_mlops_governance/01_drift_monitoring.py``.

EVIDENCE BOUNDARY
-----------------
0 phase-5 survivors of 1,403 scored specs at K = 1,780; phase 6 not opened; the declared
holdout 2026-03-01 .. 2026-08-31 is **unscored**. There is no live stream. Every function here
returns ``not_computable`` with a reason when its input does not exist; none defaults to a
pass. :func:`assert_no_holdout` refuses any frame reaching into the holdout, and a flag alone
cannot open the boundary — the marker file has to exist.

**Read this before interpreting any alert.** Phase 3 found 0 of 84 columns clearing
Benjamini-Hochberg in either spec and phase 5 found no survivor. So a falling IC or hit rate
here is a **data-quality** signal — a feed changed, the D1 as-of join broke, the panel changed
shape — and is **not** evidence that an edge decayed. There is no edge on record to decay.

THREE THINGS THIS MODULE DOES DIFFERENTLY FROM ``exness_fx_d1``, EACH FOR A MEASURED REASON
--------------------------------------------------------------------------------------------
1. **The IC is a per-index TIME-SERIES coefficient, never a cross-sectional one.** A rank or
   Pearson correlation over TWO points is +1 or -1 whatever the numbers are. That is why
   ``05_evaluation`` and ``12_model_analysis`` were forked onto the time-series statistic
   (BOT.md phases 3 and 4), why the registry's own ``ic_mean`` is null on all 356 prediction
   sets, and why kill criterion (b) reads a hit rate rather than an IC. Computing the
   cross-sectional version here would produce a coin flip on every session.
2. **Everything is per spec.** ``intraday`` and ``overnight`` have different feature matrices
   under the same ``(timestamp, symbol)`` key (measured: the two ``financial.parquet`` digests
   differ). Pooling them averages two processes.
3. **The monitored columns are declared by FAMILY, not by |IC|.** ``exness_fx_d1`` watches its
   eight largest-|IC| PROCEED columns; on this bot that would be a ranking of noise and it
   would be exactly the generation-1 calibration the mentor forbade. The families come from
   ``setup.yaml::features.families`` — the register, declared before any IC was read — and the
   columns are resolved from each family's own ``pattern``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import yaml

logger = logging.getLogger("exness_usidx_sess.monitor.drift")

BOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MONITOR_CONFIG = BOT_DIR / "monitor" / "monitor_config.yaml"
DEFAULT_RISK_CONFIG = BOT_DIR / "deploy" / "risk_config.yaml"

Status = Literal["OK", "WATCH", "ALERT"]
SPECS = ("intraday", "overnight")


def load_monitor_config(path: Path | str | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or DEFAULT_MONITOR_CONFIG).read_text())


# ---------------------------------------------------------------------------------------
# The evidence boundary
# ---------------------------------------------------------------------------------------
def holdout_bounds(setup: dict[str, Any]) -> tuple[date, date]:
    ev = setup["evaluation"]
    return (
        date.fromisoformat(str(ev["holdout_start"])),
        date.fromisoformat(str(ev["holdout_end"])),
    )


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
    when ``holdout_scored_marker`` exists on disk. A flag alone cannot open the boundary: that
    is the whole difference between a rule and an intention.
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
        f"the frame reaches {latest}, inside the declared holdout starting {start}. The holdout "
        "has not been scored, so drift monitoring may not read it; the reference and current "
        "windows come from train/validation (monitor_config.yaml::drift.reference)."
    )


# ---------------------------------------------------------------------------------------
# Which columns are watched — resolved from the REGISTER, never from a result
# ---------------------------------------------------------------------------------------
def monitored_columns(
    available: list[str], setup: dict[str, Any], *, config: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """Family name -> the columns of ``available`` that family's declared pattern matches.

    The register is ``setup.yaml::features.families`` and each family carries a ``pattern``
    such as ``ret_*d|mom_skip_recent|accel_*``. Those are glob-ish alternations, so they are
    translated to a regular expression here rather than re-typed as a column list — a second
    copy of the register is exactly the drift the phase-3 review found (a private prefix table
    that matched nothing for 21 of 84 columns).
    """
    cfg = (config or load_monitor_config())["drift"]
    wanted = list(cfg["monitored_families"])
    cap = int(cfg.get("max_columns_per_family", 4))
    families = {str(f["name"]): str(f.get("pattern") or "") for f in setup["features"]["families"]}
    unknown = [name for name in wanted if name not in families]
    if unknown:
        raise KeyError(
            f"monitor_config.yaml::drift.monitored_families names {unknown}, which are not in "
            f"setup.yaml::features.families ({sorted(families)}). The register moved and the "
            "monitor did not."
        )
    out: dict[str, list[str]] = {}
    for name in wanted:
        pattern = families[name]
        if not pattern:
            out[name] = []
            continue
        regex = re.compile(
            "^(?:" + "|".join(part.replace("*", ".*") for part in pattern.split("|")) + ")$"
        )
        # Declared column order, never |IC| order.
        out[name] = [c for c in available if regex.match(c)][:cap]
    return out


# ---------------------------------------------------------------------------------------
# PSI and K-S (26/01:317-336, 384-388)
# ---------------------------------------------------------------------------------------
def compute_psi(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, epsilon: float = 1e-6
) -> tuple[float, np.ndarray]:
    """Population Stability Index, with bins taken from the REFERENCE only.

    Letting the current sample stretch the range concentrates reference mass into fewer bins
    under genuine drift and so *reduces* sensitivity; the outer infinite edges capture current
    values outside the reference range, so a regime shift to new extremes amplifies the PSI.
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
    family: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def to_numpy(frame: pl.DataFrame, column: str) -> np.ndarray:
    return np.asarray(frame.get_column(column).drop_nulls().to_numpy(), dtype=float)


def classify(psi: float, ks_pvalue: float, thresholds: dict[str, Any]) -> Status:
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
    columns_by_family: dict[str, list[str]],
    *,
    config: dict[str, Any] | None = None,
) -> list[DriftMetric]:
    """PSI and K-S for each monitored column, tagged with the family that declared it."""
    from scipy import stats

    thresholds = (config or load_monitor_config())["drift"]
    metrics: list[DriftMetric] = []
    for family, columns in columns_by_family.items():
        for feature in columns:
            if feature not in reference_frame.columns or feature not in current_frame.columns:
                continue
            reference = to_numpy(reference_frame, feature)
            current = to_numpy(current_frame, feature)
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
                    family=family,
                )
            )
    return sorted(metrics, key=lambda m: (-m.psi if not np.isnan(m.psi) else 0.0))


def prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
) -> DriftMetric:
    """The same two statistics on the model's own output distribution."""
    from scipy import stats

    thresholds = (config or load_monitor_config())["drift"]
    psi, _ = compute_psi(
        reference_scores, current_scores, n_bins=int(thresholds["psi"]["n_bins"])
    )
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
# Rolling signal quality — the PER-INDEX TIME-SERIES IC
# ---------------------------------------------------------------------------------------
def session_metrics(stream: pl.DataFrame) -> pl.DataFrame:
    """Per-session hit rate and MSE from a ``timestamp, symbol, score, actual`` stream.

    **No cross-sectional IC.** With two instruments a per-session correlation is +1 or -1 and
    carries no information; the IC of this bot is a TIME-SERIES statistic and is computed by
    :func:`time_series_ic` over a window, not per session.
    """
    required = {"timestamp", "symbol", "score", "actual"}
    missing = required - set(stream.columns)
    if missing:
        raise ValueError(f"live stream is missing {sorted(missing)}")
    return (
        stream.group_by("timestamp")
        .agg(
            ((pl.col("score") * pl.col("actual")) > 0).mean().alias("hit_rate"),
            ((pl.col("actual") - pl.col("score")) ** 2).mean().alias("mse"),
            pl.col("score").mean().alias("score_mean"),
            pl.col("score").std().alias("score_std"),
            pl.len().alias("n_assets"),
        )
        .sort("timestamp")
    )


def time_series_ic(stream: pl.DataFrame, *, window: int | None = None) -> dict[str, Any]:
    """The information coefficient of this bot: rank-correlate score with outcome WITHIN an
    index, over time, then pool.

    This is the statistic ``05_evaluation`` and ``12_model_analysis`` were forked onto after
    the registry's cross-sectional ``ic_mean`` came back null on all 356 prediction sets
    (BOT.md phases 3 and 4). ``window`` truncates to the last ``window`` sessions.
    """
    frame = stream.sort("timestamp")
    if window:
        keep = frame["timestamp"].unique().sort().tail(window)
        frame = frame.filter(pl.col("timestamp").is_in(keep))
    per_symbol: dict[str, float | None] = {}
    for symbol in sorted(frame["symbol"].unique().to_list()):
        rows = frame.filter(pl.col("symbol") == symbol)
        if rows.height < 3:
            per_symbol[symbol] = None
            continue
        score = rows["score"].rank().to_numpy().astype(float)
        actual = rows["actual"].rank().to_numpy().astype(float)
        if np.std(score) == 0 or np.std(actual) == 0:
            per_symbol[symbol] = None
            continue
        per_symbol[symbol] = float(np.corrcoef(score, actual)[0, 1])
    values = [v for v in per_symbol.values() if v is not None]
    return {
        "per_symbol": per_symbol,
        "pooled": float(np.mean(values)) if values else None,
        "n_sessions": int(frame["timestamp"].n_unique()),
        "definition": (
            "Spearman correlation of score and outcome WITHIN one index over time, averaged "
            "across indices. NOT a cross-sectional IC: over two instruments that is +/-1."
        ),
    }


def rolling_quality(daily: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Rolling means of hit rate and MSE (26/01:465-469)."""
    min_periods = max(1, window // 3)
    return daily.with_columns(
        pl.col("hit_rate")
        .rolling_mean(window, min_samples=min_periods)
        .alias(f"rolling_hit_rate_{window}"),
        pl.col("mse").rolling_mean(window, min_samples=min_periods).alias(f"rolling_mse_{window}"),
    )


def performance_alerts(
    daily: pl.DataFrame, *, stream: pl.DataFrame | None = None, config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Baseline (first window) against current (last window), with the declared bands.

    The bands are one and two standard errors of the statistic on a TWO-instrument book; the
    arithmetic is in ``monitor_config.yaml``. They are wide on purpose: on two names only a
    large move is readable at all.
    """
    thresholds = (config or load_monitor_config())["drift"]["performance"]
    window = int(thresholds["window_sessions"])
    alerts: list[dict[str, Any]] = []
    if daily.height < 2 * window:
        return [
            {
                "metric": "all",
                "status": "NOT_COMPUTABLE",
                "reason": (
                    f"{daily.height} sessions of live stream against the {2 * window} a "
                    "baseline-vs-current comparison needs; silence here means 'cannot judge', "
                    "not 'healthy'"
                ),
            }
        ]
    head, tail = daily.head(window), daily.tail(window)
    for metric, watch_key, alert_key, direction in (
        ("hit_rate", "hit_rate_watch_drop", "hit_rate_alert_drop", "down"),
        ("mse", "mse_watch_ratio", "mse_alert_ratio", "ratio"),
    ):
        base = float(head[metric].mean())
        now = float(tail[metric].mean())
        if direction == "down":
            drop = base - now
            status = (
                "ALERT"
                if drop >= float(thresholds[alert_key])
                else "WATCH"
                if drop >= float(thresholds[watch_key])
                else "OK"
            )
            alerts.append(
                {
                    "metric": metric,
                    "baseline": base,
                    "current": now,
                    "drop": drop,
                    "status": status,
                    "watch": float(thresholds[watch_key]),
                    "alert": float(thresholds[alert_key]),
                }
            )
        else:
            ratio = now / base if base else float("nan")
            status = (
                "ALERT"
                if ratio >= float(thresholds[alert_key])
                else "WATCH"
                if ratio >= float(thresholds[watch_key])
                else "OK"
            )
            alerts.append(
                {
                    "metric": metric,
                    "baseline": base,
                    "current": now,
                    "ratio": ratio,
                    "status": status,
                    "watch": float(thresholds[watch_key]),
                    "alert": float(thresholds[alert_key]),
                }
            )
    if stream is not None and stream.height:
        sessions = stream["timestamp"].unique().sort()
        first = time_series_ic(stream.filter(pl.col("timestamp").is_in(sessions.head(window))))
        last = time_series_ic(stream.filter(pl.col("timestamp").is_in(sessions.tail(window))))
        if first["pooled"] is not None and last["pooled"] is not None:
            drop = first["pooled"] - last["pooled"]
            alerts.append(
                {
                    "metric": "time_series_ic",
                    "baseline": first["pooled"],
                    "current": last["pooled"],
                    "drop": drop,
                    "status": (
                        "ALERT"
                        if drop >= float(thresholds["ic_alert_drop"])
                        else "WATCH"
                        if drop >= float(thresholds["ic_watch_drop"])
                        else "OK"
                    ),
                    "watch": float(thresholds["ic_watch_drop"]),
                    "alert": float(thresholds["ic_alert_drop"]),
                    "note": "per-index time-series IC; NOT cross-sectional",
                }
            )
    return alerts


# ---------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------
def drift_report(
    *,
    spec: str,
    setup: dict[str, Any],
    reference_features: pl.DataFrame | None = None,
    current_features: pl.DataFrame | None = None,
    stream: pl.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    holdout_scored_marker: Path | None = None,
) -> dict[str, Any]:
    """One drift report for ONE spec. Returns ``not_computable`` rather than a default pass."""
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
    cfg = config or load_monitor_config()
    report: dict[str, Any] = {
        "spec": spec,
        "evidence_boundary": {
            "phase5_survivors": 0,
            "trial_count_K": 1780,
            "holdout_scored": bool(
                holdout_scored_marker and Path(holdout_scored_marker).exists()
            ),
            "interpretation": (
                "a falling IC or hit rate here is a DATA-QUALITY signal, not evidence that an "
                "edge decayed: 0 of 84 features cleared Benjamini-Hochberg and 0 of 1,403 "
                "specs survived phase 5, so there is no edge on record to decay"
            ),
        },
        "reference_window": cfg["drift"]["reference"],
        "config_source": str(DEFAULT_MONITOR_CONFIG),
    }
    if reference_features is None or current_features is None:
        report["features"] = {
            "status": "not_computable",
            "reason": (
                "no live feature stream exists: this bot has never been deployed. The "
                "reference window is registered research data; the current window would come "
                "from deploy/state/<spec>/<date>/ run records."
            ),
        }
    else:
        for frame in (reference_features, current_features):
            assert_no_holdout(frame, setup, holdout_scored_marker=holdout_scored_marker)
        columns = monitored_columns(list(current_features.columns), setup, config=cfg)
        metrics = summarize_feature_drift(
            reference_features, current_features, columns, config=cfg
        )
        report["features"] = {
            "status": "computed",
            "columns_by_family": columns,
            "metrics": [m.as_dict() for m in metrics],
            "alerting": [m.name for m in metrics if m.status == "ALERT"],
        }
    if stream is None or stream.height == 0:
        report["performance"] = {
            "status": "not_computable",
            "reason": "no live stream: the bot has never traded, in shadow or otherwise",
        }
    else:
        assert_no_holdout(stream, setup, holdout_scored_marker=holdout_scored_marker)
        daily = session_metrics(stream)
        report["performance"] = {
            "status": "computed",
            "sessions": daily.height,
            "alerts": performance_alerts(daily, stream=stream, config=cfg),
            "time_series_ic": time_series_ic(
                stream, window=int(cfg["drift"]["performance"]["window_sessions"])
            ),
        }
    return report


__all__ = [
    "SPECS",
    "DriftMetric",
    "Status",
    "assert_no_holdout",
    "classify",
    "compute_psi",
    "drift_report",
    "holdout_bounds",
    "load_monitor_config",
    "monitored_columns",
    "performance_alerts",
    "prediction_drift",
    "rolling_quality",
    "session_metrics",
    "summarize_feature_drift",
    "time_series_ic",
    "to_numpy",
]
