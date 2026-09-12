"""Feature, prediction and performance drift - ``26_mlops_governance/01_drift_monitoring.py``.

Adapted to this bot in three ways, each of which changes what a number means:

1. **The reference window is validation, never the holdout.** Notebook 01 streams a scored
   holdout because its case study had one. This bot's holdout (2025-09-01 .. 2026-08-31) is
   unscored *and* carries a contamination flag, so reading it here would both burn the evidence
   boundary and compare live data against a window a person has already seen.
   :func:`assert_no_holdout` enforces it on every frame that enters.
2. **Every window is counted in decision sessions**, because this bot decides twice a weekday.
3. **Every statistic is reported per sleeve as well as pooled.** London and New York are
   separate books (``PRICE_GRID_DECLARATION.md`` section 3.4), and a hit rate that is fine
   pooled and broken in one sleeve is a sleeve problem a pooled monitor cannot see. It is the
   same reason ``05_evaluation`` runs its staleness screen per session book, one book earlier
   than ``exness_fx_d1`` learnt to.

**There is no rolling IC in this module and that is deliberate.** Over two metals a
cross-sectional rank correlation takes only the values +/-1, and the registry declines to
compute one at all (``min_obs=5`` against a panel two wide,
``case_studies/utils/registry/metrics.py:113``). ``BOT.md`` kill criterion (b) says the same
thing - *"IC tren hai ten khong phai mot thong ke doc duoc"* - and uses hit rate instead.

Nothing here carries a threshold: every number arrives from ``monitor_config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import polars as pl

from bots.exness_gold_sess.monitor import load_monitor_config

Status = Literal["ok", "watch", "alert", "not_computable"]

__all__ = [
    "DriftMetric",
    "assert_no_holdout",
    "compute_psi",
    "drift_report",
    "feature_drift",
    "holdout_bounds",
    "performance_alerts",
    "prediction_drift",
    "rolling_hit_rate",
    "sleeve_metrics",
]


# ---------------------------------------------------------------------------------------
# The evidence boundary
# ---------------------------------------------------------------------------------------
def holdout_bounds(setup: dict[str, Any]) -> tuple[date, date]:
    """``(holdout_start, holdout_end)`` from ``setup.yaml::evaluation``."""
    evaluation = setup["evaluation"]
    return (
        date.fromisoformat(str(evaluation["holdout_start"])),
        date.fromisoformat(str(evaluation["holdout_end"])),
    )


def assert_no_holdout(
    frame: pl.DataFrame, setup: dict[str, Any], *, what: str, date_col: str = "timestamp"
) -> None:
    """Raise if *frame* contains a row inside the declared holdout window.

    Called on every reference frame this module reads. The holdout is scored **once**, by
    ``17_holdout_predictions`` / ``18_holdout_backtest``; a monitoring module that quietly used
    it as a reference distribution would burn it without anyone deciding to.
    """
    if frame.is_empty() or date_col not in frame.columns:
        return
    start, end = holdout_bounds(setup)
    inside = frame.filter(
        pl.col(date_col).dt.date().is_between(pl.lit(start), pl.lit(end), closed="both")
    )
    if inside.height:
        raise ValueError(
            f"{what} contains {inside.height} rows inside the holdout window {start}..{end}. "
            "The holdout is scored once by 17_holdout_predictions / 18_holdout_backtest and is "
            "never a drift reference. It additionally carries a contamination flag "
            "(setup.yaml::evaluation.legacy_v9_contaminated_window)."
        )


# ---------------------------------------------------------------------------------------
# Distributional drift
# ---------------------------------------------------------------------------------------
def compute_psi(reference: np.ndarray, live: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index between two samples, on quantile bins of the reference.

    Quantile bins rather than equal-width ones: gold and silver returns are heavy-tailed, and
    equal-width bins put almost everything in the middle bucket and make the PSI insensitive to
    exactly the tail behaviour worth watching. Returns ``nan`` when either sample is too small
    or the reference is constant - ``nan`` is reported as ``not_computable``, never as 0.
    """
    reference = reference[np.isfinite(reference)]
    live = live[np.isfinite(live)]
    if len(reference) < bins * 2 or len(live) < bins or np.ptp(reference) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    live_share = np.histogram(live, bins=edges)[0] / len(live)
    floor = 1e-6
    ref_share = np.clip(ref_share, floor, None)
    live_share = np.clip(live_share, floor, None)
    return float(np.sum((live_share - ref_share) * np.log(live_share / ref_share)))


@dataclass
class DriftMetric:
    """One column, one sleeve, one window."""

    feature: str
    sleeve: str
    n_reference: int
    n_live: int
    psi: float
    ks_statistic: float
    ks_pvalue: float
    status: Status
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _classify(psi: float, ks_pvalue: float, config: dict[str, Any]) -> Status:
    if not np.isfinite(psi):
        return "not_computable"
    if psi >= config["psi"]["alert"] or ks_pvalue < config["ks_pvalue_alert"]:
        return "alert"
    if psi >= config["psi"]["watch"]:
        return "watch"
    return "ok"


def feature_drift(
    reference: pl.DataFrame,
    live: pl.DataFrame,
    *,
    setup: dict[str, Any],
    config: dict[str, Any] | None = None,
    sleeve_col: str = "session",
) -> pl.DataFrame:
    """PSI and Kolmogorov-Smirnov per monitored column, pooled and per sleeve.

    A column listed under ``drift.constant_inside_a_book`` is reported as ``not_computable``
    inside a sleeve with the reason attached rather than as an alert: ``is_ny`` is constant
    inside ``london`` and inside ``ny`` **by design**, and an alert every window for a column
    that is behaving correctly is how a monitoring channel gets ignored.
    """
    from scipy.stats import ks_2samp

    config = (config or load_monitor_config())["drift"]
    assert_no_holdout(reference, setup, what="the drift reference window")

    watched = [c for c in config["monitored_features"] if c in reference.columns]
    constant = set(config.get("constant_inside_a_book", []))
    sleeves = ["pooled"]
    if sleeve_col in live.columns:
        sleeves += sorted(live[sleeve_col].unique().to_list())

    rows: list[DriftMetric] = []
    for sleeve in sleeves:
        ref = reference if sleeve == "pooled" else reference.filter(pl.col(sleeve_col) == sleeve)
        cur = live if sleeve == "pooled" else live.filter(pl.col(sleeve_col) == sleeve)
        for column in watched:
            if sleeve != "pooled" and column in constant:
                rows.append(
                    DriftMetric(
                        column, sleeve, ref.height, cur.height, float("nan"), float("nan"),
                        float("nan"), "not_computable",
                        "constant inside a session book by design (declared in "
                        "monitor_config.yaml::drift.constant_inside_a_book)",
                    )
                )
                continue
            ref_values = ref[column].drop_nulls().to_numpy().astype(float)
            live_values = cur[column].drop_nulls().to_numpy().astype(float)
            if len(live_values) < config["min_live_decisions"]:
                rows.append(
                    DriftMetric(
                        column, sleeve, len(ref_values), len(live_values), float("nan"),
                        float("nan"), float("nan"), "not_computable",
                        f"live window has {len(live_values)} observations, below the declared "
                        f"minimum of {config['min_live_decisions']}",
                    )
                )
                continue
            psi = compute_psi(ref_values, live_values)
            ks = ks_2samp(ref_values, live_values)
            rows.append(
                DriftMetric(
                    column, sleeve, len(ref_values), len(live_values), psi,
                    float(ks.statistic), float(ks.pvalue),
                    _classify(psi, float(ks.pvalue), config),
                )
            )
    return pl.DataFrame([r.as_dict() for r in rows])


def prediction_drift(
    reference_scores: np.ndarray,
    live_scores: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Has the model's own score distribution moved away from its validation distribution?"""
    config = (config or load_monitor_config())["drift"]["prediction"]
    psi = compute_psi(np.asarray(reference_scores, dtype=float), np.asarray(live_scores, dtype=float))
    return {
        "psi": psi,
        "threshold": config["psi_alert"],
        "status": (
            "not_computable" if not np.isfinite(psi)
            else "alert" if psi >= config["psi_alert"]
            else "ok"
        ),
        "n_reference": int(len(reference_scores)),
        "n_live": int(len(live_scores)),
    }


def score_staleness_by_sleeve(
    live: pl.DataFrame,
    *,
    score_col: str = "prediction",
    sleeve_col: str = "session",
    symbol_col: str = "symbol",
    config: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Share of consecutive live scores identical to the one before, per metal and sleeve.

    The screen ``05_evaluation`` runs on features, run on the live score. A model whose score
    repeats inside the book that trades it has stopped distinguishing sessions, whatever its
    pooled statistics say.
    """
    ceiling = (config or load_monitor_config())["drift"]["prediction"]["staleness_ceiling"]
    rows = []
    sleeves = ["pooled"] + (
        sorted(live[sleeve_col].unique().to_list()) if sleeve_col in live.columns else []
    )
    for sleeve in sleeves:
        part = live if sleeve == "pooled" else live.filter(pl.col(sleeve_col) == sleeve)
        chronological = part.sort([symbol_col, "timestamp"])
        unchanged = chronological.select(
            (pl.col(score_col) == pl.col(score_col).shift(1).over(symbol_col)).sum()
        ).item()
        comparable = chronological.select(
            pl.col(score_col).shift(1).over(symbol_col).is_not_null().sum()
        ).item()
        staleness = float(unchanged) / max(int(comparable), 1)
        rows.append(
            {
                "sleeve": sleeve,
                "rows": part.height,
                "staleness": staleness,
                "ceiling": ceiling,
                "status": "alert" if staleness > ceiling else "ok",
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------------------
def sleeve_metrics(stream: pl.DataFrame, *, sleeve_col: str = "session") -> pl.DataFrame:
    """Per decision and sleeve: how many symbols traded, the realised return, and a hit flag.

    A "hit" is a decision whose realised return had the sign the position had. On two metals
    that is the readable statistic; a rank correlation is not (see the module docstring).
    """
    return (
        stream.with_columns(
            hit=((pl.col("position").sign() * pl.col("realised_return")) > 0).cast(pl.Int8)
        )
        .group_by(["timestamp", sleeve_col], maintain_order=True)
        .agg(
            symbols=pl.len(),
            realised=pl.col("realised_return").sum(),
            hits=pl.col("hit").sum(),
        )
        .sort("timestamp")
    )


def rolling_hit_rate(
    decisions: pl.DataFrame, *, window: int, sleeve_col: str = "session"
) -> pl.DataFrame:
    """Rolling hit rate over *window* DECISIONS, per sleeve and pooled.

    ``window`` is a count of decisions, not of days: this bot decides twice a weekday, so 63
    decisions is about six trading weeks. The FX bot's identically named window is twice as
    long in wall-clock time.
    """
    frames = []
    for sleeve in ["pooled"] + sorted(decisions[sleeve_col].unique().to_list()):
        part = (
            decisions if sleeve == "pooled" else decisions.filter(pl.col(sleeve_col) == sleeve)
        ).sort("timestamp")
        frames.append(
            part.with_columns(
                pl.lit(sleeve).alias("sleeve"),
                (
                    pl.col("hits").rolling_sum(window) / pl.col("symbols").rolling_sum(window)
                ).alias("hit_rate"),
                pl.col("symbols").rolling_sum(window).alias("decisions_in_window"),
            ).select("timestamp", "sleeve", "hit_rate", "decisions_in_window")
        )
    return pl.concat(frames).sort(["sleeve", "timestamp"])


def performance_alerts(
    decisions: pl.DataFrame, *, config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Watch-level alerts on the rolling hit rate. The STOP is a breaker, not this.

    ``deploy/risk_config.yaml::breakers.hit_rate`` is the kill criterion: below 0.5 for 21
    consecutive sessions. This function fires earlier and softer, at the ``hit_rate_watch``
    level, so a person sees the deterioration before the breaker acts on it.
    """
    performance = (config or load_monitor_config())["drift"]["performance"]
    window = int(performance["hit_rate_window_decisions"])
    if decisions.height < window:
        return [
            {
                "metric": "hit_rate",
                "status": "not_computable",
                "reason": f"{decisions.height} decisions recorded, {window} needed",
            }
        ]
    rolling = rolling_hit_rate(decisions, window=window)
    alerts = []
    for sleeve in sorted(rolling["sleeve"].unique().to_list()):
        latest = rolling.filter(pl.col("sleeve") == sleeve).drop_nulls("hit_rate").tail(1)
        if latest.is_empty():
            continue
        value = float(latest["hit_rate"][0])
        alerts.append(
            {
                "metric": "hit_rate",
                "sleeve": sleeve,
                "window_decisions": window,
                "value": value,
                "watch_level": performance["hit_rate_watch"],
                "status": "watch" if value < performance["hit_rate_watch"] else "ok",
                "note": "the STOP is deploy/risk_config.yaml::breakers.hit_rate, not this alert",
            }
        )
    return alerts


# ---------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------
def drift_report(
    reference: pl.DataFrame | None,
    live: pl.DataFrame | None,
    decisions: pl.DataFrame | None,
    *,
    setup: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One report. Reports ``not_computable`` with a reason rather than defaulting to a pass.

    Today every branch takes the ``not_computable`` path, because phase 4 has fitted no model
    and there is no live stream to compare against anything. That is the correct output, and it
    is the reason the function returns a status per section rather than a single boolean.
    """
    config = config or load_monitor_config()
    report: dict[str, Any] = {"bot_id": config["bot_id"], "cadence": config["cadence"]}

    if reference is None or live is None or reference.is_empty() or live.is_empty():
        report["features"] = {
            "status": "not_computable",
            "reason": "no live feature stream exists: phase 4 has not fitted a model, so the "
            "deployment loop has never produced a scored decision (bots/exness_gold_sess/BOT.md, "
            "phase status)",
        }
    else:
        report["features"] = feature_drift(reference, live, setup=setup, config=config).to_dicts()

    if decisions is None or decisions.is_empty():
        report["performance"] = {
            "status": "not_computable",
            "reason": "no decisions have been recorded; deploy/state/runs/ is empty",
        }
    else:
        report["performance"] = performance_alerts(decisions, config=config)

    report["not_monitored"] = config.get("not_monitored", [])
    return report
