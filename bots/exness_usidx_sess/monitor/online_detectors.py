"""Online drift detection — ``26_mlops_governance/02_online_drift_detection.py``.

EVIDENCE BOUNDARY: 0 phase-5 survivors of 1,403 specs at K = 1,780, phase 6 not opened,
holdout 2026-03-01 .. 2026-08-31 unscored. There is no live stream, so every detector here
reports ``ready: false`` with the number of sessions it still needs. **Silence from a detector
means "cannot alert yet", never "no drift"** — that is what :func:`detector_readiness` exists
to say out loud.

TWO THINGS THIS BOT'S SHAPE FORCES, AND BOTH ARE RECORDED RATHER THAN SMOOTHED OVER
------------------------------------------------------------------------------------
1. **Every window is counted in DECISION SESSIONS OF ONE SPEC.** The bot takes two decisions a
   session, but they belong to two different books with two different models and two different
   registries; feeding both into one detector would halve the meaning of every window. One spec
   sees about 252 decisions a year, and the overnight spec historically fewer — 26 % of research
   sessions had no fill bar (0 % in 2026). At those rates a first ADWIN alert is possible only
   at session 168 and a first DDM alert at session 146.
2. **DDM has almost no resolution on a two-name book.** Its input is the share of instruments
   whose sign the model got wrong on a session, and with two instruments that share can only be
   0.0, 0.5 or 1.0. At the declared ``error_rate_floor`` of 0.52 a "bad session" is therefore
   exactly "both indices wrong". That is a real limitation and it is declared in
   ``monitor_config.yaml`` next to the floor, rather than papered over with a threshold that
   pretends to a resolution the data does not have.

Nothing in this bot's control plane acts on a detector alert. The breakers in
``deploy/risk_config.yaml`` do the acting; a detector is a prompt to look.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from bots.exness_usidx_sess.monitor.drift import SPECS, load_monitor_config


class ADWINStyle:
    """Two-window mean-shift detector with a cooldown between alerts (26/02:312-339)."""

    def __init__(self, *, window_size: int, sensitivity: float, cooldown_sessions: int) -> None:
        self.window_size = int(window_size)
        self.sensitivity = float(sensitivity)
        self.cooldown_sessions = int(cooldown_sessions)
        self.window: list[float] = []
        self.cooldown_remaining = 0

    @property
    def sessions_needed(self) -> int:
        """How many more sessions before the detector can compare anything at all."""
        return max(0, self.window_size * 2 - len(self.window))

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False
        if len(self.window) < self.window_size * 2:
            return False
        recent = np.asarray(self.window[-self.window_size :], dtype=float)
        prior = np.asarray(self.window[-self.window_size * 2 : -self.window_size], dtype=float)
        pooled = math.sqrt(
            prior.var(ddof=1) / len(prior) + recent.var(ddof=1) / len(recent) + 1e-12
        )
        if pooled == 0:
            return False
        if abs(recent.mean() - prior.mean()) / pooled >= self.sensitivity:
            self.cooldown_remaining = self.cooldown_sessions
            return True
        return False


class DDM:
    """Drift Detection Method on a bad-session indicator (26/02:349-379)."""

    def __init__(self, *, min_samples: int, warning_level: float, drift_level: float) -> None:
        self.min_samples = int(min_samples)
        self.warning_level = float(warning_level)
        self.drift_level = float(drift_level)
        self.reset()

    def reset(self) -> None:
        self.n_samples = 0
        self.n_errors = 0
        self.p_min = float("inf")
        self.s_min = float("inf")

    @property
    def sessions_needed(self) -> int:
        return max(0, self.min_samples - self.n_samples)

    def update(self, error: bool) -> str:
        self.n_samples += 1
        self.n_errors += int(bool(error))
        if self.n_samples < self.min_samples:
            return "normal"
        p = self.n_errors / self.n_samples
        s = math.sqrt(max(p * (1 - p) / self.n_samples, 1e-12))
        if p + s < self.p_min + self.s_min:
            self.p_min, self.s_min = p, s
        if p + s >= self.p_min + self.drift_level * self.s_min:
            return "drift"
        if p + s >= self.p_min + self.warning_level * self.s_min:
            return "warning"
        return "normal"


@dataclass
class DetectorSummary:
    """What a dashboard shows. ``ready`` is the field that stops silence reading as health."""

    detector: str
    spec: str
    alert_count: int
    first_alert: str | None
    ready: bool
    sessions_seen: int
    sessions_needed: int
    median_lag_sessions: float | None = None
    alerts: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = self.__dict__.copy()
        out["alerts"] = list(self.alerts)
        return out


def detector_readiness(n_sessions: int, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sessions still needed before either detector may alert, at the declared settings."""
    block = (config or load_monitor_config())["online_detectors"]
    calibration = int(block["calibration_sessions"])
    adwin_min = calibration + 2 * int(block["adwin"]["window_size"])
    ddm_min = calibration + int(block["ddm"]["min_samples"])
    return {
        "units": str(block["units"]),
        "sessions_seen": int(n_sessions),
        "calibration_sessions": calibration,
        "adwin_first_possible_alert_at_session": adwin_min,
        "ddm_first_possible_alert_at_session": ddm_min,
        "adwin_ready": n_sessions >= adwin_min,
        "ddm_ready": n_sessions >= ddm_min,
        "sessions_until_adwin_ready": max(0, adwin_min - int(n_sessions)),
        "sessions_until_ddm_ready": max(0, ddm_min - int(n_sessions)),
        "note": (
            "about 252 decision sessions a year PER SPEC (the overnight spec historically "
            "fewer: 26 % of research sessions had no fill bar, 0 % in 2026). Until the counts "
            "above are reached, 'no alert' means 'cannot alert yet', not 'no drift'."
        ),
    }


def error_stream(daily: pl.DataFrame, *, error_rate_floor: float) -> pl.DataFrame:
    """``timestamp, mse, direction_error_rate, bad_session`` from per-session metrics.

    On this bot ``direction_error_rate`` takes only the values 0.0, 0.5 and 1.0 (two
    instruments), so at a floor of 0.52 ``bad_session`` means "both indices wrong".
    """
    for column in ("timestamp", "mse", "hit_rate"):
        if column not in daily.columns:
            raise ValueError(f"per-session metrics are missing {column!r}")
    return daily.select(
        pl.col("timestamp"),
        pl.col("mse"),
        (1.0 - pl.col("hit_rate")).alias("direction_error_rate"),
    ).with_columns(
        (pl.col("direction_error_rate") > float(error_rate_floor)).alias("bad_session")
    )


def nearest_stress_lag(alert_dates: list[date], stress_dates: list[date]) -> float | None:
    """Signed median lag in days: positive = the detector led, negative = it lagged."""
    if not alert_dates or not stress_dates:
        return None
    lags = [(min(stress_dates, key=lambda s: abs((a - s).days)) - a).days for a in alert_dates]
    return float(np.median(lags))


def run_detectors(
    daily: pl.DataFrame | None,
    *,
    spec: str,
    stress_dates: list[date] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both detectors over one spec's per-session metrics.

    With ``daily=None`` (today's state) it returns ``not_ready`` with the reason and the
    readiness table, never a default pass.
    """
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
    cfg = config or load_monitor_config()
    block = cfg["online_detectors"]
    if daily is None or daily.height == 0:
        return {
            "spec": spec,
            "status": "not_ready",
            "reason": (
                "no live stream: this bot has never been deployed, in shadow or otherwise. "
                "0 phase-5 survivors at K = 1,780 and the holdout is unscored."
            ),
            "readiness": detector_readiness(0, config=cfg),
        }
    calibration = int(block["calibration_sessions"])
    adwin = ADWINStyle(
        window_size=int(block["adwin"]["window_size"]),
        sensitivity=float(block["adwin"]["sensitivity"]),
        cooldown_sessions=int(block["adwin"]["cooldown_sessions"]),
    )
    ddm = DDM(
        min_samples=int(block["ddm"]["min_samples"]),
        warning_level=float(block["ddm"]["warning_level"]),
        drift_level=float(block["ddm"]["drift_level"]),
    )
    errors = error_stream(daily, error_rate_floor=float(block["ddm"]["error_rate_floor"]))
    adwin_alerts: list[dict[str, Any]] = []
    ddm_alerts: list[dict[str, Any]] = []
    for i, row in enumerate(errors.iter_rows(named=True)):
        stamp = row["timestamp"]
        tripped = adwin.update(float(row["mse"]))
        state = ddm.update(bool(row["bad_session"]))
        if i < calibration:
            # Calibration sessions feed the detectors but may not raise: an alert during
            # calibration is the detector learning, not the world changing.
            continue
        if tripped:
            adwin_alerts.append({"timestamp": str(stamp), "mse": float(row["mse"])})
        if state == "drift":
            ddm_alerts.append({"timestamp": str(stamp), "state": state})
    n = errors.height

    def _dates(alerts: list[dict[str, Any]]) -> list[date]:
        out = []
        for a in alerts:
            value = a["timestamp"]
            out.append(date.fromisoformat(str(value)[:10]))
        return out

    readiness = detector_readiness(n, config=cfg)
    return {
        "spec": spec,
        "status": "computed",
        "readiness": readiness,
        "adwin": DetectorSummary(
            detector="adwin",
            spec=spec,
            alert_count=len(adwin_alerts),
            first_alert=adwin_alerts[0]["timestamp"] if adwin_alerts else None,
            ready=bool(readiness["adwin_ready"]),
            sessions_seen=n,
            sessions_needed=int(readiness["sessions_until_adwin_ready"]),
            median_lag_sessions=nearest_stress_lag(_dates(adwin_alerts), stress_dates or []),
            alerts=adwin_alerts,
        ).as_dict(),
        "ddm": DetectorSummary(
            detector="ddm",
            spec=spec,
            alert_count=len(ddm_alerts),
            first_alert=ddm_alerts[0]["timestamp"] if ddm_alerts else None,
            ready=bool(readiness["ddm_ready"]),
            sessions_seen=n,
            sessions_needed=int(readiness["sessions_until_ddm_ready"]),
            median_lag_sessions=nearest_stress_lag(_dates(ddm_alerts), stress_dates or []),
            alerts=ddm_alerts,
        ).as_dict(),
        "acts_on_alert": False,
        "note": (
            "nothing in this bot's control plane acts on a detector alert; the breakers in "
            "deploy/risk_config.yaml do the acting. A detector is a prompt to look."
        ),
    }


__all__ = [
    "DDM",
    "ADWINStyle",
    "DetectorSummary",
    "detector_readiness",
    "error_stream",
    "nearest_stress_lag",
    "run_detectors",
]
