"""Online drift detectors for exness_fx_d1 — ``26_mlops_governance/02``.

Two detectors on the live error stream, the notebook's implementations with one change that
matters on this bot:

``ADWINStyle``
    Two-window standardised mean-shift on the per-session MSE, with a cooldown so one regime
    change produces one alert.
``DDM``
    Drift Detection Method on a boolean "bad session" stream: it watches whether the
    *frequency* of bad sessions clusters, which is a different failure from the *size* of the
    error growing.

EVERY WINDOW IS COUNTED IN DECISION SESSIONS, AND HERE IS WHAT THAT COSTS
-------------------------------------------------------------------------
``exness_fx_d1`` takes **one decision per CME_FX session**, about 252 a year (BOT.md: 2,196
sessions in 8.5 years). The notebook's defaults are written for a daily stream too, but on an
intraday bot they would fill in days; here they fill in months. At the declared settings
(``monitor_config.yaml::online_detectors``):

* ADWIN-style needs ``2 x window_size`` = **42 sessions** (about two calendar months) before it
  can produce its first comparison, plus ``calibration_sessions`` = 126 sessions of warm-up
  that are fed in but may not alert. First possible alert: **session 127**, roughly six months
  after the first live decision.
* DDM needs ``min_samples`` = 20 sessions after the same 126-session calibration.

So **"the detectors did not trip" is not evidence that nothing drifted** during the first two
quarters of live running: they were not yet able to. :func:`detector_readiness` reports exactly
how many sessions are still needed, and :class:`DetectorSummary` carries ``ready`` so a
dashboard can say "not ready" instead of "OK". A monitoring layer that reports silence as
health on a stream too short to speak is worse than no monitoring layer.

The stress-lag machinery of the notebook (``nearest_stress_lag``) is kept because it is how one
judges whether a detector *led* or *lagged* a regime change; with no live stream yet there is
nothing to measure, and the function returns ``None`` rather than a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from bots.exness_fx_d1.monitor import load_monitor_config


class ADWINStyle:
    """Two-window mean-shift detector with a cooldown between alerts (26/02:312-339).

    Args:
        window_size: Sessions per window; the detector compares the last ``window_size``
            against the ``window_size`` before them.
        sensitivity: Standardised difference of means that counts as drift.
        cooldown_sessions: Sessions of silence after an alert.
    """

    def __init__(self, *, window_size: int, sensitivity: float, cooldown_sessions: int) -> None:
        self.window_size = int(window_size)
        self.sensitivity = float(sensitivity)
        self.cooldown_sessions = int(cooldown_sessions)
        self.window: list[float] = []
        self.cooldown_remaining = 0

    @property
    def sessions_needed(self) -> int:
        """How many more sessions before the detector can compare anything."""
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
        pooled = np.sqrt(prior.var(ddof=1) / len(prior) + recent.var(ddof=1) / len(recent) + 1e-12)
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
            "about 252 decision sessions a year; until the counts above are reached, "
            "'no alert' means 'cannot alert yet', not 'no drift'"
        ),
    }


def error_stream(daily: pl.DataFrame, *, error_rate_floor: float) -> pl.DataFrame:
    """``timestamp, mse, direction_error_rate, bad_session`` from per-session metrics.

    ``direction_error_rate`` is one minus the hit rate: the share of names whose sign the model
    got wrong on that session. ``bad_session`` is that rate above the declared floor, which is
    the boolean DDM consumes.
    """
    for column in ("timestamp", "mse", "hit_rate"):
        if column not in daily.columns:
            raise ValueError(f"per-session metrics are missing {column!r}")
    return daily.select(
        pl.col("timestamp"),
        pl.col("mse"),
        (1.0 - pl.col("hit_rate")).alias("direction_error_rate"),
    ).with_columns((pl.col("direction_error_rate") > float(error_rate_floor)).alias("bad_session"))


def nearest_stress_lag(alert_dates: list[date], stress_dates: list[date]) -> float | None:
    """Signed median lag in days: positive = the detector led, negative = it lagged (26/02:405)."""
    if not alert_dates or not stress_dates:
        return None
    lags = [
        (min(stress_dates, key=lambda s: abs((a - s).days)) - a).days for a in alert_dates
    ]
    return float(np.median(lags))


def run_detectors(
    daily: pl.DataFrame,
    *,
    stress_dates: list[date] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Feed the calibration window in silently, then run both detectors on the rest.

    Returns a dict with one :class:`DetectorSummary` per detector plus the readiness block, so
    a caller can never read "0 alerts" without also seeing whether an alert was possible.
    """
    block = (config or load_monitor_config())["online_detectors"]
    calibration_n = int(block["calibration_sessions"])
    stream = error_stream(daily, error_rate_floor=float(block["ddm"]["error_rate_floor"]))
    readiness = detector_readiness(stream.height, config={"online_detectors": block})

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
    rows = stream.to_dicts()
    calibration, monitoring = rows[:calibration_n], rows[calibration_n:]
    for row in calibration:
        adwin.update(float(row["mse"]))
        ddm.update(bool(row["bad_session"]))

    adwin_alerts: list[dict[str, Any]] = []
    ddm_alerts: list[dict[str, Any]] = []
    for row in monitoring:
        stamp = row["timestamp"]
        stamp = stamp.date() if hasattr(stamp, "date") else stamp
        if adwin.update(float(row["mse"])):
            adwin_alerts.append({"timestamp": str(stamp), "detector": "ADWIN-style", "value": float(row["mse"])})
        if ddm.update(bool(row["bad_session"])) == "drift":
            ddm_alerts.append(
                {"timestamp": str(stamp), "detector": "DDM", "value": float(row["direction_error_rate"])}
            )
            ddm.reset()

    def summarise(name: str, alerts: list[dict[str, Any]], ready: bool, needed: int) -> DetectorSummary:
        dates = [date.fromisoformat(a["timestamp"][:10]) for a in alerts]
        return DetectorSummary(
            detector=name,
            alert_count=len(alerts),
            first_alert=alerts[0]["timestamp"] if alerts else None,
            ready=ready,
            sessions_seen=stream.height,
            sessions_needed=needed,
            median_lag_sessions=nearest_stress_lag(dates, list(stress_dates or [])),
            alerts=alerts,
        )

    return {
        "readiness": readiness,
        "detectors": [
            summarise(
                "ADWIN-style",
                adwin_alerts,
                readiness["adwin_ready"],
                readiness["sessions_until_adwin_ready"],
            ).as_dict(),
            summarise(
                "DDM", ddm_alerts, readiness["ddm_ready"], readiness["sessions_until_ddm_ready"]
            ).as_dict(),
        ],
        "n_sessions": int(stream.height),
        "any_alert": bool(adwin_alerts or ddm_alerts),
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
