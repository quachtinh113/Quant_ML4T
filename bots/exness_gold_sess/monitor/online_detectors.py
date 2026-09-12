"""Online drift detection on the error stream - ``26_mlops_governance/02``.

Two detectors, both streaming, both on the same binary error series (*"this decision lost
money"*):

``ADWINStyle``
    An adaptive-window mean-shift test. It keeps a growing window and, at each new observation,
    looks for a split point where the two halves' means differ by more than a Hoeffding bound
    at confidence ``delta``. When it finds one, the older half is dropped: the window adapts to
    the change instead of averaging across it.
``DDM``
    Drift Detection Method. It tracks the running error rate and its standard deviation, keeps
    the minimum of ``p + s`` seen so far, and warns at ``p + s > p_min + warning_sigma * s_min``
    and declares drift at ``drift_sigma``. It is the cheaper, older test and it reacts to a
    slow deterioration that ADWIN's window can absorb.

**Every window is counted in DECISIONS, not days**, because this bot decides twice a weekday.
A window of 60 decisions is about six trading weeks here and twelve on the daily FX bot; the
unit is in every parameter name so the two are not confused.

**Nothing here trips a breaker.** A detector alert is a signal to look, not a rule to act on:
the acting rules are the lettered kill criteria in ``deploy/risk_config.yaml::breakers``. Two
detectors disagreeing is information; a detector wired straight to a kill switch is a second,
undeclared kill criterion.

Nothing here carries a threshold: every number arrives from ``monitor_config.yaml``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from bots.exness_gold_sess.monitor import load_monitor_config

__all__ = [
    "ADWINStyle",
    "DDM",
    "DetectorSummary",
    "detector_readiness",
    "error_stream",
    "run_detectors",
]


class ADWINStyle:
    """Two-window mean-shift detection with a Hoeffding bound.

    Not the exponential-histogram ADWIN of the literature: a plain sliding window is enough at
    504 observations a year, and the exact bucket structure would be complexity nobody here can
    check. The property that matters is preserved - when a split point shows a difference
    larger than the bound, the stale half is dropped rather than averaged over.
    """

    def __init__(self, *, delta: float, min_window: int) -> None:
        self.delta = float(delta)
        self.min_window = int(min_window)
        self.window: list[float] = []
        self.drift_points: list[int] = []
        self.n_seen = 0

    def _bound(self, n0: int, n1: int) -> float:
        harmonic = 1.0 / (1.0 / n0 + 1.0 / n1)
        return math.sqrt(math.log(4.0 / self.delta) / (2.0 * harmonic))

    def update(self, value: float) -> bool:
        """Append *value*; return True when a change point was detected and the window cut."""
        self.n_seen += 1
        self.window.append(float(value))
        if len(self.window) < 2 * self.min_window:
            return False
        for split in range(self.min_window, len(self.window) - self.min_window + 1):
            older, newer = self.window[:split], self.window[split:]
            mean_gap = abs(sum(older) / len(older) - sum(newer) / len(newer))
            if mean_gap > self._bound(len(older), len(newer)):
                self.window = newer
                self.drift_points.append(self.n_seen)
                return True
        return False


class DDM:
    """Drift Detection Method on a binary error stream (Gama et al.)."""

    def __init__(self, *, min_observations: int, warning_sigma: float, drift_sigma: float) -> None:
        self.min_observations = int(min_observations)
        self.warning_sigma = float(warning_sigma)
        self.drift_sigma = float(drift_sigma)
        self.n = 0
        self.errors = 0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.warnings: list[int] = []
        self.drift_points: list[int] = []

    def update(self, is_error: bool) -> str:
        """Return ``"stable"``, ``"warning"`` or ``"drift"`` for this observation."""
        self.n += 1
        self.errors += int(bool(is_error))
        if self.n < self.min_observations:
            return "stable"
        p = self.errors / self.n
        s = math.sqrt(max(p * (1.0 - p), 0.0) / self.n)
        if p + s < self.p_min + self.s_min:
            self.p_min, self.s_min = p, s
        if p + s > self.p_min + self.drift_sigma * self.s_min:
            self.drift_points.append(self.n)
            return "drift"
        if p + s > self.p_min + self.warning_sigma * self.s_min:
            self.warnings.append(self.n)
            return "warning"
        return "stable"


@dataclass
class DetectorSummary:
    """What both detectors saw over one stream."""

    n_decisions: int
    ready: bool
    readiness_required: int
    adwin_drift_points: list[int] = field(default_factory=list)
    ddm_warnings: list[int] = field(default_factory=list)
    ddm_drift_points: list[int] = field(default_factory=list)
    agree: bool | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def detector_readiness(n_decisions: int, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Is there enough history for a detector's output to mean anything?

    Below the declared count the modules report ``not_ready`` **with the count**, rather than a
    number computed on almost nothing. A detector's first alerts on a short stream are its
    warm-up, not evidence.
    """
    required = int((config or load_monitor_config())["online_detectors"]["readiness_decisions"])
    return {
        "n_decisions": int(n_decisions),
        "required": required,
        "ready": n_decisions >= required,
        "reason": None if n_decisions >= required else (
            f"{n_decisions} decisions recorded, {required} required. On this bot's cadence "
            f"({(config or load_monitor_config())['decisions_per_weekday']} decisions a weekday) "
            f"that is about {required / 10:.0f} trading weeks."
        ),
    }


def error_stream(decisions: pl.DataFrame, *, config: dict[str, Any] | None = None) -> pl.DataFrame:
    """Turn a decision log into the binary error series both detectors read.

    An error is a decision whose realised return had the opposite sign to the position. The
    ``error_rate_floor`` stops a run of near-zero outcomes from making every subsequent loss
    look like a change: a decision whose absolute realised return is below the floor is neither
    a hit nor an error and is dropped from the stream.
    """
    floor = float((config or load_monitor_config())["online_detectors"]["error_rate_floor"])
    return (
        decisions.with_columns(
            signed=(pl.col("position").sign() * pl.col("realised_return")),
        )
        .filter(pl.col("realised_return").abs() >= floor * pl.col("realised_return").abs().mean())
        .with_columns(is_error=(pl.col("signed") < 0).cast(pl.Int8))
        .select("timestamp", "session", "symbol", "is_error")
        .sort("timestamp")
    )


def run_detectors(
    decisions: pl.DataFrame | None, *, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run both detectors, pooled and per sleeve. Reports ``not_ready``, never a false pass.

    Today this returns ``not_ready`` on every sleeve: no decision has been recorded because
    phase 4 has not fitted a model and ``deploy/state/runs/`` is empty.
    """
    config = config or load_monitor_config()
    settings = config["online_detectors"]

    if decisions is None or decisions.is_empty():
        readiness = detector_readiness(0, config=config)
        return {
            "status": "not_ready",
            "readiness": readiness,
            "reason": "no decisions have been recorded; deploy/state/runs/ is empty because "
            "phase 4 has not fitted a model",
        }

    stream = error_stream(decisions, config=config)
    out: dict[str, Any] = {"status": "ok", "sleeves": {}}
    sleeves = ["pooled"] + sorted(stream["session"].unique().to_list())
    for sleeve in sleeves:
        part = stream if sleeve == "pooled" else stream.filter(pl.col("session") == sleeve)
        readiness = detector_readiness(part.height, config=config)
        adwin = ADWINStyle(
            delta=settings["adwin"]["delta"],
            min_window=settings["adwin"]["min_window_decisions"],
        )
        ddm = DDM(
            min_observations=settings["ddm"]["min_decisions"],
            warning_sigma=settings["ddm"]["warning_sigma"],
            drift_sigma=settings["ddm"]["drift_sigma"],
        )
        for value in part["is_error"].to_list():
            adwin.update(float(value))
            ddm.update(bool(value))
        summary = DetectorSummary(
            n_decisions=part.height,
            ready=readiness["ready"],
            readiness_required=readiness["required"],
            adwin_drift_points=adwin.drift_points,
            ddm_warnings=ddm.warnings,
            ddm_drift_points=ddm.drift_points,
            agree=bool(adwin.drift_points) == bool(ddm.drift_points),
            note="a detector alert is a signal to look, not a rule to act on; the acting rules "
            "are deploy/risk_config.yaml::breakers",
        )
        out["sleeves"][sleeve] = summary.as_dict()
    return out
