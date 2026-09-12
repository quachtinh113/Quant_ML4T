"""Safe model rollout for exness_fx_d1 — ``26_mlops_governance/03_safe_model_rollout.py``.

Four stages and one gate:

``shadow`` -> ``ab_capped`` -> ``staged_25`` -> ``staged_50`` -> ``full``

with the capital fraction and the minimum observation window of each stage declared in
``monitor_config.yaml::rollout.stages``. A challenger enters at ``shadow`` with no capital, and
moves one stage at a time; each move is a separate :func:`evaluate_promotion` call and each
call is recorded.

THE GATE IS FIXED BEFORE THE CHALLENGER RUNS
--------------------------------------------
``PromotionCriteria`` is frozen and is loaded from the YAML, which was written on 2026-09-07
before any challenger existed. Notebook 03's point is that a *positive* result is not a
sufficient one: its illustrative challenger improves shadow Sharpe and is still rejected,
because the improvement is below the declared hurdle. The same arithmetic applies here, plus
two conditions this bot needs and the notebook's case study did not:

* ``require_holdout_scored``: promotion is impossible while the declared holdout is unscored.
  A shadow Sharpe is not evidence; it is a measurement on data the bot chose to look at.
* ``min_holdout_dsr``: the Deflated Sharpe of the challenger on the holdout, at the trial count
  in force, must clear the level ``BOT.md`` declares (0.95, user decision 3).

Both are why :func:`evaluate_promotion` can return ``decision="NOT_COMPUTABLE"``: with no
scored holdout there is no honest way to answer, and answering "REJECT" would suggest the
question was asked and settled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bots.exness_fx_d1.monitor import load_monitor_config


# ---------------------------------------------------------------------------------------
# The declared gate
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PromotionCriteria:
    """Fixed requirements for leaving a stage (26/03:459-467 plus this bot's two)."""

    min_sharpe_improvement: float
    min_observation_sessions: int
    min_signal_correlation: float
    min_position_agreement: float
    max_drawdown_ratio: float
    require_holdout_scored: bool
    min_holdout_dsr: float

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> PromotionCriteria:
        block = (config or load_monitor_config())["rollout"]["promotion_criteria"]
        return cls(
            min_sharpe_improvement=float(block["min_sharpe_improvement"]),
            min_observation_sessions=int(block["min_observation_sessions"]),
            min_signal_correlation=float(block["min_signal_correlation"]),
            min_position_agreement=float(block["min_position_agreement"]),
            max_drawdown_ratio=float(block["max_drawdown_ratio"]),
            require_holdout_scored=bool(block["require_holdout_scored"]),
            min_holdout_dsr=float(block["min_holdout_dsr"]),
        )


@dataclass(frozen=True)
class RolloutStage:
    name: str
    capital_fraction: float
    min_sessions: int


def load_stages(config: dict[str, Any] | None = None) -> list[RolloutStage]:
    block = (config or load_monitor_config())["rollout"]["stages"]
    return [
        RolloutStage(str(s["name"]), float(s["capital_fraction"]), int(s["min_sessions"]))
        for s in block
    ]


def next_stage(current: str, config: dict[str, Any] | None = None) -> RolloutStage | None:
    stages = load_stages(config)
    names = [s.name for s in stages]
    if current not in names:
        raise ValueError(f"unknown rollout stage {current!r}; declared: {names}")
    index = names.index(current)
    return stages[index + 1] if index + 1 < len(stages) else None


# ---------------------------------------------------------------------------------------
# Shadow statistics
# ---------------------------------------------------------------------------------------
def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """26/03:383. Zero volatility is zero Sharpe, not an exception."""
    series = pd.Series(returns).dropna()
    if series.empty or series.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * series.mean() / series.std(ddof=1))


def max_drawdown(returns: pd.Series) -> float:
    """26/03:396. Negative number; 0.0 when nothing was lost."""
    series = pd.Series(returns).dropna()
    if series.empty:
        return 0.0
    curve = (1 + series).cumprod()
    return float((curve / curve.cummax() - 1).min())


def position_agreement(incumbent: pd.DataFrame, candidate: pd.DataFrame) -> float:
    """Share of (session, symbol) pairs on which the two books take the same side."""
    merged = incumbent.merge(candidate, on=["timestamp", "symbol"], suffixes=("_inc", "_can"))
    if merged.empty:
        return 0.0
    return float((np.sign(merged["weight_inc"]) == np.sign(merged["weight_can"])).mean())


def signal_correlation(incumbent: pd.DataFrame, candidate: pd.DataFrame) -> float:
    """Spearman correlation of the two score streams on the sessions they share."""
    merged = incumbent.merge(candidate, on=["timestamp", "symbol"], suffixes=("_inc", "_can"))
    if len(merged) < 3:
        return 0.0
    value = merged["score_inc"].corr(merged["score_can"], method="spearman")
    return 0.0 if pd.isna(value) else float(value)


@dataclass
class ShadowStats:
    model: str
    sessions: int
    sharpe: float
    total_return: float
    max_drawdown: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def shadow_stats(model: str, returns: pd.Series, periods_per_year: int = 252) -> ShadowStats:
    series = pd.Series(returns).dropna()
    return ShadowStats(
        model=model,
        sessions=int(series.size),
        sharpe=annualized_sharpe(series, periods_per_year),
        total_return=float((1 + series).prod() - 1) if series.size else 0.0,
        max_drawdown=max_drawdown(series),
    )


# ---------------------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------------------
@dataclass
class PromotionDecision:
    decision: str  # "PROMOTE" | "REJECT" | "NOT_COMPUTABLE"
    from_stage: str
    to_stage: str | None
    checks: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "checks": list(self.checks),
            "reasons": list(self.reasons),
        }


def evaluate_promotion(
    *,
    current_stage: str,
    incumbent: ShadowStats,
    candidate: ShadowStats,
    signal_corr: float,
    agreement: float,
    holdout_scored: bool,
    holdout_dsr: float | None = None,
    criteria: PromotionCriteria | None = None,
    config: dict[str, Any] | None = None,
) -> PromotionDecision:
    """Apply the fixed gate. Every criterion must pass; a missing input is not a pass.

    Returns ``NOT_COMPUTABLE`` when the holdout has not been scored (or its DSR is unknown)
    and the gate requires it: the question cannot be answered, and reporting REJECT would
    imply it had been.
    """
    criteria = criteria or PromotionCriteria.from_config(config)
    target = next_stage(current_stage, config)
    improvement = candidate.sharpe - incumbent.sharpe
    dd_ratio = abs(candidate.max_drawdown) / max(abs(incumbent.max_drawdown), 1e-6)
    checks = [
        {
            "criterion": "Sharpe improvement",
            "observed": improvement,
            "required": criteria.min_sharpe_improvement,
            "passed": improvement >= criteria.min_sharpe_improvement,
        },
        {
            "criterion": "Observation sessions",
            "observed": candidate.sessions,
            "required": criteria.min_observation_sessions,
            "passed": candidate.sessions >= criteria.min_observation_sessions,
        },
        {
            "criterion": "Signal correlation",
            "observed": signal_corr,
            "required": criteria.min_signal_correlation,
            "passed": signal_corr >= criteria.min_signal_correlation,
        },
        {
            "criterion": "Position agreement",
            "observed": agreement,
            "required": criteria.min_position_agreement,
            "passed": agreement >= criteria.min_position_agreement,
        },
        {
            "criterion": "Drawdown ratio",
            "observed": dd_ratio,
            "required": criteria.max_drawdown_ratio,
            "passed": dd_ratio <= criteria.max_drawdown_ratio,
        },
    ]
    reasons: list[str] = []
    if criteria.require_holdout_scored and not holdout_scored:
        reasons.append(
            "the declared holdout 2025-09-01..2026-08-31 has not been scored; a shadow Sharpe "
            "is a measurement, not evidence, and promotion cannot be decided on it"
        )
        return PromotionDecision("NOT_COMPUTABLE", current_stage, target.name if target else None, checks, reasons)
    if criteria.require_holdout_scored and holdout_dsr is None:
        reasons.append("holdout is marked scored but no Deflated Sharpe was supplied")
        return PromotionDecision("NOT_COMPUTABLE", current_stage, target.name if target else None, checks, reasons)
    if holdout_dsr is not None:
        checks.append(
            {
                "criterion": "Holdout DSR",
                "observed": holdout_dsr,
                "required": criteria.min_holdout_dsr,
                "passed": holdout_dsr >= criteria.min_holdout_dsr,
            }
        )
    if target is None:
        reasons.append(f"{current_stage} is the last stage; there is nothing to promote to")
        return PromotionDecision("REJECT", current_stage, None, checks, reasons)
    failed = [c["criterion"] for c in checks if not c["passed"]]
    if failed:
        reasons.append(f"failed: {failed}")
        return PromotionDecision("REJECT", current_stage, target.name, checks, reasons)
    return PromotionDecision("PROMOTE", current_stage, target.name, checks, reasons)


def rollout_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Where this bot stands today: not even in shadow, and why."""
    stages = load_stages(config)
    criteria = PromotionCriteria.from_config(config)
    return {
        "stages": [s.__dict__ for s in stages],
        "current_stage": None,
        "criteria": criteria.__dict__,
        "state": "not_started",
        "reason": (
            "no challenger and no incumbent: phase 5 produced no survivor at K = 3,204 and "
            "phase 7 has not run, so there is nothing to put in shadow mode"
        ),
    }


__all__ = [
    "PromotionCriteria",
    "PromotionDecision",
    "RolloutStage",
    "ShadowStats",
    "annualized_sharpe",
    "evaluate_promotion",
    "load_stages",
    "max_drawdown",
    "next_stage",
    "position_agreement",
    "rollout_status",
    "shadow_stats",
    "signal_correlation",
]
