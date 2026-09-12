"""Safe model rollout - ``26_mlops_governance/03_safe_model_rollout.py``.

A new model never replaces the running one. It walks a declared ladder:

    shadow (0 % capital) -> A/B at 10 % -> A/B at 25 % -> staged 50 % -> full

and it may only advance when the promotion gate in ``monitor_config.yaml::rollout.promotion``
is met on the stage it has just completed. **The gate is fixed in the config file before any
challenger runs**, which is the whole point of it being in a file: a gate chosen after seeing
the challenger's numbers is not a gate, it is a rationalisation.

TWO THINGS THIS BOT DOES DIFFERENTLY
------------------------------------
1. **Sharpe is annualised on 252, not 504.** ``PRICE_GRID_DECLARATION.md`` section 2: the
   engine writes its return series on a **daily** grid of about 261 rows a year, whatever the
   price bars are, so annualising on 504 would inflate every Sharpe by ``sqrt(2) = 1.414``. The
   504 is real but belongs to slot-level statistics (``decision.slots_per_year``), not to
   annualisation. The value is read from the config so this file carries no number.
2. **Every comparison is reported on ACTIVE return** against the 1/N long-both-metals book as
   well as against zero. With gold that is mandatory rather than optional: the 2017-2026 sample
   is a strong uptrend and any long-biased book looks good on raw return
   (``bots/exness_gold_sess/BOT.md``, Hypothesis).

STATUS: there is no incumbent and no challenger. Phase 4 has fitted nothing, so
:func:`rollout_status` reports ``no_incumbent`` and :func:`evaluate_promotion` reports
``not_computable`` with the reason rather than defaulting to a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from bots.exness_gold_sess.monitor import load_monitor_config

__all__ = [
    "PromotionCriteria",
    "PromotionDecision",
    "RolloutStage",
    "ShadowStats",
    "annualised_sharpe",
    "evaluate_promotion",
    "load_stages",
    "max_drawdown",
    "next_stage",
    "position_agreement",
    "rollout_status",
    "shadow_stats",
]


@dataclass(frozen=True)
class RolloutStage:
    name: str
    capital_fraction: float
    min_decisions: int


@dataclass(frozen=True)
class PromotionCriteria:
    """Read from ``monitor_config.yaml``; never constructed with literals in code."""

    periods_per_year: int
    min_sharpe_improvement: float
    max_drawdown_ratio: float
    min_decisions: int
    max_position_agreement: float
    benchmark: str
    report_active_return: bool


def load_stages(config: dict[str, Any] | None = None) -> list[RolloutStage]:
    stages = (config or load_monitor_config())["rollout"]["stages"]
    return [RolloutStage(s["name"], float(s["capital_fraction"]), int(s["min_decisions"])) for s in stages]


def load_criteria(config: dict[str, Any] | None = None) -> PromotionCriteria:
    promotion = (config or load_monitor_config())["rollout"]["promotion"]
    return PromotionCriteria(
        periods_per_year=int(promotion["periods_per_year"]),
        min_sharpe_improvement=float(promotion["min_sharpe_improvement"]),
        max_drawdown_ratio=float(promotion["max_drawdown_ratio"]),
        min_decisions=int(promotion["min_decisions"]),
        max_position_agreement=float(promotion["max_position_agreement"]),
        benchmark=str(promotion["benchmark"]),
        report_active_return=bool(promotion["report_active_return"]),
    )


def next_stage(current: str, config: dict[str, Any] | None = None) -> RolloutStage | None:
    """The stage after *current*, or ``None`` at the top of the ladder."""
    stages = load_stages(config)
    names = [s.name for s in stages]
    if current not in names:
        raise ValueError(f"unknown rollout stage {current!r}; declared: {names}")
    index = names.index(current)
    return stages[index + 1] if index + 1 < len(stages) else None


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------
def annualised_sharpe(returns: np.ndarray, *, periods_per_year: int) -> float:
    """Sharpe on the DAILY return series. See the module docstring on why 252 and not 504."""
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    curve = np.cumprod(1.0 + values)
    return float(np.min(curve / np.maximum.accumulate(curve) - 1.0))


def position_agreement(incumbent: pl.DataFrame, candidate: pl.DataFrame) -> float:
    """Share of (symbol, decision) keys on which the two models take the same side.

    Two models that agree on 95 % of positions are one model with two hashes. Promoting the
    second buys nothing and costs a rollout, so the gate refuses it.
    """
    joined = incumbent.join(candidate, on=["symbol", "timestamp"], how="inner", suffix="_cand")
    if joined.is_empty():
        return float("nan")
    return float(
        joined.select(
            (pl.col("position").sign() == pl.col("position_cand").sign()).mean()
        ).item()
    )


@dataclass
class ShadowStats:
    model: str
    n_decisions: int
    sharpe_raw: float
    sharpe_active: float
    max_drawdown: float
    benchmark: str


def shadow_stats(
    model: str,
    returns: np.ndarray,
    benchmark_returns: np.ndarray | None,
    *,
    criteria: PromotionCriteria,
) -> ShadowStats:
    """Raw and active Sharpe for one model over its shadow stage.

    ``sharpe_active`` is ``nan`` when no benchmark series is supplied, and it is reported as
    ``nan`` rather than silently falling back to the raw figure: on gold, the raw figure of any
    long-biased book over 2017-2026 is a statement about the gold price, not about the model.
    """
    raw = annualised_sharpe(returns, periods_per_year=criteria.periods_per_year)
    if benchmark_returns is None:
        active = float("nan")
    else:
        active = annualised_sharpe(
            np.asarray(returns, dtype=float) - np.asarray(benchmark_returns, dtype=float),
            periods_per_year=criteria.periods_per_year,
        )
    return ShadowStats(
        model=model,
        n_decisions=int(len(returns)),
        sharpe_raw=raw,
        sharpe_active=active,
        max_drawdown=max_drawdown(returns),
        benchmark=criteria.benchmark,
    )


@dataclass
class PromotionDecision:
    promote: bool
    from_stage: str
    to_stage: str | None
    reasons: list[str]
    incumbent: ShadowStats | None
    candidate: ShadowStats | None
    position_agreement: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "reasons": self.reasons,
            "incumbent": self.incumbent.__dict__ if self.incumbent else None,
            "candidate": self.candidate.__dict__ if self.candidate else None,
            "position_agreement": self.position_agreement,
        }


def evaluate_promotion(
    *,
    stage: str,
    incumbent: ShadowStats | None,
    candidate: ShadowStats | None,
    agreement: float | None,
    config: dict[str, Any] | None = None,
) -> PromotionDecision:
    """Apply the fixed gate. Every failed condition is listed, not just the first.

    Listing all of them matters: a challenger that fails only on decision count will pass with
    more data, and one that fails on Sharpe *and* drawdown *and* agreement will not. A decision
    that reports only the first failure hides that difference.
    """
    criteria = load_criteria(config)
    reasons: list[str] = []

    if incumbent is None or candidate is None:
        return PromotionDecision(
            promote=False,
            from_stage=stage,
            to_stage=None,
            reasons=[
                "not_computable: there is no incumbent and no challenger. Phase 4 has fitted no "
                "model (run_log/ is empty), so no model has ever been deployed to be challenged."
            ],
            incumbent=incumbent,
            candidate=candidate,
            position_agreement=agreement,
        )

    if candidate.n_decisions < criteria.min_decisions:
        reasons.append(
            f"candidate has {candidate.n_decisions} decisions, {criteria.min_decisions} required"
        )

    reference = candidate.sharpe_active if criteria.report_active_return else candidate.sharpe_raw
    incumbent_reference = (
        incumbent.sharpe_active if criteria.report_active_return else incumbent.sharpe_raw
    )
    if not np.isfinite(reference) or not np.isfinite(incumbent_reference):
        reasons.append(
            "active Sharpe is not computable for one of the two models; the gate is defined on "
            f"active return against {criteria.benchmark} and will not fall back to raw return"
        )
    elif reference - incumbent_reference < criteria.min_sharpe_improvement:
        reasons.append(
            f"active Sharpe improvement {reference - incumbent_reference:+.3f} below the "
            f"required {criteria.min_sharpe_improvement}"
        )

    if np.isfinite(candidate.max_drawdown) and np.isfinite(incumbent.max_drawdown):
        ratio = abs(candidate.max_drawdown) / max(abs(incumbent.max_drawdown), 1e-9)
        if ratio > criteria.max_drawdown_ratio:
            reasons.append(
                f"candidate drawdown is {ratio:.2f}x the incumbent's, above the permitted "
                f"{criteria.max_drawdown_ratio}"
            )

    if agreement is not None and np.isfinite(agreement) and agreement > criteria.max_position_agreement:
        reasons.append(
            f"the two models agree on {agreement:.1%} of positions, above the "
            f"{criteria.max_position_agreement:.0%} ceiling: promoting a near-identical model "
            "costs a rollout and buys nothing"
        )

    promote = not reasons
    return PromotionDecision(
        promote=promote,
        from_stage=stage,
        to_stage=next_stage(stage, config).name if promote and next_stage(stage, config) else None,
        reasons=reasons or ["every declared condition met"],
        incumbent=incumbent,
        candidate=candidate,
        position_agreement=agreement,
    )


def rollout_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ladder, and where this bot is on it. Today: nowhere, and it says why."""
    stages = load_stages(config)
    criteria = load_criteria(config)
    return {
        "bot_id": (config or load_monitor_config())["bot_id"],
        "stages": [s.__dict__ for s in stages],
        "criteria": criteria.__dict__,
        "current_stage": None,
        "status": "no_incumbent",
        "reason": "phase 4 has not fitted a model and phases 5-7 have not run, so nothing has "
        "ever been deployed. The first model to reach live starts at `shadow`, not at `full`, "
        "and it does so after 17_holdout_predictions / 18_holdout_backtest have scored the "
        "holdout once - which is itself contaminated for this bot and may CONFIRM, not SELECT.",
    }
