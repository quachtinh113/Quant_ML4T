"""Safe model rollout — ``26_mlops_governance/03_safe_model_rollout.py``.

EVIDENCE BOUNDARY: 0 phase-5 survivors of 1,403 scored specs at K = 1,780, phase 6 not opened,
holdout 2026-03-01 .. 2026-08-31 unscored. **There is no incumbent and no challenger**, so
:func:`evaluate_promotion` returns ``NOT_COMPUTABLE`` with the reason. It never defaults to a
pass, and the gate below was fixed in ``monitor_config.yaml`` on 2026-09-08, *before* any
challenger existed — a promotion decision may only read it.

THE FOUR THINGS THIS BOT ADDS TO THE NOTEBOOK'S GATE, EACH BECAUSE OF SOMETHING MEASURED
-----------------------------------------------------------------------------------------
1. ``require_holdout_scored`` and ``min_holdout_dsr``. The notebook's case study had already
   scored its holdout; this bot has not. Without these two, a shadow Sharpe would stand in for
   evidence, which is the substitution the whole evidence boundary exists to prevent.
2. **Sharpe is measured on the ACTIVE return over the 1/N long book of the two indices on the
   same session window**, not on the raw return. This is not a preference — it is what caught
   the headline of phase 5: the best raw trial (overnight GBM, Sharpe +1.587) has an ACTIVE
   Sharpe of **-0.020** against a 1/N book of +0.442. It bought the overnight drift and added
   nothing. A promotion gate on raw Sharpe would have promoted it.
3. ``trial_count_floor: 1780``. Any DSR a challenger reports must be deflated at the
   **cumulative** trial count of the bot, not of one workspace. This bot has TWO registries
   with 890 trials each, and every DSR helper in the repository reads one registry — so a
   number produced inside a workspace would be deflated at half the truth.
4. The A/B stage is recorded as **unreachable at this account's balance**. 10 % of the declared
   minimum viable allocation (3,000 USD) is 300 USD against a smallest placeable position of
   1,080.32 USD (US500) and 1,482.95 USD (USTEC), measured from ``symbol_info`` on 2026-09-08.
   Reaching that stage needs about 14,830 USD of allocation against an account whose equity was
   896.97 USD. Better recorded here than discovered by a rejected order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bots.exness_usidx_sess.monitor.drift import SPECS, load_monitor_config

BOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HOLDOUT_MARKER = BOT_DIR / "deploy" / "state" / "holdout_scored.json"


@dataclass(frozen=True)
class PromotionCriteria:
    """The gate, read from ``monitor_config.yaml`` and never written in code."""

    min_sharpe_improvement: float
    min_observation_sessions: int
    min_signal_correlation: float
    min_position_agreement: float
    max_drawdown_ratio: float
    require_holdout_scored: bool
    min_holdout_dsr: float
    trial_count_floor: int
    benchmark: str
    sharpe_measured_on: str

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
            trial_count_floor=int(block["trial_count_floor"]),
            benchmark=str(block["benchmark"]),
            sharpe_measured_on=str(block["sharpe_measured_on"]),
        )


@dataclass(frozen=True)
class RolloutStage:
    name: str
    capital_fraction: float
    min_sessions: int


def load_stages(config: dict[str, Any] | None = None) -> list[RolloutStage]:
    block = (config or load_monitor_config())["rollout"]["stages"]
    return [
        RolloutStage(
            name=str(s["name"]),
            capital_fraction=float(s["capital_fraction"]),
            min_sessions=int(s["min_sessions"]),
        )
        for s in block
    ]


def next_stage(current: str, config: dict[str, Any] | None = None) -> RolloutStage | None:
    stages = load_stages(config)
    names = [s.name for s in stages]
    if current not in names:
        raise KeyError(f"{current!r} is not a declared rollout stage: {names}")
    index = names.index(current)
    return stages[index + 1] if index + 1 < len(stages) else None


def stage_is_fundable(
    stage: RolloutStage, *, allocated: float, min_notional: dict[str, float]
) -> dict[str, Any]:
    """Can ``stage``'s capital fraction place the smallest possible position on each index?

    ``min_notional`` is ``volume_min x trade_contract_size x price`` per symbol, read from
    ``symbol_info`` and a live tick — never from a table.
    """
    budget = stage.capital_fraction * float(allocated)
    per_symbol = {
        symbol: {
            "budget": budget,
            "min_notional": float(value),
            "fundable": budget >= float(value),
            "allocation_needed": (float(value) / stage.capital_fraction)
            if stage.capital_fraction
            else 0.0,
        }
        for symbol, value in min_notional.items()
    }
    return {
        "stage": stage.name,
        "capital_fraction": stage.capital_fraction,
        "fundable": all(v["fundable"] for v in per_symbol.values()) or stage.capital_fraction == 0,
        "per_symbol": per_symbol,
    }


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------
def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float | None:
    """Annualised Sharpe. ``periods_per_year`` is 252 CASH SESSIONS, not 504 decisions.

    The bot takes two decisions a session but each spec takes one, and a spec's return series
    is one observation per session. Annualising at 504 would inflate every Sharpe by sqrt(2).
    """
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 2 or values.std(ddof=1) == 0:
        return None
    return float(values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / peak - 1.0))


def position_agreement(incumbent: pl.DataFrame, candidate: pl.DataFrame) -> float | None:
    """Share of ``(timestamp, symbol)`` rows on which the two books take the same side."""
    joined = incumbent.join(candidate, on=["timestamp", "symbol"], suffix="_cand")
    if joined.height == 0:
        return None
    same = (
        np.sign(joined["weight"].to_numpy()) == np.sign(joined["weight_cand"].to_numpy())
    ).mean()
    return float(same)


def signal_correlation(incumbent: pl.DataFrame, candidate: pl.DataFrame) -> float | None:
    joined = incumbent.join(candidate, on=["timestamp", "symbol"], suffix="_cand")
    if joined.height < 3:
        return None
    a = joined["score"].to_numpy().astype(float)
    b = joined["score_cand"].to_numpy().astype(float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class PromotionDecision:
    promote: bool
    status: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "status": self.status,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "criteria": dict(self.criteria),
        }


def evaluate_promotion(
    *,
    spec: str,
    incumbent_returns: np.ndarray | None = None,
    candidate_returns: np.ndarray | None = None,
    benchmark_returns: np.ndarray | None = None,
    incumbent_signals: pl.DataFrame | None = None,
    candidate_signals: pl.DataFrame | None = None,
    holdout_dsr: float | None = None,
    trial_count: int | None = None,
    config: dict[str, Any] | None = None,
    holdout_scored_marker: Path | None = None,
) -> PromotionDecision:
    """The gate. Returns ``NOT_COMPUTABLE`` when an input does not exist — never a pass."""
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
    criteria = PromotionCriteria.from_config(config)
    marker = Path(holdout_scored_marker or DEFAULT_HOLDOUT_MARKER)
    blockers: list[str] = []

    if criteria.require_holdout_scored and not marker.exists():
        blockers.append(
            f"the declared holdout has not been scored (no marker at {marker}); a challenger "
            "may not be promoted on a shadow Sharpe alone"
        )
    if incumbent_returns is None or candidate_returns is None:
        blockers.append(
            "there is no incumbent and no challenger: phase 5 found 0 survivors of 1,403 "
            "scored specs at K = 1,780, so nothing has ever been deployed to challenge"
        )
    if trial_count is not None and trial_count < criteria.trial_count_floor:
        blockers.append(
            f"the reported trial count {trial_count} is below the bot's cumulative K "
            f"{criteria.trial_count_floor}; this bot has TWO registries of 890 trials each and "
            "a DSR helper run inside one of them deflates at half the truth"
        )
    if holdout_dsr is None and criteria.require_holdout_scored:
        blockers.append("no holdout Deflated Sharpe Ratio was supplied")
    elif holdout_dsr is not None and holdout_dsr < criteria.min_holdout_dsr:
        blockers.append(
            f"holdout DSR {holdout_dsr:.3f} is below the declared {criteria.min_holdout_dsr}"
        )

    metrics: dict[str, Any] = {
        "sharpe_measured_on": criteria.sharpe_measured_on,
        "benchmark": criteria.benchmark,
    }
    if incumbent_returns is not None and candidate_returns is not None:
        inc = np.asarray(incumbent_returns, dtype=float)
        cand = np.asarray(candidate_returns, dtype=float)
        bench = (
            np.asarray(benchmark_returns, dtype=float)
            if benchmark_returns is not None
            else None
        )
        if criteria.sharpe_measured_on == "active_return":
            if bench is None:
                blockers.append(
                    "the gate measures Sharpe on the ACTIVE return over the 1/N long book and "
                    "no benchmark series was supplied. Phase 5's best raw trial had a Sharpe of "
                    "+1.587 and an ACTIVE Sharpe of -0.020: on raw return it would have passed."
                )
            else:
                inc, cand = inc - bench, cand - bench
        metrics["incumbent_sharpe"] = annualized_sharpe(inc)
        metrics["candidate_sharpe"] = annualized_sharpe(cand)
        metrics["n_sessions"] = int(min(len(inc), len(cand)))
        metrics["incumbent_max_drawdown"] = max_drawdown(inc)
        metrics["candidate_max_drawdown"] = max_drawdown(cand)
        if metrics["n_sessions"] < criteria.min_observation_sessions:
            blockers.append(
                f"{metrics['n_sessions']} sessions against the {criteria.min_observation_sessions} "
                "the gate requires"
            )
        a, b = metrics["incumbent_sharpe"], metrics["candidate_sharpe"]
        if a is not None and b is not None:
            metrics["sharpe_improvement"] = b - a
            if b - a < criteria.min_sharpe_improvement:
                blockers.append(
                    f"Sharpe improvement {b - a:+.3f} is below {criteria.min_sharpe_improvement}"
                )
    if incumbent_signals is not None and candidate_signals is not None:
        metrics["position_agreement"] = position_agreement(incumbent_signals, candidate_signals)
        metrics["signal_correlation"] = signal_correlation(incumbent_signals, candidate_signals)
        for key, floor in (
            ("position_agreement", criteria.min_position_agreement),
            ("signal_correlation", criteria.min_signal_correlation),
        ):
            value = metrics.get(key)
            if value is not None and value < floor:
                blockers.append(f"{key} {value:.3f} is below {floor}")

    if blockers:
        return PromotionDecision(
            promote=False,
            status="NOT_COMPUTABLE"
            if incumbent_returns is None or candidate_returns is None
            else "REFUSED",
            reasons=blockers,
            metrics=metrics,
            criteria=criteria.__dict__.copy(),
        )
    return PromotionDecision(
        promote=True,
        status="PROMOTE",
        reasons=[],
        metrics=metrics,
        criteria=criteria.__dict__.copy(),
    )


def rollout_status(
    *,
    allocated: float,
    min_notional: dict[str, float],
    current: str = "shadow",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Where the rollout stands, and whether the next stage can be funded at all."""
    cfg = config or load_monitor_config()
    stages = load_stages(cfg)
    upcoming = next_stage(current, cfg)
    return {
        "current_stage": current,
        "next_stage": upcoming.name if upcoming else None,
        "stages": [s.__dict__ for s in stages],
        "funding": [
            stage_is_fundable(s, allocated=allocated, min_notional=min_notional) for s in stages
        ],
        "evidence_boundary": {
            "phase5_survivors": 0,
            "trial_count_K": 1780,
            "holdout_scored": DEFAULT_HOLDOUT_MARKER.exists(),
            "may_advance": False,
            "reason": (
                "advancing past `shadow` needs a phase-5 survivor and a scored holdout; "
                "neither exists"
            ),
        },
    }


__all__ = [
    "PromotionCriteria",
    "PromotionDecision",
    "RolloutStage",
    "annualized_sharpe",
    "evaluate_promotion",
    "load_stages",
    "max_drawdown",
    "next_stage",
    "position_agreement",
    "rollout_status",
    "signal_correlation",
    "stage_is_fundable",
]
