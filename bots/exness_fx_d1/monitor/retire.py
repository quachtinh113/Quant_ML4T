"""The two retire rules of exness_fx_d1 — and why neither can be evaluated today.

``BOT.md``, Kill criteria, second sentence:

    *Retire when the latest re-estimate of the breakeven cost (16_costs) is below the measured
    p90 spread or the holdout PSR against the 1/N book is below 0.5 (the benchmark of a book
    of dollar pairs is the 1/N long book, not zero).*

A retire rule is not a circuit breaker. A breaker pauses and probes ``HALF_OPEN``; a retire
rule ends the bot and is evaluated by a person reading two numbers. Both numbers are missing:

* the **breakeven cost** comes from ``16_costs``, which has not run - phase 6 was not opened
  because phase 5 has no survivor at K = 3,204 (``BOT.md`` phase table, Decisions log
  2026-09-06);
* the **holdout PSR against the 1/N long book** comes from the holdout stages, which have not
  run; the holdout 2025-09-01 .. 2026-08-31 is unscored.

So :func:`check_retire_rules` reports ``NOT_COMPUTABLE`` per rule, with the reason and with
what would unblock it. It does **not** default to "pass": a retire check that silently returns
"keep running" because its input is missing is worse than no check, because it looks like one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RISK_CONFIG = BOT_DIR / "deploy" / "risk_config.yaml"


@dataclass
class RetireCheck:
    """One rule. ``verdict`` is ``RETIRE`` | ``KEEP`` | ``NOT_COMPUTABLE``."""

    rule: str
    verdict: str
    observed: float | None
    threshold: float | None
    reason: str
    unblocked_by: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RetireReport:
    checks: list[RetireCheck] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        verdicts = {c.verdict for c in self.checks}
        if "RETIRE" in verdicts:
            return "RETIRE"
        if verdicts == {"KEEP"}:
            return "KEEP"
        return "NOT_COMPUTABLE"

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "checks": [c.as_dict() for c in self.checks]}


def load_retire_config(path: Path | str | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path or DEFAULT_RISK_CONFIG).read_text())
    return raw["retire"]


def check_retire_rules(
    *,
    breakeven_cost_bps: float | None = None,
    holdout_psr_vs_benchmark: float | None = None,
    config_path: Path | str | None = None,
) -> RetireReport:
    """Evaluate both rules. A missing input yields ``NOT_COMPUTABLE``, never a pass.

    Args:
        breakeven_cost_bps: The latest re-estimate from ``16_costs``. ``None`` today.
        holdout_psr_vs_benchmark: PSR of the holdout book against the 1/N long book of the
            five pairs. ``None`` today.
    """
    config = load_retire_config(config_path)
    min_breakeven = float(config["min_breakeven_cost_bps"])
    min_psr = float(config["min_holdout_psr_vs_benchmark"])
    checks: list[RetireCheck] = []

    if breakeven_cost_bps is None:
        checks.append(
            RetireCheck(
                rule="breakeven_cost_above_measured_p90_spread",
                verdict="NOT_COMPUTABLE",
                observed=None,
                threshold=min_breakeven,
                reason=(
                    "16_costs has not run: phase 6 was not opened because no phase-5 spec is "
                    "significant at 0.95 on the active return at K = 3,204"
                ),
                unblocked_by=(
                    "a phase-5 survivor, then 14_portfolio_management / 15_risk_management / "
                    "16_costs in the experiment, plus the real Pro-account spread measurement "
                    "(BOT.md user decision 5)"
                ),
            )
        )
    else:
        retire = breakeven_cost_bps < min_breakeven
        checks.append(
            RetireCheck(
                rule="breakeven_cost_above_measured_p90_spread",
                verdict="RETIRE" if retire else "KEEP",
                observed=float(breakeven_cost_bps),
                threshold=min_breakeven,
                reason=(
                    f"breakeven cost {breakeven_cost_bps:.2f} bps "
                    f"{'below' if retire else 'above'} the measured p90 spread {min_breakeven:.2f} bps"
                ),
            )
        )

    if holdout_psr_vs_benchmark is None:
        checks.append(
            RetireCheck(
                rule="holdout_psr_vs_1_over_n_book",
                verdict="NOT_COMPUTABLE",
                observed=None,
                threshold=min_psr,
                reason=(
                    "the declared holdout 2025-09-01 .. 2026-08-31 is unscored; the holdout "
                    "stages have not run and may not be run before phases 6 and 7 are green"
                ),
                unblocked_by=(
                    "*_holdout_predictions and *_holdout_backtest, scored once, after the "
                    "portfolio, risk and cost stages are green"
                ),
            )
        )
    else:
        retire = holdout_psr_vs_benchmark < min_psr
        checks.append(
            RetireCheck(
                rule="holdout_psr_vs_1_over_n_book",
                verdict="RETIRE" if retire else "KEEP",
                observed=float(holdout_psr_vs_benchmark),
                threshold=min_psr,
                reason=(
                    f"holdout PSR against the 1/N long book {holdout_psr_vs_benchmark:.3f} "
                    f"{'below' if retire else 'at or above'} {min_psr:.2f}"
                ),
            )
        )
    return RetireReport(checks)


__all__ = ["RetireCheck", "RetireReport", "check_retire_rules", "load_retire_config"]
