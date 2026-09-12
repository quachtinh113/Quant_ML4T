"""The two retire rules of ``bots/exness_usidx_sess/BOT.md`` — NOT breakers.

A breaker pauses and probes; a retire rule ends the bot. The rules, verbatim from the (draft)
kill criteria:

    Retire when ``16_costs`` re-estimates the breakeven below the p90 spread measured on the
    **real** account, or when the holdout PSR against a 1/N long book on the same session
    window is below 0.5.

EVIDENCE BOUNDARY: **neither input exists.**

* ``16_costs`` has not run and cannot: phase 6 is not opened, because phase 5 found 0
  survivors of 1,403 scored specs at K = 1,780. So there is no breakeven cost to compare.
* The **real** Pro account has never been read. Every spread and swap figure this bot carries
  was measured on demo ``206539306`` @ ``Exness-MT5Trial7``. The rule names the real account
  on purpose, so a demo number may not be substituted for it, and
  ``risk_config.yaml::retire.real_account_measured`` is ``false``.
* The holdout 2026-03-01 .. 2026-08-31 has never been scored, so there is no holdout PSR.

Both checks therefore return ``not_computable`` **with the reason and with what would unblock
them**. Neither ever returns a pass by default: a retire rule that silently passes is worse
than no retire rule, because it reads as an all-clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RISK_CONFIG = BOT_DIR / "deploy" / "risk_config.yaml"
DEFAULT_HOLDOUT_MARKER = BOT_DIR / "deploy" / "state" / "holdout_scored.json"


@dataclass
class RetireCheck:
    name: str
    rule: str
    status: str  # "not_computable" | "pass" | "RETIRE"
    reason: str
    unblocked_by: str
    observed: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RetireReport:
    retire: bool
    checks: list[RetireCheck] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "retire": self.retire,
            "checks": [c.as_dict() for c in self.checks],
            "evidence": dict(self.evidence),
        }


def load_retire_config(path: Path | str | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path or DEFAULT_RISK_CONFIG).read_text())
    return {"retire": raw["retire"], "evidence_boundary": raw["evidence_boundary"]}


def check_retire_rules(
    *,
    breakeven_cost_bps: float | None = None,
    real_account_p90_spread_bps: float | None = None,
    holdout_psr_vs_benchmark: float | None = None,
    config_path: Path | str | None = None,
    holdout_scored_marker: Path | None = None,
) -> RetireReport:
    """Both rules. Every argument defaults to ``None`` because none of them exists today."""
    config = load_retire_config(config_path)
    block = config["retire"]
    boundary = config["evidence_boundary"]
    marker = Path(holdout_scored_marker or DEFAULT_HOLDOUT_MARKER)
    checks: list[RetireCheck] = []

    # -- rule 1: the breakeven cost against the REAL account's p90 spread ----------------
    real_measured = bool(block.get("real_account_measured", False))
    if breakeven_cost_bps is None or not real_measured or real_account_p90_spread_bps is None:
        missing = []
        if breakeven_cost_bps is None:
            missing.append(
                "16_costs has not run (phase 6 is not opened: 0 phase-5 survivors of 1,403 "
                "specs at K = 1,780), so there is no breakeven cost"
            )
        if not real_measured or real_account_p90_spread_bps is None:
            missing.append(
                "the REAL Pro account has never been read; every spread on record was measured "
                "on the demo login, and the rule names the real account on purpose"
            )
        checks.append(
            RetireCheck(
                name="breakeven_below_real_p90_spread",
                rule=(
                    "retire when 16_costs re-estimates the breakeven below the p90 spread "
                    "measured on the REAL account"
                ),
                status="not_computable",
                reason="; ".join(missing),
                unblocked_by=(
                    "a phase-5 survivor -> phase 6 -> 16_costs, AND a read-only measurement of "
                    "the real Pro account's spread at this bot's six execution hours"
                ),
                threshold=float(block["min_breakeven_cost_bps"]),
            )
        )
    else:
        checks.append(
            RetireCheck(
                name="breakeven_below_real_p90_spread",
                rule="retire when the breakeven falls below the real account's p90 spread",
                status="RETIRE" if breakeven_cost_bps < real_account_p90_spread_bps else "pass",
                reason=(
                    f"breakeven {breakeven_cost_bps:.2f} bps against a real-account p90 spread "
                    f"of {real_account_p90_spread_bps:.2f} bps"
                ),
                unblocked_by="",
                observed=float(breakeven_cost_bps),
                threshold=float(real_account_p90_spread_bps),
            )
        )

    # -- rule 2: the holdout PSR against the 1/N long book ------------------------------
    if holdout_psr_vs_benchmark is None or not marker.exists():
        checks.append(
            RetireCheck(
                name="holdout_psr_below_0_5",
                rule=(
                    "retire when the holdout PSR against a 1/N long book on the same session "
                    "window is below 0.5"
                ),
                status="not_computable",
                reason=(
                    f"the declared holdout {boundary['holdout_start']}..{boundary['holdout_end']} "
                    f"has never been scored (no marker at {marker}); scoring it before a "
                    "phase-5 survivor exists would burn it"
                ),
                unblocked_by=(
                    "a phase-5 survivor, phases 6 green, then 17_holdout_predictions and "
                    "18_holdout_backtest run ONCE — and note the operational risk: this bot has "
                    "TWO registries, so 'scored once' has to be enforced by hand across both"
                ),
                threshold=float(block["min_holdout_psr_vs_benchmark"]),
            )
        )
    else:
        floor = float(block["min_holdout_psr_vs_benchmark"])
        checks.append(
            RetireCheck(
                name="holdout_psr_below_0_5",
                rule="retire when the holdout PSR against the 1/N long book is below 0.5",
                status="RETIRE" if holdout_psr_vs_benchmark < floor else "pass",
                reason=f"holdout PSR {holdout_psr_vs_benchmark:.3f} against a floor of {floor}",
                unblocked_by="",
                observed=float(holdout_psr_vs_benchmark),
                threshold=floor,
            )
        )

    return RetireReport(
        retire=any(c.status == "RETIRE" for c in checks),
        checks=checks,
        evidence={
            "phase5_survivors": boundary.get("phase5_survivors"),
            "phase5_specs_scored": boundary.get("phase5_specs_scored"),
            "trial_count_K": boundary.get("phase5_trial_count"),
            "phase6_opened": boundary.get("phase6_opened"),
            "holdout_scored": marker.exists(),
            "real_account_measured": real_measured,
            "note": (
                "'not_computable' is not 'pass'. Neither retire rule can be evaluated, and a "
                "bot that cannot evaluate its own retirement rules has not earned a deployment."
            ),
        },
    )


__all__ = ["RetireCheck", "RetireReport", "check_retire_rules", "load_retire_config"]
