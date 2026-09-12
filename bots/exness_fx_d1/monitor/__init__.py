"""exness_fx_d1 monitoring layer (Chapter 26), the strategy tier.

``drift``
    ``26_mlops_governance/01_drift_monitoring.py``: PSI and K-S on the monitored features,
    prediction drift, rolling IC / hit rate / MSE on the live stream. **The reference window
    is train/validation, never the holdout** - notebook 01 uses a holdout stream because that
    case study had scored its holdout; this bot has not.
``online_detectors``
    ``26/02``: ADWIN-style two-window mean-shift and DDM on the error stream, with every
    window measured in **decision sessions** because this bot decides about 252 times a year.
``circuit_breakers``
    ``26/04``: the strategy, portfolio and trade tiers, on the shared
    ``CLOSED -> OPEN -> HALF_OPEN`` machine in ``bots/_shared/monitor/base.py``, one breaker
    per lettered kill criterion of ``BOT.md``. The **account** tier is not here: it lives in
    ``bots/_shared/monitor`` and stops every bot at once.
``rollout``
    ``26/03``: shadow -> capital-capped A/B -> staged allocation, with the promotion gate
    fixed in ``monitor_config.yaml`` before any challenger runs.
``retire``
    The two retire rules of ``BOT.md``. Both inputs are missing today, so the check reports
    ``not_computable`` with the reason rather than defaulting to a pass.

Thresholds: kill criteria in ``deploy/risk_config.yaml``, everything else in
``monitor_config.yaml``. Nothing in this package carries a number in code.
"""

from pathlib import Path

MONITOR_DIR = Path(__file__).resolve().parent
MONITOR_CONFIG = MONITOR_DIR / "monitor_config.yaml"


def load_monitor_config(path: Path | str | None = None) -> dict:
    """Parse ``monitor_config.yaml`` (or a test's copy of it)."""
    import yaml

    return yaml.safe_load(Path(path or MONITOR_CONFIG).read_text())


__all__ = ["MONITOR_CONFIG", "MONITOR_DIR", "load_monitor_config"]
