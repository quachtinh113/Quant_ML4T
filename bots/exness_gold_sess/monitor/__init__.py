"""exness_gold_sess monitoring layer (Chapter 26), the STRATEGY tier.

``drift``
    ``26_mlops_governance/01_drift_monitoring.py``: PSI and Kolmogorov-Smirnov on the monitored
    features, prediction drift, and rolling hit rate on the live stream. **The reference window
    is train/validation, never the holdout** - notebook 01 uses a holdout stream because that
    case study had scored its holdout; this bot has not, and its holdout additionally carries a
    contamination flag.
``online_detectors``
    ``26/02``: an ADWIN-style two-window mean-shift test and DDM on the error stream, with every
    window measured in **decision sessions**. This bot decides about 504 times a year, which is
    twice the FX bot's rate, so a window that reads as "63 days" there is 63 *decisions* here -
    about six trading weeks, not three months.
``circuit_breakers``
    ``26/04``: the strategy, portfolio, trade and system tiers, on the shared
    ``CLOSED -> OPEN -> HALF_OPEN`` machine in ``bots/_shared/monitor/base.py``. One breaker per
    lettered kill criterion of ``BOT.md``. The **account** tier is not here: it lives in
    ``bots/_shared/monitor`` and stops every bot on the login at once.
``rollout``
    ``26/03``: shadow -> capital-capped A/B -> staged allocation, with the promotion gate fixed
    in ``monitor_config.yaml`` before any challenger runs.

TWO THINGS THAT MAKE THIS BOT'S MONITORING DIFFERENT FROM THE FX BOT'S
----------------------------------------------------------------------
1. **IC is not a readable statistic here.** Over two metals a cross-sectional rank correlation
   takes the values +/-1 and the registry declines to compute it at all
   (``case_studies/exness_gold_sess/_model_reading.py``). ``BOT.md``'s kill criterion (b) says
   so in as many words and uses a rolling **hit rate** instead. Nothing in this package monitors
   a live IC, and that is deliberate rather than an omission.
2. **Two sleeves, not one book.** London and New York are separate books
   (``PRICE_GRID_DECLARATION.md`` section 3.4), so every rolling statistic is computed **per
   sleeve as well as pooled**. A hit rate that is 0.55 pooled and 0.40 in New York is a New York
   problem, and a pooled-only monitor cannot see it - the same reason ``05_evaluation`` runs its
   staleness screen per session book.

WHERE THE NUMBERS LIVE
----------------------
Kill-criterion thresholds are in ``deploy/risk_config.yaml``; everything else is in
``monitor_config.yaml``. **Nothing in this package carries a number in code.**

STATUS
------
Every module here is wired and testable, and none of it has anything to monitor yet: phase 4
has not fitted a model, so there is no live stream, no residual series and no incumbent to
challenge. Each entry point reports ``not_computable`` with the reason rather than defaulting
to a pass - the same convention ``retire`` uses for the two retirement rules whose inputs
(``16_costs`` and the holdout) do not exist.
"""

from pathlib import Path

MONITOR_DIR = Path(__file__).resolve().parent
MONITOR_CONFIG = MONITOR_DIR / "monitor_config.yaml"


def load_monitor_config(path: Path | str | None = None) -> dict:
    """Parse ``monitor_config.yaml`` (or a test's copy of it)."""
    import yaml

    return yaml.safe_load(Path(path or MONITOR_CONFIG).read_text())


__all__ = ["MONITOR_CONFIG", "MONITOR_DIR", "load_monitor_config"]
