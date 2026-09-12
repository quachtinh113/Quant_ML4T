"""exness_gold_sess deployment package: the loop, its risk configuration and its state.

``deployment_loop.py``
    The seven steps of ``25_live_trading/02_etfs_deployment_loop.py`` in the FX/CFD shape of
    ``25_live_trading/11_fx_deployment_loop.py``, adapted to a **session** cadence on two
    metals. Dry run by default; ``--arm`` is a per-session flag a person types.
``risk_config.yaml``
    ``ml4t.live.LiveRiskConfig`` plus this bot's breaker thresholds, derived line by line from
    ``BOT.md``'s kill criteria - which are still the mentor's DRAFT and carry
    ``breakers.pending_user_approval: true`` until the user approves them.
``state/``
    The persisted ``RiskState``, the audit journal, the per-cycle run records and the
    deployment artefacts. Nothing in it is committed.

**The broker adapter is not here.** Every bot uses ``bots/_shared/mt5_broker.py``; this package
contains no ``order_send`` and must never gain one (``bot-template.md`` lines 60-70).

**Two sleeves, not one book.** ``PRICE_GRID_DECLARATION.md`` section 3.4: the broker holds one
net position per symbol, so the London and New York decisions cannot be run as one pooled book
- a New York entry landing inside a London hold would be an adjustment rather than a new
position, and the time exit would close the whole thing at 17:00. The loop therefore runs one
sleeve per session, each with its own half of the allocation, and the pooled result is their
deterministic sum.
"""

from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
RISK_CONFIG = DEPLOY_DIR / "risk_config.yaml"
STATE_DIR = DEPLOY_DIR / "state"

__all__ = ["DEPLOY_DIR", "RISK_CONFIG", "STATE_DIR"]
