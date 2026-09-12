"""Monitoring primitives shared by every Exness bot (Chapter 26).

Two layers live here, and neither is bot-specific:

``base``
    The circuit-breaker state machine of ``26_mlops_governance/04_circuit_breakers.py``:
    ``BreakerState`` (CLOSED / OPEN / HALF_OPEN), ``BreakerEvent``, the ``CircuitBreaker``
    abstract base with its ``transition_breaker`` / ``advance_breaker_state`` helpers, and
    ``BreakerManager``. A bot's own tiers (``bots/<bot_id>/monitor/circuit_breakers.py``)
    subclass ``CircuitBreaker``; the state machine is written once.

``account``
    The **account-level** breakers: equity drawdown from a persisted high-water mark, daily
    equity loss, margin level, distance to the broker's stop-out, and the number of open
    positions and orders of *any* magic. None of them is about one strategy; all of them are
    properties of the trading account every bot on that login shares. When one trips, the halt
    file is written and **every** bot refuses to stage orders on its next cycle - which is the
    point: the Exness stop-out closes positions in the order the broker chooses, across bots,
    so the bots have to stop before it does (``mt5-exness-broker.md`` section 4).

**The two tiers do not overlap.** A bot never re-implements an account breaker, and an
account breaker never encodes a strategy rule. The account tier carries the **strictest**
limit any bot on the account declares (today 6 % drawdown and 2 % daily loss, from
``bots/xau_fx_mt5/BOT.md``), because one loose outer tier would let one bot's losses overtake
another bot's own rule; ``account_limits.yaml`` and ``bots/README.md`` record the table.

Thresholds are read from ``bots/_shared/monitor/account_limits.yaml``; nothing here carries a
number in code.
"""

from bots._shared.monitor.account import (
    AccountBreakerSet,
    AccountDrawdownBreaker,
    AccountGuard,
    AccountLimits,
    AccountSnapshot,
    DailyEquityLossBreaker,
    MarginLevelBreaker,
    OpenExposureBreaker,
    StopOutDistanceBreaker,
    load_account_limits,
    read_account_snapshot,
)
from bots._shared.monitor.base import (
    BreakerEvent,
    BreakerManager,
    BreakerState,
    BreakerTier,
    CircuitBreaker,
    advance_breaker_state,
    transition_breaker,
)

__all__ = [
    "AccountBreakerSet",
    "AccountDrawdownBreaker",
    "AccountGuard",
    "AccountLimits",
    "AccountSnapshot",
    "BreakerEvent",
    "BreakerManager",
    "BreakerState",
    "BreakerTier",
    "CircuitBreaker",
    "DailyEquityLossBreaker",
    "MarginLevelBreaker",
    "OpenExposureBreaker",
    "StopOutDistanceBreaker",
    "advance_breaker_state",
    "load_account_limits",
    "read_account_snapshot",
    "transition_breaker",
]
