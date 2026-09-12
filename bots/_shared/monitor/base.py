"""The circuit-breaker state machine of ``26_mlops_governance/04_circuit_breakers.py``.

Copied in behaviour, not in code: the notebook defines ``BreakerState``, ``BreakerEvent``,
``transition_breaker``, ``advance_breaker_state``, ``CircuitBreaker`` and ``BreakerManager``
inside its own cells, because nothing else imports them. Four bots and two breaker tiers
(account-level here, strategy-level in ``bots/<bot_id>/monitor/circuit_breakers.py``) do
import them, so the state machine lives in one module and every breaker is a
``check_condition`` of a few lines - which is the notebook's own conclusion ("Once that
lifecycle is standardized, each risk rule becomes a small condition").

Differences from the notebook, all deliberate:

* ``datetime.now(UTC)`` instead of the naive ``datetime.now()``: the bots time everything in
  UTC (the MT5 server clock is UTC+0, ``setup.yaml::decision.server_clock``), and comparing a
  naive event time with a bar timestamp is how a monitoring layer silently drifts by an hour.
* ``BreakerEvent`` carries a ``tier`` (trade / strategy / portfolio / system) and a
  ``kill_criterion`` string, so a trip can be read back against the bot's written kill
  criteria instead of against a class name.
* ``BreakerManager.to_records`` / ``CircuitBreaker.to_record`` serialise state for the run
  record the deployment loop writes each cycle.

Nothing in this module reads a threshold: every number arrives as a constructor argument from
the bot's YAML.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class BreakerState(Enum):
    """The three states of ``26_mlops_governance/04_circuit_breakers.py``."""

    CLOSED = auto()  # normal operation
    OPEN = auto()  # halted, no trading
    HALF_OPEN = auto()  # recovery probe, limited trading


class BreakerTier(str, Enum):
    """Which level of the defence a breaker belongs to (notebook 04 section 3).

    ``TRADE`` one order or one position; ``STRATEGY`` this bot's signal quality;
    ``PORTFOLIO`` this bot's book (drawdown, daily loss); ``SYSTEM`` the account and the
    infrastructure, shared by every bot.
    """

    TRADE = "trade"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"


@dataclass
class BreakerEvent:
    """One state change; the audit trail of the monitoring layer."""

    timestamp: datetime
    breaker_name: str
    old_state: BreakerState
    new_state: BreakerState
    reason: str
    value: float | None = None
    threshold: float | None = None
    tier: BreakerTier = BreakerTier.STRATEGY
    kill_criterion: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "breaker": self.breaker_name,
            "from": self.old_state.name,
            "to": self.new_state.name,
            "reason": self.reason,
            "value": self.value,
            "threshold": self.threshold,
            "tier": self.tier.value,
            "kill_criterion": self.kill_criterion,
        }


def transition_breaker(
    breaker: CircuitBreaker,
    new_state: BreakerState,
    reason: str,
    value: float | None = None,
    threshold: float | None = None,
    event_time: datetime | None = None,
) -> BreakerEvent:
    """Move ``breaker`` to ``new_state``, append the event, fire the callbacks."""
    event_time = event_time or datetime.now(UTC)
    event = BreakerEvent(
        timestamp=event_time,
        breaker_name=breaker.name,
        old_state=breaker.state,
        new_state=new_state,
        reason=reason,
        value=value,
        threshold=threshold,
        tier=breaker.tier,
        kill_criterion=breaker.kill_criterion,
    )
    breaker.history.append(event)
    breaker.state = new_state
    if new_state is BreakerState.OPEN and event.old_state is not BreakerState.OPEN:
        breaker.trip_count += 1
        breaker.trip_time = event_time
    # on_transition fires on every edge (the audit trail); on_trip only on an edge that
    # lands in OPEN, so manager-level alerting stays specific to trips (notebook 04).
    if breaker.on_transition:
        breaker.on_transition(event)
    if breaker.on_trip and new_state is BreakerState.OPEN:
        breaker.on_trip(event)
    return event


def advance_breaker_state(
    breaker: CircuitBreaker, event_time: datetime | None = None, **kwargs: Any
) -> BreakerState:
    """One step of the state machine: OPEN waits out the timeout, then probes HALF_OPEN."""
    event_time = event_time or datetime.now(UTC)
    if breaker.state is BreakerState.OPEN:
        if breaker.trip_time and event_time - breaker.trip_time >= breaker.recovery_timeout:
            transition_breaker(
                breaker, BreakerState.HALF_OPEN, "Recovery timeout elapsed", event_time=event_time
            )
        return breaker.state

    should_trip, reason, value, threshold = breaker.check_condition(**kwargs)
    if should_trip:
        if breaker.state is BreakerState.HALF_OPEN:
            transition_breaker(
                breaker, BreakerState.OPEN, f"Recovery failed: {reason}", value, threshold, event_time
            )
        else:
            transition_breaker(breaker, BreakerState.OPEN, reason, value, threshold, event_time)
    elif breaker.state is BreakerState.HALF_OPEN:
        transition_breaker(
            breaker, BreakerState.CLOSED, "Recovery successful", event_time=event_time
        )
    return breaker.state


class CircuitBreaker(ABC):
    """Base class: a breaker is a ``check_condition`` plus this shared lifecycle.

    Args:
        name: Stable identifier used in the run record and the alert.
        recovery_timeout: How long OPEN lasts before a HALF_OPEN probe. A daily bot's
            breakers use a timeout measured in days, not the notebook's one hour; the value
            always comes from the bot's YAML.
        tier: Which level of the defence this breaker guards.
        kill_criterion: The letter of the bot's written kill criteria this breaker implements
            (``"a"`` .. ``"f"``), or ``"retire:..."``. ``None`` means the breaker is
            infrastructure and not one of the declared criteria.
    """

    def __init__(
        self,
        name: str,
        *,
        recovery_timeout: timedelta = timedelta(days=1),
        tier: BreakerTier = BreakerTier.STRATEGY,
        kill_criterion: str | None = None,
        on_trip: Callable[[BreakerEvent], None] | None = None,
        on_transition: Callable[[BreakerEvent], None] | None = None,
    ) -> None:
        self.name = name
        self.recovery_timeout = recovery_timeout
        self.tier = tier
        self.kill_criterion = kill_criterion
        self.on_trip = on_trip
        self.on_transition = on_transition

        self.state = BreakerState.CLOSED
        self.trip_time: datetime | None = None
        self.trip_count = 0
        self.history: list[BreakerEvent] = []

    @abstractmethod
    def check_condition(self, **kwargs: Any) -> tuple[bool, str, float | None, float | None]:
        """``(should_trip, reason, observed_value, threshold)``."""

    def update(self, event_time: datetime | None = None, **kwargs: Any) -> BreakerState:
        return advance_breaker_state(self, event_time=event_time, **kwargs)

    def reset(self, event_time: datetime | None = None) -> None:
        """Manual reset to CLOSED. A person does this; no code path calls it automatically."""
        transition_breaker(self, BreakerState.CLOSED, "Manual reset", event_time=event_time)
        self.trip_time = None

    def is_open(self) -> bool:
        return self.state is BreakerState.OPEN

    def allows_trading(self) -> bool:
        return self.state in (BreakerState.CLOSED, BreakerState.HALF_OPEN)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "kill_criterion": self.kill_criterion,
            "state": self.state.name,
            "trip_count": self.trip_count,
            "trip_time": self.trip_time.isoformat() if self.trip_time else None,
            "allows_trading": self.allows_trading(),
            "recovery_timeout_s": self.recovery_timeout.total_seconds(),
        }


@dataclass
class BreakerManager:
    """Registration, one status view, one event log, one answer: may this bot trade?"""

    on_any_trip: Callable[[BreakerEvent], None] | None = None
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    event_log: list[BreakerEvent] = field(default_factory=list)

    def add_breaker(self, breaker: CircuitBreaker) -> CircuitBreaker:
        breaker.on_trip = self._make_trip_callback(breaker)
        breaker.on_transition = self.event_log.append
        self.breakers[breaker.name] = breaker
        return breaker

    def _make_trip_callback(self, breaker: CircuitBreaker) -> Callable[[BreakerEvent], None]:
        original = breaker.on_trip

        def wrapped(event: BreakerEvent) -> None:
            if original:
                original(event)
            logger.error(
                "breaker %s TRIPPED (tier %s, kill criterion %s): %s",
                event.breaker_name,
                event.tier.value,
                event.kill_criterion,
                event.reason,
            )
            if self.on_any_trip:
                self.on_any_trip(event)

        return wrapped

    def check_all(self, event_time: datetime | None = None, **kwargs: Any) -> bool:
        """Update every breaker with whatever it needs and return "trading allowed"."""
        for breaker in self.breakers.values():
            breaker.update(event_time=event_time, **kwargs)
        return self.allows_trading()

    def allows_trading(self) -> bool:
        return all(b.allows_trading() for b in self.breakers.values())

    def open_breakers(self) -> list[str]:
        return [name for name, b in self.breakers.items() if b.is_open()]

    def get_status(self) -> dict[str, dict[str, Any]]:
        return {name: b.to_record() for name, b in self.breakers.items()}

    def to_records(self) -> dict[str, Any]:
        return {
            "allows_trading": self.allows_trading(),
            "open": self.open_breakers(),
            "breakers": self.get_status(),
            "events": [e.as_dict() for e in self.event_log],
        }

    def reset_all(self, event_time: datetime | None = None) -> None:
        for breaker in self.breakers.values():
            breaker.reset(event_time=event_time)


__all__ = [
    "BreakerEvent",
    "BreakerManager",
    "BreakerState",
    "BreakerTier",
    "CircuitBreaker",
    "advance_breaker_state",
    "transition_breaker",
]
