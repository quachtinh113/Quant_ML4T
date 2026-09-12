"""Strategy-tier circuit breakers for ``exness_usidx_sess`` — ``26_mlops_governance/04``.

EVIDENCE BOUNDARY
-----------------
This bot has **no phase-5 survivor** (0 of 1,403 scored specs at K = 1,780), phase 6 is not
opened and the declared holdout 2026-03-01 .. 2026-08-31 has **never been scored**. Every
threshold below therefore comes from a *written kill criterion* or from a *measured price or
contract fact*, and **not one comes from the Sharpe distribution of the 1,403 failed trials**.
Calibrating a monitoring threshold on that distribution would be fitting a control to an
outcome that has already been shown to be noise. The module is DRY RUN ONLY: it watches a
stream that does not exist yet, and :func:`build_manager` refuses to build at all unless the
caller states that it knows the criteria are still a draft.

STATE MACHINE
-------------
``CLOSED -> OPEN -> HALF_OPEN`` lives once, in ``bots/_shared/monitor/base.py``; this module
supplies only the conditions. The **account** tier is not here — it is
``bots/_shared/monitor/account.py`` and it stops every bot on the login at once. A bot never
re-implements an account breaker.

MAP: KILL CRITERION -> BREAKER
------------------------------
Every threshold is read from ``deploy/risk_config.yaml::breakers`` /
``::missing_decision_bar``; not one number is written in this file.

===== ============================================== =========================================
Rule  BOT.md kill criterion (DRAFT)                  Implementation
===== ============================================== =========================================
(a)   drawdown from peak > 8 % of allocated capital  :class:`DrawdownBreaker`, portfolio tier
 --   daily loss (derived, 2 % of allocated)         :class:`DailyLossBreaker`, portfolio tier
(b)   rolling 63-session hit rate < 0.5 for 21       :class:`RollingHitRateBreaker`, strategy
      consecutive sessions, **counted separately     tier, built **TWICE** — one instance per
      per spec**                                     spec. Pooling the two would be exactly the
                                                     masking the criterion forbids.
(c)   realised round-trip cost > 1.5x the assumed    :class:`RealisedCostBreaker`, strategy
      cost over one month                            tier. TWO assumed costs exist and the
                                                     breaker takes the tighter; see the class.
(d)   reconciliation between the run record and      :class:`ReconciliationBreaker`, system
      ``positions_get`` disagrees                    tier. Trips at once, never probes, latches
                                                     the persistent kill switch.
(e)   a decision instant has no bar to fill at       **Asymmetric, and this is the bot's
      -> no order that session                       biggest difference from exness_fx_d1.**
                                                     *intraday*: 0 of 1,930 research sessions
                                                     missed, so a miss is anomalous ->
                                                     :class:`MissingDecisionBarBreaker`.
                                                     *overnight*: 504 of 1,930 (26 %) missed
                                                     and that is **market structure, not a
                                                     fault** -> no miss counter at all, only
                                                     :class:`OvernightFillCoverageBreaker`
                                                     against the pre-declared
                                                     ``min_fill_coverage`` floor.
(f)   realised swap per night > the declared value   :class:`SwapBreaker`, strategy tier,
      over one week                                  **per symbol**: US500 and USTEC declare
                                                     -147.4 and -592.7 points and a single
                                                     pooled threshold would be meaningless.
(g)   the daily break stops ending at 18:00 New      :class:`TradeHoursDstBreaker`, system
      York, i.e. the trading hours stop following    tier. **No other bot has this breaker**:
      ``decision.trade_hours_follow_dst_of``         no other bot's decision instants ride a
                                                     foreign DST rule while its server clock
                                                     does not.
 --   five consecutive losing round trips            :class:`ConsecutiveLossBreaker`, trade tier
 --   order round-trip latency                       :class:`LatencyBreaker`, system tier
===== ============================================== =========================================

Retirement is not a breaker: a breaker pauses and probes, a retire rule ends the bot. See
``retire.py``.

WHY THIS IS A THIRD COPY OF SOME OF THESE CLASSES, AND WHAT SHOULD HAPPEN TO IT
------------------------------------------------------------------------------
``bots/exness_fx_d1/monitor/circuit_breakers.py`` and
``bots/exness_gold_sess/monitor/circuit_breakers.py`` already carry their own
``DrawdownBreaker`` and friends, and **the two have already diverged** — different constructor
signatures (``portfolio_value=`` against ``bot_equity=``/``bot_equity_peak=``), different
reason strings, ``>=`` against ``>``. Extracting one shared implementation now would mean
editing two other bots' modules, which this builder does not own. So the per-bot module is
followed (it is the established pattern), the semantics of ``exness_fx_d1`` are followed where
a class is the same rule, and the cleanup is recorded in ``BOT.md`` as a cross-bot
recommendation: promote the seven generic breakers into ``bots/_shared/monitor/strategy.py``
and leave only (b)-per-spec, (e)-asymmetric and (g) here.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from bots._shared.monitor.base import (
    BreakerEvent,
    BreakerManager,
    BreakerState,
    BreakerTier,
    CircuitBreaker,
    transition_breaker,
)

logger = logging.getLogger("exness_usidx_sess.monitor")

BOT_ID = "exness_usidx_sess"
BOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RISK_CONFIG = BOT_DIR / "deploy" / "risk_config.yaml"

#: The two decision specs. One workspace, one model, one registry and one book each; they are
#: never pooled inside a breaker (BOT.md kill criterion (b)).
SPECS = ("intraday", "overnight")


class DraftCriteriaError(RuntimeError):
    """The kill criteria are still a draft the user has not approved."""


def load_breaker_config(path: Path | str | None = None) -> dict[str, Any]:
    """The blocks of ``deploy/risk_config.yaml`` this module reads. No threshold lives here."""
    source = Path(path or DEFAULT_RISK_CONFIG)
    raw = yaml.safe_load(source.read_text())
    return {
        "breakers": raw["breakers"],
        "missing_decision_bar": raw["missing_decision_bar"],
        "capital": raw["capital"],
        "evidence_boundary": raw["evidence_boundary"],
        "source": str(source),
    }


# ---------------------------------------------------------------------------------------
# (a) and the derived daily rule — portfolio tier
# ---------------------------------------------------------------------------------------
class DrawdownBreaker(CircuitBreaker):
    """(a) Drawdown from the running peak of **this bot's own** equity.

    The denominator is the capital allocated to this bot, not account equity: four other bots
    share the login and their profit and loss must not move this breaker. The peak is persisted
    with the run records, because a peak that resets with the process is not a peak.
    """

    def __init__(self, name: str, *, max_drawdown_pct: float, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.PORTFOLIO, kill_criterion="a", **kwargs)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.peak_value = 0.0

    def check_condition(self, portfolio_value: float | None = None, **_: Any):
        if portfolio_value is None:
            return False, "", None, self.max_drawdown_pct
        self.peak_value = max(self.peak_value, float(portfolio_value))
        if self.peak_value <= 0:
            return False, "", None, self.max_drawdown_pct
        drawdown = (self.peak_value - float(portfolio_value)) / self.peak_value
        if drawdown >= self.max_drawdown_pct:
            return (
                True,
                f"drawdown {drawdown:.2%} from peak {self.peak_value:,.2f} exceeds "
                f"{self.max_drawdown_pct:.0%} of allocated capital",
                drawdown,
                self.max_drawdown_pct,
            )
        return False, "", drawdown, self.max_drawdown_pct


class DailyLossBreaker(CircuitBreaker):
    """The derived daily-loss rule, in PERCENT so it does not go stale with the allocation.

    Derived, and the derivation is a price fact rather than a trial statistic: the measured
    median absolute session move of these two indices is 29-49 bps (BOT.md phase 1), so 2 % of
    allocated capital at gross 1.0 is four to seven times a median session.
    """

    def __init__(self, name: str, *, max_daily_loss_pct: float, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.PORTFOLIO, kill_criterion=None, **kwargs)
        self.max_daily_loss_pct = float(max_daily_loss_pct)

    def check_condition(
        self,
        day_start_value: float | None = None,
        portfolio_value: float | None = None,
        **_: Any,
    ):
        if day_start_value in (None, 0) or portfolio_value is None:
            return False, "", None, self.max_daily_loss_pct
        loss = (float(day_start_value) - float(portfolio_value)) / float(day_start_value)
        if loss >= self.max_daily_loss_pct:
            return (
                True,
                f"daily loss {loss:.2%} exceeds {self.max_daily_loss_pct:.0%} of the value at "
                "the start of the trading day",
                loss,
                self.max_daily_loss_pct,
            )
        return False, "", loss, self.max_daily_loss_pct


# ---------------------------------------------------------------------------------------
# (b) — one instance PER SPEC, strategy tier
# ---------------------------------------------------------------------------------------
class RollingHitRateBreaker(CircuitBreaker):
    """(b) A ``window`` -session rolling hit rate below ``min_hit_rate`` for ``consecutive``
    sessions **within one spec**.

    A sliding window, which ``ConsecutiveLossBreaker`` cannot express: that one counts a losing
    streak, this one counts how long an average has stayed under a line.

    ONE INSTANCE PER SPEC IS THE RULE, NOT AN OPTIMISATION. The intraday leg is a
    mean-reversion book on a six-hour hold that never pays financing; the overnight leg is a
    drift book on a seventeen-hour hold that pays -1.91 / -2.01 bps a night long. Pooling their
    outcomes lets a healthy leg carry a dead one past the breaker for the whole 21-session
    window, which is the failure BOT.md criterion (b) names in its own text.
    """

    def __init__(
        self,
        name: str,
        *,
        spec: str,
        window_sessions: int,
        min_hit_rate: float,
        consecutive_sessions: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="b", **kwargs)
        if spec not in SPECS:
            raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
        self.spec = spec
        self.window_sessions = int(window_sessions)
        self.min_hit_rate = float(min_hit_rate)
        self.consecutive_sessions = int(consecutive_sessions)
        self.outcomes: deque[bool] = deque(maxlen=self.window_sessions)
        self.sessions_below = 0

    def record_session(self, was_a_hit: bool) -> None:
        """One CLOSED session of this spec. A flat session is not recorded: it is not a bet."""
        self.outcomes.append(bool(was_a_hit))

    def hit_rate(self) -> float | None:
        if len(self.outcomes) < self.window_sessions:
            return None
        return float(np.mean(self.outcomes))

    def check_condition(self, **_: Any):
        rate = self.hit_rate()
        if rate is None:
            # Fewer than `window` closed sessions: the statistic does not exist yet. Silence
            # here means "cannot judge", never "healthy" - the run record says which.
            self.sessions_below = 0
            return False, "", None, self.min_hit_rate
        if rate < self.min_hit_rate:
            self.sessions_below += 1
        else:
            self.sessions_below = 0
        if self.sessions_below >= self.consecutive_sessions:
            return (
                True,
                f"{self.spec}: the {self.window_sessions}-session hit rate has been below "
                f"{self.min_hit_rate:.0%} for {self.sessions_below} consecutive sessions "
                f"(now {rate:.1%})",
                rate,
                self.min_hit_rate,
            )
        return False, "", rate, self.min_hit_rate


# ---------------------------------------------------------------------------------------
# (c) — strategy tier
# ---------------------------------------------------------------------------------------
@dataclass
class RoundTrip:
    """One closed round trip read from the MT5 deal history; never from the engine."""

    closed_at: datetime
    symbol: str
    notional: float
    cost_bps: float


class RealisedCostBreaker(CircuitBreaker):
    """(c) Realised round-trip cost above ``multiple`` x the assumed cost over a month.

    TWO ASSUMED COSTS EXIST FOR THIS BOT AND THE BREAKER TAKES THE TIGHTER. The backtest was
    charged 8.10 bps a round trip (2 x the declared p90 of 4.05), a figure deliberately set to
    the worst month of a sample that spans a spread regime break in 2024-10. Today this account
    quotes about 2.1 bps round trip. A breaker at 1.5 x 8.10 = 12.15 bps would never fire, so it
    would not be a control at all; a breaker at 1.5 x 2.10 = 3.15 bps is one. Both numbers are
    declared in the YAML and ``BOT.md`` records that criterion (c)'s wording has to say which
    it means when the user approves it. Neither number comes from a trial.

    The breaker ABSTAINS below ``min_round_trips``: a monthly average over three trades is
    noise, and a breaker that trips on noise is retired by the first person it wakes.
    """

    def __init__(
        self,
        name: str,
        *,
        declared_round_trip_bps: float,
        live_regime_round_trip_bps: float,
        multiple: float,
        window_days: int,
        min_round_trips: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="c", **kwargs)
        self.declared_round_trip_bps = float(declared_round_trip_bps)
        self.live_regime_round_trip_bps = float(live_regime_round_trip_bps)
        self.multiple = float(multiple)
        self.window_days = int(window_days)
        self.min_round_trips = int(min_round_trips)
        self.round_trips: list[RoundTrip] = []

    @property
    def assumed_round_trip_bps(self) -> float:
        """The tighter of the two declared assumptions; see the class docstring."""
        return min(self.declared_round_trip_bps, self.live_regime_round_trip_bps)

    @property
    def limit_bps(self) -> float:
        return self.assumed_round_trip_bps * self.multiple

    def record_round_trip(self, trip: RoundTrip) -> None:
        self.round_trips.append(trip)

    def check_condition(self, now: datetime | None = None, **_: Any):
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(days=self.window_days)
        recent = [t for t in self.round_trips if t.closed_at >= cutoff]
        if len(recent) < self.min_round_trips:
            return False, "", None, self.limit_bps
        notional = sum(t.notional for t in recent)
        if notional <= 0:
            return False, "", None, self.limit_bps
        realised = sum(t.cost_bps * t.notional for t in recent) / notional
        if realised > self.limit_bps:
            return (
                True,
                f"realised round-trip cost {realised:.2f} bps over {len(recent)} trips in "
                f"{self.window_days} days exceeds {self.multiple:g}x the assumed "
                f"{self.assumed_round_trip_bps:.2f} bps",
                realised,
                self.limit_bps,
            )
        return False, "", realised, self.limit_bps


# ---------------------------------------------------------------------------------------
# (d) — system tier, trips at once and latches
# ---------------------------------------------------------------------------------------
class ReconciliationBreaker(CircuitBreaker):
    """(d) The run record and ``positions_get`` disagree, or an unallocated magic is open.

    No HALF_OPEN probe. Every other breaker asks "is the strategy still working"; this one asks
    "do I know what I own", and probing that by trading again is not a recovery, it is a second
    order sent from a state nobody understands. ``BreakerManager`` latches the persistent kill
    switch through :func:`kill_switch_on_trip`.

    It counts EVERY magic, not just 260902: two positions carrying the retired legacy bot's
    magic 202500 were open on this login on 2026-09-08, and an unallocated magic must raise
    rather than be quietly filtered out.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        kwargs.setdefault("recovery_timeout", timedelta(days=3650))
        super().__init__(name, tier=BreakerTier.SYSTEM, kill_criterion="d", **kwargs)

    def check_condition(
        self,
        reconciliation_ok: bool | None = None,
        reconciliation_detail: str = "",
        **_: Any,
    ):
        if reconciliation_ok is None or reconciliation_ok:
            return False, "", None, None
        return True, f"reconciliation mismatch: {reconciliation_detail}", None, None

    def update(self, event_time: datetime | None = None, **kwargs: Any) -> BreakerState:
        """OPEN is terminal here: only a person calling :meth:`reset` clears it."""
        if self.state is BreakerState.OPEN:
            return self.state
        return super().update(event_time=event_time, **kwargs)


# ---------------------------------------------------------------------------------------
# (e) — ASYMMETRIC. Read the class docstrings before changing either.
# ---------------------------------------------------------------------------------------
class MissingDecisionBarBreaker(CircuitBreaker):
    """(e) for the **intraday** spec only: a run of decision instants with no bar to fill at.

    A single miss is a no-trade decision, not an incident, so it is counted and logged and the
    breaker escalates only after ``escalate_after`` misses inside ``window_sessions``.

    THIS BREAKER IS NEVER BUILT FOR THE OVERNIGHT SPEC. Measured on 1,930 research sessions the
    intraday instant missed 0 times and the overnight instant missed 504 (26 %), because the
    index shuts for the weekend at the Friday cash close and because through 2022 the daily
    break sat on that close. Escalating the overnight spec here would raise 504 incidents for a
    market structure ``setup.yaml`` already declares. The overnight spec's fault condition is
    :class:`OvernightFillCoverageBreaker` instead.
    """

    def __init__(
        self, name: str, *, spec: str, escalate_after: int, window_sessions: int, **kwargs: Any
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="e", **kwargs)
        if spec != "intraday":
            raise ValueError(
                "MissingDecisionBarBreaker is for the intraday spec only; a missing overnight "
                "bar is a NORMAL state (26 % of research sessions) and belongs to "
                "OvernightFillCoverageBreaker"
            )
        self.spec = spec
        self.escalate_after = int(escalate_after)
        self.window_sessions = int(window_sessions)
        self.recent: deque[bool] = deque(maxlen=self.window_sessions)

    def record_decision(self, *, had_bar: bool) -> None:
        self.recent.append(not bool(had_bar))

    @property
    def misses(self) -> int:
        return int(sum(self.recent))

    def check_condition(self, **_: Any):
        misses = self.misses
        if misses >= self.escalate_after:
            return (
                True,
                f"{self.spec}: {misses} decision instants without a bar inside the last "
                f"{len(self.recent)} (research sample: 0 of 1,930)",
                float(misses),
                float(self.escalate_after),
            )
        return False, "", float(misses), float(self.escalate_after)


class OvernightFillCoverageBreaker(CircuitBreaker):
    """(e) for the **overnight** spec: the fill coverage falls below the *pre-declared* floor.

    A missing overnight bar is not an incident. 504 of 1,930 research sessions had none, by
    year 2022 98 %, 2023 55 %, 2024 20 %, 2025 7 %, 2026 0 %. What WOULD be an incident is the
    coverage collapsing back towards the old regime, and the line for that was declared before
    phase 4 was run: ``setup.yaml::decision.min_fill_coverage`` = 0.60. Using the pre-declared
    floor is the whole point — a threshold chosen now, from the recent 93-100 % coverage, would
    be a number fitted to an outcome.

    The breaker abstains until the rolling window is full: fewer sessions than the window means
    the statistic does not exist, not that coverage is fine.
    """

    def __init__(
        self, name: str, *, min_fill_coverage: float, window_sessions: int, **kwargs: Any
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="e", **kwargs)
        self.min_fill_coverage = float(min_fill_coverage)
        self.window_sessions = int(window_sessions)
        self.recent: deque[bool] = deque(maxlen=self.window_sessions)

    def record_decision(self, *, had_bar: bool) -> None:
        self.recent.append(bool(had_bar))

    def coverage(self) -> float | None:
        if len(self.recent) < self.window_sessions:
            return None
        return float(np.mean(self.recent))

    def check_condition(self, **_: Any):
        coverage = self.coverage()
        if coverage is None:
            return False, "", None, self.min_fill_coverage
        if coverage < self.min_fill_coverage:
            return (
                True,
                f"overnight fill coverage {coverage:.1%} over {self.window_sessions} sessions "
                f"is below the pre-declared floor {self.min_fill_coverage:.0%} "
                "(setup.yaml::decision.min_fill_coverage)",
                coverage,
                self.min_fill_coverage,
            )
        return False, "", coverage, self.min_fill_coverage


# ---------------------------------------------------------------------------------------
# (f) — strategy tier, PER SYMBOL
# ---------------------------------------------------------------------------------------
class SwapBreaker(CircuitBreaker):
    """(f) The realised swap per night exceeds what ``setup.yaml::costs.swap`` declares.

    PER SYMBOL, because the two indices are not comparable: US500 declares -147.4 points per
    lot per night long and USTEC -592.7, and both declare 0.0 short. A single pooled tolerance
    would be four times too tight on one index and four times too loose on the other.

    ``swap_rollover3days`` is 5 = FRIDAY on both (read from ``symbol_info`` 2026-09-08), not
    Wednesday as on the FX pairs, so a Friday night is charged three times and is not a breach.
    A non-zero charge on a SHORT leg is a breach on its own: the declaration is 0.0.
    """

    def __init__(
        self,
        name: str,
        *,
        declared: dict[str, dict[str, float]],
        tolerance_points: float,
        consecutive_nights: int,
        rollover3days: str = "friday",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.STRATEGY, kill_criterion="f", **kwargs)
        self.declared = {
            symbol: {str(k): float(v) for k, v in sides.items()}
            for symbol, sides in declared.items()
        }
        self.tolerance_points = float(tolerance_points)
        self.consecutive_nights = int(consecutive_nights)
        self.rollover3days = str(rollover3days).lower()
        # symbol -> deque of (points_charged, side, was_triple_night)
        self.nights: dict[str, deque[tuple[float, str, bool]]] = {
            symbol: deque(maxlen=self.consecutive_nights) for symbol in self.declared
        }

    def record_night(
        self, symbol: str, *, points: float, side: str, triple: bool = False
    ) -> None:
        if symbol not in self.nights:
            raise KeyError(f"{symbol!r} is not in the declared swap table {sorted(self.declared)}")
        self.nights[symbol].append((float(points), str(side).lower(), bool(triple)))

    def check_condition(self, **_: Any):
        worst: tuple[float, str] | None = None
        for symbol, history in self.nights.items():
            if len(history) < self.consecutive_nights:
                continue
            breaches = 0
            excess_sum = 0.0
            for points, side, triple in history:
                declared = self.declared[symbol].get(side, 0.0) * (3.0 if triple else 1.0)
                # More negative than declared, or any charge at all where 0.0 was declared.
                excess = declared - points  # > 0 when the realised charge is worse
                if excess > self.tolerance_points:
                    breaches += 1
                    excess_sum += excess
            if breaches == self.consecutive_nights:
                mean_excess = excess_sum / breaches
                if worst is None or mean_excess > worst[0]:
                    worst = (mean_excess, symbol)
        if worst is not None:
            excess, symbol = worst
            return (
                True,
                f"{symbol}: the realised swap has been {excess:.1f} points per lot per night "
                f"worse than declared for {self.consecutive_nights} consecutive nights",
                excess,
                self.tolerance_points,
            )
        return False, "", None, self.tolerance_points


# ---------------------------------------------------------------------------------------
# (g) — system tier. No other bot has this one.
# ---------------------------------------------------------------------------------------
class TradeHoursDstBreaker(CircuitBreaker):
    """(g) The instrument's trading hours stop following ``America/New_York`` DST.

    THE FACT THIS GUARDS. ``setup.yaml`` declares two DIFFERENT time facts and they disagree
    with each other by design:

    * ``decision.server_clock.follows_dst_of: null`` — the MT5 server clock is UTC+0 all year
      (measured 2026-09-05).
    * ``decision.trade_hours_follow_dst_of: America/New_York`` — the daily CFD break ends at
      18:00 New York, measured on 39 readable, transition-free months of US500 H1 bars, which
      puts it at 21:00-22:00 UTC in summer and 22:00-23:00 UTC in winter.

    Every decision instant this bot computes is derived from the second fact. If the venue
    stops following New York DST — or moves the break — then the two decision instants, the
    execution bars, the labels and the ``blocked_execution_windows_local`` guard are all an
    hour wrong at once, silently, and the bot would keep trading through the change. That is
    why this is a SYSTEM-tier breaker and not a data-quality warning.

    One reading can be an outage or a thin evening, so it escalates only after
    ``escalate_after`` disagreeing observations inside ``window_sessions``. The check is fed
    from the H1 tape by the deployment loop, never from ``sessions_mt5.json``, which is an
    eight-week summer snapshot and would be wrong for five months of every year.
    """

    def __init__(
        self,
        name: str,
        *,
        expected_break_end_local: str,
        zone: str,
        escalate_after: int,
        window_sessions: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, kill_criterion="g", **kwargs)
        self.expected_break_end_local = time.fromisoformat(str(expected_break_end_local))
        self.zone = ZoneInfo(str(zone))
        self.zone_name = str(zone)
        self.escalate_after = int(escalate_after)
        self.window_sessions = int(window_sessions)
        self.recent: deque[tuple[date, time]] = deque(maxlen=self.window_sessions)

    def record_break_end(self, observed_utc: datetime) -> time:
        """Record the UTC instant the daily break ended; store it in venue-local time."""
        if observed_utc.tzinfo is None:
            observed_utc = observed_utc.replace(tzinfo=UTC)
        local = observed_utc.astimezone(self.zone)
        self.recent.append((local.date(), local.time().replace(second=0, microsecond=0)))
        return local.time()

    def disagreements(self) -> int:
        return sum(1 for _, t in self.recent if t != self.expected_break_end_local)

    def check_condition(self, **_: Any):
        bad = self.disagreements()
        if bad >= self.escalate_after:
            seen = sorted({t.isoformat(timespec="minutes") for _, t in self.recent})
            return (
                True,
                f"the daily break ended at {seen} {self.zone_name} on {bad} of the last "
                f"{len(self.recent)} sessions, not at "
                f"{self.expected_break_end_local.isoformat(timespec='minutes')}: the trading "
                "hours no longer follow decision.trade_hours_follow_dst_of, so every decision "
                "instant this bot computes is wrong",
                float(bad),
                float(self.escalate_after),
            )
        return False, "", float(bad), float(self.escalate_after)


# ---------------------------------------------------------------------------------------
# Not lettered: declared here rather than inferred later
# ---------------------------------------------------------------------------------------
class ConsecutiveLossBreaker(CircuitBreaker):
    """``max_consecutive`` losing round trips in a row (26/04). A pause, not a retirement."""

    def __init__(self, name: str, *, max_consecutive: int, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.TRADE, kill_criterion=None, **kwargs)
        self.max_consecutive = int(max_consecutive)
        self.streak = 0

    def record_trade(self, pnl: float) -> None:
        self.streak = self.streak + 1 if float(pnl) < 0 else 0

    def check_condition(self, **_: Any):
        if self.streak >= self.max_consecutive:
            return (
                True,
                f"{self.streak} consecutive losing round trips",
                float(self.streak),
                float(self.max_consecutive),
            )
        return False, "", float(self.streak), float(self.max_consecutive)


class LatencyBreaker(CircuitBreaker):
    """Infrastructure: the mean ``order_send`` round trip over the last ``n_recent`` orders."""

    def __init__(self, name: str, *, max_latency_ms: float, n_recent: int, **kwargs: Any) -> None:
        super().__init__(name, tier=BreakerTier.SYSTEM, kill_criterion=None, **kwargs)
        self.max_latency_ms = float(max_latency_ms)
        self.n_recent = int(n_recent)
        self.latency_history: list[float] = []

    def record_latency(self, latency_ms: float) -> None:
        self.latency_history.append(float(latency_ms))
        self.latency_history = self.latency_history[-100:]

    def check_condition(self, **_: Any):
        if not self.latency_history:
            return False, "", None, self.max_latency_ms
        mean = float(np.mean(self.latency_history[-self.n_recent :]))
        if mean > self.max_latency_ms:
            return (
                True,
                f"order latency {mean:.0f} ms exceeds {self.max_latency_ms:.0f} ms",
                mean,
                self.max_latency_ms,
            )
        return False, "", mean, self.max_latency_ms


# ---------------------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------------------
@dataclass
class BotBreakers:
    """Every strategy-tier breaker of this bot, plus the manager that answers "may I trade?"."""

    manager: BreakerManager
    drawdown: DrawdownBreaker
    daily_loss: DailyLossBreaker
    hit_rate: dict[str, RollingHitRateBreaker]  # one per spec — see criterion (b)
    cost: RealisedCostBreaker
    reconciliation: ReconciliationBreaker
    missing_bar_intraday: MissingDecisionBarBreaker
    overnight_coverage: OvernightFillCoverageBreaker
    swap: SwapBreaker
    trade_hours_dst: TradeHoursDstBreaker
    consecutive_loss: ConsecutiveLossBreaker
    latency: LatencyBreaker
    config_source: str
    pending_user_approval: bool
    evidence_boundary: dict[str, Any] = field(default_factory=dict)

    def allows_trading(self) -> bool:
        return self.manager.allows_trading()

    def kill_criteria_covered(self) -> dict[str, list[str]]:
        """Letter -> the breakers that implement it. A letter with no breaker is a gap."""
        out: dict[str, list[str]] = {}
        for name, breaker in self.manager.breakers.items():
            if breaker.kill_criterion:
                out.setdefault(breaker.kill_criterion, []).append(name)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def to_record(self) -> dict[str, Any]:
        return {
            "config_source": self.config_source,
            "pending_user_approval": self.pending_user_approval,
            "kill_criteria_covered": self.kill_criteria_covered(),
            "evidence_boundary": self.evidence_boundary,
            **self.manager.to_records(),
        }


def build_manager(
    config_path: Path | str | None = None,
    *,
    on_any_trip: Any = None,
    allow_draft_criteria: bool = False,
) -> BotBreakers:
    """Build every breaker from ``deploy/risk_config.yaml``. No threshold is written here.

    ``allow_draft_criteria`` must be passed explicitly while
    ``breakers.pending_user_approval`` is true. It is not a permission to trade — it is an
    acknowledgement that these breakers implement a draft of the kill criteria that the user
    has not approved, so a monitoring dashboard built on them is a rehearsal.
    """
    config = load_breaker_config(config_path)
    block = config["breakers"]
    missing = config["missing_decision_bar"]
    pending = bool(block.get("pending_user_approval", False))
    if pending and not allow_draft_criteria:
        raise DraftCriteriaError(
            f"{config['source']}::breakers.pending_user_approval is true: the kill criteria of "
            f"{BOT_ID} are a DRAFT the user has not approved (BOT.md, 'Kill criteria'). Pass "
            "allow_draft_criteria=True to build them for a dry run; they are not a permission "
            "to trade, and this bot has 0 phase-5 survivors and an unscored holdout."
        )

    timeout = timedelta(hours=float(block["recovery_timeout_hours"]))
    manager = BreakerManager(on_any_trip=on_any_trip)

    drawdown = DrawdownBreaker(
        "strategy_drawdown",
        max_drawdown_pct=float(block["drawdown"]["max_drawdown_pct"]),
        recovery_timeout=timeout,
    )
    daily_loss = DailyLossBreaker(
        "strategy_daily_loss",
        max_daily_loss_pct=float(block["daily_loss"]["max_daily_loss_pct"]),
        recovery_timeout=timeout,
    )
    hit_block = block["hit_rate"]
    hit_rate = {
        spec: RollingHitRateBreaker(
            f"strategy_hit_rate_{spec}",
            spec=spec,
            window_sessions=int(hit_block["window_sessions"]),
            min_hit_rate=float(hit_block["min_hit_rate"]),
            consecutive_sessions=int(hit_block["consecutive_sessions"]),
            recovery_timeout=timeout,
        )
        for spec in hit_block["per_spec"]
    }
    cost_block = block["cost"]
    cost = RealisedCostBreaker(
        "strategy_realised_cost",
        declared_round_trip_bps=float(cost_block["declared_round_trip_bps"]),
        live_regime_round_trip_bps=float(cost_block["live_regime_round_trip_bps"]),
        multiple=float(cost_block["multiple"]),
        window_days=int(cost_block["window_days"]),
        min_round_trips=int(cost_block["min_round_trips"]),
        recovery_timeout=timeout,
    )
    reconciliation = ReconciliationBreaker("system_reconciliation")
    missing_bar_intraday = MissingDecisionBarBreaker(
        "strategy_missing_decision_bar_intraday",
        spec="intraday",
        escalate_after=int(missing["intraday"]["escalate_after"]),
        window_sessions=int(missing["intraday"]["window_sessions"]),
        recovery_timeout=timeout,
    )
    overnight_coverage = OvernightFillCoverageBreaker(
        "strategy_overnight_fill_coverage",
        min_fill_coverage=float(missing["overnight"]["min_fill_coverage"]),
        window_sessions=int(missing["overnight"]["coverage_window_sessions"]),
        recovery_timeout=timeout,
    )
    swap_block = block["swap"]
    swap = SwapBreaker(
        "strategy_swap",
        declared=swap_block["declared_points_per_lot_per_night"],
        tolerance_points=float(swap_block["tolerance_points"]),
        consecutive_nights=int(swap_block["consecutive_nights"]),
        rollover3days=str(swap_block.get("rollover3days", "friday")),
        recovery_timeout=timeout,
    )
    dst_block = block["trade_hours_dst"]
    trade_hours_dst = TradeHoursDstBreaker(
        "system_trade_hours_dst",
        expected_break_end_local=dst_block["expected_break_end_local"],
        zone=dst_block["zone"],
        escalate_after=int(dst_block["escalate_after"]),
        window_sessions=int(dst_block["window_sessions"]),
        recovery_timeout=timeout,
    )
    consecutive = ConsecutiveLossBreaker(
        "trade_consecutive_loss",
        max_consecutive=int(block["consecutive_loss"]["max_consecutive"]),
        recovery_timeout=timeout,
    )
    latency = LatencyBreaker(
        "system_latency",
        max_latency_ms=float(block["latency"]["max_latency_ms"]),
        n_recent=int(block["latency"]["n_recent"]),
        recovery_timeout=timeout,
    )

    for breaker in (
        drawdown,
        daily_loss,
        *hit_rate.values(),
        cost,
        reconciliation,
        missing_bar_intraday,
        overnight_coverage,
        swap,
        trade_hours_dst,
        consecutive,
        latency,
    ):
        manager.add_breaker(breaker)

    boundary = config["evidence_boundary"]
    return BotBreakers(
        manager=manager,
        drawdown=drawdown,
        daily_loss=daily_loss,
        hit_rate=hit_rate,
        cost=cost,
        reconciliation=reconciliation,
        missing_bar_intraday=missing_bar_intraday,
        overnight_coverage=overnight_coverage,
        swap=swap,
        trade_hours_dst=trade_hours_dst,
        consecutive_loss=consecutive,
        latency=latency,
        config_source=config["source"],
        pending_user_approval=pending,
        evidence_boundary={
            "phase5_survivors": boundary.get("phase5_survivors"),
            "phase5_specs_scored": boundary.get("phase5_specs_scored"),
            "trial_count_K": boundary.get("phase5_trial_count"),
            "phase6_opened": boundary.get("phase6_opened"),
            "holdout_scored": False,
            "live_trading_permitted": False,
        },
    )


def kill_switch_on_trip(safe_broker: Any):
    """A ``BreakerManager.on_any_trip`` callback that latches the persistent kill switch."""

    def handler(event: BreakerEvent) -> None:
        logger.error("latching the kill switch after %s: %s", event.breaker_name, event.reason)
        safe_broker.enable_kill_switch(f"{event.breaker_name}: {event.reason}")

    return handler


__all__ = [
    "SPECS",
    "BotBreakers",
    "BreakerEvent",
    "BreakerState",
    "ConsecutiveLossBreaker",
    "DailyLossBreaker",
    "DraftCriteriaError",
    "DrawdownBreaker",
    "LatencyBreaker",
    "MissingDecisionBarBreaker",
    "OvernightFillCoverageBreaker",
    "RealisedCostBreaker",
    "ReconciliationBreaker",
    "RollingHitRateBreaker",
    "RoundTrip",
    "SwapBreaker",
    "TradeHoursDstBreaker",
    "build_manager",
    "kill_switch_on_trip",
    "load_breaker_config",
    "transition_breaker",
]
