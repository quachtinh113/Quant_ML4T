"""This bot's circuit breakers - ``26_mlops_governance/04_circuit_breakers.py``, strategy tier.

The ``CLOSED -> OPEN -> HALF_OPEN`` state machine is **not** written here: it lives once, in
``bots/_shared/monitor/base.py``, and every bot subclasses ``CircuitBreaker`` and supplies a
``check_condition`` of a few lines. That is the notebook's own conclusion - *"Once that
lifecycle is standardized, each risk rule becomes a small condition"* - and it is why four bots
share one machine.

ONE BREAKER PER LETTERED KILL CRITERION of ``bots/exness_gold_sess/BOT.md``, and the letter
travels into the ``BreakerEvent`` so a trip can be read back against the written criterion
rather than against a class name:

    (a) drawdown from peak > 8 % of allocated capital          -> DrawdownBreaker
        derived: 2 % in one day                                -> DailyLossBreaker
    (b) rolling 63-decision hit rate < 0.5 for 21 sessions     -> RollingHitRateBreaker
    (c) realised round trip > 1.5x assumed for a month         -> RealisedCostBreaker
    (d) an unexplained position carrying magic 260903          -> ReconciliationBreaker
    (e) the decision bar is missing within tolerance           -> MissingDecisionBarBreaker
    (f) realised nightly swap above the declared value, a week -> SwapBreaker
    (g) a metal halts or gaps through an UNDECLARED closure (refined 2026-09-08; the daily
        UTC break while the bot still holds                    -> OvernightGapBreaker

**The account tier is not here.** Equity drawdown, daily loss, margin level, distance to the
broker's stop-out and open positions across *every* magic live in
``bots/_shared/monitor/account.py``; one trip there halts **every** bot on the login. A bot
never re-implements an account breaker, and an account breaker never encodes a strategy rule.

TWO THINGS SPECIFIC TO THIS BOT
-------------------------------
* **(b) is a hit rate, not an IC.** Over two metals a cross-sectional rank correlation takes
  the values +/-1; ``BOT.md`` says so and the research side records the same fact in
  ``case_studies/exness_gold_sess/_model_reading.py``.
* **(c) and (g) are per metal.** Silver's round trip is 9.4 bps against gold's 1.2 - 7.7 times
  wider - so one pooled cost number would hide a silver blow-out behind gold's tightness. The
  same applies to a gap through the daily break.

**Every threshold comes from ``deploy/risk_config.yaml``. Nothing here carries a number**, and
:func:`build_manager` refuses to build while ``breakers.pending_user_approval`` is true, which
it is: the criteria are the mentor's draft and the user has not approved them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

from bots._shared.monitor import BreakerManager, BreakerTier, CircuitBreaker

BOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RISK_CONFIG = BOT_DIR / "deploy" / "risk_config.yaml"

__all__ = [
    "ConsecutiveLossBreaker",
    "DailyLossBreaker",
    "DrawdownBreaker",
    "LatencyBreaker",
    "MissingDecisionBarBreaker",
    "OvernightGapBreaker",
    "ReconciliationBreaker",
    "RealisedCostBreaker",
    "RollingHitRateBreaker",
    "SwapBreaker",
    "build_manager",
    "load_breaker_config",
]


def load_breaker_config(path: Path | str | None = None) -> dict[str, Any]:
    """The ``breakers`` block of ``deploy/risk_config.yaml``, with its source recorded."""
    config_path = Path(path or DEFAULT_RISK_CONFIG)
    raw = yaml.safe_load(config_path.read_text())
    block = dict(raw["breakers"])
    block["_source"] = str(config_path)
    block["_capital"] = raw["capital"]
    block["_cfd_guards"] = raw.get("cfd_guards", {})
    return block


def _tier(name: str) -> BreakerTier:
    return BreakerTier(name)


# ---------------------------------------------------------------------------------------
# (a) and its derived daily rule
# ---------------------------------------------------------------------------------------
class DrawdownBreaker(CircuitBreaker):
    """Kill criterion (a): drawdown from the peak exceeds a share of allocated capital.

    The peak is the bot's own high-water mark, persisted with the run records so it survives a
    restart - a peak that resets with the process is not a peak. Note the denominator: this is
    a fraction of the **capital allocated to this bot**, not of account equity, because
    ``xau_fx_mt5`` shares the login and its profit and loss must not move this bot's breaker.
    """

    def __init__(self, *, max_drawdown_pct: float, **kwargs: Any) -> None:
        super().__init__("drawdown", **kwargs)
        self.max_drawdown_pct = float(max_drawdown_pct)

    def check_condition(self, **kwargs: Any):
        peak = kwargs.get("bot_equity_peak")
        equity = kwargs.get("bot_equity")
        if peak in (None, 0) or equity is None:
            return False, "no bot equity history yet", None, self.max_drawdown_pct
        drawdown = max(0.0, (float(peak) - float(equity)) / float(peak))
        return (
            drawdown > self.max_drawdown_pct,
            f"drawdown {drawdown:.2%} of the allocated capital's peak",
            drawdown,
            self.max_drawdown_pct,
        )


class DailyLossBreaker(CircuitBreaker):
    """Derived from (a): loss in one trading day exceeds a share of allocated capital."""

    def __init__(self, *, max_daily_loss_pct: float, allocated: float, **kwargs: Any) -> None:
        super().__init__("daily_loss", **kwargs)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.allocated = float(allocated)

    def check_condition(self, **kwargs: Any):
        pnl = kwargs.get("day_pnl")
        if pnl is None:
            return False, "no profit and loss recorded for today", None, None
        limit = self.max_daily_loss_pct * self.allocated
        return (
            float(pnl) < -limit,
            f"day P&L {float(pnl):+.2f} against a limit of -{limit:.2f}",
            float(pnl),
            -limit,
        )


# ---------------------------------------------------------------------------------------
# (b) hit rate - NOT an IC
# ---------------------------------------------------------------------------------------
class RollingHitRateBreaker(CircuitBreaker):
    """Kill criterion (b): rolling hit rate below a floor for N consecutive sessions.

    Two properties worth being explicit about:

    * it is a **hit rate and not an information coefficient**, because an IC over two names is
      not a readable statistic (``BOT.md``; ``_model_reading.py`` measures the same thing on the
      research side);
    * it trips on **persistence**, not on one bad window. A 63-decision hit rate crosses 0.5 in
      both directions constantly on any book with no edge; requiring 21 consecutive sessions
      below the floor is what distinguishes deterioration from noise. The counter is reset by a
      single session at or above the floor, which is deliberate: a rule that never resets
      eventually trips on any book.
    """

    def __init__(self, *, window_decisions: int, floor: float, consecutive_sessions: int, **kwargs: Any) -> None:
        super().__init__("hit_rate", **kwargs)
        self.window_decisions = int(window_decisions)
        self.floor = float(floor)
        self.consecutive_sessions = int(consecutive_sessions)
        self.below_streak = 0

    def check_condition(self, **kwargs: Any):
        hit_rate = kwargs.get("rolling_hit_rate")
        n_decisions = int(kwargs.get("decisions_in_window") or 0)
        if hit_rate is None or n_decisions < self.window_decisions:
            return (
                False,
                f"{n_decisions} decisions in the window, {self.window_decisions} needed",
                None,
                self.floor,
            )
        if float(hit_rate) < self.floor:
            self.below_streak += 1
        else:
            self.below_streak = 0
        return (
            self.below_streak >= self.consecutive_sessions,
            f"hit rate {float(hit_rate):.3f} below {self.floor} for {self.below_streak} "
            f"consecutive sessions (limit {self.consecutive_sessions})",
            float(hit_rate),
            self.floor,
        )


# ---------------------------------------------------------------------------------------
# (c) realised cost, per metal
# ---------------------------------------------------------------------------------------
@dataclass
class RoundTrip:
    """One completed round trip, priced from the FILLS and not from the weight series.

    ``PRICE_GRID_DECLARATION.md`` section 3.4: the hold is a broker-level ``time_exit``
    position rule, and a rule-driven exit does not pass through the weight series, so the
    registry's ``avg_turnover`` under-reports by roughly half. Cost has to be read from
    ``fills.parquet`` / ``trades.parquet``, and live it is read from the fill records the
    adapter returns.
    """

    symbol: str
    cost_bps: float


class RealisedCostBreaker(CircuitBreaker):
    """Kill criterion (c): realised round-trip cost above a multiple of the assumed cost.

    Per metal, because silver's assumed round trip is 9.4 bps against gold's 1.2 and a pooled
    average would let a silver blow-out hide behind gold's tightness.
    """

    def __init__(self, *, multiple: float, assumed_bps: dict[str, float], **kwargs: Any) -> None:
        super().__init__("realised_cost", **kwargs)
        self.multiple = float(multiple)
        self.assumed_bps = {k: float(v) for k, v in assumed_bps.items()}

    def check_condition(self, **kwargs: Any):
        trips: list[RoundTrip] = list(kwargs.get("round_trips") or [])
        if not trips:
            return False, "no completed round trips in the window", None, None
        breaches = []
        worst_ratio = 0.0
        for symbol in sorted({t.symbol for t in trips}):
            costs = [t.cost_bps for t in trips if t.symbol == symbol]
            assumed = self.assumed_bps.get(symbol)
            if assumed in (None, 0):
                continue
            realised = sum(costs) / len(costs)
            ratio = realised / assumed
            worst_ratio = max(worst_ratio, ratio)
            if ratio > self.multiple:
                breaches.append(f"{symbol} {realised:.2f} bps = {ratio:.2f}x the assumed {assumed}")
        return (
            bool(breaches),
            "; ".join(breaches) or f"worst realised/assumed ratio {worst_ratio:.2f}x",
            worst_ratio,
            self.multiple,
        )


# ---------------------------------------------------------------------------------------
# (d) reconciliation
# ---------------------------------------------------------------------------------------
class ReconciliationBreaker(CircuitBreaker):
    """Kill criterion (d): an unexplained position carrying this bot's magic.

    Trips immediately, with no tolerance and no streak. A position under magic 260903 that this
    bot did not open is either a bug in this bot or another process using its magic, and both
    are reasons to stop rather than to trade beside it. Positions carrying **another** magic are
    reported as context and are not this bot's to touch: ``xau_fx_mt5`` (260905) trades XAUUSD
    on the same login.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("reconciliation", **kwargs)

    def check_condition(self, **kwargs: Any):
        result = kwargs.get("reconciliation") or {}
        unexplained = list(result.get("unexplained") or [])
        missing = list(result.get("missing") or [])
        mismatched = list(result.get("mismatched") or [])
        problems = len(unexplained) + len(missing) + len(mismatched)
        return (
            problems > 0,
            f"unexplained {unexplained}, missing {missing}, mismatched {mismatched} "
            f"(foreign magics present: {result.get('foreign_magics')})",
            float(problems),
            0.0,
        )


# ---------------------------------------------------------------------------------------
# (e) missing decision bar
# ---------------------------------------------------------------------------------------
class MissingDecisionBarBreaker(CircuitBreaker):
    """Kill criterion (e): the decision bar is missing within tolerance.

    One missing bar is a **no-trade for that session**, not a halt: the census measured the
    decision bar missing on 11 London and 10 New York days out of 2,467, all of them US public
    holidays and a handful of thin Fridays. It becomes a halt when it repeats - ``escalate_after``
    occurrences inside ``window_sessions`` - because at that point the data feed is the problem
    and not the calendar.
    """

    def __init__(self, *, escalate_after: int, window_sessions: int, **kwargs: Any) -> None:
        super().__init__("missing_decision_bar", **kwargs)
        self.escalate_after = int(escalate_after)
        self.window_sessions = int(window_sessions)
        self.recent: list[int] = []

    def check_condition(self, **kwargs: Any):
        check = kwargs.get("decision_bar_check") or {}
        missing = bool(check.get("symbols_missing"))
        self.recent.append(int(missing))
        self.recent = self.recent[-self.window_sessions :]
        count = sum(self.recent)
        return (
            count >= self.escalate_after,
            f"{count} sessions with a missing decision bar in the last {len(self.recent)} "
            f"(escalates at {self.escalate_after}); this session: {check.get('symbols_missing')}",
            float(count),
            float(self.escalate_after),
        )


# ---------------------------------------------------------------------------------------
# (f) swap
# ---------------------------------------------------------------------------------------
class SwapBreaker(CircuitBreaker):
    """Kill criterion (f): realised nightly swap above the declared value for a week.

    The declared value is **0.0 on both metals, read from ``symbol_info`` on the DEMO account**
    on 2026-09-07, with ``swap_mode 1`` (points) and ``swap_rollover3days 3`` (Wednesday). A
    demo reading is not the Pro account's reading. So a trip here may be the assumption's fault
    rather than the market's, and the reason string says so rather than leaving a reader to
    conclude the broker changed its terms.
    """

    def __init__(
        self, *, declared: dict[str, dict[str, float]], window_nights: int, is_demo_reading: bool, **kwargs: Any
    ) -> None:
        super().__init__("swap", **kwargs)
        self.declared = declared
        self.window_nights = int(window_nights)
        self.is_demo_reading = bool(is_demo_reading)

    def check_condition(self, **kwargs: Any):
        charged = kwargs.get("swap_points_by_symbol") or {}
        if not charged:
            return False, "no swap has been charged in the window", None, None
        breaches = []
        for symbol, points in sorted(charged.items()):
            side = kwargs.get("position_side", {}).get(symbol, "long")
            declared = float(self.declared.get(symbol, {}).get(side, 0.0))
            if abs(float(points)) > abs(declared):
                breaches.append(f"{symbol} {points} points/night against a declared {declared}")
        caveat = (
            " NOTE: the declared value is a DEMO account reading (2026-09-07); re-read it on "
            "the real Pro account before treating this trip as a change in the market."
            if self.is_demo_reading
            else ""
        )
        return (
            bool(breaches),
            ("; ".join(breaches) or "swap within the declared values") + caveat,
            float(len(breaches)),
            0.0,
        )


# ---------------------------------------------------------------------------------------
# (g) the metals-specific one
# ---------------------------------------------------------------------------------------
class OvernightGapBreaker(CircuitBreaker):
    """Kill criterion (g): a metal gaps or halts through the daily break while the bot holds.

    This breaker has no counterpart in the FX bot. The server takes a one-hour break every day
    - 21:00-22:00 UTC while the US is on DST, 22:00-23:00 when it is not - and exactly one of
    those two hourly bars prints on a given day. A position held across it takes whatever gap
    the reopen brings, with no chance to act, and gold gaps on weekend news
    (``bots/assets/XAUUSD.md``).

    Per metal, and it fires on the **held** symbol only: a gap in a metal the bot is flat in is
    market colour, not a risk event.

    **REFINED 2026-09-08 (``NY_EXIT_DECLARATION.md`` section 6.3, recorded in ``BOT.md``): after
    the New York exit fix this breaker no longer fires on the DAILY break, and phase 8 must not
    treat that silence as a fault.** The New York session close *is* the start of the break in
    both seasons (both anchored on 17:00 America/New_York), so the book now holds 7 grid bars and
    exits at the last print BEFORE the break - 20:00 UTC on daylight saving, 21:00 UTC otherwise -
    on 100 % of positions; London exits at 17:00 UTC. Measured on realised fills: **0** positions
    of either 8-hour book are held across the break, and **0** positions of the 24-hour book cross
    a weekend. Before the fix this criterion fired on 100 % of New York positions *by design*,
    368 of 1,984 of them across a whole weekend.

    What it still has to catch, and the reason the ``held_symbols`` gate is the load-bearing part
    rather than the window: an **undeclared** closure. A holiday early close (66 decision rows in
    the validation window, 33 a metal, all New York - traded normally, by declaration) leaves a
    position open across a closure the session calendar does not contain; so does a broker halt,
    and so does any gap outside the declared session calendar. Those are what (g) is for now.
    """

    def __init__(self, *, max_gap_pct: float, **kwargs: Any) -> None:
        super().__init__("overnight_gap", **kwargs)
        self.max_gap_pct = float(max_gap_pct)

    def check_condition(self, **kwargs: Any):
        gaps = kwargs.get("break_gap_pct_by_symbol") or {}
        held = set(kwargs.get("held_symbols") or [])
        breaches = [
            f"{symbol} gapped {float(gap):+.2%} through the daily break while held"
            for symbol, gap in sorted(gaps.items())
            if symbol in held and abs(float(gap)) > self.max_gap_pct
        ]
        worst = max((abs(float(g)) for s, g in gaps.items() if s in held), default=0.0)
        return (
            bool(breaches),
            "; ".join(breaches) or f"largest gap through the break while held {worst:.2%}",
            worst,
            self.max_gap_pct,
        )


# ---------------------------------------------------------------------------------------
# Generic trade and system tiers (26/04 section 3), not lettered criteria
# ---------------------------------------------------------------------------------------
class ConsecutiveLossBreaker(CircuitBreaker):
    """N losing decisions in a row. A trade-tier guard, not one of the written criteria."""

    def __init__(self, *, max_consecutive: int, **kwargs: Any) -> None:
        super().__init__("consecutive_loss", **kwargs)
        self.max_consecutive = int(max_consecutive)
        self.streak = 0

    def check_condition(self, **kwargs: Any):
        outcome = kwargs.get("last_decision_pnl")
        if outcome is None:
            return False, "no decision outcome yet", None, float(self.max_consecutive)
        self.streak = self.streak + 1 if float(outcome) < 0 else 0
        return (
            self.streak >= self.max_consecutive,
            f"{self.streak} consecutive losing decisions",
            float(self.streak),
            float(self.max_consecutive),
        )


class LatencyBreaker(CircuitBreaker):
    """A cycle that takes longer than its budget. The decision instant is a bar close and the
    order goes at the next bar's open, so a cycle that overruns is sizing on a price that has
    already moved."""

    def __init__(self, *, max_cycle_seconds: float, **kwargs: Any) -> None:
        super().__init__("latency", **kwargs)
        self.max_cycle_seconds = float(max_cycle_seconds)

    def check_condition(self, **kwargs: Any):
        elapsed = kwargs.get("cycle_seconds")
        if elapsed is None:
            return False, "no cycle timing recorded", None, self.max_cycle_seconds
        return (
            float(elapsed) > self.max_cycle_seconds,
            f"cycle took {float(elapsed):.1f} s",
            float(elapsed),
            self.max_cycle_seconds,
        )


# ---------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------
def build_manager(
    config: dict[str, Any] | None = None, *, allow_draft_criteria: bool = False
) -> BreakerManager:
    """Build every breaker from ``deploy/risk_config.yaml``. No threshold is written here.

    Refuses to build while ``breakers.pending_user_approval`` is true, which it is today: the
    kill criteria in ``BOT.md`` are the mentor's draft and the user has not approved them. A
    breaker set built on unapproved thresholds would let the bot stop - or, worse, *not* stop -
    on a rule nobody agreed to. ``allow_draft_criteria=True`` exists for the tests, which need
    the objects without claiming they are authorised.
    """
    config = config or load_breaker_config()
    if config.get("pending_user_approval", True) and not allow_draft_criteria:
        raise RuntimeError(
            f"{config.get('_source')}::breakers.pending_user_approval is true: the kill criteria "
            "in bots/exness_gold_sess/BOT.md are the mentor's DRAFT and the user has not approved "
            "them. Get the approval, record its date in BOT.md, clear the flag, then build. "
            "Pass allow_draft_criteria=True only in tests."
        )

    timeout = timedelta(hours=float(config.get("recovery_timeout_hours", 24)))
    allocated = float(config["_capital"]["allocated"])
    manager = BreakerManager()

    drawdown = config["drawdown"]
    manager.add_breaker(
        DrawdownBreaker(
            max_drawdown_pct=drawdown["max_drawdown_pct"],
            recovery_timeout=timeout,
            tier=_tier(drawdown["tier"]),
            kill_criterion=drawdown["kill_criterion"],
        )
    )
    daily = config["daily_loss"]
    manager.add_breaker(
        DailyLossBreaker(
            max_daily_loss_pct=daily["max_daily_loss_pct"],
            allocated=allocated,
            recovery_timeout=timeout,
            tier=_tier(daily["tier"]),
            kill_criterion=daily["kill_criterion"],
        )
    )
    hit = config["hit_rate"]
    manager.add_breaker(
        RollingHitRateBreaker(
            window_decisions=hit["window_decisions"],
            floor=hit["floor"],
            consecutive_sessions=hit["consecutive_sessions"],
            recovery_timeout=timeout,
            tier=_tier(hit["tier"]),
            kill_criterion=hit["kill_criterion"],
        )
    )
    cost = config["cost"]
    manager.add_breaker(
        RealisedCostBreaker(
            multiple=cost["multiple_of_assumed"],
            assumed_bps=cost["assumed_round_trip_bps"],
            recovery_timeout=timeout,
            tier=_tier(cost["tier"]),
            kill_criterion=cost["kill_criterion"],
        )
    )
    reconciliation = config["reconciliation"]
    manager.add_breaker(
        ReconciliationBreaker(
            recovery_timeout=timeout,
            tier=_tier(reconciliation["tier"]),
            kill_criterion=reconciliation["kill_criterion"],
        )
    )
    missing = config["_cfd_guards"]["missing_decision_bar"]
    manager.add_breaker(
        MissingDecisionBarBreaker(
            escalate_after=missing["escalate_after"],
            window_sessions=missing["window_sessions"],
            recovery_timeout=timeout,
            tier=BreakerTier.SYSTEM,
            kill_criterion="e",
        )
    )
    swap = config["swap"]
    manager.add_breaker(
        SwapBreaker(
            declared=swap["points_per_lot_per_night"],
            window_nights=swap["window_nights"],
            is_demo_reading=swap.get("assumed_is_demo_reading", True),
            recovery_timeout=timeout,
            tier=_tier(swap["tier"]),
            kill_criterion=swap["kill_criterion"],
        )
    )
    gap = config["overnight_gap"]
    manager.add_breaker(
        OvernightGapBreaker(
            max_gap_pct=gap["max_gap_pct_through_break"],
            recovery_timeout=timeout,
            tier=_tier(gap["tier"]),
            kill_criterion=gap["kill_criterion"],
        )
    )
    consecutive = config["consecutive_loss"]
    manager.add_breaker(
        ConsecutiveLossBreaker(
            max_consecutive=consecutive["max_consecutive_losing_decisions"],
            recovery_timeout=timeout,
            tier=_tier(consecutive["tier"]),
        )
    )
    latency = config["latency"]
    manager.add_breaker(
        LatencyBreaker(
            max_cycle_seconds=latency["max_cycle_seconds"],
            recovery_timeout=timeout,
            tier=_tier(latency["tier"]),
        )
    )
    return manager
