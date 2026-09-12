"""The two decision instants of ``exness_usidx_sess``, DERIVED — never typed.

EVIDENCE BOUNDARY: this module computes *when* the bot would decide. It says nothing about
what it would decide or whether that decision is worth anything. The bot has 0 phase-5
survivors at K = 1,780 and an unscored holdout; live trading is forbidden.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
``bots/assets/US500.md`` and ``BOT.md`` both describe the schedule as "15:00 UTC winter /
14:00 summer" and "21:00 / 20:00 UTC". **Those are outputs, not inputs**, and a code path that
carries them as constants is wrong for roughly five months of every year and on every half day.
Two independent time facts collide here and ``setup.yaml`` declares them separately on purpose:

* ``decision.server_clock`` — the MT5 server clock is UTC+0 **all year**
  (``follows_dst_of: null``, measured 2026-09-05 with ``ServerClock.measure``).
* ``decision.trade_hours_follow_dst_of: America/New_York`` — the *instrument's* trading hours
  DO follow New York DST. Measured on 39 readable, transition-free months of US500 H1 bars: the
  daily CFD break ends at 18:00 New York in every one of them.

So the venue moves twice a year and the clock the bot reads does not. Converting between them
is exactly what :class:`bots._shared.sessions.ServerClock` is for, and deriving the session
from the NYSE calendar is what picks up the 13:00 half-days that no DST rule would.

THREE SOURCES, CROSS-CHECKED AGAINST EACH OTHER
-----------------------------------------------
1. ``case_studies/exness_usidx_sess/_features.session_grid`` — the NYSE calendar through
   ``pandas_market_calendars``. **The same function ``01_feasibility_analysis``, ``02_labels``,
   ``03_financial_features`` and the point-in-time test call**, so the live schedule and the
   research schedule cannot drift apart. It carries the half-days.
2. ``bots/_shared/sessions.SESSIONS["us_cash"]`` — 09:30-16:00 ``America/New_York`` with real
   DST through ``zoneinfo``. Used as an INDEPENDENT CHECK on (1), not as a second source of
   truth: :func:`decision_instants` asserts the two agree on a full session and reports the
   difference on a half day rather than hiding it.
3. ``bots/_shared/sessions.ServerClock`` — UTC to the server clock, from the measured offset.

The H1 rounding rule is ``setup.yaml::decision``, quoted where it is applied.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from bots._shared.sessions import SESSIONS, ServerClock  # noqa: E402

#: ``setup.yaml::decision.specs``. ``intraday`` decides at the cash open (+ delay), ``overnight``
#: at the cash close. They are two books, not two horizons of one book.
SPECS = ("intraday", "overnight")

#: The bar grid these decisions live on. ``setup.yaml::decision`` is an H1 grid throughout.
BAR_MINUTES = 60


@dataclass(frozen=True)
class DecisionInstant:
    """One decision of one spec on one cash session, in three clocks and with its fill bar."""

    session: date
    spec: str
    #: UTC instant the decision bar CLOSES. This is the snapshot: everything the model may read
    #: has closed by now.
    decision_utc: datetime
    #: The same instant on the MT5 server clock (naive), through ``ServerClock.to_server``.
    decision_server: datetime
    #: The same instant in venue-local time. The DST rule lives here and nowhere else.
    decision_local: datetime
    #: UTC instant the execution bar OPENS. ``decision.execution_delay: next_bar_open``.
    exec_open_utc: datetime
    exec_open_server: datetime
    #: The venue session boundaries the instant was derived from, for the run record.
    cash_open_utc: datetime
    cash_close_utc: datetime
    half_day: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "session": self.session.isoformat(),
            "spec": self.spec,
            "decision_utc": self.decision_utc.isoformat(),
            "decision_server": self.decision_server.isoformat(),
            "decision_local": self.decision_local.isoformat(),
            "exec_open_utc": self.exec_open_utc.isoformat(),
            "exec_open_server": self.exec_open_server.isoformat(),
            "cash_open_utc": self.cash_open_utc.isoformat(),
            "cash_close_utc": self.cash_close_utc.isoformat(),
            "half_day": self.half_day,
        }


def server_clock_from_config(decision_block: dict[str, Any]) -> ServerClock:
    """``ServerClock`` from ``risk_config.yaml::decision.server_clock`` (mirrors setup.yaml)."""
    raw = dict(decision_block["server_clock"])
    return ServerClock.from_dict(raw)


def _ceil_to_bar(instant: datetime, *, bar_minutes: int = BAR_MINUTES) -> datetime:
    """The first bar CLOSE at or after ``instant``, on a grid anchored at the UTC hour."""
    anchor = instant.replace(minute=0, second=0, microsecond=0)
    while anchor < instant:
        anchor += timedelta(minutes=bar_minutes)
    return anchor


def _floor_to_bar(instant: datetime, *, bar_minutes: int = BAR_MINUTES) -> datetime:
    """The last bar CLOSE at or before ``instant``."""
    anchor = instant.replace(minute=0, second=0, microsecond=0)
    while anchor > instant:
        anchor -= timedelta(minutes=bar_minutes)
    return anchor


def cash_sessions(
    first: date, last: date, *, calendar: str, open_delay_minutes: int
) -> list[dict[str, Any]]:
    """One row per NYSE cash session between ``first`` and ``last``, from the calendar.

    Delegates to ``case_studies/exness_usidx_sess/_features.session_grid``, which is the single
    schedule implementation of this bot. Importing the case study from the deployment package
    is deliberate (Ch25 s25.1): a schedule written twice agrees on the day it is written and
    drifts on the first edit.
    """
    from case_studies.exness_usidx_sess._features import session_grid

    grid = session_grid(
        calendar,
        datetime.combine(first, time(0)),
        datetime.combine(last, time(23, 59)),
        open_delay_minutes=open_delay_minutes,
    )
    rows = grid.filter(
        (grid["session"] >= first) & (grid["session"] <= last)  # noqa: PD011
    ).to_dicts()
    return rows


def _venue_check(session_row: dict[str, Any]) -> dict[str, Any]:
    """Cross-check the calendar's open/close against ``SESSIONS['us_cash']`` + real DST.

    A full session must agree to the minute. A half day must NOT: the calendar knows the venue
    closes at 13:00 New York and the fixed 09:30-16:00 window does not, so the difference is
    reported rather than asserted away. That difference is the reason the calendar is the
    source and the session window is only the check.
    """
    window = SESSIONS["us_cash"]
    zone = ZoneInfo(window.zone)
    open_utc = session_row["open_at"].replace(tzinfo=UTC)
    close_utc = session_row["close_at"].replace(tzinfo=UTC)
    open_local = open_utc.astimezone(zone)
    close_local = close_utc.astimezone(zone)
    expected_open = datetime.combine(open_local.date(), window.start, tzinfo=zone)
    expected_close = datetime.combine(close_local.date(), window.end, tzinfo=zone)
    open_delta = (open_local - expected_open).total_seconds() / 60.0
    close_delta = (close_local - expected_close).total_seconds() / 60.0
    half_day = abs(close_delta) > 1e-9
    if abs(open_delta) > 1e-9:
        raise ValueError(
            f"{session_row['session']}: the NYSE calendar opens the cash session at "
            f"{open_local.time()} {window.zone} but bots/_shared/sessions.SESSIONS['us_cash'] "
            f"declares {window.start}. The two schedule sources disagree on a full session, "
            "which means one of them is wrong; refusing to guess which."
        )
    return {
        "open_local": open_local,
        "close_local": close_local,
        "half_day": half_day,
        "close_delta_minutes": close_delta,
    }


def decision_instants(
    session: date,
    *,
    decision_block: dict[str, Any],
    clock: ServerClock | None = None,
    specs: tuple[str, ...] | None = None,
) -> dict[str, DecisionInstant]:
    """The decision instants of ``session``, one per spec, in UTC / server / venue-local time.

    The two rules, quoted from ``setup.yaml::decision``:

    * ``us_cash_open``  = the close of the FIRST H1 bar closing at or after
      ``NYSE open + open_delay_minutes`` (30). The first half hour of the cash session is
      avoided on purpose (``bots/assets/US500.md`` s5, the opening-range window).
    * ``us_cash_close`` = the close of the LAST H1 bar closing at or before the NYSE close.

    ``execution_delay: next_bar_open`` puts the fill at the OPEN of the bar after the decision
    bar, i.e. at the decision instant itself on a contiguous H1 grid. Whether a bar is actually
    there is a separate question the deployment loop asks of the tape — "no bar, no order".
    """
    specs = specs or SPECS
    clock = clock or server_clock_from_config(decision_block)
    delay = int(decision_block["open_delay_minutes"])
    calendar = str(decision_block["session_calendar"])
    rows = cash_sessions(session, session, calendar=calendar, open_delay_minutes=delay)
    if not rows:
        raise LookupError(
            f"{session} is not an NYSE cash session ({calendar}); this bot has no decision that "
            "day even though Exness quotes the CFD"
        )
    row = rows[0]
    check = _venue_check(row)
    open_utc = row["open_at"].replace(tzinfo=UTC)
    close_utc = row["close_at"].replace(tzinfo=UTC)
    open_target = row["open_target"].replace(tzinfo=UTC)

    out: dict[str, DecisionInstant] = {}
    for spec in specs:
        if spec == "intraday":
            decision_utc = _ceil_to_bar(open_target)
        elif spec == "overnight":
            decision_utc = _floor_to_bar(close_utc)
        else:
            raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
        exec_open_utc = decision_utc  # next_bar_open on a contiguous H1 grid
        out[spec] = DecisionInstant(
            session=row["session"],
            spec=spec,
            decision_utc=decision_utc,
            decision_server=clock.to_server(decision_utc),
            decision_local=decision_utc.astimezone(ZoneInfo(SESSIONS["us_cash"].zone)),
            exec_open_utc=exec_open_utc,
            exec_open_server=clock.to_server(exec_open_utc),
            cash_open_utc=open_utc,
            cash_close_utc=close_utc,
            half_day=bool(check["half_day"]),
        )
    return out


def next_decision(
    now_utc: datetime,
    *,
    decision_block: dict[str, Any],
    clock: ServerClock | None = None,
    horizon_days: int = 10,
) -> DecisionInstant | None:
    """The next decision instant at or after ``now_utc``, across both specs.

    Returns ``None`` when no NYSE cash session falls inside ``horizon_days`` — a state a
    scheduler must be able to see, because a long holiday is not an error.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    clock = clock or server_clock_from_config(decision_block)
    delay = int(decision_block["open_delay_minutes"])
    calendar = str(decision_block["session_calendar"])
    rows = cash_sessions(
        now_utc.date(),
        now_utc.date() + timedelta(days=horizon_days),
        calendar=calendar,
        open_delay_minutes=delay,
    )
    upcoming: list[DecisionInstant] = []
    for row in rows:
        for inst in decision_instants(
            row["session"], decision_block=decision_block, clock=clock
        ).values():
            if inst.decision_utc >= now_utc:
                upcoming.append(inst)
    if not upcoming:
        return None
    return min(upcoming, key=lambda i: i.decision_utc)


def schedule_table(
    first: date,
    last: date,
    *,
    decision_block: dict[str, Any],
    clock: ServerClock | None = None,
) -> list[dict[str, Any]]:
    """Every decision instant between two dates, for the run record and for the tests."""
    clock = clock or server_clock_from_config(decision_block)
    delay = int(decision_block["open_delay_minutes"])
    calendar = str(decision_block["session_calendar"])
    rows = cash_sessions(first, last, calendar=calendar, open_delay_minutes=delay)
    out: list[dict[str, Any]] = []
    for row in rows:
        for inst in decision_instants(
            row["session"], decision_block=decision_block, clock=clock
        ).values():
            out.append(inst.as_dict())
    return sorted(out, key=lambda r: (r["session"], r["decision_utc"]))


__all__ = [
    "BAR_MINUTES",
    "SPECS",
    "DecisionInstant",
    "cash_sessions",
    "decision_instants",
    "next_decision",
    "schedule_table",
    "server_clock_from_config",
]
