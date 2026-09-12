"""The decision schedule is DERIVED, and these tests are what stops it becoming a constant.

The bot's two decision instants sit at 14:00/15:00 and 20:00/21:00 UTC *today*. Those are
outputs of a derivation, not inputs, and a test suite that asserted them as constants would
lock in a schedule that is wrong for roughly five months of every year and on every half day.
So the tests below assert the RULE — the venue's local time — and check the UTC hour only as a
consequence, in both DST regimes and on a measured half day.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from bots._shared.sessions import SESSIONS, ServerClock
from bots.exness_usidx_sess.deploy import schedule as sched

RISK_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "risk_config.yaml"
NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def decision_block() -> dict:
    return yaml.safe_load(RISK_CONFIG.read_text())["decision"]


# ---------------------------------------------------------------------------------------
# The rule, in venue-local time
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "session",
    [
        date(2025, 7, 15),  # US summer (EDT, UTC-4)
        date(2025, 1, 15),  # US winter (EST, UTC-5)
    ],
)
def test_intraday_decision_is_one_hour_after_the_cash_open_in_new_york(decision_block, session):
    """``us_cash_open`` = the first H1 bar CLOSING at or after 09:30 + 30 min New York.

    On an H1 grid anchored to the UTC hour, 10:00 New York IS an hour boundary in both DST
    regimes, so the decision bar closes at exactly 10:00 New York — an hour after the auction
    and half an hour after the opening-range window `open_delay_minutes` declares.
    """
    instant = sched.decision_instants(session, decision_block=decision_block)["intraday"]
    local = instant.decision_utc.astimezone(NY)
    assert local.time() == time(10, 0), local
    assert local.date() == session


@pytest.mark.parametrize("session", [date(2025, 7, 15), date(2025, 1, 15)])
def test_overnight_decision_is_the_cash_close_in_new_york(decision_block, session):
    """``us_cash_close`` = the last H1 bar closing at or before 16:00 New York."""
    instant = sched.decision_instants(session, decision_block=decision_block)["overnight"]
    local = instant.decision_utc.astimezone(NY)
    assert local.time() == time(16, 0), local


def test_the_utc_hour_moves_with_new_york_dst_and_that_is_the_whole_point(decision_block):
    """The venue-local rule is constant; the UTC hour is NOT. A constant here would be a bug.

    Winter 15:00 / 21:00 UTC, summer 14:00 / 20:00 UTC — exactly the figures ``BOT.md`` records
    as a DESCRIPTION of the derivation. They are asserted here as a consequence of the rule,
    which is why this test would fail loudly if the venue stopped following New York DST (kill
    criterion (g)).
    """
    winter = sched.decision_instants(date(2025, 1, 15), decision_block=decision_block)
    summer = sched.decision_instants(date(2025, 7, 15), decision_block=decision_block)
    assert winter["intraday"].decision_utc.hour == 15
    assert winter["overnight"].decision_utc.hour == 21
    assert summer["intraday"].decision_utc.hour == 14
    assert summer["overnight"].decision_utc.hour == 20


def test_half_day_closes_three_hours_early_and_is_flagged(decision_block):
    """The 13:00 New York half-days come from the CALENDAR, not from a DST rule.

    A fixed 09:30-16:00 window cannot know about them; that is precisely why
    ``session_grid`` (the NYSE calendar) is the source and ``SESSIONS['us_cash']`` is only the
    cross-check. 2025-11-28, the day after Thanksgiving, is a measured half day.
    """
    instants = sched.decision_instants(date(2025, 11, 28), decision_block=decision_block)
    assert instants["overnight"].half_day is True
    assert instants["overnight"].decision_utc.astimezone(NY).time() == time(13, 0)
    # And the intraday instant is unaffected: the venue opens as usual.
    assert instants["intraday"].decision_utc.astimezone(NY).time() == time(10, 0)


def test_a_market_holiday_has_no_decision_at_all(decision_block):
    """A day the NYSE is shut has no cash session, so the bot has no row and no order — even
    though Exness quotes the CFD."""
    with pytest.raises(LookupError, match="not an NYSE cash session"):
        sched.decision_instants(date(2025, 7, 4), decision_block=decision_block)


# ---------------------------------------------------------------------------------------
# The server clock is a SEPARATE fact from the trade hours
# ---------------------------------------------------------------------------------------
def test_server_time_equals_utc_because_the_clock_does_not_follow_dst(decision_block):
    """``server_clock.follows_dst_of`` is null (measured UTC+0 all year) while
    ``trade_hours_follow_dst_of`` is America/New_York. The two facts disagree by design, and
    the conversion has to go through ``ServerClock`` rather than being assumed either way."""
    clock = sched.server_clock_from_config(decision_block)
    assert clock.utc_offset_minutes == 0
    assert clock.follows_dst_of is None
    for session in (date(2025, 1, 15), date(2025, 7, 15)):
        for instant in sched.decision_instants(session, decision_block=decision_block).values():
            assert instant.decision_server == instant.decision_utc.replace(tzinfo=None)


def test_a_server_that_did_follow_dst_would_move_the_server_time_but_not_the_utc_instant(
    decision_block,
):
    """If Exness ever moved its clock onto New York DST, the UTC decision instant must not
    move. This is the failure the two separate declarations exist to make impossible."""
    shifted = ServerClock(
        utc_offset_minutes=120,
        measured_at=datetime(2025, 1, 15, tzinfo=UTC),
        follows_dst_of="America/New_York",
    )
    winter = sched.decision_instants(
        date(2025, 1, 15), decision_block=decision_block, clock=shifted
    )["intraday"]
    summer = sched.decision_instants(
        date(2025, 7, 15), decision_block=decision_block, clock=shifted
    )["intraday"]
    assert winter.decision_utc.hour == 15  # unchanged
    assert summer.decision_utc.hour == 14  # unchanged
    # but the server clock reading differs by the DST hour
    assert winter.decision_server.hour == 17
    assert summer.decision_server.hour == 17  # +180 in July on a +120 January measurement


# ---------------------------------------------------------------------------------------
# Cross-checks and helpers
# ---------------------------------------------------------------------------------------
def test_the_calendar_and_the_session_window_agree_on_every_full_session(decision_block):
    """``_venue_check`` raises when the two schedule sources disagree on a full session.

    Running it over a year is the check itself: if ``pandas_market_calendars`` and
    ``bots/_shared/sessions.SESSIONS['us_cash']`` ever disagreed on an open, the derivation
    would be resting on a source nobody had compared.
    """
    rows = sched.cash_sessions(
        date(2025, 1, 1), date(2025, 12, 31), calendar="NYSE", open_delay_minutes=30
    )
    assert len(rows) > 240
    half_days = 0
    for row in rows:
        check = sched._venue_check(row)  # raises on a full-session disagreement
        half_days += int(check["half_day"])
    assert 2 <= half_days <= 5, f"{half_days} half days in 2025 is not a plausible count"
    assert SESSIONS["us_cash"].start == time(9, 30)


def test_next_decision_returns_the_earliest_upcoming_instant(decision_block):
    now = datetime(2025, 7, 15, 12, 0, tzinfo=UTC)
    nxt = sched.next_decision(now, decision_block=decision_block)
    assert nxt is not None
    assert nxt.decision_utc == datetime(2025, 7, 15, 14, 0, tzinfo=UTC)
    assert nxt.spec == "intraday"
    later = sched.next_decision(
        datetime(2025, 7, 15, 15, 0, tzinfo=UTC), decision_block=decision_block
    )
    assert later.spec == "overnight"
    assert later.decision_utc == datetime(2025, 7, 15, 20, 0, tzinfo=UTC)


def test_next_decision_is_none_when_no_session_falls_inside_the_horizon(decision_block):
    """A long holiday is not an error, and a scheduler has to be able to see that state."""
    assert (
        sched.next_decision(
            datetime(2025, 12, 25, 0, 0, tzinfo=UTC),
            decision_block=decision_block,
            horizon_days=0,
        )
        is None
    )


def test_schedule_table_carries_two_instants_per_session(decision_block):
    table = sched.schedule_table(
        date(2025, 7, 14), date(2025, 7, 18), decision_block=decision_block
    )
    assert len(table) == 10  # five sessions x two specs
    assert {row["spec"] for row in table} == {"intraday", "overnight"}
    for row in table:
        assert row["exec_open_utc"] == row["decision_utc"]  # next_bar_open on a contiguous grid


# ---------------------------------------------------------------------------------------
# The rule that keeps the hours out of the code
# ---------------------------------------------------------------------------------------
def test_no_utc_decision_hour_is_hard_coded_in_the_deployment_package():
    """No source file may carry ``14:00``/``15:00``/``20:00``/``21:00`` as a UTC constant.

    They are outputs of the calendar and of New York DST. The one place a wall-clock time is
    written down is the CFD break in ``risk_config.yaml``, and it is declared in New York local
    time with its zone, which is what this test allows.
    """
    package = Path(__file__).resolve().parents[1]
    forbidden = ('"14:00"', "'14:00'", '"15:00"', "'15:00'", '"20:00"', "'20:00'", '"21:00"',
                 "'21:00'")
    offenders: list[str] = []
    for path in sorted((package / "deploy").glob("*.py")) + sorted(
        (package / "monitor").glob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        # Strip comment and docstring prose: the hours are DESCRIBED all over the package on
        # purpose. What must not exist is one used as a value.
        code = "\n".join(
            line.split("#")[0] for line in text.splitlines() if not line.strip().startswith("#")
        )
        for token in forbidden:
            if token in code:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, (
        "a UTC decision hour is written as a value in the deployment package; it must be "
        f"derived from the NYSE calendar and New York DST instead: {offenders}"
    )


def test_the_cfd_break_is_declared_in_new_york_local_time_not_utc():
    """Kill criterion (g)'s subject: the daily break ends at 18:00 New York, which is
    21:00-22:00 UTC in summer and 22:00-23:00 UTC in winter. A UTC constant would be wrong for
    five months of every year."""
    raw = yaml.safe_load(RISK_CONFIG.read_text())
    guards = raw["cfd_guards"]
    assert "blocked_execution_windows_utc" not in guards
    windows = guards["blocked_execution_windows_local"]
    assert windows and all(w["zone"] == "America/New_York" for w in windows)
    assert windows[0]["end"] == "18:00"
    assert raw["breakers"]["trade_hours_dst"]["expected_break_end_local"] == "18:00"


def test_execution_instant_is_the_decision_instant_and_the_tolerance_is_one_bar(decision_block):
    """``execution_delay: next_bar_open`` with a 60-minute tolerance — "no bar, no order"."""
    assert decision_block["execution_delay"] == "next_bar_open"
    assert int(decision_block["execution_tolerance_minutes"]) == 60
    assert int(decision_block["session_close_tolerance_minutes"]) == 60
    instant = sched.decision_instants(date(2025, 7, 15), decision_block=decision_block)["intraday"]
    assert instant.exec_open_utc - instant.decision_utc == timedelta(0)
