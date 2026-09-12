"""bots/_shared/sessions.py against the asset profiles and the 2026 DST calendar.

Runs without MetaTrader5: ``uv run pytest bots/exness_fx_d1/tests -q``.

The two Sundays that matter in 2026: US clocks go forward on 2026-03-08 and back on
2026-11-01; UK clocks go forward on 2026-03-29 and back on 2026-10-25. Between those dates
London and New York are one hour closer together than the winter and summer tables in
``bots/assets`` say, which is what a table cannot express and ``zoneinfo`` can.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import polars as pl
import pytest

from bots._shared import sessions as S
from bots._shared.sessions import ServerClock, session_flags, session_flags_frame

FX_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
WINTER_DAY = date(2026, 1, 15)  # Thursday, standard time everywhere in the north
SUMMER_DAY = date(2026, 7, 16)  # Thursday, daylight time in London and New York
US_DST_START = date(2026, 3, 8)
UK_DST_START = date(2026, 3, 29)
UK_DST_END = date(2026, 10, 25)
US_DST_END = date(2026, 11, 1)


def at(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=UTC)


def window_flags(day: date, name: str, start: time, end: time) -> None:
    """The flag is on from ``start`` (inclusive) to ``end`` (exclusive), UTC, on ``day``."""
    day_end = day if end > start else day + timedelta(days=1)
    first = datetime.combine(day, start, tzinfo=UTC)
    last = datetime.combine(day_end, end, tzinfo=UTC)
    assert session_flags(first)[name], f"{name} not on at its declared open {first}"
    assert session_flags(last - timedelta(minutes=1))[name], f"{name} off before close {last}"
    assert not session_flags(first - timedelta(minutes=1))[name], f"{name} on before open"
    assert not session_flags(last)[name], f"{name} still on at close {last}"


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("symbol", FX_SYMBOLS)
def test_profile_frontmatter_loads(symbol: str) -> None:
    profile = S.load_asset_profile(symbol)
    assert profile["symbol"] == symbol
    assert profile["bot"] == "exness_fx_d1"
    assert profile["template_case_study"] == "fx_pairs"
    assert profile["timezone"] == "UTC"
    assert set(profile["sessions"]) <= {"sydney", "tokyo", "london", "new_york", "overlap"}
    assert profile["decision_times"] and profile["avoid_windows"] and profile["news"]


def test_load_all_profiles_covers_the_bot() -> None:
    profiles = S.load_asset_profiles()
    mine = sorted(s for s, p in profiles.items() if p["bot"] == "exness_fx_d1")
    assert mine == sorted(FX_SYMBOLS)


@pytest.mark.parametrize("symbol", FX_SYMBOLS)
@pytest.mark.parametrize("season,day", [("winter", WINTER_DAY), ("summer", SUMMER_DAY)])
def test_zoneinfo_reproduces_profile_tables(symbol: str, season: str, day: date) -> None:
    """Every window the profile tabulates for a stable season comes out of the time zones."""
    profile = S.load_asset_profile(symbol)
    for name in profile["sessions"]:
        start, end = S.profile_window_utc(profile, name, season)
        window_flags(day, name, start, end)


# ---------------------------------------------------------------------------
# DST edges
# ---------------------------------------------------------------------------
def test_us_dst_moves_new_york_but_not_london() -> None:
    before, after = US_DST_START - timedelta(days=2), US_DST_START + timedelta(days=1)
    window_flags(before, "new_york", time(13), time(22))  # Friday 2026-03-06, winter
    window_flags(after, "new_york", time(12), time(21))  # Monday 2026-03-09, US summer
    window_flags(before, "london", time(8), time(17))
    window_flags(after, "london", time(8), time(17))  # UK still on GMT


def test_overlap_is_five_hours_between_the_two_dst_changes() -> None:
    gap_day = date(2026, 3, 16)  # Monday between 03-08 and 03-29
    window_flags(gap_day, "overlap", time(12), time(17))
    window_flags(WINTER_DAY, "overlap", time(13), time(17))
    window_flags(SUMMER_DAY, "overlap", time(12), time(16))


def test_uk_dst_moves_london() -> None:
    before, after = UK_DST_START - timedelta(days=2), UK_DST_START + timedelta(days=1)
    window_flags(before, "london", time(8), time(17))  # Friday 2026-03-27
    window_flags(after, "london", time(7), time(16))  # Monday 2026-03-30
    window_flags(after, "overlap", time(12), time(16))


def test_autumn_gap_between_uk_and_us_clock_changes() -> None:
    gap_day = date(2026, 10, 28)  # Wednesday: UK back on GMT, US still on EDT
    window_flags(gap_day, "london", time(8), time(17))
    window_flags(gap_day, "new_york", time(12), time(21))
    window_flags(gap_day, "overlap", time(12), time(17))
    after = US_DST_END + timedelta(days=1)
    window_flags(after, "new_york", time(13), time(22))


def test_sydney_dst_runs_the_other_way() -> None:
    window_flags(WINTER_DAY, "sydney", time(21), time(6))  # AEDT in January
    window_flags(SUMMER_DAY, "sydney", time(22), time(7))  # AEST in July
    window_flags(WINTER_DAY, "tokyo", time(0), time(9))
    window_flags(SUMMER_DAY, "tokyo", time(0), time(9))


def test_us_cash_session() -> None:
    window_flags(WINTER_DAY, "us_cash", time(14, 30), time(21))
    window_flags(SUMMER_DAY, "us_cash", time(13, 30), time(20))


# ---------------------------------------------------------------------------
# Derived flags
# ---------------------------------------------------------------------------
def test_asia_flag_is_sydney_or_tokyo_outside_london() -> None:
    assert session_flags(at(WINTER_DAY, 2))["asia"]
    assert session_flags(at(WINTER_DAY, 7, 59))["asia"]
    assert not session_flags(at(WINTER_DAY, 8))["asia"]  # London open ends Asia
    assert not session_flags(at(WINTER_DAY, 15))["asia"]


def test_edge_flags_are_the_first_and_last_half_hour() -> None:
    assert session_flags(at(WINTER_DAY, 8, 10))["edge_open"]  # London opened 08:00
    assert not session_flags(at(WINTER_DAY, 8, 30))["edge_open"]
    assert session_flags(at(WINTER_DAY, 16, 45))["edge_close"]  # London closes 17:00
    assert not session_flags(at(WINTER_DAY, 16, 20))["edge_close"]
    assert session_flags(at(WINTER_DAY, 13, 5))["edge_open"]  # New York opened 13:00
    assert session_flags(at(WINTER_DAY, 21, 40))["edge_close"]  # New York closes 22:00


def test_funding_flag_marks_the_8h_boundaries() -> None:
    for hh in (0, 8, 16):
        assert session_flags(at(WINTER_DAY, hh))["funding"]
    assert not session_flags(at(WINTER_DAY, 8, 15))["funding"]
    assert not session_flags(at(WINTER_DAY, 9))["funding"]


def test_market_open_follows_the_fx_week() -> None:
    friday, saturday, sunday = date(2026, 1, 16), date(2026, 1, 17), date(2026, 1, 18)
    assert session_flags(at(friday, 21, 59))["market_open"]
    assert not session_flags(at(friday, 22, 0))["market_open"]  # 17:00 New York, winter
    assert not session_flags(at(saturday, 12))["market_open"]
    assert not session_flags(at(sunday, 21, 59))["market_open"]
    assert session_flags(at(sunday, 22, 0))["market_open"]
    assert not any(session_flags(at(saturday, 12))[n] for n in S.SESSIONS)


def test_rollover_without_a_clock_is_the_new_york_convention() -> None:
    assert session_flags(at(WINTER_DAY, 21, 50))["rollover"]
    assert session_flags(at(WINTER_DAY, 22, 10))["rollover"]
    assert not session_flags(at(WINTER_DAY, 21, 40))["rollover"]
    assert session_flags(at(SUMMER_DAY, 21, 5))["rollover"]  # 17:00 EDT = 21:00 UTC


# ---------------------------------------------------------------------------
# Server clock
# ---------------------------------------------------------------------------
def test_fixed_offset_clock_round_trips() -> None:
    clock = ServerClock(utc_offset_minutes=120)
    server = datetime(2026, 3, 9, 0, 0)
    utc = clock.to_utc(server)
    assert utc == datetime(2026, 3, 8, 22, 0, tzinfo=UTC)
    assert clock.to_server(utc) == server
    assert clock.server_day_start_utc(date(2026, 7, 1)) == datetime(2026, 6, 30, 22, tzinfo=UTC)


def test_zero_offset_clock_is_identity() -> None:
    clock = ServerClock(0)
    assert clock.to_utc(datetime(2026, 1, 1, 5)) == datetime(2026, 1, 1, 5, tzinfo=UTC)


def test_dst_following_clock_shifts_with_its_zone() -> None:
    measured = datetime(2026, 1, 15, 12, tzinfo=UTC)
    clock = ServerClock(120, measured_at=measured, follows_dst_of="America/New_York")
    assert clock.offset_at(datetime(2026, 3, 7, 12, tzinfo=UTC)) == timedelta(minutes=120)
    assert clock.offset_at(datetime(2026, 3, 9, 12, tzinfo=UTC)) == timedelta(minutes=180)
    assert clock.to_utc(datetime(2026, 7, 1, 0, 0)) == datetime(2026, 6, 30, 21, tzinfo=UTC)
    assert clock.to_utc(datetime(2026, 1, 20, 0, 0)) == datetime(2026, 1, 19, 22, tzinfo=UTC)
    # measured in summer, the same server reads +180 and is +120 in winter
    summer = ServerClock(180, measured_at=datetime(2026, 7, 1, tzinfo=UTC), follows_dst_of="America/New_York")
    assert summer.offset_at(datetime(2026, 12, 1, tzinfo=UTC)) == timedelta(minutes=120)


def test_clock_serialises() -> None:
    clock = ServerClock(120, measured_at=datetime(2026, 1, 15, 12, tzinfo=UTC), follows_dst_of="Europe/London")
    assert ServerClock.from_dict(clock.as_dict()) == clock
    with pytest.raises(ValueError):
        ServerClock(120, follows_dst_of="America/New_York")  # needs measured_at
    with pytest.raises(ValueError):
        ServerClock(120).to_utc(datetime(2026, 1, 1, tzinfo=UTC))  # server time is naive


def test_clock_is_measured_from_a_tick_not_assumed() -> None:
    from bots._shared.testing.fake_mt5 import make_fake

    now = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    fake = make_fake(["EURUSD"], server_offset_minutes=180, now_utc=now)
    fake.initialize()
    clock = ServerClock.measure(fake, "EURUSD", now=now)
    assert clock.utc_offset_minutes == 180
    assert clock.measured_at == now
    other = make_fake(["EURUSD"], server_offset_minutes=0, now_utc=now)
    other.initialize()
    assert ServerClock.measure(other, "EURUSD", now=now).utc_offset_minutes == 0


def test_rollover_with_a_clock_is_the_server_day_boundary() -> None:
    clock = ServerClock(120, measured_at=datetime(2026, 1, 15, 12, tzinfo=UTC), follows_dst_of="America/New_York")
    assert session_flags(at(WINTER_DAY, 21, 50), clock)["rollover"]  # server midnight 22:00 UTC
    assert session_flags(at(WINTER_DAY, 22, 14), clock)["rollover"]
    assert not session_flags(at(WINTER_DAY, 22, 16), clock)["rollover"]
    assert session_flags(at(SUMMER_DAY, 20, 50), clock)["rollover"]  # 21:00 UTC in summer
    assert not session_flags(at(SUMMER_DAY, 21, 50), clock)["rollover"]


# ---------------------------------------------------------------------------
# Vectorised
# ---------------------------------------------------------------------------
def test_frame_flags_match_scalar_flags() -> None:
    stamps = [at(WINTER_DAY, h) for h in range(24)] + [at(SUMMER_DAY, h) for h in range(24)]
    frame = session_flags_frame(pl.Series("timestamp", stamps))
    assert frame.columns == ["timestamp", *S.FLAG_NAMES]
    assert frame.height == len(stamps)
    for row in frame.iter_rows(named=True):
        ts = row.pop("timestamp")
        assert row == session_flags(ts), ts
    naive = session_flags_frame([s.replace(tzinfo=None) for s in stamps[:3]])
    assert naive["london"].to_list() == [False, False, False]
