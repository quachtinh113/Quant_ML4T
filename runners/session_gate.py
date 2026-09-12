"""Decide whether NOW is a decision hour for a session-grid bot, and which leg it is.

Windows Task Scheduler fires at fixed Vietnam-clock times (UTC+7, no daylight saving).
The gold and US-index bots decide on VENUE sessions (Europe/London, America/New_York),
whose UTC hours move twice a year. So each leg is registered at BOTH its summer and its
winter UTC hour, and this gate runs first: it prints the leg whose decision hour is now
and exits 0, or exits 10 when the current hour is not a decision hour (the runner then
skips the cycle without touching the deployment loop, its state or its run records).

The venue windows come from ``bots/_shared/sessions.SESSIONS`` (ZoneInfo, DST-aware), the
same table the research features use. Decision rules mirrored here, on the UTC-aligned H1
grid the bots trade:

* gold ``london`` / ``ny``: close of the first H1 bar at or after venue open + 30 min,
  i.e. open + 60 min (``case_studies/exness_gold_sess/config/setup.yaml::decision``).
* usidx ``intraday``: close of the first H1 bar at or after NYSE cash open + 30 min
  (``bots/exness_usidx_sess/deploy/schedule.decision_instants``).
* usidx ``overnight``: close of the last H1 bar at or before the NYSE cash close.

Exchange holidays and half days are NOT known here; on those days the gate says "go" and
the deployment loop, which owns the calendar, reports no session. That is the intended
split: the gate only stops the twice-a-year off-hour firing from producing a NotReady
record every day.

Usage (from the runner .bat files):

    py -3.12 runners\\session_gate.py gold            -> prints "london" or "ny"
    py -3.12 runners\\session_gate.py usidx           -> prints "intraday" or "overnight"
    py -3.12 runners\\session_gate.py gold --at 2026-12-01T09:00Z   (dry check)
    py -3.12 runners\\session_gate.py gold --table    (this week's decision hours)

Exit codes: 0 = a leg is due now (its name is on stdout); 10 = not a decision hour;
2 = bad arguments.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bots._shared.sessions import SESSIONS, SessionWindow  # noqa: E402

EXIT_GO = 0
EXIT_SKIP = 10

# bot -> leg -> (venue session, anchor, offset minutes). "open" anchors on the session open,
# "close" on the session close; the H1 rounding is applied after the offset.
LEGS: dict[str, dict[str, tuple[str, str, int]]] = {
    "gold": {
        "london": ("london", "open", 30),
        "ny": ("new_york", "open", 30),
    },
    "usidx": {
        "intraday": ("us_cash", "open", 30),
        "overnight": ("us_cash", "close", 0),
    },
}


def _ceil_hour(ts: datetime) -> datetime:
    floored = ts.replace(minute=0, second=0, microsecond=0)
    return floored if floored == ts else floored + timedelta(hours=1)


def _floor_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def decision_utc(window: SessionWindow, day_utc: datetime, anchor: str, offset_min: int) -> datetime | None:
    """The decision instant of ``window`` on the venue-local weekday containing ``day_utc``.

    Returns None on a venue weekend.
    """
    zone = ZoneInfo(window.zone)
    local_day = day_utc.astimezone(zone).date()
    if local_day.weekday() >= 5:
        return None
    if anchor == "open":
        local = datetime.combine(local_day, window.start, tzinfo=zone)
        return _ceil_hour(local.astimezone(UTC) + timedelta(minutes=offset_min))
    local = datetime.combine(local_day, window.end, tzinfo=zone)
    return _floor_hour(local.astimezone(UTC) + timedelta(minutes=offset_min))


def due_leg(bot: str, now_utc: datetime) -> str | None:
    """The leg whose decision hour contains ``now_utc``, or None."""
    hour = _floor_hour(now_utc)
    for leg, (session, anchor, offset) in LEGS[bot].items():
        window = SESSIONS[session]
        # The venue-local date of "now" and of the previous UTC day both matter near
        # midnight (New York close is 20:00/21:00 UTC, never near midnight, but keep it
        # general).
        for probe in (hour, hour - timedelta(hours=12)):
            instant = decision_utc(window, probe, anchor, offset)
            if instant is not None and instant == hour:
                return leg
    return None


def week_table(bot: str, now_utc: datetime) -> str:
    lines = [f"{bot}: decision hours (UTC / Vietnam UTC+7) for the week of {now_utc:%Y-%m-%d}"]
    monday = _floor_hour(now_utc).replace(hour=12) - timedelta(days=now_utc.weekday())
    vn = ZoneInfo("Asia/Ho_Chi_Minh")
    for d in range(5):
        day = monday + timedelta(days=d)
        parts = []
        for leg, (session, anchor, offset) in LEGS[bot].items():
            inst = decision_utc(SESSIONS[session], day, anchor, offset)
            if inst is not None:
                parts.append(f"{leg}={inst:%H:%M}Z/{inst.astimezone(vn):%a %H:%M}VN")
        lines.append(f"  {day:%a %Y-%m-%d}: " + "  ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bot", choices=sorted(LEGS))
    parser.add_argument("--at", default=None, help="ISO UTC instant to test instead of now")
    parser.add_argument("--table", action="store_true", help="print this week's decision hours")
    args = parser.parse_args(argv)

    if args.at:
        now = datetime.fromisoformat(args.at.replace("Z", "+00:00"))
        now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    else:
        now = datetime.now(UTC)

    if args.table:
        print(week_table(args.bot, now))
        return EXIT_GO

    leg = due_leg(args.bot, now)
    if leg is None:
        print(f"skip: {now:%Y-%m-%d %H:%M}Z is not a {args.bot} decision hour", file=sys.stderr)
        return EXIT_SKIP
    print(leg)
    return EXIT_GO


if __name__ == "__main__":
    sys.exit(main())
