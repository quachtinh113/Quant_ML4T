"""Session calendar shared by the Exness bots.

Everything here is computed from real time zones through :mod:`zoneinfo`, never from a
table of UTC hours: London and New York change their clocks on different Sundays, Sydney
changes in the opposite direction, Tokyo does not change at all, and any table that
hard-codes "winter" and "summer" is wrong for the weeks in between. The tables in
``bots/assets/<SYMBOL>.md`` are the *expected* values for the two stable seasons and the
tests check this module against them; the module never reads them to compute a flag.

Three things live here:

1. :func:`load_asset_profile` / :func:`load_asset_profiles` read the YAML frontmatter of
   ``bots/assets/<SYMBOL>.md`` (sessions, decision times, avoid windows, news calendar).
2. :func:`session_flags` returns the flags for one UTC timestamp and
   :func:`session_flags_frame` does the same for a column of timestamps.
3. :class:`ServerClock` converts between the MT5 server clock and UTC. It is
   parameterised by a *measured* offset (``ServerClock.measure`` reads it from a tick) and
   optionally by the time zone whose daylight-saving rule the server follows. No server
   clock is hard-coded anywhere.

Session definitions (local time of the venue, Monday to Friday):

=========  ===================  =============  ======================================
flag       zone                 local window   reference in bots/assets (UTC)
=========  ===================  =============  ======================================
sydney     Australia/Sydney     08:00-17:00    21:00-06:00 winter, 22:00-07:00 summer
tokyo      Asia/Tokyo           09:00-18:00    00:00-09:00 all year
london     Europe/London        08:00-17:00    08:00-17:00 winter, 07:00-16:00 summer
new_york   America/New_York     08:00-17:00    13:00-22:00 winter, 12:00-21:00 summer
us_cash    America/New_York     09:30-16:00    14:30-21:00 winter, 13:30-20:00 summer
=========  ===================  =============  ======================================

Derived flags: ``overlap`` = london and new_york; ``asia`` = (sydney or tokyo) and not
london; ``edge_open`` / ``edge_close`` = inside the first / last ``EDGE_MINUTES`` of any
session above; ``funding`` = the 00:00 / 08:00 / 16:00 UTC funding instants of the
perpetual-swap venues the BTC bot mirrors; ``rollover`` = within ``ROLLOVER_MINUTES`` of the
server day boundary (swap time) when a :class:`ServerClock` is given, otherwise of the
17:00 New York interbank rollover; ``market_open`` = the FX week, Sunday 17:00 to Friday
17:00 New York time.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_PROFILE_NAME = re.compile(r"[A-Z][A-Z0-9]{2,11}")  # EURUSD, XAUUSD, US500, BTCUSD

ZONES: dict[str, str] = {
    "sydney": "Australia/Sydney",
    "tokyo": "Asia/Tokyo",
    "london": "Europe/London",
    "new_york": "America/New_York",
}

EDGE_MINUTES = 30
ROLLOVER_MINUTES = 15
FUNDING_HOURS_UTC = (0, 8, 16)
FX_ROLLOVER_LOCAL = time(17, 0)  # interbank value-date rollover, New York time
WEEKDAYS = (0, 1, 2, 3, 4)  # Monday .. Friday in the venue's local calendar

FLAG_NAMES = (
    "sydney",
    "tokyo",
    "london",
    "new_york",
    "us_cash",
    "asia",
    "overlap",
    "edge_open",
    "edge_close",
    "funding",
    "rollover",
    "market_open",
)


# ---------------------------------------------------------------------------
# Asset profiles (YAML frontmatter of bots/assets/<SYMBOL>.md)
# ---------------------------------------------------------------------------
def load_asset_profile(symbol: str, assets_dir: Path | None = None) -> dict[str, Any]:
    """Return the YAML frontmatter of ``bots/assets/<SYMBOL>.md`` as a dict.

    The frontmatter is the block between the first two ``---`` lines. Raises
    ``FileNotFoundError`` when the profile does not exist and ``ValueError`` when the file
    carries no frontmatter, so a bot cannot silently run on an instrument nobody profiled.
    """
    path = (assets_dir or ASSETS_DIR) / f"{symbol}.md"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} has no YAML frontmatter")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError(f"{path}: frontmatter is not closed") from exc
    profile = yaml.safe_load("\n".join(lines[1:end])) or {}
    if profile.get("symbol") != symbol:
        raise ValueError(f"{path}: frontmatter symbol {profile.get('symbol')!r} != {symbol!r}")
    return profile


def load_asset_profiles(
    symbols: Iterable[str] | None = None, assets_dir: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Profiles for ``symbols``, or for every ``<SYMBOL>.md`` in the assets directory."""
    root = assets_dir or ASSETS_DIR
    if symbols is None:
        symbols = sorted(
            p.stem for p in root.glob("*.md") if _PROFILE_NAME.fullmatch(p.stem) and p.stem != "README"
        )
    return {symbol: load_asset_profile(symbol, root) for symbol in symbols}


def parse_window(spec: str) -> tuple[time, time]:
    """``"08:00-17:00"`` -> ``(time(8), time(17))``. Wrap-around windows are allowed."""
    start, end = (s.strip() for s in spec.split("-"))
    return time.fromisoformat(start), time.fromisoformat(end)


def profile_window_utc(
    profile: Mapping[str, Any], session: str, season: str
) -> tuple[time, time]:
    """The UTC window a profile declares for ``session`` in ``season`` (``winter``/``summer``)."""
    return parse_window(profile["sessions"][session][season])


# ---------------------------------------------------------------------------
# Server clock
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerClock:
    """Conversion between the MT5 server clock and UTC, from a measured offset.

    ``utc_offset_minutes`` is *server minus UTC* as measured at ``measured_at`` (a UTC
    instant). When ``follows_dst_of`` names a time zone, the offset moves with that zone's
    daylight-saving rule: a server measured at +120 in January that follows
    ``America/New_York`` is +180 in July. With ``follows_dst_of=None`` the offset is fixed.

    Nothing here assumes what the Exness server clock is. Measure it with
    :meth:`measure` on a logged-in terminal, record the result in
    ``history_depth.json`` (``mt5_loader.download_mt5_bars`` does) and in ``setup.yaml``.
    """

    utc_offset_minutes: int
    measured_at: datetime | None = None
    follows_dst_of: str | None = None

    def __post_init__(self) -> None:
        if self.measured_at is not None and self.measured_at.tzinfo is None:
            raise ValueError("measured_at must be timezone-aware (UTC)")
        if self.follows_dst_of is not None and self.measured_at is None:
            raise ValueError("follows_dst_of needs measured_at to anchor the DST rule")
        if self.follows_dst_of is not None:
            ZoneInfo(self.follows_dst_of)  # fail early on an unknown zone

    def offset_at(self, ts_utc: datetime) -> timedelta:
        """Server offset in force at the UTC instant ``ts_utc``."""
        base = timedelta(minutes=self.utc_offset_minutes)
        if self.follows_dst_of is None:
            return base
        zone = ZoneInfo(self.follows_dst_of)
        assert self.measured_at is not None
        dst_now = _as_utc(ts_utc).astimezone(zone).dst() or timedelta(0)
        dst_then = self.measured_at.astimezone(zone).dst() or timedelta(0)
        return base + (dst_now - dst_then)

    def to_utc(self, server_time: datetime) -> datetime:
        """Naive server-clock time -> aware UTC instant."""
        if server_time.tzinfo is not None:
            raise ValueError("server_time is a naive server-clock reading")
        guess = server_time.replace(tzinfo=UTC) - timedelta(minutes=self.utc_offset_minutes)
        return server_time.replace(tzinfo=UTC) - self.offset_at(guess)

    def to_server(self, ts_utc: datetime) -> datetime:
        """Aware UTC instant -> naive server-clock time."""
        ts = _as_utc(ts_utc)
        return (ts + self.offset_at(ts)).replace(tzinfo=None)

    def server_day_start_utc(self, day: date) -> datetime:
        """The UTC instant at which the server day ``day`` opens (server midnight)."""
        return self.to_utc(datetime.combine(day, time(0)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "utc_offset_minutes": self.utc_offset_minutes,
            "measured_at": self.measured_at.isoformat() if self.measured_at else None,
            "follows_dst_of": self.follows_dst_of,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ServerClock:
        measured = payload.get("measured_at")
        return cls(
            utc_offset_minutes=int(payload["utc_offset_minutes"]),
            measured_at=datetime.fromisoformat(measured) if measured else None,
            follows_dst_of=payload.get("follows_dst_of"),
        )

    @classmethod
    def measure(
        cls,
        mt5: Any,
        symbol: str,
        *,
        follows_dst_of: str | None = None,
        now: datetime | None = None,
        granularity_minutes: int = 15,
    ) -> ServerClock:
        """Measure the server offset from the last tick of ``symbol``.

        MT5 reports tick times as epoch seconds on the *server* clock read as if it were
        UTC, so ``tick.time - now`` is the offset. Measure while the symbol is trading: on a
        weekend the last tick is Friday's and the reading is stale. The result is rounded to
        ``granularity_minutes`` because a tick is never exactly "now".
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"no tick for {symbol}: {mt5.last_error()}")
        now = _as_utc(now or datetime.now(UTC))
        server_now = datetime.fromtimestamp(int(tick.time), tz=UTC)
        raw = (server_now - now).total_seconds() / 60
        offset = int(round(raw / granularity_minutes) * granularity_minutes)
        return cls(utc_offset_minutes=offset, measured_at=now, follows_dst_of=follows_dst_of)


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


# ---------------------------------------------------------------------------
# Session windows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionWindow:
    """A venue session in its own local time, on local weekdays."""

    name: str
    zone: str
    start: time
    end: time

    def _local(self, ts_utc: datetime) -> datetime:
        return _as_utc(ts_utc).astimezone(ZoneInfo(self.zone))

    def contains(self, ts_utc: datetime) -> bool:
        local = self._local(ts_utc)
        return local.weekday() in WEEKDAYS and self.start <= local.time() < self.end

    def minutes_since_open(self, ts_utc: datetime) -> float | None:
        """Minutes since this session opened, or None when it is not in session."""
        if not self.contains(ts_utc):
            return None
        local = self._local(ts_utc)
        opened = local.replace(hour=self.start.hour, minute=self.start.minute, second=0)
        return (local - opened).total_seconds() / 60

    def minutes_to_close(self, ts_utc: datetime) -> float | None:
        if not self.contains(ts_utc):
            return None
        local = self._local(ts_utc)
        closes = local.replace(hour=self.end.hour, minute=self.end.minute, second=0)
        return (closes - local).total_seconds() / 60


SESSIONS: dict[str, SessionWindow] = {
    "sydney": SessionWindow("sydney", ZONES["sydney"], time(8, 0), time(17, 0)),
    "tokyo": SessionWindow("tokyo", ZONES["tokyo"], time(9, 0), time(18, 0)),
    "london": SessionWindow("london", ZONES["london"], time(8, 0), time(17, 0)),
    "new_york": SessionWindow("new_york", ZONES["new_york"], time(8, 0), time(17, 0)),
    "us_cash": SessionWindow("us_cash", ZONES["new_york"], time(9, 30), time(16, 0)),
}


def fx_market_open(ts_utc: datetime) -> bool:
    """True inside the spot-FX week: Sunday 17:00 to Friday 17:00 New York time."""
    local = _as_utc(ts_utc).astimezone(ZoneInfo(ZONES["new_york"]))
    wd, t = local.weekday(), local.time()
    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday
        return t >= FX_ROLLOVER_LOCAL
    if wd == 4:  # Friday
        return t < FX_ROLLOVER_LOCAL
    return True


def rollover_flag(ts_utc: datetime, clock: ServerClock | None = None) -> bool:
    """Within ``ROLLOVER_MINUTES`` of the swap time.

    With a :class:`ServerClock` the swap time is the server day boundary (server midnight),
    which is where MT5 brokers charge swap. Without one it is the 17:00 New York interbank
    rollover, a market convention rather than a broker fact.
    """
    ts = _as_utc(ts_utc)
    if clock is not None:
        local = clock.to_server(ts)
        anchor = datetime.combine(local.date(), time(0))
    else:
        ny = ts.astimezone(ZoneInfo(ZONES["new_york"]))
        anchor = ny.replace(
            hour=FX_ROLLOVER_LOCAL.hour, minute=FX_ROLLOVER_LOCAL.minute, second=0, microsecond=0
        )
        local = ny
    delta = abs((local - anchor).total_seconds()) / 60
    delta = min(delta, abs(1440 - delta))  # distance to the nearest boundary, either side
    return delta <= ROLLOVER_MINUTES


def funding_flag(ts_utc: datetime) -> bool:
    """The bar that opens on a funding instant (00:00, 08:00, 16:00 UTC)."""
    ts = _as_utc(ts_utc)
    return ts.hour in FUNDING_HOURS_UTC and ts.minute == 0 and ts.second == 0


def session_flags(ts_utc: datetime, clock: ServerClock | None = None) -> dict[str, bool]:
    """All session flags for one UTC instant. Naive datetimes are read as UTC."""
    ts = _as_utc(ts_utc)
    flags = {name: window.contains(ts) for name, window in SESSIONS.items()}
    flags["asia"] = (flags["sydney"] or flags["tokyo"]) and not flags["london"]
    flags["overlap"] = flags["london"] and flags["new_york"]
    flags["edge_open"] = any(
        (m := w.minutes_since_open(ts)) is not None and m < EDGE_MINUTES for w in SESSIONS.values()
    )
    flags["edge_close"] = any(
        (m := w.minutes_to_close(ts)) is not None and m <= EDGE_MINUTES for w in SESSIONS.values()
    )
    flags["funding"] = funding_flag(ts)
    flags["rollover"] = rollover_flag(ts, clock)
    flags["market_open"] = fx_market_open(ts)
    return {name: bool(flags[name]) for name in FLAG_NAMES}


def session_flags_frame(timestamps: Any, clock: ServerClock | None = None):
    """Flags for a column of timestamps as a Polars frame (``timestamp`` + one bool per flag).

    Accepts a Polars Series, a pandas Index/Series or any iterable of datetimes. Computed
    once per distinct instant, which is what bar data needs; imported lazily so this module
    stays importable without Polars.
    """
    import polars as pl

    if isinstance(timestamps, pl.Series):
        values = timestamps.to_list()
        dtype = timestamps.dtype
    else:
        values = [v.to_pydatetime() if hasattr(v, "to_pydatetime") else v for v in timestamps]
        dtype = None
    distinct = sorted({v for v in values if v is not None})
    rows = {name: [] for name in FLAG_NAMES}
    for ts in distinct:
        flags = session_flags(ts, clock)
        for name in FLAG_NAMES:
            rows[name].append(flags[name])
    table = pl.DataFrame({"timestamp": distinct, **rows})
    if dtype is not None:
        table = table.with_columns(pl.col("timestamp").cast(dtype))
    return table


__all__ = [
    "ASSETS_DIR",
    "EDGE_MINUTES",
    "FLAG_NAMES",
    "ROLLOVER_MINUTES",
    "SESSIONS",
    "ZONES",
    "ServerClock",
    "SessionWindow",
    "fx_market_open",
    "funding_flag",
    "load_asset_profile",
    "load_asset_profiles",
    "parse_window",
    "profile_window_utc",
    "rollover_flag",
    "session_flags",
    "session_flags_frame",
]
