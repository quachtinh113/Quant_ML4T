"""Session-conditional spread measured from this account's own MT5 ticks.

The ``costs.spread_bps`` block of every Exness ``setup.yaml`` is filled from a table this
module writes, never from a marketing page (mentor reference ``mt5-exness-broker.md``
section 2, ``bot-portfolio-exness.md`` section 6). One function does the measurement, one
writes the table, and both are pure on their inputs so a test can pass a fake
``MetaTrader5`` module.

``measure_spreads(symbols, days, mt5)`` pulls ``COPY_TICKS_INFO`` ticks for the last ``days``
days in ``chunk_days`` windows through ``copy_ticks_range``, converts each tick to UTC through
the measured :class:`~bots._shared.sessions.ServerClock`, computes the quoted spread in basis
points of the mid, ``(ask - bid) / ((ask + bid) / 2) * 1e4``, classifies the tick's minute
into one session bucket from :func:`~bots._shared.sessions.session_flags` and returns one
row per ``(symbol, session)`` with the tick count, the p50 and p90 spread (in bps and in
points) and the window it was measured over. Ticks are never held in memory across chunks:
each chunk is reduced to a histogram of spreads rounded to 0.01 bp per bucket, and the
quantiles are read from the accumulated histogram, which is exact at that resolution.

Session buckets, in the priority order a minute is assigned to (first match wins):

=========  ==================================================================
bucket     definition (flags from ``sessions.session_flags``)
=========  ==================================================================
rollover   within ``ROLLOVER_MINUTES`` of the server day boundary (swap time)
overlap    London and New York both in session
london     London in session, New York not
new_york   New York in session, London not
asia       Sydney or Tokyo in session, London not
other      none of the above (the gap between the New York close and Sydney)
all        every tick, whatever the session
=========  ==================================================================

``measure_spreads(..., overlay=True)`` adds three **overlapping** buckets on top of that
chain, for bots whose execution happens on the NYSE clock (``exness_usidx_sess``):

===========  ================================================================
bucket       definition
===========  ================================================================
us_cash      09:30-16:00 New York, weekdays (``sessions.SESSIONS['us_cash']``)
edge_open    the first ``EDGE_MINUTES`` of ``us_cash``
edge_close   the last ``EDGE_MINUTES`` of ``us_cash``
===========  ================================================================

A tick counted in an overlay bucket is also counted in one of ``SESSION_BUCKETS``, so the
overlay rows are excluded from the ``all`` row and their ``n_ticks`` must never be summed
with it. ``edge_open`` / ``edge_close`` here are the edges of the US cash session only,
narrower than the same-named any-session flags of ``sessions.session_flags``.

Output: ``ML4T_DATA_PATH/mt5/spreads_by_session.parquet`` and ``.json``.

**Swap (financing).** :func:`read_swaps` reads ``swap_long``, ``swap_short``,
``swap_rollover3days``, ``swap_mode``, ``point``, ``trade_contract_size``,
``currency_profit`` and ``path`` from ``symbol_info`` for a list of bare symbols;
:func:`holding_cost_points` turns them into the swap **points** a position collects between
two UTC instants (ported from the legacy ``v9_continuum/costs.py``; negative = cost):

* one charge per **server-day boundary** crossed (server midnight from a
  :class:`~bots._shared.sessions.ServerClock`, the swap time of an MT5 broker);
* the night whose ending day is ``swap_rollover3days`` (MT5 day of week, 0 = Sunday,
  3 = Wednesday for FX and metals, 5 = Friday for indices and crypto on Exness) is
  charged three times;
* Saturday and Sunday nights are not charged for FX, metals and indices (the market is
  closed), every night is charged for crypto (24/7); the class is read from ``path``
  (``Standard\\Crypto\\...``) unless ``charges_weekends`` is given;
* only ``swap_mode`` 1 (points) is implemented: it is what this account uses on every bot
  instrument; other modes raise so a silent zero can never enter the cost stage.

:func:`swap_points_to_account` converts points to the account currency:
``points x point x trade_contract_size x lots x quote_to_account``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bots._shared.mt5_loader import mt5_data_dir, to_bare, to_market_watch
from bots._shared.sessions import EDGE_MINUTES, SESSIONS, ServerClock, session_flags_frame

SESSION_BUCKETS = ("rollover", "overlap", "london", "new_york", "asia", "other")
# Overlapping buckets, opt in through ``measure_spreads(overlay=True)``. They are *not*
# part of the exclusive chain above: a tick can land in one of these and in one of
# SESSION_BUCKETS at the same time, so their tick counts must never be added to the
# ``all`` row. All three are defined on the US cash session only (indices trade the NYSE
# clock), which is narrower than the same-named any-session flags of ``sessions.py``.
OVERLAY_BUCKETS = ("us_cash", "edge_open", "edge_close")
QUANTILES = (0.5, 0.9)
BPS_RESOLUTION = 0.01  # histogram bin width in basis points
TICK_TIME_MS_FIELD = "time_msc"
SUMMARY_FIELDS = (
    "n_ticks",
    "spread_bps_p50",
    "spread_bps_p90",
    "spread_bps_mean",
    "spread_points_p50",
    "spread_points_p90",
)


def session_bucket_expr() -> pl.Expr:
    """Polars expression mapping the flag columns of ``session_flags_frame`` to one bucket."""
    return (
        pl.when(pl.col("rollover"))
        .then(pl.lit("rollover"))
        .when(pl.col("overlap"))
        .then(pl.lit("overlap"))
        .when(pl.col("london"))
        .then(pl.lit("london"))
        .when(pl.col("new_york"))
        .then(pl.lit("new_york"))
        .when(pl.col("asia"))
        .then(pl.lit("asia"))
        .otherwise(pl.lit("other"))
        .alias("session")
    )


def us_cash_overlay(minutes: list[datetime]) -> pl.DataFrame:
    """One boolean column per :data:`OVERLAY_BUCKETS` for a list of naive-UTC minutes.

    ``us_cash`` is the NYSE cash session (09:30-16:00 New York, weekdays);
    ``edge_open`` / ``edge_close`` are its first / last ``EDGE_MINUTES``. These are
    deliberately narrower than ``sessions.session_flags``'s ``edge_open`` / ``edge_close``,
    which fire on the edge of *any* venue session and would mix the Tokyo open into an
    index bot's execution cost.
    """
    window = SESSIONS["us_cash"]
    rows = {name: [] for name in OVERLAY_BUCKETS}
    for ts in minutes:
        since = window.minutes_since_open(ts)
        to_close = window.minutes_to_close(ts)
        rows["us_cash"].append(since is not None)
        rows["edge_open"].append(since is not None and since < EDGE_MINUTES)
        rows["edge_close"].append(to_close is not None and to_close <= EDGE_MINUTES)
    return pl.DataFrame({"minute": minutes, **rows}).with_columns(
        pl.col("minute").cast(pl.Datetime("us"))
    )


def minute_buckets(
    start_utc: datetime, end_utc: datetime, clock: ServerClock, *, overlay: bool = False
) -> pl.DataFrame:
    """``minute`` (naive UTC, Datetime[us]) -> ``session`` for every minute in the window.

    With ``overlay=True`` the frame also carries one boolean column per
    :data:`OVERLAY_BUCKETS`; the exclusive ``session`` column is unchanged either way.
    """
    start = start_utc.replace(second=0, microsecond=0, tzinfo=None)
    end = end_utc.replace(tzinfo=None)
    n_minutes = int((end - start).total_seconds() // 60) + 1
    minutes = [start + timedelta(minutes=i) for i in range(n_minutes)]
    flags = session_flags_frame(pl.Series("timestamp", minutes, dtype=pl.Datetime("us")), clock)
    table = flags.select(pl.col("timestamp").alias("minute"), session_bucket_expr())
    if overlay:
        table = table.join(us_cash_overlay(minutes), on="minute", how="left")
    return table


def ticks_to_frame(ticks: np.ndarray, symbol: str, clock: ServerClock) -> pl.DataFrame:
    """``copy_ticks_range`` array -> ``timestamp`` (naive UTC), ``symbol``, ``bid``, ``ask``, ``spread_bps``."""
    if TICK_TIME_MS_FIELD in (ticks.dtype.names or ()):
        server_time = pl.from_epoch(
            pl.Series("server_time", ticks[TICK_TIME_MS_FIELD].astype("int64")), time_unit="ms"
        ).cast(pl.Datetime("us"))
    else:
        server_time = pl.from_epoch(
            pl.Series("server_time", ticks["time"].astype("int64")), time_unit="s"
        ).cast(pl.Datetime("us"))
    days = server_time.dt.date().unique().to_list()
    offsets = {
        d: clock.offset_at(clock.to_utc(datetime.combine(d, datetime.min.time()))) for d in days
    }
    offset_us = pl.Series(
        "offset",
        [int(offsets[d].total_seconds() * 1_000_000) for d in server_time.dt.date().to_list()],
        dtype=pl.Int64,
    )
    timestamp = (server_time.cast(pl.Int64) - offset_us).cast(pl.Datetime("us"))
    frame = pl.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": pl.Series([symbol] * len(ticks), dtype=pl.Utf8),
            "bid": ticks["bid"].astype("float64"),
            "ask": ticks["ask"].astype("float64"),
        }
    ).filter((pl.col("bid") > 0) & (pl.col("ask") >= pl.col("bid")))
    return frame.with_columns(
        ((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2) * 1e4).alias(
            "spread_bps"
        )
    )


def _histogram(frame: pl.DataFrame, point: float | None) -> pl.DataFrame:
    """Per (symbol, session, spread bin) tick counts; ``spread_points`` when ``point`` is known."""
    binned = frame.with_columns(
        (pl.col("spread_bps") / BPS_RESOLUTION).round(0).cast(pl.Int64).alias("bin"),
    )
    if point:
        binned = binned.with_columns(
            ((pl.col("ask") - pl.col("bid")) / point).round(0).cast(pl.Int64).alias("spread_points")
        )
    else:
        binned = binned.with_columns(pl.lit(None, dtype=pl.Int64).alias("spread_points"))
    return binned.group_by(["symbol", "session", "bin", "spread_points"]).agg(pl.len().alias("n"))


def _weighted_quantile(values: np.ndarray, counts: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values, counts = values[order], counts[order]
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, q * cum[-1], side="left"))
    return float(values[min(idx, len(values) - 1)])


def summarize_histogram(hist: pl.DataFrame, *, add_all: bool = True) -> pl.DataFrame:
    """Histogram rows -> one row per (symbol, session) plus an ``all`` row per symbol.

    ``add_all=False`` skips the ``all`` row; it is what the overlapping
    :data:`OVERLAY_BUCKETS` need, whose ticks are already counted in the exclusive chain.
    """
    with_all = (
        pl.concat([hist, hist.with_columns(pl.lit("all").alias("session"))]) if add_all else hist
    )
    rows: list[dict[str, Any]] = []
    for key, part in with_all.group_by(["symbol", "session"], maintain_order=True):
        symbol, session = key
        by_bin = part.group_by("bin").agg(pl.col("n").sum()).sort("bin")
        bins = by_bin["bin"].to_numpy().astype("float64") * BPS_RESOLUTION
        counts = by_bin["n"].to_numpy().astype("int64")
        row: dict[str, Any] = {"symbol": symbol, "session": session, "n_ticks": int(counts.sum())}
        for q in QUANTILES:
            row[f"spread_bps_p{int(q * 100)}"] = round(_weighted_quantile(bins, counts, q), 2)
        row["spread_bps_mean"] = round(float((bins * counts).sum() / counts.sum()), 3)
        known = part.drop_nulls("spread_points")
        if known.height:
            by_pt = known.group_by("spread_points").agg(pl.col("n").sum())
            pts = by_pt["spread_points"].to_numpy().astype("float64")
            cnt = by_pt["n"].to_numpy().astype("int64")
            for q in QUANTILES:
                row[f"spread_points_p{int(q * 100)}"] = _weighted_quantile(pts, cnt, q)
        else:
            for q in QUANTILES:
                row[f"spread_points_p{int(q * 100)}"] = None
        rows.append(row)
    order = {name: i for i, name in enumerate(("all", *SESSION_BUCKETS, *OVERLAY_BUCKETS))}
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("session").replace_strict(order, return_dtype=pl.Int64).alias("_o"))
        .sort(["symbol", "_o"])
        .drop("_o")
    )


def measure_spreads(
    symbols: Iterable[str],
    days: int,
    mt5: Any,
    *,
    clock: ServerClock | None = None,
    end: datetime | None = None,
    chunk_days: int = 1,
    initialize: bool = True,
    overlay: bool = False,
    log: Any = print,
) -> pl.DataFrame:
    """Spread p50 / p90 in bps of mid per symbol and session bucket, from ``days`` of ticks.

    ``symbols`` are bare names; ``mt5`` is the ``MetaTrader5`` module (or a fake). The
    terminal is initialised here unless ``initialize=False``, and shut down on exit when it
    was initialised here. ``clock`` defaults to a measurement on the first symbol; measure
    on a symbol that is trading now (a 24/7 symbol on a weekend).

    ``overlay=True`` appends one extra row per symbol per :data:`OVERLAY_BUCKETS` (the US
    cash session and its two 30-minute edges). Those rows overlap the exclusive chain, so
    they are excluded from the ``all`` row and their tick counts must not be summed with it.
    """
    symbols = list(symbols)
    if initialize and not mt5.initialize():
        raise RuntimeError(f"MetaTrader5.initialize() failed: {mt5.last_error()}")
    try:
        clock = clock or ServerClock.measure(mt5, to_market_watch(symbols[0]))
        end_utc = (end or datetime.now(UTC)).astimezone(UTC)
        start_utc = end_utc - timedelta(days=days)
        buckets = minute_buckets(start_utc, end_utc, clock, overlay=overlay)
        flags = int(getattr(mt5, "COPY_TICKS_INFO", 1))
        hists: list[pl.DataFrame] = []
        overlay_hists: list[pl.DataFrame] = []
        for bare in symbols:
            mw = to_market_watch(bare)
            info = mt5.symbol_info(mw)
            if info is None:
                raise RuntimeError(f"{mw!r} is not in Market Watch ({mt5.last_error()})")
            mt5.symbol_select(mw, True)
            point = float(getattr(info, "point", 0) or 0) or None
            cursor = start_utc
            n_total = 0
            while cursor < end_utc:
                stop = min(cursor + timedelta(days=chunk_days), end_utc)
                ticks = mt5.copy_ticks_range(
                    mw, clock.to_server(cursor), clock.to_server(stop), flags
                )
                cursor = stop
                if ticks is None or len(ticks) == 0:
                    continue
                frame = ticks_to_frame(ticks, bare, clock)
                if frame.is_empty():
                    continue
                n_total += frame.height
                frame = (
                    frame.with_columns(pl.col("timestamp").dt.truncate("1m").alias("minute"))
                    .join(buckets, on="minute", how="left")
                    .with_columns(pl.col("session").fill_null("other"))
                )
                hists.append(_histogram(frame.drop(OVERLAY_BUCKETS, strict=False), point))
                for bucket in OVERLAY_BUCKETS if overlay else ():
                    part = frame.filter(pl.col(bucket).fill_null(False))
                    if part.height:
                        overlay_hists.append(
                            _histogram(part.with_columns(pl.lit(bucket).alias("session")), point)
                        )
            log(
                f"{bare}: {n_total:,} ticks over {days} days "
                f"({start_utc:%Y-%m-%d %H:%M} .. {end_utc:%Y-%m-%d %H:%M} UTC)"
            )
        if not hists:
            raise RuntimeError("no ticks returned for any symbol")
        hist = (
            pl.concat(hists)
            .group_by(["symbol", "session", "bin", "spread_points"])
            .agg(pl.col("n").sum())
        )
        summary = summarize_histogram(hist)
        if overlay_hists:
            overlay_hist = (
                pl.concat(overlay_hists)
                .group_by(["symbol", "session", "bin", "spread_points"])
                .agg(pl.col("n").sum())
            )
            summary = pl.concat(
                [summary, summarize_histogram(overlay_hist, add_all=False)], how="vertical"
            )
        return summary.with_columns(
            pl.lit(start_utc.replace(tzinfo=None)).alias("window_start_utc"),
            pl.lit(end_utc.replace(tzinfo=None)).alias("window_end_utc"),
            pl.lit(days).alias("days"),
            pl.lit(datetime.now(UTC).replace(tzinfo=None)).alias("measured_at_utc"),
            pl.lit(clock.utc_offset_minutes).alias("server_utc_offset_minutes"),
        )
    finally:
        if initialize:
            mt5.shutdown()


def write_spreads(table: pl.DataFrame, data_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Write ``spreads_by_session.parquet`` and ``.json`` under ``ML4T_DATA_PATH/mt5``."""
    out_dir = mt5_data_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "spreads_by_session.parquet"
    table.write_parquet(parquet)
    seen = set(table["session"].to_list())
    payload: dict[str, Any] = {
        "measured_at_utc": str(table["measured_at_utc"][0]),
        "window_start_utc": str(table["window_start_utc"][0]),
        "window_end_utc": str(table["window_end_utc"][0]),
        "days": int(table["days"][0]),
        "unit": "basis points of mid, quoted spread (ask - bid), COPY_TICKS_INFO ticks",
        "buckets": [b for b in ("all", *SESSION_BUCKETS, *OVERLAY_BUCKETS) if b in seen],
        "spreads": {},
    }
    for key, part in table.group_by("symbol", maintain_order=True):
        symbol = key[0] if isinstance(key, tuple) else key
        payload["spreads"][symbol] = {
            r["session"]: {k: r[k] for k in SUMMARY_FIELDS} for r in part.to_dicts()
        }
    js = out_dir / "spreads_by_session.json"
    js.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return parquet, js


def read_spreads(data_dir: Path | str | None = None) -> pl.DataFrame:
    return pl.read_parquet(mt5_data_dir(data_dir) / "spreads_by_session.parquet")


# ---------------------------------------------------------------------------
# Swap
# ---------------------------------------------------------------------------
SWAP_FIELDS = (
    "swap_long",
    "swap_short",
    "swap_rollover3days",
    "swap_mode",
    "point",
    "trade_contract_size",
    "currency_profit",
    "path",
)
SWAP_MODE_POINTS = 1
# MT5 day of week (0 = Sunday .. 6 = Saturday) from Python weekday() (0 = Monday .. 6 = Sunday)
_PY_TO_MT5_DOW = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
_CRYPTO_PATH_MARKERS = ("crypto",)


def read_swaps(symbols: Iterable[str], mt5: Any) -> dict[str, dict[str, Any]]:
    """``symbol_info`` swap parameters per **bare** symbol from a connected terminal (or a fake).

    Raises when a symbol is not in Market Watch: a missing swap is not a zero swap.
    """
    out: dict[str, dict[str, Any]] = {}
    for bare in symbols:
        mw = to_market_watch(bare)
        info = mt5.symbol_info(mw)
        if info is None:
            raise RuntimeError(f"{mw!r} is not in Market Watch ({mt5.last_error()})")
        record: dict[str, Any] = {f: getattr(info, f, None) for f in SWAP_FIELDS}
        record["market_watch"] = mw
        record["charges_weekends"] = is_crypto_path(record.get("path"))
        out[to_bare(mw)] = record
    return out


def is_crypto_path(path: Any) -> bool:
    """True when the Market Watch ``path`` (``Standard\\Crypto\\BTCUSDm``) is a crypto group."""
    return any(marker in str(path or "").lower() for marker in _CRYPTO_PATH_MARKERS)


def rollover_nights(
    entry_utc: datetime,
    exit_utc: datetime,
    *,
    clock: ServerClock,
    triple_dow: int,
    charges_weekends: bool,
) -> tuple[int, int]:
    """``(charged_nights, triple_nights)`` between two UTC instants.

    A night is a server-day boundary (server midnight) strictly after ``entry_utc`` and at
    or before ``exit_utc``. The night belongs to the server day that just ended: its weekday
    decides the weekend skip and the triple charge. ``triple_dow`` is the MT5 day of week.
    """
    entry = _aware(entry_utc)
    exit_ = _aware(exit_utc)
    if exit_ <= entry:
        return 0, 0
    day = clock.to_server(entry).date()
    nights = triples = 0
    while True:
        day = day + timedelta(days=1)
        boundary = clock.server_day_start_utc(day)
        if boundary <= entry:
            continue
        if boundary > exit_:
            break
        ended = day - timedelta(days=1)  # the server day that just ended
        if not charges_weekends and ended.weekday() in (5, 6):
            continue
        nights += 1
        if _PY_TO_MT5_DOW[ended.weekday()] == int(triple_dow):
            triples += 1
    return nights, triples


def holding_cost_points(
    symbol: str,
    side: str,
    entry_ts: datetime,
    exit_ts: datetime,
    *,
    swaps: Mapping[str, Mapping[str, Any]],
    clock: ServerClock,
    charges_weekends: bool | None = None,
) -> float:
    """Swap points collected by holding ``symbol`` on ``side`` (``"long"``/``"short"``, or
    ``"buy"``/``"sell"``) from ``entry_ts`` to ``exit_ts`` (UTC). Negative = cost.

    ``swaps`` is the mapping :func:`read_swaps` returns (or the ``symbols`` block of
    ``history_depth.json``, which carries the same fields).
    """
    params = swaps[symbol]
    mode = int(params.get("swap_mode", SWAP_MODE_POINTS))
    if mode == 0:
        return 0.0
    if mode != SWAP_MODE_POINTS:
        raise NotImplementedError(f"{symbol}: swap_mode {mode} is not points; only mode 1 is implemented")
    side_key = side.lower()
    if side_key in ("long", "buy"):
        per_night = float(params["swap_long"])
    elif side_key in ("short", "sell"):
        per_night = float(params["swap_short"])
    else:
        raise ValueError(f"side must be long/short (or buy/sell), got {side!r}")
    if charges_weekends is None:
        charges_weekends = bool(params.get("charges_weekends", is_crypto_path(params.get("path"))))
    nights, triples = rollover_nights(
        entry_ts,
        exit_ts,
        clock=clock,
        triple_dow=int(params.get("swap_rollover3days", 3)),
        charges_weekends=charges_weekends,
    )
    return per_night * (nights + 2 * triples)


def swap_points_to_account(
    points: float, params: Mapping[str, Any], lots: float, quote_to_account: float = 1.0
) -> float:
    """Points -> account currency: ``points x point x contract x lots x quote_to_account``."""
    return (
        float(points)
        * float(params["point"])
        * float(params["trade_contract_size"])
        * float(lots)
        * float(quote_to_account)
    )


def _aware(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


__all__ = [
    "BPS_RESOLUTION",
    "OVERLAY_BUCKETS",
    "QUANTILES",
    "SESSION_BUCKETS",
    "SUMMARY_FIELDS",
    "SWAP_FIELDS",
    "holding_cost_points",
    "is_crypto_path",
    "measure_spreads",
    "minute_buckets",
    "read_spreads",
    "read_swaps",
    "rollover_nights",
    "session_bucket_expr",
    "summarize_histogram",
    "swap_points_to_account",
    "ticks_to_frame",
    "us_cash_overlay",
    "write_spreads",
]
