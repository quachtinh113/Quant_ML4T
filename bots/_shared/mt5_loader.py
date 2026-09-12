"""MetaTrader 5 history as the one data source for research and live.

Two entry points, and one convention.

``download_mt5_bars`` reads bars from a logged-in terminal through ``MetaTrader5``
(imported lazily, so nothing else in the repository needs the package) and writes

    ``ML4T_DATA_PATH/mt5/daily.parquet``   D1 bars
    ``ML4T_DATA_PATH/mt5/4h.parquet``      H4 bars
    ``ML4T_DATA_PATH/mt5/1h.parquet``      H1 bars
    ``ML4T_DATA_PATH/mt5/history_depth.json``

The parquet columns are ``timestamp`` (UTC instant of the bar open), ``server_time`` (the
same bar open on the broker's clock, naive), ``symbol`` (bare name), ``open``, ``high``,
``low``, ``close``, ``volume`` (MT5 ``tick_volume``), ``spread`` (points at the bar open)
and ``real_volume``. ``history_depth.json`` records, per symbol and timeframe, the first
and last bar and the bar count, plus every ``symbol_info`` field the cost block and the
broker adapter read (contract size, volume limits, swaps, triple-swap day, digits, point),
and the measured :class:`~bots._shared.sessions.ServerClock`. The feasibility stage reads
that file to decide whether the walk-forward folds fit.

``load_mt5_bars`` mirrors ``data/fx/loader.py:load_fx_pairs`` in signature and output
shape - ``timestamp, symbol, open, high, low, close, volume`` with ``timestamp`` cast to
``pl.Date`` for daily bars - so the ``fx_pairs`` stages read MT5 history with a one-line
import change. For ``daily`` the date is the **server trading day** the bar belongs to,
which is the key MT5 itself files a D1 bar under; for ``4h`` / ``1h`` the timestamp is the
UTC bar open, the same convention as the OANDA feed. ``server_day_open_utc`` turns a daily
frame back into UTC bar-open instants when a stage needs to assign bars to sessions.

**Symbol-name convention.** Research (``setup.yaml``, parquet, labels, registry) uses the
*bare* name: ``EURUSD``. The name Market Watch shows may carry an account-type suffix
(``EURUSDm`` on this Exness account); that name is used only when talking to the
terminal, through :data:`SYMBOL_MAP`. ``MT5_SYMBOL_SUFFIX`` was read from a logged-in
terminal (``mt5.symbols_get()``, see the comment at its definition); :func:`to_market_watch`
and :func:`to_bare` are the only two places the mapping is applied.

**History requests are chunked.** ``copy_rates_range`` refuses a range that spans more bars
than the terminal's "Max bars in chart" setting (``terminal_info().maxbars``, 100 000 on
this terminal) with ``(-2, 'Terminal: Invalid params')``, which is what a 2000-2026 H1
request does. ``download_mt5_bars`` therefore splits the range into windows of at most
``maxbars`` bars each and concatenates the results.

**A range request only returns what the terminal already holds; a count request makes it
fetch more.** Measured on this account on 2026-09-07: ``copy_rates_range`` in calendar chunks
returned XAUUSD H1 from 2022-10-25 (22,838 bars) while ``copy_rates_from`` with a bar *count*
returned the same symbol from 2017-03-06 (55,000+ bars) minutes later, from the same terminal
and the same login. The range path is not wrong, it is *under-synchronised*: MT5 loads history
into the chart cache on demand, and only a count-based request triggers the pull. Any decision
about how far back a design can reach must therefore be taken on ``history_mode="deep"``
(:func:`fetch_mt5_bars_deep`, or ``download_mt5_bars(..., history_mode="deep")``), never on a
range download. The default stays ``"range"`` so previously recorded depths keep their meaning.

**8-hour bars are derived, not downloaded.** The ``MetaTrader5`` package exposes no
``TIMEFRAME_H8`` that this broker serves for these symbols (Exness publishes M1..M30, H1,
H4, D1, W1, MN1 through the terminal), so ``load_mt5_bars("8h")`` reads ``4h.parquet`` and
folds the H4 bars in pairs onto the 00:00 / 08:00 / 16:00 UTC grid
(:func:`resample_4h_to_8h`): the grid of the perpetual-swap funding clock and the decision
grid of ``exness_btc_8h``. It requires every H4 bar to open on a UTC hour divisible by four
(true when the server clock is UTC, as measured on this account) and raises otherwise,
because a misaligned fold would mix two 8-hour windows. A bucket holding a single H4 bar
(the Sunday open of an FX symbol) is kept; ``require_complete=True`` drops it.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from bots._shared.sessions import ServerClock
from data.exceptions import DataNotFoundError, MissingDependencyError
from utils.data_quality import apply_max_symbols

Frequency = Literal["daily", "4h", "1h", "8h"]
# How a history request is issued to the terminal. ``range`` = ``copy_rates_range`` in calendar
# chunks (the original path, kept as the default so recorded depths do not move); ``deep`` =
# ``copy_rates_from`` stepping backwards by a bar COUNT, which makes the terminal pull history it
# has not synchronised yet. The two do not return the same history; see the module docstring.
HistoryMode = Literal["range", "deep"]

# Read from mt5.symbols_get() on 2026-09-05, account server "Exness-MT5Trial7" (demo, Pro
# account): every bot instrument is listed as "<bare>m" under "Standard\Forex", "Standard\Crypto"
# or "Standard\Indices" (EURUSDm, XAUUSDm, BTCUSDm, US500m, USTECm). Variants that are NOT the
# bot's instruments and must not be matched by a prefix search: XAUUSD247m (24/7 gold),
# BTCUSDTm (Tether-quoted), US500_x100m / USTEC_x100m / US30_x10m (enlarged contracts).
# Re-read this on any other account or server; never guess it.
MT5_SYMBOL_SUFFIX = "m"

BARE_SYMBOLS: tuple[str, ...] = (
    # exness_fx_d1
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    # candidates named in bots/README.md and bot-portfolio-exness.md
    "USDCHF",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "EURGBP",
    # exness_gold_sess, exness_btc_8h, exness_usidx_sess
    "XAUUSD",
    "XAGUSD",
    "BTCUSD",
    "ETHUSD",
    "US500",
    "USTEC",
    "US30",
)

# Market Watch name -> bare name. Rebuilt by ``set_symbol_suffix`` when the suffix is read
# from the terminal; a Market Watch name that is not in the map falls back to stripping the
# suffix, so an unlisted instrument still resolves.
SYMBOL_MAP: dict[str, str] = {f"{s}{MT5_SYMBOL_SUFFIX}": s for s in BARE_SYMBOLS}

TIMEFRAME_ATTR: dict[str, str] = {"daily": "TIMEFRAME_D1", "4h": "TIMEFRAME_H4", "1h": "TIMEFRAME_H1"}
BAR_MINUTES: dict[str, int] = {"daily": 1440, "4h": 240, "1h": 60}
# frequencies folded from a downloaded one: derived -> (source frequency, polars interval)
DERIVED_FREQUENCIES: dict[str, tuple[str, str]] = {"8h": ("4h", "8h")}
PARQUET_COLUMNS = [
    "timestamp",
    "server_time",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "spread",
    "real_volume",
]
SYMBOL_INFO_FIELDS = (
    "path",
    "description",
    "trade_contract_size",
    "volume_min",
    "volume_step",
    "volume_max",
    "digits",
    "point",
    "swap_mode",
    "swap_long",
    "swap_short",
    "swap_rollover3days",
    "trade_mode",
    "currency_base",
    "currency_profit",
    "currency_margin",
)


# ---------------------------------------------------------------------------
# Symbol names
# ---------------------------------------------------------------------------
def set_symbol_suffix(suffix: str) -> None:
    """Rebuild :data:`SYMBOL_MAP` for the suffix read from the terminal."""
    global MT5_SYMBOL_SUFFIX
    MT5_SYMBOL_SUFFIX = suffix
    SYMBOL_MAP.clear()
    SYMBOL_MAP.update({f"{s}{suffix}": s for s in BARE_SYMBOLS})


def to_bare(market_watch_name: str) -> str:
    """``EURUSDm`` -> ``EURUSD`` (via the map, else by stripping the configured suffix)."""
    if market_watch_name in SYMBOL_MAP:
        return SYMBOL_MAP[market_watch_name]
    if MT5_SYMBOL_SUFFIX and market_watch_name.endswith(MT5_SYMBOL_SUFFIX):
        return market_watch_name[: -len(MT5_SYMBOL_SUFFIX)]
    return market_watch_name


def to_market_watch(bare: str) -> str:
    """``EURUSD`` -> the name the terminal knows (``EURUSDm`` when the suffix is ``m``)."""
    for mw, b in SYMBOL_MAP.items():
        if b == bare:
            return mw
    return f"{bare}{MT5_SYMBOL_SUFFIX}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def mt5_data_dir(data_dir: Path | str | None = None) -> Path:
    """``ML4T_DATA_PATH/mt5`` unless a directory is given (tests pass a tmp path)."""
    if data_dir is not None:
        return Path(data_dir)
    from utils import ML4T_DATA_PATH

    return Path(ML4T_DATA_PATH) / "mt5"


def _parquet_path(frequency: str, data_dir: Path | str | None) -> Path:
    """The parquet a frequency is read from; a derived frequency reads its source file."""
    source = DERIVED_FREQUENCIES.get(frequency, (frequency,))[0]
    if source not in TIMEFRAME_ATTR:
        raise ValueError(
            f"frequency must be one of {list(TIMEFRAME_ATTR) + list(DERIVED_FREQUENCIES)}, got {frequency!r}"
        )
    return mt5_data_dir(data_dir) / f"{source}.parquet"


def read_history_depth(data_dir: Path | str | None = None) -> dict[str, Any]:
    path = mt5_data_dir(data_dir) / "history_depth.json"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="MT5 history depth",
            path=path,
            download_script="bots/_shared/mt5_loader.py (download_mt5_bars)",
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _import_mt5() -> Any:
    try:
        return importlib.import_module("MetaTrader5")
    except ImportError as exc:
        raise MissingDependencyError(
            package="MetaTrader5",
            install_command="uv pip install MetaTrader5   # Windows only, needs a running terminal",
            purpose="reading bars and symbol specifications from the Exness MT5 terminal",
        ) from exc


def _rates_to_frame(rates: np.ndarray, symbol: str, clock: ServerClock) -> pl.DataFrame:
    """Structured array from ``copy_rates_range`` -> one parquet-shaped frame."""
    # MT5 reports epoch seconds on the server clock read as if it were UTC; keep that
    # reading as a naive datetime and derive the true UTC instant from the measured clock.
    server_time = pl.from_epoch(
        pl.Series("server_time", rates["time"].astype("int64")), time_unit="s"
    ).cast(pl.Datetime("us"))
    # One offset per distinct server day: cheaper than a Python call per bar and exact
    # for any clock whose rule changes only on a day boundary.
    days = server_time.dt.date().unique().to_list()
    offsets = {d: clock.offset_at(clock.to_utc(datetime.combine(d, datetime.min.time()))) for d in days}
    offset_us = pl.Series(
        "offset",
        [int(offsets[d].total_seconds() * 1_000_000) for d in server_time.dt.date().to_list()],
        dtype=pl.Int64,
    )
    timestamp = (server_time.cast(pl.Int64) - offset_us).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    return pl.DataFrame(
        {
            "timestamp": timestamp,
            "server_time": server_time,
            "symbol": pl.Series([symbol] * len(rates), dtype=pl.Utf8),
            "open": rates["open"].astype("float64"),
            "high": rates["high"].astype("float64"),
            "low": rates["low"].astype("float64"),
            "close": rates["close"].astype("float64"),
            "volume": rates["tick_volume"].astype("int64"),
            "spread": rates["spread"].astype("int64"),
            "real_volume": rates["real_volume"].astype("int64"),
        }
    ).select(PARQUET_COLUMNS)


def _symbol_info_record(info: Any) -> dict[str, Any]:
    return {field: getattr(info, field, None) for field in SYMBOL_INFO_FIELDS}


DEFAULT_MAX_BARS = 100_000  # MT5 default "Max bars in chart"; the terminal's own value is read


def _copy_rates_chunked(
    mt5: Any, mw: str, tf: str, start: datetime, end: datetime, max_bars: int
) -> np.ndarray | None:
    """``copy_rates_range`` over ``[start, end]`` in windows the terminal accepts.

    A window spans at most ``max_bars`` bars of ``tf`` on the calendar (weekends included,
    so it always holds fewer real bars than the limit). Chunks are concatenated and
    de-duplicated on ``time`` because a bar can sit on a window edge.
    """
    tf_const = getattr(mt5, TIMEFRAME_ATTR[tf])
    window = timedelta(minutes=BAR_MINUTES[tf] * max(1, max_bars - 1))
    parts: list[np.ndarray] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + window, end)
        rates = mt5.copy_rates_range(mw, tf_const, cursor, stop)
        if rates is not None and len(rates):
            parts.append(rates)
        cursor = stop + timedelta(minutes=BAR_MINUTES[tf])
    if not parts:
        return None
    merged = np.concatenate(parts)
    _, keep = np.unique(merged["time"], return_index=True)
    return merged[np.sort(keep)]


def _epoch_utc(ts: datetime) -> int:
    """Epoch seconds of a server-clock instant, read as UTC.

    ``copy_rates_from`` accepts either a ``datetime`` or epoch seconds, but the package
    converts a *naive* ``datetime`` through the machine's local zone (UTC+7 here), which
    silently shifts the cursor. Passing integers removes the ambiguity.
    """
    return int((ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts).timestamp())


def _copy_rates_deep(
    mt5: Any,
    mw: str,
    tf: str,
    start: datetime,
    end: datetime,
    max_bars: int,
    *,
    max_requests: int = 64,
) -> np.ndarray | None:
    """``copy_rates_from`` stepping backwards ``max_bars`` bars at a time until the server runs dry.

    The counterpart of :func:`_copy_rates_chunked`: same signature, same return shape, but it
    walks the history backwards by bar *count* from ``end`` instead of forwards by calendar
    range from ``start``. That is what makes the terminal pull history it has not synchronised
    yet (module docstring). ``start`` is a floor, not a request window: the walk stops as soon
    as a chunk reaches it, and the result is clipped to ``[start, end]``.

    Ported from ``experiments/xau_fx_mt5/data/build_panel.py:_copy_rates_deep`` (the bot that
    first measured the difference) so there is one implementation in the repository.
    """
    tf_const = getattr(mt5, TIMEFRAME_ATTR[tf])
    count = max(1, max_bars - 1)
    parts: list[np.ndarray] = []
    cursor = end
    for _ in range(max_requests):  # 64 x 100k H1 bars = 730 years: a bound, not a target
        rates = mt5.copy_rates_from(mw, tf_const, _epoch_utc(cursor), count)
        if rates is None or len(rates) == 0:
            break
        parts.append(rates)
        first = datetime.fromtimestamp(int(rates[0]["time"]), UTC).replace(tzinfo=None)
        if len(rates) < count or first <= start:
            break
        cursor = first - timedelta(minutes=BAR_MINUTES[tf])
    if not parts:
        return None
    # np.unique(..., return_index=True) + merged[np.sort(keep)] keeps the CONCATENATION order,
    # and the chunks were appended newest-first, so that order is not chronological. The range
    # reader returns ascending time; sort here so the two paths are interchangeable and a caller
    # that does not sort afterwards cannot silently get a scrambled series.
    merged = np.concatenate(parts)
    _, keep = np.unique(merged["time"], return_index=True)
    merged = merged[np.sort(keep)]
    merged = merged[np.argsort(merged["time"], kind="stable")]
    lo, hi = _epoch_utc(start), _epoch_utc(end)
    merged = merged[(merged["time"] >= lo) & (merged["time"] <= hi)]
    return merged if len(merged) else None


#: A week holding at least this many bars counts as dense, per timeframe. An H1 week on a 24x5
#: market prints about 115 and the D1-backfilled prefix this account serves on the H1 timeframe
#: prints six, so 100 separates them with room on both sides; it is the value
#: ``experiments/xau_fx_mt5/data/build_panel.py:789`` used and the one
#: ``case_studies/exness_gold_sess/config/setup.yaml::universe.history_start`` was measured with,
#: so it is written here rather than re-derived. The H4 and daily entries are the same rule at
#: those spacings (30 and 5 bars a full week, less a fifth for holidays).
DENSE_MIN_BARS_PER_WEEK: dict[str, int] = {"daily": 4, "4h": 24, "1h": 100}


def dense_history_start(frame: pl.DataFrame, *, min_bars_per_week: int) -> str | None:
    """First ISO week from which the bar grid stays dense, or None when no week qualifies.

    The rule of ``experiments/xau_fx_mt5/data/build_panel.py:dense_history_start``: the first week
    that is itself dense and after which at least 85 % of the next 52 weeks are dense (so a
    holiday week does not end the run). It matters because a count-based (``deep``) request makes
    the terminal pull older history from the server, and on this account the oldest part of what
    comes back is **not** an hourly grid at all - six bars a week, a daily backfill served on the
    H1 timeframe - so ``first`` and ``n_bars`` alone overstate the usable depth by three years.

    Lives here rather than in a bot-local tool because ``depth_record`` writes the answer and
    ``case_studies/exness_gold_sess/01_feasibility_analysis.py`` asserts on it: written only by
    ``bots/exness_gold_sess/tools/deepen_metals_h1.py``, the next plain call to
    ``download_mt5_bars(history_mode="deep")`` would overwrite the record without the key and the
    stage would raise ``KeyError`` on a file it had itself just refreshed.
    """
    weekly = (
        frame.with_columns(pl.col("timestamp").dt.truncate("1w").alias("_week"))
        .group_by("_week")
        .len()
        .sort("_week")
    )
    counts = weekly["len"].to_numpy()
    weeks = weekly["_week"].to_list()
    dense = counts >= min_bars_per_week
    for i in range(len(dense)):
        window = dense[i : i + 52]
        if dense[i] and len(window) >= 26 and window.sum() >= 0.85 * len(window):
            week = weeks[i]
            return str(week.date() if hasattr(week, "date") else week)
    return None


def depth_record(
    frame: pl.DataFrame,
    history_mode: HistoryMode = "range",
    *,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """The ``history_depth.json`` entry for one symbol and timeframe.

    ``history_mode`` is part of the record because the same symbol on the same account has two
    different answers depending on how it was asked (module docstring); a depth number without
    its method is not interpretable.

    So is ``dense_history_start``, and for the same reason one level down: a ``deep`` read reaches
    back further than the account serves as a real intraday grid, and the sparse prefix is the
    part a session design cannot use. ``dense_n_bars`` and ``sparse_prefix_bars`` split the count
    on that boundary. On a timeframe with no sparse prefix, ``dense_history_start`` equals
    ``first`` and ``sparse_prefix_bars`` is 0, which costs nothing to record and means the key is
    always present for a consumer to assert on.
    """
    record = {
        "first": frame["timestamp"].min().isoformat(),
        "last": frame["timestamp"].max().isoformat(),
        "first_server_day": str(frame["server_time"].min().date()),
        "last_server_day": str(frame["server_time"].max().date()),
        "n_bars": frame.height,
        "history_mode": history_mode,
    }
    threshold = DENSE_MIN_BARS_PER_WEEK.get(timeframe or "")
    start = None if threshold is None else dense_history_start(frame, min_bars_per_week=threshold)
    if start is not None:
        dense = frame.filter(pl.col("timestamp").dt.date() >= pl.lit(start).str.to_date())
        record["dense_history_start"] = start
        record["dense_n_bars"] = dense.height
        record["sparse_prefix_bars"] = frame.height - dense.height
    return record


def _copy_rates(
    mt5: Any,
    mw: str,
    tf: str,
    start: datetime,
    end: datetime,
    max_bars: int,
    history_mode: HistoryMode = "range",
) -> np.ndarray | None:
    """Dispatch to the range or the count-based reader; both return the same array shape."""
    if history_mode == "deep":
        return _copy_rates_deep(mt5, mw, tf, start, end, max_bars)
    if history_mode != "range":
        raise ValueError(f"history_mode must be 'range' or 'deep', got {history_mode!r}")
    return _copy_rates_chunked(mt5, mw, tf, start, end, max_bars)


def download_mt5_bars(
    symbols: list[str],
    timeframes: tuple[str, ...] = ("daily", "4h", "1h"),
    *,
    start: datetime = datetime(2000, 1, 1),
    end: datetime | None = None,
    clock: ServerClock | None = None,
    follows_dst_of: str | None = None,
    mt5: Any = None,
    data_dir: Path | str | None = None,
    initialize_kwargs: dict[str, Any] | None = None,
    history_mode: HistoryMode = "range",
) -> dict[str, Path]:
    """Read bars for ``symbols`` (bare names) and write the parquet files and the depth record.

    ``mt5`` defaults to the real ``MetaTrader5`` module, imported here and nowhere else;
    tests pass :class:`bots._shared.testing.fake_mt5.FakeMT5`. ``clock`` defaults to
    :meth:`ServerClock.measure` on the first symbol; pass ``follows_dst_of`` when the
    server is known to follow a venue's daylight-saving rule. ``start`` / ``end`` are on
    the server clock, which is what ``copy_rates_range`` expects.

    ``history_mode="deep"`` swaps the range reader for the count-based one, which is the only
    way to learn how far back the *server* will serve rather than how far back this terminal
    happens to be synchronised (module docstring). It is not the default because the recorded
    depths of the existing bots were taken on ``"range"``.
    """
    derived = [tf for tf in timeframes if tf in DERIVED_FREQUENCIES]
    if derived:
        raise ValueError(
            f"{derived} cannot be downloaded: MT5 serves no such timeframe here; download "
            f"{[DERIVED_FREQUENCIES[tf][0] for tf in derived]} and load_mt5_bars() folds it"
        )
    unknown = [tf for tf in timeframes if tf not in TIMEFRAME_ATTR]
    if unknown:
        raise ValueError(f"unknown timeframes {unknown}; choose from {list(TIMEFRAME_ATTR)}")
    mt5 = mt5 or _import_mt5()
    out_dir = mt5_data_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end = end or datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)

    if not mt5.initialize(**(initialize_kwargs or {})):
        raise RuntimeError(f"MetaTrader5.initialize() failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        clock = clock or ServerClock.measure(
            mt5, to_market_watch(symbols[0]), follows_dst_of=follows_dst_of
        )
        depth: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "server_clock": clock.as_dict(),
            "account": {
                "login": getattr(account, "login", None),
                "server": getattr(account, "server", None),
                "trade_mode": getattr(account, "trade_mode", None),
                "currency": getattr(account, "currency", None),
            },
            "terminal": {
                "name": getattr(terminal, "name", None),
                "company": getattr(terminal, "company", None),
                "build": getattr(terminal, "build", None),
                "maxbars": getattr(terminal, "maxbars", None),
            },
            "symbol_suffix": MT5_SYMBOL_SUFFIX,
            "symbols": {},
            "timeframes": {tf: {} for tf in timeframes},
        }
        written: dict[str, Path] = {}
        for bare in symbols:
            mw = to_market_watch(bare)
            info = mt5.symbol_info(mw)
            if info is None:
                raise RuntimeError(
                    f"{mw!r} is not in Market Watch ({mt5.last_error()}); check MT5_SYMBOL_SUFFIX"
                )
            mt5.symbol_select(mw, True)
            depth["symbols"][bare] = {"market_watch": mw, **_symbol_info_record(info)}

        max_bars = int(getattr(terminal, "maxbars", None) or DEFAULT_MAX_BARS)
        for tf in timeframes:
            frames: list[pl.DataFrame] = []
            for bare in symbols:
                mw = to_market_watch(bare)
                rates = _copy_rates(mt5, mw, tf, start, end, max_bars, history_mode)
                if rates is None or len(rates) == 0:
                    depth["timeframes"][tf][bare] = {"first": None, "last": None, "n_bars": 0}
                    continue
                frame = _rates_to_frame(rates, bare, clock)
                frames.append(frame)
                depth["timeframes"][tf][bare] = depth_record(frame, history_mode, timeframe=tf)
            if not frames:
                continue
            table = pl.concat(frames).sort(["symbol", "timestamp"])
            path = out_dir / f"{tf}.parquet"
            tmp = path.with_suffix(".parquet.tmp")
            table.write_parquet(tmp)
            tmp.replace(path)
            written[tf] = path

        depth_path = out_dir / "history_depth.json"
        depth_path.write_text(json.dumps(depth, indent=2, sort_keys=True, default=str) + "\n")
        written["history_depth"] = depth_path
        return written
    finally:
        mt5.shutdown()


def fetch_mt5_bars_deep(
    symbols: list[str],
    timeframes: tuple[str, ...] = ("1h",),
    *,
    start: datetime = datetime(2000, 1, 1),
    end: datetime | None = None,
    clock: ServerClock | None = None,
    follows_dst_of: str | None = None,
    mt5: Any = None,
    initialize_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    """Deepest history the *server* will serve, in memory: ``({tf: frame}, depth)``.

    The count-based sibling of :func:`download_mt5_bars` that writes nothing. Use it when the
    answer is wanted for a few symbols but the parquet holds many: the caller merges the deep
    frames into the existing file itself and keeps the other symbols' rows untouched. Writing
    the whole file from a two-symbol download would silently truncate the panel.

    Returns the frames in ``PARQUET_COLUMNS`` shape (identical to what ``download_mt5_bars``
    writes) and a depth mapping ``{timeframe: {symbol: depth_record}}`` carrying
    ``history_mode="deep"``.
    """
    unknown = [tf for tf in timeframes if tf not in TIMEFRAME_ATTR]
    if unknown:
        raise ValueError(f"unknown timeframes {unknown}; choose from {list(TIMEFRAME_ATTR)}")
    mt5 = mt5 or _import_mt5()
    end = end or datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)
    if not mt5.initialize(**(initialize_kwargs or {})):
        raise RuntimeError(f"MetaTrader5.initialize() failed: {mt5.last_error()}")
    try:
        terminal = mt5.terminal_info()
        clock = clock or ServerClock.measure(
            mt5, to_market_watch(symbols[0]), follows_dst_of=follows_dst_of
        )
        max_bars = int(getattr(terminal, "maxbars", None) or DEFAULT_MAX_BARS)
        frames: dict[str, pl.DataFrame] = {}
        depth: dict[str, Any] = {tf: {} for tf in timeframes}
        for tf in timeframes:
            parts: list[pl.DataFrame] = []
            for bare in symbols:
                mw = to_market_watch(bare)
                mt5.symbol_select(mw, True)
                rates = _copy_rates_deep(mt5, mw, tf, start, end, max_bars)
                if rates is None or len(rates) == 0:
                    depth[tf][bare] = {"first": None, "last": None, "n_bars": 0, "history_mode": "deep"}
                    continue
                frame = _rates_to_frame(rates, bare, clock)
                parts.append(frame)
                depth[tf][bare] = depth_record(frame, "deep", timeframe=tf)
            if parts:
                frames[tf] = pl.concat(parts).sort(["symbol", "timestamp"]).select(PARQUET_COLUMNS)
        return frames, depth
    finally:
        mt5.shutdown()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def list_mt5_symbols(frequency: Frequency = "daily", data_dir: Path | str | None = None) -> list[str]:
    path = _parquet_path(frequency, data_dir)
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="MT5 bars",
            path=path,
            download_script="bots/_shared/mt5_loader.py (download_mt5_bars)",
        )
    return pl.scan_parquet(path).select("symbol").unique().collect().to_series().sort().to_list()


def _coerce_day(value: Any, arg_name: str) -> str | None:
    """Coerce a ``start_date`` / ``end_date`` argument to a bare ``"YYYY-MM-DD"`` string.

    ``load_mt5_bars`` builds a polars string literal from ``start_date`` / ``end_date``
    (``pl.lit(...).str.to_date()`` / ``.str.to_datetime()``); passed anything but a bare
    string, that raises ``SchemaError: invalid series dtype: expected `String`, got
    `date`/`datetime[...]`...`` deep inside the query plan, naming neither argument. This
    runs first so the loader's public contract is generous about what it accepts while the
    polars expressions below only ever see a string.

    Accepts, and returns the calendar DAY of:

    - ``None`` -> ``None`` (the bound is left off, unchanged);
    - a ``datetime.date``;
    - a ``datetime.datetime``, naive or timezone-aware - an aware one is converted to UTC
      FIRST, then the time-of-day and the zone are both dropped. Only the DAY is ever used as
      a bound (module docstring: "intraday keeps the whole day"), so a time-of-day carried by
      the input does not narrow that day's window;
    - a ``str``: ``"YYYY-MM-DD"``, an ISO datetime (``"2025-08-31T12:00:00Z"``), or anything
      else whose first 10 characters parse as a calendar day - nothing past index 10 is read,
      matching ``case_studies/utils/backtest_loaders.py::_naive_end_date_string`` (read, not
      imported: this module must not depend on ``case_studies``).

    Raises:
        TypeError: ``value`` is not ``None``, a ``str``, a ``datetime.date`` or a
            ``datetime.datetime``, naming ``arg_name``.
        ValueError: ``value`` is a ``str`` whose first 10 characters are not a parseable
            ``YYYY-MM-DD`` day, naming ``arg_name``.

    Two behaviour changes from before this helper existed, neither exercised by a caller in
    this repository today (every direct caller passes a bare ``"YYYY-MM-DD"`` string or
    ``None`` - see ``bots/tests/test_mt5_loader_date_bounds.py`` module docstring for the
    grep): a ``"YYYY-MM-DD HH:MM:SS"`` string used to reach ``load_mt5_bars``'s intraday
    branch whole and be parsed there as a timestamp literal, narrowing the window to that
    exact time of day; it is now truncated to its first 10 characters, i.e. to the whole
    calendar day, same as every other string form this helper accepts. An empty string
    (``""``) used to be a falsy bound that the ``if start_date:`` / ``if end_date:`` guards in
    ``load_mt5_bars`` silently skipped, leaving that side unbounded; it now reaches this
    helper, fails ``date.fromisoformat("")``, and raises ``ValueError``.

    A daily frame's ``timestamp`` is the **server trading day** (module docstring); an
    intraday frame's ``timestamp`` is a UTC instant. This helper always returns a calendar
    day and does not know which of the two a given call will filter, so the two are compared
    as equal calendar days only because this account's server clock has measured
    ``utc_offset_minutes == 0`` (``ML4T_DATA_PATH/mt5/history_depth.json::server_clock``;
    recorded for `exness_btc_8h` at `bots/exness_btc_8h/BOT.md:27`). On an account whose
    server clock carries a non-zero offset, a server trading day and its UTC calendar day are
    different days near midnight, and a caller mixing a daily bound with an intraday one
    through this same helper would silently compare the wrong day on one side. That
    assumption is not enforced here; it holds only because it has been measured on this
    account, not because it is a property of ``load_mt5_bars`` in general.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        day = value[:10]
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError(
                f"{arg_name}={value!r} does not carry a parseable YYYY-MM-DD day in its "
                "first 10 characters"
            ) from exc
        return day
    raise TypeError(
        f"{arg_name} must be None, a str, a datetime.date or a datetime.datetime, got "
        f"{type(value).__name__}"
    )


def load_mt5_bars(
    frequency: Frequency = "daily",
    symbols: list[str] | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    max_symbols: int = 0,
    *,
    include_spread: bool = False,
    data_dir: Path | str | None = None,
    require_complete: bool = False,
) -> pl.DataFrame:
    """Load MT5 OHLCV bars in the shape of ``load_fx_pairs``.

    Args:
        frequency: ``"daily"`` (server trading day, ``timestamp`` is ``pl.Date``), ``"4h"``,
            ``"1h"`` (``timestamp`` is the UTC bar open) or ``"8h"`` (folded from ``4h`` onto
            the 00/08/16 UTC grid, see :func:`resample_4h_to_8h`).
        symbols: Optional bare symbol names (``["EURUSD", "GBPUSD"]``).
        start_date: Optional inclusive start. A bare ``"YYYY-MM-DD"`` string, an ISO datetime
            string, a ``datetime.date`` or a ``datetime.datetime`` (naive or timezone-aware,
            converted to UTC first) - only the calendar DAY is used; see :func:`_coerce_day`.
        end_date: Optional inclusive end, same accepted types as ``start_date``; intraday
            keeps the whole day.
        max_symbols: Keep the N most-observed symbols (0 = all), the repo's one reduction rule.
        include_spread: Append the ``spread`` column (points at the bar open).
        data_dir: Override ``ML4T_DATA_PATH/mt5`` (tests).
        require_complete: For ``"8h"`` only, drop buckets holding fewer than two H4 bars.

    Returns:
        DataFrame with columns ``timestamp, symbol, open, high, low, close, volume``
        (plus ``spread`` when requested), sorted by symbol and timestamp.
    """
    start_date = _coerce_day(start_date, "start_date")
    end_date = _coerce_day(end_date, "end_date")
    path = _parquet_path(frequency, data_dir)
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="MT5 bars",
            path=path,
            download_script="bots/_shared/mt5_loader.py (download_mt5_bars)",
        )
    lf = pl.scan_parquet(path)
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    if include_spread:
        columns.append("spread")

    if frequency == "daily":
        lf = lf.with_columns(pl.col("server_time").dt.date().alias("timestamp"))
        if start_date:
            lf = lf.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
        if end_date:
            lf = lf.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())
    else:
        ts_type = lf.collect_schema()["timestamp"]
        tz = ts_type.time_zone if isinstance(ts_type, pl.Datetime) else None

        def _ts_literal(date_str: str) -> pl.Expr:
            lit = pl.lit(date_str).str.to_datetime()
            return lit.dt.replace_time_zone(tz) if tz else lit

        if start_date:
            lf = lf.filter(pl.col("timestamp") >= _ts_literal(start_date))
        if end_date:
            lf = lf.filter(pl.col("timestamp") < _ts_literal(end_date) + pl.duration(days=1))

    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(list(symbols)))
    frame = lf.select(columns).sort(["symbol", "timestamp"]).collect()
    if frequency in DERIVED_FREQUENCIES:
        frame = resample_4h_to_8h(frame, require_complete=require_complete)
    return apply_max_symbols(frame, max_symbols)


def resample_4h_to_8h(bars: pl.DataFrame, *, require_complete: bool = False) -> pl.DataFrame:
    """Fold H4 bars in pairs onto the 00:00 / 08:00 / 16:00 UTC grid.

    ``open`` = first bar's open, ``high`` / ``low`` = extremes, ``close`` = last bar's close,
    ``volume`` = sum, ``spread`` (when present) = first bar's spread (the value at the 8h
    open, MT5's own convention for a bar's ``spread``). Every H4 bar must open on a UTC hour
    divisible by four; otherwise the H4 grid is offset from the 8h grid and the function
    raises rather than mixing windows. ``require_complete`` drops buckets holding fewer than
    two H4 bars.
    """
    if bars.is_empty():
        return bars
    ts = bars["timestamp"]
    if not isinstance(ts.dtype, pl.Datetime):
        raise TypeError("resample_4h_to_8h expects a Datetime timestamp (the 4h frame)")
    tz = ts.dtype.time_zone
    utc = ts if tz in (None, "UTC") else ts.dt.convert_time_zone("UTC")
    misaligned = int(((utc.dt.hour() % 4 != 0) | (utc.dt.minute() != 0)).sum())
    if misaligned:
        raise ValueError(
            f"{misaligned} H4 bars do not open on a UTC hour divisible by 4; the server's H4 grid "
            "is offset from the 00/08/16 UTC grid and cannot be folded into 8h bars"
        )
    aggs = [
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ]
    if "spread" in bars.columns:
        aggs.append(pl.col("spread").first())
    aggs.append(pl.len().alias("_n"))
    folded = (
        bars.sort(["symbol", "timestamp"])
        .with_columns(pl.col("timestamp").dt.truncate("8h").alias("_bucket"))
        .group_by(["symbol", "_bucket"], maintain_order=True)
        .agg(aggs)
        .rename({"_bucket": "timestamp"})
    )
    if require_complete:
        folded = folded.filter(pl.col("_n") >= 2)
    return folded.drop("_n").select(bars.columns).sort(["symbol", "timestamp"])


def server_day_open_utc(bars: pl.DataFrame, clock: ServerClock) -> pl.DataFrame:
    """Daily frame (``timestamp`` = server day as ``pl.Date``) -> UTC bar-open instants.

    The fx_pairs stages assign each bar to a trading session through
    ``TradingCalendar.get_sessions`` on a UTC timestamp; this is the one conversion a D1
    server-day bar needs before it can go through that code unchanged.
    """
    if bars.schema["timestamp"] != pl.Date:
        raise TypeError("server_day_open_utc expects a daily frame with a pl.Date timestamp")
    days = bars["timestamp"].unique().sort().to_list()
    opens = {d: clock.server_day_start_utc(d).replace(tzinfo=None) for d in days}
    mapping = pl.DataFrame(
        {
            "timestamp": pl.Series(days, dtype=pl.Date),
            "_open_utc": pl.Series([opens[d] for d in days], dtype=pl.Datetime("us")),
        }
    )
    return (
        bars.join(mapping, on="timestamp", how="left")
        .drop("timestamp")
        .rename({"_open_utc": "timestamp"})
        .select(bars.columns)
    )


def server_day_close_utc(day: date, clock: ServerClock) -> datetime:
    """The UTC instant at which the server day ``day`` closes: the bot's decision time."""
    return clock.server_day_start_utc(day + timedelta(days=1))


__all__ = [
    "BAR_MINUTES",
    "BARE_SYMBOLS",
    "DERIVED_FREQUENCIES",
    "MT5_SYMBOL_SUFFIX",
    "PARQUET_COLUMNS",
    "SYMBOL_MAP",
    "download_mt5_bars",
    "list_mt5_symbols",
    "load_mt5_bars",
    "mt5_data_dir",
    "read_history_depth",
    "resample_4h_to_8h",
    "server_day_close_utc",
    "server_day_open_utc",
    "set_symbol_suffix",
    "to_bare",
    "to_market_watch",
]
