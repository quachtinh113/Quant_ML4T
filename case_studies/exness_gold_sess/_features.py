"""exness_gold_sess: the one decision grid, the one price panel, the one feature construction.

Every consumer imports from here rather than carrying a copy: `01_feasibility_analysis`,
`02_labels`, `03_financial_features`, the backtest price loader
(`case_studies/utils/backtest_loaders.py`), the point-in-time test and, later, step 2 of the
deployment loop. A feature computed by two implementations is the two-pipeline divergence
Chapter 25 (section 25.1) warns about: the copies agree the day they are written and drift on
the first edit. Nothing here is a notebook; the notebooks explain and check what these
functions do.

Three things live in this module.

**The decision grid.** This bot decides twice a weekday, at a *session* snapshot rather than at
a clock time (`setup.yaml::decision.snapshots`). The rule, one sentence, both venues, both
seasons:

    the decision is the close of the first H1 bar that closes at or after
    (session open + `decision.edge_block_minutes`), and the label ends at the close of the
    last H1 bar that closes at or before the session close.

The session open and close are the venue's own local hours read through `ZoneInfo`
(`bots/_shared/sessions.py:276-282`: london 08:00-17:00 Europe/London, new_york 08:00-17:00
America/New_York), so daylight saving moves the UTC hour without a table and without a
hard-coded server hour - which matters because the *server* is UTC+0 all year and does **not**
follow New York DST (`setup.yaml::decision.server_clock`, measured). On a UTC-aligned H1 grid
the rule lands on open + 60 minutes and 8 bars later at the close in every case; both facts are
*asserted*, never assumed, in :func:`decision_grid`, because a broker that shifts its bar grid
would break them silently.

The 30-minute block is not decoration. `bots/_shared/sessions.py:64` (`EDGE_MINUTES`) flags the
first and last half hour of every session as `edge_open` / `edge_close`; the decision falls
outside the `edge_open` flag by construction, and `bots/assets/XAUUSD.md:22` is the requirement
it satisfies.

**The session panel.** One row per (symbol, decision instant):

- `close` is the decision bar's close - the price every feature is knowable at and the price
  `02_labels` seals the labels on;
- `open` is the open of the first bar after the *previous* decision, which is the price the
  previous decision was executed at (`decision.execution_delay: next_bar_open`), and `high` /
  `low` / `volume` cover everything that printed between the two decisions, so the marks are
  continuous and no hour belongs to no row;
- a session whose *decision bar* is further than `decision.session_close_tolerance_minutes`
  from where the rule puts it had no fresh price at the snapshot (an outage) and **leaves the
  panel, reported**, rather than being decided on a stale bar - the `exness_fx_d1` rule
  (`setup.yaml:46-51`, the 2018-01-31 incident) applied to a session grid;
- a session whose *endpoint bar* is missing (a holiday half-day: the broker closed the session
  early) keeps its decision row and carries `label_end_ts = null`. `02_labels` writes no
  `fwd_ret_8h` there. Dropping the whole row instead - the first version of this module - both
  emptied `features_as_of` (at the decision instant no endpoint has printed yet) and selected on
  information the decision could not have had, removing exactly the thin sessions.

**The feature construction, on two grids.** `bot-portfolio-exness.md:72-74` is the rule: a
session bot's feature windows stop at the session boundary, and anything longer than a session
is read from D1 bars. So:

- *intraday* families read H1 bars. The in-session family reads only the bars of this session
  that have closed (exactly one, by the open+1h rule); the pre-session family reads the hours
  *before* the open and is a separate family for exactly that reason - naming it separately is
  what keeps "blocked at the session edge" a statement rather than a slogan;
- *long-window* families read D1 bars and are joined **asof backward on the D1 bar's close
  instant**, never on its calendar date. The D1 bar of server day D closes at 00:00 UTC on day
  D+1, so a decision at 09:00 UTC on day D may read the bar of day D-1 and no later one. Same
  rule and same reason as `exness_fx_d1/setup.yaml:313-318`.

The payoff of splitting the grids is that a 252-bar warmup costs 252 *D1* bars, about a year,
available on this account from 2014 - so a long feature does not push `universe.history_start`
forward the way it did for `exness_fx_d1` (1,885 rows of warmup, `BOT.md:81`).

Every statistic is trailing within one symbol or taken at one instant; nothing is fitted across
the sample, which is what :func:`features_as_of` lets a test prove by rebuilding the panel from
only the bars that had closed at a decision instant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from bots._shared.mt5_loader import load_mt5_bars
from bots._shared.sessions import EDGE_MINUTES, SESSIONS
from case_studies.utils.feature_engineering import EPS, rolling_zscore
from utils.data_quality import apply_max_symbols
from utils.paths import REPO_ROOT

# One H1 bar; the decision grid is built from bar CLOSES, which are opens plus this.
BAR_MINUTES = 60
# The two venues this bot decides on, in the order the panel sorts them within a day.
VENUES: dict[str, str] = {"london": "london", "ny": "new_york"}
BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
CONTEXT_COLUMNS = [
    "session",
    "session_open_ts",
    "session_close_ts",
    "session_open_px",
    "dec_open",
    "dec_high",
    "dec_low",
    "label_end_ts",
    "label_end_close",
]
PANEL_COLUMNS = [*BAR_COLUMNS, *CONTEXT_COLUMNS]
# Inputs and bookkeeping; none of them reaches the feature matrix.
EXCLUDED = {*PANEL_COLUMNS, "log_return", "d1_close_ts", "d1_close", "gsr_close"}


# ---------------------------------------------------------------------------
# The decision grid
# ---------------------------------------------------------------------------
def _venue_window(day: date, venue: str) -> tuple[datetime, datetime]:
    """(open, close) of one venue session on one local date, as naive UTC instants."""
    window = SESSIONS[VENUES[venue]]
    zone = ZoneInfo(window.zone)
    opened = datetime.combine(day, window.start, tzinfo=zone)
    closed = datetime.combine(day, window.end, tzinfo=zone)
    return (
        opened.astimezone(UTC).replace(tzinfo=None),
        closed.astimezone(UTC).replace(tzinfo=None),
    )


def session_schedule(
    first: datetime, last: datetime, *, edge_block_minutes: int = EDGE_MINUTES
) -> pl.DataFrame:
    """Every (venue, local weekday) session between ``first`` and ``last``, with its own hours.

    Columns: ``session`` (``london`` / ``ny``), ``session_date`` (the venue's local date),
    ``session_open_ts``, ``session_close_ts``, ``decision_target_ts`` (open + the edge block).
    Both endpoints are naive UTC. Weekends are excluded on the *venue's* calendar, which is
    what makes a Friday-evening Sydney bar not a London session.
    """
    rows: list[dict] = []
    day = (first - timedelta(days=2)).date()
    stop = (last + timedelta(days=2)).date()
    while day <= stop:
        if day.weekday() < 5:
            for venue in VENUES:
                opened, closed = _venue_window(day, venue)
                rows.append(
                    {
                        "session": venue,
                        "session_date": day,
                        "session_open_ts": opened,
                        "session_close_ts": closed,
                        "decision_target_ts": opened + timedelta(minutes=edge_block_minutes),
                    }
                )
        day += timedelta(days=1)
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("session_date").cast(pl.Date),
            pl.col("session_open_ts").cast(pl.Datetime("us")),
            pl.col("session_close_ts").cast(pl.Datetime("us")),
            pl.col("decision_target_ts").cast(pl.Datetime("us")),
        )
        .filter(
            (pl.col("session_close_ts") >= pl.lit(first).cast(pl.Datetime("us")))
            & (pl.col("session_open_ts") <= pl.lit(last).cast(pl.Datetime("us")))
        )
        .sort(["session_open_ts", "session"])
    )


def decision_grid(
    bars: pl.DataFrame,
    *,
    edge_block_minutes: int = EDGE_MINUTES,
    tolerance_minutes: int = BAR_MINUTES,
    expected_horizon_bars: int = 8,
    max_dropped_share: float = 0.03,
    max_stale_decision_share: float = 0.005,
    verbose: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """Resolve every session of every symbol to a decision bar, and *separately* to an endpoint.

    ``bars`` are H1 bars with ``timestamp`` = the UTC bar open (naive). Returns one row per
    (symbol, session) whose **decision** resolves, with ``label_end_ts`` **null** where the
    endpoint does not - plus a report dict counting what happened and why. The report is what
    ``01_feasibility_analysis`` prints; nothing is dropped silently.

    **Why the two resolutions are separate, and why that is not a detail.** A decision needs only
    bars that closed at or before the decision instant; an endpoint needs bars that close eight
    hours later. Resolving them together - the first version of this function did - has two
    consequences, and both are defects:

    1. :func:`features_as_of` could never return a row. At the decision instant the endpoint bar
       has not printed, so ``endpoint_lag`` is the whole 480 minutes, the session fails the
       endpoint test, and the panel truncated at that instant loses exactly the row the caller
       asked for. A point-in-time test comparing two empty frames passes and proves nothing, and
       step 2 of the Chapter 25 deployment loop cannot build a live feature row at all - which
       contradicts the one-code-path promise this module's docstring makes.
    2. It selected on the future. The 81-95 sessions per metal that failed were US early closes,
       and an early close is knowable from a calendar *in advance*; the code learnt it only from
       the last bar never printing. Dropping the row entirely removed the decision as well as the
       label, so the panel silently excluded precisely the thin-liquidity sessions - a bias in one
       direction, invisible at every gate.

    So: the **decision** is kept when

    1. a bar closes at or after ``session_open + edge_block_minutes`` and at or before the session
       close (otherwise the venue never printed inside its own session), and
    2. that bar's close is no more than ``tolerance_minutes`` past the target, i.e. there was a
       *fresh* price at the snapshot and not a stale one from earlier.

    A failure of (2) is a **stale decision** - an outage, not a holiday, since a holiday closes a
    session early and does not delay its open - and it is held to ``max_stale_decision_share``.
    Measured on this account over 2017-02-27 .. 2026-08-31: 2 sessions per metal, 0.04 %.

    The **endpoint** is resolved on the same row, and is null unless

    3. the endpoint bar closes no more than ``tolerance_minutes`` before the session close, and
    4. it is exactly ``expected_horizon_bars`` bars after the decision, which is what makes the
       primary label a fixed 8-hour horizon in both venues and both seasons.

    A null endpoint is a session the broker closed early: the decision row stays, ``02_labels``
    writes no ``fwd_ret_8h`` and no ``dir_tb_8h`` for it, and ``max_dropped_share`` bounds how
    many of them the panel may hold. `exness_fx_d1` never meets these because CME_FX excludes US
    holidays before the count; this bot's venue windows come from a weekday rule.
    """
    frame = bars.select(BAR_COLUMNS).with_columns(
        (pl.col("timestamp") + pl.duration(minutes=BAR_MINUTES)).alias("bar_close")
    )
    if frame.is_empty():
        raise ValueError("no bars to build a decision grid from")
    schedule = session_schedule(
        frame["timestamp"].min(), frame["bar_close"].max(), edge_block_minutes=edge_block_minutes
    )
    target = schedule["decision_target_ts"].to_numpy().astype("datetime64[us]")
    s_close = schedule["session_close_ts"].to_numpy().astype("datetime64[us]")

    resolved: list[pl.DataFrame] = []
    report: dict = {"per_symbol": {}, "sessions_declared": schedule.height}
    for symbol in sorted(frame["symbol"].unique().to_list()):
        g = frame.filter(pl.col("symbol") == symbol).sort("bar_close")
        closes = g["bar_close"].to_numpy().astype("datetime64[us]")
        # first bar closing at or after the target; last bar closing at or before the close
        i_dec = np.searchsorted(closes, target, side="left")
        i_end = np.searchsorted(closes, s_close, side="right") - 1
        ok = (i_dec < len(closes)) & (i_end >= 0) & (i_dec <= i_end)
        i_dec_s, i_end_s = np.clip(i_dec, 0, len(closes) - 1), np.clip(i_end, 0, len(closes) - 1)
        cand = schedule.with_columns(
            pl.Series("decision_ts", closes[i_dec_s]).cast(pl.Datetime("us")),
            pl.Series("label_end_ts", closes[i_end_s]).cast(pl.Datetime("us")),
            pl.Series("_dec_idx", i_dec_s),
            pl.Series("_end_idx", i_end_s),
            pl.Series("_in_range", ok),
            pl.lit(symbol).alias("symbol"),
        ).filter(pl.col("_in_range"))
        cand = cand.with_columns(
            (
                (pl.col("decision_ts").dt.epoch("s") - pl.col("decision_target_ts").dt.epoch("s"))
                // 60
            ).alias("decision_lag_minutes"),
            (
                (pl.col("session_close_ts").dt.epoch("s") - pl.col("label_end_ts").dt.epoch("s"))
                // 60
            ).alias("endpoint_lag_minutes"),
            (pl.col("_end_idx") - pl.col("_dec_idx")).alias("horizon_bars"),
        )
        assert cand["decision_lag_minutes"].min() >= 0, "a decision bar closes before the edge block"
        assert cand["endpoint_lag_minutes"].min() >= 0, "an endpoint closes after the session close"
        # Step 1: the DECISION. A stale snapshot is an outage and the row leaves entirely - there
        # was no price to decide on.
        stale_decision = cand.filter(pl.col("decision_lag_minutes") > tolerance_minutes)
        kept = cand.filter(pl.col("decision_lag_minutes") <= tolerance_minutes)
        # Step 2: the ENDPOINT, on the rows that survived step 1. Failing it nulls the endpoint
        # and keeps the decision, so the panel holds the early-close sessions and only their
        # labels are missing.
        endpoint_ok = (pl.col("endpoint_lag_minutes") <= tolerance_minutes) & (
            pl.col("horizon_bars") == expected_horizon_bars
        )
        early_close = kept.filter(pl.col("endpoint_lag_minutes") > tolerance_minutes)
        short_horizon = kept.filter(
            (pl.col("endpoint_lag_minutes") <= tolerance_minutes)
            & (pl.col("horizon_bars") != expected_horizon_bars)
        )
        kept = kept.with_columns(
            pl.when(endpoint_ok).then(pl.col("label_end_ts")).otherwise(None).alias("label_end_ts"),
            pl.when(endpoint_ok).then(pl.col("_end_idx")).otherwise(None).alias("_end_idx"),
        )
        unresolved = kept.filter(pl.col("label_end_ts").is_null())
        report["per_symbol"][symbol] = {
            "sessions_with_bars": cand.height,
            "no_bar_in_session": int(schedule.height - cand.height),
            "stale_decision": stale_decision.height,
            "early_close": early_close.height,
            "off_horizon": short_horizon.height,
            "kept": kept.height,
            "endpoint_unresolved": unresolved.height,
            "labelled": kept.height - unresolved.height,
            "unresolved_dates": sorted({str(d) for d in unresolved["session_date"].to_list()}),
            "dropped_dates": sorted({str(d) for d in stale_decision["session_date"].to_list()}),
        }
        if stale_decision.height > max_stale_decision_share * cand.height:
            raise AssertionError(
                f"{symbol}: {stale_decision.height} of {cand.height} sessions had no bar within "
                f"{tolerance_minutes} minutes of the snapshot. That is not a holiday - a holiday "
                "closes a session early, it does not delay its open - it is a clock or calendar "
                "problem, or an outage large enough to be one."
            )
        if unresolved.height > max_dropped_share * cand.height:
            raise AssertionError(
                f"{symbol}: {unresolved.height} of {cand.height} sessions "
                f"({unresolved.height / cand.height:.2%}) have a decision but no endpoint "
                f"{expected_horizon_bars} bars later at the session close, above the declared "
                f"{max_dropped_share:.1%}. Early closes are expected on US public holidays; this "
                "many is a calendar problem."
            )
        if verbose and (stale_decision.height or unresolved.height):
            print(
                f"{symbol}: {stale_decision.height} of {cand.height} session(s) dropped for a "
                f"stale decision; {unresolved.height} kept with a NULL endpoint "
                f"({unresolved.height / cand.height:.2%}): {early_close.height} early close, "
                f"{short_horizon.height} off-horizon"
            )
        resolved.append(
            kept.select(
                "symbol",
                "session",
                "session_date",
                "session_open_ts",
                "session_close_ts",
                "decision_ts",
                "label_end_ts",
                "_dec_idx",
                "_end_idx",
            )
        )
    grid = pl.concat(resolved).sort(["symbol", "decision_ts"])
    # The two identities the design rests on, checked rather than assumed. On this account and
    # this bar grid they hold for every row; a broker with a shifted grid would fail here
    # instead of quietly producing a 7- or 9-hour "8h" label.
    offset = (
        grid.select(
            ((pl.col("decision_ts").dt.epoch("s") - pl.col("session_open_ts").dt.epoch("s")) // 60)
        )
        .to_series()
        .unique()
        .to_list()
    )
    span = (
        grid.drop_nulls("label_end_ts")
        .select(
            ((pl.col("label_end_ts").dt.epoch("s") - pl.col("decision_ts").dt.epoch("s")) // 60)
        )
        .to_series()
        .unique()
        .to_list()
    )
    report["decision_offset_minutes"] = sorted(offset)
    report["label_span_minutes"] = sorted(span)
    report["endpoint_unresolved"] = int(grid["label_end_ts"].null_count())
    assert offset == [BAR_MINUTES], f"decision is not open+{BAR_MINUTES}min: {sorted(offset)}"
    assert span in ([], [expected_horizon_bars * BAR_MINUTES]), f"label span is not 8h: {span}"
    return grid, report


# ---------------------------------------------------------------------------
# The session panel
# ---------------------------------------------------------------------------
def session_panel(
    bars: pl.DataFrame,
    *,
    edge_block_minutes: int = EDGE_MINUTES,
    tolerance_minutes: int = BAR_MINUTES,
    expected_horizon_bars: int = 8,
    max_dropped_share: float = 0.03,
    keep_context: bool = False,
    verbose: bool = True,
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """H1 bars -> one OHLCV row per (symbol, decision instant). See the module docstring.

    Returns ``symbol, timestamp, open, high, low, close, volume`` sorted by symbol and
    timestamp, where ``timestamp`` is the decision instant (naive UTC ``Datetime``). With
    ``keep_context`` it also returns the session name, the session's own open and close
    instants, the session's opening price, the decision bar's own O/H/L, and the label
    endpoint - everything ``02_labels`` and ``03_financial_features`` need and nothing they
    would have to recompute.
    """
    frame = bars.select(BAR_COLUMNS)
    ts_dtype = frame.schema["timestamp"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        frame = frame.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us"))).with_columns(
        (pl.col("timestamp") + pl.duration(minutes=BAR_MINUTES)).alias("bar_close")
    )
    grid, report = decision_grid(
        frame,
        edge_block_minutes=edge_block_minutes,
        tolerance_minutes=tolerance_minutes,
        expected_horizon_bars=expected_horizon_bars,
        max_dropped_share=max_dropped_share,
        verbose=verbose,
    )

    panels: list[pl.DataFrame] = []
    for symbol in sorted(grid["symbol"].unique().to_list()):
        g = frame.filter(pl.col("symbol") == symbol).sort("bar_close")
        rows = grid.filter(pl.col("symbol") == symbol).sort("decision_ts")
        closes = g["bar_close"].to_numpy().astype("datetime64[us]")
        decisions = rows["decision_ts"].to_numpy().astype("datetime64[us]")
        # Every bar is assigned to the first SURVIVING decision at or after its close, so the
        # blocks tile the tape without gaps and the open of a block is the fill price of the
        # decision before it. Bars after the last decision have no block and are dropped.
        slot = np.searchsorted(decisions, closes, side="left")
        blocks = (
            g.with_columns(pl.Series("_slot", slot))
            .filter(pl.col("_slot") < len(decisions))
            .group_by("_slot")
            .agg(
                pl.col("open").sort_by("bar_close").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").sort_by("bar_close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            )
        )
        # The decision bar's own O/H/L (the opening hour of the session), which is a different
        # thing from the block's: the block starts at the previous decision. Joined ON THE KEY,
        # never assigned by position - a left join is not order-preserving in general, and a
        # silently misaligned price column is the worst bug this file could ship.
        dec_bar = g.select(
            pl.col("bar_close").alias("decision_ts"),
            pl.col("open").alias("dec_open"),
            pl.col("high").alias("dec_high"),
            pl.col("low").alias("dec_low"),
        )
        end_bar = g.select(
            pl.col("bar_close").alias("label_end_ts"),
            pl.col("close").alias("label_end_close"),
        )
        rows = (
            rows.join(dec_bar, on="decision_ts", how="left")
            .join(end_bar, on="label_end_ts", how="left")
            .with_columns(pl.col("dec_open").alias("session_open_px"))
            .sort("decision_ts")
            .with_row_index("_slot")
        )
        panels.append(
            rows.join(blocks, on="_slot", how="inner")
            .rename({"decision_ts": "timestamp"})
            .select(PANEL_COLUMNS)
        )
    panel = pl.concat(panels).sort(["symbol", "timestamp"])
    assert panel["close"].min() > 0, "a non-positive close is not a denominator"
    out = panel if keep_context else panel.select(BAR_COLUMNS)
    return (out, report) if return_report else out


class PriceGridCannotExpressLabel(RuntimeError):
    """Raised when the session panel is asked to serve as a backtest price grid.

    See :func:`load_session_panel`. This exists so that phase 5 stops loudly rather than
    silently backtesting a five-hour hold against an eight-hour label.
    """


def load_session_panel(
    setup: Mapping,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
    keep_context: bool = False,
    verbose: bool = True,
    return_report: bool = False,
    for_backtest: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """``load_mt5_bars("1h")`` for the declared universe, aggregated by :func:`session_panel`.

    ``start_date`` defaults to ``universe.history_start`` - the first week the H1 grid is dense
    on both metals, measured, not guessed (task B0). Earlier bars exist in the parquet but are
    a D1 backfill served on the H1 timeframe, six a week, and the decision rule cannot use
    them. The MT5 parquet carries every instrument of the four bots, so the universe is always
    selected explicitly.

    ``for_backtest`` **raises**, and that is the point. The engine can only change a position at
    a row of the price grid it is given. On this panel the rows are decision instants, so a
    London weight set at 09:00 UTC is carried to the next row - the New York decision at 14:00 -
    and a New York weight is carried to the next London decision the following morning. That is a
    five-hour and a nineteen-hour hold against a label that is defined over eight. No
    ``SESSION_FILTER`` can repair it: filtering the predictions changes which rows carry a weight,
    not where the grid lets the weight change. And the two sleeves *overlap* by three hours on the
    pooled book, which a grid with one row per decision cannot represent at all.

    **The fix landed on 2026-09-08** (``bots/exness_gold_sess/PRICE_GRID_DECLARATION.md``) and this
    branch is no longer on the run path: ``_PRICE_CONFIG["exness_gold_sess"]`` names the loader
    ``exness_gold_sess_h1``, the raw MT5 H1 tape keyed on the **bar close**, and the hold is a
    broker-level ``TimeExit`` position rule inside every ``13_backtest`` spec rather than the grid
    spacing (``slot_strategy`` was rejected: it expresses an exit by ceasing to emit a weight row,
    and the engine only acts at a *prediction* timestamp, of which this bot has none at 17:00).
    The raise is **kept deliberately** as the guard against anyone registering this panel as the
    price grid again; it costs nothing while nothing calls it that way.
    """
    if for_backtest:
        raise PriceGridCannotExpressLabel(
            "case_studies/exness_gold_sess: the session panel has one row per decision instant, "
            "so the shortest position it can express is decision-to-next-decision (5 h after the "
            "London snapshot, 19 h after the New York one). The primary label fwd_ret_8h runs "
            "decision -> session close, 8 h. Backtesting on this grid would train on one horizon "
            "and trade another, and nothing downstream would notice. The registered grid is the "
            "H1 tape keyed on the bar close (loader 'exness_gold_sess_h1') and the hold is a "
            "TimeExit position rule in the backtest spec; use those "
            "(bots/exness_gold_sess/PRICE_GRID_DECLARATION.md; BOT.md, phase 5 row)."
        )
    symbols = list(symbols) if symbols is not None else sorted(setup["universe"]["symbols"])
    bars = load_mt5_bars(
        "1h",
        symbols=symbols,
        start_date=start_date or str(setup["universe"]["history_start"]),
        end_date=end_date,
    )
    result = session_panel(
        bars,
        edge_block_minutes=int(setup["decision"]["edge_block_minutes"]),
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        max_dropped_share=float(setup["decision"]["max_dropped_session_share"]),
        keep_context=keep_context,
        verbose=verbose,
        return_report=return_report,
    )
    if return_report:
        panel, report = result
        return apply_max_symbols(panel, max_symbols), report
    return apply_max_symbols(result, max_symbols)


def session_of(timestamps: pl.Series, *, edge_block_minutes: int = EDGE_MINUTES) -> pl.Series:
    """``london`` / ``ny`` for decision instants, for a consumer holding only a timestamp.

    ``13_backtest`` filters predictions on this before it builds weights; a spec key alone
    changes the hash and not the result (``case_studies/utils/signals.py:537-560`` ignores
    unknown keys), which is the trap ``PHASE1_SPEC_MENTOR.md`` section 2.a names.
    """
    values = timestamps.dt.replace_time_zone(None).to_list()
    out: list[str | None] = []
    for ts in values:
        label = None
        for venue in VENUES:
            opened, closed = _venue_window(ts.date(), venue)
            if opened + timedelta(minutes=edge_block_minutes) <= ts <= closed:
                # the decision instant of that venue, not merely a moment inside it
                if ts == opened + timedelta(minutes=BAR_MINUTES):
                    label = venue
                    break
        out.append(label)
    return pl.Series("session", out, dtype=pl.Utf8)


# ---------------------------------------------------------------------------
# Feature families, one function each; setup.yaml::features.families is the register
# ---------------------------------------------------------------------------
def intraday_features(panel: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """The session's opening hour, and the hours before the open. H1 grid only."""
    assert list(windows["session_bars"]) == [1], (
        "features.windows.session_bars must be [1]: with decision.snapshot session_open_plus_1h "
        "exactly one bar of the session has closed at the decision"
    )
    return panel.with_columns(
        (pl.col("close") / (pl.col("session_open_px") + EPS) - 1).alias("sess_ret_1h"),
        ((pl.col("dec_high") - pl.col("dec_low")) / (pl.col("dec_open") + EPS)).alias(
            "sess_range_1h"
        ),
        (
            (pl.col("close") - pl.col("dec_low"))
            / (pl.col("dec_high") - pl.col("dec_low") + EPS)
        ).alias("sess_close_pos_1h"),
    )


def pre_session_features(
    panel: pl.DataFrame, bars: pl.DataFrame, windows: Mapping, *, max_bar_age_minutes: int = BAR_MINUTES
) -> pl.DataFrame:
    """Returns over the k hours ending AT the session open, with a staleness ceiling.

    Deliberately crosses the session boundary - that is the family's hypothesis - so it is
    declared as its own family in ``setup.yaml::features.families`` rather than smuggled into
    the in-session one.

    **Every probe carries a ceiling on how old the bar it matched may be.** A backward asof join
    always returns something: on a holiday, over a weekend, or through the daily 21:00-22:00 UTC
    break it reaches back until it finds a print, so ``pre_ret_8h`` would quietly become a
    ten-hour or a sixty-hour return with the same column name. The age of the matched bar is
    measured against the instant the probe asked for, and a value older than
    ``max_bar_age_minutes`` (one H1 bar, ``features.windows.pre_session_max_bar_age_minutes``)
    is **null**, not stretched. The count of nulls per column is what the warmup census and the
    ``03`` stage report; a null is visible, a sixty-hour "eight-hour return" is not.
    """
    bars_clean = bars
    ts_dtype = bars.schema["timestamp"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        bars_clean = bars.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    closes = bars_clean.select(
        "symbol",
        (pl.col("timestamp") + pl.duration(minutes=BAR_MINUTES)).cast(pl.Datetime("us")).alias("at"),
        pl.col("close").alias("px"),
    ).sort(["symbol", "at"])
    out = panel
    ages: dict[str, str] = {}
    for k in [0, *windows["pre_session_bars"]]:
        name = "_open_px" if k == 0 else f"_pre_{k}h"
        age = f"{name}_age"
        ages[name] = age
        probe = (
            out.select(
                "symbol",
                "timestamp",
                (pl.col("session_open_ts") - pl.duration(hours=k)).cast(pl.Datetime("us")).alias("at"),
            )
            .sort(["symbol", "at"])
            .join_asof(
                closes.with_columns(pl.col("at").alias("_matched_at")),
                on="at",
                by="symbol",
                strategy="backward",
            )
            .with_columns(
                ((pl.col("at").dt.epoch("s") - pl.col("_matched_at").dt.epoch("s")) // 60).alias(age)
            )
            # The ceiling. A match older than one bar is a break, a holiday or a weekend, and the
            # honest value there is nothing.
            .select(
                "symbol",
                "timestamp",
                pl.when(pl.col(age) <= max_bar_age_minutes)
                .then(pl.col("px"))
                .otherwise(None)
                .alias(name),
                pl.col(age),
            )
        )
        # joined back on the ROW KEY, not by position: join_asof re-sorts its left frame.
        out = out.join(probe, on=["symbol", "timestamp"], how="left")
    # Every return in this repository reads later-over-earlier. ``_pre_kh`` is the close k hours
    # BEFORE the open, so the open is the numerator.
    out = out.with_columns(
        [
            (pl.col("_open_px") / (pl.col(f"_pre_{k}h") + EPS) - 1).alias(f"pre_ret_{k}h")
            for k in windows["pre_session_bars"]
        ]
    )
    return out.drop([*ages, *ages.values()])


def overnight_decomposition(panel: pl.DataFrame) -> pl.DataFrame:
    """Split the move between two of a venue's own sessions into the part outside and inside.

    Over one venue-day, ``log(close_t / close_{t-1})`` on this venue's own slots is the sum of

    * ``overnight_gap`` - this session's OPENING price against the **previous session of the same
      venue**'s close, i.e. everything that printed while this venue was shut; and
    * ``sess_ret_1h`` - the opening hour, which is what :func:`intraday_features` already carries;
    * plus the part of the previous session after its own first hour, ``prev_sess_ret``.

    Why the previous session of the *same venue* and not simply the previous row. The panel sorts
    London (decision 09:00 UTC, close 17:00) before New York (decision 14:00, close 22:00) on the
    same day, so the row before a New York decision is that morning's London row - whose endpoint
    is 17:00 UTC, **three hours after** the New York decision at 14:00. Reaching for the previous
    row's ``label_end_close`` would therefore read a price from the future on every New York row.
    Shifting within ``(symbol, session)`` reaches the same venue's previous day, whose close is
    always in the past, and the assertion below states it rather than trusting it.

    The first version of this module called ``session_open_px / (last H1 close at the open) - 1``
    the overnight gap. On a continuous tape those two prices are one tick apart: measured, its
    standard deviation was 9.6e-5 against 3.9e-3 for the opening hour, forty times smaller. It was
    a bid-ask artefact wearing the name of an overnight move.
    """
    prev_close = pl.col("label_end_close").shift(1).over(["symbol", "session"])
    prev_end_ts = pl.col("label_end_ts").shift(1).over(["symbol", "session"])
    prev_open = pl.col("session_open_px").shift(1).over(["symbol", "session"])
    out = panel.sort(["symbol", "session", "timestamp"]).with_columns(
        prev_close.alias("_prev_sess_close"),
        prev_end_ts.alias("_prev_sess_end_ts"),
        prev_open.alias("_prev_sess_open"),
    )
    stale = out.filter(
        pl.col("_prev_sess_end_ts").is_not_null() & (pl.col("_prev_sess_end_ts") > pl.col("timestamp"))
    )
    assert stale.is_empty(), (
        f"{stale.height} row(s) would read a previous-session close dated after their own "
        "decision instant; the shift is not within (symbol, session)"
    )
    return (
        out.with_columns(
            (pl.col("session_open_px") / (pl.col("_prev_sess_close") + EPS) - 1).alias(
                "overnight_gap"
            ),
            (pl.col("_prev_sess_close") / (pl.col("_prev_sess_open") + EPS) - 1).alias(
                "prev_sess_ret"
            ),
        )
        .drop(["_prev_sess_close", "_prev_sess_end_ts", "_prev_sess_open"])
        .sort(["symbol", "timestamp"])
    )


def slot_momentum(panel: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """Trailing returns on the decision grid itself: k previous slots, per symbol."""
    return panel.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(k).over("symbol") - 1).alias(f"slot_ret_{k}")
            for k in windows["prev_slots"]
        ]
    )


def d1_state(d1: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """Long-window state on the D1 grid, with the bar's CLOSE instant attached.

    ``d1`` carries ``symbol``, ``timestamp`` (the server trading day, ``pl.Date``) and OHLC.
    The close instant is the day boundary that follows it: server midnight, 00:00 UTC, because
    the server clock is UTC+0 (``setup.yaml::decision.server_clock``). Joining on this instant
    rather than on the date is the whole point-in-time content of the family.
    """
    # Same library calls and the same `.over("symbol")` idiom as
    # case_studies/exness_fx_d1/_features.py:257-307, so the two bots' columns of the same name
    # are the same statistic on a different grid.
    from ml4t.engineer.features.momentum import rsi
    from ml4t.engineer.features.volatility.garman_klass_volatility import (
        garman_klass_volatility,
    )

    close, d1_year = pl.col("close"), 252
    frame = d1.sort(["symbol", "timestamp"]).with_columns(
        (pl.col("timestamp").cast(pl.Datetime("us")) + pl.duration(days=1)).alias("d1_close_ts"),
        (close / close.shift(1).over("symbol")).log().alias("_lr"),
    )
    frame = frame.with_columns(
        *[
            (close / close.shift(w).over("symbol").clip(lower_bound=EPS) - 1).alias(f"d1_ret_{w}d")
            for w in windows["momentum"]
        ],
        *[
            (close / close.rolling_mean(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"d1_price_to_ma_{w}d"
            )
            for w in windows["moving_average"]
        ],
        *[
            (pl.col("_lr").rolling_std(w).over("symbol") * np.sqrt(d1_year)).alias(
                f"d1_vol_cc_{w}d"
            )
            for w in windows["close_to_close_volatility"]
        ],
        *[
            garman_klass_volatility(
                "open", "high", "low", "close", period=w, trading_periods=d1_year
            )
            .over("symbol")
            .alias(f"d1_vol_gk_{w}d")
            for w in windows["garman_klass"]
        ],
        *[rsi("close", period=p).over("symbol").alias(f"d1_rsi_{p}") for p in windows["rsi"]],
        *[
            (close / close.rolling_max(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"d1_max_dd_{w}d"
            )
            for w in windows["drawdown"]
        ],
    )
    z = int(windows["zscore"])
    frame = frame.with_columns(
        *[
            rolling_zscore(f"d1_ret_{h}d", z, "symbol").alias(f"zscore_{h}d")
            for h in windows["zscore_horizons"]
        ]
    )
    return frame.drop("_lr")


RATIO_LEGS = ("XAUUSD", "XAGUSD")


def gold_silver_ratio(
    d1: pl.DataFrame, windows: Mapping, *, reference: pl.DataFrame | None = None
) -> pl.DataFrame:
    """The ratio, its trailing z-score, and each metal's rolling beta to the other. D1 grid.

    The feature both asset profiles name first (``bots/assets/XAGUSD.md:128``,
    ``XAUUSD.md:129``; Ch08 ``03_structural_cross_instrument_features``). It is a *cross-asset*
    family with two members, which is why it is a ratio and a beta and not a rank: a percentile
    over two names takes two values (``setup.yaml::features.ranked`` is empty for this reason).

    ``reference`` carries the D1 closes of ``features.reference_symbols`` - names that are
    **priced but not traded**. Both legs of a ratio have to be observable; only one of them has to
    be in ``universe.symbols``. Phase 6 may decide silver cannot pay its own p90 spread
    (``bots/assets/XAGUSD.md:144``), and without this argument that decision would also delete the
    family both profiles name first, because the pivot would find one column. The earlier version
    returned three all-null columns in that case, silently.
    """
    frames = [d1.select("timestamp", "symbol", "close")]
    if reference is not None and not reference.is_empty():
        frames.append(reference.select("timestamp", "symbol", "close"))
    wide = (
        pl.concat(frames)
        .unique(subset=["timestamp", "symbol"], keep="first")
        .pivot(index="timestamp", on="symbol", values="close")
    )
    missing = [leg for leg in RATIO_LEGS if leg not in wide.columns]
    if missing:
        raise ValueError(
            f"the gold-silver ratio needs D1 closes for {list(RATIO_LEGS)}; {missing} is neither "
            "in universe.symbols nor in features.reference_symbols. Declare it as a reference "
            "symbol rather than letting the family return nulls."
        )
    z, b = int(windows["ratio_zscore"]), int(windows["ratio_beta"])
    wide = (
        wide.sort("timestamp")
        .with_columns((pl.col("XAUUSD") / (pl.col("XAGUSD") + EPS)).alias("gsr"))
        .with_columns(
            ((pl.col("gsr") - pl.col("gsr").rolling_mean(z)) / (pl.col("gsr").rolling_std(z) + EPS))
            .alias(f"gsr_z_{z}d"),
            (pl.col("XAUUSD") / pl.col("XAUUSD").shift(1)).log().alias("_au"),
            (pl.col("XAGUSD") / pl.col("XAGUSD").shift(1)).log().alias("_ag"),
        )
        .with_columns(
            (
                pl.rolling_cov(pl.col("_au"), pl.col("_ag"), window_size=b)
                / (pl.col("_ag").rolling_var(b) + EPS)
            ).alias("_beta_au_on_ag"),
            (
                pl.rolling_cov(pl.col("_au"), pl.col("_ag"), window_size=b)
                / (pl.col("_au").rolling_var(b) + EPS)
            ).alias("_beta_ag_on_au"),
        )
    )
    beta = pl.concat(
        [
            wide.select(
                "timestamp",
                pl.lit("XAUUSD").alias("symbol"),
                pl.col("_beta_au_on_ag").alias(f"gsr_beta_{b}d"),
            ),
            wide.select(
                "timestamp",
                pl.lit("XAGUSD").alias("symbol"),
                pl.col("_beta_ag_on_au").alias(f"gsr_beta_{b}d"),
            ),
        ]
    )
    return d1.join(
        wide.select("timestamp", "gsr", f"gsr_z_{z}d"), on="timestamp", how="left"
    ).join(beta, on=["timestamp", "symbol"], how="left")


#: Raw bar columns of the D1 frame. They are inputs to the D1 families, never features: a D1
#: close is a price level, and a level joined onto a panel that already carries ``close`` collides
#: with it and arrives as ``close_right``. The first version of this module excluded only the
#: panel's own names, so five raw D1 columns (``open_right`` .. ``volume_right``) reached the
#: feature matrix carrying yesterday's gold price as a "feature".
D1_RAW_COLUMNS = frozenset(
    {"open", "high", "low", "close", "volume", "spread", "real_volume", "server_time", "tick_volume"}
)


def join_d1_asof(panel: pl.DataFrame, d1_features: pl.DataFrame) -> pl.DataFrame:
    """Asof-backward join of the D1 families onto the decision grid, on the D1 bar's CLOSE.

    For a decision at 09:00 UTC on day D the newest usable D1 bar is day D-1's, which closed at
    00:00 UTC on day D. Joining on the bar's calendar *date* instead would hand the decision a
    bar that closes fifteen hours after it - the single most common lookahead in a bot that
    mixes grids. Same rule as ``exness_fx_d1/setup.yaml:313-318``.

    ``d1_close_ts`` is carried onto the panel as an audited column: the point-in-time test asserts
    ``d1_close_ts <= timestamp`` on every row, and reads the age off it (about nine hours at a
    London decision, fourteen at a New York one). It is excluded from the feature matrix by
    ``EXCLUDED``; it is evidence, not a feature.
    """
    d1_cols = [
        c
        for c in d1_features.columns
        if c not in D1_RAW_COLUMNS and c not in {"symbol", "timestamp", "d1_close_ts"}
    ]
    right = d1_features.select(
        ["symbol", pl.col("d1_close_ts").alias("at"), *d1_cols]
    ).sort(["symbol", "at"])
    # The D1 bar each row actually read is carried through as ``d1_close_ts``: a join on the bar's
    # calendar date instead of its close would put this fifteen hours in the future, and nothing
    # else in the matrix would show it.
    right = right.with_columns(pl.col("at").alias("d1_close_ts"))
    joined = (
        panel.with_columns(pl.col("timestamp").alias("at"))
        .sort(["symbol", "at"])
        .join_asof(right, on="at", by="symbol", strategy="backward")
        .drop("at")
        .sort(["symbol", "timestamp"])
    )
    late = joined.filter(
        pl.col("d1_close_ts").is_not_null() & (pl.col("d1_close_ts") > pl.col("timestamp"))
    )
    assert late.is_empty(), (
        f"{late.height} row(s) read a D1 bar whose close is after the decision instant; the join "
        "is not backward on d1_close_ts"
    )
    return joined


# ---------------------------------------------------------------------------
# Session and calendar state
# ---------------------------------------------------------------------------
#: Flags of ``bots/_shared/sessions.py`` that are CONSTANT at every decision instant of this
#: grid, and are therefore asserted rather than shipped. Measured over the 8,615 development
#: decision instants of 2017-02-27 .. 2025-08-29: ``london`` and ``market_open`` are true on every
#: one (the New York decision at 13:00/14:00 UTC is inside London's 07:00-16:00/08:00-17:00
#: window), and ``edge_open``, ``edge_close``, ``rollover``, ``sydney``, ``us_cash`` and ``asia``
#: are false on every one. ``edge_open`` false is the 30-minute edge block of
#: ``bots/assets/XAUUSD.md:22`` holding by construction; a constant column carries nothing and
#: would only add a rank-deficient regressor, so the fact is a test
#: (``bots/exness_gold_sess/tests/test_sessions.py``), not a feature.
CONSTANT_DECISION_FLAGS = {
    "london": True,
    "market_open": True,
    "edge_open": False,
    "edge_close": False,
    "rollover": False,
    "sydney": False,
    "us_cash": False,
    "asia": False,
}


def _nfp_release_utc(day: date) -> datetime | None:
    """The Non-Farm Payrolls release instant on ``day``, or None if there is none.

    The rule, not a table: the Bureau of Labor Statistics releases the Employment Situation at
    08:30 in New York on the **first Friday of the month**, so the UTC instant is 12:30 under EDT
    and 13:30 under EST - read through ``ZoneInfo`` and never hard-coded, the same discipline the
    session windows use. This is the only one of the three releases
    ``PHASE1_SPEC_MENTOR.md`` section 2.f names that is a rule; CPI and FOMC dates are announced
    rather than derived, and ``setup.yaml`` declares them PLANNED for want of a calendar file.

    The rule has known exceptions - the release moves when the first Friday is a holiday, and it
    moved during the 2013 and 2018-19 shutdowns - so the family's ``failure_mode`` says so. A flag
    that is wrong on a handful of months is still knowable in advance, which is the property that
    matters here.
    """
    if day.weekday() != 4 or day.day > 7:
        return None
    zone = ZoneInfo(SESSIONS["new_york"].zone)
    local = datetime.combine(day, time(8, 30), tzinfo=zone)
    return local.astimezone(UTC).replace(tzinfo=None)


def session_state(panel: pl.DataFrame) -> pl.DataFrame:
    """State columns read off the decision instant alone: venue, season, weekday, news window.

    ``role: state`` in the register. None of these is a signal: they say what kind of decision
    this is, and a model reads them beside a signal to learn that the signal behaves differently
    on a New York slot, in the winter season, or on a payroll Friday. Every one is knowable in
    advance - a venue's DST calendar and the first Friday of a month are not market data - so the
    family's ``lag`` is 0 and its ``lookback`` is 0.

    ``is_ny`` is constant inside a filtered book and carries nothing there; that is the family's
    declared failure mode and it is what the per-book staleness screen in ``05_evaluation``
    reports rather than discovers.
    """
    rows = panel.select("symbol", "timestamp", "session", "session_open_ts", "session_close_ts")
    values: list[dict] = []
    for symbol, ts, sess, open_ts, close_ts in rows.iter_rows():
        venue = SESSIONS[VENUES[sess]]
        # The venue's own UTC offset on this date. A London session opens 08:00 UTC in GMT and
        # 07:00 UTC in BST; the server never moves, the venue does.
        offset_h = (
            datetime.combine(open_ts.date(), time(12), tzinfo=ZoneInfo(venue.zone))
            .utcoffset()
            .total_seconds()
            / 3600
        )
        nfp = _nfp_release_utc(open_ts.date())
        values.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "is_ny": float(sess == "ny"),
                "is_venue_dst": float(offset_h == _standard_offset_hours(venue.zone) + 1),
                "dow": float(open_ts.weekday()),
                # the release lands inside the decision bar itself: the bar the decision is taken
                # on already contains it, so the decision reads a post-release price
                "news_nfp_decision_bar": float(
                    nfp is not None and ts - timedelta(minutes=BAR_MINUTES) <= nfp < ts
                ),
                # the release lands after the decision and before the session close: the position
                # is held through it
                "news_nfp_label_window": float(nfp is not None and ts <= nfp < close_ts),
            }
        )
    return panel.join(pl.DataFrame(values), on=["symbol", "timestamp"], how="left")


def _standard_offset_hours(zone: str) -> float:
    """The zone's winter (standard) UTC offset in hours, read from the zone database."""
    return (
        datetime(2020, 1, 15, 12, tzinfo=ZoneInfo(zone)).utcoffset().total_seconds() / 3600
    )


def build_features(
    panel: pl.DataFrame,
    bars: pl.DataFrame,
    d1: pl.DataFrame,
    windows: Mapping,
    *,
    d1_reference: pl.DataFrame | None = None,
    macro_vintages: pl.DataFrame | None = None,
    macro_legs: Mapping[str, str] | None = None,
    releases: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Every family of ``setup.yaml::features.families`` that has data, on one panel.

    ``panel`` must carry the context columns (``session_panel(keep_context=True)``); ``bars``
    are the H1 bars the panel was built from; ``d1`` are the D1 bars of the same symbols;
    ``d1_reference`` are the D1 bars of ``features.reference_symbols`` - priced, not traded -
    which the gold-silver ratio needs when a leg is not in the universe.

    The two PLANNED families of the register are optional inputs, ``None`` by default, so the
    matrix is **unchanged** until they are switched on (generation 2, block 3,
    ``GEN2_DECLARATION_2026-09-10.md`` section 3): ``macro_vintages`` with ``macro_legs`` builds
    *real yields and the dollar* (:func:`real_yields_dollar`) and ``releases`` builds *CPI and
    FOMC release windows* (:func:`release_windows`). Both read publication-dated inputs and
    refuse observation-dated ones.
    """
    built = intraday_features(panel, windows)
    built = pre_session_features(
        built,
        bars,
        windows,
        max_bar_age_minutes=int(windows.get("pre_session_max_bar_age_minutes", BAR_MINUTES)),
    )
    built = overnight_decomposition(built)
    built = slot_momentum(built, windows)
    built = session_state(built)
    d1_features = gold_silver_ratio(d1_state(d1, windows), windows, reference=d1_reference)
    built = join_d1_asof(built, d1_features)
    if macro_vintages is not None:
        if not macro_legs:
            raise ValueError("macro_vintages were given without macro_legs (one series id a leg)")
        built = real_yields_dollar(built, macro_vintages, legs=macro_legs, windows=windows)
    if releases is not None:
        built = release_windows(
            built,
            releases,
            window_minutes=int(windows.get("release_window_minutes", RELEASE_WINDOW_MINUTES)),
        )
    return built.sort(["symbol", "timestamp"])


def feature_columns(built: pl.DataFrame) -> list[str]:
    return [c for c in built.columns if c not in EXCLUDED and not c.startswith("_")]


def warmup_expectations(
    windows: Mapping,
    *,
    macro_legs: Mapping[str, str] | None = None,
    release_kinds: Sequence[str] = (),
) -> dict[str, dict[str, int]]:
    """Longest chain of trailing bars each column reads, **keyed by the grid it counts on**.

    The two PLANNED families add entries only when they are switched on (``macro_legs`` /
    ``release_kinds`` non-empty), so a stage that builds without them audits the same set of
    columns it did before. Their columns are joined onto the panel from a *published* timeline
    that starts years before the panel does, so on the panel they are dense from row 1 and are
    listed at 1 slot, the way the state family is.

    The floor the warmup audit holds each column to. The return value is two dictionaries, not
    one, and the split is the whole point of a two-grid design:

    * ``"decision slots"`` - columns built on the panel itself. Their warmup is counted in rows
      of this panel, two a weekday, and :func:`case_studies.utils.feature_engineering.warmup_audit`
      can be run on the panel directly.
    * ``"D1 bars"`` - columns built on the D1 frame and asof-joined. Their warmup is counted in
      D1 bars and has to be audited **on the D1 frame**, before the join. Auditing them on the
      panel is meaningless and, worse, it *raises*: this bot reads D1 from 2014 while the panel
      starts in 2017, so ``d1_ret_252d`` is already dense at panel row 1 and an audit expecting
      252 rows of warmup would report a column "populated from fewer bars than its window spans".
      Mixing the units is what makes a two-grid design look like it needs a year of H1 history
      when what it needs is a year of D1 history.

    The state family (``is_*``, ``dow``, ``news_*``) reads one instant and has no warmup at all;
    it is listed at 1 slot so the audit still proves the columns are not null everywhere.
    """
    panel: dict[str, int] = {
        "sess_ret_1h": 1,
        "sess_range_1h": 1,
        "sess_close_pos_1h": 1,
        # both need the previous session OF THE SAME VENUE, which is two slots back
        "overnight_gap": 2,
        "prev_sess_ret": 2,
        "is_ny": 1,
        "is_venue_dst": 1,
        "dow": 1,
        "news_nfp_decision_bar": 1,
        "news_nfp_label_window": 1,
    }
    panel |= {f"pre_ret_{k}h": 1 for k in windows["pre_session_bars"]}
    panel |= {f"slot_ret_{k}": k for k in windows["prev_slots"]}
    daily: dict[str, int] = {}
    daily |= {f"d1_ret_{w}d": w for w in windows["momentum"]}
    daily |= {f"d1_price_to_ma_{w}d": w for w in windows["moving_average"]}
    daily |= {f"d1_vol_cc_{w}d": w + 1 for w in windows["close_to_close_volatility"]}
    daily |= {f"d1_vol_gk_{w}d": w for w in windows["garman_klass"]}
    daily |= {f"d1_rsi_{w}": w for w in windows["rsi"]}
    daily |= {f"d1_max_dd_{w}d": w for w in windows["drawdown"]}
    daily |= {f"zscore_{h}d": int(windows["zscore"]) + h for h in windows["zscore_horizons"]}
    daily["gsr"] = 1
    daily[f"gsr_z_{int(windows['ratio_zscore'])}d"] = int(windows["ratio_zscore"])
    daily[f"gsr_beta_{int(windows['ratio_beta'])}d"] = int(windows["ratio_beta"]) + 1
    if macro_legs:
        for column in macro_column_names(macro_legs, windows):
            panel[column] = 1
    for kind in release_kinds:
        for column in release_column_names(kind):
            panel[column] = 1
    return {"decision slots": panel, "D1 bars": daily}


# ---------------------------------------------------------------------------
# Point-in-time harness: rebuild the panel from what had closed at an instant
# ---------------------------------------------------------------------------
def bars_closed_by(
    bars: pl.DataFrame, at: datetime, bar_minutes: int = BAR_MINUTES
) -> pl.DataFrame:
    """Only the bars whose CLOSE is at or before ``at``. The definition of knowable."""
    ts_dtype = bars.schema["timestamp"]
    if getattr(ts_dtype, "time_zone", None) is None and at.tzinfo is not None:
        at_cmp = at.replace(tzinfo=None)
    elif getattr(ts_dtype, "time_zone", None) is not None and at.tzinfo is None:
        at_cmp = at.replace(tzinfo=UTC)
    else:
        at_cmp = at
    return bars.filter(pl.col("timestamp") + pl.duration(minutes=bar_minutes) <= pl.lit(at_cmp))


def features_as_of(
    at: datetime,
    bars: pl.DataFrame,
    d1: pl.DataFrame,
    windows: Mapping,
    *,
    d1_reference: pl.DataFrame | None = None,
    macro_vintages: pl.DataFrame | None = None,
    macro_legs: Mapping[str, str] | None = None,
    releases: pl.DataFrame | None = None,
    **panel_kwargs,
) -> pl.DataFrame:
    """Rebuild the feature row(s) dated ``at`` using only data that had closed by ``at``.

    The harness ``tests/test_lookahead.py`` compares against the batch panel: identical values
    prove that no column read a bar printed after the decision. It is the same code path, not a
    reimplementation - which is the only kind of point-in-time test worth running.

    It returns **one row per symbol quoting at that instant**, and the caller must check that:
    a comparison of two empty frames passes and proves nothing. Before the decision/endpoint split
    in :func:`decision_grid` this function could only ever return zero rows, because at the
    decision instant the session's endpoint bar has not printed and the whole session was dropped.
    """
    h1 = bars_closed_by(bars, at)
    d1_pit = bars_closed_by(
        d1.with_columns(pl.col("timestamp").cast(pl.Datetime("us"))), at, bar_minutes=24 * 60
    ).with_columns(pl.col("timestamp").cast(pl.Date))
    ref_pit = None
    if d1_reference is not None:
        ref_pit = bars_closed_by(
            d1_reference.with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            at,
            bar_minutes=24 * 60,
        ).with_columns(pl.col("timestamp").cast(pl.Date))
    panel = session_panel(h1, keep_context=True, verbose=False, **panel_kwargs)
    # The macro inputs are truncated on their PUBLICATION date, not their observation date: a
    # vintage or a schedule entry is knowable from `available_at`, and what a decision at `at`
    # may read is what was available on the decision's own date.
    macro_pit = (
        None if macro_vintages is None else vintages_available_by(macro_vintages, at.date())
    )
    releases_pit = None if releases is None else releases_known_by(releases, at.date())
    built = build_features(
        panel,
        h1,
        d1_pit,
        windows,
        d1_reference=ref_pit,
        macro_vintages=macro_pit,
        macro_legs=macro_legs,
        releases=releases_pit,
    )
    at_naive = at.astimezone(UTC).replace(tzinfo=None) if at.tzinfo is not None else at
    return built.filter(pl.col("timestamp") == pl.lit(at_naive).cast(pl.Datetime("us")))


# ---------------------------------------------------------------------------
# The declared closure calendar (generation 2, block 1)
# ---------------------------------------------------------------------------
#: ``bots/exness_gold_sess/calendar/closures_metals.parquet``, written only by
#: ``bots/exness_gold_sess/tools/build_closure_calendar.py``. Two sources and no hand-typed date:
#: ``exchange_calendars:<id>@<version>`` rows are the package's declaration, ``tape:*`` rows are
#: what the development tape showed (``decision_grid``'s ``unresolved_dates`` and the census
#: Mon-Fri holes). ``setup.yaml::decision.closure_calendar`` documents the file; nothing in this
#: module's decision grid, panel or feature construction reads it - the panel is sealed and its
#: digest must not move (``GEN2_DECLARATION_2026-09-10.md`` L0.3).
CLOSURE_CALENDAR_PATH = (
    REPO_ROOT / "bots" / "exness_gold_sess" / "calendar" / "closures_metals.parquet"
)
CLOSURE_COLUMNS = ["date", "symbol", "kind", "close_utc", "source", "source_version", "reconciled_on"]
CLOSURE_KINDS = ("closed", "early_close")


def declared_closures(
    path: Path | str | None = None,
    *,
    symbols: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
    package_only: bool = False,
    verify: bool = True,
) -> pl.DataFrame:
    """The declared closure calendar, read-only and checked against its sidecar digest.

    Returns the frame with :data:`CLOSURE_COLUMNS`, sorted by ``symbol, date, source``.
    ``package_only`` keeps the ``exchange_calendars`` rows and drops the ``tape:*`` ones - what a
    consumer wants when it must not learn a closure from the tape (kill criterion (g) in
    ``BOT.md``, the block-5 early-close filter). With ``verify`` the file's content digest is
    recomputed and compared with the sidecar ``write_artifact`` left beside it, so a hand-edited
    parquet is refused rather than read.
    """
    from case_studies.utils.artifact_digest import read_digest, value_digest

    path = Path(path) if path is not None else CLOSURE_CALENDAR_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; build it with "
            "bots/exness_gold_sess/tools/build_closure_calendar.py"
        )
    frame = pl.read_parquet(path)
    missing = [c for c in CLOSURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks the declared columns {missing}")
    if verify:
        record = read_digest(path)
        digest = value_digest(frame)
        if digest != record["digest"]:
            raise RuntimeError(
                f"{path} does not match its sidecar: content digest {digest} != recorded "
                f"{record['digest']}. The file is written only by build_closure_calendar.py."
            )
    unknown = sorted(set(frame["kind"].unique().to_list()) - set(CLOSURE_KINDS))
    if unknown:
        raise ValueError(f"{path} carries kinds outside {CLOSURE_KINDS}: {unknown}")
    if package_only:
        frame = frame.filter(pl.col("source").str.starts_with("exchange_calendars:"))
    if symbols is not None:
        frame = frame.filter(pl.col("symbol").is_in(list(symbols)))
    if kinds is not None:
        frame = frame.filter(pl.col("kind").is_in(list(kinds)))
    return frame.select(CLOSURE_COLUMNS).sort(["symbol", "date", "source"])


# ---------------------------------------------------------------------------
# The two PLANNED families (generation 2, block 3): code and tests exist, the data does not
# ---------------------------------------------------------------------------
#: ``bots/_shared/macro_config.yaml``, the one place a FRED series id lives. Its ``gold`` block
#: is this bot's (legs, lag, raw file); the ``carry`` block beside it is ``exness_fx_d1``'s and is
#: not read here. ``GEN2_DECLARATION_2026-09-10.md`` ruling 3.1: both families stay
#: ``pattern: PLANNED`` in ``setup.yaml`` until ``FRED_API_KEY`` has a value; nothing below is
#: called by ``03_financial_features`` today, so ``features/financial.parquet`` does not move.
MACRO_CONFIG_PATH = REPO_ROOT / "bots" / "_shared" / "macro_config.yaml"
#: The ALFRED raw table shape (``data/macro/download_alfred.py``): one row per observation date
#: with the vintage it was first published in. The aligned panel drops ``vintage_date`` and is
#: refused - it forward-fills on the OBSERVATION date and would date a value before anyone saw it.
MACRO_VINTAGE_COLUMNS = ["series", "timestamp", "vintage_date", "value"]
#: A release-calendar row: what is released (``cpi`` / ``fomc``), the instant it is released
#: (naive UTC) and the date the schedule entry was announced - the publication date of a
#: schedule, without which a flag "a release falls inside this hold" is not point-in-time.
RELEASE_COLUMNS = ["kind", "release_ts", "announced_on"]
#: ``bots/assets/XAUUSD.md`` avoid window around CPI / FOMC: +-30 minutes. Read from
#: ``setup.yaml::features.windows.release_window_minutes`` by the stage; the constant is the
#: same number so the function has a default that matches the declaration.
RELEASE_WINDOW_MINUTES = 30
MACRO_DEFAULT_ZSCORE_WINDOW = 252
MACRO_DOLLAR_CHANGE_WINDOW = 21


def load_gold_macro_config(path: Path | str | None = None) -> dict:
    """The ``gold`` block of ``bots/_shared/macro_config.yaml``: series ids, lag, raw file.

    Raises rather than defaulting, the way ``exness_fx_d1/_features.load_macro_config`` does
    for its ``carry`` block: a macro column built on an id nobody declared is a number with
    no provenance, and a lag below one day would claim to know the release hour.
    """
    import yaml

    config = yaml.safe_load(Path(path or MACRO_CONFIG_PATH).read_text(encoding="utf-8"))
    gold = config.get("gold")
    if not gold or not gold.get("legs"):
        raise ValueError(f"no gold block with legs in {path or MACRO_CONFIG_PATH}")
    for key in ("availability_lag_days", "raw_file"):
        if key not in gold:
            raise ValueError(f"gold block is missing {key!r}")
    if int(gold["availability_lag_days"]) < 1:
        raise ValueError(
            "availability_lag_days must be at least one day: an ALFRED vintage is dated by the "
            "day of release and says nothing about the hour, and the London decision of this "
            "bot is taken before the New York morning"
        )
    return gold


def load_macro_vintages(
    gold: Mapping, *, storage_dir: Path | str | None = None
) -> pl.DataFrame:
    """The ALFRED initial-release RAW table for the gold legs, with ``available_at``.

    Only the raw file is accepted (see :data:`MACRO_VINTAGE_COLUMNS`). ``available_at`` is
    ``vintage_date + availability_lag_days`` and is the date from which a decision may read the
    value. Raises ``FileNotFoundError`` with the download command when the file is absent.
    """
    from bots._shared.mt5_loader import mt5_data_dir

    directory = (
        Path(storage_dir)
        if storage_dir is not None
        else mt5_data_dir().parent / str(gold.get("storage_path", "mt5/macro_exness_gold_sess"))
    )
    path = Path(directory) / str(gold["raw_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"gold macro vintages not downloaded: {path}. Run "
            "`uv run python case_studies/exness_gold_sess/_check_gold_series.py` and then "
            "`uv run python data/macro/download_alfred.py --config "
            "bots/exness_gold_sess/macro_config.yaml`"
        )
    return prepare_macro_vintages(pl.read_parquet(path), lag_days=int(gold["availability_lag_days"]))


def prepare_macro_vintages(frame: pl.DataFrame, *, lag_days: int) -> pl.DataFrame:
    """Validate a vintage table and add ``available_at = vintage_date + lag_days``.

    Refuses a frame without ``vintage_date`` (an observation-dated panel) and a lag below one
    day, for the reasons :func:`load_gold_macro_config` gives.
    """
    missing = [c for c in MACRO_VINTAGE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"not an ALFRED raw table (missing {missing}); an observation-dated macro panel is "
            "not point-in-time and is refused"
        )
    if int(lag_days) < 1:
        raise ValueError("availability_lag_days must be at least one day")
    return (
        frame.select(MACRO_VINTAGE_COLUMNS)
        .with_columns(
            pl.col("timestamp").cast(pl.Date),
            pl.col("vintage_date").cast(pl.Date),
            pl.col("value").cast(pl.Float64),
            pl.col("series").str.to_lowercase(),
        )
        .with_columns((pl.col("vintage_date") + pl.duration(days=int(lag_days))).alias("available_at"))
        .sort(["series", "available_at", "timestamp"])
    )


def vintages_available_by(vintages: pl.DataFrame, on: date) -> pl.DataFrame:
    """Only the vintages a decision dated ``on`` could read: ``available_at <= on``."""
    if "available_at" not in vintages.columns:
        raise ValueError("vintages carry no available_at; pass them through prepare_macro_vintages")
    return vintages.filter(pl.col("available_at") <= pl.lit(on))


def macro_column_names(legs: Mapping[str, str], windows: Mapping) -> list[str]:
    """The columns :func:`real_yields_dollar` emits for ``legs``, in emission order."""
    z = int(windows.get("real_yield_zscore", MACRO_DEFAULT_ZSCORE_WINDOW))
    names: list[str] = []
    for leg in legs:
        if leg == "dollar":
            names += [f"macro_{leg}_ret_{MACRO_DOLLAR_CHANGE_WINDOW}d", f"macro_{leg}_z_{z}d"]
        else:
            names += [f"macro_{leg}", f"macro_{leg}_chg_{MACRO_DOLLAR_CHANGE_WINDOW}d", f"macro_{leg}_z_{z}d"]
    return names


def _trailing_zscore(expr: pl.Expr, window: int) -> pl.Expr:
    """``(x - trailing mean) / trailing std`` over ``window`` rows of ONE series, no partition.

    The shared :func:`case_studies.utils.feature_engineering.rolling_zscore` partitions by an
    entity column; a published macro series has no entity, so the same trailing arithmetic is
    written once here with the same ``EPS`` floor.
    """
    return (expr - expr.rolling_mean(window)) / expr.rolling_std(window).clip(lower_bound=EPS)


def _published_timeline(vintages: pl.DataFrame, series_id: str) -> pl.DataFrame:
    """One value per publication date for one series: the latest observation published that day.

    Trailing statistics are taken on THIS timeline - the sequence a reader of FRED would have
    seen day by day - and never on the observation timeline, so a window of 252 means the last
    252 publications and a value never depends on a later one.
    """
    published = (
        vintages.filter(pl.col("series") == str(series_id).lower())
        .sort(["available_at", "timestamp"])
        .group_by("available_at", maintain_order=True)
        .agg(pl.col("value").last().alias("value"))
        .sort("available_at")
    )
    if published.is_empty():
        raise ValueError(f"no vintages for series {series_id!r} in the macro table")
    return published


def real_yields_dollar(
    panel: pl.DataFrame,
    vintages: pl.DataFrame,
    *,
    legs: Mapping[str, str],
    windows: Mapping,
) -> pl.DataFrame:
    """The *real yields and the dollar* family, joined on the PUBLICATION date.

    ``legs`` maps a leg name to a FRED series id (``real_yield: DFII10``, ``dollar: DTWEXBGS``
    in ``macro_config.yaml``). Per leg, on its own published timeline: the last published level
    (real yield only - the dollar index level is a scale, not a state), its change over the last
    21 publications (a log return for the dollar index) and its trailing z-score over
    ``windows.real_yield_zscore`` publications. Every statistic is trailing on the published
    series and is then joined ``asof backward`` on the decision DATE against ``available_at``, so
    a decision reads the last value published at or before its own date and never a later one.
    ``bots/assets/XAUUSD.md`` names real yields as the primary driver of both metals.
    """
    if "available_at" not in vintages.columns:
        raise ValueError("vintages carry no available_at; pass them through prepare_macro_vintages")
    z = int(windows.get("real_yield_zscore", MACRO_DEFAULT_ZSCORE_WINDOW))
    k = MACRO_DOLLAR_CHANGE_WINDOW
    grid = (
        panel.select(pl.col("timestamp").dt.date().alias("_decision_date"))
        .unique()
        .sort("_decision_date")
    )
    for leg, series_id in legs.items():
        published = _published_timeline(vintages, series_id)
        if leg == "dollar":
            level = pl.col("value").log()
            published = published.with_columns(
                (level - level.shift(k)).alias(f"macro_{leg}_ret_{k}d"),
                _trailing_zscore(level, z).alias(f"macro_{leg}_z_{z}d"),
            ).drop("value")
        else:
            published = published.with_columns(
                pl.col("value").alias(f"macro_{leg}"),
                (pl.col("value") - pl.col("value").shift(k)).alias(f"macro_{leg}_chg_{k}d"),
                _trailing_zscore(pl.col("value"), z).alias(f"macro_{leg}_z_{z}d"),
            ).drop("value")
        grid = grid.join_asof(
            published, left_on="_decision_date", right_on="available_at", strategy="backward"
        ).drop("available_at")
    return (
        panel.with_columns(pl.col("timestamp").dt.date().alias("_decision_date"))
        .join(grid, on="_decision_date", how="left")
        .drop("_decision_date")
    )


def prepare_releases(frame: pl.DataFrame, *, lag_days: int = 1) -> pl.DataFrame:
    """Validate a release calendar and add ``available_at = announced_on + lag_days``.

    A frame without ``announced_on`` is refused: a release date whose announcement date is
    unknown cannot be shown to have been knowable at a decision, which is exactly why a
    hand-typed table of dates was rejected in the register (``setup.yaml``, PLANNED rows).
    """
    missing = [c for c in RELEASE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"not a point-in-time release calendar (missing {missing}); a schedule with no "
            "announcement date is refused"
        )
    if int(lag_days) < 1:
        raise ValueError("a schedule's availability lag must be at least one day")
    return (
        frame.select(RELEASE_COLUMNS)
        .with_columns(
            pl.col("kind").cast(pl.String).str.to_lowercase(),
            pl.col("release_ts").cast(pl.Datetime("us")),
            pl.col("announced_on").cast(pl.Date),
        )
        .with_columns((pl.col("announced_on") + pl.duration(days=int(lag_days))).alias("available_at"))
        .sort(["kind", "release_ts"])
    )


def releases_known_by(releases: pl.DataFrame, on: date) -> pl.DataFrame:
    """Only the schedule entries a decision dated ``on`` could know: ``available_at <= on``."""
    if "available_at" not in releases.columns:
        raise ValueError("releases carry no available_at; pass them through prepare_releases")
    return releases.filter(pl.col("available_at") <= pl.lit(on))


def release_column_names(kind: str) -> list[str]:
    return [f"news_{kind}_decision_window", f"news_{kind}_label_window"]


def release_windows(
    panel: pl.DataFrame,
    releases: pl.DataFrame,
    *,
    window_minutes: int = RELEASE_WINDOW_MINUTES,
) -> pl.DataFrame:
    """The *CPI and FOMC release windows* family: two flags per release kind, knowable in advance.

    For every kind in ``releases`` (``cpi``, ``fomc`` when built from the FRED release
    calendar): ``news_<kind>_decision_window`` is 1.0 when a release falls within
    +-``window_minutes`` of the decision instant (the avoid window of ``bots/assets/XAUUSD.md``),
    ``news_<kind>_label_window`` is 1.0 when a release falls inside the hold - after the decision
    and at or before the session close - the same second flag the payroll rule produces. A
    schedule entry counts only if it was announced before the decision's date
    (``available_at <= decision date``); an entry announced later is invisible to that decision
    even though it is on the calendar, which is what makes the flag point-in-time rather than
    merely historical.
    """
    if "available_at" not in releases.columns:
        raise ValueError("releases carry no available_at; pass them through prepare_releases")
    needed = {"symbol", "timestamp", "session_close_ts"}
    if not needed <= set(panel.columns):
        raise KeyError(f"release_windows needs {sorted(needed)} on the panel (keep_context=True)")
    kinds = sorted(releases["kind"].unique().to_list())
    keys = panel.select("symbol", "timestamp", "session_close_ts").with_columns(
        pl.col("timestamp").dt.date().alias("_decision_date")
    )
    half = timedelta(minutes=int(window_minutes))
    out = keys
    for kind in kinds:
        rows = releases.filter(pl.col("kind") == kind).select("release_ts", "available_at")
        joined = (
            keys.join(rows, how="cross")
            .filter(pl.col("available_at") <= pl.col("_decision_date"))
            .with_columns(
                (
                    (pl.col("release_ts") >= pl.col("timestamp") - half)
                    & (pl.col("release_ts") <= pl.col("timestamp") + half)
                ).alias("_near"),
                (
                    (pl.col("release_ts") > pl.col("timestamp"))
                    & (pl.col("release_ts") <= pl.col("session_close_ts"))
                ).alias("_hold"),
            )
            .group_by(["symbol", "timestamp"])
            .agg(
                pl.col("_near").any().cast(pl.Float64).alias(f"news_{kind}_decision_window"),
                pl.col("_hold").any().cast(pl.Float64).alias(f"news_{kind}_label_window"),
            )
        )
        out = out.join(joined, on=["symbol", "timestamp"], how="left").with_columns(
            pl.col(f"news_{kind}_decision_window").fill_null(0.0),
            pl.col(f"news_{kind}_label_window").fill_null(0.0),
        )
    return panel.join(
        out.drop("session_close_ts", "_decision_date"), on=["symbol", "timestamp"], how="left"
    )
