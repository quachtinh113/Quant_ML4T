"""exness_usidx_sess: the one session panel and the one feature construction, shared by every consumer.

Research (`01_feasibility_analysis`, `02_labels`, `03_financial_features`,
`04_model_based_features`), the backtest price loader (`case_studies/utils/backtest_loaders.py`),
the point-in-time tests (`bots/exness_usidx_sess/tests/`) and, later, step 2 of the deployment
loop all import from here rather than carrying a copy. A feature computed by two implementations
is the two-pipeline divergence Chapter 25 (section 25.1) warns about: the copies agree on the day
they are written and drift on the first edit. Nothing here is a notebook; the notebooks explain
and check what these functions do.

Two things live in this module, as in ``case_studies/exness_fx_d1/_features.py``, which this is a
Route B fork of.

**The session grid and the two panels.** MT5 serves one-hour bars stamped at their UTC open on a
UTC+0 server (``setup.yaml::decision.server_clock``). This bot decides **twice** per NYSE cash
session, so there are two panels, one per label spec, and each is one row per index per session:

``intraday``
    decision at ``us_cash_open``: the close of the first H1 bar that closes at or after the NYSE
    open plus ``decision.open_delay_minutes`` (15:00 UTC in winter, 14:00 in summer). The order
    goes at that bar's successor's open and the position is closed at ``us_cash_close``.

``overnight``
    decision at ``us_cash_close``: the close of the last H1 bar that closes at or before the NYSE
    close (21:00 UTC winter, 20:00 summer, 18:00/17:00 on a half day). The order goes at that
    bar's successor's open and the position is closed at the next session's ``us_cash_open``
    decision instant.

Both instants come from the ``NYSE`` calendar at run time, never from a table: the cash session
follows New York DST and the half-days (13:00 New York) move it three hours. The **server clock**
does not follow DST (``server_clock.follows_dst_of: null``) but the **trading hours of the CFD**
do, which is a separate declaration (``decision.trade_hours_follow_dst_of``) that
``01_feasibility_analysis`` asserts on the bars on every run.

Every panel row carries three groups of columns and they must not be confused:

* ``open, high, low, close, volume`` are the **decision window**: every bar that closed after the
  previous decision instant and at or before this one. ``close`` is the decision bar's close, so
  every feature built from these columns is knowable at ``decision_ts`` - which is the property
  ``features_as_of`` exists to prove. This is the same rule ``exness_fx_d1`` applies to its H4
  bars, and it is why the panel's ``close`` is **not** the label's endpoint.
* ``exec_ts`` / ``exec_open`` are the price the order fills at: the open of the bar that opens at
  ``decision_ts`` (``decision.execution_delay: next_bar_open``). A session whose execution bar is
  missing gets nulls here and **no order is placed**; the rate is reported by
  ``01_feasibility_analysis`` and asserted by ``bots/exness_usidx_sess/tests/test_sessions.py``.
* ``label_ts`` / ``label_close`` are where the label ends. ``02_labels`` seals
  ``fwd_ret_<spec> = label_close / exec_open - 1`` on ``label_ts`` and writes nothing else.

The mentor's work order described a panel whose ``open`` is the fill price and whose ``close`` is
the label endpoint. That shape cannot carry the features: ``build_features`` reads ``close`` with
rolling windows that include the current row, so a ``close`` dated after the decision would put
the label's own endpoint inside every momentum column. The three groups above keep both, under
names that say which is which.

**The feature construction.** The families ``setup.yaml::features`` registers, built the way
``case_studies/exness_fx_d1`` builds its shared block, with the two changes this universe needs:
the cross-sectional rank family is gone (two instruments cannot be ranked - ``mapping.class:
time_series_threshold``), and two families are added that only an index session has: the
**overnight / intraday decomposition** (Ch08 01, ``08_financial_features/01_price_volume_features.py``)
and the **opening range** the decision waits out.

The decomposition is the one place the two specs differ, and the difference is knowability, not
arithmetic. ``intraday_ret(d)`` finishes at the cash close of session ``d``; the ``overnight``
spec decides at that instant and may read it, the ``intraday`` spec decides six hours earlier and
may not, so for that spec the column is shifted one session. ``overnight_ret(d)`` finishes at the
cash open and both specs may read it. Getting this wrong is invisible in the batch panel and is
exactly what ``features_as_of`` catches: rebuild from the bars that had closed at ``decision_ts``
and the unshifted column comes back null.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
from ml4t.diagnostic.splitters.calendar import TradingCalendar
from ml4t.engineer.features.momentum import rsi
from ml4t.engineer.features.volatility.garman_klass_volatility import garman_klass_volatility

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.utils.feature_engineering import (
    EPS,
    drawdown_block,
    momentum_volatility_block,
    rolling_zscore,
)
from utils.data_quality import apply_max_symbols

BAR_MINUTES = 60
SPECS = ("intraday", "overnight")
BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
WINDOW_COLUMNS = ["open", "high", "low", "close", "volume"]
# Realized volatility over the decision window and over the NYSE cash session, both summed
# from the H1 log returns while the panel is being aggregated. They are panel INPUTS, like
# session_open_px: what reaches the matrix is what session_vol_features makes of them.
REALIZED_VOL_COLUMNS = ["rv_window", "rv_cash"]
OPENING_RANGE_COLUMNS = ["or_open", "or_high", "or_low", "or_close"]
EXECUTION_COLUMNS = ["decision_ts", "exec_ts", "exec_open", "label_ts", "label_close"]
PANEL_COLUMNS = [
    "symbol",
    "timestamp",
    *WINDOW_COLUMNS,
    *REALIZED_VOL_COLUMNS,
    *OPENING_RANGE_COLUMNS,
    *EXECUTION_COLUMNS,
]
# The inputs the features are made of, and the intermediate log return. None of them reaches the
# feature matrix, and neither does anything that is dated after the decision instant.
EXCLUDED = {
    *BAR_COLUMNS,
    *REALIZED_VOL_COLUMNS,
    *OPENING_RANGE_COLUMNS,
    *EXECUTION_COLUMNS,
    "log_return",
    "session_open_px",
    "session_close_px",
}
# Every column built on the DAILY parquet carries this prefix, so that no name can be
# confused with the same statistic taken on the session grid and the register can claim the
# two blocks with separate patterns and separate `lag` declarations.
DAILY_PREFIX = "d1_"
# The USTEC / US500 ratio the `structural spread` family is built on.
SPREAD_NUMERATOR = "USTEC"
SPREAD_DENOMINATOR = "US500"


# ---------------------------------------------------------------------------
# The session grid
# ---------------------------------------------------------------------------
def _naive_utc(frame: pl.DataFrame, column: str = "timestamp") -> pl.DataFrame:
    dtype = frame.schema[column]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        frame = frame.with_columns(
            pl.col(column).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    return frame.with_columns(pl.col(column).cast(pl.Datetime("us")))


def session_grid(
    calendar: str,
    first: datetime,
    last: datetime,
    *,
    open_delay_minutes: int,
) -> pl.DataFrame:
    """One row per cash session between ``first`` and ``last`` (padded a week each side).

    Columns: ``session`` (the date the venue files the session under), ``open_at`` and
    ``close_at`` (the declared cash open and close as naive UTC instants) and
    ``open_target``, the instant the ``intraday`` decision bar must close at or after.

    Everything is read from the calendar, so New York DST and the 13:00 half-days are handled
    by ``pandas_market_calendars`` rather than by a table in this repository. The winter and
    summer figures in ``bots/assets/US500.md`` are a description of what this returns, not an
    input to it.
    """
    schedule = TradingCalendar(calendar).calendar.schedule(
        pd.Timestamp(first) - pd.Timedelta(days=7), pd.Timestamp(last) + pd.Timedelta(days=7)
    )
    grid = pl.DataFrame(
        {
            "session": pd.Series(schedule.index.date),
            "open_at": schedule["market_open"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .to_numpy(),
            "close_at": schedule["market_close"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .to_numpy(),
        }
    ).with_columns(
        pl.col("session").cast(pl.Date),
        pl.col("open_at").cast(pl.Datetime("us")),
        pl.col("close_at").cast(pl.Datetime("us")),
    )
    return grid.with_columns(
        (pl.col("open_at") + pl.duration(minutes=open_delay_minutes)).alias("open_target")
    ).sort("session")


def _decision_instants(
    bars: pl.DataFrame,
    grid: pl.DataFrame,
    *,
    spec: str,
    tolerance_minutes: int,
    verbose: bool,
) -> pl.DataFrame:
    """Per symbol and session, the two decision instants and the two session prices.

    ``open_decision_ts`` is the close of the first bar closing at or after ``open_target``;
    ``close_decision_ts`` the close of the last bar closing at or before ``close_at``. Both are
    found in the bars, never assumed: an instant whose bar is further than ``tolerance_minutes``
    from its target had no fresh price and is **nulled**, the same rule ``exness_fx_d1`` applies
    at 240 minutes for its H4 grid.

    Which of the two may be null and which may not is what ``spec`` decides, and getting it
    wrong breaks the point-in-time rebuild rather than the batch panel. The ``intraday`` panel
    decides at ``open_decision_ts`` and needs ``close_decision_ts`` only for its label endpoint;
    at the moment of that decision the endpoint has not printed yet, so requiring it would drop
    exactly the session ``features_as_of`` is asked to rebuild. A session whose REQUIRED instant
    is missing is dropped; a session whose other instant is missing keeps its features and
    carries no label.

    ``session_open_px`` is the open of the bar covering the cash open and ``session_close_px``
    the close of the ``close_decision`` bar; the overnight / intraday decomposition is built on
    those two.
    """
    closes = bars.select(
        "symbol", "bar_close", pl.col("open").alias("_o"), pl.col("close").alias("_c")
    ).sort("bar_close")
    opens = bars.select("symbol", "timestamp", pl.col("open").alias("_bar_open")).sort("timestamp")

    rows: list[pl.DataFrame] = []
    for symbol in sorted(bars["symbol"].unique().to_list()):
        sym_closes = closes.filter(pl.col("symbol") == symbol).drop("symbol")
        sym_opens = opens.filter(pl.col("symbol") == symbol).drop("symbol")
        part = (
            grid.sort("open_target")
            .join_asof(
                sym_closes.select("bar_close", pl.col("_c").alias("_open_dec_close")),
                left_on="open_target",
                right_on="bar_close",
                strategy="forward",
            )
            .rename({"bar_close": "open_decision_ts"})
            .sort("close_at")
            .join_asof(
                sym_closes.select(
                    "bar_close", pl.col("_c").alias("session_close_px")
                ),
                left_on="close_at",
                right_on="bar_close",
                strategy="backward",
            )
            .rename({"bar_close": "close_decision_ts"})
            # The bar covering the cash open: the last bar that OPENS at or before it. On an
            # H1 grid and a UTC+0 server that bar opens exactly 30 minutes before the NYSE
            # cash open (13:00 UTC against a 13:30 summer open, 14:00 against 14:30 in
            # winter), so `session_open_px` is the price THIRTY MINUTES BEFORE the auction,
            # not the session's first traded price. It is the cut point the overnight and
            # intraday legs are measured from, so the two legs still sum to the session, but
            # the cut sits half an hour early: `overnight_ret` is a PRE-OPEN gap and
            # `intraday_ret` carries the last half hour of pre-open with it. That is a
            # property of an H1 grid, not a choice, and it is declared rather than rounded
            # away - a finer grid would move it, and moving it would change every label.
            .sort("open_at")
            .join_asof(
                sym_opens.select("timestamp", pl.col("_bar_open").alias("session_open_px")),
                left_on="open_at",
                right_on="timestamp",
                strategy="backward",
            )
            .rename({"timestamp": "session_open_bar_ts"})
            .with_columns(pl.lit(symbol).alias("symbol"))
        )
        rows.append(part)
    stamped = pl.concat(rows, how="vertical").sort(["symbol", "session"])

    lag_open = (
        pl.col("open_decision_ts").dt.epoch("s") - pl.col("open_target").dt.epoch("s")
    ) // 60
    lag_close = (
        pl.col("close_at").dt.epoch("s") - pl.col("close_decision_ts").dt.epoch("s")
    ) // 60
    stamped = stamped.with_columns(
        lag_open.alias("open_lag_minutes"), lag_close.alias("close_lag_minutes")
    )
    assert (
        stamped.filter(pl.col("close_lag_minutes") < 0).height == 0
    ), "a close-decision bar closes after the declared cash close"
    assert (
        stamped.filter(pl.col("open_lag_minutes") < 0).height == 0
    ), "an open-decision bar closes before the opening-range window ends"

    ok_open = pl.col("open_lag_minutes").is_not_null() & (
        pl.col("open_lag_minutes") <= tolerance_minutes
    )
    # The close decision has to be a LATER bar than the open decision; on a truncated rebuild
    # both resolve to the last bar seen, and one bar cannot be two instants.
    ok_close = (
        pl.col("close_lag_minutes").is_not_null()
        & (pl.col("close_lag_minutes") <= tolerance_minutes)
        & (pl.col("close_decision_ts") > pl.col("open_decision_ts"))
    )
    stamped = stamped.with_columns(
        pl.when(ok_open).then(pl.col("open_decision_ts")).alias("open_decision_ts"),
        pl.when(ok_open).then(pl.col("_open_dec_close")).alias("_open_dec_close"),
        pl.when(ok_close).then(pl.col("close_decision_ts")).alias("close_decision_ts"),
        pl.when(ok_close).then(pl.col("session_close_px")).alias("session_close_px"),
    )
    required = "open_decision_ts" if spec == "intraday" else "close_decision_ts"
    complete = stamped.filter(pl.col(required).is_not_null())
    dropped = stamped.height - complete.height
    if dropped and verbose:
        print(
            f"{dropped} of {stamped.height} symbol-sessions dropped for want of a bar within "
            f"{tolerance_minutes} minutes of the {spec} decision instant"
        )
    return complete.drop("open_lag_minutes", "close_lag_minutes")


def _assign_windows(bars: pl.DataFrame, instants: pl.DataFrame, decision_col: str) -> pl.DataFrame:
    """Aggregate every bar into the decision window it closed in.

    A bar belongs to the session whose decision instant is the first at or after the bar's
    close, so every bar sits in exactly one window and a window is everything that printed
    between two consecutive decisions - the rule ``exness_fx_d1._features.session_panel``
    applies to its H4 bars, on this bot's grid instead.
    """
    parts: list[pl.DataFrame] = []
    for symbol in sorted(instants["symbol"].unique().to_list()):
        sym_bars = bars.filter(pl.col("symbol") == symbol).sort("bar_close").with_columns(
            pl.col("close").log().diff().alias("_lr")
        )
        sym_inst = (
            instants.filter(pl.col("symbol") == symbol)
            .select("session", decision_col)
            .drop_nulls()
            .sort(decision_col)
        )
        if sym_bars.is_empty() or sym_inst.is_empty():
            continue
        edges = sym_inst[decision_col].to_numpy().astype("datetime64[us]")
        bar_close = sym_bars["bar_close"].to_numpy().astype("datetime64[us]")
        idx = np.searchsorted(edges, bar_close, side="left")
        keep = idx < len(edges)
        sessions = sym_inst["session"].to_numpy().astype("datetime64[D]")
        assigned = np.full(len(idx), np.datetime64("NaT", "D"), dtype="datetime64[D]")
        assigned[keep] = sessions[idx[keep]]
        parts.append(
            sym_bars.with_columns(pl.Series("timestamp", assigned).cast(pl.Date))
            .drop_nulls("timestamp")
            .group_by(["symbol", "timestamp"], maintain_order=True)
            .agg(
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                # Realized volatility of the window: the root sum of squared hourly log
                # returns of everything that printed between the previous decision and this
                # one. The first return of a window is measured from the previous window's
                # last close, which is inside this window's own span, so the continuous
                # series and not a within-group difference is the right input.
                pl.col("_lr").pow(2).sum().sqrt().alias("rv_window"),
            )
        )
    if not parts:
        raise ValueError("no bars fall inside any decision window")
    return pl.concat(parts, how="vertical").sort(["symbol", "timestamp"])


def _cash_session_realized_vol(bars: pl.DataFrame, instants: pl.DataFrame) -> pl.DataFrame:
    """Per symbol and session, the realized volatility of the bars INSIDE the cash session.

    The window is ``(open_at, close_at]`` of the NYSE cash session, so the return that crosses
    the opening auction is deliberately outside it: that move is the overnight leg and the
    ``overnight and intraday`` family already carries it. The log returns are therefore taken
    within the session rather than from the continuous series.

    Sessions are disjoint and ordered the same way by both edges, so the bar-to-session
    assignment is one searchsorted on the closing edge plus a check against the opening one.
    """
    parts: list[pl.DataFrame] = []
    for symbol in sorted(instants["symbol"].unique().to_list()):
        sym_bars = bars.filter(pl.col("symbol") == symbol).sort("bar_close")
        sym_inst = (
            instants.filter(pl.col("symbol") == symbol)
            .select("session", "open_at", "close_at")
            .sort("open_at")
        )
        if sym_bars.is_empty() or sym_inst.is_empty():
            continue
        opens = sym_inst["open_at"].to_numpy().astype("datetime64[us]")
        closes = sym_inst["close_at"].to_numpy().astype("datetime64[us]")
        bar_close = sym_bars["bar_close"].to_numpy().astype("datetime64[us]")
        idx = np.searchsorted(closes, bar_close, side="left")
        inside = (idx < len(closes)) & (bar_close > opens[np.clip(idx, 0, len(opens) - 1)])
        sessions = sym_inst["session"].to_numpy().astype("datetime64[D]")
        assigned = np.full(len(idx), np.datetime64("NaT", "D"), dtype="datetime64[D]")
        assigned[inside] = sessions[idx[inside]]
        parts.append(
            sym_bars.with_columns(pl.Series("session", assigned).cast(pl.Date))
            .drop_nulls("session")
            .with_columns(
                pl.col("close").log().diff().over("session").alias("_lr_cash")
            )
            .group_by(["symbol", "session"], maintain_order=True)
            .agg(pl.col("_lr_cash").pow(2).sum().sqrt().alias("rv_cash"))
        )
    if not parts:
        return pl.DataFrame(
            schema={"symbol": pl.String, "session": pl.Date, "rv_cash": pl.Float64}
        )
    return pl.concat(parts, how="vertical")


def session_panel(
    bars: pl.DataFrame,
    *,
    spec: str,
    calendar: str,
    tolerance_minutes: int,
    open_delay_minutes: int,
    bar_minutes: int = BAR_MINUTES,
    keep_execution: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """One-hour bars (``timestamp`` = UTC bar open) -> one row per index and cash session.

    ``spec`` is ``"intraday"`` or ``"overnight"`` and picks which of the session's two decision
    instants this panel is built on. See the module docstring for the three column groups.
    """
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
    frame = _naive_utc(bars.select(BAR_COLUMNS)).with_columns(
        (pl.col("timestamp") + pl.duration(minutes=bar_minutes)).alias("bar_close")
    )
    if frame.is_empty():
        raise ValueError("no bars to aggregate")

    grid = session_grid(
        calendar,
        frame["timestamp"].min(),
        frame["bar_close"].max(),
        open_delay_minutes=open_delay_minutes,
    )
    instants = _decision_instants(
        frame, grid, spec=spec, tolerance_minutes=tolerance_minutes, verbose=verbose
    )
    decision_col = "open_decision_ts" if spec == "intraday" else "close_decision_ts"

    windows = _assign_windows(frame, instants, decision_col).join(
        _cash_session_realized_vol(frame, instants).rename({"session": "timestamp"}),
        on=["symbol", "timestamp"],
        how="left",
    )

    # The opening-range bar: the bar that covers the cash open. Its four prices are knowable at
    # both decision instants, so they ride on both panels unchanged.
    opening = frame.join(
        instants.select("symbol", "session", "session_open_bar_ts"),
        left_on=["symbol", "timestamp"],
        right_on=["symbol", "session_open_bar_ts"],
        how="inner",
    ).select(
        "symbol",
        pl.col("session").alias("timestamp"),
        pl.col("open").alias("or_open"),
        pl.col("high").alias("or_high"),
        pl.col("low").alias("or_low"),
        pl.col("close").alias("or_close"),
    )

    # Where the order fills, and where the label ends.
    if spec == "intraday":
        label_ts = pl.col("close_decision_ts")
        label_px = pl.col("session_close_px")
        ends = instants.select(
            "symbol",
            pl.col("session").alias("timestamp"),
            pl.col(decision_col).alias("decision_ts"),
            label_ts.alias("label_ts"),
            label_px.alias("label_close"),
            "session_open_px",
            "session_close_px",
        )
    else:
        # The overnight position is closed at the NEXT session's open decision, which is the
        # first instant at which the intraday spec would re-enter. Both are on the same grid,
        # so "next" is the next row of this symbol's own session series.
        ends = instants.sort(["symbol", "session"]).select(
            "symbol",
            pl.col("session").alias("timestamp"),
            pl.col(decision_col).alias("decision_ts"),
            pl.col("open_decision_ts").shift(-1).over("symbol").alias("label_ts"),
            pl.col("_open_dec_close").shift(-1).over("symbol").alias("label_close"),
            "session_open_px",
            "session_close_px",
        )

    # `decision.execution_delay: next_bar_open`: the first bar that OPENS at or after the
    # decision instant, and no more than `tolerance_minutes` after it. The tolerance is not
    # slack, it is the width of one bar: on this account the daily Globex break sits right on
    # the cash close in some regimes (in 2022 the break ran 20:00-22:00 UTC in summer and
    # 21:00-23:00 in winter, two hours; from 2023 it is one hour and starts an hour after the
    # cash close), and the index shuts for the weekend AT the Friday cash close. A session
    # whose next bar is on the far side of a break or of the weekend has no fill within a bar
    # of the decision, and the rule is then NO ORDER - never a fill hours later at a price the
    # decision did not see. The rate is reported here and by 01_feasibility_analysis.
    execution = (
        frame.select(
            "symbol", pl.col("timestamp").alias("exec_ts"), pl.col("open").alias("exec_open")
        )
        .sort("exec_ts")
    )
    panel = (
        windows.join(opening, on=["symbol", "timestamp"], how="left")
        .join(ends, on=["symbol", "timestamp"], how="left")
        .sort("decision_ts")
        .join_asof(
            execution,
            left_on="decision_ts",
            right_on="exec_ts",
            by="symbol",
            strategy="forward",
            tolerance=timedelta(minutes=tolerance_minutes),
        )
        .sort(["symbol", "timestamp"])
    )
    missing_exec = panel.filter(pl.col("decision_ts").is_not_null() & pl.col("exec_open").is_null())
    if missing_exec.height and verbose:
        print(
            f"{missing_exec.height} of {panel.height} symbol-sessions have no bar opening "
            f"within {tolerance_minutes} minutes of the decision instant: no order is placed "
            "on those (setup.yaml::decision.execution_delay; bots/exness_usidx_sess/BOT.md)"
        )
    columns = (
        PANEL_COLUMNS
        if keep_execution
        else ["symbol", "timestamp", *WINDOW_COLUMNS, *REALIZED_VOL_COLUMNS]
    )
    return panel.select([*columns, "session_open_px", "session_close_px"]).sort(
        ["symbol", "timestamp"]
    )


def load_session_panel(
    setup: Mapping,
    *,
    spec: str,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
    keep_execution: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """``load_mt5_bars("1h")`` for the declared universe, aggregated by :func:`session_panel`.

    ``start_date`` defaults to ``universe.history_start`` (2022-10-25, the first H1 bar this
    account serves for either index); ``symbols`` to ``universe.symbols``. The MT5 parquet
    carries every instrument of every bot, so the universe is always selected explicitly.
    """
    symbols = list(symbols) if symbols is not None else sorted(setup["universe"]["symbols"])
    bars = load_mt5_bars(
        "1h",
        symbols=symbols,
        start_date=start_date or str(setup["universe"]["history_start"]),
        end_date=end_date,
    )
    decision = setup["decision"]
    panel = session_panel(
        bars,
        spec=spec,
        calendar=decision["session_calendar"],
        tolerance_minutes=int(decision["session_close_tolerance_minutes"]),
        open_delay_minutes=int(decision["open_delay_minutes"]),
        keep_execution=keep_execution,
        verbose=verbose,
    )
    return apply_max_symbols(panel, max_symbols)


# ---------------------------------------------------------------------------
# Feature families, one function each; setup.yaml::features.families is the register
# ---------------------------------------------------------------------------
def momentum_features(df: pl.DataFrame, windows: Mapping, periods_per_year: float) -> pl.DataFrame:
    """The shared trailing block, plus the two differences this case study builds on it."""
    df = momentum_volatility_block(
        df,
        entity="symbol",
        return_windows=windows["momentum"],
        volatility_windows=windows["close_to_close_volatility"],
        periods_per_year=periods_per_year,
    ).rename({f"vol_{w}d": f"vol_cc_{w}d" for w in windows["close_to_close_volatility"]})
    held = pl.col("close").shift(windows["skip_recent"]).over("symbol")
    start = pl.col("close").shift(windows["momentum"][-1]).over("symbol")
    short, mid, long = windows["momentum"][1], windows["momentum"][3], windows["momentum"][-1]
    # A Sharpe ratio over one session divides a return by the standard deviation of one
    # observation. The shared block emits one for every declared return window, and this case
    # study declares a one-session return because the single session IS the horizon it trades;
    # the ratio built on it is not a quantity. It is dropped by name so that the register's
    # `sharpe_*d` pattern claims only the windows that mean something, rather than left as an
    # all-null column for the null policy to delete the whole matrix on.
    degenerate = [f"sharpe_{w}d" for w in windows["momentum"] if w < 2]
    df = df.drop(degenerate, strict=False)
    return df.with_columns(
        (held / start.clip(lower_bound=EPS) - 1).alias("mom_skip_recent"),
        (pl.col(f"ret_{short}d") - pl.col(f"ret_{mid}d")).alias(f"accel_{short}_{mid}"),
        (pl.col(f"ret_{mid}d") - pl.col(f"ret_{long}d")).alias(f"accel_{mid}_{long}"),
    )


def volatility_features(df: pl.DataFrame, windows: Mapping, periods_per_year: float) -> pl.DataFrame:
    """Garman-Klass deviation at the declared windows, and the ratio between them."""
    short, long = windows["garman_klass"][0], windows["garman_klass"][-1]
    return df.with_columns(
        garman_klass_volatility(
            "open", "high", "low", "close", period=w, trading_periods=periods_per_year
        )
        .over("symbol")
        .alias(f"vol_gk_{w}d")
        for w in windows["garman_klass"]
    ).with_columns(
        (pl.col(f"vol_gk_{short}d") / pl.col(f"vol_gk_{long}d").clip(lower_bound=EPS)).alias(
            f"vol_ratio_{short}_{long}"
        )
    )


def mean_reversion_features(df: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """Trailing z-scores of the multi-horizon returns, and two range positions."""
    bb, close = windows["bollinger"], pl.col("close")
    hi = {w: close.rolling_max(w).over("symbol") for w in windows["channel"]}
    lo = {w: close.rolling_min(w).over("symbol") for w in windows["channel"]}
    mid = close.rolling_mean(bb).over("symbol")
    sd = close.rolling_std(bb).over("symbol")
    return df.with_columns(
        *[
            rolling_zscore(f"ret_{h}d", windows["zscore"], "symbol").alias(f"zscore_{h}d")
            for h in windows["zscore_horizons"]
        ],
        *[
            ((close - lo[w]) / (hi[w] - lo[w]).clip(lower_bound=EPS)).alias(f"channel_pos_{w}d")
            for w in windows["channel"]
        ],
        pl.when(sd > 0).then((close - (mid - 2 * sd)) / (4 * sd)).alias(f"bollinger_pctb_{bb}d"),
    )


def drawdown_range_and_oscillators(df: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """Distance below the trailing peak, normalized range, RSI and trend ratios."""
    close = pl.col("close")
    rng = (pl.col("high") - pl.col("low")) / close.clip(lower_bound=EPS)
    return drawdown_block(df, entity="symbol", windows=windows["drawdown"]).with_columns(
        *[rng.rolling_mean(w).over("symbol").alias(f"avg_range_{w}d") for w in windows["range"]],
        *[rsi("close", period=p).over("symbol").alias(f"rsi_{p}d") for p in windows["rsi"]],
        *[
            (close / close.rolling_mean(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"price_to_ma_{w}d"
            )
            for w in windows["moving_average"]
        ],
    )


def opening_range_features(df: pl.DataFrame) -> pl.DataFrame:
    """The return and the range of the H1 bar that straddles the cash open.

    On this grid that bar runs from 30 minutes before the NYSE cash open to 30 minutes after
    it, so ``or_ret`` is half pre-open drift and half opening auction rather than the first
    hour of the session. It is named for what it is used as - the state the auction left - and
    the register says which sixty minutes it reads.

    That bar has already printed at both decision instants (the ``intraday`` decision is taken
    at the close of the bar after it), so it is knowable on both panels without a shift.
    """
    return df.with_columns(
        (pl.col("or_close") / pl.col("or_open").clip(lower_bound=EPS) - 1).alias("or_ret"),
        ((pl.col("or_high") - pl.col("or_low")) / pl.col("or_close").clip(lower_bound=EPS)).alias(
            "or_range"
        ),
    )


def overnight_intraday_features(df: pl.DataFrame, windows: Mapping, spec: str) -> pl.DataFrame:
    """The two legs of an index session, and their trailing sums (Ch08 01).

    ``overnight_ret(d) = session_open_px(d) / session_close_px(d-1) - 1`` finishes THIRTY MINUTES
    BEFORE the cash open, because ``session_open_px`` is the open of the H1 bar straddling it
    (see :func:`_decision_instants`); it is a pre-open gap rather than an open-to-open one, and
    it is knowable at both decision instants. ``intraday_ret(d) = session_close_px(d) /
    session_open_px(d) - 1`` therefore carries that last half hour of pre-open with it, and
    finishes at the cash close: the ``overnight`` spec decides there
    and may read it, the ``intraday`` spec decides earlier and may not, so on that panel the
    column is shifted one session before anything is built on it. ``gap`` is the overnight leg
    measured against the previous session's own dispersion, which is what makes a two-point
    move on a quiet index and on a violent one comparable.
    """
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
    prev_close = pl.col("session_close_px").shift(1).over("symbol")
    overnight = pl.col("session_open_px") / prev_close.clip(lower_bound=EPS) - 1
    intraday = pl.col("session_close_px") / pl.col("session_open_px").clip(lower_bound=EPS) - 1
    df = df.with_columns(overnight.alias("_overnight"), intraday.alias("_intraday"))
    if spec == "intraday":
        df = df.with_columns(pl.col("_intraday").shift(1).over("symbol").alias("_intraday"))
    out = df.with_columns(
        pl.col("_overnight").alias("overnight_ret_1d"),
        pl.col("_intraday").alias("intraday_ret_1d"),
    )
    for h in windows["overnight_intraday"]:
        if h == 1:
            continue
        out = out.with_columns(
            pl.col("_overnight").rolling_sum(h).over("symbol").alias(f"overnight_ret_{h}d"),
            pl.col("_intraday").rolling_sum(h).over("symbol").alias(f"intraday_ret_{h}d"),
        )
    dispersion = pl.col("_intraday").rolling_std(windows["overnight_intraday"][-1]).over("symbol")
    return out.with_columns(
        (pl.col("_overnight") / dispersion.clip(lower_bound=EPS)).alias("gap_z")
    ).drop("_overnight", "_intraday")


def structural_spread(df: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """The USTEC / US500 ratio at three horizons, and the rolling correlation of the two.

    With two instruments this is the whole cross-asset block: there is no factor to estimate,
    only the one relative price the pair defines (Ch08 03, ``bots/assets/USTEC.md`` section 7).
    Both legs are the decision-window close, so the ratio is knowable at the decision instant.
    """
    wide = (
        df.filter(pl.col("symbol").is_in([SPREAD_NUMERATOR, SPREAD_DENOMINATOR]))
        .select("timestamp", "symbol", "close")
        .pivot(on="symbol", index="timestamp", values="close")
        .sort("timestamp")
    )
    if SPREAD_NUMERATOR not in wide.columns or SPREAD_DENOMINATOR not in wide.columns:
        # A run restricted to one symbol (MAX_SYMBOLS) has no spread; the family is absent
        # rather than filled with nulls, which is what the null policy would otherwise drop.
        return df
    ratio = pl.col(SPREAD_NUMERATOR) / pl.col(SPREAD_DENOMINATOR).clip(lower_bound=EPS)
    wide = wide.with_columns(ratio.alias("_ratio"))
    wide = wide.with_columns(
        (pl.col("_ratio") / pl.col("_ratio").shift(h) - 1).alias(f"spread_ret_{h}d")
        for h in windows["structural"]
    ).select("timestamp", *[f"spread_ret_{h}d" for h in windows["structural"]])
    return (
        df.join(wide, on="timestamp", how="left")
        .sort(["symbol", "timestamp"])
        .with_columns(
            pl.rolling_corr(
                pl.col("ret_1d"),
                pl.col(f"spread_ret_{windows['structural'][0]}d"),
                window_size=windows["structural_exposure"],
            )
            .over("symbol")
            .alias("spread_corr_1d")
        )
    )


def session_vol_features(df: pl.DataFrame, windows: Mapping, spec: str) -> pl.DataFrame:
    """Realized volatility over the decision window and inside the cash session (Ch08 s8.2).

    ``rv_window`` is the root sum of squared one-hour log returns over everything that printed
    between the previous decision and this one, so it is the dispersion the position just lived
    through and it is complete at the decision by construction. ``rv_cash`` is the same sum
    taken over the bars *inside* the NYSE cash session, which is a different quantity: the cash
    session is where the volume is, and on the ``overnight`` panel the decision is taken at its
    close, so the whole of it is knowable. On the ``intraday`` panel the decision is taken six
    hours earlier and today's cash session has barely started, so the column is shifted one
    session - exactly the asymmetry :func:`overnight_intraday_features` handles for
    ``intraday_ret``, and exactly what ``features_as_of`` catches if it is got wrong.

    The raw ``rv_window`` and ``rv_cash`` columns are panel inputs and stay in ``EXCLUDED``; what
    reaches the matrix is the current value under an explicit ``_1d`` name, the trailing means
    over the declared windows, and the ratio of the current window to its trailing month, which
    reads the regime rather than the level.
    """
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
    if spec == "intraday":
        df = df.with_columns(pl.col("rv_cash").shift(1).over("symbol").alias("rv_cash"))
    out = df.with_columns(
        pl.col("rv_window").alias("rv_window_1d"), pl.col("rv_cash").alias("rv_cash_1d")
    )
    for w in windows["realized_vol"]:
        out = out.with_columns(
            pl.col("rv_window").rolling_mean(w).over("symbol").alias(f"rv_window_{w}d"),
            pl.col("rv_cash").rolling_mean(w).over("symbol").alias(f"rv_cash_{w}d"),
        )
    longest = windows["realized_vol"][-1]
    return out.with_columns(
        (
            pl.col("rv_window") / pl.col(f"rv_window_{longest}d").clip(lower_bound=EPS)
        ).alias(f"rv_ratio_1_{longest}")
    )


def calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Seven columns that are functions of the session date and of nothing else.

    Every one of them is pure date arithmetic, which is what makes the family safe: it reads no
    price, no later session and - deliberately - no count of the sessions the month turned out
    to hold. A "position of this session among its month's sessions" column would have to know
    how many sessions the month ends up with, and on a panel truncated at a decision instant
    that count is smaller than it is in the batch build; the column would then take two
    different values for one session, which is a lookahead wearing a calendar's clothes.
    ``month_progress`` uses the day of the month over the length of the month instead, which the
    Gregorian calendar fixes years in advance.

    The weekday and the month enter as sine and cosine pairs rather than as integers or dummies:
    a linear model reading ``weekday = 5`` as five times ``weekday = 1`` is reading an ordering
    that is not there, and five dummies on a two-instrument panel spend degrees of freedom this
    sample cannot afford. ``is_opex`` is the third Friday of the month, the monthly index-option
    expiry (``bots/assets/US500.md``): a Friday whose day of the month falls between the
    fifteenth and the twenty-first is the third Friday, with no calendar lookup needed.
    """
    ts = pl.col("timestamp")
    weekday, month = ts.dt.weekday(), ts.dt.month()
    day, days_in_month = ts.dt.day(), ts.dt.month_end().dt.day()
    return df.with_columns(
        (2 * np.pi * (weekday - 1) / 5).sin().alias("dow_sin"),
        (2 * np.pi * (weekday - 1) / 5).cos().alias("dow_cos"),
        (2 * np.pi * (month - 1) / 12).sin().alias("month_sin"),
        (2 * np.pi * (month - 1) / 12).cos().alias("month_cos"),
        ((weekday == 5) & day.is_between(15, 21)).cast(pl.Float64).alias("is_opex"),
        ((day <= 3) | (day >= days_in_month - 2)).cast(pl.Float64).alias("turn_of_month"),
        ((day - 1) / (days_in_month - 1).clip(lower_bound=1)).alias("month_progress"),
    )


def daily_features(
    daily: pl.DataFrame, daily_windows: Mapping, periods_per_year: float
) -> pl.DataFrame:
    """The long-window families, computed on the D1 grid instead of this bot's session grid.

    The H1 history this account serves starts 2022-10-25 and leaves 96 sessions of warmup before
    the earliest fold, so nothing built on the session grid may look back further than 63
    sessions. A quarter is short for a trend feature and a year is not reachable at all. The D1
    file starts 2019-07-16 and holds 2,191 bars per index, which warms a 252-bar window more than
    three years before the panel opens - so the long windows come from there, and they come with
    a declared staleness rather than with a promise.

    Saturday and Sunday rows are dropped first. A Sunday D1 bar on this UTC+0 server holds only
    the two evening hours 22:00 and 23:00 UTC, which is the week's re-open and not a trading day;
    leaving it in would put a two-hour bar into a 252-bar window as though it were a session.
    Close-to-close returns over the Monday-to-Friday subset lose nothing by it, because a return
    reads two levels and the Sunday evening move is already inside Monday's close.

    Every produced column is prefixed ``d1_`` so that no name here can be confused with the
    same statistic taken on the session grid, and so that the register can claim the two blocks
    with separate patterns and separate ``lag`` declarations.
    """
    frame = (
        daily.select(BAR_COLUMNS)
        .filter(pl.col("timestamp").dt.weekday() <= 5)
        .unique(subset=["symbol", "timestamp"], keep="last")
        .sort(["symbol", "timestamp"])
    )
    if frame.is_empty():
        raise ValueError("no daily bars to build the long-window families from")
    frame = momentum_volatility_block(
        frame,
        entity="symbol",
        return_windows=daily_windows["momentum"],
        volatility_windows=daily_windows["volatility"],
        periods_per_year=periods_per_year,
    )
    frame = drawdown_block(frame, entity="symbol", windows=daily_windows["drawdown"])
    close = pl.col("close")
    short, long = daily_windows["volatility"][0], daily_windows["volatility"][-1]
    fast, slow = daily_windows["momentum"][1], daily_windows["momentum"][-1]
    hi = {w: close.rolling_max(w).over("symbol") for w in daily_windows["channel"]}
    lo = {w: close.rolling_min(w).over("symbol") for w in daily_windows["channel"]}
    frame = frame.with_columns(
        *[
            (close / close.rolling_mean(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"price_to_ma_{w}d"
            )
            for w in daily_windows["moving_average"]
        ],
        *[
            ((close - lo[w]) / (hi[w] - lo[w]).clip(lower_bound=EPS)).alias(f"channel_pos_{w}d")
            for w in daily_windows["channel"]
        ],
        (pl.col(f"vol_{short}d") / pl.col(f"vol_{long}d").clip(lower_bound=EPS)).alias(
            f"vol_ratio_{short}_{long}"
        ),
        (pl.col(f"ret_{fast}d") - pl.col(f"ret_{slow}d")).alias(f"accel_{fast}_{slow}"),
    )
    produced = [c for c in frame.columns if c not in {*BAR_COLUMNS, "log_return"}]
    return frame.select(
        "symbol", "timestamp", *[pl.col(c).alias(f"{DAILY_PREFIX}{c}") for c in produced]
    )


def daily_factor(
    df: pl.DataFrame, daily: pl.DataFrame, daily_windows: Mapping, periods_per_year: float
) -> pl.DataFrame:
    """Join :func:`daily_features` onto the session grid at the last D1 bar that had CLOSED.

    This is the one join in the construction that crosses two frequencies, so the rule it applies
    is the whole point of it. A D1 bar dated ``d`` on this server is exactly the 00:00-24:00 UTC
    aggregate of that day's H1 bars - verified on 26 sessions of US500 in June 2025, where the
    open, high, low and close of the daily bar and of the hourly aggregate agree to 1e-6 - so it
    has finished printing at 00:00 UTC on ``d + 1`` and not a minute earlier. Every decision
    instant of this bot falls between 14:00 and 21:00 UTC, so the freshest bar a decision on day
    ``d`` may read is dated ``d - 1``, and the join is an as-of join on that availability instant
    rather than on the date. Joining on the date instead would hand each session a bar containing
    the very hours it is trying to forecast.

    The staleness is declared, not hidden: every family this builds carries ``lag: 1`` in the
    register, ``03_financial_features`` asserts that no joined bar is dated on or after its own
    decision day, and ``features_as_of`` rebuilds the join from truncated daily bars.
    """
    features = (
        daily_features(daily, daily_windows, periods_per_year)
        .with_columns(
            (pl.col("timestamp").cast(pl.Datetime("us")) + pl.duration(days=1)).alias(
                "available_from"
            )
        )
        .drop("timestamp")
        .sort("available_from")
    )
    return (
        df.sort("decision_ts")
        .join_asof(
            features,
            left_on="decision_ts",
            right_on="available_from",
            by="symbol",
            strategy="backward",
        )
        .drop("available_from")
        .sort(["symbol", "timestamp"])
    )


def load_daily_bars(
    setup: Mapping,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """``load_mt5_bars("daily")`` for the declared universe, from ``daily_history_start``.

    ``end_date`` is the caller's responsibility and matters: the daily file runs past the holdout
    boundary like every other price file, and a development-only build has to say so.
    """
    symbols = list(symbols) if symbols is not None else sorted(setup["universe"]["symbols"])
    return load_mt5_bars(
        "daily",
        symbols=symbols,
        start_date=start_date or str(setup["universe"]["daily_history_start"]),
        end_date=end_date,
    )


def build_features(
    prices: pl.DataFrame,
    *,
    spec: str,
    windows: Mapping,
    daily_windows: Mapping,
    ranked: Sequence[str],
    periods_per_year: float,
    daily: pl.DataFrame,
) -> pl.DataFrame:
    """The whole construction, as one function every consumer calls.

    ``ranked`` is accepted and must be empty: two instruments carry no cross-section to rank
    within (``setup.yaml::mapping.class: time_series_threshold``), and a rank over two rows is
    a coin flip dressed as a feature. Passing a non-empty list raises rather than producing one.

    The last step converts NaN to null. Some library calls fill their warmup with NaN and
    others with null, and a NaN is a missing value dressed as a number: ``is_not_null`` reads it
    as present, so the warmup audit and the coverage figure would score an empty stretch as
    covered.
    """
    if list(ranked):
        raise ValueError(
            f"{list(ranked)} were declared as cross-sectional ranks, but this case study has "
            "two instruments and mapping.class time_series_threshold; there is nothing to rank"
        )
    # `decision_ts` is kept through the construction because the daily join needs the instant
    # rather than the session date, and it is in EXCLUDED so it never reaches the matrix.
    keep = [
        c
        for c in prices.columns
        if c
        in {
            *BAR_COLUMNS,
            *REALIZED_VOL_COLUMNS,
            *OPENING_RANGE_COLUMNS,
            "decision_ts",
            "session_open_px",
            "session_close_px",
        }
    ]
    df = (
        prices.select(keep)
        .sort(["symbol", "timestamp"])
        .pipe(momentum_features, windows, periods_per_year)
        .pipe(volatility_features, windows, periods_per_year)
        .pipe(mean_reversion_features, windows)
        .pipe(drawdown_range_and_oscillators, windows)
        .pipe(opening_range_features)
        .pipe(overnight_intraday_features, windows, spec)
        .pipe(session_vol_features, windows, spec)
        .pipe(structural_spread, windows)
        .pipe(calendar_features)
        .pipe(daily_factor, daily, daily_windows, periods_per_year)
    )
    return df.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))


def feature_columns(built: pl.DataFrame) -> list[str]:
    return [c for c in built.columns if c not in EXCLUDED]


def warmup_expectations(windows: Mapping, spec: str = "overnight") -> dict[str, int]:
    """The declared warmup, per audited column, from the window register and the spec.

    ``spec`` matters for one family. ``overnight_ret`` is built on a one-session shift of
    ``session_close_px`` and so always costs one row more than its window; ``intraday_ret`` is
    built inside one session and costs exactly its window - except on the ``intraday`` panel,
    where it is shifted a session because the decision is taken before it resolves (see
    :func:`overnight_intraday_features`). One row of difference, and it is the difference
    between an honest feature and a leak.
    """
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, not {spec!r}")
    intraday_shift = 1 if spec == "intraday" else 0
    return {
        f"zscore_{windows['zscore_horizons'][-1]}d": windows["zscore"]
        + windows["zscore_horizons"][-1],
        f"ret_{windows['momentum'][-1]}d": windows["momentum"][-1],
        f"sharpe_{windows['momentum'][-1]}d": windows["momentum"][-1],
        f"vol_gk_{windows['garman_klass'][-1]}d": windows["garman_klass"][-1],
        f"price_to_ma_{windows['moving_average'][-1]}d": windows["moving_average"][-1],
        f"channel_pos_{windows['channel'][-1]}d": windows["channel"][-1],
        f"max_dd_{windows['drawdown'][0]}d": windows["drawdown"][0],
        f"rsi_{windows['rsi'][0]}d": windows["rsi"][0],
        f"overnight_ret_{windows['overnight_intraday'][-1]}d": windows["overnight_intraday"][-1]
        + 1,
        f"intraday_ret_{windows['overnight_intraday'][-1]}d": windows["overnight_intraday"][-1]
        + intraday_shift,
        f"spread_ret_{windows['structural'][-1]}d": windows["structural"][-1],
        "spread_corr_1d": windows["structural_exposure"] + windows["structural"][0],
        # rv_window is complete on the panel's first row; rv_cash costs one row more on the
        # intraday panel, where it is shifted a session because the decision is taken before
        # the cash session it measures has finished.
        f"rv_window_{windows['realized_vol'][-1]}d": windows["realized_vol"][-1],
        f"rv_cash_{windows['realized_vol'][-1]}d": windows["realized_vol"][-1]
        + intraday_shift,
    }


# ---------------------------------------------------------------------------
# Point-in-time recomputation
# ---------------------------------------------------------------------------
def bars_closed_by(
    bars: pl.DataFrame, decision_ts: datetime, bar_minutes: int = BAR_MINUTES
) -> pl.DataFrame:
    """The bars a trader at ``decision_ts`` (naive UTC) could have seen: those that had closed."""
    ts_cmp = decision_ts.replace(tzinfo=None) if decision_ts.tzinfo is not None else decision_ts
    return _naive_utc(bars).filter(
        pl.col("timestamp") + pl.duration(minutes=bar_minutes) <= pl.lit(ts_cmp)
    )


def daily_bars_closed_by(daily: pl.DataFrame, decision_ts: datetime) -> pl.DataFrame:
    """The D1 bars that had finished printing at ``decision_ts``.

    A D1 bar dated ``d`` covers 00:00-24:00 UTC of ``d`` on this server, so it is complete at
    00:00 UTC on ``d + 1``; the bar dated on the decision's own day is still open and is
    withheld here, which is what makes the point-in-time rebuild a test of the daily join and
    not only of the hourly one.
    """
    cutoff = pl.lit(decision_ts).cast(pl.Datetime("us"))
    return daily.filter(
        (pl.col("timestamp").cast(pl.Datetime("us")) + pl.duration(days=1)) <= cutoff
    )


def features_as_of(
    bars: pl.DataFrame,
    decision_ts: datetime,
    setup: Mapping,
    *,
    spec: str,
    daily_bars: pl.DataFrame,
) -> pl.DataFrame:
    """Rebuild the feature panel from the bars that had closed at ``decision_ts``.

    The last session of the result is the one decided at ``decision_ts``; its feature row is
    what the batch panel must reproduce for that session, and ``test_lookahead.py`` asserts
    that it does. The gap between the two would be information from after the decision.
    """
    decision = setup["decision"]
    prices = session_panel(
        bars_closed_by(bars, decision_ts),
        spec=spec,
        calendar=decision["session_calendar"],
        tolerance_minutes=int(decision["session_close_tolerance_minutes"]),
        open_delay_minutes=int(decision["open_delay_minutes"]),
        verbose=False,
    )
    features = setup["features"]
    return build_features(
        prices,
        spec=spec,
        windows=features["windows"],
        daily_windows=features["daily_windows"],
        ranked=features.get("ranked") or [],
        periods_per_year=setup["evaluation"]["periods_per_year"],
        daily=daily_bars_closed_by(daily_bars, decision_ts),
    )


EVENING_HOURS_UTC = (19, 20, 21, 22, 23)


def daily_break_by_month(
    bars: pl.DataFrame,
    symbol: str,
    *,
    zone: str = "America/New_York",
    quiet_share: float = 0.2,
    search_hours: Sequence[int] = EVENING_HOURS_UTC,
) -> pl.DataFrame:
    """Per calendar month, the hours of the day this symbol does **not** print a bar in.

    The evidence behind ``decision.trade_hours_follow_dst_of``. Only Monday to Thursday are
    counted, so the Friday close and the Sunday open - which are week boundaries, not the daily
    break - cannot be mistaken for it. An hour is part of the break when it carries fewer than
    ``quiet_share`` of the bars the median hour of that month carries.

    Returns ``month``, ``is_dst`` (of ``zone`` on the fifteenth), ``break_utc`` and ``break_ny``
    (the same hours in ``zone``'s local time) and ``break_end_ny``, the hour the break ends in
    ``zone``'s local time. That last column is the invariant: the daily break of these index
    CFDs ends at 18:00 New York in every month of this sample, in both daylight-saving regimes
    and across the 2023 change from a two-hour break to a one-hour one. The UTC hour it falls on
    moves by one between summer and winter, which is exactly the claim
    ``decision.trade_hours_follow_dst_of`` makes and the reason a hard-coded 21:00 or 22:00 is
    wrong for half the year.

    The search is restricted to ``search_hours`` (19:00-23:00 UTC by default). That window is a
    declared search space, not a declared answer: the daily break of a US index on a UTC+0
    server sits in the evening in both regimes, and looking outside it only picks up hours that
    a holiday or a thin week happened to leave quiet.

    ``has_dst_transition`` marks a month in which ``zone`` changes offset. Such a month contains
    both regimes at once, so its quiet hours are a mixture and mean nothing; the caller exempts
    it rather than guessing. A month whose quiet set is empty comes back with a null
    ``break_end_ny`` and is exempt for the same reason.
    """
    frame = (
        _naive_utc(bars)
        .filter((pl.col("symbol") == symbol) & (pl.col("timestamp").dt.weekday() <= 4))
        .with_columns(
            pl.col("timestamp").dt.truncate("1mo").alias("month"),
            pl.col("timestamp").dt.hour().alias("hour"),
        )
    )
    rows: list[dict] = []
    for (month,), part in frame.group_by("month", maintain_order=True):
        counts = dict(
            zip(
                part.group_by("hour").agg(pl.len().alias("n"))["hour"].to_list(),
                part.group_by("hour").agg(pl.len().alias("n"))["n"].to_list(),
                strict=True,
            )
        )
        if not counts:
            continue
        ordered = sorted(counts.values())
        median = ordered[len(ordered) // 2]
        break_utc = [h for h in sorted(search_hours) if counts.get(h, 0) < quiet_share * median]
        mid = pd.Timestamp(month.year, month.month, 15, tz=zone)
        offset_hours = int(mid.utcoffset().total_seconds() // 3600)
        first = pd.Timestamp(month.year, month.month, 1, tz=zone)
        last = first + pd.offsets.MonthEnd(0)
        break_ny = sorted((h + offset_hours) % 24 for h in break_utc)
        rows.append(
            {
                "month": month,
                "is_dst": bool(mid.dst()),
                "has_dst_transition": first.utcoffset() != last.utcoffset(),
                "break_utc": break_utc,
                "break_ny": break_ny,
                "break_end_ny": (max(break_ny) + 1) if break_ny else None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "month": pl.Datetime("us"),
            "is_dst": pl.Boolean,
            "has_dst_transition": pl.Boolean,
            "break_utc": pl.List(pl.Int64),
            "break_ny": pl.List(pl.Int64),
            "break_end_ny": pl.Int64,
        },
    ).sort("month")


__all__ = [
    "BAR_COLUMNS",
    "BAR_MINUTES",
    "DAILY_PREFIX",
    "EXCLUDED",
    "EXECUTION_COLUMNS",
    "OPENING_RANGE_COLUMNS",
    "PANEL_COLUMNS",
    "REALIZED_VOL_COLUMNS",
    "SPECS",
    "SPREAD_DENOMINATOR",
    "SPREAD_NUMERATOR",
    "WINDOW_COLUMNS",
    "bars_closed_by",
    "build_features",
    "calendar_features",
    "daily_bars_closed_by",
    "daily_factor",
    "daily_features",
    "drawdown_range_and_oscillators",
    "EVENING_HOURS_UTC",
    "daily_break_by_month",
    "feature_columns",
    "features_as_of",
    "load_daily_bars",
    "load_session_panel",
    "mean_reversion_features",
    "momentum_features",
    "opening_range_features",
    "overnight_intraday_features",
    "session_grid",
    "session_panel",
    "session_vol_features",
    "structural_spread",
    "volatility_features",
    "warmup_expectations",
]
