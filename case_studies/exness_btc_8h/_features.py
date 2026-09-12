"""exness_btc_8h: the one decision grid, the one price panel, the one feature construction.

Every consumer imports from here rather than carrying a copy: `01_feasibility_analysis`,
`02_labels`, `03_financial_features`, `04_model_based_features`, `05_evaluation`, the
point-in-time test in `bots/exness_btc_8h/tests/` and, later, the backtest price loader and
step 2 of the deployment loop. A feature computed by two implementations is the two-pipeline
divergence Chapter 25 (section 25.1) warns about: the copies agree the day they are written and
drift on the first edit. Nothing here is a notebook; the notebooks explain and check what these
functions do.

Three things live in this module.

**The decision grid.** This bot decides three times a day, at 00:00, 08:00 and 16:00 UTC, every
day of the week. Unlike every other bot on this account the rule needs no venue, no session and
no calendar, because the instrument never stops quoting: the decision instant is simply the
CLOSE of an eight-hour bar. MetaTrader 5 serves no H8 timeframe, so the bars are folded from the
account's H4 bars by :func:`bots._shared.mt5_loader.resample_4h_to_8h`, which refuses a grid
whose bars do not open on a UTC hour divisible by four.

    the decision is the close of an 8-hour bar; the label ends at the close of the next one.

:func:`decision_grid` keys the panel on that **close** rather than on the bar's open, so the
`timestamp` column of every artifact IS the decision instant and the claim "this bot decides at
00:00 / 08:00 / 16:00 UTC" is literally true of the data. Two facts follow and both are
*asserted* rather than assumed, because a broker that shifted its bar grid would break them
silently:

- the set of decision hours is exactly ``{0, 8, 16}``;
- consecutive decisions are exactly eight hours apart on the development window, so
  ``fwd_ret_8h`` is a return between two adjacent rows and nothing has to be dropped for a
  holiday, an early close or a weekend. There are none.

**The price panel.** One row per decision instant:

- ``close`` is the decision bar's close - the price every feature is knowable at and the price
  ``02_labels`` seals the labels on;
- ``open``, ``high``, ``low`` and ``volume`` are the **decision bar's own**, i.e. the bar that
  ran ``t - 8h .. t``. So every column of a row dated ``t`` was knowable at ``t``, which is what
  makes the panel safe to build features on without any further filtering.
- the price the decision is EXECUTED at is therefore *not* on this row. ``execution_delay:
  next_bar_open`` fills at the open of the bar that starts at ``t``, which is the ``open`` of the
  row dated ``t + 8h``. On a 24/7 tape that instant IS ``t`` (``decision.execution_gap_minutes:
  0``), so the fill is simultaneous with the decision - but the two prices are a bid-ask tick
  apart, and ``02_labels`` measures the gap rather than assuming it away. Nothing here shifts a
  price backward to hide it.

**The feature construction, on two grids.** ``bot-portfolio-exness.md`` is the rule: a window
longer than the decision horizon is read from a coarser bar rather than from hundreds of fine
ones. So:

- *slot* families read the eight-hour decision bars. Their windows are counted in decision
  slots, three a day;
- *D1* families read daily bars and are joined **asof backward on the D1 bar's CLOSE instant**
  (00:00 UTC, because the server is UTC+0), never on its calendar date. For a decision at 08:00
  UTC on day D the newest usable D1 bar is day D-1's, which closed at 00:00 on day D. Joining on
  the date instead is the single most common lookahead in a bot that mixes grids, and it is the
  one thing in this module that is checked with an assertion inside the join itself
  (:func:`join_d1_asof`).

The three **cross-asset** families - gold, the dollar and the US index - sit on the D1 grid and
take the same join, which is what makes them point-in-time for free: XAUUSD, the five FX majors
and USTEC all stop quoting at the weekend while BTCUSD does not, so a backward asof join carries
a Friday close into a Saturday decision. That is exactly what a trader at that instant could see,
so it is correct rather than stale - and the age of the bar each row read is carried through as
``d1_close_ts`` so the point-in-time test can measure it instead of trusting this paragraph.

**What is deliberately absent.** Every cross-sectional mechanism of the templates. This is one
instrument: a percentile over one name is the constant 0.5 at every decision instant, which is
not stale but degenerate, and a cross-sectional information coefficient over one row is
undefined. ``setup.yaml::features.ranked`` is empty, ``mapping.class`` is
``per_asset_signal_timing``, and the statistic ``05_evaluation`` reads is the pooled panel IC of
:mod:`case_studies.exness_btc_8h._model_reading`, which on one name reduces to that name's own
Spearman correlation through time with a Newey-West interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
import polars as pl
from ml4t.engineer.features.momentum import rsi
from ml4t.engineer.features.volatility.garman_klass_volatility import garman_klass_volatility

from bots._shared.mt5_loader import load_mt5_bars
from case_studies.utils.feature_engineering import EPS, rolling_zscore

#: One decision slot. The panel is keyed on the bar CLOSE, so a row dated ``t`` was built from
#: the bar that ran ``t - SLOT_MINUTES .. t``.
SLOT_MINUTES = 8 * 60
#: The three UTC hours a decision may fall on. Asserted, never assumed.
DECISION_HOURS = (0, 8, 16)
#: MT5 serves no H8 timeframe; ``load_mt5_bars("8h")`` folds pairs of H4 bars.
SOURCE_FREQUENCY = "4h"

#: Cross-asset partners. Priced, never traded: none of them is in ``universe.symbols``, no label
#: is built on them and no order is ever placed in them (``setup.yaml::features.reference_symbols``).
GOLD_SYMBOL = "XAUUSD"
USIDX_SYMBOL = "USTEC"
#: The five dollar pairs of ``exness_fx_d1``, all in the same MT5 parquet. A pair quoting the
#: dollar second enters the signed proxy with its return negated.
DOLLAR_PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD")

BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
#: Columns that are inputs or evidence, never features. ``d1_close_ts`` is the audited proof that
#: the D1 join was backward (see :func:`join_d1_asof`); ``log_return`` is an intermediate whose
#: value dated ``t`` is the primary label dated one slot earlier.
EXCLUDED = {*BAR_COLUMNS, "d1_close_ts", "log_return", "bar_open_ts"}
#: Raw bar columns of a D1 frame. They are inputs to the D1 families, never features: a D1 close
#: is a price level, and a level joined onto a panel that already carries ``close`` arrives as
#: ``close_right`` and reaches the matrix as "yesterday's price" - the defect
#: ``exness_gold_sess/_features.py`` records against its own first version.
D1_RAW_COLUMNS = frozenset(
    {"open", "high", "low", "close", "volume", "spread", "real_volume", "server_time", "tick_volume"}
)
#: Trading days a year on the D1 grid of a 24/7 instrument. Used only to annualise the D1
#: volatility columns, so a number is needed and 365 is the honest one for this tape.
D1_PERIODS_PER_YEAR = 365


# ---------------------------------------------------------------------------
# The decision grid and the price panel
# ---------------------------------------------------------------------------
def decision_grid(
    bars: pl.DataFrame,
    *,
    expect_hours: Sequence[int] = DECISION_HOURS,
    slot_minutes: int = SLOT_MINUTES,
    verbose: bool = True,
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """Key the 8-hour bars on their CLOSE and check that the grid is the one declared.

    ``bars`` is the output of ``load_mt5_bars("8h", ...)``: one row per bar, ``timestamp`` at the
    bar's UTC **open**. The returned panel carries ``timestamp`` at the bar's **close**, which is
    the decision instant, plus ``bar_open_ts`` so a reader can see which bar a row came from.

    Three properties are checked here rather than in a notebook, because every stage calls this
    function and a notebook check protects only the notebook that runs it:

    1. every decision hour is in ``expect_hours``;
    2. no duplicate ``(symbol, timestamp)``;
    3. the gaps between consecutive decisions are reported. On this account the development
       window has none other than one slot (measured 2026-09-08: 9,371 consecutive slots from
       2018-02-16 with zero missing), but a gap is *reported* rather than asserted away, because
       a re-download that lost a week must be visible and not fatal to a diagnostic notebook.
    """
    if bars.is_empty():
        empty = bars.with_columns(pl.col("timestamp").alias("bar_open_ts"))
        return (empty, {"rows": 0, "gaps": []}) if return_report else empty
    frame = bars
    dtype = frame.schema["timestamp"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        frame = frame.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    panel = (
        frame.with_columns(
            pl.col("timestamp").alias("bar_open_ts"),
            (pl.col("timestamp") + pl.duration(minutes=slot_minutes)).alias("_close_ts"),
        )
        .drop("timestamp")
        .rename({"_close_ts": "timestamp"})
        .sort(["symbol", "timestamp"])
    )
    hours = sorted(panel["timestamp"].dt.hour().unique().to_list())
    assert set(hours) <= set(expect_hours), (
        f"decision instants fall on UTC hours {hours}; setup.yaml declares {list(expect_hours)}. "
        "The 8-hour grid is folded from H4 bars by mt5_loader.resample_4h_to_8h, so an hour "
        "outside this set means the account's H4 grid moved off the four-hour boundary."
    )
    assert panel.select(["symbol", "timestamp"]).is_duplicated().sum() == 0, (
        "two rows carry the same (symbol, decision instant)"
    )
    step = pl.duration(minutes=slot_minutes)
    gaps = (
        panel.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").shift(1).over("symbol")).alias("_d")
        )
        .filter(pl.col("_d").is_not_null() & (pl.col("_d") != step))
        .select("symbol", "timestamp", "_d")
    )
    report = {
        "rows": panel.height,
        "symbols": sorted(panel["symbol"].unique().to_list()),
        "first": panel["timestamp"].min(),
        "last": panel["timestamp"].max(),
        "decision_hours": hours,
        "n_gaps": gaps.height,
        "gaps": [
            (str(r["symbol"]), str(r["timestamp"]), int(r["_d"].total_seconds() // 60))
            for r in gaps.head(20).iter_rows(named=True)
        ],
    }
    if verbose:
        print(
            f"decision grid: {report['rows']:,} slots, {report['first']} -> {report['last']} UTC, "
            f"hours {report['decision_hours']}, {report['n_gaps']} gap(s) other than one slot"
        )
        for symbol, ts, minutes in report["gaps"]:
            print(f"    gap before {symbol} {ts}: {minutes} minutes")
    keep = ["symbol", "timestamp", "bar_open_ts", "open", "high", "low", "close", "volume"]
    # The broker's own per-bar spread, in points, when the caller asked load_mt5_bars for it.
    # It is the IN-SAMPLE cost record (setup.yaml::costs.spread_bps_in_sample) and it is NEVER a
    # feature: EXCLUDED does not list it because feature_columns() only sees the panel after
    # build_features, which does not carry it forward - 01_feasibility_analysis and 16_costs ask
    # for it explicitly and nothing else does.
    if "spread" in panel.columns:
        keep.append("spread")
    panel = panel.select(keep)
    return (panel, report) if return_report else panel


def load_decision_panel(
    setup: Mapping,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """``load_mt5_bars("8h")`` on the declared window, keyed on the decision instant."""
    symbols = list(symbols or sorted(setup["universe"]["symbols"]))
    start = start_date or str(setup["universe"]["history_start"])
    end = end_date or str(setup["evaluation"]["holdout_end"])
    bars = load_mt5_bars("8h", symbols=symbols, start_date=start, end_date=end).with_columns(
        pl.col("timestamp").dt.replace_time_zone(None)
    )
    return decision_grid(bars, verbose=verbose)


def load_d1(
    setup: Mapping,
    symbols: Sequence[str],
    *,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Daily bars from ``features.grid.long_window_history_start``, not from ``history_start``.

    The head start is the payoff of the two-grid design: a 315-D1-bar chain is already dense when
    the 8-hour panel opens, so no long-window family pushes ``universe.history_start`` forward.
    """
    start = str(setup["features"]["grid"].get("long_window_history_start", "2014-01-01"))
    end = end_date or str(setup["evaluation"]["holdout_end"])
    return load_mt5_bars("daily", symbols=list(symbols), start_date=start, end_date=end)


# ---------------------------------------------------------------------------
# Slot-grid families
# ---------------------------------------------------------------------------
def slot_momentum(panel: pl.DataFrame, windows: Mapping, slots_per_year: float) -> pl.DataFrame:
    """Trailing returns on the decision grid, their differences, and their Sharpe ratios."""
    close = pl.col("close")
    horizons = list(windows["prev_slots"])
    out = panel.with_columns(
        (close / close.shift(1).over("symbol").clip(lower_bound=EPS)).log().alias("log_return")
    ).with_columns(
        [
            (close / close.shift(k).over("symbol").clip(lower_bound=EPS) - 1).alias(f"slot_ret_{k}")
            for k in horizons
        ]
    )
    sharpe_windows = [k for k in horizons if k >= 21]
    out = out.with_columns(
        [
            (
                pl.col("log_return").rolling_mean(k).over("symbol")
                / pl.col("log_return").rolling_std(k).over("symbol").clip(lower_bound=EPS)
                * np.sqrt(slots_per_year)
            ).alias(f"slot_sharpe_{k}")
            for k in sharpe_windows
        ]
    )
    pairs = list(zip(horizons[:-1], horizons[1:], strict=False))
    return out.with_columns(
        [
            (pl.col(f"slot_ret_{a}") - pl.col(f"slot_ret_{b}")).alias(f"slot_accel_{a}_{b}")
            for a, b in pairs
        ]
    )


def slot_volatility(panel: pl.DataFrame, windows: Mapping, slots_per_year: float) -> pl.DataFrame:
    """Garman-Klass and close-to-close dispersion on the decision grid, and ratios between them."""
    gk = list(windows["garman_klass"])
    out = panel.with_columns(
        [
            garman_klass_volatility(
                "open", "high", "low", "close", period=w, trading_periods=int(slots_per_year)
            )
            .over("symbol")
            .alias(f"slot_vol_gk_{w}")
            for w in gk
        ]
        + [
            (pl.col("log_return").rolling_std(w).over("symbol") * np.sqrt(slots_per_year)).alias(
                f"slot_vol_cc_{w}"
            )
            for w in windows["close_to_close_volatility"]
        ]
    )
    ratios = list(zip(gk[:-1], gk[1:], strict=False))
    return out.with_columns(
        [
            (
                pl.col(f"slot_vol_gk_{a}") / pl.col(f"slot_vol_gk_{b}").clip(lower_bound=EPS)
            ).alias(f"slot_vol_ratio_{a}_{b}")
            for a, b in ratios
        ]
    )


def slot_mean_reversion(panel: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """Trailing z-scores of the multi-horizon slot returns, channel position and Bollinger %B."""
    close, bb = pl.col("close"), int(windows["bollinger"])
    z = int(windows["zscore"])
    hi = {w: close.rolling_max(w).over("symbol") for w in windows["channel"]}
    lo = {w: close.rolling_min(w).over("symbol") for w in windows["channel"]}
    mid = close.rolling_mean(bb).over("symbol")
    sd = close.rolling_std(bb).over("symbol")
    return panel.with_columns(
        [rolling_zscore(f"slot_ret_{h}", z, "symbol").alias(f"slot_zscore_{h}") for h in windows["zscore_horizons"]]
        + [
            ((close - lo[w]) / (hi[w] - lo[w]).clip(lower_bound=EPS)).alias(f"slot_channel_pos_{w}")
            for w in windows["channel"]
        ]
        + [
            pl.when(sd > 0)
            .then((close - (mid - 2 * sd)) / (4 * sd))
            .alias(f"slot_bollinger_pctb_{bb}")
        ]
    )


def slot_oscillator_and_range(panel: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """RSI, price against a moving average, drawdown below the trailing peak, mean bar range."""
    close = pl.col("close")
    rng = (pl.col("high") - pl.col("low")) / close.clip(lower_bound=EPS)
    return panel.with_columns(
        [rsi("close", period=p).over("symbol").alias(f"slot_rsi_{p}") for p in windows["rsi"]]
        + [
            (close / close.rolling_mean(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"slot_price_to_ma_{w}"
            )
            for w in windows["moving_average"]
        ]
        + [
            (close / close.rolling_max(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"slot_max_dd_{w}"
            )
            for w in windows["drawdown"]
        ]
        + [rng.rolling_mean(w).over("symbol").alias(f"slot_avg_range_{w}") for w in windows["range"]]
    )


# ---------------------------------------------------------------------------
# D1 families, including the three cross-asset ones
# ---------------------------------------------------------------------------
def d1_state(d1: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """Long-window state on the D1 grid, with the bar's CLOSE instant attached.

    ``d1`` carries ``symbol``, ``timestamp`` (the server trading day) and OHLC. The close instant
    is the day boundary that follows it - server midnight, 00:00 UTC, because the server clock is
    UTC+0 (``setup.yaml::decision.server_clock``, measured). Joining on this instant rather than
    on the date is the whole point-in-time content of the family.
    """
    close = pl.col("close")
    frame = d1.sort(["symbol", "timestamp"]).with_columns(
        (pl.col("timestamp").cast(pl.Datetime("us")) + pl.duration(days=1)).alias("d1_close_ts"),
        (close / close.shift(1).over("symbol").clip(lower_bound=EPS)).log().alias("_lr"),
    )
    frame = frame.with_columns(
        [
            (close / close.shift(w).over("symbol").clip(lower_bound=EPS) - 1).alias(f"d1_ret_{w}d")
            for w in windows["d1_momentum"]
        ]
        + [
            (close / close.rolling_mean(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"d1_price_to_ma_{w}d"
            )
            for w in windows["d1_moving_average"]
        ]
        + [
            (pl.col("_lr").rolling_std(w).over("symbol") * np.sqrt(D1_PERIODS_PER_YEAR)).alias(
                f"d1_vol_cc_{w}d"
            )
            for w in windows["d1_close_to_close_volatility"]
        ]
        + [
            garman_klass_volatility(
                "open", "high", "low", "close", period=w, trading_periods=D1_PERIODS_PER_YEAR
            )
            .over("symbol")
            .alias(f"d1_vol_gk_{w}d")
            for w in windows["d1_garman_klass"]
        ]
        + [rsi("close", period=p).over("symbol").alias(f"d1_rsi_{p}") for p in windows["d1_rsi"]]
        + [
            (close / close.rolling_max(w).over("symbol").clip(lower_bound=EPS) - 1).alias(
                f"d1_max_dd_{w}d"
            )
            for w in windows["d1_drawdown"]
        ]
    )
    z = int(windows["d1_zscore"])
    return frame.with_columns(
        [
            rolling_zscore(f"d1_ret_{h}d", z, "symbol").alias(f"d1_zscore_{h}d")
            for h in windows["d1_zscore_horizons"]
        ]
    ).drop("_lr")


def _partner_series(reference: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """``timestamp``, ``_px`` for one reference symbol's D1 closes, or an empty frame."""
    part = reference.filter(pl.col("symbol") == symbol).select(
        "timestamp", pl.col("close").alias("_px")
    )
    return part.sort("timestamp")


def _exposure_block(
    d1: pl.DataFrame,
    partner: pl.DataFrame,
    *,
    prefix: str,
    horizons: Sequence[int],
    beta_window: int,
) -> pl.DataFrame:
    """``{prefix}_ret_{h}d`` and ``{prefix}_beta_{w}d`` on the D1 grid.

    ``partner`` is ``timestamp`` + ``_px``. Its own return is measured on ITS grid, then carried
    onto this instrument's D1 grid by a backward fill of the *price* - a forward-filled price is
    point-in-time (it is the last print a trader could see), a forward-filled return would not
    be. The partners quote 5 days a week and this instrument quotes 7, so the fill is doing real
    work on every Saturday and Sunday row.

    The beta is this instrument's own D1 log return regressed on the partner's, over
    ``beta_window`` D1 bars: ``cov / var``, both trailing, so it is a rolling exposure and not a
    single full-sample number.
    """
    if partner.is_empty():
        cols = [pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_ret_{h}d") for h in horizons]
        cols.append(pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_beta_{beta_window}d"))
        return d1.with_columns(cols)
    dates = d1.select("timestamp").unique().sort("timestamp")
    aligned = (
        dates.join(partner, on="timestamp", how="left")
        .sort("timestamp")
        .with_columns(pl.col("_px").forward_fill())
    )
    aligned = aligned.with_columns(
        [
            (pl.col("_px") / pl.col("_px").shift(h).clip(lower_bound=EPS) - 1).alias(
                f"{prefix}_ret_{h}d"
            )
            for h in horizons
        ]
        + [(pl.col("_px") / pl.col("_px").shift(1).clip(lower_bound=EPS)).log().alias("_plr")]
    ).drop("_px")
    joined = d1.join(aligned, on="timestamp", how="left").sort(["symbol", "timestamp"])
    own = (pl.col("close") / pl.col("close").shift(1).over("symbol").clip(lower_bound=EPS)).log()
    return (
        joined.with_columns(own.alias("_olr"))
        .with_columns(
            (
                pl.rolling_cov(pl.col("_olr"), pl.col("_plr"), window_size=beta_window).over("symbol")
                / pl.col("_plr").rolling_var(beta_window).over("symbol").clip(lower_bound=EPS)
            ).alias(f"{prefix}_beta_{beta_window}d")
        )
        .drop(["_olr", "_plr"])
    )


def dollar_proxy(reference: pl.DataFrame, pairs: Sequence[str] = DOLLAR_PAIRS) -> pl.DataFrame:
    """A signed broad-dollar proxy from the five FX majors' D1 closes: ``timestamp``, ``_px``.

    The dollar is the first three letters of ``USDJPY`` and the last three of ``EURUSD``, so a
    pair quoting the dollar SECOND enters the average with its return negated. The signed daily
    average is then compounded into an index level, because :func:`_exposure_block` measures a
    partner's multi-horizon return from a price series and a factor has no price of its own.
    """
    present = [p for p in pairs if p in set(reference["symbol"].unique().to_list())]
    if not present:
        return pl.DataFrame(schema={"timestamp": reference.schema["timestamp"], "_px": pl.Float64})
    base = pl.col("symbol").str.head(3)
    signed = (
        reference.filter(pl.col("symbol").is_in(present))
        .sort(["symbol", "timestamp"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol").clip(lower_bound=EPS))
            .log()
            .alias("_lr")
        )
        .with_columns(
            pl.when(base == "USD").then(pl.col("_lr")).otherwise(-pl.col("_lr")).alias("_usd")
        )
        .group_by("timestamp")
        .agg(pl.col("_usd").mean().alias("_usd"), pl.len().alias("_n"))
        .sort("timestamp")
    )
    return signed.with_columns(
        pl.col("_usd").fill_null(0.0).cum_sum().exp().alias("_px")
    ).select("timestamp", "_px")


def cross_asset(d1: pl.DataFrame, reference: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """The three cross-asset families, all on the D1 grid: gold, the dollar, the US index."""
    frame = _exposure_block(
        d1,
        _partner_series(reference, GOLD_SYMBOL),
        prefix="gold",
        horizons=windows["gold"],
        beta_window=int(windows["gold_beta"]),
    )
    frame = _exposure_block(
        frame,
        dollar_proxy(reference),
        prefix="usd",
        horizons=windows["dollar"],
        beta_window=int(windows["dollar_beta"]),
    )
    return _exposure_block(
        frame,
        _partner_series(reference, USIDX_SYMBOL),
        prefix="usidx",
        horizons=windows["usidx"],
        beta_window=int(windows["usidx_beta"]),
    )


def join_d1_asof(panel: pl.DataFrame, d1_features: pl.DataFrame) -> pl.DataFrame:
    """Asof-backward join of every D1 family onto the decision grid, on the D1 bar's CLOSE.

    For a decision at 08:00 UTC on day D the newest usable D1 bar is day D-1's, which closed at
    00:00 UTC on day D. Joining on the bar's calendar *date* instead would hand the decision a
    bar that closes sixteen hours after it. The assertion at the end is what makes this a check
    and not a comment.

    ``d1_close_ts`` is carried onto the panel as audited evidence, excluded from the feature
    matrix by :data:`EXCLUDED`: the point-in-time test asserts ``d1_close_ts <= timestamp`` on
    every row and reads the bar's age off it (0 hours at the 00:00 decision, 8 at 08:00, 16 at
    16:00).
    """
    d1_cols = [
        c
        for c in d1_features.columns
        if c not in D1_RAW_COLUMNS and c not in {"symbol", "timestamp", "d1_close_ts"}
    ]
    right = (
        d1_features.select(["symbol", pl.col("d1_close_ts").alias("at"), *d1_cols])
        .with_columns(pl.col("at").alias("d1_close_ts"))
        .sort(["symbol", "at"])
    )
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
# Swap and calendar state
# ---------------------------------------------------------------------------
#: MT5 day of week (0 = Sunday .. 6 = Saturday) for ``symbol_info.swap_rollover3days``. Polars'
#: ``dt.weekday()`` is 1 = Monday .. 7 = Sunday, so the map is written out rather than guessed.
_PL_WEEKDAY_TO_MT5_DOW = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 0}


def swap_and_calendar_state(
    panel: pl.DataFrame,
    *,
    rollover3days_dow: int,
    slot_minutes: int = SLOT_MINUTES,
) -> pl.DataFrame:
    """State columns read off the decision instant alone: slot, weekday, and what it costs to hold.

    None of these is a signal - they say what *kind* of decision this is, and a model reads them
    beside a signal to learn that the signal behaves differently on the slot that pays the swap.
    Every value is knowable weeks in advance (a weekday and a UTC hour are not market data), so
    the family's ``lag`` is 0 and its ``lookback`` is 0.

    ``pays_swap_night`` is the one that earns the family its place. The server is UTC+0, so swap
    is charged at 00:00 UTC; a position opened at a decision instant and closed at the next one
    crosses that boundary if and only if the decision was the 16:00 one. Measured against
    :func:`bots._shared.costs_mt5.holding_cost_points` on the real ``symbol_info`` record:
    Sun 16:00 -> Mon 00:00 charges -1,638.6 points long, Mon 00:00 -> 08:00 charges 0.0.
    ``pays_triple_swap`` marks the nights whose ending server day is
    ``symbol_info.swap_rollover3days`` - **5, Friday**, on this symbol, where FX and metals on the
    same account report 3, Wednesday.

    The two columns are a deterministic function of ``slot_of_day`` on an unbroken grid, and that
    redundancy is declared in ``setup.yaml::features.families`` rather than hidden: F5's
    redundancy tree is what decides whether both survive, not the author.
    """
    exit_ts = pl.col("timestamp") + pl.duration(minutes=slot_minutes)
    crosses_midnight = exit_ts.dt.date() > pl.col("timestamp").dt.date()
    # The server day that ENDS at the midnight crossed, i.e. the decision's own date.
    ending_dow = pl.col("timestamp").dt.weekday().replace_strict(_PL_WEEKDAY_TO_MT5_DOW, default=None)
    return panel.with_columns(
        pl.col("timestamp").dt.hour().cast(pl.Float64).alias("slot_of_day"),
        pl.col("timestamp").dt.weekday().cast(pl.Float64).alias("dow"),
        (pl.col("timestamp").dt.weekday() >= 6).cast(pl.Float64).alias("is_weekend"),
        (pl.col("timestamp").dt.hour() == 16).cast(pl.Float64).alias("is_us_hours"),
        crosses_midnight.cast(pl.Float64).alias("pays_swap_night"),
        (crosses_midnight & (ending_dow == int(rollover3days_dow)))
        .cast(pl.Float64)
        .alias("pays_triple_swap"),
    )


def session_book_of(timestamps: pl.Series) -> pl.Series:
    """``no_swap_night`` / ``us_hours`` for decision instants, for a consumer holding only a timestamp.

    A TWO-WAY PARTITION of the decision grid, not three values: ``all`` is not a label this
    function returns, it is the union of both, which is what the ``__pooled__`` bucket of
    ``_model_reading.prediction_staleness_by_book`` already means and what a stage that skips
    ``session_filter`` entirely trades. ``setup.yaml::decision.session_filter_definitions``
    declares the same split in words: ``no_swap_night`` = {00:00, 08:00} UTC, ``us_hours`` =
    {16:00} UTC. Analogue of ``case_studies/exness_gold_sess/_features.py::session_of`` for a
    venue-session bot; this bot has one venue (the grid itself) and the two-way split is a
    property of the swap clock, not of a market's opening hours.

    ``13_backtest`` filters predictions on this (through the case-study-scoped resolver
    `case_studies/utils/backtest_runner.py::apply_session_filter` is expected to dispatch to -
    see the proposed diff in `bots/exness_btc_8h/BOT.md` Open questions) before it builds
    weights; a spec key alone changes the hash and not the result
    (`case_studies/utils/signals.py:537-560` ignores unknown keys), which is the trap
    `PHASE1_SPEC_MENTOR.md` section 2.a names on the sibling bot.
    """
    hours = timestamps.dt.replace_time_zone(None).dt.hour()
    unexpected = sorted(set(hours.to_list()) - {0, 8, 16})
    if unexpected:
        raise ValueError(
            f"session_book_of received decision hour(s) {unexpected} outside the declared "
            "{0, 8, 16} UTC grid (decision.snapshots_utc); the book cannot be resolved and must "
            "not be guessed"
        )
    return pl.Series(
        "session_book",
        ["us_hours" if h == 16 else "no_swap_night" for h in hours.to_list()],
        dtype=pl.Utf8,
    )


# ---------------------------------------------------------------------------
# The whole construction
# ---------------------------------------------------------------------------
def build_features(
    panel: pl.DataFrame,
    d1: pl.DataFrame,
    windows: Mapping,
    *,
    reference: pl.DataFrame | None = None,
    slots_per_year: float = 1095.0,
    rollover3days_dow: int = 5,
) -> pl.DataFrame:
    """Every family of ``setup.yaml::features.families`` that has data, on one panel.

    The last step converts NaN to null. Some library calls fill their warmup with NaN and others
    with null, and a NaN is a missing value dressed as a number: ``is_not_null`` reads it as
    present, so the warmup audit and the coverage figure would score an empty stretch as covered.
    """
    built = (
        panel.sort(["symbol", "timestamp"])
        .pipe(slot_momentum, windows, slots_per_year)
        .pipe(slot_volatility, windows, slots_per_year)
        .pipe(slot_mean_reversion, windows)
        .pipe(slot_oscillator_and_range, windows)
        .pipe(swap_and_calendar_state, rollover3days_dow=rollover3days_dow)
    )
    d1_features = d1_state(d1, windows)
    if reference is not None and not reference.is_empty():
        d1_features = cross_asset(d1_features, reference, windows)
    built = join_d1_asof(built, d1_features)
    return built.sort(["symbol", "timestamp"]).with_columns(
        pl.col(pl.Float32, pl.Float64).fill_nan(None)
    )


def feature_columns(built: pl.DataFrame) -> list[str]:
    return [c for c in built.columns if c not in EXCLUDED and not c.startswith("_")]


def warmup_expectations(windows: Mapping) -> dict[str, dict[str, int]]:
    """Longest chain of trailing bars each column reads, **keyed by the grid it counts on**.

    Two dictionaries, not one, and the split is the whole point of a two-grid design:

    * ``"decision slots"`` - columns built on the 8-hour panel. Their warmup is counted in rows
      of that panel, three a day, and
      :func:`case_studies.utils.feature_engineering.warmup_audit` can be run on it directly.
    * ``"D1 bars"`` - columns built on the D1 frame and asof-joined. Their warmup is counted in
      D1 bars and has to be audited **on the D1 frame**, before the join. Auditing them on the
      panel is meaningless and, worse, it raises: the D1 frame is read from 2014 while the panel
      opens in 2018, so ``d1_ret_252d`` is already dense at panel row 1 and an audit expecting
      252 rows of warmup would report a column "populated from fewer bars than its window spans".

    The state family (``slot_of_day``, ``dow``, ``is_*``, ``pays_*``) reads one instant and has
    no warmup at all; it is listed at 1 slot so the audit still proves the columns are not null
    everywhere.
    """
    horizons = list(windows["prev_slots"])
    slot: dict[str, int] = {f"slot_ret_{k}": k for k in horizons}
    slot |= {f"slot_sharpe_{k}": k + 1 for k in horizons if k >= 21}
    slot |= {
        f"slot_accel_{a}_{b}": max(a, b)
        for a, b in zip(horizons[:-1], horizons[1:], strict=False)
    }
    slot |= {f"slot_vol_gk_{w}": w for w in windows["garman_klass"]}
    slot |= {f"slot_vol_cc_{w}": w + 1 for w in windows["close_to_close_volatility"]}
    gk = list(windows["garman_klass"])
    slot |= {
        f"slot_vol_ratio_{a}_{b}": max(a, b) for a, b in zip(gk[:-1], gk[1:], strict=False)
    }
    slot |= {
        f"slot_zscore_{h}": int(windows["zscore"]) + h for h in windows["zscore_horizons"]
    }
    slot |= {f"slot_channel_pos_{w}": w for w in windows["channel"]}
    slot[f"slot_bollinger_pctb_{int(windows['bollinger'])}"] = int(windows["bollinger"])
    slot |= {f"slot_rsi_{p}": p for p in windows["rsi"]}
    slot |= {f"slot_price_to_ma_{w}": w for w in windows["moving_average"]}
    slot |= {f"slot_max_dd_{w}": w for w in windows["drawdown"]}
    slot |= {f"slot_avg_range_{w}": w for w in windows["range"]}
    slot |= dict.fromkeys(
        ["slot_of_day", "dow", "is_weekend", "is_us_hours", "pays_swap_night", "pays_triple_swap"],
        1,
    )
    daily: dict[str, int] = {f"d1_ret_{w}d": w for w in windows["d1_momentum"]}
    daily |= {f"d1_price_to_ma_{w}d": w for w in windows["d1_moving_average"]}
    daily |= {f"d1_vol_cc_{w}d": w + 1 for w in windows["d1_close_to_close_volatility"]}
    daily |= {f"d1_vol_gk_{w}d": w for w in windows["d1_garman_klass"]}
    daily |= {f"d1_rsi_{p}": p for p in windows["d1_rsi"]}
    daily |= {f"d1_max_dd_{w}d": w for w in windows["d1_drawdown"]}
    daily |= {
        f"d1_zscore_{h}d": int(windows["d1_zscore"]) + h for h in windows["d1_zscore_horizons"]
    }
    for prefix, key, beta_key in (
        ("gold", "gold", "gold_beta"),
        ("usd", "dollar", "dollar_beta"),
        ("usidx", "usidx", "usidx_beta"),
    ):
        daily |= {f"{prefix}_ret_{h}d": h for h in windows[key]}
        daily[f"{prefix}_beta_{int(windows[beta_key])}d"] = int(windows[beta_key]) + 1
    return {"decision slots": slot, "D1 bars": daily}


# ---------------------------------------------------------------------------
# Point-in-time harness: rebuild the panel from what had closed at an instant
# ---------------------------------------------------------------------------
def bars_closed_by(
    bars: pl.DataFrame, at: datetime, bar_minutes: int = SLOT_MINUTES
) -> pl.DataFrame:
    """Only the bars whose CLOSE is at or before ``at``. The definition of knowable.

    ``bars`` is a frame keyed on the bar's **open** (what ``load_mt5_bars`` returns). A frame
    already keyed on the close - the decision panel - must be filtered with ``bar_minutes=0``.
    """
    frame = bars
    dtype = frame.schema["timestamp"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        frame = frame.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    if isinstance(dtype, pl.Date):
        frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    return frame.filter(pl.col("timestamp") + pl.duration(minutes=bar_minutes) <= pl.lit(at))


def features_as_of(
    at: datetime,
    bars: pl.DataFrame,
    d1: pl.DataFrame,
    windows: Mapping,
    *,
    reference: pl.DataFrame | None = None,
    slots_per_year: float = 1095.0,
    rollover3days_dow: int = 5,
) -> pl.DataFrame:
    """Rebuild the feature row(s) dated ``at`` using only data that had closed by ``at``.

    ``bars`` are the 8-hour bars keyed on their OPEN (straight from ``load_mt5_bars("8h")``);
    ``d1`` and ``reference`` are daily frames. Every one of them is truncated on its own bar
    length, so the harness tests the grid arithmetic as well as the feature construction: a
    column that read a D1 bar closing after ``at`` would differ from the batch row here.

    The comparison in ``bots/exness_btc_8h/tests/test_lookahead.py`` is against the batch panel;
    identical values prove that no column read a bar printed after the decision. It is the same
    code path, not a reimplementation - which is the only kind of point-in-time test worth
    running. The caller must check that the result is non-empty: a comparison of two empty
    frames passes and proves nothing.
    """
    slot_pit = bars_closed_by(bars, at, bar_minutes=SLOT_MINUTES)
    d1_pit = bars_closed_by(
        d1.with_columns(pl.col("timestamp").cast(pl.Datetime("us"))), at, bar_minutes=24 * 60
    ).with_columns(pl.col("timestamp").cast(pl.Date))
    ref_pit = None
    if reference is not None:
        ref_pit = bars_closed_by(
            reference.with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            at,
            bar_minutes=24 * 60,
        ).with_columns(pl.col("timestamp").cast(pl.Date))
    panel = decision_grid(slot_pit, verbose=False)
    built = build_features(
        panel,
        d1_pit,
        windows,
        reference=ref_pit,
        slots_per_year=slots_per_year,
        rollover3days_dow=rollover3days_dow,
    )
    return built.filter(pl.col("timestamp") == pl.lit(at).cast(pl.Datetime("us")))


__all__ = [
    "BAR_COLUMNS",
    "DECISION_HOURS",
    "DOLLAR_PAIRS",
    "D1_PERIODS_PER_YEAR",
    "EXCLUDED",
    "GOLD_SYMBOL",
    "SLOT_MINUTES",
    "SOURCE_FREQUENCY",
    "USIDX_SYMBOL",
    "bars_closed_by",
    "build_features",
    "cross_asset",
    "d1_state",
    "decision_grid",
    "dollar_proxy",
    "feature_columns",
    "features_as_of",
    "join_d1_asof",
    "load_d1",
    "load_decision_panel",
    "slot_mean_reversion",
    "slot_momentum",
    "slot_oscillator_and_range",
    "slot_volatility",
    "swap_and_calendar_state",
    "warmup_expectations",
]
