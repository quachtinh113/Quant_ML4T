"""exness_fx_d1: the one price panel and the one feature construction, shared by every consumer.

Research (`03_financial_features`, `04_model_based_features`), the backtest price loader
(`case_studies/utils/backtest_loaders.py`), the point-in-time test
(`bots/exness_fx_d1/tests/test_lookahead.py`) and, later, step 2 of the deployment loop all
import from here rather than carrying a copy. A feature computed by two implementations is the
two-pipeline divergence Chapter 25 (section 25.1) warns about: the copies agree on the day they
are written and drift on the first edit. Nothing here is a notebook; the notebooks explain and
check what these functions do.

Two things live in this module.

**The session panel.** MT5 serves four-hour bars stamped at their UTC open on a UTC+0 server
(``setup.yaml::decision.server_clock``). The strategy decides once per ``CME_FX`` session, at the
close of the last bar that closes at or before the session's declared close (``decision.snapshot:
h4_close_2000utc``, the 16:00-20:00 UTC bar), which is the rule ``01_feasibility_analysis``
checks and ``02_labels.to_daily_sessions`` applies to the close. :func:`session_panel` applies the
same rule and adds the other four bar fields:

- a bar belongs to the session whose declared close is the first at or after the bar's *close*,
  so every bar sits in exactly one session and a session is everything that printed between two
  consecutive decision instants. The 20:00-24:00 UTC bar, which straddles the 5PM rollover,
  therefore opens the *next* session: that session's ``open`` is the price the previous decision
  was executed at (``decision.execution_delay: next_bar_open``), and its high and low cover the
  thin post-New York hours a by-open assignment would leave in no session at all;
- ``close`` is the close of the session's last bar, the decision bar, and is identical to the
  close ``02_labels`` sealed the labels on (asserted by ``test_lookahead.py`` against the label
  files); ``decision_ts`` is that bar's close instant, the moment every feature is knowable;
- a pair-session whose last bar closes more than ``decision.session_close_tolerance_minutes``
  ahead of the declared close had no fresh price at the decision (a server outage, 2018-01-31
  on every pair) and leaves the panel, reported rather than decided on a stale bar.

**The feature construction.** The families ``setup.yaml::features`` registers, built exactly as
``case_studies/fx_pairs/03_financial_features`` builds them, with three changes this universe
needs: the dollar proxy reads the bare MT5 symbol names (``EURUSD``, not ``EUR_USD``); a gold
factor is added from ``XAUUSD`` on the same decision bar, the cross-instrument feature
``bots/assets/AUDUSD.md`` names; and a **carry** family is available from the policy-rate
differential of the two legs of each pair, the effect every asset profile names first. Every
statistic is trailing within one pair or taken within one decision date; nothing is fitted
across the sample, which is what :func:`features_as_of` lets a test prove by rebuilding the
panel from the bars that had closed at a decision instant.

**Carry is the one family whose input is not a bar.** A price is knowable the moment it prints;
a policy rate is knowable only when the statistical agency publishes it, and for the four
monthly legs of this universe that is weeks after the month the rate describes (Ch08 s8.4).
:func:`load_carry_vintages` therefore reads the ALFRED *initial release* table, which carries a
``vintage_date`` per observation, and :func:`published_rates` joins it backward onto the session
grid on ``available_at = vintage_date + availability_lag_days``. The aligned macro panel is
refused outright, because it is forward-filled on the observation date and would date a monthly
rate up to eight weeks before anyone could see it. The ids, the lag and the sign convention are
declared in ``bots/_shared/macro_config.yaml``; the family is only built when
``setup.yaml::features.windows`` declares a ``carry`` window, so a bot without the data is not
silently given a column of nulls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from ml4t.diagnostic.splitters.calendar import TradingCalendar
from ml4t.engineer.features.momentum import rsi
from ml4t.engineer.features.volatility.garman_klass_volatility import garman_klass_volatility

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir
from case_studies.utils.feature_engineering import (
    EPS,
    cross_sectional_percentile,
    drawdown_block,
    momentum_volatility_block,
    rolling_zscore,
)
from utils.data_quality import apply_max_symbols
from utils.paths import REPO_ROOT

BAR_MINUTES = 240
GOLD_SYMBOL = "XAUUSD"
BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
PANEL_COLUMNS = [*BAR_COLUMNS, "decision_ts"]
# The one place the carry series ids, the sign convention and the availability lag are declared.
MACRO_CONFIG_PATH = REPO_ROOT / "bots" / "_shared" / "macro_config.yaml"
VINTAGE_COLUMNS = ["series", "timestamp", "vintage_date", "value"]
# The inputs the features are made of, and the intermediate log return, which dated t is the
# primary label dated one session earlier. None of them reaches the feature matrix.
EXCLUDED = {*BAR_COLUMNS, "decision_ts", "log_return"}


# ---------------------------------------------------------------------------
# The session panel
# ---------------------------------------------------------------------------
def declared_session_closes(calendar: TradingCalendar, first: datetime, last: datetime) -> pl.DataFrame:
    """Every session between ``first`` and ``last`` (padded a week each side), with its declared close.

    ``session`` is the calendar date the venue files the session under, ``close_at`` the declared
    close as a naive UTC instant. Both sides of every comparison below are naive UTC.
    """
    schedule = calendar.calendar.schedule(
        pd.Timestamp(first) - pd.Timedelta(days=7), pd.Timestamp(last) + pd.Timedelta(days=7)
    )
    return (
        pl.DataFrame(
            {
                "session": pd.Series(schedule.index.date),
                "close_at": schedule["market_close"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
            }
        )
        .with_columns(pl.col("session").cast(pl.Date), pl.col("close_at").cast(pl.Datetime("us")))
        .sort("close_at")
    )


def session_panel(
    bars: pl.DataFrame,
    *,
    calendar: str,
    tolerance_minutes: int,
    bar_minutes: int = BAR_MINUTES,
    keep_decision_ts: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Four-hour bars (``timestamp`` = UTC bar open) -> one OHLCV row per pair and session.

    See the module docstring for the rule. Raises when a bar closes after the session close it
    was assigned to (impossible by construction, so it is a check on the construction) and when
    more than one percent of pair-sessions lack a fresh bar (a clock or calendar problem, not an
    outage). Returns ``symbol, timestamp (pl.Date), open, high, low, close, volume`` plus
    ``decision_ts`` when asked for, sorted by pair and session.
    """
    frame = bars.select(BAR_COLUMNS)
    ts_dtype = frame.schema["timestamp"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        frame = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us"))).with_columns(
        (pl.col("timestamp") + pl.duration(minutes=bar_minutes)).alias("bar_close")
    )
    if frame.is_empty():
        raise ValueError("no bars to aggregate")

    closes = declared_session_closes(
        TradingCalendar(calendar), frame["timestamp"].min(), frame["bar_close"].max()
    )
    close_at = closes["close_at"].to_numpy().astype("datetime64[us]")
    bar_close = frame["bar_close"].to_numpy().astype("datetime64[us]")
    # The session whose declared close is the first at or after the bar's close. A bar closing
    # after the last declared close has no session and is dropped (the padding above makes that
    # only the bars past the end of the calendar).
    idx = np.searchsorted(close_at, bar_close, side="left")
    in_range = idx < len(close_at)
    session = np.full(len(idx), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    session_dates = closes["session"].to_numpy().astype("datetime64[D]")
    session[in_range] = session_dates[idx[in_range]]
    stamped = (
        frame.with_columns(pl.Series("timestamp_session", session).cast(pl.Date))
        .drop_nulls("timestamp_session")
        .join(closes.rename({"session": "timestamp_session"}), on="timestamp_session", how="left")
    )
    late = stamped.filter(pl.col("bar_close") > pl.col("close_at"))
    if late.height:
        raise AssertionError(f"{late.height} bars close after the session close they were assigned to")

    grouped = (
        stamped.sort(["symbol", "bar_close"])
        .group_by(["symbol", "timestamp_session"], maintain_order=True)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("bar_close").last().alias("decision_ts"),
            pl.col("close_at").first().alias("close_at"),
        )
        .rename({"timestamp_session": "timestamp"})
        .with_columns(
            ((pl.col("close_at").dt.epoch("s") - pl.col("decision_ts").dt.epoch("s")) // 60).alias(
                "lag_minutes"
            )
        )
    )
    assert grouped["lag_minutes"].min() >= 0, "a decision bar closes after the declared session close"
    stale = grouped.filter(pl.col("lag_minutes") > tolerance_minutes)
    if stale.height > 0.01 * grouped.height:
        raise AssertionError(
            f"{stale.height} of {grouped.height} pair-sessions lack a bar within {tolerance_minutes} "
            "minutes of the declared close: a clock or calendar problem, not an outage"
        )
    if stale.height and verbose:
        dropped = sorted({str(d) for d in stale["timestamp"].to_list()})
        print(f"{stale.height} pair-sessions dropped for want of a bar within tolerance: {dropped}")
    panel = (
        grouped.filter(pl.col("lag_minutes") <= tolerance_minutes)
        .select(PANEL_COLUMNS)
        .sort(["symbol", "timestamp"])
    )
    return panel if keep_decision_ts else panel.drop("decision_ts")


def load_session_panel(
    setup: Mapping,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
    keep_decision_ts: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """``load_mt5_bars("4h")`` for the declared universe, aggregated by :func:`session_panel`.

    ``start_date`` defaults to ``universe.history_start``, the first server day on which every
    pair carries the full six-bar grid; ``symbols`` to ``universe.symbols``. The MT5 parquet
    carries every instrument of the four bots, so the universe is always selected explicitly.
    """
    symbols = list(symbols) if symbols is not None else sorted(setup["universe"]["symbols"])
    bars = load_mt5_bars(
        "4h",
        symbols=symbols,
        start_date=start_date or str(setup["universe"]["history_start"]),
        end_date=end_date,
    )
    panel = session_panel(
        bars,
        calendar=setup["decision"]["session_calendar"],
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        keep_decision_ts=keep_decision_ts,
        verbose=verbose,
    )
    return apply_max_symbols(panel, max_symbols)


# ---------------------------------------------------------------------------
# Feature families, one function each; setup.yaml::features.families is the register
# ---------------------------------------------------------------------------
def momentum_features(df: pl.DataFrame, windows: Mapping, periods_per_year: float) -> pl.DataFrame:
    """The shared trailing block, plus the differences this case study builds on it."""
    df = momentum_volatility_block(
        df,
        entity="symbol",
        return_windows=windows["momentum"],
        volatility_windows=windows["close_to_close_volatility"],
        periods_per_year=periods_per_year,
    ).rename({f"vol_{w}d": f"vol_cc_{w}d" for w in windows["close_to_close_volatility"]})
    held = pl.col("close").shift(windows["skip_recent"]).over("symbol")
    start = pl.col("close").shift(windows["momentum"][-1]).over("symbol")
    return df.with_columns(
        (held / start.clip(lower_bound=EPS) - 1).alias("mom_skip_recent"),
        (pl.col("ret_21d") - pl.col("ret_63d")).alias("accel_21_63"),
        (pl.col("ret_63d") - pl.col("ret_126d")).alias("accel_63_126"),
    )


def volatility_features(df: pl.DataFrame, windows: Mapping, periods_per_year: float) -> pl.DataFrame:
    """Garman-Klass deviation at four windows, and the two ratios between them."""
    short, mid, long = (windows["garman_klass"][i] for i in (0, 1, -1))
    return df.with_columns(
        garman_klass_volatility("open", "high", "low", "close", period=w, trading_periods=periods_per_year)
        .over("symbol")
        .alias(f"vol_gk_{w}d")
        for w in windows["garman_klass"]
    ).with_columns(
        (pl.col(f"vol_gk_{short}d") / pl.col(f"vol_gk_{mid}d").clip(lower_bound=EPS)).alias(
            f"vol_ratio_{short}_{mid}"
        ),
        (pl.col(f"vol_gk_{mid}d") / pl.col(f"vol_gk_{long}d").clip(lower_bound=EPS)).alias(
            f"vol_ratio_{mid}_{long}"
        ),
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
            (close / close.rolling_mean(w).over("symbol").clip(lower_bound=EPS) - 1).alias(f"price_to_ma_{w}d")
            for w in windows["moving_average"]
        ],
    )


def dollar_factor(df: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """The signed dollar proxy at three horizons, and each pair's correlation with it.

    Bare MT5 names: the dollar is the first three letters of ``USDJPY`` and the last three of
    ``EURUSD``, and a pair quoting the dollar second enters the average with its return negated.
    """
    base, quote = pl.col("symbol").str.head(3), pl.col("symbol").str.tail(3)
    signed = (
        df.filter((base == "USD") | (quote == "USD"))
        .with_columns(
            pl.when(base == "USD").then(pl.col("log_return")).otherwise(-pl.col("log_return")).alias("_usd")
        )
        .group_by("timestamp")
        .agg(pl.col("_usd").mean().alias("usd_factor_1d"))
        .sort("timestamp")
    )
    signed = signed.with_columns(
        pl.col("usd_factor_1d").rolling_sum(h).alias(f"usd_factor_{h}d") for h in windows["dollar"]
    )
    return (
        df.join(signed, on="timestamp", how="left")
        .sort(["symbol", "timestamp"])
        .with_columns(
            pl.rolling_corr(pl.col(f"ret_{h}d"), pl.col(f"usd_factor_{h}d"), window_size=windows["dollar_exposure"])
            .over("symbol")
            .alias(f"usd_corr_{h}d")
            for h in windows["dollar"]
        )
    )


def gold_factor(df: pl.DataFrame, gold: pl.DataFrame, windows: Mapping) -> pl.DataFrame:
    """Gold's trailing return at two horizons, and each pair's rolling correlation with it.

    ``gold`` is the ``XAUUSD`` session panel on the same decision bar. Gold keeps its own hours
    (a daily break and metal holidays), so its close is carried onto the pairs' session grid as
    the last print at or before each decision instant, which is what a trader at that instant
    could see; the return is then measured on that aligned series. A forward-filled *price* is
    point-in-time; a forward-filled return would not be.
    """
    dates = df.select("timestamp").unique().sort("timestamp")
    aligned = (
        dates.join(
            gold.filter(pl.col("symbol") == GOLD_SYMBOL).select("timestamp", pl.col("close").alias("_gold")),
            on="timestamp",
            how="left",
        )
        .sort("timestamp")
        .with_columns(pl.col("_gold").forward_fill())
    )
    aligned = aligned.with_columns(
        (pl.col("_gold") / pl.col("_gold").shift(h) - 1).alias(f"gold_ret_{h}d") for h in windows["gold"]
    ).drop("_gold")
    return (
        df.join(aligned, on="timestamp", how="left")
        .sort(["symbol", "timestamp"])
        .with_columns(
            pl.rolling_corr(pl.col(f"ret_{h}d"), pl.col(f"gold_ret_{h}d"), window_size=windows["gold_exposure"])
            .over("symbol")
            .alias(f"gold_corr_{h}d")
            for h in windows["gold"]
        )
    )


# ---------------------------------------------------------------------------
# Carry: policy-rate differentials, read at their publication date
# ---------------------------------------------------------------------------
def load_macro_config(path: Path | str | None = None) -> dict:
    """The ``carry`` block of ``bots/_shared/macro_config.yaml``: ids, sign rule, lag.

    Kept in YAML rather than in this module so a series id exists in exactly one place - the
    same file the downloader is pointed at. Raises rather than defaulting: a carry family
    built on an id nobody declared is a number with no provenance.
    """
    config = yaml.safe_load(Path(path or MACRO_CONFIG_PATH).read_text(encoding="utf-8"))
    carry = config.get("carry")
    if not carry or not carry.get("legs"):
        raise ValueError(f"no carry block with legs in {path or MACRO_CONFIG_PATH}")
    for key in ("availability_lag_days", "raw_file"):
        if key not in carry:
            raise ValueError(f"carry block is missing {key!r}")
    if int(carry["availability_lag_days"]) < 1:
        raise ValueError(
            "availability_lag_days must be at least one day: an ALFRED vintage is dated by the "
            "day of release and says nothing about the hour, and this bot decides at 20:00 UTC"
        )
    return carry


def load_carry_vintages(
    carry: Mapping, *, storage_dir: Path | str | None = None
) -> pl.DataFrame:
    """The ALFRED initial-release RAW table, with the date each value became usable.

    Returns ``series, timestamp, vintage_date, value, available_at`` where ``available_at`` is
    ``vintage_date + carry["availability_lag_days"]``.

    Only the *raw* file is accepted. ``data/macro/download_alfred.py::build_aligned_panel``
    writes an aligned panel too, and that one is pivoted on the OBSERVATION date, drops
    ``vintage_date`` and forward-fills on a daily calendar: for a monthly policy rate the
    observation date is the first of the month while the publication date is weeks later, so
    the aligned panel dates every value up to about eight weeks before anyone could see it.
    A frame without ``vintage_date`` is therefore refused rather than used.
    """
    directory = Path(storage_dir) if storage_dir is not None else mt5_data_dir().parent / "mt5" / "macro_exness_fx_d1"
    path = Path(directory) / str(carry["raw_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"carry vintages not downloaded: {path}. Run "
            "`uv run python data/macro/download_alfred.py --config bots/_shared/macro_config.yaml`"
        )
    frame = pl.read_parquet(path)
    missing = [c for c in VINTAGE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is not the ALFRED raw table (missing {missing}); the aligned panel is "
            "forward-filled by observation date and is not point-in-time"
        )
    lag = int(carry["availability_lag_days"])
    return (
        frame.select(VINTAGE_COLUMNS)
        .with_columns(
            pl.col("timestamp").cast(pl.Date),
            pl.col("vintage_date").cast(pl.Date),
            pl.col("value").cast(pl.Float64),
            pl.col("series").str.to_lowercase(),
        )
        .with_columns((pl.col("vintage_date") + pl.duration(days=lag)).alias("available_at"))
        .sort(["series", "available_at", "timestamp"])
    )


def published_rates(vintages: pl.DataFrame, legs: Mapping, dates: pl.Series) -> pl.DataFrame:
    """One column per currency: the last rate published at or before each session date.

    ``dates`` are the decision dates of the panel. For every currency the vintages are collapsed
    to one value per ``available_at`` - the latest observation published that day - and joined
    backward onto the session grid, so a session between two publications reads the earlier one
    and never the later. A currency whose first publication is after a session leaves a null
    there, which the null policy removes; it is not filled.
    """
    grid = pl.DataFrame({"timestamp": dates.unique().sort()}).with_columns(
        pl.col("timestamp").cast(pl.Date)
    )
    for currency, series_id in legs.items():
        column = f"_rate_{currency}"
        published = (
            vintages.filter(pl.col("series") == str(series_id).lower())
            .sort(["available_at", "timestamp"])
            .group_by("available_at", maintain_order=True)
            .agg(pl.col("value").last().alias(column))
            .sort("available_at")
        )
        if published.is_empty():
            raise ValueError(f"no vintages for {currency} ({series_id}) in the carry table")
        grid = grid.join_asof(
            published, left_on="timestamp", right_on="available_at", strategy="backward"
        ).drop("available_at")
    return grid


def carry_factor(
    df: pl.DataFrame, vintages: pl.DataFrame, *, legs: Mapping, windows: Mapping
) -> pl.DataFrame:
    """The policy-rate differential of each pair, its changes and its trailing z-score.

    ``carry(XXXYYY) = rate(XXX) - rate(YYY)`` in percent per annum, each leg at its own last
    published vintage, so a positive value means a long position in the pair earns the
    differential. The universe holds both arrangements - the dollar is the quote currency of
    ``EURUSD`` and the base currency of ``USDJPY`` - and the formula above is what makes the two
    comparable; writing "foreign minus United States" instead would flip the sign on two of the
    five pairs and hand the ranking a systematic error nothing downstream could see.

    ``carry_chg_{w}d`` is the change in the differential over ``w`` sessions, which is what a
    policy surprise looks like on this column; ``carry_z_{z}d`` standardizes the level against
    its own trailing window, because a differential of one point means something different in a
    decade of zero rates and in one of five per cent.
    """
    currencies = set(legs)
    needed = {s[:3] for s in df["symbol"].unique()} | {s[3:6] for s in df["symbol"].unique()}
    unknown = sorted(needed - currencies)
    if unknown:
        raise ValueError(f"no policy-rate leg declared for {unknown} in macro_config.yaml")
    grid = published_rates(vintages, legs, df["timestamp"])
    base, quote = pl.col("symbol").str.head(3), pl.col("symbol").str.tail(3)
    base_rate = pl.coalesce(
        [pl.when(base == currency).then(pl.col(f"_rate_{currency}")) for currency in legs]
    )
    quote_rate = pl.coalesce(
        [pl.when(quote == currency).then(pl.col(f"_rate_{currency}")) for currency in legs]
    )
    out = (
        df.join(grid, on="timestamp", how="left")
        .sort(["symbol", "timestamp"])
        .with_columns((base_rate - quote_rate).alias("carry_diff"))
        .drop([f"_rate_{currency}" for currency in legs])
    )
    return out.with_columns(
        *[
            (pl.col("carry_diff") - pl.col("carry_diff").shift(w).over("symbol")).alias(
                f"carry_chg_{w}d"
            )
            for w in windows["carry"]
        ],
        rolling_zscore("carry_diff", windows["carry_zscore"], "symbol").alias(
            f"carry_z_{windows['carry_zscore']}d"
        ),
    )


def build_features(
    prices: pl.DataFrame,
    *,
    windows: Mapping,
    ranked: Sequence[str],
    periods_per_year: float,
    gold: pl.DataFrame | None = None,
    carry_vintages: pl.DataFrame | None = None,
    carry_legs: Mapping | None = None,
) -> pl.DataFrame:
    """The whole construction, as one function every consumer calls.

    The last step converts NaN to null. Some library calls fill their warmup with NaN and
    others with null, and a NaN is a missing value dressed as a number: ``is_not_null`` reads
    it as present, so the warmup audit and the coverage figure would score an empty stretch
    as covered.
    """
    df = (
        prices.select(BAR_COLUMNS)
        .sort(["symbol", "timestamp"])
        .pipe(momentum_features, windows, periods_per_year)
        .pipe(volatility_features, windows, periods_per_year)
        .pipe(mean_reversion_features, windows)
        .pipe(drawdown_range_and_oscillators, windows)
        .pipe(dollar_factor, windows)
    )
    if gold is not None:
        df = gold_factor(df, gold, windows)
    if carry_vintages is not None:
        if carry_legs is None:
            raise ValueError("carry_vintages given without carry_legs")
        df = carry_factor(df, carry_vintages, legs=carry_legs, windows=windows)
    df = df.with_columns(cross_sectional_percentile(col, "timestamp").alias(f"rank_{col}") for col in ranked)
    return df.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))


def feature_columns(built: pl.DataFrame) -> list[str]:
    return [c for c in built.columns if c not in EXCLUDED]


def warmup_expectations(windows: Mapping) -> dict[str, int]:
    """The declared warmup, per audited column, from the window register alone."""
    expected = {
        "zscore_126d": windows["zscore"] + windows["zscore_horizons"][-1],
        "usd_corr_63d": windows["dollar_exposure"] + windows["dollar"][-1],
        "ret_252d": windows["momentum"][-1],
        "sharpe_252d": windows["momentum"][-1],
        "vol_gk_252d": windows["garman_klass"][-1],
        "price_to_ma_252d": windows["moving_average"][-1],
        "channel_pos_126d": windows["channel"][-1],
        "max_dd_63d": windows["drawdown"][0],
        "rsi_14d": windows["rsi"][0],
    }
    if "gold" in windows:
        expected[f"gold_corr_{windows['gold'][-1]}d"] = windows["gold_exposure"] + windows["gold"][-1]
    if "carry" in windows:
        # `carry_diff` itself has no warmup - the rate is published before the session that
        # reads it - so only the two derived columns are audited.
        expected[f"carry_chg_{windows['carry'][-1]}d"] = windows["carry"][-1]
        expected[f"carry_z_{windows['carry_zscore']}d"] = windows["carry_zscore"]
    return expected


# ---------------------------------------------------------------------------
# Point-in-time recomputation
# ---------------------------------------------------------------------------
def bars_closed_by(bars: pl.DataFrame, decision_ts: datetime, bar_minutes: int = BAR_MINUTES) -> pl.DataFrame:
    """The bars a trader at ``decision_ts`` (naive UTC) could have seen: those that had closed."""
    frame = bars
    ts_dtype = frame.schema["timestamp"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        frame = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    return frame.filter(pl.col("timestamp") + pl.duration(minutes=bar_minutes) <= pl.lit(decision_ts))


def vintages_published_by(vintages: pl.DataFrame, decision_ts: datetime | date) -> pl.DataFrame:
    """The macro values a trader at ``decision_ts`` could have read: those already available.

    The filter is on ``available_at`` (the publication date plus the declared lag), never on the
    observation date the value describes. A monthly policy rate observed on the first of the
    month is published weeks later, so filtering on ``timestamp`` would let a decision read a
    number that did not exist yet - which is the whole failure mode this column is here to make
    impossible.
    """
    cutoff = decision_ts.date() if isinstance(decision_ts, datetime) else decision_ts
    return vintages.filter(pl.col("available_at") <= pl.lit(cutoff))


def features_as_of(
    bars: pl.DataFrame,
    decision_ts: datetime,
    setup: Mapping,
    *,
    gold_bars: pl.DataFrame | None = None,
    carry_vintages: pl.DataFrame | None = None,
    carry_legs: Mapping | None = None,
) -> pl.DataFrame:
    """Rebuild the feature panel from the bars that had closed at ``decision_ts``.

    The last session of the result is the one decided at ``decision_ts``; its feature row is what
    the batch panel must reproduce for that session, and ``test_lookahead.py`` asserts that it
    does. The gap between the two would be information from after the decision.

    The carry legs are truncated the same way, on their publication date rather than on the date
    they describe (:func:`vintages_published_by`), so the comparison tests the vintage join as
    well as the bar aggregation.
    """
    calendar = setup["decision"]["session_calendar"]
    tolerance = int(setup["decision"]["session_close_tolerance_minutes"])
    prices = session_panel(bars_closed_by(bars, decision_ts), calendar=calendar, tolerance_minutes=tolerance, verbose=False)
    gold = None
    if gold_bars is not None:
        gold = session_panel(
            bars_closed_by(gold_bars, decision_ts), calendar=calendar, tolerance_minutes=tolerance, verbose=False
        )
    features = setup["features"]
    return build_features(
        prices,
        windows=features["windows"],
        ranked=features["ranked"],
        periods_per_year=setup["evaluation"]["periods_per_year"],
        gold=gold,
        carry_vintages=(
            None if carry_vintages is None else vintages_published_by(carry_vintages, decision_ts)
        ),
        carry_legs=carry_legs,
    )


__all__ = [
    "BAR_COLUMNS",
    "BAR_MINUTES",
    "EXCLUDED",
    "GOLD_SYMBOL",
    "MACRO_CONFIG_PATH",
    "PANEL_COLUMNS",
    "VINTAGE_COLUMNS",
    "bars_closed_by",
    "build_features",
    "carry_factor",
    "declared_session_closes",
    "dollar_factor",
    "drawdown_range_and_oscillators",
    "feature_columns",
    "features_as_of",
    "gold_factor",
    "load_carry_vintages",
    "load_macro_config",
    "load_session_panel",
    "mean_reversion_features",
    "momentum_features",
    "published_rates",
    "session_panel",
    "volatility_features",
    "vintages_published_by",
    "warmup_expectations",
]
