"""Data-quality checks on the MT5 one-hour history of exness_usidx_sess (roadmap phase 2).

Modelled on ``02_financial_data_universe/13_data_quality_framework.py`` (OHLC invariants, gap
detection, deduplication, outlier flags) and applied to ``ML4T_DATA_PATH/mt5/1h.parquet`` for
the two indices of ``setup.yaml``, on the development history only (the holdout window is never
read). Hard invariants are asserted; the rest is printed as findings for ``BOT.md`` (run with
``-s`` to see them):

- keys unique, every bar on the one-hour UTC grid, no bar off the hour;
- ``high >= max(open, close)``, ``low <= min(open, close)``, ``close > 0``;
- the parquet against ``history_depth.json``;
- the hourly grid against the sessions the NYSE calendar declares: how many cash sessions carry
  a complete set of hourly bars, and how many hours a typical session is missing;
- ``spread`` (points at the bar open) against the p90 the tick measurement recorded in
  ``spreads_by_session.json``, overall and on the two decision bars;
- flat bars and hourly return outliers.

Unlike the FX bot's version, this one does **not** test the bar grid against
``sessions_mt5.json``. That file is an eight-week summer snapshot of the trading hours, and this
instrument's hours move with New York daylight saving (``test_sessions.py``), so the snapshot
would mark half the sample as missing. The session-level coverage check below is the replacement.

Skipped when the MT5 history is not downloaded.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_usidx_sess/tests/test_data_quality.py -q -s
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta

import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import mt5_data_dir, read_history_depth
from utils.data_quality import check_ohlc_invariants
from utils.paths import REPO_ROOT

CASE_STUDY_ID = "exness_usidx_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
# The point value of both indices is 0.01, so a spread quoted in points becomes a price
# by dividing by 100 - the same conversion setup.yaml's swap block applies when it turns
# -147.4 points into -1.474 USD at trade_contract_size 1.
SPREAD_POINT_VALUE_DIVISOR = 100.0
# The declared cost is a p50/p90 pair rounded to two decimals, so a comparison against a
# freshly computed quantile needs the width of that rounding and nothing more.
TOLERANCE_BPS = 0.01
# The bars this bot fills on, in both DST regimes and on a half day. Kept here rather than read
# from setup.yaml so that a declaration edited to make this test pass has to edit the test too.
EXECUTION_HOURS_UTC = [14, 15, 17, 18, 20, 21]
RETURN_OUTLIER = 0.01  # an hourly move larger than this is listed
RETURN_ERROR = 0.10  # ... and larger than this is a data error
# An index CFD prints 6 or 7 hourly bars inside a full 6.5-hour NYSE cash session (the open and
# the close both fall inside an hour). A session missing more than one of them had an outage.
MIN_CASH_HOURS = 6


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def window(setup: dict) -> tuple[date, date]:
    start = date.fromisoformat(str(setup["universe"]["history_start"]))
    dev_end = date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1)
    return start, dev_end


@pytest.fixture(scope="module")
def bars(setup: dict, window: tuple[date, date]) -> pl.DataFrame:
    path = mt5_data_dir() / "1h.parquet"
    if not path.exists():
        pytest.skip(f"MT5 history not downloaded: {path}")
    start, dev_end = window
    return (
        pl.read_parquet(path)
        .filter(pl.col("symbol").is_in(sorted(setup["universe"]["symbols"])))
        .with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
        .filter(
            (pl.col("timestamp") >= datetime.combine(start, time(0)))
            & (pl.col("timestamp") < datetime.combine(dev_end + timedelta(days=1), time(0)))
        )
        .sort(["symbol", "timestamp"])
    )


@pytest.fixture(scope="module")
def spreads() -> dict:
    path = mt5_data_dir() / "spreads_by_session.json"
    if not path.exists():
        pytest.skip(f"spread table not recorded: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Keys, grid, invariants
# ---------------------------------------------------------------------------
def test_keys_unique_and_on_the_hourly_grid(bars: pl.DataFrame) -> None:
    dupes = bars.select(["symbol", "timestamp"]).is_duplicated().sum()
    off_grid = bars.filter(
        (pl.col("timestamp").dt.minute() != 0) | (pl.col("timestamp").dt.second() != 0)
    )
    print(
        f"\n{bars.height:,} bars, {bars['symbol'].n_unique()} indices, "
        f"{bars['timestamp'].min()} .. {bars['timestamp'].max()}"
    )
    print(f"duplicate keys: {dupes}; bars off the 1h UTC grid: {off_grid.height}")
    assert dupes == 0
    assert off_grid.height == 0


def test_ohlc_invariants_hold(bars: pl.DataFrame) -> None:
    table = check_ohlc_invariants(bars)
    print("\n", table)
    assert (table["valid_pct"] == 100.0).all(), table
    assert bars.filter(pl.col("close") <= 0).height == 0


def test_history_depth_matches_parquet(setup: dict) -> None:
    depth = read_history_depth()
    full = pl.read_parquet(mt5_data_dir() / "1h.parquet")
    for symbol in sorted(setup["universe"]["symbols"]):
        recorded = depth["timeframes"]["1h"][symbol]
        rows = full.filter(pl.col("symbol") == symbol)
        assert rows.height == recorded["n_bars"], (symbol, rows.height, recorded["n_bars"])
        assert rows["timestamp"].min().isoformat() == recorded["first"], symbol
        assert rows["timestamp"].max().isoformat() == recorded["last"], symbol
        print(f"\n{symbol}: {rows.height:,} H1 bars, {recorded['first']} .. {recorded['last']}")


# ---------------------------------------------------------------------------
# Coverage of the cash session
# ---------------------------------------------------------------------------
def test_every_cash_session_carries_its_hourly_bars(bars: pl.DataFrame, setup: dict) -> None:
    """The hours the bot actually reads are the cash-session hours; those must be complete.

    Bars outside the cash session are used for features and never for a decision, so a gap there
    is not a defect of this bot. A gap inside the cash session is: it removes a decision instant
    or an execution price.
    """
    pytest.importorskip("ml4t.diagnostic", reason="the NYSE calendar needs the full environment")
    from case_studies.exness_usidx_sess._features import session_grid

    decision = setup["decision"]
    grid = session_grid(
        decision["session_calendar"],
        bars["timestamp"].min(),
        bars["timestamp"].max(),
        open_delay_minutes=int(decision["open_delay_minutes"]),
    ).filter(
        (pl.col("close_at") >= bars["timestamp"].min())
        & (pl.col("close_at") <= bars["timestamp"].max())
    )
    full_days = grid.filter(
        (pl.col("close_at") - pl.col("open_at")).dt.total_minutes() > 300
    )
    for symbol in sorted(bars["symbol"].unique().to_list()):
        sym = bars.filter(pl.col("symbol") == symbol).select("timestamp")
        counts = (
            full_days.join_where(
                sym,
                pl.col("timestamp") >= pl.col("open_at"),
                pl.col("timestamp") < pl.col("close_at"),
            )
            .group_by("session")
            .agg(pl.len().alias("hours"))
        )
        joined = full_days.join(counts, on="session", how="left").with_columns(
            pl.col("hours").fill_null(0)
        )
        thin = joined.filter(pl.col("hours") < MIN_CASH_HOURS)
        share = thin.height / joined.height
        print(
            f"\n{symbol}: {joined.height:,} full cash sessions, {thin.height} carry fewer than "
            f"{MIN_CASH_HOURS} hourly bars ({share:.2%}); "
            f"median hours per session {joined['hours'].median():.0f}"
        )
        if thin.height:
            print("  thin sessions:", sorted(str(d) for d in thin["session"].to_list())[:20])
        assert share <= 0.01, f"{symbol}: {share:.1%} of cash sessions are missing hourly bars"


# ---------------------------------------------------------------------------
# Spreads, flat bars, return outliers
# ---------------------------------------------------------------------------
def test_declared_spread_covers_the_development_tape(bars: pl.DataFrame, setup: dict) -> None:
    """The declared cost has to cover what this account's own tape quoted, over the whole sample.

    REWRITTEN 2026-09-08 (second pass). The test this replaces compared the bar spread with five
    times the p90 of a thirty-day tick window and passed with room to spare while the declared
    cost was wrong by a factor of four: the worst month in the file quotes 174 points against a
    threshold of 5 x 51 = 255, so no history this account could have produced would ever have
    failed it. A test that cannot fail is not evidence.

    What is asserted instead is the thing that actually matters to a backtest: the number
    ``setup.yaml::costs.spread_bps.indices`` declares, which is what the engine charges, must be
    at least the p90 the tape quotes over the development window - and the MEDIAN bar must sit
    inside it too, so a declaration that covers the tail by ignoring the body cannot pass either.
    Both are checked per symbol, in basis points of the bar's own close, because the point value
    is fixed and the index level is not.
    """
    declared_p90 = float(setup["costs"]["spread_bps"]["indices"][-1])
    declared_p50 = float(setup["costs"]["spread_bps"]["indices"][0])
    # Only the hours this bot crosses in. The declaration is the cost of a CROSSING, and this
    # bot crosses at the bar opening on each of its two decision instants - 14:00 or 15:00 UTC
    # for the intraday spec and 20:00 or 21:00 for the overnight one, with 17:00 and 18:00 on
    # the half days. Measuring over all 24 hours would charge the strategy for the spread of
    # sessions it never trades in, which is conservative in the wrong direction: it is not
    # more correct, only larger.
    frame = bars.filter(pl.col("timestamp").dt.hour().is_in(EXECUTION_HOURS_UTC)).with_columns(
        (1e4 * (pl.col("spread") / SPREAD_POINT_VALUE_DIVISOR) / pl.col("close")).alias("bps")
    )
    for symbol in sorted(frame["symbol"].unique().to_list()):
        sym = frame.filter(pl.col("symbol") == symbol)
        p50, p90 = float(sym["bps"].median()), float(sym["bps"].quantile(0.9))
        worst_month = (
            sym.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.col("bps").quantile(0.9).alias("p90"))
            .sort("p90", descending=True)
            .row(0, named=True)
        )
        print(
            f"\n{symbol}: development tape p50 {p50:.2f} / p90 {p90:.2f} bps against a declared "
            f"[{declared_p50:.2f}, {declared_p90:.2f}]; worst month "
            f"{worst_month['month'].date()} at p90 {worst_month['p90']:.2f} bps; "
            f"median bar spread {sym['spread'].median():.0f} points, max {sym['spread'].max():.0f}"
        )
        assert declared_p90 >= p90 - TOLERANCE_BPS, (
            f"{symbol}: costs.spread_bps.indices declares {declared_p90:.2f} bps but the "
            f"development tape quotes a p90 of {p90:.2f}; the engine would under-price its trades"
        )
        assert declared_p50 >= p50 - TOLERANCE_BPS, (
            f"{symbol}: the declared p50 {declared_p50:.2f} bps is below the tape's median "
            f"{p50:.2f}; a declaration that covers the tail but not the body still under-prices"
        )
        assert declared_p90 >= worst_month["p90"] - TOLERANCE_BPS, (
            f"{symbol}: the worst month of the development window quotes a p90 of "
            f"{worst_month['p90']:.2f} bps against a declared {declared_p90:.2f}"
        )


def test_the_spread_regime_break_is_still_where_setup_says_it_is(
    bars: pl.DataFrame, setup: dict
) -> None:
    """The 2024-10 regime break is a declared fact about this account, so it is checked.

    ``costs.spread_bps_by_year.regime_break`` says the quoted spread stepped down by about four
    times in October 2024. If a later download moves that month, or the step disappears, every
    number in the cost block was computed on a different tape and has to be recomputed - which
    this test says out loud rather than leaving to be noticed.
    """
    declared = str(setup["costs"]["spread_bps_by_year"]["regime_break"])
    monthly = (
        bars.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
        .group_by(["symbol", "month"])
        .agg(pl.col("spread").median().alias("p50"))
        .sort(["symbol", "month"])
        .with_columns((pl.col("p50") / pl.col("p50").shift(1).over("symbol")).alias("step"))
    )
    for symbol in sorted(monthly["symbol"].unique().to_list()):
        sym = monthly.filter(pl.col("symbol") == symbol).drop_nulls("step")
        biggest = sym.sort("step").row(0, named=True)
        found = biggest["month"].strftime("%Y-%m")
        print(
            f"\n{symbol}: the largest month-on-month fall in the median quoted spread is "
            f"{found} at x{biggest['step']:.2f} (declared regime break {declared})"
        )
        assert found == declared, (
            f"{symbol}: the spread regime break is at {found}, not the declared {declared}; "
            "the whole of costs.spread_bps_by_year was measured on a different tape"
        )
        assert biggest["step"] < 0.5, (
            f"{symbol}: the declared regime break at {declared} is only a x{biggest['step']:.2f} "
            "step; the two-regime cost story no longer describes this file"
        )


def test_flat_bars_are_rare(bars: pl.DataFrame) -> None:
    for symbol in sorted(bars["symbol"].unique().to_list()):
        sym = bars.filter(pl.col("symbol") == symbol)
        flat = sym.filter(pl.col("high") == pl.col("low"))
        print(f"\n{symbol}: {flat.height} flat bars of {sym.height:,} ({flat.height / sym.height:.3%})")
        assert flat.height <= 0.01 * sym.height, f"{symbol}: {flat.height} flat bars"


def test_hourly_return_outliers(bars: pl.DataFrame) -> None:
    returns = bars.sort(["symbol", "timestamp"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret")
    ).drop_nulls("ret")
    listed = returns.filter(pl.col("ret").abs() > RETURN_OUTLIER)
    errors = returns.filter(pl.col("ret").abs() > RETURN_ERROR)
    print(
        f"\nhourly moves over {RETURN_OUTLIER:.0%}: {listed.height} of {returns.height:,}; "
        f"largest {returns['ret'].abs().max():.3%}"
    )
    if listed.height:
        print(
            listed.sort(pl.col("ret").abs(), descending=True)
            .select("symbol", "timestamp", "ret")
            .head(10)
        )
    assert errors.height == 0, f"{errors.height} hourly moves exceed {RETURN_ERROR:.0%}"
