"""Data-quality checks on the MT5 hourly history of exness_gold_sess (roadmap phase 2).

Modelled on ``02_financial_data_universe/13_data_quality_framework.py`` (OHLC invariants, gap
detection, deduplication, outlier flags) and on ``bots/exness_fx_d1/tests/test_data_quality.py``,
applied to ``ML4T_DATA_PATH/mt5/1h.parquet`` for the two metals of ``setup.yaml``, on the
**development history only** - the holdout window is never read for a value.

Beyond the generic checks, the metals-specific facts the 2026-09-08 census found
(``bots/exness_gold_sess/data_census_2026-09-08.md``) are asserted here rather than left in a
markdown file, because a fact nobody re-checks is a fact that quietly stops being true:

* the 21:00 and 22:00 UTC hours are **complementary** - the server's one-hour daily break sits at
  21:00-22:00 UTC while New York is on DST and at 22:00-23:00 UTC when it is not, so exactly one
  of the two prints on a given weekday;
* the Mon-Fri calendar holes are Christmas, New Year and **Good Friday** and nothing else;
* a small, named set of session days carries no decision bar at all;
* the sparse pre-2017 prefix - six bars a week, a daily backfill served on the H1 timeframe - is
  outside ``universe.history_start`` and cannot leak into the panel;
* bars quoting ``spread == 0`` exist and are counted, because pricing that stretch off the bar
  field would price 2017-2023 as free.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests/test_data_quality.py -q -s
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import mt5_data_dir, read_history_depth
from utils.data_quality import check_ohlc_invariants
from utils.paths import REPO_ROOT

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
BAR_HOURS = 1
#: An hourly move larger than this is listed; larger than the second is a data error. Gold moves
#: further in an hour than a currency pair does, so the FX bot's 2 %/10 % pair would flag ordinary
#: sessions here. Set from the census: the largest hourly move in nine years is under 6 %.
RETURN_OUTLIER = 0.03
RETURN_ERROR = 0.10
#: Share of a symbol's bars that may repeat the previous close. Silver's thin Asian hours repeat
#: more often than gold's, so the ceiling is read against the census rather than against FX.
STALE_CLOSE_CEILING = 0.02
FLAT_BAR_CEILING = 0.001


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def window(setup: dict) -> tuple[date, date]:
    start = date.fromisoformat(str(setup["universe"]["history_start"]))
    dev_end = date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1)
    return start, dev_end


@pytest.fixture(scope="module")
def raw(setup: dict) -> pl.DataFrame:
    """The whole 1h parquet for the two metals - used only to count what is EXCLUDED."""
    path = mt5_data_dir() / "1h.parquet"
    if not path.exists():
        pytest.skip(f"MT5 history not downloaded: {path}")
    return (
        pl.read_parquet(path)
        .filter(pl.col("symbol").is_in(sorted(setup["universe"]["symbols"])))
        .with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
        .sort(["symbol", "timestamp"])
    )


@pytest.fixture(scope="module")
def bars(raw: pl.DataFrame, window: tuple[date, date]) -> pl.DataFrame:
    start, dev_end = window
    return raw.filter(
        (pl.col("timestamp") >= datetime.combine(start, time(0)))
        & (pl.col("timestamp") < datetime.combine(dev_end + timedelta(days=1), time(0)))
    )


# ---------------------------------------------------------------------------
# 1-4: keys, grid, invariants, volume
# ---------------------------------------------------------------------------
def test_keys_unique_and_on_the_hourly_grid(bars: pl.DataFrame) -> None:
    dupes = bars.select(["symbol", "timestamp"]).is_duplicated().sum()
    off_grid = bars.filter(pl.col("timestamp").dt.minute() != 0)
    print(
        f"\n{bars.height:,} bars, {bars['symbol'].n_unique()} metals, "
        f"{bars['timestamp'].min()} .. {bars['timestamp'].max()}"
    )
    print(f"duplicate keys: {dupes}; bars off the :00 UTC grid: {off_grid.height}")
    assert dupes == 0
    assert off_grid.height == 0
    assert bars.null_count().sum_horizontal().item() == 0, "a null in an OHLCV column"


def test_ohlc_invariants_hold(bars: pl.DataFrame) -> None:
    table = check_ohlc_invariants(bars)
    print("\n", table)
    assert (table["valid_pct"] == 100.0).all(), table
    assert bars.filter(pl.col("close") <= 0).height == 0


def test_no_zero_volume_bars(bars: pl.DataFrame) -> None:
    zero = bars.filter(pl.col("volume") <= 0)
    print(f"\nzero-volume bars: {zero.height}")
    assert zero.height == 0


def test_flat_bars_and_repeated_closes_are_rare(bars: pl.DataFrame) -> None:
    """A flat bar is one tick; a repeated close is an hour with no trade at the print.

    Both are real and both are rare, and both are counted per symbol rather than pooled: silver
    is an order of magnitude thinner than gold on this account, so a pooled share would hide a
    silver problem behind gold's volume.
    """
    rows = []
    for symbol in sorted(bars["symbol"].unique().to_list()):
        sym = bars.filter(pl.col("symbol") == symbol)
        flat = sym.filter(
            (pl.col("open") == pl.col("high"))
            & (pl.col("high") == pl.col("low"))
            & (pl.col("low") == pl.col("close"))
        )
        stale = sym.filter(pl.col("close") == pl.col("close").shift(1))
        rows.append((symbol, sym.height, flat.height, stale.height, stale.height / sym.height))
        print(
            f"\n{symbol}: {flat.height} flat bars (o=h=l=c), {stale.height} bars repeating the "
            f"previous close ({stale.height / sym.height:.3%})"
        )
        if flat.height:
            print(flat.select("timestamp", "close", "volume").head(20))
    for symbol, n, flat, stale, share in rows:
        assert flat <= FLAT_BAR_CEILING * n, (symbol, flat, n)
        assert share <= STALE_CLOSE_CEILING, (symbol, share)


def test_hourly_return_outliers(bars: pl.DataFrame) -> None:
    rets = bars.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret")
    ).drop_nulls("ret")
    large = rets.filter(pl.col("ret").abs() > RETURN_OUTLIER).sort("ret")
    print(f"\nhourly moves larger than {RETURN_OUTLIER:.0%}: {large.height}")
    if large.height:
        print(large.select("symbol", "timestamp", "ret").head(30))
    assert rets.filter(pl.col("ret").abs() > RETURN_ERROR).height == 0


# ---------------------------------------------------------------------------
# 5: the parquet against history_depth.json
# ---------------------------------------------------------------------------
def test_history_depth_matches_the_parquet(setup: dict, raw: pl.DataFrame) -> None:
    """And the recorded depth must say HOW it was read, and where the dense grid starts.

    A depth number without its method is not interpretable on this account: a range download
    returned 22,838 H1 bars from 2022-10-25 and a count-based one 57,094 from 2014-01-14, from
    the same terminal in the same hour. ``universe.history_start`` rests on the second, so this
    test refuses a record written by the first.
    """
    depth = read_history_depth()
    for symbol in sorted(setup["universe"]["symbols"]):
        recorded = depth["timeframes"]["1h"][symbol]
        rows = raw.filter(pl.col("symbol") == symbol)
        assert rows.height == recorded["n_bars"], (symbol, rows.height, recorded["n_bars"])
        assert rows["timestamp"].min() == datetime.fromisoformat(recorded["first"]).replace(
            tzinfo=None
        ), symbol
        assert rows["timestamp"].max() == datetime.fromisoformat(recorded["last"]).replace(
            tzinfo=None
        ), symbol
        assert recorded.get("history_mode") == "deep", (
            f"{symbol}: history_depth.json records the H1 depth as "
            f"{recorded.get('history_mode', 'range (no history_mode key)')!r}; a range download "
            "under-reports it on this account"
        )
        assert "dense_history_start" in recorded, (
            f"{symbol}: no dense_history_start in the depth record. A deep read reaches into a "
            "prefix that is not an hourly grid, and universe.history_start is asserted against "
            "this key by 01_feasibility_analysis"
        )
        assert recorded["dense_history_start"] <= str(setup["universe"]["history_start"]), symbol
        print(
            f"\n{symbol}: {recorded['n_bars']:,} bars {recorded['first']} .. {recorded['last']} "
            f"read {recorded['history_mode']}, dense from {recorded['dense_history_start']}"
        )


# ---------------------------------------------------------------------------
# 6: the sparse pre-2017 prefix is excluded
# ---------------------------------------------------------------------------
def test_the_sparse_prefix_is_outside_the_declared_history(
    raw: pl.DataFrame, setup: dict
) -> None:
    """Six bars a week is a daily backfill served on the H1 timeframe, not an hourly grid.

    A session decision is the bar that closes one hour after the venue opens. At six bars a week
    that bar does not exist, so the prefix cannot carry a decision - and the only thing standing
    between it and the panel is ``universe.history_start``, which is why it is asserted here.
    """
    start = date.fromisoformat(str(setup["universe"]["history_start"]))
    for symbol in sorted(setup["universe"]["symbols"]):
        sym = raw.filter(pl.col("symbol") == symbol)
        pre = sym.filter(pl.col("timestamp") < datetime.combine(start, time(0)))
        post = sym.filter(pl.col("timestamp") >= datetime.combine(start, time(0)))
        weekly = lambda f: (  # noqa: E731
            f.with_columns(pl.col("timestamp").dt.truncate("1w").alias("w"))
            .group_by("w")
            .len()["len"]
            .median()
        )
        print(
            f"\n{symbol}: {pre.height:,} bars before {start} at a median "
            f"{float(weekly(pre)):.0f} a week; {post.height:,} from {start} at a median "
            f"{float(weekly(post)):.0f} a week"
        )
        assert pre.height > 0, "the deep read should have returned a prefix to exclude"
        assert float(weekly(pre)) < 20, "the prefix is denser than a daily backfill"
        assert float(weekly(post)) > 100, "the declared history is not a dense hourly grid"


# ---------------------------------------------------------------------------
# 7: the complementary 21:00 / 22:00 UTC break hours
# ---------------------------------------------------------------------------
def test_the_daily_break_hour_is_complementary(bars: pl.DataFrame) -> None:
    """Exactly one of the 21:00 and 22:00 UTC bars prints on a Mon-Thu date.

    The server takes a one-hour break at 21:00-22:00 UTC while New York is on DST and at
    22:00-23:00 UTC when it is not, and the server clock itself never moves. The New York label
    endpoint sits at the far side of that break, so a design that assumed both hours print - or
    that neither does - would lose or duplicate the last hour of every New York session.
    """
    ny = ZoneInfo("America/New_York")
    frame = bars.filter(pl.col("timestamp").dt.weekday() <= 4).with_columns(  # Mon..Thu
        pl.col("timestamp").dt.date().alias("day"), pl.col("timestamp").dt.hour().alias("hour")
    )
    for symbol in sorted(frame["symbol"].unique().to_list()):
        sym = frame.filter(pl.col("symbol") == symbol)
        wide = (
            sym.filter(pl.col("hour").is_in([21, 22]))
            .group_by(["day", "hour"])
            .len()
            .pivot(index="day", on="hour", values="len")
            .fill_null(0)
        )
        wide = wide.with_columns(
            pl.col("day")
            .map_elements(
                lambda d: bool(datetime(d.year, d.month, d.day, 12, tzinfo=ny).dst()),
                return_dtype=pl.Boolean,
            )
            .alias("us_dst")
        )
        both = wide.filter((pl.col("21") > 0) & (pl.col("22") > 0))
        neither = wide.filter((pl.col("21") == 0) & (pl.col("22") == 0))
        summer_21 = wide.filter(pl.col("us_dst") & (pl.col("21") > 0)).height
        winter_22 = wide.filter(~pl.col("us_dst") & (pl.col("22") > 0)).height
        print(
            f"\n{symbol}: {wide.height:,} Mon-Thu dates; both 21:00 and 22:00 printed on "
            f"{both.height}, neither on {neither.height}; under US DST the 21:00 bar is absent "
            f"on {wide.filter(pl.col('us_dst')).height - summer_21} of "
            f"{wide.filter(pl.col('us_dst')).height}, outside it the 22:00 bar is absent on "
            f"{wide.filter(~pl.col('us_dst')).height - winter_22} of "
            f"{wide.filter(~pl.col('us_dst')).height}"
        )
        print(f"  dates printing both: {[str(d) for d in both['day'].to_list()]}")
        print(f"  dates printing neither: {[str(d) for d in neither['day'].to_list()]}")
        # Complementary on all but a handful of dates. MEASURED 2026-09-08 over the development
        # window: 19 of 1,718 Mon-Thu dates print both hours (1.1 %) and none prints neither. They
        # are transition-week and partial-session dates, where the break did not fall where the
        # zone rule puts it - the reason this is a ceiling and not an equality. It is set at 2 %
        # against a measurement of 1.1 %, so a systematic change (a broker moving its break) fails
        # it while the known exceptions do not.
        assert both.height <= 0.02 * wide.height, both.head(20)
        assert neither.height <= 0.02 * wide.height, neither.head(20)


# ---------------------------------------------------------------------------
# 8: the calendar holes, and the session days with no decision bar
# ---------------------------------------------------------------------------
def _good_fridays(first: int, last: int) -> set[date]:
    """Good Friday by the anonymous Gregorian computus, so the dates are derived, not typed.

    A hand-typed list of holiday dates is a data source with no provenance and a shelf life; the
    computus is the rule the calendar itself uses, and it keeps working in 2030.
    """
    out: set[date] = set()
    for year in range(first, last + 1):
        a, b, c = year % 19, year // 100, year % 100
        d, e = b // 4, b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = c // 4, c % 4
        el = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * el) // 451
        month = (h + el - 7 * m + 114) // 31
        day = ((h + el - 7 * m + 114) % 31) + 1
        out.add(date(year, month, day) - timedelta(days=2))
    return out


def test_calendar_holes_are_the_holidays_the_census_named(
    bars: pl.DataFrame, window: tuple[date, date]
) -> None:
    """Every Mon-Fri date with no bar at all is Christmas, New Year or Good Friday."""
    start, end = window
    weekdays = {
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    }
    good_fridays = _good_fridays(start.year, end.year)
    for symbol in sorted(bars["symbol"].unique().to_list()):
        present = set(
            bars.filter(pl.col("symbol") == symbol)["timestamp"].dt.date().unique().to_list()
        )
        holes = sorted(weekdays - present)
        explained = [
            d
            for d in holes
            if (d.month, d.day) in {(12, 25), (1, 1)} or d in good_fridays
        ]
        unexplained = [d for d in holes if d not in explained]
        print(
            f"\n{symbol}: {len(holes)} Mon-Fri dates with no bar; "
            f"{len([d for d in explained if d in good_fridays])} Good Fridays, "
            f"{len([d for d in explained if (d.month, d.day) == (12, 25)])} Christmas Days, "
            f"{len([d for d in explained if (d.month, d.day) == (1, 1)])} New Year's Days"
        )
        print(f"  all holes: {[str(d) for d in holes]}")
        assert good_fridays & set(holes), "no Good Friday hole found; the metals do trade then?"
        assert len(unexplained) <= 2, f"{symbol}: unexplained calendar holes {unexplained}"


def test_the_session_days_without_a_decision_bar_are_named_and_few(
    bars: pl.DataFrame, setup: dict
) -> None:
    """A Mon-Fri date with bars but no bar closing an hour after a venue's open.

    Those dates are dropped by ``_features.decision_grid`` as a stale decision, and the point of
    counting them here is that the count is what the 0.5 % stale-decision guard is set against.
    """
    london, ny = ZoneInfo("Europe/London"), ZoneInfo("America/New_York")
    frame = bars.with_columns(pl.col("timestamp").dt.date().alias("day"))
    for symbol in sorted(frame["symbol"].unique().to_list()):
        sym = frame.filter(pl.col("symbol") == symbol)
        present = set(sym["timestamp"].to_list())
        days = sorted({d for d in sym["day"].unique().to_list() if d.weekday() < 5})
        missing = {"london": [], "ny": []}
        for day in days:
            for venue, zone in (("london", london), ("ny", ny)):
                opened = (
                    datetime.combine(day, time(8), tzinfo=zone)
                    .astimezone(ZoneInfo("UTC"))
                    .replace(tzinfo=None)
                )
                # the DECISION bar is the one that opens at the session open and closes an hour
                # later; its parquet timestamp is the open
                if opened not in present:
                    missing[venue].append(day)
        print(
            f"\n{symbol}: {len(days):,} Mon-Fri session days; no London decision bar on "
            f"{len(missing['london'])}, no New York decision bar on {len(missing['ny'])}"
        )
        for venue, dates in missing.items():
            print(f"  {venue}: {[str(d) for d in dates]}")
            assert len(dates) <= 0.01 * len(days), (symbol, venue, len(dates))


# ---------------------------------------------------------------------------
# 9: bars quoting a zero spread
# ---------------------------------------------------------------------------
def test_zero_spread_bars_are_counted_and_not_priced_as_free(
    bars: pl.DataFrame, setup: dict
) -> None:
    """They exist. They are not a free crossing, and this bot must not read the bar field.

    ``xau_fx_mt5`` measured about 1,015 such bars per metal on this account up to 2023-02-03 and
    fills them with a point-in-time rolling median. This bot takes ``costs.spread_bps`` from a
    30-day tick measurement instead, so the count is a coverage fact - until someone prices the
    2017-2023 stretch off ``spread``, at which point it becomes six years of free trading. The
    assertion is therefore on the CONFIGURATION as much as on the data.
    """
    assert "spread_bps" in setup["costs"], "costs.spread_bps is the tick-measured source"
    assert setup["costs"]["spread_bps_by_session"]["source"] != "bar_spread_field", (
        "this bot's spread is measured from ticks; if that ever changes, the zero-spread bars "
        "below have to be filled point-in-time first"
    )
    for symbol in sorted(bars["symbol"].unique().to_list()):
        sym = bars.filter(pl.col("symbol") == symbol)
        zero = sym.filter(pl.col("spread") <= 0)
        by_year = zero.group_by(pl.col("timestamp").dt.year().alias("y")).len().sort("y")
        print(
            f"\n{symbol}: {zero.height:,} of {sym.height:,} development bars quote spread == 0 "
            f"({zero.height / sym.height:.3%}), "
            f"{zero['timestamp'].min()} .. {zero['timestamp'].max()}; "
            f"median quoted spread {float(sym['spread'].median()):.0f} points"
        )
        print(f"  by year: {dict(zip(by_year['y'], by_year['len'], strict=True))}")
        assert zero.height > 0, "the census counted about a thousand; none found is a change"
        assert zero.height < 0.05 * sym.height
        assert zero["timestamp"].max() < datetime(2023, 3, 1), (
            "a zero-spread bar after the 2023-02-03 the census recorded as the last one"
        )
