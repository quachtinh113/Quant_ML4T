"""The registered price grid, and the hold the engine actually realises on it.

Rewritten on 2026-09-08 - **inverted**, not replaced - when
``bots/exness_gold_sess/PRICE_GRID_DECLARATION.md`` landed. The file it replaces characterised an
open defect: ``_PRICE_CONFIG["exness_gold_sess"]`` registered the *session panel* as the price
grid, one row per decision instant, so the shortest hold an engine could express was
decision-to-next-decision - five hours after the London snapshot, nineteen after the New York one -
against a primary label ``fwd_ret_8h`` that runs eight. The loader raised rather than serve it.

What replaced it, and what this file now measures:

* the grid is the **raw MT5 H1 tape keyed on the bar close** (loader ``exness_gold_sess_h1``,
  ``timestamp + 60 min``), not cropped to session hours, so every decision instant is a row of it;
* the hold is **not** the grid spacing and **not** ``slot_strategy``. It is a broker position rule,
  ``TimeExit(max_bars=HOLD_BARS)``, declared in the ``risk`` block of every ``13_backtest`` spec
  and therefore inside ``backtest_hash``;
* ``setup.yaml::backtest.price_grid_expresses_primary_label`` is now ``true``.

The load-bearing tests below assert on the **realised** holding period of a real
``run_backtest(..., register=False)`` - never on a config value. That is what makes them a
calibration of ``HOLD_BARS`` rather than a restatement of it.

**Rewritten again on 2026-09-08 from** ``bots/exness_gold_sess/NY_EXIT_DECLARATION.md``, which
corrects sections 3.2 and 3.3 of the price-grid declaration and, in its section 0, corrects a fact
this file previously asserted. The old ``test_the_new_york_book_exits_after_its_label_endpoint``
characterised the New York book as broken and said, in its docstring, that *"the metals stop
quoting between 21:00 and 22:00 UTC every day"*. **That is false**, and this bot's own ``BOT.md``
refutes it (Decisions log 2026-09-07, measured over 997 session-days a metal): the broker's daily
one-hour break sits at **21:00-22:00 UTC while the US is on daylight saving and at 22:00-23:00 UTC
when it is not**, so exactly one of the 21:00 / 22:00 prints exists on any given day.

The corrected fact is stronger, and it is what closes every bar-count exit:

    **The New York session close IS the start of the broker's daily break, in both seasons**,
    because both are anchored on 17:00 America/New_York - ``bots/_shared/sessions.py`` gives
    ``new_york`` 08:00-17:00 local and ``bots/assets/XAUUSD.md:110,188`` puts the break at
    17:00-18:00 local. Summer: close 21:00 UTC, break 21:00-22:00. Winter: close 22:00 UTC, break
    22:00-23:00.

With ``execution_price: open`` + ``execution_mode: next_bar`` on a close-keyed grid, a fill at the
label endpoint ``E`` needs the row keyed ``E + 60min``. In New York that interval **is** the break,
in either season, so no bar count fills at the endpoint: 8 bars fills at the first print *after*
the break (60 minutes late, or Sunday 22:00 UTC after a Friday) and 7 fills at the last print
*before* it (60 minutes early, every ordinary day, in both seasons). 8 was rejected because it
trips this bot's own kill criterion (g) on 100 % of New York positions by design; 7 lands in the
measured ``new_york`` spread bucket and errs *inside* the label window.

So the tests below assert the **derived** constant, not a typed one. ``HOLD_BARS[book]`` is
computed here by ``case_studies.exness_gold_sess._hold.derive_hold_bars`` - the same function
``13_backtest`` calls - and comes out **london 8 / ny 7** with a measured shortfall of **0 / 60**
minutes against the sealed ``label_end_ts``.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests/test_backtest_grid.py -q -s
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml

from bots._shared.mt5_loader import mt5_data_dir
from utils.paths import REPO_ROOT, get_case_study_dir

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"

#: MEASURED 2026-09-08 and DECLARED rather than filtered (``_hold.derive_hold_bars``): 102 New
#: York endpoints dated before the earliest fold ``train_start`` (2018-08-30) give 8 bars instead
#: of 7, because the broker's daily break did not sit at the New York close in 2017 - it was at
#: 00:00-01:00 UTC in February and early March, it stayed on 22:00-23:00 UTC through US daylight
#: saving in March and April, and 16 evenings of 2017 had no gap at all. It has been anchored on
#: the New York close in both seasons since 2017-05-08. No fold of this bot fits on, scores on or
#: walks any of those rows. The number is asserted, so a change in it turns this suite red.
PRE_FOLD_BREAK_REGIME = {"ny": 102}

#: One quarter inside fold 0's validation window (2024-08-30 .. 2025-08-28), a clear eight months
#: before the holdout boundary. Short enough to run in the suite, long enough to contain Fridays
#: in both daylight-saving regimes.
WINDOW = ("2024-10-01", "2024-12-31")


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


def _require_data(setup: dict) -> None:
    if not (mt5_data_dir() / "1h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/1h.parquet)")
    label = setup["labels"]["primary"]
    if not (get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{label}.parquet").exists():
        pytest.skip("02_labels has not run in this output directory")


def _derive(setup: dict, label: str) -> tuple[dict[str, int], pl.DataFrame]:
    """``HOLD_BARS[book]`` by the rule of declaration section 1.2, on the development window.

    Nothing here is typed. The rule runs on the **sealed session panel** and the **registered
    price grid** between ``universe.history_start`` and the day before
    ``evaluation.holdout_start`` - ``assert_no_holdout`` refuses any frame that reaches further -
    and it raises rather than returning a guess if the number is not constant inside a book.
    """
    import pandas as pd

    from bots._shared.mt5_loader import load_mt5_bars
    from case_studies.exness_gold_sess._features import session_panel
    from case_studies.exness_gold_sess._hold import (
        assert_no_holdout,
        derive_hold_bars,
        development_window,
    )
    from case_studies.utils.backtest_loaders import load_backtest_prices
    from utils.cv_splits import generate_cv_splits

    lo, hi = development_window(setup)
    grid = load_backtest_prices(CASE_STUDY_ID, start_date=lo, end_date=hi)
    bars = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    )
    if bars.schema["timestamp"].time_zone is not None:
        bars = bars.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    panel = session_panel(bars, keep_context=True, verbose=False)
    assert_no_holdout(setup, grid, panel)

    labels_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{setup['labels']['primary']}.parquet"
    timeline = (
        pl.read_parquet(labels_path)
        .select(pl.col("timestamp").dt.date().alias("timestamp"))
        .unique()
        .sort("timestamp")
    )
    folds = [
        {
            "fold": int(split["fold"]),
            **{
                name: str(pd.Timestamp(split[name]).date())
                for name in ("train_start", "train_end", "val_start", "val_end")
            },
        }
        for split in generate_cv_splits(
            timeline,
            case_study_id=CASE_STUDY_ID,
            label_buffer=setup["labels"]["buffer"],
            date_col="timestamp",
        )
    ]
    return derive_hold_bars(
        grid,
        panel,
        label=label,
        horizons=setup["labels"]["horizons"],
        books=list(setup["decision"]["session_filter_values"]),
        folds=folds,
        pre_fold_disagreements=PRE_FOLD_BREAK_REGIME,
    )


@pytest.fixture(scope="module")
def realised(setup: dict) -> dict:
    """Run one ``register=False`` backtest per session book and return its closed trades.

    Deterministic by construction: a constant positive score on both metals at every decision
    instant of the book, so the ``fixed_threshold`` signal opens a position in every slot and the
    only thing that can decide when it closes is the ``TimeExit`` rule. ``register=False`` is not
    optional - this test writes no registry row, and the run log of this bot is still empty.
    """
    _require_data(setup)
    import time

    from case_studies.exness_gold_sess._features import session_of
    from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices
    from case_studies.utils.backtest_presets import build_backtest_spec
    from case_studies.utils.backtest_runner import observed_periods_per_year, run_backtest

    label = setup["labels"]["primary"]
    hold_bars = int(str(setup["labels"]["horizons"][label]).rstrip("Hh"))
    assert WINDOW[1] < str(setup["evaluation"]["holdout_start"]), "the probe window is validation"
    hold_by_book, derived_rows = _derive(setup, label)

    labels_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{label}.parquet"
    prices = load_backtest_prices(CASE_STUDY_ID, start_date=WINDOW[0], end_date=WINDOW[1])
    assert prices["timestamp"].max().date().isoformat() < str(
        setup["evaluation"]["holdout_start"]
    ), "the price window must not reach the holdout"
    frames = pl.read_parquet(labels_path).filter(
        (pl.col("timestamp") >= prices["timestamp"].min())
        & (pl.col("timestamp") <= prices["timestamp"].max())
    )
    instants = frames["timestamp"].unique().sort()
    sessions = pl.DataFrame({"timestamp": instants, "session": session_of(instants)})
    assert sessions["session"].null_count() == 0, "a decision instant resolved to no venue"

    case_config = get_backtest_config(CASE_STUDY_ID)
    bar_index = prices.select("timestamp").unique().sort("timestamp").with_row_index("bar_i")
    out: dict = {
        "hold_bars_declared": hold_bars,
        "hold_by_book": hold_by_book,
        "derived_rows": derived_rows,
        "prices": prices,
        "sessions": sessions,
        "labels": frames,
        "books": {},
    }
    for book in setup["decision"]["session_filter_values"]:
        predictions = (
            frames.join(sessions, on="timestamp", how="left")
            .filter(pl.col("session") == book)
            .select(
                "timestamp",
                "symbol",
                pl.lit(1.0).alias("y_score"),
                pl.col(label).alias("y_true"),
            )
        )
        signal = {
            "method": "fixed_threshold",
            "threshold": 0.0,
            "long_short": True,
            "session_filter": book,
        }
        # The DERIVED per-book hold, not the declared horizon: NY_EXIT_DECLARATION.md 1.2.
        risk = {"position_rules": [{"type": "time_exit", "bars": hold_by_book[book]}]}
        spec = build_backtest_spec(
            CASE_STUDY_ID,
            case_config,
            prices=prices,
            prediction_hash=f"hold_probe_{book}",
            initial_cash=case_config.initial_cash,
            signal=signal,
            risk=risk,
            chapter="16",
            label=label,
        )
        started = time.perf_counter()
        result = run_backtest(
            CASE_STUDY_ID,
            f"hold_probe_{book}",
            spec,
            prices=prices,
            predictions=predictions,
            label=label,
            register=False,
            initial_cash=case_config.initial_cash,
            calendar=case_config.calendar,
        )
        elapsed = time.perf_counter() - started
        trades = (
            result.engine_result.to_trades_dataframe()
            .join(
                bar_index.rename({"timestamp": "entry_time", "bar_i": "entry_i"}),
                on="entry_time",
                how="left",
            )
            .join(
                bar_index.rename({"timestamp": "exit_time", "bar_i": "exit_i"}),
                on="exit_time",
                how="left",
            )
            .with_columns(
                (pl.col("exit_i") - pl.col("entry_i")).alias("hold_bars"),
                (
                    (pl.col("exit_time").dt.epoch("s") - pl.col("entry_time").dt.epoch("s")) // 60
                ).alias("hold_minutes"),
                # next_bar_open fills at the OPEN of the row, which is one bar before its key.
                (pl.col("entry_time") - pl.duration(minutes=60)).alias("entry_price_ts"),
                (pl.col("exit_time") - pl.duration(minutes=60)).alias("exit_price_ts"),
            )
            .filter(pl.col("status") == "closed")
        )
        assert trades.height > 0, f"the {book} book opened no position; nothing was measured"
        out["books"][book] = {
            "trades": trades,
            "predictions": predictions,
            "seconds": elapsed,
            "observed_periods_per_year": observed_periods_per_year(result.daily_returns),
            "result_daily_returns": result.daily_returns,
            "metrics": dict(result.metrics),
            "engine_metrics": dict(result.engine_result.metrics),
            "daily_rows": result.daily_returns.height,
        }
        print(
            f"\n[{book}] {predictions.height} prediction rows -> {trades.height} closed positions "
            f"in {elapsed:.2f}s; daily_returns {result.daily_returns.height} rows, "
            f"observed_periods_per_year "
            f"{out['books'][book]['observed_periods_per_year']:.2f}"
        )
    return out


# ---------------------------------------------------------------------------------------------
# The declaration, inverted
# ---------------------------------------------------------------------------------------------
def test_setup_declares_that_the_grid_now_expresses_the_label(setup: dict) -> None:
    assert setup["backtest"]["price_grid_expresses_primary_label"] is True, (
        "if this is false again the H1 grid has been rolled back: invert this file rather than "
        "flipping the key"
    )
    assert setup["backtest"]["price_grid_fix"] == (
        "register_h1_close_keyed_grid_and_hold_with_position_time_exit"
    )
    # The hold mechanism is a position rule, so the two time_exit arms that counted DECISION SLOTS
    # are gone; a 10-bar arm on the H1 grid could never fire under an 8-bar baseline hold and
    # would still be counted as a trial by the Deflated Sharpe Ratio.
    arms = {arm["name"] for arm in setup["backtest"]["sweep"]["risk_controls"]["position"]}
    assert {"time_exit_2", "time_exit_10"}.isdisjoint(arms)
    assert "hold_2h" in arms
    # The pooled book is arithmetic, not a third engine run (declaration section 3.4).
    assert setup["backtest"]["sweep"]["session_books"] == ["london", "ny"]
    assert setup["backtest"]["sweep"]["pooled_book"]["method"] == "sleeve_sum"
    # 504 moved from the annualisation factor to the slot census; the return grid is daily.
    assert setup["evaluation"]["periods_per_year"] == 252
    assert setup["decision"]["slots_per_year"] == 504


def test_the_loader_serves_the_h1_grid_and_still_guards_the_panel(setup: dict) -> None:
    """The registry points at the H1 grid; the old panel loader is kept, and it still raises.

    Keeping the raising branch is the point: it is no longer on the run path, so it costs nothing,
    and it is the thing that stops a future reader re-registering the decision panel as the price
    grid without noticing what that does to the hold.
    """
    from case_studies.exness_gold_sess._features import (
        PriceGridCannotExpressLabel,
        load_session_panel,
    )
    from case_studies.utils.backtest_loaders import _PRICE_CONFIG

    assert _PRICE_CONFIG[CASE_STUDY_ID]["loader"] == "exness_gold_sess_h1"
    with pytest.raises(PriceGridCannotExpressLabel, match="8 h"):
        load_session_panel(setup, for_backtest=True)


def test_the_registered_grid_is_hourly_and_keyed_on_the_bar_close(setup: dict) -> None:
    """Two facts one query apart: the spacing is 60 minutes, and decisions land ON rows.

    Keyed on the bar *open* the grid would be just as hourly, and every decision instant would
    fall between two rows - so the second assertion is the one that pins the convention, and with
    it the fill price (``execution_price: open`` + ``execution_mode: next_bar`` fills at the open
    of the row after the decision, which on a close-keyed grid is the decision instant's own price).
    """
    _require_data(setup)
    from case_studies.utils.backtest_loaders import load_backtest_prices

    label = setup["labels"]["primary"]
    labels_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{label}.parquet"
    prices = load_backtest_prices(CASE_STUDY_ID, start_date=WINDOW[0], end_date=WINDOW[1])
    gaps = (
        prices.sort(["symbol", "timestamp"])
        .with_columns(
            (
                (
                    pl.col("timestamp").shift(-1).over("symbol").dt.epoch("s")
                    - pl.col("timestamp").dt.epoch("s")
                )
                // 60
            ).alias("gap")
        )
        .drop_nulls("gap")
    )
    modal = int(gaps.group_by("gap").len().sort("len", descending=True)["gap"][0])
    assert modal == 60, f"the registered grid is not hourly; modal spacing {modal} minutes"

    keys = prices.select("timestamp", "symbol").unique()
    decisions = (
        pl.read_parquet(labels_path)
        .select("timestamp", "symbol")
        .unique()
        .filter(
            (pl.col("timestamp") >= prices["timestamp"].min())
            & (pl.col("timestamp") <= prices["timestamp"].max())
        )
    )
    off_grid = decisions.join(keys, on=["timestamp", "symbol"], how="anti")
    print(
        f"\ngrid {prices.height:,} rows, modal spacing {modal} min; "
        f"{decisions.height:,} decision instants in window, {off_grid.height} off the grid"
    )
    assert off_grid.height == 0, (
        f"{off_grid.height} decision instants are not rows of the price grid "
        f"(first: {off_grid.head(3).to_dicts()}): the grid is keyed on the bar OPEN and every "
        "fill would be an hour late"
    )


# ---------------------------------------------------------------------------------------------
# The realised hold - measured on trades, never read from config
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sealed_panel(setup: dict) -> pl.DataFrame:
    """The sealed session panel over the development window: the endpoint every claim is against.

    ``label_end_ts`` is the close of the last H1 bar closing at or before the venue's session
    close - the very instant ``02_labels`` computed ``fwd_ret_8h`` over - and ``label_end_close``
    is the price at it. Neither is read from the trades; that is the point.
    """
    _require_data(setup)
    from bots._shared.mt5_loader import load_mt5_bars
    from case_studies.exness_gold_sess._features import session_panel
    from case_studies.exness_gold_sess._hold import assert_no_holdout, development_window

    lo, hi = development_window(setup)
    bars = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    )
    if bars.schema["timestamp"].time_zone is not None:
        bars = bars.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    panel = session_panel(bars, keep_context=True, verbose=False)
    assert_no_holdout(setup, panel)
    return panel.select(
        pl.col("timestamp").alias("decision_ts"),
        "symbol",
        "session",
        "label_end_ts",
        "label_end_close",
    )


def _join_sealed(trades: pl.DataFrame, panel: pl.DataFrame, book: str) -> pl.DataFrame:
    """Attach each trade to the sealed decision it came from, and difference the two instants.

    The entry fill lands on the row keyed ``decision + 60min``, so ``entry_price_ts`` recovers the
    decision instant; ``exit_price_ts`` recovers the exit fill instant the same way. That second
    identity is an assumption about the engine until
    ``test_the_exit_fill_price_is_measured_against_the_grid_row_not_assumed`` measures the exit
    PRICE against the grid row's open - which is exactly why that test exists.
    """
    joined = trades.join(
        panel.filter(pl.col("session") == book),
        left_on=["entry_price_ts", "symbol"],
        right_on=["decision_ts", "symbol"],
        how="left",
    )
    unmatched = joined.filter(pl.col("session").is_null())
    assert unmatched.height == 0, (
        f"{book}: {unmatched.height} fills did not match a decision of the sealed panel "
        f"(first {unmatched.select('symbol', 'entry_time').head(3).to_dicts()})"
    )
    return joined.with_columns(
        (
            (pl.col("label_end_ts").dt.epoch("s") - pl.col("exit_price_ts").dt.epoch("s")) // 60
        ).alias("label_end_minus_exit_fill")
    )


def test_the_hold_constant_is_derived_and_not_typed(setup: dict, realised: dict) -> None:
    """Point 1 of declaration section 5: the constant is an OUTPUT of a rule.

    ``_hold.derive_hold_bars`` re-runs the section-1.2 rule over the whole development window and
    raises if ``i1 - i0`` is not one number inside a book across both metals, both seasons and
    every fold. The two assertions here are the ones that make a silent regression impossible:
    London must equal the declared label horizon, and New York must be **strictly shorter**. If
    the broker ever drops its daily break, New York would come out at 8 as well and this turns
    red rather than quietly trading an hour it never measured.
    """
    label = setup["labels"]["primary"]
    declared = int(str(setup["labels"]["horizons"][label]).rstrip("Hh"))
    hold = realised["hold_by_book"]
    rows = realised["derived_rows"]
    print(f"\nDERIVED HOLD_BARS = {hold} (declared {label} horizon: {declared} bars)")
    print(
        rows.filter(pl.col("fold") != -2)
        .group_by(["session", "symbol", "season"])
        .agg(
            pl.col("hold_bars").unique().sort().alias("hold_bars"),
            pl.col("exit_gap_minutes").unique().sort().alias("gap_min"),
            pl.len().alias("rows"),
        )
        .sort(["session", "symbol", "season"])
    )
    assert set(hold) == set(setup["decision"]["session_filter_values"])
    assert hold["london"] == declared == 8, (
        f"London is the continuous-tape case and must reduce to the declared horizon; got {hold}"
    )
    assert hold["ny"] < hold["london"], (
        f"the New York hold is not shorter than London's ({hold}). Either the broker's daily "
        "break no longer coincides with the New York session close, or the derivation is reading "
        "a grid that has the break bar in it - re-read NY_EXIT_DECLARATION.md section 0."
    )
    for book, bars in hold.items():
        assert bars <= declared, f"{book}: a hold may not run past the label window ({bars})"


def test_the_realised_hold_is_the_derived_constant_on_every_position(
    setup: dict, realised: dict
) -> None:
    """Point 2: mechanism. ``hold_bars``, ``bars_held`` and ``exit_reason``, both books, 100 %.

    Measured on the trades of a real ``register=False`` backtest, never read from config. 5 and
    19 - the holds the session panel could express - are dead, and so is the 8-bar New York hold
    that used to fill after the break.
    """
    hold = realised["hold_by_book"]
    for book, measured in realised["books"].items():
        trades = measured["trades"]
        expected = hold[book]
        distribution = trades.group_by("hold_bars").len().sort("hold_bars").to_dicts()
        engine_held = trades.group_by("bars_held").len().sort("bars_held").to_dicts()
        print(
            f"\n[{book}] expected {expected} bars; grid hold_bars {distribution}; "
            f"engine bars_held {engine_held}"
        )
        assert trades.filter(pl.col("hold_bars") != expected).height == 0, (
            f"{book}: {distribution} - the realised hold is not {expected} price-grid bars on "
            "100 % of positions"
        )
        assert trades.filter(pl.col("bars_held") != expected).height == 0, engine_held
        assert set(trades["exit_reason"].unique()) == {"time_stop"}, (
            f"{book}: something other than the TimeExit rule closed a position "
            f"({trades.group_by('exit_reason').len().to_dicts()})"
        )


def test_both_books_exit_at_the_last_fill_instant_before_their_label_endpoint(
    setup: dict, realised: dict, sealed_panel: pl.DataFrame
) -> None:
    """Point 3, the main proposition, on the realised FILL INSTANT of both books.

    Criterion 1.4 of the declaration, generalised from the "480 minutes" of section 3.2 that the
    New York tape cannot satisfy: the exit fill instant is the **latest fill instant at or before
    the sealed** ``label_end_ts``, and the shortfall is at most one grid bar on 100 % of positions
    that carry an endpoint. The two numbers are *printed*, and compared with what the section-1.2
    rule predicted for that book rather than with a literal - so London's 0 and New York's 60 are
    each proved twice, once by the rule and once by the engine, and neither is written here.
    """
    hold = realised["hold_by_book"]
    predicted = {
        book: int(
            realised["derived_rows"]
            .filter((pl.col("session") == book) & (pl.col("fold") != -2))
            .get_column("exit_gap_minutes")
            .unique()
            .item()
        )
        for book in hold
    }
    declared = setup["backtest"]["exit_gap_minutes_by_book"]
    for book, measured in realised["books"].items():
        joined = _join_sealed(measured["trades"], sealed_panel, book)
        with_endpoint = joined.filter(pl.col("label_end_ts").is_not_null())
        exempt = joined.height - with_endpoint.height
        offsets = (
            with_endpoint.group_by("label_end_minus_exit_fill")
            .len()
            .sort("len", descending=True)
            .to_dicts()
        )
        print(
            f"\n[{book}] label_end_ts - exit_fill_instant (minutes): {offsets}; "
            f"{exempt} position(s) exempt for a null endpoint; rule predicted "
            f"{predicted[book]}, setup.yaml declares {declared['by_book'][book]}"
        )
        assert with_endpoint.height > 0, f"{book}: nothing with a sealed endpoint was measured"
        bad = with_endpoint.filter(
            (pl.col("label_end_minus_exit_fill") < 0)
            | (pl.col("label_end_minus_exit_fill") > 60)
        )
        assert bad.height == 0, (
            f"{book}: {bad.height} exits fall outside 0 <= label_end_ts - exit_fill_instant <= 60 "
            f"minutes ({offsets}). An exit past the endpoint crosses a closure the label never "
            "contained."
        )
        realised_gap = sorted(with_endpoint["label_end_minus_exit_fill"].unique().to_list())
        assert realised_gap == [predicted[book]], (
            f"{book}: the engine realises {realised_gap} minutes of shortfall and the section-1.2 "
            f"rule predicts {predicted[book]}. The rule and the engine disagree, so one of them "
            "is not describing this tape."
        )
        assert realised_gap[0] == int(declared["by_book"][book]), (
            f"{book}: setup.yaml::backtest.exit_gap_minutes_by_book says "
            f"{declared['by_book'][book]} and the engine realises {realised_gap[0]}"
        )
        assert (
            with_endpoint.filter(pl.col("hold_minutes") != 60 * hold[book]).height
            <= with_endpoint.height
        )
    assert declared["derivation"] == "last_grid_fill_at_or_before_label_end"


def test_no_position_of_either_book_crosses_a_closure(
    setup: dict, realised: dict, sealed_panel: pl.DataFrame
) -> None:
    """Point 4, the weekend-carry killer, stated on the CLOCK and not on the calendar.

    ``hold_minutes == 60 * hold_bars`` says that no closed interval of the tape - the nightly
    break, a weekend, a holiday - fell inside the hold, because if one had, the same number of
    grid rows would have spanned more clock minutes. The declared early-close set is exempt from
    this and only from this; it is still asserted on the bar count by the test above.

    The count of positions held longer than a day is printed as well as asserted, so a regression
    is visible as a number rather than only as a red test. Before the New York fix it was 368 of
    1,984 over the full validation window (Friday entries carried to Sunday 22:00 UTC, three
    nights of swap and a weekend gap the label never contained); it is 0 now.
    """
    hold = realised["hold_by_book"]
    for book, measured in realised["books"].items():
        joined = _join_sealed(measured["trades"], sealed_panel, book)
        exempt = joined.filter(pl.col("label_end_ts").is_null())
        checked = joined.filter(pl.col("label_end_ts").is_not_null())
        minutes = checked.group_by("hold_minutes").len().sort("len", descending=True).to_dicts()
        overnight = checked.filter(pl.col("hold_minutes") > 24 * 60)
        print(
            f"\n[{book}] hold_minutes {minutes}; {exempt.height} early-close position(s) exempt "
            f"{exempt.select('symbol', 'entry_time', 'hold_minutes').to_dicts()}; "
            f"positions held longer than a day: {overnight.height} of {checked.height}"
        )
        assert checked.filter(pl.col("hold_minutes") != 60 * hold[book]).height == 0, (
            f"{book}: {minutes} - a closure fell inside a hold of {hold[book]} bars"
        )
        assert overnight.height == 0, (
            f"{book}: {overnight.height} positions are carried more than 24 hours "
            f"({overnight.group_by(pl.col('entry_time').dt.weekday()).len().to_dicts()}). The "
            "weekend carry is back."
        )


def test_the_exit_fill_price_is_measured_against_the_grid_row_not_assumed(
    setup: dict, realised: dict, sealed_panel: pl.DataFrame
) -> None:
    """Point 5: the gap nobody in this repository had measured - the fill PRICE, not the instant.

    Every instant test in this file derives ``exit_price_ts = exit_time - 60min`` from the belief
    that ``execution_price: open`` + ``execution_mode: next_bar`` fills at the **open** of the
    exit row. If the engine actually filled at that row's **close**, "0 minutes of difference"
    would be a consequence of the assumption rather than a measurement of the engine.

    So this joins the trade's realised ``exit_price`` to the registered grid on
    ``(symbol, exit_time)`` and *measures* which column it is, with and without the declared
    slippage. The answer is printed and asserted; for London the same join must also land on the
    sealed panel's ``label_end_close``.
    """
    from case_studies.utils.backtest_loaders import get_backtest_config

    slippage_bps = float(get_backtest_config(CASE_STUDY_ID).slippage_bps)
    grid = realised["prices"].select("timestamp", "symbol", "open", "close")
    verdicts: dict[str, str] = {}
    for book, measured in realised["books"].items():
        joined = (
            measured["trades"]
            .join(
                grid.rename({"timestamp": "exit_time"}),
                on=["exit_time", "symbol"],
                how="left",
            )
            .with_columns(
                (pl.col("exit_price") / pl.col("open") - 1.0).alias("vs_open"),
                (pl.col("exit_price") / pl.col("close") - 1.0).alias("vs_close"),
            )
        )
        assert joined.filter(pl.col("open").is_null()).height == 0, (
            f"{book}: an exit_time is not a row of the registered grid"
        )
        summary = {
            "max |exit_price/open - 1| bps": float(joined["vs_open"].abs().max()) * 1e4,
            "max |exit_price/close - 1| bps": float(joined["vs_close"].abs().max()) * 1e4,
            "median exit_price/open - 1 bps": float(joined["vs_open"].median()) * 1e4,
            "median exit_price/close - 1 bps": float(joined["vs_close"].median()) * 1e4,
            "declared slippage bps": slippage_bps,
        }
        print(f"\n[{book}] exit fill price against the grid row: {summary}")
        on_open = float(joined["vs_open"].abs().max()) * 1e4
        on_close = float(joined["vs_close"].abs().max()) * 1e4
        verdicts[book] = "open" if on_open < on_close else "close"
        assert verdicts[book] == "open", (
            f"{book}: the engine fills at the exit row's {verdicts[book]}, not its open "
            f"({summary}). Every instant assertion in this file derives exit_price_ts = "
            "exit_time - 60min from the opposite belief and would have to be re-derived."
        )
        assert on_open <= max(2.0 * slippage_bps, 1e-6) + 1e-6, (
            f"{book}: exit_price differs from the grid open by up to {on_open:.4f} bps, more than "
            f"the round trip of the declared {slippage_bps} bps slippage - the engine is applying "
            f"something this bot has not declared ({summary})"
        )

    london = _join_sealed(realised["books"]["london"]["trades"], sealed_panel, "london")
    with_endpoint = london.filter(pl.col("label_end_close").is_not_null())
    diff_bps = (
        (with_endpoint["exit_price"] / with_endpoint["label_end_close"] - 1.0).abs().max() * 1e4
    )
    print(
        f"[london] exit_price against the sealed panel's label_end_close: max "
        f"{diff_bps:.4f} bps over {with_endpoint.height} positions"
    )
    assert diff_bps <= max(2.0 * slippage_bps, 1e-6) + 1e-6, (
        "the London exit price is not the sealed label endpoint's close; the same join that "
        "proves the fill instant has to prove the fill price"
    )


def test_the_early_close_set_is_printed_never_filtered(
    setup: dict, realised: dict, sealed_panel: pl.DataFrame
) -> None:
    """Point 6: the declared early-close residual, counted against BOT.md and printed row by row.

    These are the sessions the broker closed early, so ``label_end_ts`` is null and no
    ``fwd_ret_8h`` exists. They are **traded normally** (``PRICE_GRID_DECLARATION.md`` section
    3.3: filtering on a null endpoint is a point-in-time violation, and the honest repair is a
    published holiday calendar, which this repository does not have yet). They are exempt from
    the two clock assertions and from nothing else.
    """
    from case_studies.utils.backtest_loaders import load_backtest_prices_for

    prices = load_backtest_prices_for(
        CASE_STUDY_ID, setup["labels"]["primary"], split="validation"
    )
    lo, hi = prices["timestamp"].min(), prices["timestamp"].max()
    early = (
        sealed_panel.filter(pl.col("label_end_ts").is_null())
        .filter((pl.col("decision_ts") >= lo) & (pl.col("decision_ts") <= hi))
        .select("symbol", "session", "decision_ts")
        .sort(["symbol", "decision_ts"])
    )
    per_symbol = early.group_by("symbol").len().sort("symbol").to_dicts()
    per_book = early.group_by("session").len().sort("session").to_dicts()
    print(
        f"\nEarly-close decision rows in the validation price window {lo} -> {hi}: "
        f"{early.height} ({per_symbol}, {per_book})"
    )
    with pl.Config(tbl_rows=min(max(early.height, 1), 80)):
        print(early)
    # RE-MEASURED AND RE-RECORDED 2026-09-08, after the A-prime re-publication of the three label
    # parquets and the clean 04 -> 05 -> 06 -> 07 pass (RULINGS_2026-09-08.md execution step 8).
    # The recorded count was 66 (33 a metal), measured BEFORE A-prime; A-prime publishes every
    # label on the decision grid's key set instead of dropping its nulls, so two more early-close
    # New York decisions are now inside the validation price window. Measured today over
    # 2021-09-01 01:00 -> 2025-08-29 21:00: 68 rows, 34 a metal, 34 distinct dates, all New York.
    # The number is re-recorded rather than relaxed: it sizes the exemption the two clock tests
    # grant, so it has to be a count and not a bound.
    assert early.height == 68, (
        f"BOT.md records 68 early-close rows in the validation window (re-measured 2026-09-08), "
        f"34 a metal, all New York; this run measures {early.height} ({per_symbol}, {per_book}). "
        "Re-measure and re-record before running anything - the count is what the exemption in "
        "the clock tests is sized on."
    )
    assert {row["len"] for row in per_symbol} == {34}, per_symbol
    assert per_book == [{"session": "ny", "len": 68}], per_book


# ---------------------------------------------------------------------------------------------
# The 24-hour book: the one exit nobody had measured
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def realised_24h(setup: dict) -> dict:
    """The same harness as ``realised``, on ``fwd_ret_24h`` with its declared Friday rule.

    ``NY_EXIT_DECLARATION.md`` section 6.1: this book has a crippled version of the New York
    problem and it had never been measured. Under ``drop_friday`` the Thursday position has no
    Friday weight to overwrite it, so what closes it is the **backstop** - and 24 grid rows are
    about 25 clock hours, because every day of grid swallows one hour at the break. The backstop
    is therefore derived by the section-1.2 rule under ``derive_backstop_bars`` rather than typed,
    and this measures what the engine then really does with it.
    """
    _require_data(setup)
    import pandas as pd

    from bots._shared.mt5_loader import load_mt5_bars
    from case_studies.exness_gold_sess._features import session_of, session_panel
    from case_studies.exness_gold_sess._hold import (
        assert_no_holdout,
        derive_backstop_bars,
        development_window,
    )
    from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices
    from case_studies.utils.backtest_presets import build_backtest_spec
    from case_studies.utils.backtest_runner import run_backtest
    from utils.cv_splits import generate_cv_splits

    label = "fwd_ret_24h"
    labels_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{label}.parquet"
    if not labels_path.exists():
        pytest.skip("02_labels has not written fwd_ret_24h in this output directory")

    lo, hi = development_window(setup)
    dev_grid = load_backtest_prices(CASE_STUDY_ID, start_date=lo, end_date=hi)
    bars = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    )
    if bars.schema["timestamp"].time_zone is not None:
        bars = bars.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    panel = session_panel(bars, keep_context=True, verbose=False)
    assert_no_holdout(setup, dev_grid, panel)
    primary_labels = pl.read_parquet(
        get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{setup['labels']['primary']}.parquet"
    )
    timeline = (
        primary_labels.select(pl.col("timestamp").dt.date().alias("timestamp"))
        .unique()
        .sort("timestamp")
    )
    folds = [
        {
            "fold": int(split["fold"]),
            **{
                name: str(pd.Timestamp(split[name]).date())
                for name in ("train_start", "train_end", "val_start", "val_end")
            },
        }
        for split in generate_cv_splits(
            timeline,
            case_study_id=CASE_STUDY_ID,
            label_buffer=setup["labels"]["buffer"],
            date_col="timestamp",
        )
    ]
    backstop, derived_rows = derive_backstop_bars(
        dev_grid,
        panel,
        label=label,
        horizons=setup["labels"]["horizons"],
        books=list(setup["decision"]["session_filter_values"]),
        folds=folds,
    )

    case_config = get_backtest_config(CASE_STUDY_ID)
    # A whole day of hold needs more than one quarter of grid to leave closed positions behind.
    prices = load_backtest_prices(CASE_STUDY_ID, start_date=WINDOW[0], end_date=WINDOW[1])
    frames = pl.read_parquet(labels_path).filter(
        (pl.col("timestamp") >= prices["timestamp"].min())
        & (pl.col("timestamp") <= prices["timestamp"].max())
    )
    instants = frames["timestamp"].unique().sort()
    sessions = pl.DataFrame({"timestamp": instants, "session": session_of(instants)})
    bar_index = prices.select("timestamp").unique().sort("timestamp").with_row_index("bar_i")
    out: dict = {"backstop": backstop, "derived_rows": derived_rows, "books": {}}
    for book in setup["decision"]["session_filter_values"]:
        predictions = (
            frames.join(sessions, on="timestamp", how="left")
            .filter(pl.col("session") == book)
            .select(
                "timestamp", "symbol", pl.lit(1.0).alias("y_score"), pl.col(label).alias("y_true")
            )
        )
        signal = {
            "method": "fixed_threshold",
            "threshold": 0.0,
            "long_short": True,
            "session_filter": book,
            "drop_friday": True,
        }
        risk = {"position_rules": [{"type": "time_exit", "bars": backstop[book]}]}
        spec = build_backtest_spec(
            CASE_STUDY_ID,
            case_config,
            prices=prices,
            prediction_hash=f"backstop_probe_{book}",
            initial_cash=case_config.initial_cash,
            signal=signal,
            risk=risk,
            chapter="16",
            label=label,
        )
        result = run_backtest(
            CASE_STUDY_ID,
            f"backstop_probe_{book}",
            spec,
            prices=prices,
            predictions=predictions,
            label=label,
            register=False,
            initial_cash=case_config.initial_cash,
            calendar=case_config.calendar,
        )
        trades = (
            result.engine_result.to_trades_dataframe()
            .join(
                bar_index.rename({"timestamp": "entry_time", "bar_i": "entry_i"}),
                on="entry_time",
                how="left",
            )
            .join(
                bar_index.rename({"timestamp": "exit_time", "bar_i": "exit_i"}),
                on="exit_time",
                how="left",
            )
            .with_columns(
                (pl.col("exit_i") - pl.col("entry_i")).alias("hold_bars"),
                (
                    (pl.col("exit_time").dt.epoch("s") - pl.col("entry_time").dt.epoch("s")) // 60
                ).alias("hold_minutes"),
                (pl.col("entry_time") - pl.duration(minutes=60)).alias("entry_price_ts"),
                (pl.col("exit_time") - pl.duration(minutes=60)).alias("exit_price_ts"),
            )
            .filter(pl.col("status") == "closed")
        )
        out["books"][book] = {"trades": trades, "predictions": predictions}
    return out


def test_the_twenty_four_hour_backstop_is_derived_and_its_exit_is_measured(
    setup: dict, realised_24h: dict
) -> None:
    """Declaration section 6.1, both halves: derive the backstop, then measure the real exit.

    ``13_backtest`` used to type 24 for this book. The section-1.2 rule says **23**: a day of
    clock is 24 grid steps and the recurring break removes one of them, so ``i1 - i0`` on a tape
    whose only closure is that break is 23 - measured on 2,770 of 2,870 sealed endpoints in each
    book, with a 0-minute exit gap and 1,440 clock minutes on every one of them.

    What this then asserts on realised fills, which is the half nobody had run:

    * no Friday decision reaches the engine (``drop_friday`` is the declared weekday rule);
    * the realised hold is the derived backstop on 100 % of closed positions, and the exit reason
      is the time stop - so the backstop really is what closes this book, exactly as section 6.1
      says it is "satisfied by accident rather than by design";
    * no position is carried over a weekend.
    """
    backstop = realised_24h["backstop"]
    print(f"\nDERIVED fwd_ret_24h backstop (was a typed 24): {backstop}")
    assert set(backstop.values()) == {23}, (
        f"the section-1.2 rule no longer gives 23 bars for the 24-hour book ({backstop}); the "
        "recurring break is the only closure it is supposed to absorb"
    )
    for book, measured in realised_24h["books"].items():
        trades = measured["trades"]
        weekdays = sorted(
            measured["predictions"]["timestamp"].dt.weekday().unique().to_list()
        )
        bars_dist = trades.group_by("hold_bars").len().sort("hold_bars").to_dicts()
        minutes_dist = (
            trades.group_by("hold_minutes").len().sort("len", descending=True).to_dicts()
        )
        # RE-MEASURED AND RE-STATED 2026-09-08 (RULINGS_2026-09-08.md execution step 8). The
        # `> 48 h` proxy was standing in for "carried over a weekend", and on the A-prime panel it
        # catches something else: exactly one New York position, XAUUSD entered 2024-12-24 15:00
        # and exited 2024-12-26 18:00 - 23 bars, 3,060 minutes - which crosses **Christmas Day**,
        # not a weekend. The tape does not print on 25 December and this repository has no
        # published holiday calendar (PRICE_GRID_DECLARATION.md section 3.3), so the grid has no
        # way to know. The proxy is therefore replaced by the proposition it stood for - does a
        # Saturday or a Sunday fall inside the hold - and the holiday carry is counted separately
        # and pinned at its measured value so it cannot grow in silence. It is an instance of this
        # bot's own kill criterion (g), "a symbol halts or gaps across a closure that was NOT
        # declared while the bot is holding", and it is recorded in BOT.md open questions.
        long_holds = trades.filter(pl.col("hold_minutes") > 48 * 60)
        weekend = trades.filter(
            pl.int_ranges(
                pl.col("entry_time").dt.date().dt.epoch("d") + 1,
                pl.col("exit_time").dt.date().dt.epoch("d") + 1,
            )
            # 1970-01-01 was a THURSDAY, so epoch-day 0 is weekday index 3 with Monday = 0;
            # Saturday and Sunday are 5 and 6. Verified against the measured output: with the
            # off-by-one (+4) this flagged 11 London positions whose hold is exactly 1,440
            # minutes, which cannot cross a weekend at all.
            .list.eval((pl.element() + 3) % 7 >= 5)
            .list.any()
        )
        holiday_carry = long_holds.join(
            weekend.select("entry_time", "symbol"), on=["entry_time", "symbol"], how="anti"
        )
        print(
            f"[{book}] long holds (> 48 h) {long_holds.height}; of those crossing a weekend "
            f"{weekend.height}, crossing a NON-WEEKEND closure {holiday_carry.height}"
        )
        if holiday_carry.height:
            print(holiday_carry.select("symbol", "entry_time", "exit_time", "hold_minutes"))
        print(
            f"[{book}] {measured['predictions'].height} prediction rows on weekdays {weekdays} -> "
            f"{trades.height} closed positions; hold_bars {bars_dist}; hold_minutes "
            f"{minutes_dist}; carried over a weekend {weekend.height}"
        )
        # RE-STATED AND RE-MEASURED 2026-09-08 (RULINGS_2026-09-08.md execution step 8). Until
        # A-prime the label parquet held only rows whose label was defined, so "no Friday reaches
        # the engine" could be read off the PREDICTION FRAME. A-prime publishes every decision
        # instant with a null where the label is undefined, so the frame now carries 964 London
        # and 966 New York Friday rows - and **0 of them carry a non-null fwd_ret_24h** (measured
        # today). The old assertion was therefore testing the label parquet's null policy, not
        # `drop_friday`. What has to be true, and is now asserted on realised fills instead, is
        # that `drop_friday` keeps every Friday decision out of the BOOK.
        friday_rows = measured["predictions"].filter(
            pl.col("timestamp").dt.weekday() == 5
        )
        friday_labelled = friday_rows.filter(pl.col("y_true").is_not_null())
        print(
            f"[{book}] Friday prediction rows {friday_rows.height}, of which carrying a non-null "
            f"fwd_ret_24h {friday_labelled.height} (A-prime pads the parquet; 0 expected)"
        )
        assert friday_labelled.is_empty(), (
            "a Friday decision carries a sealed fwd_ret_24h - the label's own weekday rule moved"
        )
        friday_entries = trades.filter(pl.col("entry_time").dt.weekday() == 5)
        assert friday_entries.is_empty(), (
            f"{book}: {friday_entries.height} closed positions of the 24-hour book were entered "
            "on a Friday, so drop_friday did not remove them from the book"
        )
        assert trades.height > 0, f"the {book} 24-hour book closed nothing"
        assert trades.filter(pl.col("hold_bars") != backstop[book]).height == 0, (
            f"{book}: the realised hold is not the derived backstop of {backstop[book]} bars "
            f"({bars_dist})"
        )
        assert set(trades["exit_reason"].unique()) == {"time_stop"}, (
            f"{book}: something other than the backstop closed a 24-hour position "
            f"({trades.group_by('exit_reason').len().to_dicts()})"
        )
        assert weekend.height == 0, (
            f"{book}: {weekend.height} positions of the 24-hour book are carried across a weekend "
            "- the Friday rule was supposed to make that impossible"
        )
        expected_holiday_carry = {"london": 0, "ny": 1}[book]
        assert holiday_carry.height == expected_holiday_carry, (
            f"{book}: {holiday_carry.height} positions are held across a closure that is not a "
            f"weekend; {expected_holiday_carry} was measured on 2026-09-08 (the ny one is "
            "XAUUSD over Christmas Day 2024). A change here is kill criterion (g) moving, not a "
            "number to re-record without reading BOT.md open questions."
        )


# ---------------------------------------------------------------------------------------------
# calendar.data_frequency: an inert key, asserted so that it can no longer lie
# ---------------------------------------------------------------------------------------------
def test_what_the_engine_resolves_from_calendar_data_frequency_is_measured(
    setup: dict, realised: dict
) -> None:
    """NY_EXIT_DECLARATION.md section 4.a: keep the value, and make it a CHECKED fact.

    The declaration's premise was that ``calendar.data_frequency`` is decorative - that the
    engine would resolve ``feed.data_frequency`` from the cadence and land on ``daily``. Half of
    that is true and this test records which half:

    * ``_infer_data_frequency("session")`` really is ``"daily"``, and no line of *this repository*
      reads the ``calendar`` key;
    * but ``EngineBacktestConfig`` records ``data_frequency`` as **explicitly provided** when the
      preset's calendar block carries it (``ml4t/backtest/config.py:774``), declines to overwrite
      it from the feed spec (``:788-790``) and pushes it back over the feed in
      ``resolved_feed_spec`` (``:820-822``). The resolved feed is therefore ``1h``.

    So the key is the authority, not decoration, and the two things it could move are asserted
    here instead of assumed: ``enforce_sessions`` is off (so no bar is skipped by it), and the
    annualisation of everything **this repository registers** still comes from
    ``evaluation.periods_per_year`` on the DAILY return frame.
    """
    from case_studies.utils.backtest_loaders import get_backtest_config
    from case_studies.utils.backtest_presets import _infer_data_frequency, build_backtest_spec
    from case_studies.utils.backtest_runner import reconcile_periods_per_year

    case_config = get_backtest_config(CASE_STUDY_ID)
    spec = build_backtest_spec(
        CASE_STUDY_ID,
        case_config,
        prices=realised["prices"],
        prediction_hash="frequency_probe",
        initial_cash=case_config.initial_cash,
        signal={"method": "fixed_threshold", "threshold": 0.0, "session_filter": "london"},
        chapter="16",
        label=setup["labels"]["primary"],
    )
    backtest_config = spec["backtest_config"]
    declared = backtest_config["calendar"]["data_frequency"]
    resolved = backtest_config["feed"]["data_frequency"]
    inferred = _infer_data_frequency(case_config.cadence)
    enforce = backtest_config["calendar"].get("enforce_sessions", False)
    print(
        f"\ncalendar.data_frequency={declared!r}; cadence={case_config.cadence!r} would infer "
        f"{inferred!r}; the engine RESOLVES feed.data_frequency={resolved!r}; "
        f"enforce_sessions={enforce}"
    )
    assert declared == "1h", "base.yaml no longer declares the hourly grid it registers"
    assert inferred == "daily", (
        f"_infer_data_frequency({case_config.cadence!r}) is now {inferred!r}; the base.yaml "
        "comment reasons from 'daily' and has to be re-measured"
    )
    assert resolved == declared, (
        f"the engine resolves feed.data_frequency={resolved!r} while the calendar block declares "
        f"{declared!r}. ml4t/backtest/config.py:774,788-790,820-822 used to make the calendar "
        "block the authority; if that changed, the base.yaml comment is now wrong."
    )
    assert enforce is False, (
        "enforce_sessions is on, so feed.data_frequency now SKIPS BARS (engine.py:181,320). "
        "That is a behaviour change this bot has never measured."
    )

    declared_ppy = int(setup["evaluation"]["periods_per_year"])
    for book, measured in realised["books"].items():
        observed = measured["observed_periods_per_year"]
        kept = reconcile_periods_per_year(
            declared_ppy, measured["result_daily_returns"], case_study=CASE_STUDY_ID
        )
        engine_sharpe = measured["engine_metrics"].get("sharpe")
        registered_sharpe = measured["metrics"].get("sharpe")
        print(
            f"[{book}] daily_returns {measured['daily_rows']} rows, observed {observed:.2f} "
            f"periods/yr against the declared {declared_ppy}; reconciler keeps {kept}. "
            f"engine_result.metrics['sharpe']={engine_sharpe} (annualised by the engine at the "
            f"feed frequency, NOT registered) against run_backtest metrics['sharpe']="
            f"{registered_sharpe} (annualised here on the daily grid, REGISTERED)"
        )
        assert kept == declared_ppy, (
            f"{book}: reconcile_periods_per_year moved the annualisation to {kept} from the "
            f"declared {declared_ppy} (observed {observed:.2f}). The H1 price grid is not "
            "supposed to touch the DAILY return grid."
        )
        assert 0.5 <= observed / declared_ppy <= 2.0, (
            f"{book}: {observed:.2f} observed periods a year against {declared_ppy} declared - "
            "outside the band the reconciler keeps"
        )
        assert engine_sharpe is not None and registered_sharpe is not None
        assert engine_sharpe != registered_sharpe, (
            f"{book}: the engine's own Sharpe and the registered Sharpe are now identical "
            f"({engine_sharpe}). They were measured apart on 2026-09-08 precisely because the "
            "engine annualises on the feed frequency and this repository annualises on the "
            "daily return grid; if they have converged, one of the two paths has changed and "
            "the base.yaml comment has to be re-measured."
        )


# ---------------------------------------------------------------------------------------------
# The rebalance schedule, and why the pinned step is 1
# ---------------------------------------------------------------------------------------------
def test_the_rebalance_schedule_keeps_every_decision_instant(setup: dict, realised: dict) -> None:
    """The cheapest test that catches the quietest error in the declaration's table.

    ``labels.rebalance_step`` is a repository-pinned key that enters no hash, so a wrong value is
    reused silently for ever. At the pinned step of 1 the schedule of a one-venue book keeps every
    decision instant; the counterfactual at step 2 is asserted in the same test, because that is
    what makes the pin load-bearing rather than decorative:

    * on the ``london`` book, whose schedule is one slot a day, step 2 trades every OTHER day - a
      48-hour cadence for a 24-hour label;
    * on the pooled instant list ``[London d1, NY d1, London d2, ...]``, ``gather_every(2)`` keeps
      indices 0, 2, 4 ... which is **one venue a day**. Measured over the whole development panel
      (4,817 instants): at step 1 the schedule keeps all of them and 2,366 of 2,451 days carry
      both venues; at step 2 it keeps 2,409 and **0 days carry both** - every traded day is a
      single-venue day, and *which* venue drifts with the gaps in the tape (1,369 London against
      1,040 New York) rather than being chosen.

    RE-MEASURED 2026-09-08 on the A-prime panel (RULINGS_2026-09-08.md execution step 8). The
    counterfactual did not change its shape, only its numbers, because A-prime removed the gaps
    that used to thin the instant list: **4,905** pooled instants over **2,453** days; step 1
    keeps all 4,905 and **2,452 of 2,453 days carry both venues** (one day carries one); step 2
    keeps **2,453**, **0 days carry both**, and the surviving venue is **240 London against 2,213
    New York** - still drifting with the tape rather than chosen, and now lopsided rather than
    near-even. The pooled counterfactual is computed on the whole development instant list, which
    is what this docstring always described; computing it on the short `realised` probe window
    instead makes it collapse to one venue and says nothing about the pin. That is the
    nasdaq100_microstructure failure,
      "the values did not change; what they count did", and the reason a per-label step is
      unusable on this bot: no value but 1 is right on all three books.
    """
    from case_studies.exness_gold_sess._features import session_of
    from case_studies.utils.backtest_loaders import (
        get_backtest_config,
        get_rebalance_step,
        resolve_decision_schedule,
    )

    cadence = setup["decision"]["cadence"]
    calendar = get_backtest_config(CASE_STUDY_ID).calendar
    steps = {
        label: get_rebalance_step(CASE_STUDY_ID, label) for label in setup["labels"]["horizons"]
    }
    print(f"\ndeclared rebalance_step: {steps}")
    assert set(steps.values()) == {1}, (
        f"PRICE_GRID_DECLARATION.md section 2 pins the step to 1 on all three labels; got {steps}"
    )
    step = steps[setup["labels"]["primary"]]

    # The pooled counterfactual runs on the WHOLE development instant list, not on the `realised`
    # fixture's probe window: on 65 days `gather_every(2)` keeps London every time and the venue
    # mix says nothing (measured 2026-09-08). The sealed primary label parquet is that list.
    pooled = (
        pl.read_parquet(
            get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{setup['labels']['primary']}.parquet"
        )["timestamp"]
        .unique()
        .sort()
    )
    sessions = realised["sessions"]
    for book in setup["decision"]["session_filter_values"]:
        instants = (
            sessions.filter(pl.col("session") == book).get_column("timestamp").unique().sort()
        )
        kept = resolve_decision_schedule(instants, cadence, step, calendar)
        print(f"  {book}: {instants.len()} instants -> schedule keeps {kept.len()}")
        assert kept.len() == instants.len(), (
            f"{book}: the schedule dropped {instants.len() - kept.len()} decision instants at "
            "step 1"
        )
        thinned = resolve_decision_schedule(instants, cadence, 2, calendar)
        assert thinned.len() < instants.len(), (
            f"{book}: a step of 2 did not thin the schedule, so this test proves nothing about "
            "the pin"
        )

    kept_pooled = resolve_decision_schedule(pooled, cadence, step, calendar)
    assert kept_pooled.len() == pooled.len(), (
        f"the pooled schedule kept {kept_pooled.len()} of {pooled.len()} decision instants at "
        "step 1"
    )
    venues_kept = set(session_of(kept_pooled).unique().to_list())
    assert venues_kept == set(setup["decision"]["session_filter_values"]), (
        f"the pooled schedule at step 1 covers only {venues_kept}"
    )
    def _days_carrying_both(schedule: pl.Series) -> tuple[int, int]:
        frame = pl.DataFrame({"ts": schedule, "venue": session_of(schedule)})
        per_day = frame.group_by(pl.col("ts").dt.date().alias("day")).agg(
            pl.col("venue").n_unique().alias("venues")
        )
        return (
            per_day.filter(pl.col("venues") == 2).height,
            per_day.filter(pl.col("venues") == 1).height,
        )

    both_at_1, one_at_1 = _days_carrying_both(kept_pooled)
    thinned_pooled = resolve_decision_schedule(pooled, cadence, 2, calendar)
    both_at_2, one_at_2 = _days_carrying_both(thinned_pooled)
    venue_mix = (
        pl.DataFrame({"venue": session_of(thinned_pooled)}).group_by("venue").len().sort("venue")
    )
    print(
        f"  pooled: step 1 keeps {kept_pooled.len()} instants over {both_at_1} two-venue days and "
        f"{one_at_1} one-venue days; step 2 would keep {thinned_pooled.len()} over {both_at_2} "
        f"two-venue days and {one_at_2} one-venue days, venue mix {venue_mix.to_dicts()}"
    )
    assert both_at_1 > one_at_1, (
        "at the pinned step of 1 the pooled schedule should carry both venues on most days; it "
        f"carries both on {both_at_1} and one on {one_at_1}"
    )
    assert both_at_2 == 0 and one_at_2 > 0, (
        "a step of 2 no longer reduces every traded day of the pooled schedule to a single venue "
        f"({both_at_2} two-venue days, {one_at_2} one-venue days); the reason the step is pinned "
        "to 1 has changed and the declaration has to be re-read"
    )
    assert len(set(venue_mix["venue"].to_list())) == 2, (
        "at step 2 the surviving venue no longer drifts between London and New York, so the "
        "schedule is now a property of the strategy rather than of the gaps in the tape - "
        f"re-read PRICE_GRID_DECLARATION.md section 2 (measured today: {venue_mix.to_dicts()}; "
        "re-measured 2026-09-08 as 240 London / 2,213 New York)"
    )
