"""Where each session book's hold comes from: **a rule**, not two typed constants.

Written 2026-09-08 from ``bots/exness_gold_sess/NY_EXIT_DECLARATION.md`` section 1.2, which
corrects sections 3.2 and 3.3 of ``PRICE_GRID_DECLARATION.md``. It is imported by
``case_studies/exness_gold_sess/13_backtest.py`` and by
``bots/exness_gold_sess/tests/test_backtest_grid.py`` so that the stage and the test cannot
disagree about the number - a second implementation is the Chapter 25 divergence in miniature.

**The fact the rule turns on.** The New York session close *is* the start of the broker's daily
one-hour break, in both seasons, because both are anchored on 17:00 America/New_York:
``bots/_shared/sessions.py`` gives ``new_york`` 08:00-17:00 local, and the break is 17:00-18:00
local (``bots/assets/XAUUSD.md:110,188``). Summer: close 21:00 UTC, break 21:00-22:00 UTC.
Winter: close 22:00 UTC, break 22:00-23:00 UTC. So it is *not* "the metals stop quoting between
21:00 and 22:00 UTC every day" - that sentence was written into a test docstring on 2026-09-08
and this bot's own BOT.md (Decisions log 2026-09-07, 630 of 997 session-days) refutes it.

**The rule** (declaration section 1.2), stated once and applied to both venues:

    ``HOLD_BARS[book]`` is the number of price-grid rows between the **entry row** and the
    **last row whose fill instant is at or before the sealed** ``label_end_ts``.

Precisely, on the ``bar_index`` of the registered grid (which is keyed on the BAR CLOSE):

* ``i0`` = the index of the first row whose key is ``> d`` - the row the entry fills at, because
  ``execution_price: open`` + ``execution_mode: next_bar`` fills at the *open* of that row, and
  the open of a close-keyed row is the price at ``key - 60min``;
* ``i1`` = the index of the last row whose key is ``<= label_end_ts + 60min``, i.e. whose fill
  instant ``key - 60min`` is at or before the endpoint;
* ``N = i1 - i0``.

On a continuous tape ``i1`` is the row keyed ``E + 60`` and ``N`` is the horizon in bars - which
is why London comes out at 8 with a zero-minute gap and nothing about it changes. In New York the
interval ``[E, E + 60]`` **is** the break, so that row does not exist in either season, ``i1`` is
the last print before the break, and ``N`` falls out one lower with a 60-minute gap. Nobody types
7 or 8 anywhere; both are printed by :func:`derive_hold_bars`.

**Execution constraints, from the declaration:**

* the derivation runs on the **sealed panel** and the **registered price grid** over
  ``universe.history_start -> evaluation.holdout_start`` only. It must never read the holdout;
  :func:`development_window` is the single place that boundary is expressed.
* rows whose ``label_end_ts`` is null (the declared early closes) are **excluded from the
  derivation** - they have no endpoint to measure against - but they still **trade** with the
  same constant. ``PRICE_GRID_DECLARATION.md`` section 3.3 is unchanged.
* there is no point-in-time violation: ``label_end_ts`` here is the *venue's session close*,
  known in advance from the calendar exactly like the Friday rule, and what enters the spec is
  **one constant per book**, never a per-row lookup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

#: One row of the registered price grid. The grid is hourly and keyed on the bar close.
BAR_MINUTES = 60

#: The clock whose daylight saving moves both the New York session close and the broker's daily
#: break, and therefore the only "season" this derivation can be a function of.
SEASON_ZONE = "America/New_York"

#: Written into ``setup.yaml::backtest.exit_gap_minutes_by_book.derivation`` so a later reader
#: finds the claim and the rule that produced it in the same place.
DERIVATION_KEY = "last_grid_fill_at_or_before_label_end"


def development_window(setup: Mapping) -> tuple[str, str]:
    """``(start, end)`` of the development window, as ``YYYY-MM-DD`` strings.

    The end is the **last day before** ``evaluation.holdout_start``. Every loader in this module
    is given this window and nothing else, so no derivation here can read the holdout even by
    accident.
    """
    from datetime import date

    holdout_start = str(setup["evaluation"]["holdout_start"])
    y, m, d = (int(part) for part in holdout_start.split("-"))
    last_dev_day = date(y, m, d) - timedelta(days=1)
    return str(setup["universe"]["history_start"]), last_dev_day.isoformat()


def assert_no_holdout(setup: Mapping, *frames: pl.DataFrame, column: str = "timestamp") -> None:
    """Refuse to derive anything from a frame that reaches into the holdout.

    The boundary is stated once, here, and every caller passes its frames through it. One
    deliberate subtlety: the registered grid is keyed on the **bar close**, so the row keyed
    ``holdout_start 00:00`` is the bar that ran 23:00-24:00 on the last development day. It is
    development data under a holdout-dated key, and the comparison is therefore ``<=`` midnight
    of ``holdout_start`` rather than ``<`` its date - which is exactly the kind of off-by-one a
    prose comment would hide.
    """
    boundary = pl.lit(str(setup["evaluation"]["holdout_start"])).str.to_datetime()
    for frame in frames:
        if frame.is_empty() or column not in frame.columns:
            continue
        past = frame.filter(pl.col(column) > boundary)
        if past.height:
            raise RuntimeError(
                f"{past.height} of {frame.height} rows are dated after "
                f"{setup['evaluation']['holdout_start']} (max "
                f"{frame.get_column(column).max()}): the hold derivation may not read the holdout"
            )


def _season(instants: pl.Series) -> pl.Series:
    """``"us_dst"`` / ``"us_std"`` per instant, from the venue's own clock, never from a table."""
    zone = ZoneInfo(SEASON_ZONE)
    return pl.Series(
        "season",
        [
            "us_dst" if ts.replace(tzinfo=UTC).astimezone(zone).dst() else "us_std"
            for ts in instants.to_list()
        ],
    )


def label_endpoints(
    panel: pl.DataFrame,
    *,
    label: str,
    horizons: Mapping[str, str],
) -> pl.DataFrame:
    """The (symbol, book, decision, sealed endpoint) frame the rule is measured over.

    Two label shapes, one function, because the *rule* is one rule and only the endpoint
    differs - which is the whole point of section 1.2:

    ``fwd_ret_8h`` / ``dir_tb_8h``
        the endpoint is the session panel's own ``label_end_ts``: the close of the last H1 bar
        that closes at or before the venue's session close. Null on an early close, and those
        rows are dropped here (declaration section 1.2) while still trading (section 3.3).
    ``fwd_ret_24h``
        the endpoint is the decision **two slots ahead**, kept only where that span is exactly
        24 hours - byte-for-byte the rule ``02_labels.py:231-240`` sealed the label with. Every
        other row (Friday, and the holiday gaps) is null there and is dropped here too.
    """
    horizon_minutes = int(str(horizons[label]).rstrip("Hh")) * 60
    frame = panel.select("symbol", "session", "timestamp", "label_end_ts").sort(
        ["symbol", "timestamp"]
    )
    if horizon_minutes <= 24 * 60 // 2:  # an intraday label: the session close is the endpoint
        endpoints = frame.rename({"timestamp": "decision_ts", "label_end_ts": "end_ts"})
    else:
        slots_ahead = horizon_minutes // (24 * 60 // 2)
        endpoints = (
            frame.with_columns(
                pl.col("timestamp").shift(-slots_ahead).over("symbol").alias("_ahead")
            )
            .with_columns(
                pl.when(
                    (pl.col("_ahead").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")) // 60
                    == horizon_minutes
                )
                .then(pl.col("_ahead"))
                .otherwise(None)
                .alias("end_ts")
            )
            .rename({"timestamp": "decision_ts"})
            .select("symbol", "session", "decision_ts", "end_ts")
        )
    return endpoints.drop_nulls("end_ts")


def measure_hold(grid: pl.DataFrame, endpoints: pl.DataFrame) -> pl.DataFrame:
    """Apply the section-1.2 rule row by row. Returns one row per measurable endpoint.

    Columns: ``symbol, session, decision_ts, end_ts, season, entry_key, entry_fill_instant,
    exit_key, exit_fill_instant, hold_bars, exit_gap_minutes, hold_minutes``.

    A row is *measurable* only when the loaded grid brackets it - there is a key after the
    decision and the grid reaches at least ``end_ts + 60min``. That excludes at most the last
    session or two of the load window, and it is a property of the window rather than of the
    tape, so the excluded count is returned in the frame's metadata via
    :func:`derive_hold_bars` and printed rather than swallowed.
    """
    pieces: list[pl.DataFrame] = []
    for symbol in sorted(endpoints["symbol"].unique().to_list()):
        keys = (
            grid.filter(pl.col("symbol") == symbol)
            .get_column("timestamp")
            .unique()
            .sort()
            .to_numpy()
            .astype("datetime64[us]")
        )
        rows = endpoints.filter(pl.col("symbol") == symbol).sort("decision_ts")
        if not len(keys) or rows.is_empty():
            continue
        d = rows["decision_ts"].to_numpy().astype("datetime64[us]")
        end = rows["end_ts"].to_numpy().astype("datetime64[us]")
        end_plus = end + np.timedelta64(BAR_MINUTES, "m").astype("timedelta64[us]")
        i0 = np.searchsorted(keys, d, side="right")  # first key STRICTLY after the decision
        i1 = np.searchsorted(keys, end_plus, side="right") - 1  # last key <= end + 60min
        bracketed = (i0 < len(keys)) & (i1 >= 0) & (keys[-1] >= end_plus) & (keys[0] <= d)
        i0c, i1c = np.clip(i0, 0, len(keys) - 1), np.clip(i1, 0, len(keys) - 1)
        pieces.append(
            rows.with_columns(
                pl.Series("entry_key", keys[i0c]).cast(pl.Datetime("us")),
                pl.Series("exit_key", keys[i1c]).cast(pl.Datetime("us")),
                pl.Series("hold_bars", (i1 - i0).astype("int64")),
                pl.Series("_bracketed", bracketed),
            ).filter(pl.col("_bracketed"))
        )
    if not pieces:
        raise RuntimeError("no endpoint of any symbol is bracketed by the loaded price grid")
    out = pl.concat(pieces).with_columns(
        (pl.col("entry_key") - pl.duration(minutes=BAR_MINUTES)).alias("entry_fill_instant"),
        (pl.col("exit_key") - pl.duration(minutes=BAR_MINUTES)).alias("exit_fill_instant"),
    )
    return (
        out.with_columns(
            _season(out.get_column("decision_ts")),
            ((pl.col("end_ts").dt.epoch("s") - pl.col("exit_fill_instant").dt.epoch("s")) // 60)
            .cast(pl.Int64)
            .alias("exit_gap_minutes"),
            ((pl.col("exit_key").dt.epoch("s") - pl.col("entry_key").dt.epoch("s")) // 60)
            .cast(pl.Int64)
            .alias("hold_minutes"),
        )
        .drop("_bracketed")
        .sort(["session", "symbol", "decision_ts"])
    )


def fold_span(folds: Sequence[Mapping] | None) -> tuple[str, str] | None:
    """``(earliest train_start, latest val_end)`` of the declared fold ladder, or ``None``.

    This is the tape **any fold of this bot can reach**: nothing before the earliest
    ``train_start`` is fitted on, scored on, or walked by the engine. It is the fold ladder
    authored on 2026-09-07 (``BOT.md``, branch A of the phase-1 spec: 4 folds of train P3Y /
    val P1Y), so it is a boundary declared long before the hold was ever measured - which is
    the only thing that makes it a rule rather than a choice made after seeing the answer.
    """
    if not folds:
        return None
    return (
        min(str(split["train_start"]) for split in folds),
        max(str(split["val_end"]) for split in folds),
    )


def assign_validation_fold(
    measured: pl.DataFrame, *, folds: Sequence[Mapping] | None
) -> pl.DataFrame:
    """Tag each measured row with the id of the validation fold containing it, or ``-1``.

    The declaration asks for constancy "across every symbol, both seasons and every fold". A
    row that falls in no validation window is training-only; it is tagged ``-1`` rather than
    dropped, so the constancy assertion covers the whole development window and not merely the
    parts a fold happens to validate on. Rows before the earliest ``train_start`` are tagged
    ``-2``: no fold reaches them at all, and :func:`derive_hold_bars` treats them separately
    for the reason its docstring records.
    """
    if not folds:
        return measured.with_columns(pl.lit(-1, dtype=pl.Int64).alias("fold"))
    span = fold_span(folds)
    day = pl.col("decision_ts").dt.date()
    fold_id = (
        pl.when(day < pl.lit(span[0]).str.to_date())
        .then(pl.lit(-2, dtype=pl.Int64))
        .otherwise(pl.lit(-1, dtype=pl.Int64))
    )
    for split in folds:
        start = pl.lit(str(split["val_start"])).str.to_date()
        end = pl.lit(str(split["val_end"])).str.to_date()
        fold_id = (
            pl.when((day >= start) & (day <= end))
            .then(pl.lit(int(split["fold"]), dtype=pl.Int64))
            .otherwise(fold_id)
        )
    return measured.with_columns(fold_id.alias("fold"))


class HoldIsNotConstant(RuntimeError):
    """The section-1.2 rule did not produce one number per book. Stop and report.

    This is the loud failure the declaration asks for: if the broker ever moves or removes its
    daily break, or a venue's hours change, the rule stops collapsing to a constant and this
    raises instead of the sweep quietly trading a hold nobody derived.
    """


def derive_hold_bars(
    grid: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    label: str,
    horizons: Mapping[str, str],
    books: Sequence[str],
    folds: Sequence[Mapping] | None = None,
    require_constant: bool = True,
    pre_fold_disagreements: Mapping[str, int] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, int], pl.DataFrame]:
    """``HOLD_BARS[book]`` by the rule of declaration section 1.2, plus the measured detail.

    ``require_constant``
        the declaration's assertion for the two 8-hour labels: ``N`` must be one value inside a
        book across every symbol, both seasons and every fold, or :class:`HoldIsNotConstant` is
        raised. ``fwd_ret_24h`` is the one label this cannot hold for - the distance between two
        same-venue decisions moves with daylight saving - so section 6.1 of the declaration
        derives its **backstop** from the same measurement with ``require_constant=False``.

    ``pre_fold_disagreements``
        MEASURED 2026-09-08 and declared here rather than filtered away. Rows dated before the
        earliest ``train_start`` of the fold ladder (fold 3, ``2018-08-30``) are tagged
        ``fold == -2``: **no fold of this bot fits on them, scores on them or walks them**. On
        the tape any fold can reach the section-1.2 rule collapses to one number per book. On
        the 2017 tape it does not, and the cause is measured rather than guessed - the broker's
        daily break did not sit at the New York close then (February and early March 2017 it sat
        at 00:00-01:00 UTC; through US daylight saving in March and April it stayed on
        22:00-23:00 UTC; on 16 days of 2017 the evening had no gap at all), and it has been
        anchored on the New York close in both seasons since **2017-05-08**, which is what
        ``BOT.md``'s 2026-09-07 census measured over its own window.

        So the pre-fold disagreement is neither hidden nor silently dropped: the caller states
        the count it expects per book, this function measures it, and any change - a different
        count, a different book, a different value - raises. Passing ``None`` means "I expect
        none", and any disagreement raises. That is the loud failure the declaration asks for,
        with an escape that is itself a checked number.

    Returns ``(hold_bars_by_book, measured_rows)``. Nothing is typed: every number in the
    returned dict comes out of ``i1 - i0`` and is printed when ``verbose``.
    """
    endpoints = label_endpoints(panel, label=label, horizons=horizons)
    endpoints = endpoints.filter(pl.col("session").is_in(list(books)))
    measured = assign_validation_fold(measure_hold(grid, endpoints), folds=folds)
    span = fold_span(folds)
    if verbose and span:
        print(f"  fold ladder spans {span[0]} -> {span[1]}; rows before {span[0]} carry fold -2")

    hold: dict[str, int] = {}
    for book in books:
        rows = measured.filter(pl.col("session") == book)
        if rows.is_empty():
            raise HoldIsNotConstant(
                f"book {book!r} has no measurable {label} endpoint in the development window; "
                "the rule of NY_EXIT_DECLARATION.md section 1.2 cannot be evaluated"
            )
        reachable = rows.filter(pl.col("fold") != -2)
        pre_fold = rows.filter(pl.col("fold") == -2)
        distinct = sorted(reachable.get_column("hold_bars").unique().to_list())
        if verbose:
            share = (
                reachable.group_by("hold_bars")
                .agg(pl.len().alias("rows"))
                .sort("hold_bars")
                .to_dicts()
            )
            gaps = (
                reachable.group_by("exit_gap_minutes")
                .agg(pl.len().alias("rows"))
                .sort("exit_gap_minutes")
                .to_dicts()
            )
            minutes = (
                reachable.group_by("hold_minutes")
                .agg(pl.len().alias("rows"))
                .sort("hold_minutes")
                .to_dicts()
            )
            print(
                f"  [{label} / {book}] i1 - i0 over {reachable.height:,} sealed endpoints any "
                f"fold reaches ({reachable['symbol'].n_unique()} symbols, "
                f"{reachable['season'].n_unique()} seasons, "
                f"{reachable.filter(pl.col('fold') >= 0)['fold'].n_unique()} validation folds): "
                f"{share}; label_end_ts - exit_fill_instant {gaps}; hold_minutes {minutes}"
            )
        if require_constant and len(distinct) != 1:
            by_group = (
                reachable.group_by(["symbol", "season", "fold"])
                .agg(
                    pl.col("hold_bars").n_unique().alias("distinct"),
                    pl.col("hold_bars").min().alias("min"),
                    pl.col("hold_bars").max().alias("max"),
                    pl.len().alias("rows"),
                )
                .filter(pl.col("distinct") > 1)
                .sort(["symbol", "season", "fold"])
            )
            raise HoldIsNotConstant(
                f"the section-1.2 rule gives {distinct} bars inside book {book!r} for label "
                f"{label!r} on the tape the folds reach, not one constant. Per "
                f"(symbol, season, fold): {by_group.to_dicts()[:8]}. Stop and report: the "
                "broker's session or break calendar has moved and the hold may not be guessed."
            )
        value = int(distinct[0]) if len(distinct) == 1 else int(min(distinct))
        hold[book] = value

        disagreeing = pre_fold.filter(pl.col("hold_bars") != value)
        expected = int((pre_fold_disagreements or {}).get(book, 0))
        if verbose and pre_fold.height:
            dates = sorted({str(d) for d in disagreeing["decision_ts"].dt.date().to_list()})
            print(
                f"    pre-fold tape ({pre_fold.height:,} endpoints before {span[0] if span else '?'}): "
                f"{disagreeing.height} disagree with {value} on "
                f"{len(dates)} dates{' ' + str(dates[:6]) + ' ...' if dates else ''}"
            )
        if disagreeing.height != expected:
            raise HoldIsNotConstant(
                f"book {book!r} label {label!r}: {disagreeing.height} endpoints before the "
                f"earliest fold train_start disagree with the derived {value} bars, and "
                f"{expected} were declared. Values "
                f"{disagreeing.group_by('hold_bars').len().sort('hold_bars').to_dicts()}, dates "
                f"{sorted({str(d) for d in disagreeing['decision_ts'].dt.date().to_list()})[:12]}. "
                "Re-measure the broker's break calendar and re-declare the count before running "
                "anything: NY_EXIT_DECLARATION.md section 1.2 forbids guessing this number."
            )
    if verbose:
        print(f"  DERIVED HOLD_BARS[{label}] = {hold} (nothing here was typed)")
    return hold, measured


def exit_gap_minutes_by_book(
    measured: pl.DataFrame, books: Sequence[str], *, include_pre_fold: bool = False
) -> dict[str, int]:
    """The residual declaration section 2 asks to be published: measured, not asserted.

    ``label_end_ts - exit_fill_instant`` in minutes, per book, asserted to be a **single value**
    on 100 % of the endpoints any fold reaches - which is stronger than reporting a mode.

    **One discrepancy inside the declaration, resolved here rather than coded around.**
    Section 1.4 criterion 2 writes the pass band as ``0 <= label_end_ts - exit_fill_instant <
    60`` while section 2 declares ``exit_gap_minutes_by_book: {london: 0, ny: 60}``, and 60 is
    not below 60. The measurement settles which one is the typo: fill instants live on a
    60-minute lattice (the open of a close-keyed row is the price one bar earlier), so the only
    values available at or before the endpoint are 0, 60, 120 ... A strict ``< 60`` would force
    every book to 0 and contradict the number section 2 publishes. The operative criterion is
    therefore **at most one grid bar**, ``0 <= gap <= 60``, which is exactly what section 0
    describes for New York ("sớm 60 phút") - and it is asserted here.
    """
    out: dict[str, int] = {}
    for book in books:
        rows = measured.filter(pl.col("session") == book)
        if not include_pre_fold and "fold" in rows.columns:
            rows = rows.filter(pl.col("fold") != -2)
        bad = rows.filter(
            (pl.col("exit_gap_minutes") < 0) | (pl.col("exit_gap_minutes") > BAR_MINUTES)
        )
        if bad.height:
            raise HoldIsNotConstant(
                f"book {book!r}: {bad.height} of {rows.height} endpoints fall outside "
                f"0 <= label_end_ts - exit_fill_instant <= {BAR_MINUTES} minutes "
                f"({bad.group_by('exit_gap_minutes').len().sort('exit_gap_minutes').to_dicts()})"
            )
        values = sorted(rows.get_column("exit_gap_minutes").unique().to_list())
        if len(values) != 1:
            raise HoldIsNotConstant(
                f"book {book!r} has more than one exit gap on the tape the folds reach: "
                f"{rows.group_by('exit_gap_minutes').len().sort('exit_gap_minutes').to_dicts()}. "
                "The residual declared in setup.yaml::backtest.exit_gap_minutes_by_book would "
                "be a mode rather than a fact."
            )
        out[book] = int(values[0])
    return out


def derive_backstop_bars(
    grid: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    label: str,
    horizons: Mapping[str, str],
    books: Sequence[str],
    folds: Sequence[Mapping] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, int], pl.DataFrame]:
    """The same rule, for the one label whose ``N`` cannot be a constant: ``fwd_ret_24h``.

    Declaration section 6.1: *"Trước khi sweep nhãn 24 h: đo bằng đúng harness mục 5, và **dẫn
    xuất** backstop theo cùng luật mục 1.2 thay vì gõ 24."* So the 24 that ``13_backtest`` used
    to type is replaced by whatever ``i1 - i0`` gives - and what it gives is **not** 24.

    ``N`` varies here for a reason the 8-hour labels never meet: the interval between the entry
    row and the exit row spans a whole day, so it contains the recurring one-hour break *and*
    whatever public holidays fall inside it. The extra rule that turns a distribution back into
    a derivation, stated before the number is read:

        the backstop is ``N`` on a tape whose only closure inside the hold is the **recurring
        daily break**. Every row that differs must differ because the tape had *more* holes,
        never fewer - which is checked, not assumed, from
        ``missing = hold_minutes / 60 - hold_bars``.

    A row with the same number of holes and a different ``N``, or with *fewer* holes and a
    different ``N``, is an unexplained deviation and raises :class:`HoldIsNotConstant`.
    """
    endpoints = label_endpoints(panel, label=label, horizons=horizons)
    endpoints = endpoints.filter(pl.col("session").is_in(list(books)))
    measured = assign_validation_fold(measure_hold(grid, endpoints), folds=folds).with_columns(
        (pl.col("hold_minutes") // BAR_MINUTES - pl.col("hold_bars")).alias("missing_keys")
    )
    span = fold_span(folds)
    if verbose and span:
        print(f"  fold ladder spans {span[0]} -> {span[1]}; rows before {span[0]} carry fold -2")

    backstop: dict[str, int] = {}
    for book in books:
        rows = measured.filter((pl.col("session") == book) & (pl.col("fold") != -2))
        if rows.is_empty():
            raise HoldIsNotConstant(f"book {book!r} has no measurable {label} endpoint")
        counts = rows.group_by("hold_bars").agg(pl.len().alias("rows")).sort("rows", descending=True)
        modal_n = int(counts["hold_bars"][0])
        modal_holes = int(
            rows.filter(pl.col("hold_bars") == modal_n)
            .group_by("missing_keys")
            .agg(pl.len().alias("rows"))
            .sort("rows", descending=True)["missing_keys"][0]
        )
        unexplained = rows.filter(
            (pl.col("hold_bars") != modal_n) & (pl.col("missing_keys") <= modal_holes)
        )
        if verbose:
            print(
                f"  [{label} / {book}] i1 - i0 over {rows.height:,} sealed endpoints any fold "
                f"reaches: {counts.sort('hold_bars').to_dicts()}; holes inside the hold "
                f"{rows.group_by('missing_keys').agg(pl.len().alias('rows')).sort('missing_keys').to_dicts()}; "
                f"exit gap {rows.group_by('exit_gap_minutes').agg(pl.len().alias('rows')).sort('exit_gap_minutes').to_dicts()}; "
                f"hold_minutes {rows.group_by('hold_minutes').agg(pl.len().alias('rows')).sort('hold_minutes').to_dicts()}"
            )
            print(
                f"    modal N {modal_n} carries exactly {modal_holes} missing key(s) - the "
                f"recurring daily break; {rows.filter(pl.col('hold_bars') != modal_n).height} "
                "rows differ and every one of them has MORE holes"
            )
        if unexplained.height:
            raise HoldIsNotConstant(
                f"book {book!r} label {label!r}: {unexplained.height} endpoints give a hold "
                f"other than {modal_n} bars without extra closures in the tape "
                f"({unexplained.group_by(['hold_bars', 'missing_keys']).len().to_dicts()[:8]}). "
                "The backstop is not derivable and may not be typed."
            )
        backstop[book] = modal_n
    if verbose:
        print(f"  DERIVED BACKSTOP[{label}] = {backstop} (the typed 24 was never measured)")
    return backstop, measured


# ---------------------------------------------------------------------------
# Generation 2, block 2: the tradable session exit that seals ``fwd_ret_sess``
# ---------------------------------------------------------------------------
def tradable_exit(panel: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """The last H1 close at or before ``label_end_ts`` at which a ``next_bar_open`` exit fills.

    ``bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md`` section 2, ruling 2.1 - *one rule,
    two constants*, the same shape as the section-1.2 rule above::

        tradable_exit_ts = max{ c : decision_ts < c <= label_end_ts, a bar CLOSES at c and a
                                bar OPENS at c }

    A bar that opens at ``c`` is the grid row keyed ``c + 60`` (the grid is keyed on the bar
    close), and ``execution_price: open`` + ``execution_mode: next_bar`` fills at the open of
    that row, which is the price at ``c`` (``PRICE_GRID_DECLARATION.md`` section 1.1). So ``c`` is
    the last instant at which the position can still be closed at a price the label reads.
    ``fwd_ret_sess = close(c) / close(decision_ts) - 1`` is then the return over exactly the
    window the book trades. Where the tape is continuous through the endpoint (London) ``c`` is
    ``label_end_ts`` itself and the label equals ``fwd_ret_8h``; where the endpoint is the start
    of the daily break (New York, both seasons) no bar opens at ``label_end_ts``, ``c`` is one
    bar earlier, and the label equals the ``fwd_ret_7h_probe`` of ``_report_phase5.py``. Nothing
    here types 7 or 8: :func:`derive_hold_bars` measures the same instants on the registered grid
    and ``tests/test_labels_gen2.py`` asserts the two agree.

    Point-in-time: ``label_end_ts`` is the venue's session close, known from the calendar; whether
    a bar opens at an instant is known at that instant - which is at or after the endpoint, so a
    LABEL (sealed at its endpoint) may read it and a feature may not. Rows without an endpoint
    carry nulls. Returns one row per panel row: ``symbol, timestamp, label_end_ts,
    tradable_exit_ts, tradable_exit_close``. Only bar timestamps and the close price at the exit
    instant are read.
    """
    step = np.timedelta64(BAR_MINUTES, "m").astype("timedelta64[us]")
    pieces: list[pl.DataFrame] = []
    for symbol in sorted(panel["symbol"].unique().to_list()):
        rows = (
            panel.filter(pl.col("symbol") == symbol)
            .select("symbol", "timestamp", "label_end_ts")
            .sort("timestamp")
        )
        g = bars.filter(pl.col("symbol") == symbol).sort("timestamp")
        opens = g["timestamp"].to_numpy().astype("datetime64[us]")
        closes = opens + step
        # an exit can fill at a close instant only if a bar OPENS there (the next grid row exists)
        fillable = np.isin(closes, opens)
        exits, exit_px = closes[fillable], g["close"].to_numpy()[fillable]
        d = rows["timestamp"].to_numpy().astype("datetime64[us]")
        has_end = rows["label_end_ts"].is_not_null().to_numpy()
        # a null endpoint is replaced by the decision itself, which no candidate can precede
        end = rows["label_end_ts"].fill_null(rows["timestamp"]).to_numpy().astype("datetime64[us]")
        if len(exits) == 0:
            ok = np.zeros(len(rows), dtype=bool)
            idx = np.zeros(len(rows), dtype=int)
        else:
            idx = np.searchsorted(exits, end, side="right") - 1
            ok = has_end & (idx >= 0)
            idx = np.clip(idx, 0, len(exits) - 1)
            ok &= exits[idx] > d
        exit_ts = np.where(ok, exits[idx] if len(exits) else np.datetime64("NaT", "us"), np.datetime64("NaT", "us"))
        exit_close = np.where(ok, exit_px[idx] if len(exits) else np.nan, np.nan)
        pieces.append(
            rows.with_columns(
                pl.Series("tradable_exit_ts", exit_ts.astype("datetime64[us]")).cast(pl.Datetime("us")),
                pl.Series("tradable_exit_close", exit_close.astype("float64"), dtype=pl.Float64).fill_nan(None),
            )
        )
    out = pl.concat(pieces).sort(["symbol", "timestamp"])
    sealed = out.drop_nulls("tradable_exit_ts")
    if sealed.height:
        assert (sealed["tradable_exit_ts"] > sealed["timestamp"]).all(), "an exit before its decision"
        assert (sealed["tradable_exit_ts"] <= sealed["label_end_ts"]).all(), "an exit after its endpoint"
    return out
