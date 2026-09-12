"""Contract tests for the mentor-gated `apply_session_filter` diff (fleet gate item #4, 2026-09-10).

`case_studies/utils/backtest_runner.py::apply_session_filter` is a SHARED, single-writer file this
bot's builder may not edit until the orchestrator says "go" (serialization: `exness_fx_d1`'s
builder is editing `case_studies/utils/` and `utils/` right now). These tests are written AHEAD of
that diff landing, per the mentor's five-item list verbatim, so the diff can be graded the moment
it lands rather than reviewed by eye. Two kinds of test live here:

* Tests 1 and 2 exercise CURRENT, unmodified behaviour (gold's own dispatch, and every other
  registered case study's refusal) and PASS TODAY - they are real regression protection, not
  placeholders, and the diff must not change what they assert.
* Tests 3, 4 and 5 are `exness_btc_8h`-specific and cannot pass until the diff lands (this
  case study is not registered yet); they SKIP with an explicit reason
  (`"waiting for apply_session_filter diff (fleet gate item #4)"`) detected by probing the
  live error message, and start running for real the moment the registration exists - no manual
  un-skipping required.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/test_session_filter_diff.py -q
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from case_studies.exness_gold_sess._features import VENUES, BAR_MINUTES, _venue_window
from case_studies.exness_gold_sess._features import session_of as gold_session_of
from case_studies.utils.backtest_loaders import _PRICE_CONFIG
from case_studies.utils.backtest_runner import apply_session_filter

CASE_STUDY_ID = "exness_btc_8h"
NOT_REGISTERED_SUBSTRING = "must register its own session resolver"


def _btc_session_filter_status() -> tuple[bool, str]:
    """``(supported, detail)``: is `exness_btc_8h` registered in `apply_session_filter` yet?

    Probes with a real, tiny frame rather than importing private state, so the detection is
    correct however the diff implements the dispatch (a new `elif`, a registry dict, ...).
    """
    probe = pl.DataFrame(
        {"timestamp": [datetime(2026, 1, 5, 0, 0, 0)], "symbol": ["BTCUSD"]}
    )
    try:
        apply_session_filter(probe, CASE_STUDY_ID, {"session_filter": "all"})
    except ValueError as exc:
        if NOT_REGISTERED_SUBSTRING in str(exc):
            return False, str(exc)
        # A different ValueError (e.g. an undeclared-book message) means the case study IS
        # registered and this probe simply used a book/frame combination the new code rejects
        # for its own reason - still "supported" for the purpose of gating the tests below.
        return True, str(exc)
    return True, "no error"


BTC_SUPPORTED, _BTC_STATUS_DETAIL = _btc_session_filter_status()
SKIP_REASON = (
    "waiting for apply_session_filter diff (fleet gate item #4) to register exness_btc_8h; "
    f"current probe: {_BTC_STATUS_DETAIL}"
)


def _gold_fixture() -> pl.DataFrame:
    """A small, REAL gold predictions frame: one London and one NY decision instant on a
    Monday and on a Friday, for both metals. Built from `_venue_window` rather than typed
    UTC hours, so the fixture is correct across DST without guessing.
    """
    monday = date(2026, 1, 5)
    friday = date(2026, 1, 9)
    rows = []
    for day in (monday, friday):
        for venue in VENUES:
            opened, _closed = _venue_window(day, venue)
            decision = opened + timedelta(minutes=BAR_MINUTES)
            for symbol in ("XAUUSD", "XAGUSD"):
                rows.append({"timestamp": decision, "symbol": symbol, "venue": venue, "day": day})
    frame = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    return frame


class TestLegacyGoldEqualityItem1:
    """(1) legacy equality for exness_gold_sess books london/ny/drop_friday, frame equality.

    `apply_session_filter` reads `exness_gold_sess`'s OWN `session_filter_values` through
    `load_setup_config`, which resolves via `get_case_study_dir` and therefore via
    `ML4T_OUTPUT_DIR`. That env var is process-global: when the whole bot suite runs with
    `ML4T_OUTPUT_DIR` pointed at THIS bot's experiment workspace (the documented convention,
    e.g. `~/ml4t/experiments/exness_btc_8h`), a plain call would read gold's config from inside
    that workspace instead of gold's own repository copy and see `session_filter_values: []`.
    Every test in this class clears `ML4T_OUTPUT_DIR` for its own duration so "legacy equality"
    means the SAME thing whether this file runs alone or inside the full suite.
    """

    @pytest.fixture(autouse=True)
    def _read_golds_own_repository_config(self, monkeypatch):
        monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("ML4T_CHAPTER_OUTPUT_DIR", raising=False)

    def test_london_book_matches_direct_session_of_filter(self):
        frame = _gold_fixture().select("timestamp", "symbol")
        expected = frame.filter(gold_session_of(frame.get_column("timestamp")) == "london")
        actual = apply_session_filter(frame, "exness_gold_sess", {"session_filter": "london"})
        assert actual.sort(actual.columns).equals(expected.sort(expected.columns))

    def test_ny_book_matches_direct_session_of_filter(self):
        frame = _gold_fixture().select("timestamp", "symbol")
        expected = frame.filter(gold_session_of(frame.get_column("timestamp")) == "ny")
        actual = apply_session_filter(frame, "exness_gold_sess", {"session_filter": "ny"})
        assert actual.sort(actual.columns).equals(expected.sort(expected.columns))

    def test_drop_friday_matches_direct_weekday_filter(self):
        frame = _gold_fixture().select("timestamp", "symbol")
        actual = apply_session_filter(
            frame, "exness_gold_sess", {"session_filter": "london", "drop_friday": True}
        )
        direct_london = frame.filter(
            (gold_session_of(frame.get_column("timestamp")) == "london")
            & (frame.get_column("timestamp").dt.weekday() != 5)
        )
        assert actual.sort(actual.columns).equals(direct_london.sort(direct_london.columns))
        # The fixture's Friday row is dropped by drop_friday and would otherwise have survived
        # as a "london" decision - assert the fixture actually exercises the rule rather than
        # passing vacuously because no Friday row existed.
        friday_london = frame.filter(
            (gold_session_of(frame.get_column("timestamp")) == "london")
            & (frame.get_column("timestamp").dt.weekday() == 5)
        )
        assert friday_london.height > 0
        assert actual.height == direct_london.height < (
            direct_london.height + friday_london.height
        )


class TestEveryOtherCaseStudyStillRaisesItem2:
    """(2) every other registered case study still raises with the old message."""

    @pytest.mark.parametrize(
        "case_study",
        sorted(set(_PRICE_CONFIG) - {"exness_gold_sess", "exness_btc_8h"}),
    )
    def test_still_raises_old_message(self, case_study):
        probe = pl.DataFrame(
            {"timestamp": [datetime(2026, 1, 5, 0, 0, 0)], "symbol": ["X"]}
        )
        with pytest.raises(ValueError, match=NOT_REGISTERED_SUBSTRING):
            apply_session_filter(probe, case_study, {"session_filter": "some_book"})


def _btc_probe_frame() -> pl.DataFrame:
    """Nine real decision instants (3 days x {00:00, 08:00, 16:00} UTC) for BTCUSD."""
    from datetime import datetime

    start = datetime(2026, 1, 5, 0, 0, 0)
    rows = [
        {"timestamp": start + timedelta(days=day_offset, hours=hour), "symbol": "BTCUSD"}
        for day_offset in range(3)
        for hour in (0, 8, 16)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


@pytest.mark.skipif(not BTC_SUPPORTED, reason=SKIP_REASON)
class TestBtcAllIsAnExplicitNoOpItem3:
    """(3) 'all' explicit no-op: rows(all) == rows(no_swap_night) + rows(us_hours), union == frame."""

    def test_all_book_is_complete_partition(self):
        frame = _btc_probe_frame()
        all_rows = apply_session_filter(frame, CASE_STUDY_ID, {"session_filter": "all"})
        no_swap = apply_session_filter(frame, CASE_STUDY_ID, {"session_filter": "no_swap_night"})
        us_hours = apply_session_filter(frame, CASE_STUDY_ID, {"session_filter": "us_hours"})
        assert all_rows.height == frame.height, "'all' must be a true no-op: same row count"
        assert all_rows.sort(all_rows.columns).equals(frame.sort(frame.columns))
        assert all_rows.height == no_swap.height + us_hours.height
        union = pl.concat([no_swap, us_hours]).sort(["timestamp", "symbol"])
        assert union.equals(all_rows.sort(["timestamp", "symbol"]))


@pytest.mark.skipif(not BTC_SUPPORTED, reason=SKIP_REASON)
class TestUnknownBookStillRaisesItem4:
    """(4) an undeclared book still raises at backtest_runner.py:888-893's declared-book gate."""

    def test_unknown_book_raises(self):
        frame = _btc_probe_frame()
        with pytest.raises(ValueError, match="is not one of the declared"):
            apply_session_filter(frame, CASE_STUDY_ID, {"session_filter": "london"})


@pytest.mark.skipif(not BTC_SUPPORTED, reason=SKIP_REASON)
class TestThreeBooksPairwiseDifferentItem5:
    """(5) the three books yield pairwise-different frames."""

    def test_pairwise_different(self):
        frame = _btc_probe_frame()
        books = ["all", "no_swap_night", "us_hours"]
        results = {
            book: apply_session_filter(frame, CASE_STUDY_ID, {"session_filter": book})
            for book in books
        }
        for i, a in enumerate(books):
            for b in books[i + 1 :]:
                fa, fb = results[a].sort(results[a].columns), results[b].sort(results[b].columns)
                assert not (fa.height == fb.height and fa.equals(fb)), (
                    f"books {a!r} and {b!r} produced identical frames"
                )


@pytest.mark.skipif(not BTC_SUPPORTED, reason=SKIP_REASON)
class TestAllValueIsDeclaredNotHardCodedMaterial1:
    """Fleet mentor material #1 (2026-09-10, combined re-gate): the "whole grid" value must be a
    DECLARATION (`decision.session_filter_all_value`), not `apply_session_filter` hard-coding
    `case_study == "exness_btc_8h" and book == "all"`. Proven behaviourally, not by reading the
    source: with the declaration monkeypatched away, `"all"` stops being a no-op and is resolved
    against `session_book_of` like any other value - which raises, because that resolver never
    returns `"all"`. If a future edit reintroduces the hard-code, this test does not notice the
    declaration was ignored - it notices the FUNCTION'S BEHAVIOUR stopped depending on it.
    """

    def test_setup_yaml_declares_the_all_value(self):
        import yaml

        from utils.paths import get_case_study_dir

        setup = yaml.safe_load(
            (get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text()
        )
        assert setup["decision"]["session_filter_all_value"] == "all"
        assert (
            setup["decision"]["session_filter_all_value"] in setup["decision"]["session_filter_values"]
        )

    def test_behaviour_follows_the_declaration_not_a_hardcoded_case_study_check(self, monkeypatch):
        import case_studies.utils.backtest_runner as backtest_runner_module
        from utils.artifact_specs import load_setup_config as real_load_setup_config

        def _load_setup_without_all_value(case_study: str):
            cfg = real_load_setup_config(case_study)
            if case_study == CASE_STUDY_ID:
                cfg = dict(cfg)
                cfg["decision"] = {
                    k: v for k, v in cfg["decision"].items() if k != "session_filter_all_value"
                }
            return cfg

        monkeypatch.setattr(
            "utils.artifact_specs.load_setup_config", _load_setup_without_all_value
        )
        frame = _btc_probe_frame()
        # "all" is still a DECLARED book (session_filter_values), so it clears that check; with
        # no session_filter_all_value to name it a no-op, it is resolved against
        # session_book_of - which never returns "all" - and the empty result raises.
        with pytest.raises(ValueError, match="emptied a"):
            backtest_runner_module.apply_session_filter(
                frame, CASE_STUDY_ID, {"session_filter": "all"}
            )
        # Control: with the declaration intact (no monkeypatch active outside this test), "all"
        # is still a no-op - proven by every other test in this class/module, re-asserted here
        # directly so this test is self-contained.
        monkeypatch.undo()
        result = apply_session_filter(frame, CASE_STUDY_ID, {"session_filter": "all"})
        assert result.height == frame.height
