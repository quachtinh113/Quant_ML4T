"""Contract tests for the mentor-gated `_PRICE_CONFIG` diff (fleet gate item #5, 2026-09-10).

`case_studies/utils/backtest_loaders.py::_PRICE_CONFIG` and its `_load_via_canonical` dispatch are
SHARED, single-writer files this bot's builder may not edit until the orchestrator says "go".
These tests are written AHEAD of the diff landing, per the mentor's five-item list verbatim. The
mentor's correction to the first proposal is load-bearing: the real schema is
``entity_col/time_col/close_col/ohlcv/loader/drop_cols`` (+ optional ``price_lineage``), not
``symbols/timeframe/close_keyed`` - every assertion below reads the REAL schema.

* Tests 3, 4 exercise this bot's OWN existing functions (`bots._shared.mt5_loader.load_mt5_bars`,
  `case_studies.exness_btc_8h._features.decision_grid`) directly, independent of the registration
  - the proposed loader branch is designed to call exactly these, so these two pass TODAY and
  double as the specification the loader branch must satisfy.
* Tests 1, 2 need the registration itself and SKIP until `"exness_btc_8h" in _PRICE_CONFIG`.
* Test 5 is a real regression snapshot of the untouched entries, passes TODAY, and must keep
  passing after the diff (additive only).

    uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/test_price_config_diff.py -q
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from case_studies.exness_btc_8h._features import DECISION_HOURS, SLOT_MINUTES, decision_grid
from case_studies.utils.backtest_loaders import _PRICE_CONFIG

CASE_STUDY_ID = "exness_btc_8h"
REQUIRED_KEYS = {"entity_col", "time_col", "close_col", "ohlcv", "loader", "drop_cols"}
OPTIONAL_KEYS = {"price_lineage", "rename_cols", "filter", "prices_file"}
PRICE_SUPPORTED = CASE_STUDY_ID in _PRICE_CONFIG
SKIP_REASON = (
    "waiting for _PRICE_CONFIG diff (fleet gate item #5) to register exness_btc_8h "
    f"(currently registered: {sorted(_PRICE_CONFIG)})"
)


@pytest.mark.skipif(not PRICE_SUPPORTED, reason=SKIP_REASON)
class TestKeySetEqualityAndDispatchItem1:
    """(1) key-set equality with existing _PRICE_CONFIG entries; loader resolves in dispatch."""

    def test_required_keys_present_and_no_unknown_keys(self):
        entry = _PRICE_CONFIG[CASE_STUDY_ID]
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, f"missing required _PRICE_CONFIG keys: {missing}"
        unknown = set(entry) - REQUIRED_KEYS - OPTIONAL_KEYS
        assert not unknown, f"unexpected _PRICE_CONFIG keys (real schema drifted?): {unknown}"

    def test_entity_time_close_columns_match_the_sealed_panel_shape(self):
        entry = _PRICE_CONFIG[CASE_STUDY_ID]
        assert entry["entity_col"] == "symbol"
        assert entry["time_col"] == "timestamp"
        assert entry["close_col"] == "close"
        assert entry["ohlcv"] is True

    def test_loader_resolves_in_dispatch(self):
        """The declared loader name must be a real branch of `_load_via_canonical`, not a typo.

        Probed with an impossible window (`end_date` before `history_start`) so the branch is
        exercised without pulling a real, possibly large, price frame - the dispatcher's own
        fallback (`case_studies/utils/backtest_loaders.py:1092-1093`) raises
        ``ValueError("Unknown loader: ...")`` when no branch matches the name; anything else
        (including an empty/short result, or a data error from the impossible window) means the
        branch exists and was reached.
        """
        from case_studies.utils.backtest_loaders import _load_via_canonical

        loader_name = _PRICE_CONFIG[CASE_STUDY_ID]["loader"]
        assert loader_name, "exness_btc_8h must declare a loader, not the materialized-parquet path"
        try:
            _load_via_canonical(loader_name, start_date="2018-01-01", end_date="2018-01-02")
        except ValueError as exc:
            if str(exc).startswith("Unknown loader:"):
                pytest.fail(f"loader {loader_name!r} has no branch in _load_via_canonical")
        except Exception:
            pass  # any other failure means the branch was reached and did real work


@pytest.mark.skipif(not PRICE_SUPPORTED, reason=SKIP_REASON)
class TestGridIdentityItem2:
    """(2) grid identity: price timestamps == the 8h decision grid 02_labels uses, one row per
    instant per symbol - this is what makes `backtest.price_grid_expresses_primary_label` a
    MEASUREMENT rather than a declared claim taken on faith.
    """

    def test_price_grid_equals_label_grid_one_row_per_instant(self):
        import yaml

        from case_studies.utils.backtest_loaders import load_backtest_prices_for
        from utils.paths import get_case_study_dir

        setup = yaml.safe_load(
            (get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text()
        )
        if not setup["backtest"]["price_grid_expresses_primary_label"]:
            pytest.skip("setup.yaml declares price_grid_expresses_primary_label: false")
        primary_label = setup["labels"]["primary"]
        label_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{primary_label}.parquet"
        if not label_path.exists():
            pytest.skip(f"labels not built yet: {label_path}")
        label_frame = pl.read_parquet(label_path)

        prices = load_backtest_prices_for(CASE_STUDY_ID, primary_label, split="validation")
        assert prices.select("timestamp", "symbol").n_unique() == prices.height, (
            "one row per (symbol, decision instant) - a folded-twice bar would duplicate a key"
        )
        window_lo, window_hi = prices["timestamp"].min(), prices["timestamp"].max()
        decision_rows = label_frame.select("timestamp", "symbol").unique().filter(
            (pl.col("timestamp") >= window_lo) & (pl.col("timestamp") <= window_hi)
        )
        off_grid = decision_rows.join(
            prices.select("timestamp", "symbol").unique(), on=["timestamp", "symbol"], how="anti"
        )
        assert off_grid.is_empty(), (
            f"{off_grid.height} decision instant(s) of the sealed primary label are not a row "
            "of the registered price grid; the price loader and 02_labels have drifted apart"
        )


class TestCloseKeyedGridItem3:
    """(3) close-keyed via bots/_shared/mt5_loader.resample_4h_to_8h, (t-8h, t]."""

    def test_decision_grid_close_keyed_on_load_mt5_bars_8h(self):
        from bots._shared.mt5_loader import DataNotFoundError, load_mt5_bars

        try:
            bars = load_mt5_bars("8h", symbols=["BTCUSD"], start_date="2024-01-01", end_date="2024-01-08")
        except DataNotFoundError:
            pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/4h.parquet)")
        if bars.is_empty():
            pytest.skip("no BTCUSD 8h bars in the probed window")
        panel = decision_grid(bars, verbose=False)
        # decision instant (timestamp) is the bar's OPEN + SLOT_MINUTES = the bar's CLOSE, i.e.
        # the panel row keys the bar spanning (t - 8h, t].
        gap_minutes = (panel["timestamp"] - panel["bar_open_ts"]).dt.total_minutes()
        assert (gap_minutes == SLOT_MINUTES).all()
        assert set(panel["timestamp"].dt.hour().unique().to_list()) <= set(DECISION_HOURS)


class TestPitBoundaryItem4:
    """(4) PIT/boundary - no row >= holdout_start (2025-09-01) on the development path."""

    def test_naive_end_date_bound_leaks_one_close_keyed_row_past_holdout_start(self):
        """DOCUMENTS A REAL LEAK, found while writing this test (2026-09-10).

        `load_mt5_bars(end_date=...)` bounds the raw, OPEN-keyed H4 bars inclusively through
        the whole `end_date` day. `decision_grid` then shifts the timestamp forward by
        `SLOT_MINUTES` (480) to key on the bar's CLOSE. The last H4 pair of `end_date` = the
        day before `holdout_start` therefore folds into an 8h bar OPENING at 16:00 on that day
        and CLOSING - i.e. decision-keyed - at 00:00 on `holdout_start` itself: `>=
        holdout_start`, not `<`. Bounding `load_mt5_bars`'s `end_date` alone is NOT sufficient
        to keep the development path strictly before the holdout; this is what the proposed
        loader branch in `bots/exness_btc_8h/BOT.md` Open question 5 must additionally filter
        on the CLOSE side, explicitly, after `decision_grid` runs - not rely on `end_date`
        alone. This test pins the leak so the fix cannot regress silently.
        """
        from bots._shared.mt5_loader import DataNotFoundError, load_mt5_bars

        holdout_start = "2025-09-01"
        try:
            bars = load_mt5_bars(
                "8h", symbols=["BTCUSD"], start_date="2025-08-25", end_date="2025-08-31"
            )
        except DataNotFoundError:
            pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/4h.parquet)")
        if bars.is_empty():
            pytest.skip("no BTCUSD 8h bars in the probed window")
        panel = decision_grid(bars, verbose=False)
        boundary = pl.lit(holdout_start).str.to_datetime()
        leaked = panel.filter(pl.col("timestamp") >= boundary)
        assert leaked.height == 1, (
            f"expected exactly the one known-leaked row (the 16:00 slot's close, folded past "
            f"midnight into {holdout_start}); got {leaked.height}. If this is 0, the loader's "
            "upstream behaviour changed and the explicit close-side clip below may no longer "
            "be necessary - re-read before deleting it from the proposed diff."
        )

        def clip_to_development(frame: "pl.DataFrame") -> "pl.DataFrame":
            """What the proposed loader branch must do: an EXPLICIT close-side clip."""
            return frame.filter(pl.col("timestamp") < boundary)

        clipped = clip_to_development(panel)
        assert clipped.filter(pl.col("timestamp") >= boundary).is_empty()
        assert clipped.height == panel.height - 1
        assert str(clipped["timestamp"].max())[:10] < holdout_start


class TestEndDateCoercionMaterial3:
    """Fleet mentor material #3 (2026-09-10, combined re-gate): the PIT clip's `end_date` must
    be coerced EXPLICITLY (date / str / tz-aware datetime) so a naive-vs-aware comparison can
    never raise, and the clip must only run when `end_date` is given at all.
    """

    def test_date_object(self):
        from datetime import date

        from case_studies.utils.backtest_loaders import _naive_end_date_string

        assert _naive_end_date_string(date(2025, 8, 31)) == "2025-08-31"

    def test_naive_datetime(self):
        from datetime import datetime

        from case_studies.utils.backtest_loaders import _naive_end_date_string

        assert _naive_end_date_string(datetime(2025, 8, 31, 13, 45)) == "2025-08-31"

    def test_timezone_aware_datetime_is_converted_to_utc_first(self):
        from datetime import datetime, timedelta, timezone

        from case_studies.utils.backtest_loaders import _naive_end_date_string

        # 2025-09-01 02:00 in UTC+5 is still 2025-08-31 21:00 UTC - the UTC calendar day, not
        # the local one, must come out, or a caller in a positive-offset zone would silently
        # clip one day too LATE (re-opening exactly the holdout leak this material follows).
        eastern5 = timezone(timedelta(hours=5))
        aware = datetime(2025, 9, 1, 2, 0, tzinfo=eastern5)
        assert _naive_end_date_string(aware) == "2025-08-31"

    def test_bare_date_string(self):
        from case_studies.utils.backtest_loaders import _naive_end_date_string

        assert _naive_end_date_string("2025-08-31") == "2025-08-31"

    def test_iso_datetime_string_reads_only_the_calendar_day(self):
        from case_studies.utils.backtest_loaders import _naive_end_date_string

        assert _naive_end_date_string("2025-08-31T00:00:00+00:00") == "2025-08-31"

    @pytest.mark.parametrize(
        "end_date_value",
        [
            "2025-08-31",
            date(2025, 8, 31),
            datetime(2025, 8, 31, 12, 0),
            datetime(2025, 8, 31, 12, 0, tzinfo=timezone.utc),
        ],
        ids=["str", "date", "naive_datetime", "aware_datetime"],
    )
    @pytest.mark.skipif(not PRICE_SUPPORTED, reason=SKIP_REASON)
    def test_loader_branch_accepts_every_end_date_type_without_raising(self, end_date_value):
        """Integration proof: none of the four input types reach a naive-vs-aware comparison
        error inside the actual dispatch branch (unit tests above cover the coercion alone).
        Any OTHER exception is a real failure of this test, not tolerated silently.
        """
        from bots._shared.mt5_loader import DataNotFoundError
        from case_studies.utils.backtest_loaders import _load_via_canonical

        loader_name = _PRICE_CONFIG[CASE_STUDY_ID]["loader"]
        try:
            _load_via_canonical(
                loader_name, start_date="2025-08-25", end_date=end_date_value
            )
        except DataNotFoundError:
            pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/4h.parquet)")


class TestLegacyEntriesUnchangedItem5:
    """(5) legacy - existing _PRICE_CONFIG entries and the fx_pairs path unchanged."""

    # Captured 2026-09-10, before this bot's registration was proposed. A real regression
    # snapshot: additive registration must not touch any existing entry's dict.
    _FX_PAIRS_SNAPSHOT = {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "close",
        "ohlcv": True,
        "loader": "fx_pairs",
        "drop_cols": [],
    }
    _EXNESS_GOLD_SESS_LOADER = "exness_gold_sess_h1"
    _EXPECTED_EXISTING_KEYS = {
        "etfs",
        "crypto_perps_funding",
        "nasdaq100_microstructure",
        "sp500_equity_option_analytics",
        "us_firm_characteristics",
        "fx_pairs",
        "exness_fx_d1",
        "exness_gold_sess",
        "exness_usidx_sess",
        "xau_fx_mt5",
        "cme_futures",
        "sp500_options",
        "us_equities_panel",
    }

    def test_fx_pairs_entry_unchanged(self):
        assert _PRICE_CONFIG["fx_pairs"] == self._FX_PAIRS_SNAPSHOT

    def test_exness_gold_sess_loader_name_unchanged(self):
        assert _PRICE_CONFIG["exness_gold_sess"]["loader"] == self._EXNESS_GOLD_SESS_LOADER

    def test_existing_keys_all_still_present(self):
        missing = self._EXPECTED_EXISTING_KEYS - set(_PRICE_CONFIG)
        assert not missing, f"an existing _PRICE_CONFIG entry disappeared: {missing}"

    def test_registration_is_additive_only(self):
        """The only new key relative to the 2026-09-10 snapshot may be exness_btc_8h."""
        extra = set(_PRICE_CONFIG) - self._EXPECTED_EXISTING_KEYS
        assert extra <= {CASE_STUDY_ID}, f"unexpected new _PRICE_CONFIG key(s): {extra - {CASE_STUDY_ID}}"
