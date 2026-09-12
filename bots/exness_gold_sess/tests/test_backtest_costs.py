"""The engine cost block of exness_gold_sess, and the ``metals`` CFD pair class it needs.

Before this class existed, ``case_studies.utils.backtest_loaders._cfd_pair_class`` returned
``"major_pairs"`` for any symbol containing ``"USD"`` - which XAUUSD and XAGUSD both do. A
metals-only ``spread_bps`` block therefore raised ("no range for 'major_pairs'"), and the way
round it - declaring an FX range the bot does not trade - would have priced silver at about a
quarter of its measured spread **in silence**. Measured 2026-09-07 on this account: XAGUSD p90
4.70 bps a crossing against 0.6-1.3 bps for the five FX pairs.

The FX case studies must not move: ``fx_pairs`` has no ``commission_per_lot`` and keeps the
generic fallback, and ``exness_fx_d1``'s five pairs still resolve to ``major_pairs``, so their
registered backtest identities are unchanged. That is what the last two tests hold.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests -q
"""

from __future__ import annotations

import pytest
import yaml

from utils.paths import REPO_ROOT

pytest.importorskip("ml4t.backtest")

from case_studies.utils.backtest_loaders import (  # noqa: E402
    _cfd_pair_class,
    _normalize_cfd_costs,
    _normalize_costs,
    get_backtest_config,
)

CASE_STUDY_ID = "exness_gold_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"


def _setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


def test_metals_are_their_own_pair_class():
    assert _cfd_pair_class("XAUUSD") == "metals"
    assert _cfd_pair_class("XAGUSD") == "metals"
    assert _cfd_pair_class("XAUUSD247") == "metals"  # the 24/7 gold variant, same cost family
    assert _cfd_pair_class("xagusd") == "metals"  # case is not a classification


def test_the_fx_classes_are_unchanged():
    """The regression guard: no FX symbol may fall into the new branch."""
    for pair in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"):
        assert _cfd_pair_class(pair) == "major_pairs"
    for cross in ("EURJPY", "GBPJPY", "EURGBP"):
        assert _cfd_pair_class(cross) == "cross_pairs"


def test_declared_block_is_the_cfd_layout_with_only_a_metals_range():
    costs = _setup()["costs"]
    assert costs["components"] == ["spread", "commission", "swap"]
    assert costs["commission_per_lot"] == 0
    # metals only: declaring an FX range this universe does not trade would be a wrong number
    # one edit away from being charged.
    assert set(costs["spread_bps"]) == {"metals"}


def test_engine_costs_are_the_measured_metals_spread_and_no_commission():
    setup = _setup()
    commission_bps, slippage_bps = _normalize_costs(setup["costs"], CASE_STUDY_ID)
    assert commission_bps == 0.0
    # The top of the declared range, which is silver's measured p90: the whole book is charged
    # the more expensive leg, deliberately (PHASE1_SPEC_MENTOR.md section 2.e).
    assert slippage_bps == pytest.approx(float(setup["costs"]["spread_bps"]["metals"][-1]))
    assert slippage_bps == pytest.approx(4.70)


def test_the_declared_round_trips_match_the_measured_per_session_table():
    """``round_trip_p90_bps`` is twice the p90 of the traded sessions, per metal."""
    table = _setup()["costs"]["spread_bps_by_session"]
    for symbol, expected in table["round_trip_p90_bps"].items():
        traded = [table[symbol][bucket][1] for bucket in ("london", "new_york")]
        assert 2 * max(traded) == pytest.approx(float(expected), abs=1e-9)
    # silver is the expensive leg by a factor the design has to survive
    ratio = table["round_trip_p90_bps"]["XAGUSD"] / table["round_trip_p90_bps"]["XAUUSD"]
    assert ratio > 7.0


def test_a_metals_universe_without_a_metals_range_is_refused():
    """The failure mode this class replaces: an unpriced universe must raise, not default."""
    with pytest.raises(ValueError, match="no range for 'metals'"):
        _normalize_cfd_costs(
            {"commission_per_lot": 0, "spread_bps": {"major_pairs": [0.6, 1.3]}}, CASE_STUDY_ID
        )


def test_get_backtest_config_reads_the_same_numbers():
    config = get_backtest_config(CASE_STUDY_ID)
    assert (config.commission_bps, config.slippage_bps) == (0.0, 4.70)
    assert config.execution_delay == "next_bar_open"
    assert config.long_short is True  # mapping.position_state_space carries both tokens
    assert config.share_type == "integer"
    assert config.cadence == "session"


def test_exness_fx_d1_costs_did_not_move():
    """The neighbouring bot's registered identities must be unaffected by the new class."""
    config = get_backtest_config("exness_fx_d1")
    assert (config.commission_bps, config.slippage_bps) == (0.0, 1.3)
