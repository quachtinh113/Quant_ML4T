"""The engine cost block of exness_fx_d1 comes from the measured CFD ``costs`` block (phase 5).

``case_studies.utils.backtest_loaders.get_backtest_config`` used to fall through to the generic
``5 + 2`` bps fallback for a block keyed ``spread_bps`` / ``commission_per_lot`` / ``swap`` (the
layout of ``mt5-exness-broker.md`` section 2), charging 14 bps a round trip against the 2.6 bps
``01_feasibility_analysis`` prices. The CFD branch charges the top of the declared spread range
for the pair classes in the universe as the slippage leg, zero commission on a Pro account, and
leaves the swap (a holding cost per night) to the cost stage. The template ``fx_pairs`` block has
no ``commission_per_lot`` and keeps the fallback, so its registered identities do not move.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_fx_d1/tests/test_backtest_costs.py -q
"""

from __future__ import annotations

import pytest
import yaml

from utils.paths import REPO_ROOT

pytest.importorskip("ml4t.backtest")

from case_studies.utils.backtest_loaders import (  # noqa: E402
    _normalize_cfd_costs,
    _normalize_costs,
    get_backtest_config,
)

CASE_STUDY_ID = "exness_fx_d1"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"


def _setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


def test_declared_block_is_the_cfd_layout():
    costs = _setup()["costs"]
    assert costs["components"] == ["spread", "commission", "swap"]
    assert costs["commission_per_lot"] == 0
    assert set(costs["spread_bps"]) >= {"major_pairs", "cross_pairs"}


def test_engine_costs_are_the_measured_spread_and_no_commission():
    setup = _setup()
    commission_bps, slippage_bps = _normalize_costs(setup["costs"], CASE_STUDY_ID)
    assert commission_bps == 0.0
    # Five dollar pairs: the top of the major range, never the template cross range.
    assert slippage_bps == pytest.approx(float(setup["costs"]["spread_bps"]["major_pairs"][-1]))
    assert slippage_bps < min(setup["costs"]["spread_bps"]["cross_pairs"])
    # One crossing pays the declared spread; a round trip is what 01 prices (2 * top of range).
    assert 2 * slippage_bps == pytest.approx(2 * 1.3)


def test_get_backtest_config_reads_the_same_numbers():
    config = get_backtest_config(CASE_STUDY_ID)
    assert (config.commission_bps, config.slippage_bps) == (0.0, 1.3)
    assert config.execution_delay == "next_bar_open"
    assert config.long_short is True
    # Generation 3 (2026-09-10, GEN3_DECLARATION_2026-09-10.md, OQ 32(b)): was "integer".
    assert config.share_type == "mt5_lot"


def test_non_zero_per_lot_commission_is_refused_not_guessed():
    costs = dict(_setup()["costs"])
    costs["commission_per_lot"] = 3.5
    with pytest.raises(ValueError, match="commission_per_lot"):
        _normalize_cfd_costs(costs, CASE_STUDY_ID)


def test_template_fx_pairs_block_keeps_the_fallback():
    template = yaml.safe_load(
        (REPO_ROOT / "case_studies" / "fx_pairs" / "config" / "setup.yaml").read_text(encoding="utf-8")
    )["costs"]
    assert "commission_per_lot" not in template
    assert _normalize_costs(template, "fx_pairs") == (5.0, 2.0)
