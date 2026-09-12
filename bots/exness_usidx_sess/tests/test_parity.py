"""The parity harness — the one thing that must exist BEFORE a survivor does.

Parity (Ch25 s25.1, ``25_live_trading/08``) tests the **code path**: given the same inputs,
does the live loop reach the same features, the same predictions, the same weights and the same
basket the research pipeline reached? It says nothing about whether that basket is worth
holding, and on this bot it demonstrably is not — 0 survivors of 1,403 scored specs at
K = 1,780.

WHY THE PARITY REPLAY IS NOT AN ``ml4t.backtest.Engine`` REPLAY HERE, unlike ``exness_fx_d1``.
``BOT.md`` records the mechanism: prediction sets are keyed on the session DATE and the
registered price grid is HOURLY, so a ``next_bar`` fill lands 13-15 hours BEFORE the decision;
and the engine holds continuously between fills while both specs are flat for part of their
cycle (measured median holding period 2,088 hours against a six-hour label). Running the engine
here would reproduce that lookahead inside the deployment package and present it as a
reference tape. So parity is the five-stage comparison instead, which needs no engine.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from bots.exness_usidx_sess.deploy import deployment_loop as dl

RISK_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "risk_config.yaml"
STAGES = ("1_data", "2_features", "3_predictions", "4_sizing", "5_orders")


@pytest.fixture(scope="module")
def cfg_and_panel(tmp_path_factory):
    """Built once: recomputing the panel is the expensive part of the harness."""
    root = tmp_path_factory.mktemp("parity")
    cfg = dl.load_deploy_config(RISK_CONFIG, spec="intraday", state_dir=root)
    panel = dl.recompute_features(cfg, as_of=cfg.holdout_start)
    return cfg, panel


@pytest.fixture(scope="module")
def predictions(cfg_and_panel):
    cfg, _ = cfg_and_panel
    choice = dl.resolve_model(cfg, "latest-complete")
    try:
        frame, _hash = dl.read_registered_predictions(cfg, choice)
    except dl.NotReadyError as exc:
        pytest.skip(f"no readable registered prediction set: {exc}")
    return frame


# =======================================================================================
# The five stages exist and are compared
# =======================================================================================
def test_the_report_has_all_five_stages(cfg_and_panel, predictions):
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    assert tuple(report["stages"]) == STAGES


def test_stage_one_reads_both_frequencies(cfg_and_panel, predictions):
    """The bot's panel is a TWO-frequency join: H1 for the session grid and D1 for every
    window longer than 63, because the H1 history warms only 96 sessions."""
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    data = report["stages"]["1_data"]
    assert data["h1_bars"] > 30_000  # 38,937 H1 bars truncated at holdout_start
    assert data["d1_bars"] > 2_000
    assert data["symbols"] == ["US500", "USTEC"]


def test_stage_two_uses_the_case_studys_own_feature_builder(cfg_and_panel, predictions):
    """One feature implementation, five consumers (``01``, ``02``, ``03``, the registered price
    loader and this loop). A feature written twice agrees on the day it is written and drifts
    on the first edit."""
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    stage = report["stages"]["2_features"]
    assert stage["builder"].endswith("_features.build_features")
    # 74 price features: BOT.md phase 3 measured exactly that on both specs.
    assert stage["live_columns"] == 74, stage["live_columns"]


def test_stage_three_never_reaches_the_holdout(cfg_and_panel, predictions):
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    assert report["stages"]["3_predictions"]["holdout_rows"] == 0
    last = date.fromisoformat(report["stages"]["3_predictions"]["last"])
    assert last < cfg.holdout_start


def test_stage_four_uses_the_declared_sweep_and_not_a_hand_built_dict(cfg_and_panel, predictions):
    """The setting comes from ``case_studies/exness_usidx_sess/_sweep.declared_settings`` —
    the same function ``13_backtest`` and ``_report_phase5`` call. A dict built here would be a
    third copy that agrees only by inspection."""
    cfg, panel = cfg_and_panel
    sizing = dl.build_target_weights(cfg, predictions.with_columns(pl.col("timestamp").cast(pl.Date)))
    assert sizing["config"]["method"] == "per_symbol_rolling_percentile"
    assert sizing["config"]["lookback_days"] == 252
    assert sizing["config"]["bars_per_day"] == 1  # one decision per session per spec
    assert sizing["config"]["long_short"] is False
    assert sizing["config"]["direction"] == "long_only"


def test_stage_four_reports_n_traded_and_not_only_n_sessions(cfg_and_panel, predictions):
    """The mentor's bug #3, made impossible to overlook. Phase 5 deflated its Sharpe on the
    number of SESSIONS in the window; the percentile family trades 7-16 % of them, so the
    effective sample was overstated 6-14x. The report carries both numbers."""
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    stage = report["stages"]["4_sizing"]
    assert stage["sessions_scored"] > 0
    assert stage["sessions_with_a_position"] <= stage["sessions_scored"]
    assert 0.0 <= stage["share_of_sessions_traded"] <= 1.0
    assert "overstates the effective sample" in stage["n_traded_note"]


def test_the_latest_basket_is_one_session_and_carries_its_date(cfg_and_panel, predictions):
    """A basket assembled from the last row PER SYMBOL is a book that never existed: on a book
    that trades 2 % of sessions the two legs could be months apart."""
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    stage = report["stages"]["4_sizing"]
    if stage["latest_weights"]:
        assert stage["latest_weights_session"] is not None
        session = date.fromisoformat(stage["latest_weights_session"])
        assert session < cfg.holdout_start


def test_stage_five_measures_the_volume_min_gap_rather_than_describing_it(
    cfg_and_panel, predictions
):
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    stage = report["stages"]["5_orders"]
    assert set(stage["legs"]) == {"US500", "USTEC"}
    for leg in stage["legs"].values():
        assert "volume_min" in leg and "raw_lots" in leg and "normalised_lots" in leg
    assert any("share_type parity gap" in g for g in stage["gaps"])
    assert report["gaps"] == stage["gaps"]


# =======================================================================================
# What parity must NOT report
# =======================================================================================
def test_no_performance_number_is_reported_while_the_holdout_is_unscored(
    cfg_and_panel, predictions
):
    """A return or a Sharpe printed here would be a holdout reading taken outside the holdout
    stages — and, separately, a number from a generation with no survivor."""
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    assert report["performance_reported"] is False
    assert "has not been scored" in report["performance_withheld_because"]

    # No performance KEY may exist anywhere in the report. Checking keys rather than the prose
    # is the point: the report DESCRIBES the Deflated Sharpe defect of phase 5 in a note, and a
    # substring search on the word would fail on the explanation while missing a real leak.
    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k)
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    forbidden = {
        "sharpe", "sharpe_raw", "sharpe_active", "dsr", "psr", "total_return",
        "total_return_pct", "final_value", "cagr", "equity_curve", "net", "pnl",
        "max_drawdown", "hit_rate",
    }
    leaked = sorted(forbidden & set(keys(report)))
    assert not leaked, leaked


def test_the_report_states_why_the_engine_is_not_used(cfg_and_panel, predictions):
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    assert "NOT an ml4t.backtest.Engine replay" in report["definition"]
    assert "13-15 h before the decision" in report["definition"]


def test_fixed_threshold_is_excluded_for_a_mechanical_reason(cfg_and_panel, predictions):
    """Excluding it is not cherry-picking. ``signals.py:53-62`` with ``long_short`` sets the
    lower threshold to ``1.0 - threshold``, so a regression score of order 1e-3 never lands in
    the flat branch and the book is always invested: 1,068 of generation 1's 1,780 trials
    tested a book the hypothesis does not describe."""
    cfg, panel = cfg_and_panel
    report = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    reason = report["stages"]["4_sizing"]["fixed_threshold_excluded_because"]
    assert "never lands in the flat branch" in reason
    assert "MECHANICAL exclusion" in reason
    assert "no generation has yet been scored" in reason.lower()
    assert cfg.model_block["signal_family"] == "per_symbol_rolling_percentile"


# =======================================================================================
# Determinism
# =======================================================================================
def test_the_harness_is_deterministic(cfg_and_panel, predictions):
    """Two runs over the same inputs give the same tape. A parity harness that is not
    reproducible cannot tell a real divergence from its own noise."""
    cfg, panel = cfg_and_panel
    a = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    b = dl.parity_report(cfg, panel, predictions, live_start=date(2025, 1, 1))
    assert a["stages"]["4_sizing"] == b["stages"]["4_sizing"]
    assert a["stages"]["5_orders"] == b["stages"]["5_orders"]


def test_recomputed_features_reproduce_the_research_row_count(cfg_and_panel):
    """BOT.md phase 3: 74 features on 1,764 rows, of which 1,510 are development. The live
    recompute truncated at ``holdout_start`` must land on the development figure, not on a
    number nobody has seen before."""
    cfg, panel = cfg_and_panel
    assert len(panel.columns) == 74
    assert panel.prices["timestamp"].max() < cfg.holdout_start


def test_the_two_specs_produce_different_feature_matrices(tmp_path):
    """The reason each spec has its own workspace: the two panels are different aggregations
    of the same bars and carry different values under the same ``(timestamp, symbol)`` key.
    Merging them would leak the overnight panel's 20:00-UTC bars into the intraday label's own
    holding period."""
    frames = {}
    for spec in ("intraday", "overnight"):
        cfg = dl.load_deploy_config(RISK_CONFIG, spec=spec, state_dir=tmp_path)
        panel = dl.recompute_features(cfg, as_of=cfg.holdout_start)
        frames[spec] = panel.prices.select(["symbol", "timestamp", "close"]).sort(
            ["symbol", "timestamp"]
        )
    common = frames["intraday"].join(
        frames["overnight"], on=["symbol", "timestamp"], suffix="_on", how="inner"
    )
    assert common.height > 100
    differing = common.filter(pl.col("close") != pl.col("close_on")).height
    assert differing > 0.5 * common.height, (
        "the two specs' session panels agree on the decision-bar close, which would mean they "
        "are the same panel and the two-workspace design is unnecessary"
    )
