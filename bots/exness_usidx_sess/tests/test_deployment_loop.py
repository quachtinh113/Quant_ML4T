"""The deployment loop, against ``bots/_shared/testing/fake_mt5.py`` — never a live terminal.

WHAT THESE TESTS ARE FOR. The bot has 0 phase-5 survivors of 1,403 scored specs at K = 1,780,
phase 6 is not opened and the holdout is unscored, so none of this is evidence of an edge.
What the tests establish is that the CODE PATH is right and, more importantly, that it
**cannot be made to place an order**: every refusal below is asserted, not assumed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from bots._shared import MAGIC_ALLOCATION, RESERVED_MAGICS
from bots.exness_usidx_sess.deploy import deployment_loop as dl

RISK_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "risk_config.yaml"


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(RISK_CONFIG.read_text())


@pytest.fixture
def cfg(tmp_path) -> dl.DeployConfig:
    return dl.load_deploy_config(RISK_CONFIG, spec="intraday", state_dir=tmp_path)


@pytest.fixture
def fake(cfg):
    return dl.make_fake_terminal(cfg)


# =======================================================================================
# Configuration: the cross-checks that stop the live loop drifting from the research
# =======================================================================================
def test_magic_is_the_allocated_one_and_is_not_reserved(cfg):
    assert cfg.magic == MAGIC_ALLOCATION["exness_usidx_sess"] == 260902
    assert cfg.magic not in RESERVED_MAGICS


def test_magic_260902_is_taken_by_no_other_bot():
    """``bots/tests/test_magic_numbers.py`` globs every ``deploy/risk_config.yaml``; this is
    the same assertion from inside the bot, so the collision is caught here too."""
    others = {}
    for path in sorted(Path(__file__).resolve().parents[2].glob("*/deploy/risk_config.yaml")):
        block = yaml.safe_load(path.read_text())
        others[block["bot_id"]] = int(block["magic"])
    assert others["exness_usidx_sess"] == 260902
    assert len(set(others.values())) == len(others), others


def test_config_refuses_a_universe_that_does_not_match_setup_yaml(tmp_path):
    block = yaml.safe_load(RISK_CONFIG.read_text())
    block["live_risk_config"]["allowed_assets"] = ["US500"]
    path = tmp_path / "risk.yaml"
    path.write_text(yaml.safe_dump(block))
    with pytest.raises(ValueError, match="allowed_assets"):
        dl.load_deploy_config(path, spec="intraday", state_dir=tmp_path)


def test_config_refuses_a_decision_block_that_drifts_from_setup_yaml(tmp_path):
    """The live loop must decide at the instant the research panel was built on. A silent
    divergence here would make every parity comparison meaningless."""
    block = yaml.safe_load(RISK_CONFIG.read_text())
    block["decision"]["open_delay_minutes"] = 15
    path = tmp_path / "risk.yaml"
    path.write_text(yaml.safe_dump(block))
    with pytest.raises(ValueError, match="open_delay_minutes"):
        dl.load_deploy_config(path, spec="intraday", state_dir=tmp_path)


def test_config_refuses_a_holdout_window_that_drifts_from_setup_yaml(tmp_path):
    block = yaml.safe_load(RISK_CONFIG.read_text())
    block["evidence_boundary"]["holdout_start"] = "2025-01-01"
    path = tmp_path / "risk.yaml"
    path.write_text(yaml.safe_dump(block))
    with pytest.raises(ValueError, match="holdout_start"):
        dl.load_deploy_config(path, spec="intraday", state_dir=tmp_path)


def test_execution_mode_may_only_be_paper(tmp_path):
    block = yaml.safe_load(RISK_CONFIG.read_text())
    block["execution"]["mode"] = "live"
    path = tmp_path / "risk.yaml"
    path.write_text(yaml.safe_dump(block))
    with pytest.raises(ValueError, match="only be 'paper'"):
        dl.load_deploy_config(path, spec="intraday", state_dir=tmp_path)


def test_the_two_specs_get_two_state_directories(tmp_path):
    a = dl.load_deploy_config(RISK_CONFIG, spec="intraday", state_dir=tmp_path)
    b = dl.load_deploy_config(RISK_CONFIG, spec="overnight", state_dir=tmp_path)
    assert a.state_dir != b.state_dir
    assert a.label == "fwd_ret_intraday" and b.label == "fwd_ret_overnight"


# =======================================================================================
# The evidence boundary
# =======================================================================================
def test_the_evidence_stamp_says_what_the_bot_has_not_proved(cfg):
    stamp = cfg.evidence_stamp()
    assert stamp["phase5_survivors"] == 0
    assert stamp["phase5_specs_scored"] == 1403
    assert stamp["trial_count_K"] == 1780
    assert stamp["phase6_opened"] is False
    assert stamp["holdout_scored"] is False
    assert stamp["kill_criteria_approved"] is False
    assert stamp["live_trading_permitted"] is False


def test_holdout_is_not_scored_and_a_flag_cannot_say_otherwise(cfg):
    """``holdout_scored`` reads a MARKER FILE the holdout stages write. There is no argument,
    environment variable or config key that can make it true."""
    assert cfg.holdout_scored is False
    assert not (Path(dl.DEPLOY_DIR) / "state" / "holdout_scored.json").exists()


def test_a_refit_window_is_clamped_before_the_holdout(cfg):
    assert dl.clamp_training_window(cfg, date(2026, 12, 31)) == cfg.holdout_start - timedelta(
        days=1
    )
    assert dl.clamp_training_window(cfg, date(2025, 1, 1)) == date(2025, 1, 1)


def test_the_default_decision_date_is_the_last_day_before_the_holdout(cfg, fake, tmp_path):
    record = dl.run_cycle(
        spec="intraday",
        risk_config=RISK_CONFIG,
        state_dir=tmp_path,
        mt5_module=fake,
        account_halt_file=tmp_path / "halt.json",
    )
    assert date.fromisoformat(record["decision_date"]) < cfg.holdout_start
    assert record["holdout_window_touched"] is False


# =======================================================================================
# Arming is refused
# =======================================================================================
def test_arm_is_refused_while_the_kill_criteria_are_a_draft(fake, tmp_path):
    with pytest.raises(PermissionError, match="--arm refused"):
        dl.run_cycle(
            spec="intraday",
            arm=True,
            risk_config=RISK_CONFIG,
            state_dir=tmp_path,
            mt5_module=fake,
            account_halt_file=tmp_path / "halt.json",
        )


def test_a_refit_is_refused_because_no_survivor_exists(cfg, fake, tmp_path):
    """``retrain`` never invents a model. Refitting a registered training run that failed
    phase 5 would be presenting a failed trial as a deployment."""
    with pytest.raises(dl.NotReadyError, match="0 survivors"):
        dl.retrain(cfg, None, dl.ModelChoice("abc123", "abc123", None, None))


def test_the_safe_broker_layer_is_shadow_and_arming_can_only_make_it_safer(cfg):
    unarmed = dl.build_live_risk_config(cfg, armed=False)
    armed = dl.build_live_risk_config(cfg, armed=True)
    assert unarmed.execution_mode.value == "shadow"
    # The YAML itself declares shadow, so even arming cannot move it out of shadow.
    assert armed.execution_mode.value == "shadow"
    assert unarmed.shadow_mode is True


def test_live_risk_config_carries_the_declared_limits(cfg, raw):
    block = raw["live_risk_config"]
    risk = dl.build_live_risk_config(cfg, armed=False)
    assert risk.max_drawdown_pct == float(block["max_drawdown_pct"]) == 0.08
    assert risk.max_daily_loss == float(block["max_daily_loss"])
    assert risk.max_positions == 2
    assert risk.allowed_assets == {"US500", "USTEC"}
    assert risk.fail_on_reconciliation_mismatch is True
    assert risk.kill_switch_enabled is False


# =======================================================================================
# The parity gap: fractional research, volume_min live
# =======================================================================================
def test_order_parity_measures_the_volume_min_rejection(cfg):
    """The bot's largest parity gap, measured rather than described: at a weight of 1.0 on
    900 USD of allocation both legs round BELOW ``volume_min`` and the adapter rejects them.
    They are not smaller trades; they are no trade."""
    report = dl.order_parity(cfg, {"US500": 1.0, "USTEC": 1.0})
    for symbol in ("US500", "USTEC"):
        leg = report["legs"][symbol]
        assert leg["raw_lots"] > 0
        assert leg["normalised_lots"] == 0.0
        assert leg["rejected_by_volume_min"] is True
    assert any("share_type parity gap" in g for g in report["gaps"])
    assert report["legs"]["US500"]["min_notional_account"] == pytest.approx(1080.32, abs=0.01)
    assert report["legs"]["USTEC"]["min_notional_account"] == pytest.approx(1482.95, abs=0.01)


def test_order_parity_reads_symbol_info_when_a_broker_is_connected(cfg, fake):
    """Contract size, ``volume_min`` and the price come from ``symbol_info`` and a live tick,
    never from a table. The fallback figures in the YAML are only for an offline run."""
    from bots._shared.mt5_broker import MT5Broker

    broker = MT5Broker(magic=cfg.magic, mt5=fake)
    asyncio.run(broker.connect())  # MT5Broker.connect is a coroutine
    report = dl.order_parity(cfg, {"US500": 1.0, "USTEC": 0.0}, broker=broker)
    assert report["legs"]["US500"]["price_source"] == "symbol_info_tick"
    assert report["legs"]["US500"]["volume_min"] == 0.14
    assert report["legs"]["USTEC"]["contract_size"] == 1.0


def test_a_position_large_enough_to_place_is_not_rejected(cfg):
    """The gap is a FLOOR, not a blanket refusal: raise the weight enough and the leg sizes."""
    from dataclasses import replace

    bigger = replace(cfg, risk={**cfg.risk, "capital": {**cfg.risk["capital"]}})
    bigger.risk["capital"]["allocated"] = 20_000.0
    report = dl.order_parity(bigger, {"US500": 1.0, "USTEC": 1.0})
    assert report["legs"]["US500"]["normalised_lots"] > 0.14
    assert report["legs"]["USTEC"]["normalised_lots"] > 0.05


def test_capital_viability_blocks_at_the_declared_allocation(cfg):
    viability = dl.capital_viability(cfg, None, account_equity=896.97)
    assert viability["viable"] is False
    assert any("min_viable_allocated" in r for r in viability["reasons"])
    assert viability["share_type_in_research"] == "fractional"
    # The arithmetic is in the record, not only in a prose reason.
    assert viability["per_symbol"]["USTEC"]["allocation_needed_for_equal_weight_leg"] == (
        pytest.approx(2965.9, abs=0.1)
    )


# =======================================================================================
# The CFD guard, in venue-local time
# =======================================================================================
def test_the_blocked_window_is_converted_from_new_york_and_moves_with_dst(cfg, fake):
    """17:00-18:00 New York is 21:00-22:00 UTC in summer and 22:00-23:00 UTC in winter. The
    guard reports both the local window and today's UTC equivalent so a reader can check it."""
    from datetime import UTC

    from bots._shared.mt5_broker import MT5Broker

    broker = MT5Broker(magic=cfg.magic, mt5=fake)
    asyncio.run(broker.connect())  # MT5Broker.connect is a coroutine
    summer = dl.cfd_execution_guard(
        cfg, broker, cfg.symbols, datetime(2025, 7, 15, 21, 30, tzinfo=UTC)
    )
    winter = dl.cfd_execution_guard(
        cfg, broker, cfg.symbols, datetime(2025, 1, 15, 21, 30, tzinfo=UTC)
    )
    assert summer["windows"][0]["blocked"] is True
    assert summer["windows"][0]["utc_equivalent_today"] == ["21:00", "22:00"]
    assert winter["windows"][0]["blocked"] is False
    assert winter["windows"][0]["utc_equivalent_today"] == ["22:00", "23:00"]
    assert not summer["ok"]
    assert winter["ok"]


def test_the_spread_guard_uses_the_recent_regime_and_passes_on_todays_quote(cfg, fake, raw):
    from datetime import UTC

    from bots._shared.mt5_broker import MT5Broker

    broker = MT5Broker(magic=cfg.magic, mt5=fake)
    asyncio.run(broker.connect())  # MT5Broker.connect is a coroutine
    guard = dl.cfd_execution_guard(
        cfg, broker, cfg.symbols, datetime(2025, 7, 15, 15, 0, tzinfo=UTC)
    )
    assert guard["ok"] is True
    assumed = raw["cfd_guards"]["assumed_spread_p90_bps"]
    # The guard's p90 is the RECENT-REGIME figure (1.05 / 0.87), not the 4.05 bps the backtest
    # charges: a guard set four times above the live quote would never fire.
    assert assumed["US500"] == 1.05 and assumed["USTEC"] == 0.87
    assert guard["spreads"]["US500"]["spread_bps"] < assumed["US500"]


def test_a_wide_print_is_refused(cfg, fake):
    from datetime import UTC

    from bots._shared.mt5_broker import MT5Broker

    fake.quotes["US500m"] = (7700.0, 7710.0)  # ~13 bps, far above 2 x 1.05
    broker = MT5Broker(magic=cfg.magic, mt5=fake)
    asyncio.run(broker.connect())  # MT5Broker.connect is a coroutine
    guard = dl.cfd_execution_guard(
        cfg, broker, ["US500"], datetime(2025, 7, 15, 15, 0, tzinfo=UTC)
    )
    assert guard["ok"] is False
    assert "above 2x the assumed" in guard["reasons"][0]


# =======================================================================================
# The decision-bar guard, ASYMMETRIC between the two specs
# =======================================================================================
def test_a_missing_intraday_bar_escalates_and_a_missing_overnight_bar_does_not(tmp_path):
    """Kill criterion (e). Measured: 0 of 1,930 research sessions missed the intraday instant
    and 504 of 1,930 missed the overnight one. Treating them alike would page a human 504
    times for market structure ``setup.yaml`` already declares."""
    import polars as pl

    for spec, escalates in (("intraday", True), ("overnight", False)):
        cfg = dl.load_deploy_config(RISK_CONFIG, spec=spec, state_dir=tmp_path)
        panel = dl.FeaturePanel(
            prices=pl.DataFrame(
                {
                    "symbol": ["US500", "USTEC"],
                    "timestamp": [date(2026, 2, 27)] * 2,
                    "exec_open": [None, None],
                }
            ).with_columns(pl.col("exec_open").cast(pl.Float64)),
            features=pl.DataFrame(),
            daily=pl.DataFrame(),
            bars=pl.DataFrame(),
            columns=[],
            spec=spec,
        )
        check = dl.decision_bar_check(cfg, panel, decision_date=date(2026, 2, 27))
        assert check["missing"] == ["US500", "USTEC"]
        assert check["escalates"] is escalates, spec
        if not escalates:
            assert "STRUCTURAL" in check["classification"]


# =======================================================================================
# The whole cycle
# =======================================================================================
def test_the_default_cycle_stops_after_parity_and_stages_nothing(cfg, fake, tmp_path):
    record = dl.run_cycle(
        spec="intraday",
        risk_config=RISK_CONFIG,
        state_dir=tmp_path,
        mt5_module=fake,
        account_halt_file=tmp_path / "halt.json",
    )
    assert record["deployable"] is False
    assert record["signal_status"] == "no_deployable_signal"
    assert "step7_staging" not in record
    assert "step 6" in record["stopped_after"]
    assert record["evidence"]["live_trading_permitted"] is False
    # The run record is on disk and carries the boundary.
    written = json.loads(Path(record["run_record_path"]).read_text())
    assert written["evidence"]["phase5_survivors"] == 0


@pytest.mark.parametrize("spec", ["intraday", "overnight"])
def test_the_plumbing_cycle_reaches_step_seven_and_accepts_nothing(fake, tmp_path, spec):
    record = dl.run_cycle(
        spec=spec,
        model_selector="latest-complete",
        risk_config=RISK_CONFIG,
        state_dir=tmp_path,
        mt5_module=fake,
        account_halt_file=tmp_path / "halt.json",
    )
    staging = record["step7_staging"]
    assert record["signal_status"] == "plumbing_only"
    assert record["deployable"] is False
    assert staging["armed"] is False
    assert staging["execution_mode_broker"] == "paper"
    assert staging["execution_mode_safe"] == "shadow"
    assert staging["orders_can_reach_broker"] is False
    assert staging["accepted_basket"] == []
    assert all(leg["status"] in ("dry_run", "rounds_to_zero", "flat") for leg in staging["legs"])
    # Every reason the loop refuses, in the record rather than in a log line.
    blocked = " | ".join(staging["blocked_by"])
    assert "0 survivors" in blocked
    assert "has not been scored" in blocked
    assert "DRAFT" in blocked
    assert "not armed" in blocked


def test_the_reconciliation_step_recognises_the_reserved_legacy_magic(fake, tmp_path):
    """Two positions of the retired magic 202500 were open on this login on 2026-09-08. They
    are RESERVED, so they are recognised rather than raising; an UNALLOCATED magic must raise."""
    record = dl.run_cycle(
        spec="intraday",
        model_selector="latest-complete",
        risk_config=RISK_CONFIG,
        state_dir=tmp_path,
        mt5_module=fake,
        account_halt_file=tmp_path / "halt.json",
    )
    rec = record["step7_staging"]["reconciliation"]
    assert rec["magic"] == 260902
    assert rec["unknown_magics"] == []
    assert "202500" in rec["note"]


def test_an_unallocated_magic_blocks_the_cycle(fake, tmp_path):
    from types import SimpleNamespace

    fake.positions[999] = SimpleNamespace(
        ticket=999,
        symbol="US500m",
        magic=777777,  # allocated to nothing and not reserved
        volume=0.14,
        type=0,
        price_open=7716.0,
        price_current=7716.0,
        profit=0.0,
        swap=0.0,
        time=0,
        comment="",
    )
    record = dl.run_cycle(
        spec="intraday",
        model_selector="latest-complete",
        risk_config=RISK_CONFIG,
        state_dir=tmp_path,
        mt5_module=fake,
        account_halt_file=tmp_path / "halt.json",
    )
    staging = record["step7_staging"]
    assert staging["reconciliation"]["clean"] is False
    assert "777777" in staging["reconciliation"]["error"]
    assert staging["accepted_basket"] == []


def test_the_run_record_stamps_the_trial_count(fake, tmp_path):
    record = dl.run_cycle(
        spec="intraday",
        risk_config=RISK_CONFIG,
        state_dir=tmp_path,
        mt5_module=fake,
        account_halt_file=tmp_path / "halt.json",
    )
    assert record["evidence"]["trial_count_K"] == 1780
    assert record["breakers"]["pending_user_approval"] is True


def test_the_registry_of_each_spec_is_a_different_file(cfg, raw):
    registries = raw["evidence_boundary"]["registries"]
    assert set(registries) == {"intraday", "overnight"}
    assert registries["intraday"] != registries["overnight"]
    assert "usidx_intraday" in registries["intraday"]
    assert "usidx_overnight" in registries["overnight"]
