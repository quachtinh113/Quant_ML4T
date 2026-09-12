"""bots/exness_fx_d1/deploy/deployment_loop.py — dry run end to end on the fake MT5.

No terminal, no ``order_send``, no network. What is exercised:

* the config gate (decision block, magic, execution mode, evidence boundary must agree with
  ``setup.yaml`` and the allocation table);
* **the evidence boundary**: a deployment refit is clamped before ``holdout_start`` and the
  clamp is asserted, and the parity replay withholds performance while the holdout is unscored;
* the decision-bar rule (BOT.md open question 9 / kill criterion (e)) and the no-trade-day log;
* the seven steps end to end, ``--model none`` stopping after the parity step and
  ``--model latest-complete`` exercising the staging path as plumbing;
* ``armed=False`` refusing to send anything, and the lot arithmetic that would have been sent;
* reconciliation of both views (this magic against ``RiskState``, every magic against the
  allocation table);
* the run record and everything ``26_mlops_governance`` needs from it.

The end-to-end tests read the experiment's registry and labels, so they need
``ML4T_OUTPUT_DIR`` to point at an experiment that has run ``02_labels``, ``06``/``07`` and the
MT5 four-hour parquet. Where that is absent (the Windows download environment) they skip and
the pure-unit tests still run.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

pytest.importorskip("ml4t.live", reason="the deployment loop needs the ml4t research environment")

from bots._shared import MAGIC_ALLOCATION  # noqa: E402
from bots._shared.mt5_loader import to_market_watch  # noqa: E402
from bots.exness_fx_d1.deploy import deployment_loop as loop  # noqa: E402

MAGIC = MAGIC_ALLOCATION["exness_fx_d1"]


# ---------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg():
    return loop.load_deploy_config()


def _experiment_ready(cfg) -> str | None:
    """Why the end-to-end tests cannot run here, or ``None`` when they can."""
    if not loop.registry_path(cfg).exists():
        return f"no registry at {loop.registry_path(cfg)}; set ML4T_OUTPUT_DIR to an experiment"
    if not (cfg.case_dir / "labels").exists():
        return f"no labels in {cfg.case_dir}; run 02_labels in the experiment"
    root = os.environ.get("ML4T_DATA_PATH")
    if root and not (Path(root) / "mt5" / "4h.parquet").exists():
        return "no MT5 four-hour parquet"
    return None


@pytest.fixture(scope="module")
def live(cfg):
    reason = _experiment_ready(cfg)
    if reason:
        pytest.skip(reason)
    return cfg


@pytest.fixture
def fake(cfg):
    return loop.make_fake_terminal(cfg)


def viable_config(tmp_path: Path) -> Path:
    """A copy of the shipped config funded at `capital.min_viable_allocated`.

    The shipped `capital.allocated` is 900 USD, the equity measured on the demo login on
    2026-09-07, and at that size every k=1 leg rounds below `volume_min` (see
    `TestTheShippedAllocationCannotFundTheBook`). Sizing and order-parity assertions need a
    book that can actually be sized, so they run against this copy; nothing else changes.
    """
    raw = yaml.safe_load(loop.DEFAULT_RISK_CONFIG.read_text())
    raw["capital"]["allocated"] = float(raw["capital"]["min_viable_allocated"])
    path = tmp_path / "risk_config_viable.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.fixture
def sandbox(tmp_path):
    """State, halt file and high-water mark all inside the test's own directory."""
    return {
        "state_dir": tmp_path / "state",
        "halt_file": tmp_path / "monitor" / "account_halt.json",
    }


def _cycle(cfg, fake, sandbox, **kwargs):
    return loop.run_cycle(mt5=fake, state_dir=sandbox["state_dir"], halt_file=sandbox["halt_file"], **kwargs)


# =======================================================================================
# The configuration gate
# =======================================================================================
class TestTheConfigurationGate:
    def test_the_decision_block_must_match_setup_yaml(self, cfg):
        assert cfg.risk["decision"]["snapshot"] == cfg.setup["decision"]["snapshot"]
        assert cfg.snapshot_utc == time(20, 0)
        assert cfg.tolerance_minutes == 240

    def test_a_drifted_decision_block_is_refused(self, cfg, tmp_path):
        raw = yaml.safe_load(cfg.risk_path.read_text())
        raw["decision"]["snapshot_utc"] = "08:00"
        path = tmp_path / "risk.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="different instant from the research pipeline"):
            loop.load_deploy_config(path)

    def test_the_magic_must_be_the_allocated_one(self, cfg, tmp_path):
        assert cfg.magic == MAGIC == 260901
        raw = yaml.safe_load(cfg.risk_path.read_text())
        raw["magic"] = 202500  # the retired legacy bot's default
        path = tmp_path / "risk.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="allocation table"):
            loop.load_deploy_config(path)

    def test_live_execution_mode_is_refused_outright(self, cfg, tmp_path):
        raw = yaml.safe_load(cfg.risk_path.read_text())
        raw["execution"]["mode"] = "live"
        path = tmp_path / "risk.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="only run in paper mode until phase 7"):
            loop.load_deploy_config(path)

    def test_the_evidence_boundary_must_match_setup_yaml(self, cfg, tmp_path):
        raw = yaml.safe_load(cfg.risk_path.read_text())
        raw["evidence_boundary"]["holdout_start"] = "2024-01-01"
        path = tmp_path / "risk.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="evidence_boundary.holdout_start"):
            loop.load_deploy_config(path)

    def test_the_shipped_config_is_a_dry_run_with_no_model(self, cfg):
        assert cfg.risk["execution"]["armed"] is False
        assert cfg.live_risk_block["execution_mode"] == "shadow"
        # One switch: shadow_mode is derived by LiveRiskConfig and must not be in the YAML,
        # because `shadow_mode: true` beside a non-shadow execution_mode raises
        # ExecutionModeError (ml4t/live/safety.py:153-160).
        assert "shadow_mode" not in cfg.live_risk_block
        assert cfg.model_block["training_hash"] is None
        assert loop.SUBMIT_ORDERS is False
        assert loop.MODEL_SELECTOR == "none"

    def test_the_two_layers_carry_different_modes_on_purpose(self, cfg):
        """SafeBroker is `shadow`; the broker adapter is `paper`.

        `shadow` stops SafeBroker from routing the order at all. `paper` is what makes
        `MT5Broker._guard_mode` run `assert_paper_trading()` and prove the account is a demo
        (mt5_broker.py:461-467), so anything that ever got past SafeBroker would still have to
        clear that. Setting the broker layer to `shadow` would fail closed with the wrong
        error.
        """
        assert cfg.live_risk_block["execution_mode"] == "shadow"
        assert cfg.risk["execution"]["mode"] == "paper"

    def test_the_order_rate_cap_matches_one_daily_rotation(self, cfg):
        # k=1 long-short: 2 closes + 2 opens.
        assert cfg.live_risk_block["max_orders_per_minute"] == 4


# =======================================================================================
# The evidence boundary
# =======================================================================================
class TestTheEvidenceBoundary:
    def test_the_holdout_marker_is_absent_so_the_boundary_is_shut(self, cfg):
        assert cfg.holdout_scored is False
        assert cfg.holdout_start == date(2025, 9, 1)

    def test_the_training_window_is_clamped_before_the_holdout(self, live):
        panel = loop.recompute_features(live)
        window = loop.training_window(live, panel, "fwd_ret_1d")
        assert window["train_end"] < live.holdout_start
        assert window["train_start"] < window["train_end"]

    def test_a_panel_reaching_into_the_holdout_still_clamps(self, live):
        panel = loop.recompute_features(live, as_of=date(2026, 8, 28))
        window = loop.training_window(live, panel, "fwd_ret_1d")
        assert window["clamped_to_holdout_boundary"] is True
        assert window["train_end"] < live.holdout_start

    def test_the_default_decision_date_stays_out_of_the_holdout(self, live):
        panel = loop.recompute_features(live)
        assert panel.decision_date < live.holdout_start
        assert panel.record["holdout_window_touched"] is False

    def test_registered_validation_predictions_never_reach_the_holdout(self, live):
        choice = loop.resolve_model(live, "none")
        predictions = loop.read_registered_predictions(live, choice)
        assert predictions["timestamp"].max() < live.holdout_start

    def test_the_parity_replay_withholds_performance(self, live, fake, sandbox):
        run = _cycle(live, fake, sandbox, model_selector="none")
        parity = run["steps"]["6_parity"]
        assert parity["performance_reported"] is False
        assert "final_value" not in parity
        assert "total_return_pct" not in parity
        assert "holdout" in parity["performance_withheld_because"]


# =======================================================================================
# The decision bar (open question 9 / kill criterion (e))
# =======================================================================================
class TestTheDecisionBarRule:
    def test_a_healthy_session_reports_a_lag_inside_the_tolerance(self, live):
        panel = loop.recompute_features(live)
        check = loop.decision_bar_check(live, panel)
        assert check["ok"] is True
        assert 0 <= check["max_lag_minutes"] <= live.tolerance_minutes
        assert len(check["per_symbol"]) == len(live.universe)

    def test_a_missing_pair_is_a_no_trade_day(self, live):
        panel = loop.recompute_features(live)
        panel.missing_symbols = ["USDJPY"]
        check = loop.decision_bar_check(live, panel)
        assert check["ok"] is False
        assert "no decision bar for ['USDJPY']" in check["reasons"][0]

    def test_a_stale_bar_beyond_the_tolerance_is_a_no_trade_day(self, live):
        panel = loop.recompute_features(live)
        stale = panel.prices.with_columns(
            pl.when(pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date))
            .then(pl.col("decision_ts") - pl.duration(hours=12))
            .otherwise(pl.col("decision_ts"))
            .alias("decision_ts")
        )
        panel.prices = stale
        check = loop.decision_bar_check(live, panel)
        assert check["ok"] is False
        assert "above the 240 min tolerance" in check["reasons"][0]

    def test_a_bar_closing_after_the_snapshot_is_lookahead_and_is_refused(self, live):
        panel = loop.recompute_features(live)
        ahead = panel.prices.with_columns(
            pl.when(pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date))
            .then(pl.col("decision_ts") + pl.duration(hours=2))
            .otherwise(pl.col("decision_ts"))
            .alias("decision_ts")
        )
        panel.prices = ahead
        check = loop.decision_bar_check(live, panel)
        assert check["ok"] is False
        assert "after the" in check["reasons"][0]

    def test_a_no_trade_day_is_logged_and_escalates_only_when_it_repeats(self, cfg, tmp_path):
        state = tmp_path / "state"
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, state)
        first = loop.record_no_trade_day(config, date(2026, 1, 5), "16:00 UTC bar missing")
        assert first["action"] == "no_trade_today"
        assert first["count_in_window"] == 1
        assert first["escalated"] is False
        for day in (6, 7):
            last = loop.record_no_trade_day(config, date(2026, 1, day), "bar missing")
        assert last["count_in_window"] == 3
        assert last["escalated"] is True
        logged = json.loads((state / "no_trade_days.json").read_text())["days"]
        assert [d["date"] for d in logged] == ["2026-01-05", "2026-01-06", "2026-01-07"]

    def test_the_same_day_is_not_logged_twice(self, cfg, tmp_path):
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, tmp_path / "s")
        loop.record_no_trade_day(config, date(2026, 1, 5), "x")
        second = loop.record_no_trade_day(config, date(2026, 1, 5), "x")
        assert second["count_in_window"] == 1


# =======================================================================================
# Model resolution
# =======================================================================================
class TestModelResolution:
    def test_none_resolves_to_a_registered_validation_prediction_set(self, live):
        choice = loop.resolve_model(live, "none")
        assert choice.mode == "none"
        assert choice.prediction_hash
        assert choice.label in {"fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d"}

    def test_latest_complete_is_the_same_row_labelled_plumbing(self, live):
        assert loop.resolve_model(live, "latest-complete").mode == "plumbing"

    def test_an_unregistered_hash_is_an_error_not_a_fallback(self, live):
        with pytest.raises(LookupError, match="not registered"):
            loop.resolve_model(live, "deadbeefcafe")

    def test_a_missing_registry_is_an_error(self, cfg, tmp_path):
        broken = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, tmp_path, tmp_path)
        with pytest.raises(FileNotFoundError, match="no registry"):
            loop.resolve_model(broken, "latest-complete")


# =======================================================================================
# Sizing: units, lots and the granularity floor
# =======================================================================================
class TestSizingAndTheGranularityFloor:
    def test_units_are_notional_over_the_account_value_of_one_unit(self):
        units = loop.target_units(
            {"EURUSD": 1.0, "AUDUSD": -1.0}, {"EURUSD": 1.16, "AUDUSD": 0.65}, 10_000.0
        )
        assert units["EURUSD"] == pytest.approx(10_000 / 1.16)
        assert units["AUDUSD"] == pytest.approx(-10_000 / 0.65)

    def test_a_jpy_quoted_pair_is_sized_on_its_unit_value_not_its_quote(self, cfg, fake):
        """One unit of USDJPY is one US dollar, not one yen.

        An MT5 lot is 100,000 units of the BASE currency, so 0.024 lots of USDJPY is 2,400 USD
        of exposure. Dividing the notional by the quote would ask for 154x too few units and
        therefore 154x too small a position. `unit_values` reads `currency_base` /
        `currency_profit` from `symbol_info`: USD is the base, so a unit is worth exactly
        1.00 USD and no second quote is consulted at all.
        """
        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=MAGIC, mt5=fake)
        broker.account_snapshot = {"currency": "USD"}
        values = loop.unit_values(broker, {"USDJPY": 154.304, "EURUSD": 1.1629})
        assert values["USDJPY"] == 1.0
        assert values["EURUSD"] == pytest.approx(1.1629)
        units = loop.target_units({"USDJPY": 1.0}, values, 10_000.0)
        assert units["USDJPY"] == pytest.approx(10_000.0)

    def test_a_pair_without_a_price_is_dropped_rather_than_guessed(self):
        assert loop.target_units({"EURUSD": 1.0}, {}, 10_000.0) == {}

    def test_a_genuine_cross_whose_leg_is_not_quoted_is_dropped_not_sized_on_a_guess(self, cfg):
        """EURJPY on a USD account: neither leg is USD, so it needs the USDJPY cross."""
        from bots._shared.mt5_broker import MT5Broker
        from bots._shared.mt5_loader import to_market_watch
        from bots._shared.testing.fake_mt5 import FakeMT5, default_symbol_info

        name = to_market_watch("EURJPY")
        info = default_symbol_info(name, currency_base="EUR", currency_profit="JPY")
        only_the_cross = FakeMT5({}, symbols={name: info})
        only_the_cross.initialize()
        broker = MT5Broker(magic=MAGIC, mt5=only_the_cross)
        broker.account_snapshot = {"currency": "USD"}
        assert loop.unit_values(broker, {"EURJPY": 168.0}) == {}

    def test_the_minimum_tradable_notional_is_reported_per_pair(self, cfg, fake):
        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=MAGIC, mt5=fake)
        report = loop.min_tradable_notional(broker, ["EURUSD"], {"EURUSD": 1.16})
        entry = report["EURUSD"]
        # volume_min 0.01 x contract 100,000 = 1,000 units, about 1,160 USD on EURUSD.
        assert entry["min_units"] == pytest.approx(1_000.0)
        assert entry["min_notional_account"] == pytest.approx(1_160.0)

    def test_a_leg_below_the_floor_rounds_to_zero_and_is_refused_not_shrunk(self, cfg, fake):
        import asyncio

        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=MAGIC, mt5=fake)
        leg = asyncio.run(
            loop._stage_leg(None, broker, symbol="EURUSD", units=500.0, price=1.16, weight=0.05, blocked=False)
        )
        assert leg["status"] == "rounds_to_zero"
        assert leg["lots"] == 0.0
        assert "below volume_min" in leg["detail"]


# =======================================================================================
# CFD execution guards
# =======================================================================================
class TestTheCfdExecutionGuards:
    def test_the_post_new_york_gap_is_a_blocked_execution_window(self, cfg, fake):
        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=MAGIC, mt5=fake)
        inside = datetime(2026, 9, 7, 21, 30, tzinfo=UTC)
        guard = loop.cfd_execution_guard(cfg, broker, ["EURUSD"], inside)
        assert guard["ok"] is False
        assert "post-New-York gap" in guard["reasons"][0]

    def test_a_spread_above_twice_the_assumed_p90_blocks_the_leg(self, cfg, fake):
        from bots._shared.mt5_broker import MT5Broker

        fake.quotes[to_market_watch("EURUSD")] = (1.1000, 1.1010)  # ~9 bps
        broker = MT5Broker(magic=MAGIC, mt5=fake)
        guard = loop.cfd_execution_guard(cfg, broker, ["EURUSD"], datetime(2026, 9, 7, 20, 0, tzinfo=UTC))
        assert guard["ok"] is False
        assert "above 2x the assumed p90" in guard["reasons"][0]
        assert guard["spreads"]["EURUSD"]["blocked"] is True

    def test_a_normal_in_session_spread_passes(self, cfg, fake):
        from bots._shared.mt5_broker import MT5Broker

        fake.quotes[to_market_watch("EURUSD")] = (1.16000, 1.16007)  # 0.6 bps
        broker = MT5Broker(magic=MAGIC, mt5=fake)
        guard = loop.cfd_execution_guard(cfg, broker, ["EURUSD"], datetime(2026, 9, 7, 20, 0, tzinfo=UTC))
        assert guard["ok"] is True

    def test_a_closed_market_refuses_because_nothing_quotes(self, cfg):
        from bots._shared.mt5_broker import MT5Broker
        from bots._shared.testing.fake_mt5 import FakeMT5

        empty = FakeMT5({})
        empty.initialize()
        broker = MT5Broker(magic=MAGIC, mt5=empty)
        guard = loop.cfd_execution_guard(cfg, broker, ["EURUSD"], datetime(2026, 9, 7, 20, 0, tzinfo=UTC))
        assert guard["ok"] is False
        assert "FX week is closed" in guard["reasons"][-1]


# =======================================================================================
# LiveRiskConfig
# =======================================================================================
class TestTheLiveRiskConfig:
    def test_the_shipped_yaml_actually_constructs_a_LiveRiskConfig(self, cfg, tmp_path):
        """The trap this test exists for: shadow_mode x execution_mode.

        `LiveRiskConfig.__post_init__` raises `ExecutionModeError` when `shadow_mode=True`
        sits beside a non-shadow `execution_mode` (ml4t/live/safety.py:153-160), and it
        validates every limit as a positive number and every percentage on the open interval
        (0, 1). A YAML that fails any of those would blow up on the first cycle, not in
        review, so it is asserted here before anything else in the loop is exercised.
        """
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, tmp_path)
        built = loop.build_live_risk_config(config, armed=False)
        from ml4t.live import ExecutionMode, LiveRiskConfig

        assert isinstance(built, LiveRiskConfig)
        assert built.execution_mode is ExecutionMode.SHADOW
        assert built.shadow_mode is True  # derived, never passed
        assert 0.0 < built.max_drawdown_pct < 1.0
        assert built.max_daily_loss > 0

    def test_the_loss_limits_are_the_kill_criteria(self, cfg, tmp_path):
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, tmp_path)
        built = loop.build_live_risk_config(config, armed=False)
        breakers = cfg.risk["breakers"]
        # (a) drawdown from peak > 8 % of allocated capital, enforced pre-trade here and
        # again by DrawdownBreaker in monitor/circuit_breakers.py.
        assert built.max_drawdown_pct == pytest.approx(0.08)
        assert built.max_drawdown_pct == pytest.approx(breakers["drawdown"]["max_drawdown_pct"])
        # the derived daily loss, as an absolute figure against the minimum viable allocation
        assert built.max_daily_loss == pytest.approx(
            breakers["daily_loss"]["max_daily_loss_pct"] * cfg.risk["capital"]["min_viable_allocated"]
        )

    def test_every_limit_comes_from_the_yaml(self, cfg, tmp_path):
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, tmp_path)
        built = loop.build_live_risk_config(config, armed=False)
        block = cfg.live_risk_block
        assert built.max_order_shares == block["max_order_shares"]
        assert built.max_position_value == block["max_position_value"]
        assert built.max_orders_per_minute == block["max_orders_per_minute"]
        assert built.allowed_assets == set(block["allowed_assets"])
        assert built.fail_on_reconciliation_mismatch is True

    def test_not_arming_forces_shadow_mode(self, cfg, tmp_path):
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, tmp_path)
        assert loop.build_live_risk_config(config, armed=False).shadow_mode is True

    def test_arming_cannot_switch_shadow_mode_off_while_the_yaml_says_true(self, cfg, tmp_path):
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, tmp_path)
        armed = loop.build_live_risk_config(config, armed=True)
        assert armed.shadow_mode is True, "arming must never be able to loosen the config"

    def test_the_state_and_journal_files_land_in_the_state_dir(self, cfg, tmp_path):
        config = loop.DeployConfig(cfg.risk, cfg.setup, cfg.risk_path, cfg.case_dir, tmp_path)
        built = loop.build_live_risk_config(config, armed=False)
        assert Path(built.state_file).parent == tmp_path.resolve()
        assert Path(built.journal_file).parent == tmp_path.resolve()


# =======================================================================================
# The cycle, end to end on the fake MT5
# =======================================================================================
class TestTheDefaultDryRunStopsAfterTheParityStep:
    @pytest.fixture(scope="class")
    def run(self, request):
        cfg = loop.load_deploy_config()
        reason = _experiment_ready(cfg)
        if reason:
            pytest.skip(reason)
        tmp = request.getfixturevalue("tmp_path_factory").mktemp("default_cycle")
        return loop.run_cycle(
            model_selector="none",
            mt5=loop.make_fake_terminal(cfg),
            state_dir=tmp / "state",
            halt_file=tmp / "halt.json",
        )

    def test_it_reports_no_deployable_signal(self, run):
        assert run["signal_status"] == "no_deployable_signal"
        assert run["deployable"] is False
        assert "K = 3,204" in run["not_deployable_because"]
        assert run["trial_count_K"] == 3204

    def test_it_ends_after_step_six(self, run):
        assert run["ended_after_step"] == 6
        assert run["steps"]["7_stage"]["skipped"] is True

    def test_it_does_not_refit(self, run):
        assert run["steps"]["3_retrain"]["skipped"] is True
        assert "No refit, no data extension, no holdout read." in run["steps"]["3_retrain"]["reason"]
        assert run["steps"]["4_persist"]["skipped"] is True

    def test_the_features_come_from_the_research_module(self, run):
        assert run["steps"]["2_features"]["feature_source"].startswith("case_studies/exness_fx_d1/_features.py")

    def test_the_parity_tape_uses_the_research_engine_configuration(self, run):
        engine = run["steps"]["6_parity"]["engine"]
        assert engine["source"].endswith("get_backtest_config('exness_fx_d1')")
        assert engine["commission_bps"] == 0.0  # Pro account
        assert engine["slippage_bps"] == pytest.approx(1.3)  # top of the measured spread range
        # Mentor fleet gate, 2026-09-10, item B: mt5_lot -> "fractional" (not "integer"), mirrored
        # from case_studies/utils/backtest_presets.py -- see GEN3_DECLARATION_2026-09-10.md.
        assert engine["share_type"] == "fractional"
        assert engine["long_short"] is True

    def test_the_run_record_is_written_and_reloadable(self, run):
        path = Path(run["run_record"])
        assert path.exists()
        record = json.loads(path.read_text())
        assert record["magic"] == MAGIC
        assert record["decision_date"] == run["decision_date"]
        assert record["holdout"]["scored"] is False


class TestThePlumbingRunExercisesTheStagingPath:
    @pytest.fixture(scope="class")
    def run(self, request):
        cfg = loop.load_deploy_config()
        reason = _experiment_ready(cfg)
        if reason:
            pytest.skip(reason)
        tmp = request.getfixturevalue("tmp_path_factory").mktemp("plumbing_cycle")
        return loop.run_cycle(
            model_selector="latest-complete",
            arm=True,  # even armed, a plumbing run may not send
            mt5=loop.make_fake_terminal(cfg),
            state_dir=tmp / "state",
            halt_file=tmp / "halt.json",
            risk_config=viable_config(tmp),
        )

    def test_it_is_labelled_plumbing_and_is_not_deployable(self, run):
        assert run["signal_status"] == "plumbing_only"
        assert run["deployable"] is False
        assert run["ended_after_step"] == 7

    def test_arming_a_plumbing_run_still_sends_nothing(self, run):
        stage = run["steps"]["7_stage"]
        assert stage["armed"] is False
        assert stage["shadow_mode"] is True
        assert stage["orders_can_reach_broker"] is False
        assert stage["accepted_basket"] == []
        assert stage["attempted_basket"] == []
        assert all(leg["status"] == "dry_run" for leg in stage["legs"])
        assert stage["legs"], "the plumbing run staged no leg at all"

    def test_the_refusal_reasons_are_recorded(self, run):
        blocked = run["steps"]["7_stage"]["blocked_by"]
        assert any("plumbing run" in r for r in blocked)
        assert any("not armed" in r for r in blocked)

    def test_the_account_is_verified_as_a_demo_account(self, run):
        assert run["steps"]["7_stage"]["account"]["trade_mode"] == 0

    def test_reconciliation_covers_both_views(self, run):
        rec = run["steps"]["7_stage"]["reconciliation"]
        assert rec["clean"] is True
        assert rec["magic"] == MAGIC
        assert "live_positions_this_magic" in rec
        assert rec["unknown_magics"] == []

    def test_the_account_breakers_ran_and_allowed_trading(self, run):
        breakers = run["steps"]["7_stage"]["account_breakers"]
        assert breakers["allows_trading"] is True
        assert set(breakers["breakers"]) == {
            "account_drawdown",
            "account_daily_loss",
            "account_margin",
            "account_stop_out",
            "account_exposure",
        }

    def test_the_legs_carry_units_lots_and_the_contract_size_read_from_symbol_info(self, run):
        from bots._shared.mt5_broker import normalize_lot
        from bots._shared.mt5_loader import to_market_watch
        from bots._shared.testing.fake_mt5 import default_symbol_info

        for leg in run["steps"]["7_stage"]["legs"]:
            assert leg["contract_size"] == 100_000.0
            info = default_symbol_info(to_market_watch(leg["symbol"]))
            assert leg["lots"] == normalize_lot(
                abs(leg["units"]) / leg["contract_size"], info, min_lot_policy="reject"
            )
            assert leg["side"] in ("buy", "sell")

    def test_the_capital_is_viable_at_the_minimum_declared_allocation(self, run):
        viability = run["steps"]["7_stage"]["capital_viability"]
        assert viability["viable"] is True
        assert viability["reasons"] == []

    def test_the_run_record_carries_what_chapter_26_needs(self, run):
        stage = run["steps"]["7_stage"]
        assert set(stage) >= {
            "intended_basket",
            "attempted_basket",
            "accepted_basket",
            "failed_basket",
            "min_tradable",
            "cfd_guard",
            "capital_viability",
        }
        assert run["decision_bar_lag_minutes"] is not None
        assert run["model"]["prediction_hash"]
        assert run["holdout_window_touched"] is False


class TestAnUnexplainedPositionStopsTheCycle:
    def test_a_position_with_an_unallocated_magic_blocks_staging(self, live, sandbox):
        from types import SimpleNamespace

        fake = loop.make_fake_terminal(live)
        fake.positions = {
            9: SimpleNamespace(
                ticket=9,
                symbol=to_market_watch("EURUSD"),
                magic=777777,  # in no bot's allocation and not the reserved legacy magic
                type=0,
                volume=0.10,
                price_open=1.16,
                price_current=1.16,
                profit=0.0,
                swap=0.0,
                comment="",
                time=0,
                time_msc=0,
            )
        }
        run = _cycle(live, fake, sandbox, model_selector="latest-complete")
        stage = run["steps"]["7_stage"]
        assert stage["reconciliation"]["clean"] is False
        assert "unallocated magic" in stage["reconciliation"]["error"]
        assert any("unallocated magic" in r for r in stage["blocked_by"])
        assert stage["accepted_basket"] == []

    def test_a_position_of_another_bots_magic_is_reported_not_treated_as_foreign(self, live, sandbox):
        from types import SimpleNamespace

        fake = loop.make_fake_terminal(live)
        fake.positions = {
            8: SimpleNamespace(
                ticket=8,
                symbol=to_market_watch("EURUSD"),
                magic=MAGIC_ALLOCATION["exness_btc_8h"],
                type=0,
                volume=0.10,
                price_open=1.16,
                price_current=1.16,
                profit=0.0,
                swap=0.0,
                comment="",
                time=0,
                time_msc=0,
            )
        }
        run = _cycle(live, fake, sandbox, model_selector="latest-complete")
        rec = run["steps"]["7_stage"]["reconciliation"]
        assert rec["clean"] is True
        assert rec["account_positions_by_magic"] == {str(MAGIC_ALLOCATION["exness_btc_8h"]): 1}


class TestAnAccountHaltStopsThisBot:
    def test_a_halt_file_written_by_another_bot_blocks_staging(self, live, sandbox):
        halt = sandbox["halt_file"]
        halt.parent.mkdir(parents=True, exist_ok=True)
        halt.write_text(json.dumps({"raised_by_bot": "exness_btc_8h", "reason": "margin level"}))
        run = _cycle(live, loop.make_fake_terminal(live), sandbox, model_selector="latest-complete")
        blocked = run["steps"]["7_stage"]["blocked_by"]
        assert any("account halt file present" in r for r in blocked)
        assert run["steps"]["7_stage"]["accepted_basket"] == []


class TestTheShippedAllocationCannotFundTheBook:
    """Measured 2026-09-07: the demo login holds 898.56 USD, and 900 USD cannot buy a lot.

    `volume_min 0.01 x trade_contract_size 100,000` = 1,000 units, about 1,163 USD of EURUSD
    at the price read from the terminal that day. A k=1 leg at 900 USD is 0.0077 lots, which
    `normalize_lot` rounds down to zero and rejects. The loop must say so, not emit a basket
    of refusals that reads like a broker problem.
    """

    def test_the_shipped_config_declares_an_allocation_below_its_own_minimum(self, cfg):
        assert cfg.allocated == pytest.approx(900.0)
        assert cfg.allocated < float(cfg.risk["capital"]["min_viable_allocated"])

    def test_capital_viability_reports_the_arithmetic_and_refuses(self, cfg, fake):
        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=MAGIC, mt5=fake)
        report = loop.capital_viability(cfg, broker, {"EURUSD": 1.1629}, account_equity=898.56)
        assert report["viable"] is False
        entry = report["per_pair"]["EURUSD"]
        assert entry["rounds_to_zero"] is True
        assert entry["leg_lots"] == pytest.approx(900.0 / 1.1629 / 100_000, rel=1e-6)
        assert entry["min_notional_account"] == pytest.approx(1_162.9, rel=1e-4)
        assert any("below volume_min" in r for r in report["reasons"])
        assert any("min_viable_allocated" in r for r in report["reasons"])

    def test_an_allocation_above_the_account_equity_is_refused(self, cfg, fake):
        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=MAGIC, mt5=fake)
        report = loop.capital_viability(cfg, broker, {"EURUSD": 1.1629}, account_equity=100.0)
        assert any("exceeds the account equity" in r for r in report["reasons"])

    def test_at_the_minimum_viable_allocation_it_becomes_viable(self, tmp_path, fake):
        from bots._shared.mt5_broker import MT5Broker

        config = loop.load_deploy_config(viable_config(tmp_path))
        broker = MT5Broker(magic=MAGIC, mt5=fake)
        report = loop.capital_viability(config, broker, {"EURUSD": 1.1629}, account_equity=5_000.0)
        assert report["viable"] is True
        assert report["per_pair"]["EURUSD"]["rounds_to_zero"] is False

    def test_a_cycle_on_the_shipped_config_blocks_staging_and_says_why(self, live, sandbox):
        run = _cycle(live, loop.make_fake_terminal(live), sandbox, model_selector="latest-complete")
        stage = run["steps"]["7_stage"]
        assert stage["capital_viability"]["viable"] is False
        assert any("below volume_min" in r for r in stage["blocked_by"])
        assert stage["accepted_basket"] == []
        assert stage["attempted_basket"] == []
        # Not every pair rounds to zero at 900 USD - AUDUSD at 0.65 gives 1,385 units, which
        # is 0.0138 lots and survives the round-down to 0.01. That is exactly why the
        # viability check is a whole-book gate and not a per-leg one: a book that can open one
        # leg of a two-legged spread is worse than a book that opens neither.
        assert any(leg["status"] == "rounds_to_zero" for leg in stage["legs"])
        assert all(leg["status"] in ("rounds_to_zero", "dry_run") for leg in stage["legs"])


class TestAFakeTerminalNeverTouchesTheSharedMonitorDirectory:
    """A synthetic account must not write facts about the real login.

    Measured on 2026-09-07: a fake-terminal smoke wrote `peak_equity: 10000` into
    `ML4T_DATA_PATH/mt5/monitor/account_hwm.json` while the real demo login held 898.56 USD.
    The next real cycle would have read that peak as a 91 % drawdown and halted every bot on
    the account.
    """

    def test_the_fake_is_recognised(self, cfg, fake):
        assert loop._is_fake_terminal(fake) is True
        assert loop._is_fake_terminal(None) is False
        assert loop._is_fake_terminal(object()) is False

    def test_a_fake_run_keeps_the_halt_and_peak_files_in_its_own_state_directory(self, live, tmp_path):
        state = tmp_path / "state"
        run = loop.run_cycle(
            model_selector="latest-complete",
            mt5=loop.make_fake_terminal(live),
            state_dir=state,
        )
        record = run["steps"]["7_stage"]["account_breakers"]
        assert Path(record["halt_file"]).parent == state.resolve() or Path(
            record["halt_file"]
        ).is_relative_to(state)
        shared = os.environ.get("ML4T_DATA_PATH")
        if shared:
            assert not str(record["halt_file"]).startswith(str(Path(shared) / "mt5" / "monitor"))

    def test_an_explicit_halt_file_is_still_honoured(self, live, sandbox):
        run = _cycle(live, loop.make_fake_terminal(live), sandbox, model_selector="latest-complete")
        assert run["steps"]["7_stage"]["account_breakers"]["halt_file"] == str(sandbox["halt_file"])
