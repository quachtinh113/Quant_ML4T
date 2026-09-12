"""bots/exness_fx_d1/monitor: drift diagnostics, online detectors and the rollout gate.

Synthetic streams only - no MT5, no registry, no network. The point of these tests is not that
PSI is computed correctly in the abstract (that is the notebook's own function) but that the
three rules this bot added hold:

* the drift reference window is train/validation and the holdout is refused;
* the online detectors report **readiness**, so "no alert" on a short stream cannot be read as
  "no drift";
* the rollout gate is fixed before a challenger runs, and returns ``NOT_COMPUTABLE`` while the
  holdout is unscored rather than pretending the question was settled.
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from bots.exness_fx_d1.monitor import drift, load_monitor_config, online_detectors, rollout

SETUP = {"evaluation": {"holdout_start": "2025-09-01", "holdout_end": "2026-08-31"}}
SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY"]


def frame(start: date, n_sessions: int, *, shift: float = 0.0, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sessions):
        stamp = start + timedelta(days=i)
        for symbol in SYMBOLS:
            rows.append(
                {
                    "timestamp": stamp,
                    "symbol": symbol,
                    "usd_corr_21d": float(rng.normal(shift, 1.0)),
                    "usd_corr_63d": float(rng.normal(shift, 1.0)),
                    "gold_corr_21d": float(rng.normal(0.0, 1.0)),
                    "gold_corr_63d": float(rng.normal(0.0, 1.0)),
                    "mom_skip_recent": float(rng.normal(0.0, 1.0)),
                    "zscore_21d": float(rng.normal(0.0, 1.0)),
                    "vol_gk_21d": float(rng.normal(0.0, 1.0)),
                    "price_to_ma_21d": float(rng.normal(0.0, 1.0)),
                }
            )
    return pl.DataFrame(rows)


def stream(start: date, n_sessions: int, *, hit: float, seed: int = 0) -> pl.DataFrame:
    """A ``timestamp, symbol, score, actual`` stream whose hit rate is about ``hit``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sessions):
        stamp = start + timedelta(days=i)
        for symbol in SYMBOLS:
            score = float(rng.normal(0, 0.001))
            right = rng.random() < hit
            actual = abs(rng.normal(0, 0.003)) * (1 if (score > 0) == right else -1)
            rows.append({"timestamp": stamp, "symbol": symbol, "score": score, "actual": actual})
    return pl.DataFrame(rows)


# =======================================================================================
# The evidence boundary
# =======================================================================================
class TestTheDriftReferenceIsValidationNeverTheHoldout:
    def test_the_declared_reference_window_ends_before_the_holdout(self):
        block = load_monitor_config()["drift"]["reference"]
        assert date.fromisoformat(block["end"]) < date.fromisoformat(SETUP["evaluation"]["holdout_start"])

    def test_a_frame_reaching_into_the_holdout_is_refused(self):
        with pytest.raises(ValueError, match="inside the declared holdout"):
            drift.assert_no_holdout(frame(date(2025, 8, 20), 30), SETUP)

    def test_the_allow_flag_alone_cannot_open_the_boundary(self, tmp_path):
        with pytest.raises(ValueError, match="inside the declared holdout"):
            drift.assert_no_holdout(
                frame(date(2025, 8, 20), 30),
                SETUP,
                allow_holdout=True,
                holdout_scored_marker=tmp_path / "absent.json",
            )

    def test_it_opens_once_the_holdout_has_actually_been_scored(self, tmp_path):
        marker = tmp_path / "holdout_scored.json"
        marker.write_text("{}")
        drift.assert_no_holdout(
            frame(date(2025, 8, 20), 30), SETUP, allow_holdout=True, holdout_scored_marker=marker
        )

    def test_a_pre_holdout_frame_always_passes(self):
        drift.assert_no_holdout(frame(date(2024, 9, 2), 60), SETUP)

    def test_the_report_refuses_a_current_slice_inside_the_holdout(self):
        with pytest.raises(ValueError, match="inside the declared holdout"):
            drift.drift_report(
                reference_features=frame(date(2024, 9, 2), 60),
                current_features=frame(date(2025, 9, 2), 60),
                setup=SETUP,
            )


# =======================================================================================
# PSI, K-S and the bands
# =======================================================================================
class TestFeatureDrift:
    def test_psi_is_near_zero_for_two_draws_of_the_same_distribution(self):
        rng = np.random.default_rng(0)
        psi, _ = drift.compute_psi(rng.normal(size=5_000), rng.normal(size=5_000))
        assert psi < 0.05

    def test_psi_grows_with_a_mean_shift(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(size=5_000)
        small, _ = drift.compute_psi(reference, rng.normal(0.2, 1.0, size=5_000))
        large, _ = drift.compute_psi(reference, rng.normal(1.5, 1.0, size=5_000))
        assert 0 < small < large
        assert large > 0.25

    def test_a_shift_to_new_extremes_is_amplified_not_lost(self):
        """Outer infinite edges: current values outside the reference range must count."""
        reference = np.linspace(-1, 1, 1_000)
        current = np.linspace(5, 7, 1_000)
        psi, _ = drift.compute_psi(reference, current)
        assert psi > 1.0

    def test_the_bands_are_the_declared_ones(self):
        config = load_monitor_config()
        assert drift.classify(0.30, 1.0, config["drift"]) == "ALERT"
        assert drift.classify(0.15, 1.0, config["drift"]) == "WATCH"
        assert drift.classify(0.03, 1e-6, config["drift"]) == "WATCH"  # small PSI, clear K-S
        assert drift.classify(0.01, 1.0, config["drift"]) == "OK"

    def test_a_shifted_feature_shows_up_as_alert_in_the_summary(self):
        reference = frame(date(2024, 9, 2), 120, seed=1)
        current = frame(date(2025, 3, 1), 60, shift=2.0, seed=2)
        metrics = drift.summarize_feature_drift(
            reference, current, ["usd_corr_21d", "gold_corr_21d"]
        )
        by_name = {m.name: m for m in metrics}
        assert by_name["usd_corr_21d"].status == "ALERT"
        assert by_name["gold_corr_21d"].status == "OK"
        assert metrics[0].name == "usd_corr_21d"  # sorted by PSI

    def test_prediction_drift_uses_the_same_bands(self):
        rng = np.random.default_rng(3)
        metric = drift.prediction_drift(rng.normal(size=2_000), rng.normal(3.0, 1.0, size=2_000))
        assert metric.status == "ALERT"
        assert metric.name == "prediction_score"


# =======================================================================================
# Rolling signal quality
# =======================================================================================
class TestRollingSignalQuality:
    def test_session_metrics_need_the_four_columns(self):
        with pytest.raises(ValueError, match="missing"):
            drift.session_metrics(pl.DataFrame({"timestamp": [date(2024, 1, 1)]}))

    def test_hit_rate_is_recovered_from_a_synthetic_stream(self):
        daily = drift.session_metrics(stream(date(2024, 9, 2), 200, hit=0.8, seed=4))
        assert daily.height == 200
        assert float(daily["hit_rate"].mean()) == pytest.approx(0.8, abs=0.05)

    def test_rolling_means_appear_only_once_there_is_enough_history(self):
        daily = drift.session_metrics(stream(date(2024, 9, 2), 80, hit=0.5, seed=5))
        rolled = drift.rolling_quality(daily, window=63)
        assert rolled["rolling_hit_rate_63"][0] is None
        assert rolled["rolling_hit_rate_63"][-1] is not None

    def test_a_falling_hit_rate_raises_an_alert(self):
        good = stream(date(2024, 9, 2), 63, hit=0.9, seed=6)
        bad = stream(date(2024, 12, 4), 63, hit=0.2, seed=7)
        daily = drift.session_metrics(pl.concat([good, bad]))
        alerts = {a["metric"]: a for a in drift.performance_alerts(daily)}
        assert alerts["rolling_hit_rate"]["status"] == "ALERT"

    def test_a_stable_stream_stays_ok(self):
        daily = drift.session_metrics(stream(date(2024, 9, 2), 126, hit=0.55, seed=8))
        alerts = {a["metric"]: a for a in drift.performance_alerts(daily)}
        assert alerts["rolling_hit_rate"]["status"] in ("OK", "WATCH")


class TestBuildingTheLiveStreamFromRunRecords:
    def test_scores_join_the_next_session_return(self):
        records = [
            {
                "steps": {
                    "5_predict": {
                        "latest_cross_section": [
                            {"timestamp": "2024-09-02", "symbol": "EURUSD", "score": 0.001},
                            {"timestamp": "2024-09-02", "symbol": "GBPUSD", "score": -0.001},
                        ]
                    }
                }
            }
        ]
        panel = pl.DataFrame(
            {
                "timestamp": [date(2024, 9, 2), date(2024, 9, 3)] * 2,
                "symbol": ["EURUSD", "EURUSD", "GBPUSD", "GBPUSD"],
                "close": [1.10, 1.11, 1.30, 1.29],
            }
        ).sort(["symbol", "timestamp"])
        built = drift.build_live_stream(records, panel)
        assert built.height == 2
        eur = built.filter(pl.col("symbol") == "EURUSD")
        assert float(eur["actual"][0]) == pytest.approx(1.11 / 1.10 - 1)

    def test_an_empty_record_list_yields_an_empty_typed_frame(self):
        built = drift.build_live_stream([], pl.DataFrame({"timestamp": [], "symbol": [], "close": []}))
        assert built.height == 0
        assert set(built.columns) == {"timestamp", "symbol", "score", "actual"}


# =======================================================================================
# Online detectors
# =======================================================================================
class TestOnlineDetectorsOnASessionStream:
    def test_windows_are_declared_in_decision_sessions(self):
        block = load_monitor_config()["online_detectors"]
        assert block["units"] == "decision_sessions"

    def test_readiness_says_how_long_before_an_alert_is_even_possible(self):
        readiness = online_detectors.detector_readiness(0)
        assert readiness["adwin_ready"] is False
        assert readiness["ddm_ready"] is False
        assert readiness["sessions_until_adwin_ready"] == readiness["adwin_first_possible_alert_at_session"]
        assert "cannot alert yet" in readiness["note"]

    def test_a_short_stream_reports_not_ready_rather_than_healthy(self):
        daily = drift.session_metrics(stream(date(2024, 9, 2), 40, hit=0.5, seed=9))
        result = online_detectors.run_detectors(daily)
        assert result["any_alert"] is False
        assert all(d["ready"] is False for d in result["detectors"])
        assert all(d["sessions_needed"] > 0 for d in result["detectors"])

    def _adwin(self):
        block = load_monitor_config()["online_detectors"]["adwin"]
        return online_detectors.ADWINStyle(
            window_size=int(block["window_size"]),
            sensitivity=float(block["sensitivity"]),
            cooldown_sessions=int(block["cooldown_sessions"]),
        )

    def test_adwin_trips_on_a_sustained_jump_in_error_size(self):
        trips = 0
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            detector = self._adwin()
            for _ in range(60):
                detector.update(abs(rng.normal(1.0, 0.05)))
            detector.cooldown_remaining = 0
            trips += any(detector.update(abs(rng.normal(5.0, 0.05))) for _ in range(42))
        assert trips == 20, "a five-fold jump in error size must trip the detector every time"

    def test_the_declared_false_alarm_rate_on_a_stationary_stream_is_what_the_yaml_says(self):
        """sensitivity 1.4 is loose on purpose; measure it rather than assume silence.

        The YAML says an ADWIN alert is a prompt to look, not a reason to stop, because on a
        stationary stream the two-window statistic clears 1.4 on roughly one comparison in
        six. This test pins that: most stationary runs raise at least one alert, and nothing
        in the control plane acts on one.
        """
        alarms = 0
        for seed in range(30):
            rng = np.random.default_rng(200 + seed)
            detector = self._adwin()
            alarms += any(detector.update(abs(rng.normal(1.0, 0.05))) for _ in range(120))
        assert 0 < alarms <= 30
        # No breaker reads a detector alert: the acting rules are in deploy/risk_config.yaml.
        from bots.exness_fx_d1.monitor.circuit_breakers import build_manager

        assert not [b for b in build_manager().manager.breakers if "adwin" in b or "drift" in b]

    def test_the_cooldown_turns_one_regime_change_into_one_alert(self):
        detector = online_detectors.ADWINStyle(window_size=5, sensitivity=1.0, cooldown_sessions=21)
        for _ in range(20):
            detector.update(1.0 + 0.01 * np.random.default_rng(11).normal())
        alerts = sum(detector.update(10.0 + 0.01 * i) for i in range(30))
        assert alerts <= 2

    def test_ddm_reports_drift_when_bad_sessions_cluster(self):
        block = load_monitor_config()["online_detectors"]["ddm"]
        detector = online_detectors.DDM(
            min_samples=int(block["min_samples"]),
            warning_level=float(block["warning_level"]),
            drift_level=float(block["drift_level"]),
        )
        for _ in range(100):
            detector.update(False)
        statuses = [detector.update(True) for _ in range(60)]
        assert "drift" in statuses

    def test_error_stream_turns_hit_rate_into_a_bad_session_flag(self):
        daily = drift.session_metrics(stream(date(2024, 9, 2), 10, hit=0.2, seed=12))
        errors = online_detectors.error_stream(daily, error_rate_floor=0.52)
        assert set(errors.columns) == {"timestamp", "mse", "direction_error_rate", "bad_session"}
        assert bool(errors["bad_session"].any())

    def test_stress_lag_is_signed_and_none_without_stress_dates(self):
        assert online_detectors.nearest_stress_lag([date(2024, 1, 10)], []) is None
        assert online_detectors.nearest_stress_lag([date(2024, 1, 10)], [date(2024, 1, 20)]) == 10.0
        assert online_detectors.nearest_stress_lag([date(2024, 1, 20)], [date(2024, 1, 10)]) == -10.0


# =======================================================================================
# Rollout
# =======================================================================================
class TestTheRolloutGateIsFixedBeforeTheChallengerRuns:
    def test_the_stages_start_with_no_capital_and_end_at_full(self):
        stages = rollout.load_stages()
        assert stages[0].name == "shadow" and stages[0].capital_fraction == 0.0
        assert stages[-1].name == "full" and stages[-1].capital_fraction == 1.0
        assert [s.capital_fraction for s in stages] == sorted(s.capital_fraction for s in stages)

    def test_the_criteria_are_frozen_and_read_from_the_yaml(self):
        criteria = rollout.PromotionCriteria.from_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            criteria.min_sharpe_improvement = 0.0  # the gate cannot be moved after the fact
        assert criteria.require_holdout_scored is True
        assert criteria.min_holdout_dsr == pytest.approx(0.95)

    def test_promotion_is_not_computable_while_the_holdout_is_unscored(self):
        incumbent = rollout.shadow_stats("incumbent", pd.Series(np.zeros(200)))
        candidate = rollout.shadow_stats("candidate", pd.Series(np.full(200, 0.01)))
        decision = rollout.evaluate_promotion(
            current_stage="shadow",
            incumbent=incumbent,
            candidate=candidate,
            signal_corr=0.9,
            agreement=0.9,
            holdout_scored=False,
        )
        assert decision.decision == "NOT_COMPUTABLE"
        assert "has not been scored" in decision.reasons[0]

    def test_a_positive_but_small_improvement_is_still_rejected(self):
        rng = np.random.default_rng(13)
        base = pd.Series(rng.normal(0.0002, 0.005, 200))
        incumbent = rollout.shadow_stats("incumbent", base)
        candidate = rollout.shadow_stats("candidate", base + 0.00001)
        decision = rollout.evaluate_promotion(
            current_stage="shadow",
            incumbent=incumbent,
            candidate=candidate,
            signal_corr=0.9,
            agreement=0.9,
            holdout_scored=True,
            holdout_dsr=0.99,
        )
        assert decision.decision == "REJECT"
        assert "Sharpe improvement" in str(decision.reasons)

    def test_everything_passing_promotes_to_the_next_stage(self):
        rng = np.random.default_rng(14)
        incumbent = rollout.shadow_stats("incumbent", pd.Series(rng.normal(0.0, 0.005, 200)))
        candidate = rollout.shadow_stats("candidate", pd.Series(rng.normal(0.0015, 0.005, 200)))
        decision = rollout.evaluate_promotion(
            current_stage="shadow",
            incumbent=incumbent,
            candidate=candidate,
            signal_corr=0.9,
            agreement=0.9,
            holdout_scored=True,
            holdout_dsr=0.99,
        )
        assert decision.decision == "PROMOTE"
        assert decision.to_stage == "ab_capped"

    def test_a_holdout_dsr_below_the_declared_level_blocks_promotion(self):
        rng = np.random.default_rng(15)
        incumbent = rollout.shadow_stats("incumbent", pd.Series(rng.normal(0.0, 0.005, 200)))
        candidate = rollout.shadow_stats("candidate", pd.Series(rng.normal(0.0015, 0.005, 200)))
        decision = rollout.evaluate_promotion(
            current_stage="shadow",
            incumbent=incumbent,
            candidate=candidate,
            signal_corr=0.9,
            agreement=0.9,
            holdout_scored=True,
            holdout_dsr=0.10,
        )
        assert decision.decision == "REJECT"
        assert "Holdout DSR" in str(decision.reasons)

    def test_this_bot_is_not_even_in_shadow_and_says_why(self):
        status = rollout.rollout_status()
        assert status["state"] == "not_started"
        assert "no survivor" in status["reason"]

    def test_helpers_do_not_explode_on_empty_input(self):
        assert rollout.annualized_sharpe(pd.Series(dtype=float)) == 0.0
        assert rollout.max_drawdown(pd.Series(dtype=float)) == 0.0
        assert rollout.position_agreement(
            pd.DataFrame(columns=["timestamp", "symbol", "weight"]),
            pd.DataFrame(columns=["timestamp", "symbol", "weight"]),
        ) == 0.0
