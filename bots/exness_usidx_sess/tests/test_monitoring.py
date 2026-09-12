"""Drift, online detectors, rollout and retirement — ``26_mlops_governance/01``-``04``.

THE TEST THAT MATTERS MOST IN THIS FILE is ``test_no_threshold_is_calibrated_from_the_failed
_generation``. The mentor's ruling calls calibrating a monitoring threshold from generation
1's Sharpe distribution "the easiest mistake to make", and it would be fatal: it would fit a
control to an outcome already shown to be noise (0 survivors of 1,403 scored specs at
K = 1,780). Every band in ``monitor_config.yaml`` is therefore either a declared kill criterion
or the sampling noise of the statistic on a TWO-instrument book, and the arithmetic is
reproduced here.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from bots.exness_usidx_sess.monitor import drift, online_detectors, retire, rollout

BOT_DIR = Path(__file__).resolve().parents[1]
MONITOR_CONFIG = BOT_DIR / "monitor" / "monitor_config.yaml"
RISK_CONFIG = BOT_DIR / "deploy" / "risk_config.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(MONITOR_CONFIG.read_text())


@pytest.fixture(scope="module")
def setup() -> dict:
    from utils.paths import get_case_study_dir

    return yaml.safe_load(
        (Path(get_case_study_dir("exness_usidx_sess")) / "config" / "setup.yaml").read_text()
    )


def _stream(n: int, *, seed: int = 0, start: date = date(2025, 1, 1)) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        stamp = start + timedelta(days=i)
        for symbol in ("US500", "USTEC"):
            score = float(rng.normal(0, 0.003))
            rows.append(
                {
                    "timestamp": stamp,
                    "symbol": symbol,
                    "score": score,
                    "actual": float(rng.normal(0, 0.01)),
                }
            )
    return pl.DataFrame(rows)


# =======================================================================================
# The rule the mentor named
# =======================================================================================
def test_no_threshold_is_calibrated_from_the_failed_generation(config):
    """The bands are one and two standard errors of the statistic on a TWO-name book, and the
    arithmetic is reproduced from first principles here rather than trusted.

    hit rate: sd per session = sqrt(p(1-p)/2) = 0.3536 at p = 0.5; a 63-session mean has a
              standard error of 0.3536/sqrt(63) = 0.0446.
    IC:       a 63-session rank correlation has a standard error of about 1/sqrt(63) = 0.126;
              pooling two indices divides it by sqrt(2) -> 0.089.
    """
    perf = config["drift"]["performance"]
    window = int(perf["window_sessions"])
    hit_se = math.sqrt(0.25 / 2) / math.sqrt(window)
    ic_se = (1.0 / math.sqrt(window)) / math.sqrt(2)
    assert perf["hit_rate_watch_drop"] == pytest.approx(hit_se, abs=0.005)
    assert perf["hit_rate_alert_drop"] == pytest.approx(2 * hit_se, abs=0.01)
    assert perf["ic_watch_drop"] == pytest.approx(ic_se, abs=0.005)
    assert perf["ic_alert_drop"] == pytest.approx(2 * ic_se, abs=0.01)
    # And they are far above the notebook's cross-sectional bands, which would alert on chance.
    assert perf["hit_rate_watch_drop"] > 0.005
    assert perf["ic_watch_drop"] > 0.010


def test_the_monitored_features_come_from_the_register_not_from_an_ic_ranking(config, setup):
    """0 of 84 columns cleared Benjamini-Hochberg in either spec, so an |IC| ranking over them
    is a ranking of noise. The families are named from ``setup.yaml::features.families``."""
    declared = {str(f["name"]) for f in setup["features"]["families"]}
    watched = set(config["drift"]["monitored_families"])
    assert watched <= declared, watched - declared
    assert len(watched) >= 6


def test_a_family_that_leaves_the_register_breaks_the_monitor_loudly(setup):
    """If the register moves and the monitor does not, the dashboard would silently watch
    nothing. The resolver raises instead."""
    broken = {
        "drift": {
            "monitored_families": ["a family nobody declared"],
            "max_columns_per_family": 2,
        }
    }
    with pytest.raises(KeyError, match="register moved"):
        drift.monitored_columns(["ret_1d"], setup, config=broken)


def test_families_resolve_to_real_columns(setup, config):
    """The pattern in the register is translated to a regex; a second copy of the column list
    would drift the moment phase 3 registered a new family (as it did: 21 of 84 columns matched
    nothing in the private prefix table that used to exist)."""
    available = [
        "ret_1d", "ret_5d", "ret_21d", "mom_skip_recent", "zscore_21d", "vol_gk_21d",
        "overnight_ret_1d", "intraday_ret_1d", "gap_1d", "d1_vol_63d", "or_ret", "not_a_feature",
    ]
    resolved = drift.monitored_columns(available, setup, config=config)
    flat = [c for cols in resolved.values() for c in cols]
    assert "not_a_feature" not in flat
    assert flat, resolved
    cap = int(config["drift"]["max_columns_per_family"])
    assert all(len(cols) <= cap for cols in resolved.values())


# =======================================================================================
# The evidence boundary
# =======================================================================================
def test_assert_no_holdout_refuses_a_frame_that_reaches_into_the_holdout(setup):
    frame = pl.DataFrame({"timestamp": [date(2026, 3, 15)]})
    with pytest.raises(ValueError, match="inside the declared holdout"):
        drift.assert_no_holdout(frame, setup)


def test_a_flag_alone_cannot_open_the_evidence_boundary(setup, tmp_path):
    """``allow_holdout=True`` is honoured only when the marker FILE exists. An intention is
    not a rule."""
    frame = pl.DataFrame({"timestamp": [date(2026, 3, 15)]})
    with pytest.raises(ValueError):
        drift.assert_no_holdout(frame, setup, allow_holdout=True)
    marker = tmp_path / "holdout_scored.json"
    marker.write_text("{}")
    drift.assert_no_holdout(
        frame, setup, allow_holdout=True, holdout_scored_marker=marker
    )  # no raise


def test_the_reference_window_ends_before_the_holdout(config, setup):
    end = date.fromisoformat(str(config["drift"]["reference"]["end"]))
    start = date.fromisoformat(str(setup["evaluation"]["holdout_start"]))
    assert end < start


# =======================================================================================
# The IC is a time-series statistic, not a cross-sectional one
# =======================================================================================
def test_session_metrics_carries_no_cross_sectional_ic():
    """A rank correlation over TWO points is +1 or -1 whatever the numbers are. The registry's
    own ``ic_mean`` came back null on all 356 prediction sets for exactly this reason."""
    daily = drift.session_metrics(_stream(40))
    assert "ic" not in daily.columns
    assert {"hit_rate", "mse", "n_assets"} <= set(daily.columns)
    assert daily["n_assets"].max() == 2


def test_time_series_ic_is_per_index_and_pooled():
    stream = _stream(120, seed=7)
    result = drift.time_series_ic(stream, window=63)
    assert set(result["per_symbol"]) == {"US500", "USTEC"}
    assert result["n_sessions"] == 63
    assert -1.0 <= result["pooled"] <= 1.0
    assert "NOT a cross-sectional IC" in result["definition"]


def test_time_series_ic_recovers_a_planted_relationship():
    """A monitor that cannot see a signal it is shown is not a monitor."""
    rows = []
    for i in range(120):
        stamp = date(2025, 1, 1) + timedelta(days=i)
        for symbol in ("US500", "USTEC"):
            score = (i % 20) / 20.0 - 0.5
            rows.append(
                {"timestamp": stamp, "symbol": symbol, "score": score, "actual": score * 2}
            )
    result = drift.time_series_ic(pl.DataFrame(rows))
    assert result["pooled"] > 0.9


# =======================================================================================
# PSI and the report
# =======================================================================================
def test_psi_is_zero_on_the_same_sample_and_large_on_a_shifted_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5_000)
    b = rng.normal(0, 1, 5_000)
    shifted = rng.normal(3, 1, 5_000)
    assert drift.compute_psi(a, b)[0] < 0.05
    assert drift.compute_psi(a, shifted)[0] > 0.25


def test_classify_uses_the_declared_bands(config):
    thresholds = config["drift"]
    assert drift.classify(0.0, 0.9, thresholds) == "OK"
    assert drift.classify(float(thresholds["psi"]["watch"]), 0.9, thresholds) == "WATCH"
    assert drift.classify(float(thresholds["psi"]["alert"]), 0.9, thresholds) == "ALERT"
    # A small but statistically clear move is a WATCH, not silence.
    assert drift.classify(0.05, 0.0001, thresholds) == "WATCH"


@pytest.mark.parametrize("spec", ["intraday", "overnight"])
def test_drift_report_is_not_computable_today_and_says_why(setup, spec):
    report = drift.drift_report(spec=spec, setup=setup)
    assert report["spec"] == spec
    assert report["features"]["status"] == "not_computable"
    assert report["performance"]["status"] == "not_computable"
    assert "never been deployed" in report["features"]["reason"]
    assert report["evidence_boundary"]["phase5_survivors"] == 0
    assert "DATA-QUALITY signal" in report["evidence_boundary"]["interpretation"]


def test_drift_report_refuses_an_unknown_spec(setup):
    with pytest.raises(ValueError, match="spec must be one of"):
        drift.drift_report(spec="london", setup=setup)


def test_performance_alerts_abstain_on_a_short_stream(config):
    daily = drift.session_metrics(_stream(30))
    alerts = drift.performance_alerts(daily, config=config)
    assert alerts[0]["status"] == "NOT_COMPUTABLE"
    assert "cannot judge" in alerts[0]["reason"]


def test_performance_alerts_fire_on_a_real_collapse(config):
    """Baseline 100 % hit rate, current 0 %: a drop of 1.0 against an alert band of 0.09."""
    window = int(config["drift"]["performance"]["window_sessions"])
    rows = []
    for i in range(2 * window):
        stamp = date(2025, 1, 1) + timedelta(days=i)
        sign = 1.0 if i < window else -1.0
        for symbol in ("US500", "USTEC"):
            rows.append(
                {"timestamp": stamp, "symbol": symbol, "score": 0.01, "actual": 0.01 * sign}
            )
    daily = drift.session_metrics(pl.DataFrame(rows))
    alerts = {a["metric"]: a for a in drift.performance_alerts(daily, config=config)}
    assert alerts["hit_rate"]["status"] == "ALERT"
    assert alerts["hit_rate"]["drop"] == pytest.approx(1.0)


# =======================================================================================
# Online detectors
# =======================================================================================
def test_detector_readiness_says_cannot_alert_rather_than_no_drift(config):
    readiness = online_detectors.detector_readiness(0, config=config)
    assert readiness["adwin_ready"] is False
    assert readiness["ddm_ready"] is False
    assert "cannot alert yet" in readiness["note"]
    assert readiness["units"] == "decision_sessions_per_spec"
    block = config["online_detectors"]
    assert readiness["adwin_first_possible_alert_at_session"] == int(
        block["calibration_sessions"]
    ) + 2 * int(block["adwin"]["window_size"])


def test_run_detectors_reports_not_ready_with_no_stream(config):
    result = online_detectors.run_detectors(None, spec="intraday", config=config)
    assert result["status"] == "not_ready"
    assert "never been deployed" in result["reason"]
    assert result["readiness"]["sessions_seen"] == 0


def test_adwin_needs_two_full_windows_before_it_can_say_anything(config):
    block = config["online_detectors"]["adwin"]
    det = online_detectors.ADWINStyle(
        window_size=int(block["window_size"]),
        sensitivity=float(block["sensitivity"]),
        cooldown_sessions=int(block["cooldown_sessions"]),
    )
    assert det.sessions_needed == 2 * int(block["window_size"])
    rng = np.random.default_rng(1)
    for _ in range(2 * int(block["window_size"]) - 1):
        assert det.update(float(rng.normal())) is False
    assert det.sessions_needed == 1


def test_adwin_sees_a_regime_change(config):
    block = config["online_detectors"]["adwin"]
    det = online_detectors.ADWINStyle(
        window_size=int(block["window_size"]),
        sensitivity=float(block["sensitivity"]),
        cooldown_sessions=int(block["cooldown_sessions"]),
    )
    rng = np.random.default_rng(3)
    for _ in range(2 * int(block["window_size"])):
        det.update(float(rng.normal(0, 1)))
    fired = any(det.update(float(rng.normal(10, 1))) for _ in range(int(block["window_size"])))
    assert fired


def test_ddm_only_sees_both_indices_wrong(config):
    """A real limitation, declared rather than smoothed over: with two instruments the
    per-session error rate takes only 0.0, 0.5 and 1.0, so a floor of 0.52 means "both
    wrong"."""
    block = config["online_detectors"]["ddm"]
    floor = float(block["error_rate_floor"])
    assert 0.5 < floor < 1.0
    assert "0, 0.5 and 1" in block["error_rate_resolution_note"]
    daily = pl.DataFrame(
        {"timestamp": [date(2025, 1, 1), date(2025, 1, 2)], "mse": [1.0, 1.0],
         "hit_rate": [0.5, 0.0]}
    )
    errors = online_detectors.error_stream(daily, error_rate_floor=floor)
    assert errors["bad_session"].to_list() == [False, True]


def test_run_detectors_over_a_stationary_stream_stays_quiet(config):
    daily = drift.session_metrics(_stream(200, seed=11))
    result = online_detectors.run_detectors(daily, spec="overnight", config=config)
    assert result["status"] == "computed"
    assert result["acts_on_alert"] is False
    # 200 sessions against a 126-session calibration: the detectors are ready but the stream is
    # stationary, so at most a handful of ADWIN prompts and no DDM drift.
    assert result["ddm"]["alert_count"] == 0
    assert result["adwin"]["alert_count"] <= 3


# =======================================================================================
# Rollout
# =======================================================================================
def test_promotion_is_not_computable_because_there_is_no_challenger():
    decision = rollout.evaluate_promotion(spec="intraday")
    assert decision.promote is False
    assert decision.status == "NOT_COMPUTABLE"
    assert any("0 survivors" in r for r in decision.reasons)
    assert any("holdout has not been scored" in r for r in decision.reasons)


def test_the_gate_measures_sharpe_on_the_active_return(config):
    """The convention that caught phase 5's headline: the best raw trial had Sharpe +1.587 and
    an ACTIVE Sharpe of -0.020 against a 1/N book of +0.442. A raw-return gate would promote
    the overnight beta."""
    criteria = rollout.PromotionCriteria.from_config(config)
    assert criteria.sharpe_measured_on == "active_return"
    assert "1/N long book" in criteria.benchmark
    assert criteria.require_holdout_scored is True
    assert criteria.min_holdout_dsr == 0.95
    assert criteria.trial_count_floor == 1780


def test_the_gate_refuses_a_raw_comparison_with_no_benchmark(tmp_path):
    marker = tmp_path / "holdout_scored.json"
    marker.write_text("{}")
    rng = np.random.default_rng(5)
    decision = rollout.evaluate_promotion(
        spec="intraday",
        incumbent_returns=rng.normal(0, 0.01, 100),
        candidate_returns=rng.normal(0.001, 0.01, 100),
        holdout_dsr=0.99,
        trial_count=1780,
        holdout_scored_marker=marker,
    )
    assert decision.promote is False
    assert any("no benchmark series was supplied" in r for r in decision.reasons)


def test_the_gate_refuses_a_dsr_deflated_at_one_workspaces_trial_count(tmp_path):
    """Two registries hold 890 trials each; a DSR helper run inside one deflates at half the
    truth. The floor is the bot's cumulative K."""
    marker = tmp_path / "holdout_scored.json"
    marker.write_text("{}")
    decision = rollout.evaluate_promotion(
        spec="overnight", trial_count=890, holdout_dsr=0.99, holdout_scored_marker=marker
    )
    assert any("890" in r and "cumulative K" in r for r in decision.reasons)


def test_the_ab_stage_cannot_be_funded_at_this_accounts_balance(config):
    """10 % of the declared minimum viable allocation is 300 USD against a smallest placeable
    position of 1,080.32 (US500) and 1,482.95 (USTEC)."""
    stages = {s.name: s for s in rollout.load_stages(config)}
    report = rollout.stage_is_fundable(
        stages["ab_capped"],
        allocated=3000.0,
        min_notional={"US500": 1080.32, "USTEC": 1482.95},
    )
    assert report["fundable"] is False
    assert report["per_symbol"]["USTEC"]["allocation_needed"] == pytest.approx(14829.5, abs=1.0)


def test_the_shadow_stage_is_always_fundable_because_it_risks_nothing(config):
    stages = {s.name: s for s in rollout.load_stages(config)}
    assert stages["shadow"].capital_fraction == 0.0
    report = rollout.stage_is_fundable(
        stages["shadow"], allocated=900.0, min_notional={"US500": 1080.32}
    )
    assert report["fundable"] is True


def test_rollout_status_refuses_to_advance(config):
    status = rollout.rollout_status(
        allocated=900.0, min_notional={"US500": 1080.32, "USTEC": 1482.95}, config=config
    )
    assert status["current_stage"] == "shadow"
    assert status["next_stage"] == "ab_capped"
    assert status["evidence_boundary"]["may_advance"] is False


def test_annualisation_is_252_sessions_not_504_decisions():
    """The bot takes two decisions a session but each SPEC takes one, and a spec's return
    series is one observation per session. Annualising at 504 would inflate every Sharpe by
    sqrt(2)."""
    series = np.full(252, 0.001)
    series[::2] = -0.0005
    a = rollout.annualized_sharpe(series, periods_per_year=252)
    b = rollout.annualized_sharpe(series, periods_per_year=504)
    assert b / a == pytest.approx(math.sqrt(2), rel=1e-6)


# =======================================================================================
# Retirement
# =======================================================================================
def test_both_retire_rules_are_not_computable_and_neither_defaults_to_a_pass():
    report = retire.check_retire_rules(config_path=RISK_CONFIG)
    assert report.retire is False
    assert {c.status for c in report.checks} == {"not_computable"}
    assert all(c.unblocked_by for c in report.checks)
    assert "not 'pass'" in report.evidence["note"]


def test_the_breakeven_rule_names_the_real_account_and_it_has_never_been_read():
    report = retire.check_retire_rules(config_path=RISK_CONFIG)
    check = next(c for c in report.checks if c.name == "breakeven_below_real_p90_spread")
    assert "REAL Pro account has never been read" in check.reason
    assert "16_costs has not run" in check.reason
    assert report.evidence["real_account_measured"] is False


def test_the_holdout_rule_records_the_two_registry_risk():
    report = retire.check_retire_rules(config_path=RISK_CONFIG)
    check = next(c for c in report.checks if c.name == "holdout_psr_below_0_5")
    assert "TWO registries" in check.unblocked_by
    assert "scored once" in check.unblocked_by


def test_a_supplied_measurement_can_still_retire_the_bot(tmp_path):
    """The rules are not decorative: given both inputs they decide."""
    marker = tmp_path / "holdout_scored.json"
    marker.write_text("{}")
    block = yaml.safe_load(RISK_CONFIG.read_text())
    block["retire"]["real_account_measured"] = True
    path = tmp_path / "risk.yaml"
    path.write_text(yaml.safe_dump(block))
    report = retire.check_retire_rules(
        breakeven_cost_bps=0.5,
        real_account_p90_spread_bps=1.05,
        holdout_psr_vs_benchmark=0.2,
        config_path=path,
        holdout_scored_marker=marker,
    )
    assert report.retire is True
    assert {c.status for c in report.checks} == {"RETIRE"}
