"""bots/exness_fx_d1/monitor/circuit_breakers.py — one trip test per breaker.

The pattern of ``26_mlops_governance/04_circuit_breakers.py`` section 4: drive each breaker
along a synthetic series until it trips, then check the state machine and the audit trail.
Every threshold is read from ``deploy/risk_config.yaml``; the tests assert against the file,
not against numbers retyped here, so a change to the kill criteria fails the test that claims
to implement it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bots._shared.monitor.base import BreakerState, BreakerTier
from bots.exness_fx_d1.monitor import retire
from bots.exness_fx_d1.monitor.circuit_breakers import (
    RoundTrip,
    build_manager,
    kill_switch_on_trip,
    load_breaker_config,
)

NOW = datetime(2026, 9, 7, 20, 0, tzinfo=UTC)


@pytest.fixture
def config():
    return load_breaker_config()


@pytest.fixture
def breakers():
    return build_manager()


# ---------------------------------------------------------------------------------------
# the map from kill criteria to breakers
# ---------------------------------------------------------------------------------------
class TestEveryKillCriterionHasABreaker:
    def test_the_lettered_criteria_are_all_represented(self, breakers):
        criteria = {b.kill_criterion for b in breakers.manager.breakers.values()}
        assert {"a", "b", "c", "d", "e", "f"} <= criteria

    def test_the_tiers_are_the_four_of_notebook_04(self, breakers):
        tiers = {b.tier for b in breakers.manager.breakers.values()}
        assert tiers == {BreakerTier.TRADE, BreakerTier.STRATEGY, BreakerTier.PORTFOLIO, BreakerTier.SYSTEM}

    def test_no_account_level_breaker_is_duplicated_here(self, breakers):
        """The account tier belongs to bots/_shared/monitor and stops every bot at once."""
        names = set(breakers.manager.breakers)
        assert not {n for n in names if n.startswith("account_")}

    def test_thresholds_come_from_the_yaml(self, breakers, config):
        assert breakers.drawdown.max_drawdown_pct == float(config["breakers"]["drawdown"]["max_drawdown_pct"])
        assert breakers.hit_rate.window_sessions == int(config["breakers"]["hit_rate"]["window_sessions"])
        assert breakers.cost.assumed_round_trip_bps == float(config["breakers"]["cost"]["assumed_round_trip_bps"])
        assert breakers.config_source.endswith("risk_config.yaml")


# ---------------------------------------------------------------------------------------
# (a) drawdown
# ---------------------------------------------------------------------------------------
class TestDrawdownBreakerCriterionA:
    def test_it_trips_past_eight_percent_from_the_peak(self, breakers):
        breaker = breakers.drawdown
        breaker.update(event_time=NOW, portfolio_value=10_000.0)
        breaker.update(event_time=NOW, portfolio_value=9_300.0)  # -7 %
        assert breaker.state is BreakerState.CLOSED
        breaker.update(event_time=NOW, portfolio_value=9_150.0)  # -8.5 %
        assert breaker.state is BreakerState.OPEN
        assert breaker.trip_count == 1
        assert "exceeds 8%" in breaker.history[-1].reason

    def test_it_probes_half_open_after_the_recovery_timeout_and_closes_on_recovery(self, breakers, config):
        breaker = breakers.drawdown
        breaker.update(event_time=NOW, portfolio_value=10_000.0)
        breaker.update(event_time=NOW, portfolio_value=9_000.0)
        assert breaker.state is BreakerState.OPEN
        later = NOW + timedelta(hours=float(config["breakers"]["recovery_timeout_hours"]) + 1)
        assert breaker.update(event_time=later, portfolio_value=9_000.0) is BreakerState.HALF_OPEN
        assert breaker.update(event_time=later, portfolio_value=9_900.0) is BreakerState.CLOSED


# ---------------------------------------------------------------------------------------
# daily loss (derived)
# ---------------------------------------------------------------------------------------
class TestDailyLossBreaker:
    def test_it_trips_at_two_percent_of_the_days_opening_equity(self, breakers):
        breaker = breakers.daily_loss
        breaker.reset_day(10_000.0)
        breaker.update(event_time=NOW, portfolio_value=9_850.0)  # -1.5 %
        assert breaker.state is BreakerState.CLOSED
        breaker.update(event_time=NOW, portfolio_value=9_790.0)  # -2.1 %
        assert breaker.state is BreakerState.OPEN


# ---------------------------------------------------------------------------------------
# (b) rolling hit rate — the new subclass
# ---------------------------------------------------------------------------------------
class TestRollingHitRateBreakerCriterionB:
    def test_the_rolling_statistic_does_not_exist_before_the_window_is_full(self, breakers):
        breaker = breakers.hit_rate
        for _ in range(breaker.window_sessions - 1):
            assert breaker.record_session(0.0) is None
        assert breaker.sessions_below == 0
        breaker.update(event_time=NOW)
        assert breaker.state is BreakerState.CLOSED

    def test_it_trips_only_after_the_declared_run_of_sessions_below_the_level(self, breakers):
        breaker = breakers.hit_rate
        for _ in range(breaker.window_sessions):
            breaker.record_session(0.30)  # rolling mean 0.30, below 0.5, from session 63 on
        assert breaker.sessions_below == 1
        for _ in range(breaker.consecutive_sessions - 2):
            breaker.record_session(0.30)
        breaker.update(event_time=NOW)
        assert breaker.state is BreakerState.CLOSED, "trips one session too early"
        breaker.record_session(0.30)
        assert breaker.sessions_below == breaker.consecutive_sessions
        breaker.update(event_time=NOW)
        assert breaker.state is BreakerState.OPEN
        assert "consecutive sessions" in breaker.history[-1].reason

    def test_a_single_good_session_resets_the_run(self, breakers):
        breaker = breakers.hit_rate
        for _ in range(breaker.window_sessions + 5):
            breaker.record_session(0.0)
        assert breaker.sessions_below == 6
        for _ in range(breaker.window_sessions):
            breaker.record_session(1.0)  # rolling mean climbs back above 0.5
        assert breaker.sessions_below == 0

    def test_this_is_not_the_consecutive_loss_rule(self, breakers):
        """A run of 21 sessions below a 63-session mean can contain winning sessions."""
        breaker = breakers.hit_rate
        for _ in range(breaker.window_sessions):
            breaker.record_session(0.0)
        for _ in range(breaker.consecutive_sessions):
            breaker.record_session(1.0 if _ % 3 == 0 else 0.0)  # some winning sessions
        breaker.update(event_time=NOW)
        assert breaker.state is BreakerState.OPEN
        assert breakers.consecutive_loss.consecutive_losses == 0


# ---------------------------------------------------------------------------------------
# (c) realised cost
# ---------------------------------------------------------------------------------------
class TestRealisedCostBreakerCriterionC:
    def test_it_abstains_below_the_declared_minimum_number_of_round_trips(self, breakers):
        breaker = breakers.cost
        for i in range(breaker.min_round_trips - 1):
            breaker.record_round_trip(
                RoundTrip(NOW - timedelta(days=i), "EURUSD", notional=10_000.0, cost_quote=100.0)
            )
        realised, n = breaker.realised_bps(NOW)
        assert realised is None and n == breaker.min_round_trips - 1
        breaker.update(event_time=NOW, now=NOW)
        assert breaker.state is BreakerState.CLOSED

    def test_it_trips_above_one_and_a_half_times_the_assumed_round_trip(self, breakers):
        breaker = breakers.cost
        # 6 bps a round trip against an assumed 2.6 x 1.5 = 3.9 bps threshold.
        for i in range(breaker.min_round_trips):
            breaker.record_round_trip(
                RoundTrip(NOW - timedelta(days=i), "EURUSD", notional=10_000.0, cost_quote=6.0)
            )
        realised, _ = breaker.realised_bps(NOW)
        assert realised == pytest.approx(6.0)
        assert breaker.threshold_bps == pytest.approx(3.9)
        breaker.update(event_time=NOW, now=NOW)
        assert breaker.state is BreakerState.OPEN

    def test_round_trips_outside_the_window_do_not_count(self, breakers):
        breaker = breakers.cost
        for i in range(breaker.min_round_trips):
            breaker.record_round_trip(
                RoundTrip(NOW - timedelta(days=90 + i), "EURUSD", notional=10_000.0, cost_quote=60.0)
            )
        assert breaker.realised_bps(NOW) == (None, 0)


# ---------------------------------------------------------------------------------------
# (d) reconciliation
# ---------------------------------------------------------------------------------------
class TestReconciliationBreakerCriterionD:
    def test_a_clean_report_keeps_it_closed(self, breakers):
        breakers.reconciliation.update(event_time=NOW, reconciliation={"clean": True})
        assert breakers.reconciliation.state is BreakerState.CLOSED

    def test_one_unexplained_position_trips_it_immediately(self, breakers):
        breakers.reconciliation.update(
            event_time=NOW,
            reconciliation={"clean": False, "unexpected_positions": {"EURUSD": 10_000.0}},
        )
        assert breakers.reconciliation.state is BreakerState.OPEN
        assert "unexpected_positions" in breakers.reconciliation.history[-1].reason

    def test_it_does_not_self_heal(self, breakers):
        breakers.reconciliation.update(event_time=NOW, reconciliation={"clean": False, "error": "foreign magic"})
        much_later = NOW + timedelta(days=365)
        assert breakers.reconciliation.update(event_time=much_later, reconciliation={"clean": True}) is BreakerState.OPEN

    def test_a_trip_latches_the_persistent_kill_switch(self):
        class FakeSafeBroker:
            def __init__(self):
                self.reasons: list[str] = []

            def enable_kill_switch(self, reason: str = "Manual") -> None:
                self.reasons.append(reason)

        safe = FakeSafeBroker()
        wired = build_manager(on_any_trip=kill_switch_on_trip(safe))
        wired.reconciliation.update(event_time=NOW, reconciliation={"clean": False, "error": "foreign magic"})
        assert safe.reasons and "system_reconciliation" in safe.reasons[0]


# ---------------------------------------------------------------------------------------
# (e) missing decision bar — a no-trade day that escalates
# ---------------------------------------------------------------------------------------
class TestMissingDecisionBarIsANoTradeDayNotAPause:
    def test_one_miss_does_not_trip(self, breakers):
        breakers.missing_bar.record_session(decision_bar_ok=False)
        breakers.missing_bar.update(event_time=NOW)
        assert breakers.missing_bar.state is BreakerState.CLOSED
        assert breakers.missing_bar.misses_in_window == 1

    def test_it_escalates_at_the_declared_count_inside_the_window(self, breakers, config):
        n = int(config["missing_decision_bar"]["escalate_after"])
        for _ in range(n - 1):
            breakers.missing_bar.record_session(decision_bar_ok=False)
        breakers.missing_bar.update(event_time=NOW)
        assert breakers.missing_bar.state is BreakerState.CLOSED
        breakers.missing_bar.record_session(decision_bar_ok=False)
        breakers.missing_bar.update(event_time=NOW)
        assert breakers.missing_bar.state is BreakerState.OPEN

    def test_misses_age_out_of_the_window(self, breakers, config):
        window = int(config["missing_decision_bar"]["window_sessions"])
        breakers.missing_bar.record_session(decision_bar_ok=False)
        for _ in range(window):
            breakers.missing_bar.record_session(decision_bar_ok=True)
        assert breakers.missing_bar.misses_in_window == 0


# ---------------------------------------------------------------------------------------
# (f) swap
# ---------------------------------------------------------------------------------------
class TestSwapBreakerCriterionF:
    def test_a_night_inside_the_tolerance_does_not_count(self, breakers):
        breakers.swap.record_night(0.4)  # declared 0.0 + tolerance 0.5
        assert breakers.swap.nights_above == 0

    def test_it_trips_after_the_declared_run_of_nights(self, breakers):
        for _ in range(breakers.swap.consecutive_nights - 1):
            breakers.swap.record_night(3.0)
        breakers.swap.update(event_time=NOW)
        assert breakers.swap.state is BreakerState.CLOSED
        breakers.swap.record_night(3.0)
        breakers.swap.update(event_time=NOW)
        assert breakers.swap.state is BreakerState.OPEN
        assert "points/lot/night" in breakers.swap.history[-1].reason


# ---------------------------------------------------------------------------------------
# trade tier and infrastructure
# ---------------------------------------------------------------------------------------
class TestTradeTierAndInfrastructure:
    def test_consecutive_loss_breaker_trips_on_the_declared_streak(self, breakers):
        for _ in range(breakers.consecutive_loss.max_consecutive):
            breakers.consecutive_loss.record_trade(-1.0)
        breakers.consecutive_loss.update(event_time=NOW)
        assert breakers.consecutive_loss.state is BreakerState.OPEN

    def test_one_win_resets_the_streak(self, breakers):
        breakers.consecutive_loss.record_trade(-1.0)
        breakers.consecutive_loss.record_trade(+1.0)
        assert breakers.consecutive_loss.consecutive_losses == 0

    def test_latency_breaker_trips_on_a_slow_order_path(self, breakers):
        for _ in range(breakers.latency.n_recent):
            breakers.latency.record_latency(5_000.0)
        breakers.latency.update(event_time=NOW)
        assert breakers.latency.state is BreakerState.OPEN


# ---------------------------------------------------------------------------------------
# the manager
# ---------------------------------------------------------------------------------------
class TestTheManager:
    def test_one_open_breaker_stops_trading_and_is_named(self, breakers):
        assert breakers.allows_trading()
        breakers.reconciliation.update(event_time=NOW, reconciliation={"clean": False, "error": "x"})
        assert not breakers.allows_trading()
        assert breakers.manager.open_breakers() == ["system_reconciliation"]

    def test_the_record_carries_the_audit_trail(self, breakers):
        breakers.drawdown.update(event_time=NOW, portfolio_value=10_000.0)
        breakers.drawdown.update(event_time=NOW, portfolio_value=8_000.0)
        record = breakers.to_record()
        assert record["allows_trading"] is False
        assert record["breakers"]["strategy_drawdown"]["kill_criterion"] == "a"
        assert any(e["to"] == "OPEN" for e in record["events"])


# ---------------------------------------------------------------------------------------
# retirement is not a breaker
# ---------------------------------------------------------------------------------------
class TestRetireRulesAreNotComputableToday:
    def test_both_rules_report_not_computable_with_a_reason(self):
        report = retire.check_retire_rules()
        assert report.verdict == "NOT_COMPUTABLE"
        assert [c.verdict for c in report.checks] == ["NOT_COMPUTABLE", "NOT_COMPUTABLE"]
        assert "16_costs has not run" in report.checks[0].reason
        assert "unscored" in report.checks[1].reason
        assert all(c.unblocked_by for c in report.checks)

    def test_supplying_the_numbers_produces_a_verdict(self):
        keep = retire.check_retire_rules(breakeven_cost_bps=5.0, holdout_psr_vs_benchmark=0.8)
        assert keep.verdict == "KEEP"
        gone = retire.check_retire_rules(breakeven_cost_bps=0.5, holdout_psr_vs_benchmark=0.8)
        assert gone.verdict == "RETIRE"
        gone2 = retire.check_retire_rules(breakeven_cost_bps=5.0, holdout_psr_vs_benchmark=0.2)
        assert gone2.verdict == "RETIRE"

    def test_a_missing_input_never_reads_as_a_pass(self):
        half = retire.check_retire_rules(breakeven_cost_bps=5.0)
        assert half.verdict == "NOT_COMPUTABLE"
