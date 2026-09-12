"""One trip test per breaker, plus the three things that make this bot's set different.

The three: kill criterion (b) is built PER SPEC, criterion (e) is ASYMMETRIC between the two
specs, and criterion (g) exists on no other bot. Each of those has its own test below, because
each is a rule someone could "simplify" away and only a test would notice.

Nothing here calibrates a threshold. Every number the tests use is read from
``deploy/risk_config.yaml``; the tests assert the BEHAVIOUR at whatever the config declares.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from bots._shared.monitor.base import BreakerState, BreakerTier
from bots.exness_usidx_sess.monitor import circuit_breakers as cb

RISK_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "risk_config.yaml"


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(RISK_CONFIG.read_text())


@pytest.fixture
def breakers() -> cb.BotBreakers:
    return cb.build_manager(RISK_CONFIG, allow_draft_criteria=True)


# ---------------------------------------------------------------------------------------
# The draft gate
# ---------------------------------------------------------------------------------------
def test_build_manager_refuses_while_the_kill_criteria_are_a_draft():
    """The criteria are still awaiting the user, and a monitoring layer built on a draft is a
    rehearsal. The refusal makes that explicit rather than implicit."""
    with pytest.raises(cb.DraftCriteriaError, match="DRAFT"):
        cb.build_manager(RISK_CONFIG)


def test_every_lettered_kill_criterion_has_at_least_one_breaker(breakers):
    covered = breakers.kill_criteria_covered()
    assert set(covered) == {"a", "b", "c", "d", "e", "f", "g"}, covered
    # (b) and (e) are covered by TWO breakers each, for the reasons the criteria give.
    assert len(covered["b"]) == 2, covered["b"]
    assert len(covered["e"]) == 2, covered["e"]


def test_the_bot_writes_no_account_breaker(breakers):
    """The account tier is ``bots/_shared/monitor`` and stops every bot at once. A bot that
    re-implemented one would have two different answers to the same question."""
    assert not [n for n in breakers.manager.breakers if n.startswith("account_")]


def test_the_evidence_boundary_is_in_the_record(breakers):
    record = breakers.to_record()
    assert record["pending_user_approval"] is True
    assert record["evidence_boundary"]["phase5_survivors"] == 0
    assert record["evidence_boundary"]["trial_count_K"] == 1780
    assert record["evidence_boundary"]["live_trading_permitted"] is False


# ---------------------------------------------------------------------------------------
# (a) and the derived daily rule
# ---------------------------------------------------------------------------------------
def test_drawdown_breaker_trips_at_the_declared_fraction(breakers, raw):
    limit = float(raw["breakers"]["drawdown"]["max_drawdown_pct"])
    b = breakers.drawdown
    assert b.tier is BreakerTier.PORTFOLIO
    assert b.update(portfolio_value=1000.0) is BreakerState.CLOSED
    assert b.update(portfolio_value=1000.0 * (1 - limit / 2)) is BreakerState.CLOSED
    assert b.update(portfolio_value=1000.0 * (1 - limit)) is BreakerState.OPEN
    assert b.history[-1].kill_criterion == "a"


def test_daily_loss_breaker_works_in_percent_so_it_does_not_go_stale(breakers, raw):
    limit = float(raw["breakers"]["daily_loss"]["max_daily_loss_pct"])
    b = breakers.daily_loss
    assert b.update(day_start_value=900.0, portfolio_value=900.0) is BreakerState.CLOSED
    assert b.update(day_start_value=900.0, portfolio_value=900.0 * (1 - limit)) is (
        BreakerState.OPEN
    )


# ---------------------------------------------------------------------------------------
# (b) — per spec, and the masking it prevents
# ---------------------------------------------------------------------------------------
def test_hit_rate_breaker_is_built_once_per_spec(breakers):
    assert set(breakers.hit_rate) == {"intraday", "overnight"}
    assert "strategy_hit_rate_intraday" in breakers.manager.breakers
    assert "strategy_hit_rate_overnight" in breakers.manager.breakers


def test_hit_rate_breaker_abstains_until_the_window_is_full(breakers, raw):
    b = breakers.hit_rate["intraday"]
    window = int(raw["breakers"]["hit_rate"]["window_sessions"])
    for _ in range(window - 1):
        b.record_session(False)
    assert b.hit_rate() is None
    assert b.update() is BreakerState.CLOSED  # silence means "cannot judge", not "healthy"


def test_hit_rate_breaker_trips_only_after_the_declared_run(breakers, raw):
    block = raw["breakers"]["hit_rate"]
    window, need = int(block["window_sessions"]), int(block["consecutive_sessions"])
    b = breakers.hit_rate["overnight"]
    for _ in range(window):
        b.record_session(False)
    for i in range(need - 1):
        assert b.update() is BreakerState.CLOSED, i
    assert b.update() is BreakerState.OPEN
    assert "overnight" in b.history[-1].reason


def test_one_dead_spec_cannot_be_masked_by_a_healthy_one(breakers, raw):
    """The reason criterion (b) says "counted separately per spec". Feeding a pooled series
    would let a 100 %-hit intraday leg carry a 0 %-hit overnight leg to a 50 % pooled rate,
    which never trips."""
    block = raw["breakers"]["hit_rate"]
    window, need = int(block["window_sessions"]), int(block["consecutive_sessions"])
    good, bad = breakers.hit_rate["intraday"], breakers.hit_rate["overnight"]
    for _ in range(window):
        good.record_session(True)
        bad.record_session(False)
    for _ in range(need):
        good.update()
        bad.update()
    assert good.state is BreakerState.CLOSED
    assert bad.state is BreakerState.OPEN
    assert breakers.allows_trading() is False


# ---------------------------------------------------------------------------------------
# (c) — two assumed costs, the tighter binds
# ---------------------------------------------------------------------------------------
def test_realised_cost_breaker_uses_the_tighter_of_the_two_assumptions(breakers, raw):
    block = raw["breakers"]["cost"]
    b = breakers.cost
    assert b.assumed_round_trip_bps == min(
        float(block["declared_round_trip_bps"]), float(block["live_regime_round_trip_bps"])
    )
    assert b.limit_bps == pytest.approx(b.assumed_round_trip_bps * float(block["multiple"]))
    # And it is genuinely tighter than the backtest's own charge, which is the point.
    assert b.limit_bps < float(block["declared_round_trip_bps"])


def test_realised_cost_breaker_abstains_below_the_minimum_number_of_round_trips(breakers, raw):
    b = breakers.cost
    now = datetime(2026, 9, 8, tzinfo=UTC)
    for _ in range(int(raw["breakers"]["cost"]["min_round_trips"]) - 1):
        b.record_round_trip(cb.RoundTrip(now, "US500", 1000.0, 99.0))
    assert b.update(now=now) is BreakerState.CLOSED


def test_realised_cost_breaker_trips_on_a_notional_weighted_average(breakers, raw):
    b = breakers.cost
    now = datetime(2026, 9, 8, tzinfo=UTC)
    for _ in range(int(raw["breakers"]["cost"]["min_round_trips"])):
        b.record_round_trip(cb.RoundTrip(now, "US500", 1000.0, b.limit_bps * 2))
    assert b.update(now=now) is BreakerState.OPEN


def test_realised_cost_breaker_ignores_trips_outside_the_window(breakers, raw):
    b = breakers.cost
    now = datetime(2026, 9, 8, tzinfo=UTC)
    stale = now - timedelta(days=int(raw["breakers"]["cost"]["window_days"]) + 1)
    for _ in range(int(raw["breakers"]["cost"]["min_round_trips"]) * 2):
        b.record_round_trip(cb.RoundTrip(stale, "US500", 1000.0, 999.0))
    assert b.update(now=now) is BreakerState.CLOSED


# ---------------------------------------------------------------------------------------
# (d) — trips at once, never probes
# ---------------------------------------------------------------------------------------
def test_reconciliation_breaker_trips_immediately_and_never_probes(breakers):
    b = breakers.reconciliation
    assert b.tier is BreakerTier.SYSTEM
    assert b.update(reconciliation_ok=True) is BreakerState.CLOSED
    assert b.update(reconciliation_ok=False, reconciliation_detail="ghost leg") is (
        BreakerState.OPEN
    )
    # Ten years later it is still OPEN: "do I know what I own" is not probed by trading again.
    assert b.update(event_time=datetime.now(UTC) + timedelta(days=365)) is BreakerState.OPEN
    b.reset()
    assert b.state is BreakerState.CLOSED


# ---------------------------------------------------------------------------------------
# (e) — THE ASYMMETRY. This is the bot's biggest difference from exness_fx_d1.
# ---------------------------------------------------------------------------------------
def test_a_missing_bar_breaker_cannot_be_built_for_the_overnight_spec():
    """504 of 1,930 research sessions had no overnight fill bar. Counting them as incidents
    would page a human for market structure ``setup.yaml`` already declares."""
    with pytest.raises(ValueError, match="NORMAL state"):
        cb.MissingDecisionBarBreaker(
            "x", spec="overnight", escalate_after=3, window_sessions=21
        )


def test_intraday_missing_bar_escalates_after_the_declared_run(breakers, raw):
    block = raw["missing_decision_bar"]["intraday"]
    b = breakers.missing_bar_intraday
    for _ in range(int(block["escalate_after"]) - 1):
        b.record_decision(had_bar=False)
    assert b.update() is BreakerState.CLOSED
    b.record_decision(had_bar=False)
    assert b.update() is BreakerState.OPEN
    assert b.history[-1].kill_criterion == "e"


def test_overnight_coverage_tolerates_the_structural_miss_rate(breakers, raw):
    """26 % missing is the research sample's own rate. The breaker must not trip on it."""
    block = raw["missing_decision_bar"]["overnight"]
    window = int(block["coverage_window_sessions"])
    b = breakers.overnight_coverage
    for i in range(window):
        b.record_decision(had_bar=(i % 4 != 0))  # 25 % missing
    assert b.coverage() == pytest.approx(0.75, abs=0.02)
    assert b.update() is BreakerState.CLOSED


def test_overnight_coverage_trips_below_the_pre_declared_floor(breakers, raw):
    """The floor is ``setup.yaml::decision.min_fill_coverage`` (0.60), declared before phase 4
    and therefore not chosen from an outcome."""
    block = raw["missing_decision_bar"]["overnight"]
    window = int(block["coverage_window_sessions"])
    floor = float(block["min_fill_coverage"])
    b = breakers.overnight_coverage
    for i in range(window):
        b.record_decision(had_bar=(i % 2 == 0))  # 50 %, below the floor
    assert b.coverage() < floor
    assert b.update() is BreakerState.OPEN


def test_overnight_coverage_floor_matches_the_research_declaration():
    """If ``setup.yaml`` ever moves ``min_fill_coverage``, this test fails rather than letting
    the live floor and the research floor drift apart."""
    from utils.paths import get_case_study_dir

    setup = yaml.safe_load(
        (Path(get_case_study_dir("exness_usidx_sess")) / "config" / "setup.yaml").read_text()
    )
    raw = yaml.safe_load(RISK_CONFIG.read_text())
    assert float(raw["missing_decision_bar"]["overnight"]["min_fill_coverage"]) == float(
        setup["decision"]["min_fill_coverage"]
    )


# ---------------------------------------------------------------------------------------
# (f) — per symbol
# ---------------------------------------------------------------------------------------
def test_swap_breaker_is_declared_per_symbol(breakers, raw):
    declared = raw["breakers"]["swap"]["declared_points_per_lot_per_night"]
    assert set(breakers.swap.declared) == {"US500", "USTEC"}
    assert breakers.swap.declared["US500"]["long"] == float(declared["US500"]["long"])
    assert breakers.swap.declared["USTEC"]["short"] == 0.0
    assert breakers.swap.rollover3days == "friday"


def test_swap_breaker_tolerates_the_declared_charge_and_the_friday_triple(breakers, raw):
    b = breakers.swap
    nights = int(raw["breakers"]["swap"]["consecutive_nights"])
    for i in range(nights):
        triple = i == 4  # Friday
        b.record_night("US500", points=-147.4 * (3 if triple else 1), side="long", triple=triple)
    assert b.update() is BreakerState.CLOSED


def test_swap_breaker_trips_when_the_broker_changes_its_terms(breakers, raw):
    b = breakers.swap
    nights = int(raw["breakers"]["swap"]["consecutive_nights"])
    tolerance = float(raw["breakers"]["swap"]["tolerance_points"])
    for _ in range(nights):
        b.record_night("USTEC", points=-592.7 - tolerance * 3, side="long")
    assert b.update() is BreakerState.OPEN
    assert "USTEC" in b.history[-1].reason


def test_any_swap_charged_on_a_short_leg_is_a_breach(breakers, raw):
    """Both indices declare ``short: 0.0``. A charge there is a change of terms, not noise."""
    b = breakers.swap
    nights = int(raw["breakers"]["swap"]["consecutive_nights"])
    tolerance = float(raw["breakers"]["swap"]["tolerance_points"])
    for _ in range(nights):
        b.record_night("US500", points=-(tolerance + 10.0), side="short")
    assert b.update() is BreakerState.OPEN


def test_swap_breaker_refuses_an_undeclared_symbol(breakers):
    with pytest.raises(KeyError):
        breakers.swap.record_night("XAUUSD", points=-1.0, side="long")


# ---------------------------------------------------------------------------------------
# (g) — no other bot has this one
# ---------------------------------------------------------------------------------------
def test_trade_hours_dst_breaker_accepts_both_dst_regimes(breakers):
    """The break RUNS 21:00-22:00 UTC in summer and 22:00-23:00 UTC in winter, so it ENDS at
    22:00 UTC in summer and 23:00 UTC in winter — the same venue-local instant, 18:00 New York
    (BOT.md, measured on 39 transition-free months of US500 H1 bars). A breaker written in UTC
    would trip twice a year on the calendar alone.

    Note which two UTC hours these are: the START of the break is 21:00 UTC in summer, an hour
    the ``exness_fx_d1`` config blocks for a different reason, and confusing the two is exactly
    the mistake this test exists to catch."""
    b = breakers.trade_hours_dst
    b.record_break_end(datetime(2025, 7, 15, 22, 0, tzinfo=UTC))  # EDT -> 18:00 New York
    b.record_break_end(datetime(2025, 1, 15, 23, 0, tzinfo=UTC))  # EST -> 18:00 New York
    assert b.disagreements() == 0
    assert b.update() is BreakerState.CLOSED


def test_trade_hours_dst_breaker_trips_when_the_venue_stops_following_new_york(breakers, raw):
    b = breakers.trade_hours_dst
    need = int(raw["breakers"]["trade_hours_dst"]["escalate_after"])
    for i in range(need):
        b.record_break_end(datetime(2025, 7, 15 + i, 23, 0, tzinfo=UTC))  # 19:00 New York
    assert b.update() is BreakerState.OPEN
    assert b.history[-1].kill_criterion == "g"
    assert "decision instant" in b.history[-1].reason


def test_one_odd_reading_does_not_trip_the_dst_breaker(breakers):
    b = breakers.trade_hours_dst
    b.record_break_end(datetime(2025, 7, 14, 22, 0, tzinfo=UTC))  # 18:00 New York, correct
    b.record_break_end(datetime(2025, 7, 15, 23, 0, tzinfo=UTC))  # 19:00 New York, one outage
    assert b.disagreements() == 1
    assert b.update() is BreakerState.CLOSED


# ---------------------------------------------------------------------------------------
# Not lettered
# ---------------------------------------------------------------------------------------
def test_consecutive_loss_and_latency(breakers, raw):
    b = breakers.consecutive_loss
    for _ in range(int(raw["breakers"]["consecutive_loss"]["max_consecutive"])):
        b.record_trade(-1.0)
    assert b.update() is BreakerState.OPEN
    lat = breakers.latency
    for _ in range(int(raw["breakers"]["latency"]["n_recent"])):
        lat.record_latency(float(raw["breakers"]["latency"]["max_latency_ms"]) * 2)
    assert lat.update() is BreakerState.OPEN


def test_recovery_timeout_comes_from_the_yaml_and_a_probe_can_clear_a_breaker(breakers, raw):
    hours = float(raw["breakers"]["recovery_timeout_hours"])
    b = breakers.drawdown
    assert b.recovery_timeout == timedelta(hours=hours)
    start = datetime(2026, 9, 8, tzinfo=UTC)
    b.update(portfolio_value=1000.0, event_time=start)
    b.update(portfolio_value=500.0, event_time=start)
    assert b.state is BreakerState.OPEN
    b.update(event_time=start + timedelta(hours=hours))
    assert b.state is BreakerState.HALF_OPEN
    b.update(portfolio_value=1000.0, event_time=start + timedelta(hours=hours + 1))
    assert b.state is BreakerState.CLOSED


def test_kill_switch_callback_latches_on_any_trip():
    latched: list[str] = []

    class FakeSafe:
        def enable_kill_switch(self, reason: str) -> None:
            latched.append(reason)

    manager = cb.build_manager(
        RISK_CONFIG,
        allow_draft_criteria=True,
        on_any_trip=cb.kill_switch_on_trip(FakeSafe()),
    )
    manager.reconciliation.update(reconciliation_ok=False, reconciliation_detail="ghost")
    assert latched and "system_reconciliation" in latched[0]
