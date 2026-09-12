"""pytest suite for dashboard/build_dashboard.py.

Runs with:
    py -3.12 -m pytest dashboard\\tests -q

Every fixture here builds its own tmp_path tree; nothing in this suite reads or writes
any real bot state, any real log file, the real reviews.jsonl, or touches MT5. The two
guard tests at the bottom assert the module never references an order-placing MT5 API
and never writes outside its own directory.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_dashboard as dash  # noqa: E402


def _spec(bot_id: str) -> dash.BotSpec:
    return next(b for b in dash.BOT_SPECS if b.bot_id == bot_id)


def _touch(path: Path, content: str, mtime_offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    t = time.time() + mtime_offset
    os.utime(path, (t, t))


# ---------------------------------------------------------------------------
# Log tail parser
# ---------------------------------------------------------------------------
class TestLogParser:
    def test_parses_success_block(self):
        text = (
            "======\n"
            "[Sat 09/12/2026 13:47:06.25] Running exness_btc_8h\n"
            "======\n"
            '{"status": "success", "action": "hold", "reason": "cycle_completed_shadow"}\n'
            "Exit code: 0\n"
        )
        records = dash.parse_log_records(text, limit=10)
        assert len(records) == 1
        r = records[0]
        assert r.exit_code == 0
        assert r.skipped is False
        assert r.leg is None
        assert "cycle_completed_shadow" in r.reason

    def test_parses_skipped_block(self):
        text = (
            "======\n[Sat 09/12/2026 14:28:21.02] Running exness_gold_sess\n======\n"
            "skip: 2026-09-12 07:28Z is not a gold decision hour\n"
            "Skipped: not a gold decision hour (DST off-hour firing). Exit code: 10\n"
        )
        records = dash.parse_log_records(text, limit=10)
        assert len(records) == 1
        r = records[0]
        assert r.exit_code == 10
        assert r.skipped is True
        assert r.reason == "not a gold decision hour (DST off-hour firing)"

    def test_parses_leg_and_error_block(self):
        text = (
            "======\n[Sat 09/12/2026 13:53:08.03] Running exness_gold_sess\n======\n"
            "Decision leg: london\n"
            "Traceback (most recent call last):\n"
            "ModuleNotFoundError: No module named 'ml4t'\n"
            "Exit code: 1\n"
        )
        records = dash.parse_log_records(text, limit=10)
        r = records[0]
        assert r.leg == "london"
        assert r.exit_code == 1
        assert "ModuleNotFoundError" in r.reason

    def test_most_recent_first_and_limit(self):
        blocks = "".join(
            f"======\n[Sat 09/12/2026 1{i}:00:00.00] Running exness_btc_8h\n======\nExit code: 0\n"
            for i in range(3)
        )
        records = dash.parse_log_records(blocks, limit=2)
        assert len(records) == 2
        # block index 2 (last in file) must come first (most recent first)
        assert records[0].ts_raw.startswith("Sat 09/12/2026 12:")
        assert records[1].ts_raw.startswith("Sat 09/12/2026 11:")

    def test_no_header_lines_returns_empty(self):
        assert dash.parse_log_records("nothing here\nExit code: 0\n") == []

    def test_read_log_records_missing_file(self, tmp_path):
        records, err = dash.read_log_records(tmp_path / "nope.log", limit=10)
        assert records == []
        assert "not found" in err

    def test_read_log_records_empty_file(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("", encoding="utf-8")
        records, err = dash.read_log_records(p, limit=10)
        assert records == []
        assert "empty" in err


# ---------------------------------------------------------------------------
# State record layouts
# ---------------------------------------------------------------------------
class TestStateLayouts:
    def test_flat_run(self, tmp_path):
        spec = _spec("exness_btc_8h")
        state_dir = tmp_path / "bots" / spec.bot_id / "deploy" / "state"
        _touch(state_dir / "run_1.json", json.dumps({"n": 1}), mtime_offset=-10)
        _touch(state_dir / "run_2.json", json.dumps({"n": 2}), mtime_offset=0)
        path, data, err = dash.find_latest_state_record(tmp_path, spec)
        assert err is None
        assert data == {"n": 2}
        assert path.name == "run_2.json"

    def test_date_dir(self, tmp_path):
        spec = _spec("exness_fx_d1")
        state_dir = tmp_path / "bots" / spec.bot_id / "deploy" / "state"
        _touch(state_dir / "2023-12-29" / "run_a.json", json.dumps({"day": "old"}), mtime_offset=-10)
        _touch(state_dir / "2026-09-12" / "run_b.json", json.dumps({"day": "new"}), mtime_offset=0)
        path, data, err = dash.find_latest_state_record(tmp_path, spec)
        assert err is None
        assert data == {"day": "new"}

    def test_date_dir_picks_by_mtime_not_directory_name(self, tmp_path):
        """A decision_date directory name can be stale (data lag) while still holding the
        most recently WRITTEN record - mtime, not the folder's date string, must win."""
        spec = _spec("exness_fx_d1")
        state_dir = tmp_path / "bots" / spec.bot_id / "deploy" / "state"
        _touch(state_dir / "2026-09-01" / "run_old.json", json.dumps({"which": "alphabetically-later-but-older"}), mtime_offset=-100)
        _touch(state_dir / "2023-12-29" / "run_new.json", json.dumps({"which": "alphabetically-earlier-but-newer"}), mtime_offset=0)
        _, data, _ = dash.find_latest_state_record(tmp_path, spec)
        assert data == {"which": "alphabetically-earlier-but-newer"}

    def test_spec_date_dir(self, tmp_path):
        spec = _spec("exness_usidx_sess")
        state_dir = tmp_path / "bots" / spec.bot_id / "deploy" / "state"
        _touch(state_dir / "intraday" / "2026-02-27" / "state.json", json.dumps({"spec": "intraday"}), mtime_offset=-10)
        _touch(state_dir / "overnight" / "2026-09-10" / "state.json", json.dumps({"spec": "overnight"}), mtime_offset=0)
        path, data, err = dash.find_latest_state_record(tmp_path, spec)
        assert err is None
        assert data == {"spec": "overnight"}

    def test_runs_dir(self, tmp_path):
        spec = _spec("exness_gold_sess")
        state_dir = tmp_path / "bots" / spec.bot_id / "deploy" / "state"
        _touch(state_dir / "runs" / "2026-09-11T000000.0.json", json.dumps({"n": "first"}), mtime_offset=-10)
        _touch(state_dir / "runs" / "2026-09-12T065752.3.json", json.dumps({"n": "second"}), mtime_offset=0)
        path, data, err = dash.find_latest_state_record(tmp_path, spec)
        assert err is None
        assert data == {"n": "second"}

    def test_missing_state_dir_is_tolerated(self, tmp_path):
        spec = _spec("exness_btc_8h")
        path, data, err = dash.find_latest_state_record(tmp_path, spec)
        assert path is None
        assert data is None
        assert "no state/run record found" in err

    def test_corrupt_json_reports_error(self, tmp_path):
        spec = _spec("exness_btc_8h")
        state_dir = tmp_path / "bots" / spec.bot_id / "deploy" / "state"
        _touch(state_dir / "run_1.json", "{not valid json", mtime_offset=0)
        path, data, err = dash.find_latest_state_record(tmp_path, spec)
        assert data is None
        assert "JSON parse failed" in err


# ---------------------------------------------------------------------------
# reviews.jsonl latest-per-bot
# ---------------------------------------------------------------------------
class TestReviews:
    def test_latest_per_bot_and_malformed_line_tolerance(self, tmp_path):
        p = tmp_path / "reviews.jsonl"
        lines = [
            json.dumps({"ts": "2026-09-10T10:00:00+00:00", "bot": "exness_btc_8h", "verdict": "not_evidence_yet"}),
            "{this is not json",
            json.dumps({"ts": "2026-09-10T23:11:13+00:00", "bot": "exness_btc_8h", "verdict": "met"}),
            json.dumps({"ts": "2026-09-09T23:34:08+00:00", "bot": "exness_gold_sess", "verdict": "not_evidence_yet"}),
            "",
        ]
        p.write_text("\n".join(lines), encoding="utf-8")
        latest, err = dash.read_latest_reviews(p, ["exness_btc_8h", "exness_gold_sess", "exness_usidx_sess"])
        assert err is None
        assert latest["exness_btc_8h"]["verdict"] == "met"
        assert latest["exness_gold_sess"]["verdict"] == "not_evidence_yet"
        assert "exness_usidx_sess" not in latest

    def test_missing_file(self, tmp_path):
        latest, err = dash.read_latest_reviews(tmp_path / "nope.jsonl", ["exness_btc_8h"])
        assert latest == {}
        assert "not found" in err

    def test_unsortable_timestamps_still_pick_last_in_file_order(self, tmp_path):
        p = tmp_path / "reviews.jsonl"
        lines = [
            json.dumps({"ts": "not-a-date", "bot": "exness_gold_sess", "verdict": "first"}),
            json.dumps({"ts": "also-not-a-date", "bot": "exness_gold_sess", "verdict": "second"}),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")
        latest, _ = dash.read_latest_reviews(p, ["exness_gold_sess"])
        assert latest["exness_gold_sess"]["verdict"] == "second"


# ---------------------------------------------------------------------------
# risk_config.yaml flag extraction
# ---------------------------------------------------------------------------
class TestRiskConfigFlags:
    def test_extract_exec_mode_full(self):
        cfg = {
            "shadow_mode": True,
            "execution": {"mode": "paper", "armed": False},
            "live_risk_config": {"execution_mode": "shadow"},
            "breakers": {"pending_user_approval": True},
        }
        em = dash.extract_exec_mode(cfg)
        assert em == {
            "shadow_mode": True, "execution_mode_broker": "paper", "armed": False,
            "execution_mode_live_risk": "shadow", "pending_user_approval": True,
        }

    def test_extract_exec_mode_missing_fields_are_none(self):
        em = dash.extract_exec_mode({"execution": {"mode": "paper"}})
        assert em["shadow_mode"] is None
        assert em["armed"] is None
        assert em["pending_user_approval"] is None

    def test_extract_exec_mode_none_config(self):
        em = dash.extract_exec_mode(None)
        assert all(v is None for v in em.values())

    def test_safe_read_yaml_tolerates_missing(self, tmp_path):
        data, err = dash.safe_read_yaml(tmp_path / "nope.yaml")
        assert data is None
        assert "not found" in err

    def test_safe_read_yaml_tolerates_corrupt(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("key: [unclosed", encoding="utf-8")
        data, err = dash.safe_read_yaml(p)
        assert data is None
        assert err is not None

    def test_build_evidence_reads_evidence_boundary(self, tmp_path):
        spec = _spec("exness_fx_d1")
        (tmp_path / "bots" / spec.bot_id).mkdir(parents=True)
        risk_cfg = {
            "evidence_boundary": {"holdout_start": "2025-09-01", "holdout_end": "2026-08-31", "holdout_scored_marker": "state/holdout_scored.json"},
            "model": {"trial_count_at_deployment": 5340},
        }
        ev = dash.build_evidence(root=tmp_path, spec=spec, risk_cfg=risk_cfg, state_record=None, latest_review=None)
        assert ev.dsr_k == 5340
        assert ev.dsr_k_source == "risk_config.yaml"
        assert ev.dsr_k_unverified is False
        assert ev.holdout_start == "2025-09-01"
        assert ev.holdout_end == "2026-08-31"
        assert ev.holdout_scored is False  # marker file does not exist

    def test_build_evidence_state_record_takes_priority_over_risk_config(self, tmp_path):
        spec = _spec("exness_usidx_sess")
        (tmp_path / "bots" / spec.bot_id).mkdir(parents=True)
        risk_cfg = {"phase5_trial_count": 999}
        state_record = {"evidence": {"trial_count_K": 1780, "phase5_survivors": 0, "phase5_specs_scored": 1403}}
        ev = dash.build_evidence(root=tmp_path, spec=spec, risk_cfg=risk_cfg, state_record=state_record, latest_review=None)
        assert ev.dsr_k == 1780
        assert ev.dsr_k_source == "latest deploy state record"
        assert ev.dsr_k_unverified is False
        assert ev.survivors == 0
        assert ev.fdr_n == 1403
        assert ev.fdr_n_source == "latest deploy state record (specs scored)"

    def test_build_evidence_never_fabricates_positive_survivor_count_from_hyphenated_prose(self, tmp_path):
        """Regression: 'generation-3 survivor' must not be read as '3 survivors'."""
        spec = _spec("exness_fx_d1")
        bot_dir = tmp_path / "bots" / spec.bot_id
        bot_dir.mkdir(parents=True)
        (bot_dir / "BOT.md").write_text(
            "every generation-3 survivor is re-run; phase-5 survivor count: 0 survivors of 1,068",
            encoding="utf-8",
        )
        ev = dash.build_evidence(root=tmp_path, spec=spec, risk_cfg=None, state_record=None, latest_review=None)
        assert ev.survivors == 0
        assert "literal" in ev.survivors_source

    def test_build_evidence_three_counters_are_separate_and_botmd_fallback_is_unverified(self, tmp_path):
        """BLOCKING findings 1 and 6: DSR-K, FDR-n and K5 must be three distinct fields
        (never collapsed into one ambiguous "K"), and a value found only by grepping the
        whole BOT.md must be marked unverified rather than shown as a trustworthy number."""
        spec = _spec("exness_btc_8h")
        bot_dir = tmp_path / "bots" / spec.bot_id
        bot_dir.mkdir(parents=True)
        (bot_dir / "BOT.md").write_text(
            "Some unrelated K = 999 mentioned early on, before the ledger.\n\n"
            "| 5 Backtest and inference | **NOT OPENED, generation 1 CLOSED before this phase** | 2026-09-10 | evidence |\n\n"
            "## Trials counted for the Deflated Sharpe Ratio\n\n"
            "| Date | What was tried | Count |\n|---|---|---|\n"
            "| 2026-09-11 | a rule-wording fix, not a trial | **K = 0; FDR-n unchanged (73 then 356)** |\n\n"
            "Cumulative K = 0. K5 = 2,136 remains a declared-but-undrawn number.\n",
            encoding="utf-8",
        )
        ev = dash.build_evidence(root=tmp_path, spec=spec, risk_cfg=None, state_record=None, latest_review=None)
        # Trials-table extraction wins over the earlier, unrelated "K = 999" - and is trusted.
        assert ev.dsr_k == 0
        assert ev.dsr_k_unverified is False
        assert "Trials table" in ev.dsr_k_source
        assert ev.fdr_n == 356
        assert ev.fdr_n_unverified is False
        assert ev.k5 == 2136
        assert ev.k5_unverified is False
        assert ev.phase5_status == "NOT OPENED, generation 1 CLOSED before this phase"

    def test_build_evidence_whole_document_fallback_is_marked_unverified(self, tmp_path):
        spec = _spec("exness_gold_sess")
        bot_dir = tmp_path / "bots" / spec.bot_id
        bot_dir.mkdir(parents=True)
        (bot_dir / "BOT.md").write_text("no Trials heading here, just K = 2,514 somewhere in prose", encoding="utf-8")
        ev = dash.build_evidence(root=tmp_path, spec=spec, risk_cfg=None, state_record=None, latest_review=None)
        assert ev.dsr_k == 2514
        assert ev.dsr_k_unverified is True
        assert "free-text search" in ev.dsr_k_source

    def test_extract_trials_table_block_returns_none_without_heading(self):
        assert dash.extract_trials_table_block("no heading here at all") is None

    def test_extract_phase5_status(self):
        text = "| 5 Backtest | **NOT OPENED, generation 1 CLOSED** | 2026-09-10 | note |"
        assert dash.extract_phase5_status(text) == "NOT OPENED, generation 1 CLOSED"

    def test_extract_fdr_n_takes_last_number(self):
        assert dash.extract_fdr_n("FDR-n at the IC tier = 73 (phase 3) then 356 (phase 4), spent") == 356
        assert dash.extract_fdr_n("FDR-n unchanged (73 then 356)") == 356
        assert dash.extract_fdr_n("no FDR mention here") is None

    def test_extract_decision_date_top_level(self):
        assert dash.extract_decision_date({"decision_date": "2026-09-12"}) == "2026-09-12"

    def test_extract_decision_date_nested_fx_d1_shape(self):
        record = {"steps": {"2_features": {"decision_date": "2023-12-29"}}}
        assert dash.extract_decision_date(record) == "2023-12-29"

    def test_extract_decision_date_missing(self):
        assert dash.extract_decision_date({"other": 1}) is None
        assert dash.extract_decision_date(None) is None


class TestVerdictBadge:
    def test_met_is_never_green(self):
        html = dash.verdict_badge("met")
        assert "badge-green" not in html
        assert "phase gate met (process), not evidence of an edge" in html

    def test_not_evidence_yet_is_amber(self):
        html = dash.verdict_badge("not_evidence_yet")
        assert "badge-amber" in html

    def test_unknown_verdict_is_never_green(self):
        assert "badge-green" not in dash.verdict_badge("info")
        assert "badge-green" not in dash.verdict_badge("anything-else")
        assert "badge-green" not in dash.verdict_badge(None)


class TestUnverifiedCell:
    def test_unverified_value_is_struck_through_and_labelled(self):
        html = dash._unverified_cell(42, "BOT.md (free-text search)", True)
        assert "<s" in html
        assert "unverified" in html

    def test_verified_value_is_plain(self):
        html = dash._unverified_cell(42, "risk_config.yaml", False)
        assert "<s class=" not in html
        assert "42" in html

    def test_none_value(self):
        html = dash._unverified_cell(None, "not found", False)
        assert "n/a" in html


class TestMt5PathResolution:
    """resolve_single_terminal_path() never calls mt5.initialize() itself - it only lists
    OS processes - so these tests never touch a real MT5 terminal."""

    def test_no_path_exactly_one_running(self, monkeypatch):
        monkeypatch.setattr(dash, "list_terminal_executable_paths", lambda timeout=5.0: [r"D:\MT5\terminal64.exe"])
        path, reason = dash.resolve_single_terminal_path(None)
        assert path == r"D:\MT5\terminal64.exe"
        assert reason is None

    def test_no_path_zero_running_refuses(self, monkeypatch):
        monkeypatch.setattr(dash, "list_terminal_executable_paths", lambda timeout=5.0: [])
        path, reason = dash.resolve_single_terminal_path(None)
        assert path is None
        assert "0 or >1 terminals running" in reason

    def test_no_path_multiple_running_refuses(self, monkeypatch):
        monkeypatch.setattr(dash, "list_terminal_executable_paths", lambda timeout=5.0: [r"C:\A\terminal64.exe", r"C:\B\terminal64.exe"])
        path, reason = dash.resolve_single_terminal_path(None)
        assert path is None
        assert "0 or >1 terminals running" in reason

    def test_explicit_path_running(self, monkeypatch):
        monkeypatch.setattr(dash, "list_terminal_executable_paths", lambda timeout=5.0: [r"C:\A\terminal64.exe", r"C:\B\terminal64.exe"])
        path, reason = dash.resolve_single_terminal_path(r"C:\B\terminal64.exe")
        assert path == r"C:\B\terminal64.exe"
        assert reason is None

    def test_explicit_path_not_running_refuses(self, monkeypatch):
        monkeypatch.setattr(dash, "list_terminal_executable_paths", lambda timeout=5.0: [r"C:\A\terminal64.exe"])
        path, reason = dash.resolve_single_terminal_path(r"C:\Other\terminal64.exe")
        assert path is None
        assert "not among" in reason

    def test_process_listing_unavailable(self, monkeypatch):
        monkeypatch.setattr(dash, "list_terminal_executable_paths", lambda timeout=5.0: None)
        path, reason = dash.resolve_single_terminal_path(None)
        assert path is None
        assert "cannot determine" in reason

    def test_positions_never_carry_a_profit_field(self):
        import inspect
        source = inspect.getsource(dash.read_mt5_snapshot)
        assert '"profit"' not in source


class TestFooterSafetySummary:
    def _view(self, bot_id, armed, broker_mode, live_mode):
        spec = next(b for b in dash.BOT_SPECS if b.bot_id == bot_id)
        return {"spec": spec, "exec_mode": {"armed": armed, "execution_mode_broker": broker_mode, "execution_mode_live_risk": live_mode}}

    def test_all_shadow_no_armed_is_green(self):
        views = [self._view(b.bot_id, False, "paper", "shadow") for b in dash.BOT_SPECS]
        html = dash.render_footer_safety_summary(views)
        assert "badge-red" not in html
        assert "badge-green" in html

    def test_armed_bot_is_red(self):
        views = [self._view(b.bot_id, False, "paper", "shadow") for b in dash.BOT_SPECS]
        views[0]["exec_mode"]["armed"] = True
        html = dash.render_footer_safety_summary(views)
        assert "badge-red" in html
        assert views[0]["spec"].bot_id in html

    def test_live_execution_mode_is_red(self):
        views = [self._view(b.bot_id, False, "paper", "shadow") for b in dash.BOT_SPECS]
        views[0]["exec_mode"]["execution_mode_live_risk"] = "live"
        html = dash.render_footer_safety_summary(views)
        assert "badge-red" in html


# ---------------------------------------------------------------------------
# Task Scheduler output parsing
# ---------------------------------------------------------------------------
class TestTaskScheduler:
    def test_parses_single_object(self):
        raw = json.dumps({"Name": "Exness_Bot_BTC_8h", "Found": True, "State": "Ready", "LastRunTime": "2026-09-12T06:00:00+00:00", "LastTaskResult": 0, "NextRunTime": "2026-09-12T15:01:01+07:00"})
        result = dash.parse_task_scheduler_json(raw)
        assert result["Exness_Bot_BTC_8h"]["found"] is True
        assert result["Exness_Bot_BTC_8h"]["state"] == "Ready"
        assert result["Exness_Bot_BTC_8h"]["last_task_result"] == 0

    def test_parses_array(self):
        raw = json.dumps([
            {"Name": "A", "Found": True, "State": "Ready", "LastRunTime": None, "LastTaskResult": 267011, "NextRunTime": None},
            {"Name": "B", "Found": False, "State": None, "LastRunTime": None, "LastTaskResult": None, "NextRunTime": None},
        ])
        result = dash.parse_task_scheduler_json(raw)
        assert result["A"]["last_task_result"] == 267011
        assert result["B"]["found"] is False

    def test_garbage_raises(self):
        with pytest.raises(Exception):
            dash.parse_task_scheduler_json("not json at all")


# ---------------------------------------------------------------------------
# Traffic-light rules
# ---------------------------------------------------------------------------
class TestTrafficLight:
    READY = {"found": True, "state": "Ready"}

    def _call(self, **kwargs):
        defaults = dict(
            window_exit_codes=[0], last_skipped=False, log_error=None, task_info=self.READY,
            fleet_halted=False, fleet_foreign_magic=False,
        )
        defaults.update(kwargs)
        return dash.traffic_light(**defaults)

    def test_green(self):
        light, _ = self._call(window_exit_codes=[0])
        assert light == "green"

    def test_amber_on_exit_1(self):
        light, reasons = self._call(window_exit_codes=[1])
        assert light == "amber"
        assert any("exited 1" in r for r in reasons)

    def test_amber_on_skipped(self):
        light, reasons = self._call(window_exit_codes=[10], last_skipped=True)
        assert light == "amber"
        assert any("Skipped" in r for r in reasons)

    def test_red_on_halt(self):
        light, reasons = self._call(fleet_halted=True)
        assert light == "red"
        assert any("halt" in r for r in reasons)

    def test_red_on_foreign_magic(self):
        light, reasons = self._call(fleet_foreign_magic=True)
        assert light == "red"
        assert any("magic" in r for r in reasons)

    def test_red_on_missing_task(self):
        light, reasons = self._call(task_info=None)
        assert light == "red"
        assert any("missing" in r for r in reasons)

    def test_red_on_disabled_task(self):
        light, _ = self._call(task_info={"found": True, "state": "Disabled"})
        assert light == "red"

    def test_red_on_unexpected_exit_code(self):
        light, reasons = self._call(window_exit_codes=[2])
        assert light == "red"
        assert any("not in {0, 1, 10}" in r for r in reasons)

    def test_amber_on_no_run_history(self):
        light, reasons = self._call(window_exit_codes=[], log_error="log file not found")
        assert light == "amber"

    def test_red_takes_priority_over_amber(self):
        """Halt present AND last exit 1 -> still red, not amber."""
        light, reasons = self._call(window_exit_codes=[1], fleet_halted=True)
        assert light == "red"

    def test_blocking_finding_5_amber_when_older_run_in_window_errored(self):
        """Regression: fx_d1's real run history was [0, 1, 1, 1, 1] (most-recent-first) and
        an earlier version of this dashboard rendered it green because it only looked at
        window_exit_codes[0]. Any 1 anywhere in the displayed window must force amber."""
        light, reasons = self._call(window_exit_codes=[0, 1, 1, 1, 1])
        assert light == "amber"
        assert any("4 run(s)" in r for r in reasons)

    def test_green_requires_the_whole_window_clean(self):
        light, _ = self._call(window_exit_codes=[0, 0, 10, 0])
        assert light == "green"

    def test_red_on_armed_true(self):
        light, reasons = self._call(armed=True)
        assert light == "red"
        assert any("armed is True" in r for r in reasons)

    def test_red_on_non_paper_shadow_execution_mode(self):
        light, reasons = self._call(execution_modes=["live"])
        assert light == "red"
        assert any("not in {paper, shadow}" in r for r in reasons)

    def test_execution_modes_paper_and_shadow_do_not_trip_red(self):
        light, _ = self._call(execution_modes=["paper", "shadow"])
        assert light == "green"

    def test_red_on_live_trading_permitted_while_holdout_unscored(self):
        light, reasons = self._call(live_trading_permitted=True, holdout_scored=False)
        assert light == "red"
        assert any("live_trading_permitted" in r for r in reasons)

    def test_live_trading_permitted_true_with_holdout_scored_true_is_not_this_red_reason(self):
        light, reasons = self._call(live_trading_permitted=True, holdout_scored=True)
        assert not any("live_trading_permitted" in r for r in reasons)


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------
class TestGuards:
    SOURCE = Path(dash.__file__).read_text(encoding="utf-8")

    def test_dont_write_bytecode_is_set(self):
        """BLOCKING finding 3: importlib-loading runners/session_gate.py and importing
        bots._shared wrote .pyc files outside dashboard\\ (runners\\__pycache__,
        bots\\__pycache__, bots\\_shared\\__pycache__). sys.dont_write_bytecode must be set
        before any of that happens - checked two ways: the flag is actually True in this
        process (proving it took effect, since build_dashboard is already imported), and it
        is set textually before the first non-stdlib import in the source."""
        assert sys.dont_write_bytecode is True
        set_pos = self.SOURCE.index("sys.dont_write_bytecode = True")
        bots_import_pos = self.SOURCE.index("import bots._shared")
        assert set_pos < bots_import_pos

    def test_never_calls_order_send(self):
        assert "order_send(" not in self.SOURCE

    def test_never_calls_mt5_login(self):
        assert "mt5.login(" not in self.SOURCE
        assert ".login(" not in self.SOURCE

    def test_never_imports_deployment_loop_module(self):
        import re
        assert not re.search(r"^\s*(from|import)\s+[\w.]*deployment_loop", self.SOURCE, re.MULTILINE)

    def test_atomic_write_refuses_outside_dashboard_dir(self, tmp_path):
        outside = tmp_path / "elsewhere.html"
        with pytest.raises(RuntimeError):
            dash.atomic_write_text(outside, "<html></html>")
        assert not outside.exists()

    def test_atomic_write_succeeds_inside_dashboard_dir(self, tmp_path, monkeypatch):
        # Redirect DASHBOARD_DIR to a tmp dir for this one test so it never touches the
        # real dashboard/dashboard.html.
        monkeypatch.setattr(dash, "DASHBOARD_DIR", tmp_path)
        target = tmp_path / "sub" / "out.html"
        dash.atomic_write_text(target, "<html>ok</html>")
        assert target.read_text(encoding="utf-8") == "<html>ok</html>"

    def test_never_reads_dotenv_files(self):
        # A substring check on ".env" would false-positive on this module's own docstrings
        # (which document the guard in prose). The actual mechanism to forbid is importing
        # python-dotenv or calling load_dotenv() - see v9_dashboard/README.md's "Design
        # decisions" for why that specific call is the hazard.
        assert "load_dotenv" not in self.SOURCE
        assert "import dotenv" not in self.SOURCE
        assert "from dotenv" not in self.SOURCE


# ---------------------------------------------------------------------------
# Fixed-hour / session-gate decision instants
# ---------------------------------------------------------------------------
class TestDecisionInstants:
    def test_next_fixed_instants_btc_daily(self):
        spec = _spec("exness_btc_8h")
        now = datetime(2026, 9, 12, 7, 47, tzinfo=timezone.utc)  # Saturday
        instants = dash.next_fixed_instants(spec, now, count=3)
        assert [i["decision_utc"].hour for i in instants] == [8, 16, 0]

    def test_next_fixed_instants_fx_skips_weekend(self):
        spec = _spec("exness_fx_d1")
        now = datetime(2026, 9, 12, 7, 47, tzinfo=timezone.utc)  # Saturday
        instants = dash.next_fixed_instants(spec, now, count=1)
        assert instants[0]["decision_utc"].weekday() == 0  # Monday
        assert instants[0]["decision_utc"].hour == 20

    def test_session_gate_module_loads_and_produces_instants(self):
        root = Path(dash.DEFAULT_ROOT)
        if not root.exists():
            pytest.skip("real ML_4_Bot root not present on this machine")
        gate_mod = dash.load_session_gate(root)
        assert gate_mod is not None
        now = datetime(2026, 9, 12, 7, 47, tzinfo=timezone.utc)
        instants = dash.next_session_gate_instants(gate_mod, "gold", now, n_weekdays=2)
        assert len(instants) > 0
        assert all(it["leg"] in ("london", "ny") for it in instants)
