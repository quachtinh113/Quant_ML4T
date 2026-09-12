"""dashboard - read-only HTML operator dashboard for the 4 Exness bots.

    exness_btc_8h  (BTCUSD,               magic 260904)
    exness_fx_d1   (5 FX pairs,           magic 260901)
    exness_usidx_sess (US500/USTEC,       magic 260902)
    exness_gold_sess  (XAUUSD/XAGUSD,     magic 260903)

All four bots closed Phase 5 with 0 survivors (see each bot's BOT.md Trials table).
None is deployable. This dashboard never claims otherwise: it shows shadow_mode,
execution_mode, pending_user_approval and the mentor's own verdicts verbatim, and it
never prints a PnL or "profit" figure derived from a run these bots' own risk_config
withholds pending an unscored holdout.

Reads (all tolerant of missing/corrupt - rendered as "n/a" with a reason):
  * bots/<bot>/deploy/risk_config.yaml            - YAML, yaml.safe_load
  * bots/<bot>/deploy/state/**                    - JSON run/state records, one layout
    per bot (see BOT_SPECS[*]["state_kind"] below); the LATEST by file mtime is used
  * bots/<bot>/monitor/monitor_config.yaml        - YAML (drift/rollout thresholds;
    circuit-breaker thresholds themselves live in risk_config.yaml, see dashboard/README.md)
  * bots/<bot>/monitor/circuit_breakers.py        - TEXT, regex for `class X(CircuitBreaker)`
    names only; never imported
  * bots/<bot>/BOT.md                             - TEXT, best-effort regex fallback for
    the trial count K / survivor count / holdout window when risk_config.yaml and the
    latest state record do not carry them (see extract_evidence())
  * logs/<bot>.log                                - TEXT, tailed and parsed into per-cycle
    blocks (never truncated on disk, never rewritten)
  * D:\\05_Quant\\merg\\data_lake\\reviews.jsonl    - JSONL, READ ONLY, latest entry per bot
  * .../mt5/monitor/account_halt.json             - existence = fleet halt
  * .../mt5/monitor/account_hwm.json              - JSON, tolerant
  * runners/session_gate.py                       - imported via importlib (read-only
    module: due_leg/decision_utc/LEGS/SESSIONS), NEVER bots/<bot>/deploy/deployment_loop.py
  * bots/_shared/__init__.py                      - imported (MAGIC_ALLOCATION,
    RESERVED_MAGICS); pure dict/function definitions, no I/O, no .env
  * Windows Task Scheduler, via `Get-ScheduledTask`/`Get-ScheduledTaskInfo` (PowerShell,
    read-only, no admin rights needed)
  * MT5, READ-ONLY, only with --mt5 (default OFF): mt5.initialize() with NO path argument
    (these bots' own MT5Broker connects the same way - see bots/_shared/mt5_broker.py -
    there is no single configured terminal64.exe path for this fleet), account_info(),
    positions_get(), terminal_info(), shutdown(). Never mt5.login, never order_send, never
    any position-modifying call. Exactly like dashboard.html's sibling v9_dashboard,
    initialize() is only ever called after an OS-level check
    (Get-CimInstance Win32_Process -Filter "Name='terminal64.exe'") confirms a terminal is
    ALREADY running; if that cannot be confirmed, initialize() is never called, because it
    launches a terminal when none is running.

Writes:
  * dashboard/dashboard.html, atomically (tmp file + os.replace), and ONLY inside this
    script's own directory - atomic_write_text() asserts this at runtime and raises rather
    than write elsewhere.

Never touches: bots/**/deploy/state/** (read-only), bots/**/deploy/deployment_loop.py
(never imported - it may write state/orders), run_log/registry.db, any holdout file,
data_lake/reviews.jsonl (read-only), .env* (never read), and it never calls
mt5.order_send / mt5.login / any order-placing API, and never launches terminal64.exe.
"""
from __future__ import annotations

import sys

# Set before any other import (including the local `bots._shared` / runners/session_gate.py
# imports further down): this process must never write a .pyc anywhere, and in particular
# never outside dashboard\ (bots\__pycache__, bots\_shared\__pycache__,
# runners\__pycache__\session_gate.cpython-312.pyc were all found after an earlier run - see
# dashboard/README.md). dont_write_bytecode disables bytecode caching for every import made
# by this interpreter process from here on, stdlib and local alike.
sys.dont_write_bytecode = True

import argparse
import importlib.util
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is expected on py -3.12 for this workspace
    yaml = None

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # pragma: no cover - optional, only used with --mt5
    mt5 = None

DASHBOARD_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = r"D:\05_Quant\ML_4_Bot"
DEFAULT_REVIEWS = r"D:\05_Quant\merg\data_lake\reviews.jsonl"
DEFAULT_HALT_FILE = r"D:\05_Quant\machine-learning-for-trading\data\mt5\monitor\account_halt.json"
DEFAULT_HWM_FILE = r"D:\05_Quant\machine-learning-for-trading\data\mt5\monitor\account_hwm.json"

VN_OFFSET = timedelta(hours=7)  # Asia/Ho_Chi_Minh, fixed offset, no DST
VN_TZ_LABEL = "VN (UTC+7)"

TASK_NAMES = {
    "exness_btc_8h": "Exness_Bot_BTC_8h",
    "exness_fx_d1": "Exness_Bot_FX_D1",
    "exness_usidx_sess": "Exness_Bot_USIDX_Sess",
    "exness_gold_sess": "Exness_Bot_Gold_Sess",
}

# Windows Task Scheduler LastTaskResult codes worth naming; anything else is shown as a raw
# hex/decimal code rather than guessed at.
TASK_RESULT_LABELS = {
    0: "success (exit 0)",
    1: "error (exit 1)",
    10: "skipped - not a decision hour (exit 10)",
    267009: "currently running (0x41301)",
    267011: "has not run yet (0x41303)",
}

RESERVED_MAGICS_FALLBACK = {202500: "legacy V9 Continuum bot (retired)"}


# ---------------------------------------------------------------------------
# Bot registry - the shape differs per bot (state layout, decision schedule); this table is
# the single place that says how, so the rest of the file is bot-agnostic.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BotSpec:
    bot_id: str
    label_vi: str
    symbols: Tuple[str, ...]
    magic: int
    cadence_vi: str
    state_kind: str  # "flat_run" | "date_dir" | "spec_date_dir" | "runs_dir"
    specs: Tuple[str, ...] = ()  # only for spec_date_dir (usidx: intraday/overnight)
    has_monitor_dir: bool = True
    decision_kind: str = "fixed"  # "fixed_daily_hours" | "fixed_weekday_hour" | "session_gate"
    decision_hours_utc: Tuple[int, ...] = ()
    decision_weekday_only: bool = False
    gate_key: str = ""  # "usidx" | "gold", for decision_kind == "session_gate"

    @property
    def log_name(self) -> str:
        return {
            "exness_btc_8h": "btc_8h.log",
            "exness_fx_d1": "fx_d1.log",
            "exness_usidx_sess": "usidx_sess.log",
            "exness_gold_sess": "gold_sess.log",
        }[self.bot_id]

    @property
    def task_name(self) -> str:
        return TASK_NAMES[self.bot_id]


BOT_SPECS: Tuple[BotSpec, ...] = (
    BotSpec(
        bot_id="exness_btc_8h",
        label_vi="BTC 8h",
        symbols=("BTCUSD",),
        magic=260904,
        cadence_vi="Lưới 8 giờ (00/08/16 UTC), 7 ngày/tuần",
        state_kind="flat_run",
        decision_kind="fixed_daily_hours",
        decision_hours_utc=(0, 8, 16),
        decision_weekday_only=False,
    ),
    BotSpec(
        bot_id="exness_fx_d1",
        label_vi="FX D1",
        symbols=("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"),
        magic=260901,
        cadence_vi="Hằng ngày 20:00 UTC (Thứ 2 - Thứ 6)",
        state_kind="date_dir",
        decision_kind="fixed_daily_hours",
        decision_hours_utc=(20,),
        decision_weekday_only=True,
    ),
    BotSpec(
        bot_id="exness_usidx_sess",
        label_vi="US Index Sess",
        symbols=("US500", "USTEC"),
        magic=260902,
        cadence_vi="2 chân trong ngày: intraday (NYSE mở +30') và overnight (NYSE đóng)",
        state_kind="spec_date_dir",
        specs=("intraday", "overnight"),
        decision_kind="session_gate",
        gate_key="usidx",
    ),
    BotSpec(
        bot_id="exness_gold_sess",
        label_vi="Gold/Silver Sess",
        symbols=("XAUUSD", "XAGUSD"),
        magic=260903,
        cadence_vi="2 chân theo phiên: london (London mở +1h), ny (New York mở +1h)",
        state_kind="runs_dir",
        decision_kind="session_gate",
        gate_key="gold",
    ),
)
# exness_btc_8h has no bots/exness_btc_8h/monitor/ dir - its breakers live only in risk_config.
BOT_SPECS = tuple(
    b if b.bot_id != "exness_btc_8h" else BotSpec(**{**b.__dict__, "has_monitor_dir": False})
    for b in BOT_SPECS
)


# ---------------------------------------------------------------------------
# Generic, tolerant readers - every one of these returns None/[] on any failure, never raises
# ---------------------------------------------------------------------------
def safe_read_text(path: Path) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(parsed_dict_or_None, error_reason_or_None)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, "file not found"
    except Exception as e:
        return None, f"read failed: {type(e).__name__}: {e}"
    try:
        data = json.loads(text)
    except Exception as e:
        return None, f"JSON parse failed: {type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return None, "JSON root is not an object"
    return data, None


def safe_read_yaml(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if yaml is None:
        return None, "PyYAML not importable"
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, "file not found"
    except Exception as e:
        return None, f"read failed: {type(e).__name__}: {e}"
    try:
        data = yaml.safe_load(text)
    except Exception as e:
        return None, f"YAML parse failed: {type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return None, "YAML root is not a mapping"
    return data, None


def get_in(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_decision_date(state_record: Optional[Dict[str, Any]]) -> Optional[str]:
    """The decision date the LATEST state record itself claims to be for - e.g. fx_d1's
    steps.2_features.decision_date - shown as its own row (BLOCKING finding 7) so an
    operator can see when the record's underlying market data is actually from, independent
    of when the cycle happened to run."""
    if not isinstance(state_record, dict):
        return None
    for path in (("decision_date",), ("steps", "2_features", "decision_date"), ("step2_features", "decision_date")):
        v = get_in(state_record, *path)
        if isinstance(v, str) and v:
            return v
    return None


def extract_exec_mode(risk_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Execution-mode flags read straight from risk_config.yaml - never re-derived, never
    guessed. A field absent from the YAML (e.g. no 'pending_user_approval' block for a bot
    that has no draft kill criteria) renders as None, distinct from an explicit False."""
    return {
        "shadow_mode": risk_cfg.get("shadow_mode") if risk_cfg else None,
        "execution_mode_broker": get_in(risk_cfg, "execution", "mode"),
        "armed": get_in(risk_cfg, "execution", "armed"),
        "execution_mode_live_risk": get_in(risk_cfg, "live_risk_config", "execution_mode"),
        "pending_user_approval": get_in(risk_cfg, "breakers", "pending_user_approval"),
    }


def find_latest_by_mtime(patterns: Sequence[Tuple[Path, str]]) -> Optional[Path]:
    """The file matching any of `patterns` (each a (base_dir, glob_pattern) pair, where
    glob_pattern may itself contain multiple wildcard segments, e.g. "*/run_*.json") with
    the newest mtime, or None if none exist / all are unreadable."""
    candidates: List[Tuple[float, Path]] = []
    for base_dir, glob_pat in patterns:
        try:
            if not base_dir.exists():
                continue
            for p in base_dir.glob(glob_pat):
                try:
                    candidates.append((p.stat().st_mtime, p))
                except OSError:
                    continue
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Per-bot latest state/run record (layout differs - see docstring)
# ---------------------------------------------------------------------------
def find_latest_state_record(root: Path, spec: BotSpec) -> Tuple[Optional[Path], Optional[Dict[str, Any]], Optional[str]]:
    """(path, parsed_dict, error_reason). All three may legitimately be None/None/"reason"."""
    deploy_dir = root / "bots" / spec.bot_id / "deploy"
    state_dir = deploy_dir / "state"
    if spec.state_kind == "flat_run":
        latest = find_latest_by_mtime([(state_dir, "run_*.json")])
    elif spec.state_kind == "date_dir":
        latest = find_latest_by_mtime([(state_dir, "*/run_*.json")])
    elif spec.state_kind == "spec_date_dir":
        latest = find_latest_by_mtime([(state_dir, f"{s}/*/state.json") for s in spec.specs])
    elif spec.state_kind == "runs_dir":
        latest = find_latest_by_mtime([(state_dir / "runs", "*.json")])
    else:  # pragma: no cover - defensive, BOT_SPECS is fixed
        return None, None, f"unknown state_kind {spec.state_kind!r}"
    if latest is None:
        return None, None, "no state/run record found under deploy/state/"
    data, err = safe_read_json(latest)
    return latest, data, err


# ---------------------------------------------------------------------------
# Log tail parsing - splits a runner log into per-invocation blocks on the
# "[<timestamp>] Running <bot>" header line each run_*.bat writes.
# ---------------------------------------------------------------------------
HEADER_RE = re.compile(r"\[(?P<ts>[^\]]+)\]\s+Running\s+(?P<bot>\S+)")
EXIT_CODE_RE = re.compile(r"Exit code:\s*(-?\d+)")
SKIPPED_RE = re.compile(r"Skipped:\s*(.+)")
LEG_RE = re.compile(r"Decision (?:leg|spec):\s*(\S+)")
NOT_READY_RE = re.compile(r'"not_ready":\s*"([^"]{1,220})')
SIGNAL_STATUS_RE = re.compile(r"SIGNAL STATUS:\s*([^\r\n]+)")
REASON_JSON_RE = re.compile(r'"reason":\s*"([^"]{1,220})"')
STOPPED_AFTER_RE = re.compile(r'"stopped_after":\s*"([^"]{1,220})"')
STOPPED_AT_RE = re.compile(r'"stopped_at":\s*"([^"]{1,160})"')
ERROR_LINE_RE = re.compile(r"([A-Za-z_.]+Error:\s*[^\r\n]+)")
INFO_LINE_RE = re.compile(r"\[INFO\][^\r\n]+")


@dataclass
class LogRecord:
    ts_raw: str
    leg: Optional[str]
    exit_code: Optional[int]
    skipped: bool
    reason: str


def _strip_exit_suffix(s: str) -> str:
    return re.sub(r"\.?\s*Exit code:\s*-?\d+\s*$", "", s).strip()


def summarize_reason(block_text: str, exit_code: Optional[int], skip_reason: Optional[str]) -> str:
    if skip_reason:
        return _strip_exit_suffix(skip_reason)[:220]
    m = NOT_READY_RE.search(block_text)
    if m:
        return m.group(1)[:220]
    m = SIGNAL_STATUS_RE.search(block_text)
    if m:
        return m.group(1).strip()[:220]
    all_reasons = REASON_JSON_RE.findall(block_text)
    if all_reasons:
        return all_reasons[-1][:220]
    m = STOPPED_AFTER_RE.search(block_text)
    if m:
        return m.group(1)[:220]
    m = STOPPED_AT_RE.search(block_text)
    if m:
        return m.group(1)[:220]
    errs = ERROR_LINE_RE.findall(block_text)
    if errs:
        return errs[-1][:220]
    infos = INFO_LINE_RE.findall(block_text)
    if infos:
        return infos[-1][:220]
    if exit_code == 0:
        return "completed"
    return "see log for detail"


def parse_log_records(text: str, limit: int = 10) -> List[LogRecord]:
    """The last `limit` runner invocations, most-recent first."""
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return []
    blocks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((m.group("ts"), text[start:end]))
    records: List[LogRecord] = []
    for ts_raw, block_text in blocks[-limit:]:
        codes = EXIT_CODE_RE.findall(block_text)
        exit_code = int(codes[-1]) if codes else None
        skip_m = SKIPPED_RE.search(block_text)
        skip_reason = skip_m.group(1) if skip_m else None
        skipped = bool(skip_reason) or exit_code == 10
        leg_m = LEG_RE.search(block_text)
        reason = summarize_reason(block_text, exit_code, skip_reason)
        records.append(LogRecord(
            ts_raw=ts_raw.strip(),
            leg=leg_m.group(1) if leg_m else None,
            exit_code=exit_code,
            skipped=skipped,
            reason=reason,
        ))
    records.reverse()  # most recent first
    return records


def read_log_records(log_path: Path, limit: int = 10) -> Tuple[List[LogRecord], Optional[str]]:
    text = safe_read_text(log_path)
    if text is None:
        return [], "log file not found or unreadable"
    if not text.strip():
        return [], "log file is empty"
    records = parse_log_records(text, limit=limit)
    if not records:
        return [], "log file has no recognisable 'Running <bot>' header lines"
    return records, None


# ---------------------------------------------------------------------------
# reviews.jsonl - READ ONLY, latest entry per bot
# ---------------------------------------------------------------------------
def read_latest_reviews(reviews_path: Path, bot_ids: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """{bot_id: latest_entry_dict} for whichever of `bot_ids` have at least one entry.
    Malformed lines are skipped, not fatal. Returns ({}, reason) if the file itself is
    missing/unreadable."""
    try:
        lines = Path(reviews_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return {}, "reviews.jsonl not found"
    except Exception as e:
        return {}, f"read failed: {type(e).__name__}: {e}"

    latest: Dict[str, Dict[str, Any]] = {}
    latest_ts: Dict[str, datetime] = {}
    wanted = set(bot_ids)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if not isinstance(entry, dict):
            continue
        bot = entry.get("bot")
        if bot not in wanted:
            continue
        ts_raw = entry.get("ts")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            ts = None
        if bot not in latest:
            latest[bot] = entry
            if ts is not None:
                latest_ts[bot] = ts
            continue
        prev_ts = latest_ts.get(bot)
        if ts is not None and (prev_ts is None or ts >= prev_ts):
            latest[bot] = entry
            latest_ts[bot] = ts
        elif ts is None and prev_ts is None:
            latest[bot] = entry  # last one in file order wins when neither has a usable ts
    return latest, None


# ---------------------------------------------------------------------------
# BOT.md best-effort text fallback (only used when risk_config.yaml / the latest state
# record do not carry a structured trial count, survivor count or holdout window).
#
# This fleet's bots track (at least) THREE different counters, not one "trial count K" -
# see bots/exness_btc_8h/BOT.md's "Trials counted for the Deflated Sharpe Ratio" section
# and its "three-counter rule": an IC-level screen is not a Sharpe-based selection, so it
# does not move the Deflated Sharpe Ratio's K.
#   DSR-K:  the count the Deflated Sharpe Ratio actually divides by (a Sharpe-based
#           selection stage, e.g. a backtest sweep). Can legitimately be 0 even after a
#           screen scored hundreds of columns, if nothing reached a Sharpe-producing stage.
#   FDR-n:  the size of the multiple-testing family at the IC/screening tier (Benjamini-
#           Hochberg n) - a DIFFERENT, usually larger, number than DSR-K.
#   K5:     a declared-but-undrawn number for a LATER stage that never opened (e.g. the
#           backtest population phase 5 would spend if it were ever reached).
# Rendering these as one ambiguous "K" (as an earlier version of this dashboard did) made
# a closed-with-0-survivors generation (DSR-K=0) look identical to "no evidence gathered at
# all", when in fact hundreds of screening trials (FDR-n) were spent and a much larger K5
# was declared and deliberately never drawn. All three are shown separately, every one
# labelled with its source, and a free-text match (as opposed to a structured field or the
# bot's own Trials table) is marked UNVERIFIED rather than presented as a number to trust.
# ---------------------------------------------------------------------------
DSR_K_RE = re.compile(r"(?<![A-Za-z0-9])K\s*=\s*([\d][\d,]*)")
FDR_N_RE = re.compile(r"FDR-n[^\d]{0,40}?([\d][\d,]*)(?:\s*\(phase\s*\d+\))?\s*(?:then\s*([\d][\d,]*))?", re.IGNORECASE)
K5_RE = re.compile(r"K5\s*=\s*([\d][\d,]*)", re.IGNORECASE)
# The phase-status table's "Phase 5" row: "| 5 <title> | <status> | <date> | <evidence> |".
# First match wins (the phase-status table is conventionally near the top of BOT.md; later
# "| 5 ... |" lines are usually unrelated table cells that happen to start with the digit).
PHASE5_ROW_RE = re.compile(r"^\|\s*5\b[^|]*\|\s*([^|]+)\|", re.MULTILINE)
TRIALS_HEADING_RE = re.compile(r"^#{1,3}\s*Trials.*$", re.MULTILINE)
NEXT_HEADING_RE = re.compile(r"^#{1,3}\s+\S", re.MULTILINE)
# Deliberately NOT a generic "(\d+) survivor" pattern: this corpus is full of hyphenated
# ordinals immediately before the word ("phase-5 survivor", "generation-3 survivor") whose
# trailing digit a naive `\d+\s*surviv` regex greedily binds to, producing a confidently
# wrong count (e.g. reading "3" out of "generation-3 survivor" when the true count is 0).
# Every one of this fleet's bots closed phase 5 at 0 survivors (see BOT_SPECS docstring /
# each BOT.md Trials table), so the only text-search claim safe enough to surface is the
# literal, unambiguous "0 survivor(s)" - anything else falls back to "not found" rather
# than a fabricated positive number.
ZERO_SURVIVOR_RE = re.compile(r"(?<![\w-])0\s+surviv", re.IGNORECASE)
HOLDOUT_RE = re.compile(r"[Hh]oldout[^\n]{0,20}(\d{4}-\d{2}-\d{2})[^\n]{0,25}(\d{4}-\d{2}-\d{2})")


def last_match_int(pattern: re.Pattern, text: str) -> Optional[int]:
    ms = pattern.findall(text)
    if not ms:
        return None
    try:
        return int(ms[-1].replace(",", ""))
    except (ValueError, AttributeError):
        return None


def extract_fdr_n(text: str) -> Optional[int]:
    """The LAST number in a 'FDR-n ... = 73 (phase 3) then 356 (phase 4)' style clause -
    the cumulative, current count, not the first (superseded) one. None if no such clause."""
    matches = list(FDR_N_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    raw = m.group(2) or m.group(1)
    try:
        return int(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def find_zero_survivors(text: str) -> Optional[int]:
    """0 if the text contains an unambiguous literal '0 survivor(s)', else None (never a
    fabricated positive count from prose - see ZERO_SURVIVOR_RE comment)."""
    return 0 if ZERO_SURVIVOR_RE.search(text) else None


def extract_holdout_snippet(text: str) -> Optional[Tuple[str, str]]:
    m = HOLDOUT_RE.search(text)
    if not m:
        return None
    return m.group(1), m.group(2)


def extract_phase5_status(text: str) -> Optional[str]:
    """The Status column of the phase-status table's Phase 5 row, e.g. btc's
    '**NOT OPENED, generation 1 CLOSED before this phase**' - stripped of markdown bold
    markers, truncated. None if no '| 5 ... | ... |' row is found at all."""
    m = PHASE5_ROW_RE.search(text)
    if not m:
        return None
    return m.group(1).strip().strip("*").strip()[:400]


def extract_trials_table_block(text: str) -> Optional[str]:
    """The last row of the '## Trials counted for the Deflated Sharpe Ratio' table PLUS the
    summary paragraph immediately following it (K5's mention is often in that paragraph, not
    in the table itself - see btc BOT.md's 'Cumulative K = 0 ... K5 = 2,136 ...' sentence).
    None if the bot's BOT.md carries no such heading (only exness_btc_8h does today)."""
    m = TRIALS_HEADING_RE.search(text)
    if not m:
        return None
    tail = text[m.end():]
    nxt = NEXT_HEADING_RE.search(tail)
    block = tail[: nxt.start()] if nxt else tail
    rows = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
    if not rows:
        return block[:4000] or None
    last_row = rows[-1]
    after = block[block.rfind(last_row) + len(last_row):]
    return (last_row + "\n" + after)[:4000]


# ---------------------------------------------------------------------------
# Evidence assembly: layers structured sources (state record, risk_config.yaml, the bot's
# own Trials table) over free-text search, in that priority order, and always records WHICH
# source won AND whether that source is trustworthy enough to show as a plain number
# (`*_unverified = False`) or must be flagged (`*_unverified = True`, rendered struck-through
# with an explicit "unverified" label - see render_evidence_card).
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    # Three SEPARATE counters - see the long comment above DSR_K_RE for why they must never
    # be collapsed into one ambiguous "trial count K".
    dsr_k: Optional[int] = None
    dsr_k_source: str = "not found"
    dsr_k_unverified: bool = False
    fdr_n: Optional[int] = None
    fdr_n_source: str = "not found"
    fdr_n_unverified: bool = False
    k5: Optional[int] = None
    k5_source: str = "not found"
    k5_unverified: bool = False
    phase5_status: Optional[str] = None
    phase5_status_source: str = "not found"
    survivors: Optional[int] = None
    survivors_source: str = "not found"
    specs_scored: Optional[int] = None
    phase6_opened: Optional[bool] = None
    holdout_start: Optional[str] = None
    holdout_end: Optional[str] = None
    holdout_scored: Optional[bool] = None
    holdout_source: str = "not found"
    kill_criteria_approved: Optional[bool] = None
    live_trading_permitted: Optional[bool] = None
    note: Optional[str] = None
    latest_review: Optional[Dict[str, Any]] = None


def build_evidence(
    *, root: Path, spec: BotSpec, risk_cfg: Optional[Dict[str, Any]],
    state_record: Optional[Dict[str, Any]], latest_review: Optional[Dict[str, Any]],
) -> Evidence:
    ev = Evidence(latest_review=latest_review)

    # 1) the latest deploy state/run record's own evidence-shaped fields (usidx nests them
    #    under "evidence"; fx_d1 carries them at top level; btc/gold's records do not carry
    #    them at all, which is itself an honest fact, not a bug).
    sr = state_record or {}
    ev_block = sr.get("evidence") if isinstance(sr.get("evidence"), dict) else sr

    def sr_get(*keys):
        for k in keys:
            if isinstance(ev_block, dict) and k in ev_block and ev_block[k] is not None:
                return ev_block[k]
            if k in sr and sr[k] is not None:
                return sr[k]
        return None

    # trial_count_K in a state record IS the DSR-K for that bot (fx_d1/usidx: the K their
    # phase-5 backtest sweep actually ran at) - a structured field, not free text, so it is
    # trusted (unverified=False).
    k_from_state = sr_get("trial_count_K")
    if isinstance(k_from_state, (int, float)):
        ev.dsr_k = int(k_from_state)
        ev.dsr_k_source = "latest deploy state record"

    surv = sr_get("phase5_survivors")
    if isinstance(surv, (int, float)):
        ev.survivors = int(surv)
        ev.survivors_source = "latest deploy state record"

    specs_scored = sr_get("phase5_specs_scored")
    if isinstance(specs_scored, (int, float)):
        ev.specs_scored = int(specs_scored)
        # "specs scored" is this bot's FDR-n equivalent: the size of the family screened at
        # the multiple-testing tier, structured and trusted, same as DSR-K above.
        ev.fdr_n = int(specs_scored)
        ev.fdr_n_source = "latest deploy state record (specs scored)"

    p6 = sr_get("phase6_opened")
    if isinstance(p6, bool):
        ev.phase6_opened = p6

    hs = sr_get("holdout_scored")
    if isinstance(hs, bool):
        ev.holdout_scored = hs
        ev.holdout_source = "latest deploy state record"
    hw = sr_get("holdout_window")
    if isinstance(hw, (list, tuple)) and len(hw) == 2:
        ev.holdout_start, ev.holdout_end = str(hw[0]), str(hw[1])
        ev.holdout_source = "latest deploy state record"

    kca = sr_get("kill_criteria_approved")
    if isinstance(kca, bool):
        ev.kill_criteria_approved = kca
    ltp = sr_get("live_trading_permitted")
    if isinstance(ltp, bool):
        ev.live_trading_permitted = ltp

    note = sr_get("note") or sr.get("not_deployable_because")
    if isinstance(note, str):
        ev.note = note

    # 2) risk_config.yaml structured fields, only filling gaps the state record left. Also
    #    structured, also trusted.
    rc = risk_cfg or {}
    if ev.dsr_k is None:
        tc = rc.get("phase5_trial_count")
        if tc is None:
            tc = get_in(rc, "model", "trial_count_at_deployment")
        if isinstance(tc, (int, float)):
            ev.dsr_k = int(tc)
            ev.dsr_k_source = "risk_config.yaml"

    if ev.holdout_start is None:
        hs_start = get_in(rc, "evidence_boundary", "holdout_start")
        hs_end = get_in(rc, "evidence_boundary", "holdout_end")
        if hs_start and hs_end:
            ev.holdout_start, ev.holdout_end = str(hs_start), str(hs_end)
            ev.holdout_source = "risk_config.yaml::evidence_boundary"
            marker = get_in(rc, "evidence_boundary", "holdout_scored_marker")
            if marker and ev.holdout_scored is None:
                marker_path = root / "bots" / spec.bot_id / "deploy" / str(marker)
                ev.holdout_scored = marker_path.exists()

    if ev.holdout_start is None:
        retire_status = get_in(rc, "retire", "holdout_status")
        if isinstance(retire_status, str) and retire_status.strip():
            ev.note = (ev.note + " | " if ev.note else "") + f"retire.holdout_status: {retire_status.strip()}"
            ev.holdout_source = "risk_config.yaml::retire.holdout_status (free text, no explicit dates)"

    if ev.kill_criteria_approved is None:
        pua = get_in(rc, "breakers", "pending_user_approval")
        if isinstance(pua, bool):
            ev.kill_criteria_approved = not pua

    # 3) BOT.md's own "Trials counted for the Deflated Sharpe Ratio" table, LAST row (plus
    #    its summary paragraph) - the bot's own authoritative ledger, preferred over any
    #    other free-text search and NOT marked unverified, though it is still a text parse
    #    (there is no machine-readable form of this table).
    botmd_text = safe_read_text(root / "bots" / spec.bot_id / "BOT.md")
    trials_block = extract_trials_table_block(botmd_text) if botmd_text else None
    if trials_block:
        if ev.dsr_k is None:
            k = last_match_int(DSR_K_RE, trials_block)
            if k is not None:
                ev.dsr_k, ev.dsr_k_source = k, "BOT.md Trials table (last row)"
        if ev.fdr_n is None:
            n = extract_fdr_n(trials_block)
            if n is not None:
                ev.fdr_n, ev.fdr_n_source = n, "BOT.md Trials table (last row)"
        if ev.k5 is None:
            k5 = last_match_int(K5_RE, trials_block)
            if k5 is not None:
                ev.k5, ev.k5_source = k5, "BOT.md Trials table (last row / summary paragraph)"

    # 4) BOT.md phase-status table's "Phase 5" row (verbatim status text, not a number).
    if botmd_text:
        status = extract_phase5_status(botmd_text)
        if status:
            ev.phase5_status = status
            ev.phase5_status_source = "BOT.md phase-status table (first '| 5 ...' row)"

    # 5) latest reviews.jsonl entry, free-text fallback - marked UNVERIFIED.
    if latest_review:
        blob = " ".join(str(latest_review.get(f, "")) for f in ("blocking", "notes", "next_step"))
        if ev.dsr_k is None:
            k = last_match_int(DSR_K_RE, blob)
            if k is not None:
                ev.dsr_k, ev.dsr_k_source, ev.dsr_k_unverified = k, "reviews.jsonl (latest entry, free text)", True
        if ev.fdr_n is None:
            n = extract_fdr_n(blob)
            if n is not None:
                ev.fdr_n, ev.fdr_n_source, ev.fdr_n_unverified = n, "reviews.jsonl (latest entry, free text)", True
        if ev.k5 is None:
            k5 = last_match_int(K5_RE, blob)
            if k5 is not None:
                ev.k5, ev.k5_source, ev.k5_unverified = k5, "reviews.jsonl (latest entry, free text)", True
        if ev.survivors is None:
            s = find_zero_survivors(blob)
            if s is not None:
                ev.survivors = s
                ev.survivors_source = "reviews.jsonl (latest entry, literal '0 survivor(s)' match)"

    # 6) BOT.md whole-document free-text search, last resort - marked UNVERIFIED. Preferring
    #    the Trials-table extraction (step 3) over this is exactly BLOCKING finding 6: a
    #    number found by grepping the entire narrative for "K = <n>" can land on any
    #    generation's interim figure, not the bot's own current ledger entry.
    if botmd_text:
        if ev.dsr_k is None:
            k = last_match_int(DSR_K_RE, botmd_text)
            if k is not None:
                ev.dsr_k, ev.dsr_k_source, ev.dsr_k_unverified = k, "BOT.md (free-text search)", True
        if ev.fdr_n is None:
            n = extract_fdr_n(botmd_text)
            if n is not None:
                ev.fdr_n, ev.fdr_n_source, ev.fdr_n_unverified = n, "BOT.md (free-text search)", True
        if ev.k5 is None:
            k5 = last_match_int(K5_RE, botmd_text)
            if k5 is not None:
                ev.k5, ev.k5_source, ev.k5_unverified = k5, "BOT.md (free-text search)", True
        if ev.survivors is None:
            s = find_zero_survivors(botmd_text)
            if s is not None:
                ev.survivors = s
                ev.survivors_source = "BOT.md (literal '0 survivor(s)' match, best-effort)"
        if ev.holdout_start is None:
            snippet = extract_holdout_snippet(botmd_text)
            if snippet:
                ev.holdout_start, ev.holdout_end = snippet
                ev.holdout_source = "BOT.md (text search, best-effort)"

    return ev


# ---------------------------------------------------------------------------
# session_gate.py - imported (never bots/<bot>/deploy/deployment_loop.py)
# ---------------------------------------------------------------------------
def load_session_gate(root: Path):
    path = root / "runners" / "session_gate.py"
    if not path.exists():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("_dashboard_session_gate", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def next_session_gate_instants(gate_mod, gate_key: str, now_utc: datetime, n_weekdays: int = 5) -> List[Dict[str, Any]]:
    """[{date, leg, decision_utc}, ...] for the next `n_weekdays` weekdays (Mon-Fri),
    starting today, one row per leg per weekday that has an instant (venue weekends -> none)."""
    out: List[Dict[str, Any]] = []
    day = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    found_days = 0
    guard = 0
    while found_days < n_weekdays and guard < 30:
        guard += 1
        if day.weekday() < 5:
            found_days += 1
            for leg, (session, anchor, offset) in gate_mod.LEGS[gate_key].items():
                window = gate_mod.SESSIONS[session]
                inst = gate_mod.decision_utc(window, day, anchor, offset)
                if inst is not None:
                    out.append({"date": day.date().isoformat(), "leg": leg, "decision_utc": inst})
        day += timedelta(days=1)
    return out


def next_fixed_instants(spec: BotSpec, now_utc: datetime, count: int = 10) -> List[Dict[str, Any]]:
    """The next `count` chronological fixed-UTC-hour decision instants for btc/fx_d1."""
    out: List[Dict[str, Any]] = []
    probe = now_utc.replace(minute=0, second=0, microsecond=0)
    guard = 0
    while len(out) < count and guard < 24 * 10:
        guard += 1
        probe += timedelta(hours=1)
        if probe.hour not in spec.decision_hours_utc:
            continue
        if spec.decision_weekday_only and probe.weekday() >= 5:
            continue
        out.append({"date": probe.date().isoformat(), "leg": None, "decision_utc": probe})
    return out


# ---------------------------------------------------------------------------
# Task Scheduler (PowerShell, read-only)
# ---------------------------------------------------------------------------
_TASK_PS_TEMPLATE = r"""
$names = @({names})
$out = foreach ($n in $names) {{
    $found = $false; $state = $null; $lastRun = $null; $lastResult = $null; $nextRun = $null
    try {{
        $t = Get-ScheduledTask -TaskName $n -ErrorAction Stop
        $found = $true
        $state = [string]$t.State
        $i = Get-ScheduledTaskInfo -TaskName $n -ErrorAction SilentlyContinue
        if ($i) {{
            if ($i.LastRunTime -and $i.LastRunTime.Year -gt 1601) {{ $lastRun = $i.LastRunTime.ToString('o') }}
            $lastResult = $i.LastTaskResult
            if ($i.NextRunTime -and $i.NextRunTime.Year -gt 1601) {{ $nextRun = $i.NextRunTime.ToString('o') }}
        }}
    }} catch {{ }}
    [PSCustomObject]@{{ Name=$n; Found=$found; State=$state; LastRunTime=$lastRun; LastTaskResult=$lastResult; NextRunTime=$nextRun }}
}}
@($out) | ConvertTo-Json -Compress
"""


def parse_task_scheduler_json(raw: str) -> Dict[str, Dict[str, Any]]:
    """Pure parser for the JSON `_TASK_PS_TEMPLATE` prints - factored out of
    read_task_scheduler() so tests can feed it synthetic PowerShell output without
    shelling out. Raises ValueError on unparseable input; callers catch it."""
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        parsed = [parsed]
    result: Dict[str, Dict[str, Any]] = {}
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict) or "Name" not in item:
            continue
        result[item["Name"]] = {
            "found": bool(item.get("Found")),
            "state": item.get("State"),
            "last_run_time": item.get("LastRunTime"),
            "last_task_result": item.get("LastTaskResult"),
            "next_run_time": item.get("NextRunTime"),
        }
    return result


def read_task_scheduler(task_names: Sequence[str], timeout: float = 15.0) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """{task_name: {found, state, last_run_time, last_task_result, next_run_time}}."""
    if sys.platform != "win32":
        return {}, "not running on Windows; Task Scheduler is unavailable"
    ps_names = ", ".join(f"'{n}'" for n in task_names)
    script = _TASK_PS_TEMPLATE.format(names=ps_names)
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stderr=subprocess.DEVNULL, text=True, timeout=timeout,
        )
    except Exception as e:
        return {}, f"PowerShell query failed: {type(e).__name__}: {e}"
    try:
        result = parse_task_scheduler_json(out)
    except Exception as e:
        return {}, f"could not parse Task Scheduler output: {type(e).__name__}: {e}"
    return result, None


# ---------------------------------------------------------------------------
# Fleet-level: account halt / high-water mark
# ---------------------------------------------------------------------------
def read_account_halt(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"halted": False, "data": None, "error": None}
    data, err = safe_read_json(p)
    return {"halted": True, "data": data, "error": err}


def read_account_hwm(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return safe_read_json(Path(path))


# ---------------------------------------------------------------------------
# MT5, read-only, never-launch (mirrors v9_dashboard's design exactly - see its README)
# ---------------------------------------------------------------------------
def list_terminal_executable_paths(timeout: float = 5.0) -> Optional[List[str]]:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='terminal64.exe'\" | "
                "Select-Object -ExpandProperty ExecutablePath",
            ],
            stderr=subprocess.DEVNULL, text=True, timeout=timeout,
        )
    except Exception:
        return None
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def resolve_single_terminal_path(explicit_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(confirmed_running_path, refusal_reason) - mt5.initialize() must only ever be called
    with a path THIS function proved is already running (BLOCKING finding 4: pass the
    proven-running path explicitly, never rely on "no path = attach to whatever is
    running", which cannot tell two terminals apart and could attach to the wrong one).

    - `explicit_path` given (--mt5-path): that exact path, only if it is one of the
      currently running terminal64.exe processes.
    - `explicit_path` absent: the one path, IFF exactly one terminal64.exe process is
      running at all. 0 or more than 1 is refused - this dashboard does not guess which
      terminal a bot would have used."""
    paths = list_terminal_executable_paths()
    if paths is None:
        return None, "cannot determine whether a terminal is running (process listing unavailable); not launching one for safety"
    if explicit_path:
        target = os.path.normcase(os.path.normpath(str(explicit_path)))
        if any(os.path.normcase(os.path.normpath(p)) == target for p in paths):
            return explicit_path, None
        return None, f"--mt5-path is not among the {len(paths)} running terminal64.exe process(es); not launching it"
    if len(paths) == 1:
        return paths[0], None
    return None, f"n/a: 0 or >1 terminals running (found {len(paths)}); not launching or guessing which one"


def _empty_mt5_snap(error: str, terminal_running: Optional[bool] = None) -> Dict[str, Any]:
    return {
        "reachable": False, "error": error, "account": None, "positions": [], "terminal": None,
        "terminal_running": terminal_running, "login_mismatch": None, "server_mismatch": None,
    }


def read_mt5_snapshot(
    mt5_path: Optional[str], *, expected_login: Optional[int] = None, expected_server: Optional[str] = None,
) -> Dict[str, Any]:
    if mt5 is None:
        return _empty_mt5_snap("MetaTrader5 package not importable", terminal_running=None)
    resolved_path, refuse_reason = resolve_single_terminal_path(mt5_path)
    if resolved_path is None:
        return _empty_mt5_snap(refuse_reason, terminal_running=None)
    try:
        ok = mt5.initialize(path=resolved_path)
    except Exception as e:
        return _empty_mt5_snap(f"mt5.initialize raised {type(e).__name__}: {e}", terminal_running=True)
    if not ok:
        try:
            err = mt5.last_error()
        except Exception:
            err = None
        return _empty_mt5_snap(f"mt5.initialize failed: {err}", terminal_running=True)
    try:
        acc = None
        try:
            info = mt5.account_info()
        except Exception:
            info = None
        if info is not None:
            acc = {
                "login": getattr(info, "login", None), "server": getattr(info, "server", None),
                "trade_mode": getattr(info, "trade_mode", None), "balance": getattr(info, "balance", None),
                "equity": getattr(info, "equity", None), "margin": getattr(info, "margin", None),
                "margin_free": getattr(info, "margin_free", None),
            }
        term = None
        try:
            tinfo = mt5.terminal_info()
        except Exception:
            tinfo = None
        if tinfo is not None:
            term = {
                "connected": getattr(tinfo, "connected", None), "trade_allowed": getattr(tinfo, "trade_allowed", None),
                "ping_last": getattr(tinfo, "ping_last", None), "build": getattr(tinfo, "build", None),
            }
        try:
            raw_positions = mt5.positions_get() or []
        except Exception:
            raw_positions = []
        # NO 'profit' field (BLOCKING finding 4): this dashboard never surfaces a per-position
        # or aggregate PnL figure, by design - see module docstring and dashboard/README.md.
        positions = [{
            "ticket": getattr(p, "ticket", None), "symbol": getattr(p, "symbol", None),
            "type": getattr(p, "type", None), "volume": getattr(p, "volume", None),
            "price_open": getattr(p, "price_open", None), "price_current": getattr(p, "price_current", None),
            "swap": getattr(p, "swap", None), "magic": getattr(p, "magic", None),
        } for p in raw_positions]
        login_mismatch = bool(acc and expected_login is not None and acc.get("login") != expected_login)
        server_mismatch = bool(acc and expected_server is not None and acc.get("server") != expected_server)
        return {
            "reachable": True, "error": None, "account": acc, "positions": positions, "terminal": term,
            "terminal_running": True, "resolved_path": resolved_path,
            "login_mismatch": login_mismatch, "server_mismatch": server_mismatch,
            "expected_login": expected_login, "expected_server": expected_server,
        }
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def read_mt5_snapshot_with_timeout(
    mt5_path: Optional[str], timeout: float, *, expected_login: Optional[int] = None, expected_server: Optional[str] = None,
) -> Dict[str, Any]:
    box: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["value"] = read_mt5_snapshot(mt5_path, expected_login=expected_login, expected_server=expected_server)
        except Exception as e:
            box["value"] = _empty_mt5_snap(f"mt5 read raised {type(e).__name__}: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return _empty_mt5_snap(f"terminal unresponsive (no response within {timeout:g}s)")
    return box.get("value") or _empty_mt5_snap("mt5 read produced no result")


# ---------------------------------------------------------------------------
# Circuit breaker names from monitor/circuit_breakers.py (TEXT, regex only, never imported)
# ---------------------------------------------------------------------------
BREAKER_CLASS_RE = re.compile(r"^class (\w+)\(CircuitBreaker\)", re.MULTILINE)


def read_breaker_class_names(root: Path, spec: BotSpec) -> List[str]:
    if not spec.has_monitor_dir:
        return []
    text = safe_read_text(root / "bots" / spec.bot_id / "monitor" / "circuit_breakers.py")
    if not text:
        return []
    return BREAKER_CLASS_RE.findall(text)


# ---------------------------------------------------------------------------
# Traffic light
# ---------------------------------------------------------------------------
def traffic_light(
    *, window_exit_codes: Sequence[Optional[int]], last_skipped: bool, log_error: Optional[str],
    task_info: Optional[Dict[str, Any]], fleet_halted: bool, fleet_foreign_magic: bool,
    armed: Optional[bool] = None, execution_modes: Sequence[Optional[str]] = (),
    live_trading_permitted: Optional[bool] = None, holdout_scored: Optional[bool] = None,
) -> Tuple[str, List[str]]:
    """`window_exit_codes` is every exit code currently shown in the run-history table
    (most-recent-first), not just the latest one - BLOCKING finding 5: a bot whose most
    recent run happened to exit 0 but whose four runs before that all errored must not read
    as green. `last_skipped` still only describes the single latest line (a Skipped run is
    routine and never on its own worse than amber - see README)."""
    reasons: List[str] = []
    task_found = bool(task_info and task_info.get("found"))
    task_state = task_info.get("state") if task_info else None
    codes = [c for c in window_exit_codes if c is not None]
    bad_codes = sorted({c for c in codes if c not in (0, 1, 10)})
    ones = [c for c in codes if c == 1]

    if fleet_halted:
        reasons.append("account_halt.json is present (fleet-wide halt)")
    if fleet_foreign_magic:
        reasons.append("a position on the account carries a magic not in MAGIC_ALLOCATION or RESERVED_MAGICS")
    if not task_found or (task_state and task_state.lower() in ("disabled",)):
        reasons.append(f"scheduled task missing or disabled (state={task_state!r}, found={task_found})")
    if bad_codes:
        reasons.append(f"exit code(s) {bad_codes} in the displayed run history are not in {{0, 1, 10}}")
    if armed is True:
        reasons.append("execution.armed is True")
    bad_modes = sorted({m for m in execution_modes if m is not None and str(m).lower() not in ("paper", "shadow")})
    if bad_modes:
        reasons.append(f"execution mode {bad_modes} is not in {{paper, shadow}}")
    if live_trading_permitted is True and holdout_scored is not True:
        reasons.append("live_trading_permitted is True while the holdout is not (yet) scored")
    if reasons:
        return "red", reasons

    if ones:
        reasons.append(f"{len(ones)} run(s) in the displayed run history exited 1 (error/NotReady)")
    if last_skipped:
        reasons.append("the latest log line is a Skipped (not-a-decision-hour) run")
    if reasons:
        return "amber", reasons

    if codes and codes[0] == 0 and task_state == "Ready":
        return "green", ["every run in the displayed history exited 0 or 10, the latest exited 0, and the scheduled task is Ready"]

    if not codes:
        reasons.append(f"no run history: {log_error or 'no log records'}")
        return "amber", reasons

    reasons.append(f"exit codes {codes}, task state {task_state!r} - not a clear green")
    return "amber", reasons


# ---------------------------------------------------------------------------
# View model assembly (mostly-pure: takes already-read inputs, does no I/O itself except
# the two file reads that are cheap and bot-scoped: risk_config.yaml re-parse and BOT.md,
# both inside build_evidence())
# ---------------------------------------------------------------------------
def collect_bot(
    *, root: Path, spec: BotSpec, now_utc: datetime, reviews_by_bot: Dict[str, Dict[str, Any]],
    task_info_by_name: Dict[str, Dict[str, Any]], gate_mod, fleet_halted: bool, fleet_foreign_magic: bool,
    log_tail: int,
) -> Dict[str, Any]:
    deploy_dir = root / "bots" / spec.bot_id / "deploy"
    risk_cfg_path = deploy_dir / "risk_config.yaml"
    risk_cfg, risk_cfg_err = safe_read_yaml(risk_cfg_path)

    state_path, state_record, state_err = find_latest_state_record(root, spec)

    log_path = root / "logs" / spec.log_name
    log_records, log_err = read_log_records(log_path, limit=log_tail)
    last_record = log_records[0] if log_records else None

    latest_review = reviews_by_bot.get(spec.bot_id)

    evidence = build_evidence(root=root, spec=spec, risk_cfg=risk_cfg, state_record=state_record, latest_review=latest_review)

    exec_mode = extract_exec_mode(risk_cfg)

    if spec.decision_kind == "session_gate" and gate_mod is not None:
        try:
            next_instants = next_session_gate_instants(gate_mod, spec.gate_key, now_utc)
        except Exception:
            next_instants = []
    elif spec.decision_kind == "session_gate":
        next_instants = []
    else:
        next_instants = next_fixed_instants(spec, now_utc)

    task_info = task_info_by_name.get(spec.task_name)

    light, light_reasons = traffic_light(
        window_exit_codes=[r.exit_code for r in log_records],
        last_skipped=last_record.skipped if last_record else False,
        log_error=log_err,
        task_info=task_info,
        fleet_halted=fleet_halted,
        fleet_foreign_magic=fleet_foreign_magic,
        armed=exec_mode.get("armed"),
        execution_modes=[exec_mode.get("execution_mode_broker"), exec_mode.get("execution_mode_live_risk")],
        live_trading_permitted=evidence.live_trading_permitted,
        holdout_scored=evidence.holdout_scored,
    )

    breakers_cfg = get_in(risk_cfg, "breakers", default={}) if isinstance(risk_cfg, dict) else {}
    breaker_classes = read_breaker_class_names(root, spec)

    return {
        "spec": spec,
        "risk_cfg": risk_cfg,
        "risk_cfg_err": risk_cfg_err,
        "state_path": state_path,
        "state_record": state_record,
        "state_err": state_err,
        "log_records": log_records,
        "log_err": log_err,
        "last_record": last_record,
        "latest_review": latest_review,
        "evidence": evidence,
        "exec_mode": exec_mode,
        "next_instants": next_instants,
        "task_info": task_info,
        "traffic_light": light,
        "traffic_light_reasons": light_reasons,
        "breakers_cfg": breakers_cfg if isinstance(breakers_cfg, dict) else {},
        "breaker_classes": breaker_classes,
    }


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
def _esc(v: Any) -> str:
    if v is None:
        return "n/a"
    return escape(str(v))


def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return _esc(v)


def _fmt_dt_utc_vn(dt: Optional[datetime]) -> str:
    if dt is None:
        return "n/a"
    vn = dt + VN_OFFSET
    return f"{dt.strftime('%Y-%m-%d %H:%M')} UTC / {vn.strftime('%Y-%m-%d %H:%M')} VN"


def _parse_iso_maybe(s: Any) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _badge(text: str, color: str = "") -> str:
    cls = f"badge badge-{color}" if color else "badge"
    return f'<span class="{cls}">{escape(str(text))}</span>'


def _bool_badge(v: Any, *, true_is_good: bool = True) -> str:
    if v is True:
        return _badge("true", "green" if true_is_good else "red")
    if v is False:
        return _badge("false", "red" if true_is_good else "green")
    return _badge("n/a")


def _light_badge(light: str) -> str:
    label = {"green": "XANH / GREEN", "amber": "VÀNG / AMBER", "red": "ĐỎ / RED"}.get(light, light.upper())
    return _badge(label, light)


def verdict_badge(verdict: Optional[str]) -> str:
    """A reviews.jsonl 'verdict' rendered so that 'met' can NEVER be read as a pass on the
    underlying hypothesis: 'met' means the *process gate* (numbers verified, procedure
    followed) passed, not that an edge was found - every bot in this fleet closed Phase 5
    with 0 survivors regardless of how many of its process gates were 'met'. Neutral (no
    color) badge, explicit label - never green, for any verdict value."""
    if verdict is None:
        return _badge("n/a")
    if verdict == "met":
        return _badge("phase gate met (process), not evidence of an edge")  # neutral
    if verdict == "not_evidence_yet":
        return _badge("not_evidence_yet", "amber")
    return _badge(str(verdict))  # neutral for "info" and anything else - never assume green


def _unverified_cell(value: Optional[int], source: str, unverified: bool) -> str:
    """A best-effort text-search number, struck through and explicitly labelled
    'unverified' (BLOCKING finding 6) so it cannot be read as a value to trust the way a
    structured-source number can. Structured/Trials-table numbers render as plain text."""
    if value is None:
        return f'n/a <span class="muted">({_esc(source)})</span>'
    if unverified:
        return (
            f'<s class="unverified-value">{_esc(value)}</s> {_badge("unverified", "amber")} '
            f'<span class="muted">({_esc(source)} - verify against the BOT.md Trials table)</span>'
        )
    return f'{_esc(value)} <span class="muted">({_esc(source)})</span>'


def render_evidence_card(view: Dict[str, Any]) -> str:
    ev: Evidence = view["evidence"]
    review = view["latest_review"]
    rows = []
    rows.append(("Phase 5 status (BOT.md phase table)", _esc(ev.phase5_status) if ev.phase5_status else f'n/a <span class="muted">({_esc(ev.phase5_status_source)})</span>'))
    rows.append(("DSR-K (Sharpe-based selection count)", _unverified_cell(ev.dsr_k, ev.dsr_k_source, ev.dsr_k_unverified)))
    rows.append(("FDR-n (screening family size)", _unverified_cell(ev.fdr_n, ev.fdr_n_source, ev.fdr_n_unverified)))
    rows.append(("K5 (declared, later-stage, undrawn)", _unverified_cell(ev.k5, ev.k5_source, ev.k5_unverified)))
    rows.append(("Survivors", f'{_esc(ev.survivors)} <span class="muted">({_esc(ev.survivors_source)})</span>'))
    if ev.specs_scored is not None:
        rows.append(("Specs scored", _esc(ev.specs_scored)))
    if ev.phase6_opened is not None:
        rows.append(("Phase 6 opened", _bool_badge(ev.phase6_opened, true_is_good=False)))
    holdout_cell = "n/a"
    if ev.holdout_start and ev.holdout_end:
        scored_txt = _bool_badge(ev.holdout_scored, true_is_good=False) if ev.holdout_scored is not None else _badge("unknown")
        holdout_cell = f'{_esc(ev.holdout_start)} &rarr; {_esc(ev.holdout_end)} &middot; sealed/scored: {scored_txt} <span class="muted">({_esc(ev.holdout_source)})</span>'
    rows.append(("Holdout window", holdout_cell))
    if ev.kill_criteria_approved is not None:
        rows.append(("Kill criteria approved", _bool_badge(ev.kill_criteria_approved)))
    if ev.live_trading_permitted is not None:
        rows.append(("Live trading permitted", _bool_badge(ev.live_trading_permitted)))
    if ev.note:
        rows.append(("Ghi chú (note)", _esc(ev.note)[:400]))

    if review:
        ts = _esc(review.get("ts"))
        agent = _esc(review.get("agent"))
        phase = _esc(review.get("phase"))
        blocking = review.get("blocking") or []
        blocking_txt = "; ".join(str(b) for b in blocking)[:400] if blocking else ""
        review_html = (
            f'<div><b>{ts}</b> &middot; agent {agent} &middot; phase {phase} &middot; verdict {verdict_badge(review.get("verdict"))}</div>'
        )
        if blocking_txt:
            review_html += f'<div class="muted">blocking: {escape(blocking_txt)}</div>'
    else:
        review_html = '<div class="muted">n/a &mdash; không có review nào cho bot này trong reviews.jsonl (0 entries)</div>'

    table = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return (
        '<h3>Bằng chứng (Evidence status)</h3>'
        f'<table>{table}</table>'
        '<h4>Latest mentor verdict (reviews.jsonl)</h4>'
        f'{review_html}'
    )


def render_execution_mode_card(view: Dict[str, Any]) -> str:
    em = view["exec_mode"]
    rows = [
        ("shadow_mode", _bool_badge(em["shadow_mode"], true_is_good=True) if em["shadow_mode"] is not None else _badge("n/a")),
        ("execution.mode (broker layer)", _esc(em["execution_mode_broker"])),
        ("execution.armed", _bool_badge(em["armed"], true_is_good=False) if em["armed"] is not None else _badge("n/a")),
        ("live_risk_config.execution_mode", _esc(em["execution_mode_live_risk"])),
        (
            "breakers.pending_user_approval",
            (_badge("draft, pending approval", "amber") if em["pending_user_approval"] is True
             else (_bool_badge(em["pending_user_approval"], true_is_good=False) if em["pending_user_approval"] is not None else _badge("n/a - not declared for this bot"))),
        ),
    ]
    table = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    err_note = f'<div class="muted">risk_config.yaml: {_esc(view["risk_cfg_err"])}</div>' if view["risk_cfg_err"] else ""
    return f'<h3>Chế độ thực thi (Execution mode)</h3>{err_note}<table>{table}</table>'


def render_last_run_card(view: Dict[str, Any]) -> str:
    sr = view["state_record"]
    if sr is None:
        body = f'<div class="muted">n/a &mdash; {_esc(view["state_err"])}</div>'
    else:
        rows = []
        ts = sr.get("finished_at_utc") or sr.get("run_finished_at") or sr.get("finished_at") or sr.get("timestamp")
        rows.append(("Thời điểm (finished at)", _esc(ts)))
        decision_date = extract_decision_date(sr)
        rows.append(("Decision date (record's own)", _esc(decision_date) if decision_date else 'n/a <span class="muted">(not present in this record)</span>'))
        for key in ("status", "action", "signal_status", "deployable", "not_ready", "stopped_after", "stopped_at"):
            if key in sr and sr[key] is not None:
                val = sr[key]
                if isinstance(val, str) and len(val) > 300:
                    val = val[:300] + "…"
                rows.append((key, _esc(val)))
        if not rows:
            rows.append(("raw keys", _esc(", ".join(sr.keys()))))
        body = "<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"
        body += f'<div class="muted">nguồn: {_esc(view["state_path"])}</div>'
    return f'<h3>Lần chạy gần nhất (Last run, state record)</h3>{body}'


def render_run_history_table(view: Dict[str, Any]) -> str:
    records: List[LogRecord] = view["log_records"]
    if not records:
        return f'<h3>Lịch sử chạy (Run history)</h3><div class="muted">n/a &mdash; {_esc(view["log_err"])}</div>'
    rows = []
    for r in records:
        row_class = ' class="muted-row"' if r.skipped else ""
        exit_badge = (
            _badge(str(r.exit_code), "green") if r.exit_code == 0
            else _badge(str(r.exit_code), "amber") if r.exit_code in (1, 10)
            else _badge(str(r.exit_code), "red") if r.exit_code is not None else _badge("n/a")
        )
        rows.append(
            f"<tr{row_class}><td>{_esc(r.ts_raw)}</td><td>{_esc(r.leg or '-')}</td>"
            f"<td>{exit_badge}</td><td>{_esc(r.reason)}</td></tr>"
        )
    table = (
        '<table><tr><th style="width:auto">Thời gian (machine clock)</th><th style="width:auto">Leg/Spec</th>'
        '<th style="width:auto">Exit</th><th style="width:auto">Lý do (one-line reason)</th></tr>'
        + "".join(rows) + "</table>"
    )
    return f'<h3>Lịch sử chạy (Run history, {len(records)} lần gần nhất)</h3>{table}'


def render_next_instants(view: Dict[str, Any]) -> str:
    instants = view["next_instants"]
    if not instants:
        return '<h3>Giờ quyết định kế tiếp (Next decision instants)</h3><div class="muted">n/a</div>'
    rows = []
    for it in instants[:20]:
        leg = it.get("leg") or "-"
        rows.append(f"<tr><td>{_esc(it['date'])}</td><td>{_esc(leg)}</td><td>{_fmt_dt_utc_vn(it['decision_utc'])}</td></tr>")
    table = (
        '<table><tr><th style="width:auto">Ngày</th><th style="width:auto">Leg/Spec</th>'
        '<th style="width:auto">Giờ quyết định (UTC / VN)</th></tr>' + "".join(rows) + "</table>"
    )
    return f'<h3>Giờ quyết định kế tiếp (Next decision instants)</h3>{table}'


def render_task_scheduler(view: Dict[str, Any], task_err: Optional[str]) -> str:
    info = view["task_info"]
    spec: BotSpec = view["spec"]
    if task_err and not info:
        return f'<h3>Task Scheduler ({_esc(spec.task_name)})</h3><div class="muted">n/a &mdash; {_esc(task_err)}</div>'
    if not info:
        return f'<h3>Task Scheduler ({_esc(spec.task_name)})</h3><div class="muted">n/a &mdash; task not found</div>'
    if not info.get("found"):
        return f'<h3>Task Scheduler ({_esc(spec.task_name)})</h3><div class="muted">{_badge("missing", "red")} task not registered</div>'
    last_run = _parse_iso_maybe(info.get("last_run_time"))
    next_run = _parse_iso_maybe(info.get("next_run_time"))
    code = info.get("last_task_result")
    code_label = TASK_RESULT_LABELS.get(code, f"code {code}") if code is not None else "n/a"
    # Task Scheduler's "never run" sentinel LastRunTime is not always .NET's DateTime.MinValue
    # (year 1601); on this host it has been observed as a 1999 placeholder. LastTaskResult
    # 267011 (0x41303, "task has not yet run") is the reliable signal - if it is set, the
    # last-run timestamp (whatever placeholder value it holds) is not shown as if it were real.
    has_not_run = code == 267011
    state = info.get("state")
    state_badge = _badge(state, "green") if state == "Ready" else _badge(state, "amber" if state else "")
    last_run_cell = (
        "chưa từng chạy qua Task Scheduler (has not run yet)" if has_not_run
        else (_fmt_dt_utc_vn(last_run) if last_run else "n/a")
    )
    rows = [
        ("State", state_badge),
        ("Last run", last_run_cell),
        ("Last result", _esc(code_label)),
        ("Next run", _fmt_dt_utc_vn(next_run) if next_run else "n/a"),
    ]
    table = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<h3>Task Scheduler ({_esc(spec.task_name)})</h3><table>{table}</table>'


def render_breakers(view: Dict[str, Any]) -> str:
    cfg = view["breakers_cfg"]
    classes = view["breaker_classes"]
    pending = cfg.get("pending_user_approval") if isinstance(cfg, dict) else None
    header = ""
    if pending is True:
        header = f'<div class="banner banner-amber">{_badge("draft, pending approval", "amber")} — kill criteria have NOT been approved by the user for this bot</div>'
    if not cfg:
        body = '<div class="muted">n/a &mdash; no breakers: block in risk_config.yaml</div>'
    else:
        rows = []
        for name, val in cfg.items():
            if name in ("pending_user_approval", "recovery_timeout_hours"):
                continue
            if isinstance(val, dict):
                summary = "; ".join(f"{k}={v}" for k, v in val.items())
            else:
                summary = str(val)
            rows.append(f"<tr><td>{_esc(name)}</td><td>{_esc(summary)[:200]}</td></tr>")
        body = (
            '<table><tr><th style="width:auto">Breaker</th><th style="width:auto">Threshold (risk_config.yaml)</th></tr>'
            + "".join(rows) + "</table>"
        )
    classes_line = f'<div class="muted">Implemented breaker classes (monitor/circuit_breakers.py): {_esc(", ".join(classes)) if classes else "n/a (no monitor/ dir for this bot; breakers embedded directly in risk_config.yaml)"}</div>'
    return f'<h3>Circuit breakers</h3>{header}{body}{classes_line}'


def render_bot_card(view: Dict[str, Any]) -> str:
    spec: BotSpec = view["spec"]
    light = view["traffic_light"]
    reasons = view["traffic_light_reasons"]
    identity_rows = [
        ("Symbols", ", ".join(spec.symbols)),
        ("Magic", str(spec.magic)),
        ("Cadence", spec.cadence_vi),
    ]
    identity_table = "".join(f"<tr><th>{k}</th><td>{_esc(v)}</td></tr>" for k, v in identity_rows)
    reasons_html = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
    return f"""
<div class="card full bot-card bot-card-{light}">
  <div class="bot-header">
    <h2>{_esc(spec.label_vi)} &mdash; {_esc(spec.bot_id)}</h2>
    {_light_badge(light)}
  </div>
  <details class="light-reasons"><summary>Vì sao đèn này? (why this light)</summary><ul>{reasons_html}</ul></details>
  <div class="grid">
    <div class="subcard"><h3>Identity</h3><table>{identity_table}</table></div>
    <div class="subcard">{render_execution_mode_card(view)}</div>
    <div class="subcard">{render_evidence_card(view)}</div>
    <div class="subcard">{render_last_run_card(view)}</div>
    <div class="subcard full">{render_run_history_table(view)}</div>
    <div class="subcard">{render_next_instants(view)}</div>
    <div class="subcard">{render_task_scheduler(view, view.get("task_err"))}</div>
    <div class="subcard full">{render_breakers(view)}</div>
  </div>
</div>
"""


def render_fleet_header(
    *, now_utc: datetime, halt: Dict[str, Any], hwm: Optional[Dict[str, Any]], hwm_err: Optional[str],
    mt5_snap: Dict[str, Any], magic_allocation: Dict[str, int], reserved_magics: Dict[int, str],
    magic_source_err: Optional[str],
) -> Tuple[str, bool]:
    """Returns (html, foreign_magic_present)."""
    now_vn = now_utc + VN_OFFSET
    banners = []
    if halt["halted"]:
        reason = get_in(halt["data"], "reason") if halt["data"] else None
        halted_at = get_in(halt["data"], "halted_at") if halt["data"] else None
        extra = f' reason: {_esc(reason)}, halted_at: {_esc(halted_at)}' if reason or halted_at else (f' ({_esc(halt["error"])})' if halt["error"] else " (file present, empty/unreadable)")
        banners.append(f'<div class="banner banner-red">DỪNG KHẨN CẤP (ACCOUNT HALT) ĐANG BẬT{extra}</div>')
    else:
        banners.append('<div class="banner banner-green">Account halt: KHÔNG có (no halt file)</div>')

    hwm_line = "n/a"
    if hwm:
        hwm_line = f'peak_equity {_fmt_num(hwm.get("peak_equity"))} &middot; updated_at {_esc(hwm.get("updated_at"))}'
    elif hwm_err:
        hwm_line = f'n/a ({_esc(hwm_err)})'

    reachable = mt5_snap.get("reachable")
    foreign_present = False
    account_mismatch = False
    mt5_html = ""
    foreign_magic_banner = ""
    if not mt5_snap.get("attempted", True):
        mt5_html = '<div class="muted">MT5: bị tắt (--no-mt5 mặc định). Dùng --mt5 để bật (chỉ đọc).</div>'
        # BLOCKING finding 4: state this on the page itself, not only in README.md - an
        # operator reading only the page must not assume a foreign position would be caught.
        foreign_magic_banner = (
            '<div class="banner banner-amber">Bộ phát hiện magic lạ (foreign-magic detector): '
            'ĐANG TẮT &mdash; cần chạy lại với <code>--mt5</code> để đọc vị thế MT5 thật</div>'
        )
    elif not reachable:
        mt5_html = f'<div class="muted">MT5: n/a &mdash; {_esc(mt5_snap.get("error"))}</div>'
        foreign_magic_banner = (
            '<div class="banner banner-amber">Bộ phát hiện magic lạ (foreign-magic detector): '
            'KHÔNG chạy được (MT5 không đọc được) &mdash; xem lý do ở trên</div>'
        )
    else:
        acc = mt5_snap.get("account") or {}
        login_mismatch = bool(mt5_snap.get("login_mismatch"))
        server_mismatch = bool(mt5_snap.get("server_mismatch"))
        account_mismatch = login_mismatch or server_mismatch
        login_cell = _esc(acc.get("login"))
        if login_mismatch:
            expected_login_txt = "expected " + str(mt5_snap.get("expected_login"))
            login_cell += " " + _badge(expected_login_txt, "red")
        server_cell = _esc(acc.get("server"))
        if server_mismatch:
            expected_server_txt = "expected " + str(mt5_snap.get("expected_server"))
            server_cell += " " + _badge(expected_server_txt, "red")
        acc_rows = [
            ("Login", login_cell), ("Server", server_cell),
            ("Trade mode", _esc(acc.get("trade_mode"))), ("Equity", _fmt_num(acc.get("equity"))),
            ("Margin free", _fmt_num(acc.get("margin_free"))),
        ]
        acc_table = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in acc_rows)
        by_magic: Dict[Any, List[Dict[str, Any]]] = {}
        for p in mt5_snap.get("positions", []):
            by_magic.setdefault(p.get("magic"), []).append(p)
        pos_rows = []
        for magic, plist in sorted(by_magic.items(), key=lambda kv: (kv[0] is None, kv[0])):
            if magic in reserved_magics:
                label = f"{reserved_magics[magic]} (magic {magic})"
                cls = ""
            else:
                owner = next((bid for bid, m in magic_allocation.items() if m == magic), None)
                if owner:
                    label = f"{owner} (magic {magic})"
                    cls = ""
                else:
                    label = f"UNKNOWN/FOREIGN magic {magic}"
                    cls = ' class="foreign-magic"'
                    foreign_present = True
            pos_rows.append(f'<tr{cls}><td>{_esc(label)}</td><td>{len(plist)}</td></tr>')
        pos_table = '<table><tr><th style="width:auto">Magic (chủ sở hữu)</th><th style="width:auto">Số vị thế</th></tr>' + "".join(pos_rows or ['<tr><td colspan="2" class="muted">none</td></tr>']) + "</table>"
        mt5_html = f'<table>{acc_table}</table>{pos_table}'

    if account_mismatch:
        banners.append('<div class="banner banner-red">MT5 ACCOUNT MISMATCH &mdash; login/server khác với --expected-login/--expected-server</div>')

    banners_html = "".join(banners) + foreign_magic_banner
    if magic_source_err:
        banners_html += f'<div class="banner banner-amber">MAGIC_ALLOCATION: {_esc(magic_source_err)} (dùng bảng dự phòng, chỉ đọc)</div>'

    html = f"""
<div class="header">
  <h1>Bảng điều khiển vận hành 4 Bot Exness &mdash; Fleet Operator Dashboard</h1>
  <div class="muted">
    UTC {now_utc.strftime('%Y-%m-%d %H:%M:%S')} &middot; VN {now_vn.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7) &middot;
    Chỉ đọc (read-only) &mdash; không đặt lệnh, không khởi chạy MT5 terminal
  </div>
</div>
{banners_html}
<div class="grid">
  <div class="card"><h2>Đỉnh vốn (Account high-water mark)</h2><div>{hwm_line}</div></div>
  <div class="card"><h2>MT5 snapshot</h2>{mt5_html}</div>
</div>
"""
    return html, foreign_present


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{interval}">
<title>Exness 4-Bot Fleet Dashboard</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #f5f6f8; --fg: #14181f; --card: #ffffff; --border: #d8dde3;
    --muted: #6b7280; --green: #16794a; --green-bg: #e3f6ec;
    --amber: #92620a; --amber-bg: #fdf1d6; --red: #a3181f; --red-bg: #fbe1e2;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f1216; --fg: #e7eaee; --card: #171b21; --border: #2a2f37;
      --muted: #9aa3ae; --green: #4ade80; --green-bg: #10331f;
      --amber: #f5b544; --amber-bg: #3a2b0a; --red: #f87171; --red-bg: #3a1315;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-variant-numeric: tabular-nums; font-size: 14px; line-height: 1.5;
  }}
  h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
  h2 {{ font-size: 15px; margin: 0 0 8px 0; }}
  h3 {{ font-size: 13px; margin: 10px 0 6px 0; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); }}
  h4 {{ font-size: 12px; margin: 8px 0 4px 0; color: var(--muted); }}
  .muted {{ color: var(--muted); }}
  .header {{ margin-bottom: 12px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-bottom: 14px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
  .subcard {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }}
  .full {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--border); vertical-align: top; word-break: break-word; }}
  tr:last-child th, tr:last-child td {{ border-bottom: none; }}
  th {{ font-weight: 500; color: var(--muted); width: 42%; white-space: nowrap; }}
  .muted-row td {{ color: var(--muted); font-style: italic; }}
  .foreign-magic {{ background: var(--red-bg); color: var(--red); font-weight: 600; }}
  .unverified-value {{ color: var(--muted); text-decoration: line-through; }}
  .banner {{ padding: 8px 12px; border-radius: 8px; font-weight: 600; margin-bottom: 8px; }}
  .banner-red {{ background: var(--red-bg); color: var(--red); }}
  .banner-amber {{ background: var(--amber-bg); color: var(--amber); }}
  .banner-green {{ background: var(--green-bg); color: var(--green); }}
  .badge {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; background: var(--border); }}
  .badge-green {{ background: var(--green-bg); color: var(--green); }}
  .badge-amber {{ background: var(--amber-bg); color: var(--amber); }}
  .badge-red {{ background: var(--red-bg); color: var(--red); }}
  .bot-card {{ border-width: 2px; }}
  .bot-card-green {{ border-color: var(--green); }}
  .bot-card-amber {{ border-color: var(--amber); }}
  .bot-card-red {{ border-color: var(--red); }}
  .bot-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; }}
  .light-reasons {{ margin-bottom: 10px; font-size: 13px; }}
  .light-reasons summary {{ cursor: pointer; color: var(--muted); }}
  footer {{ margin-top: 16px; color: var(--muted); font-size: 12px; }}
  footer code {{ font-size: 11px; }}
  @media (max-width: 480px) {{
    body {{ padding: 8px; font-size: 13px; }}
    th {{ width: auto; }}
  }}
</style>
</head>
<body>
{fleet_header}
{bot_cards}
<footer>
  root: <code>{root}</code> &middot; reviews.jsonl: <code>{reviews}</code> &middot;
  built: {built_utc} UTC<br>
  {safety_summary}<br>
  Về trang này (tĩnh, không phụ thuộc dữ liệu bot): không gửi lệnh (no order_send), không khởi chạy MT5 terminal, không ghi file ngoài thư mục <code>dashboard\\</code>.
</footer>
</body>
</html>
"""


def render_footer_safety_summary(bot_views: List[Dict[str, Any]]) -> str:
    """BLOCKING finding 2: the old footer's 'no bot is armed; all are shadow' claim was a
    hardcoded string that would stay on the page saying that even if a bot's risk_config.yaml
    changed. This derives the claim from each bot's own extract_exec_mode() dict every build,
    so a real armed=true or a non-paper/shadow execution mode shows up here in red rather than
    being contradicted by a stale static sentence."""
    armed_bots = [v["spec"].bot_id for v in bot_views if v["exec_mode"].get("armed") is True]
    non_shadow_bots = []
    for v in bot_views:
        em = v["exec_mode"]
        modes = [str(m).lower() for m in (em.get("execution_mode_broker"), em.get("execution_mode_live_risk")) if m is not None]
        if any(m not in ("paper", "shadow") for m in modes):
            non_shadow_bots.append(v["spec"].bot_id)

    armed_txt = (
        _badge(f"CẢNH BÁO: armed=true tại {', '.join(armed_bots)}", "red") if armed_bots
        else _badge("execution.armed = false ở cả 4 bot (risk_config.yaml)", "green")
    )
    shadow_txt = (
        _badge(f"chế độ thực thi ngoài paper/shadow tại {', '.join(non_shadow_bots)}", "red") if non_shadow_bots
        else _badge("cả 4 bot khai báo execution mode paper/shadow (risk_config.yaml)", "green")
    )
    return (
        f'{armed_txt} &middot; {shadow_txt} &middot; '
        'Không có PnL nào được suy diễn từ holdout chưa chấm điểm (evidence_boundary/retire.holdout_status của từng bot).'
    )


def render_html(*, root: Path, reviews_path: Path, now_utc: datetime, fleet_header_html: str, bot_views: List[Dict[str, Any]], interval: int) -> str:
    cards = "".join(render_bot_card(v) for v in bot_views)
    return PAGE_TEMPLATE.format(
        interval=interval, fleet_header=fleet_header_html, bot_cards=cards,
        root=_esc(root), reviews=_esc(reviews_path), built_utc=now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        safety_summary=render_footer_safety_summary(bot_views),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def atomic_write_text(path: Path, content: str) -> None:
    """Writes ONLY inside DASHBOARD_DIR - refuses (raises) otherwise. This is the runtime
    enforcement of "never writes outside its own folder", not just a docstring promise."""
    path = Path(path).resolve()
    try:
        path.relative_to(DASHBOARD_DIR)
    except ValueError:
        raise RuntimeError(f"refusing to write outside {DASHBOARD_DIR}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def load_magic_allocation(root: Path) -> Tuple[Dict[str, int], Dict[int, str], Optional[str]]:
    """Imports bots._shared (pure dict/function definitions, no I/O, no .env) for
    MAGIC_ALLOCATION/RESERVED_MAGICS. Falls back to a transcribed copy if that ever fails,
    and says so - never silently wrong."""
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import bots._shared as shared  # type: ignore
        return dict(shared.MAGIC_ALLOCATION), dict(shared.RESERVED_MAGICS), None
    except Exception as e:
        fallback = {b.bot_id: b.magic for b in BOT_SPECS}
        return fallback, dict(RESERVED_MAGICS_FALLBACK), f"could not import bots._shared ({type(e).__name__}: {e})"


def build_once(args: argparse.Namespace) -> str:
    root = Path(args.root)
    now_utc = datetime.now(timezone.utc)

    magic_allocation, reserved_magics, magic_err = load_magic_allocation(root)
    gate_mod = load_session_gate(root)

    halt = read_account_halt(Path(args.halt_file))
    hwm, hwm_err = read_account_hwm(Path(args.hwm_file))

    if args.mt5:
        mt5_snap = read_mt5_snapshot_with_timeout(
            args.mt5_path, args.mt5_timeout, expected_login=args.expected_login, expected_server=args.expected_server,
        )
        mt5_snap["attempted"] = True
    else:
        mt5_snap = {"attempted": False, "reachable": False, "error": "disabled (pass --mt5 to enable)", "account": None, "positions": []}

    reviews_by_bot, reviews_err = read_latest_reviews(Path(args.reviews), [b.bot_id for b in BOT_SPECS])

    task_info_by_name, task_err = read_task_scheduler([b.task_name for b in BOT_SPECS])

    fleet_header_html, foreign_magic_present = render_fleet_header(
        now_utc=now_utc, halt=halt, hwm=hwm, hwm_err=hwm_err, mt5_snap=mt5_snap,
        magic_allocation=magic_allocation, reserved_magics=reserved_magics, magic_source_err=magic_err,
    )
    if reviews_err:
        fleet_header_html += f'<div class="banner banner-amber">reviews.jsonl: {_esc(reviews_err)}</div>'
    if task_err and not task_info_by_name:
        fleet_header_html += f'<div class="banner banner-amber">Task Scheduler: {_esc(task_err)}</div>'

    bot_views = []
    for spec in BOT_SPECS:
        view = collect_bot(
            root=root, spec=spec, now_utc=now_utc, reviews_by_bot=reviews_by_bot,
            task_info_by_name=task_info_by_name, gate_mod=gate_mod,
            fleet_halted=halt["halted"], fleet_foreign_magic=foreign_magic_present,
            log_tail=args.log_tail,
        )
        view["task_err"] = task_err
        bot_views.append(view)

    html = render_html(
        root=root, reviews_path=Path(args.reviews), now_utc=now_utc,
        fleet_header_html=fleet_header_html, bot_views=bot_views, interval=args.interval,
    )
    atomic_write_text(Path(args.out), html)
    return html


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only operator dashboard for the 4 Exness bots")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--out", default=str(DASHBOARD_DIR / "dashboard.html"))
    p.add_argument("--reviews", default=DEFAULT_REVIEWS)
    p.add_argument("--halt-file", default=DEFAULT_HALT_FILE)
    p.add_argument("--hwm-file", default=DEFAULT_HWM_FILE)
    p.add_argument("--mt5", dest="mt5", action="store_true", default=False, help="Enable a read-only MT5 snapshot (default: off)")
    p.add_argument("--no-mt5", dest="mt5", action="store_false", help="Explicitly disable MT5 (default already off)")
    p.add_argument("--mt5-path", default=None, help="Optional terminal64.exe path to require running; default requires EXACTLY ONE running terminal64.exe process (see resolve_single_terminal_path)")
    p.add_argument("--mt5-timeout", type=float, default=10.0)
    p.add_argument("--expected-login", type=int, default=206539306, help="MT5 login this fleet's demo account is expected to be (bots/exness_fx_d1/deploy/risk_config.yaml)")
    p.add_argument("--expected-server", default="Exness-MT5Trial7", help="MT5 server this fleet's demo account is expected to be")
    p.add_argument("--interval", type=int, default=60, help="HTML meta-refresh seconds")
    p.add_argument("--log-tail", type=int, default=10, help="Number of recent runner invocations to show per bot")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    build_once(args)
    print(f"dashboard: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
