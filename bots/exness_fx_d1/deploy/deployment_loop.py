"""exness_fx_d1 deployment loop — DRY RUN ONLY, no deployable signal exists.

The seven steps of ``25_live_trading/02_etfs_deployment_loop.py``, with
``11_fx_deployment_loop.py`` as the FX worked example, on MetaTrader 5 instead of Alpaca:

1. **Refresh** the MT5 four-hour history through ``bots/_shared/mt5_loader`` (opt-in, Windows
   terminal only); otherwise the existing parquet is read and its coverage reported.
2. **Recompute** the session panel and the features through
   ``case_studies/exness_fx_d1/_features.py`` - ``session_panel(..., keep_decision_ts=True)``
   and ``build_features``, the same functions ``03_financial_features``, the backtest price
   loader and ``tests/test_lookahead.py`` call. One feature implementation, three consumers
   (BOT.md, Decisions log 2026-09-05): a feature written twice agrees on the day it is written
   and drifts on the first edit (Ch25 section 25.1).
3. **Retrain** the model a ``--model <training_hash>`` names, from the preset YAML the registry
   row points at. No hyperparameter is typed in this file.
4. **Persist** to ``deploy/state/<decision date>/``; deployment artefacts never enter the
   research registry.
5. **Predict** the cross-sections of the window.
6. **Replay** that window through ``ml4t.backtest.Engine`` - the parity tape - with the engine
   configured from ``get_backtest_config("exness_fx_d1")`` (the CFD cost branch), not from
   constants retyped here.
7. **Stage** the latest scheduled top-k long-short basket through
   ``SafeBroker(MT5Broker(magic=...), LiveRiskConfig(...))``. ``shadow_mode`` is on, so
   ``SafeBroker`` books the fills into a ``VirtualPortfolio`` and the adapter's ``order_send``
   is never reached.

THE EVIDENCE BOUNDARY IS THE FIRST THING THIS MODULE DEFENDS
------------------------------------------------------------
``25_live_trading/02`` deliberately extends its data past the book's frozen cut ("this
notebook is the one place where the data extends past that cut"). **This bot may not.** Its
declared holdout, 2025-09-01 to 2026-08-31, has not been scored, so:

* a deployment refit is clamped to end strictly before ``evaluation.holdout_start`` and the
  clamp is **asserted**, in the dry run as much as anywhere else;
* the default decision date is the last session before ``holdout_start``; reaching into the
  holdout window needs an explicit ``--as-of`` and is stamped ``holdout_window_touched`` in
  the run record;
* the parity replay reports the order tape and never a return or a Sharpe.

WHAT MAY RUN TODAY
------------------
Phase 5 registered K = 3,204 trials across three strategy cohorts (long-only cross-sectional
top-k, long-short cross-sectional sleeves, long-only per-pair time-series percentile) and no
spec of any cohort is significant at 0.95 on the active return over the 1/N long book; phase 6
is not opened and phase 7 has not run. So:

``--model none`` (default)
    Steps 1, 2, 5, 6. Predictions are **read** from the latest complete registered
    *validation* prediction set (population ``exness_fx_d1:validation-predictions``): no
    refit, no data extension, no holdout. The loop stops after the parity step and says why.
``--model latest-complete``
    The same, and then step 7 in dry run so the staging path is exercised end to end. The run
    record is stamped ``signal_status: plumbing_only``.
``--model <training_hash>``
    Refit that registered run on the clamped window and run all seven steps. Still a dry run
    unless a person passes ``--arm``, and ``deployable`` stays false until phase 7 passes.

Run from the repository root::

    uv run python bots/exness_fx_d1/deploy/deployment_loop.py --fake-mt5
    uv run python bots/exness_fx_d1/deploy/deployment_loop.py --model latest-complete --fake-mt5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import lightgbm  # noqa: F401  # GBM libraries import before scikit-learn (ml4t skill rule 8)
import numpy as np
import polars as pl
import yaml

from bots._shared import MAGIC_ALLOCATION, RESERVED_MAGICS
from bots._shared.monitor import AccountGuard, load_account_limits, read_account_snapshot
from bots._shared.mt5_broker import MT5Broker
from bots._shared.mt5_loader import load_mt5_bars, to_market_watch

logger = logging.getLogger("exness_fx_d1.deploy")

# ---------------------------------------------------------------------------------------
# Parameters block (the module equivalent of a notebook's `tags=["parameters"]` cell).
# Production defaults, overridable only from the command line - never edited to true here.
# 25_live_trading/02:121 (SUBMIT_PAPER_ORDERS = False) and 11_fx_deployment_loop.py:134.
# ---------------------------------------------------------------------------------------
SUBMIT_ORDERS = False  # explicit opt-in per session; the default is an offline dry run
REFRESH_DATA = False  # pulling new bars needs a Windows terminal and is not reproducible
MODEL_SELECTOR = "none"  # no deployable signal exists; see the module docstring

BOT_ID = "exness_fx_d1"
DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_RISK_CONFIG = DEPLOY_DIR / "risk_config.yaml"
STATE_DIR = DEPLOY_DIR / "state"

_ISO_DURATION_UNITS = {"Y": "years", "M": "months", "W": "weeks", "D": "days"}


# =======================================================================================
# Configuration
# =======================================================================================
@dataclass(frozen=True)
class DeployConfig:
    """``risk_config.yaml`` and ``setup.yaml``, parsed and cross-checked."""

    risk: dict[str, Any]
    setup: dict[str, Any]
    risk_path: Path
    case_dir: Path
    state_dir: Path

    @property
    def magic(self) -> int:
        return int(self.risk["magic"])

    @property
    def universe(self) -> list[str]:
        return sorted(self.setup["universe"]["symbols"])

    @property
    def allocated(self) -> float:
        return float(self.risk["capital"]["allocated"])

    @property
    def snapshot_utc(self) -> time:
        return time.fromisoformat(str(self.setup["decision"]["snapshot_utc"]))

    @property
    def tolerance_minutes(self) -> int:
        return int(self.setup["decision"]["session_close_tolerance_minutes"])

    @property
    def holdout_start(self) -> date:
        return date.fromisoformat(str(self.setup["evaluation"]["holdout_start"]))

    @property
    def holdout_scored(self) -> bool:
        return self.holdout_marker.exists()

    @property
    def holdout_marker(self) -> Path:
        return self.state_dir / Path(self.risk["evidence_boundary"]["holdout_scored_marker"]).name

    @property
    def model_block(self) -> dict[str, Any]:
        return dict(self.risk.get("model") or {})

    @property
    def live_risk_block(self) -> dict[str, Any]:
        return dict(self.risk["live_risk_config"])

    @property
    def cfd_guards(self) -> dict[str, Any]:
        return dict(self.risk.get("cfd_guards") or {})


def load_deploy_config(
    risk_config: Path | str | None = None,
    *,
    case_study: str = BOT_ID,
    state_dir: Path | str | None = None,
) -> DeployConfig:
    """Read both declarations and refuse a decision block, magic or mode that has drifted.

    The live loop must apply the research decision rule and the allocated magic; both are
    written down twice (once for the stages or the allocation table, once for the operator),
    so the loop checks they agree instead of trusting that somebody kept them in step.
    """
    from utils.paths import get_case_study_dir

    risk_path = Path(risk_config) if risk_config is not None else DEFAULT_RISK_CONFIG
    risk = yaml.safe_load(risk_path.read_text())
    case_dir = get_case_study_dir(case_study)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())

    if str(risk.get("case_study")) != case_study:
        raise ValueError(f"{risk_path.name} names case study {risk.get('case_study')!r}, not {case_study!r}")

    declared, actual = risk.get("decision", {}), setup["decision"]
    for key in ("snapshot", "snapshot_utc", "session_calendar", "session_close_tolerance_minutes"):
        if str(declared.get(key)) != str(actual.get(key)):
            raise ValueError(
                f"{risk_path.name} declares decision.{key}={declared.get(key)!r} but setup.yaml "
                f"declares {actual.get(key)!r}; the live loop would decide at a different "
                "instant from the research pipeline"
            )
    boundary = risk.get("evidence_boundary", {})
    for key, setup_key in (("holdout_start", "holdout_start"), ("holdout_end", "holdout_end")):
        if str(boundary.get(key)) != str(setup["evaluation"][setup_key]):
            raise ValueError(
                f"{risk_path.name} declares evidence_boundary.{key}={boundary.get(key)!r} but "
                f"setup.yaml declares {setup['evaluation'][setup_key]!r}"
            )

    magic = int(risk["magic"])
    allocated = MAGIC_ALLOCATION.get(BOT_ID)
    if magic != allocated:
        raise ValueError(
            f"{risk_path.name} declares magic {magic}, the allocation table in "
            f"bots/_shared/__init__.py allocates {allocated} to {BOT_ID}"
        )
    if magic in RESERVED_MAGICS:
        raise ValueError(f"magic {magic} is reserved: {RESERVED_MAGICS[magic]}")
    if str(risk["execution"]["mode"]) != "paper":
        raise ValueError(
            f"{risk_path.name} declares execution.mode={risk['execution']['mode']!r}; this bot "
            "may only run in paper mode until phase 7 passes (README.md)"
        )
    return DeployConfig(
        risk=risk,
        setup=setup,
        risk_path=risk_path,
        case_dir=case_dir,
        state_dir=Path(state_dir) if state_dir is not None else STATE_DIR,
    )


def _parse_iso_duration(text: str) -> Any:
    """``P3Y`` / ``P18M`` / ``P90D`` -> a ``pandas.DateOffset``. Only what setup.yaml uses."""
    import pandas as pd

    raw = str(text).strip().upper()
    if not raw.startswith("P") or len(raw) < 3:
        raise ValueError(f"unsupported duration {text!r}; expected an ISO period such as P3Y")
    number, unit = raw[1:-1], raw[-1]
    if unit not in _ISO_DURATION_UNITS or not number.isdigit():
        raise ValueError(f"unsupported duration {text!r}")
    return pd.DateOffset(**{_ISO_DURATION_UNITS[unit]: int(number)})


# =======================================================================================
# Step 1 — refresh
# =======================================================================================
def refresh_data(
    cfg: DeployConfig,
    *,
    do_refresh: bool = REFRESH_DATA,
    mt5: Any = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Bring the MT5 four-hour history up to date, or report the coverage of what is there.

    Opt-in exactly as ``REFRESH_DATA`` is in ``25_live_trading/02``: a loop that silently
    re-downloads is not reproducible, and the download needs a Windows terminal the research
    environment does not have. **A refresh extends the data past the holdout boundary but
    never past the training clamp** - the clamp is enforced in :func:`training_window`, not
    here, so the two cannot drift apart.
    """
    from case_studies.exness_fx_d1._features import GOLD_SYMBOL

    symbols = [*cfg.universe, GOLD_SYMBOL]
    record: dict[str, Any] = {"requested_symbols": symbols, "refreshed": bool(do_refresh)}
    if do_refresh:
        from bots._shared.mt5_loader import download_mt5_bars

        written = download_mt5_bars(symbols, timeframes=("4h",), mt5=mt5, data_dir=data_dir)
        record["written"] = {k: str(v) for k, v in written.items()}
    bars = load_mt5_bars("4h", symbols=symbols, data_dir=data_dir)
    per_symbol = (
        bars.group_by("symbol")
        .agg(pl.len().alias("bars"), pl.col("timestamp").max().alias("last_bar_utc"))
        .sort("symbol")
    )
    record["bars"] = int(bars.height)
    record["coverage"] = [
        {"symbol": r["symbol"], "bars": int(r["bars"]), "last_bar_utc": str(r["last_bar_utc"])}
        for r in per_symbol.to_dicts()
    ]
    record["last_bar_utc"] = str(bars["timestamp"].max())
    return record


# =======================================================================================
# Step 2 — recompute features (the research code path, not a copy of it)
# =======================================================================================
@dataclass
class FeaturePanel:
    prices: pl.DataFrame
    features: pl.DataFrame
    feature_columns: list[str]
    decision_date: date
    decision_ts: datetime | None
    quoting_symbols: list[str]
    missing_symbols: list[str]
    record: dict[str, Any] = field(default_factory=dict)


def recompute_features(
    cfg: DeployConfig,
    *,
    as_of: date | None = None,
    data_dir: Path | str | None = None,
    verbose: bool = False,
) -> FeaturePanel:
    """Session panel + feature matrix from ``case_studies/exness_fx_d1/_features.py``.

    ``as_of`` pins the decision date. The default is the last session strictly before
    ``holdout_start`` while the holdout is unscored: the loop does not wander into the
    evidence window just because more bars exist.
    """
    from case_studies.exness_fx_d1._features import (
        GOLD_SYMBOL,
        build_features,
        feature_columns,
        load_session_panel,
    )

    end_date = str(as_of) if as_of is not None else None
    prices = load_session_panel(
        cfg.setup,
        symbols=cfg.universe,
        start_date=str(cfg.setup["universe"]["history_start"]),
        end_date=end_date,
        keep_decision_ts=True,
        verbose=verbose,
    )
    gold = load_session_panel(
        cfg.setup,
        symbols=[GOLD_SYMBOL],
        start_date=str(cfg.setup["universe"]["history_start"]),
        end_date=end_date,
        verbose=verbose,
    )
    block = cfg.setup["features"]
    built = build_features(
        prices.drop("decision_ts"),
        windows=block["windows"],
        ranked=block["ranked"],
        periods_per_year=cfg.setup["evaluation"]["periods_per_year"],
        gold=gold,
    )
    cols = feature_columns(built)

    sessions = sorted(prices["timestamp"].unique().to_list())
    if as_of is not None:
        decision_date = as_of
    elif cfg.holdout_scored:
        decision_date = sessions[-1]
    else:
        before = [d for d in sessions if d < cfg.holdout_start]
        if not before:
            raise ValueError("no session before holdout_start; nothing may be decided on")
        decision_date = before[-1]

    latest = prices.filter(pl.col("timestamp") == pl.lit(decision_date).cast(pl.Date))
    quoting = sorted(latest["symbol"].to_list())
    missing = sorted(set(cfg.universe) - set(quoting))
    decision_ts = latest["decision_ts"].max() if latest.height else None
    record = {
        "decision_date": str(decision_date),
        "decision_ts_utc": str(decision_ts) if decision_ts else None,
        "declared_snapshot_utc": str(cfg.snapshot_utc),
        "quoting_symbols": quoting,
        "missing_symbols": missing,
        "n_sessions": len(sessions),
        "n_feature_rows": int(built.height),
        "n_feature_columns": len(cols),
        "feature_source": "case_studies/exness_fx_d1/_features.py (session_panel + build_features)",
        "holdout_window_touched": bool(decision_date >= cfg.holdout_start),
    }
    return FeaturePanel(prices, built, cols, decision_date, decision_ts, quoting, missing, record)


def decision_bar_check(cfg: DeployConfig, panel: FeaturePanel) -> dict[str, Any]:
    """Kill criterion (e) / open question 9. Same rule as research, measured in minutes.

    ``_features.session_panel`` has already dropped a pair-session whose newest bar closes
    further ahead of the declared close than ``session_close_tolerance_minutes``; this
    function reports the lag it left, so ``lag_minutes`` lands in the run record whether or
    not the check passes. Three ways to fail: a pair that did not quote at all, a bar that
    closes **after** the declared snapshot (which would be lookahead), and a lag above the
    declared tolerance.
    """
    snapshot = datetime.combine(panel.decision_date, cfg.snapshot_utc)
    tolerance = cfg.tolerance_minutes
    per_symbol: list[dict[str, Any]] = []
    latest = panel.prices.filter(pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date))
    for row in latest.to_dicts():
        lag = (snapshot - row["decision_ts"]).total_seconds() / 60.0
        per_symbol.append(
            {"symbol": row["symbol"], "decision_ts_utc": str(row["decision_ts"]), "lag_minutes": lag}
        )
    reasons: list[str] = []
    if panel.missing_symbols:
        reasons.append(f"no decision bar for {panel.missing_symbols} on {panel.decision_date}")
    if latest.height == 0:
        reasons.append(f"no session row at all for {panel.decision_date}")
    for entry in per_symbol:
        if entry["lag_minutes"] < 0:
            reasons.append(f"{entry['symbol']}: bar closes after the {snapshot} UTC snapshot")
        elif entry["lag_minutes"] > tolerance:
            reasons.append(
                f"{entry['symbol']}: newest bar closed {entry['lag_minutes']:.0f} min before the "
                f"snapshot, above the {tolerance} min tolerance"
            )
    return {
        "ok": not reasons,
        "snapshot_utc": str(snapshot),
        "tolerance_minutes": tolerance,
        "per_symbol": per_symbol,
        "max_lag_minutes": max((e["lag_minutes"] for e in per_symbol), default=None),
        "reasons": reasons,
    }


def record_no_trade_day(cfg: DeployConfig, decision_date: date, reason: str) -> dict[str, Any]:
    """(e) is a no-trade day, not a breaker; it escalates only when it repeats.

    The declared escalation is ``missing_decision_bar.escalate_after`` misses inside
    ``window_sessions``. One missing bar is a broker outage (2018-01-31); a run of them is a
    broken feed and the strategy tier picks it up as a breaker.
    """
    block = cfg.risk["missing_decision_bar"]
    path = cfg.state_dir / "no_trade_days.json"
    history: list[dict[str, Any]] = []
    if path.exists():
        history = json.loads(path.read_text()).get("days", [])
    entry = {"date": str(decision_date), "reason": reason, "logged_at": datetime.now(UTC).isoformat()}
    if not any(d["date"] == entry["date"] for d in history):
        history.append(entry)
    history = history[-int(block["window_sessions"]) * 4 :]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"days": history}, indent=2))
    recent = history[-int(block["window_sessions"]) :]
    return {
        "action": str(block["action"]),
        "count_in_window": len(recent),
        "escalate_after": int(block["escalate_after"]),
        "window_sessions": int(block["window_sessions"]),
        "escalated": len(recent) >= int(block["escalate_after"]),
        "log": str(path),
    }


# =======================================================================================
# Step 3 — resolve the model, then retrain it
# =======================================================================================
@dataclass(frozen=True)
class ModelChoice:
    """A registered run, resolved to what a refit or a replay needs."""

    mode: str  # "none" | "plumbing" | "refit"
    training_hash: str | None = None
    prediction_hash: str | None = None
    family: str | None = None
    config_name: str | None = None
    label: str | None = None
    created_at: str | None = None
    feature_names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "training_hash": self.training_hash,
            "prediction_hash": self.prediction_hash,
            "family": self.family,
            "config_name": self.config_name,
            "label": self.label,
            "registered_at": self.created_at,
            "n_feature_names": len(self.feature_names),
        }


def registry_path(cfg: DeployConfig) -> Path:
    return cfg.case_dir / "run_log" / "registry.db"


def _open_registry(cfg: DeployConfig) -> sqlite3.Connection:
    db_path = registry_path(cfg)
    if not db_path.exists():
        raise FileNotFoundError(
            f"no registry at {db_path}; run the experiment's stages with ML4T_OUTPUT_DIR set "
            "before asking the deployment loop for a model"
        )
    return sqlite3.connect(str(db_path))


def resolve_model(
    cfg: DeployConfig, selector: str | None, *, prediction_hash: str | None = None
) -> ModelChoice:
    """``none`` -> replay only; ``latest-complete`` -> plumbing; a hash -> that registered run.

    A model may come only from the registry: a deployed model has to be one the research
    pipeline produced and counted as a trial. An unknown hash is an error, never a fallback.
    ``prediction_hash`` names a specific registered **validation** prediction set for the
    replay modes instead of taking the most recent one.
    """
    selector = selector if selector not in (None, "") else "none"
    if selector in ("none", "latest-complete"):
        db = _open_registry(cfg)
        query = (
            "SELECT p.prediction_hash, t.training_hash, t.family, t.config_name, t.label, "
            "       p.created_at, t.spec_json "
            "FROM prediction_sets p JOIN training_runs t ON t.training_hash = p.training_hash "
            "WHERE p.split = 'validation'"
        )
        try:
            if prediction_hash:
                row = db.execute(
                    query + " AND p.prediction_hash = ?", (prediction_hash,)
                ).fetchone()
                if row is None:
                    raise LookupError(
                        f"prediction set {prediction_hash!r} is not a registered validation set "
                        f"in {registry_path(cfg)}"
                    )
            else:
                rows = db.execute(query + " ORDER BY p.created_at DESC").fetchall()
                row = None
                pred_dir = cfg.case_dir / "run_log" / "predictions"
                for r in rows:
                    if (pred_dir / r[0] / "predictions.parquet").exists():
                        row = r
                        break
                if row is None and rows:
                    row = rows[0]
        finally:
            db.close()
        if row is None:
            raise LookupError(f"{registry_path(cfg)} holds no validation prediction sets")
        spec = json.loads(row[6]) if row[6] else {}
        return ModelChoice(
            mode="none" if selector == "none" else "plumbing",
            training_hash=row[1],
            prediction_hash=row[0],
            family=row[2],
            config_name=row[3],
            label=row[4],
            created_at=row[5],
            feature_names=list(spec.get("computation", {}).get("feature_names") or []),
        )
    db = _open_registry(cfg)
    try:
        row = db.execute(
            "SELECT training_hash, family, config_name, label, created_at, spec_json "
            "FROM training_runs WHERE training_hash = ?",
            (selector,),
        ).fetchone()
    finally:
        db.close()
    if row is None:
        raise LookupError(f"training hash {selector!r} is not registered in {registry_path(cfg)}")
    spec = json.loads(row[5]) if row[5] else {}
    return ModelChoice(
        mode="refit",
        training_hash=row[0],
        family=row[1],
        config_name=row[2],
        label=row[3],
        created_at=row[4],
        feature_names=list(spec.get("computation", {}).get("feature_names") or []),
    )


@dataclass
class TrainedModel:
    """The deployment artefact: one fit on one window, not the research walk-forward."""

    estimator: Any
    feature_columns: list[str]
    imputer: Any | None
    scaler: Any | None
    metadata: dict[str, Any]

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = frame.select(self.feature_columns).to_numpy().astype(float)
        if self.imputer is not None:
            matrix = self.imputer.transform(matrix)
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        return np.asarray(self.estimator.predict(matrix), dtype=float)


def training_window(cfg: DeployConfig, panel: FeaturePanel, label: str) -> dict[str, Any]:
    """The walk-forward geometry rolled forward, then clamped at the evidence boundary.

    ``25_live_trading/02`` cuts training at ``LABEL_AVAILABLE_AS_OF``, the last date whose
    forward label realises before the live window; the same rule applies here with the label's
    own horizon. Two clamps on top:

    * the window is ``evaluation.train_size`` long, so the deployment fit sees what the folds
      saw, rolled forward, rather than the whole sample;
    * while ``holdout_scored`` is false it ends **strictly before**
      ``evaluation.holdout_start`` - and the caller asserts that, because a refit that quietly
      spends the holdout ends the bot's evidence chain with nobody noticing.
    """
    import pandas as pd

    horizon = _label_horizon_sessions(cfg, label)
    sessions = sorted(panel.prices["timestamp"].unique().to_list())
    if len(sessions) <= horizon:
        raise ValueError(f"panel has {len(sessions)} sessions, fewer than the {horizon}-session horizon")
    train_end = sessions[-horizon - 1]
    clamped = False
    if not cfg.holdout_scored:
        before = [d for d in sessions if d < cfg.holdout_start]
        if not before:
            raise ValueError("no session before holdout_start; nothing may be trained on")
        if train_end >= cfg.holdout_start:
            train_end, clamped = before[-1], True
    offset = _parse_iso_duration(cfg.setup["evaluation"]["train_size"])
    train_start = (pd.Timestamp(train_end) - offset).date()
    window = {
        "train_start": train_start,
        "train_end": train_end,
        "label_horizon_sessions": horizon,
        "train_size": str(cfg.setup["evaluation"]["train_size"]),
        "clamped_to_holdout_boundary": clamped,
        "holdout_start": str(cfg.holdout_start),
        "holdout_scored": cfg.holdout_scored,
    }
    if not cfg.holdout_scored:
        assert train_end < cfg.holdout_start, (
            f"deployment refit would train through {train_end}, inside the unscored holdout "
            f"starting {cfg.holdout_start}"
        )
    return window


def _label_horizon_sessions(cfg: DeployConfig, label: str) -> int:
    steps = cfg.setup["labels"].get("rebalance_step") or {}
    if label in steps:
        return int(steps[label])
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot read a horizon out of label {label!r}")
    return int(digits)


def load_label_frame(cfg: DeployConfig, label: str) -> pl.DataFrame:
    path = cfg.case_dir / "labels" / f"{label}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"label {label!r} not found at {path}; run 02_labels in the experiment "
            "(ML4T_OUTPUT_DIR) before deploying a model trained on it"
        )
    return pl.read_parquet(path)


def retrain(cfg: DeployConfig, panel: FeaturePanel, choice: ModelChoice) -> TrainedModel:
    """One fit of the chosen preset on the window :func:`training_window` returns.

    The preset is read through the repository's own loader from
    ``case_studies/config/<family>/<config_name>.yaml``, so the hyperparameters are the ones
    the registry row was fitted with and none of them is typed here. Preprocessing follows the
    family, matching what the research stages record: ``linear`` uses the median imputer and
    the standard scaler (``median-imputer-standard-scaler/v1``), ``gbm`` feeds LightGBM the raw
    float matrix (``lightgbm-native-float32/v1``).
    """
    from case_studies.utils.registry import load_preset

    window = training_window(cfg, panel, choice.label)
    labels = load_label_frame(cfg, choice.label)
    wanted = choice.feature_names or panel.feature_columns
    available = [c for c in wanted if c in panel.features.columns]
    missing = [c for c in wanted if c not in panel.features.columns]
    if missing:
        raise ValueError(
            f"the registered model needs {len(missing)} feature column(s) the deployment "
            f"recompute does not produce: {missing[:6]}{'...' if len(missing) > 6 else ''}. "
            "Model-based features (04_model_based_features) are not recomputed live; deploy a "
            "financial-only model or add the model-based stage to the loop."
        )

    joined = panel.features.join(labels, on=["timestamp", "symbol"], how="inner")
    training = joined.filter(
        (pl.col("timestamp") >= pl.lit(window["train_start"]).cast(pl.Date))
        & (pl.col("timestamp") <= pl.lit(window["train_end"]).cast(pl.Date))
    ).drop_nulls(subset=[choice.label])
    if training.height == 0:
        raise ValueError(f"no training rows between {window['train_start']} and {window['train_end']}")
    observed_end = training["timestamp"].max()
    if not cfg.holdout_scored:
        assert observed_end < cfg.holdout_start, (
            f"training rows reach {observed_end}, inside the unscored holdout "
            f"starting {cfg.holdout_start}"
        )

    x = training.select(available).to_numpy().astype(float)
    y = training[choice.label].to_numpy().astype(float)
    preset = load_preset(choice.family, choice.config_name)
    estimator, imputer, scaler = _fit_family(choice, preset, x, y)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "bot_id": BOT_ID,
        "source_training_hash": choice.training_hash,
        "family": choice.family,
        "config_name": choice.config_name,
        "label": choice.label,
        "n_train_rows": int(training.height),
        "n_features": len(available),
        "feature_columns": available,
        "train_window": {k: str(v) for k, v in window.items()},
        "observed_train_end": str(observed_end),
        "preprocessing": "median-imputer-standard-scaler/v1" if imputer is not None else "lightgbm-native-float32/v1",
        "deployment_fit": "single fit on the rolled-forward walk-forward window, not the research CV",
    }
    return TrainedModel(estimator, available, imputer, scaler, metadata)


def _fit_family(choice: ModelChoice, preset: dict[str, Any], x: np.ndarray, y: np.ndarray):
    """Fit one family. Every parameter comes from ``preset``; none is written here."""
    if choice.family == "gbm":
        from case_studies.utils.gbm import create_model

        model = create_model(preset.get("library", "lightgbm"), preset.get("params"), device="cpu")
        model.fit(x, y)
        return model, None, None
    if choice.family == "linear":
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        from utils.modeling import resolve_linear_params

        imputer = SimpleImputer(strategy="median").fit(x)
        scaler = StandardScaler().fit(imputer.transform(x))
        design = scaler.transform(imputer.transform(x))
        model = _linear_estimator(preset, resolve_linear_params(preset, design, y))
        model.fit(design, y)
        return model, imputer, scaler
    raise NotImplementedError(
        f"family {choice.family!r} has no deployment fit; this bot ran 06_linear and 07_gbm only"
    )


def _linear_estimator(preset: dict[str, Any], params: dict[str, Any]):
    from sklearn import linear_model

    cls_name = preset.get("model_class") or preset.get("class") or "LinearRegression"
    estimator_cls = getattr(linear_model, cls_name, None)
    if estimator_cls is None:
        raise ValueError(f"unknown linear model class {cls_name!r} in preset {preset.get('config_name')!r}")
    return estimator_cls(**params)


# =======================================================================================
# Step 4 — persist
# =======================================================================================
def persist(model: TrainedModel, *, state_dir: Path, decision_date: date) -> Path:
    """Write the artefact under ``deploy/state/<decision date>/``; never into the registry."""
    target = Path(state_dir) / str(decision_date)
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "model.pkl", "wb") as fh:
        pickle.dump(model.estimator, fh)
    if model.imputer is not None:
        with open(target / "imputer.pkl", "wb") as fh:
            pickle.dump(model.imputer, fh)
    if model.scaler is not None:
        with open(target / "scaler.pkl", "wb") as fh:
            pickle.dump(model.scaler, fh)
    (target / "feature_columns.json").write_text(json.dumps(model.feature_columns, indent=2))
    (target / "training_metadata.json").write_text(json.dumps(model.metadata, indent=2, default=str))
    return target


# =======================================================================================
# Step 5 — predict (refit) or read the registered validation predictions (plumbing)
# =======================================================================================
def predict(model: TrainedModel, panel: FeaturePanel, *, live_start: date) -> pl.DataFrame:
    """Score every cross-section from ``live_start`` on; drop all-null feature rows."""
    live = panel.features.filter(pl.col("timestamp") >= pl.lit(live_start).cast(pl.Date)).select(
        ["timestamp", "symbol", *model.feature_columns]
    )
    live = live.filter(~pl.all_horizontal([pl.col(c).is_null() for c in model.feature_columns]))
    if live.height == 0:
        raise ValueError(f"no scorable rows from {live_start}")
    return live.select(["timestamp", "symbol"]).with_columns(
        pl.Series("score", model.predict(live))
    )


def read_registered_predictions(cfg: DeployConfig, choice: ModelChoice) -> pl.DataFrame:
    """The plumbing path: a registered **validation** prediction set, read, never refitted.

    No refit, no data extension, no holdout: the rows were produced by the research stages on
    the validation split (2021-09-06 .. 2025-08-28) and are already counted in K = 3,204.
    """
    from case_studies.utils.registry import read_predictions

    frame = read_predictions(BOT_ID, choice.prediction_hash, case_dir=cfg.case_dir)
    frame = frame.rename({c: "score" for c in ("y_score", "prediction") if c in frame.columns})
    out = frame.select(
        pl.col("timestamp").cast(pl.Date), pl.col("symbol"), pl.col("score").cast(pl.Float64)
    ).sort(["timestamp", "symbol"])
    if not cfg.holdout_scored:
        latest = out["timestamp"].max()
        assert latest < cfg.holdout_start, (
            f"registered prediction set {choice.prediction_hash} reaches {latest}, inside the "
            f"unscored holdout starting {cfg.holdout_start}"
        )
    return out


# =======================================================================================
# Step 6 — the parity tape
# =======================================================================================
class TopKLongShortStrategy:
    """The book phase 5 traded: equal-weight top-k long, bottom-k short.

    Written against ``ml4t.backtest.Strategy``'s ``on_data`` contract exactly as
    ``25_live_trading/02``'s ``CrossSectionalRidgeStrategy`` is, with the short sleeve
    ``setup.yaml::mapping.position_state_space: long_short`` declares. The base class is bound
    at call time so this module imports without ``ml4t``.
    """

    def __init__(
        self,
        predictions: pl.DataFrame,
        *,
        top_k: int,
        rebalance_every: int,
        symbols: list[str],
        long_short: bool,
    ):
        self.predictions = predictions
        self.top_k = int(top_k)
        self.rebalance_every = int(rebalance_every)
        self.symbols = list(symbols)
        self.long_short = bool(long_short)
        self._bars_seen = 0
        self.signal_log: list[dict[str, Any]] = []
        self.rebalance_log: list[dict[str, Any]] = []

    def on_start(self, broker) -> None:  # noqa: ARG002 - Strategy interface
        self._bars_seen = 0
        self.signal_log = []
        self.rebalance_log = []

    def on_end(self, broker) -> None:  # noqa: ARG002 - Strategy interface
        return None

    def on_data(self, timestamp, data, context, broker) -> None:  # noqa: ARG002 - interface
        self._bars_seen += 1
        if (self._bars_seen - 1) % self.rebalance_every != 0:
            return
        targets = self.target_weights(timestamp)
        if targets is None:
            return
        self.rebalance_log.append(
            {
                "timestamp": timestamp,
                "long": sorted(s for s, w in targets.items() if w > 0),
                "short": sorted(s for s, w in targets.items() if w < 0),
                "weights": dict(targets),
            }
        )
        prices = {s: data[s]["close"] for s in self.symbols if s in data}
        account_value = broker.get_account_value()
        for symbol in self.symbols:
            position = broker.get_position(symbol)
            held = float(position.quantity) if position is not None else 0.0
            weight = targets.get(symbol, 0.0)
            if weight != 0.0 and symbol not in prices:
                continue  # held through a gap rather than exited on a missing bar
            target_qty = round(account_value * weight / prices[symbol]) if weight != 0.0 else 0.0
            delta = target_qty - held
            if abs(delta) < 1:
                continue
            side = self._side(delta)
            broker.submit_order(symbol, abs(int(delta)), side=side)
            self.signal_log.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": side.value,
                    "delta": int(delta),
                    "target_qty": int(target_qty),
                    "weight": weight,
                }
            )

    def target_weights(self, timestamp) -> dict[str, float] | None:
        """The ranking rule the sweep used, exposed so the live leg reuses it verbatim."""
        stamp = timestamp.date() if hasattr(timestamp, "date") else timestamp
        scores = self.predictions.filter(
            pl.col("timestamp") == pl.lit(stamp).cast(self.predictions.schema["timestamp"])
        ).sort("score", descending=True)
        needed = self.top_k * (2 if self.long_short else 1)
        if scores.height < needed:
            return None
        names = scores["symbol"].to_list()
        weights = {s: 1.0 / self.top_k for s in names[: self.top_k]}
        if self.long_short:
            weights.update({s: -1.0 / self.top_k for s in names[-self.top_k :]})
        return weights

    @staticmethod
    def _side(delta: float):
        from ml4t.backtest import OrderSide

        return OrderSide.BUY if delta > 0 else OrderSide.SELL


def _make_strategy(predictions: pl.DataFrame, **kwargs: Any):
    from ml4t.backtest import Strategy

    bound = type("BoundTopKLongShortStrategy", (TopKLongShortStrategy, Strategy), {})
    return bound(predictions, **kwargs)


def replay_parity(
    cfg: DeployConfig,
    panel: FeaturePanel,
    predictions: pl.DataFrame,
    *,
    live_start: date,
    label: str,
) -> dict[str, Any]:
    """Replay the window through ``ml4t.backtest.Engine``: the offline reference tape.

    The engine spec is the **research** spec, read from
    ``case_studies/utils/backtest_loaders.get_backtest_config("exness_fx_d1")`` - the CFD
    branch (BOT.md 2026-09-06): commission 0 on a Pro account, slippage = the top of the
    measured spread range, integer share type, ``NEXT_BAR`` on the session ``open``
    (``decision.execution_delay: next_bar_open``), long/short from ``setup.yaml::mapping``.
    Nothing is retyped here.

    While the holdout is unscored the tape's **performance** is not reported: a final value
    printed here would be a reading taken outside the holdout stages. The baskets, the order
    deltas and the schedule - everything parity needs - are reported in full.
    """
    from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode
    from ml4t.backtest.config import ShareType, SlippageType

    from case_studies.utils.backtest_loaders import get_backtest_config

    research = get_backtest_config(BOT_ID)
    prices = (
        panel.prices.filter(pl.col("timestamp") >= pl.lit(live_start).cast(pl.Date))
        .select(["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
        .sort(["timestamp", "symbol"])
    )
    preds = predictions.with_columns(pl.col("timestamp").cast(pl.Date))
    long_short = bool(cfg.model_block.get("long_short", research.long_short))
    rebalance_every = int(cfg.setup["labels"]["rebalance_step"][label])
    strategy = _make_strategy(
        preds,
        top_k=int(cfg.model_block.get("top_k", 1)),
        rebalance_every=rebalance_every,
        symbols=sorted(preds["symbol"].unique().to_list()),
        long_short=long_short,
    )
    # Generation 3 (2026-09-10, bots/exness_fx_d1/GEN3_DECLARATION_2026-09-10.md, OQ 32(b)):
    # research.share_type moved from "integer" to "mt5_lot" (case_studies/utils/backtest_runner.py
    # ::_apply_lot_floor_if_declared's gate). ml4t.backtest.ShareType has no "mt5_lot" member
    # (only FRACTIONAL / INTEGER); ShareType("mt5_lot") raises. Mentor fleet gate, 2026-09-10,
    # item B: mapped to "fractional", not "integer", mirroring case_studies/utils/
    # backtest_presets.py::build_resolved_backtest_config -- a second, engine-level INTEGER
    # truncation on top of the lot-grid rounding _apply_lot_floor_if_declared already applies is
    # a residual, avoidable discretisation error (declared and bounded in
    # GEN3_DECLARATION_2026-09-10.md, "Mentor fleet-gate addendum", item 1), not a broker
    # constraint the lot grid did not already enforce. The MT5 lot floor itself stays a
    # weights-level guard applied in research sizing, not a share-rounding mode this engine has.
    _engine_share_type = "fractional" if research.share_type == "mt5_lot" else research.share_type
    config = BacktestConfig(
        initial_cash=cfg.allocated,
        execution_mode=ExecutionMode.NEXT_BAR,
        share_type=ShareType(_engine_share_type),
        commission_rate=research.commission_bps / 10_000.0,
        slippage_type=SlippageType.PERCENTAGE,
        slippage_rate=research.slippage_bps / 10_000.0,
        allow_short_selling=long_short,
    )
    results = Engine(feed=DataFeed(prices_df=prices), strategy=strategy, config=config).run()
    tape = [
        {
            "date": _date_key(e["timestamp"]),
            "symbol": e["symbol"],
            "side": e["side"],
            "delta": int(e["delta"]),
            "target_qty": int(e["target_qty"]),
            "weight": float(e["weight"]),
        }
        for e in strategy.signal_log
    ]
    record: dict[str, Any] = {
        "live_window_start": str(live_start),
        "n_sessions": int(prices["timestamp"].n_unique()),
        "n_rebalances": len(strategy.rebalance_log),
        "n_signals": len(tape),
        "tape": tape,
        "engine": {
            "source": "case_studies.utils.backtest_loaders.get_backtest_config('exness_fx_d1')",
            "execution_mode": "next_bar",
            "share_type": _engine_share_type,
            "commission_bps": research.commission_bps,
            "slippage_bps": research.slippage_bps,
            "long_short": long_short,
            "initial_cash": cfg.allocated,
            "rebalance_every": rebalance_every,
        },
        "performance_reported": cfg.holdout_scored,
    }
    if strategy.rebalance_log:
        last = strategy.rebalance_log[-1]
        record["last_rebalance"] = {
            "date": _date_key(last["timestamp"]),
            "long": last["long"],
            "short": last["short"],
            "weights": last["weights"],
        }
    if cfg.holdout_scored:
        record["final_value"] = float(results["final_value"])
        record["total_return_pct"] = float(results["total_return_pct"])
    else:
        record["performance_withheld_because"] = (
            "the declared holdout 2025-09-01..2026-08-31 has not been scored by the holdout "
            "stages; a return printed here would be a holdout reading taken outside them"
        )
    record["_strategy"] = strategy
    return record


def _date_key(stamp: Any) -> str:
    return stamp.date().isoformat() if hasattr(stamp, "date") else str(stamp)[:10]


# =======================================================================================
# Step 7 — stage the basket through SafeBroker
# =======================================================================================
def build_live_risk_config(cfg: DeployConfig, *, armed: bool):
    """``LiveRiskConfig`` from the YAML. Arming can only ever make the config *safer*.

    Every limit is a field of ``ml4t.live.safety.LiveRiskConfig`` - including the loss limits,
    the staleness guard, the asset restrictions, the reconciliation flag and the journal - so
    nothing here re-implements a control the class already has.
    """
    from ml4t.live import ExecutionMode as LiveExecutionMode
    from ml4t.live import LiveRiskConfig

    block = cfg.live_risk_block
    # One switch. `shadow_mode` is never read from the YAML and never passed: the dataclass
    # derives it from `execution_mode` (safety.py:153-160), and passing both is how a config
    # ends up raising `ExecutionModeError`. Not arming forces SHADOW, so arming can only ever
    # move the mode in the safe direction.
    mode = LiveExecutionMode(str(block.get("execution_mode") or "shadow"))
    if not armed:
        mode = LiveExecutionMode.SHADOW
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return LiveRiskConfig(
        max_position_value=float(block["max_position_value"]),
        max_position_shares=float(block["max_position_shares"]),
        max_total_exposure=float(block["max_total_exposure"]),
        max_positions=int(block["max_positions"]),
        max_order_value=float(block["max_order_value"]),
        max_order_shares=float(block["max_order_shares"]),
        max_orders_per_minute=int(block["max_orders_per_minute"]),
        max_daily_loss=float(block["max_daily_loss"]),
        max_drawdown_pct=float(block["max_drawdown_pct"]),
        max_price_deviation_pct=float(block["max_price_deviation_pct"]),
        max_data_staleness_seconds=float(block["max_data_staleness_seconds"]),
        dedup_window_seconds=float(block["dedup_window_seconds"]),
        allowed_assets=set(block.get("allowed_assets") or []),
        blocked_assets=set(block.get("blocked_assets") or []),
        execution_mode=mode,
        kill_switch_enabled=bool(block.get("kill_switch_enabled", False)),
        allow_reducing_risk_when_killed=bool(block.get("allow_reducing_risk_when_killed", True)),
        halt_on_reducing_risk_failure=bool(block.get("halt_on_reducing_risk_failure", True)),
        fail_on_reconciliation_mismatch=bool(block.get("fail_on_reconciliation_mismatch", True)),
        state_file=str((cfg.state_dir / Path(block["state_file"]).name).resolve()),
        journal_file=str((cfg.state_dir / Path(block["journal_file"]).name).resolve())
        if block.get("journal_file")
        else None,
        fail_on_journal_error=bool(block.get("fail_on_journal_error", True)),
    )


class ForeignPositionError(RuntimeError):
    """Kill criterion (d): a position exists that no bot on this account can explain."""


async def reconcile_async(safe, broker: MT5Broker, *, magic: int) -> dict[str, Any]:
    """Compare **both** views against the persisted state, as the criterion requires.

    * ``MT5Broker.positions`` (filtered on this bot's magic) against ``SafeBroker``'s persisted
      ``RiskState`` / ``VirtualPortfolio`` (``preview_reconciliation_async``): a difference here
      is *this bot's* unexplained position and is kill criterion (d).
    * ``MT5Broker.all_positions()`` (every magic) against the allocation table: a position
      carrying a magic no bot owns and that is not the reserved legacy magic is an unexplained
      position on the shared account, and is escalated the same way.

    Raises :class:`ForeignPositionError` on the second case; the first is reported to the
    caller, which refuses to trade when ``fail_on_reconciliation_mismatch`` is set.
    """
    known = set(MAGIC_ALLOCATION.values()) | set(RESERVED_MAGICS)
    preview = await safe.preview_reconciliation_async()
    own = {asset: pos.quantity for asset, pos in broker.positions.items()}
    everything = broker.all_positions()
    foreign = [p for p in everything if int(p["magic"]) not in known]
    by_magic: dict[str, int] = {}
    for position in everything:
        by_magic[str(position["magic"])] = by_magic.get(str(position["magic"]), 0) + 1
    record = {
        "clean": bool(preview["clean"]) and not foreign,
        "persisted_positions": preview["persisted_positions"],
        "live_positions_this_magic": own,
        "missing_positions": preview["missing_positions"],
        "unexpected_positions": preview["unexpected_positions"],
        "quantity_mismatches": preview["quantity_mismatches"],
        "missing_pending_orders": preview["missing_pending_orders"],
        "unexpected_pending_orders": preview["unexpected_pending_orders"],
        "account_positions_by_magic": by_magic,
        "unknown_magics": sorted({int(p["magic"]) for p in foreign}),
        "magic": magic,
    }
    if foreign:
        raise ForeignPositionError(
            f"positions with unallocated magic(s) {record['unknown_magics']} are open on this "
            f"account (known: {sorted(known)}); kill criterion (d) - no trading until a person "
            "explains them"
        )
    return record


def cfd_execution_guard(cfg: DeployConfig, broker: MT5Broker, symbols: list[str], now: datetime) -> dict[str, Any]:
    """The CFD stop before kill criterion (c): bad window, wide spread, closed market.

    Kill criterion (c) is a monthly statistic. This is the per-order check that stops a fill
    in the 21:00-22:00 UTC gap (setup.yaml's ``other`` bucket, p90 4.3-9.5 bps against an
    in-session 0.63-1.28) and a fill at a spread more than the declared multiple of the
    assumed p90. The live spread comes from ``symbol_info_tick``, never from a table.
    """
    guards = cfg.cfd_guards
    reasons: list[str] = []
    for window in guards.get("blocked_execution_windows_utc") or []:
        start, end = time.fromisoformat(window["start"]), time.fromisoformat(window["end"])
        if start <= now.time() < end:
            reasons.append(
                f"execution window {window['start']}-{window['end']} UTC is blocked: {window['reason']}"
            )
    spreads: dict[str, Any] = {}
    assumed = guards.get("assumed_spread_p90_bps") or {}
    multiple = float(guards.get("max_spread_multiple_of_p90", 2.0))
    market_open = False
    for symbol in symbols:
        tick = broker.tick(symbol)
        if tick is None or not tick.bid or not tick.ask:
            spreads[symbol] = {"spread_bps": None, "note": "no tick"}
            continue
        mid = (float(tick.bid) + float(tick.ask)) / 2
        spread_bps = (float(tick.ask) - float(tick.bid)) / mid * 10_000.0
        limit = assumed.get(symbol)
        entry = {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread_bps": spread_bps,
            "assumed_p90_bps": limit,
            "limit_bps": None if limit is None else limit * multiple,
            "tick_time": int(getattr(tick, "time", 0)),
        }
        if limit is not None and spread_bps > limit * multiple:
            entry["blocked"] = True
            reasons.append(
                f"{symbol}: live spread {spread_bps:.2f} bps above {multiple:g}x the assumed "
                f"p90 {limit:.2f} bps"
            )
        spreads[symbol] = entry
        market_open = True
    if guards.get("refuse_when_market_closed") and not market_open:
        reasons.append("no pair is quoting: the FX week is closed")
    return {"ok": not reasons, "reasons": reasons, "spreads": spreads, "checked_at": now.isoformat()}


def unit_values(
    broker: MT5Broker, prices: dict[str, float], *, account_currency: str | None = None
) -> dict[str, float]:
    """Account-currency value of ONE unit of each pair, from ``symbol_info`` and the price.

    Three cases, and the first two need no second quote at all - which matters, because a
    conversion that depends on another instrument's live tick is a dependency that fails at
    the worst moment:

    * ``currency_profit == account currency`` (``EURUSD``, ``GBPUSD``, ``AUDUSD`` on a USD
      account): one unit is worth the price, 1.16 USD.
    * ``currency_base == account currency`` (``USDJPY``, ``USDCAD``): one unit of the base IS
      one unit of account currency, so the value is exactly **1.00**, whatever the quote says.
      This is the case that makes a naive ``notional / price`` conversion **into MT5 lots**
      wrong by a factor of the quote: an MT5 lot is 100,000 units of the BASE currency, so
      0.10 lots of ``USDJPY`` is 10,000 USD and not 10,000 / 154 USD.
    * a genuine cross (``EURJPY``): neither leg is the account currency, so the terminal is
      asked through :meth:`MT5Broker.unit_value_account`, and a pair whose cross is not quoted
      is **dropped** rather than sized on a guess.

    All five pairs of this bot's universe fall into the first two cases.
    """
    account_ccy = account_currency or (broker.account_snapshot.get("currency") or "USD")
    out: dict[str, float] = {}
    for symbol, price in prices.items():
        if not price:
            continue
        info = broker.symbol_info(symbol)
        if info is None:
            logger.error("%s: not in Market Watch; cannot value one unit", symbol)
            continue
        if str(info.currency_profit) == account_ccy:
            out[symbol] = float(price)
            continue
        if str(info.currency_base) == account_ccy:
            out[symbol] = 1.0
            continue
        try:
            value = float(broker.unit_value_account(symbol, float(price)))
        except Exception as exc:  # noqa: BLE001 - a missing cross must not size a position
            logger.error("%s: cannot value one unit in account currency (%s)", symbol, exc)
            continue
        if value > 0:
            out[symbol] = value
    return out


def target_units(weights: dict[str, float], unit_value: dict[str, float], equity: float) -> dict[str, float]:
    """Signed units of base currency per pair; the adapter turns them into lots.

    ``weight x equity`` is a notional in account currency, and dividing by the
    **account-currency value of one unit** gives units of the base currency, which is what
    ``execution.share_type: integer`` means.

    Note what this is NOT: dividing by the raw quote. On the MT5 side that is right only when
    the quote currency is the account currency, because a lot is 100,000 units of the **base**
    currency: 0.10 lots of ``USDJPY`` is 10,000 USD of exposure, so ``notional / 154`` units
    would be 154x too small a position.

    The research engine is a different, self-consistent world and is **not** wrong here. It
    treats ``price`` as the account-currency value of one unit on both sides - it sizes
    ``target_notional / price`` (``ml4t/backtest/preopen.py:800``) and it values
    ``quantity * price`` (``broker.py:2087``, ``preopen.py:793``) - so a USDJPY leg holds few
    "units" but the right notional and the right P&L. Measured on the generation-2 winner's
    ``fills.parquet``, the median leg notional is 110,277 USD on USDJPY against 100,028 on
    EURUSD. So the engine's ``quantity`` and an MT5 ``quantity`` are simply different
    quantities, not the same number computed two ways, and the parity gate compares baskets
    and sides rather than raw unit counts.
    """
    return {
        symbol: float(weight) * float(equity) / float(unit_value[symbol])
        for symbol, weight in weights.items()
        if unit_value.get(symbol)
    }


def min_tradable_notional(broker: MT5Broker, symbols: list[str], prices: dict[str, float]) -> dict[str, Any]:
    """``volume_min x trade_contract_size`` per pair, and what it costs in account currency.

    An operational fact a dry run has to make measurable: 0.01 lots x 100,000 = 1,000 units,
    about 1,163 USD of EURUSD and 1,000 USD of USDJPY (one unit of USDJPY is one US dollar).
    Allocate less than that per leg and the loop produces 0-lot orders that ``normalize_lot``
    rejects - silently, unless this is printed.
    """
    out: dict[str, Any] = {}
    for symbol in symbols:
        info = broker.symbol_info(symbol)
        if info is None:
            out[symbol] = {"note": f"{to_market_watch(symbol)} not in Market Watch"}
            continue
        units = float(info.volume_min) * float(info.trade_contract_size)
        price = prices.get(symbol)
        value = (
            None if price is None else unit_values(broker, {symbol: float(price)}).get(symbol)
        )
        out[symbol] = {
            "volume_min": float(info.volume_min),
            "volume_step": float(info.volume_step),
            "contract_size": float(info.trade_contract_size),
            "min_units": units,
            "unit_value_account": value,
            "min_notional_account": None if value is None else units * value,
        }
    return out


def capital_viability(
    cfg: DeployConfig,
    broker: MT5Broker,
    prices: dict[str, float],
    *,
    account_equity: float | None = None,
) -> dict[str, Any]:
    """Can this bot's declared allocation actually place an order on this account?

    Two ways it cannot, and both are silent unless something says so out loud:

    * **The lot floor.** ``volume_min x trade_contract_size`` is 1,000 units on these pairs,
      about 1,163 USD of EURUSD notional at the price measured on 2026-09-07. A design leg
      below that rounds to zero and ``normalize_lot`` rejects it, so the cycle would produce a
      basket of refusals that reads like a broker problem.
    * **The account.** An allocation larger than the account's equity is a declaration the
      account cannot honour; the margin guard would refuse the second leg, not the first.

    Returns the arithmetic either way, and the caller blocks staging when ``viable`` is false.
    """
    top_k = int(cfg.model_block.get("top_k", 1))
    leg_notional = cfg.allocated / max(top_k, 1)
    per_pair: dict[str, Any] = {}
    reasons: list[str] = []
    values = unit_values(broker, prices)
    for symbol in sorted(prices):
        info = broker.symbol_info(symbol)
        value = values.get(symbol)
        if info is None or not value:
            per_pair[symbol] = {"note": "no symbol_info, no price, or no cross to value a unit"}
            continue
        contract = float(info.trade_contract_size)
        step, minimum = float(info.volume_step), float(info.volume_min)
        units = leg_notional / value
        lots = units / contract
        entry = {
            "leg_notional": leg_notional,
            "unit_value_account": value,
            "leg_units": units,
            "leg_lots": lots,
            "volume_min": minimum,
            "volume_step": step,
            "min_notional_account": minimum * contract * value,
            "rounds_to_zero": lots < minimum,
        }
        if entry["rounds_to_zero"]:
            reasons.append(
                f"{symbol}: a {leg_notional:,.0f} {cfg.risk['capital']['currency']} leg is "
                f"{lots:.4f} lots, below volume_min {minimum}"
            )
        per_pair[symbol] = entry
    declared_min = float(cfg.risk["capital"].get("min_viable_allocated", 0.0))
    if declared_min and cfg.allocated < declared_min:
        reasons.append(
            f"capital.allocated {cfg.allocated:,.0f} is below the declared "
            f"min_viable_allocated {declared_min:,.0f}"
        )
    if account_equity is not None and cfg.allocated > float(account_equity):
        reasons.append(
            f"capital.allocated {cfg.allocated:,.0f} exceeds the account equity "
            f"{float(account_equity):,.2f}"
        )
    return {
        "allocated": cfg.allocated,
        "currency": cfg.risk["capital"]["currency"],
        "top_k": top_k,
        "min_viable_allocated": declared_min,
        "account_equity": account_equity,
        "per_pair": per_pair,
        "viable": not reasons,
        "reasons": reasons,
    }


async def stage_orders(
    cfg: DeployConfig,
    *,
    weights: dict[str, float],
    prices: dict[str, float],
    armed: bool,
    mt5: Any,
    account_guard: AccountGuard,
    blocked_reasons: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Open the ``SafeBroker`` session, reconcile, guard, then stage (or refuse to stage).

    ``armed=False`` (the default) forces ``shadow_mode``: ``SafeBroker`` books the fills into
    its ``VirtualPortfolio`` and the adapter is never asked to send anything. Every refusal is
    recorded per leg so the run record shows what *would* have been sent.
    """
    from ml4t.live import SafeBroker

    now = now or datetime.now(UTC)
    broker = MT5Broker(
        magic=cfg.magic,
        execution_mode=str(cfg.risk["execution"]["mode"]),
        armed_live=False,
        max_margin_fraction=float(cfg.risk["execution"]["max_margin_fraction"]),
        deviation_points=int(cfg.risk["execution"]["deviation_points"]),
        close_deviation_points=int(cfg.risk["execution"]["close_deviation_points"]),
        min_lot_policy=str(cfg.risk["execution"]["min_lot_policy"]),
        comment=BOT_ID,
        mt5=mt5,
    )
    risk_config = build_live_risk_config(cfg, armed=armed)
    safe = SafeBroker(broker, risk_config)
    record: dict[str, Any] = {
        "magic": cfg.magic,
        "execution_mode": str(cfg.risk["execution"]["mode"]),
        "armed": bool(armed),
        "shadow_mode": bool(risk_config.shadow_mode),
        "orders_can_reach_broker": bool(armed and not risk_config.shadow_mode),
        "state_file": risk_config.state_file,
        "max_orders_per_minute": risk_config.max_orders_per_minute,
    }
    reasons = list(blocked_reasons or [])
    try:
        await safe.connect()
        broker.assert_paper_trading()
        record["account"] = broker.account_summary()

        snapshot = read_account_snapshot(broker)
        account_guard.start_day(snapshot)
        account_ok = account_guard.check(snapshot)
        record["account_snapshot"] = snapshot.as_dict()
        record["account_breakers"] = account_guard.to_record()

        try:
            record["reconciliation"] = await reconcile_async(safe, broker, magic=cfg.magic)
        except ForeignPositionError as exc:
            record["reconciliation"] = {"clean": False, "error": str(exc)}
            reasons.append(str(exc))

        guard = cfd_execution_guard(cfg, broker, sorted(weights), now)
        record["cfd_guard"] = guard
        record["min_tradable"] = min_tradable_notional(broker, sorted(weights), prices)
        viability = capital_viability(cfg, broker, prices, account_equity=snapshot.equity)
        record["capital_viability"] = viability
        if not viability["viable"]:
            reasons.extend(viability["reasons"])

        if account_guard.is_halted():
            reasons.append(f"account halt file present at {account_guard.halt_file}")
        if not account_ok:
            reasons.append(f"account breaker(s) open: {account_guard.manager.open_breakers()}")
        rec = record.get("reconciliation") or {}
        if not rec.get("clean", False) and risk_config.fail_on_reconciliation_mismatch:
            reasons.append("reconciliation is not clean (kill criterion (d))")
        if not guard["ok"]:
            reasons.extend(guard["reasons"])
        if not armed:
            reasons.append("not armed: --arm was not given in this session")
        record["blocked_by"] = reasons

        units = target_units(weights, unit_values(broker, prices), cfg.allocated)
        legs = [
            await _stage_leg(
                safe,
                broker,
                symbol=symbol,
                units=units.get(symbol, 0.0),
                price=prices.get(symbol),
                weight=weights[symbol],
                blocked=bool(reasons),
            )
            for symbol in sorted(weights)
        ]
        record["legs"] = legs
        record["intended_basket"] = sorted(leg["symbol"] for leg in legs)
        record["attempted_basket"] = sorted(
            leg["symbol"] for leg in legs if leg["status"] in ("submitted", "rejected")
        )
        record["accepted_basket"] = sorted(leg["symbol"] for leg in legs if leg["status"] == "submitted")
        record["failed_basket"] = sorted(leg["symbol"] for leg in legs if leg["status"] == "rejected")
    finally:
        try:
            safe.close_persistence()
        finally:
            await safe.disconnect()
    return record


async def _stage_leg(
    safe,
    broker: MT5Broker,
    *,
    symbol: str,
    units: float,
    price: float | None,
    weight: float,
    blocked: bool,
) -> dict[str, Any]:
    """One leg. Lots come from ``symbol_info``, never from a table."""
    from ml4t.backtest import OrderSide, OrderStatus

    from bots._shared.mt5_broker import normalize_lot

    info = broker.symbol_info(symbol)
    contract = float(getattr(info, "trade_contract_size", 0.0)) if info is not None else 0.0
    lots = (
        normalize_lot(abs(units) / contract, info, min_lot_policy=broker.min_lot_policy)
        if contract
        else 0.0
    )
    leg: dict[str, Any] = {
        "symbol": symbol,
        "weight": float(weight),
        "ref_price": price,
        "units": float(units),
        "contract_size": contract,
        "lots": lots,
        "side": "buy" if units > 0 else "sell",
    }
    if contract == 0.0:
        leg["status"] = "unknown_symbol"
        return leg
    if price is None or price <= 0:
        leg["status"] = "no_ref_price"
        return leg
    if lots <= 0:
        leg["status"] = "rounds_to_zero"
        leg["detail"] = (
            f"{abs(units):,.0f} units = {abs(units) / contract:.4f} lots, below volume_min "
            f"{getattr(info, 'volume_min', None)}"
        )
        return leg
    if blocked:
        leg["status"] = "dry_run"
        return leg
    # Prime the staleness / price-deviation guard from a live tick before every order
    # (mt5-exness-broker.md section 3).
    tick = broker.tick(symbol)
    reference = (float(tick.bid) + float(tick.ask)) / 2 if tick is not None else float(price)
    safe.record_market_snapshot(symbol, reference)
    order = await safe.submit_order_async(
        asset=symbol,
        quantity=lots * contract,
        side=OrderSide.BUY if units > 0 else OrderSide.SELL,
    )
    leg["reference_price"] = reference
    leg["status"] = "submitted" if order.status is not OrderStatus.REJECTED else "rejected"
    leg["order_status"] = order.status.value
    leg["order_id"] = order.order_id
    leg["filled_quantity"] = float(order.filled_quantity)
    leg["rejection_reason"] = order.rejection_reason
    return leg


# =======================================================================================
# The cycle
# =======================================================================================
def run_cycle(
    *,
    model_selector: str | None = MODEL_SELECTOR,
    prediction_selector: str | None = None,
    arm: bool = SUBMIT_ORDERS,
    refresh: bool = REFRESH_DATA,
    as_of: date | None = None,
    live_start: date | None = None,
    risk_config: Path | str | None = None,
    state_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    mt5: Any = None,
    account_limits: Path | str | None = None,
    halt_file: Path | str | None = None,
    case_study: str = BOT_ID,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One deployment cycle. Returns the run record it writes."""
    started = datetime.now(UTC)
    cfg = load_deploy_config(risk_config, case_study=case_study, state_dir=state_dir)
    run: dict[str, Any] = {
        "bot_id": BOT_ID,
        "run_started_at": started.isoformat(),
        "risk_config": str(cfg.risk_path),
        "case_study_dir": str(cfg.case_dir),
        "magic": cfg.magic,
        # The three facts a reader of this file must not have to infer.
        "signal_status": "no_deployable_signal",
        "deployable": False,
        "not_deployable_because": (
            "phase 5 gate not met at K = 3,204: no spec of the three cohorts (long-only top-k, "
            "long-short sleeves, per-pair time-series percentile; 0 of 1,068 in each) is "
            "significant at 0.95 on the active return over the 1/N long book, and not one "
            "reaches an active DSR of 0.5; phase 6 is not opened and the holdout is unscored "
            "(BOT.md phase table)"
        ),
        "trial_count_K": int(cfg.model_block.get("trial_count_at_deployment", 0)),
        "holdout": {
            "start": str(cfg.holdout_start),
            "end": str(cfg.setup["evaluation"]["holdout_end"]),
            "scored": cfg.holdout_scored,
        },
        "steps": {},
    }

    # 1 ---------------------------------------------------------------------------------
    run["steps"]["1_refresh"] = refresh_data(cfg, do_refresh=refresh, mt5=mt5, data_dir=data_dir)

    # 3a: resolve the model first, because in plumbing mode it fixes the decision date -----
    choice = resolve_model(
        cfg,
        model_selector if model_selector is not None else cfg.model_block.get("training_hash"),
        prediction_hash=prediction_selector,
    )
    run["model"] = choice.as_dict()
    if choice.mode == "plumbing":
        run["signal_status"] = "plumbing_only"

    # 2 ---------------------------------------------------------------------------------
    predictions: pl.DataFrame | None = None
    if choice.mode in ("none", "plumbing"):
        predictions = read_registered_predictions(cfg, choice)
        as_of = as_of or predictions["timestamp"].max()
    panel = recompute_features(cfg, as_of=as_of, data_dir=data_dir)
    bar = decision_bar_check(cfg, panel)
    run["steps"]["2_features"] = {**panel.record, "decision_bar": bar}
    run["decision_date"] = str(panel.decision_date)
    run["decision_bar_lag_minutes"] = bar["max_lag_minutes"]
    run["holdout_window_touched"] = panel.record["holdout_window_touched"]
    if not bar["ok"]:
        run["steps"]["2_features"]["no_trade_day"] = record_no_trade_day(
            cfg, panel.decision_date, "; ".join(bar["reasons"])
        )

    # 3 / 4 / 5 -------------------------------------------------------------------------
    if choice.mode == "refit":
        model = retrain(cfg, panel, choice)
        run["steps"]["3_retrain"] = {"training_metadata": model.metadata}
        run["steps"]["4_persist"] = {
            "artefact_dir": str(persist(model, state_dir=cfg.state_dir, decision_date=panel.decision_date))
        }
        window_start = live_start or _first_session_after(panel, model.metadata["train_window"]["train_end"])
        predictions = predict(model, panel, live_start=window_start)
    else:
        run["steps"]["3_retrain"] = {
            "skipped": True,
            "reason": (
                "no refit in this mode: predictions are READ from the registered validation "
                f"prediction set {choice.prediction_hash} (population "
                f"{cfg.model_block.get('plumbing_prediction_population')}). No refit, no data "
                "extension, no holdout read."
            ),
        }
        run["steps"]["4_persist"] = {"skipped": True, "reason": "nothing was fitted"}
        window_start = live_start or predictions["timestamp"].min()
    latest = predictions.filter(
        pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date)
    ).sort("score", descending=True)
    run["steps"]["5_predict"] = {
        "source": "refit" if choice.mode == "refit" else f"registry:{choice.prediction_hash}",
        "window_start": str(window_start),
        "n_predictions": int(predictions.height),
        "n_dates": int(predictions["timestamp"].n_unique()),
        "latest_cross_section": latest.to_dicts(),
    }

    # 6 ---------------------------------------------------------------------------------
    parity = replay_parity(cfg, panel, predictions, live_start=window_start, label=choice.label)
    strategy = parity.pop("_strategy")
    run["steps"]["6_parity"] = parity

    # 7 ---------------------------------------------------------------------------------
    if choice.mode == "none":
        run["steps"]["7_stage"] = {
            "skipped": True,
            "reason": (
                "stopped after the parity step: no deployable signal exists. Use "
                "--model latest-complete to exercise the staging path as plumbing, or "
                "--model <training_hash> to deploy a specific registered run."
            ),
        }
        run["ended_after_step"] = 6
        return _write_run_record(run, cfg.state_dir, panel.decision_date)

    weights = strategy.target_weights(panel.decision_date) or {}
    prices = {
        row["symbol"]: float(row["close"])
        for row in panel.prices.filter(
            pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date)
        ).to_dicts()
    }
    # A fake terminal must never touch the shared account-monitor directory: its account is
    # synthetic, and a high-water mark or a halt file written from it would be read by the
    # OTHER bots as a fact about the real login. (Measured: a fake-terminal smoke wrote
    # peak_equity 10,000 next to a real account holding 898.56, which reads as a 91 % drawdown
    # and halts everything.) Unless a path is given, a fake run keeps both files in its own
    # state directory.
    if halt_file is None and _is_fake_terminal(mt5):
        halt_file = cfg.state_dir / "account_halt.json"
    guard = AccountGuard(
        load_account_limits(account_limits), halt_file=halt_file, bot_id=BOT_ID
    )
    blocked: list[str] = []
    if not bar["ok"]:
        blocked.append("no decision bar: " + "; ".join(bar["reasons"]))
    if choice.mode == "plumbing":
        blocked.append("plumbing run: replayed validation predictions are not a live signal")
    armed = bool(arm) and choice.mode == "refit"
    staging = asyncio.run(
        stage_orders(
            cfg,
            weights=weights,
            prices=prices,
            armed=armed,
            mt5=mt5,
            account_guard=guard,
            blocked_reasons=blocked,
            now=now,
        )
    )
    staging["intended_weights"] = weights
    staging["parity_agrees_with_last_rebalance"] = _parity_agrees(parity, weights, panel.decision_date)
    run["steps"]["7_stage"] = staging
    run["ended_after_step"] = 7
    return _write_run_record(run, cfg.state_dir, panel.decision_date)


def _is_fake_terminal(mt5: Any) -> bool:
    """True for ``bots._shared.testing.fake_mt5.FakeMT5`` (and subclasses of it)."""
    return mt5 is not None and any(
        cls.__module__.endswith("testing.fake_mt5") and cls.__name__ == "FakeMT5"
        for cls in type(mt5).__mro__
    )


def _first_session_after(panel: FeaturePanel, train_end: Any) -> date:
    end = date.fromisoformat(str(train_end))
    sessions = [d for d in sorted(panel.prices["timestamp"].unique().to_list()) if d > end]
    if not sessions:
        raise ValueError("no session after the training window; nothing to predict")
    return sessions[0]


def _parity_agrees(parity: dict[str, Any], weights: dict[str, float], decision_date: date) -> bool | None:
    """Does the staged basket equal the basket the offline replay chose on the same date?"""
    last = parity.get("last_rebalance")
    if not last or last["date"] != str(decision_date):
        return None
    staged_long = sorted(s for s, w in weights.items() if w > 0)
    staged_short = sorted(s for s, w in weights.items() if w < 0)
    return staged_long == last["long"] and staged_short == last["short"]


def _write_run_record(run: dict[str, Any], state_root: Path, decision_date: date) -> dict[str, Any]:
    run["run_finished_at"] = datetime.now(UTC).isoformat()
    target = Path(state_root) / str(decision_date)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(json.dumps(run, indent=2, default=str))
    run["run_record"] = str(path)
    return run


# =======================================================================================
# CLI
# =======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exness_fx_d1 deployment loop",
        description="Seven-step deployment cycle. Dry run only; no deployable signal exists.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_SELECTOR,
        help=(
            "'none' (default: steps 1-6 off the registered validation predictions, then stop), "
            "'latest-complete' (the same plus the staging path, labelled plumbing), or a "
            "registered training hash to refit and run all seven steps"
        ),
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="replay a specific registered validation prediction set instead of the newest one",
    )
    parser.add_argument("--arm", action="store_true", help="arm order sending for THIS session (a person types this)")
    parser.add_argument("--refresh", action="store_true", help="pull new MT5 bars first (Windows terminal only)")
    parser.add_argument("--as-of", default=None, help="decide as of this session date (YYYY-MM-DD)")
    parser.add_argument("--live-window-start", default=None, help="first session of the parity replay (YYYY-MM-DD)")
    parser.add_argument("--risk-config", default=None, help="override deploy/risk_config.yaml")
    parser.add_argument("--state-dir", default=None, help="override deploy/state/")
    parser.add_argument("--data-dir", default=None, help="override ML4T_DATA_PATH/mt5")
    parser.add_argument(
        "--halt-file",
        default=None,
        help="override the shared account halt file (a smoke run should not touch the real one)",
    )
    parser.add_argument(
        "--fake-mt5",
        action="store_true",
        help="run against bots/_shared/testing/fake_mt5.FakeMT5 instead of a terminal",
    )
    return parser


def make_fake_terminal(cfg: DeployConfig):
    """A ``FakeMT5`` carrying this bot's Market Watch names, for a terminal-free dry run."""
    from bots._shared.testing.fake_mt5 import make_fake

    fake = make_fake([to_market_watch(s) for s in cfg.universe], n_days=5, server_offset_minutes=0)
    fake.initialize()
    return fake


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("ml4t").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    cfg = load_deploy_config(args.risk_config, state_dir=args.state_dir)
    run = run_cycle(
        model_selector=args.model,
        prediction_selector=args.predictions,
        arm=args.arm,
        refresh=args.refresh,
        as_of=date.fromisoformat(args.as_of) if args.as_of else None,
        live_start=date.fromisoformat(args.live_window_start) if args.live_window_start else None,
        risk_config=args.risk_config,
        state_dir=args.state_dir,
        data_dir=args.data_dir,
        halt_file=args.halt_file,
        mt5=make_fake_terminal(cfg) if args.fake_mt5 else None,
    )
    header = {k: v for k, v in run.items() if k != "steps"}
    print(json.dumps(header, indent=2, default=str))
    for name, step in run["steps"].items():
        trimmed = {k: v for k, v in step.items() if k not in ("tape", "latest_cross_section", "coverage", "per_symbol")}
        print(f"\n[{name}]\n{json.dumps(trimmed, indent=2, default=str)[:2500]}")
    print(f"\nSIGNAL STATUS: {run['signal_status']} — deployable={run['deployable']}")
    print(run["not_deployable_because"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
