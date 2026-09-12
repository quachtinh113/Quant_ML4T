"""``exness_usidx_sess`` deployment loop — DRY RUN ONLY. NO DEPLOYABLE SIGNAL EXISTS.

======================================================================================
EVIDENCE BOUNDARY — THE FIRST THING THIS MODULE DEFENDS
======================================================================================
* **Phase 5 ran in full and found 0 survivors of 1,403 scored specs at K = 1,780.** Not one
  trial reaches a Deflated Sharpe Ratio of 0.50 on the active return over a 1/N long book, let
  alone the drafted 0.95. At K = 1,780 the expected maximum annualised Sharpe from luck alone
  is +4.006 and the best trial reaches +1.587.
* **Phase 6 is not opened.** **Phase 7 is sealed**: the declared holdout 2026-03-01 ..
  2026-08-31 has never been read by any stage.
* The hypothesis, the kill criteria and the DSR benchmark are all a **draft awaiting the
  user**.

THIS FILE IS NOT PERMISSION TO TRADE. It exists because a parity harness and a safety layer
test the **code path**, not the edge (Ch25 s25.1), and because the mentor's ruling of
2026-09-08 authorised exactly that and nothing more. Running it cannot place an order:
``--arm`` is refused while the kill criteria are a draft, ``LiveRiskConfig.execution_mode`` is
``shadow`` (orders are booked into a ``VirtualPortfolio`` and never reach the adapter) and
``MT5Broker.execution_mode`` is ``paper`` (``assert_paper_trading`` refuses a real account).
Precedent followed exactly: ``bots/exness_fx_d1/BOT.md`` phases 8 and 9.
======================================================================================

THE SEVEN STEPS OF ``25_live_trading/02_etfs_deployment_loop.py``, on MT5 instead of Alpaca,
with ``11_fx_deployment_loop.py`` as the CFD worked example:

1. **Refresh** the MT5 H1 and D1 history through ``bots/_shared/mt5_loader`` (opt-in, needs a
   Windows terminal); otherwise the existing parquet is read and its coverage reported.
2. **Recompute** the session panel and the features through
   ``case_studies/exness_usidx_sess/_features.py`` — ``session_panel`` and ``build_features``,
   the same functions ``01``, ``02``, ``03``, the registered price loader and
   ``tests/test_lookahead.py`` call. One feature implementation, five consumers.
3. **Retrain** the model a ``--model <training_hash>`` names, from the preset the registry row
   points at. No hyperparameter is typed in this file.
4. **Persist** to ``deploy/state/<spec>/<decision date>/``; deployment artefacts never enter
   the research registry.
5. **Predict** the sessions of the live window.
6. **Replay** the parity tape — see PARITY below. This is the step that has to exist before a
   survivor does.
7. **Stage** the basket through ``SafeBroker(MT5Broker(magic=260902), LiveRiskConfig)``.

PARITY — AND WHY IT IS **NOT** AN ``ml4t.backtest.Engine`` REPLAY HERE
---------------------------------------------------------------------
``exness_fx_d1`` replays its window through ``ml4t.backtest.Engine`` and compares the tape.
This bot cannot: BOT.md records, with the mechanism, that **the engine cannot produce a valid
backtest for it** — prediction sets are keyed on the session DATE while the registered price
grid is HOURLY, so a ``next_bar`` fill lands 13-15 hours BEFORE the decision; and the engine
holds continuously between fills while both specs are flat for part of their cycle (measured
median holding period 2,088 hours against a six-hour label). Replaying through the engine here
would reproduce that lookahead inside the deployment package and call it a reference.

So parity is done the way ``25_live_trading/08`` defines it — **the five stages of the
pipeline, compared research-against-live**, which is what parity means and which needs no
engine at all:

    1 data        the bars the research panel was built from vs the bars the loop reads now
    2 features    ``features/financial.parquet`` vs ``build_features`` recomputed here
    3 predictions the registered validation prediction set vs the model re-run here
    4 sizing      ``signals.build_target_weights_from_config`` on both — the SAME function
                  ``13_backtest`` sends the engine, so the signal semantics are identical
    5 orders      weights -> index units -> lots through ``symbol_info``: the stage where the
                  fractional research book meets ``volume_min`` and legs are REJECTED

Stage 5 is where this bot's largest parity gap lives and the harness measures it rather than
describing it: ``execution.share_type`` is ``fractional`` in the engine and ``volume_min`` is
0.14 lot on US500 / 0.05 on USTEC at the adapter, so a target below the floor is **rejected,
not scaled down**, and at the account's measured 897 USD of equity every leg is below it.

WHAT MAY RUN TODAY
------------------
``--model none`` (default)
    Steps 1, 2, 5, 6. Predictions are **read** from a registered *validation* prediction set:
    no refit, no data extension, no holdout read. The loop stops after parity and says why.
``--model latest-complete``
    The same, then step 7 in dry run so the staging path is exercised end to end. Stamped
    ``signal_status: plumbing_only``.
``--model <training_hash>``
    Refit that registered run on the clamped window and run all seven steps. Still a dry run.

Run from the repository root::

    uv run python bots/exness_usidx_sess/deploy/deployment_loop.py --spec intraday --fake-mt5
    uv run python bots/exness_usidx_sess/deploy/deployment_loop.py \
        --spec overnight --model latest-complete --fake-mt5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import lightgbm  # noqa: F401  # GBM libraries import before scikit-learn (ml4t skill rule 8)
import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from bots._shared import MAGIC_ALLOCATION, RESERVED_MAGICS  # noqa: E402
from bots._shared.monitor import (  # noqa: E402
    AccountGuard,
    load_account_limits,
    read_account_snapshot,
)
from bots._shared.mt5_broker import MT5Broker, normalize_lot  # noqa: E402
from bots._shared.mt5_loader import load_mt5_bars, to_market_watch  # noqa: E402
from bots.exness_usidx_sess.deploy import schedule as sched  # noqa: E402
from bots.exness_usidx_sess.monitor import circuit_breakers as cb  # noqa: E402

logger = logging.getLogger("exness_usidx_sess.deploy")

# ---------------------------------------------------------------------------------------
# Parameters block (the module equivalent of a notebook's `tags=["parameters"]` cell).
# Production defaults, overridable only from the command line — never edited to true here.
# 25_live_trading/02:121 (SUBMIT_PAPER_ORDERS = False) and 11_fx_deployment_loop.py:134.
# ---------------------------------------------------------------------------------------
SUBMIT_ORDERS = False  # explicit opt-in per session; the default is an offline dry run
REFRESH_DATA = False  # pulling new bars needs a Windows terminal and is not reproducible
MODEL_SELECTOR = "none"  # no deployable signal exists; see the module docstring
SPEC = "intraday"  # one book per run: the two specs never share a workspace or a registry

BOT_ID = "exness_usidx_sess"
CASE_STUDY = "exness_usidx_sess"
DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_RISK_CONFIG = DEPLOY_DIR / "risk_config.yaml"
STATE_DIR = DEPLOY_DIR / "state"


class NotReadyError(RuntimeError):
    """A step cannot run because an artefact it needs does not exist. It never fakes one."""


class EvidenceBoundaryError(RuntimeError):
    """Something tried to read, or report on, the unscored holdout."""


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
    spec: str

    # -- shortcuts -------------------------------------------------------------------
    @property
    def magic(self) -> int:
        return int(self.risk["magic"])

    @property
    def symbols(self) -> list[str]:
        return sorted(self.setup["universe"]["symbols"])

    @property
    def allocated(self) -> float:
        return float(self.risk["capital"]["allocated"])

    @property
    def min_viable_allocated(self) -> float:
        return float(self.risk["capital"]["min_viable_allocated"])

    @property
    def live_risk_block(self) -> dict[str, Any]:
        return self.risk["live_risk_config"]

    @property
    def cfd_guards(self) -> dict[str, Any]:
        return self.risk["cfd_guards"]

    @property
    def decision_block(self) -> dict[str, Any]:
        return self.risk["decision"]

    @property
    def model_block(self) -> dict[str, Any]:
        return self.risk["model"]

    @property
    def boundary(self) -> dict[str, Any]:
        return self.risk["evidence_boundary"]

    @property
    def holdout_start(self) -> date:
        return date.fromisoformat(str(self.boundary["holdout_start"]))

    @property
    def holdout_end(self) -> date:
        return date.fromisoformat(str(self.boundary["holdout_end"]))

    @property
    def holdout_scored(self) -> bool:
        """True only when the holdout stages have written their marker. A flag cannot do it."""
        marker = self.risk_path.parent / Path(str(self.boundary["holdout_scored_marker"])).name
        alt = DEPLOY_DIR / str(self.boundary["holdout_scored_marker"])
        return marker.exists() or alt.exists()

    @property
    def label(self) -> str:
        return f"fwd_ret_{self.spec}"

    def evidence_stamp(self) -> dict[str, Any]:
        """Stamped into EVERY run record. A live number read without this is meaningless."""
        return {
            "bot_id": BOT_ID,
            "spec": self.spec,
            "phase5_survivors": self.boundary.get("phase5_survivors"),
            "phase5_specs_scored": self.boundary.get("phase5_specs_scored"),
            "trial_count_K": self.boundary.get("phase5_trial_count"),
            "phase6_opened": self.boundary.get("phase6_opened"),
            "holdout_window": [str(self.holdout_start), str(self.holdout_end)],
            "holdout_scored": self.holdout_scored,
            "kill_criteria_approved": not bool(
                self.risk["breakers"].get("pending_user_approval", True)
            ),
            "live_trading_permitted": False,
            "note": (
                "0 survivors of 1,403 scored specs at K = 1,780; phase 6 not opened; holdout "
                "sealed. This package is a DRY RUN of the code path and is not evidence of an "
                "edge or permission to trade."
            ),
        }


def load_deploy_config(
    risk_config: Path | str | None = None,
    *,
    spec: str = SPEC,
    state_dir: Path | str | None = None,
) -> DeployConfig:
    """Load and CROSS-CHECK the two declarations. A disagreement raises, it never wins."""
    from utils.paths import get_case_study_dir

    risk_path = Path(risk_config or DEFAULT_RISK_CONFIG)
    risk = yaml.safe_load(risk_path.read_text())
    case_dir = Path(get_case_study_dir(str(risk["case_study"])))
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())

    if spec not in sched.SPECS:
        raise ValueError(f"spec must be one of {sched.SPECS}, not {spec!r}")

    # -- the checks that stop a config drifting away from the research declaration -----
    allocated = MAGIC_ALLOCATION.get(str(risk["bot_id"]))
    if allocated is None:
        raise KeyError(
            f"{risk['bot_id']!r} has no entry in bots/_shared/__init__.py::MAGIC_ALLOCATION"
        )
    if int(risk["magic"]) != allocated:
        raise ValueError(
            f"{risk_path}::magic is {risk['magic']} but MAGIC_ALLOCATION allocates "
            f"{allocated} to {risk['bot_id']}"
        )
    if int(risk["magic"]) in RESERVED_MAGICS:
        raise ValueError(f"magic {risk['magic']} is reserved: {RESERVED_MAGICS[risk['magic']]}")
    declared = sorted(risk["live_risk_config"]["allowed_assets"])
    if declared != sorted(setup["universe"]["symbols"]):
        raise ValueError(
            f"live_risk_config.allowed_assets {declared} does not match "
            f"setup.yaml::universe.symbols {sorted(setup['universe']['symbols'])}"
        )
    if str(risk["execution"]["mode"]) != "paper":
        raise ValueError(
            "execution.mode may only be 'paper' until the holdout has been scored once; "
            f"it is {risk['execution']['mode']!r}"
        )
    dec, sdec = risk["decision"], setup["decision"]
    for key in ("open_delay_minutes", "session_calendar", "session_close_tolerance_minutes",
                "trade_hours_follow_dst_of"):
        if str(dec[key]) != str(sdec[key]):
            raise ValueError(
                f"risk_config.yaml::decision.{key} = {dec[key]!r} but "
                f"setup.yaml::decision.{key} = {sdec[key]!r}; the live loop must decide at the "
                "same instant the research panel was built on"
            )
    if str(dec["server_clock"]["utc_offset_minutes"]) != str(
        sdec["server_clock"]["utc_offset_minutes"]
    ):
        raise ValueError("the two declarations disagree about the measured server offset")
    for key in ("holdout_start", "holdout_end"):
        if str(risk["evidence_boundary"][key]) != str(setup["evaluation"][key]):
            raise ValueError(
                f"evidence_boundary.{key} = {risk['evidence_boundary'][key]!r} but "
                f"setup.yaml::evaluation.{key} = {setup['evaluation'][key]!r}"
            )

    root = Path(state_dir) if state_dir else STATE_DIR
    resolved = root / spec
    resolved.mkdir(parents=True, exist_ok=True)
    return DeployConfig(
        risk=risk,
        setup=setup,
        risk_path=risk_path,
        case_dir=case_dir,
        state_dir=resolved,
        spec=spec,
    )


# =======================================================================================
# Step 1 — refresh the history
# =======================================================================================
def refresh_data(cfg: DeployConfig, *, refresh: bool) -> dict[str, Any]:
    """Read (or optionally pull) the H1 and D1 tapes and report their coverage.

    ``refresh`` needs a Windows terminal and is off by default: a deployment cycle that pulls
    new bars is not reproducible, and every dry run in this package must be.
    """
    record: dict[str, Any] = {"refreshed": False, "frequencies": {}}
    # ``mt5_loader``'s own frequency names, not MT5's timeframe constants: "daily" and "1h".
    # The two-frequency panel is this bot's own design (BOT.md, Decisions log 2026-09-08): the
    # H1 history warms only 96 sessions, so every window longer than 63 comes from the D1 file
    # joined at the last bar whose UTC day had CLOSED.
    frequencies = {
        "1h": str(cfg.setup["universe"]["history_start"]),
        "daily": str(cfg.setup["universe"]["daily_history_start"]),
    }
    if refresh:  # pragma: no cover - needs a terminal
        from bots._shared.mt5_loader import download_mt5_bars

        for freq in frequencies:
            download_mt5_bars(cfg.symbols, freq)
        record["refreshed"] = True
    for freq, start in frequencies.items():
        bars = load_mt5_bars(freq, symbols=cfg.symbols, start_date=start)
        record["frequencies"][freq] = {
            "rows": bars.height,
            "symbols": sorted(bars["symbol"].unique().to_list()),
            "first": str(bars["timestamp"].min()),
            "last": str(bars["timestamp"].max()),
        }
    return record


# =======================================================================================
# Step 2 — recompute the panel and the features
# =======================================================================================
@dataclass
class FeaturePanel:
    """The live recompute. Both frames come from the case study's own functions."""

    prices: pl.DataFrame
    features: pl.DataFrame
    daily: pl.DataFrame
    bars: pl.DataFrame
    columns: list[str]
    spec: str


def recompute_features(cfg: DeployConfig, *, as_of: date | None = None) -> FeaturePanel:
    """``session_panel`` + ``build_features``, the SAME functions the research stages call.

    A feature written twice agrees on the day it is written and drifts on the first edit
    (Ch25 s25.1). This function must therefore never contain a feature definition — only the
    calls.
    """
    from case_studies.exness_usidx_sess._features import (
        build_features,
        feature_columns,
        load_daily_bars,
        session_panel,
    )

    decision = cfg.setup["decision"]
    bars = load_mt5_bars(
        "1h",
        symbols=cfg.symbols,
        start_date=str(cfg.setup["universe"]["history_start"]),
        end_date=None if as_of is None else str(as_of),
    )
    daily = load_daily_bars(cfg.setup, symbols=cfg.symbols)
    prices = session_panel(
        bars,
        spec=cfg.spec,
        calendar=str(decision["session_calendar"]),
        tolerance_minutes=int(decision["session_close_tolerance_minutes"]),
        open_delay_minutes=int(decision["open_delay_minutes"]),
        verbose=False,
    )
    features_block = cfg.setup["features"]
    built = build_features(
        prices,
        spec=cfg.spec,
        windows=features_block["windows"],
        daily_windows=features_block["daily_windows"],
        ranked=features_block.get("ranked") or [],
        periods_per_year=cfg.setup["evaluation"]["periods_per_year"],
        daily=daily,
    )
    return FeaturePanel(
        prices=prices,
        features=built,
        daily=daily,
        bars=bars,
        columns=feature_columns(built),
        spec=cfg.spec,
    )


# =======================================================================================
# The decision-bar guard — ASYMMETRIC, and that is kill criterion (e)
# =======================================================================================
def decision_bar_check(
    cfg: DeployConfig, panel: FeaturePanel, *, decision_date: date
) -> dict[str, Any]:
    """Is there a bar to fill at for this spec on this session? "No bar, no order."

    The two specs are NOT treated alike, because the measurement does not treat them alike:
    the intraday instant missed 0 of 1,930 research sessions and the overnight instant missed
    504 (26 %). A miss on the intraday spec is an incident; a miss on the overnight spec is a
    Friday, or 2022. See ``monitor/circuit_breakers.py``, criterion (e).
    """
    instants = sched.decision_instants(
        decision_date, decision_block=cfg.decision_block, specs=(cfg.spec,)
    )
    instant = instants[cfg.spec]
    row = panel.prices.filter(pl.col("timestamp") == pl.lit(decision_date).cast(pl.Date))
    have = {}
    for symbol in cfg.symbols:
        srow = row.filter(pl.col("symbol") == symbol)
        exec_open = None
        if srow.height and "exec_open" in srow.columns:
            value = srow["exec_open"][0]
            exec_open = None if value is None else float(value)
        have[symbol] = exec_open is not None and exec_open > 0
    missing = sorted(s for s, ok in have.items() if not ok)
    structural = cfg.spec == "overnight"
    return {
        "decision_date": str(decision_date),
        "spec": cfg.spec,
        "instant": instant.as_dict(),
        "fillable": {s: bool(v) for s, v in have.items()},
        "missing": missing,
        "has_any_fill": bool(missing != cfg.symbols),
        # The one sentence that keeps a human from paging on a market structure:
        "classification": (
            "no_trade_this_decision (STRUCTURAL for the overnight spec: 26 % of research "
            "sessions have no fill bar because the index shuts for the weekend at the Friday "
            "cash close; this is not an incident)"
            if missing and structural
            else "no_trade_this_decision (ANOMALOUS for the intraday spec: 0 of 1,930 research "
            "sessions missed a fill bar)"
            if missing
            else "fillable"
        ),
        "escalates": bool(missing) and not structural,
    }


# =======================================================================================
# Steps 3-5 — model, persistence, predictions
# =======================================================================================
@dataclass(frozen=True)
class ModelChoice:
    selector: str
    training_hash: str | None
    population: str | None
    registry: Path | None


def registry_path(cfg: DeployConfig) -> Path:
    """This SPEC's registry. TWO REGISTRIES, ONE BOT — see the module note in BOT.md."""
    import os

    out = os.environ.get("ML4T_OUTPUT_DIR")
    if out:
        candidate = Path(out) / CASE_STUDY / "run_log" / "registry.db"
        if candidate.exists():
            return candidate
    declared = cfg.boundary.get("registries", {}).get(cfg.spec)
    if declared:
        candidate = Path(declared)
        if not candidate.is_absolute():
            candidate = Path.home() / "ml4t" / candidate
        if candidate.exists():
            return candidate
    raise NotReadyError(
        f"no registry for spec {cfg.spec!r}: set ML4T_OUTPUT_DIR to that spec's workspace, or "
        f"make {cfg.boundary.get('registries', {}).get(cfg.spec)} reachable. The two specs are "
        "two workspaces and no single registry answers for this bot"
    )


def resolve_model(cfg: DeployConfig, selector: str | None) -> ModelChoice:
    """What (if anything) may be deployed. ``none`` is the honest default and says why."""
    selector = selector or "none"
    if selector == "none":
        return ModelChoice(
            selector="none",
            training_hash=None,
            population=cfg.model_block["plumbing_prediction_population"].get(cfg.spec),
            registry=None,
        )
    try:
        path = registry_path(cfg)
    except NotReadyError:
        path = None
    if selector == "latest-complete":
        return ModelChoice(
            selector=selector,
            training_hash=None,
            population=cfg.model_block["plumbing_prediction_population"].get(cfg.spec),
            registry=path,
        )
    return ModelChoice(selector=selector, training_hash=selector, population=None, registry=path)


def clamp_training_window(cfg: DeployConfig, end: date) -> date:
    """A deployment refit may NEVER end at or after ``holdout_start``, and this asserts it.

    ``25_live_trading/02`` deliberately extends its data past the book's frozen cut ("this
    notebook is the one place where the data extends past that cut"). **This bot may not**: its
    cut is an unscored holdout.
    """
    limit = cfg.holdout_start - timedelta(days=1)
    clamped = min(end, limit)
    if clamped >= cfg.holdout_start:  # pragma: no cover - defensive
        raise EvidenceBoundaryError(
            f"a refit ending {clamped} would read into the unscored holdout "
            f"{cfg.holdout_start}..{cfg.holdout_end}"
        )
    return clamped


def read_registered_predictions(
    cfg: DeployConfig, choice: ModelChoice
) -> tuple[pl.DataFrame, str]:
    """Read a registered VALIDATION prediction set through the REGISTRY'S OWN reader.

    This is what makes the parity harness runnable before a survivor exists: the predictions
    already exist in the registry, they were scored on validation folds only, and reading them
    adds nothing to K (a prediction set is not a trial until it is turned into a position).

    ``case_studies.utils.registry.read_predictions`` is used rather than a bare
    ``read_parquet``, because it also normalises the artefact's column names to the canonical
    schema (``prediction`` -> ``y_score``, ``actual`` -> ``y_true``) that
    ``signals.build_target_weights_from_config`` expects. Reading the parquet directly would
    hand the sizing function a column it does not know and silently size nothing.
    """
    from case_studies.utils.registry import read_predictions

    path = choice.registry or registry_path(cfg)
    case_dir = path.parent.parent
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = list(
            con.execute(
                "SELECT prediction_hash, training_hash, checkpoint_kind, checkpoint_value "
                "FROM prediction_sets WHERE split = 'validation' ORDER BY created_at"
            )
        )
    except sqlite3.OperationalError as exc:  # pragma: no cover - schema drift
        raise NotReadyError(f"{path}: {exc}") from exc
    finally:
        con.close()
    if not rows:
        raise NotReadyError(f"{path} holds no validation prediction set")
    errors: list[str] = []
    for row in rows:
        try:
            frame = read_predictions(CASE_STUDY, row["prediction_hash"], case_dir=case_dir)
        except (FileNotFoundError, OSError) as exc:  # artefact pruned or not synced
            errors.append(f"{row['prediction_hash']}: {exc}")
            continue
        frame = frame.with_columns(pl.col("timestamp").cast(pl.Date))
        holdout = frame.filter(pl.col("timestamp") >= pl.lit(cfg.holdout_start).cast(pl.Date))
        if holdout.height:
            raise EvidenceBoundaryError(
                f"prediction set {row['prediction_hash']} carries {holdout.height} rows at or "
                f"after holdout_start {cfg.holdout_start}; a validation set must not"
            )
        return frame, str(row["prediction_hash"])
    raise NotReadyError(
        f"{path}: {len(rows)} validation prediction sets are registered but none of their "
        f"artefacts is readable from here (the workspaces live in WSL2 and are gitignored): "
        f"{errors[:3]}"
    )


def retrain(cfg: DeployConfig, panel: FeaturePanel, choice: ModelChoice) -> Any:
    """Step 3. Refuses rather than inventing a model, because there is no survivor to refit."""
    raise NotReadyError(
        f"--model {choice.selector!r} asks for a refit, but phase 5 found 0 survivors of 1,403 "
        f"scored specs at K = {cfg.boundary['phase5_trial_count']} and phase 6 is not opened, "
        "so no registered training run is a deployable model. Refitting one anyway would be "
        "presenting a failed trial as a deployment. Use --model none (parity only) or "
        "--model latest-complete (plumbing only)."
    )


def persist(cfg: DeployConfig, payload: dict[str, Any], *, decision_date: date) -> Path:
    """Step 4. Deployment artefacts NEVER enter the research registry."""
    out = cfg.state_dir / decision_date.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "state.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# =======================================================================================
# Step 6 — PARITY: the five stages of 25_live_trading/08, research against live
# =======================================================================================
def build_target_weights(
    cfg: DeployConfig, predictions: pl.DataFrame, *, decision_date: date | None = None
) -> dict[str, Any]:
    """Stage 4 of parity: the SAME ``signals`` function ``13_backtest`` sends the engine.

    ``fixed_threshold`` is deliberately not used. The mentor proved that with ``long_short``
    the lower threshold is ``1.0 - threshold``, so a regression score of order 1e-3 falls into
    the SHORT branch whenever it does not clear the upper one: the book is never flat and 60 %
    of generation 1's trials tested an always-invested book instead of the declared one
    (``case_studies/utils/signals.py:53-62``, warned at ``:36-37``). That is a MECHANICAL
    defect, not a bad result, which is why avoiding it here is not cherry-picking.
    """
    from case_studies.exness_usidx_sess._sweep import declared_settings
    from case_studies.utils.signals import build_target_weights_from_config

    block = cfg.model_block
    family = str(block["signal_family"])
    lookback = int(block["signal_lookback"])
    # The setting is SELECTED FROM the declared sweep, never constructed here. Building the
    # dict by hand is how generation 1 nearly planned one identity and reported another
    # (case_studies/exness_usidx_sess/_sweep.py, module docstring): one function, three
    # consumers - 13_backtest, _report_phase5 and this loop.
    candidates = declared_settings(cfg.setup, family=family)
    matching = [c for c in candidates if int(c.get("lookback_days", -1)) == lookback]
    if not matching:
        raise NotReadyError(
            f"lookback {lookback} is not in the declared sweep for family {family!r}: "
            f"{[c.get('lookback_days') for c in candidates]}. The live book must be a book the "
            "backtest actually ran."
        )
    config = matching[0]
    if bool(config.get("long_short", False)) != bool(block["long_short"]):
        raise ValueError(
            f"risk_config.yaml::model.long_short is {block['long_short']} but the declared "
            f"sweep runs this family with long_short={config.get('long_short')}. The live book "
            "must be the book that was backtested."
        )
    weights = build_target_weights_from_config(predictions, config)
    if "asset" in weights.columns and "symbol" not in weights.columns:
        weights = weights.rename({"asset": "symbol"})
    weights = weights.with_columns(pl.col("timestamp").cast(pl.Date))
    if decision_date is not None:
        weights = weights.filter(pl.col("timestamp") == pl.lit(decision_date))
    # THE LATEST BASKET IS ONE SESSION'S BASKET, not the last row per symbol.
    # `group_by(symbol).last()` would assemble a basket out of weights decided on different
    # days - a book that never existed - and on a book that trades 2 % of sessions the two
    # legs could be months apart. The basket is the weights of the LAST session the book held
    # anything, and that session's date is reported next to it so it can never be mistaken for
    # today's.
    latest: dict[str, float] = {}
    latest_session = None
    if weights.height:
        latest_session = weights["timestamp"].max()
        latest = {
            row["symbol"]: float(row["weight"])
            for row in weights.filter(pl.col("timestamp") == latest_session).to_dicts()
        }
    return {
        "config": config,
        "frame": weights,
        "latest": latest,
        "latest_session": str(latest_session) if latest_session is not None else None,
    }


def parity_report(
    cfg: DeployConfig,
    panel: FeaturePanel,
    predictions: pl.DataFrame,
    *,
    live_start: date,
    broker: MT5Broker | None = None,
) -> dict[str, Any]:
    """The parity tape: five stages compared, and NOT one performance number.

    While the holdout is unscored, a return or a Sharpe printed here would be a reading taken
    outside the holdout stages. The baskets, the sides, the schedule and the sizing — everything
    parity actually needs — are reported in full.
    """
    report: dict[str, Any] = {
        "definition": (
            "25_live_trading/08: data, features, predictions, sizing, orders — compared "
            "research-against-live. NOT an ml4t.backtest.Engine replay: BOT.md records that "
            "the engine cannot express this bot (date-keyed targets on an hourly grid fill "
            "13-15 h before the decision; a continuous hold against a 6 h label)."
        ),
        "live_start": str(live_start),
        "spec": cfg.spec,
        "stages": {},
        "performance_reported": False,
        "performance_withheld_because": (
            f"the declared holdout {cfg.holdout_start}..{cfg.holdout_end} has not been scored "
            "by the holdout stages; a return printed here would be a holdout reading taken "
            "outside them. Phase 5 also found 0 survivors of 1,403 specs at K = 1,780."
        ),
    }

    # -- stage 1: data -------------------------------------------------------------------
    window = panel.prices.filter(pl.col("timestamp") >= pl.lit(live_start).cast(pl.Date))
    report["stages"]["1_data"] = {
        "h1_bars": panel.bars.height,
        "d1_bars": panel.daily.height,
        "sessions_in_window": int(window["timestamp"].n_unique()),
        "symbols": sorted(window["symbol"].unique().to_list()),
        "first_session": str(window["timestamp"].min()),
        "last_session": str(window["timestamp"].max()),
        "source": "bots/_shared/mt5_loader.load_mt5_bars",
    }

    # -- stage 2: features ----------------------------------------------------------------
    research_matrix = cfg.case_dir / "features" / "financial.parquet"
    stage2: dict[str, Any] = {
        "live_columns": len(panel.columns),
        "live_rows": panel.features.height,
        "builder": "case_studies/exness_usidx_sess/_features.build_features",
        "research_matrix": str(research_matrix),
        "research_matrix_present": research_matrix.exists(),
    }
    if research_matrix.exists():
        research = pl.read_parquet(research_matrix)
        shared = sorted(set(research.columns) & set(panel.features.columns))
        stage2["research_columns"] = len(research.columns)
        stage2["shared_columns"] = len(shared)
        stage2["columns_only_in_research"] = sorted(
            set(research.columns) - set(panel.features.columns)
        )[:20]
        stage2["columns_only_in_live"] = sorted(
            set(panel.features.columns) - set(research.columns)
        )[:20]
        stage2["agrees_on_schema"] = not stage2["columns_only_in_research"] and not (
            stage2["columns_only_in_live"]
        )
    else:
        stage2["note"] = (
            "the research matrix lives in the WSL2 experiment workspace and is gitignored; "
            "run this loop with ML4T_OUTPUT_DIR set to that workspace to compare it"
        )
    report["stages"]["2_features"] = stage2

    # -- stage 3: predictions --------------------------------------------------------------
    preds = predictions.with_columns(pl.col("timestamp").cast(pl.Date))
    report["stages"]["3_predictions"] = {
        "rows": preds.height,
        "sessions": int(preds["timestamp"].n_unique()),
        "symbols": sorted(preds["symbol"].unique().to_list()),
        "first": str(preds["timestamp"].min()),
        "last": str(preds["timestamp"].max()),
        "source": "registered validation prediction set (no refit, no holdout)",
        "holdout_rows": int(
            preds.filter(pl.col("timestamp") >= pl.lit(cfg.holdout_start).cast(pl.Date)).height
        ),
    }
    if report["stages"]["3_predictions"]["holdout_rows"]:  # pragma: no cover - guarded upstream
        raise EvidenceBoundaryError("the prediction set reaches into the holdout")

    # -- stage 4: sizing --------------------------------------------------------------------
    sizing = build_target_weights(cfg, preds)
    frame = sizing["frame"]
    n_sessions = int(preds["timestamp"].n_unique())
    invested = int(frame["timestamp"].n_unique()) if frame.height else 0
    report["stages"]["4_sizing"] = {
        "signal_config": sizing["config"],
        "function": "case_studies/utils/signals.build_target_weights_from_config",
        "weight_rows": frame.height,
        "sessions_with_a_position": invested,
        "sessions_scored": n_sessions,
        # The mentor's bug #3: `n` is sessions, not trades. Reported explicitly so it can never
        # be mistaken for the sample size of a Sharpe ratio again.
        "share_of_sessions_traded": round(invested / n_sessions, 4) if n_sessions else None,
        "n_traded_note": (
            "the number of sessions this book actually holds a position on. Phase 5 computed "
            "its Deflated Sharpe on the number of SESSIONS in the window, which overstates the "
            "effective sample by 6-14x on the percentile family (mentor review, bug #3)."
        ),
        "latest_weights": sizing["latest"],
        "latest_weights_session": sizing["latest_session"],
        "latest_weights_note": (
            "the basket of ONE session - the last the book held anything - not the last row "
            "per symbol. On a book that trades 2 % of sessions those are different things, and "
            "the second is a basket that never existed"
        ),
        "fixed_threshold_excluded_because": (
            "GENERATION 1 tested a book the hypothesis does not describe on that family: "
            "signals.py:53-62 with long_short sets the lower threshold to 1.0 - threshold, so "
            "a regression score of order 1e-3 never lands in the flat branch and 1,068 of the "
            "1,780 trials ran an always-invested book. setup.yaml now declares "
            "threshold_grid.fixed_threshold_convention: signed, which fixes it, but NO "
            "generation has yet been scored with the correction. Until one has, the live book "
            "is the family whose recorded generation actually tested what was declared. This "
            "is a MECHANICAL exclusion, not a choice made after seeing which family lost."
        ),
    }

    # -- stage 5: orders --------------------------------------------------------------------
    report["stages"]["5_orders"] = order_parity(cfg, sizing["latest"], broker=broker)
    report["gaps"] = [g for g in report["stages"]["5_orders"].get("gaps", [])]
    return report


def order_parity(
    cfg: DeployConfig, weights: dict[str, float], *, broker: MT5Broker | None = None
) -> dict[str, Any]:
    """Stage 5: weights -> index units -> lots. **The bot's largest parity gap lives here.**

    ``setup.yaml::execution.share_type`` is ``fractional``: the research engine holds any
    weight it likes. MT5 does not. ``volume_min x trade_contract_size`` is 0.14 index units on
    US500 and 0.05 on USTEC, ``volume_step`` is 0.01, and ``MT5Broker.normalize_lot`` rounds
    DOWN and REJECTS below ``volume_min``. So a research fill below the floor has no live
    counterpart at all — it is not a smaller trade, it is no trade.

    Every number here is read from ``symbol_info`` and a live tick when a broker is connected,
    and falls back to the figures MEASURED on 2026-09-08 and recorded in ``risk_config.yaml``
    otherwise. Nothing is assumed.
    """
    capital = cfg.risk["capital"]
    fallback_price = capital["measured_price_2026_09_08"]
    fallback_min = capital["volume_min"]
    equity = cfg.allocated
    legs: dict[str, Any] = {}
    gaps: list[str] = []
    for symbol in cfg.symbols:
        weight = float(weights.get(symbol, 0.0))
        info = broker.symbol_info(symbol) if broker is not None else None
        tick = broker.tick(symbol) if broker is not None else None
        contract = float(getattr(info, "trade_contract_size", 0.0)) or float(
            capital["contract_size"]
        )
        volume_min = float(getattr(info, "volume_min", 0.0)) or float(fallback_min[symbol])
        volume_step = float(getattr(info, "volume_step", 0.0)) or float(capital["volume_step"])
        price = (
            float(tick.ask)
            if tick is not None and getattr(tick, "ask", 0)
            else float(fallback_price[symbol])
        )
        target_units = weight * equity / price if price else 0.0
        raw_lots = abs(target_units) / contract if contract else 0.0
        lots = (
            normalize_lot(
                raw_lots,
                info
                if info is not None
                else _FallbackInfo(volume_min, volume_step, contract),
                min_lot_policy=str(cfg.risk["execution"]["min_lot_policy"]),
            )
            if raw_lots
            else 0.0
        )
        min_notional = volume_min * contract * price
        leg = {
            "weight": weight,
            "price_source": "symbol_info_tick" if tick is not None else "measured 2026-09-08",
            "price": price,
            "contract_size": contract,
            "volume_min": volume_min,
            "volume_step": volume_step,
            "target_units": target_units,
            "raw_lots": raw_lots,
            "normalised_lots": lots,
            "min_notional_account": min_notional,
            "rejected_by_volume_min": bool(raw_lots > 0 and lots == 0.0),
        }
        if leg["rejected_by_volume_min"]:
            gaps.append(
                f"{symbol}: a research weight of {weight:+.4f} sizes {raw_lots:.4f} lots, below "
                f"volume_min {volume_min}; the adapter REJECTS it rather than scaling it down, "
                f"so the research fill has no live counterpart (min notional {min_notional:,.2f} "
                f"{capital['currency'] if 'currency' in capital else 'USD'})"
            )
        legs[symbol] = leg
    smallest = {s: legs[s]["min_notional_account"] for s in legs}
    gaps.append(
        "share_type parity gap: setup.yaml::execution.share_type is 'fractional' while the "
        f"adapter floor is volume_min ({fallback_min}); the smallest placeable position is "
        f"{ {k: round(v, 2) for k, v in smallest.items()} } and the account's measured equity "
        "on 2026-09-08 was 896.97 USD, so EVERY leg is below the floor at the current balance"
    )
    return {"legs": legs, "gaps": gaps, "equity_used": equity}


@dataclass(frozen=True)
class _FallbackInfo:
    """``symbol_info``-shaped stand-in built from the measured figures in the YAML."""

    volume_min: float
    volume_step: float
    trade_contract_size: float
    volume_max: float = 1000.0


# =======================================================================================
# Step 7 — stage through SafeBroker
# =======================================================================================
def build_live_risk_config(cfg: DeployConfig, *, armed: bool):
    """``LiveRiskConfig`` from the YAML. Arming can only ever make the config SAFER."""
    from ml4t.live import ExecutionMode as LiveExecutionMode
    from ml4t.live import LiveRiskConfig

    block = cfg.live_risk_block
    # ONE SWITCH. `shadow_mode` is derived by `__post_init__` from `execution_mode`; passing
    # both is how a config raises ExecutionModeError. Not arming forces SHADOW.
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


def cfd_execution_guard(
    cfg: DeployConfig, broker: MT5Broker, symbols: list[str], now: datetime
) -> dict[str, Any]:
    """Bad window, wide spread, closed market — checked in VENUE-LOCAL time.

    THE BLOCKED WINDOW IS CONVERTED, NOT TYPED. ``blocked_execution_windows_local`` declares
    17:00-18:00 ``America/New_York``, which is 21:00-22:00 UTC in summer and 22:00-23:00 UTC in
    winter. A UTC constant would be wrong for five months of every year — that is kill
    criterion (g)'s subject, and hard-coding it here would defeat the breaker that watches it.
    """
    guards = cfg.cfd_guards
    reasons: list[str] = []
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    windows: list[dict[str, Any]] = []
    for window in guards.get("blocked_execution_windows_local") or []:
        zone = ZoneInfo(str(window["zone"]))
        local = now.astimezone(zone)
        start = time.fromisoformat(str(window["start"]))
        end = time.fromisoformat(str(window["end"]))
        blocked = start <= local.time() < end
        windows.append(
            {
                "zone": str(window["zone"]),
                "local_now": local.isoformat(),
                "utc_equivalent_today": [
                    datetime.combine(local.date(), start, tzinfo=zone)
                    .astimezone(UTC)
                    .strftime("%H:%M"),
                    datetime.combine(local.date(), end, tzinfo=zone)
                    .astimezone(UTC)
                    .strftime("%H:%M"),
                ],
                "blocked": blocked,
                "reason": window.get("reason"),
            }
        )
        if blocked:
            reasons.append(
                f"execution window {window['start']}-{window['end']} {window['zone']} is "
                f"blocked: {window.get('reason')}"
            )
    spreads: dict[str, Any] = {}
    assumed = guards.get("assumed_spread_p90_bps") or {}
    multiple = float(guards.get("max_spread_multiple_of_p90", 2.0))
    quoting = False
    for symbol in symbols:
        tick = broker.tick(symbol)
        if tick is None or not getattr(tick, "bid", 0) or not getattr(tick, "ask", 0):
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
        }
        if limit is not None and spread_bps > limit * multiple:
            entry["blocked"] = True
            reasons.append(
                f"{symbol}: live spread {spread_bps:.2f} bps above {multiple:g}x the assumed "
                f"p90 {limit:.2f} bps"
            )
        spreads[symbol] = entry
        quoting = True
    if guards.get("refuse_when_market_closed") and not quoting:
        reasons.append("neither index is quoting")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "windows": windows,
        "spreads": spreads,
        "checked_at": now.isoformat(),
    }


def capital_viability(
    cfg: DeployConfig, broker: MT5Broker | None, *, account_equity: float | None = None
) -> dict[str, Any]:
    """Can this bot's declared allocation place an order on this account at all?

    Two ways it cannot and both are silent unless something says so out loud: the ALLOCATION is
    below the lot floor, or the ACCOUNT is. Both are true today.
    """
    capital = cfg.risk["capital"]
    reasons: list[str] = []
    per_symbol: dict[str, Any] = {}
    for symbol in cfg.symbols:
        info = broker.symbol_info(symbol) if broker is not None else None
        tick = broker.tick(symbol) if broker is not None else None
        volume_min = float(getattr(info, "volume_min", 0.0)) or float(
            capital["volume_min"][symbol]
        )
        contract = float(getattr(info, "trade_contract_size", 0.0)) or float(
            capital["contract_size"]
        )
        price = (
            float(tick.ask)
            if tick is not None and getattr(tick, "ask", 0)
            else float(capital["measured_price_2026_09_08"][symbol])
        )
        min_notional = volume_min * contract * price
        # The declared book is equal-weight over 2 symbols, so a same-side session runs each
        # leg at 0.5 of allocated: that is the binding requirement.
        needed = min_notional / 0.5
        per_symbol[symbol] = {
            "volume_min": volume_min,
            "contract_size": contract,
            "price": price,
            "min_notional_account": min_notional,
            "allocation_needed_for_equal_weight_leg": needed,
            "fits_in_allocated": cfg.allocated >= needed,
        }
        if cfg.allocated < needed:
            reasons.append(
                f"{symbol}: an equal-weight leg needs {needed:,.2f} of allocated capital "
                f"(volume_min {volume_min} x contract {contract:g} x price {price:,.2f} at "
                f"weight 0.5) but capital.allocated is {cfg.allocated:,.2f}"
            )
        if account_equity is not None and account_equity < min_notional:
            reasons.append(
                f"{symbol}: the smallest placeable position is {min_notional:,.2f} of notional "
                f"against account equity {account_equity:,.2f}"
            )
    if cfg.allocated < cfg.min_viable_allocated:
        reasons.append(
            f"capital.allocated {cfg.allocated:,.2f} is below the declared "
            f"min_viable_allocated {cfg.min_viable_allocated:,.2f}"
        )
    return {
        "viable": not reasons,
        "reasons": reasons,
        "per_symbol": per_symbol,
        "allocated": cfg.allocated,
        "min_viable_allocated": cfg.min_viable_allocated,
        "account_equity": account_equity,
        "share_type_in_research": cfg.setup["execution"]["share_type"],
    }


async def reconcile_async(safe, broker: MT5Broker, *, magic: int) -> dict[str, Any]:
    """Kill criterion (d): compare BOTH views — this magic, and every magic on the login."""
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
        "account_positions_by_magic": by_magic,
        "unknown_magics": sorted({int(p["magic"]) for p in foreign}),
        "magic": magic,
        "note": (
            "two positions of the RETIRED legacy magic 202500 were open on this login on "
            "2026-09-08; 202500 is in RESERVED_MAGICS so they are recognised, not foreign"
        ),
    }
    if foreign:
        raise ForeignPositionError(
            f"positions with unallocated magic(s) {record['unknown_magics']} are open (known: "
            f"{sorted(known)}); kill criterion (d) — no trading until a person explains them"
        )
    return record


async def stage_basket_async(
    cfg: DeployConfig,
    weights: dict[str, float],
    *,
    armed: bool,
    mt5_module: Any = None,
    breakers: cb.BotBreakers | None = None,
    account_halt_file: Path | str | None = None,
) -> dict[str, Any]:
    """Step 7. Every leg is ``dry_run`` unless a person armed the session AND nothing blocks."""
    from ml4t.backtest import OrderSide, OrderStatus
    from ml4t.live import SafeBroker

    risk_config = build_live_risk_config(cfg, armed=armed)
    broker = MT5Broker(
        magic=cfg.magic,
        execution_mode=str(cfg.risk["execution"]["mode"]),
        armed_live=False,  # never; live needs phase 7 and an explicit human action
        max_margin_fraction=float(cfg.risk["execution"]["max_margin_fraction"]),
        deviation_points=int(cfg.risk["execution"]["deviation_points"]),
        close_deviation_points=int(cfg.risk["execution"]["close_deviation_points"]),
        min_lot_policy=str(cfg.risk["execution"]["min_lot_policy"]),
        comment=f"ml4t-{BOT_ID[:12]}",
        mt5=mt5_module,
    )
    safe = SafeBroker(broker, risk_config)
    record: dict[str, Any] = {
        "armed": armed,
        "execution_mode_broker": broker.execution_mode,
        "execution_mode_safe": risk_config.execution_mode.value,
        "shadow_mode": getattr(risk_config, "shadow_mode", None),
        "orders_can_reach_broker": bool(
            armed and risk_config.execution_mode.value != "shadow"
        ),
    }
    reasons: list[str] = []
    await safe.connect()
    try:
        record["account"] = broker.account_summary()
        snapshot = read_account_snapshot(broker)
        # The halt file and the persisted high-water mark are SHARED ACCOUNT STATE: a trip
        # stops every bot on the login. Tests point them at a tmp path so a dry run can never
        # write a peak or a halt that another bot would read as real.
        guard = AccountGuard(
            load_account_limits(), halt_file=account_halt_file, bot_id=BOT_ID
        )
        guard.start_day(snapshot)
        account_ok = guard.check(snapshot)
        record["account_snapshot"] = snapshot.as_dict()
        record["account_breakers"] = guard.to_record()

        try:
            record["reconciliation"] = await reconcile_async(safe, broker, magic=cfg.magic)
        except ForeignPositionError as exc:
            record["reconciliation"] = {"clean": False, "error": str(exc)}
            reasons.append(str(exc))

        now = datetime.now(UTC)
        record["cfd_guard"] = cfd_execution_guard(cfg, broker, cfg.symbols, now)
        record["capital_viability"] = capital_viability(
            cfg, broker, account_equity=snapshot.equity
        )
        record["order_parity"] = order_parity(cfg, weights, broker=broker)

        if guard.is_halted():
            reasons.append(f"account halt file present at {guard.halt_file}")
        if not account_ok:
            reasons.append(f"account breaker(s) open: {guard.manager.open_breakers()}")
        rec = record.get("reconciliation") or {}
        if not rec.get("clean", False) and risk_config.fail_on_reconciliation_mismatch:
            reasons.append("reconciliation is not clean (kill criterion (d))")
        if not record["cfd_guard"]["ok"]:
            reasons.extend(record["cfd_guard"]["reasons"])
        if not record["capital_viability"]["viable"]:
            reasons.extend(record["capital_viability"]["reasons"])
        if breakers is not None and not breakers.allows_trading():
            reasons.append(f"strategy breaker(s) open: {breakers.manager.open_breakers()}")
        if breakers is not None and breakers.pending_user_approval:
            reasons.append(
                "the kill criteria are a DRAFT the user has not approved (BOT.md)"
            )
        if not cfg.holdout_scored:
            reasons.append(
                f"the holdout {cfg.holdout_start}..{cfg.holdout_end} has not been scored"
            )
        if int(cfg.boundary.get("phase5_survivors", 0)) == 0:
            reasons.append(
                f"phase 5 found 0 survivors of {cfg.boundary.get('phase5_specs_scored')} scored "
                f"specs at K = {cfg.boundary.get('phase5_trial_count')}"
            )
        if not armed:
            reasons.append("not armed: --arm was not given in this session")
        record["blocked_by"] = reasons

        legs = []
        parity_legs = record["order_parity"]["legs"]
        for symbol in cfg.symbols:
            leg = dict(parity_legs[symbol])
            leg["symbol"] = symbol
            lots = float(leg["normalised_lots"])
            if lots <= 0:
                leg["status"] = (
                    "rounds_to_zero" if leg["raw_lots"] > 0 else "flat"
                )
                legs.append(leg)
                continue
            if reasons:
                leg["status"] = "dry_run"
                legs.append(leg)
                continue
            tick = broker.tick(symbol)  # pragma: no cover - unreachable in a dry run
            reference = (
                (float(tick.bid) + float(tick.ask)) / 2 if tick is not None else leg["price"]
            )
            safe.record_market_snapshot(symbol, reference)
            order = await safe.submit_order_async(
                asset=symbol,
                quantity=lots * float(leg["contract_size"]),
                side=OrderSide.BUY if leg["target_units"] > 0 else OrderSide.SELL,
            )
            leg["status"] = (
                "submitted" if order.status is not OrderStatus.REJECTED else "rejected"
            )
            leg["order_status"] = order.status.value
            leg["order_id"] = order.order_id
            legs.append(leg)
        record["legs"] = legs
        record["intended_basket"] = sorted(
            leg["symbol"] for leg in legs if leg["status"] != "flat"
        )
        record["accepted_basket"] = sorted(
            leg["symbol"] for leg in legs if leg.get("status") == "submitted"
        )
    finally:
        try:
            safe.close_persistence()
        finally:
            await safe.disconnect()
    return record


# =======================================================================================
# The cycle
# =======================================================================================
def run_cycle(
    *,
    spec: str = SPEC,
    model_selector: str | None = MODEL_SELECTOR,
    arm: bool = SUBMIT_ORDERS,
    refresh: bool = REFRESH_DATA,
    as_of: date | None = None,
    live_start: date | None = None,
    risk_config: Path | str | None = None,
    state_dir: Path | str | None = None,
    mt5_module: Any = None,
    account_halt_file: Path | str | None = None,
) -> dict[str, Any]:
    """One deployment cycle for ONE spec. Returns the run record; writes it to ``state/``."""
    cfg = load_deploy_config(risk_config, spec=spec, state_dir=state_dir)
    record: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "evidence": cfg.evidence_stamp(),
        "config_source": str(cfg.risk_path),
        "magic": cfg.magic,
    }

    # -- arming is refused outright while the criteria are a draft ---------------------
    pending = bool(cfg.risk["breakers"].get("pending_user_approval", True))
    if arm and pending:
        raise PermissionError(
            "--arm refused: the kill criteria of this bot are a DRAFT the user has not "
            "approved (BOT.md, 'Kill criteria'), phase 5 found 0 survivors of 1,403 specs at "
            "K = 1,780 and the holdout is unscored. Arming would be staging orders against a "
            "rule set nobody agreed to."
        )
    armed = bool(arm)

    breakers = cb.build_manager(cfg.risk_path, allow_draft_criteria=True)
    record["breakers"] = breakers.to_record()

    # -- step 1 ------------------------------------------------------------------------
    record["step1_data"] = refresh_data(cfg, refresh=refresh)

    # -- the decision date: last session BEFORE the holdout unless a person overrides ---
    default_as_of = cfg.holdout_start - timedelta(days=1)
    decision_date = as_of or default_as_of
    record["decision_date"] = str(decision_date)
    record["holdout_window_touched"] = bool(decision_date >= cfg.holdout_start)
    if record["holdout_window_touched"]:
        logger.warning(
            "--as-of %s is inside the unscored holdout %s..%s; the run record is stamped",
            decision_date,
            cfg.holdout_start,
            cfg.holdout_end,
        )

    # -- step 2 ------------------------------------------------------------------------
    panel = recompute_features(cfg, as_of=None if as_of else cfg.holdout_start - timedelta(days=1))
    record["step2_features"] = {
        "rows": panel.features.height,
        "columns": len(panel.columns),
        "sessions": int(panel.features["timestamp"].n_unique()),
        "builder": "case_studies/exness_usidx_sess/_features",
    }

    # -- the decision-bar guard --------------------------------------------------------
    sessions = sorted(panel.prices["timestamp"].unique().to_list())
    if decision_date not in sessions:
        near = [s for s in sessions if s <= decision_date]
        decision_date = near[-1] if near else decision_date
        record["decision_date"] = str(decision_date)
        record["decision_date_note"] = "moved back to the last session the panel carries"
    bar = decision_bar_check(cfg, panel, decision_date=decision_date)
    record["decision_bar"] = bar
    if cfg.spec == "intraday":
        breakers.missing_bar_intraday.record_decision(had_bar=bar["has_any_fill"])
    else:
        breakers.overnight_coverage.record_decision(had_bar=bar["has_any_fill"])

    # -- steps 3-5 ---------------------------------------------------------------------
    choice = resolve_model(cfg, model_selector)
    record["model"] = {
        "selector": choice.selector,
        "training_hash": choice.training_hash,
        "population": choice.population,
        "refit": False,
    }
    if choice.selector not in ("none", "latest-complete"):
        retrain(cfg, panel, choice)  # always raises: no survivor exists to refit

    try:
        predictions, prediction_hash = read_registered_predictions(cfg, choice)
        record["step5_predictions"] = {
            "rows": predictions.height,
            "prediction_hash": prediction_hash,
            "source": "registered validation prediction set (registry.read_predictions)",
            "adds_to_K": 0,
            "adds_to_K_because": (
                "a prediction set is not a trial until it is turned into a position, and this "
                "one was already registered and already counted in K = 1,780"
            ),
        }
    except NotReadyError as exc:
        record["step5_predictions"] = {"rows": 0, "blocked": str(exc)}
        record["signal_status"] = "no_deployable_signal"
        record["deployable"] = False
        record["stopped_after"] = "step 5 (no readable registered prediction set)"
        return _write(cfg, record, decision_date)

    # -- step 6: PARITY ----------------------------------------------------------------
    window_start = live_start or (decision_date - timedelta(days=365))
    record["step6_parity"] = parity_report(
        cfg, panel, predictions, live_start=window_start
    )
    record["signal_status"] = (
        "plumbing_only" if choice.selector == "latest-complete" else "no_deployable_signal"
    )
    record["deployable"] = False

    weights = record["step6_parity"]["stages"]["4_sizing"]["latest_weights"]
    record["staged_basket_is_historical"] = {
        "session": record["step6_parity"]["stages"]["4_sizing"]["latest_weights_session"],
        "note": (
            "this is the last basket the REGISTERED VALIDATION predictions produced, not a "
            "decision taken today. Step 7 exercises the staging path with it; it is plumbing, "
            "not a trading decision, and every leg is dry_run"
        ),
    }

    # -- step 7 ------------------------------------------------------------------------
    if choice.selector == "none":
        record["stopped_after"] = (
            "step 6 (parity). --model none does not stage: there is no deployable signal, so "
            "exercising the staging path would be theatre. Use --model latest-complete to "
            "exercise it as plumbing."
        )
        return _write(cfg, record, decision_date)

    record["step7_staging"] = asyncio.run(
        stage_basket_async(
            cfg,
            weights,
            armed=armed,
            mt5_module=mt5_module,
            breakers=breakers,
            account_halt_file=account_halt_file,
        )
    )
    return _write(cfg, record, decision_date)


def _write(cfg: DeployConfig, record: dict[str, Any], decision_date: date) -> dict[str, Any]:
    record["finished_at"] = datetime.now(UTC).isoformat()
    record["run_record_path"] = str(persist(cfg, record, decision_date=decision_date))
    return record


# =======================================================================================
# CLI
# =======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "exness_usidx_sess deployment loop — DRY RUN ONLY. 0 phase-5 survivors at "
            "K = 1,780, phase 6 not opened, holdout unscored. This cannot place an order."
        )
    )
    parser.add_argument("--spec", choices=sorted(sched.SPECS), default=SPEC)
    parser.add_argument("--model", default=MODEL_SELECTOR)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--live-start", type=date.fromisoformat, default=None)
    parser.add_argument("--risk-config", default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--arm",
        action="store_true",
        help="REFUSED while the kill criteria are a draft; kept so the refusal is testable",
    )
    parser.add_argument("--fake-mt5", action="store_true", help="run against bots/_shared/testing")
    parser.add_argument("--schedule", action="store_true", help="print the decision schedule only")
    return parser


def make_fake_terminal(cfg: DeployConfig):
    """A fake ``MetaTrader5`` with THIS bot's contract sizes, so a dry run measures the floor."""
    from bots._shared.testing.fake_mt5 import FakeMT5, default_symbol_info

    capital = cfg.risk["capital"]
    symbols = {}
    for symbol in cfg.symbols:
        name = to_market_watch(symbol)
        symbols[name] = default_symbol_info(
            name,
            trade_contract_size=float(capital["contract_size"]),
            volume_min=float(capital["volume_min"][symbol]),
            volume_step=float(capital["volume_step"]),
            volume_max=1000.0,
            digits=2,
            point=0.01,
            swap_mode=1,
            swap_long=-147.4 if symbol == "US500" else -592.7,
            swap_short=0.0,
            swap_rollover3days=5,
            currency_base="USD",
            currency_profit="USD",
            currency_margin="USD",
        )
    fake = FakeMT5(symbols=symbols)
    fake.account.update({"balance": 894.24, "equity": 896.97, "margin": 1.38, "leverage": 2000})
    for symbol in cfg.symbols:
        price = float(capital["measured_price_2026_09_08"][symbol])
        fake.quotes[to_market_watch(symbol)] = (price - 0.4, price)
    return fake


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_deploy_config(args.risk_config, spec=args.spec, state_dir=args.state_dir)

    if args.schedule:
        today = date.today()
        table = sched.schedule_table(
            today, today + timedelta(days=10), decision_block=cfg.decision_block
        )
        print(json.dumps({"evidence": cfg.evidence_stamp(), "schedule": table}, indent=2))
        return 0

    record = run_cycle(
        spec=args.spec,
        model_selector=args.model,
        arm=args.arm,
        refresh=args.refresh,
        as_of=args.as_of,
        live_start=args.live_start,
        risk_config=args.risk_config,
        state_dir=args.state_dir,
        mt5_module=make_fake_terminal(cfg) if args.fake_mt5 else None,
    )
    print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
