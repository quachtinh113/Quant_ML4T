"""exness_gold_sess deployment loop - the seven steps of Chapter 25, on a session cadence.

WHAT THIS IS
------------
``25_live_trading/02_etfs_deployment_loop.py`` gives the shape (refresh data -> features ->
refit -> persist -> predict -> strategy -> offline reference tape -> stage orders ->
reconcile -> run record) and ``25_live_trading/11_fx_deployment_loop.py`` gives the FX/CFD
variant of it. This module is that loop for two metals decided **twice a weekday** rather than
one basket decided once a day.

WHAT IS NOT HERE, AND MUST NEVER BE
-----------------------------------
``order_send``. Every bot on this account routes through the ONE shared adapter
``bots/_shared/mt5_broker.py::MT5Broker``, wrapped in ``ml4t.live.SafeBroker`` with the
``LiveRiskConfig`` from ``deploy/risk_config.yaml``. Four copies of an order path would drift;
the contract is identical for every bot and only the data differs (``bots/README.md``,
``bot-template.md`` lines 60-70).

TWO SLEEVES, NOT ONE BOOK
-------------------------
``bots/exness_gold_sess/PRICE_GRID_DECLARATION.md`` section 3.4: a broker holds **one net
position per symbol**. On a pooled book the New York decision at 14:00 UTC lands in the middle
of the London hold (which still has three hours to run), so the position is *adjusted* rather
than opened, ``bars_held`` keeps counting from the London entry, and the time-exit rule closes
the **whole** position at 17:00 - killing the New York leg after three hours instead of eight.
Nothing in the repository can express two overlapping holds of the same symbol on one netting
account.

So this loop runs **one sleeve per session**, each carrying its own half of the allocation
(``risk_config.yaml::capital.sleeves``), and the pooled result is the deterministic sum of the
two. A reader who expected one net book should read that section before changing anything here.

STATE OF THE BOT - WHY THIS CANNOT RUN END TO END TODAY
-------------------------------------------------------
Phase 4 has not run. ``case_studies/exness_gold_sess/run_log/`` is empty, no training run and
no prediction set exists, the price-grid fix has not landed, phases 5-7 have not happened and
the holdout is unscored. Steps 3, 4, 5, 6 and 7 below therefore **cannot be executed**, and
they are written as explicit stubs that RAISE ``NotReadyError`` naming what is missing. They
are not faked, not filled with a placeholder model, and not silently skipped: a stub that
returns a plausible number is worse than one that stops.

What DOES run today, and is worth running:
  * step 1  refresh / load the H1 and D1 bars (offline by default),
  * step 2  rebuild the decision-grid features at the live instant through the SAME code the
            research stages use (``case_studies/exness_gold_sess/_features.features_as_of``),
            which is the parity property phase 2 tested,
  * step 8  the pre-flight: connect read-only, assert the account is a demo in paper mode,
            check the magic against the allocation table, read the account-tier halt file,
            verify the contract geometry against ``symbol_info`` rather than assuming it,
  * step 9  reconcile: list every position carrying magic 260903 and compare with what this
            bot believes it holds - kill criterion (d),
  * step 10 write the run record.

Run it with ``--preflight`` to do exactly that.

USAGE
-----
    uv run python bots/exness_gold_sess/deploy/deployment_loop.py --preflight
    uv run python bots/exness_gold_sess/deploy/deployment_loop.py --cycle          # dry run
    uv run python bots/exness_gold_sess/deploy/deployment_loop.py --cycle --arm    # per session

``--arm`` is refused while ``risk_config.yaml::breakers.pending_user_approval`` is true, which
it is: the kill criteria are the mentor's draft and the user has not approved them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from bots._shared import MAGIC_ALLOCATION, RESERVED_MAGICS
from bots._shared.monitor import AccountGuard, load_account_limits, read_account_snapshot
from bots._shared.mt5_broker import MT5Broker

logger = logging.getLogger("exness_gold_sess.deploy")

# ---------------------------------------------------------------------------------------
# Parameters block - the module equivalent of a notebook's `tags=["parameters"]` cell.
# Production defaults. Never edited to true here; the command line is the only way to change
# them, for one session at a time. 25_live_trading/02:121 and 11_fx_deployment_loop.py:134.
# ---------------------------------------------------------------------------------------
SUBMIT_ORDERS = False  # explicit opt-in per session; the default is an offline dry run
REFRESH_DATA = False  # pulling new bars needs a logged-in Windows terminal
MODEL_SELECTOR = "none"  # no prediction set exists; see the module docstring

BOT_ID = "exness_gold_sess"
CASE_STUDY = "exness_gold_sess"
DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_RISK_CONFIG = DEPLOY_DIR / "risk_config.yaml"
STATE_DIR = DEPLOY_DIR / "state"


class NotReadyError(RuntimeError):
    """A step that cannot run because a phase this bot has not reached must run first.

    Raised rather than returning a placeholder. Every message names the phase, the artifact
    and the file that would produce it, so a reader learns what is missing rather than that
    "something failed".
    """


# =======================================================================================
# Configuration
# =======================================================================================
@dataclass(frozen=True)
class DeployConfig:
    """``risk_config.yaml`` and the case study's ``setup.yaml``, parsed and cross-checked."""

    bot_id: str
    case_study: str
    magic: int
    shadow_mode: bool
    execution: dict[str, Any]
    contract: dict[str, Any]
    capital: dict[str, Any]
    live_risk: dict[str, Any]
    cfd_guards: dict[str, Any]
    breakers: dict[str, Any]
    retire: dict[str, Any]
    setup: dict[str, Any]
    source: Path

    @property
    def symbols(self) -> list[str]:
        return sorted(self.setup["universe"]["symbols"])

    @property
    def sleeves(self) -> dict[str, float]:
        return dict(self.capital["sleeves"])

    @property
    def min_units(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.contract["min_units"].items()}

    @property
    def kill_criteria_approved(self) -> bool:
        return not bool(self.breakers.get("pending_user_approval", True))


def load_deploy_config(path: Path | str | None = None) -> DeployConfig:
    """Read both YAML files and cross-check the things that must agree.

    Cross-checks, each of which has been a real incident somewhere in this repository:

    * the magic matches ``bots/_shared/__init__.py::MAGIC_ALLOCATION`` for this bot, and is
      not one of ``RESERVED_MAGICS``. Two bots sharing a magic reconcile each other's
      positions and close them;
    * the allowed assets are exactly ``setup.yaml::universe.symbols``. A risk config that
      permits a symbol the research never modelled is a config that permits an unmodelled
      trade;
    * the sleeve weights sum to 1.0;
    * ``execution.mode`` is ``paper``. Nothing may set it to ``live`` before phase 7.
    """
    from utils.paths import get_case_study_dir

    config_path = Path(path or DEFAULT_RISK_CONFIG)
    raw = yaml.safe_load(config_path.read_text())
    setup = yaml.safe_load(
        (get_case_study_dir(CASE_STUDY) / "config" / "setup.yaml").read_text()
    )

    magic = int(raw["magic"])
    allocated = MAGIC_ALLOCATION.get(raw["bot_id"])
    if allocated is None:
        raise ValueError(
            f"{raw['bot_id']!r} has no entry in bots/_shared/__init__.py::MAGIC_ALLOCATION"
        )
    if magic != allocated:
        raise ValueError(
            f"{config_path} declares magic {magic} but the allocation table gives "
            f"{allocated} for {raw['bot_id']!r}; the table is the single source of truth"
        )
    if magic in RESERVED_MAGICS:
        raise ValueError(f"magic {magic} is reserved: {RESERVED_MAGICS[magic]}")

    declared_assets = sorted(raw["live_risk_config"]["allowed_assets"])
    universe = sorted(setup["universe"]["symbols"])
    if declared_assets != universe:
        raise ValueError(
            f"allowed_assets {declared_assets} != setup.yaml universe {universe}; a risk "
            "config that permits a symbol the research never modelled permits an unmodelled trade"
        )

    sleeves = dict(raw["capital"]["sleeves"])
    if abs(sum(sleeves.values()) - 1.0) > 1e-9:
        raise ValueError(f"capital.sleeves must sum to 1.0, got {sleeves}")

    mode = str(raw["execution"]["mode"])
    if mode != "paper":
        raise ValueError(
            f"execution.mode is {mode!r}; this bot may only be 'paper' until phase 7 has "
            "scored the holdout and the user has approved the kill criteria"
        )

    return DeployConfig(
        bot_id=raw["bot_id"],
        case_study=raw["case_study"],
        magic=magic,
        shadow_mode=bool(raw["shadow_mode"]),
        execution=raw["execution"],
        contract=raw["contract"],
        capital=raw["capital"],
        live_risk=raw["live_risk_config"],
        cfd_guards=raw.get("cfd_guards", {}),
        breakers=raw.get("breakers", {}),
        retire=raw.get("retire", {}),
        setup=setup,
        source=config_path,
    )


# =======================================================================================
# STEP 1 - Refresh data (25_live_trading/02:125)
# =======================================================================================
def refresh_data(cfg: DeployConfig, *, refresh: bool = REFRESH_DATA) -> dict[str, Any]:
    """Load the H1 and D1 bars the features are built from; optionally pull new ones first.

    Pulling needs a logged-in MetaTrader 5 terminal, which exists only on the Windows side of
    this setup, so the default is to read what ``bots/_shared/mt5_loader`` last wrote. The
    returned dict records which it was, because "the features were built on stale bars" is a
    thing a run record has to be able to answer.
    """
    from bots._shared.mt5_loader import load_mt5_bars

    if refresh:
        raise NotReadyError(
            "refresh_data(refresh=True) needs a logged-in MetaTrader 5 terminal. On this setup "
            "MT5 runs only on Windows with the global Python 3.12 (bots/exness_gold_sess/BOT.md, "
            "phase 0). Download with bots/_shared/mt5_loader.download_mt5_bars there, then run "
            "this loop against the refreshed parquet."
        )

    h1 = load_mt5_bars(frequency="1h", symbols=cfg.symbols)
    d1 = load_mt5_bars(frequency="daily", symbols=cfg.symbols)
    return {
        "refreshed": False,
        "h1_rows": h1.height,
        "h1_last": str(h1["timestamp"].max()),
        "d1_rows": d1.height,
        "d1_last": str(d1["timestamp"].max()),
        "h1": h1,
        "d1": d1,
    }


# =======================================================================================
# STEP 2 - Load inputs and compute features (25_live_trading/02:158)
# =======================================================================================
@dataclass
class LivePanel:
    """The feature rows for one decision instant, and the provenance of how they were built."""

    decision_ts: datetime
    session: str
    rows: pl.DataFrame
    feature_columns: list[str]
    bars_used_through: datetime


def recompute_features(cfg: DeployConfig, bars: dict[str, Any], *, as_of: datetime) -> LivePanel:
    """Rebuild the feature row at *as_of* from ONLY the bars closed by then.

    This calls ``case_studies.exness_gold_sess._features.features_as_of``, which is the same
    code path ``03_financial_features`` uses in batch. That is the whole point and it is not a
    convenience: phase 2 asserted that the matrix rebuilt from only the bars closed at eight
    sampled instants - two per (venue x DST season) - equals the batch row on all 33 columns
    to a maximum absolute difference of **0.0**
    (``bots/exness_gold_sess/tests/test_lookahead.py``). A deployment loop with its own second
    implementation of the features would make that test prove nothing about what trades.

    Note what this does NOT include: the ten model-based columns of
    ``04_model_based_features`` (Kalman, HMM, ARIMA). Those are refitted per fold in research,
    and the live equivalent is step 3, which is a stub.
    """
    from case_studies.exness_gold_sess import _features

    windows = cfg.setup.get("features", {}).get("windows", {})
    panel = _features.features_as_of(as_of, bars=bars["h1"], d1=bars["d1"], windows=windows)
    if panel.is_empty():
        raise NotReadyError(
            f"no decision row exists at {as_of:%Y-%m-%d %H:%M} UTC. Either it is not a declared "
            "decision instant (session open + 60 minutes for london / new_york), or the decision "
            "bar is missing - which is kill criterion (e): do not place an order this session."
        )
    session = str(panel["session"][0]) if "session" in panel.columns else "unknown"
    feature_columns = [c for c in panel.columns if c not in ("symbol", "timestamp", "session")]
    return LivePanel(
        decision_ts=as_of,
        session=session,
        rows=panel,
        feature_columns=feature_columns,
        bars_used_through=as_of,
    )


def decision_bar_check(cfg: DeployConfig, panel: LivePanel) -> dict[str, Any]:
    """Kill criterion (e): every symbol must have a decision row, or the session is a no-trade."""
    present = sorted(panel.rows["symbol"].unique().to_list())
    missing = sorted(set(cfg.symbols) - set(present))
    guard = cfg.cfd_guards.get("missing_decision_bar", {})
    return {
        "decision_ts": str(panel.decision_ts),
        "session": panel.session,
        "symbols_present": present,
        "symbols_missing": missing,
        "action": guard.get("action", "no_trade_session") if missing else "proceed",
        "kill_criterion": "e" if missing else None,
    }


# =======================================================================================
# STEP 3 - Build the training matrix and refit (25_live_trading/02:189)
#          *** STUB: needs a model that does not exist ***
# =======================================================================================
def retrain(cfg: DeployConfig, panel: LivePanel) -> Any:
    """STUB. Refit the selected configuration on data up to the decision instant.

    Cannot run: phase 4 has not fitted anything. What has to exist first, in order:

    1. ``labels.rebalance_step`` re-authored per ``PRICE_GRID_DECLARATION.md`` section 2, and
       the H1 price grid registered in ``case_studies/utils/backtest_loaders.py::_PRICE_CONFIG``
       (both are repository-pinned changes that need an empty run log, which is why they come
       first);
    2. ``case_studies/exness_gold_sess/06_linear.py`` and ``07_gbm.py`` run, publishing the two
       canonical validation populations (104 training runs, 419 prediction sets);
    3. ``13_backtest`` through ``16_costs`` run, and a configuration SELECTED on validation
       backtest Sharpe at the cumulative trial count - not on an IC, and not here;
    4. ``17_holdout_predictions`` / ``18_holdout_backtest`` scored once.

    Only then is there a name to put in ``MODEL_SELECTOR``. When it exists, this function
    reproduces the selected training spec from the registry (family, config, checkpoint,
    feature list, preprocessing identity) and refits it on the window ending at the decision
    instant, with the SAME purge the research folds used - and the lookahead guard of
    ``25_live_trading/02:213`` asserts that the training window ends strictly before the
    decision bar.
    """
    raise NotReadyError(
        "step 3 (refit) has no model to refit. run_log/ is empty: 0 training runs, 0 prediction "
        "sets, K = 0 trials spent. Fit the population with case_studies/exness_gold_sess/"
        "06_linear.py and 07_gbm.py, select on validation backtest Sharpe in 13_backtest, score "
        "the holdout once in 17/18, then name the winner in MODEL_SELECTOR. Do not put a "
        "placeholder model here."
    )


# =======================================================================================
# STEP 4 - Persist the deployment artefact (25_live_trading/02:253)
#          *** STUB: nothing to persist until step 3 runs ***
# =======================================================================================
def persist(model: Any, *, state_dir: Path, decision_ts: datetime) -> Path:
    """STUB. Write the refitted model plus its full provenance beside the run record."""
    raise NotReadyError(
        "step 4 (persist) has nothing to persist: step 3 raised. The artefact must carry the "
        "training hash, the prediction hash of the selected member, the feature list and its "
        "digests, and the preprocessing identity, so that a live prediction can be traced back "
        "to a registered validation result."
    )


# =======================================================================================
# STEP 5 - Predict the live window (25_live_trading/02:303)
#          *** STUB ***
# =======================================================================================
def predict(model: Any, panel: LivePanel) -> pl.DataFrame:
    """STUB. Score the decision row with the refitted model."""
    raise NotReadyError("step 5 (predict) has no model; step 3 raised.")


# =======================================================================================
# STEP 6 - Build the strategy (25_live_trading/02:328)
#          *** STUB for the signal; the SIZING rules below are real and testable ***
# =======================================================================================
def build_targets(cfg: DeployConfig, predictions: pl.DataFrame, sleeve: str) -> dict[str, float]:
    """STUB for the signal. Turn scores into per-symbol target OUNCES for one sleeve.

    The signal method is declared in ``setup.yaml::backtest.sweep.signal_specs`` and there are
    two: ``fixed_threshold`` at 0.0 and ``per_symbol_rolling_percentile`` at the 80th
    percentile over 63 days. **Which of the two** this bot trades is a phase-5 selection that
    has not been made, so this function cannot pick one; picking one here would be a selection
    made outside the backtest.

    What is NOT a stub, and is implemented below in :func:`floor_to_placeable`, is the sizing
    arithmetic - because that part does not depend on the model and it is where this bot is
    most likely to diverge from its own backtest.
    """
    raise NotReadyError(
        "step 6 (signal) cannot choose between the two declared signal specs "
        "(fixed_threshold_0, per_symbol_p80) - that choice is made in 13_backtest on validation "
        "Sharpe at the cumulative trial count, and 13_backtest has not run."
    )


def floor_to_placeable(
    cfg: DeployConfig, targets: dict[str, float]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Round every target DOWN to a volume the broker will actually accept.

    This is real, it is testable without a model, and it is the single most likely source of
    backtest/live divergence on this bot. ``setup.yaml::execution.share_type`` is ``integer``
    and the engine's unit is ``lots x trade_contract_size`` = **ounces**:

        XAUUSD  volume_min 0.01 x contract_size 100  = 1 oz   -> every integer is placeable
        XAGUSD  volume_min 0.01 x contract_size 5000 = 50 oz  -> only multiples of 50 are

    So a backtest that fills **73 oz of silver** is filling a volume that cannot be sent. The
    adapter's ``normalize_lot`` rounds down, which turns 73 into 50 - a 32 % under-fill - and
    turns 40 into 0, which is REJECTED rather than bumped up (``min_lot_policy: reject``).

    Returns the floored targets and one adjustment record per symbol that moved, so the run
    record carries the divergence rather than hiding it. ``tests/test_parity.py`` asserts this
    against the backtest's own fills (``PHASE1_SPEC_MENTOR.md`` section 2.e item 5).
    """
    minimum = cfg.min_units
    floored: dict[str, float] = {}
    adjustments: list[dict[str, Any]] = []
    for symbol, target in targets.items():
        step = minimum.get(symbol)
        if step is None:
            raise ValueError(f"no min_units declared for {symbol}; refusing to guess a lot step")
        sign = -1.0 if target < 0 else 1.0
        magnitude = abs(float(target))
        placeable = sign * (int(magnitude // step) * step)
        floored[symbol] = placeable
        if placeable != target:
            adjustments.append(
                {
                    "symbol": symbol,
                    "requested_units": target,
                    "placeable_units": placeable,
                    "granularity_units": step,
                    "dropped_units": target - placeable,
                    "rejected": placeable == 0.0 and magnitude > 0.0,
                    "reason": (
                        f"{symbol} is placeable only in multiples of {step} oz "
                        f"(volume_min {cfg.contract['volume_min']} x contract_size "
                        f"{cfg.contract['contract_size'][symbol]})"
                    ),
                }
            )
    return floored, adjustments


# =======================================================================================
# STEP 7 - Offline reference tape (25_live_trading/02:422)
#          *** STUB: needs the price grid fix and a registered backtest ***
# =======================================================================================
def replay_parity(cfg: DeployConfig, targets: dict[str, float]) -> dict[str, Any]:
    """STUB. Replay the same decision through ``ml4t.backtest.Engine`` and compare.

    This is the parity check of ``25_live_trading/08_pipeline_verification.py``: the offline
    engine and the live path must produce the same weights for the same decision, or one of
    them is not doing what the other thinks.

    It cannot run today for a reason that is separate from the missing model:
    ``_features.load_session_panel(for_backtest=True)`` **raises**, on purpose, because the
    registered price grid is the session panel and cannot express the 8-hour hold
    (``setup.yaml::backtest.price_grid_expresses_primary_label: false``). The fix is
    ``PRICE_GRID_DECLARATION.md`` - register the H1 bars and express the hold as a broker-level
    ``time_exit`` position rule - and it has not landed.
    """
    raise NotReadyError(
        "step 7 (offline reference tape) cannot run: the registered price grid does not express "
        "the primary label (setup.yaml::backtest.price_grid_expresses_primary_label is false and "
        "_features.load_session_panel(for_backtest=True) raises). Land "
        "bots/exness_gold_sess/PRICE_GRID_DECLARATION.md first."
    )


# =======================================================================================
# STEP 8 - Pre-flight and order staging through SafeBroker (25_live_trading/02:462)
# =======================================================================================
def build_live_risk_config(cfg: DeployConfig, *, armed: bool):
    """Construct ``ml4t.live.LiveRiskConfig`` from the YAML, with nothing defaulted silently.

    ``armed`` never reaches the dataclass as a field: arming is expressed by leaving
    ``shadow_mode`` false and setting ``execution_mode`` to the declared mode. A run that is
    not armed gets ``shadow_mode=True`` and ``execution_mode='shadow'``, which is what makes
    the default of every run a dry run.
    """
    from ml4t.live import LiveRiskConfig

    declared = dict(cfg.live_risk)
    state_dir = STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)

    if armed and not cfg.kill_criteria_approved:
        raise NotReadyError(
            "refusing to arm: deploy/risk_config.yaml::breakers.pending_user_approval is true. "
            "The kill criteria in bots/exness_gold_sess/BOT.md are the mentor's DRAFT and the "
            "user has not approved them. A bot may not send an order under a rule nobody agreed "
            "to. Get the approval, record its date in BOT.md, then clear the flag."
        )

    return LiveRiskConfig(
        max_position_value=float(declared["max_position_value"]),
        max_position_shares=float(declared["max_position_shares"]),
        max_total_exposure=float(declared["max_total_exposure"]),
        max_positions=int(declared["max_positions"]),
        max_order_value=float(declared["max_order_value"]),
        max_order_shares=float(declared["max_order_shares"]),
        max_orders_per_minute=int(declared["max_orders_per_minute"]),
        max_daily_loss=float(declared["max_daily_loss"]),
        max_drawdown_pct=float(declared["max_drawdown_pct"]),
        max_price_deviation_pct=float(declared["max_price_deviation_pct"]),
        max_data_staleness_seconds=float(declared["max_data_staleness_seconds"]),
        dedup_window_seconds=float(declared["dedup_window_seconds"]),
        allowed_assets=set(declared["allowed_assets"]),
        blocked_assets=set(declared["blocked_assets"]),
        shadow_mode=not armed,
        execution_mode="shadow" if not armed else str(cfg.execution["mode"]),
        kill_switch_enabled=bool(declared["kill_switch_enabled"]),
        allow_reducing_risk_when_killed=bool(declared["allow_reducing_risk_when_killed"]),
        halt_on_reducing_risk_failure=bool(declared["halt_on_reducing_risk_failure"]),
        fail_on_reconciliation_mismatch=bool(declared["fail_on_reconciliation_mismatch"]),
        state_file=str(state_dir / declared["state_file"]),
        journal_file=str(state_dir / declared["journal_file"]),
        fail_on_journal_error=bool(declared["fail_on_journal_error"]),
    )


def verify_contract_geometry(cfg: DeployConfig, broker: MT5Broker) -> list[dict[str, Any]]:
    """Read contract size, volume step and swap from ``symbol_info``; never assume them.

    ``mt5-exness-broker.md``: symbol names, contract sizes and swaps are READ from MT5, not
    assumed. This compares what the terminal says against what ``risk_config.yaml`` declares
    and returns one row per symbol. A mismatch on ``trade_contract_size`` silently rescales
    every position this bot sizes, so it is an error and not a warning.
    """
    rows = []
    for symbol in cfg.symbols:
        info = broker.symbol_info(symbol)
        declared_size = float(cfg.contract["contract_size"][symbol])
        actual_size = float(getattr(info, "trade_contract_size", float("nan")))
        declared_min = float(cfg.contract["volume_min"])
        actual_min = float(getattr(info, "volume_min", float("nan")))
        row = {
            "symbol": symbol,
            "declared_contract_size": declared_size,
            "actual_contract_size": actual_size,
            "declared_volume_min": declared_min,
            "actual_volume_min": actual_min,
            "actual_volume_step": float(getattr(info, "volume_step", float("nan"))),
            "swap_long": float(getattr(info, "swap_long", float("nan"))),
            "swap_short": float(getattr(info, "swap_short", float("nan"))),
            "swap_mode": getattr(info, "swap_mode", None),
            "swap_rollover3days": getattr(info, "swap_rollover3days", None),
            "min_units_declared": cfg.min_units[symbol],
            "min_units_actual": actual_min * actual_size,
        }
        row["contract_size_matches"] = actual_size == declared_size
        row["min_units_matches"] = row["min_units_actual"] == row["min_units_declared"]
        rows.append(row)
    mismatched = [r for r in rows if not (r["contract_size_matches"] and r["min_units_matches"])]
    if mismatched:
        raise ValueError(
            "symbol_info disagrees with deploy/risk_config.yaml::contract on "
            f"{[r['symbol'] for r in mismatched]}: {mismatched}. A wrong contract size rescales "
            "every position this bot sizes; fix the config against the terminal, not the "
            "other way round."
        )
    return rows


def preflight(cfg: DeployConfig, *, armed: bool = False) -> dict[str, Any]:
    """Everything that can be checked before a model exists. Read-only; sends no order.

    Order matters: the account-tier halt is read FIRST, because if it is set no bot on this
    login may stage anything and the rest of the checks are moot.
    """
    from ml4t.live import SafeBroker

    report: dict[str, Any] = {
        "bot_id": cfg.bot_id,
        "magic": cfg.magic,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "armed_requested": armed,
    }

    limits = load_account_limits()
    guard = AccountGuard(limits)
    halted = guard.is_halted() if hasattr(guard, "is_halted") else None
    report["account_halt"] = halted
    if halted:
        report["result"] = "REFUSED: the account-tier halt file is set; every bot stops"
        return report

    broker = MT5Broker(magic=cfg.magic, execution_mode=str(cfg.execution["mode"]))
    broker.ensure_connected()
    try:
        broker.assert_paper_trading()
        report["assert_paper_trading"] = "passed (the account is a demo)"
        account = broker.account_summary()
        report["account"] = {
            k: account.get(k)
            for k in ("login", "server", "currency", "balance", "equity", "margin_free", "leverage")
        }
        report["account_snapshot"] = read_account_snapshot(broker).__dict__
        report["contract_geometry"] = verify_contract_geometry(cfg, broker)

        risk = build_live_risk_config(cfg, armed=armed)
        safe = SafeBroker(broker, risk)
        report["safe_broker"] = {
            "shadow_mode": risk.shadow_mode,
            "execution_mode": str(risk.execution_mode),
            "state_file": risk.state_file,
        }
        report["reconciliation"] = reconcile(cfg, broker, believed={})
        report["result"] = (
            "PRE-FLIGHT PASSED. No order was staged: there is no model. "
            "shadow_mode=" + str(risk.shadow_mode)
        )
        del safe
    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()
    return report


def stage_orders(cfg: DeployConfig, safe_broker: Any, targets: dict[str, float]) -> list[dict]:
    """STUB above the sizing floor: there is nothing to stage until step 6 produces targets.

    When it does, this is the only place an order is created, and it does three things in this
    order for every symbol, none of which may be skipped:

    1. ``safe_broker.record_market_snapshot(symbol, price)`` from a **fresh**
       ``symbol_info_tick``. Without it the staleness guard blocks the order
       (``10_safety_risk_demo.py:352-354``);
    2. :func:`floor_to_placeable`, so silver is a multiple of 50 oz and a sub-lot target is a
       no-trade rather than a rounded-up trade;
    3. the order through ``SafeBroker``, never through ``MT5Broker`` directly - the pre-trade
       limits are the whole reason the wrapper exists.
    """
    raise NotReadyError("step 8 (stage orders) has no targets; step 6 raised.")


# =======================================================================================
# STEP 9 - Reconcile (25_live_trading/02:616)
# =======================================================================================
def reconcile(
    cfg: DeployConfig, broker: MT5Broker, believed: dict[str, float]
) -> dict[str, Any]:
    """Compare what the broker holds under magic 260903 with what this bot believes it holds.

    Kill criterion (d): *"reconciliation cua SafeBroker tim thay mot vi the khong giai thich
    duoc mang magic 260903"* - an unexplained position carrying this bot's magic is a halt, not
    a warning. Runs at startup as well as after staging, which is the Chapter 25 rule: a bot
    that reconciles only after trading cannot detect what happened while it was down.

    Note ``all_positions``: the account also carries ``xau_fx_mt5`` (magic 260905) trading
    XAUUSD and, historically, the retired legacy bot's magic 202500. Positions of another magic
    are reported as context and are **not** this bot's to touch.
    """
    ours = broker.positions()
    foreign = [p for p in broker.all_positions() if int(p.get("magic", -1)) != cfg.magic]
    unexplained = sorted(set(ours) - set(believed))
    missing = sorted(set(believed) - set(ours))
    mismatched = [
        {
            "symbol": symbol,
            "broker_units": getattr(ours[symbol], "quantity", None),
            "believed_units": believed[symbol],
        }
        for symbol in sorted(set(ours) & set(believed))
        if getattr(ours[symbol], "quantity", None) != believed[symbol]
    ]
    return {
        "magic": cfg.magic,
        "broker_positions": {s: getattr(p, "quantity", None) for s, p in ours.items()},
        "believed_positions": believed,
        "unexplained": unexplained,
        "missing": missing,
        "mismatched": mismatched,
        "foreign_magics": sorted({int(p.get("magic", -1)) for p in foreign}),
        "foreign_position_count": len(foreign),
        "halt": bool(unexplained or missing or mismatched),
        "kill_criterion": "d" if (unexplained or missing or mismatched) else None,
    }


# =======================================================================================
# STEP 10 - Persist the run record (25_live_trading/02:676)
# =======================================================================================
def write_run_record(record: dict[str, Any], *, state_dir: Path = STATE_DIR) -> Path:
    """One JSON per cycle under ``deploy/state/runs/``. Chapter 26 reads these back.

    Everything the monitoring layer needs has to be in here, because nothing else records it:
    the decision instant and its session, which bars were used, what was staged, what the
    granularity floor dropped, the reconciliation result and every breaker's state.
    """
    runs = state_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = str(record.get("decision_ts") or datetime.now(UTC).isoformat())
    safe_stamp = stamp.replace(":", "").replace(" ", "T")[:19]
    path = runs / f"{safe_stamp}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    return path


# =======================================================================================
# The cycle
# =======================================================================================
def run_cycle(
    cfg: DeployConfig,
    *,
    as_of: datetime | None = None,
    armed: bool = SUBMIT_ORDERS,
    refresh: bool = REFRESH_DATA,
) -> dict[str, Any]:
    """One decision cycle, per sleeve. Stops at the first step that is not ready and says so.

    It does not swallow :class:`NotReadyError`: the record carries which step stopped and why,
    and the caller sees a non-zero exit. A loop that logged "step 3 skipped" and carried on
    would eventually stage an order with no model behind it.
    """
    record: dict[str, Any] = {
        "bot_id": cfg.bot_id,
        "magic": cfg.magic,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "armed": armed,
        "sleeves": cfg.sleeves,
        "steps": {},
    }
    try:
        bars = refresh_data(cfg, refresh=refresh)
        record["steps"]["1_refresh_data"] = {
            k: v for k, v in bars.items() if k not in ("h1", "d1")
        }

        at = as_of or datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        panel = recompute_features(cfg, bars, as_of=at)
        record["decision_ts"] = str(panel.decision_ts)
        record["session"] = panel.session
        record["steps"]["2_features"] = {
            "rows": panel.rows.height,
            "feature_columns": len(panel.feature_columns),
            "bars_used_through": str(panel.bars_used_through),
            "decision_bar_check": decision_bar_check(cfg, panel),
        }

        model = retrain(cfg, panel)  # raises NotReadyError today
        record["steps"]["3_refit"] = "ok"
        persist(model, state_dir=STATE_DIR, decision_ts=panel.decision_ts)
        predictions = predict(model, panel)
        for sleeve in cfg.sleeves:
            targets = build_targets(cfg, predictions, sleeve)
            floored, adjustments = floor_to_placeable(cfg, targets)
            record["steps"].setdefault("6_targets", {})[sleeve] = {
                "targets": floored,
                "granularity_adjustments": adjustments,
            }
        replay_parity(cfg, floored)
    except NotReadyError as exc:
        record["stopped_at"] = _current_step(record)
        record["not_ready"] = str(exc)
        logger.warning("cycle stopped at %s: %s", record["stopped_at"], exc)
    record["finished_at_utc"] = datetime.now(UTC).isoformat()
    record["run_record"] = str(write_run_record(record))
    return record


def _current_step(record: dict[str, Any]) -> str:
    done = list(record.get("steps", {}))
    return f"after {done[-1]}" if done else "before step 1"


# =======================================================================================
# CLI
# =======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preflight", action="store_true", help="read-only checks; no order")
    parser.add_argument("--cycle", action="store_true", help="run one decision cycle")
    parser.add_argument(
        "--arm",
        action="store_true",
        help="permit orders FOR THIS SESSION ONLY; refused while the kill criteria are a draft",
    )
    parser.add_argument("--as-of", default=None, help="decision instant, ISO UTC")
    parser.add_argument("--refresh-data", action="store_true", help="needs a Windows MT5 terminal")
    parser.add_argument("--risk-config", default=None, help="override deploy/risk_config.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_deploy_config(args.risk_config)

    if args.arm and not cfg.kill_criteria_approved:
        print(
            "REFUSED: --arm is not available. deploy/risk_config.yaml::breakers."
            "pending_user_approval is true - the kill criteria in BOT.md are the mentor's DRAFT "
            "and the user has not approved them.",
            file=sys.stderr,
        )
        return 2

    if args.preflight:
        print(json.dumps(preflight(cfg, armed=args.arm), indent=2, default=str))
        return 0
    if args.cycle:
        as_of = datetime.fromisoformat(args.as_of).replace(tzinfo=UTC) if args.as_of else None
        record = run_cycle(cfg, as_of=as_of, armed=args.arm, refresh=args.refresh_data)
        print(json.dumps(record, indent=2, default=str))
        return 1 if record.get("not_ready") else 0

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
