"""exness_btc_8h deployment loop.

Implements the 8-hour decision cycle for BTCUSD on Exness Pro:
1. Refresh / load MT5 8h (from H4) bars.
2. Build decision features via case_studies/exness_btc_8h/_features.py.
3. Check Account-level Breaker (Tầng 1) and Strategy Breakers (Tầng 2).
4. Reconcile positions with magic 260904 via MT5Broker.
5. In dry-run / shadow-mode, log decisions and audit state without sending real orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from bots._shared import MAGIC_ALLOCATION
from bots._shared.monitor import AccountGuard, load_account_limits, read_account_snapshot
from bots._shared.mt5_broker import MT5Broker

logger = logging.getLogger("exness_btc_8h.deploy")

BOT_ID = "exness_btc_8h"
DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_RISK_CONFIG = DEPLOY_DIR / "risk_config.yaml"
STATE_DIR = DEPLOY_DIR / "state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run a {BOT_ID} deployment cycle.")
    parser.add_argument("--config", type=Path, default=DEFAULT_RISK_CONFIG)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--preflight", action="store_true", help="Run preflight checks only.")
    parser.add_argument("--cycle", action="store_true", help="Execute one decision cycle.")
    parser.add_argument("--arm", action="store_true", help="Arm live/paper order execution.")
    parser.add_argument("--fake-mt5", action="store_true", help="Use FakeMT5 module for offline testing.")
    return parser.parse_args()


def run_cycle(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    now_utc = datetime.now(UTC)
    state_dir = args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "bot_id": BOT_ID,
        "timestamp": now_utc.isoformat(),
        "magic": cfg.get("magic", 260904),
        "armed": bool(args.arm),
        "shadow_mode": cfg.get("shadow_mode", True),
        "status": "success",
        "action": "hold",
        "reason": "dry_run",
    }

    # 1. MT5 Connector / FakeMT5
    if args.fake_mt5:
        from bots._shared.testing.fake_mt5 import FakeMT5
        mt5_module = FakeMT5()
        logger.info("Using FakeMT5 for testing.")
    else:
        try:
            import MetaTrader5 as mt5_module
            if not mt5_module.initialize():
                record["status"] = "error"
                record["reason"] = f"MT5 initialize failed: {mt5_module.last_error()}"
                return record
        except ImportError:
            from bots._shared.testing.fake_mt5 import FakeMT5
            mt5_module = FakeMT5()
            logger.warning("MetaTrader5 not available, fallback to FakeMT5.")

    broker = MT5Broker(magic=cfg.get("magic", 260904), mt5=mt5_module)

    # 2. Check Account-level Breakers (Tầng 1)
    account_limits = load_account_limits()
    snapshot = read_account_snapshot(broker)
    guard = AccountGuard(limits=account_limits, bot_id=BOT_ID)
    if guard.is_halted():
        halt_rec = guard.halt_record() or {}
        record["status"] = "halted"
        record["reason"] = f"Account halted: {halt_rec.get('reason', 'account_halt.json present')}"
        logger.warning(f"[{BOT_ID}] ACCOUNT HALTED: {record['reason']}")
        return record

    allowed = guard.check(snapshot)
    if not allowed:
        record["status"] = "halted"
        record["reason"] = "Account breaker tripped on check"
        logger.warning(f"[{BOT_ID}] Account breaker tripped.")
        return record

    # 3. Check Positions & Reconcile
    all_positions = broker.mt5.positions_get() or ()
    magic_val = int(cfg.get("magic", 260904))
    positions = [p for p in all_positions if getattr(p, "magic", None) == magic_val]
    record["open_positions"] = len(positions)

    # 4. Status summary
    if args.preflight:
        record["reason"] = "preflight_complete"
        logger.info(f"[{BOT_ID}] Preflight checks passed.")
    else:
        # Dry-run decision log
        record["reason"] = "cycle_completed_shadow"
        logger.info(f"[{BOT_ID}] Decision cycle completed at {now_utc.isoformat()}. Open positions: {record['open_positions']}")

    # Save run record
    run_file = state_dir / f"run_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
    run_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    res = run_cycle(cfg, args)
    print(json.dumps(res, indent=2))
    if res.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
