"""Measure spread by session and re-read symbol_info for BTCUSD (Exness MT5).

Preparation task for phase 1 of ``exness_btc_8h`` (2026-09-08). Windows only: the
``MetaTrader5`` package needs a running, logged-in terminal. Run it from the repository
root with the isolated interpreter the other Exness bots use for MT5 work (the project env
does not build on Windows, ``docs/installation.md:115-118``)::

    PYTHONPATH=<repo root> uv run --no-project --python 3.12 \
        --with MetaTrader5==5.0.5640 --with polars==1.41.1 --with numpy --with pyarrow \
        --with pyyaml --with python-dotenv --with pandas \
        python bots/exness_btc_8h/tools/measure_btc_costs.py

Same shape as ``bots/exness_gold_sess/tools/measure_metals_costs.py`` (2026-09-07), which
appended the two metals to the five FX rows measured on 2026-09-05. This one appends
BTCUSD and touches no existing row.

What it does, in order:

1. connects read-only (there is no ``order_send`` anywhere in this file);
2. measures the :class:`~bots._shared.sessions.ServerClock` on BTCUSD itself - the symbol
   trades 24/7, so its last tick is "now" on any day, weekend included;
3. runs :func:`bots._shared.costs_mt5.measure_spreads` over ``DAYS`` days of
   ``COPY_TICKS_INFO`` ticks;
4. reads the full ``symbol_info`` record and the swap parameters, including the
   ``charges_weekends`` flag :func:`bots._shared.costs_mt5.read_swaps` derives from the
   Market Watch path (``Standard\\Crypto\\BTCUSDm`` -> every night is charged, no weekend
   exemption);
5. prints the swap points the crypto rule implies over one 8h bar, one night and one week
   from :func:`bots._shared.costs_mt5.holding_cost_points`, so ``setup.yaml`` can carry a
   measured holding cost rather than a guess;
6. backs up ``spreads_by_session.{json,parquet}`` as ``*.bak_<STAMP>`` and rewrites them
   with the existing rows **unchanged** plus BTCUSD appended.

Nothing is selected, fitted or backtested here; this only records what the account costs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import MetaTrader5 as mt5  # noqa: E402

from bots._shared.costs_mt5 import (  # noqa: E402
    SESSION_BUCKETS,
    SUMMARY_FIELDS,
    holding_cost_points,
    measure_spreads,
    read_spreads,
    read_swaps,
    swap_points_to_account,
)
from bots._shared.mt5_loader import to_market_watch  # noqa: E402
from bots._shared.sessions import ServerClock  # noqa: E402

SYMBOLS = ("BTCUSD",)
CLOCK_SYMBOL = "BTCUSDm"  # 24/7, so the last tick is "now" on any day
DAYS = 30
STAMP = "2026-09-08"
DATA_DIR = Path(os.environ.get("ML4T_DATA_PATH", REPO_ROOT / "data")) / "mt5"

SYMBOL_INFO_FIELDS = (
    "name",
    "path",
    "description",
    "currency_base",
    "currency_profit",
    "currency_margin",
    "trade_contract_size",
    "volume_min",
    "volume_step",
    "volume_max",
    "digits",
    "point",
    "spread",
    "spread_float",
    "trade_tick_size",
    "trade_tick_value",
    "trade_stops_level",
    "trade_freeze_level",
    "trade_mode",
    "trade_calc_mode",
    "filling_mode",
    "swap_mode",
    "swap_long",
    "swap_short",
    "swap_rollover3days",
    "session_deals",
    "session_buy_orders",
    "session_sell_orders",
    "volume",
    "time",
    "bid",
    "ask",
)


def dump_symbol_info() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for bare in SYMBOLS:
        mw = to_market_watch(bare)
        info = mt5.symbol_info(mw)
        if info is None:
            raise RuntimeError(f"{mw!r} is not in Market Watch ({mt5.last_error()})")
        out[bare] = {f: getattr(info, f, None) for f in SYMBOL_INFO_FIELDS}
    return out


def merge_and_write(new_rows: pl.DataFrame) -> tuple[Path, Path]:
    """Append ``new_rows`` to the existing table; never touch a row already there."""
    parquet = DATA_DIR / "spreads_by_session.parquet"
    js = DATA_DIR / "spreads_by_session.json"
    for path in (parquet, js):
        backup = path.with_name(path.name + f".bak_{STAMP}")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"backup: {backup}")

    old_table = read_spreads(DATA_DIR)
    keep = old_table.filter(~pl.col("symbol").is_in(list(SYMBOLS)))
    print(f"existing rows kept: {keep.height} ({keep['symbol'].n_unique()} symbols)")
    merged = pl.concat([keep, new_rows.select(keep.columns)], how="vertical")
    merged.write_parquet(parquet)

    payload = json.loads(js.read_text(encoding="utf-8"))
    payload["buckets"] = list(("all", *SESSION_BUCKETS))
    windows = payload.get("windows", {})
    for key, part in merged.group_by("symbol", maintain_order=True):
        symbol = key[0] if isinstance(key, tuple) else key
        payload["spreads"][symbol] = {
            r["session"]: {k: r[k] for k in SUMMARY_FIELDS} for r in part.to_dicts()
        }
        first = part.row(0, named=True)
        windows[symbol] = {
            "window_start_utc": str(first["window_start_utc"]),
            "window_end_utc": str(first["window_end_utc"]),
            "days": int(first["days"]),
            "measured_at_utc": str(first["measured_at_utc"]),
            "server_utc_offset_minutes": int(first["server_utc_offset_minutes"]),
        }
    payload["windows"] = windows
    payload["windows_note"] = (
        "The top-level window_* / measured_at_utc keys describe the first row of the "
        "parquet (the 2026-09-05 FX measurement) and are kept for backward compatibility; "
        "read `windows` for the per-symbol window."
    )
    js.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return parquet, js


def report_holding_cost(swaps: dict, clock: ServerClock, mid: float) -> dict:
    """Swap points and USD the crypto rule charges over the windows this bot holds for.

    ``2026-09-07`` is a Monday, so the week case crosses seven server midnights of which one
    is ``swap_rollover3days`` (Friday on this symbol, MT5 day-of-week 5).
    """
    bare = SYMBOLS[0]
    base = datetime(2026, 9, 7, 0, 0, tzinfo=UTC)  # Monday 00:00 UTC = server midnight
    cases = {
        "one 8h bar, no midnight (Mon 00:00 -> 08:00)": (base, base + timedelta(hours=8)),
        "one 8h bar over midnight (Sun 16:00 -> Mon 00:00)": (base - timedelta(hours=8), base),
        "one day (Mon 00:00 -> Tue 00:00)": (base, base + timedelta(days=1)),
        "one week (Mon 00:00 -> next Mon 00:00)": (base, base + timedelta(days=7)),
    }
    params = swaps[bare]
    out = {}
    for name, (a, b) in cases.items():
        row = {}
        for side in ("long", "short"):
            points = holding_cost_points(bare, side, a, b, swaps=swaps, clock=clock)
            row[f"{side}_points"] = points
            row[f"{side}_usd_per_001_lot"] = swap_points_to_account(points, params, lots=0.01)
            if mid:
                row[f"{side}_bps_of_notional"] = (
                    row[f"{side}_usd_per_001_lot"] / (0.01 * float(params["trade_contract_size"]) * mid) * 1e4
                )
        out[name] = row
    return out


def main() -> int:
    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5.initialize() failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        print(
            f"account {account.login} @ {account.server} trade_mode={account.trade_mode} "
            f"{account.currency} leverage 1:{account.leverage}; terminal {mt5.version()} "
            f"connected={terminal.connected} maxbars={terminal.maxbars}"
        )
        if account.trade_mode != 0:
            raise RuntimeError("this script is read-only but refuses to run on a live account")

        clock = ServerClock.measure(mt5, CLOCK_SYMBOL)
        stored = json.loads((DATA_DIR / "sessions_mt5.json").read_text(encoding="utf-8"))
        print(
            f"server clock measured on {CLOCK_SYMBOL}: {clock.utc_offset_minutes:+d} min; "
            f"recorded 2026-09-05: {stored['server_clock']}"
        )

        info = dump_symbol_info()
        swaps = read_swaps(SYMBOLS, mt5)
        for bare in SYMBOLS:
            print(f"\n--- symbol_info {bare} ---")
            for key, value in info[bare].items():
                print(f"  {key:24s} {value}")
            print(f"  charges_weekends         {swaps[bare]['charges_weekends']}")

        bid, ask = info[SYMBOLS[0]]["bid"], info[SYMBOLS[0]]["ask"]
        mid = (float(bid) + float(ask)) / 2 if bid and ask else 0.0
        print(f"\n--- holding cost, crypto rule: every night incl. weekends, x3 on the rollover day ---")
        print(f"    (mid {mid:,.2f}; USD figures are for one 0.01 lot = {0.01 * float(info[SYMBOLS[0]]['trade_contract_size']):.2f} BTC)")
        holding = report_holding_cost(swaps, clock, mid)
        for name, values in holding.items():
            print(
                f"  {name:52s} long {values['long_points']:>12.1f} pts "
                f"({values['long_usd_per_001_lot']:>8.3f} USD, {values.get('long_bps_of_notional', 0):>7.1f} bps)"
                f"   short {values['short_points']:>8.1f} pts"
            )

        print(f"\n--- measure_spreads({list(SYMBOLS)}, days={DAYS}) ---")
        table = measure_spreads(SYMBOLS, DAYS, mt5, clock=clock, initialize=False)
        with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=220, ascii_tables=True):
            print(table)

        parquet, js = merge_and_write(table)
        print(f"\nwrote {parquet}")
        print(f"wrote {js}")

        out = DATA_DIR / f"symbol_info_btc_{STAMP}.json"
        out.write_text(
            json.dumps(
                {
                    "measured_at_utc": datetime.now(UTC).isoformat(),
                    "account": {
                        "login": account.login,
                        "server": account.server,
                        "trade_mode": account.trade_mode,
                        "currency": account.currency,
                        "leverage": account.leverage,
                    },
                    "server_clock": clock.as_dict(),
                    "symbols": info,
                    "swaps": swaps,
                    "holding_cost": holding,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
