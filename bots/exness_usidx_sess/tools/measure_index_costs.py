"""Measure spread by session and re-read symbol_info for US500 and USTEC (Exness MT5).

Preparation task for phase 1 of ``exness_usidx_sess`` (2026-09-08). Windows only: the
``MetaTrader5`` package needs a running, logged-in terminal. Run it from the repository
root with the isolated interpreter the ``exness_fx_d1`` bot uses for MT5 work (the project
env does not build on Windows, ``docs/installation.md:115-118``)::

    PYTHONPATH=<repo root> uv run --no-project --python 3.14 \
        --with MetaTrader5 --with polars==1.41.1 --with numpy --with pyarrow \
        --with pyyaml --with python-dotenv --with pandas \
        python bots/exness_usidx_sess/tools/measure_index_costs.py

What it does, in order:

1. connects read-only (there is no ``order_send`` anywhere in this file);
2. measures the :class:`~bots._shared.sessions.ServerClock` on the 24/7 crypto symbol and
   compares it with the one recorded in ``ML4T_DATA_PATH/mt5/sessions_mt5.json``;
3. runs :func:`bots._shared.costs_mt5.measure_spreads` over ``DAYS`` days of
   ``COPY_TICKS_INFO`` ticks for the two indices, the same call the five FX pairs of
   ``exness_fx_d1`` were measured with on 2026-09-05, but with ``overlay=True`` so the
   table also carries the ``us_cash`` / ``edge_open`` / ``edge_close`` buckets this bot
   executes in (the p90 of those buckets is what phase 6 must beat, not the 24h ``all``);
4. reads the full ``symbol_info`` record of both symbols;
5. backs up ``spreads_by_session.{json,parquet}`` as ``*.bak_<STAMP>`` and rewrites them
   with the existing rows **unchanged** plus the two index symbols appended. Every row of
   the parquet carries its own measurement window, so the FX rows keep their 2026-09-05
   window and the index rows carry today's. The JSON keeps its top-level scalars (they
   describe the first row, i.e. the FX window) and gains a ``windows`` block giving the
   window and ``measured_at`` per symbol.

Nothing is selected, fitted or backtested here; this only records what the account costs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import MetaTrader5 as mt5  # noqa: E402

from bots._shared.costs_mt5 import (  # noqa: E402
    OVERLAY_BUCKETS,
    SESSION_BUCKETS,
    SUMMARY_FIELDS,
    measure_spreads,
    read_spreads,
    read_swaps,
)
from bots._shared.mt5_loader import to_market_watch  # noqa: E402
from bots._shared.sessions import ServerClock  # noqa: E402

SYMBOLS = ("US500", "USTEC")
CLOCK_SYMBOL = "BTCUSDm"  # 24/7, so the last tick is "now" on any weekday
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
    payload["buckets"] = list(("all", *SESSION_BUCKETS, *OVERLAY_BUCKETS))
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


def main() -> int:
    # The Windows console this runs on is cp1252; polars writes a micro sign in the
    # Datetime[us] dtype header, which cp1252 cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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

        print(f"\n--- measure_spreads({list(SYMBOLS)}, days={DAYS}) ---")
        table = measure_spreads(SYMBOLS, DAYS, mt5, clock=clock, initialize=False, overlay=True)
        # ASCII table: the Windows console this runs on is cp1252 and cannot encode the
        # Unicode box characters polars draws by default.
        with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=220, ascii_tables=True):
            print(table)

        parquet, js = merge_and_write(table)
        print(f"\nwrote {parquet}")
        print(f"wrote {js}")

        out = DATA_DIR / f"symbol_info_indices_{STAMP}.json"
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
