"""Weekend vs weekday spread on BTCUSD, from this account's own ticks (Exness MT5).

``bots/_shared/costs_mt5.py::measure_spreads`` buckets a tick by *venue session*, and every
venue window in ``bots/_shared/sessions.py`` is weekday-only (``WEEKDAYS = (0, 1, 2, 3, 4)``
in the venue's local calendar). For a 24/7 symbol that puts the whole weekend into the
``other`` bucket together with the weekday gaps between sessions, so the table
``measure_btc_costs.py`` writes cannot answer "what does BTCUSD cost on a Saturday".

This bot trades 24/7 and its asset profile (``bots/assets/BTCUSD.md`` section 5) tells it to
avoid the weekend because "thanh khoan mong, spread CFD rong". That is a claim about this
account, so it is measured here rather than assumed, over the same 30-day tick window and
with the same ``COPY_TICKS_INFO`` reader, split four ways:

    weekday / saturday / sunday / weekend (saturday + sunday), UTC calendar days.

Windows only; run it the same way as ``measure_btc_costs.py``::

    PYTHONIOENCODING=utf-8 PYTHONPATH=<repo root> uv run --no-project --python 3.12 \
        --with MetaTrader5==5.0.5640 --with polars==1.41.1 --with numpy --with pyarrow \
        python bots/exness_btc_8h/tools/measure_btc_weekend_spread.py

Read-only: there is no ``order_send`` in this file. Writes
``ML4T_DATA_PATH/mt5/spread_weekend_BTCUSD_<STAMP>.json`` and touches no shared table.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import MetaTrader5 as mt5  # noqa: E402

from bots._shared.costs_mt5 import ticks_to_frame  # noqa: E402
from bots._shared.mt5_loader import to_market_watch  # noqa: E402
from bots._shared.sessions import ServerClock  # noqa: E402

SYMBOL = "BTCUSD"
DAYS = 30
STAMP = "2026-09-08"
DATA_DIR = Path(os.environ.get("ML4T_DATA_PATH", REPO_ROOT / "data")) / "mt5"


def summarize(frame: pl.DataFrame, point: float) -> dict:
    """p50 / p90 / mean of the quoted spread, in bps of mid and in points."""
    if frame.is_empty():
        return {"n_ticks": 0}
    bps = frame["spread_bps"].to_numpy()
    pts = np.asarray(bps) / 1e4 * frame["mid"].to_numpy() / point
    return {
        "n_ticks": int(frame.height),
        "spread_bps_p50": round(float(np.percentile(bps, 50)), 3),
        "spread_bps_p90": round(float(np.percentile(bps, 90)), 3),
        "spread_bps_mean": round(float(np.mean(bps)), 3),
        "spread_points_p50": round(float(np.percentile(pts, 50)), 1),
        "spread_points_p90": round(float(np.percentile(pts, 90)), 1),
    }


def main() -> int:
    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5.initialize() failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        if account.trade_mode != 0:
            raise RuntimeError("read-only script, but it refuses to run on a live account")
        print(f"account {account.login} @ {account.server} trade_mode={account.trade_mode}")

        mw = to_market_watch(SYMBOL)
        info = mt5.symbol_info(mw)
        if info is None:
            raise RuntimeError(f"{mw!r} is not in Market Watch ({mt5.last_error()})")
        mt5.symbol_select(mw, True)
        point = float(info.point)
        clock = ServerClock.measure(mt5, mw)
        flags = int(getattr(mt5, "COPY_TICKS_INFO", 1))

        end_utc = datetime.now(UTC)
        start_utc = end_utc - timedelta(days=DAYS)
        cursor, parts = start_utc, []
        while cursor < end_utc:
            stop = min(cursor + timedelta(days=1), end_utc)
            ticks = mt5.copy_ticks_range(mw, clock.to_server(cursor), clock.to_server(stop), flags)
            cursor = stop
            if ticks is None or len(ticks) == 0:
                continue
            frame = ticks_to_frame(ticks, SYMBOL, clock)
            if not frame.is_empty():
                parts.append(
                    frame.select(
                        "timestamp",
                        "spread_bps",
                        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
                    )
                )
        if not parts:
            raise RuntimeError("no ticks returned")
        ticks_frame = pl.concat(parts).with_columns(
            pl.col("timestamp").dt.weekday().alias("dow")  # polars: 1 = Monday .. 7 = Sunday
        )
        print(f"{ticks_frame.height:,} ticks, {start_utc:%Y-%m-%d %H:%M} .. {end_utc:%Y-%m-%d %H:%M} UTC")

        buckets = {
            "all": ticks_frame,
            "weekday": ticks_frame.filter(pl.col("dow") <= 5),
            "saturday": ticks_frame.filter(pl.col("dow") == 6),
            "sunday": ticks_frame.filter(pl.col("dow") == 7),
            "weekend": ticks_frame.filter(pl.col("dow") >= 6),
        }
        out = {
            "symbol": SYMBOL,
            "market_watch": mw,
            "point": point,
            "days": DAYS,
            "window_start_utc": start_utc.isoformat(),
            "window_end_utc": end_utc.isoformat(),
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "account": {"login": account.login, "server": account.server, "trade_mode": account.trade_mode},
            "note": (
                "UTC calendar-day split. measure_spreads' session buckets are weekday-only "
                "(sessions.py WEEKDAYS), so its `other` bucket mixes the weekend with the "
                "weekday gaps between venue sessions; this file separates them."
            ),
            "buckets": {name: summarize(part, point) for name, part in buckets.items()},
        }
        for name, values in out["buckets"].items():
            print(f"  {name:9s} {values}")
        path = DATA_DIR / f"spread_weekend_{SYMBOL}_{STAMP}.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
