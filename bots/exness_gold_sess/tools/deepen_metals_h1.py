"""Task B0 — re-read XAUUSD/XAGUSD H1 with the COUNT-based path and extend the data layer.

Why (mentor spec ``bots/exness_gold_sess/PHASE1_SPEC_MENTOR.md`` §0): the "H1 only goes back to
2022-10-25" number that made ``exness_fx_d1`` reject an H1 design is an artefact of
``copy_rates_range``, which serves only the bars the terminal has already synchronised. A
count-based ``copy_rates_from`` makes the terminal pull the older history from the server; the
``xau_fx_mt5`` bot measured 2017-03-06 for XAUUSD H1 on this same login. XAGUSD has never been
measured this way anywhere in the repository.

Windows only, terminal must be running and logged in. Run from the repository root with the
isolated interpreter (the project env does not build on Windows)::

    PYTHONPATH=<repo root> uv run --no-project --python 3.12 \
        --with MetaTrader5==5.0.5640 --with polars==1.41.1 --with numpy --with pyarrow \
        --with pyyaml --with python-dotenv --with pandas --with tzdata \
        python bots/exness_gold_sess/tools/deepen_metals_h1.py

What it does, in order (read-only against the account; there is no ``order_send`` here):

1. connects, measures the :class:`~bots._shared.sessions.ServerClock`;
2. reads H1 for both metals through :func:`bots._shared.mt5_loader.fetch_mt5_bars_deep`
   (``copy_rates_from`` + epoch seconds + a bar count), capped at the last bar the existing
   panel already holds so the RIGHT edge of the panel does not move — only the left is extended;
3. reports bars / first / last / ``dense_history_start`` per symbol, the last computed with the
   ``min_bars_per_week=100`` rule of ``experiments/xau_fx_mt5/data/build_panel.py:789``;
4. backs up ``1h.parquet`` as ``1h.parquet.bak_2026-09-07b``, replaces ONLY the two metals' rows
   and verifies with an exact frame comparison that every other symbol's rows are unchanged;
5. updates ``history_depth.json`` for the two metals (``history_mode: "deep"`` plus a note).

Nothing is fitted, selected or backtested; this only records what the account will serve.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bots._shared.mt5_loader import (  # noqa: E402
    fetch_mt5_bars_deep,
    mt5_data_dir,
    to_market_watch,
)
from bots._shared.sessions import ServerClock  # noqa: E402

SYMBOLS = ["XAUUSD", "XAGUSD"]
CLOCK_SYMBOL = "BTCUSD"  # 24/7, so the last tick is "now" on any weekday
STAMP = "2026-09-07b"
# The last H1 bar the existing panel holds for these symbols (measured before this run). Capping
# the request here keeps the right edge of the panel identical to the FX symbols', so the only
# change this task makes to the data layer is history added on the LEFT.
END_SERVER = datetime(2026, 9, 4, 20, 0)
HISTORY_FLOOR = datetime(2000, 1, 1)  # a floor for the backwards walk, not a request window
DENSE_MIN_BARS_PER_WEEK = 100


def dense_history_start(frame: pl.DataFrame, *, min_bars_per_week: int = DENSE_MIN_BARS_PER_WEEK) -> str | None:
    """First ISO week from which the H1 grid stays dense.

    Verbatim rule of ``experiments/xau_fx_mt5/data/build_panel.py:dense_history_start``: the first
    week that is itself dense and after which at least 85 % of the next 52 weeks are dense
    (holiday weeks allowed). A D1-backfilled or thinly synchronised prefix is one bar a day and
    fails the threshold, which is exactly what this is for.
    """
    weekly = (
        frame.with_columns(pl.col("timestamp").dt.truncate("1w").alias("week"))
        .group_by("week")
        .len()
        .sort("week")
    )
    counts = weekly["len"].to_numpy()
    weeks = weekly["week"].to_list()
    dense = counts >= min_bars_per_week
    for i in range(len(dense)):
        window = dense[i : i + 52]
        if dense[i] and len(window) >= 26 and window.sum() >= 0.85 * len(window):
            return str(weeks[i].date())
    return None


def bars_per_week_head(frame: pl.DataFrame, n: int = 8) -> list[tuple[str, int]]:
    weekly = (
        frame.with_columns(pl.col("timestamp").dt.truncate("1w").alias("week"))
        .group_by("week")
        .len()
        .sort("week")
        .head(n)
    )
    return [(str(w.date()), int(c)) for w, c in zip(weekly["week"].to_list(), weekly["len"].to_list())]


def main() -> int:
    import MetaTrader5 as mt5

    data_dir = mt5_data_dir()
    h1_path = data_dir / "1h.parquet"
    depth_path = data_dir / "history_depth.json"
    before = pl.read_parquet(h1_path)
    print(f"[panel] {h1_path}: {before.height:,} rows, {before['symbol'].n_unique()} symbols")

    # ---- 1. clock -------------------------------------------------------------------------
    if not mt5.initialize():
        raise SystemExit(f"MetaTrader5.initialize() failed: {mt5.last_error()}")
    try:
        acct = mt5.account_info()
        print(f"[account] login={acct.login} server={acct.server} trade_mode={acct.trade_mode}")
        clock = ServerClock.measure(mt5, to_market_watch(CLOCK_SYMBOL), follows_dst_of=None)
        print(f"[clock] utc_offset_minutes={clock.utc_offset_minutes} measured_at={clock.measured_at}")
    finally:
        mt5.shutdown()

    # ---- 2. count-based read --------------------------------------------------------------
    frames, depth_new = fetch_mt5_bars_deep(
        SYMBOLS, ("1h",), start=HISTORY_FLOOR, end=END_SERVER, clock=clock, mt5=mt5
    )
    deep = frames["1h"]
    report: dict[str, Any] = {}
    for bare in SYMBOLS:
        g = deep.filter(pl.col("symbol") == bare)
        dhs = dense_history_start(g)
        report[bare] = {
            "n_bars": g.height,
            "first": str(g["timestamp"].min()),
            "last": str(g["timestamp"].max()),
            "dense_history_start": dhs,
            "first_weeks": bars_per_week_head(g),
        }
        old = before.filter(pl.col("symbol") == bare)
        print(
            f"[deep] {bare}: {g.height:,} bars  {g['timestamp'].min()} -> {g['timestamp'].max()}"
            f"  dense_history_start={dhs}  (range path had {old.height:,} from {old['timestamp'].min()})"
        )
        print(f"        first weeks (bars/week): {report[bare]['first_weeks']}")

    # overlap check: do the two methods agree where they overlap?
    for bare in SYMBOLS:
        old = before.filter(pl.col("symbol") == bare)
        new = deep.filter(pl.col("symbol") == bare)
        common = old.join(new, on="timestamp", how="inner", suffix="_new")
        diff = common.filter(
            (pl.col("open") != pl.col("open_new"))
            | (pl.col("high") != pl.col("high_new"))
            | (pl.col("low") != pl.col("low_new"))
            | (pl.col("close") != pl.col("close_new"))
        )
        missing_old = old.join(new.select("timestamp"), on="timestamp", how="anti")
        report[bare]["overlap_rows"] = common.height
        report[bare]["overlap_ohlc_mismatch"] = diff.height
        report[bare]["in_old_not_in_deep"] = missing_old.height
        print(
            f"[overlap] {bare}: {common.height:,} shared bars, {diff.height} OHLC mismatches, "
            f"{missing_old.height} bars present in the old file but not in the deep read"
        )

    # ---- 3. backup and merge --------------------------------------------------------------
    backup = h1_path.with_suffix(f".parquet.bak_{STAMP}")
    shutil.copy2(h1_path, backup)
    print(f"[backup] {backup}")

    others = before.filter(~pl.col("symbol").is_in(SYMBOLS))
    after = pl.concat([others, deep.select(before.columns)]).sort(["symbol", "timestamp"])
    tmp = h1_path.with_suffix(".parquet.tmp")
    after.write_parquet(tmp)
    tmp.replace(h1_path)
    print(f"[write] {h1_path}: {before.height:,} -> {after.height:,} rows")

    # ---- 4. verify the other symbols are untouched ------------------------------------------
    check_before = pl.read_parquet(backup).filter(~pl.col("symbol").is_in(SYMBOLS)).sort(
        ["symbol", "timestamp"]
    )
    check_after = pl.read_parquet(h1_path).filter(~pl.col("symbol").is_in(SYMBOLS)).sort(
        ["symbol", "timestamp"]
    )
    identical = check_before.equals(check_after)
    print(
        f"[verify] non-metal rows: {check_before.height:,} before, {check_after.height:,} after, "
        f"exact frame equality = {identical}"
    )
    if not identical:
        raise SystemExit("ABORT: non-metal rows changed; restore from the backup")

    # ---- 5. history_depth.json --------------------------------------------------------------
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    for bare in SYMBOLS:
        rec = dict(depth_new["1h"][bare])
        rec["dense_history_start"] = report[bare]["dense_history_start"]
        rec["note"] = (
            "count-based copy_rates_from (mt5_loader.fetch_mt5_bars_deep, history_mode='deep'), "
            f"read {datetime.now(UTC).date()}; the other symbols and timeframes in this file were "
            "read with copy_rates_range in calendar chunks and are NOT comparable depths"
        )
        depth["timeframes"]["1h"][bare] = rec
    depth["history_mode_note"] = (
        "Entries without a 'history_mode' key were read with copy_rates_range (chunked), which "
        "returns only the bars the terminal had already synchronised. XAUUSD/XAGUSD H1 carry "
        "history_mode='deep' (copy_rates_from by bar count) and reach further back as a result."
    )
    depth_path.write_text(json.dumps(depth, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[depth] {depth_path} updated for {SYMBOLS}")

    out = Path(__file__).resolve().parent / f"deepen_metals_h1_{STAMP}.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
