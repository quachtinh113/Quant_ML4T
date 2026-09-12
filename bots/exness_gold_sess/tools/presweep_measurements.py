"""Two pre-sweep measurements that spend no trial and write no registry row.

``PHASE567_DECLARATION.md`` D6.3 and D6.4. Both read the **price path** and the **fill ledger** of
``run_backtest(..., register=False)`` runs; neither reads a Sharpe, a P&L or an information
coefficient, neither selects anything, and neither writes to ``registry.db``. The registry row
counts are printed before and after so that claim is checkable rather than asserted.

**D6.3 - the swap census that can unblock `16_costs` without the real account.**
Count the positions whose HOLDING WINDOW contains the server's 00:00 UTC rollover, per (label,
book). The server is UTC+0; London runs 09:00-17:00 UTC and New York 13:00/14:00 to 20:00/21:00
UTC, both inside one server day, and after ``NY_EXIT_DECLARATION.md`` the New York book holds 7
bars, so no 8-hour position is carried overnight. If the count is **0** for ``fwd_ret_8h`` and
``dir_tb_8h`` then the swap of those two labels is **zero by construction, independently of the
account**, and ``costs.swap.not_applicable_labels`` may be declared with this measured count and
its date. If it is not zero, nothing is declared and `16_costs` stays blocked on a real-account
read.

**D6.4 - the risk-arm activation pre-check.**
For each declared arm, the share of positions whose H1 price path between the entry fill and the
exit fill would TRIGGER it. Activation is a property of the price path, not of profit, so this is
not a peek - and D6.4 forbids the converse absolutely: an arm may never be kept or dropped for its
Sharpe. An arm that triggers on **0 %** of positions in **both** books is degenerate and is
removed from the sweep before it runs, with this measurement and its date recorded. That is the
same reason ``time_exit_10`` was deleted: a rule that cannot fire is still a trial the Deflated
Sharpe Ratio divides by.

Direction, stated because it changes the reading: no carrier exists (``backtest_runs`` is 0), so
the probe scores a constant +1 on both metals at every decision instant of the book. That opens a
position in **every** slot, which is a strict SUPERSET of any carrier's positions, and each path
is evaluated **both** as a long and as a short. An arm that fires on 0 % of this superset in both
directions cannot fire for any carrier.

Rule semantics are taken from ``ml4t.backtest.risk`` rather than invented:

* ``StopLoss(pct)`` long triggers when a bar's ``low <= entry_price * (1 - pct)``; short when a
  bar's ``high >= entry_price * (1 + pct)``.
* ``TrailingStop(pct)`` defaults to the LAGGED water mark: at bar *t* the level is the high water
  mark **through bar t-1**, so the bar that sets a new extreme cannot also be stopped out by it.
* ``TimeExit(bars)`` triggers whenever the position would otherwise be held longer than ``bars``.

Run (WSL2, from the repository root)::

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess uv run python \
        bots/exness_gold_sess/tools/presweep_measurements.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bots._shared.mt5_loader import load_mt5_bars  # noqa: E402
from case_studies.exness_gold_sess._features import session_of, session_panel  # noqa: E402
from case_studies.exness_gold_sess._hold import (  # noqa: E402
    assert_no_holdout,
    derive_backstop_bars,
    derive_hold_bars,
    development_window,
)
from case_studies.utils.backtest_loaders import (  # noqa: E402
    get_backtest_config,
    load_backtest_prices,
)
from case_studies.utils.backtest_presets import build_backtest_spec  # noqa: E402
from case_studies.utils.backtest_runner import run_backtest  # noqa: E402
from utils.cv_splits import generate_cv_splits  # noqa: E402
from utils.paths import get_case_study_dir  # noqa: E402

CASE_STUDY_ID = "exness_gold_sess"
BAR_MINUTES = 60

case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text(encoding="utf-8"))
DB = case_dir / "run_log" / "registry.db"

TRACKED_TABLES = (
    "backtest_runs",
    "backtest_metrics",
    "backtest_fold_metrics",
    "cohort_metrics",
    "candidate_sets",
    "candidate_set_members",
    "official_populations",
    "official_population_members",
    "prediction_sets",
    "training_runs",
)


def registry_counts(tag: str) -> dict[str, int]:
    if not DB.exists():
        print(f"[{tag}] no registry.db at {DB}")
        return {}
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    out = {
        t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in TRACKED_TABLES
        if t in have
    }
    con.close()
    print(f"[{tag}] registry rows: {out}")
    return out


# --------------------------------------------------------------------------- the hold, derived
def derive_holds() -> tuple[dict[tuple[str, str], int], list[str], pl.DataFrame, pl.DataFrame]:
    import pandas as pd

    lo, hi = development_window(setup)
    grid = load_backtest_prices(CASE_STUDY_ID, start_date=lo, end_date=hi)
    bars = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    )
    if bars.schema["timestamp"].time_zone is not None:
        bars = bars.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    panel = session_panel(bars, keep_context=True, verbose=False)
    assert_no_holdout(setup, grid, panel, bars)

    primary = setup["labels"]["primary"]
    timeline = (
        pl.read_parquet(case_dir / "labels" / f"{primary}.parquet")
        .select(pl.col("timestamp").dt.date().alias("timestamp"))
        .unique()
        .sort("timestamp")
    )
    folds = [
        {
            "fold": int(s["fold"]),
            **{
                n: str(pd.Timestamp(s[n]).date())
                for n in ("train_start", "train_end", "val_start", "val_end")
            },
        }
        for s in generate_cv_splits(
            timeline,
            case_study_id=CASE_STUDY_ID,
            label_buffer=setup["labels"]["buffer"],
            date_col="timestamp",
        )
    ]
    books = list(setup["backtest"]["sweep"]["session_books"])
    horizons = setup["labels"]["horizons"]
    hold: dict[tuple[str, str], int] = {}
    for label in sorted(horizons):
        bars_h = int(str(horizons[label]).rstrip("Hh"))
        if bars_h > 12:
            derived, _ = derive_backstop_bars(
                grid, panel, label=label, horizons=horizons, books=books, folds=folds,
                verbose=False,
            )
        else:
            derived, _ = derive_hold_bars(
                grid, panel, label=label, horizons=horizons, books=books, folds=folds,
                pre_fold_disagreements={"ny": 102}, verbose=False,
            )
        for book, n in derived.items():
            hold[label, book] = n
    val_lo = min(f["val_start"] for f in folds)
    val_hi = max(f["val_end"] for f in folds)
    print(f"derived HOLD_BARS: {hold}")
    print(f"validation window across the four folds: {val_lo} -> {val_hi}")
    return hold, books, grid, pl.DataFrame({"val_lo": [val_lo], "val_hi": [val_hi]})


# --------------------------------------------------------------------------- the probe runs
def probe_trades(
    label: str, book: str, hold_bars: int, prices: pl.DataFrame, val_lo: str, val_hi: str
) -> pl.DataFrame:
    """One ``register=False`` run: a constant +1 score in every slot of the book."""
    labels_path = case_dir / "labels" / f"{label}.parquet"
    frame = pl.read_parquet(labels_path).filter(
        (pl.col("timestamp") >= pl.lit(val_lo).str.to_datetime())
        & (pl.col("timestamp") <= pl.lit(val_hi).str.to_datetime() + pl.duration(days=1))
    )
    instants = frame["timestamp"].unique().sort()
    sessions = pl.DataFrame({"timestamp": instants, "session": session_of(instants)})
    predictions = (
        frame.join(sessions, on="timestamp", how="left")
        .filter(pl.col("session") == book)
        .select(
            "timestamp",
            "symbol",
            pl.lit(1.0).alias("y_score"),
            pl.col(label).cast(pl.Float64).alias("y_true"),
        )
        .drop_nulls("y_true")
    )
    signal = {
        "method": "fixed_threshold",
        "threshold": 0.0,
        "long_short": True,
        "session_filter": book,
    }
    if label == "fwd_ret_24h":
        signal["drop_friday"] = True
    risk = {"position_rules": [{"type": "time_exit", "bars": hold_bars}]}
    cfg = get_backtest_config(CASE_STUDY_ID)
    spec = build_backtest_spec(
        CASE_STUDY_ID,
        cfg,
        prices=prices,
        prediction_hash=f"presweep_{label}_{book}",
        initial_cash=cfg.initial_cash,
        signal=signal,
        risk=risk,
        chapter="16",
        label=label,
    )
    result = run_backtest(
        CASE_STUDY_ID,
        f"presweep_{label}_{book}",
        spec,
        prices=prices,
        predictions=predictions,
        label=label,
        register=False,
        initial_cash=cfg.initial_cash,
        calendar=cfg.calendar,
    )
    trades = result.engine_result.to_trades_dataframe().filter(pl.col("status") == "closed")
    return trades.with_columns(
        (pl.col("entry_time") - pl.duration(minutes=BAR_MINUTES)).alias("entry_fill_ts"),
        (pl.col("exit_time") - pl.duration(minutes=BAR_MINUTES)).alias("exit_fill_ts"),
        pl.lit(label).alias("label"),
        pl.lit(book).alias("book"),
    )


# --------------------------------------------------------------------------- D6.3 swap census
def midnight_crossings(trades: pl.DataFrame) -> dict:
    """Positions whose holding window (entry fill, exit fill] contains a 00:00 UTC rollover."""
    if trades.is_empty():
        return {"positions": 0, "crossing_00utc": 0, "share": float("nan"), "nights_total": 0}
    t = trades.with_columns(
        (
            pl.col("exit_fill_ts").dt.date().cast(pl.Int32)
            - pl.col("entry_fill_ts").dt.date().cast(pl.Int32)
        ).alias("date_steps")
    )
    crossing = int((t["date_steps"] > 0).sum())
    return {
        "positions": int(t.height),
        "crossing_00utc": crossing,
        "share": round(crossing / t.height, 6),
        "nights_total": int(t["date_steps"].sum()),
        "max_nights_one_position": int(t["date_steps"].max()),
    }


# --------------------------------------------------------------------------- D6.4 activation
def price_paths(prices: pl.DataFrame) -> dict[str, dict]:
    """Per symbol: the sorted grid keys and the O/H/L arrays, for slicing one hold at a time."""
    out: dict[str, dict] = {}
    for symbol, sub in prices.sort(["symbol", "timestamp"]).group_by("symbol"):
        s = symbol[0] if isinstance(symbol, tuple) else symbol
        out[s] = {
            "ts": sub["timestamp"].to_numpy().astype("datetime64[us]"),
            "open": sub["open"].to_numpy().astype(float),
            "high": sub["high"].to_numpy().astype(float),
            "low": sub["low"].to_numpy().astype(float),
        }
    return out


def arm_activation(trades: pl.DataFrame, paths: dict[str, dict], arms: list[dict]) -> pl.DataFrame:
    """Share of positions each arm would trigger on, per (arm, book, metal, direction).

    The evaluated path is the grid rows keyed ``[entry_time, exit_time)`` - the bars the position
    is live through - and the entry price is the OPEN of the entry row, which is what
    ``execution_price: open`` + ``execution_mode: next_bar`` fills at.
    """
    rows: list[dict] = []
    for symbol, sub in trades.group_by("symbol"):
        sym = symbol[0] if isinstance(symbol, tuple) else symbol
        if sym not in paths:
            continue
        p = paths[sym]
        entry_ts = sub["entry_time"].to_numpy().astype("datetime64[us]")
        exit_ts = sub["exit_time"].to_numpy().astype("datetime64[us]")
        i0 = np.searchsorted(p["ts"], entry_ts, side="left")
        i1 = np.searchsorted(p["ts"], exit_ts, side="left")
        n = len(i0)
        counters = {(a["name"], d): 0 for a in arms for d in ("long", "short")}
        bars_seen = np.zeros(n, dtype=int)
        for k in range(n):
            a, b = int(i0[k]), int(i1[k])
            if b <= a or a >= len(p["ts"]):
                continue
            hi = p["high"][a:b]
            lo = p["low"][a:b]
            entry_px = float(p["open"][a])
            bars_seen[k] = b - a
            if entry_px <= 0:
                continue
            run_max = np.maximum.accumulate(hi)
            run_min = np.minimum.accumulate(lo)
            # LAGGED water mark: at bar t the level uses the extreme THROUGH bar t-1, seeded at
            # the entry price. `ml4t.backtest.risk.TrailingStop`, default TrailStopTiming.
            hwm = np.concatenate(([entry_px], run_max[:-1]))
            lwm = np.concatenate(([entry_px], run_min[:-1]))
            for arm in arms:
                name, kind = arm["name"], arm["type"]
                if kind == "stop_loss":
                    pct = float(arm["threshold"])
                    if np.any(lo <= entry_px * (1.0 - pct)):
                        counters[name, "long"] += 1
                    if np.any(hi >= entry_px * (1.0 + pct)):
                        counters[name, "short"] += 1
                elif kind == "trailing_stop":
                    pct = float(arm["threshold"])
                    if np.any(lo <= hwm * (1.0 - pct)):
                        counters[name, "long"] += 1
                    if np.any(hi >= lwm * (1.0 + pct)):
                        counters[name, "short"] += 1
                elif kind == "time_exit":
                    if (b - a) > int(arm["bars"]):
                        counters[name, "long"] += 1
                        counters[name, "short"] += 1
        for (name, direction), hits in counters.items():
            rows.append(
                {
                    "arm": name,
                    "label": sub["label"][0],
                    "book": sub["book"][0],
                    "symbol": sym,
                    "direction": direction,
                    "positions": int(n),
                    "median_bars": int(np.median(bars_seen)) if n else 0,
                    "triggers": hits,
                    "activation_rate": round(hits / n, 6) if n else float("nan"),
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    before = registry_counts("BEFORE")
    hold, books, dev_grid, window = derive_holds()
    val_lo, val_hi = window["val_lo"][0], window["val_hi"][0]
    dev_lo, dev_hi = development_window(setup)
    prices = load_backtest_prices(CASE_STUDY_ID, start_date=val_lo, end_date=dev_hi)
    assert_no_holdout(setup, prices)
    print(
        f"price frame for the probes: {prices.height:,} rows, "
        f"{prices['timestamp'].min()} -> {prices['timestamp'].max()} "
        f"(holdout_start {setup['evaluation']['holdout_start']})"
    )
    paths = price_paths(prices)
    arms = list(setup["backtest"]["sweep"]["risk_controls"]["position"])
    print(f"declared risk arms: {[a['name'] for a in arms]}")

    all_trades = []
    swap_rows = []
    for label in sorted(setup["labels"]["horizons"]):
        for book in books:
            t = probe_trades(label, book, hold[label, book], prices, val_lo, val_hi)
            if t.is_empty():
                print(f"  {label} / {book}: NO CLOSED POSITION")
                continue
            all_trades.append(t)
            census = midnight_crossings(t)
            swap_rows.append({"label": label, "book": book, "hold_bars": hold[label, book], **census})
            print(f"  {label} / {book}: {t.height:,} closed positions, hold {hold[label, book]} bars")

    print("\n" + "=" * 96)
    print("D6.3 - SWAP CENSUS: positions whose holding window contains 00:00 UTC (server = UTC+0)")
    print("=" * 96)
    swap = pl.DataFrame(swap_rows)
    with pl.Config(tbl_cols=20, tbl_width_chars=200, tbl_rows=20):
        print(swap)
    eight_hour = swap.filter(pl.col("label").is_in(["fwd_ret_8h", "dir_tb_8h"]))
    total_8h = int(eight_hour["crossing_00utc"].sum()) if eight_hour.height else -1
    print(
        f"\n8-hour labels (`fwd_ret_8h`, `dir_tb_8h`): {total_8h} of "
        f"{int(eight_hour['positions'].sum())} positions cross 00:00 UTC."
    )
    if total_8h == 0:
        print(
            "=> ZERO. The swap of those two labels is 0 BY CONSTRUCTION, independently of the "
            "account, so `costs.swap.not_applicable_labels: [fwd_ret_8h, dir_tb_8h]` may be "
            "declared with this count and today's date, and `16_costs` is unblocked FOR THOSE "
            "TWO LABELS ONLY. `fwd_ret_24h` stays blocked on a real-account read."
        )
    else:
        print(
            "=> NOT ZERO. Nothing is declared, no gate is relaxed, and `16_costs` stays blocked "
            "on a read-only measurement of the real Pro account (D6.3)."
        )

    print("\n" + "=" * 96)
    print("D6.4 - RISK-ARM ACTIVATION PRE-CHECK (a property of the price path, not of P&L)")
    print("=" * 96)
    trades = pl.concat(all_trades, how="vertical_relaxed")
    act = pl.concat(
        [arm_activation(t, paths, arms) for t in all_trades], how="vertical_relaxed"
    )
    per_arm_book = (
        act.group_by("arm", "book", "symbol", "direction")
        .agg(
            pl.col("positions").sum().alias("positions"),
            pl.col("triggers").sum().alias("triggers"),
        )
        .with_columns((pl.col("triggers") / pl.col("positions")).alias("activation_rate"))
        .sort("arm", "book", "symbol", "direction")
    )
    with pl.Config(tbl_cols=20, tbl_width_chars=220, tbl_rows=200):
        print("\n-- full table: arm x book x metal x direction, all labels pooled --")
        print(per_arm_book)
        print("\n-- arm x book x metal x direction x label --")
        print(act.sort("arm", "label", "book", "symbol", "direction"))

    # `position_directions` = positions x 2, because every path is scored once as a long and once
    # as a short; the rate is therefore the mean of the long and the short activation rate.
    degenerate = (
        per_arm_book.group_by("arm")
        .agg(
            pl.col("triggers").sum().alias("triggers"),
            pl.col("positions").sum().alias("position_directions"),
        )
        .with_columns((pl.col("triggers") / pl.col("position_directions")).alias("activation_rate"))
        .sort("activation_rate")
    )
    print("\n-- verdict per arm (0 in BOTH books and BOTH directions => degenerate) --")
    with pl.Config(tbl_rows=20):
        print(degenerate)
    dead = degenerate.filter(pl.col("triggers") == 0)["arm"].to_list()
    print(
        f"\nDEGENERATE ARMS (activation 0 everywhere): {dead or 'none'}. D6.4: these are removed "
        "from the sweep BEFORE `15_risk_management` runs, on this measurement and its date, never "
        "on a Sharpe. Each surviving arm still needs a line in deploy/risk_config.yaml saying "
        "what it does live, or it is not swept."
    )
    print(
        f"\ntotal closed positions measured: {trades.height:,} over "
        f"{trades['label'].n_unique()} labels x {trades['book'].n_unique()} books"
    )
    after = registry_counts("AFTER")
    if before and after and before != after:
        raise SystemExit(f"THE REGISTRY MOVED: {before} -> {after}. register=False was not honoured")
    print("registry unchanged: register=False honoured, 0 rows written.")


if __name__ == "__main__":
    main()
