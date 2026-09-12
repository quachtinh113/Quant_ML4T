"""Phase-5 registry report for exness_gold_sess: K, the null, the distribution, the diagnostics.

Not a stage, and deliberately not a selection surface. A **read-only** report over whatever
`13_backtest` has registered, forked from `case_studies/exness_fx_d1/_report_phase5.py` and changed
in exactly the six places `bots/exness_gold_sess/PHASE567_DECLARATION.md` D5.5 lists:

===========================================  ==========================================================
change                                       why
===========================================  ==========================================================
``LABELS`` are this bot's three               `fwd_ret_8h`, `fwd_ret_24h`, `dir_tb_8h`
benchmark folds H1 -> D1 before returns      the registered grid is hourly (D5.3); `pct_change` on
                                             the H1 frame gives a series 24x too dense and every
                                             active return after it is rubbish
``_spec_fields`` -> ``method/book/holdN``    `top_k` does not exist on this bot, and the book and
                                             the hold **must** appear in the grouping key
turnover: ledger + notional, never weights   a `TimeExit` exit never passes through the weight
                                             series, so `avg_turnover` and any reconstruction from
                                             `weights.parquet` under-report by about half (D5.2)
the 838 pooled series join ``member_stats``  K of the record is 2,514, not 1,676 (D5.3 / D5.4)
``ONE_LEGGED_THRESHOLD`` 0.5, renamed        on a rising gold sample the number to publish is the
                                             **exposure gate**: a book that is directional is
                                             reported as directional, never as dollar-neutral
===========================================  ==========================================================

**The import ban is the mechanism, not a promise.** D5.6 requires that no `CandidateSet`, no
`OfficialPopulation` and no `run_backtests` be reachable from this file, so that nothing here can
create a candidate, publish a population or run a backtest. The ban is asserted against this
module's own source at import time by :func:`_assert_no_selection_imports`, so deleting the
sentence from the docstring does not delete the guard. The registry is opened through
``sqlite3.connect("file:...?mode=ro", uri=True)``: a write attempt raises rather than succeeding
quietly.

**Print order is fixed by D5.3 and is an anti-peek device.** (1) K, `member_digest`, membership
per label and book; (2) `variance_trials` and the `expected_max_sharpe` **under the null** for raw
and for active; (3) only then the Sharpe distribution and any spec name. A reader therefore sees
what 2,514 pure-noise strategies would deliver before seeing which spec led.

**It degrades honestly.** Run against a registry with zero `backtest_runs` rows it still computes
everything that does not depend on a backtest - the benchmark, the H1->D1 falsifier, the New York
7-hour probe on the sealed panel - and prints `NOT COMPUTABLE` with the reason for everything that
does, instead of printing a zero.

Run from the repository root in the research environment (WSL2)::

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess uv run python \
        case_studies/exness_gold_sess/_report_phase5.py

Conventions (D5.3): `PPY = 252`; `confidence_level = 0.95`, `is_significant = probability >= 0.95`;
`variance_trials` is the sample variance (ddof=1) of the annualised Sharpe over **all** members of
the record's cohort, on the same return definition as the statistic being deflated; both the
`16_strategy_simulation/12_dsr_validation.py` formula and `ml4t.diagnostic` are printed and neither
is preferred for being the prettier number. The **gate is the active return** over the 1/N long
book of the two metals.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
)
from case_studies.utils.uncertainty import cohort_member_digest
from utils.paths import get_case_study_dir

CS = "exness_gold_sess"
PPY = 252
CONFIDENCE_LEVEL = 0.95
EULER_MASCHERONI = 0.5772156649015329

#: D5.5 change 1. `dir_tb_8h` is listed even though its 13 registered linear sets are VOID (the
#: lookahead leak of 2026-09-08): the report says what the registry holds, and a label omitted
#: here would silently leave its trials out of K.
LABELS = ["fwd_ret_8h", "fwd_ret_24h", "dir_tb_8h"]

#: The two ENGINE books. The pooled book is not an engine run (`setup.yaml::backtest.sweep
#: .pooled_book`), so it never appears in this list; it is built arithmetically in
#: :func:`pooled_members`.
ENGINE_BOOKS = ["london", "ny"]

#: D5.5 change 6. Threshold unchanged at 0.5; what changed is the name it is reported under. A
#: spec whose share of invested sessions with |net| / gross above this is above it is a
#: **DIRECTIONAL** book and is reported as one. On a 2017-2025 gold sample with a pronounced
#: uptrend that distinction is the difference between an edge and a long position.
ONE_LEGGED_THRESHOLD = 0.5

#: Declared in advance (D5.1), printed before anything is measured, and never recomputed from
#: what happens to be in the registry: 419 prediction sets x 2 engine books x 2 signal specs =
#: 1,676 registered, plus 419 x 1 pooled book x 2 signal specs = 838 arithmetic, K = 2,514.
K_DECLARED = 2514
K_DECLARED_REGISTERED = 1676
K_DECLARED_POOLED = 838

#: RULINGS_2026-09-08.md, addendum sections 2, 4, 5 and 6, written before the sweep was restarted.
#: Exactly three declared identities cannot be run, and they are NAMED here rather than deducted
#: from K: prediction `e7979e003e48` (`dir_tb_8h` / linear / `logistic_l1_C0.001`) carries a
#: constant score of 0.0 on all 4,060 rows because its L1 penalty at C = 0.001 zeroes every
#: coefficient, and a constant score can never exceed its own rolling percentile. So the
#: `per_symbol_p80` member of each engine book has no target weight at any rebalance, and the
#: pooled pair of those two sleeves has nothing to sum. The same prediction set under
#: `fixed_threshold_0` DOES run - `signals.py:53-62` sets `lower_threshold = 1.0 - threshold`, so a
#: constant 0.0 is short on every row - and it is a full member with a Sharpe like any other.
#:
#: These three names enter `member_digest` as literal strings, by the same naming device D5.3
#: declared for the pooled members. They carry no return series, so they are absent from
#: `variance_trials` and from every Sharpe distribution below. K stays 2,514.
UNRUNNABLE_LABEL = "dir_tb_8h"
UNRUNNABLE_PREDICTION = "e7979e003e48"
UNRUNNABLE_METHOD = "per_symbol_rolling_percentile"
UNRUNNABLE_SPEC_NAME = "per_symbol_p80"
UNRUNNABLE_NAMES = (
    f"unrunnable:{UNRUNNABLE_PREDICTION}/london/{UNRUNNABLE_SPEC_NAME}",
    f"unrunnable:{UNRUNNABLE_PREDICTION}/ny/{UNRUNNABLE_SPEC_NAME}",
    f"unrunnable:pooled:{UNRUNNABLE_PREDICTION}/{UNRUNNABLE_SPEC_NAME}",
)
#: The one pooled pair D5.4 is allowed to be missing, named in advance so the assertion still
#: stops at 836 pairs and still stops if a DIFFERENT pair is the absent one.
PERMITTED_MISSING_POOLED_PAIR = (UNRUNNABLE_LABEL, UNRUNNABLE_PREDICTION, UNRUNNABLE_METHOD)
K_SCORABLE = K_DECLARED - len(UNRUNNABLE_NAMES)
K_POOLED_BUILDABLE = K_DECLARED_POOLED - 1
K_REGISTERED_RUNNABLE = K_DECLARED_REGISTERED - 2

#: The names of the objects this file may not import, asserted against its own source.
BANNED_NAMES = ("CandidateSet", "OfficialPopulation", "run_backtests")


def _assert_no_selection_imports() -> None:
    """D5.6: this report is diagnostic, so the selection machinery is not reachable from it.

    Checked two ways, because either alone is defeatable: the module's own source may not
    *import* the banned names, and they may not be bound in its namespace at run time.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for name in BANNED_NAMES:
            if name in stripped:
                raise RuntimeError(
                    f"_report_phase5.py imports {name!r} on line {line!r}. D5.6 forbids it: this "
                    "report is a diagnostic, not a selection surface."
                )
    bound = [name for name in BANNED_NAMES if name in globals()]
    if bound:
        raise RuntimeError(f"{bound} are bound in this module's namespace; D5.6 forbids it")


_assert_no_selection_imports()
print(
    f"import ban asserted: {', '.join(BANNED_NAMES)} are neither imported nor bound. This report "
    "writes 0 rows to the registry and creates no candidate, population or backtest."
)

case_dir = get_case_study_dir(CS)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text(encoding="utf-8"))
db = case_dir / "run_log" / "registry.db"
print(f"case dir : {case_dir}")
print(f"registry : {db} (exists={db.exists()})")
if not db.exists():
    raise SystemExit(
        "no registry.db under this ML4T_OUTPUT_DIR; nothing has been registered for this bot"
    )

# Read-only by URI, so a stray write raises `sqlite3.OperationalError: attempt to write a readonly
# database` instead of succeeding. `mode=ro` also refuses to create the file.
con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)


def q(sql: str) -> pl.DataFrame:
    return pl.read_database(sql, con)


def _table_names() -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


TABLES = _table_names()
counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in sorted(TABLES)}
print("\n== registry row counts ==")
for name, n in counts.items():
    if n:
        print(f"  {name:36s} {n:>8,}")
print(
    "  (tables with 0 rows: "
    + ", ".join(t for t, n in counts.items() if not n)
    + ")"
)

# ---------------------------------------------------------------- the evidence boundary, first
preds = q(
    "SELECT p.prediction_hash, t.label, t.family, t.config_name, p.checkpoint_kind, "
    "p.checkpoint_value, p.split FROM prediction_sets p "
    "JOIN training_runs t ON p.training_hash = t.training_hash"
)
holdout_rows = preds.filter(pl.col("split") == "holdout").height
print(
    f"\nprediction sets by split: {preds.group_by('split').len().to_dicts()}; "
    f"holdout rows: {holdout_rows}"
)
if holdout_rows:
    raise SystemExit("holdout prediction sets exist; the evidence boundary was crossed")
print(
    "0 prediction sets carry split == 'holdout': the holdout window "
    f"{setup['evaluation']['holdout_start']} .. {setup['evaluation']['holdout_end']} is UNBURNT."
)

runs_all = q("SELECT * FROM backtest_runs")
metrics_all = q("SELECT * FROM backtest_metrics")
HAVE_BACKTESTS = runs_all.height > 0
print(f"backtest_runs rows: {runs_all.height} | backtest_metrics rows: {metrics_all.height}")


# ---------------------------------------------------------------- spec fields (D5.5 change 3)
def _spec_fields(spec_json: str) -> dict:
    """D5.5 change 3: `spec_key = method/session_filter/holdN`.

    `top_k` does not exist on this bot - the signal is per-asset timing over two names, not a
    cross-sectional rank - so the fx_pairs key would collapse every member onto one string. The
    **book** and the **hold** are what distinguish two members that share a prediction set, so
    both are inside the key. `pair_key` is the same identity with the book removed: it is what
    the 838 pooled pairs of D5.4 are matched on.
    """
    spec = json.loads(spec_json)
    strat = spec.get("strategy", spec)
    signal = strat["signal"]
    risk = strat.get("risk") or {}
    rules = risk.get("position_rules") or []
    hold = int(rules[0]["bars"]) if rules and "bars" in rules[0] else -1
    bc = spec["backtest_config"]
    method = str(signal.get("method", ""))
    book = str(signal.get("session_filter", ""))
    # Everything of the signal that is NOT the book, so two sleeves of one pooled pair share it.
    params = {
        k: v
        for k, v in sorted(signal.items())
        if k not in ("session_filter",) and not isinstance(v, dict | list)
    }
    # The HOLD is not in `pair_key`, and that is a correction rather than a convenience. The hold
    # is derived per (label, BOOK) - NY_EXIT_DECLARATION.md section 1.2, london 8 bars and New
    # York 7 on the 8-hour labels - so it is part of the book's identity, not of the signal spec.
    # Leaving it in the pair key would give the two sleeves of ONE declared spec two different
    # keys on `dir_tb_8h` and `fwd_ret_8h`, no pair would ever match, and D5.4 - which pairs on
    # "(label, prediction_hash, signal spec name)" and nothing else - could never be executed.
    # `spec_key` keeps the hold, because there the book is present too.
    return {
        "method": method,
        "book": book,
        "hold_bars": hold,
        "spec_key": f"{method}/{book}/hold{hold}",
        "pair_key": f"{method}/" + json.dumps(params, sort_keys=True),
        "long_short": bool(signal.get("long_short", False)),
        "drop_friday": bool(signal.get("drop_friday", False)),
        "step": int(strat.get("rebalance", {}).get("step", 1)),
        "cadence": str(strat.get("rebalance", {}).get("cadence", "")),
        "costs": json.dumps(
            {
                "commission": bc["commission"]["model"],
                "commission_rate": bc["commission"]["rate"],
                "slippage": bc["slippage"]["model"],
                "slippage_rate": round(bc["slippage"]["rate"], 8),
            },
            sort_keys=True,
        ),
        "fill_timing": (
            f"{bc['execution']['execution_mode']}/{bc['execution']['execution_price']}/"
            f"mark={bc['execution']['mark_price']}/short={bc['account']['allow_short_selling']}/"
            f"{bc['position_sizing']['share_type']}/cal={bc['calendar']['calendar']}"
        ),
    }


if HAVE_BACKTESTS:
    fields = pl.DataFrame(
        [
            {"backtest_hash": h, **_spec_fields(s)}
            for h, s in zip(runs_all["backtest_hash"], runs_all["spec_json"], strict=True)
        ]
    )
    runs_all = runs_all.join(fields, on="backtest_hash", how="left").join(
        preds.drop("split"), on="prediction_hash", how="left"
    )

cfg = get_backtest_config(CS)
slip = cfg.slippage_bps / 1e4
print(
    f"\nengine costs: commission {cfg.commission_bps:.2f} bps, slippage {cfg.slippage_bps:.2f} bps "
    f"per crossing ({2 * cfg.slippage_bps:.2f} bps round trip), initial_cash {cfg.initial_cash:,.0f}"
)


# ---------------------------------------------------------------- benchmark (D5.5 change 2)
def daily_benchmark(label: str) -> tuple[pl.DataFrame, dict, dict]:
    """1/N long both metals, on returns taken **after** folding the H1 grid to one bar a day.

    D5.3, and this is the one place the fx_pairs original is wrong for this bot. Its grid is
    daily, so `pct_change().over("symbol")` on the price frame is a daily close-to-close return.
    Here the registered grid is the raw H1 tape keyed on the bar close, so the same call would
    produce an HOURLY return series about 24x too dense - a 24x understatement of the volatility
    a day carries, hence a Sharpe inflated by roughly sqrt(24), and an active return joined to
    the strategy's DAILY series on a date key would be joining two different things.

    So: last bar of each calendar date **per symbol** -> close-to-close by symbol -> mean over
    the two metals. The falsifier is printed beside it: the row count the wrong route would have
    produced, and their ratio.
    """
    px = load_backtest_prices_for(CS, label, split="validation")
    n_symbols = px["symbol"].n_unique()
    # Per SYMBOL, so the falsifier compares like with like: how many hourly returns ONE metal
    # would yield against how many daily ones it actually yields.
    wrong_n = (px.drop_nulls("close").height - n_symbols) // max(n_symbols, 1)
    eod = (
        px.sort(["symbol", "timestamp"])
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by(["symbol", "date"])
        .agg(pl.col("close").last().alias("close"), pl.col("timestamp").last().alias("bar_ts"))
        .sort(["symbol", "date"])
    )
    r = (
        eod.with_columns(pl.col("close").pct_change().over("symbol").alias("ret"))
        .drop_nulls("ret")
        .group_by("date")
        .agg(pl.col("ret").mean().alias("ew"), pl.len().alias("n_metals"))
        .sort("date")
    )
    ew = r["ew"].to_numpy()
    sd = float(np.std(ew, ddof=1))
    info = {
        "sharpe": round(float(np.mean(ew) / sd * np.sqrt(PPY)), 3) if sd > 0 else 0.0,
        "cagr": round(float(np.mean(ew) * PPY), 4),
        "vol": round(float(sd * np.sqrt(PPY)), 4),
        "n_periods": int(len(ew)),
        "window": [str(r["date"].min()), str(r["date"].max())],
    }
    falsifier = {
        "h1_returns_per_symbol_if_pct_change_on_H1": int(wrong_n),
        "folded_daily_returns_per_symbol": int(len(ew)),
        "density_ratio": round(wrong_n / max(len(ew), 1), 2),
    }
    return r.select("date", "ew"), info, falsifier


print("\n== benchmark: 1/N long both metals, H1 FOLDED TO D1 before returns (D5.3) ==")
print(
    "The three label rows are identical BY CONSTRUCTION and that is the check, not a bug: "
    "the registered price grid is the same H1 tape for every label - the label only decides "
    "which rows carry a prediction - so a 1/N long book of the two metals is ONE series. A "
    "label that moved this number would mean the grid had been filtered by the label, which "
    "is what PRICE_GRID_DECLARATION.md forbids."
)
bench: dict[str, dict] = {}
bench_series: dict[str, pl.DataFrame] = {}
for _label in LABELS:
    try:
        series, info, falsifier = daily_benchmark(_label)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  {_label}: NOT COMPUTABLE - {type(exc).__name__}: {exc}")
        continue
    bench_series[_label] = series
    bench[_label] = info
    print(f"  {_label}: {info}")
    print(
        "     falsifier: pct_change on the H1 frame would give "
        f"{falsifier['h1_returns_per_symbol_if_pct_change_on_H1']:,} returns per metal against "
        f"{falsifier['folded_daily_returns_per_symbol']:,} folded daily ones "
        f"(x{falsifier['density_ratio']}), so it would report a Sharpe roughly "
        f"sqrt({falsifier['density_ratio']}) = {np.sqrt(falsifier['density_ratio']):.1f}x too "
        "high and would not join to a daily strategy series at all"
    )


# ---------------------------------------------------------------- statistics helpers
def sharpe(x: np.ndarray) -> float:
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def _expected_max_sharpe(variance_trials: float, n_trials: int) -> float:
    """The Sharpe a cohort of `n_trials` pure-noise strategies is expected to hand its winner."""
    if n_trials <= 1 or variance_trials <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - (1.0 / n_trials))
    z2 = stats.norm.ppf(1.0 - (np.exp(-1.0) / n_trials))
    weight = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return float(np.sqrt(variance_trials) * weight)


def dsr_notebook(observed_sharpe, skewness, kurtosis, n_samples, n_trials, variance_trials):
    """`16_strategy_simulation/12_dsr_validation.py::deflated_sharpe_ratio`, annualized inputs."""
    annualizer = np.sqrt(PPY)
    observed_native = observed_sharpe / annualizer
    variance_native = max(variance_trials, 0.0) / PPY
    expected_max = _expected_max_sharpe(variance_native, n_trials)
    variance = 1.0 - skewness * observed_native + ((kurtosis - 1.0) / 4.0) * observed_native**2
    z = (
        (observed_native - expected_max)
        * np.sqrt(max(n_samples - 1, 1))
        / np.sqrt(max(variance, 1e-12))
    )
    p = float(stats.norm.cdf(z))
    return {
        "dsr": p,
        "z": float(z),
        "expected_max_sharpe": expected_max * annualizer,
        "is_significant": p >= CONFIDENCE_LEVEL,
    }


def psr_notebook(observed_sr, benchmark_sr, n_observations, skewness, kurtosis):
    annualizer = np.sqrt(PPY)
    o, b = observed_sr / annualizer, benchmark_sr / annualizer
    variance = 1.0 - skewness * o + ((kurtosis - 1.0) / 4.0) * o**2
    se = np.sqrt(max(variance, 1e-12) / max(n_observations - 1, 1))
    return float(stats.norm.cdf((o - b) / se))


def dsr_library(sr_ann, skew, exkurt, n, K, var_trials_ann):
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics

    lib = deflated_sharpe_ratio_from_statistics(
        observed_sharpe=sr_ann / np.sqrt(PPY),
        n_samples=n,
        n_trials=K,
        variance_trials=var_trials_ann / PPY if K > 1 else 0.0,
        skewness=skew,
        excess_kurtosis=exkurt,
        periods_per_year=PPY,
    )
    return {
        "dsr": float(lib.probability),
        "p_value": float(lib.p_value),
        "expected_max_sharpe": float(lib.expected_max_sharpe) * np.sqrt(PPY),
        "min_trl": float(lib.min_trl),
        "adequate_sample": bool(lib.has_adequate_sample),
        "is_significant": float(lib.probability) >= CONFIDENCE_LEVEL,
    }


def moments(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "sharpe": sharpe(x),
        "skew": float(stats.skew(x)),
        "exkurt": float(stats.kurtosis(x)),
    }


# ---------------------------------------------------------------- artifacts (D5.2)
BT_DIR = case_dir / "run_log" / "backtest"
REQUIRED_ARTIFACTS = (
    "daily_returns.parquet",
    "trades.parquet",
    "fills.parquet",
    "portfolio_state.parquet",
    "weights.parquet",
)


def artifact_census(hashes: list[str]) -> pl.DataFrame:
    """D5.2: every member must carry these five files. Missing ones are named, not summarised."""
    rows = []
    for h in hashes:
        d = BT_DIR / h
        rows.append(
            {"backtest_hash": h, **{a: (d / a).exists() for a in REQUIRED_ARTIFACTS}}
        )
    return pl.DataFrame(rows)


def read_returns(bh: str) -> pl.DataFrame:
    return pl.read_parquet(BT_DIR / bh / "daily_returns.parquet").with_columns(
        pl.col("timestamp").cast(pl.Date).alias("date")
    )


def book_shape(bh: str) -> dict:
    """The EXPOSURE GATE input (D5.5 change 6): how directional the engine actually held it."""
    path = BT_DIR / bh / "portfolio_state.parquet"
    if not path.exists():
        return {
            "one_legged_share": float("nan"),
            "gross_over_equity": float("nan"),
            "invested_sessions": 0,
        }
    ps = pl.read_parquet(path)
    invested = ps.filter(pl.col("gross_exposure") > 0)
    if invested.height == 0:
        return {
            "one_legged_share": float("nan"),
            "gross_over_equity": float("nan"),
            "invested_sessions": 0,
        }
    ratio = (invested["net_exposure"] / invested["gross_exposure"]).abs()
    return {
        "one_legged_share": float((ratio > ONE_LEGGED_THRESHOLD).mean()),
        "gross_over_equity": float((invested["gross_exposure"] / invested["equity"]).median()),
        "invested_sessions": int(invested.height),
    }


def turnover_from_ledger(row: dict) -> float:
    """D5.5 change 4, and the ONLY turnover this bot reads.

    `total_slippage / slippage_rate` is the notional the engine actually charged on, so it is
    exactly linear in what costs are computed from. The registry's `avg_turnover` is built from
    the weight series (`backtest_runner.py:1610-1620`), and a `TimeExit` exit never writes a
    weight row, so it misses about half the trading of this bot. It is printed beside this number
    under the label `UNDER-REPORTS - do not use`, never instead of it.
    """
    return float(
        (row["total_slippage"] / slip) / cfg.initial_cash / max(int(row["n_periods"]), 1)
    )


def notional_traded(bh: str) -> float:
    """D5.5 change 4: sum |qty x price| over `fills.parquet`, the second number of D5.2."""
    path = BT_DIR / bh / "fills.parquet"
    if not path.exists():
        return float("nan")
    f = pl.read_parquet(path)
    qty = next((c for c in ("quantity", "qty", "shares", "size", "units") if c in f.columns), None)
    px = next((c for c in ("price", "fill_price", "avg_price") if c in f.columns), None)
    if qty is None or px is None:
        return float("nan")
    return float((f[qty].abs() * f[px].abs()).sum())


def member_stats(bh: str, label: str) -> dict:
    ret = read_returns(bh)
    r = ret["daily_return"].to_numpy()
    raw = moments(r)
    if label in bench_series:
        joined = ret.join(bench_series[label], on="date", how="inner")
        act = moments((joined["daily_return"] - joined["ew"]).to_numpy())
    else:
        act = {"n": 0, "sharpe": float("nan"), "skew": float("nan"), "exkurt": float("nan")}
    return {
        "backtest_hash": bh,
        "member_name": bh,
        "kind": "registered",
        **book_shape(bh),
        "n": raw["n"],
        "sharpe_raw": raw["sharpe"],
        "skew_raw": raw["skew"],
        "exkurt_raw": raw["exkurt"],
        "n_active": act["n"],
        "sharpe_active": act["sharpe"],
        "skew_active": act["skew"],
        "exkurt_active": act["exkurt"],
    }


# ---------------------------------------------------------------- the pooled book (D5.4)
def pooled_members(runs: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    """The 838 pooled series: paired, asserted bijective, summed 0.5 / 0.5, outer-joined on date.

    D5.4, executed literally::

        pair on (label, prediction_hash, signal spec identity without the book) -> 419 x 2 = 838
        pooled(d) = 0.5 x r_london(d) + 0.5 x r_ny(d)
        join OUTER on `date`; a missing sleeve contributes 0.0 (that sleeve held CASH that day)

    Three assertions, not hopes: (a) the pairing is a **bijection** - every london member has
    exactly one ny partner on the same (label, prediction, spec) and vice versa, and an unmatched
    member stops the report; (b) the number of ONE-SLEEVE days is counted and printed, and on
    `fwd_ret_24h` it has to agree with the filtered Friday set; (c) a pooled member gets no
    `backtest_runs` row, no `OfficialPopulation` membership and no `cohort_metrics` row - it
    exists only as a numpy array inside this process, which is why it can never be a carrier.

    **Assertion (a), amended 2026-09-08** (RULINGS_2026-09-08.md addendum, section 6). The count
    it expects is **837**, not 838, and the single pair permitted to be absent is NAMED IN ADVANCE
    in :data:`PERMITTED_MISSING_POOLED_PAIR`. This is not a threshold moved after seeing a result:
    it fires on a **structural** event - which (label, prediction, spec) triples have two sleeves -
    it names the exception before the report runs, and it therefore still stops at 836 pairs, and
    still stops if the absent pair is a different one. The absent pair is counted in K under its
    literal `unrunnable:` name; it is not deducted.
    """
    if runs.is_empty():
        return pl.DataFrame(), []
    key = ["label", "prediction_hash", "pair_key", "method"]
    lon = runs.filter(pl.col("book") == "london").select([*key, "backtest_hash"])
    ny = runs.filter(pl.col("book") == "ny").select([*key, "backtest_hash"])
    pairs = lon.join(ny, on=key, how="full", suffix="_ny", coalesce=True)
    unmatched = pairs.filter(
        pl.col("backtest_hash").is_null() | pl.col("backtest_hash_ny").is_null()
    )
    if unmatched.height:
        raise SystemExit(
            f"the pooled pairing is NOT a bijection: {unmatched.height} of {pairs.height} pairs "
            f"have only one sleeve (first: {unmatched.head(3).to_dicts()}). D5.4 stops here."
        )
    if lon.height != ny.height or pairs.height != lon.height:
        raise SystemExit(
            f"pooled pairing sizes disagree: london {lon.height}, ny {ny.height}, "
            f"pairs {pairs.height}"
        )
    # The declared set of pairs: every (label, prediction) the registry knows, crossed with every
    # signal spec `setup.yaml` declares. Built from the DECLARATION and the prediction identities,
    # never from what happens to have run, so a member that vanished is a missing pair here rather
    # than a smaller denominator.
    declared_methods = sorted(
        {str(spec["method"]) for spec in setup["backtest"]["sweep"]["signal_specs"]}
    )
    prediction_ids = {
        (row["label"], row["prediction_hash"])
        for row in runs.select("label", "prediction_hash").unique().iter_rows(named=True)
    }
    declared_triples = {
        (label, prediction_hash, method)
        for (label, prediction_hash) in prediction_ids
        for method in declared_methods
    }
    observed_triples = {
        (row["label"], row["prediction_hash"], row["method"])
        for row in pairs.select("label", "prediction_hash", "method").iter_rows(named=True)
    }
    missing_triples = sorted(declared_triples - observed_triples)
    if len(declared_triples) != K_DECLARED_POOLED:
        raise SystemExit(
            f"the declared pooled cross product is {len(declared_triples)} pairs, not the "
            f"{K_DECLARED_POOLED} of D5.1 ({len(prediction_ids)} prediction identities x "
            f"{len(declared_methods)} signal specs). D5.4 stops here."
        )
    if pairs.height != K_POOLED_BUILDABLE:
        raise SystemExit(
            f"the pooled pairing built {pairs.height} pairs, not the {K_POOLED_BUILDABLE} "
            f"expected ({K_DECLARED_POOLED} declared minus the one pair named in advance as "
            f"unbuildable, {PERMITTED_MISSING_POOLED_PAIR}). Missing: {missing_triples[:5]}. "
            "D5.4 stops here."
        )
    if missing_triples != [PERMITTED_MISSING_POOLED_PAIR]:
        raise SystemExit(
            f"the absent pooled pair is {missing_triples}, and the only pair permitted to be "
            f"absent is {PERMITTED_MISSING_POOLED_PAIR} (RULINGS_2026-09-08.md addendum, section "
            "6). A different absence is a different fact about the search and is not covered by "
            "that declaration. D5.4 stops here."
        )
    print(
        f"D5.4 pairing: {pairs.height} pooled pairs built of {len(declared_triples)} declared; "
        f"the one absent pair is {PERMITTED_MISSING_POOLED_PAIR}, named in advance, counted in K "
        f"as {UNRUNNABLE_NAMES[2]!r} and NOT deducted."
    )
    rows, detail = [], []
    for p in pairs.iter_rows(named=True):
        lh, nh, label = p["backtest_hash"], p["backtest_hash_ny"], p["label"]
        rl = read_returns(lh).select("date", pl.col("daily_return").alias("l"))
        rn = read_returns(nh).select("date", pl.col("daily_return").alias("n"))
        j = (
            rl.join(rn, on="date", how="full", coalesce=True)
            .with_columns(
                pl.col("l").fill_null(0.0), pl.col("n").fill_null(0.0),
                pl.col("l").is_null().alias("_l_missing"),
                pl.col("n").is_null().alias("_n_missing"),
            )
            .sort("date")
        )
        one_sleeve = int(
            rl.join(rn, on="date", how="full", coalesce=True)
            .select(
                (pl.col("l").is_null() | pl.col("n").is_null()).sum()
            )
            .item()
        )
        pooled = (0.5 * j["l"] + 0.5 * j["n"]).to_numpy()
        raw = moments(pooled)
        if label in bench_series:
            jb = (
                pl.DataFrame({"date": j["date"], "pooled": pooled})
                .join(bench_series[label], on="date", how="inner")
            )
            act = moments((jb["pooled"] - jb["ew"]).to_numpy())
        else:
            act = {"n": 0, "sharpe": float("nan"), "skew": float("nan"), "exkurt": float("nan")}
        name = f"pooled:{lh}+{nh}"
        rows.append(
            {
                "backtest_hash": None,
                "member_name": name,
                "kind": "pooled",
                "one_legged_share": float("nan"),
                "gross_over_equity": float("nan"),
                "invested_sessions": 0,
                "n": raw["n"],
                "sharpe_raw": raw["sharpe"],
                "skew_raw": raw["skew"],
                "exkurt_raw": raw["exkurt"],
                "n_active": act["n"],
                "sharpe_active": act["sharpe"],
                "skew_active": act["skew"],
                "exkurt_active": act["exkurt"],
            }
        )
        detail.append(
            {
                "member_name": name,
                "label": label,
                "pair_key": p["pair_key"],
                "london": lh,
                "ny": nh,
                "days": raw["n"],
                "one_sleeve_days": one_sleeve,
            }
        )
    return pl.DataFrame(rows), detail


# ================================================================ (1) K, digest, membership
print("\n" + "=" * 100)
print("D5.3 PRINT ORDER, STEP 1 OF 3: the trial count and the cohort it names")
print("=" * 100)
print(
    f"K DECLARED IN ADVANCE = {K_DECLARED:,} = {K_DECLARED_REGISTERED:,} registered engine "
    f"backtests + {K_DECLARED_POOLED:,} arithmetic pooled series "
    "(PHASE567_DECLARATION.md D5.1; 419 prediction sets x 2 books x 2 signal specs, plus "
    "419 x 1 pooled book x 2 signal specs)."
)
print(
    f"Of those {K_DECLARED:,} declared members, {K_SCORABLE:,} can be scored "
    f"({K_REGISTERED_RUNNABLE:,} engine + {K_POOLED_BUILDABLE:,} pooled) and "
    f"{len(UNRUNNABLE_NAMES)} are named and never deleted: a constant score cannot exceed its own "
    "rolling percentile, so prediction "
    f"{UNRUNNABLE_PREDICTION} x {UNRUNNABLE_SPEC_NAME} produces no position in either book and "
    "its pooled pair has nothing to sum (RULINGS_2026-09-08.md addendum, sections 2 and 4). K "
    "counts the search that was declared, not the members that happened to complete, so K DOES "
    "NOT MOVE."
)

member_stats_df = pl.DataFrame()
pooled_detail: list[dict] = []
if HAVE_BACKTESTS:
    census = artifact_census(runs_all["backtest_hash"].to_list())
    missing = census.filter(
        ~pl.all_horizontal([pl.col(a) for a in REQUIRED_ARTIFACTS])
    )
    print(
        f"artifact census over {census.height} members: "
        + ", ".join(f"{a} {int(census[a].sum())}" for a in REQUIRED_ARTIFACTS)
    )
    if missing.height:
        print(f"  MEMBERS MISSING AN ARTIFACT ({missing.height}): {missing.head(10).to_dicts()}")
    reg_rows = [
        member_stats(r["backtest_hash"], r["label"])
        for r in runs_all.select("backtest_hash", "label").iter_rows(named=True)
    ]
    pooled_df, pooled_detail = pooled_members(runs_all)
    member_stats_df = pl.concat(
        [pl.DataFrame(reg_rows), pooled_df], how="vertical_relaxed"
    )
    # The cohort NAMES are the declared search, and the three identities that cannot produce a
    # position are members of it under literal `unrunnable:` names - the same naming device D5.3
    # used for the pooled members (RULINGS_2026-09-08.md addendum, section 5). They carry no
    # return series, so they never reach `member_stats_df`, `variance_trials` or any Sharpe
    # distribution; they are in the digest and in K because they were searched.
    scored_names = member_stats_df["member_name"].to_list()
    names = [*scored_names, *UNRUNNABLE_NAMES]
    if len(set(names)) != len(names):
        raise SystemExit("a cohort member name is duplicated; the digest would collapse it")
    digest = cohort_member_digest(names)
    n_reg = int((member_stats_df["kind"] == "registered").sum())
    n_pool = int((member_stats_df["kind"] == "pooled").sum())
    print(
        f"K = {K_DECLARED:,} DECLARED and it does not move (D5.1, held to it by the addendum of "
        f"2026-09-08): {len(names):,} names = {n_reg:,} registered + {n_pool:,} pooled + "
        f"{len(UNRUNNABLE_NAMES)} unrunnable-but-searched; member_digest = {digest}"
    )
    for unrunnable_name in UNRUNNABLE_NAMES:
        print(f"    cohort member with no return series, counted in K: {unrunnable_name}")
    if len(names) != K_DECLARED:
        print(
            f"  NOTE: the cohort names count {len(names):,}, not the declared {K_DECLARED:,}. K "
            "is NOT lowered to fit: the declared budget stands and the difference has to be "
            "explained in BOT.md before any DSR is quoted."
        )
    if n_reg != K_REGISTERED_RUNNABLE or n_pool != K_POOLED_BUILDABLE:
        print(
            f"  NOTE: {n_reg:,} registered and {n_pool:,} pooled members carry a return series, "
            f"against the {K_REGISTERED_RUNNABLE:,} and {K_POOLED_BUILDABLE:,} the addendum "
            "declared. The difference has to be explained in BOT.md before any DSR is quoted."
        )
    print("\nmembership per label and per book (registered members only):")
    print(runs_all.group_by("label", "book").len().sort("label", "book"))
    one_sleeve_by_label = (
        pl.DataFrame(pooled_detail)
        .group_by("label")
        .agg(
            pl.len().alias("pairs"),
            pl.col("one_sleeve_days").sum().alias("one_sleeve_days_total"),
            pl.col("one_sleeve_days").max().alias("one_sleeve_days_max"),
        )
        .sort("label")
    )
    print("\npooled pairs and ONE-SLEEVE days per label (D5.4 assertion b):")
    print(one_sleeve_by_label)
else:
    digest = None
    print(
        "K MEASURED = 0 registered + 0 pooled. `member_digest` is NOT COMPUTABLE: a digest is "
        "taken over member NAMES, and no backtest identity exists yet. Every Deflated Sharpe "
        "Ratio in this report is therefore NOT COMPUTABLE as well - a DSR without its cohort "
        "digest states no cohort and is not a result (D5.3)."
    )
    print(
        "membership per label and per book: NOT COMPUTABLE - `backtest_runs` is empty. The "
        "prediction sets that would feed it are: "
        + str(preds.group_by("label").len().sort("label").to_dicts())
    )

# ================================================================ (2) the null, before any name
print("\n" + "=" * 100)
print("D5.3 PRINT ORDER, STEP 2 OF 3: what NOISE alone delivers at this trial count")
print("=" * 100)
var_raw = var_active = float("nan")
if member_stats_df.height > 1:
    var_raw = float(np.var(member_stats_df["sharpe_raw"].to_numpy(), ddof=1))
    active_arr = member_stats_df["sharpe_active"].to_numpy()
    active_arr = active_arr[np.isfinite(active_arr)]
    var_active = float(np.var(active_arr, ddof=1)) if len(active_arr) > 1 else float("nan")
    K_USE = K_DECLARED
    print(
        f"variance_trials over the {member_stats_df.height:,} members that HAVE a return series "
        f"(ddof=1, annualised Sharpe): raw {var_raw:.4f}, active {var_active:.4f}"
    )
    # RULINGS_2026-09-08.md addendum, section 5: the two numbers are different and both are
    # printed. The sentence below is the one the addendum requires verbatim, in its own words and
    # with its own thousands separator, so a reader can match it against the ruling character for
    # character.
    n_series = f"{member_stats_df.height:,}".replace(",", ".")
    k_formula = f"{K_DECLARED:,}".replace(",", ".")
    print(
        f'  "variance_trials trên {n_series} thành viên có chuỗi; '
        f"K trong công thức DSR = {k_formula}; ba identity không sinh vị "
        'thế và không đóng góp một Sharpe nào"'
    )
    print(
        f"  (variance_trials is computed on {member_stats_df.height:,} members with a return "
        f"series; the K the DSR formula divides by is {K_DECLARED:,}; the "
        f"{len(UNRUNNABLE_NAMES)} named identities produce no position and contribute no Sharpe.)"
    )
    for kind, v in (("raw", var_raw), ("active", var_active)):
        if np.isfinite(v):
            print(
                f"  expected_max_sharpe under the NULL at K={K_USE:,} on {kind:6s}: "
                f"{_expected_max_sharpe(v / PPY, K_USE) * np.sqrt(PPY):.3f} - a cohort of "
                f"{K_USE:,} strategies with NO edge is expected to hand its winner this "
                "annualised Sharpe. Any observed Sharpe must be read against it first."
            )
else:
    print(
        "variance_trials: NOT COMPUTABLE - it is the sample variance of the annualised Sharpe "
        "ACROSS members, and there are fewer than two members."
    )
    print(
        "expected_max_sharpe under the null: NOT COMPUTABLE for the same reason. It is the number "
        "that has to be printed BEFORE any spec name, so nothing below may be read as a result."
    )

# ================================================================ (3) the distribution and names
print("\n" + "=" * 100)
print("D5.3 PRINT ORDER, STEP 3 OF 3: the Sharpe distribution and the leading specs")
print("=" * 100)


def dist(s: pl.Series) -> dict:
    s = s.drop_nulls()
    if s.len() == 0:
        return {"n": 0}
    return {
        "n": s.len(),
        "min": round(float(s.min()), 3),
        "median": round(float(s.median()), 3),
        "p90": round(float(s.quantile(0.9)), 3),
        "max": round(float(s.max()), 3),
        "share_positive": round(float((s > 0).mean()), 3),
    }


if HAVE_BACKTESTS:
    joined = (
        runs_all.join(metrics_all, on="backtest_hash", how="left")
        .join(
            member_stats_df.filter(pl.col("kind") == "registered").drop("member_name", "kind"),
            on="backtest_hash",
            how="left",
        )
    )
    print("registered raw Sharpe, all:", dist(joined["sharpe_raw"]))
    print("registered active Sharpe, all:", dist(joined["sharpe_active"]))
    for keys, sub in sorted(joined.group_by("label", "spec_key"), key=lambda kv: kv[0]):
        print(f"  {keys}: raw {dist(sub['sharpe_raw'])} | active {dist(sub['sharpe_active'])}")
    pooled_only = member_stats_df.filter(pl.col("kind") == "pooled")
    print("pooled raw Sharpe:", dist(pooled_only["sharpe_raw"]))
    print("pooled active Sharpe:", dist(pooled_only["sharpe_active"]))

    directional = joined.filter(pl.col("one_legged_share") > ONE_LEGGED_THRESHOLD)
    print(
        f"\nEXPOSURE GATE (D5.5 change 6) at |net|/gross > {ONE_LEGGED_THRESHOLD} on more than "
        f"half the invested sessions: {directional.height} of {joined.height} registered specs "
        "are DIRECTIONAL books and are reported as such, never as dollar-neutral. On a sample "
        "whose underlying rose over the window, a directional book's raw Sharpe is mostly the "
        "underlying."
    )

    print("\n== top 5 by ACTIVE Sharpe (the gate statistic) ==")
    top = joined.sort("sharpe_active", descending=True, nulls_last=True).head(5)
    rows = []
    for row in top.iter_rows(named=True):
        # The DSR of each leading spec, both formulas, both return definitions, at K = 2,514 and
        # the cohort's own `variance_trials` (D5.3) - printed beside the Sharpe so no leader is read
        # without its deflation. `None` where the cohort variance is undefined.
        dsr_cols: dict[str, float | None] = {
            "dsr_active_notebook": None,
            "dsr_active_library": None,
            "dsr_raw_notebook": None,
            "dsr_raw_library": None,
        }
        if np.isfinite(var_active) and row["n_active"] > 2 and np.isfinite(row["sharpe_active"]):
            dsr_cols["dsr_active_notebook"] = round(
                dsr_notebook(
                    row["sharpe_active"], row["skew_active"], row["exkurt_active"] + 3.0,
                    row["n_active"], K_DECLARED, var_active,
                )["dsr"], 4,
            )
            dsr_cols["dsr_active_library"] = round(
                dsr_library(
                    row["sharpe_active"], row["skew_active"], row["exkurt_active"],
                    row["n_active"], K_DECLARED, var_active,
                )["dsr"], 4,
            )
        if np.isfinite(var_raw) and row["n"] > 2 and np.isfinite(row["sharpe_raw"]):
            dsr_cols["dsr_raw_notebook"] = round(
                dsr_notebook(
                    row["sharpe_raw"], row["skew_raw"], row["exkurt_raw"] + 3.0,
                    row["n"], K_DECLARED, var_raw,
                )["dsr"], 4,
            )
            dsr_cols["dsr_raw_library"] = round(
                dsr_library(
                    row["sharpe_raw"], row["skew_raw"], row["exkurt_raw"],
                    row["n"], K_DECLARED, var_raw,
                )["dsr"], 4,
            )
        rows.append(
            {
                "backtest_hash": row["backtest_hash"],
                "label": row["label"],
                "family": row["family"],
                "preset": row["config_name"],
                "ckpt": row["checkpoint_value"],
                "spec_key": row["spec_key"],
                "sharpe_net_registry": round(row["sharpe"], 3) if row["sharpe"] is not None else None,
                "sharpe_raw": round(row["sharpe_raw"], 3),
                "sharpe_active": round(row["sharpe_active"], 3),
                **dsr_cols,
                "max_dd": round(row["max_drawdown"], 4) if row["max_drawdown"] is not None else None,
                "trades": int(row["num_trades"]) if row["num_trades"] is not None else None,
                "turnover_ledger": round(turnover_from_ledger(row), 4),
                "notional_traded": round(notional_traded(row["backtest_hash"]), 0),
                "avg_turnover_registry_UNDER_REPORTS_do_not_use": (
                    round(row["avg_turnover"], 4) if row["avg_turnover"] is not None else None
                ),
                "one_legged_share": round(row["one_legged_share"], 3),
            }
        )
    with pl.Config(tbl_cols=30, tbl_width_chars=280, tbl_rows=10):
        print(pl.DataFrame(rows))

    print(
        "\nNOTE ON TURNOVER (D5.2): `turnover_ledger` = (total_slippage / slippage_rate) / "
        "initial_cash / n_periods and `notional_traded` = sum |qty x price| over fills.parquet "
        "are the two numbers this bot reads. `avg_turnover` is printed above ONLY under the "
        "label UNDER-REPORTS - do not use: it is computed from the weight series and a TimeExit "
        "exit never passes through one."
    )

    print("\n== the gate: DSR on the ACTIVE return at K = 2,514, over the whole cohort ==")
    if np.isfinite(var_active):
        # Both formulas, on the same inputs and the same K (D5.3: print both, prefer neither).
        # `dsr_notebook` is the `16_strategy_simulation/12_dsr_validation.py` formula and takes the
        # raw kurtosis; `dsr_library` is `ml4t.diagnostic` and takes the EXCESS kurtosis. Neither
        # count replaces the other; the gate sentence quotes both with their label.
        n_sig_active = n_sig_raw = 0
        n_sig_active_lib = n_sig_raw_lib = 0
        for st in member_stats_df.iter_rows(named=True):
            if st["n_active"] > 2 and np.isfinite(st["sharpe_active"]):
                n_sig_active += dsr_notebook(
                    st["sharpe_active"], st["skew_active"], st["exkurt_active"] + 3.0,
                    st["n_active"], K_DECLARED, var_active,
                )["is_significant"]
                n_sig_active_lib += dsr_library(
                    st["sharpe_active"], st["skew_active"], st["exkurt_active"],
                    st["n_active"], K_DECLARED, var_active,
                )["is_significant"]
            if st["n"] > 2 and np.isfinite(st["sharpe_raw"]):
                n_sig_raw += dsr_notebook(
                    st["sharpe_raw"], st["skew_raw"], st["exkurt_raw"] + 3.0,
                    st["n"], K_DECLARED, var_raw,
                )["is_significant"]
                n_sig_raw_lib += dsr_library(
                    st["sharpe_raw"], st["skew_raw"], st["exkurt_raw"],
                    st["n"], K_DECLARED, var_raw,
                )["is_significant"]
        print(
            f"at K = {K_DECLARED:,} (member_digest {digest}): {n_sig_active} of {K_DECLARED:,} "
            f"specs have DSR >= {CONFIDENCE_LEVEL} on the ACTIVE return; {n_sig_raw} on the raw "
            "return [dsr_notebook - the 12_dsr_validation formula]. THE GATE IS THE ACTIVE "
            "RETURN. The Deflated Sharpe Ratio of every member was "
            f"computed at K = {K_DECLARED:,}; the {len(UNRUNNABLE_NAMES)} named unrunnable "
            "members produce no return series, so they can never be among the significant - "
            f"which is why the count above is out of {K_DECLARED:,} and the scan ran over the "
            f"{member_stats_df.height:,} members that have one."
        )
        print(
            f"at K = {K_DECLARED:,} (member_digest {digest}): {n_sig_active_lib} of "
            f"{K_DECLARED:,} specs have DSR >= {CONFIDENCE_LEVEL} on the ACTIVE return; "
            f"{n_sig_raw_lib} on the raw return [dsr_library - "
            "ml4t.diagnostic.deflated_sharpe_ratio_from_statistics, same inputs, same K, same "
            "variance_trials]. Both formulas are reported and neither is preferred (D5.3)."
        )
    else:
        print("NOT COMPUTABLE: variance_trials is undefined.")
else:
    print(
        "NOT COMPUTABLE - `backtest_runs` is empty, so there is no Sharpe distribution, no "
        "leading spec, no exposure gate, no turnover and no Deflated Sharpe Ratio. This is the "
        "honest state of the record today, not a zero."
    )
    print(
        f"What the sweep will produce when it is authorised: {K_DECLARED_REGISTERED:,} registered "
        f"members and {K_DECLARED_POOLED:,} pooled series, at a measured median of 2.68 s per "
        f"engine run = about {K_DECLARED_REGISTERED * 2.68 / 60:.0f} minutes of engine time."
    )


# ================================================================ D5.6 three-level diagnostic
print("\n" + "=" * 100)
print("D5.6 SESSION DIAGNOSTIC, THREE LEVELS - READ ONLY")
print("=" * 100)
print(
    "This table is a DIAGNOSTIC, NOT A SELECTION SURFACE. No model, no spec, no book and no "
    "metal may be chosen or dropped because of a cell in it. It is computed in "
    "_report_phase5.py, it writes 0 rows to the registry, and no CandidateSet or "
    "OfficialPopulation is built from it - the import ban asserted at the top of this file is "
    "what enforces that."
)


def level_1(hashes_by_book: dict[str, str], label: str) -> pl.DataFrame:
    """Level 1 - the book: london / ny / pooled, side by side on one row set."""
    rows = []
    series: dict[str, np.ndarray] = {}
    for book, bh in hashes_by_book.items():
        ret = read_returns(bh)
        series[book] = ret["daily_return"].to_numpy()
        m = metrics_all.filter(pl.col("backtest_hash") == bh)
        st = member_stats(bh, label)
        row = {
            "book": book,
            "n_days": int(ret.height),
            "sharpe_raw": round(st["sharpe_raw"], 3),
            "sharpe_active": round(st["sharpe_active"], 3),
            "max_dd": round(float(m["max_drawdown"][0]), 4) if m.height else None,
            "turnover_ledger": round(turnover_from_ledger(m.row(0, named=True)), 4)
            if m.height
            else None,
            "trades": int(m["num_trades"][0]) if m.height else None,
            "exposure_gate_share": round(st["one_legged_share"], 3),
        }
        rows.append(row)
    if len(series) == 2:
        lon = read_returns(hashes_by_book["london"]).select(
            "date", pl.col("daily_return").alias("l")
        )
        nyr = read_returns(hashes_by_book["ny"]).select("date", pl.col("daily_return").alias("n"))
        j = (
            lon.join(nyr, on="date", how="full", coalesce=True)
            .with_columns(pl.col("l").fill_null(0.0), pl.col("n").fill_null(0.0))
            .sort("date")
        )
        pooled = (0.5 * j["l"] + 0.5 * j["n"]).to_numpy()
        act = float("nan")
        if label in bench_series:
            jb = pl.DataFrame({"date": j["date"], "p": pooled}).join(
                bench_series[label], on="date", how="inner"
            )
            act = sharpe((jb["p"] - jb["ew"]).to_numpy())
        rows.append(
            {
                "book": "pooled (arithmetic)",
                "n_days": int(len(pooled)),
                "sharpe_raw": round(sharpe(pooled), 3),
                "sharpe_active": round(act, 3),
                "max_dd": None,
                "turnover_ledger": None,
                "trades": None,
                "exposure_gate_share": None,
            }
        )
    return pl.DataFrame(rows)


def level_2(bh: str) -> pl.DataFrame:
    """Level 2 - book x metal: gross and net P&L, trades, notional, hit rate, from trades.parquet."""
    path = BT_DIR / bh / "trades.parquet"
    if not path.exists():
        return pl.DataFrame()
    t = pl.read_parquet(path)
    pnl = next((c for c in ("pnl", "net_pnl", "profit", "realized_pnl") if c in t.columns), None)
    fee = next((c for c in ("commission", "fees", "cost") if c in t.columns), None)
    if pnl is None:
        return pl.DataFrame()
    aggs = [
        pl.len().alias("trades"),
        pl.col(pnl).sum().alias("pnl_net"),
        (pl.col(pnl) > 0).mean().alias("hit_rate"),
    ]
    if fee:
        aggs.append((pl.col(pnl) + pl.col(fee)).sum().alias("pnl_gross"))
    return t.group_by("symbol").agg(aggs).sort("symbol")


def level_3(bh: str, label: str) -> pl.DataFrame:
    """Level 3 - book x metal x validation year: hit rate, mean net per slot, Sharpe."""
    path = BT_DIR / bh / "trades.parquet"
    if not path.exists():
        return pl.DataFrame()
    t = pl.read_parquet(path)
    pnl = next((c for c in ("pnl", "net_pnl", "profit", "realized_pnl") if c in t.columns), None)
    tcol = next((c for c in ("entry_time", "entry_ts", "timestamp") if c in t.columns), None)
    if pnl is None or tcol is None:
        return pl.DataFrame()
    return (
        t.with_columns(pl.col(tcol).dt.year().alias("year"))
        .group_by("symbol", "year")
        .agg(
            pl.len().alias("trades"),
            (pl.col(pnl) > 0).mean().alias("hit_rate"),
            pl.col(pnl).mean().alias("mean_net_per_slot"),
            (pl.col(pnl).mean() / pl.col(pnl).std()).alias("per_trade_sharpe"),
        )
        .sort("symbol", "year")
    )


if HAVE_BACKTESTS and member_stats_df.height:
    top5 = (
        runs_all.join(
            member_stats_df.filter(pl.col("kind") == "registered").drop("member_name", "kind"),
            on="backtest_hash",
            how="left",
        )
        .sort("sharpe_active", descending=True, nulls_last=True)
        .head(5)
    )
    for row in top5.iter_rows(named=True):
        bh, label, book = row["backtest_hash"], row["label"], row["book"]
        print(f"\n-- {bh} {label} {row['spec_key']} --")
        partner = runs_all.filter(
            (pl.col("label") == label)
            & (pl.col("prediction_hash") == row["prediction_hash"])
            & (pl.col("pair_key") == row["pair_key"])
        )
        by_book = {
            r["book"]: r["backtest_hash"] for r in partner.iter_rows(named=True)
        }
        print("LEVEL 1 (book):")
        print(level_1(by_book, label))
        print("LEVEL 2 (book x metal):")
        print(level_2(bh))
        print("LEVEL 3 (book x metal x validation year):")
        print(level_3(bh, label))
else:
    print(
        "\nLEVEL 1 / 2 / 3: NOT COMPUTABLE - they read `daily_returns.parquet`, `trades.parquet` "
        "and `portfolio_state.parquet` of registered members, and `backtest_runs` is empty. "
        "Nothing is substituted for them."
    )


# ---------------------------------------------------------------- the New York 7-hour probe
print("\n" + "=" * 100)
print("D5.6 ADDITION, REQUIRED BY NY_EXIT_DECLARATION.md: the New York 7-hour probe")
print("=" * 100)
print(
    "The New York book trains on an 8-hour label and TRADES a 7-hour window (hold 7 bars, exit "
    "fill 60 minutes before the sealed `label_end_ts`, because the New York session close IS the "
    "start of the broker's daily break in both seasons). So 1,257 of the 2,514 trials will be "
    "RANKED by a statistic measured over a window they do not trade. The probe below is built "
    "from the SEALED panel - the close of the H1 bar keyed `label_end_ts - 60min`, exactly the "
    "bar the book exits at - so it is not a new label, it moves no digest, and it is REPORTED, "
    "never selected on."
)


def ny_probe() -> pl.DataFrame | None:
    from bots._shared.mt5_loader import load_mt5_bars
    from case_studies.exness_gold_sess._features import session_panel
    from case_studies.exness_gold_sess._hold import assert_no_holdout, development_window

    lo, hi = development_window(setup)
    bars = load_mt5_bars(
        "1h", symbols=sorted(setup["universe"]["symbols"]), start_date=lo, end_date=hi
    )
    if bars.schema["timestamp"].time_zone is not None:
        bars = bars.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    panel = session_panel(bars, keep_context=True, verbose=False)
    assert_no_holdout(setup, bars, panel)
    closes = bars.select(
        pl.col("symbol"),
        (pl.col("timestamp") + pl.duration(minutes=60)).alias("probe_end_ts"),
        pl.col("close").alias("probe_close"),
    )
    out = (
        panel.drop_nulls("label_end_ts")
        .with_columns((pl.col("label_end_ts") - pl.duration(minutes=60)).alias("probe_end_ts"))
        .join(closes, on=["symbol", "probe_end_ts"], how="inner")
        .with_columns(
            (pl.col("label_end_close") / pl.col("close") - 1).alias("fwd_ret_8h"),
            (pl.col("probe_close") / pl.col("close") - 1).alias("fwd_ret_7h_probe"),
        )
        .select("symbol", "session", "timestamp", "fwd_ret_8h", "fwd_ret_7h_probe")
    )
    return out


try:
    probe = ny_probe()
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    probe = None
    print(f"NOT COMPUTABLE: {type(exc).__name__}: {exc}")

if probe is not None:
    rows = []
    for (book, symbol), sub in sorted(probe.group_by("session", "symbol"), key=lambda kv: kv[0]):
        a = sub["fwd_ret_8h"].to_numpy()
        b = sub["fwd_ret_7h_probe"].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        rows.append(
            {
                "book": book,
                "symbol": symbol,
                "n": int(len(a)),
                "pearson": round(float(np.corrcoef(a, b)[0, 1]), 4),
                "spearman": round(float(stats.spearmanr(a, b).statistic), 4),
                "same_sign_share": round(float(np.mean(np.sign(a) == np.sign(b))), 4),
                "sd_8h_bps": round(float(np.std(a, ddof=1) * 1e4), 1),
                "sd_7h_bps": round(float(np.std(b, ddof=1) * 1e4), 1),
                "mean_abs_gap_bps": round(float(np.mean(np.abs(a - b)) * 1e4), 1),
            }
        )
    with pl.Config(tbl_cols=20, tbl_width_chars=200):
        print(pl.DataFrame(rows))
    print(
        "The LONDON rows are a CONTRAST, not a residual: London holds 8 bars and its exit fill "
        "lands exactly on the sealed endpoint (gap 0 minutes), so its 7-hour probe is a "
        "counterfactual and its lower correlation only says the 16:00-17:00 UTC hour it DOES "
        "trade carries real variance."
    )
    print(
        "Read: the NEW YORK rows are the ones that matter - a correlation well below 1 there "
        "means the statistic the sweep ranks New York on is measuring a different hour than the "
        "book trades. IC of a prediction set on the 7-hour probe requires a CARRIER, which "
        "requires a registered backtest; with `backtest_runs` empty it is NOT COMPUTABLE and is "
        "deferred rather than substituted."
    )

con.close()
print("\nregistry connection closed. Rows written by this report: 0.")
