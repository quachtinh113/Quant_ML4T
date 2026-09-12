"""Phase-5 registry report for exness_fx_d1: populations, Sharpe distribution, benchmark, DSR.

Not a stage. A read-only report over the validation backtests `13_backtest` registered, one
section per generation of every baseline population the bot has published. It writes nothing into
the registry.

Two populations, because two signal families were run and a different ``method`` is a different
strategy rather than a later generation of the same one:

- ``exness_fx_d1:equal-weight-baselines`` - cross-sectional top-k. Generation 1 long-only
  (``b57aa4a93f72``), generation 2 long-short (``52365960d8ca``, the tip of that chain).
- ``exness_fx_d1:timeseries-percentile-baselines`` - the Chapter 16 per-pair time-series framing
  (`per_symbol_rolling_percentile`, long only), generation 1.

The trial count ``K`` accumulates over the **bot**, so it is the total membership of every
generation of every population, and every Deflated Sharpe Ratio below is computed at that total.

Run from the repository root in the research environment (WSL2), with the experiment selected::

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_fx_d1 uv run python \
        case_studies/exness_fx_d1/_report_phase5.py > experiments/exness_fx_d1/notebooks/phase5_report_gen2.txt

What it reports (the numbers `bots/exness_fx_d1/BOT.md` phase 5 cites):

- the registry populations and the members of every generation (generation 1 must stay
  queryable after generation 2 supersedes it); 0 holdout rows asserted;
- per cohort: counts per label / family / setting; the distribution of the validation net
  Sharpe (median, p90, max, share > 0);
- the equal-weight 1/N long benchmark of the five pairs on each label's validation window;
- the top 5 specs of the tip generation: net Sharpe (registry), gross Sharpe (net + the ledger's
  cost drag), CI95 (block bootstrap, registry), max drawdown, turnover from the ledger
  (``total_slippage / rate / initial_cash / n_periods``) and from the fixed
  ``target_weight_turnover`` on the scheduled targets of ``weights.parquet`` (never the
  ``avg_turnover`` column of generation-1 rows, which is wrong: BOT.md open question 16), trades;
- a sample weight frame of the tip generation (offsetting long and short weights, net exposure);
- for the best spec of each generation: PSR vs 0, PSR vs the benchmark, DSR at the cumulative
  trial count K (all generations together) with the formula of
  ``16_strategy_simulation/12_dsr_validation.py`` and with ``ml4t.diagnostic``, on (i) the raw
  return and (ii) the ACTIVE return over the 1/N long book; ``is_significant`` at 0.95;
- the count of tip-generation specs whose active-return DSR at K is >= 0.95 (the gate).

Conventions: annualization 252; DSR ``confidence_level`` 0.95 (``is_significant = probability >=
0.95``); the variance of trials is the sample variance of the annualized Sharpe across every
member of every generation, on the same return definition (raw or active) as the statistic.
"""

from __future__ import annotations

import json
import sqlite3
import sys

import numpy as np
import polars as pl
from scipy import stats

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    resolve_decision_schedule,
    target_weight_turnover,
)
from case_studies.utils.registry import load_backtest_metrics, load_backtest_runs
from utils.paths import get_case_study_dir

CS = "exness_fx_d1"
PPY = 252
CONFIDENCE_LEVEL = 0.95
EULER_MASCHERONI = 0.5772156649015329
BASELINE_POPULATIONS = [
    f"{CS}:equal-weight-baselines",
    f"{CS}:timeseries-percentile-baselines",
]
LABELS = ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d"]
# Declared before the time-series family ran (setup.yaml::backtest.sweep.timeseries_percentile_grid
# and 13_backtest, open question 23): a spec whose share of invested sessions with
# |net| / gross above this threshold exceeds it is a DIRECTIONAL book and is reported as one. The
# engine holds every book on a cash account, so a long-short target it cannot fund comes out
# one-legged, and reading such a spec as dollar-neutral would be reading a book that never existed.
ONE_LEGGED_THRESHOLD = 0.5

case_dir = get_case_study_dir(CS)
db = case_dir / "run_log" / "registry.db"
print("registry:", db)

# ---------------------------------------------------------------- populations and generations
con = sqlite3.connect(str(db))
pop_cols = [r[1] for r in con.execute("pragma table_info(official_populations)")]
pops = pl.read_database("SELECT * FROM official_populations", con)
print("official_populations columns:", pop_cols)
with pl.Config(tbl_rows=40, tbl_width_chars=200):
    print(pops.select([c for c in pops.columns if c not in ("members",)]))


def _generation_chain(frame: pl.DataFrame) -> list[str]:
    """Order the generations of one population along its `supersedes` chain."""
    if "supersedes" not in frame.columns:
        return frame.get_column("population_hash").to_list()
    by_hash = {r["population_hash"]: r for r in frame.iter_rows(named=True)}
    roots = [h for h, r in by_hash.items() if not r["supersedes"]]
    chain: list[str] = []
    cursor = roots[0] if roots else frame.row(0, named=True)["population_hash"]
    while cursor:
        chain.append(cursor)
        nxt = [h for h, r in by_hash.items() if r["supersedes"] == cursor]
        cursor = nxt[0] if nxt else None
    return chain


# Every generation of every baseline population, in publication order, as one flat list of
# cohorts. K is the total membership of that list: the trial count accumulates over the bot, not
# over a population name, so a second signal family adds to K exactly as a second generation does.
cohorts: list[dict] = []
members_by_gen: dict[str, set[str]] = {}
for _name in BASELINE_POPULATIONS:
    _frame = pops.filter(pl.col("name") == _name)
    if _frame.is_empty():
        print(f"\n{_name}: not published")
        continue
    _chain = _generation_chain(_frame)
    print(f"\n{_name}: {len(_chain)} generation(s): {_chain}")
    for g, h in enumerate(_chain, start=1):
        m = pl.read_database(
            f"SELECT member_hash FROM official_population_members WHERE population_hash = '{h}'",
            con,
        )
        members_by_gen[h] = set(m.get_column("member_hash").to_list())
        cohorts.append({"population": _name, "generation": g, "hash": h})
        print(f"  generation {g} {h}: {len(members_by_gen[h])} members")
if not cohorts:
    sys.exit("no baseline population published")
generations = [c["hash"] for c in cohorts]
cohort_label = {
    c["hash"]: f"{c['population'].split(':', 1)[1]} generation {c['generation']}" for c in cohorts
}
overlap = set.intersection(*members_by_gen.values()) if len(members_by_gen) > 1 else set()
print(f"\nmembers shared between any two cohorts: {len(overlap)}")

preds = pl.read_database(
    "SELECT p.prediction_hash, t.label, t.family, t.config_name, p.checkpoint_kind, "
    "p.checkpoint_value, p.split FROM prediction_sets p JOIN training_runs t "
    "ON p.training_hash = t.training_hash",
    con,
)
holdout_rows = preds.filter(pl.col("split") == "holdout").height
print(f"prediction sets by split: {preds.group_by('split').len().to_dicts()}; holdout rows: {holdout_rows}")
assert holdout_rows == 0, "holdout prediction sets exist; the evidence boundary was crossed"
con.close()

runs_all = load_backtest_runs(CS)
metrics_all = load_backtest_metrics(CS)
print(f"backtest_runs rows: {runs_all.height} | backtest_metrics rows: {metrics_all.height}")


def _spec_fields(spec_json: str) -> dict:
    spec = json.loads(spec_json)
    strat = spec.get("strategy", spec)
    signal = strat["signal"]
    bc = spec["backtest_config"]
    # `top_k` belongs to the cross-sectional family only; the per-pair family is parameterized by
    # its quantile and its lookback instead. `spec_key` is the one column every grouping below
    # uses, so a family without `top_k` is described rather than crashed on.
    method = str(signal.get("method", ""))
    top_k = signal.get("top_k")
    if method == "per_symbol_rolling_percentile":
        spec_key = (
            f"q{float(signal['long_q']):.2f}/L{int(signal['lookback_days'])}"
            f"/bpd{int(signal.get('bars_per_day', 390))}/{signal.get('direction', 'long_only')}"
        )
    else:
        spec_key = f"top_k={top_k}"
    return {
        "method": method,
        "spec_key": spec_key,
        "top_k": int(top_k) if top_k is not None else -1,
        "long_q": float(signal["long_q"]) if "long_q" in signal else float("nan"),
        "lookback_days": int(signal["lookback_days"]) if "lookback_days" in signal else -1,
        "bars_per_day": int(signal["bars_per_day"]) if "bars_per_day" in signal else -1,
        "direction": str(signal.get("direction", "long_only")),
        "long_short": bool(signal.get("long_short", False)),
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


fields = pl.DataFrame(
    [{"backtest_hash": h, **_spec_fields(s)} for h, s in zip(runs_all["backtest_hash"], runs_all["spec_json"])]
)
runs_all = runs_all.join(fields, on="backtest_hash", how="left")

cfg = get_backtest_config(CS)
slip = cfg.slippage_bps / 1e4
print(
    f"engine costs: commission {cfg.commission_bps:.2f} bps, slippage {cfg.slippage_bps:.2f} bps "
    f"per crossing, initial_cash {cfg.initial_cash:,.0f}"
)


def gen_table(pop_hash: str) -> pl.DataFrame:
    """One row per member: spec fields, prediction lineage and registry metrics."""
    hashes = list(members_by_gen[pop_hash])
    runs = runs_all.filter(pl.col("backtest_hash").is_in(hashes))
    table = (
        runs.select(
            "backtest_hash", "prediction_hash", "method", "spec_key", "top_k", "long_q",
            "lookback_days", "bars_per_day", "direction", "long_short", "step", "cadence",
            "costs", "fill_timing", "stage", "git_commit", "elapsed_s",
        )
        .join(preds.drop("split"), on="prediction_hash", how="left")
        .join(metrics_all, on="backtest_hash", how="left")
    )
    assert table.height == len(hashes), (table.height, len(hashes))
    assert table["sharpe"].null_count() == 0
    return table


tables = {h: gen_table(h) for h in generations}

# ---------------------------------------------------------------- benchmark: 1/N long, daily
print("\n== equal-weight benchmark (1/N long the five pairs, daily close-to-close, validation window) ==")
bench: dict[str, dict] = {}
bench_series: dict[str, pl.DataFrame] = {}
# The full decision grid per label, kept because the turnover reconstruction below has to resolve
# the rebalance schedule over it and NOT over a spec's own weight frame. The cross-sectional books
# hold something on every session, so the two coincided; a per-pair time-series book is flat on a
# third of its sessions and writes no weight row there, and `gather_every(step)` over that sparse
# series would count every fifth *held* session instead of every fifth session. That is the same
# sparse-frame mistake that made `avg_turnover` wrong on the generation-1 rows (open question 16).
# `backtest_runner._run_engine` resolves over `predictions["timestamp"].unique()`; this is that
# grid.
decision_grid: dict[str, pl.Series] = {}
for label in LABELS:
    px = load_backtest_prices_for(CS, label, split="validation")
    decision_grid[label] = pl.Series("ts", px["timestamp"].unique().sort().to_list())
    r = (
        px.sort("symbol", "timestamp")
        .with_columns(pl.col("close").pct_change().over("symbol").alias("ret"))
        .drop_nulls("ret")
        .group_by("timestamp")
        .agg(pl.col("ret").mean().alias("ew"), pl.len().alias("n"))
        .sort("timestamp")
        .with_columns(pl.col("timestamp").cast(pl.Date).alias("date"))
    )
    ew = r["ew"].to_numpy()
    sr = float(np.mean(ew) / np.std(ew, ddof=1) * np.sqrt(PPY))
    bench[label] = {
        "sharpe": round(sr, 3),
        "cagr": round(float(np.mean(ew) * PPY), 4),
        "vol": round(float(np.std(ew, ddof=1) * np.sqrt(PPY)), 4),
        "n_periods": int(len(ew)),
        "window": [str(r["date"].min()), str(r["date"].max())],
    }
    bench_series[label] = r.select("date", "ew")
    print(label, bench[label])


# ---------------------------------------------------------------- statistics helpers
def sharpe(x: np.ndarray) -> float:
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def _expected_max_sharpe(variance_trials: float, n_trials: int) -> float:
    if n_trials <= 1 or variance_trials <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - (1.0 / n_trials))
    z2 = stats.norm.ppf(1.0 - (np.exp(-1.0) / n_trials))
    weight = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return float(np.sqrt(variance_trials) * weight)


def dsr_notebook(observed_sharpe, skewness, kurtosis, n_samples, n_trials, variance_trials):
    """16_strategy_simulation/12_dsr_validation.py::deflated_sharpe_ratio (annualized inputs)."""
    annualizer = np.sqrt(PPY)
    observed_native = observed_sharpe / annualizer
    variance_native = max(variance_trials, 0.0) / PPY
    expected_max = _expected_max_sharpe(variance_native, n_trials)
    variance = 1.0 - skewness * observed_native + ((kurtosis - 1.0) / 4.0) * observed_native**2
    z = (observed_native - expected_max) * np.sqrt(max(n_samples - 1, 1)) / np.sqrt(max(variance, 1e-12))
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


# ---------------------------------------------------------------- per-member raw and active series
def read_returns(bh: str) -> pl.DataFrame:
    return pl.read_parquet(case_dir / "run_log" / "backtest" / bh / "daily_returns.parquet").with_columns(
        pl.col("timestamp").cast(pl.Date).alias("date")
    )


def book_shape(bh: str) -> dict:
    """How the engine actually held the book: sessions with one sleeve missing, gross / equity.

    The pinned `config/backtest/base.yaml` declares `allow_short_selling: true` without
    `allow_leverage`, so the engine runs a cash account that sizes a short by the cash on hand;
    when a rotation closes the short leg while the long leg stays invested there is no cash left
    and the new short is filled for a few units only (BOT.md open question 23). The share of
    invested sessions with |net| / gross > 0.5 counts those one-legged sessions.
    """
    ps = pl.read_parquet(case_dir / "run_log" / "backtest" / bh / "portfolio_state.parquet")
    invested = ps.filter(pl.col("gross_exposure") > 0)
    if invested.height == 0:
        return {"one_legged_share": float("nan"), "gross_over_equity": float("nan"), "invested_sessions": 0}
    ratio = (invested["net_exposure"] / invested["gross_exposure"]).abs()
    return {
        "one_legged_share": float((ratio > 0.5).mean()),
        "gross_over_equity": float((invested["gross_exposure"] / invested["equity"]).median()),
        "invested_sessions": int(invested.height),
    }


def member_stats(bh: str, label: str) -> dict:
    ret = read_returns(bh)
    r = ret["daily_return"].to_numpy()
    joined = ret.join(bench_series[label], on="date", how="inner")
    a = (joined["daily_return"] - joined["ew"]).to_numpy()
    raw, act = moments(r), moments(a)
    return {
        "backtest_hash": bh,
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


print("\nreading the return series of every member of every generation ...")
stats_rows = []
for h in generations:
    for row in tables[h].select("backtest_hash", "label").iter_rows(named=True):
        stats_rows.append({"population_hash": h, **member_stats(row["backtest_hash"], row["label"])})
member_stats_df = pl.DataFrame(stats_rows)
K_total = member_stats_df.height
var_raw = float(np.var(member_stats_df["sharpe_raw"].to_numpy(), ddof=1))
var_active = float(np.var(member_stats_df["sharpe_active"].to_numpy(), ddof=1))
print(
    f"K (cumulative trials, all generations) = {K_total}; variance of the annualized Sharpe across "
    f"trials: raw {var_raw:.4f}, active {var_active:.4f}"
)
for h in generations:
    sub = member_stats_df.filter(pl.col("population_hash") == h)
    diff = (sub["sharpe_raw"] - tables[h].join(sub, on="backtest_hash")["sharpe"]).abs().max()
    print(f"  {h}: recomputed raw Sharpe agrees with the registry to {diff:.2e}")


# ---------------------------------------------------------------- per-generation sections
def dist(s: pl.Series) -> dict:
    return {
        "n": s.len(),
        "min": round(float(s.min()), 3),
        "median": round(float(s.median()), 3),
        "p90": round(float(s.quantile(0.9)), 3),
        "max": round(float(s.max()), 3),
        "share_positive": round(float((s > 0).mean()), 3),
    }


def turnover_from_ledger(row: dict) -> float:
    """Notional traded per session as a fraction of initial cash, from the fill ledger."""
    return float((row["total_slippage"] / slip) / cfg.initial_cash / max(int(row["n_periods"]), 1))


def turnover_from_weights(bh: str, cadence: str, step: int, label: str) -> tuple[float, int]:
    """Fixed `target_weight_turnover` on the scheduled targets (the runner's rule).

    Returns the summed |dw| over the run and the number of scheduled targets that moved the book.
    The schedule is resolved over the label's whole decision grid, exactly as
    `backtest_runner._run_engine` does, and only then intersected with the weight frame - see the
    note beside `decision_grid`.
    """
    w = pl.read_parquet(case_dir / "run_log" / "backtest" / bh / "weights.parquet")
    schedule = resolve_decision_schedule(decision_grid[label], cadence, step, cfg.calendar)
    scheduled = w.filter(pl.col("timestamp").is_in(schedule.implode()))
    t = target_weight_turnover(scheduled)
    return float(t["turnover"].sum()), int((t["turnover"] > 0).sum())


def gross_sharpe_ledger(bh: str, row: dict) -> tuple[float, float]:
    r = read_returns(bh)["daily_return"].to_numpy()
    drag = (row["total_commission"] + row["total_slippage"]) / cfg.initial_cash / max(int(row["n_periods"]), 1)
    return sharpe(r + drag), float(drag * PPY)


def report_best(tag: str, row: dict, st: dict, K: int) -> dict:
    label = row["label"]
    out = {"tag": tag, "backtest_hash": row["backtest_hash"], "K": K}
    print(
        f"\n-- {tag}: {row['backtest_hash']} {row['family']} {row['config_name']} @{row['checkpoint_value']} "
        f"{label} {row['spec_key']} long_short={row['long_short']} direction={row['direction']}"
    )
    print(
        f"   raw: Sharpe {st['sharpe_raw']:.3f} (registry {row['sharpe']:.3f}), n {st['n']}, skew {st['skew_raw']:.3f}, "
        f"excess kurtosis {st['exkurt_raw']:.3f}; active vs 1/N ({bench[label]['sharpe']}): Sharpe "
        f"{st['sharpe_active']:.3f}, n {st['n_active']}, skew {st['skew_active']:.3f}, excess kurtosis {st['exkurt_active']:.3f}"
    )
    psr0 = psr_notebook(st["sharpe_raw"], 0.0, st["n"], st["skew_raw"], st["exkurt_raw"] + 3.0)
    psrb = psr_notebook(st["sharpe_raw"], bench[label]["sharpe"], st["n"], st["skew_raw"], st["exkurt_raw"] + 3.0)
    psra = psr_notebook(st["sharpe_active"], 0.0, st["n_active"], st["skew_active"], st["exkurt_active"] + 3.0)
    print(f"   PSR vs 0: {psr0:.4f}; PSR vs benchmark Sharpe {bench[label]['sharpe']}: {psrb:.4f}; PSR of the active return vs 0: {psra:.4f}")
    out.update({"psr_vs_0": psr0, "psr_vs_benchmark": psrb, "psr_active_vs_0": psra})
    for kind, sr, sk, ek, n, var in (
        ("raw", st["sharpe_raw"], st["skew_raw"], st["exkurt_raw"], st["n"], var_raw),
        ("active", st["sharpe_active"], st["skew_active"], st["exkurt_active"], st["n_active"], var_active),
    ):
        nb = dsr_notebook(sr, sk, ek + 3.0, n, K, var)
        lib = dsr_library(sr, sk, ek, n, K, var)
        print(
            f"   DSR on {kind:6s} return at K={K}: notebook {nb['dsr']:.4f} (z {nb['z']:.2f}, expected max Sharpe "
            f"{nb['expected_max_sharpe']:.3f}, is_significant@{CONFIDENCE_LEVEL} {nb['is_significant']}); "
            f"ml4t-diagnostic {lib['dsr']:.4f} (p {lib['p_value']:.4f}, expected max {lib['expected_max_sharpe']:.3f}, "
            f"minTRL {lib['min_trl']:.0f}, adequate sample {lib['adequate_sample']}, is_significant {lib['is_significant']})"
        )
        out[f"dsr_{kind}_notebook"] = nb
        out[f"dsr_{kind}_library"] = lib
    return out


summary: dict = {
    "K": K_total,
    "populations": BASELINE_POPULATIONS,
    "generations": {},
    "benchmark": bench,
}
for _cohort in cohorts:
    g, h = _cohort["generation"], _cohort["hash"]
    table = tables[h]
    st_df = member_stats_df.filter(pl.col("population_hash") == h)
    print(f"\n\n================ {cohort_label[h]}: {h} ({table.height} members) ================")
    print("method =", table["method"].unique().to_list(), "| settings =", sorted(table["spec_key"].unique().to_list()))
    print("long_short =", table["long_short"].unique().to_list(), "| direction =", table["direction"].unique().to_list())
    print("distinct cost blocks:", table["costs"].unique().to_list())
    print("distinct fill timings:", table["fill_timing"].unique().to_list())
    print("stages:", table["stage"].unique().to_list(), "| git commits:", table["git_commit"].unique().to_list())
    print("elapsed_s: total %.0f, median %.2f" % (table["elapsed_s"].sum(), table["elapsed_s"].median()))

    print("\n== counts per label / family / setting ==")
    with pl.Config(tbl_rows=20):
        print(
            table.group_by("label", "family", "spec_key")
            .agg(pl.len().alias("n"), pl.col("num_trades").sum().cast(pl.Int64).alias("trades"))
            .sort("label", "family", "spec_key")
        )

    print("\n== validation net Sharpe distribution (registry, net of %.1f bps per crossing) ==" % cfg.slippage_bps)
    print("all:", dist(table["sharpe"]))
    for (label, spec_key), sub in sorted(table.group_by("label", "spec_key"), key=lambda kv: kv[0]):
        print(f"{label} {spec_key}:", dist(sub["sharpe"]))
    for (label, fam), sub in sorted(table.group_by("label", "family"), key=lambda kv: kv[0]):
        print(f"{label} {fam}:", dist(sub["sharpe"]))
    print("\n== active Sharpe over the 1/N long book: distribution ==")
    joined = table.join(st_df, on="backtest_hash")
    print("all:", dist(joined["sharpe_active"]))
    for (label, spec_key), sub in sorted(joined.group_by("label", "spec_key"), key=lambda kv: kv[0]):
        print(f"{label} {spec_key}:", dist(sub["sharpe_active"]))
    print("\n== book shape from portfolio_state (share of invested sessions with |net|/gross > 0.5; median gross/equity) ==")
    print(
        "all: one_legged_share median %.3f, p90 %.3f, max %.3f; gross/equity median %.3f, min %.3f, max %.3f"
        % (
            joined["one_legged_share"].median(), joined["one_legged_share"].quantile(0.9), joined["one_legged_share"].max(),
            joined["gross_over_equity"].median(), joined["gross_over_equity"].min(), joined["gross_over_equity"].max(),
        )
    )
    for (label, spec_key), sub in sorted(joined.group_by("label", "spec_key"), key=lambda kv: kv[0]):
        print(
            f"{label} {spec_key}: one_legged_share median {sub['one_legged_share'].median():.3f}, "
            f"max {sub['one_legged_share'].max():.3f}; gross/equity median {sub['gross_over_equity'].median():.3f}"
        )
    # The exposure gate, declared before the sweep ran (open question 23). It costs no trial: it
    # is a measurement of what the engine held, printed here rather than left as a footnote, so a
    # number in the tables above is never read as a dollar-neutral book when it was not one.
    directional = joined.filter(pl.col("one_legged_share") > ONE_LEGGED_THRESHOLD)
    print(
        f"EXPOSURE GATE at |net|/gross > {ONE_LEGGED_THRESHOLD} on more than half the invested "
        f"sessions: {directional.height} of {joined.height} specs are DIRECTIONAL books and are "
        "reported as such, never as dollar-neutral"
    )
    if table["long_short"].unique().to_list() == [False]:
        print(
            "  (this cohort is declared long only, so gross <= 1 and net == gross by "
            "construction: the cash account holds exactly the book the signal targeted)"
        )

    print("\n== top 5 specs by validation net Sharpe (raw) ==")
    rows = []
    for row in table.sort("sharpe", descending=True).head(5).iter_rows(named=True):
        g_sr, drag = gross_sharpe_ledger(row["backtest_hash"], row)
        t_sum, n_changes = turnover_from_weights(
            row["backtest_hash"], row["cadence"], row["step"], row["label"]
        )
        st = st_df.filter(pl.col("backtest_hash") == row["backtest_hash"]).row(0, named=True)
        rows.append(
            {
                "backtest_hash": row["backtest_hash"],
                "family": row["family"],
                "preset": row["config_name"],
                "ckpt": row["checkpoint_value"],
                "label": row["label"],
                "setting": row["spec_key"],
                "sharpe_net": round(row["sharpe"], 3),
                "sharpe_gross_ledger": round(g_sr, 3),
                "cost_drag_ann": round(drag, 4),
                "sharpe_active": round(st["sharpe_active"], 3),
                "sharpe_ci95": f"[{row['sharpe_ci95_lo']:.2f}, {row['sharpe_ci95_hi']:.2f}]",
                "psr_pvalue": round(row["psr_pvalue"], 3) if row["psr_pvalue"] is not None else None,
                "max_dd": round(row["max_drawdown"], 4),
                "cagr": round(row["cagr"], 4),
                "vol": round(row["volatility"], 4),
                "turnover_ledger": round(turnover_from_ledger(row), 3),
                "turnover_weights_per_session": round(t_sum / max(int(row["n_periods"]), 1), 3),
                "turnover_weights_per_rebalance": round(t_sum / max(n_changes, 1), 3),
                "rebalances_changed": n_changes,
                "avg_turnover_registry": round(row["avg_turnover"], 4),
                "trades": int(row["num_trades"]),
                "n_periods": int(row["n_periods"]),
                "one_legged_share": round(st["one_legged_share"], 3),
                "gross_over_equity": round(st["gross_over_equity"], 3),
            }
        )
    with pl.Config(tbl_cols=30, tbl_width_chars=280, tbl_rows=20):
        print(pl.DataFrame(rows))

    print("\n== top 5 specs by ACTIVE Sharpe over the 1/N long book ==")
    with pl.Config(tbl_cols=30, tbl_width_chars=200, tbl_rows=20):
        print(
            joined.sort("sharpe_active", descending=True)
            .head(5)
            .select("backtest_hash", "family", "config_name", "checkpoint_value", "label", "spec_key", "sharpe", "sharpe_active", "max_drawdown", "num_trades")
        )

    # sample weight frame: offsetting sleeves and net exposure
    sample_hash = table.sort("sharpe", descending=True).row(0, named=True)["backtest_hash"]
    w = pl.read_parquet(case_dir / "run_log" / "backtest" / sample_hash / "weights.parquet")
    ps = pl.read_parquet(case_dir / "run_log" / "backtest" / sample_hash / "portfolio_state.parquet")
    per_ts = w.group_by("timestamp").agg(
        pl.col("weight").filter(pl.col("weight") > 0).sum().alias("long"),
        pl.col("weight").filter(pl.col("weight") < 0).sum().alias("short"),
        pl.col("weight").sum().alias("net"),
        pl.len().alias("names"),
    )
    invested = ps.filter(pl.col("gross_exposure") > 0)
    net_ratio = (invested["net_exposure"] / invested["gross_exposure"]).abs()
    print(f"\n== sample weights {sample_hash}: {w.height} rows, negative weights {int((w['weight'] < 0).sum())}, "
          f"sessions with a target {per_ts.height}; per-session long sum {per_ts['long'].min():.2f}..{per_ts['long'].max():.2f}, "
          f"short sum {per_ts['short'].min():.2f}..{per_ts['short'].max():.2f}, net {per_ts['net'].min():.2f}..{per_ts['net'].max():.2f}; "
          f"portfolio_state |net|/gross on invested sessions: median {net_ratio.median():.4f}, max {net_ratio.max():.4f} "
          f"({invested.height} of {ps.height} sessions invested)")
    print(w.head(6))

    # best of this generation: DSR at the cumulative K
    best_raw = table.sort("sharpe", descending=True).row(0, named=True)
    st_best = st_df.filter(pl.col("backtest_hash") == best_raw["backtest_hash"]).row(0, named=True)
    best_out = report_best(f"{cohort_label[h]} best by raw Sharpe", best_raw, st_best, K_total)
    best_active_hash = joined.sort("sharpe_active", descending=True).row(0, named=True)["backtest_hash"]
    active_out = None
    if best_active_hash != best_raw["backtest_hash"]:
        row_a = table.filter(pl.col("backtest_hash") == best_active_hash).row(0, named=True)
        st_a = st_df.filter(pl.col("backtest_hash") == best_active_hash).row(0, named=True)
        active_out = report_best(f"{cohort_label[h]} best by active Sharpe", row_a, st_a, K_total)

    # by validation year for the best raw
    ret = read_returns(best_raw["backtest_hash"])
    yr = ret.with_columns(pl.col("date").dt.year().alias("year")).group_by("year").agg(
        (pl.col("daily_return").mean() / pl.col("daily_return").std() * np.sqrt(PPY)).alias("sharpe"),
        pl.len().alias("n"),
    ).sort("year")
    print("   best by raw Sharpe, by validation year:", yr.to_dicts())

    # the gate: how many specs of this generation are significant at K on the active return
    n_sig_active = n_sig_raw = n_half_active = 0
    for st in st_df.iter_rows(named=True):
        nb_a = dsr_notebook(st["sharpe_active"], st["skew_active"], st["exkurt_active"] + 3.0, st["n_active"], K_total, var_active)
        nb_r = dsr_notebook(st["sharpe_raw"], st["skew_raw"], st["exkurt_raw"] + 3.0, st["n"], K_total, var_raw)
        n_sig_active += nb_a["is_significant"]
        n_half_active += nb_a["dsr"] > 0.5
        n_sig_raw += nb_r["is_significant"]
    print(
        f"\n== gate, {cohort_label[h]} at K={K_total} (notebook formula, own moments): specs with DSR >= {CONFIDENCE_LEVEL} "
        f"on the ACTIVE return: {n_sig_active} of {st_df.height}; on the raw return: {n_sig_raw} of {st_df.height}; "
        f"active DSR > 0.5: {n_half_active}"
    )
    summary["generations"][h] = {
        "cohort": cohort_label[h],
        "generation": g,
        "members": table.height,
        "method": table["method"].unique().to_list(),
        "settings": sorted(table["spec_key"].unique().to_list()),
        "long_short": table["long_short"].unique().to_list(),
        "directional_specs": directional.height,
        "best_raw": {k: v for k, v in best_raw.items() if k != "spec_json" and not isinstance(v, float)} | {"sharpe": best_raw["sharpe"]},
        "best_raw_dsr": best_out,
        "best_active_dsr": active_out,
        "n_significant_active": n_sig_active,
        "n_significant_raw": n_sig_raw,
    }

print("\n\n== summary (json) ==")
print(json.dumps(summary, default=str)[:6000])
