"""Phase-5 evaluation and report for exness_usidx_sess: both specs, the swap ledger, K and the DSR.

Not a stage. It reads BOTH workspaces' registries, because this bot is one bot in two experiment
workspaces (one per spec) and K accumulates over the BOT, while every registry helper in this
repository reads ONE registry. See bots/exness_usidx_sess/BOT.md, "the two-registry rule".

WHAT THIS FILE IS, AND WHAT THE ENGINE IS
-----------------------------------------
CORRECTED 2026-09-08 (third pass), after a mentor review of phases 3 to 5. The version this
replaces opened by explaining why the engine could not be used at all. That was true of the
`engine` execution mode and is not true of the run as a whole: `13_backtest.py` now passes
`execution_mode="vectorized"` to `build_backtest_spec` (the keyword has always been there,
case_studies/utils/backtest_presets.py), which is the same weights-times-forward-return
arithmetic on the shared code path, so every trial of the sweep registers a `backtest_runs` row
and the phase-5 gate of the roadmap is met by the registry rather than by this file.

So there are two artefacts of this sweep and they answer different questions.

* The REGISTRY (`backtest_runs`, both workspaces) is the auditable record: identity, lineage,
  population membership, metrics. Its cost model is the shared one - `_run_vectorized` charges
  `turnover x rate`, i.e. `|dw|` between consecutive rebalances, at the single conservative rate
  the engine resolves from setup.yaml.
* THIS FILE is the exact economics of the declared strategy: a full ROUND TRIP on every session
  a symbol is held (both specs close their own position at the end of their own window, so there
  is no turnover to net), the financing of the real nights the overnight spec crosses, and the
  spread of the REGIME each session actually traded in. Where the two disagree, the reason is
  named in BOT.md rather than averaged over.

THE ARITHMETIC
--------------
The strategy holds exactly one label per decision, so its return is arithmetic and not a
simulation:

    net(t) = SUM_symbol [ w(t,s) * label(t,s)
                          - |w(t,s)| * 2 * spread_bps(t,s) / 1e4
                          + max(w(t,s), 0) * swap_bps(t,s) / 1e4 ]

with the weights built by the same case_studies.utils.signals.build_target_weights_from_config
the backtest stage sends the engine, from the same signal dicts (`_sweep.declared_settings`, one
function imported by both files so the two cannot drift apart).

COSTS, AND THE CORRECTION TO THEM
---------------------------------
`spread_bps(t, s)` is now per session and per symbol, under the rule
`setup.yaml::costs.spread_bps_charge_rule`: the conservative declared p90 (4.05 bps) before
2024-11-01, which covers the pre-2024-10 regime and the break month itself, and the declared
`recent_regime_p90` of that symbol after it (US500 1.05, USTEC 0.87). Both numbers were already
in `setup.yaml` before phase 5 ran; the rule adds only the boundary between them.

The version this replaces charged 4.05 bps on every session of both specs. The validation
windows are 2024-08-29 .. 2026-02-26 (intraday) and 2024-11-05 .. 2026-02-26 (overnight), which
lie almost entirely AFTER the break, so it charged roughly four to five times the cost its own
evaluation window measured. The old justification - that the engine takes one number - did not
apply to a scorer that never called the engine. It still applies to the engine run, which is why
`backtest_runs` is charged the conservative rate and this file is not.

The swap is charged only on the overnight spec and only on a long leg, per real night, tripled
on a Friday, through bots/_shared/costs_mt5.holding_cost_points.

WHAT IS COUNTED, AND ON HOW MANY LEGS
-------------------------------------
`n` is the number of sessions in a member's own scoring window - the union of its walk-forward
validation windows - and a flat session is an observation of zero, not a missing row. `n_traded`
is the number of those sessions on which the book actually held something, and `n_legs` the
number of symbol-sessions. They are not the same statistic and the difference is large: the best
raw member of generation 1 rested on 31 legs over 302 sessions while its `min_trl` was computed
against n = 302. `setup.yaml` declares `backtest.sweep.min_traded_sessions` and every headline
below is reported both with and without that floor.

THE BENCHMARK, AND THE THREE CONVENTIONS IT IS COMPUTED UNDER
-------------------------------------------------------------
    DRAFT FOR THE USER - NOT APPROVED. bots/exness_usidx_sess/BOT.md proposes the active return
    over a 1/N long book measured on the SAME session window as the spec, at a Deflated Sharpe
    Ratio of 0.95. The user has not approved that convention. Everything below is reported at it
    AND against SR = 0, with the trial count K printed beside every figure. No result below is
    described as passing a gate.

The benchmark is the passive alternative a trader actually had: hold both indices equally
weighted over exactly the window the spec holds. It pays the same financing on the overnight
spec and no spread at all, because a buy-and-hold book crosses twice over the whole window
rather than twice a session.

A book that is flat on 85 to 93 % of its sessions and a book invested 100 % of the time are two
different risk budgets, so the active return is reported under three conventions and the reader
is told which is which. NONE of them is a selection criterion here: the same members are
reported under all three, and nothing is chosen by comparing them.

    active     = net - bench                     the OLD convention: the benchmark is fully
                                                 invested whatever the strategy did.
    active_gx  = net - gross_exposure(t) * bench the benchmark is scaled by the strategy's own
                                                 realised GROSS exposure that session. This is
                                                 the convention the mentor named. On a
                                                 long-short session (gross 2.0) it levers the
                                                 long benchmark twice, which is why the third
                                                 is reported beside it.
    active_nx  = net - net_exposure(t) * bench   the benchmark is scaled by the strategy's own
                                                 realised NET exposure, the directional beta the
                                                 book actually carried; 0 on a flat session and
                                                 0 on a balanced long-short one.

Run from the repository root in the research environment::

    uv run python case_studies/exness_usidx_sess/_report_phase5.py

Conventions: annualization 252; DSR confidence_level 0.95; the variance of trials is the sample
variance of the annualized Sharpe across every trial with a DEFINED Sharpe on THAT return
definition. Each column is dropped on its OWN nulls: joining them removed a trial from the raw
statistics because its active series happened to be constant, which is a reason that has nothing
to do with its raw return.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

CS = "exness_usidx_sess"
PPY = 252
CONFIDENCE_LEVEL = 0.95
EULER_MASCHERONI = 0.5772156649015329
WORKSPACES = {
    "intraday": Path.home() / "ml4t/experiments/usidx_intraday",
    "overnight": Path.home() / "ml4t/experiments/usidx_overnight",
}
OUT = Path.home() / "ml4t/experiments/usidx_phase5_members.parquet"

# A return series whose standard deviation is below this is constant for any practical purpose,
# and its Sharpe ratio is not a number. It happens here for two different reasons, and the
# version of this file that the mentor reviewed named only the first and named it wrongly.
#   (a) A book that holds the SAME thing on every session of its window. Its raw return moves
#       with the market, but its ACTIVE return against a 1/N book it equals is the same constant
#       every session.
#   (b) A book that NEVER OPENS A POSITION on its window. Its raw return is 0.0 on every
#       session, so both the raw and the active series are constant. Most of the long-only
#       per_symbol_rolling_percentile degenerates are this, and it is the opposite situation
#       from (a): not "holds both indices always" but "holds nothing ever".
# Reporting 1e16 for either would poison the variance of trials the Deflated Sharpe Ratio
# divides by, so they are reported as undefined and counted in K. `n_traded` tells them apart,
# and the report prints the split rather than asserting a cause.
CONSTANT_RETURN_SD = 1e-9


# ---------------------------------------------------------------- statistics helpers
# The four functions below are copied verbatim from case_studies/exness_fx_d1/_report_phase5.py so
# that a Deflated Sharpe Ratio reported for this bot and one reported for that bot are the same
# statistic computed the same way.
def sharpe(x: np.ndarray) -> float | None:
    sd = float(np.std(x, ddof=1))
    if sd < CONSTANT_RETURN_SD:
        return None
    return float(np.mean(x) / sd * np.sqrt(PPY))


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
    z = (
        (observed_native - expected_max)
        * np.sqrt(max(n_samples - 1, 1))
        / np.sqrt(max(variance, 1e-12))
    )
    p = float(stats.norm.cdf(z))
    return {
        "dsr": p,
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


def dist(s: pl.Series) -> dict:
    v = s.drop_nulls().to_numpy()
    if not len(v):
        return {}
    return {
        "n": int(len(v)),
        "median": round(float(np.median(v)), 3),
        "p90": round(float(np.quantile(v, 0.9)), 3),
        "max": round(float(np.max(v)), 3),
        "min": round(float(np.min(v)), 3),
        "share_gt_0": round(float((v > 0).mean()), 3),
    }


# ---------------------------------------------------------------- the swap ledger
def swap_bps_table(spec: str, setup: dict) -> pl.DataFrame:
    """Financing owed per symbol-session for a LONG position, in basis points of notional.

    Zero on the intraday spec by construction: it is entered after the cash open and closed at the
    cash close, so no server midnight falls inside the holding period. On the overnight spec it is
    the swap of the REAL nights between the execution instant and the label instant - one on a
    normal session and three when the Friday triple falls inside - which is why the count comes
    from rollover_nights rather than being assumed to be one.
    """
    from bots._shared.costs_mt5 import holding_cost_points
    from bots._shared.sessions import ServerClock

    from case_studies.exness_usidx_sess._features import load_session_panel

    panel = load_session_panel(setup, spec=spec, verbose=False)
    if spec == "intraday":
        return panel.select("symbol", "timestamp").with_columns(pl.lit(0.0).alias("swap_bps"))

    cfg = setup["costs"]["swap"]
    clock = ServerClock.from_dict(setup["decision"]["server_clock"])
    swaps = {
        symbol: {
            "swap_long": float(v["long"]),
            "swap_short": float(v["short"]),
            "swap_mode": 1,
            "swap_rollover3days": 5 if str(cfg["rollover3days"]).lower() == "friday" else 3,
            "charges_weekends": False,
        }
        for symbol, v in cfg["points_per_lot_per_night"].items()
    }
    rows = []
    for r in panel.drop_nulls(["exec_ts", "label_ts", "exec_open"]).iter_rows(named=True):
        pts = holding_cost_points(
            r["symbol"],
            "long",
            r["exec_ts"],
            r["label_ts"],
            swaps=swaps,
            clock=clock,
            charges_weekends=False,
        )
        # points -> price: the point value is 0.01 on both indices. price -> bps: / price * 1e4.
        rows.append(
            {
                "symbol": r["symbol"],
                "timestamp": r["timestamp"],
                "swap_bps": 1e4 * (pts / 100.0) / float(r["exec_open"]),
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- one workspace
def load_workspace(spec: str, root: Path) -> dict:
    os.environ["ML4T_OUTPUT_DIR"] = str(root)
    case_dir = root / CS
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    label = setup["labels"]["primary"]
    holdout_start = pl.lit(str(setup["evaluation"]["holdout_start"])).str.to_date()
    labels_frame = pl.read_parquet(case_dir / "labels" / f"{label}.parquet").filter(
        pl.col("timestamp") < holdout_start
    )

    from case_studies.utils.registry import load_prediction_sets, load_training_runs

    sets = load_prediction_sets(CS)
    training = load_training_runs(CS).select("training_hash", "family", "config_name")
    members = sets.join(training, on="training_hash", how="left").unique(subset=["prediction_hash"])
    print(
        f"\n===== workspace {spec}: {members.height} prediction sets, label {label}, "
        f"{labels_frame.height} development label rows ====="
    )
    return {
        "spec": spec,
        "setup": setup,
        "label_col": label,
        "labels": labels_frame,
        "swap": swap_bps_table(spec, setup),
        "members": members,
    }


def score(ws: dict, weights: pl.DataFrame, sessions: pl.DataFrame) -> pl.DataFrame:
    """Per-session net return of a weight book, exactly: w * label - round trip + swap.

    Also carries the two exposure series the benchmark conventions need and the counts the
    sample size is really made of: `legs`, the number of symbol-sessions with a position.
    """
    from case_studies.exness_usidx_sess._sweep import with_spread_bps

    if "asset" in weights.columns and "symbol" not in weights.columns:
        weights = weights.rename({"asset": "symbol"})
    joined = (
        weights.join(ws["labels"], on=["timestamp", "symbol"], how="inner")
        .join(ws["swap"], on=["timestamp", "symbol"], how="left")
        .with_columns(pl.col("swap_bps").fill_null(0.0))
    )
    # The spread is per session and per symbol under the declared regime rule, not one number
    # for a window that straddles a fourfold change in the tape.
    joined = with_spread_bps(ws["setup"], joined)
    held = (
        joined.with_columns(
            (
                pl.col("weight") * pl.col(ws["label_col"])
                - pl.col("weight").abs() * pl.col("spread_bps") * 2.0 / 1e4
                + pl.when(pl.col("weight") > 0)
                .then(pl.col("weight") * pl.col("swap_bps") / 1e4)
                .otherwise(0.0)
            ).alias("leg")
        )
        .group_by("timestamp")
        .agg(
            pl.col("leg").sum().alias("net"),
            pl.col("weight").abs().sum().alias("gross"),
            pl.col("weight").sum().alias("net_exposure"),
            pl.len().alias("legs"),
        )
    )
    # A session on which the book held nothing earns ZERO, and it is an observation. The weight
    # frame carries a row only for a symbol that was held, so a naive group-by silently drops
    # every flat session and with it every zero from the sample the Sharpe ratio is computed over.
    # On the settings that trade rarely that is catastrophic rather than cosmetic: the first
    # version of this function reported a Sharpe of 6.16 over 22 observations for a strategy that
    # was flat on 280 of the 302 sessions in its window.
    return (
        sessions.join(held, on="timestamp", how="left")
        .with_columns(
            pl.col("net").fill_null(0.0),
            pl.col("gross").fill_null(0.0),
            pl.col("net_exposure").fill_null(0.0),
            pl.col("legs").fill_null(0).cast(pl.Int64),
        )
        .sort("timestamp")
    )


def benchmark(ws: dict) -> pl.DataFrame:
    """1/N long both indices over exactly this spec's own window, net of financing, no spread."""
    return (
        ws["labels"]
        .join(ws["swap"], on=["timestamp", "symbol"], how="left")
        .with_columns(pl.col("swap_bps").fill_null(0.0))
        .with_columns((pl.col(ws["label_col"]) + pl.col("swap_bps") / 1e4).alias("leg"))
        .group_by("timestamp")
        .agg(pl.col("leg").mean().alias("bench"))
        .sort("timestamp")
    )


# ---------------------------------------------------------------- main
def main() -> None:
    from case_studies.exness_usidx_sess._sweep import declared_settings, min_traded_sessions
    from case_studies.utils.registry import read_predictions
    from case_studies.utils.signals import build_target_weights_from_config

    rows: list[dict] = []
    benches: dict[str, pl.DataFrame] = {}
    floor: int | None = None
    for spec, root in WORKSPACES.items():
        ws = load_workspace(spec, root)
        spec_floor = min_traded_sessions(ws["setup"])
        if floor is None:
            floor = spec_floor
        elif floor != spec_floor:
            raise RuntimeError(
                "the two workspaces declare different min_traded_sessions "
                f"({floor} vs {spec_floor}): one bot, one floor"
            )
        # The benchmark is restricted to the window the prediction sets cover, so the active
        # return subtracts a benchmark measured on the strategy's own sessions and not a longer
        # one. The comment that stood here ASSERTED that all the sets of one workspace share the
        # same walk-forward validation windows and then took the window from row 0 anyway; the
        # loop below now CHECKS it on every member, because both measurement bugs this file has
        # already had came from an unchecked assumption about the session grid.
        first_hash = ws["members"].row(0, named=True)["prediction_hash"]
        first = read_predictions(CS, first_hash)
        window = first.select("timestamp").unique().sort("timestamp")
        reference_grid = tuple(window["timestamp"].to_list())
        b = window.join(benchmark(ws), on="timestamp", how="inner").sort("timestamp")
        if b.height != window.height:
            raise RuntimeError(
                f"{spec}: the 1/N benchmark is missing {window.height - b.height} of the "
                f"{window.height} sessions the prediction grid covers; an inner join would have "
                "measured the active return on a shorter window than the raw return"
            )
        benches[spec] = b
        arr = b["bench"].to_numpy()
        bsr = sharpe(arr)
        print(
            f"  1/N long benchmark on this spec's own window: {len(arr)} sessions, "
            f"Sharpe {bsr:+.3f}, mean {np.mean(arr) * 1e4:+.2f} bps/session, "
            f"{'net of financing' if spec == 'overnight' else 'no financing owed'}"
        )
        print(
            f"  window {b['timestamp'].min()} .. {b['timestamp'].max()}; "
            f"declared minimum traded sessions {spec_floor}"
        )
        settings = declared_settings(ws["setup"])
        print(f"  {len(settings)} declared settings x {ws['members'].height} prediction sets")
        for m in ws["members"].iter_rows(named=True):
            preds = read_predictions(CS, m["prediction_hash"])
            if preds is None or preds.is_empty():
                continue
            # The denominator of every statistic below is the set of sessions this prediction set
            # was SCORED on - the union of its walk-forward validation windows - and not the whole
            # development window. Padding with zeros from before the model existed would add
            # hundreds of flat observations to the track record.
            sessions = preds.select("timestamp").unique().sort("timestamp")
            grid = tuple(sessions["timestamp"].to_list())
            if grid != reference_grid:
                raise RuntimeError(
                    f"{spec}: prediction set {m['prediction_hash']} is scored on a different "
                    f"session grid from {first_hash} ({len(grid)} sessions "
                    f"{grid[0]}..{grid[-1]} against {len(reference_grid)} sessions "
                    f"{reference_grid[0]}..{reference_grid[-1]}). The benchmark window is taken "
                    "from one member, so every member has to share it; it does not, so this run "
                    "stops instead of comparing two windows"
                )
            for setting in settings:
                series = score(ws, build_target_weights_from_config(preds, setting), sessions)
                if series.height < 60:
                    continue
                merged = series.join(b, on="timestamp", how="inner")
                if merged.height != series.height:
                    raise RuntimeError("the benchmark does not cover this member's whole window")
                raw = moments(series["net"].to_numpy())
                act = moments((merged["net"] - merged["bench"]).to_numpy())
                act_gx = moments((merged["net"] - merged["gross"] * merged["bench"]).to_numpy())
                act_nx = moments(
                    (merged["net"] - merged["net_exposure"] * merged["bench"]).to_numpy()
                )
                rows.append(
                    {
                        "spec": spec,
                        "family": m["family"],
                        "config_name": m["config_name"],
                        "checkpoint": m["checkpoint_value"],
                        "prediction_hash": m["prediction_hash"],
                        "method": setting["method"],
                        "setting": str(
                            {
                                k: v
                                for k, v in setting.items()
                                if k not in ("method", "bars_per_day", "direction")
                            }
                        ),
                        "n": raw["n"],
                        # The counts the sample size is really made of. `n` is sessions in the
                        # window; `n_traded` is sessions on which a position existed.
                        "n_traded": int((series["legs"] > 0).sum()),
                        "n_legs": int(series["legs"].sum()),
                        "avg_gross": float(series["gross"].mean()),
                        "sharpe_raw": raw["sharpe"],
                        "skew_raw": raw["skew"],
                        "exkurt_raw": raw["exkurt"],
                        "n_active": act["n"],
                        "sharpe_active": act["sharpe"],
                        "skew_active": act["skew"],
                        "exkurt_active": act["exkurt"],
                        "sharpe_active_gx": act_gx["sharpe"],
                        "skew_active_gx": act_gx["skew"],
                        "exkurt_active_gx": act_gx["exkurt"],
                        "sharpe_active_nx": act_nx["sharpe"],
                        "skew_active_nx": act_nx["skew"],
                        "exkurt_active_nx": act_nx["exkurt"],
                    }
                )

    table = pl.DataFrame(rows)
    if table.is_empty():
        print("no trials scored")
        return
    K = table.height
    K_CUMULATIVE = 2 * K
    # Per column, not jointly. A trial whose ACTIVE series is constant but whose RAW series is not
    # has a defined raw Sharpe, and dropping it from the raw statistics because of the other
    # column removes it from the variance of trials and from the raw significance count for a
    # reason that has nothing to do with the raw return (measured on the version this replaces:
    # intraday leaves_31_huber at checkpoint 300, sharpe_raw -1.1465 with sharpe_active null).
    stat_cols = ("sharpe_raw", "sharpe_active", "sharpe_active_gx", "sharpe_active_nx")
    scored = {col: table.drop_nulls([col]) for col in stat_cols}
    variance = {col: float(np.var(frame[col].to_numpy(), ddof=1)) for col, frame in scored.items()}

    print(f"\n== K (trials of THIS generation: both specs, both families, every setting) = {K} ==")
    print(
        f"== K cumulative over the bot's GENERATIONS = {K_CUMULATIVE}. Generation 1 asked the "
        f"same {K}\n   questions under the mirrored short threshold - an always-invested book - "
        "and this generation\n   asks them under the signed one, on the same data and the same "
        "folds, so the conservative\n   count is the sum. Every DSR below is printed at BOTH "
        f"K = {K} and K = {K_CUMULATIVE}; neither may be\n   quoted without the other."
    )
    for col, frame in scored.items():
        print(
            f"  {col:18s}: {frame.height} of {K} trials have a defined Sharpe "
            f"({K - frame.height} constant), variance of the annualized Sharpe "
            f"{variance[col]:.4f}"
        )
    print(
        "\n  A constant return series is counted in K - it was tried - and excluded from the\n"
        "  variance of trials and the significance counts, where it has nothing to say. Two\n"
        "  different books produce one: a book that holds the same thing on EVERY session (its\n"
        "  active return against a benchmark it equals is constant), and a book that NEVER\n"
        "  OPENS a position (its raw return is 0.0 on every session). n_traded tells them apart:"
    )
    degenerate = table.filter(pl.col("sharpe_raw").is_null() | pl.col("sharpe_active").is_null())
    print(
        degenerate.group_by("spec", "method")
        .agg(
            pl.len().alias("degenerate"),
            (pl.col("n_traded") == 0).sum().alias("never_traded"),
            (pl.col("n_traded") == pl.col("n")).sum().alias("traded_every_session"),
        )
        .sort("spec", "method")
    )
    print(table.group_by("spec", "method").agg(pl.len().alias("trials")).sort("spec", "method"))

    print("\n== how much of the time these books are invested, and on how many legs ==")
    print(
        table.group_by("spec", "method")
        .agg(
            pl.len().alias("trials"),
            pl.col("n").median().alias("sessions"),
            pl.col("n_traded").median().alias("median_n_traded"),
            pl.col("n_legs").median().alias("median_n_legs"),
            pl.col("avg_gross").median().round(3).alias("median_avg_gross"),
            (pl.col("n_traded") >= floor).sum().alias("clears_floor"),
        )
        .sort("spec", "method")
    )
    eligible = table.filter(pl.col("n_traded") >= floor)
    print(
        f"\n  {eligible.height} of {K} trials clear the declared floor of {floor} traded "
        f"sessions ({K - eligible.height} do not).\n  The floor was declared in setup.yaml "
        "BEFORE this re-scoring ran and is NOT a filter on K: every\n  trial is counted in K "
        "whatever its leg count. It decides only which members may be read as\n  candidates."
    )

    print("\n== distribution of the validation Sharpe, net of the regime costs ==")
    for col in stat_cols:
        print(f"  {col:18s} all:", dist(scored[col][col]))
    for (spec, method), sub in table.group_by(["spec", "method"], maintain_order=True):
        for col in ("sharpe_raw", "sharpe_active", "sharpe_active_gx"):
            print(f"  {col:18s} {spec:9s} {method:30s}", dist(sub[col]))

    show = [
        "spec", "family", "config_name", "checkpoint", "method", "setting",
        "n", "n_traded", "n_legs", "sharpe_raw", "sharpe_active", "sharpe_active_gx",
    ]
    with pl.Config(tbl_rows=12, tbl_width_chars=240):
        for tag, frame in (("ALL trials", table), (f"trials with n_traded >= {floor}", eligible)):
            for col in ("sharpe_raw", "sharpe_active"):
                print(f"\n== top 6 by {col}, {tag} ==")
                sub = frame.drop_nulls([col])
                if sub.is_empty():
                    print("   none")
                    continue
                print(sub.sort(col, descending=True).head(6).select(show))

    print("\n== PSR and DSR, for the best trial on each definition ==")
    for tag, frame in (("ALL trials", table), (f"n_traded >= {floor}", eligible)):
        for label_txt, col in (
            ("best by raw Sharpe", "sharpe_raw"),
            ("best by active Sharpe", "sharpe_active"),
            ("best by exposure-matched active Sharpe", "sharpe_active_gx"),
        ):
            sub = frame.drop_nulls([col])
            if sub.is_empty():
                print(f"\n-- {tag} / {label_txt}: no trial with a defined Sharpe")
                continue
            best = sub.sort(col, descending=True).row(0, named=True)
            bsr = sharpe(benches[best["spec"]]["bench"].to_numpy())
            print(
                f"\n-- {tag} / {label_txt}: {best['spec']} / {best['family']} "
                f"{best['config_name']} ckpt {best['checkpoint']} / {best['method']} "
                f"{best['setting']}"
            )
            print(
                f"   n {best['n']} sessions, n_traded {best['n_traded']}, "
                f"n_legs {best['n_legs']}, mean gross exposure {best['avg_gross']:.3f}"
            )
            print(
                f"   raw       Sharpe {best['sharpe_raw']:+.3f}  "
                f"skew {best['skew_raw']:+.2f}  excess kurtosis {best['exkurt_raw']:+.2f}"
            )
            for name, c in (
                ("active   ", "sharpe_active"),
                ("active_gx", "sharpe_active_gx"),
                ("active_nx", "sharpe_active_nx"),
            ):
                value = best[c]
                shown = f"{value:+.3f}" if value is not None else "undefined (constant)"
                print(f"   {name} Sharpe {shown}")
            print(
                f"   the 1/N long book on the same window: Sharpe {bsr:+.3f}, "
                "invested 100 % of the time"
            )
            print(
                "   PSR vs 0 (raw) "
                f"{psr_notebook(best['sharpe_raw'], 0.0, best['n'], best['skew_raw'], best['exkurt_raw'] + 3.0):.4f}"
                " | PSR vs the benchmark "
                f"{psr_notebook(best['sharpe_raw'], bsr, best['n'], best['skew_raw'], best['exkurt_raw'] + 3.0):.4f}"
            )
            if best["sharpe_active"] is not None:
                print(
                    "   PSR of the active return vs 0 "
                    f"{psr_notebook(best['sharpe_active'], 0.0, best['n_active'], best['skew_active'], best['exkurt_active'] + 3.0):.4f}"
                )
            for kind, sr, sk, ek, n, col_var in (
                ("raw      ", best["sharpe_raw"], best["skew_raw"], best["exkurt_raw"], best["n"], "sharpe_raw"),
                ("active   ", best["sharpe_active"], best["skew_active"], best["exkurt_active"], best["n_active"], "sharpe_active"),
                ("active_gx", best["sharpe_active_gx"], best["skew_active_gx"], best["exkurt_active_gx"], best["n_active"], "sharpe_active_gx"),
            ):
                if sr is None:
                    print(f"   DSR({kind}): undefined (constant return series)")
                    continue
                for k_value in (K, K_CUMULATIVE):
                    nb = dsr_notebook(sr, sk, ek + 3.0, n, k_value, variance[col_var])
                    lib = dsr_library(sr, sk, ek, n, k_value, variance[col_var])
                    print(
                        f"   DSR({kind}) at K={k_value}: notebook {nb['dsr']:.4f} "
                        f"(E[max SR] {nb['expected_max_sharpe']:+.3f}), library {lib['dsr']:.4f}, "
                        f"min track record {lib['min_trl']:.0f} sessions vs n {n} "
                        f"(n_traded {best['n_traded']}), adequate sample {lib['adequate_sample']}"
                    )

    print(f"\n== how many trials reach {CONFIDENCE_LEVEL}, under the DRAFT convention ==")
    for k_value in (K, K_CUMULATIVE):
        print(f"  at K = {k_value}:")
        for col in stat_cols:
            frame = scored[col]
            skew_col = col.replace("sharpe", "skew")
            kurt_col = col.replace("sharpe", "exkurt")
            n_col = "n" if col == "sharpe_raw" else "n_active"
            hits = eligible_hits = 0
            for r in frame.iter_rows(named=True):
                d = dsr_notebook(
                    r[col], r[skew_col], r[kurt_col] + 3.0, r[n_col], k_value, variance[col]
                )
                hits += d["is_significant"]
                eligible_hits += d["is_significant"] and r["n_traded"] >= floor
            print(
                f"    {col:18s} DSR >= {CONFIDENCE_LEVEL}: {hits} of {frame.height} with a "
                f"defined Sharpe ({K} tried); of those, {eligible_hits} also clear "
                f"n_traded >= {floor}"
            )
    print(
        "\n   The 0.95 level, the 1/N benchmark and all three active-return conventions are a\n"
        "   DRAFT the user has not approved. The counts are stated at that level; they are not a\n"
        "   gate anyone has agreed to, and the raw figures beside them are what a different\n"
        "   convention re-scores from. No member was chosen by comparing the conventions."
    )
    table.write_parquet(OUT)
    print(f"\nwrote the per-trial table to {OUT}")


if __name__ == "__main__":
    main()
