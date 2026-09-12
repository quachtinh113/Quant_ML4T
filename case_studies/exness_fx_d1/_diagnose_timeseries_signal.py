"""Zero-trial diagnostic for the per-pair time-series signal family (user decision 6).

Not a stage, not a backtest, not a trial. A read-only measurement over prediction sets that
`06_linear` and `07_gbm` already registered, run BEFORE `13_backtest` opens the family, to
answer the two questions Chapter 16 notebook `08_signal_method_comparison` asks of a signal
method (`16_strategy_simulation/08_signal_method_comparison.py`, sections on the activation
and transition rate):

1. **Does the rule fire at all?** `case_studies/utils/signals.py::per_symbol_rolling_percentile_signal`
   defaults to ``bars_per_day = 390`` - the number of minute bars in a United States equity
   session - and computes its trailing quantile over ``W = lookback_days * bars_per_day`` rows
   with ``min_samples = W // 2``. This bot takes one decision per session, so the correct value
   is ``bars_per_day = 1``. Left at the default, ``W`` is 390 times too long, ``min_samples`` is
   never reached on the 1,031 validation sessions, every threshold is null, every signal is 0 -
   and the sweep still completes, registering a thousand books that never traded. The
   diagnostic below measures the activation rate under both settings so the difference is a
   number in a log rather than a belief.
2. **How often does the book change?** The transition rate is what the cost model will be
   charged on, and it is knowable before a single backtest runs.

It also prints the warmup each setting costs (``min_samples`` sessions per pair with a null
threshold and therefore no position) and the number of pairs held per session, which is what
decides whether the book the cash account has to hold is one position or five.

Nothing is written: no registry row, no artifact, no population. Run from the repository root
with the experiment selected::

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_fx_d1 uv run python \
        case_studies/exness_fx_d1/_diagnose_timeseries_signal.py
"""

from __future__ import annotations

import sqlite3

import polars as pl

from case_studies.utils.registry import read_predictions
from case_studies.utils.signals import (
    _signals_to_equal_weights,
    per_symbol_rolling_percentile_signal,
)
from utils.paths import get_case_study_dir

CS = "exness_fx_d1"
LABELS = ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d"]
# The grid the family will be run on, and the wrong default it must not be run on.
LOOKBACK_DAYS = [63, 252]
LONG_Q = 0.80
BARS_PER_DAY_CORRECT = 1
BARS_PER_DAY_DEFAULT = 390  # signals.py default: minute bars of a US equity session

case_dir = get_case_study_dir(CS)
db = case_dir / "run_log" / "registry.db"
print("registry:", db)

con = sqlite3.connect(str(db))
preds = pl.read_database(
    "SELECT p.prediction_hash, t.label, t.family, t.config_name, p.checkpoint_value, p.split "
    "FROM prediction_sets p JOIN training_runs t ON p.training_hash = t.training_hash "
    "WHERE p.split = 'validation'",
    con,
)
con.close()
print(f"validation prediction sets: {preds.height}")
assert preds.height > 0, "no validation prediction sets registered"

# One representative prediction set per label and family: the diagnostic measures the SIGNAL
# RULE, not a model, so it needs a score series of the right shape, not the whole catalog.
sample = (
    preds.sort("label", "family", "config_name", "checkpoint_value")
    .group_by("label", "family", maintain_order=True)
    .head(1)
)
print("\nrepresentative prediction sets:")
print(sample)


def diagnose(scores: pl.DataFrame, *, lookback_days: int, bars_per_day: int) -> dict:
    """Activation and transition rate of the per-pair rolling-percentile rule, long-only."""
    signalled = per_symbol_rolling_percentile_signal(
        scores,
        long_q=LONG_Q,
        lookback_days=lookback_days,
        bars_per_day=bars_per_day,
        signal_type="long_only",
    )
    n_rows = signalled.height
    active = int((signalled["signal"] != 0).sum())
    changes = (
        signalled.sort(["symbol", "timestamp"])
        .with_columns(
            (pl.col("signal") != pl.col("signal").shift(1).over("symbol")).alias("_changed")
        )
        .drop_nulls("_changed")
    )
    weights = _signals_to_equal_weights(signalled)
    per_session = (
        weights.group_by("timestamp")
        .agg(pl.len().alias("names"), pl.col("weight").sum().alias("gross"))
        .sort("timestamp")
    )
    sessions = signalled["timestamp"].n_unique()
    return {
        "lookback_days": lookback_days,
        "bars_per_day": bars_per_day,
        "window_rows": lookback_days * bars_per_day,
        "min_samples": (lookback_days * bars_per_day) // 2,
        "pair_sessions": n_rows,
        "sessions": sessions,
        "activation_rate": round(active / n_rows, 4) if n_rows else 0.0,
        "transition_rate": round(float(changes["_changed"].mean()), 4) if changes.height else 0.0,
        "sessions_invested": per_session.height,
        "invested_share": round(per_session.height / sessions, 4) if sessions else 0.0,
        "names_median": float(per_session["names"].median()) if per_session.height else 0.0,
        "names_max": int(per_session["names"].max()) if per_session.height else 0,
        "gross_max": round(float(per_session["gross"].max()), 4) if per_session.height else 0.0,
    }


rows = []
for row in sample.iter_rows(named=True):
    scores = read_predictions(CS, row["prediction_hash"]).select("timestamp", "symbol", "y_score")
    for bars_per_day in (BARS_PER_DAY_CORRECT, BARS_PER_DAY_DEFAULT):
        for lookback in LOOKBACK_DAYS:
            rows.append(
                {
                    "label": row["label"],
                    "family": row["family"],
                    "config": row["config_name"],
                    **diagnose(scores, lookback_days=lookback, bars_per_day=bars_per_day),
                }
            )

table = pl.DataFrame(rows)
with pl.Config(tbl_cols=20, tbl_rows=60, tbl_width_chars=240):
    print("\n== activation and transition rate by setting ==")
    print(table)

wrong = table.filter(pl.col("bars_per_day") == BARS_PER_DAY_DEFAULT)
right = table.filter(pl.col("bars_per_day") == BARS_PER_DAY_CORRECT)
print(
    f"\nbars_per_day={BARS_PER_DAY_DEFAULT} (the library default): activation rate "
    f"{wrong['activation_rate'].min()}..{wrong['activation_rate'].max()}, sessions invested "
    f"{wrong['sessions_invested'].min()}..{wrong['sessions_invested'].max()}"
)
print(
    f"bars_per_day={BARS_PER_DAY_CORRECT} (this bot): activation rate "
    f"{right['activation_rate'].min()}..{right['activation_rate'].max()}, sessions invested "
    f"{right['sessions_invested'].min()}..{right['sessions_invested'].max()}, "
    f"transition rate {right['transition_rate'].min()}..{right['transition_rate'].max()}"
)
if float(wrong["activation_rate"].max()) > 0.0:
    print(
        "NOTE: the library default fires on this history, so the silent-no-trade trap does not "
        "reproduce here; the declared bars_per_day: 1 is still the only correct value."
    )
else:
    print(
        "CONFIRMED: at the library default nothing ever fires and the sweep would register "
        "books that never trade. The family is run at bars_per_day: 1."
    )
if float(right["activation_rate"].min()) <= 0.0:
    raise SystemExit(
        "the declared setting never fires either: do not open the sweep, the family is empty"
    )
print("\nGATE: the declared setting fires and turns over. The family may be opened.")
