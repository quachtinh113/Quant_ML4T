# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exness BTC 8h (exness_btc_8h): Features Built From Fitted Models
#
# Every column in `03_financial_features` is a formula: the same arithmetic on any window of any
# series. The three columns here are different - each is the output of a **model that has to be
# estimated**, and an estimated feature is point-in-time only if the estimate never saw the rows
# it is applied to.
#
# So the discipline is the whole notebook. Each model is fitted on a fold's **training** slots
# alone, and its recursion is then run forward across training and validation together **without
# re-estimating** - because a recursion carries state, and restarting it at the first validation
# slot would throw away everything it had learned. What is emitted is the **filtered** quantity,
# the one that reads only observations up to each slot; a smoothed one would produce
# better-looking features and no tradable ones, and there is an assertion that says which of the
# two this is.
#
# Two models, chosen for what this instrument does rather than copied:
#
# | Model | What it says | Why here |
# |---|---|---|
# | GARCH(1,1) | the conditional volatility of the next eight hours, and how far the realised move was from it | Bitcoin's volatility clusters hard, and a signal of a given size means different things at 30 % and 90 % annualised |
# | Two-state Gaussian HMM | the filtered probability that the market is in its turbulent state | the calm and turbulent states of this instrument differ by more than a factor of two in dispersion, and every cost in `16_costs` is charged in basis points against that |
#
# There is deliberately no Kalman trend and no ARIMA here, and the reason is a measurement rather
# than a preference: `01_feasibility_analysis` found the eight-hour return's own autocorrelation
# inside the white-noise band at every lag but one, and `bots/exness_fx_d1/BOT.md` records that its
# Kalman observation-noise MLE collapsed to zero in every fold, making the emitted column a
# per-fold constant. A model that cannot say anything is a column a downstream fit still has to
# spend a degree of freedom on.
#
# ## Learning objectives
#
# - Lay out a walk-forward fold ladder before anything is fitted, and prove every variant label's
#   validation window opens after the primary label's training window closes
# - Fit a volatility model inside a fold and filter it forward without re-estimating
# - Show that deleting the second half of a training window does not move the first half's outputs
# - Emit one artifact carrying a fold id, so a downstream model reads the vintage that belongs to
#   the fold it is scored on
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 9. Route B fork of
# [`exness_gold_sess/04_model_based_features`](../exness_gold_sess/04_model_based_features.ipynb).
# Reads the decision panel through `_features.load_decision_panel` and
# `labels/fwd_ret_8h.parquet` (for its timeline only - **no label value is read here, and no
# information coefficient is computed**); writes `features/model_based.parquet` with a
# `.digest.json` sidecar.

# %%
"""exness_btc_8h: Features Built From Fitted Models (Route B fork of exness_gold_sess)."""

import logging
import warnings

import numpy as np
import pandas as pd
import polars as pl
from hmmlearn.hmm import GaussianHMM
from IPython.display import display
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from case_studies.exness_btc_8h._features import decision_grid
from case_studies.research.holdout import build_holdout_cv
from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.cv_window import assert_variant_folds_are_out_of_sample
from case_studies.utils.temporal import filtered_state_probs, sort_states_by_variance
from bots._shared.mt5_loader import load_mt5_bars
from utils.artifact_specs import load_setup_config, resolve_label_buffer
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
set_global_seeds(42)

CASE_STUDY_ID = "exness_btc_8h"

# %% [markdown]
# Everything in the next cell can be overridden without editing the file, which is how a reader
# runs a smaller version first. Each one trades runtime for scope.

# %% tags=["parameters"]
MAX_FOLDS = 0
N_HMM_RESTARTS = 10
GARCH_MAXITER = 400
START_DATE = None

# %% [markdown]
# ## 1. Configuration and the price grid
#
# The panel is built by the one function every stage of this case study uses, so the slot a model
# is fitted on is the slot the label was sealed on. Two constants below are read from `setup.yaml`
# rather than typed: the volatility z-score window is the **shortest** declared close-to-close
# window, and the training-data floor is one year of decision slots - `decision.slots_per_year`,
# never `evaluation.periods_per_year`. Those two keys say different things on this bot (1,095
# slots a year against 365 days a year for annualisation), and reading the wrong one here would
# set the floor at a third of a year.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
LABELS_DIR = CASE_DIR / "labels"
FEATURES_DIR = CASE_DIR / "features"
SETUP = load_setup_config(CASE_STUDY_ID)
HMM_N_STATES = 2
HMM_STABILITY_REL_TOL = 1e-3
HMM_SCALE = 100.0
SLOTS_PER_DAY = len(SETUP["decision"]["snapshots_utc"])
SLOTS_PER_YEAR = int(SETUP["decision"]["slots_per_year"])
MIN_TRAIN_SLOTS = SLOTS_PER_YEAR
MIN_VAL_SLOTS = 10
VOL_WINDOW = int(min(SETUP["features"]["windows"]["close_to_close_volatility"]))
VOL_COL = f"slot_vol_{VOL_WINDOW}"
GARCH_Z_WINDOW = int(SETUP["features"]["windows"]["zscore"])
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)
_EVAL = load_evaluation_config(CASE_STUDY_ID)
HOLDOUT_START = pd.Timestamp(_EVAL["holdout_start"]).date()
HOLDOUT_END = pd.Timestamp(_EVAL["holdout_end"]).date()
HISTORY_START = START_DATE or str(SETUP["universe"]["history_start"])
SYMBOLS = sorted(SETUP["universe"]["symbols"])
assert len(SYMBOLS) == SETUP["universe"]["n_assets"]

print(f"Primary label {PRIMARY_LABEL}, purge buffer {LABEL_BUFFER}; holdout {HOLDOUT_START} to {HOLDOUT_END}")
print(
    f"{SLOTS_PER_DAY} decision slots a day, {SLOTS_PER_YEAR} a year; a fold needs at least "
    f"{MIN_TRAIN_SLOTS} training slots and {MIN_VAL_SLOTS} validation slots"
)
print(
    f"The regime model reads a {VOL_WINDOW}-slot volatility; the GARCH z-score is taken against a "
    f"{GARCH_Z_WINDOW}-slot trailing window (both from setup.yaml::features.windows)"
)

# %%
bars = load_mt5_bars(
    "8h", symbols=SYMBOLS, start_date=HISTORY_START, end_date=str(HOLDOUT_END)
).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
prices = decision_grid(bars, verbose=False).sort(["symbol", "timestamp"])
all_slots = sorted(prices["timestamp"].unique().to_list())
print(
    f"Loaded {prices.height:,} rows, {len(SYMBOLS)} instrument, {len(all_slots):,} decision slots, "
    f"{all_slots[0]} to {all_slots[-1]}"
)

# %% [markdown]
# ## 2. The walk-forward folds, derived before anything is fitted
#
# The folds are laid out on **dates**, which on this grid is a convenience rather than a decision:
# with three slots a day and no session sleeves there is nothing a mid-day boundary would split.
# It matters for a different reason - `01_feasibility_analysis` printed the same ladder from the
# same function, so a fold id names the same window everywhere.
#
# A holdout fold is appended. It is not a choice: `build_holdout_cv` reads
# `evaluation.holdout_start/holdout_end` and trains on everything before them less one label
# buffer, so this stage can emit a feature vintage for the rows phase 7 will score **without any
# stage here reading a holdout outcome**. Nothing in this notebook computes a label or an
# information coefficient.
#
# The first assertion is the one `exness_gold_sess` had to add after the defect surfaced two
# stages later: the label parquet's key set must be the decision grid's key set. The library
# derives the canonical fold from the label parquet's timeline and thereby treats that parquet as
# the trading calendar - a proxy that is only true while every decision also resolves a label.
# `02_labels` publishes both labels on the full decision grid with nulls where a rule leaves them
# undefined, which turns the proxy into an equality; this is that equality placed where it can
# break.

# %%
label_frame = pl.read_parquet(LABELS_DIR / f"{PRIMARY_LABEL}.parquet")
timeline = (
    label_frame.select(pl.col("timestamp").dt.date().alias("timestamp")).unique().sort("timestamp")
)
_panel_keys = set(zip(prices["timestamp"].to_list(), prices["symbol"].to_list(), strict=True))
_label_keys = set(
    zip(label_frame["timestamp"].to_list(), label_frame["symbol"].to_list(), strict=True)
)
assert _label_keys == _panel_keys, (
    f"the {PRIMARY_LABEL} parquet is not published on the decision grid: "
    f"{len(_panel_keys - _label_keys):,} panel keys are missing from it and "
    f"{len(_label_keys - _panel_keys):,} of its keys are not on the panel"
)
print(
    f"timeline {timeline.height:,} decision dates, derived from {len(_label_keys):,} "
    f"{PRIMARY_LABEL} keys that are exactly the {len(_panel_keys):,} keys of the decision panel"
)

raw_folds = generate_cv_splits(
    timeline, case_study_id=CASE_STUDY_ID, label_buffer=LABEL_BUFFER, date_col="timestamp"
)
folds = [
    {
        "fold": int(split["fold"]),
        **{
            name: pd.Timestamp(split[name]).date()
            for name in ("train_start", "train_end", "val_start", "val_end")
        },
    }
    for split in raw_folds
]
if MAX_FOLDS:
    folds = folds[:MAX_FOLDS]
VALIDATION_FOLD_IDS = {f["fold"] for f in folds}
_derived = build_holdout_cv(
    {"label": PRIMARY_LABEL, "computation": {"cv": {"folds": folds}}},
    case_study=CASE_STUDY_ID,
    timeline=timeline["timestamp"].to_list(),
    label=PRIMARY_LABEL,
)["folds"][0]
holdout_fold = {
    "fold": int(_derived["fold"]),
    **{
        name: pd.Timestamp(_derived[name]).date()
        for name in ("train_start", "train_end", "val_start", "val_end")
    },
}
HOLDOUT_FOLD_ID = holdout_fold["fold"]
assert holdout_fold["train_end"] < HOLDOUT_START, (
    "the holdout fold trains through the boundary, so the last training label resolves inside the "
    "window it is meant to be judged against"
)
assert (holdout_fold["val_start"], holdout_fold["val_end"]) == (HOLDOUT_START, HOLDOUT_END)
folds.append(holdout_fold)
for f in folds:
    f["n_train"] = sum(f["train_start"] <= d.date() <= f["train_end"] for d in all_slots)
    f["n_val"] = sum(f["val_start"] <= d.date() <= f["val_end"] for d in all_slots)
    tag = "  <- HOLDOUT vintage, scored once in phase 7" if f["fold"] == HOLDOUT_FOLD_ID else ""
    print(
        f"  Fold {f['fold']}: train {f['train_start']}..{f['train_end']} ({f['n_train']:,} slots), "
        f"validate {f['val_start']}..{f['val_end']} ({f['n_val']:,} slots){tag}"
    )

# %% [markdown]
# ### One ladder, two labels
#
# This artifact carries **one** fold set, built above on the primary label's geometry, and a model
# trained on the variant label reads it by `fold` id. So the values it reads for fold F were fitted
# through the **primary** label's `train_end`, and the variant's own validation window has to open
# after that or the model is scored on slots its features already saw.

# %%
variant_gaps = pl.DataFrame(assert_variant_folds_are_out_of_sample(CASE_STUDY_ID, PRIMARY_LABEL))
with pl.Config(tbl_rows=variant_gaps.height):
    display(variant_gaps.sort(["label", "fold"]))
_gaps = set(variant_gaps["gap"].to_list())
assert len(_gaps) == 1, (
    "the configured labels do not sit on one fold ladder: the gap between the primary's train_end "
    f"and a variant's val_start varies -> {sorted(_gaps)}"
)
print(
    f"{variant_gaps.height} (variant, fold) rows, every gap {variant_gaps['gap'].min()} - one "
    f"ladder, and that single value is {PRIMARY_LABEL}'s own purge"
)

# %% [markdown]
# ### The purge gap, measured in slots of this grid
#
# `labels.buffer` is declared as a duration - 24 hours - because `evaluation.calendar: crypto` maps
# to no session calendar. What matters here is how many **decision slots** sit between the last
# training slot and the first validation slot: it must be at least the label horizon (one slot for
# `fwd_ret_8h`) plus one of embargo, or the last training row's outcome would be known inside the
# validation window.

# %%
MIN_PURGE_SLOTS = int(SETUP["labels"]["rebalance_step"][PRIMARY_LABEL]) + 1
gap_rows = []
for f in folds:
    for symbol in SYMBOLS:
        grid = prices.filter(pl.col("symbol") == symbol)["timestamp"].to_list()
        last_train = max(i for i, d in enumerate(grid) if d.date() <= f["train_end"])
        first_val = min(i for i, d in enumerate(grid) if d.date() >= f["val_start"])
        gap_rows.append(
            {
                "fold": f["fold"],
                "symbol": symbol,
                "last_train_slot": grid[last_train],
                "first_val_slot": grid[first_val],
                "gap_slots": first_val - last_train,
                "calendar_days": (f["val_start"] - f["train_end"]).days,
            }
        )
purge_gaps = pl.DataFrame(gap_rows).sort(["fold", "symbol"])
with pl.Config(tbl_rows=purge_gaps.height):
    display(purge_gaps)
assert (purge_gaps["gap_slots"] >= MIN_PURGE_SLOTS).all(), (
    f"a train -> validation gap is below {MIN_PURGE_SLOTS} slots: "
    f"{purge_gaps.filter(pl.col('gap_slots') < MIN_PURGE_SLOTS).to_dicts()}"
)
print(
    f"purge gap {purge_gaps['gap_slots'].min()}-{purge_gaps['gap_slots'].max()} decision slots "
    f"(>= {MIN_PURGE_SLOTS}), {purge_gaps['calendar_days'].min()}-"
    f"{purge_gaps['calendar_days'].max()} calendar days"
)

# %% [markdown]
# ## 3. How volatile the next eight hours are expected to be
#
# A GARCH(1,1) on the decision-slot return: the conditional variance of slot $t$ is
#
# $$\sigma_t^2 = \omega + \alpha \, \varepsilon_{t-1}^2 + \beta \, \sigma_{t-1}^2$$
#
# The three parameters are fitted by maximum likelihood **on the training slots alone**, over a
# parameterisation that keeps them positive and stationary without a constraint solver, and the
# recursion then runs forward across training and validation together without re-estimating.
#
# Three columns come out, and each answers a different question:
#
# - `garch_vol` - the conditional volatility itself, annualised on the decision grid;
# - `garch_vol_z` - where that sits against its own trailing window, so a model reads "unusually
#   turbulent for this instrument" rather than a level whose scale drifted through the sample;
# - `garch_shock` - the last realised return divided by the volatility that had been forecast for
#   it. It is a **surprise**, not a direction, and it is the column with any chance of being a
#   signal rather than a state.
#
# All three are computed from information available at the slot: `garch_shock` at slot $t$ uses
# the return that ended at $t$ and the forecast made for it at $t-1$.

# %%
def _garch_filter(eps: np.ndarray, omega: float, alpha: float, beta: float, var0: float) -> np.ndarray:
    """Conditional variance path of a GARCH(1,1); ``var[t]`` is the forecast made at ``t-1``."""
    var = np.empty(len(eps))
    prev = var0
    for i in range(len(eps)):
        var[i] = prev
        prev = omega + alpha * eps[i] ** 2 + beta * prev
    return var


def _garch_nll(theta: np.ndarray, eps: np.ndarray, var0: float) -> float:
    """Negative Gaussian log-likelihood, in a parameterisation that cannot leave the simplex.

    ``alpha`` and ``beta`` come from a softmax-like map so that ``alpha + beta < 1`` by
    construction: the alternative - a bounded optimiser with an inequality constraint - stops at
    the boundary and reports a fit that is really a failure.
    """
    log_omega, a_raw, b_raw = theta
    omega = float(np.exp(log_omega))
    total = 1.0 / (1.0 + np.exp(-a_raw))          # alpha + beta, in (0, 1)
    share = 1.0 / (1.0 + np.exp(-b_raw))          # alpha's share of it
    alpha, beta = total * share, total * (1.0 - share)
    var = _garch_filter(eps, omega, alpha, beta, var0)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return 1e12
    return float(0.5 * np.sum(np.log(var) + eps**2 / var))


def _garch_params(theta: np.ndarray) -> tuple[float, float, float]:
    log_omega, a_raw, b_raw = theta
    total = 1.0 / (1.0 + np.exp(-a_raw))
    share = 1.0 / (1.0 + np.exp(-b_raw))
    return float(np.exp(log_omega)), float(total * share), float(total * (1.0 - share))


slot_series = (
    prices.sort(["symbol", "timestamp"])
    .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret"))
    .with_columns(pl.col("ret").rolling_std(VOL_WINDOW).over("symbol").alias(VOL_COL))
    .drop_nulls(subset=["ret", VOL_COL])
)
ANNUALISE = float(np.sqrt(SLOTS_PER_YEAR))

garch_rows, garch_params = [], []
for f in folds:
    for symbol in SYMBOLS:
        sym = slot_series.filter(pl.col("symbol") == symbol).sort("timestamp")
        dates = sym["timestamp"].to_list()
        ret = sym["ret"].to_numpy().astype(float)
        train_idx = [i for i, d in enumerate(dates) if f["train_start"] <= d.date() <= f["train_end"]]
        path_idx = [i for i, d in enumerate(dates) if f["train_start"] <= d.date() <= f["val_end"]]
        if len(train_idx) < MIN_TRAIN_SLOTS:
            continue
        eps_train = ret[train_idx] - ret[train_idx].mean()
        var0 = float(eps_train.var())
        result = minimize(
            _garch_nll,
            x0=np.array([np.log(var0 * 0.05), 2.2, -1.4]),
            args=(eps_train, var0),
            method="Nelder-Mead",
            options={"maxiter": GARCH_MAXITER, "xatol": 1e-8, "fatol": 1e-8},
        )
        omega, alpha, beta = _garch_params(result.x)
        mu = float(ret[train_idx].mean())
        eps_path = ret[path_idx] - mu
        var_path = _garch_filter(eps_path, omega, alpha, beta, var0)
        vol = np.sqrt(var_path)
        for j, i in enumerate(path_idx):
            if not (dates[i].date() <= f["train_end"] or dates[i].date() >= f["val_start"]):
                continue
            window = vol[max(0, j - GARCH_Z_WINDOW) : j + 1]
            centre, spread = float(window.mean()), float(window.std())
            garch_rows.append(
                {
                    "timestamp": dates[i],
                    "symbol": symbol,
                    "fold": f["fold"],
                    "garch_vol": float(vol[j] * ANNUALISE),
                    "garch_vol_z": float((vol[j] - centre) / spread) if spread > 0 else 0.0,
                    "garch_shock": float(eps_path[j] / vol[j]) if vol[j] > 0 else 0.0,
                }
            )
        garch_params.append(
            {
                "fold": f["fold"],
                "symbol": symbol,
                "omega": omega,
                "alpha": alpha,
                "beta": beta,
                "persistence": alpha + beta,
                "long_run_vol": float(np.sqrt(omega / max(1 - alpha - beta, 1e-12)) * ANNUALISE),
                "converged": bool(result.success),
                "train_slots": len(train_idx),
            }
        )
garch_df = pl.DataFrame(garch_rows)
display(pl.DataFrame(garch_params).sort(["fold", "symbol"]))
print(f"GARCH features: {len(garch_df):,} rows over {len(garch_params)} fold fits")
print(
    "Read `persistence` first: alpha + beta near one is the volatility clustering this instrument "
    "is known for, and `long_run_vol` is what the fit thinks the unconditional level is - a number "
    "that should be recognisable as Bitcoin's annualised volatility and not as anything else."
)

# %% [markdown]
# **The seal: the fit uses training rows only, and the filter is forward.** Deleting the second
# half of a fold's training observations must not move the first half's conditional variances,
# because the recursion at slot $t$ reads only slots up to $t$. A smoother would fail this.

# %%
_seal_sym = slot_series.filter(
    (pl.col("symbol") == SYMBOLS[0]) & (pl.col("timestamp").dt.date() < HOLDOUT_START)
).sort("timestamp")
_seal_ret = _seal_sym["ret"].to_numpy().astype(float)
_seal_idx = [
    i
    for i, d in enumerate(_seal_sym["timestamp"].to_list())
    if folds[0]["train_start"] <= d.date() <= folds[0]["train_end"]
]
_seal_eps = _seal_ret[_seal_idx] - _seal_ret[_seal_idx].mean()
_seal_var0 = float(_seal_eps.var())
_p = garch_params[0]
_full = _garch_filter(_seal_eps, _p["omega"], _p["alpha"], _p["beta"], _seal_var0)
_cut = len(_seal_eps) // 2
_half = _garch_filter(_seal_eps[:_cut], _p["omega"], _p["alpha"], _p["beta"], _seal_var0)
_drift = float(np.abs(_full[:_cut] - _half).max())
assert _drift == 0.0, f"the conditional variance moved by {_drift:.2e} - the filter is not forward"
_validation_rows = garch_df.filter(pl.col("fold").is_in(VALIDATION_FOLD_IDS))
assert _validation_rows["timestamp"].dt.date().max() < HOLDOUT_START
print(
    f"Deleting the last {len(_seal_eps) - _cut} of {len(_seal_eps)} training observations moves "
    f"the first {_cut} conditional variances by {_drift:.2e}"
)

# %% [markdown]
# ## 4. When the market is calm and when it is turbulent
#
# A two-state Gaussian HMM on the instrument's own decision-slot return and its trailing
# volatility, both scaled by 100 so the library's fixed covariance constants stay small relative to
# the data. The model is fitted on the training slots alone with ten restarts, the highest
# **stable** likelihood is kept (a restart whose final EM step falls is discarded, not averaged
# in), the two states are ordered by variance so "turbulent" means the same thing across folds, and
# the **filtered** probability - the one that reads only observations up to each slot - is run
# forward.
#
# Two columns: the filtered probability of the turbulent state, and its change over one slot, which
# is the transition rather than the level.

# %%
def fit_best_hmm(x_train: np.ndarray) -> tuple[GaussianHMM, float, int]:
    """The highest-likelihood stable training-only fit, over N_HMM_RESTARTS starting points."""
    best_ll, best_model, unstable = -np.inf, None, 0
    for seed in range(N_HMM_RESTARTS):
        try:
            with threadpool_limits(limits=1):
                model = GaussianHMM(
                    n_components=HMM_N_STATES,
                    covariance_type="full",
                    n_iter=100,
                    random_state=seed,
                    tol=1e-4,
                ).fit(x_train)
            history = list(model.monitor_.history)
            final_delta = history[-1] - history[-2] if len(history) >= 2 else 0.0
            scale = max(abs(history[-2]) if len(history) >= 2 else 1.0, 1.0)
            if final_delta < -HMM_STABILITY_REL_TOL * scale:
                unstable += 1
                continue
            score = model.score(x_train)
            if np.isfinite(score) and score > best_ll:
                best_ll, best_model = score, model
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("no stable HMM fit")
    return best_model, best_ll, unstable


def extract_hmm_features(fold: dict, symbol: str) -> tuple[list[dict], dict | None]:
    sym = slot_series.filter(pl.col("symbol") == symbol).sort("timestamp")
    dates = sym["timestamp"].to_list()
    arr = sym.select(["ret", VOL_COL]).to_numpy() * HMM_SCALE
    train_idx = [i for i, d in enumerate(dates) if fold["train_start"] <= d.date() <= fold["train_end"]]
    val_idx = [i for i, d in enumerate(dates) if fold["val_start"] <= d.date() <= fold["val_end"]]
    path_idx = [i for i, d in enumerate(dates) if fold["train_start"] <= d.date() <= fold["val_end"]]
    if len(train_idx) < MIN_TRAIN_SLOTS or len(val_idx) < MIN_VAL_SLOTS:
        return [], None
    model, score, unstable = fit_best_hmm(arr[train_idx])
    order = sort_states_by_variance(model)
    filtered = filtered_state_probs(model, arr[path_idx])
    path_dates = [dates[i] for i in path_idx]
    lag = int(SETUP["labels"]["rebalance_step"][PRIMARY_LABEL])
    rows = [
        {
            "timestamp": d,
            "symbol": symbol,
            "fold": fold["fold"],
            "hmm_regime_prob_high_vol": float(filtered[i, order[1]]),
            "hmm_regime_transition": (
                float(filtered[i, order[1]] - filtered[i - lag, order[1]]) if i >= lag else None
            ),
        }
        for i, d in enumerate(path_dates)
        if d.date() <= fold["train_end"] or d.date() >= fold["val_start"]
    ]
    transitions = model.transmat_[np.ix_(order, order)]
    return rows, {
        "fold": fold["fold"],
        "symbol": symbol,
        "persist_low_vol": float(transitions[0, 0]),
        "persist_high_vol": float(transitions[1, 1]),
        "log_likelihood": float(score),
        "unstable_restarts": unstable,
        "rows": len(rows),
    }


hmm_results, hmm_params = [], []
for fold in folds:
    for symbol in SYMBOLS:
        rows, params = extract_hmm_features(fold, symbol)
        hmm_results.extend(rows)
        if params is not None:
            hmm_params.append(params)
    print(f"  HMM fold {fold['fold']}: {sum(r['fold'] == fold['fold'] for r in hmm_results):,} rows")
hmm_df = pl.DataFrame(hmm_results)
display(pl.DataFrame(hmm_params).sort(["fold", "symbol"]))
print(
    f"HMM features: {len(hmm_df):,} rows; unstable restarts discarded: "
    f"{sum(p['unstable_restarts'] for p in hmm_params)}"
)
print(
    "The two persistence numbers are the diagonal of the transition matrix on the decision grid: "
    "the expected stay in each state is 1 / (1 - persistence) slots, so divide by "
    f"{SLOTS_PER_DAY} for days."
)

# %% [markdown]
# **The seal, again: filtered, not smoothed.**

# %%
_seal_arr = _seal_sym.select(["ret", VOL_COL]).to_numpy() * HMM_SCALE
_seal_model, _, _ = fit_best_hmm(_seal_arr[_seal_idx])
_obs = _seal_arr[_seal_idx]
_hcut = len(_obs) // 2
_hmm_drift = float(
    np.abs(
        filtered_state_probs(_seal_model, _obs)[:_hcut]
        - filtered_state_probs(_seal_model, _obs[:_hcut])
    ).max()
)
assert _hmm_drift < 1e-10, f"filtered probabilities moved by {_hmm_drift:.2e} - not filtered"
_hmm_validation = hmm_df.filter(pl.col("fold").is_in(VALIDATION_FOLD_IDS))
assert _hmm_validation["timestamp"].dt.date().max() < HOLDOUT_START
print(
    f"Regime checks hold; deleting the last {len(_obs) - _hcut} of {len(_obs)} observations moves "
    f"the first {_hcut} probabilities by {_hmm_drift:.2e}"
)

# %% [markdown]
# ## 5. Bring the two sets together and write the artifact
#
# The two models emit on slightly different slots - the HMM loses the volatility warmup and the
# transition column loses one more - so the join is an outer one on `(timestamp, symbol, fold)`
# and the coverage of each block is reported rather than assumed. Every **validation** row must
# carry every column: a null there is a slot a downstream model would silently train around.

# %%
temporal_df = garch_df.join(
    hmm_df, on=["timestamp", "symbol", "fold"], how="full", coalesce=True
).sort(["symbol", "timestamp", "fold"])
temporal_cols = [c for c in temporal_df.columns if c not in {"timestamp", "symbol", "fold"}]
assert temporal_df.select(pl.struct("timestamp", "symbol", "fold").is_duplicated().sum()).item() == 0
val_windows = pl.DataFrame(
    [{"fold": f["fold"], "val_start": f["val_start"], "val_end": f["val_end"]} for f in folds]
).with_columns(pl.col("val_start").cast(pl.Date), pl.col("val_end").cast(pl.Date))
validation_rows = temporal_df.join(val_windows, on="fold", how="inner").filter(
    pl.col("timestamp").dt.date().is_between(pl.col("val_start"), pl.col("val_end"), closed="both")
)
val_nulls = {
    c: validation_rows[c].null_count() for c in temporal_cols if validation_rows[c].null_count()
}
assert not val_nulls, f"a validation row is missing a feature value: {val_nulls}"
coverage = (
    temporal_df.group_by("fold")
    .agg(
        pl.len().alias("rows"),
        *[pl.col(c).is_not_null().mean().round(4).alias(c) for c in temporal_cols],
    )
    .sort("fold")
)
with pl.Config(tbl_rows=coverage.height, tbl_cols=len(temporal_cols) + 2):
    display(coverage)
print(f"Merged: {len(temporal_df):,} rows, {len(temporal_cols)} columns {temporal_cols}")
print(f"validation rows {len(validation_rows):,} with 0 nulls")

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
record = write_artifact(
    temporal_df,
    FEATURES_DIR / "model_based.parquet",
    keys=["timestamp", "symbol", "fold"],
    written_by="case_studies/exness_btc_8h/04_model_based_features.py",
    inputs={
        "decision_panel": value_digest(prices.select(["symbol", "timestamp", "close"])),
        f"labels:{PRIMARY_LABEL}": value_digest(label_frame),
    },
    metadata={
        "fold_geometry": [
            {
                "fold": f["fold"],
                **{n: str(f[n]) for n in ("train_start", "train_end", "val_start", "val_end")},
            }
            for f in folds
        ],
        "purge_gap_slots": {
            f"fold_{r['fold']}_{r['symbol']}": int(r["gap_slots"])
            for r in purge_gaps.iter_rows(named=True)
        },
        "holdout_fold": HOLDOUT_FOLD_ID,
        "models": (
            "GARCH(1,1) by Nelder-Mead MLE on the decision-slot return; GaussianHMM(2) on "
            f"[ret, {VOL_COL}] - both fitted on training slots only and filtered forward"
        ),
    },
)
print(f"Wrote features/model_based.parquet, shape {temporal_df.shape}, digest {record['digest']}")

# %% [markdown]
# ## Key takeaways
#
# - A fitted feature is point-in-time only if the fit never saw the rows it is applied to, and the
#   only way to know that is to fit inside the fold and prove the recursion is filtered.
# - **Filtered, not smoothed** is the whole difference, and it is one assertion per model. A
#   smoother produces better-looking features and no tradable ones.
# - **One ladder for every label.** A downstream model reads this artifact by fold id, so a variant
#   label whose validation window opened before the primary's training window closed would be
#   scored on slots its features had already seen - which is why the check is here and not two
#   stages later.
# - **Choose the models from a measurement.** The Kalman trend and the ARIMA that the sibling bots
#   fit are absent here because the eight-hour return has almost no linear structure to model and
#   because the same Kalman collapsed to a per-fold constant on `exness_fx_d1`; a column that
#   cannot say anything still costs a degree of freedom.
#
# **Next**: the evaluation, which is where any of this is tested against the labels for the first
# time - and the first stage of this bot that has to work around a one-name panel in every
# statistic it computes.
