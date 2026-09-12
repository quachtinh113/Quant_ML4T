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
# # Exness Gold Sessions (exness_gold_sess): Features Built From Fitted Models
#
# Chapter 8's features are arithmetic on past prices. This notebook builds a different kind: each
# column here is the output of a model whose parameters were themselves estimated from price
# history, so the window those parameters came from is part of what the feature knows. A feature
# like that is point-in-time only if the fit never saw the rows it is applied to, which is what the
# walk-forward geometry in section 2 is for and what the three seal checks prove.
#
# Three models, one per section, all on the **decision grid** - two slots a weekday, the same grid
# `02_labels` sealed the labels on and `03_financial_features` built its matrix on:
#
# | Model | What it splits out | Why it belongs on a metals session bot |
# |---|---|---|
# | Local linear trend (Kalman) | a slowly-moving level and the quoting noise around it | the level is where the metal "is"; the innovation is how far this session's close surprised it |
# | Two-state Gaussian HMM | a calm regime and a turbulent one, per metal | gold and silver differ enough in volatility that one regime model over both would describe silver and ignore gold |
# | ARIMA(1,0,1) | a one-step-ahead return forecast, and what it missed | the residual is a surprise measure, which is a different question from direction |
#
# **Learning objectives**
# - Derive the walk-forward folds from `setup.yaml` before anything is fitted, and measure the
#   purge gap in slots of the decision grid rather than in calendar days
# - Fit each estimator on a fold's training slots alone and run it **filtered** - not smoothed -
#   forward across the validation slots
# - Prove the absence of look-ahead by re-running each recursion on a truncated series and
#   checking the earlier values do not move
#
# **Book reference**: Chapter 9, Sections 9.2 (Kalman), 9.3 (ARIMA), 9.5 (HMM)
#
# **Prerequisites**: `02_labels.py` (its timestamps set the fold geometry) and the same MT5 H1
# history every other stage reads. Route B fork of
# [`exness_fx_d1/04_model_based_features`](../exness_fx_d1/04_model_based_features.ipynb) with two
# differences: the grid is the session decision grid, and there is no dollar factor - a signed
# average over two metals is not a factor, so the regime model is fitted **per metal** on its own
# return and volatility, the shape `xau_fx_mt5_d1` uses.
#
# **Output contract**: `features/model_based.parquet`, keys `timestamp, symbol, fold`. `fold`
# records which fit produced the row and is not itself a feature. Every value is computed from
# observations up to and including its own slot.

# %%
"""exness_gold_sess: Features Built From Fitted Models (Route B fork of exness_fx_d1)."""

import logging
import warnings

import numpy as np
import pandas as pd
import polars as pl
from hmmlearn.hmm import GaussianHMM
from IPython.display import display
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from threadpoolctl import threadpool_limits

from case_studies.exness_gold_sess._features import load_session_panel
from case_studies.research.holdout import build_holdout_cv
from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.cv_window import assert_variant_folds_are_out_of_sample
from case_studies.utils.temporal import filtered_state_probs, sort_states_by_variance
from utils.artifact_specs import load_setup_config, resolve_label_buffer
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

CASE_STUDY_ID = "exness_gold_sess"

# %% [markdown]
# Everything in the next cell can be overridden without editing the file, which is how a reader
# runs a smaller version first. Each one trades runtime for scope.

# %% tags=["parameters"]
MAX_SYMBOLS = 0
MAX_FOLDS = 0
KALMAN_MAXITER = 300
N_HMM_RESTARTS = 10
START_DATE = None

# %% [markdown]
# ## 1. Configuration and the price grid
#
# The panel is built by the one function every stage of this case study uses, so the slot a model
# is fitted on is the slot the label was sealed on. Two constants below are read from
# `setup.yaml` rather than typed: the Kalman trend is measured against the **middle** of the three
# declared moving-average windows (the shortest sits inside the filter's own responsiveness and
# the longest is slower than a validation year), and the regime model is given the shortest
# declared close-to-close volatility window.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
LABELS_DIR = CASE_DIR / "labels"
FEATURES_DIR = CASE_DIR / "features"
SETUP = load_setup_config(CASE_STUDY_ID)
ARIMA_ORDER = (1, 0, 1)
HMM_N_STATES = 2
HMM_STABILITY_REL_TOL = 1e-3
HMM_SCALE = 100.0
SLOTS_PER_DAY = len(SETUP["decision"]["snapshots"])
#: A year of decision slots. Every "enough training data" floor below is written in this unit, so
#: the numbers mean the same thing they do on a daily bot: one year of history.
#: Reads `decision.slots_per_year`, NOT `evaluation.periods_per_year`
#: (bots/exness_gold_sess/PRICE_GRID_DECLARATION.md section 5, 2026-09-08). The two used to be
#: the same key holding 504; they now say different things - 504 slots a year here, 252 DAYS a
#: year for the annualisation of the engine's return series. The value this line reads is
#: unchanged at 504, which is why the digest 4b1a0239330a0e7d is unchanged and was re-asserted
#: after this edit.
SLOTS_PER_YEAR = int(SETUP["decision"]["slots_per_year"])
MIN_TRAIN_SLOTS = SLOTS_PER_YEAR
MIN_VAL_SLOTS = 10
# The two windows below are DAILY windows in the feature register, and they are applied here to
# the DECISION grid, so the same number means half as many days. That is intentional and it is
# stated: the models in this notebook read the decision series, not the daily one, and a window
# named in days would have to be doubled to mean the same span - which is the kind of silent
# unit conversion the two-grid design exists to make visible.
KALMAN_TREND_WINDOW = int(sorted(SETUP["features"]["windows"]["moving_average"])[1])
VOL_WINDOW = int(min(SETUP["features"]["windows"]["close_to_close_volatility"]))
VOL_COL = f"slot_vol_{VOL_WINDOW}"
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)
_EVAL = load_evaluation_config(CASE_STUDY_ID)
HOLDOUT_START = pd.Timestamp(_EVAL["holdout_start"]).date()
HOLDOUT_END = pd.Timestamp(_EVAL["holdout_end"]).date()
HISTORY_START = START_DATE or str(SETUP["universe"]["history_start"])

print(f"Primary label {PRIMARY_LABEL}, purge buffer {LABEL_BUFFER}; holdout "
      f"{HOLDOUT_START} to {HOLDOUT_END}")
print(f"{SLOTS_PER_DAY} decision slots a weekday, {SLOTS_PER_YEAR} a year; a fold needs at least "
      f"{MIN_TRAIN_SLOTS} training slots and {MIN_VAL_SLOTS} validation slots per metal")
print(f"Kalman trend measured against a {KALMAN_TREND_WINDOW}-slot moving average; the regime "
      f"model reads a {VOL_WINDOW}-slot volatility")

# %%
prices = load_session_panel(
    SETUP,
    symbols=sorted(SETUP["universe"]["symbols"]),
    start_date=HISTORY_START,
    end_date=str(HOLDOUT_END),
    verbose=False,
)
SYMBOLS = sorted(SETUP["universe"]["symbols"])
assert len(SYMBOLS) == SETUP["universe"]["n_assets"]
assert set(SYMBOLS) <= set(prices["symbol"].unique().to_list())
if MAX_SYMBOLS:
    SYMBOLS = SYMBOLS[:MAX_SYMBOLS]
    prices = prices.filter(pl.col("symbol").is_in(SYMBOLS))
n_symbols = len(SYMBOLS)
prices = prices.sort(["symbol", "timestamp"])
all_slots = sorted(prices["timestamp"].unique().to_list())
print(f"Loaded {prices.height:,} rows, {n_symbols} metals, {len(all_slots):,} decision slots, "
      f"{all_slots[0]} to {all_slots[-1]}")

# %% [markdown]
# ## 2. The walk-forward folds, derived before anything is fitted
#
# The folds are laid out on **dates**, not on slots: a boundary falling between the London and the
# New York slot of one day would split a day in half for no reason and train the two sleeves of
# the book on different samples. That is the same call `01_feasibility_analysis` printed and
# `05_evaluation` will make again, so a fold id names the same window everywhere.
#
# A holdout fold is appended. It is not a choice: `build_holdout_cv` reads
# `evaluation.holdout_start/holdout_end` and trains on everything before them less one label
# buffer, so this stage can emit a feature vintage for the rows phase 7 will score **without any
# stage here reading a holdout outcome**. Nothing in this notebook computes a label or an
# information coefficient.

# %%
label_frame = pl.read_parquet(LABELS_DIR / f"{PRIMARY_LABEL}.parquet")
timeline = (
    label_frame.select(pl.col("timestamp").dt.date().alias("timestamp")).unique().sort("timestamp")
)
# The library defines the canonical fold as the fold derived from the label parquet's timeline
# (`case_studies/utils/cv_window.py:95-179`, `:164-167`), which silently treats that parquet as the
# **trading calendar**. That is a proxy, and it is only true while every decision the bot makes
# also resolves a label. On this grid it was not: US early closes and one Labor Day print a
# decision bar and no eight-hour endpoint, so `01_feasibility_analysis` (which builds its timeline
# from the panel, `01:578-580`) and this stage disagreed about fold 3's `train_start`. A-prime
# turns the proxy into an equality by publishing every label on the decision grid
# (`02_labels.py`, `bots/exness_gold_sess/FOLD_GEOMETRY_DECLARATION.md`), and this assertion is
# that equality placed exactly where it can break.
_panel_keys = set(zip(prices["timestamp"].to_list(), prices["symbol"].to_list(), strict=True))
_label_keys = set(zip(label_frame["timestamp"].to_list(), label_frame["symbol"].to_list(), strict=True))
assert _label_keys == _panel_keys, (
    f"the {PRIMARY_LABEL} parquet is not published on the decision grid: "
    f"{len(_panel_keys - _label_keys):,} panel keys are missing from it and "
    f"{len(_label_keys - _panel_keys):,} of its keys are not on the panel. The fold ladder this "
    "stage stamps would then be derived from the label's solvability rather than from the "
    "calendar the bot decides on"
)
assert set(timeline["timestamp"].to_list()) == {d.date() for d in prices["timestamp"].to_list()}, (
    "the fold timeline and the session panel disagree on which dates exist"
)
print(
    f"timeline {timeline.height:,} decision dates, derived from {len(_label_keys):,} "
    f"{PRIMARY_LABEL} keys that are exactly the {len(_panel_keys):,} keys of the session panel"
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
    "the holdout fold trains through the boundary, so the last training label resolves inside "
    "the window it is meant to be judged against"
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
# ### One ladder, three labels
#
# This artifact carries **one** fold set, built above on the primary label's geometry, and a model
# trained on a variant label reads it by `fold` id. So the values it reads for fold F were fitted
# through the **primary** label's `train_end`, and the variant's own validation window has to open
# after that or the model is scored on sessions its features already saw. The check that says so
# is `assert_variant_folds_are_out_of_sample`, and the four sibling case studies with a stage 04
# all call it here (`fx_pairs/04:534`, `exness_fx_d1/04:528`, `xau_fx_mt5/04:507`, `etfs/04:390`);
# this fork had dropped the call, which is why the defect surfaced two stages later at
# `06_linear`'s `request.resolve()` instead of here.
#
# After A-prime the gap is not merely positive: it is **the same gap on all eight rows**, because
# the three labels sit on one timeline and therefore on one ladder. A test that only asked for
# `gap > 0` would pass on a geometry in which each label still drifted.

# %%
variant_gaps = pl.DataFrame(assert_variant_folds_are_out_of_sample(CASE_STUDY_ID, PRIMARY_LABEL))
if variant_gaps.is_empty():
    # Generation 2 (bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md section 2):
    # `labels.variants` is empty, so there is no variant ladder to compare with the primary's.
    # Stated rather than skipped: the one ladder this artifact stamps IS the primary label's
    # own, and no model of another label reads it by fold id. The assertion below is what runs
    # the day a variant is declared again. (Guard added 2026-09-10 for the zero-variant case;
    # it reads no data and moves no value, so the artifact digest cannot depend on it.)
    print(f"0 variant labels declared: the fold ladder is {PRIMARY_LABEL}'s own and no variant "
          "reads it by fold id")
else:
    with pl.Config(tbl_rows=variant_gaps.height):
        display(variant_gaps.sort(["label", "fold"]))
    _gaps = set(variant_gaps["gap"].to_list())
    assert len(_gaps) == 1, (
        "the configured labels do not sit on one fold ladder: the gap between the primary's "
        f"train_end and a variant's val_start varies -> {sorted(_gaps)}"
    )
    print(
        f"{variant_gaps.height} (variant, fold) rows, every gap {variant_gaps['gap'].min()} - one "
        f"ladder, and that single value is {PRIMARY_LABEL}'s own purge"
    )

# %% [markdown]
# ### The purge gap, measured in slots of this grid
#
# `labels.buffer` is declared in sessions of the evaluation calendar. What matters on this grid is
# how many **decision slots** sit between the last training slot and the first validation slot on
# each metal's own series: it must be at least the label horizon - one slot for `fwd_ret_8h` -
# plus one of embargo, or the last training row's outcome would be known inside the validation
# window.

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
print(f"purge gap {purge_gaps['gap_slots'].min()}-{purge_gaps['gap_slots'].max()} decision slots "
      f"(>= {MIN_PURGE_SLOTS} = a {SETUP['labels']['rebalance_step'][PRIMARY_LABEL]}-slot horizon "
      f"+ 1 embargo), {purge_gaps['calendar_days'].min()}-{purge_gaps['calendar_days'].max()} "
      "calendar days")

# %% [markdown]
# ## 3. Where the price level is, and how fast it is moving
#
# A local linear trend model treats the level as hidden and each observed close as a noisy reading
# of it. Three variances decide how much it believes each: the observation noise, the level noise
# and the slope noise. They are fitted by maximum likelihood **on the training slots alone**, over
# their logarithms so the positivity constraint is removed rather than enforced, and the recursion
# then runs forward across training and validation together without re-estimating - because a
# recursion carries state, and restarting it at the first validation slot would throw away
# everything it had learned about where the level was.

# %%
def kalman_local_linear(
    prices_arr: np.ndarray,
    observation_noise: float = 0.01,
    level_noise: float = 0.001,
    slope_noise: float = 0.001,
) -> dict[str, np.ndarray]:
    """Local linear trend Kalman filter: level, slope, innovation, uncertainty, log-likelihood."""
    n = len(prices_arr)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[level_noise, 0.0], [0.0, slope_noise]])
    R = np.array([[observation_noise]])
    x = np.array([prices_arr[0], 0.0])
    P = np.eye(2) * 10.0
    levels, slopes = np.zeros(n), np.zeros(n)
    innovations, uncertainties = np.zeros(n), np.zeros(n)
    log_lik = 0.0
    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        y = prices_arr[t] - H @ x_pred
        S = H @ P_pred @ H.T + R
        log_lik += -0.5 * (np.log(2 * np.pi * S[0, 0]) + y[0] ** 2 / S[0, 0])
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred
        levels[t], slopes[t] = x[0], x[1]
        innovations[t], uncertainties[t] = y[0], P[0, 0]
    return {
        "level": levels,
        "slope": slopes,
        "innovation": innovations,
        "uncertainty": uncertainties,
        "log_likelihood": log_lik,
    }


def fit_kalman_mle(train_prices: np.ndarray, maxiter: int = 300) -> tuple[float, float, float]:
    """Estimate the three noise sizes on the training slots. Started from their own scale."""

    def negative_log_likelihood(params: np.ndarray) -> float:
        return -kalman_local_linear(train_prices, *np.exp(params))["log_likelihood"]

    variance = max(float(np.var(np.diff(train_prices))), 1e-10)
    x0 = np.log([variance * 0.5, variance * 0.1, variance * 0.01])
    opt = minimize(negative_log_likelihood, x0, method="Nelder-Mead", options={"maxiter": maxiter})
    return tuple(np.exp(opt.x))


# %% [markdown]
# Four columns come out of it. `kalman_trend` is how far the fitted level sits from a moving
# average of the log price; `kalman_slope` is the drift the model currently believes in;
# `kalman_slope_zscore` puts that drift on the scale of its own **training-window** spread;
# `kalman_innovation` is the gap between the observed close and what the model expected before
# seeing it.
#
# **A fifth column, `kalman_smoothness`, was REMOVED on 2026-09-08**
# (`bots/exness_gold_sess/RULINGS_2026-09-08.md` ruling 2.1), and the reason is computable from
# this stage alone, with no label anywhere near it. It was `1.0 / (uncertainty + 1e-10)`, where
# `uncertainty` is `P[0,0]` of the filter. In a linear-Gaussian Kalman filter the covariance
# recursion `P_pred = F P Fᵀ + Q ; S = H P_pred Hᵀ + R ; K = P_pred Hᵀ S⁻¹ ; P = (I − K H) P_pred`
# **has no price in it**: `P` depends only on `F, H, Q, R` and converges geometrically to the
# steady state of the Riccati equation. So the column reports where Nelder-Mead stopped, not
# anything about the metal - and `observation_noise` is unidentified here (the likelihood is flat;
# the MLE lands between 1e-18 and 1e-72 and moves by a factor of 288 for a two-row change in the
# training window).
#
# Measured on the sealed artifact before removing it, so the record carries the fact and not the
# theory: every one of the 24,740 values lay in `1e10 × (1 − 5.55e-6, 1]`, the coefficient of
# variation inside a (fold, symbol) cell was between 2e-11 and 1.2e-7, and **22.0 % of rows were
# exactly `1e10`** - which is `1/1e-10`, the epsilon in the denominator above, i.e. the column was
# reporting the guard constant rather than the filter. `exness_fx_d1` recorded the same collapse
# independently (`bots/exness_fx_d1/BOT.md:185`, R ≈ 1e-17, "a per-fold constant 1e10").
#
# The removal is not a number: written beside the decision so a later reader can check that -
# `05`'s verdict on this column was **PROCEED**, its pooled panel IC **−0.00674** (HAC t −0.300),
# **rank 37 of 43** by |IC|. **If it had ranked first it would still have been removed.**
#
# The path's first slot is walked through and not emitted. A recursion starts somewhere, and it
# starts at the first observed price with a slope of zero, so on that slot the forecast equals the
# observation by construction: the innovation is identically zero and the slope is the zero it was
# initialised to. Those are the prior showing through, not a session on which the model happened to
# be right. Slots inside the purge gap are walked through and emitted to neither split.

# %%
def extract_kalman_features(fold: dict, symbol: str) -> tuple[list[dict], dict | None]:
    sym = prices.filter(pl.col("symbol") == symbol).sort("timestamp")
    dates = sym["timestamp"].to_list()
    log_prices = np.log(sym["close"].to_numpy())
    train_mask = [fold["train_start"] <= d.date() <= fold["train_end"] for d in dates]
    val_mask = [fold["val_start"] <= d.date() <= fold["val_end"] for d in dates]
    path_mask = [fold["train_start"] <= d.date() <= fold["val_end"] for d in dates]
    if sum(train_mask) < MIN_TRAIN_SLOTS or sum(val_mask) < MIN_VAL_SLOTS:
        return [], None
    path_dates = [d for d, keep in zip(dates, path_mask, strict=True) if keep]
    path_prices = log_prices[path_mask]
    opt = fit_kalman_mle(log_prices[train_mask], maxiter=KALMAN_MAXITER)
    filtered = kalman_local_linear(path_prices, *opt)
    train_idx = np.array([d.date() <= fold["train_end"] for d in path_dates])
    slope_mean = float(np.mean(filtered["slope"][train_idx]))
    slope_std = float(np.std(filtered["slope"][train_idx])) + 1e-10
    moving_average = (
        pl.Series(path_prices).rolling_mean(KALMAN_TREND_WINDOW, min_samples=1).to_numpy()
    )
    rows = [
        {
            "timestamp": path_dates[i],
            "symbol": symbol,
            "fold": fold["fold"],
            "kalman_trend": float(filtered["level"][i] - moving_average[i]),
            "kalman_slope": float(filtered["slope"][i]),
            "kalman_slope_zscore": float((filtered["slope"][i] - slope_mean) / slope_std),
            "kalman_innovation": float(filtered["innovation"][i]),
            # kalman_smoothness removed 2026-09-08 (ruling 2.1). `filtered["uncertainty"]` is
            # still computed and returned by the filter - deleting an OUTPUT column must not
            # touch the fit, and the acceptance signature says every remaining value has to
            # reproduce bit for bit.
        }
        for i in range(len(path_dates))
        if i > 0 and (path_dates[i].date() <= fold["train_end"] or path_dates[i].date() >= fold["val_start"])
    ]
    params = {
        "fold": fold["fold"],
        "symbol": symbol,
        "observation_noise": float(opt[0]),
        "level_noise": float(opt[1]),
        "slope_noise": float(opt[2]),
        "train_slots": int(sum(train_mask)),
    }
    return rows, params


kalman_results, kalman_params = [], []
for fold in folds:
    for symbol in SYMBOLS:
        rows, params = extract_kalman_features(fold, symbol)
        kalman_results.extend(rows)
        if params is not None:
            kalman_params.append(params)
    print(f"  Kalman fold {fold['fold']}: "
          f"{sum(r['fold'] == fold['fold'] for r in kalman_results):,} rows")
kalman_df = pl.DataFrame(kalman_results)
display(pl.DataFrame(kalman_params).sort(["fold", "symbol"]))
print(f"Kalman features: {len(kalman_df):,} rows over {n_symbols} metals x {len(folds)} folds")

# %% [markdown]
# **The seal.** Deleting the second half of a fold's path must not move the first half's filtered
# values. A smoother would fail this; a filter cannot. It is run on the pre-holdout series, because
# a test that read held-back slots to prove they are held back would report on a series the
# validation folds may not see.

# %%
for fold in folds:
    rows = kalman_df.filter(pl.col("fold") == fold["fold"])
    assert rows["timestamp"].dt.date().min() >= fold["train_start"], fold["fold"]
    assert rows["timestamp"].dt.date().max() <= fold["val_end"], fold["fold"]
    inside_gap = rows.filter(
        (pl.col("timestamp").dt.date() > fold["train_end"])
        & (pl.col("timestamp").dt.date() < fold["val_start"])
    )
    assert inside_gap.is_empty(), f"fold {fold['fold']}: a row inside the purge gap"
_validation_rows = kalman_df.filter(pl.col("fold").is_in(VALIDATION_FOLD_IDS))
assert _validation_rows["timestamp"].dt.date().max() < HOLDOUT_START, (
    "the Kalman filter emitted a holdout-dated row on a validation fold"
)
_seal = prices.filter(
    (pl.col("symbol") == SYMBOLS[0]) & (pl.col("timestamp").dt.date() < HOLDOUT_START)
).sort("timestamp")
_seal_prices = np.log(_seal["close"].to_numpy())
_seal_train = _seal_prices[
    [folds[0]["train_start"] <= d.date() <= folds[0]["train_end"] for d in _seal["timestamp"]]
]
_opt = fit_kalman_mle(_seal_train, maxiter=KALMAN_MAXITER)
_cut = len(_seal_train) // 2
_drift = float(
    np.abs(
        kalman_local_linear(_seal_train, *_opt)["level"][:_cut]
        - kalman_local_linear(_seal_train[:_cut], *_opt)["level"]
    ).max()
)
assert _drift < 1e-10, f"the filtered level moved by {_drift:.2e} - that is a smoother"
print(f"Kalman checks hold on {len(folds)} folds x {n_symbols} metals; deleting the last "
      f"{len(_seal_train) - _cut} of {len(_seal_train)} training slots moves the first {_cut} "
      f"levels by {_drift:.2e}")

# %% [markdown]
# ## 4. When each metal is calm and when it is turbulent
#
# A two-state Gaussian HMM on the metal's own decision-slot return and its trailing volatility,
# both scaled by 100 so the library's fixed covariance constants stay small relative to the data.
# The model is fitted on the training slots alone with ten restarts, the highest **stable**
# likelihood is kept (a restart whose final EM step falls is discarded, not averaged in), the two
# states are ordered by variance so "turbulent" means the same thing across folds, and the
# **filtered** probability - the one that reads only observations up to each slot - is run forward.
#
# Per metal, not pooled. `exness_fx_d1` fits one regime model on a dollar factor because its five
# pairs share one; two metals do not make a factor, and silver's volatility is roughly twice
# gold's, so a pooled fit would describe silver and call gold permanently calm.

# %%
slot_series = (
    prices.sort(["symbol", "timestamp"])
    .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret"))
    .with_columns(pl.col("ret").rolling_std(VOL_WINDOW).over("symbol").alias(VOL_COL))
    .drop_nulls(subset=["ret", VOL_COL])
)


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
    print(f"  HMM fold {fold['fold']}: "
          f"{sum(r['fold'] == fold['fold'] for r in hmm_results):,} rows")
hmm_df = pl.DataFrame(hmm_results)
display(pl.DataFrame(hmm_params).sort(["fold", "symbol"]))
print(f"HMM features: {len(hmm_df):,} rows; unstable restarts discarded: "
      f"{sum(p['unstable_restarts'] for p in hmm_params)}")

# %% [markdown]
# **The seal: filtered, not smoothed.** Deleting the second half of a fold's training observations
# must not move the first half's probabilities. A smoothed posterior would.

# %%
_seal_sym = slot_series.filter(
    (pl.col("symbol") == SYMBOLS[0]) & (pl.col("timestamp").dt.date() < HOLDOUT_START)
).sort("timestamp")
_seal_arr = _seal_sym.select(["ret", VOL_COL]).to_numpy() * HMM_SCALE
_seal_train_idx = [
    i
    for i, d in enumerate(_seal_sym["timestamp"].to_list())
    if folds[0]["train_start"] <= d.date() <= folds[0]["train_end"]
]
_seal_model, _, _ = fit_best_hmm(_seal_arr[_seal_train_idx])
_obs = _seal_arr[_seal_train_idx]
_cut = len(_obs) // 2
_hmm_drift = float(
    np.abs(
        filtered_state_probs(_seal_model, _obs)[:_cut]
        - filtered_state_probs(_seal_model, _obs[:_cut])
    ).max()
)
assert _hmm_drift < 1e-10, f"filtered probabilities moved by {_hmm_drift:.2e} - not filtered"
_validation_rows = hmm_df.filter(pl.col("fold").is_in(VALIDATION_FOLD_IDS))
assert _validation_rows["timestamp"].dt.date().max() < HOLDOUT_START
print(f"Regime checks hold; deleting the last {len(_obs) - _cut} of {len(_obs)} observations "
      f"moves the first {_cut} probabilities by {_hmm_drift:.2e}")

# %% [markdown]
# ## 5. What the return model did not see coming
#
# An ARIMA(1,0,1) on the decision-slot returns, fitted on the training slots and **extended** over
# the path with `apply(..., refit=False)`. The forecast itself is one column; the more interesting
# two are what it missed - the residual, and the residual on the scale of the training window's own
# residual spread, which is a surprise measure rather than a direction.

# %%
def extract_arima_features(fold: dict, symbol: str) -> list[dict]:
    sym = prices.filter(pl.col("symbol") == symbol).sort("timestamp")
    dates = sym["timestamp"].to_list()
    close = sym["close"].to_numpy()
    rets = np.diff(close) / close[:-1]
    ret_dates = dates[1:]
    train_mask = [fold["train_start"] <= d.date() <= fold["train_end"] for d in ret_dates]
    val_mask = [fold["val_start"] <= d.date() <= fold["val_end"] for d in ret_dates]
    path_mask = [fold["train_start"] <= d.date() <= fold["val_end"] for d in ret_dates]
    if sum(train_mask) < MIN_TRAIN_SLOTS or sum(val_mask) < MIN_VAL_SLOTS:
        return []
    path_dates = [d for d, keep in zip(ret_dates, path_mask, strict=True) if keep]
    path_rets = rets[path_mask]
    fit = ARIMA(rets[train_mask], order=ARIMA_ORDER).fit()
    resid_std = float(np.std(fit.resid)) + 1e-10
    predicted = fit.apply(path_rets, refit=False).predict(start=0, end=len(path_rets) - 1)
    return [
        {
            "timestamp": d,
            "symbol": symbol,
            "fold": fold["fold"],
            "arima_forecast": float(predicted[i]),
            "arima_residual": float(path_rets[i] - predicted[i]),
            "arima_residual_zscore": float((path_rets[i] - predicted[i]) / resid_std),
        }
        for i, d in enumerate(path_dates)
        if d.date() <= fold["train_end"] or d.date() >= fold["val_start"]
    ]


arima_results = []
for fold in folds:
    for symbol in SYMBOLS:
        arima_results.extend(extract_arima_features(fold, symbol))
    print(f"  ARIMA fold {fold['fold']}: "
          f"{sum(r['fold'] == fold['fold'] for r in arima_results):,} rows")
arima_df = pl.DataFrame(arima_results)
print(f"ARIMA features: {len(arima_df):,} rows")

# %% [markdown]
# **Two seals here, because truncation alone would not catch a re-fit.** A model re-estimated on
# the longer series is still a forward pass over it, so the parameter vectors are compared element
# by element as well as the forecasts.

# %%
_seal = prices.filter(
    (pl.col("symbol") == SYMBOLS[0]) & (pl.col("timestamp").dt.date() < HOLDOUT_START)
).sort("timestamp")
_close = _seal["close"].to_numpy()
_rets = np.diff(_close) / _close[:-1]
_ret_dates = _seal["timestamp"].to_list()[1:]
_train = _rets[[folds[0]["train_start"] <= d.date() <= folds[0]["train_end"] for d in _ret_dates]]
_path = _rets[[folds[0]["train_start"] <= d.date() <= folds[0]["val_end"] for d in _ret_dates]]
_fit = ARIMA(_train, order=ARIMA_ORDER).fit()
_applied = _fit.apply(_path, refit=False)
_param_drift = float(np.abs(np.asarray(_fit.params) - np.asarray(_applied.params)).max())
assert _param_drift == 0.0, f"apply() re-estimated: parameters moved by {_param_drift:.2e}"
_cut = len(_path) // 2
_full = np.asarray(_applied.predict(start=0, end=len(_path) - 1))
_prefix = np.asarray(_fit.apply(_path[:_cut], refit=False).predict(start=0, end=_cut - 1))
_arima_drift = float(np.abs(_full[:_cut] - _prefix).max())
assert _arima_drift < 1e-10, f"forecasts moved by {_arima_drift:.2e} - not a forward pass"
_validation_rows = arima_df.filter(pl.col("fold").is_in(VALIDATION_FOLD_IDS))
assert _validation_rows["timestamp"].dt.date().max() < HOLDOUT_START
print(f"Return-model checks hold; extending {SYMBOLS[0]} fold {folds[0]['fold']} from "
      f"{len(_train)} to {len(_path)} observations moves the parameters by {_param_drift:.2e}, "
      f"and deleting the last {len(_path) - _cut} moves the first {_cut} forecasts by "
      f"{_arima_drift:.2e}")

# %% [markdown]
# ## 6. Bring the three sets together and write the artifact
#
# The three models emit on slightly different slots - the Kalman filter drops each path's first
# slot, ARIMA loses one to differencing, the HMM loses the volatility warmup - so the join is an
# outer one on `(timestamp, symbol, fold)` and the coverage of each block is reported rather than
# assumed. Every **validation** row must carry every column: a null there is a slot a downstream
# model would silently train around.

# %%
temporal_df = (
    kalman_df.join(hmm_df, on=["timestamp", "symbol", "fold"], how="full", coalesce=True)
    .join(arima_df, on=["timestamp", "symbol", "fold"], how="full", coalesce=True)
    .sort(["symbol", "timestamp", "fold"])
)
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
    written_by="case_studies/exness_gold_sess/04_model_based_features.py",
    inputs={
        "session_panel": value_digest(prices),
        f"labels:{PRIMARY_LABEL}": value_digest(label_frame),
    },
    metadata={
        "fold_geometry": [
            {
                "fold": f["fold"],
                **{
                    n: str(f[n]) for n in ("train_start", "train_end", "val_start", "val_end")
                },
            }
            for f in folds
        ],
        "purge_gap_slots": {
            f"fold_{r['fold']}_{r['symbol']}": int(r["gap_slots"])
            for r in purge_gaps.iter_rows(named=True)
        },
        "holdout_fold": HOLDOUT_FOLD_ID,
        "models": (
            "Kalman local linear trend on log close; GaussianHMM(2) per metal on "
            f"[ret, {VOL_COL}]; ARIMA{ARIMA_ORDER} on slot returns - all on the decision grid"
        ),
    },
)
print(f"Wrote features/model_based.parquet, shape {temporal_df.shape}, digest {record['digest']}")

# %% [markdown]
# ## Key takeaways
#
# - A fitted feature is point-in-time only if the fit never saw the rows it is applied to, and the
#   only way to know that is to fit inside the fold and prove the recursion is filtered.
# - **Filtered, not smoothed** is the whole difference, and it is one line in each of the three
#   sections and one assertion. A smoother produces better-looking features and no tradable ones.
# - The purge gap is measured in slots of **this** grid. A buffer declared in sessions of a
#   calendar means something different on a grid with two decisions a weekday, and the number that
#   matters is the one measured on the series the model actually reads.
# - Two metals are not a factor. The regime model is fitted per metal, and saying why is what
#   stops the next reader from "simplifying" it into one pooled fit.
# - This stage computes no label and no information coefficient. `05_evaluation` scores every
#   column, on validation rows only.
