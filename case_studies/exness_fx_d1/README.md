# Case Study: exness_fx_d1 (Exness MT5, daily cross-sectional FX)

Research side of the bot `exness_fx_d1` (contract and memory: [`bots/exness_fx_d1/BOT.md`](../../bots/exness_fx_d1/BOT.md);
instrument profiles: [`bots/assets/`](../../bots/assets/README.md)). A Route B fork of
[`case_studies/fx_pairs`](../fx_pairs/README.md): the same pipeline, the same three labels and the same model presets,
with three things changed - the data source (MetaTrader 5 history through
[`bots/_shared/mt5_loader.py`](../../bots/_shared/mt5_loader.py) instead of OANDA), the decision bar (the last MT5
H4 bar closing before the New York rollover, 20:00 UTC on this UTC+0 server, instead of an OANDA bar aligned to the
rollover) and the universe (five dollar pairs instead of twenty G10 pairs).

Hypothesis, kill criteria and the measured cost table are owned by `BOT.md`; this README only describes the pipeline.
Both are TODO for the user at the time of writing.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | FX spot CFDs on Exness MT5 (`EURUSD` `GBPUSD` `USDJPY` `AUDUSD` `USDCAD`, bare names; Market Watch suffix handled in the loader) |
| Frequency | Daily, one decision per `CME_FX` session at the close of the last MT5 H4 bar before the New York rollover (`decision.snapshot: h4_close_2000utc`; the server D1 bar failed the session checks on 2026-09-05, see `bots/exness_fx_d1/BOT.md`) |
| Universe | 5 dollar pairs; `top_k_grid: [1, 2]` per label |
| History | Full H4 grid from 2017-03-01 on all five pairs (`universe.history_start`; before that the server serves one H4 bar a day, D1 backfill); D1 from 2014-01-14, H1 from 2022-10-25 (`history_depth.json`, Exness-MT5Trial7, 2026-09-05) |
| Primary Label | `fwd_ret_1d` (variants `fwd_ret_5d`, `fwd_ret_21d`), unchanged from `fx_pairs` |
| CV Folds | 4 x 3Y train / 1Y val, holdout 2025-09-01 to 2026-08-31 (declared 2026-09-05 from the measured H4 history depth, before any training run) |
| Cost Model | Material: spread + swap (Pro account, no commission); spreads measured 2026-09-05 from 30 days of this account's ticks by session (`bots/_shared/costs_mt5.py`) |

## Data

`bots/_shared/mt5_loader.py:download_mt5_bars` reads D1 / H4 / H1 bars from a logged-in terminal and writes
`ML4T_DATA_PATH/mt5/{daily,4h,1h}.parquet` plus `history_depth.json` (history depth, contract size, volume limits,
swaps, triple-swap day, and the measured server clock). `load_mt5_bars(frequency, ...)` returns the same shape as
`data/fx/loader.py:load_fx_pairs`. The stages read the H4 bars (UTC bar open through `decision.server_clock`),
assign them to `CME_FX` sessions and take as each session's close the last bar that closes at or before the
calendar close, within `decision.session_close_tolerance_minutes` (one bar). The server D1 bar is downloaded but
not used as the decision bar: on this UTC+0 server it closes 2-3 hours after the rollover and prints a Sunday bar.
`bots/_shared/costs_mt5.py:measure_spreads` writes `mt5/spreads_by_session.{parquet,json}` from tick history.

## Pipeline

Every stage from `01` to `19` except the deep-learning and causal ones is in place, copied from `fx_pairs`: `01`-`05`
change the loader and the clock, `06`, `07` and `12`-`19` change the study id, the population lineage and the prose,
and nothing else. **`14`-`19` were copied on 2026-09-07 and have not been run** - phase 5 has no survivor at the
cumulative trial count, and `16_costs` additionally refuses to price a swap the real Pro account has not reported
(`BOT.md` phases 5-7, user decision 5). Their `SUPERSEDES_*` declarations are empty because this bot's registry holds
no generation of those populations yet; a copied `fx_pairs` hash would claim to supersede a snapshot that does not
exist here and the first run would be refused. The candidate set `exness_fx_d1:holdout-candidates` that `17`, `18` and
`19` resolve is written by **`15_risk_management`**, so the order `13` -> `14` -> `15` -> `16` -> `17`/`18`/`19` has no
short cut.

The session panel and the feature construction live in one importable module, [`_features.py`](_features.py), so the
notebooks, the backtest price loader (`case_studies/utils/backtest_loaders.py`, loader `exness_fx_d1`), the point-in-time
test (`bots/exness_fx_d1/tests/test_lookahead.py`) and the deployment loop compute the same numbers.

Three read-only helpers sit beside the stages. None is a stage, none writes to the registry, and none costs a trial:
[`_report_phase5.py`](_report_phase5.py) is the phase-5 number report (populations, generations, Sharpe distributions,
benchmark, exposure gate, Deflated Sharpe at the cumulative trial count);
[`_diagnose_timeseries_signal.py`](_diagnose_timeseries_signal.py) measures the activation and transition rate of a
signal rule on registered prediction sets **before** a sweep is opened on it (Ch16 `08_signal_method_comparison`);
[`_check_carry_series.py`](_check_carry_series.py) runs the four acceptance checks on the FRED/ALFRED carry legs of
[`bots/_shared/macro_config.yaml`](../../bots/_shared/macro_config.yaml) before any macro history is fetched.

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Server-clock alignment with the session calendar, universe breadth at the decision bar, independent bets, assumed round-trip cost per pair, move-to-cost by horizon, return persistence, history depth per symbol, walk-forward folds | none |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 1-, 5- and 21-session forward returns on the session close (last H4 bar before the rollover) | `labels/fwd_ret_{1d,5d,21d}.parquet` + digest sidecars |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | The `fx_pairs` register (momentum, risk-adjusted momentum, mean reversion, volatility, range and drawdown, oscillators, dollar factor, cross-sectional ranks) plus a gold factor from `XAUUSD` on the same decision bar; warmup audit, holdout-withholding seal and a rebuild of the panel from the bars closed at sampled decision instants. No carry (see Known limitations in the notebook), no oil (no instrument downloaded), no session flags (constant at a fixed daily snapshot) | `features/financial.parquet` + digest sidecar |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Walk-forward Kalman level/slope, dollar-regime HMM, ARIMA surprise, per fold plus the holdout fold; truncation seals | `features/model_based.parquet` + digest sidecar |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7-9 | Feature-label IC triage across the five pairs on the four validation folds, Newey-West and BH-FDR, fold agreement, horizon profile, redundancy groups | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | The `fx_pairs` penalty grid (OLS, 11 Ridge, 8 Lasso, 8 ElasticNet) on all three labels; menu `config/training/<label>.yaml`, presets `case_studies/config/{ols,ridge,lasso,elastic_net}/` | Training runs and prediction sets in the experiment's `run_log/registry.db`, population `exness_fx_d1-linear-validation-v1` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | The `fx_pairs` LightGBM grid (5 capacities x 3 objectives, 10 checkpoints each) on the CPU; `modeling.gbm.num_threads` declared in the experiment's `setup.yaml` for the WSL host | Training runs, prediction sets per checkpoint, boosters and fold metrics; population `exness_fx_d1-gbm-validation-v1` |
| Tabular / sequence DL | `08_tabular_dl` ... `10a_dl_lstm` (to copy) | Ch12-13 | Declared in the menus, no stage yet; `12` names them as declared exclusions | -- |
| Causal | `11_causal_dml` (to copy) | Ch15 | DML on `mom_skip_recent` | `causal_runs` row |
| Model Analysis | [`12_model_analysis`](12_model_analysis.ipynb) | -- | Population-vs-menu check, cross-family IC, fold stability, agreement, buckets, conformal coverage (embargo entries in `case_studies/utils/conformal.py`) | nothing |
| Backtest | [`13_backtest`](13_backtest.ipynb) | Ch16 | Two signal families over the same complete validation prediction sets (linear and GBM, every checkpoint), selected by the `SIGNAL_FAMILY` parameter and each publishing its own population: **`cross_sectional_top_k`** = long-short top-1 / top-2 equal-weight sleeves, and **`per_pair_timeseries`** = the Ch16 `08_signal_method_comparison` per-pair rolling-percentile book (`signals.py::per_symbol_rolling_percentile_signal`, `bars_per_day: 1`, `long_q` and `lookback_days` from `setup.yaml::backtest.sweep.timeseries_percentile_grid`, long only because the engine holds a cash account). Both run through the shared engine on the MT5 session panel (`case_studies/utils/backtest_loaders.py`, loader `exness_fx_d1`: the `next_bar_open` fill is the open of the bar starting at the decision instant, the mark is the decision bar's close); costs from `setup.yaml` (`commission_per_lot: 0`, spread 1.3 bps per crossing, swap 0.0 on the demo); freezes the prediction and baseline populations; the total membership of every generation of every population is the DSR trial count in `BOT.md` | `backtest_runs`, `backtest_metrics`, populations `exness_fx_d1:validation-predictions`, `exness_fx_d1:equal-weight-baselines`, `exness_fx_d1:timeseries-percentile-baselines` |
| Portfolio | [`14_portfolio_management`](14_portfolio_management.ipynb) | Ch17 | Allocators (score-weighted, inverse-vol, risk parity, MVO-LW, HRP, conformal) over the top-10 baseline configurations per label. **Copied 2026-09-07, not run** | `backtest_runs`, population `exness_fx_d1:allocation-backtests`, candidate sets `exness_fx_d1:<label>:equal-weight-candidates` |
| Risk | [`15_risk_management`](15_risk_management.ipynb) | Ch19 | The declared stop-loss / trailing / time-exit overlays on one strategy per label; also writes the candidate set `exness_fx_d1:holdout-candidates` that `17`-`19` resolve, so it is on the critical path to the holdout. **Copied 2026-09-07, not run** | `backtest_runs`, population `exness_fx_d1:risk-backtests`, candidate sets |
| Costs | [`16_costs`](16_costs.ipynb) | Ch18 | One cost-sensitivity curve per label over `backtest.sweep.cost_grid_bps`, plus a guard that refuses to price a swap the account has not reported (`costs.swap.measured_on_account` must be `real`) or a non-zero swap the percentage model cannot charge per crossing. Breakeven is read against the p90 of the measured in-session spread. **Copied 2026-09-07, not run**: blocked on the phase-5 gate and on the real Pro-account readings (user decision 5) | `backtest_runs`, population `exness_fx_d1:cost-backtests` |
| Holdout | [`17_holdout_predictions`](17_holdout_predictions.ipynb), [`18_holdout_backtest`](18_holdout_backtest.ipynb) | Ch16-20 | Scored once, after `16_costs` is green AND the real Pro-account costs exist. **Copied 2026-09-07, not run; the holdout is sealed** | holdout prediction sets and backtest rows |
| Strategy Analysis | [`19_strategy_analysis`](19_strategy_analysis.ipynb) | Ch20 | Validation and holdout with intervals, DSR with the trial count, per-session trade diagnostics (level 3). **Copied 2026-09-07, not run** | `cohort_metrics`, `backtest_paired_metrics` |

## Run

Always inside the experiment, never against this directory's `run_log/`:

```bash
uv run python scripts/create_experiment.py --cs exness_fx_d1 --output experiments/exness_fx_d1
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python case_studies/exness_fx_d1/01_feasibility_analysis.py
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python case_studies/exness_fx_d1/02_labels.py
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python case_studies/exness_fx_d1/03_financial_features.py
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python case_studies/exness_fx_d1/04_model_based_features.py
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python case_studies/exness_fx_d1/05_evaluation.py
uv run --with pytest==9.0.3 python -m pytest bots/exness_fx_d1/tests -q   # point-in-time and data-quality tests
# Model stages open the study through open_study(); without a workspace that is the maintainer's
# regeneration path, which needs symlinked artifact directories. Inside the experiment pass the
# experiment root as WORKSPACE (same value as ML4T_OUTPUT_DIR) through the parameters cell:
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python -m papermill   case_studies/exness_fx_d1/06_linear.ipynb experiments/exness_fx_d1/notebooks/06_linear.ipynb   -p WORKSPACE "$PWD/experiments/exness_fx_d1" --cwd "$PWD"
# same for 07_gbm, 12_model_analysis and 13_backtest; on the WSL host LightGBM needs libgomp on
# LD_LIBRARY_PATH (bots/exness_fx_d1/BOT.md, Decisions log)
#
# 13_backtest takes the signal family through the parameters cell. The committed default is the
# per-pair time-series family; the cross-sectional one reproduces the equal-weight generations:
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python -m papermill   case_studies/exness_fx_d1/13_backtest.ipynb experiments/exness_fx_d1/notebooks/13_backtest_timeseries.ipynb   -p WORKSPACE "$PWD/experiments/exness_fx_d1" -p SIGNAL_FAMILY per_pair_timeseries --cwd "$PWD"
# Read-only helpers (no registry write, no trial):
uv run python case_studies/exness_fx_d1/_check_carry_series.py                       # needs FRED_API_KEY
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python case_studies/exness_fx_d1/_diagnose_timeseries_signal.py
ML4T_OUTPUT_DIR=experiments/exness_fx_d1 uv run python case_studies/exness_fx_d1/_report_phase5.py
```

The carry family (user decision 4) is fetched and turned on with:

```bash
uv run python data/macro/download.py --config bots/_shared/macro_config.yaml --list          # no key needed
uv run python case_studies/exness_fx_d1/_check_carry_series.py                               # (a)-(d) per leg
uv run python data/macro/download_alfred.py --config bots/_shared/macro_config.yaml          # -> data/mt5/macro_exness_fx_d1/
# then declare features.windows.carry, features.windows.carry_zscore and a `carry` family row in
# config/setup.yaml and re-run 03 -> 04 -> 05 -> 06 -> 07 -> 12 -> 13. The feature matrix changes,
# so 06/07 publish new prediction sets and 13 must declare
# SUPERSEDES_VALIDATION_PREDICTIONS = "930dcbfa4e57" and
# SUPERSEDES_EQUAL_WEIGHT_BASELINES = "52365960d8ca".
```

Both stages refuse to run until `download_mt5_bars` has written the MT5 parquet files and `history_depth.json`, and
until `setup.yaml::decision.server_clock` matches the clock recorded there. The pipeline runs in WSL2 (`docs/installation.md`);
the download runs on Windows Python with `MetaTrader5` against the logged-in terminal.
