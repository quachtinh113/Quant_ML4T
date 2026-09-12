# exness_btc_8h — research pipeline

Route B research directory for the bot [`bots/exness_btc_8h/BOT.md`](../../bots/exness_btc_8h/BOT.md):
**one** crypto CFD, `BTCUSD`, on an Exness MetaTrader 5 account, decided three times a day at
00:00 / 08:00 / 16:00 UTC, **every day of the week**.

Authored 2026-09-08 with `run_log/` empty and no `registry.db`. Stages 01–05 have run; **no model
has been fitted, no backtest has been run and the holdout has never been opened.** The bot stops
at the end of phase 3 until its Hypothesis and Kill criteria are written and approved: no trial
may be counted before there is a falsification line to count it against.

## Where it came from, and what it is not

Forked from [`case_studies/exness_gold_sess`](../exness_gold_sess), **not** from
`crypto_perps_funding`, although `bots/README.md` names the latter. Three reasons, all readable in
that case study: its own README says the release pipeline is not supported and notebooks 13–17 are
not an end-to-end strategy; every one of its six feature families reads the perpetual's premium
index or its funding settlements, neither of which a CFD has; and its execution is `same_bar` at
the funding instant, which without funding is a one-bar lookahead. What was taken from it is
arithmetic and convention only — 1,095 slots a year, `calendar: crypto` with
`periods_per_year: 365`, and the shape of `labels.rebalance_step` for an 8-hour schedule.

## What is different from every sibling bot on this account

| | `exness_fx_d1` | `exness_gold_sess` | **`exness_btc_8h`** |
|---|---|---|---|
| Instruments | 5 | 2 | **1** |
| Decision grid | 1 a session, 5 days | 2 a weekday | **3 a day, 7 days** |
| Cross-section | rank over 5 (STOPped) | none (2 names) | **none — a percentile over one name is a constant** |
| IC statistic | cross-sectional | pooled panel | **time series, with a HAC interval *and* a block bootstrap** |
| Swap on the demo | 0.0 | 0.0 | **−1,638.6 points/lot/night long, every night, ×3 on Friday** |
| Spread | ~1.3 bps, flat | 0.6 / 4.7 bps, flat | **1.3 bps live, 17.4 bps p90 in sample — the same points against a 15× price move** |

Two of those rows are the whole design.

**One instrument means every cross-sectional mechanism is removed, not weakened.**
`features.ranked` is empty, `mapping.class` is `per_asset_signal_timing`, and the allocation
dimension of Ch17 is declared empty because every allocator that spreads weight across names is
the identity on one. Three implementations in this repository return an *empty result* rather than
an error on a one-name panel (`feature_engineering`'s `min_cross_section`, the registry's
hard-coded `min_obs=5`, `exness_fx_d1/05_evaluation`'s `MIN_PERIODS`) and a fourth
(`plot_persistence`'s right panel) raises on an axis limit. `05_evaluation` documents all four and
uses a time-series statistic instead.

**Two cost readings that disagree by 11×, and the choice of which one the engine charges.** The
tick measurement taken on 2026-09-08 says the spread is a flat 1,000 points — 1.3 bps at today's
price, identical in every session bucket, at the swap rollover and at the weekend. The broker's own
per-bar record says the development window cost a median of 6.3 bps and a p90 of 17.4, because the
quote is a near-constant number of *points* while the price rose from 6,700 to 103,500.
`costs.spread_bps` therefore carries the **in-sample** range — it is what
`backtest_loaders::_normalize_cfd_costs` turns into the engine's slippage leg — and the tick table
is what a live order and the go-live gate are read against.

## Layout

```
config/setup.yaml            every declaration, with the measurement or file:line behind each number
config/backtest/base.yaml    engine defaults (repository-pinned; edit only on an empty run_log/)
config/training/*.yaml       the model menu, DECLARED and NOT RUN (43 configurations a label)
_features.py                 the one decision grid, price panel and feature construction
_model_reading.py            how a ONE-NAME panel is read: time-series IC, HAC + block bootstrap
01_feasibility_analysis.py   the grid, the two cost readings, IC*, the fold ladder
02_labels.py                 fwd_ret_8h and fwd_ret_24h, sealed on the decision grid
03_financial_features.py     68 columns on two grids (8h slots + D1 asof-joined on the bar close)
04_model_based_features.py   GARCH(1,1) and a two-state HMM, fitted per fold, filtered forward
05_evaluation.py             one decision per column, with two intervals and an FDR adjustment
```

Tests live outside this directory, in
[`bots/exness_btc_8h/tests/`](../../bots/exness_btc_8h/tests/): the point-in-time harness, the
data-quality census and a regression test for the shared CFD cost classifier.

## Running it

```bash
uv run python scripts/create_experiment.py --cs exness_btc_8h --output experiments/exness_btc_8h
export ML4T_OUTPUT_DIR=$PWD/experiments/exness_btc_8h
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python case_studies/exness_btc_8h/01_feasibility_analysis.py
# ... 02 .. 05 in order
uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/ -q
```

On the WSL2 host this project runs on, `LightGBM` needs `LD_LIBRARY_PATH=$HOME/omp` to reach
`libgomp.so.1`; the phase 0–3 stages do not import it, but `06_linear` onward will.

Needs MT5 history in `ML4T_DATA_PATH/mt5/` — `4h.parquet` (folded to 8h by
`mt5_loader.resample_4h_to_8h`), `daily.parquet` and `history_depth.json` — written by
`bots/_shared/mt5_loader.py` from a logged-in Exness terminal. There is no CI fixture, which is why
every stage is `skip: true` in `tests/overrides.yaml` with that reason recorded.

## What the first run found

Measured 2026-09-08 on the demo login `206539306 @ Exness-MT5Trial7`:

- **The grid is complete.** 8,261 development decision slots from 2018-02-16, zero missing, zero
  folded from a single H4 bar, every one on {00, 08, 16} UTC. The fill instant equals the decision
  instant, so `next_bar_open` costs no waiting time on this instrument.
- **Financing falls on one slot in three.** The 16:00 decision's hold crosses server midnight and
  pays −2.08 bps of notional on the long side; the other two pay nothing. Fridays pay triple.
  Annualised, a continuously held long pays 9.7 % of notional.
- **IC\* is the number to read everything against.** A signal needs a rank correlation of about
  **0.095** on the development window (`01`, `02`) and **0.114** on the validation slots `05`
  scored — against the in-sample spread plus a night of financing — before it pays for itself.
  The largest |IC| among 73 columns is 0.041, i.e. **0.36×** the validation-slot figure — and **0 of 73 survive a
  Benjamini-Hochberg adjustment** over the set. Three clear 5 % under both a Newey-West interval
  and a block bootstrap, which agree on all 73 columns.
- **The strongest single relationship is a one-day reversal.** `slot_zscore_3` and `slot_ret_3`
  both carry IC ≈ −0.04 with the same sign in all four validation folds; `01_feasibility_analysis`
  had already found the three-slot autocorrelation at −0.084 against a white-noise band of 0.022.

None of that is a strategy. It is what phase 4 has to beat, and phase 4 does not open until the
falsification line is approved.
