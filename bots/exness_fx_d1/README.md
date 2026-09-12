# exness_fx_d1 — operating manual

The contract, the evidence and the decisions live in [`BOT.md`](BOT.md). This file is the
runbook: when the loop runs, what it needs, how a person arms it, how to stop it, and what is
forbidden until the research gates are green.

> **STATE ON 2026-09-08: THIS BOT HAS NO DEPLOYABLE SIGNAL AND MAY NOT TRADE.**
> Phase 5 registered **K = 3,204** trials across three strategy cohorts (long-only
> cross-sectional top-k, long-short cross-sectional sleeves, long-only per-pair time-series
> percentile) and **0 of 1,068 in each** is significant at 0.95 on the active return over the
> 1/N long book; not one spec anywhere reaches an active DSR of 0.5. Phase 6 (portfolio, costs, risk) is not
> opened and the declared holdout, 2025-09-01 to 2026-08-31, is **unscored**. Everything in
> `deploy/` and `monitor/` is deployment engineering built and tested ahead of a signal; it is
> not permission to trade, and the loop refuses to send an order.

---

## 1. What the deployment loop does

`deploy/deployment_loop.py` is the seven steps of `25_live_trading/02_etfs_deployment_loop.py`
on MetaTrader 5:

| # | Step | Where it comes from |
|---|---|---|
| 1 | Refresh MT5 four-hour history | `bots/_shared/mt5_loader.py` (opt-in, `--refresh`) |
| 2 | Recompute session panel and features | `case_studies/exness_fx_d1/_features.py`, the **same functions** the research stages call |
| 3 | Retrain the named model | preset YAML the registry row points at, on a walk-forward window clamped before the holdout |
| 4 | Persist the artefact | `deploy/state/<decision date>/` |
| 5 | Predict the cross-section | the refit, or a registered validation prediction set |
| 6 | Replay through `ml4t.backtest.Engine` | the parity tape; engine spec from `get_backtest_config("exness_fx_d1")` |
| 7 | Stage the basket | `SafeBroker(MT5Broker(magic=260901), LiveRiskConfig)`, shadow mode |

Three run modes:

```bash
# default: steps 1, 2, 5, 6 off a registered VALIDATION prediction set, then stop.
uv run python bots/exness_fx_d1/deploy/deployment_loop.py

# plumbing: the same, plus step 7, so the staging path is exercised. Labelled
# signal_status: plumbing_only. Still sends nothing.
uv run python bots/exness_fx_d1/deploy/deployment_loop.py --model latest-complete

# a named registered training run: refit (clamped before the holdout) and all seven steps.
uv run python bots/exness_fx_d1/deploy/deployment_loop.py --model <training_hash>
```

Useful flags: `--fake-mt5` (no terminal at all), `--as-of YYYY-MM-DD`, `--predictions <hash>`,
`--state-dir`, `--halt-file`, `--data-dir`, `--refresh`, `--arm`.

Always run from the repository root, with `ML4T_OUTPUT_DIR` pointing at the experiment that
holds the labels and the registry:

```bash
export ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_fx_d1
uv run python bots/exness_fx_d1/deploy/deployment_loop.py --fake-mt5
```

---

## 2. Schedule

| | |
|---|---|
| Decision instant | **20:00 UTC**, the close of the last MT5 H4 bar that closes at or before the `CME_FX` session close. The server clock is UTC+0 all year, so 20:00 UTC = 20:00 server. |
| Capital | `capital.allocated` 900 USD (the equity measured on the demo login 2026-09-07) against `capital.min_viable_allocated` 2,400 USD. **The bot cannot place an order at the current balance**; the loop reports `capital_viability.viable: false` and blocks staging. |
| Execution | the open of that same 20:00 UTC bar (`decision.execution_delay: next_bar_open`), inside the New York session |
| Cadence | one decision per `CME_FX` session, about 252 a year |
| When to run the loop | at **20:05 UTC** on each `CME_FX` session, five minutes after the decision bar closes, so the terminal has the bar. Earlier and the bar is not there; much later and the fill drifts away from the price the backtest assumed. |
| Never run | Friday 21:00 UTC to Sunday 21:00 UTC (the FX week is closed; the loop refuses because nothing quotes) and inside **21:00-22:00 UTC** on any day (the `other` spread bucket, p90 4.3-9.5 bps against an in-session 0.63-1.28; the loop refuses) |
| Retrain cadence | monthly (`model.refit_every_days: 31`), not per decision |

A missed session is a **no-trade day**, not a catch-up: the loop decides on the bar that closed
at 20:00 UTC and nothing else.

---

## 3. Credentials and environment

* **No password is ever stored.** `MT5Broker` calls `mt5.initialize()` against a terminal a
  person has already logged in, and then verifies `login`, `server` and `trade_mode`. A
  password passed through `initialize_kwargs` is rejected by the adapter.
* **The MetaTrader5 package runs on Windows only.** The research environment (features,
  registry, engine) builds on WSL2/Linux. In practice: the terminal and any refresh run on
  Windows, the loop runs where `ml4t` is installed. See BOT.md phase 0.
* `ML4T_DATA_PATH` must point at the shared data root (`data/mt5/*.parquet`,
  `data/mt5/monitor/`).
* `ML4T_OUTPUT_DIR` must point at the experiment (labels, registry).
* No API key is needed. `.env` holds no broker secret for this bot.

Account: Exness **Pro**, currently the demo login `Exness-MT5Trial7`. `execution.mode: paper`
in `deploy/risk_config.yaml` makes `MT5Broker.assert_paper_trading()` refuse anything that is
not a demo account, at connect and before every order.

---

## 4. Arming procedure

Arming is a **person, in a session, on purpose**. There is no configuration file that arms
this bot and no environment variable that does it either.

1. Confirm the gates. `--arm` is meaningless today and the loop ignores it: a plumbing run can
   never send, and there is no survivor to name with `--model <hash>`. Before arming is even a
   question, phases 6 and 7 must be green (see section 7).
2. Read the last run record: `deploy/state/<date>/run_*.json`. Check `signal_status`,
   `deployable`, `decision_bar_lag_minutes`, `steps.7_stage.reconciliation.clean` and
   `steps.7_stage.account_breakers.allows_trading`.
3. Confirm no account halt: `data/mt5/monitor/account_halt.json` must not exist.
4. Arm for one run only:

   ```bash
   uv run python bots/exness_fx_d1/deploy/deployment_loop.py --model <training_hash> --arm
   ```

   Even then, `live_risk_config.shadow_mode: true` in `deploy/risk_config.yaml` keeps every
   order inside `SafeBroker`'s `VirtualPortfolio`. Turning shadow mode off is a **separate,
   deliberate edit** of that file, made when phase 7 has passed and recorded in BOT.md's
   Decisions log. Arming can only ever make the configuration safer, never looser - the loop
   asserts that.
5. Live (a real Pro account) additionally needs `execution.mode: live` and
   `armed_live=True` passed to `MT5Broker` for the session; the adapter refuses a demo account
   in live mode and a real account in paper mode. Today `load_deploy_config` **refuses any
   value of `execution.mode` other than `paper`.**

---

## 5. Kill switch and how to stop the bot

Three levels, in the order they bite:

1. **Per-bot circuit breakers** (`monitor/circuit_breakers.py`, thresholds in
   `deploy/risk_config.yaml::breakers`). Any one OPEN and the loop stages nothing. Wiring
   `kill_switch_on_trip(safe_broker)` latches `SafeBroker`'s persistent kill switch, which
   survives a restart and must be cleared by hand.
2. **Account-level breakers** (`bots/_shared/monitor/`). Equity drawdown 6 %, daily equity loss
   2 %, margin level, distance to the broker stop-out, open positions and gross notional across
   every magic. A trip writes `data/mt5/monitor/account_halt.json` and **every bot on the login
   stops**. Note two things measured on 2026-09-07: the terminal reports `margin_so_so = 0.0`,
   i.e. **no stop-out level**, so the declared `fallback_stop_out_pct` is what protects the
   account; and one position carrying the retired legacy bot's magic **202500** was open, which
   is why the exposure breaker counts every magic and not only the bots'.
3. **Manual halt**, right now, without running anything:

   ```python
   from bots._shared.monitor import AccountGuard, load_account_limits
   AccountGuard(load_account_limits(), bot_id="operator").write_halt("manual stop: <reason>")
   ```

   (`write_halt` takes the reason positionally; the snapshot argument is optional.)

   To resume, a person clears it with a reason and a name:

   ```python
   AccountGuard(load_account_limits(), bot_id="operator").clear_halt(
       reason="<what was fixed>", cleared_by="<who>"
   )
   ```

   The cleared record is archived next to the halt file, never deleted.

Closing positions is **not** automatic. Nothing in this package closes a book on a breaker
trip; a breaker stops new orders. Closing is an operator action through the terminal or
`MT5Broker.close_position_async`, and it is deliberate that a machine does not liquidate on a
threshold it might have mis-measured.

### Kill criteria, and what implements each

| BOT.md criterion | Implementation |
|---|---|
| (a) drawdown from peak > 8 % of allocated capital | `DrawdownBreaker`, portfolio tier |
| (b) rolling 63-session hit rate < 0.5 for 21 consecutive sessions | `RollingHitRateBreaker` (a sliding window, not a losing streak) |
| (c) realised round-trip cost > 1.5x the assumed 2.6 bp for a month | `RealisedCostBreaker`, fed from the MT5 deal history |
| (d) reconciliation finds an unexplained position | `ReconciliationBreaker`: trips immediately, no recovery probe, latches the kill switch |
| (e) the decision bar is missing within tolerance | **not a breaker** - a no-trade day, logged to `deploy/state/no_trade_days.json`; `MissingDecisionBarBreaker` escalates only after 3 misses in 21 sessions |
| (f) realised swap above the declared value for a week | `SwapBreaker`, fed from the rollover history |
| retire: breakeven cost below the measured p90 spread | `monitor/retire.py` - reports `NOT_COMPUTABLE`: `16_costs` has not run |
| retire: holdout PSR against the 1/N book below 0.5 | `monitor/retire.py` - reports `NOT_COMPUTABLE`: the holdout is unscored |
| derived: daily loss 2 % of allocated capital | `DailyLossBreaker` (BOT.md names a daily loss without a number; the derivation is in `risk_config.yaml`) |
| derived: 5 consecutive losing round trips | `ConsecutiveLossBreaker` |

---

## 6. Monitoring

| File | What it does | Chapter |
|---|---|---|
| `monitor/drift.py` | PSI, K-S, prediction drift, rolling IC / hit rate / MSE. **Reference window is train/validation (2024-09-02 .. 2025-08-28), never the holdout**; it refuses a slice inside the unscored holdout. | 26/01 |
| `monitor/online_detectors.py` | ADWIN-style and DDM, windows in **decision sessions**. Reports `ready`: at ~252 decisions a year the first alert is possible only around session 168, so "no alert" before that means "cannot alert", not "no drift". | 26/02 |
| `monitor/rollout.py` | shadow -> capital-capped A/B -> staged -> full, promotion gate fixed in `monitor_config.yaml` before any challenger runs; returns `NOT_COMPUTABLE` while the holdout is unscored. | 26/03 |
| `monitor/circuit_breakers.py` | The tiers above, on the shared CLOSED/OPEN/HALF_OPEN machine. | 26/04 |
| `monitor/retire.py` | The two retire rules; `NOT_COMPUTABLE` with a reason and what unblocks it. | - |
| `bots/_shared/monitor/` | The account tier, shared by every bot. | 26/04 |

Two operational facts worth knowing before reading any dashboard:

* **The 0.01-lot floor, and the account cannot clear it today.** `volume_min 0.01 x
  trade_contract_size 100,000` = 1,000 units: about **1,172 USD** on EURUSD, **1,352 USD** on
  GBPUSD, **1,000 USD** on USDJPY and USDCAD (one unit of those is one US dollar), **649 USD**
  on AUDUSD. A leg smaller than that rounds to zero lots and is *rejected*, not shrunk. The
  demo login held **898.56 USD** when it was read on 2026-09-07, so at `capital.allocated: 900`
  four of the five pairs round to zero and the loop blocks staging with
  `capital_viability.viable: false`. A paper cycle needs at least
  `capital.min_viable_allocated` = **2,400 USD**. Every run record prints
  `steps.7_stage.capital_viability` and `steps.7_stage.min_tradable` per pair.
* **Lots are sized on the value of one unit, not on the quote.** An MT5 lot is 100,000 units
  of the **base** currency, so 0.024 lots of USDJPY is 2,400 USD of exposure and a 2,400 USD
  leg is 2,400 units, not 2,400/154 = 16. `unit_values` reads `currency_base` /
  `currency_profit` from `symbol_info`. The research engine uses its own, self-consistent
  convention (`price` is the account-currency value of one unit on both sizing and
  valuation), so its JPY legs carry the right notional and the two "quantities" are simply
  different quantities - which is why the parity gate compares baskets and sides and not unit
  counts (BOT.md open question 30).
* **The drift bands are not the notebook's.** This book holds five names; a 63-session mean hit
  rate has a standard error of about 0.028, so the notebook's 0.010 alert would fire on chance
  alone. The bands in `monitor_config.yaml` are one and two standard errors of the statistic on
  *this* book, with the arithmetic written out.

---

## 7. What is NOT allowed until phase 7 passes

Absolute, no exceptions without a dated user decision recorded in BOT.md:

1. **No live account.** `execution.mode` may only be `paper`; `load_deploy_config` raises
   otherwise. No real Pro-account order, of any size, for any reason.
2. **No paper cycle against a signal.** There is no survivor. A paper cycle would be trading a
   book the evidence does not support; the only permitted cycles are the dry runs above.
3. **No `shadow_mode: false`.** Editing that key is a phase-7 action.
4. **No topping the account up to make it trade.** Raising `capital.allocated` above the
   account's equity, or above `min_viable_allocated`, is a funding decision for the user and a
   dated entry in BOT.md - not a config edit made to get past a red `capital_viability`.
5. **The holdout stays shut.** No stage, notebook, script or loop may train on, score on or
   report a return over 2025-09-01 .. 2026-08-31 before the holdout stages run once. The loop
   clamps and asserts; `monitor/drift.py` refuses; the parity replay withholds performance.
   The marker file `deploy/state/holdout_scored.json` is written by the holdout stages and by
   nothing else.
6. **No stage 14-19 run on the phase-5 population.** Phase 6 is not opened (BOT.md Decisions
   log 2026-09-06): allocators, overlays and cost curves on a signal the DSR does not show
   would add trials without evidence.
7. **No writing into a downloaded `run_log/`** and no editing of the four pinned `setup.yaml`
   keys or `config/backtest/base.yaml`.
8. **No new magic number.** 260901 comes from `bots/_shared/__init__.py::MAGIC_ALLOCATION`;
   changing it orphans this bot's positions.

What *is* allowed now: dry runs, plumbing runs, unit tests, read-only MT5 calls
(`account_info`, `symbol_info`, `symbol_info_tick`, `positions_get`), and the real Pro-account
**read-only** cost measurement BOT.md user decision 5 asks for.

---

## 8. Tests

```bash
export ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_fx_d1
uv run --with pytest==9.0.3 python -m pytest bots/exness_fx_d1/tests -q
```

| File | Covers |
|---|---|
| `test_parity.py` | 25/08: data -> features -> predictions -> sizing -> orders, plus determinism |
| `test_deployment_loop.py` | dry run end to end on the fake MT5, the config gate, the evidence boundary, the decision-bar rule, `armed=False`, reconciliation of both views, the account halt, the run record |
| `test_circuit_breakers.py` | one trip test per breaker, the kill-criteria map, the retire rules |
| `test_drift.py` | PSI / K-S bands, the holdout refusal, detector readiness, the rollout gate |
| `test_account_breakers.py` | the account tier, the halt file, the magic allocation table |
| `test_lookahead.py`, `test_data_quality.py`, `test_mt5_*.py`, `test_costs_mt5.py`, `test_swaps_mt5.py`, `test_backtest_costs.py`, `test_sessions.py` | the earlier phases |

Nothing in the suite opens a terminal, sends an order or reads the holdout.

---

## 9. Definition of done (`bot-template.md`), as of 2026-09-07

The checklist the mentor skill applies to a finished bot, marked honestly. Four of nine are
met; the five that are not are all downstream of the same fact, that phase 5 has no survivor.

| | Item | State |
|---|---|---|
| [x] | `README.md` states the economic hypothesis, the cadence, the label, the assumed costs and the kill criteria | met — the hypothesis, cadence, label and costs are in `BOT.md`; this file carries the cadence, the kill-criteria map and the operating rules |
| [x] | `setup.yaml` declares walk-forward folds and the holdout before any training run exists | met 2026-09-05: 4 folds P3Y/P1Y, holdout 2025-09-01 → 2026-08-31 declared before the first fit; `01_feasibility_analysis` asserts "holdout untouched" |
| [ ] | All stages through `17_costs` ran inside an experiment; registry rows exist for each | **not met** — `01`–`07`, `12`, `13` ran; `14`–`16` were deliberately not opened (BOT.md Decisions log 2026-09-06: no spec is significant at K = 3,204, so allocators, overlays and cost curves would add trials without evidence) |
| [ ] | Breakeven cost exceeds assumed cost; DSR reported with the trial count | **not met on the first half, met on the second** — no breakeven cost exists (`16_costs` not run); the DSR *is* reported with its trial count: at K = 3,204 the best spec of any cohort is the time-series winner `4ebb02db5ac9` (active Sharpe 0.543, CI95 [-0.18, 1.72]), active DSR 0.0916 (notebook) / 0.0009 (library), `is_significant` False at 0.95; the generation-2 winner `fecd7db8d885` re-deflates to raw 0.0348 / active 0.0283 |
| [ ] | Holdout scored once; `strategy_assessment.json` archived | **not met, and deliberately so** — the holdout is untouched and stays that way until phases 6 and 7 are green |
| [ ] | Deployment loop runs end to end in dry-run and paper; parity tape matches staged orders | **half met** — dry run: yes, end to end on the fake MT5, and the parity tape matches the staged basket on the decision date (`test_parity.py`, `parity_agrees_with_last_rebalance is True`). Paper: **not run**, and may not be: a paper cycle needs a signal |
| [x] | `SafeBroker` limits, reconciliation and kill switch tested | met — `LiveRiskConfig` built from YAML and asserted field by field, reconciliation of both views (this magic against `RiskState`, every magic against the allocation table) with a foreign-magic case, and the kill-switch latch driven by `ReconciliationBreaker` |
| [x] | Drift dashboard, online detectors, circuit breakers and rollout protocol in place | met as code — all four exist, are configured from YAML and are tested on synthetic streams. Not validated against a live stream, because there is none (BOT.md open question 28) |
| [ ] | Retrain / pause / retire rules written down and automated | **half met** — pause rules (the breakers) are written and automated; the retrain cadence is declared (`model.refit_every_days: 31`) but never exercised on a real hash; the two retire rules are written and return `NOT_COMPUTABLE` because `16_costs` and the holdout are missing |

What unblocks the rest, in the order BOT.md's open question 18 declares: the time-series
mapping **ran on 2026-09-07 and did not clear the gate** (population
`exness_fx_d1:timeseries-percentile-baselines` `1836e5594858`, 1,068 specs); the carry family
is **blocked because `FRED_API_KEY` is present but empty** in both `.env` files, so no FRED or
ALFRED call resolves; the real Pro-account cost measurement is blocked on the user. Each is a
new trial on top of K = 3,204, and each is judged against the approved falsification line
before phase 6 is opened.
