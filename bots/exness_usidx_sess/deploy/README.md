# `exness_usidx_sess` — deployment operating manual

> ## THIS BOT MAY NOT TRADE
>
> * **Phase 5 ran in full and found 0 survivors of 1,403 scored specs at K = 1,780.** Not one
>   trial reaches a Deflated Sharpe Ratio of 0.50 on the active return over a 1/N long book of
>   US500 and USTEC, let alone the drafted 0.95. At K = 1,780 the expected maximum annualised
>   Sharpe from luck alone is +4.006; the best trial reaches +1.587.
> * **Phase 6 is not opened. Phase 7 is sealed**: the declared holdout 2026-03-01 .. 2026-08-31
>   has never been read by any stage.
> * The hypothesis, the kill criteria and the DSR benchmark are all a **draft awaiting the
>   user** (`BOT.md`, "Awaiting user approval").
> * **This account cannot fund the bot at any weight**: equity 896.97 USD against a smallest
>   placeable position of 1,080.32 USD (US500) and 1,482.95 USD (USTEC).
>
> This package exists because a parity harness and a safety layer test the **code path**, not
> the edge. It is **DRY RUN ONLY** and it is not permission to trade. Same wording, same
> discipline as `bots/exness_fx_d1/BOT.md` phases 8 and 9.

## What is in here

| File | What it is |
|---|---|
| `deployment_loop.py` | The seven steps of `25_live_trading/02_etfs_deployment_loop.py` on MetaTrader 5, with the parity harness of `25_live_trading/08` wired in |
| `schedule.py` | The two decision instants per NYSE cash session, **derived** from the calendar and from `bots/_shared/sessions`, converted through `ServerClock` |
| `risk_config.yaml` | `LiveRiskConfig`, the CFD guards, the breaker thresholds and the evidence boundary. Every number carries its derivation |
| `state/<spec>/<date>/` | One run record per cycle, per spec, plus the persisted `RiskState` and the audit journal |

The broker adapter is **not** here: every bot uses `bots/_shared/mt5_broker.py`. The account
tier is **not** here: it is `bots/_shared/monitor/`. The strategy breakers are in
`../monitor/circuit_breakers.py`.

## How to run it (dry run)

From the repository root, in the WSL2 environment, with `ML4T_OUTPUT_DIR` unset:

```bash
export LD_LIBRARY_PATH=$HOME/omp          # lightgbm needs libgomp in this clone
uv run python bots/exness_usidx_sess/deploy/deployment_loop.py --spec intraday  --fake-mt5
uv run python bots/exness_usidx_sess/deploy/deployment_loop.py --spec overnight --model latest-complete --fake-mt5
uv run python bots/exness_usidx_sess/deploy/deployment_loop.py --schedule      # print the schedule only
```

`--spec` is not optional in spirit: **one run is one book**. The two specs have two feature
matrices, two models, two registries and two decision instants, and they are never netted
before the broker sees them.

Modes:

* `--model none` (default) — steps 1, 2, 5, 6. Predictions are **read** from a registered
  *validation* prediction set: no refit, no data extension, no holdout read. Stops after
  parity and says why.
* `--model latest-complete` — the same, then step 7 in dry run so the staging path is
  exercised end to end. Stamped `signal_status: plumbing_only`.
* `--model <training_hash>` — **always raises**. Refitting a spec that failed phase 5 would be
  presenting a failed trial as a deployment.
* `--arm` — **refused** (`PermissionError`) while the kill criteria are a draft. The flag
  exists so the refusal is testable.

## The schedule, and why no UTC hour appears in the code

Two time facts collide here and `setup.yaml` declares them **separately** on purpose:

* the **server clock** is UTC+0 all year (`server_clock.follows_dst_of: null`, measured
  2026-09-05 with `ServerClock.measure`);
* the **instrument's trading hours** follow New York DST
  (`trade_hours_follow_dst_of: America/New_York`, measured on 39 transition-free months of H1
  bars: the daily CFD break ends at 18:00 New York in every one of them).

So `15:00 / 21:00 UTC` (winter) and `14:00 / 20:00 UTC` (summer) are **outputs**, not inputs. A
UTC constant would be wrong for roughly five months of every year and on every half day.
`schedule.py` derives both instants from the NYSE calendar through
`case_studies/exness_usidx_sess/_features.session_grid` — the same function `01`, `02`, `03`
and the point-in-time test call — cross-checks them against
`bots/_shared/sessions.SESSIONS["us_cash"]`, and converts with `ServerClock.to_server`. The
CFD break guard is declared as **17:00-18:00 America/New_York** and converted at run time.
`tests/test_schedule.py` greps the whole package and fails if a UTC decision hour is written as
a value.

## What blocks a real cycle today

The run record lists every reason on every cycle. As of 2026-09-08 there are nine:

1. `US500`: an equal-weight leg needs 2,160.64 USD of allocated capital; `allocated` is 900.00
2. `US500`: the smallest placeable position is 1,080.32 USD against equity 896.97
3. `USTEC`: an equal-weight leg needs 2,965.91 USD
4. `USTEC`: the smallest placeable position is 1,482.95 USD
5. `capital.allocated` 900.00 is below `min_viable_allocated` 3,000.00
6. the kill criteria are a DRAFT the user has not approved
7. the holdout 2026-03-01..2026-08-31 has not been scored
8. phase 5 found 0 survivors of 1,403 scored specs at K = 1,780
9. not armed

Reasons 1-5 are a **funding** decision (open question 10) and reasons 6-8 are an **evidence**
decision. Fixing the funding does not touch the evidence: there would still be no survivor.

## The parity gap you have to know about

`setup.yaml::execution.share_type` is `fractional` — the research engine holds any weight it
likes. MT5 does not. `volume_min x trade_contract_size`, read from `symbol_info` on 2026-09-08:

| Symbol | `volume_min` | contract | ask | smallest position |
|---|---|---|---|---|
| `US500m` | 0.14 | 1.0 | 7,716.56 | **1,080.32 USD** |
| `USTECm` | 0.05 | 1.0 | 29,659.06 | **1,482.95 USD** |

`MT5Broker.normalize_lot` rounds **down** and **rejects** below `volume_min`. A research fill
under the floor therefore has **no live counterpart at all** — it is not a smaller trade, it is
no trade. Every cycle measures this (`order_parity`) and reports it under
`step6_parity.stages.5_orders.gaps`; `capital_viability` blocks staging with the arithmetic.

## What is forbidden until a survivor exists AND the holdout has been scored once

* `armed_live` — anywhere, for any reason.
* Shadow mode with capital. `shadow` here means orders are booked into a `VirtualPortfolio` and
  never reach the adapter. Allocating capital to a shadow book is a live deployment with a
  different name.
* Reading the holdout. The refit window is clamped strictly before `holdout_start` and the
  clamp is asserted; the parity replay withholds every performance key until
  `state/holdout_scored.json` exists, and a flag alone cannot create it.
* Calibrating **any** monitoring threshold from generation 1's Sharpe distribution. Thresholds
  come from the declared kill criteria or from measured price facts. See
  `../monitor/monitor_config.yaml` and `tests/test_monitoring.py`.

## If a survivor ever does appear

In this order, none of it optional:

1. Fix the four mechanical defects the mentor listed (`fixed_threshold` semantics — already
   declared as `signed` in `setup.yaml` but never yet scored; regime-conditional costs;
   `n_traded` and a minimum-trades floor; the benchmark exposure convention).
2. Declare a **smaller K in `setup.yaml` before generation 2 runs**, on a revised hypothesis.
   Never by re-scoring the existing 1,780 at a smaller K.
3. Phase 6 (`14`-`16`), on the real Pro account's costs.
4. Phase 7, scored **once** — and note the operational risk: this bot has **two registries**,
   so "scored once" has to be enforced by hand across both.
5. Only then: a paper cycle on a funded account, and only then a conversation about arming.
