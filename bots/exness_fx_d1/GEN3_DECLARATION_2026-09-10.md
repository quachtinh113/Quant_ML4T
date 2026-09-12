# GEN3_DECLARATION_2026-09-10 — generation 3 of `exness_fx_d1`

Modelled on `bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md`. Written **before** the first backtest of
generation 3 runs. Repo root `D:/05_Quant/machine-learning-for-trading` (branch `exness-bots`). Registry at
declaration time (WSL, `~/ml4t/experiments/exness_fx_d1`): `backtest_runs` **3,204** (0 holdout), `official
populations` 2 relevant to phase 5 (`exness_fx_d1:equal-weight-baselines` tip generation 2 `52365960d8ca`,
`exness_fx_d1:timeseries-percentile-baselines` tip generation 1 `1836e5594858`), `training_runs`/`prediction_sets`
unchanged (no refit in this round).

State this declaration inherits, not re-argued: phase 5 gate **not met** on any of the three existing cohorts at
K = 3,204 (`bots/exness_fx_d1/BOT.md` Phase status row 5); the DSR gate is `confidence_level = 0.95` on the active
return over the 1/N long book, notebook formula and `ml4t.diagnostic` library formula both reported (user decision
2026-09-06); holdout `2025-09-01 .. 2026-08-31` never read. USER DECISION 2026-09-10 (OQ 32) approved three
generation-3 items; this declaration executes (a) and (b) now, defers (c) (Nautilus confirmation) to after a
survivor is found.

## Mentor fleet-gate addendum, 2026-09-10 (added before the DSR table is ever read)

**(1) Residual double-discretisation under `ShareType.INTEGER` on USDJPY/USDCAD — known, conservative, undeclared
in gen 3.** `_apply_lot_floor_if_declared` returns a WEIGHT sized to an exact MT5 lot count; the engine then
re-derives its own share count from that weight (`ml4t/backtest/preopen.py`: `raw = target_notional /
(open_price * multiplier)`, rounded under `ShareType.INTEGER` — measured directly in the installed engine source,
read-only, not synced: `preopen.py:491/500` do `float(int(policy_fill))` and `_round_quantity`'s
`RoundingPolicy.TOWARD_ZERO` branch does `float(math.trunc(value))` — **INTEGER truncates toward zero (floors a
long position), never rounds to nearest**). For a quote-currency-is-account-currency symbol (EURUSD/GBPUSD/AUDUSD)
this second rounding is a no-op relative to the lot-floor weight: `shares = lots * contract_size * price / price =
lots * contract_size`, an integer whenever `lots` is a multiple of `volume_step` and `contract_size` is a whole
number, so nothing new is lost. For a base-currency-is-account-currency symbol (USDJPY/USDCAD) it is **not**:
`shares = lots * contract_size / price`, which is generally non-integer (price is not a divisor of 100,000), so a
SECOND truncation applies on top of the lot-grid rounding `_apply_lot_floor_if_declared` already did. Relative
error bound `price / (lots * contract_size)` (one truncated unit over the exact share count): **≤ 0.3 % per leg at
real registered size** (a 1-lot USDJPY leg at price ≈150: `shares_exact = 100,000/150 ≈ 666.7`, error `≈
1/666.7 ≈ 0.15 %`), **≈ −10 % at the 0.01-lot floor itself** (`shares_exact = 0.01*100,000/150 ≈ 6.667 → 6`, a
downward bias of `(6.667−6)/6.667 ≈ 10 %`, always shrinking the position, never enlarging it). This is declared
here as a known, bounded, **conservative** (always shrinks, never inflates a position) discretisation of
generation 3 that the lot-floor fix does not eliminate; §B below records the fix chosen for the *next* registered
run (not retrofitted onto generation 3, which is already running).

**(2) The lot floor is currently OUTSIDE `backtest_hash`.** `sizing.min_lot_policy` and `costs.contract.*` (the
keys `_apply_lot_floor_if_declared` reads) are **not** part of `strategy_spec` today — the function reads them by
re-opening `setup.yaml` directly (`case_studies/utils/backtest_runner.py::_apply_lot_floor_if_declared`), not
through anything `plan_backtests`/`run_backtests` hash. Generation 3's `backtest_hash`es moved for a **different**
reason: `execution.share_type` (a declared spec-hash input, `setup.yaml`'s own top comment) and
`account.allow_leverage` (part of `strategy_spec["backtest_config"]["account"]`, which is hashed). Toggling
`sizing.min_lot_policy` or a `costs.contract.*` value alone, with `share_type`/`account` unchanged, would **not**
move the hash today — a silent identity gap. §B below moves these keys into the hashed spec for the next
registered run.

**§B — fix for the NEXT registered run (implemented and tested on the WINDOWS tree only, NOT applied to the
generation-3 run already in flight).** Two decisions, both made, both tested:

1. **`mt5_lot` maps to `ShareType.FRACTIONAL`, not `INTEGER`** (`case_studies/utils/backtest_presets.py`, mirrored
   in `bots/exness_fx_d1/deploy/deployment_loop.py:944-961`). Reason: the lot grid (`volume_step`) is already the
   discretisation `_apply_lot_floor_if_declared` enforces upstream on the weights frame; a second, engine-level
   `INTEGER` truncation on top of that is not a broker constraint the lot grid did not already apply — it is
   exactly the residual error addendum item 1 measures (≤0.3%/leg at real size, ≈−10% at the 0.01-lot floor).
   `FRACTIONAL` removes that second truncation.
2. **`sizing.min_lot_policy` and `costs.contract.*` now enter the hashed `strategy_spec`.** A new shared reader,
   `case_studies/utils/backtest_loaders.py::read_mt5_lot_sizing_declaration(case_study_id)`, returns the
   declaration (or `None` when not opted in) and is the single source of truth for both consumers:
   `_apply_lot_floor_if_declared` (applies it — unchanged) and `build_resolved_backtest_config` (**new**: records
   it in `resolved.metadata["mt5_lot_sizing"]` plus `resolved.metadata["execution_share_type_declared"] =
   "mt5_lot"` — the DECLARED value, distinct from the translated engine `position_sizing.share_type` value —
   whenever `case_config.share_type == "mt5_lot"`). `backtest_config.metadata` is part of what
   `case_studies/utils/registry/specs.py::backtest_hash_from_parts`/`_hashable_strategy_spec` hashes (only
   `preset_path` is excluded), so this metadata injection moves `backtest_hash` when `min_lot_policy` toggles —
   proven directly against `backtest_hash_from_parts` in `test_toggling_min_lot_policy_changes_the_hashed_metadata_and_the_backtest_hash`.
   A case study that never declared `mt5_lot` (`fx_pairs`, `etfs`) takes neither branch: its resolved metadata has
   neither new key, proven in `test_a_case_study_without_mt5_lot_gets_no_new_metadata_keys[fx_pairs|etfs]` — legacy
   spec, and therefore legacy hash, byte-identical.

**INTEGER rounding semantics, measured read-only in the installed engine source (`~/ml4t/.venv/.../ml4t/backtest/`,
no sync):** `preopen.py:491,500` do `float(int(policy_fill))`; `preopen.py::_round_quantity`'s
`RoundingPolicy.TOWARD_ZERO` branch does `float(math.trunc(value))`. Both are **truncation toward zero**, never
round-to-nearest — for a long position this always shrinks it, matching the addendum's "conservative" framing (a
downward bias, never an inflated position).

**Files changed for §B (Windows tree only)**: `case_studies/utils/backtest_loaders.py` (new
`read_mt5_lot_sizing_declaration`), `case_studies/utils/backtest_presets.py` (fractional translation +
metadata injection), `bots/exness_fx_d1/deploy/deployment_loop.py` (mirrored fractional translation),
`bots/exness_fx_d1/tests/test_lot_floor_currency.py` (+4 tests), `bots/exness_fx_d1/tests/test_deployment_loop.py`
/ `test_parity.py` (assertions updated `integer -> fractional`).

**§C — two material items from the mentor fleet gate, same round, implemented in
`case_studies/utils/backtest_runner.py::_apply_lot_floor_if_declared`:**

1. **Fail loud on an unclassifiable symbol.** A symbol is "covered" only when declared in
   `costs.contract.account_currency_is_base_for` OR its quote currency (read only from the standard six-letter FX
   convention, `symbol[3:6]`, never a decimals-based guess) equals the declared `costs.account_currency` (a new,
   opt-in key; `exness_fx_d1/setup.yaml` declares `USD`). An uncovered symbol (a cross pair, or any symbol when
   `costs.account_currency` is undeclared) **raises** rather than silently applying an unverified convention — the
   same "declare or raise" discipline `case_studies/utils/spread_costs.py::_resolve_declared_map` already uses.
   Tests: `test_a_cross_pair_neither_declared_nor_quote_matching_raises`,
   `test_undeclared_account_currency_raises_for_a_quote_only_symbol`,
   `test_every_real_exness_fx_d1_symbol_is_covered`, and (rewritten to match the new, safer behaviour)
   `test_usdjpy_without_the_currency_declaration_now_raises_instead_of_silently_mispricing`.
2. **The `price is None -> return weight unchanged` silent path now raises** for a non-zero weight (a weight of
   exactly `0.0` still passes through with no price needed — nothing to size). Consistent with the "declare or
   raise" convention elsewhere in this module: a leg the join could not price is a data problem to surface, not a
   reason to skip the lot floor silently for that one row. Tests: `test_a_non_zero_weight_with_no_price_raises`,
   `test_a_zero_weight_with_no_price_still_passes_through`.

**Fleet convention, recorded**: every case study opting into `execution.share_type: mt5_lot` must also declare
`costs.account_currency`, and every symbol in its universe must be either base-covered
(`account_currency_is_base_for`) or quote-covered (its own six-letter code's quote matches `costs.account_currency`)
— an uncovered symbol is a configuration error, not a silently-approximated one.

**`xau_fx_mt5`'s own test fixture updated** (`experiments/xau_fx_mt5/tests/test_execution_guards.py::
_fake_backtest_config`, Windows tree only): gained `"account_currency": "USD"` so its XAUUSD fixture (quote USD)
stays classified exactly as it always implicitly was — required by C(1) above, legacy-preserving, not a behaviour
change to what that suite tests. Verified: `experiments/xau_fx_mt5/tests/test_execution_guards.py` **37/37**
passed in an isolated hardlink-broken snapshot (`~/ml4t_stage_gen4`, not `~/ml4t`) after every §B/§C change.

## Mentor re-gate addendum, 2026-09-10 (added before any DSR table is read)

**(D1) Generation 3, once registered, is NOT reproducible on the tree that carries §B/§C.** §B changed two things
that are spec-hash inputs of every future computation: the engine `ShareType` resolution (`mt5_lot -> fractional`,
was `-> integer`) and `backtest_config.metadata` (now carries `mt5_lot_sizing` / `execution_share_type_declared`
for any case study declaring `mt5_lot`). Generation 3's rows, registered by the sweep running under the
PRE-§B/§C code (`ShareType.INTEGER`, no lot-sizing metadata), carry hashes that code no longer computes: the
moment §B/§C are synced into `~/ml4t`, re-submitting the identical nominal spec (same signal, same account model,
same lot-floor declaration) produces a **DIFFERENT** `backtest_hash` (both because `ShareType` resolves differently
and because the metadata payload differs) and is therefore registered as a **NEW** generation, not a re-run of
generation 3. Generation 3 is hereby declared a **frozen generation**: its engine semantics
(`ShareType.INTEGER`, the double-discretisation of addendum item 1, no hashed lot-sizing metadata) exist only in
its own registered rows from this point forward, not in the code. Re-running the same nominal spec under §B/§C is
**generation 4** of both populations, **K += 2,136** (1,068 + 1,068, the same shape as generation 3), to be
declared BEFORE that run the same way this file declared generation 3 before its own first fit — not implied,
not assumed, not silently absorbed into "the same experiment, corrected code."

**(D2) Guard (d)'s fixed-`initial_cash` limitation, declared as a semantic limit, not a bug.**
`_apply_lot_floor_if_declared` receives `initial_cash` as the single scalar `run_backtest` was called with (the
backtest's STARTING cash, e.g. 100,000 — `case_studies/utils/backtest_runner.py:1512`,
`weights = _apply_lot_floor_if_declared(weights, prices, case_study, initial_cash)`, called once, outside any
per-timestamp loop) and applies it to EVERY row of the `weights` frame across every rebalance date in the
backtest. The ENGINE, however, sizes each rebalance against **running equity**, which drifts from `initial_cash`
as P&L accrues over the backtest. So the lot-floor guard's "is this leg above 0.01 lots" decision is calibrated
against a **fixed reference notional**, not the broker's live constraint at that point in the backtest — a leg
that would clear the floor against the account's actual (drifted) equity may be evaluated against the smaller or
larger starting notional instead, and vice versa. The error this introduces is **bounded by how far equity has
drifted from `initial_cash`** by that rebalance date (a validation-window Sharpe near 0 to 0.7 on a book that
rarely exceeds ~20% drawdown or ~20% gain implies a bound on the order of that same percentage, not unbounded).
This is a **semantic limitation of the guard as designed** (it floors against the backtest's declared starting
size, matching how `setup.yaml::execution.initial_cash` is used everywhere else in this case study — "the
research book is scale-free," `deploy/risk_config.yaml`'s own wording), not an implementation bug; a
running-equity-aware floor would need `_apply_lot_floor_if_declared` to receive a per-timestamp equity series,
which no caller supplies today. Recorded here rather than fixed, because fixing it is a different, larger change
(a new function signature, a new caller contract) than the fleet gate's B/C items, and generation 3 (and its
declared generation-4 successor under §D1) both run at `execution.initial_cash: 100_000`, a single value for the
whole backtest, where this limitation is real but its bound is the same order as the equity drift already visible
in every winner's reported drawdown/CAGR figures in this file's own generation-2 and time-series sections.

**Tests, all run in the isolated snapshot `~/ml4t_stage_gen4` (never `~/ml4t`, per the constraint not to disturb
the running sweep or the shared WSL tree while other bots' work is in flight there):**
`bots/exness_fx_d1/tests/test_lot_floor_currency.py` **22 passed**; full `bots/exness_fx_d1/tests` (excluding the
`ML4T_OUTPUT_DIR`-only tests) **285 passed, 54 skipped**; `test_deployment_loop.py` + `test_parity.py` with
`ML4T_OUTPUT_DIR` set **85 passed**; `experiments/xau_fx_mt5/tests/test_execution_guards.py` **37 passed**.

Guards this declaration is checked against (`mentor-protocol.md`): **Multiple testing** (K accumulates, declared
before the run, never cut on results), **Evidence boundary** (holdout not read; `deploy/risk_config.yaml`'s own
evidence-boundary block untouched), **Point-in-time** (no change to any label, feature or fold boundary),
**Parity** (the deployment loop's parity replay is kept consistent with the research engine's new ShareType, see
§3), **Costs** (unchanged — commission/slippage/swap block untouched, the added guard in `13_backtest.py` still
refuses a mismatch), **Safety** (`deploy/risk_config.yaml::execution.mode` stays `paper`, `armed: false`,
`live_risk_config.execution_mode: shadow`; nothing here arms anything).

---

## 0. What changes, and why every registered `backtest_hash` moves

**(a) Research engine account model.** `case_studies/exness_fx_d1/config/backtest/base.yaml::account` gains
`allow_leverage: true` and `initial_margin: 0.0005` (Exness leverage cap 1:2000 on this account, `BOT.md` Identity
table), plus `long_maintenance_margin: 0.00025` / `short_maintenance_margin: 0.0003` (scaled from the engine's
Reg-T defaults by the same ratio `initial_margin` was scaled by, so `maintenance < initial` and `short tighter than
long` both still hold — `ml4t.backtest.config.BacktestConfig.validate` requires the former). `ml4t.backtest.broker`
derives `account_type = "margin"` from `allow_leverage=True` (was `"crypto"`, a cash account, from
`allow_short_selling=True` alone). This is exactly what OQ 23 (`BOT.md`) found missing: the generation-2 winner's
book was one-legged on 48 % of its invested sessions because a cash account cannot fund a new short leg until the
old one's proceeds land. A margin account holds both legs of a `top_k = 1` or `top_k = 2` long-short target book at
once. Verified (WSL smoke, `build_resolved_backtest_config("exness_fx_d1", ...)`, no registry write):
`account_type = "margin"`, `allow_leverage = True`, `initial_margin = 0.0005`, `engine_cfg.validate(warn=False) ==
[]` (no margin-parameter issue).

**(b) Broker lot floor in research sizing.** `case_studies/exness_fx_d1/config/setup.yaml::execution.share_type`
moves from `integer` to `mt5_lot`, gating `case_studies/utils/backtest_runner.py::_apply_lot_floor_if_declared`
(added 2026-09-10 for `xau_fx_mt5`, execution-guards phase, shared and reused here, not reimplemented — task rule).
`sizing.min_lot_policy: reject` and `costs.contract.{volume_min: 0.01, volume_step: 0.01, volume_max: 200.0}` are
declared (real numbers from `bots/assets/*.md`, §1 below). A leg below 1,000 units (0.01 lot) is rounded down to
the lot grid and **dropped** (weight → 0.0) when it rounds to zero — never bumped up — the same rule
`deploy/risk_config.yaml::execution.min_lot_policy: reject` already applies on the live (dry-run) path.

**(c) Nautilus confirmation.** Deferred. Per OQ 32(c): every generation-3 **survivor** (a spec at DSR ≥ 0.95 on
active return at the new cumulative K) is re-run with `exports/fxd1_nautilus_check/replicate.py` before phase 6
opens. Not run in this round because there is no survivor yet to confirm; §7 records the deferral explicitly.

**Why every registered `backtest_hash` moves.** `setup.yaml`'s own top comment on the `execution:` block: *"cash +
share_type are spec-hash inputs"* — changing `share_type` alone invalidates every backtest identity of this case
study. `account.allow_leverage` is read into `strategy_spec["backtest_config"]["account"]`, which
`resolved_allow_short_selling` reads and which is part of what `plan_backtests`/`run_backtests` hash. Both
generation-1 registered items (long-only top-k `b57aa4a93f72`, already superseded within its own population; the
per-pair time-series generation 1 `1836e5594858`) and generation 2 (long-short `52365960d8ca`) stay in the registry
byte-for-byte, immutable, still counted at K = 3,204 and still re-deflatable — this declaration produces new rows
under new hashes, it does not touch old ones.

**Additive code changes required, and why they are additive/opt-in** (task rule: "make ONLY an additive opt-in
change with a legacy-unchanged behavioural test... and say so"). Two defects were found while wiring (a)/(b), both
fixed with the same discipline as the execution-guards work: an early return / a value check that changes nothing
for any case study that does not declare the new key, proven by a legacy-unchanged test.

1. **Units-of-base-currency bug in `_apply_lot_floor_if_declared` (OQ 30 corollary).** The shared helper assumes
   every symbol's QUOTE currency is the account currency (EURUSD/GBPUSD/AUDUSD — correct). For USDJPY/USDCAD, whose
   BASE currency (USD) already **is** the account currency, dividing by price a second time shrinks the implied lot
   count by ~150x (USDJPY) / ~1.38x (USDCAD) — it would have dropped almost every USDJPY/USDCAD leg regardless of
   true size, the same class of bug `bots/_shared/mt5_broker.py::unit_values` already fixed on the LIVE path
   2026-09-07. Fix: `costs.contract.account_currency_is_base_for: [USDJPY, USDCAD]`, read only when present;
   undeclared or empty reproduces the original arithmetic exactly (test:
   `test_legacy_path_unchanged_when_account_currency_is_base_for_absent/_is_empty`, and the pre-existing
   `xau_fx_mt5` guard-(d) suite, 37/37 unchanged).
2. **`mt5_lot` is not a valid `ml4t.backtest.ShareType`** (only `FRACTIONAL`/`INTEGER`). Declaring
   `execution.share_type: mt5_lot` for real (as opposed to the fixture-only usage the helper's own tests used)
   would crash `ShareType(sizing_cfg.get("share_type", "integer"))` at engine-config build time, in **two**
   consumers: `case_studies/utils/backtest_presets.py::build_resolved_backtest_config` (the research sweep) and
   `bots/exness_fx_d1/deploy/deployment_loop.py`'s parity-replay engine (a second, independent construction site
   found while checking for the same class of bug). Fix in both: `mt5_lot` maps to `"integer"` for the ENGINE's own
   share rounding — the lot floor itself is a weights-level guard applied before the engine ever sees the weights,
   not a share-rounding mode the engine needs a new member for. Every other declared `share_type` passes through
   unchanged (test: `test_mt5_lot_share_type_resolves_to_a_valid_engine_share_type`,
   `test_legacy_share_types_pass_through_unchanged`; `bots/exness_fx_d1/tests/test_parity.py` and
   `test_deployment_loop.py` updated to assert the translated value, not reverted).

No change to `case_studies/utils/backtest_runner.py` beyond these two additive branches; no change to any other
case study's registered hashes (verified: `xau_fx_mt5`'s own `execution.share_type: integer` is untouched, its
guard-(d) test suite 37/37 green after the edit).

---

## 1. Real numbers declared, not guessed (`bots/assets/*.md`, verified against `symbol_info` on the demo login
   `Exness-MT5Trial7`, 2026-09-05/07 — `BOT.md` "Contract facts")

| Symbol | `trade_contract_size` | `volume_min` | `volume_step` | `volume_max` | quote currency | base currency |
|---|---|---|---|---|---|---|
| EURUSD | 100,000 | 0.01 | 0.01 | 200.0 | USD | EUR |
| GBPUSD | 100,000 | 0.01 | 0.01 | 200.0 | USD | GBP |
| AUDUSD | 100,000 | 0.01 | 0.01 | 200.0 | USD | AUD |
| USDCAD | 100,000 | 0.01 | 0.01 | 200.0 | CAD | **USD** |
| USDJPY | 100,000 | 0.01 | 0.01 | **300.0** | JPY | **USD** |

`costs.contract.volume_max` is declared as **one global scalar** (the shared helper reads `contract["volume_max"]`
directly, not per symbol) at **200.0**, the tighter of the two real values, rather than 300 (USDJPY's real
ceiling): the ceiling this guard enforces is then never looser than any pair's real one. At this bot's capital
scale (a k=1 leg is a small fraction of one lot; the live side's own `min_viable_allocated` is 2,400 USD) no leg
comes close to either ceiling, so this binds nothing today. Recorded so a future reader does not read 200 as
USDJPY's own number.

---

## 2. K accounting

```
K before this declaration (BOT.md, Phase status row 5, 2026-09-07)     = 3,204
  = 1,068 long-only top-k       (exness_fx_d1:equal-weight-baselines gen 1, b57aa4a93f72, superseded, frozen)
  + 1,068 long-short sleeves    (exness_fx_d1:equal-weight-baselines gen 2, 52365960d8ca, current tip -> re-run)
  + 1,068 per-pair time-series  (exness_fx_d1:timeseries-percentile-baselines gen 1, 1836e5594858, current tip -> re-run)

Generation-3 population, declared before either fit runs:
  equal-weight-baselines generation 3  (long_short=True, the SAME spec as generation 2's tip,
                                          re-run only because share_type/account moved)
    = 534 prediction sets x top_k {1, 2} = 1,068
  timeseries-percentile-baselines generation 2  (long_only, the SAME spec as generation 1's tip,
                                          re-run only because share_type moved -- the lot floor
                                          applies to it too, the account model does not since it
                                          never exceeds 100% gross)
    = 534 prediction sets x lookback_days {63, 252} = 1,068
  => generation-3 population = 1,068 + 1,068 = 2,136

K after this run = 3,204 + 2,136 = 5,340
```

**Not re-run, and why.** The original long-only top-k generation (`b57aa4a93f72`) is already superseded within its
own population by generation 2 (long-short); it stays frozen, historical, immutable, and is already counted in
K = 3,204 (and will be re-deflated at K = 5,340 alongside every other generation, per
`_report_phase5.py`'s own rule: "K accumulates over the bot ... walks every generation of every population"). It is
not economically affected by `allow_leverage` (it never exceeds 100 % gross) and re-running it under new engine
settings would add a third trial family to answer a question nobody asked (OQ 32 only approves fixing the
long-short cohort's cash-account defect and applying the lot floor everywhere). Re-running it is **not** ruled out
for a future round; it is simply not part of this declaration, and if a future round wants it, that is its own
+1,068 to declare before running.

**Alternative reading, recorded and rejected.** A stricter reading of "run the `13_backtest` population (all three
cohorts as generation 2 did)" could mean re-running the long-only top-k too, for a generation-3 population of
3,204 and K after = 6,408. Rejected here because (i) the long-only cohort's own hash is not what OQ 32 was written
to fix, (ii) reviving an already-superseded spec as a new generation of the same population without a matching user
decision would be a trial spent on a question the declaration does not ask, and (iii) the two-population reading
(2,136) is the minimal set whose hashes are stale for a *reason OQ 32 names* — the account model for the long-short
family, the lot floor for both currently-tip families. If the mentor or the user wants the wider reading, that is a
one-line change to `SIGNAL_FAMILY`/`long_short` in a third `13_backtest.py` invocation and +1,068 to declare first.

**DSR gate unchanged**: `confidence_level = 0.95`, `is_significant = probability >= 0.95`, reported on the active
return (strategy minus the 1/N long book of the five pairs) and vs `SR = 0`, both the notebook formula
(`16_strategy_simulation/12_dsr_validation.py`) and `ml4t.diagnostic.evaluation.stats
.deflated_sharpe_ratio_from_statistics`, at the new cumulative K for every winner (this generation's and every
earlier generation's, re-deflated). **Holdout untouched** — `2025-09-01 .. 2026-08-31` is not read by this
declaration or the run it authorizes.

---

## 3. Files changed (this declaration)

| File | Change |
|---|---|
| `case_studies/utils/backtest_runner.py` | `_apply_lot_floor_if_declared`: additive `account_currency_is_base_for` branch (fix 1 above) |
| `case_studies/utils/backtest_runner.py` (second patch, **2026-09-10T14:23Z, in code before generation 3's restart and therefore part of the generation this ran under**) | `_apply_lot_floor_if_declared`'s `weights.join(prices)`: cast the join key to the price frame's `Datetime` time unit before joining (`weights` carried `Datetime(us)`, `prices` `Datetime(ms)` — the first real exercise of this join on production data; generation 3's first launch, 14:08:30Z, crashed on this exact `SchemaError` 11 minutes in, registry unaffected, 0 rows written), then cast the OUTPUT back to the original `weights` dtype so callers see no change. Test: `test_mismatched_datetime_time_units_do_not_raise_and_output_keeps_the_weights_dtype`. Generation 3 (both legs, restarted 14:23:49Z) ran entirely under this patch; a frozen generation's manifest is not complete without it |
| `case_studies/utils/backtest_presets.py` | additive `mt5_lot -> integer` ShareType translation (fix 2 above; superseded for the NEXT run by the fleet-gate addendum's `-> fractional`, §B above — generation 3 itself ran under `integer`) |
| `case_studies/exness_fx_d1/config/backtest/base.yaml` | `account.allow_leverage: true`, `initial_margin: 0.0005`, `long_maintenance_margin: 0.00025`, `short_maintenance_margin: 0.0003` |
| `case_studies/exness_fx_d1/config/setup.yaml` | `execution.share_type: mt5_lot`; new `sizing.min_lot_policy: reject`; new `costs.contract.{volume_min, volume_step, volume_max, account_currency_is_base_for}` |
| `case_studies/exness_fx_d1/13_backtest.py` | parameters cell: `SUPERSEDES_EQUAL_WEIGHT_BASELINES = "52365960d8ca"` (was `"b57aa4a93f72"`), `SUPERSEDES_TIMESERIES_BASELINES = "1836e5594858"` (was `""`); two markdown addenda documenting the account-model/lot-floor change (history not rewritten) |
| `bots/exness_fx_d1/deploy/deployment_loop.py` | additive `mt5_lot -> integer` translation at the parity-replay engine's own `ShareType(...)` call (fix 2, second consumer; superseded for the NEXT run by `-> fractional`, §B) |
| `bots/exness_fx_d1/tests/test_backtest_costs.py` | `config.share_type` assertion updated `integer -> mt5_lot` |
| `bots/exness_fx_d1/tests/test_parity.py` | engine/`research.share_type` comparison updated to go through the same translation (later re-updated for `fractional`, §B) |
| `bots/exness_fx_d1/tests/test_deployment_loop.py` | `engine["share_type"]` assertion updated to match the translated value (later re-updated for `fractional`, §B) |
| `bots/exness_fx_d1/tests/test_lot_floor_currency.py` | **new**; grew from 11 tests (this declaration) to 22 across the same-day fleet-gate rounds (§B/§C additions: dtype-mismatch, fail-loud coverage/currency (C(i)), no-price-raises (C(ii)), hash-inclusion, tautological-test fix) |
| `case_studies/utils/backtest_loaders.py` | **new** (fleet gate item B, after generation 3's own run): `read_mt5_lot_sizing_declaration`, the shared reader `_apply_lot_floor_if_declared` and `build_resolved_backtest_config` both use — not present in the code generation 3 ran under |
| `case_studies/exness_fx_d1/config/setup.yaml` (second patch, fleet gate item C(i), after generation 3's own run) | `costs.account_currency: USD` — not present in the code generation 3 ran under |
| `experiments/xau_fx_mt5/tests/test_execution_guards.py` | `_fake_backtest_config` fixture gained `"account_currency": "USD"` (required by C(i), legacy-preserving) |
| `experiments/exness_fx_d1/exness_fx_d1/config/{setup.yaml,backtest/base.yaml}` (Windows, config-only mirror) | mirrored from the case-study source |
| `~/ml4t/experiments/exness_fx_d1/exness_fx_d1/config/{setup.yaml,backtest/base.yaml}` (WSL, the running workspace) | mirrored from the case-study source — `get_case_study_dir` resolves config reads here when `ML4T_OUTPUT_DIR` is set, `get_case_study_source_dir` (base.yaml preset path) always reads the source repo regardless |
| `bots/exness_fx_d1/GEN3_DECLARATION_2026-09-10.md` | this file |
| `bots/exness_fx_d1/BOT.md` | Decisions log row, Trials row (declared before run), Phase 5 status placeholder, OQ 26/31/32 updated |

**Identity version**: unchanged (`identity_version` is not bumped; the hash inputs that moved — `share_type`,
`account.allow_leverage` — are ordinary spec-hash fields already covered by the existing hashing scheme, not a
schema-level change to what a `backtest_hash` is computed over).

---

## 4. Tests, run before the sweep (WSL, `ML4T_DATA_PATH=/mnt/d/05_Quant/machine-learning-for-trading/data`)

- `bots/exness_fx_d1/tests/test_lot_floor_currency.py` — **11 passed** (new file).
- `bots/exness_fx_d1/tests` (whole suite) — **291 passed**.
- `experiments/xau_fx_mt5/tests/test_execution_guards.py` — **37 passed** (legacy-unchanged proof for the shared
  helper's other adopter; unaffected by anything declared here).
- Engine-config smoke (`build_resolved_backtest_config("exness_fx_d1", ...)`, no registry write):
  `account_type = margin`, `allow_leverage = True`, `initial_margin = 0.0005`, `share_type = ShareType.INTEGER`,
  `validate(warn=False) == []`.

---

## 5. Execution boundary this declaration authorizes

Run `13_backtest.py` twice in WSL2, `ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_fx_d1`, headless papermill, source
commit stamped in `runtime.json`, `register=True` (`RUN_SWEEP=True`), holdout NEVER loaded (`SPLIT="validation"`
throughout, unchanged):

1. `SIGNAL_FAMILY="cross_sectional_top_k"` (equal-weight-baselines generation 3, long-short).
2. `SIGNAL_FAMILY="per_pair_timeseries"` (timeseries-percentile-baselines generation 2, long-only).

Then `_report_phase5.py` at the new cumulative K = 5,340, printing `variance_trials` and `expected_max_sharpe`
before the Sharpe table (BOT.md gate wording), per cohort: count of specs with active-return DSR ≥ 0.95, best spec
hash. **Not opened by this declaration**: phase 6 (`14`-`16`), phase 7 (`17`-`19`), Nautilus confirmation (OQ 32(c),
deferred to §0(c) above). If any generation-3 spec clears the gate, this declaration stops there and reports —
gating is the mentor's job, Nautilus confirmation is the next builder step, neither happens in this round.

---

## 6. Open questions closed or updated by this declaration

- **OQ 32** — items (a) and (b) EXECUTED (this declaration and the run it authorizes); item (c) still deferred,
  restated as a concrete precondition in §0(c) and §5 ("Not opened").
- **OQ 31** (demo login not idle, legacy bot's position) — the legacy V9 bot (magic 202500) was stopped today
  (tasks disabled, process tree terminated, 0 positions on the shared demo login); demo equity 891.15 USD; the
  user will reset it to 10,000 USD (pending, not done as of this declaration). `capital.min_viable_allocated: 2400`
  in `deploy/risk_config.yaml` is unaffected by this declaration (a live/deploy number, not a research-engine one)
  and stays tied to the pending reset — recorded in BOT.md, not changed here.
- **OQ 26** (0.01-lot floor against the declared allocation) — the same floor is now also enforced in the
  **research** sweep (item (b) above), not only in the live dry run; the number itself (1,000 units ≈ 1,160 USD)
  is unchanged.
- **OQ 23** (cash-account one-legged books) — item (a) is the fix this open question asked for; whether it
  actually clears the one-legged share to ~0 is a measurement the generation-3 run reports, not assumed here.
