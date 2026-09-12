# Case Study: exness_usidx_sess (Exness MT5, US index CFDs, two decisions per cash session)

Research side of the bot `exness_usidx_sess` (contract and memory:
[`bots/exness_usidx_sess/BOT.md`](../../bots/exness_usidx_sess/BOT.md); instrument profiles:
[`bots/assets/US500.md`](../../bots/assets/US500.md), [`bots/assets/USTEC.md`](../../bots/assets/USTEC.md)).
A Route B fork of [`case_studies/exness_fx_d1`](../exness_fx_d1/README.md), which is itself a fork of
[`fx_pairs`](../fx_pairs/README.md). Four things change:

1. **Two decisions per session, not one.** The New York cash session is split into an intraday leg and an overnight
   leg (Chapter 8 section 8.1), and each leg is its own label, its own panel and its own spec.
2. **Two instruments, so no cross-section.** `fx_pairs` and `exness_fx_d1` rank five pairs against each other;
   ranking two indices is a spread trade. `mapping.class` is `time_series_threshold` and each index is compared with
   its own score distribution (Chapter 16 `04_single_asset_ml4t_backtest` / `05_stateful_strategies`).
3. **The bar is H1 and the calendar is `NYSE`**, including its half-days, instead of H4 on `CME_FX`.
4. **Financing is a material cost.** A long index position pays about 2 bps a night on this account and a short pays
   nothing; the FX demo was swap-free.

Hypothesis, kill criteria and the measured cost table are owned by `BOT.md`; this README only describes the pipeline.
Both are drafts awaiting the user at the time of writing.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | US index CFDs on Exness MT5 (`US500` `USTEC`, bare names; the Market Watch suffix `m` is handled in the loader) |
| Frequency | Two decisions per `NYSE` cash session, both on the H1 grid and both derived from the calendar: the cash open plus `decision.open_delay_minutes` (15:00 UTC winter, 14:00 summer) and the cash close (21:00 / 20:00 UTC, 18:00 / 17:00 on a half day) |
| Universe | 2 indices; no `top_k` grid, a threshold on each index's own score |
| History | H1 from 2022-10-25 on both symbols (22,554 / 22,553 bars, `history_depth.json`, Exness-MT5Trial7, read 2026-09-05); D1 from 2019-07-16. The H1 depth is the server's, not a download limit (`terminal_info.maxbars` is 100,000) |
| Labels | `fwd_ret_intraday` (primary) and `fwd_ret_overnight`; together they are the whole session and they do not overlap |
| CV Folds | 3 x 18M train / 6M val, holdout 2026-03-01 to 2026-08-31 (declared 2026-09-08 from the measured H1 depth, before any training run) |
| Cost Model | Material: spread + swap (Pro account, no commission). Spreads measured 2026-09-08 from 30 days of this account's own ticks (2.79 M US500 + 19.69 M USTEC), swap read from `symbol_info` the same day |

## Data

`bots/_shared/mt5_loader.py:download_mt5_bars` reads D1 / H4 / H1 bars from a logged-in terminal and writes
`ML4T_DATA_PATH/mt5/{daily,4h,1h}.parquet` plus `history_depth.json`.
`bots/exness_usidx_sess/tools/measure_index_costs.py` measures the spread by session bucket for these two symbols
and appends them to `mt5/spreads_by_session.{parquet,json}` without touching the rows the other bots measured.

Three facts about this instrument were measured rather than assumed, and each of them contradicts something a
reasonable person would have hard-coded:

- **The server clock does not follow daylight saving; the instrument's trading hours do.** The clock is UTC+0 all
  year (measured 2026-09-05). The daily Globex break ends at 18:00 New York in every readable month of the sample,
  which puts it on 21:00 UTC in summer and 22:00 UTC in winter. `decision.trade_hours_follow_dst_of` is that claim
  and `01_feasibility_analysis` asserts it on the bars on every run. `sessions_mt5.json` is an eight-week summer
  snapshot and must not be used for this.
- **The daily break used to be two hours and used to sit on the cash close.** Through 2022 it ran 16:00-18:00 New
  York; from 2023 it is 17:00-18:00.
- **The index shuts for the weekend at the Friday cash close.** Together with the point above, that leaves the
  `overnight` spec with no execution bar on 26 % of its index-sessions over the whole sample - 98 % in 2022, 55 % in
  2023, 7 % in 2025 and 0 % in 2026. The rule is **no bar within one bar of the decision, no order**.

## Pipeline

`01_feasibility_analysis` and `02_labels` are in place. `03` onward are phase 3 and later.

The two session panels and the feature construction live in one importable module, [`_features.py`](_features.py),
so the notebooks, the backtest price loader (`case_studies/utils/backtest_loaders.py`, loader
`exness_usidx_sess_h1`), the point-in-time tests (`bots/exness_usidx_sess/tests/`) and, later, the deployment loop
compute the same numbers.

Each panel row carries three groups of columns and they are deliberately not merged:

| Group | Columns | What it is |
|---|---|---|
| decision window | `open` `high` `low` `close` `volume` | every bar that closed after the previous decision instant and at or before this one; `close` is the decision bar's close, so every feature built on it is knowable at `decision_ts` |
| execution | `exec_ts` `exec_open` | the first bar opening within one bar of the decision, and its open - the price the order fills at. Null means no order |
| label | `label_ts` `label_close` | where the position is unwound. `02_labels` seals `fwd_ret_<spec> = label_close / exec_open - 1` |

The mentor's work order described a panel whose `open` is the fill price and whose `close` is the label endpoint.
That shape cannot carry the features: `build_features` reads `close` with rolling windows that include the current
row, so a `close` dated after the decision would put the label's own endpoint inside every momentum column.

## The one place the two specs differ

`intraday_ret(d)` finishes at the cash close. The `overnight` spec decides there and may read it; the `intraday`
spec decides six hours earlier and may not, so on that panel the column is shifted one session. `overnight_ret(d)`
finishes at the cash open and both may read it. The batch panel cannot show the difference; the point-in-time test
can, because a rebuild from the bars that had closed at the decision instant returns null where the unshifted column
would have had a value.

## Registrations

`exness_usidx_sess` is registered in `case_studies/utils/backtest_loaders.py` (`ALL_CASE_STUDIES`, `_PRICE_CONFIG`,
the `exness_usidx_sess_h1` loader branch and the `indices` class of `_cfd_pair_class`), `case_studies/utils/analytics.py`
(five dictionaries) and `case_studies/utils/conformal.py` (one embargo step per label). `tests/overrides.yaml` skips
both stages in CI: they need MT5 history that no fixture carries.

`_cfd_pair_class` needed the new class because neither `US500` nor `USTEC` contains `USD`; without it both resolve
to `cross_pairs` and an index universe is silently priced at the FX crosses' 3-8 bps against a measured 0.66 / 0.39.

## What is not here

- No VIX instrument exists on this account and this repository carries no point-in-time FOMC calendar, so two
  feature families `bots/assets/US500.md` section 7 names cannot be built. Recorded as a known limitation.
- Windows longer than about 60 sessions cannot be warmed up on the H1 grid. They have to come from the D1 parquet
  (2019-07-16 onward) joined onto the session grid at the last D1 close at or before the decision instant, which is
  phase 3 work.
- The spread and the swap were measured on a **demo** account. `16_costs` is blocked until the real Pro account is
  read (the same user decision that blocks `exness_fx_d1`).
