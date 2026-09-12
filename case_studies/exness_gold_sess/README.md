# exness_gold_sess — gold and silver, two decisions a weekday

A Route B case study for the bot `bots/exness_gold_sess/BOT.md`. Everything that defines it is in
[`config/setup.yaml`](config/setup.yaml); this file says what the study is for, what its history
can and cannot show, and what would stop it.

Forked from `case_studies/exness_fx_d1` (itself a Route B fork of `fx_pairs`), which had already
been adapted to MT5 history, a UTC+0 server clock and a single-code-path `_features.py`. The id is
separate so this bot's registry never mixes with a published baseline.

## Hypothesis (mentor DRAFT — awaiting the user's approval, `bots/exness_gold_sess/BOT.md`)

Gold and silver on this CFD account are priced by the same forces — US real yields and the dollar
(`bots/assets/XAUUSD.md:66`, Ch08 `04_fundamentals_macro_calendar`) — and are correlated at about
0.8 (`bots/assets/XAGUSD.md:140`). So this bot does **not** look for a cross-sectional ranking
edge: two names do not make a cross-section. Its edge, if it has one, is (i) **intraday
structure** — flow concentrates at the London open and the New York open, and the range and the
spread of those sessions differ from Asia's (`bots/assets/XAUUSD.md:146`), so a forecast of the
return from the session snapshot to the session close may carry information a daily forecast
blurs — and (ii) **mean reversion of the gold/silver ratio** when one leg moves first
(`bots/assets/XAGUSD.md:128`, Ch08 `03`). On the other side is order flow that must trade
regardless of price: miner and industrial hedging, ETF rebalancing, and the twice-daily LBMA
price (`bots/assets/XAUUSD.md:89`).

## Market, cadence, labels

| | |
|---|---|
| Instruments | `XAUUSD`, `XAGUSD` (bare names; the Market Watch suffix `m` lives only in `bots/_shared/mt5_loader.py`) |
| Data | MT5 H1 and D1 bars, Exness demo `206539306 @ Exness-MT5Trial7`, server clock UTC+0 (measured) |
| Decision grid | **two a weekday**: the close of the first H1 bar closing at or after (venue open + 30 min), for London and for New York. On a UTC-aligned hourly grid that is open + 60 min in both seasons |
| Execution | next bar open |
| Session filter | `london`, `ny`, and the pooled book. **Not** `overlap`: with these snapshots it selects exactly the `ny` rows, so it would be a duplicate spec and a wasted trial |
| Primary label | **`fwd_ret_sess`** (generation 2, 2026-09-10) — decision close to the last tradable bar close at or before the session close (`_hold.tradable_exit`): 8 bars in London (= `fwd_ret_8h`), 7 in New York (= the generation-1 seven-hour probe). Generation 1 trained on `fwd_ret_8h` — decision close to session close |
| Variants | none in generation 2. Generation-1 labels stay sealed on disk, unrepublished: `fwd_ret_8h` (left the menu: its New York window was never traded), `fwd_ret_24h` (deferred until the real account's swap is read), `dir_tb_8h` (closed after its trial gate failed) — `bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md` section 2 |
| Walk-forward | 4 folds, 3 years training + 1 year validation, on `FX` calendar dates |
| Holdout | `2025-09-01` → `2026-08-31`, declared before the first training run, scored **once** |

## Assumed costs (measured, not quoted)

Measured 2026-09-07 from 30 days of `COPY_TICKS_INFO` ticks through `bots/_shared/costs_mt5.py`
(6.41 M ticks gold, 1.56 M silver), the same call and buckets the five FX pairs of `exness_fx_d1`
were measured with. The quoted spread on this account is a **constant in points** — 260 points on
gold, 30 on silver — in every session bucket and at every percentile of the window.

| | XAUUSD | XAGUSD |
|---|---|---|
| Spread p90, per crossing | 0.60 bps | 4.70 bps |
| **Round trip at the p90** | **1.2 bps** | **9.4 bps** |
| Commission | 0 (Pro account) | 0 |
| Swap (demo) | 0.0 / 0.0 points | 0.0 / 0.0 points |
| Contract, min order in units | 100 oz, 1 oz | 5,000 oz, **50 oz** |

Silver charges 7.8× gold in basis points while its sessions move about twice as far. The
`other` (off-session) bucket is empty on both metals, so unlike the FX pairs there is no
expensive out-of-session bucket to avoid. Swap is a **demo** reading and must be re-read on the
real Pro account before `16_costs`; it is deliberately not part of any label.

## What this history can and cannot show

**Depth, measured (task B0, 2026-09-07).** H1 back to **2017-02-27** on *both* metals — 56,123
dense bars on gold and 56,103 on silver — read with a count-based `copy_rates_from` request. The
"2022-10-25" recorded before that was the terminal's chart cache, not the account: a calendar-range
request returns only the bars the terminal had already synchronised. Bars before 2017-02-27 exist
but are six a week, a D1 backfill served on the H1 timeframe, and are excluded by
`universe.history_start`.

**It can show**: that the point-in-time construction holds; the relation between cost and session
range per venue and per metal; an information-coefficient estimate with a HAC standard error at
the measured label overlap; and a breakeven cost.

**It cannot show**: evidence across monetary regimes. Gold trends up through most of 2017–2026, so
a book tilted long looks good on raw return for that reason and not because of a model. Every
statistic must therefore be reported on the **active** return over a 1/N long book of the two
metals as well as against SR = 0 — the same rule `exness_fx_d1` adopted (`BOT.md:139`). And the
minimum track record needed to resolve a small edge is far longer than what is observed:
`exness_fx_d1` needed 6,081 sessions against 1,011 observed (`BOT.md:83`).

**Contamination.** The legacy bot `v9_continuum` traded `XAUUSD` (not `XAGUSD`) on this same login
until 2026-09-05, so its behaviour over the holdout window has been seen by a human.
`evaluation.legacy_v9_contaminated_window` declares it: the holdout may **confirm**, it may not
**select**, and no parameter of this bot may be chosen because it resembles a v9 rule.

## Kill criteria (mentor DRAFT — awaiting the user's approval)

Pause when: drawdown from peak > 8 % of the bot's allocated capital; or the rolling 63-decision
hit rate of the running book is below 0.5 for 21 consecutive sessions; or the realised round-trip
cost exceeds 1.5× the cost in `setup.yaml::costs` for a calendar month; or reconciliation finds an
unexplained position carrying magic `260903`; or the decision bar is missing within tolerance (no
order that session); or the realised swap exceeds the declared value for a week; or an instrument
halts or gaps across the daily 21:00–22:00 UTC break while a position is open.

Retire when the latest breakeven estimate from `16_costs` falls below the measured p90 round trip
of the traded session, or when the holdout PSR against the 1/N long book is below 0.5.

## Pipeline

| Stage | Status | What it does |
|---|---|---|
| `01_feasibility_analysis.py` | present | resolves the session rule to decision instants and checks it; cost against move; IC\*; folds |
| `02_labels.py` | present | the three labels, the 24-hour guard, the session-cut triple barrier, overlap and N_eff, baseline IC |
| `03_financial_features.py` | present | the six built feature families on two grids (H1 in-session, D1 asof-joined on the bar close) |
| `04_model_based_features.py` | present | Kalman, per-metal HMM and ARIMA on the decision grid, one fold set, digest `4b1a0239330a0e7d` |
| `05_evaluation.py` | present | the pooled-panel IC triage of 43 candidates, per-book staleness, HAC and BH control |
| `06_linear.py`, `07_gbm.py`, `12_model_analysis.py` | present, **run 2026-09-08** (clean re-fit after the `dir_tb_8h` leak and the `kalman_smoothness` removal) | 216 training runs, **419** validation prediction sets (178 `fwd_ret_8h` / 178 `fwd_ret_24h` / 63 `dir_tb_8h`); `12` exits 0 on the full menu; `dir_tb_8h` trial gate (2) evaluated and **failed** (+0.02386 <= +0.03055), its 378 trials SPENT; 0 of 419 sets exceed the minimum detectable IC 0.054 (`bots/exness_gold_sess/BOT.md`) |
| `13_backtest.py` | present, **run 2026-09-08, guard amended 2026-09-09, re-run verified green 2026-09-10** (0 computed, 1,674 served, population `2f37b3b51a42`, `require_complete()` green; the 2026-09-09 attempt was interrupted before its run cell and verified nothing) | two engine books (`london`, `ny`) over every complete validation prediction set, `TimeExit(HOLD_BARS)` in every spec, Friday rule on the 24-hour label, early closes printed and traded. **1,674** registered engine backtests in population `2f37b3b51a42` (supersedes the incomplete `cf0cb9ef5f25`), 3 named unrunnable identities, K = 2,514. **Gate D5.3 evaluated 2026-09-08: 0 of 2,514 specs have DSR >= 0.95 on the active return** (`member_digest 765830e9fe95a8e06167ca89317282c02e6f9dcaf4e00685bf1bdcc5817e287b`); phase 6 NOT open |
| `14_portfolio_management.py` … `19_strategy_analysis.py` | not forked; **phase 6 NOT OPEN** (D5.3 gate not met, 2026-09-08/09) | phase 6 onward; `14` is CLOSED for this generation (D6.1); `17`-`19` did not run and the holdout 2025-09-01 -> 2026-08-31 is **unburnt** (0 `prediction_sets` with `split == 'holdout'`); `19` is also where the pooled book is built as a 50/50 sleeve sum |

`_features.py` is the single implementation of the decision grid, the session panel and the
feature families; every stage, the backtest price loader and the point-in-time test import it
rather than carrying a copy.

**The registered price grid is not the session panel.** Since 2026-09-08
(`bots/exness_gold_sess/PRICE_GRID_DECLARATION.md`) `_PRICE_CONFIG["exness_gold_sess"]` names the
loader `exness_gold_sess_h1`: the raw MT5 H1 tape of both metals, keyed on the **bar close**
(`timestamp + 60 min`), uncropped. Every decision instant is therefore a row of the grid, and the
holding period is expressed by a broker-level `TimeExit` position rule inside every backtest spec -
not by the grid spacing and not by `slot_strategy`, whose exit could never execute here because the
engine only acts at a prediction timestamp. `load_session_panel(for_backtest=True)` still raises,
as the guard against registering the panel again.

## Running it

```bash
uv run python scripts/create_experiment.py --cs exness_gold_sess --output experiments/exness_gold_sess
ML4T_OUTPUT_DIR=experiments/exness_gold_sess uv run python case_studies/exness_gold_sess/01_feasibility_analysis.py
ML4T_OUTPUT_DIR=experiments/exness_gold_sess uv run python case_studies/exness_gold_sess/02_labels.py
```

`run_log/` starts empty and no baseline is ever downloaded into it. The two repository-pinned keys
`labels.rebalance_step` and `labels.classification_eval_label` were authored on 2026-09-07 while it
was empty and may not be edited now that it is not.
