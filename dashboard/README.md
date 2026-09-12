# dashboard

A local, read-only HTML operator dashboard for the 4 Exness bots in this workspace
(`exness_btc_8h`, `exness_fx_d1`, `exness_usidx_sess`, `exness_gold_sess`). It replaces
`runners/check_fleet_status.py`'s log tail with one page: evidence status, execution
mode, last run, run history, next decision instants, Task Scheduler state, circuit
breakers, and a traffic light per bot - plus a fleet header (account halt, high-water
mark, optional MT5 snapshot).

**None of these four bots is deployable.** All four closed Phase 5 with 0 survivors
(see each bot's `BOT.md` Trials table). This dashboard never prints a PnL or "profit"
figure derived from a run these bots' own `risk_config.yaml` withholds pending an
unscored holdout - there is no such tile on the page. It shows `shadow_mode`,
`execution_mode`, `pending_user_approval` and the mentor's verdicts verbatim instead.

Design pattern borrowed from `D:\05_Quant\v9_dashboard` (a local read-only dashboard for
the retired V9 bot): a Python script builds a static `dashboard.html`, never writes
outside its own folder, never calls `order_send`, never launches the MT5 terminal, and
only ever reads MT5 read-only after an OS-level check that a terminal is already
running. The *schema* here is new (4 bots, 4 different state layouts, evidence/K/verdict
tracking); the *safety design* is the same.

## What it reads

- `bots/<bot>/deploy/risk_config.yaml` - YAML (`yaml.safe_load`). Never imported.
- `bots/<bot>/deploy/state/**` - the **latest** JSON run/state record, by file mtime.
  The layout differs per bot and is declared once, in `BOT_SPECS` in
  `build_dashboard.py`:
  - `exness_btc_8h`: `state/run_*.json`
  - `exness_fx_d1`: `state/<date>/run_*.json`
  - `exness_usidx_sess`: `state/<spec>/<date>/state.json`, `spec` in `{intraday, overnight}`
  - `exness_gold_sess`: `state/runs/<timestamp>.json`

  "Latest" is always by file **mtime**, never by the date embedded in the path - a
  bot's `decision_date` can lag real wall-clock time (stale market data), and the most
  recently *written* record is what an operator wants, not the one with the
  alphabetically largest date string.
- `bots/<bot>/monitor/circuit_breakers.py` - TEXT, regex for `class X(CircuitBreaker)`
  names only. **Never imported.** `exness_btc_8h` has no `monitor/` directory at all -
  its breakers are declared directly in `risk_config.yaml::breakers`, same as the other
  three; `monitor_config.yaml` (where it exists) holds drift/rollout thresholds, not
  breaker thresholds - see the comment at the top of each bot's own
  `monitor/monitor_config.yaml` for why (breakers are the kill criteria and live next to
  `LiveRiskConfig` in `risk_config.yaml`, not next to the drift monitor).
- `bots/<bot>/BOT.md` - TEXT, **best-effort regex fallback only**, used only when
  `risk_config.yaml` and the latest state record do not already carry a structured
  trial count / survivor count / holdout window. See "Where the evidence numbers come
  from" below - this is the part of the dashboard most worth treating with suspicion.
- `logs/<bot>.log` - TEXT, split into per-invocation blocks on the
  `[<timestamp>] Running <bot>` header line every `runners/run_*.bat` writes, then the
  last N blocks are parsed for exit code, leg/spec, and skip/error text.
- `D:\05_Quant\merg\data_lake\reviews.jsonl` - **READ ONLY**, JSONL, the latest entry
  per bot (by `ts`, tolerant of malformed lines).
- `D:\05_Quant\machine-learning-for-trading\data\mt5\monitor\account_halt.json` -
  **existence alone** means the fleet is halted; content is best-effort parsed for a
  `reason`/`halted_at` but a halt is reported even if the file is empty or corrupt.
- `D:\05_Quant\machine-learning-for-trading\data\mt5\monitor\account_hwm.json` - JSON,
  tolerant.
- `runners/session_gate.py` - **imported** via `importlib` (never
  `bots/<bot>/deploy/deployment_loop.py`, which may write state or, if armed, orders).
  Only `LEGS`, `SESSIONS`, and `decision_utc()` are used, to compute gold/usidx's next
  decision instants for the next 5 weekdays with the same DST-aware logic the runner
  `.bat` files use to decide whether to skip a cycle.
- `bots/_shared/__init__.py` - **imported** for `MAGIC_ALLOCATION` / `RESERVED_MAGICS`.
  This module is pure dict/function definitions with no I/O and no `.env` read (unlike
  `config.settings` in the v9 dashboard's target, which is why v9_dashboard parses that
  one as text instead of importing it - see its README). If the import ever fails for
  any reason, the dashboard falls back to a transcribed copy and says so in an amber
  banner rather than silently using stale numbers.
- Windows Task Scheduler, via `Get-ScheduledTask` / `Get-ScheduledTaskInfo`
  (PowerShell, no admin rights needed) for each of the 4 tasks
  (`Exness_Bot_BTC_8h`, `Exness_Bot_FX_D1`, `Exness_Bot_USIDX_Sess`,
  `Exness_Bot_Gold_Sess`): state, last run time, last result, next run time.
- MT5, **read-only, only with `--mt5`** (default OFF): `mt5.initialize(path=...)` -
  **always with an explicit path**, resolved by `resolve_single_terminal_path()` -
  `account_info()`, `positions_get()`, `terminal_info()`, `shutdown()`. See "Never
  launches the terminal" below for how the path is resolved and why a mismatch (0 or
  more than 1 terminal running) refuses rather than guesses.

## What it never does

- Never writes anywhere outside `D:\05_Quant\ML_4_Bot\dashboard\` - `atomic_write_text()`
  resolves its target path and **raises** if it is not inside this folder. This is a
  runtime assertion, not just a docstring promise (see
  `dashboard/tests/test_build_dashboard.py::TestGuards::test_atomic_write_refuses_outside_dashboard_dir`).
- Never imports `bots/<bot>/deploy/deployment_loop.py` (it may write state and, if
  armed, place an order) or any of the four bots' `monitor/circuit_breakers.py` /
  `monitor/retire.py` modules (read as text only).
- Never writes to `bots/**/deploy/state/**`, `run_log/registry.db`, any holdout file,
  or `data_lake/reviews.jsonl` (read only, never modified).
- Never reads any `.env*` file, and never imports anything that would (`load_dotenv`
  is never called - see `dashboard/tests`'s guard test).
- Never calls `mt5.order_send`, `mt5.login`, or any order/position-modifying API.
- **Never launches the MT5 terminal, and never calls `mt5.initialize()` with a path it
  has not itself proven is already running.** `resolve_single_terminal_path()`
  (`Get-CimInstance Win32_Process -Filter "Name='terminal64.exe'"`, no admin rights
  needed) requires **exactly one** running `terminal64.exe` process before it will
  return a path at all:
  - `--mt5-path` given: that exact path, only if it is one of the running processes.
  - `--mt5-path` absent: the one path, **only if exactly one** `terminal64.exe` is
    running anywhere. Zero or more than one refuses outright (rendered
    `"n/a: 0 or >1 terminals running"`) - this dashboard does not guess which terminal
    a bot would have used, and an earlier version's "attach to whatever is running, no
    path" behaviour could have attached to the wrong one on a machine running more than
    one terminal (e.g. the retired V9 bot's `D:\05_Quant\MT5_V9\terminal64.exe`).
  Only once one path is proven running does `mt5.initialize(path=<that path>)` get
  called - never with no path, and never with an unconfirmed one.
- Never derives or prints a return, Sharpe, or PnL/profit figure - not even a
  per-position one. The MT5 snapshot's position records carry no `profit` field at all
  (dropped at the source in `read_mt5_snapshot()`, not merely hidden by the renderer),
  and the fleet header shows only account equity/margin and position **counts** by
  magic, because these bots' own `risk_config.yaml` explicitly withholds performance
  figures until an unscored holdout is scored (`evidence_boundary` /
  `retire.holdout_status` in each `risk_config.yaml`).
- **This process never writes a `.pyc` anywhere** (not just inside `dashboard\`):
  `sys.dont_write_bytecode = True` is set at the top of `build_dashboard.py`, before
  the `bots._shared` import and the `importlib` load of `runners/session_gate.py` -
  both of which, without it, wrote bytecode caches into `bots\__pycache__`,
  `bots\_shared\__pycache__` and `runners\__pycache__` (outside this folder) on every
  run. That was a real bug in an earlier version of this file, found in mentor audit;
  it is fixed, not just documented.

## How to run it

```cmd
runners\run_dashboard.bat
```
or the same file directly:
```cmd
dashboard\run_dashboard.bat
```
builds `dashboard\dashboard.html` once (`py -3.12 build_dashboard.py`) and opens it in
the default browser. It does not loop; re-run it (or refresh the page - it carries a
`<meta http-equiv="refresh">` at `--interval` seconds, default 60) to get a new
snapshot.

Direct invocation with the defaults spelled out:

```cmd
py -3.12 dashboard\build_dashboard.py ^
  --root         "D:\05_Quant\ML_4_Bot" ^
  --out          "D:\05_Quant\ML_4_Bot\dashboard\dashboard.html" ^
  --reviews      "D:\05_Quant\merg\data_lake\reviews.jsonl" ^
  --halt-file    "D:\05_Quant\machine-learning-for-trading\data\mt5\monitor\account_halt.json" ^
  --hwm-file     "D:\05_Quant\machine-learning-for-trading\data\mt5\monitor\account_hwm.json" ^
  --interval 60 ^
  --log-tail 10 ^
  --no-mt5
```

Add `--mt5` to enable the read-only MT5 snapshot (off by default; when off, the page
itself says the foreign-magic detector is off, not only this README). Add `--mt5-path
"C:\path\to\terminal64.exe"` to require one specific terminal instead of requiring
exactly one running terminal of any path. `--mt5-timeout` (default 10s) bounds how
long a hung terminal can block the build. `--expected-login` (default `206539306`) and
`--expected-server` (default `Exness-MT5Trial7`, both from
`bots/exness_fx_d1/deploy/risk_config.yaml`'s measured-account comments) are compared
against the connected account when `--mt5` is on; a mismatch renders a red banner and
badge, it does not silently substitute the expected values.

## How to interpret the traffic lights

Computed per bot in `traffic_light()`, in this priority order (first match wins), over
the **whole displayed run-history window** (`--log-tail`, default 10 records), not just
the single latest one:

| Light | Condition |
|---|---|
| 🔴 Red | `account_halt.json` exists (fleet-wide halt), **or** an MT5 position exists whose magic is not in `MAGIC_ALLOCATION`/`RESERVED_MAGICS` (only checked with `--mt5`), **or** the bot's Task Scheduler task is missing/disabled, **or** any exit code in the displayed window is not in `{0, 1, 10}`, **or** `execution.armed` is `true`, **or** `execution.mode`/`live_risk_config.execution_mode` is anything other than `paper`/`shadow`, **or** `live_trading_permitted` is `true` while the holdout is not (yet) `scored` |
| 🟡 Amber | **Any** exit code in the displayed window is `1` (not only the latest one - a bot whose last run happened to succeed after four straight failures is amber, not green), **or** the most recent log line is a `Skipped` (not-a-decision-hour) run, **or** there is no run history to judge at all |
| 🟢 Green | Every exit code in the displayed window is `0` or `10`, the latest is `0`, **and** the Task Scheduler task state is `Ready` |

Every card has a "Vì sao đèn này? (why this light)" `<details>` disclosure listing the
exact reason(s) computed for that bot - the rule is never a black box.

Exit code `10` = the runner's `session_gate.py` said "not a decision hour" and skipped
the cycle without touching the deployment loop (see `README_RUNNER.md` section 2) - a
routine, expected outcome for `exness_gold_sess`/`exness_usidx_sess` most hours of the
day, not a fault. That is why a `Skipped` run is amber, not red, and why its row in the
run-history table is rendered muted rather than in the normal row style.

## Where the evidence numbers come from (read this before trusting a number)

Each bot card's "Bằng chứng (Evidence status)" section shows **three separate
counters**, never one ambiguous "trial count K" (an earlier version of this dashboard
did that, and it made a closed-with-0-survivors generation look identical to "nothing
was ever measured" - a mentor-audit BLOCKING finding):

| Row | What it means |
|---|---|
| **DSR-K** | The count the Deflated Sharpe Ratio actually divides by - a Sharpe-based selection stage (a backtest sweep). Can legitimately be **0** even after hundreds of columns were screened, if nothing reached a Sharpe-producing stage (e.g. `exness_btc_8h`: generation 1 closed at the Phase-4 R1 gate, Phase 5 never opened, DSR-K = 0). |
| **FDR-n** | The size of the multiple-testing family at the IC/screening tier (Benjamini-Hochberg `n`) - a different, usually much larger, number than DSR-K. |
| **K5** | A declared-but-undrawn number for a later stage that never opened (e.g. the backtest population Phase 5 would have spent). |

Plus a **"Phase 5 status"** row quoting the Status column of the bot's own
phase-status table verbatim (`extract_phase5_status()`, the first `"| 5 ... | ... |"`
row found in `BOT.md`) - e.g. `exness_btc_8h`: *"NOT OPENED, generation 1 CLOSED before
this phase"*.

Sources, in priority order, **recorded per counter, not once for the whole card**:

1. The **latest deploy state/run record** (`bots/<bot>/deploy/state/...`), when it
   embeds the field directly (`trial_count_K` -> DSR-K, `phase5_specs_scored` -> FDR-n,
   `phase5_survivors`, `holdout_window`, `holdout_scored`). `exness_fx_d1` and
   `exness_usidx_sess` currently do this - the most reliable source when present.
2. `risk_config.yaml`'s own structured fields (`evidence_boundary.holdout_start/end`,
   `model.trial_count_at_deployment`, `phase5_trial_count`, `retire.holdout_status`).
3. The bot's own **`BOT.md` "Trials counted for the Deflated Sharpe Ratio" table**,
   its **last row** plus the summary paragraph immediately after it (K5's mention is
   often in that paragraph, not the table - see `exness_btc_8h`'s "Cumulative K = 0 ...
   K5 = 2,136 ..." sentence). This is a text parse, but of the bot's own authoritative
   ledger, so it is **not** marked unverified.
4. The **latest `reviews.jsonl` entry**'s `blocking`/`notes`/`next_step` text, free-text
   search - **marked UNVERIFIED**.
5. `BOT.md`'s **whole document**, free-text search, last resort - **marked
   UNVERIFIED**. (BLOCKING finding 6: grepping the entire narrative for `K = <n>` can
   land on any generation's superseded interim figure, not the bot's current ledger
   entry - preferring the Trials-table extraction, step 3, over this exists precisely
   to avoid that.)

**An UNVERIFIED number is rendered struck through with an explicit "unverified" badge**
(`_unverified_cell()`), never as a plain trustworthy value - e.g.
`exness_gold_sess`'s DSR-K (no Trials-table heading exists in its `BOT.md` yet) shows
as ~~2514~~ `unverified`, not `2514`.

**Survivors are never reported as a positive number from free text.** All four bots'
Phase 5 gates closed at 0 survivors, and this corpus's prose is full of hyphenated
ordinals immediately before the word "survivor" (`"generation-3 survivor"`,
`"phase-5 survivor"`) whose trailing digit a naive `(\d+)\s*survivor` regex would
happily (and wrongly) read as a count. The text-search fallback therefore only ever
matches the unambiguous literal `0 survivor(s)` and reports "not found" for anything
else - see `ZERO_SURVIVOR_RE` in `build_dashboard.py` and the regression test
`test_build_evidence_never_fabricates_positive_survivor_count_from_hyphenated_prose`.

**A `reviews.jsonl` verdict of `"met" never renders green.`** `"met"` means the
*process gate* passed (numbers verified, procedure followed) - it says nothing about
whether an edge exists, and every bot in this fleet has closed Phase 5 with 0 survivors
regardless of how many of its process gates were `"met"`. `verdict_badge()` renders
`"met"` as a neutral badge labelled *"phase gate met (process), not evidence of an
edge"*; no verdict value ever gets the green badge class.

`exness_usidx_sess` has **zero** entries in `reviews.jsonl` as of this writing; its
evidence card says so plainly (`"n/a — không có review nào cho bot này"`) rather than
inventing one.

## Decision date vs. run time

The "Last run" card also shows a **"Decision date (record's own)"** row -
`extract_decision_date()` reads the record's own `decision_date` field (top-level, or
nested at `steps.2_features.decision_date` for `exness_fx_d1`'s layout) - separately
from "Thời điểm (finished at)", the wall-clock time the cycle *ran*. The two can differ
by days when market data is stale (a cycle can run today against a decision bar from a
much earlier date); conflating them would hide that staleness.

## Files

- `build_dashboard.py` - the builder. Python 3.12, stdlib + PyYAML (+ `MetaTrader5`
  imported inside a `try/except`, optional, only used with `--mt5`).
- `run_dashboard.bat` - builds once, then opens `dashboard.html`. Identical to
  `runners\run_dashboard.bat` (that one just `call`s this one).
- `dashboard.html` - the generated output (not meant to be hand-edited; regenerated
  every run; overwritten atomically via a `.tmp` file + `os.replace`).
- `tests/test_build_dashboard.py` - pytest suite (79 tests): one class per parser
  (log tail, each of the 4 state layouts, `reviews.jsonl` latest-per-bot,
  `risk_config.yaml` flag extraction, the three-counter/unverified evidence logic,
  verdict-badge neutrality, Task Scheduler JSON parsing, `resolve_single_terminal_path`
  monkeypatched process-listing, the footer safety summary, the traffic-light rules
  including the whole-window exit-code and armed/execution-mode/live-permission red
  conditions, decision-instant math) plus a `TestGuards` class asserting the module
  never calls `order_send(`/`mt5.login(`, never imports `deployment_loop`, never
  imports `dotenv`, sets `sys.dont_write_bytecode` before its local imports, and cannot
  write outside its own folder. Run with:
  ```cmd
  py -3.12 -m pytest dashboard\tests -q
  ```

## Assumptions worth flagging

- The `[<timestamp>]` header each `run_*.bat` writes via cmd.exe's `%DATE% %TIME%` is
  shown **verbatim, labelled "machine clock"** - it is not reinterpreted as UTC or VN
  time, because that depends on the host OS's locale/timezone setting, which this
  script has no way to verify from inside a log file.
- Windows Task Scheduler's "never run" sentinel `LastRunTime` has been observed on this
  host as a `1999-11-30` placeholder rather than .NET's usual `1601` epoch. The
  dashboard trusts `LastTaskResult == 267011` (`0x41303`, "task has not yet run") to
  decide whether to show "has not run yet" instead of trusting any particular sentinel
  year in `LastRunTime`.
- `exness_btc_8h` has no `bots/exness_btc_8h/monitor/` directory; its breakers are read
  from `risk_config.yaml::breakers` exactly like the other three, it just has no
  separate `circuit_breakers.py` to also list class names from.
- The footer's safety summary (armed / execution-mode badges) is **derived fresh from
  each bot's `extract_exec_mode()` dict every build** (`render_footer_safety_summary()`)
  - it is not a hardcoded sentence. If any bot's `risk_config.yaml` ever declares
  `armed: true` or an execution mode outside `{paper, shadow}`, the footer badge turns
  red and names the bot; it does not keep asserting "no bot is armed" regardless of
  what the files say (a mentor-audit BLOCKING finding against an earlier version).
