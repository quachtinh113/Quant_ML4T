# Data census — `exness_gold_sess` H1 panel (XAUUSD, XAGUSD)

Measured **2026-09-08** from `ML4T_DATA_PATH/mt5/1h.parquet` **after task B0 deepened it** (`bots/exness_gold_sess/tools/deepen_metals_h1.py`, count-based `copy_rates_from` on the Exness demo terminal `Exness-MT5Trial7`, login 206539306). It therefore covers the whole development window and **supersedes `data_census_2026-09-07.md`**, which was taken from the terminal's chart cache and reached back only to 2022-10-25 - two thirds of the sample the folds are laid on had never been quality-checked. Experiment-independent: no stage, registry, label or model is involved. Server clock UTC+0 without DST, so server hour == UTC hour. Code at the bottom of this file (`bots/exness_gold_sess/tools/h1_census.py`).

## 1. Coverage of the whole file

| symbol | H1 bars | first H1 (UTC) | last H1 (UTC) | H1 days with bars | D1 | H4 |
|---|---|---|---|---|---|---|
| XAUUSD | 57,094 | 2014-01-14 00:00:00+00:00 | 2026-09-04 20:00:00+00:00 | 3,892 | 3,892 (2014-01-14 00:00:00+00:00 -> 2026-09-04 00:00:00+00:00) | 16,148 (2014-01-14 00:00:00+00:00 -> 2026-09-04 20:00:00+00:00) |
| XAGUSD | 57,079 | 2014-01-12 00:00:00+00:00 | 2026-09-04 20:00:00+00:00 | 3,892 | 3,892 (2014-01-12 00:00:00+00:00 -> 2026-09-04 00:00:00+00:00) | 16,150 (2014-01-12 00:00:00+00:00 -> 2026-09-04 20:00:00+00:00) |

The H1 file now begins in 2014 on both metals, but the first three years are **not an hourly grid**: six bars a week, a daily backfill the server returns on the H1 timeframe. `universe.history_start: 2017-02-27` is the first week both metals print a real hourly grid (`bots/_shared/mt5_loader.dense_history_start`, `min_bars_per_week=100`), and **every section from 2 onward reads only from that date** - the window the folds and the holdout are laid on. Section 1b counts what is excluded.

## 1b. The sparse pre-2017 prefix, and why it is excluded

| symbol | bars before 2017-02-27 | first | last | median bars/week (sparse) | bars from 2017-02-27 | median bars/week (dense) |
|---|---|---|---|---|---|---|
| XAUUSD | 971 | 2014-01-14 00:00:00+00:00 | 2017-02-26 00:00:00+00:00 | 6 | 56,123 | 115 |
| XAGUSD | 976 | 2014-01-12 00:00:00+00:00 | 2017-02-26 23:00:00+00:00 | 6 | 56,103 | 115 |

A session decision needs the bar that closes one hour after the venue opens. At six bars a week that bar does not exist, so the prefix cannot carry a decision at all - it is excluded by `universe.history_start`, not by a filter anywhere downstream. Every section below reads only the dense part.

## 1d. Bars quoting a zero spread

`spread` is the quoted spread in points at the bar open. A bar carrying zero is not a free crossing: it is a bar the server wrote without a spread. `xau_fx_mt5` measured about 1,015 such H1 bars per metal on this account up to 2023-02-03 (`case_studies/xau_fx_mt5/config/setup.yaml:92-95`) and fills them with a point-in-time rolling median rather than pricing them at zero. This bot does **not** read the per-bar spread at all - `costs.spread_bps` comes from the 30-day tick measurement - so the count below is a coverage fact rather than a cost input; it becomes one the moment anyone prices the 2017-2023 stretch off the bar field.

| symbol | dense H1 bars | spread == 0 | share | first | last | by year | median spread (points) |
|---|---|---|---|---|---|---|---|
| XAUUSD | 56,123 | 1,081 | 1.926% | 2017-02-27 00:00:00+00:00 | 2023-02-03 21:00:00+00:00 | 2017:1060, 2019:15, 2021:2, 2023:4 | 200 |
| XAGUSD | 56,103 | 1,074 | 1.914% | 2017-02-27 00:00:00+00:00 | 2023-02-03 21:00:00+00:00 | 2017:1053, 2019:15, 2021:2, 2023:4 | 37 |

## 2. Bars per weekday and server hour — XAUUSD

Number of H1 bars stamped at each (weekday, UTC hour) over the whole history. A zero means the server printed no bar in that slot; a count well below the weekday's modal count means the slot exists in only one DST season.

| weekday | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 491 | 490 | 490 | 489 | 489 | 490 | 491 | 491 | 491 | 491 | 490 | 491 | 491 | 491 | 491 | 491 | 491 | 485 | 465 | 459 | 449 | 148 | 325 | 491 |
| Tue | 493 | 493 | 493 | 493 | 493 | 493 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 492 | 490 | 490 | 170 | 328 | 490 |
| Wed | 492 | 492 | 492 | 492 | 492 | 492 | 492 | 493 | 493 | 493 | 493 | 493 | 493 | 493 | 493 | 492 | 492 | 492 | 490 | 489 | 489 | 167 | 325 | 492 |
| Thu | 492 | 493 | 493 | 493 | 493 | 493 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 493 | 486 | 482 | 480 | 159 | 320 | 483 |
| Fri | 484 | 484 | 483 | 483 | 483 | 483 | 483 | 483 | 483 | 483 | 484 | 483 | 484 | 484 | 484 | 484 | 484 | 480 | 480 | 472 | 470 | 152 | 3 | 0 |
| Sat | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sun | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 264 | 454 |

Hours 00-20 are printed on essentially every Mon-Fri date. Hours **21** and **22** are complementary: exactly one of them is printed on a given weekday, because the one-hour daily break sits at 21:00-22:00 UTC while the US is on DST and at 22:00-23:00 UTC while it is not. Friday ends at 21:00 UTC (no 22:00 or 23:00 bar) and Sunday only re-opens at 22:00 UTC. Saturday is empty.

### 2b. The same, as hour spans — XAUUSD

| weekday | hour spans with bars (UTC) | bars/hour min-max | bars | hours |
|---|---|---|---|---|
| Mon | 00:00-24:00 | 148-491 | 11,161 | 24 |
| Tue | 00:00-24:00 | 170-495 | 11,358 | 24 |
| Wed | 00:00-24:00 | 167-493 | 11,316 | 24 |
| Thu | 00:00-24:00 | 159-494 | 11,294 | 24 |
| Fri | 00:00-23:00 | 3-484 | 10,276 | 23 |
| Sat | - | - | 0 | 0 |
| Sun | 22:00-24:00 | 264-454 | 718 | 2 |

## 2. Bars per weekday and server hour — XAGUSD

Number of H1 bars stamped at each (weekday, UTC hour) over the whole history. A zero means the server printed no bar in that slot; a count well below the weekday's modal count means the slot exists in only one DST season.

| weekday | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 491 | 490 | 490 | 489 | 489 | 490 | 491 | 491 | 491 | 491 | 490 | 491 | 491 | 491 | 491 | 491 | 491 | 485 | 465 | 459 | 450 | 148 | 324 | 491 |
| Tue | 493 | 493 | 493 | 493 | 493 | 493 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 495 | 492 | 490 | 490 | 168 | 325 | 489 |
| Wed | 492 | 492 | 492 | 492 | 492 | 492 | 492 | 493 | 493 | 493 | 493 | 493 | 493 | 493 | 493 | 492 | 492 | 491 | 490 | 489 | 489 | 167 | 324 | 490 |
| Thu | 492 | 493 | 493 | 493 | 493 | 493 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 494 | 493 | 486 | 481 | 480 | 159 | 317 | 482 |
| Fri | 484 | 484 | 483 | 483 | 483 | 483 | 483 | 483 | 483 | 483 | 484 | 483 | 484 | 484 | 484 | 484 | 484 | 481 | 480 | 471 | 470 | 147 | 3 | 0 |
| Sat | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sun | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 264 | 454 |

Hours 00-20 are printed on essentially every Mon-Fri date. Hours **21** and **22** are complementary: exactly one of them is printed on a given weekday, because the one-hour daily break sits at 21:00-22:00 UTC while the US is on DST and at 22:00-23:00 UTC while it is not. Friday ends at 21:00 UTC (no 22:00 or 23:00 bar) and Sunday only re-opens at 22:00 UTC. Saturday is empty.

### 2b. The same, as hour spans — XAGUSD

| weekday | hour spans with bars (UTC) | bars/hour min-max | bars | hours |
|---|---|---|---|---|
| Mon | 00:00-24:00 | 148-491 | 11,161 | 24 |
| Tue | 00:00-24:00 | 168-495 | 11,352 | 24 |
| Wed | 00:00-24:00 | 167-493 | 11,312 | 24 |
| Thu | 00:00-24:00 | 159-494 | 11,289 | 24 |
| Fri | 00:00-23:00 | 3-484 | 10,271 | 23 |
| Sat | - | - | 0 | 0 |
| Sun | 22:00-24:00 | 264-454 | 718 | 2 |

## 3. Missing calendar days

| symbol | calendar days in range | days with bars | days with no bar | by weekday | Mon-Fri holes | the Mon-Fri holes |
|---|---|---|---|---|---|---|
| XAUUSD | 3,477 | 2,921 | 556 | Sat 496, Sun 42, Fri 11, Mon 2, Tue 2, Wed 2, Thu 1 | 18 | 2017-12-25, 2018-01-01, 2018-02-01, 2018-12-25, 2019-01-01, 2019-04-19, 2019-12-25, 2020-01-01, 2020-04-10, 2020-12-25, 2021-01-01, 2021-04-02 ... |
| XAGUSD | 3,477 | 2,921 | 556 | Sat 496, Sun 42, Fri 11, Mon 2, Tue 2, Wed 2, Thu 1 | 18 | 2017-12-25, 2018-01-01, 2018-02-01, 2018-12-25, 2019-01-01, 2019-04-19, 2019-12-25, 2020-01-01, 2020-04-10, 2020-12-25, 2021-01-01, 2021-04-02 ... |

Saturdays are expected (the market is shut). Sundays carry only the 22:00-24:00 re-open. The Mon-Fri holes are the market holidays listed above; each one is a date on which no session decision can be taken.

## 4. Integrity of the H1 bars

| symbol | bars | duplicate timestamps | OHLC invariant violations | zero-volume bars | flat bars (high==low) | null OHLCV | off the :00 grid | tick volume min/median/max |
|---|---|---|---|---|---|---|---|---|
| XAUUSD | 56,123 | 0 | 0 | 0 | 18 | 0 | 0 | 1 / 3,921 / 236,819 |
| XAGUSD | 56,103 | 0 | 0 | 0 | 12 | 0 | 0 | 1 / 839 / 129,993 |

## 5. Do the decision bars exist on every session day?

The London-open bar is 07:00 UTC on BST dates and 08:00 UTC on GMT dates; the New York open bar is 12:00 UTC on EDT dates and 13:00 UTC on EST dates. A **session day** is a Mon-Fri date with at least one bar; a Sunday carries only the 22:00-24:00 re-open and can never hold a session decision, so Sundays are counted separately and excluded from the decision-bar check.

| symbol | days with bars | of which Sunday re-opens | session days (Mon-Fri) | with a London-open bar | missing | with a NY-open bar | missing | summer / winter session days |
|---|---|---|---|---|---|---|---|---|
| XAUUSD | 2,921 | 454 | 2,467 | 2,456 | 11 | 2,457 | 10 | 1,483 / 984 |
| XAGUSD | 2,921 | 454 | 2,467 | 2,456 | 11 | 2,457 | 10 | 1,483 / 984 |

- **XAUUSD** — session days without a London-open bar (11): 2017-04-14, 2018-02-02, 2018-03-30, 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01; without a NY-open bar (10): 2017-04-14, 2018-03-30, 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01

- **XAUUSD** — session days on which the exit price could not be formed: London close 1 (2018-01-31); 21:00 UTC 79 (2017-05-29, 2017-07-04, 2017-09-04, 2017-11-23, 2017-11-24, 2018-01-15, 2018-01-31, 2018-02-19, 2018-05-28, 2018-07-04, 2018-09-03, 2018-11-22, 2018-11-23, 2018-12-24, 2019-01-21, 2019-02-18, 2019-05-27, 2019-07-04, 2019-09-02, 2019-11-28)

- **XAGUSD** — session days without a London-open bar (11): 2017-04-14, 2018-02-02, 2018-03-30, 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01; without a NY-open bar (10): 2017-04-14, 2018-03-30, 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01

- **XAGUSD** — session days on which the exit price could not be formed: London close 1 (2018-01-31); 21:00 UTC 78 (2017-07-04, 2017-09-04, 2017-11-23, 2017-11-24, 2018-01-15, 2018-01-31, 2018-02-19, 2018-05-28, 2018-07-04, 2018-09-03, 2018-11-22, 2018-11-23, 2018-12-24, 2019-01-21, 2019-02-18, 2019-05-27, 2019-07-04, 2019-09-02, 2019-11-28, 2019-11-29)

## 6. Realised session moves vs the measured round trip

Return from the **open of the decision bar** to the price at the session-exit instant (`price_at` = open of that hour's bar, else close of the previous bar), in basis points, absolute value, over the whole H1 history. The round trip is 2 x the in-session p90 spread measured on 2026-09-07 from 30 days of this account's ticks (`data/mt5/spreads_by_session.json`): XAUUSD 0.60 bps, XAGUSD 4.70 bps per crossing. No cost other than the spread is charged here (Pro account, commission 0, measured swap 0.0/0.0).

| symbol | window | n session days | median abs return (bps) | p25 | p75 | p90 | round trip (bps) | median / round trip | share of days above the round trip |
|---|---|---|---|---|---|---|---|---|---|
| XAUUSD | London open -> London close (17:00 GMT / 16:00 BST) | 2,455 | 37.8 | 16.7 | 70.8 | 115.2 | 1.2 | 31.5x | 0.983 |
| XAUUSD | NY open -> 21:00 UTC | 2,378 | 37.2 | 17.7 | 70.1 | 118.2 | 1.2 | 31.0x | 0.986 |
| XAGUSD | London open -> London close (17:00 GMT / 16:00 BST) | 2,455 | 70.3 | 31.5 | 135.4 | 220.8 | 9.4 | 7.5x | 0.930 |
| XAGUSD | NY open -> 21:00 UTC | 2,379 | 71.3 | 32.9 | 129.3 | 222.2 | 9.4 | 7.6x | 0.931 |

Read this as the cost feasibility ratio the phase-1 specification has to beat: a signal has to capture a fraction of the median move larger than one round trip.

### 6b. How often the exit price came from the fallback

| symbol | London close used prev-bar close | 21:00 UTC used prev-bar close |
|---|---|---|
| XAUUSD | 0 | 1582 |
| XAGUSD | 0 | 1590 |

`prev-bar close` means the bar stamped at the exit hour does not exist and the close of the preceding bar (the same instant) was used. For the 21:00 UTC exit this is the summer regime, where the server's daily break is 21:00-22:00 UTC and no 21:00 bar is printed.

## 7. Code used

`bots/exness_gold_sess/tools/h1_census.py`

```python
"""H1 census of XAUUSD and XAGUSD from ``ML4T_DATA_PATH/mt5/1h.parquet``.

Preparation task for phase 0-1 of ``exness_gold_sess`` (2026-09-07). Reads only; writes one
markdown report, ``bots/exness_gold_sess/data_census_2026-09-07.md``. No stage, no registry,
no experiment: these are experiment-independent facts about the raw MT5 panel, used to
decide in phase 1 whether an H1 session bot on these two symbols is feasible at all.

Run it from the repository root with the isolated MT5-side interpreter (the project env does
not build on Windows)::

    PYTHONIOENCODING=utf-8 PYTHONPATH=<repo root> uv run --no-project --python 3.12 \
        --with polars==1.41.1 --with pyarrow \
        python bots/exness_gold_sess/tools/h1_census.py

Conventions used throughout (stated so the numbers can be reproduced):

* ``timestamp`` in the parquet is the **UTC instant of the bar open**; the Exness server
  clock was measured at UTC+0 with no DST (``bots/assets/XAUUSD.md`` section 12), so server
  hour == UTC hour.
* Session hours follow ``bots/assets/XAUUSD.md`` section 2, resolved per date from the real
  time zones rather than a hard-coded table: London open/close = 08:00/17:00 UTC when
  ``Europe/London`` is on GMT, 07:00/16:00 UTC when it is on BST; New York open = 13:00 UTC
  when ``America/New_York`` is on EST, 12:00 UTC on EDT. The New York exit instant is
  **21:00 UTC** in both seasons (the task's definition; in winter the NY session runs one
  hour longer, to 22:00 UTC).
* ``price_at(day, hour)`` = the **open** of the H1 bar stamped ``day hour:00`` when that bar
  exists, else the **close** of the bar stamped ``hour-1`` (the same instant), else null.
  The fallback counts are reported so the reader knows which was used.
* A "trading day" is a calendar date on which the symbol has at least one H1 bar.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("ML4T_DATA_PATH", REPO_ROOT / "data")) / "mt5"
OUT = REPO_ROOT / "bots" / "exness_gold_sess" / "data_census_2026-09-08.md"
SYMBOLS = ("XAUUSD", "XAGUSD")
# universe.history_start: the first week both metals print a real hourly grid. Everything before
# it is a D1 backfill served on the H1 timeframe (six bars a week) and cannot carry a session
# decision, so every measurement below section 1b reads only from this date.
DENSE_START = date(2017, 2, 27)
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")


def london_hours(day: date) -> tuple[int, int]:
    """(open hour, close hour) in UTC for the London session on ``day``."""
    noon = datetime(day.year, day.month, day.day, 12, tzinfo=ZoneInfo("UTC"))
    summer = bool(noon.astimezone(LONDON).dst())
    return (7, 16) if summer else (8, 17)


def ny_open_hour(day: date) -> int:
    noon = datetime(day.year, day.month, day.day, 12, tzinfo=ZoneInfo("UTC"))
    return 12 if noon.astimezone(NEW_YORK).dst() else 13


def load(freq: str) -> pl.DataFrame:
    return pl.read_parquet(DATA_DIR / f"{freq}.parquet").filter(pl.col("symbol").is_in(SYMBOLS))


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:  # noqa: C901 - one linear report
    h1_all = load("1h").with_columns(
        pl.col("timestamp").dt.date().alias("day"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("iso_wd"),  # 1 = Monday
    )
    # Sections 2-6 read the DENSE window only; section 1 and 1b read the whole file.
    h1 = h1_all.filter(pl.col("day") >= DENSE_START)
    d1 = load("daily")
    h4 = load("4h")
    parts: list[str] = []

    parts.append(
        "# Data census — `exness_gold_sess` H1 panel (XAUUSD, XAGUSD)\n\n"
        "Measured **2026-09-08** from `ML4T_DATA_PATH/mt5/1h.parquet` **after task B0 deepened "
        "it** (`bots/exness_gold_sess/tools/deepen_metals_h1.py`, count-based `copy_rates_from` "
        "on the Exness demo terminal `Exness-MT5Trial7`, login 206539306). It therefore covers "
        "the whole development window and **supersedes `data_census_2026-09-07.md`**, which was "
        "taken from the terminal's chart cache and reached back only to 2022-10-25 - two thirds "
        "of the sample the folds are laid on had never been quality-checked. "
        "Experiment-independent: no stage, registry, label or model is involved. Server clock "
        "UTC+0 without DST, so server hour == UTC hour. Code at the bottom of this file "
        "(`bots/exness_gold_sess/tools/h1_census.py`).\n"
    )

    # ---------------------------------------------------------------- 1 coverage
    rows = []
    for s in SYMBOLS:
        p = h1_all.filter(pl.col("symbol") == s)
        pd1 = d1.filter(pl.col("symbol") == s)
        ph4 = h4.filter(pl.col("symbol") == s)
        rows.append(
            [
                s,
                f"{p.height:,}",
                str(p["timestamp"].min()),
                str(p["timestamp"].max()),
                f"{p['day'].n_unique():,}",
                f"{pd1.height:,} ({pd1['timestamp'].min()} -> {pd1['timestamp'].max()})",
                f"{ph4.height:,} ({ph4['timestamp'].min()} -> {ph4['timestamp'].max()})",
            ]
        )
    parts.append(
        "## 1. Coverage of the whole file\n\n"
        + md_table(
            ["symbol", "H1 bars", "first H1 (UTC)", "last H1 (UTC)", "H1 days with bars", "D1", "H4"],
            rows,
        )
        + "\n\nThe H1 file now begins in 2014 on both metals, but the first three years are "
        "**not an hourly grid**: six bars a week, a daily backfill the server returns on the H1 "
        "timeframe. `universe.history_start: 2017-02-27` is the first week both metals print a "
        "real hourly grid (`bots/_shared/mt5_loader.dense_history_start`, "
        "`min_bars_per_week=100`), and **every section from 2 onward reads only from that "
        "date** - the window the folds and the holdout are laid on. Section 1b counts what is "
        "excluded.\n"
    )

    # ---------------------------------------------------------------- 2 weekday x hour
    for s in SYMBOLS:
        p = h1.filter(pl.col("symbol") == s)
        grid = (
            p.group_by(["iso_wd", "hour"])
            .agg(pl.len().alias("n"))
            .sort(["iso_wd", "hour"])
        )
        counts = {(r["iso_wd"], r["hour"]): r["n"] for r in grid.iter_rows(named=True)}
        matrix = [
            [WEEKDAYS[wd - 1]] + [str(counts.get((wd, hh), 0)) for hh in range(24)]
            for wd in range(1, 8)
        ]
        parts.append(
            f"## 2. Bars per weekday and server hour — {s}\n\n"
            "Number of H1 bars stamped at each (weekday, UTC hour) over the whole history. "
            "A zero means the server printed no bar in that slot; a count well below the "
            "weekday's modal count means the slot exists in only one DST season.\n\n"
            + md_table(["weekday"] + [f"{h:02d}" for h in range(24)], matrix)
            + "\n\nHours 00-20 are printed on essentially every Mon-Fri date. Hours **21** and "
            "**22** are complementary: exactly one of them is printed on a given weekday, "
            "because the one-hour daily break sits at 21:00-22:00 UTC while the US is on DST "
            "and at 22:00-23:00 UTC while it is not. Friday ends at 21:00 UTC (no 22:00 or "
            "23:00 bar) and Sunday only re-opens at 22:00 UTC. Saturday is empty.\n"
        )
        rows = []
        for wd in range(1, 8):
            part = grid.filter(pl.col("iso_wd") == wd).sort("hour")
            if part.is_empty():
                rows.append([WEEKDAYS[wd - 1], "-", "-", "0", "0"])
                continue
            hours = part["hour"].to_list()
            counts = part["n"].to_list()
            spans, start, prev = [], hours[0], hours[0]
            for hh in hours[1:]:
                if hh == prev + 1:
                    prev = hh
                    continue
                spans.append((start, prev))
                start = prev = hh
            spans.append((start, prev))
            rows.append(
                [
                    WEEKDAYS[wd - 1],
                    ", ".join(f"{a:02d}:00-{b + 1:02d}:00" for a, b in spans),
                    f"{min(counts)}-{max(counts)}",
                    f"{sum(counts):,}",
                    f"{part.height}",
                ]
            )
        parts.append(
            f"### 2b. The same, as hour spans — {s}\n\n"
            + md_table(
                ["weekday", "hour spans with bars (UTC)", "bars/hour min-max", "bars", "hours"],
                rows,
            )
            + "\n"
        )

    # ---------------------------------------------------------------- 3 calendar gaps
    rows = []
    for s in SYMBOLS:
        p = h1.filter(pl.col("symbol") == s)
        first, last = p["day"].min(), p["day"].max()
        present = set(p["day"].unique().to_list())
        n_days = (last - first).days + 1
        missing = [first + timedelta(days=i) for i in range(n_days)]
        missing = [d for d in missing if d not in present]
        by_wd: dict[str, int] = {}
        for d in missing:
            by_wd[WEEKDAYS[d.weekday()]] = by_wd.get(WEEKDAYS[d.weekday()], 0) + 1
        weekday_missing = [d for d in missing if d.weekday() < 5]
        rows.append(
            [
                s,
                f"{n_days:,}",
                f"{len(present):,}",
                f"{len(missing):,}",
                ", ".join(f"{k} {v}" for k, v in sorted(by_wd.items(), key=lambda kv: -kv[1])),
                f"{len(weekday_missing)}",
                ", ".join(str(d) for d in weekday_missing[:12])
                + (" ..." if len(weekday_missing) > 12 else ""),
            ]
        )
    parts.append(
        "## 3. Missing calendar days\n\n"
        + md_table(
            [
                "symbol",
                "calendar days in range",
                "days with bars",
                "days with no bar",
                "by weekday",
                "Mon-Fri holes",
                "the Mon-Fri holes",
            ],
            rows,
        )
        + "\n\nSaturdays are expected (the market is shut). Sundays carry only the 22:00-24:00 "
        "re-open. The Mon-Fri holes are the market holidays listed above; each one is a date on "
        "which no session decision can be taken.\n"
    )

    # ---------------------------------------------------------------- 4 integrity
    rows = []
    for s in SYMBOLS:
        p = h1.filter(pl.col("symbol") == s)
        dup = p.height - p["timestamp"].n_unique()
        ohlc = p.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
        ).height
        zero_vol = p.filter(pl.col("volume") == 0).height
        flat = p.filter(pl.col("high") == pl.col("low")).height
        nulls = p.select(
            pl.sum_horizontal(
                [pl.col(c).is_null().sum() for c in ("open", "high", "low", "close", "volume")]
            )
        ).item()
        off_grid = p.filter(pl.col("timestamp").dt.minute() != 0).height
        rows.append(
            [
                s,
                f"{p.height:,}",
                dup,
                ohlc,
                zero_vol,
                flat,
                nulls,
                off_grid,
                f"{p['volume'].min():,} / {int(p['volume'].median()):,} / {p['volume'].max():,}",
            ]
        )
    parts.append(
        "## 4. Integrity of the H1 bars\n\n"
        + md_table(
            [
                "symbol",
                "bars",
                "duplicate timestamps",
                "OHLC invariant violations",
                "zero-volume bars",
                "flat bars (high==low)",
                "null OHLCV",
                "off the :00 grid",
                "tick volume min/median/max",
            ],
            rows,
        )
        + "\n"
    )

    # ------------------------------------------------- 5 decision bars and session returns
    detail: dict[str, dict] = {}
    for s in SYMBOLS:
        p = h1.filter(pl.col("symbol") == s)
        bars = {
            (r["day"], r["hour"]): (r["open"], r["close"]) for r in p.iter_rows(named=True)
        }
        days = sorted({d for d, _ in bars})

        def price_at(day: date, hour: int) -> tuple[float | None, str]:
            if (day, hour) in bars:
                return bars[(day, hour)][0], "open"
            if (day, hour - 1) in bars:
                return bars[(day, hour - 1)][1], "prev_close"
            return None, "missing"

        # A Sunday carries only the 22:00-24:00 re-open, so it can never hold a session
        # decision bar. Session days are the Mon-Fri days that have at least one bar.
        session_days = [d for d in days if d.weekday() < 5]
        sunday_days = [d for d in days if d.weekday() == 6]

        stats = {
            "n_days": len(days),
            "n_session_days": len(session_days),
            "n_sundays": len(sunday_days),
            "lon_open_missing": [],
            "ny_open_missing": [],
            "lon_close_fallback": 0,
            "ny_close_fallback": 0,
            "lon_exit_missing": [],
            "ny_exit_missing": [],
            "lon_ret": [],
            "ny_ret": [],
            "lon_summer": 0,
            "lon_winter": 0,
        }
        for d in session_days:
            lo, lc = london_hours(d)
            no = ny_open_hour(d)
            stats["lon_summer" if lo == 7 else "lon_winter"] += 1
            if (d, lo) not in bars:
                stats["lon_open_missing"].append(d)
            if (d, no) not in bars:
                stats["ny_open_missing"].append(d)
            if (d, lo) in bars:
                px, how = price_at(d, lc)
                if how == "prev_close":
                    stats["lon_close_fallback"] += 1
                if px is None:
                    stats["lon_exit_missing"].append(d)
                else:
                    stats["lon_ret"].append(px / bars[(d, lo)][0] - 1.0)
            if (d, no) in bars:
                px, how = price_at(d, 21)
                if how == "prev_close":
                    stats["ny_close_fallback"] += 1
                if px is None:
                    stats["ny_exit_missing"].append(d)
                else:
                    stats["ny_ret"].append(px / bars[(d, no)][0] - 1.0)
        detail[s] = stats

    rows = []
    for s in SYMBOLS:
        st = detail[s]
        rows.append(
            [
                s,
                f"{st['n_days']:,}",
                f"{st['n_sundays']:,}",
                f"{st['n_session_days']:,}",
                f"{st['n_session_days'] - len(st['lon_open_missing']):,}",
                f"{len(st['lon_open_missing'])}",
                f"{st['n_session_days'] - len(st['ny_open_missing']):,}",
                f"{len(st['ny_open_missing'])}",
                f"{st['lon_summer']:,} / {st['lon_winter']:,}",
            ]
        )
    parts.append(
        "## 5. Do the decision bars exist on every session day?\n\n"
        "The London-open bar is 07:00 UTC on BST dates and 08:00 UTC on GMT dates; the New York "
        "open bar is 12:00 UTC on EDT dates and 13:00 UTC on EST dates. A **session day** is a "
        "Mon-Fri date with at least one bar; a Sunday carries only the 22:00-24:00 re-open and "
        "can never hold a session decision, so Sundays are counted separately and excluded from "
        "the decision-bar check.\n\n"
        + md_table(
            [
                "symbol",
                "days with bars",
                "of which Sunday re-opens",
                "session days (Mon-Fri)",
                "with a London-open bar",
                "missing",
                "with a NY-open bar",
                "missing",
                "summer / winter session days",
            ],
            rows,
        )
        + "\n"
    )
    for s in SYMBOLS:
        st = detail[s]
        miss_l, miss_n = st["lon_open_missing"], st["ny_open_missing"]
        if miss_l or miss_n:
            parts.append(
                f"- **{s}** — session days without a London-open bar ({len(miss_l)}): "
                + (", ".join(str(d) for d in miss_l) if miss_l else "none")
                + f"; without a NY-open bar ({len(miss_n)}): "
                + (", ".join(str(d) for d in miss_n) if miss_n else "none")
                + "\n"
            )
        else:
            parts.append(
                f"- **{s}**: both decision bars exist on **every** session day "
                f"({st['n_session_days']:,} of {st['n_session_days']:,}).\n"
            )
        parts.append(
            f"- **{s}** — session days on which the exit price could not be formed: London close "
            f"{len(st['lon_exit_missing'])}"
            + (
                " (" + ", ".join(str(d) for d in st["lon_exit_missing"][:20]) + ")"
                if st["lon_exit_missing"]
                else ""
            )
            + f"; 21:00 UTC {len(st['ny_exit_missing'])}"
            + (
                " (" + ", ".join(str(d) for d in st["ny_exit_missing"][:20]) + ")"
                if st["ny_exit_missing"]
                else ""
            )
            + "\n"
        )

    def q(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        return xs[min(int(p * len(xs)), len(xs) - 1)]

    rows = []
    spread_p90 = {"XAUUSD": 0.60, "XAGUSD": 4.70}
    for s in SYMBOLS:
        st = detail[s]
        for name, key, hours in (
            ("London open -> London close (17:00 GMT / 16:00 BST)", "lon_ret", "close"),
            ("NY open -> 21:00 UTC", "ny_ret", "21:00"),
        ):
            xs = [abs(x) * 1e4 for x in st[key]]
            med = q(xs, 0.5)
            rt = 2 * spread_p90[s]
            rows.append(
                [
                    s,
                    name,
                    f"{len(xs):,}",
                    f"{med:.1f}",
                    f"{q(xs, 0.25):.1f}",
                    f"{q(xs, 0.75):.1f}",
                    f"{q(xs, 0.9):.1f}",
                    f"{rt:.1f}",
                    f"{med / rt:.1f}x",
                    f"{sum(1 for x in xs if x > rt) / len(xs):.3f}",
                ]
            )
    parts.append(
        "## 6. Realised session moves vs the measured round trip\n\n"
        "Return from the **open of the decision bar** to the price at the session-exit instant "
        "(`price_at` = open of that hour's bar, else close of the previous bar), in basis points, "
        "absolute value, over the whole H1 history. The round trip is 2 x the in-session p90 "
        "spread measured on 2026-09-07 from 30 days of this account's ticks "
        "(`data/mt5/spreads_by_session.json`): XAUUSD 0.60 bps, XAGUSD 4.70 bps per crossing. "
        "No cost other than the spread is charged here (Pro account, commission 0, measured swap "
        "0.0/0.0).\n\n"
        + md_table(
            [
                "symbol",
                "window",
                "n session days",
                "median abs return (bps)",
                "p25",
                "p75",
                "p90",
                "round trip (bps)",
                "median / round trip",
                "share of days above the round trip",
            ],
            rows,
        )
        + "\n\nRead this as the cost feasibility ratio the phase-1 specification has to beat: a "
        "signal has to capture a fraction of the median move larger than one round trip.\n"
    )

    fb = md_table(
        ["symbol", "London close used prev-bar close", "21:00 UTC used prev-bar close"],
        [
            [s, detail[s]["lon_close_fallback"], detail[s]["ny_close_fallback"]]
            for s in SYMBOLS
        ],
    )
    parts.append(
        "### 6b. How often the exit price came from the fallback\n\n"
        + fb
        + "\n\n`prev-bar close` means the bar stamped at the exit hour does not exist and the "
        "close of the preceding bar (the same instant) was used. For the 21:00 UTC exit this is "
        "the summer regime, where the server's daily break is 21:00-22:00 UTC and no 21:00 bar is "
        "printed.\n"
    )

    # ------------------------------------------------- 1b the sparse pre-2017 prefix
    rows = []
    for s_ in SYMBOLS:
        p = h1_all.filter(pl.col("symbol") == s_)
        pre = p.filter(pl.col("day") < DENSE_START)
        post = p.filter(pl.col("day") >= DENSE_START)
        weekly_pre = (
            pre.with_columns(pl.col("timestamp").dt.truncate("1w").alias("w"))
            .group_by("w").len()["len"]
        )
        weekly_post = (
            post.with_columns(pl.col("timestamp").dt.truncate("1w").alias("w"))
            .group_by("w").len()["len"]
        )
        rows.append(
            [
                s_,
                f"{pre.height:,}",
                str(pre["timestamp"].min()) if pre.height else "-",
                str(pre["timestamp"].max()) if pre.height else "-",
                f"{float(weekly_pre.median()):.0f}" if pre.height else "-",
                f"{post.height:,}",
                f"{float(weekly_post.median()):.0f}",
            ]
        )
    parts.append(
        "## 1b. The sparse pre-2017 prefix, and why it is excluded\n\n"
        + md_table(
            [
                "symbol",
                "bars before 2017-02-27",
                "first",
                "last",
                "median bars/week (sparse)",
                "bars from 2017-02-27",
                "median bars/week (dense)",
            ],
            rows,
        )
        + "\n\nA session decision needs the bar that closes one hour after the venue opens. At "
        "six bars a week that bar does not exist, so the prefix cannot carry a decision at all - "
        "it is excluded by `universe.history_start`, not by a filter anywhere downstream. Every "
        "section below reads only the dense part.\n"
    )

    # ------------------------------------------------- 1c spread == 0 bars
    rows = []
    for s_ in SYMBOLS:
        p = h1.filter((pl.col("symbol") == s_) & (pl.col("day") >= DENSE_START))
        zero = p.filter(pl.col("spread") <= 0)
        by_year = (
            zero.group_by(pl.col("timestamp").dt.year().alias("y")).len().sort("y")
        )
        rows.append(
            [
                s_,
                f"{p.height:,}",
                f"{zero.height:,}",
                f"{zero.height / max(p.height, 1):.3%}",
                str(zero["timestamp"].min()) if zero.height else "-",
                str(zero["timestamp"].max()) if zero.height else "-",
                ", ".join(f"{r[0]}:{r[1]}" for r in by_year.iter_rows()) if zero.height else "-",
                f"{float(p['spread'].median()):.0f}",
            ]
        )
    parts.append(
        "## 1d. Bars quoting a zero spread\n\n"
        "`spread` is the quoted spread in points at the bar open. A bar carrying zero is not a "
        "free crossing: it is a bar the server wrote without a spread. `xau_fx_mt5` measured "
        "about 1,015 such H1 bars per metal on this account up to 2023-02-03 "
        "(`case_studies/xau_fx_mt5/config/setup.yaml:92-95`) and fills them with a point-in-time "
        "rolling median rather than pricing them at zero. This bot does **not** read the per-bar "
        "spread at all - `costs.spread_bps` comes from the 30-day tick measurement - so the count "
        "below is a coverage fact rather than a cost input; it becomes one the moment anyone "
        "prices the 2017-2023 stretch off the bar field.\n\n"
        + md_table(
            [
                "symbol",
                "dense H1 bars",
                "spread == 0",
                "share",
                "first",
                "last",
                "by year",
                "median spread (points)",
            ],
            rows,
        )
        + "\n"
    )

    # Sections 1b and 1d are computed last (they need the frames the earlier sections built) but
    # they belong beside section 1, because they are what says which part of the file the rest of
    # the report reads. Move them rather than leaving the reader to find them at the bottom.
    parts = parts[:1] + parts[1:2] + parts[-2:] + parts[2:-2]

    code = Path(__file__).read_text(encoding="utf-8")
    parts.append("## 7. Code used\n\n`bots/exness_gold_sess/tools/h1_census.py`\n\n```python\n" + code + "\n```\n")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    for block in parts[:-1]:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```
