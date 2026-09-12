# Data census — `exness_gold_sess` H1 panel (XAUUSD, XAGUSD)

Measured 2026-09-07 from `ML4T_DATA_PATH/mt5/1h.parquet` (downloaded from the Exness demo terminal `Exness-MT5Trial7`, login 206539306, on 2026-09-05 by `bots/_shared/mt5_loader.py`). Experiment-independent: no stage, registry, label or model is involved, and no holdout is defined yet. Server clock UTC+0 without DST, so server hour == UTC hour. Code at the bottom of this file (`bots/exness_gold_sess/tools/h1_census.py`).

## 1. Coverage and the 2022-10-25 H1 start

| symbol | H1 bars | first H1 (UTC) | last H1 (UTC) | H1 days with bars | D1 | H4 |
|---|---|---|---|---|---|---|
| XAUUSD | 22,838 | 2022-10-25 01:00:00+00:00 | 2026-09-04 20:00:00+00:00 | 1,202 | 3,892 (2014-01-14 00:00:00+00:00 -> 2026-09-04 00:00:00+00:00) | 16,148 (2014-01-14 00:00:00+00:00 -> 2026-09-04 20:00:00+00:00) |
| XAGUSD | 22,831 | 2022-10-25 01:00:00+00:00 | 2026-09-04 20:00:00+00:00 | 1,202 | 3,892 (2014-01-12 00:00:00+00:00 -> 2026-09-04 00:00:00+00:00) | 16,150 (2014-01-12 00:00:00+00:00 -> 2026-09-04 20:00:00+00:00) |

The H1 history starts **2022-10-25** on both symbols while D1 and H4 reach back to 2014-01: the terminal serves a shallower H1 depth, not a data error. Any H1 walk-forward design has ~3.9 years of bars to split, which is what the phase-1 specification has to fit its folds and its holdout into; a D1 or H4 line of the same bot would have ~12.6 years.

## 2. Bars per weekday and server hour — XAUUSD

Number of H1 bars stamped at each (weekday, UTC hour) over the whole history. A zero means the server printed no bar in that slot; a count well below the weekday's modal count means the slot exists in only one DST season.

| weekday | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 192 | 189 | 181 | 60 | 128 | 201 |
| Tue | 201 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 200 | 200 | 71 | 129 | 200 |
| Wed | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 199 | 198 | 198 | 69 | 129 | 200 |
| Thu | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 198 | 196 | 194 | 66 | 126 | 198 |
| Fri | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 195 | 195 | 192 | 191 | 68 | 0 | 0 |
| Sat | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sun | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 129 | 197 |

Hours 00-20 are printed on essentially every Mon-Fri date. Hours **21** and **22** are complementary: exactly one of them is printed on a given weekday, because the one-hour daily break sits at 21:00-22:00 UTC while the US is on DST and at 22:00-23:00 UTC while it is not. Friday ends at 21:00 UTC (no 22:00 or 23:00 bar) and Sunday only re-opens at 22:00 UTC. Saturday is empty.

### 2b. The same, as hour spans — XAUUSD

| weekday | hour spans with bars (UTC) | bars/hour min-max | bars | hours |
|---|---|---|---|---|
| Mon | 00:00-24:00 | 60-201 | 4,497 | 24 |
| Tue | 00:00-24:00 | 71-202 | 4,637 | 24 |
| Wed | 00:00-24:00 | 69-200 | 4,593 | 24 |
| Thu | 00:00-24:00 | 66-200 | 4,578 | 24 |
| Fri | 00:00-22:00 | 68-198 | 4,207 | 22 |
| Sat | - | - | 0 | 0 |
| Sun | 22:00-24:00 | 129-197 | 326 | 2 |

## 2. Bars per weekday and server hour — XAGUSD

Number of H1 bars stamped at each (weekday, UTC hour) over the whole history. A zero means the server printed no bar in that slot; a count well below the weekday's modal count means the slot exists in only one DST season.

| weekday | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 197 | 196 | 196 | 196 | 196 | 196 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 197 | 192 | 189 | 181 | 60 | 128 | 201 |
| Tue | 201 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 202 | 200 | 200 | 71 | 129 | 200 |
| Wed | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 199 | 198 | 198 | 69 | 129 | 200 |
| Thu | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 198 | 195 | 194 | 66 | 126 | 198 |
| Fri | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 195 | 195 | 191 | 191 | 68 | 0 | 0 |
| Sat | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sun | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 129 | 197 |

Hours 00-20 are printed on essentially every Mon-Fri date. Hours **21** and **22** are complementary: exactly one of them is printed on a given weekday, because the one-hour daily break sits at 21:00-22:00 UTC while the US is on DST and at 22:00-23:00 UTC while it is not. Friday ends at 21:00 UTC (no 22:00 or 23:00 bar) and Sunday only re-opens at 22:00 UTC. Saturday is empty.

### 2b. The same, as hour spans — XAGUSD

| weekday | hour spans with bars (UTC) | bars/hour min-max | bars | hours |
|---|---|---|---|---|
| Mon | 00:00-24:00 | 60-201 | 4,492 | 24 |
| Tue | 00:00-24:00 | 71-202 | 4,637 | 24 |
| Wed | 00:00-24:00 | 69-200 | 4,593 | 24 |
| Thu | 00:00-24:00 | 66-200 | 4,577 | 24 |
| Fri | 00:00-22:00 | 68-198 | 4,206 | 22 |
| Sat | - | - | 0 | 0 |
| Sun | 22:00-24:00 | 129-197 | 326 | 2 |

## 3. Missing calendar days

| symbol | calendar days in range | days with bars | days with no bar | by weekday | Mon-Fri holes | the Mon-Fri holes |
|---|---|---|---|---|---|---|
| XAUUSD | 1,411 | 1,202 | 209 | Sat 201, Sun 4, Fri 4 | 4 | 2023-04-07, 2024-03-29, 2025-04-18, 2026-04-03 |
| XAGUSD | 1,411 | 1,202 | 209 | Sat 201, Sun 4, Fri 4 | 4 | 2023-04-07, 2024-03-29, 2025-04-18, 2026-04-03 |

Saturdays are expected (the market is shut). Sundays carry only the 22:00-24:00 re-open. The Mon-Fri holes are the market holidays listed above; each one is a date on which no session decision can be taken.

## 4. Integrity of the H1 bars

| symbol | bars | duplicate timestamps | OHLC invariant violations | zero-volume bars | flat bars (high==low) | null OHLCV | off the :00 grid | tick volume min/median/max |
|---|---|---|---|---|---|---|---|---|
| XAUUSD | 22,838 | 0 | 0 | 0 | 0 | 0 | 0 | 192 / 6,449 / 236,819 |
| XAGUSD | 22,831 | 0 | 0 | 0 | 0 | 0 | 0 | 17 / 1,169 / 129,993 |

## 5. Do the decision bars exist on every session day?

The London-open bar is 07:00 UTC on BST dates and 08:00 UTC on GMT dates; the New York open bar is 12:00 UTC on EDT dates and 13:00 UTC on EST dates. A **session day** is a Mon-Fri date with at least one bar; a Sunday carries only the 22:00-24:00 re-open and can never hold a session decision, so Sundays are counted separately and excluded from the decision-bar check.

| symbol | days with bars | of which Sunday re-opens | session days (Mon-Fri) | with a London-open bar | missing | with a NY-open bar | missing | summer / winter session days |
|---|---|---|---|---|---|---|---|---|
| XAUUSD | 1,202 | 197 | 1,005 | 997 | 8 | 997 | 8 | 571 / 434 |
| XAGUSD | 1,202 | 197 | 1,005 | 997 | 8 | 997 | 8 | 571 / 434 |

- **XAUUSD** — session days without a London-open bar (8): 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01; without a NY-open bar (8): 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01

- **XAUUSD** — session days on which the exit price could not be formed: London close 0; 21:00 UTC 33 (2022-11-24, 2022-11-25, 2023-01-16, 2023-02-20, 2023-05-29, 2023-06-19, 2023-07-04, 2023-09-04, 2023-11-23, 2023-11-24, 2024-01-15, 2024-02-19, 2024-05-27, 2024-06-19, 2024-07-04, 2024-09-02, 2024-11-28, 2024-11-29, 2024-12-24, 2025-01-20)

- **XAGUSD** — session days without a London-open bar (8): 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01; without a NY-open bar (8): 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01, 2024-12-25, 2025-01-01, 2025-12-25, 2026-01-01

- **XAGUSD** — session days on which the exit price could not be formed: London close 0; 21:00 UTC 33 (2022-11-24, 2022-11-25, 2023-01-16, 2023-02-20, 2023-05-29, 2023-06-19, 2023-07-04, 2023-09-04, 2023-11-23, 2023-11-24, 2024-01-15, 2024-02-19, 2024-05-27, 2024-06-19, 2024-07-04, 2024-09-02, 2024-11-28, 2024-11-29, 2024-12-24, 2025-01-20)

## 6. Realised session moves vs the measured round trip

Return from the **open of the decision bar** to the price at the session-exit instant (`price_at` = open of that hour's bar, else close of the previous bar), in basis points, absolute value, over the whole H1 history. The round trip is 2 x the in-session p90 spread measured on 2026-09-07 from 30 days of this account's ticks (`data/mt5/spreads_by_session.json`): XAUUSD 0.60 bps, XAGUSD 4.70 bps per crossing. No cost other than the spread is charged here (Pro account, commission 0, measured swap 0.0/0.0).

| symbol | window | n session days | median abs return (bps) | p25 | p75 | p90 | round trip (bps) | median / round trip | share of days above the round trip |
|---|---|---|---|---|---|---|---|---|---|
| XAUUSD | London open -> London close (17:00 GMT / 16:00 BST) | 997 | 44.3 | 19.3 | 81.1 | 131.6 | 1.2 | 36.9x | 0.984 |
| XAUUSD | NY open -> 21:00 UTC | 964 | 43.5 | 19.7 | 84.5 | 137.7 | 1.2 | 36.3x | 0.988 |
| XAGUSD | London open -> London close (17:00 GMT / 16:00 BST) | 997 | 85.7 | 35.9 | 167.2 | 256.5 | 9.4 | 9.1x | 0.950 |
| XAGUSD | NY open -> 21:00 UTC | 964 | 88.8 | 39.7 | 161.3 | 270.5 | 9.4 | 9.4x | 0.953 |

Read this as the cost feasibility ratio the phase-1 specification has to beat: a signal has to capture a fraction of the median move larger than one round trip.

### 6b. How often the exit price came from the fallback

| symbol | London close used prev-bar close | 21:00 UTC used prev-bar close |
|---|---|---|
| XAUUSD | 0 | 630 |
| XAGUSD | 0 | 630 |

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
OUT = REPO_ROOT / "bots" / "exness_gold_sess" / "data_census_2026-09-07.md"
SYMBOLS = ("XAUUSD", "XAGUSD")
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
    h1 = load("1h").with_columns(
        pl.col("timestamp").dt.date().alias("day"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("iso_wd"),  # 1 = Monday
    )
    d1 = load("daily")
    h4 = load("4h")
    parts: list[str] = []

    parts.append(
        "# Data census — `exness_gold_sess` H1 panel (XAUUSD, XAGUSD)\n\n"
        "Measured 2026-09-07 from `ML4T_DATA_PATH/mt5/1h.parquet` (downloaded from the Exness "
        "demo terminal `Exness-MT5Trial7`, login 206539306, on 2026-09-05 by "
        "`bots/_shared/mt5_loader.py`). Experiment-independent: no stage, registry, label or "
        "model is involved, and no holdout is defined yet. Server clock UTC+0 without DST, so "
        "server hour == UTC hour. Code at the bottom of this file "
        "(`bots/exness_gold_sess/tools/h1_census.py`).\n"
    )

    # ---------------------------------------------------------------- 1 coverage
    rows = []
    for s in SYMBOLS:
        p = h1.filter(pl.col("symbol") == s)
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
        "## 1. Coverage and the 2022-10-25 H1 start\n\n"
        + md_table(
            ["symbol", "H1 bars", "first H1 (UTC)", "last H1 (UTC)", "H1 days with bars", "D1", "H4"],
            rows,
        )
        + "\n\nThe H1 history starts **2022-10-25** on both symbols while D1 and H4 reach back to "
        "2014-01: the terminal serves a shallower H1 depth, not a data error. Any H1 walk-forward "
        "design has ~3.9 years of bars to split, which is what the phase-1 specification has to "
        "fit its folds and its holdout into; a D1 or H4 line of the same bot would have ~12.6 "
        "years.\n"
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
