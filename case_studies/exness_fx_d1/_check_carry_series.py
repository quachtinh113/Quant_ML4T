"""Acceptance checks for the carry legs, to run the moment `FRED_API_KEY` has a value.

Not a stage and not a trial: it downloads nothing and writes nothing. It asks FRED and ALFRED
four questions about every id in `bots/_shared/macro_config.yaml::carry.legs`, and only a leg
that answers all four may be used to build the carry family:

  (a) the id resolves on the FRED ``series`` endpoint;
  (b) ALFRED returns vintage dates for it - ``download_alfred.py::fetch_vintage_dates`` raises
      ``RuntimeError`` on an empty list, which aborts the whole fetch, so one leg without
      vintages costs every leg;
  (c) ``observation_start`` is at or before ``--coverage-from`` (default 2016-01-01), so the leg
      covers ``universe.history_start`` (2017-03-01) together with the feature warmup;
  (d) ``observation_end`` is within ``--max-staleness-days`` (default 90) of today.

(d) is the check that matters most and the one a "does the id resolve?" probe misses entirely.
Several OECD "Immediate Rates" series were discontinued on FRED. A discontinued leg still
resolves, still has vintages, still has decades of history - and forward-fills one constant from
the day it stopped to the end of the sample. The carry column built on it would still show an
information coefficient, and it would carry no information at all. A leg failing (b) or (d) is
replaced from the alternates listed in `macro_config.yaml`, not carried forward.

The key is read through `utils.downloading.load_dotenv` / `require_env`, the same path the
downloaders use, and is never printed.

    uv run python case_studies/exness_fx_d1/_check_carry_series.py
    uv run python case_studies/exness_fx_d1/_check_carry_series.py --config bots/_shared/macro_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

import yaml

from utils.downloading import load_dotenv, require_env
from utils.paths import REPO_ROOT

FRED_API = "https://api.stlouisfed.org/fred"
USER_AGENT = "ml4t-carry-acceptance/1.0"
DEFAULT_CONFIG = REPO_ROOT / "bots" / "_shared" / "macro_config.yaml"


def _request(endpoint: str, params: dict, api_key: str) -> dict:
    query = dict(params, api_key=api_key, file_type="json")
    url = f"{FRED_API}/{endpoint}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)


def check_leg(series_id: str, api_key: str, *, coverage_from: date, max_staleness: int) -> dict:
    """The four acceptance checks for one leg. Never raises; a failure is a reported row."""
    row: dict = {"series_id": series_id, "resolves": False, "has_vintages": False}
    try:
        meta = _request("series", {"series_id": series_id}, api_key)["seriess"][0]
    except urllib.error.HTTPError as exc:
        row["error"] = f"HTTP {exc.code}"
        return row
    except Exception as exc:  # noqa: BLE001 - the reason is reported, not raised
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["resolves"] = True
    row["title"] = meta["title"]
    row["frequency"] = meta["frequency_short"]
    row["observation_start"] = meta["observation_start"]
    row["observation_end"] = meta["observation_end"]
    row["last_updated"] = meta["last_updated"][:10]
    start = date.fromisoformat(meta["observation_start"])
    end = date.fromisoformat(meta["observation_end"])
    row["covers_history"] = start <= coverage_from
    row["staleness_days"] = (date.today() - end).days
    row["is_current"] = row["staleness_days"] <= max_staleness
    try:
        payload = _request(
            "series/vintagedates",
            {"series_id": series_id, "limit": 10, "sort_order": "desc"},
            api_key,
        )
        vintages = payload.get("vintage_dates", [])
        row["has_vintages"] = bool(vintages)
        row["vintage_count"] = int(payload.get("count", len(vintages)))
        row["latest_vintage"] = vintages[0] if vintages else None
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"vintages: {type(exc).__name__}: {exc}"
    row["accepted"] = bool(
        row["resolves"] and row["has_vintages"] and row["covers_history"] and row["is_current"]
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--coverage-from", type=str, default="2016-01-01")
    parser.add_argument("--max-staleness-days", type=int, default=90)
    args = parser.parse_args()

    load_dotenv()
    api_key = require_env(
        "FRED_API_KEY", "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
    )
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    legs = config["carry"]["legs"]
    coverage_from = date.fromisoformat(args.coverage_from)

    print(f"config: {args.config}")
    print(f"checked at: {datetime.now().isoformat(timespec='seconds')}")
    print(
        f"acceptance: resolves, has vintages, observation_start <= {coverage_from}, "
        f"observation_end within {args.max_staleness_days} days\n"
    )
    rows = []
    for currency, series_id in legs.items():
        row = check_leg(
            str(series_id),
            api_key,
            coverage_from=coverage_from,
            max_staleness=args.max_staleness_days,
        )
        row["currency"] = currency
        rows.append(row)
        verdict = "ACCEPT" if row.get("accepted") else "REJECT"
        print(f"{verdict}  {currency} {series_id}")
        if not row["resolves"]:
            print(f"         (a) does not resolve: {row.get('error')}")
            continue
        print(f"         {row['title'][:78]}")
        print(
            f"         (a) resolves, {row['frequency']}  "
            f"(b) vintages {row.get('vintage_count', 0)}, latest {row.get('latest_vintage')}  "
            f"(c) from {row['observation_start']} -> {row['covers_history']}  "
            f"(d) to {row['observation_end']}, {row['staleness_days']} days old "
            f"-> {row['is_current']}"
        )

    rejected = [r for r in rows if not r.get("accepted")]
    print(f"\n{len(rows) - len(rejected)} of {len(rows)} legs accepted")
    if rejected:
        print(
            "REJECTED: "
            + ", ".join(f"{r['currency']} ({r['series_id']})" for r in rejected)
            + ". Replace each from the alternates in macro_config.yaml and re-run this check "
            "BEFORE `download_alfred.py`, which aborts on the first id ALFRED cannot serve."
        )
        return 1
    print(
        "All legs accepted. Next: "
        "`uv run python data/macro/download_alfred.py --config "
        f"{args.config}`, then declare the carry family in "
        "case_studies/exness_fx_d1/config/setup.yaml (features.windows.carry, "
        "features.windows.carry_zscore, a `carry` row in features.families) and re-run 03."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
