"""The declared phase-5 sweep of `exness_usidx_sess`, in one place.

Three declarations live here, all of them read from `config/setup.yaml` and none of them
invented by the caller:

* :func:`declared_settings` - the signal dicts. `13_backtest.py` sends them to the engine and
  `_report_phase5.py` scores the same dicts vectorised. Generation 1 built that list twice, in
  two files, and the two agreed only by inspection; a divergence would have planned one identity
  and reported another. One function, imported by both, makes that class of error impossible.
* :func:`with_spread_bps` - the per-session spread charge, under the regime rule
  `costs.spread_bps_charge_rule` declares. Only the vectorised report can use it: the engine
  charges one number for the whole run and that number stays the conservative one.
* :func:`min_traded_sessions` - the declared floor on the number of sessions a member's book
  actually opened a position on, below which the member is counted in K but is not eligible to
  be read as a candidate.

Nothing here selects, scores or ranks anything.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl


def declared_settings(setup: dict, family: str | None = None) -> list[dict[str, Any]]:
    """The declared settings of the sweep, in the order `setup.yaml` declares them.

    `family` narrows to one signal family (`13_backtest` publishes one population per family);
    the default returns the whole declared sweep, which is what K counts per prediction set.
    """
    grid = setup["backtest"]["sweep"]["threshold_grid"]
    convention = str(grid["fixed_threshold_convention"])
    if convention not in ("probability_mirror", "signed"):
        raise RuntimeError(
            f"threshold_grid.fixed_threshold_convention is {convention!r}: expected "
            "'probability_mirror' or 'signed' (case_studies/utils/signals.py)"
        )
    fixed = [
        {
            "method": "fixed_threshold",
            "threshold": float(t),
            "long_short": True,
            # Part of the identity. Without it the library mirrors the short threshold about
            # 0.5 and a regression score can never be flat; see the setup.yaml declaration.
            "threshold_convention": convention,
        }
        for t in grid["fixed_threshold"]
    ]
    pct = grid["per_symbol_rolling_percentile"]
    bars_per_day = int(pct["bars_per_day"])
    if bars_per_day != 1:
        raise RuntimeError(
            f"per_symbol_rolling_percentile.bars_per_day is {bars_per_day}: this bot takes one "
            "decision per session per spec, and the library default of 390 makes min_samples "
            "unreachable on the validation window, so every threshold is null and every book "
            "never trades while the sweep completes and registers"
        )
    percentile = [
        {
            "method": "per_symbol_rolling_percentile",
            "long_q": float(pct["long_q"]),
            "lookback_days": int(lookback),
            "bars_per_day": bars_per_day,
            # Long only. The pinned config/backtest/base.yaml declares allow_short_selling
            # without allow_leverage, so the engine runs a cash account and a short is sized by
            # the cash on hand. A short leg here would double what this family adds to K and has
            # not been approved.
            "long_short": False,
            "direction": "long_only",
        }
        for lookback in pct["lookback_days"]
    ]
    by_family = {"fixed_threshold": fixed, "per_symbol_rolling_percentile": percentile}
    if family is None:
        return fixed + percentile
    if family not in by_family:
        raise ValueError(f"unknown signal family {family!r}: {sorted(by_family)}")
    return by_family[family]


def min_traded_sessions(setup: dict) -> int:
    return int(setup["backtest"]["sweep"]["min_traded_sessions"])


def with_spread_bps(
    setup: dict,
    frame: pl.DataFrame,
    *,
    time_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Add a `spread_bps` column: the charge PER CROSSING for that symbol on that session.

    The rule is `costs.spread_bps_charge_rule` and it adds no number of its own. Before
    `post_break_from` - the pre-2024-10 regime and the break month itself, which cannot be
    assigned to either side - the charge is the conservative `costs.spread_bps.indices[-1]`.
    From that date the charge is `costs.spread_bps_by_year.recent_regime_p90[symbol]`, the
    worst of that symbol's 2025 and 2026 year p90s at this bot's execution hours.

    A symbol with no declared post-break rate stops the run. Charging it the conservative rate
    instead would be a measurement nobody made, and charging it zero would be worse.
    """
    costs = setup["costs"]
    rule = costs["spread_bps_charge_rule"]
    if rule.get("pre_break_bps") != "from_spread_bps_indices_p90":
        raise RuntimeError(
            f"spread_bps_charge_rule.pre_break_bps is {rule.get('pre_break_bps')!r}: this "
            "function implements 'from_spread_bps_indices_p90' only"
        )
    if rule.get("post_break_bps") != "from_recent_regime_p90":
        raise RuntimeError(
            f"spread_bps_charge_rule.post_break_bps is {rule.get('post_break_bps')!r}: this "
            "function implements 'from_recent_regime_p90' only"
        )
    if rule.get("unknown_session") != "refuse":
        raise RuntimeError("spread_bps_charge_rule.unknown_session must be 'refuse'")

    pre_break = float(costs["spread_bps"]["indices"][-1])
    post_break = {
        str(symbol): float(value)
        for symbol, value in costs["spread_bps_by_year"]["recent_regime_p90"].items()
    }
    cutoff = date.fromisoformat(str(rule["post_break_from"]))

    missing = sorted(set(frame[symbol_col].unique().to_list()) - set(post_break))
    if missing:
        raise RuntimeError(
            f"no declared post-break spread for {missing}: "
            "costs.spread_bps_by_year.recent_regime_p90 has "
            f"{sorted(post_break)}; declare it or stop scoring these symbols"
        )
    return frame.with_columns(
        pl.when(pl.col(time_col).cast(pl.Date) < cutoff)
        .then(pl.lit(pre_break, dtype=pl.Float64))
        .otherwise(
            pl.col(symbol_col).replace_strict(post_break, return_dtype=pl.Float64)
        )
        .alias("spread_bps")
    )
