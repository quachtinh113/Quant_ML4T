"""Real per-bar spread -> fractional trading cost, shared across case studies.

Guard (c), pinned 2026-09-10 (`bots/xau_fx_mt5/BOT.md`, Decisions log; source: the V9 P&L
bridge, machine record `data_lake/reviews.jsonl`, agent=quant-orchestrator, phase=pnl-bridge).
V9 Continuum embedded a **$3,453 one-way MODEL spread** in its backtest versus **$7,372 real
round-trip** measured from its own broker's ticks — a factor of ~2.1x understated cost that
alone accounted for a material share of the $11,998 P&L bridge on the V9 elite6 population's
$10k account.

This module is intentionally **case-study-agnostic**: no symbol list, no bot id, no market
convention beyond "the caller supplies a `spread` column in *points* and a `point` size per
symbol". `case_studies/utils/backtest_runner.py` calls it ONLY when a case study's own
`setup.yaml` declares `costs.spread_source: data_spread_column` AND
`costs.spread_charge: round_trip` (see `run_backtest`'s cost_spec construction and
`_run_vectorized`'s `cost_model == "spread_column"` branch); every other case study never
imports this module at backtest time.

Two steps, kept separate so each is independently testable:

1. :func:`fill_zero_spread` — a terminal that recorded no spread on some bars must not have
   those bars priced as free. Fills with the trailing median of *positive* spreads over
   ``window`` prior rows (strictly trailing: row *t*'s own spread never enters its own fill
   value), and leaves ``null`` (never 0) when no positive spread exists yet in that window.
   Point in time by construction (only past rows). Ported from the bot-local
   `experiments/xau_fx_mt5/data/costs_pit.py::fill_spread_cost`, generalised to arbitrary
   column names so it has no dependency on any one case study's schema.
2. :func:`real_spread_cost_fraction` — converts a per-bar spread in *points* to a **fractional**
   cost (dimensionless, same units as `commission_bps / 1e4`): ``spread_points * point / price``.
   Point size is read from the caller-supplied ``point_by_symbol`` mapping — which in turn is
   sourced from `setup.yaml::costs.contract.point`, itself derived from the symbol's `digits`
   at declaration time (point = 10**-digits) — **never assumed** or hardcoded here. This is the
   unit-trap guard: a metals/JPY symbol quoting 3 decimals has point = 0.001, not the 5-decimal
   default 0.00001 most FX pairs use; passing the wrong map silently overstates or understates
   cost by orders of magnitude (a 260-point XAUUSD spread is $0.26/oz at point=0.001, not
   $2.60 at an assumed point=0.01).
3. :func:`commission_cost_fraction` — converts a flat, account-wide per-lot commission
   (``setup.yaml::costs.commission_per_lot``) to the same fractional units, via
   ``commission_per_lot / (contract_size * price)``. ``contract_size_by_symbol`` follows the
   same declared-default-or-raise rule as ``point_by_symbol`` below.

**No silent code-invented default anywhere in this module** (mentor gate finding B5, 2026-09-10):
a symbol missing from ``point_by_symbol`` / ``contract_size_by_symbol`` RAISES unless the
mapping itself declares a ``"default"`` entry (the ``setup.yaml::costs.contract.point.default`` /
``costs.contract.contract_size.default`` convention already used elsewhere in this bot's config)
— a declared default is a real, written-down number; a function parameter default of ``0.0``
silently prices an unmapped symbol as free, which is exactly the defect guard (c) exists to stop.

The **round-trip** convention (charged as HALF the quoted spread at the entry turnover event and
HALF again at the exit turnover event, so a full round trip totals ONE quoted spread —
``setup.yaml::costs``: "Round trip = one quoted spread" — never the full spread at each leg) is
the caller's responsibility, not this module's — see `backtest_runner.py`'s `spread_column` cost
branch, which mirrors the existing `per_share_plus_spread` branch's turnover-drag shape exactly.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

FILL_WINDOW = 480  # trailing bars; matches experiments/xau_fx_mt5/data/costs_pit.py::FILL_WINDOW


def _resolve_declared_map(
    values: Mapping[str, float], symbols: set[str], *, symbol_col: str, what: str
) -> pl.Expr:
    """A polars expression mapping ``symbol_col`` -> a declared value in ``values``.

    RAISES if any symbol in ``symbols`` is absent from ``values`` and ``values`` carries no
    ``"default"`` entry. A ``"default"`` entry, when present, IS a declared value (written in
    the case study's own setup.yaml) and is used as the fallback for every symbol not
    explicitly listed — this is not a code-invented default, it is the caller's own
    declaration. Mentor gate finding B5, 2026-09-10: no function-parameter default is permitted
    to silently price an unmapped symbol as free.

    ALSO RAISES if ANY declared value in ``values`` (a symbol entry or the ``"default"`` entry)
    is <= 0 (mentor gate finding M2, 2026-09-10): a *declared* ``0`` is still a config error, not
    a real point size or contract size, and would price every symbol it covers as free exactly
    as a code-invented ``0.0`` would — "declared" only means "written down", not "validated". Every
    entry is checked, not just the ones the current frame's symbols happen to use, so a bad
    ``"default": 0`` is caught even before any symbol falls back to it.
    """
    non_positive = {k: v for k, v in values.items() if float(v) <= 0}
    if non_positive:
        raise ValueError(
            f"declared {what} must be > 0, got {non_positive} -- a declared value of 0 (or "
            f"negative) would still price the affected symbol(s) as free (guard c, M2)"
        )
    declared_default = "default" in values
    missing = symbols - set(values)
    if missing and not declared_default:
        raise ValueError(
            f"no {what} declared for {sorted(missing)}, and no 'default' entry either -- "
            f"a missing {what} must never fall back to a code-invented value (guard c, B5)"
        )
    if declared_default:
        return pl.col(symbol_col).replace_strict(
            {k: v for k, v in values.items() if k != "default"},
            default=float(values["default"]),
            return_dtype=pl.Float64,
        )
    return pl.col(symbol_col).replace_strict(dict(values), return_dtype=pl.Float64)


def fill_zero_spread(
    frame: pl.DataFrame,
    *,
    symbol_col: str = "symbol",
    time_col: str = "timestamp",
    spread_col: str = "spread_points",
    out_col: str | None = None,
    window: int = FILL_WINDOW,
) -> pl.DataFrame:
    """Add ``out_col`` (default ``f"{spread_col}_filled"``): the trailing median of positive
    ``spread_col`` values over ``window`` prior rows wherever ``spread_col`` is <= 0, else
    ``spread_col`` itself. Never 0: a bar with no positive spread in its trailing window gets
    ``null``, which the caller must treat as "cannot be priced", never as "free".
    """
    out_col = out_col or f"{spread_col}_filled"
    f = frame.sort([symbol_col, time_col])
    positive = pl.when(pl.col(spread_col) > 0).then(pl.col(spread_col)).otherwise(None)
    trailing = positive.shift(1).rolling_median(window_size=window, min_samples=1).over(symbol_col)
    out = f.with_columns(
        pl.when(pl.col(spread_col) > 0)
        .then(pl.col(spread_col).cast(pl.Float64))
        .otherwise(trailing)
        .alias(out_col)
    )
    assert out.filter(pl.col(out_col) <= 0).height == 0, "a bar was priced as free"
    return out


def real_spread_cost_fraction(
    prices: pl.DataFrame,
    *,
    spread_col: str = "spread_points",
    price_col: str = "close",
    symbol_col: str = "symbol",
    time_col: str = "timestamp",
    point_by_symbol: Mapping[str, float],
    fill_window: int = FILL_WINDOW,
    out_col: str = "spread_cost_fraction",
) -> pl.DataFrame:
    """``[time_col, symbol_col, out_col]``: one full quoted spread as a fraction of price.

    ``out_col`` = ``spread_points * point / price`` where ``point`` comes from
    ``point_by_symbol[symbol]`` — NEVER assumed from the number of decimals in a price string,
    always the caller-declared, digits-derived value. A symbol present in ``prices`` but absent
    from ``point_by_symbol`` (and with no ``"default"`` entry in it) RAISES — see
    :func:`_resolve_declared_map` (guard c, B5).

    Rows whose ``spread_col`` is null after :func:`fill_zero_spread` (no positive spread in the
    trailing window) are dropped from the output, not zero-filled: the caller (``_run_vectorized``)
    must treat an unpriceable bar as unpriceable, never as free (guard c, B2) — it raises if any
    such bar coincides with an actual turnover event.
    """
    filled = fill_zero_spread(
        prices.select([time_col, symbol_col, spread_col]),
        symbol_col=symbol_col,
        time_col=time_col,
        spread_col=spread_col,
        out_col="_spread_filled",
        window=fill_window,
    ).drop_nulls("_spread_filled")

    priced = prices.select([time_col, symbol_col, price_col]).join(
        filled, on=[time_col, symbol_col], how="inner"
    )
    symbols_in_frame = set(priced[symbol_col].unique().to_list())
    point_expr = _resolve_declared_map(
        point_by_symbol, symbols_in_frame, symbol_col=symbol_col, what="point size"
    )
    return priced.filter(pl.col(price_col).is_not_null() & (pl.col(price_col) > 0)).select(
        time_col,
        symbol_col,
        (pl.col("_spread_filled") * point_expr / pl.col(price_col)).alias(out_col),
    )


def commission_cost_fraction(
    prices: pl.DataFrame,
    *,
    price_col: str = "close",
    symbol_col: str = "symbol",
    time_col: str = "timestamp",
    commission_per_lot: float,
    contract_size_by_symbol: Mapping[str, float],
    out_col: str = "commission_cost_fraction",
) -> pl.DataFrame:
    """``[time_col, symbol_col, out_col]``: a flat, account-wide per-lot commission
    (``setup.yaml::costs.commission_per_lot``, e.g. $0 on this account) as a fraction of price,
    via ``commission_per_lot / (contract_size * price)`` — the same unit conversion
    :func:`real_spread_cost_fraction` uses for the spread, so both terms sum directly into one
    turnover-drag fraction.

    ``commission_per_lot == 0.0`` short-circuits to an all-zero column without requiring
    ``contract_size_by_symbol`` to cover every symbol (0 / anything is 0; no point raising on a
    contract size this account's zero commission will never multiply). A nonzero
    ``commission_per_lot`` with a symbol missing from ``contract_size_by_symbol`` (and no
    declared ``"default"``) RAISES, exactly like :func:`real_spread_cost_fraction` (guard c, B5).
    """
    base = prices.select([time_col, symbol_col, price_col]).filter(
        pl.col(price_col).is_not_null() & (pl.col(price_col) > 0)
    )
    if commission_per_lot == 0.0:
        return base.select(time_col, symbol_col, pl.lit(0.0).alias(out_col))
    symbols_in_frame = set(base[symbol_col].unique().to_list())
    size_expr = _resolve_declared_map(
        contract_size_by_symbol, symbols_in_frame, symbol_col=symbol_col, what="contract size"
    )
    return base.select(
        time_col,
        symbol_col,
        (commission_per_lot / (size_expr * pl.col(price_col))).alias(out_col),
    )


__all__ = [
    "FILL_WINDOW",
    "commission_cost_fraction",
    "fill_zero_spread",
    "real_spread_cost_fraction",
]
