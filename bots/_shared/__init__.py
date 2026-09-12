"""Code shared by every Exness bot: session calendar, MT5 loader, costs, broker adapter.

Modules
-------
sessions
    Session flags (asia, tokyo, london, new_york, overlap, us_cash, edge_open, edge_close,
    funding, rollover, market_open) for a UTC timestamp, computed from real time zones via
    ``zoneinfo``; the YAML frontmatter of ``bots/assets/<SYMBOL>.md``; and ``ServerClock``,
    the server-time conversion parameterised by a measured offset.
mt5_loader
    ``download_mt5_bars`` (MetaTrader5 imported lazily) and ``load_mt5_bars``, which mirrors
    ``data/fx/loader.py:load_fx_pairs`` so the fx_pairs stages read MT5 history unchanged;
    ``"8h"`` is folded from the H4 bars onto the 00/08/16 UTC grid (MT5 serves no H8).
costs_mt5
    ``measure_spreads``: p50 / p90 quoted spread in bps of mid per symbol and session bucket
    from this account's ticks; ``write_spreads`` writes ``mt5/spreads_by_session.parquet``;
    ``read_swaps`` / ``holding_cost_points``: swap from ``symbol_info`` and the nightly /
    triple / weekend arithmetic of a holding period.
mt5_broker
    ``MT5Broker``: the one MetaTrader 5 adapter every bot wraps in ``SafeBroker``; a bot passes
    its own magic number. ``ml4t`` types are used when installed, stand-ins otherwise.
monitor
    Account-level circuit breakers (Chapter 26) and the shared breaker state machine. The
    account tier stops **every** bot at once through a halt file; per-strategy breakers live
    in ``bots/<bot_id>/monitor/``.
testing.fake_mt5
    A fake ``MetaTrader5`` module for unit tests (rates, ticks, ``order_send``, positions,
    orders, margin); nothing here needs a terminal.

Magic numbers
-------------
``MAGIC_ALLOCATION`` below is the **single** allocation table for the MT5 accounts these bots
share. Every order a bot sends carries its magic and every position query filters on it, so
two bots sharing a number would each read the other's positions as its own and try to close
them. The table lives here rather than in four ``risk_config.yaml`` files so a collision is a
merge conflict instead of a live incident; ``bots/README.md`` repeats it for readers.

Scheme: ``26 09 NN`` = year 2026, month 09, bot index. ``202500`` is the retired legacy V9
Continuum bot's default (``bots/exness_fx_d1/BOT.md`` open question 3) and is reserved, never
reused, so a stray position of that magic is recognisable.
"""

# bot_id -> magic. Decided 2026-09-07; `xau_fx_mt5` had no magic recorded before this table.
MAGIC_ALLOCATION: dict[str, int] = {
    "exness_fx_d1": 260901,
    "exness_usidx_sess": 260902,
    "exness_gold_sess": 260903,
    "exness_btc_8h": 260904,
    "xau_fx_mt5": 260905,
}

# Not ours. Reserved so nothing is ever allocated on top of a legacy position.
RESERVED_MAGICS: dict[int, str] = {202500: "legacy V9 Continuum bot (retired)"}


def magic_for(bot_id: str) -> int:
    """The magic number allocated to ``bot_id``; a bot never invents its own."""
    try:
        return MAGIC_ALLOCATION[bot_id]
    except KeyError as exc:  # pragma: no cover - a typo in a config, caught at startup
        raise KeyError(
            f"{bot_id!r} has no magic number in bots/_shared/__init__.py::MAGIC_ALLOCATION; "
            f"allocated: {sorted(MAGIC_ALLOCATION)}"
        ) from exc


def assert_magics_unique() -> None:
    """Fail loudly if the allocation table ever collides with itself or with a reservation."""
    values = list(MAGIC_ALLOCATION.values())
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate magic numbers in MAGIC_ALLOCATION: {MAGIC_ALLOCATION}")
    clash = set(values) & set(RESERVED_MAGICS)
    if clash:
        raise ValueError(f"magic numbers collide with a reserved magic: {sorted(clash)}")


assert_magics_unique()
