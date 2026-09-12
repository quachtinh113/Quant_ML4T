"""exness_btc_8h: 8-hour single-instrument BTCUSD CFD bot on Exness MT5 (see BOT.md).

The package marker is load-bearing rather than decorative. Without it pytest resolves
`bots/exness_btc_8h/tests/test_data_quality.py` to the top-level package name `tests`, which
collides with the repository's own `tests/` package and with the identically named module in every
sibling bot: `ModuleNotFoundError: No module named 'tests.test_data_quality'` on collection.
`bots/exness_fx_d1/__init__.py` exists for the same reason; `exness_gold_sess` and
`exness_usidx_sess` still lack theirs and their suites error under a whole-tree `pytest bots`
(measured 2026-09-08, and reproduced with this bot excluded, so it is not caused by it).
"""
