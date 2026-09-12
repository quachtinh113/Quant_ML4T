"""Deployment side of the Exness bot portfolio (see bots/README.md).

Research lives in ``case_studies/<bot_id>/``; this package holds what is not research: the
shared MT5 loader, session calendar and broker adapter (``bots._shared``), and per-bot
deployment, monitoring and tests (``bots/<bot_id>/``).
"""
