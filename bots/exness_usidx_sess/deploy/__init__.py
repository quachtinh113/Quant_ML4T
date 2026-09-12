"""exness_usidx_sess deployment package — DRY RUN ONLY.

EVIDENCE BOUNDARY: this bot has **0 phase-5 survivors of 1,403 scored specs at K = 1,780**,
phase 6 is not opened and the declared holdout 2026-03-01 .. 2026-08-31 has never been scored.
Nothing in this package is evidence of an edge and nothing in it is permission to trade.

``deployment_loop.py``
    The seven steps of ``25_live_trading/02_etfs_deployment_loop.py`` on MetaTrader 5, with
    the parity harness of ``25_live_trading/08`` wired in. It cannot place an order: ``--arm``
    is refused while the kill criteria are a draft, the ``SafeBroker`` layer is ``shadow`` and
    the adapter layer is ``paper``.
``schedule.py``
    The two decision instants per NYSE cash session, DERIVED from the calendar and from
    ``bots/_shared/sessions`` and converted through ``ServerClock``. The UTC hours 14/15/20/21
    appear nowhere as an input: the trade hours follow New York DST and the server clock does
    not, so a UTC constant would be wrong for five months of every year.
``risk_config.yaml``
    ``LiveRiskConfig``, the CFD guards and the breaker thresholds, derived line by line from
    ``BOT.md``'s (draft) kill criteria (a)-(g). Every number carries its derivation.
``state/``
    The persisted ``RiskState``, the audit journal and one run record per cycle, per spec.

The broker adapter is **not** here: every bot uses ``bots/_shared/mt5_broker.py``.
"""
