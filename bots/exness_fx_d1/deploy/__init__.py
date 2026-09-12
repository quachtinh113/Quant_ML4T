"""exness_fx_d1 deployment package: the loop, its risk configuration and its state.

``deployment_loop.py`` is the seven steps of ``25_live_trading/02_etfs_deployment_loop.py``
adapted to MetaTrader 5; ``risk_config.yaml`` is the ``LiveRiskConfig`` and the breaker
thresholds, derived line by line from ``BOT.md``'s kill criteria; ``state/`` holds the
persisted ``RiskState``, the per-cycle run records and the deployment artefacts.

The broker adapter is **not** here: every bot uses ``bots/_shared/mt5_broker.py``.
"""
