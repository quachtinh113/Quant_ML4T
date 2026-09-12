"""exness_usidx_sess — the US index session bot (US500, USTEC) on Exness MT5.

EVIDENCE BOUNDARY: 0 phase-5 survivors of 1,403 scored specs at K = 1,780; phase 6 not opened;
the declared holdout 2026-03-01 .. 2026-08-31 is UNSCORED. Nothing in this package is evidence
of an edge and nothing in it is permission to trade. See ``BOT.md``.

``deploy/``   the seven-step deployment loop, its schedule derivation and its risk config
``monitor/``  the Chapter 26 layer: breakers, drift, online detectors, rollout, retirement
``tests/``    everything above, against ``bots/_shared/testing/fake_mt5.py`` — never a terminal
``tools/``    one-off measurement scripts

The research pipeline is ``case_studies/exness_usidx_sess/``; the broker adapter is the shared
``bots/_shared/mt5_broker.py``.
"""
