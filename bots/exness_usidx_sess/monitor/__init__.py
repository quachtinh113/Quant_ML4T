"""exness_usidx_sess monitoring layer (Chapter 26) — nothing to monitor yet, and it says so.

EVIDENCE BOUNDARY: 0 phase-5 survivors of 1,403 scored specs at K = 1,780, phase 6 not opened,
holdout 2026-03-01 .. 2026-08-31 unscored. There is no live stream, no incumbent and no
challenger. Every module here reports ``not_computable`` with a reason rather than a default
pass, and **not one threshold is calibrated from the Sharpe distribution of the 1,403 failed
trials** — they come from the declared kill criteria and from measured price facts.

``circuit_breakers``
    One breaker per lettered kill criterion (a)-(g) on the SHARED state machine
    ``bots/_shared/monitor/base.py``. (b) is built TWICE, one per spec, because the criterion
    says so; (e) is asymmetric, because a missing overnight bar is market structure and a
    missing intraday bar is an incident; (g) exists on no other bot.
``drift``
    ``26_mlops_governance/01``: PSI, K-S, prediction drift and the rolling per-index
    time-series IC. The reference window is train/validation only and ``assert_no_holdout``
    refuses a slice inside the holdout.
``online_detectors``
    ``26_mlops_governance/02``: ADWIN-style and DDM, every window counted in DECISIONS.
``rollout``
    ``26_mlops_governance/03``: shadow -> capped A/B -> staged -> full, the promotion gate
    fixed in YAML BEFORE any challenger exists, and ``NOT_COMPUTABLE`` while the holdout is
    unscored.
``retire``
    The two retire rules of ``BOT.md``. Both report ``not_computable`` today.

The **account** tier is not here: ``bots/_shared/monitor/`` stops every bot on the login at
once. This bot writes no account breaker and re-implements none.
"""
