"""Regression test for the shared CFD cost classifier and the 8-hour cadence multiplier.

This file sits in `bots/exness_btc_8h/tests/` and not beside the module it tests, for the same
reason `bots/tests/test_magic_numbers.py` does: the change is in SHARED code
(`case_studies/utils/backtest_loaders.py`), the bot that needed it is this one, and what has to be
proved is a property of the *set* of bots - that adding a crypto class moved nothing for the FX,
metals or cross universes that were already priced by it.

**The defect it closes.** `_cfd_pair_class` used to return `"major_pairs"` for any symbol whose
name contains `"USD"`. `BTCUSD` contains `"USD"`. So a crypto universe declaring
`costs.spread_bps: {crypto: [...]}` would have raised "no range for 'major_pairs'" - and a
universe that copied the FX template's `major_pairs: [0.6, 1.3]` instead would have been priced at
1.3 basis points a crossing when the measured in-sample p90 on this account is 17.4. That is the
same failure the `metals` branch was added to stop, recorded in the comment above
`_METAL_PREFIXES`: "would have priced gold at 1.3 bps in silence".

**The second change.** `_CADENCE_CALENDAR_DAYS_PER_PERIOD` had no `8_hour` token, so
`_calendar_days_per_period` fell back to 1.5 calendar days per period - four and a half times the
true 1/3 - and the allocator warmup prefix of every load would have been over-provisioned by that
factor. The entry is additive: no other case study declares `8_hour`, so no existing window moves.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/test_cfd_pair_class.py -q
"""

from __future__ import annotations

import pytest
import yaml

pytest.importorskip("polars")

from case_studies.utils.backtest_loaders import (  # noqa: E402
    _CADENCE_CALENDAR_DAYS_PER_PERIOD,
    _calendar_days_per_period,
    _cfd_pair_class,
    _intraday_cadence_interval,
    _normalize_costs,
)
from utils.paths import REPO_ROOT  # noqa: E402

# The five FX majors of exness_fx_d1, the two metals of exness_gold_sess, the two index CFDs of
# exness_usidx_sess and the eight pairs of xau_fx_mt5 - every symbol any bot on this account
# trades. Their classes must be exactly what they were before the crypto branch was added.
UNCHANGED = {
    "EURUSD": "major_pairs",
    "GBPUSD": "major_pairs",
    "USDJPY": "major_pairs",
    "AUDUSD": "major_pairs",
    "USDCAD": "major_pairs",
    "NZDUSD": "major_pairs",
    "USDCHF": "major_pairs",
    "XAUUSD": "metals",
    "XAGUSD": "metals",
    "XPTUSD": "metals",
    "XPDUSD": "metals",
    "EURGBP": "cross_pairs",
    "EURJPY": "cross_pairs",
    # `indices` was added to the same function by exness_usidx_sess on 2026-09-08, concurrently
    # with the `crypto` branch this file exists for. Recorded here rather than left as
    # `cross_pairs`, which is what this table said when it was first written and what the code
    # returned an hour earlier: two bots extending one classifier is exactly the situation this
    # test is for.
    "US500": "indices",
    "USTEC": "indices",
}


def test_btcusd_is_crypto_and_not_a_dollar_pair() -> None:
    """The one line the whole file exists for."""
    assert _cfd_pair_class("BTCUSD") == "crypto"
    assert "USD" in "BTCUSD", "the trap is that the name contains USD; if it stops, re-read this test"
    # The Market Watch name on this account carries the account suffix.
    assert _cfd_pair_class("BTCUSDm") == "crypto"
    assert _cfd_pair_class("btcusd") == "crypto", "the classifier upper-cases its input"


@pytest.mark.parametrize(("symbol", "expected"), sorted(UNCHANGED.items()))
def test_existing_universes_keep_their_class(symbol: str, expected: str) -> None:
    """No symbol any other bot trades changed class, so no registered identity moved."""
    assert _cfd_pair_class(symbol) == expected


def test_the_crypto_branch_is_a_prefix_and_not_a_substring() -> None:
    """A substring test would reclassify names that merely contain a ticker."""
    assert _cfd_pair_class("ETHUSD") == "crypto"
    assert _cfd_pair_class("XRPUSD") == "crypto"
    # Neither of these starts with a crypto prefix, and neither may be reclassified.
    assert _cfd_pair_class("USDBTC") == "major_pairs"
    assert _cfd_pair_class("EURUSD") == "major_pairs"


def test_the_declared_cost_block_resolves_to_the_in_sample_spread() -> None:
    """`_normalize_costs` charges the top of `costs.spread_bps.crypto`, not the live tick number.

    The engine's slippage leg comes from here, so this is the number every Ch16-19 backtest of
    2018-2025 pays. It has to be the IN-SAMPLE p90 (17.4 bps), because the same spread in points
    was five times more expensive in basis points when the price was a fifth of today's.
    """
    setup_path = REPO_ROOT / "case_studies" / "exness_btc_8h" / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text(encoding="utf-8"))
    commission_bps, slippage_bps = _normalize_costs(setup["costs"], "exness_btc_8h")
    declared_top = float(setup["costs"]["spread_bps"]["crypto"][-1])
    tick_p90 = float(setup["costs"]["spread_bps_by_session"]["BTCUSD"]["all"][1])
    assert commission_bps == 0.0, "the Pro account charges no commission"
    assert slippage_bps == declared_top
    assert slippage_bps > 5 * tick_p90, (
        f"the engine would charge {slippage_bps} bps a crossing while the live tick p90 is "
        f"{tick_p90}; costs.spread_bps must carry the in-sample range, not today's"
    )
    print(f"\ncommission {commission_bps} bps, slippage {slippage_bps} bps a crossing")


def test_the_eight_hour_cadence_resolves_to_the_declared_grid() -> None:
    """`8_hour` parses as an intraday cadence and lands on 00:00 / 08:00 / 16:00 UTC."""
    interval = _intraday_cadence_interval("8_hour")
    assert interval is not None and interval.total_seconds() == 8 * 3600
    # _on_the_clock keeps timestamps whose second-of-day is a whole multiple of the interval.
    day = 24 * 3600
    kept = [h for h in range(24) if (h * 3600) % int(interval.total_seconds()) == 0]
    assert kept == [0, 8, 16], f"an 8-hour cadence keeps hours {kept}"
    assert day % int(interval.total_seconds()) == 0, "8 hours must divide a day exactly"


def test_the_eight_hour_cadence_has_a_calendar_multiplier() -> None:
    """Without the table entry the warmup prefix would be over-provisioned 4.5-fold.

    `_calendar_days_per_period` takes a CASE STUDY, not a cadence token: it reads
    `decision.entry_cadence > decision.cadence > decision.bar_frequency` from that study's
    setup.yaml and looks the result up in the table. So the test drives it the way the loader does
    - through this bot's own declaration - and checks the table separately.
    """
    assert _CADENCE_CALENDAR_DAYS_PER_PERIOD["8_hour"] == pytest.approx(1.0 / 3.0)
    assert _calendar_days_per_period("exness_btc_8h") == pytest.approx(1.0 / 3.0), (
        "this bot's cadence token does not resolve in the table, so every allocator warmup "
        "prefix would be read 4.5x too wide from the 1.5-day fallback"
    )
    # The addition is additive: every token that was there keeps its value.
    assert _CADENCE_CALENDAR_DAYS_PER_PERIOD["daily_ny_close"] == 1.5
    assert _CADENCE_CALENDAR_DAYS_PER_PERIOD["8_hour_funding_aligned"] == pytest.approx(1.0 / 3.0)
    assert _CADENCE_CALENDAR_DAYS_PER_PERIOD["15_minute"] == pytest.approx((1.0 / 26.0) * 1.4)
    assert "a_token_nobody_declares" not in _CADENCE_CALENDAR_DAYS_PER_PERIOD
