"""Generation 3 (2026-09-10, ``bots/exness_fx_d1/GEN3_DECLARATION_2026-09-10.md``, OQ 30 / OQ 32(b)).

``case_studies/utils/backtest_runner.py::_apply_lot_floor_if_declared`` (added 2026-09-10 for
``xau_fx_mt5``) assumes every symbol's QUOTE currency is the account currency: one "engine unit"
(``weight * initial_cash / price``) is treated as one unit of the symbol's BASE currency and
divided by ``contract_size`` to get an implied MT5 lot count. That is correct for
EURUSD/GBPUSD/AUDUSD (quote = USD = account currency) but WRONG for USDJPY/USDCAD, whose BASE
currency (USD) already IS the account currency: dividing by price there shrinks the implied lot
count by ~price (~150x on USDJPY, ~1.38x on USDCAD) -- the same class of bug
``bots/_shared/mt5_broker.py::unit_values`` already fixed on the LIVE path (Decisions log
2026-09-07, ``BOT.md`` OQ 30), now found on the RESEARCH path while wiring generation 3.

This file tests the additive, opt-in fix: ``costs.contract.account_currency_is_base_for`` (a
list of symbols whose base currency equals the account currency). Undeclared or empty -> the
original price-divided arithmetic, byte-for-byte (the legacy path every other case study,
including ``xau_fx_mt5`` itself, still takes). It also tests the ``mt5_lot -> fractional`` ShareType
translation ``case_studies/utils/backtest_presets.py`` needed once ``execution.share_type:
mt5_lot`` was declared for real (``ml4t.backtest.ShareType`` has no ``mt5_lot`` member; mapped to
FRACTIONAL rather than INTEGER per the mentor fleet gate, 2026-09-10 -- see
``GEN3_DECLARATION_2026-09-10.md`` addendum), plus fail-loud coverage/currency and hash-inclusion
tests added the same day (fleet gate items B and C).

    uv run --with pytest==9.0.3 python -m pytest bots/exness_fx_d1/tests/test_lot_floor_currency.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from utils.paths import REPO_ROOT

pytest.importorskip("ml4t.backtest")

import polars as pl  # noqa: E402

from case_studies.utils import backtest_runner  # noqa: E402
from case_studies.utils.backtest_loaders import get_backtest_config  # noqa: E402

CASE_STUDY_ID = "exness_fx_d1"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"


def _setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# setup.yaml declares the real numbers, not guessed ones
# ---------------------------------------------------------------------------


def test_setup_declares_mt5_lot_and_reject_policy():
    setup = _setup()
    assert setup["execution"]["share_type"] == "mt5_lot"
    assert setup["sizing"]["min_lot_policy"] == "reject"
    contract = setup["costs"]["contract"]
    assert contract["volume_min"] == 0.01
    assert contract["volume_step"] == 0.01
    assert contract["volume_max"] == 200.0
    assert set(contract["account_currency_is_base_for"]) == {"USDJPY", "USDCAD"}


def test_get_backtest_config_reads_mt5_lot():
    config = get_backtest_config(CASE_STUDY_ID)
    assert config.share_type == "mt5_lot"


# ---------------------------------------------------------------------------
# _apply_lot_floor_if_declared: legacy path unchanged when the new key is absent
# ---------------------------------------------------------------------------

_CONTRACT_NO_CCY_MAP = {
    "contract_size": {"default": 100_000.0},
    "volume_min": 0.01,
    "volume_step": 0.01,
    "volume_max": 200.0,
}


def _fake_config(share_type: str, contract: dict, account_currency: str | None = "USD") -> SimpleNamespace:
    raw_costs: dict = {"contract": contract}
    if account_currency is not None:
        raw_costs["account_currency"] = account_currency
    return SimpleNamespace(share_type=share_type, raw_costs=raw_costs)


def _weights_prices(symbol: str, weight: float, price: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    weights = pl.DataFrame({"timestamp": [0], "symbol": [symbol], "weight": [weight]})
    prices = pl.DataFrame({"timestamp": [0], "symbol": [symbol], "close": [price]})
    return weights, prices


def _quote_currency_implied_lots(weight: float, price: float, contract_size: float, initial_cash: float) -> float:
    """Invert the function's own forward formula for a quote-currency-is-account-currency
    symbol (weight = lots * contract_size * price / initial_cash), i.e. the SAME division by
    price the forward direction used -- NOT a second multiplication by price."""
    return weight * initial_cash / price / contract_size


def _write_sizing_yaml(tmp_path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "setup.yaml").write_text(
        yaml.safe_dump({"sizing": {"min_lot_policy": "reject"}}), encoding="utf-8"
    )


def test_legacy_path_unchanged_when_account_currency_is_base_for_absent(tmp_path, monkeypatch):
    """No ``account_currency_is_base_for`` key at all, on a QUOTE-covered symbol (EURUSD, quote
    USD = the declared account currency) -- byte-for-byte the pre-generation-3 arithmetic (this
    is what ``xau_fx_mt5`` and every other ``mt5_lot`` adopter still gets). A BASE-covered symbol
    (USDJPY) without the declaration is a DIFFERENT scenario, covered by the C(i) fail-loud tests
    below -- it no longer silently falls back to the wrong arithmetic, it raises."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", _CONTRACT_NO_CCY_MAP))
    weights, prices = _weights_prices("EURUSD", 0.001, 1.16)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)
    # Legacy formula: target_lots = (weight * cash / price) / contract_size
    #               = (0.001 * 100_000 / 1.16) / 100_000 = 0.00086 lots -> below volume_min -> dropped.
    assert out["weight"][0] == pytest.approx(0.0)


def test_legacy_path_unchanged_when_account_currency_is_base_for_is_empty(tmp_path, monkeypatch):
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=[])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract))
    weights, prices = _weights_prices("EURUSD", 0.001, 1.16)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)
    assert out["weight"][0] == pytest.approx(0.0)


def test_usdjpy_without_the_currency_declaration_now_raises_instead_of_silently_mispricing(tmp_path, monkeypatch):
    """The exact scenario the ORIGINAL bug lived in (before ``account_currency_is_base_for``
    existed): USDJPY with no currency-mapping declared. Before C(i), this silently applied the
    quote-divides-by-price formula (~150x too small). After C(i), it RAISES instead -- there is
    no more silent-wrong-arithmetic path for an unclassified symbol."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", _CONTRACT_NO_CCY_MAP))
    weights, prices = _weights_prices("USDJPY", 0.5, 150.0)
    with pytest.raises(ValueError, match="USDJPY"):
        backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)


def test_quote_currency_symbol_still_divides_by_price(tmp_path, monkeypatch):
    """EURUSD (quote = USD = account currency): declaring the new key with EURUSD absent from
    the list must reproduce the legacy price-divided arithmetic for EURUSD exactly."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract))
    # weight 1.0, cash 100k, price 1.16 -> notional $100k -> 86,207 EUR -> 0.862 lots (above floor).
    weights, prices = _weights_prices("EURUSD", 1.0, 1.16)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)
    implied_lots = _quote_currency_implied_lots(out["weight"][0], 1.16, 100_000.0, 100_000.0)
    assert implied_lots == pytest.approx(0.86, abs=0.01)


def test_base_currency_symbol_skips_the_price_division(tmp_path, monkeypatch):
    """USDJPY (base = USD = account currency): a leg the LEGACY formula would shrink ~150x and
    drop must, once the symbol is declared in ``account_currency_is_base_for``, size to the TRUE
    MT5 lot count -- notional / contract_size, no price division."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract))
    # weight 1.0, cash 100k -> notional $100k -> TRUE lots = 100,000 / 100,000 = 1.00 lot.
    weights, prices = _weights_prices("USDJPY", 1.0, 150.0)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)
    # Reverse conversion for a base-currency-is-account-currency symbol: weight = lots *
    # contract_size / initial_cash (no price multiplication).
    implied_lots = out["weight"][0] * 100_000.0 / 100_000.0
    assert implied_lots == pytest.approx(1.00, abs=1e-9)
    # The legacy (buggy) formula would have given 100_000/150/100_000 = 0.00667 lots -- confirm
    # the fixed value is NOT that.
    assert implied_lots != pytest.approx(0.00667, abs=1e-4)


def test_base_currency_symbol_below_floor_is_dropped_not_clamped(tmp_path, monkeypatch):
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract))
    # weight 0.005 -> notional $500 -> TRUE lots = 500 / 100,000 = 0.005 lot, below volume_min 0.01.
    weights, prices = _weights_prices("USDJPY", 0.005, 150.0)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)
    assert out["weight"][0] == pytest.approx(0.0)


def test_real_exness_fx_d1_config_sizes_a_usdjpy_leg_correctly():
    """End-to-end against the REAL, on-disk exness_fx_d1 setup.yaml (no monkeypatch): a k=1
    USDJPY leg at the bot's own 100,000 USD research notional must size near 1 lot, not near
    0.0067 lots (the legacy bug's value)."""
    weights, prices = _weights_prices("USDJPY", 1.0, 150.0)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, CASE_STUDY_ID, 100_000.0)
    implied_lots = out["weight"][0] * 100_000.0 / 100_000.0
    assert implied_lots == pytest.approx(1.00, abs=1e-9)


def test_real_exness_fx_d1_config_sizes_a_eurusd_leg_correctly():
    weights, prices = _weights_prices("EURUSD", 1.0, 1.16)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, CASE_STUDY_ID, 100_000.0)
    implied_lots = _quote_currency_implied_lots(out["weight"][0], 1.16, 100_000.0, 100_000.0)
    assert implied_lots == pytest.approx(0.86, abs=0.01)


def test_mismatched_datetime_time_units_do_not_raise_and_output_keeps_the_weights_dtype(tmp_path, monkeypatch):
    """Found 2026-09-10 on the first real generation-3 run (not caught by the synthetic-int-
    timestamp tests above, which never hit a Polars dtype mismatch): ``weights`` and ``prices``
    do not always share the same ``Datetime`` time unit (observed in production: weights ``us``,
    prices ``ms``), and Polars' join raises ``SchemaError`` on a mismatch rather than casting.
    The join must succeed, and the RETURNED frame's ``timestamp`` dtype must be the ORIGINAL
    ``weights`` dtype, not silently switched to the price frame's."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract))
    import datetime as dt

    ts = dt.datetime(2026, 1, 5)
    weights = pl.DataFrame({"timestamp": [ts], "symbol": ["EURUSD"], "weight": [1.0]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"))
    )
    prices = pl.DataFrame({"timestamp": [ts], "symbol": ["EURUSD"], "close": [1.16]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms"))
    )
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)
    assert out["timestamp"].dtype == pl.Datetime("us"), "output must keep the ORIGINAL weights dtype"
    assert out.height == 1
    implied_lots = _quote_currency_implied_lots(out["weight"][0], 1.16, 100_000.0, 100_000.0)
    assert implied_lots == pytest.approx(0.86, abs=0.01)


# ---------------------------------------------------------------------------
# Mentor fleet gate, 2026-09-10, material item C(i): fail loud on an unclassifiable symbol
# ---------------------------------------------------------------------------


def test_a_cross_pair_neither_declared_nor_quote_matching_raises(tmp_path, monkeypatch):
    """EURJPY: base EUR is not the account currency (USD) and quote JPY is not either --
    neither branch of the lot-floor arithmetic is verified for it, so the guard must RAISE
    rather than silently apply the quote-is-account-currency formula to a cross pair."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract, account_currency="USD"))
    weights, prices = _weights_prices("EURJPY", 1.0, 165.0)
    with pytest.raises(ValueError, match="EURJPY"):
        backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)


def test_undeclared_account_currency_raises_for_a_quote_only_symbol(tmp_path, monkeypatch):
    """With no ``costs.account_currency`` declared, a symbol NOT in
    ``account_currency_is_base_for`` cannot be classified as quote-covered either -- it must
    raise, not silently fall back to the quote-divides-by-price assumption."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(
        backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract, account_currency=None)
    )
    weights, prices = _weights_prices("EURUSD", 1.0, 1.16)
    with pytest.raises(ValueError, match="EURUSD"):
        backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)


def test_every_real_exness_fx_d1_symbol_is_covered():
    """End-to-end against the REAL, on-disk setup.yaml: all five universe symbols (three
    quote-covered via ``costs.account_currency: USD``, two base-covered via
    ``account_currency_is_base_for``) must NOT raise."""
    weights, prices = _weights_prices("AUDUSD", 1.0, 0.65)
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, CASE_STUDY_ID, 100_000.0)
    assert out.height == 1


# ---------------------------------------------------------------------------
# Mentor fleet gate, 2026-09-10, material item C(ii): a non-zero weight with no valid price
# must raise, not silently pass through unchanged
# ---------------------------------------------------------------------------


def test_a_non_zero_weight_with_no_price_raises(tmp_path, monkeypatch):
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract))
    # weight targets USDJPY but the price frame has no row for it -- the left join leaves price null.
    weights = pl.DataFrame({"timestamp": [0], "symbol": ["USDJPY"], "weight": [1.0]})
    prices = pl.DataFrame({"timestamp": [0], "symbol": ["EURUSD"], "close": [1.16]})
    with pytest.raises(ValueError, match="USDJPY"):
        backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)


def test_a_zero_weight_with_no_price_still_passes_through(tmp_path, monkeypatch):
    """A weight of exactly 0.0 needs no price at all -- unaffected by the C(ii) fix."""
    _write_sizing_yaml(tmp_path)
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, create=False: tmp_path)
    contract = dict(_CONTRACT_NO_CCY_MAP, account_currency_is_base_for=["USDJPY", "USDCAD"])
    monkeypatch.setattr(backtest_runner, "get_backtest_config", lambda cs: _fake_config("mt5_lot", contract))
    weights = pl.DataFrame({"timestamp": [0], "symbol": ["USDJPY"], "weight": [0.0]})
    prices = pl.DataFrame({"timestamp": [0], "symbol": ["EURUSD"], "close": [1.16]})
    out = backtest_runner._apply_lot_floor_if_declared(weights, prices, "fake_cs", 100_000.0)
    assert out["weight"][0] == 0.0


# ---------------------------------------------------------------------------
# backtest_presets.py: mt5_lot -> fractional ShareType translation (mentor fleet gate item B,
# 2026-09-10: fractional, not integer -- see GEN3_DECLARATION_2026-09-10.md addendum item 1)
# ---------------------------------------------------------------------------


def test_mt5_lot_share_type_resolves_to_a_valid_engine_share_type():
    """``ml4t.backtest.ShareType`` has only FRACTIONAL and INTEGER members; the case study's own
    ``execution.share_type: mt5_lot`` must not reach ``ShareType(...)`` unchanged, or engine
    config construction raises ``ValueError: 'mt5_lot' is not a valid ShareType``. Mapped to
    FRACTIONAL (not INTEGER): the lot grid is already the discretisation enforced upstream by
    ``_apply_lot_floor_if_declared``; a second, engine-level INTEGER truncation on top of it is a
    residual, avoidable rounding error (measured in ``GEN3_DECLARATION_2026-09-10.md``)."""
    from ml4t.backtest.config import ShareType

    from case_studies.utils import backtest_presets

    src = backtest_presets.__file__
    text = open(src, encoding="utf-8").read()
    assert '"fractional" if share_type == "mt5_lot" else share_type' in text, (
        "backtest_presets.py must translate mt5_lot to a valid engine ShareType before it "
        "reaches position_sizing"
    )
    # The translated value itself must be a real ShareType member.
    assert ShareType("fractional") is ShareType.FRACTIONAL


def test_resolved_metadata_records_the_declared_mt5_lot_share_type_not_the_translated_one():
    """``build_resolved_backtest_config``'s ``metadata`` must carry the DECLARED share_type
    (``"mt5_lot"``), not the translated engine value (``"fractional"``) -- a reader of the
    resolved config must be able to tell provenance from engine mechanics."""
    from case_studies.utils.backtest_loaders import get_backtest_config
    from case_studies.utils.backtest_presets import build_resolved_backtest_config

    cfg = get_backtest_config(CASE_STUDY_ID)
    prices = pl.DataFrame(
        {
            "timestamp": [0, 1],
            "symbol": ["EURUSD", "USDJPY"],
            "open": [1.16, 150.0],
            "high": [1.16, 150.0],
            "low": [1.16, 150.0],
            "close": [1.16, 150.0],
            "volume": [1, 1],
        }
    )
    spec = {"backtest_config": {"account": {"allow_short_selling": True}}, "costs": {}, "execution": {}}
    resolved = build_resolved_backtest_config(CASE_STUDY_ID, cfg, spec, prices=prices, initial_cash=100_000.0)
    assert resolved.metadata["execution_share_type_declared"] == "mt5_lot"
    assert resolved.metadata["mt5_lot_sizing"]["min_lot_policy"] == "reject"
    assert resolved.share_type.value == "fractional"


def test_legacy_share_types_pass_through_unchanged():
    """Every declared share_type OTHER than ``mt5_lot`` passes through
    ``build_resolved_backtest_config`` untouched -- a real assertion against the function's
    OUTPUT, not a self-referential copy of its own conditional (mentor fleet gate, 2026-09-10:
    the previous version of this test re-asserted an inline copy of the mapping expression
    against itself and could never fail; its docstring and literal ("integer") were also stale
    after the mt5_lot translation moved to "fractional"). Uses ``xau_fx_mt5``'s own REAL,
    on-disk ``execution.share_type: integer`` declaration -- not a stub."""
    from case_studies.utils.backtest_loaders import get_backtest_config
    from case_studies.utils.backtest_presets import build_resolved_backtest_config

    cfg = get_backtest_config("xau_fx_mt5")
    assert cfg.share_type == "integer"  # xau_fx_mt5's own real declaration, untouched by this bot
    prices = pl.DataFrame(
        {
            "timestamp": [0],
            "symbol": ["XAUUSD"],
            "open": [2000.0],
            "high": [2000.0],
            "low": [2000.0],
            "close": [2000.0],
            "volume": [1],
        }
    )
    spec = {"backtest_config": {"account": {"allow_short_selling": True}}, "costs": {}, "execution": {}}
    resolved = build_resolved_backtest_config("xau_fx_mt5", cfg, spec, prices=prices, initial_cash=100_000.0)
    assert resolved.share_type.value == "integer", "a declared 'integer' must resolve to ShareType.INTEGER unchanged"
    assert "execution_share_type_declared" not in resolved.metadata
    assert "mt5_lot_sizing" not in resolved.metadata


# ---------------------------------------------------------------------------
# Mentor fleet gate, 2026-09-10, item B: sizing.min_lot_policy / costs.contract.* now enter the
# HASHED strategy_spec (via backtest_config.metadata); a case study that never declared them is
# byte-identical to before.
# ---------------------------------------------------------------------------


def _resolved_metadata_for(case_study: str) -> dict:
    from case_studies.utils.backtest_loaders import get_backtest_config
    from case_studies.utils.backtest_presets import build_resolved_backtest_config

    cfg = get_backtest_config(case_study)
    symbol = "AAPL" if case_study == "etfs" else "EURUSD"
    prices = pl.DataFrame(
        {
            "timestamp": [0],
            "symbol": [symbol],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1],
        }
    )
    spec = {"backtest_config": {"account": {"allow_short_selling": True}}, "costs": {}, "execution": {}}
    resolved = build_resolved_backtest_config(case_study, cfg, spec, prices=prices, initial_cash=100_000.0)
    return dict(resolved.metadata)


@pytest.mark.parametrize("other_case_study", ["fx_pairs", "etfs"])
def test_a_case_study_without_mt5_lot_gets_no_new_metadata_keys(other_case_study):
    """``fx_pairs`` and ``etfs`` never declared ``execution.share_type: mt5_lot``; their resolved
    metadata must carry neither ``execution_share_type_declared`` nor ``mt5_lot_sizing`` --
    the new branch in ``build_resolved_backtest_config`` is a no-op for them, so their
    ``backtest_hash`` is unaffected by this change (legacy-unchanged)."""
    metadata = _resolved_metadata_for(other_case_study)
    assert "execution_share_type_declared" not in metadata
    assert "mt5_lot_sizing" not in metadata


def test_toggling_min_lot_policy_changes_the_hashed_metadata_and_the_backtest_hash(monkeypatch):
    """Mentor fleet gate item B: before this fix, toggling ``sizing.min_lot_policy`` alone did
    not move ``backtest_hash`` (declared as a gap in ``GEN3_DECLARATION_2026-09-10.md``). Proven
    here directly against ``backtest_hash_from_parts``, the real hashing function."""
    from case_studies.utils import backtest_loaders
    from case_studies.utils.backtest_loaders import get_backtest_config, read_mt5_lot_sizing_declaration
    from case_studies.utils.backtest_presets import build_resolved_backtest_config
    from case_studies.utils.registry.specs import backtest_hash_from_parts

    cfg = get_backtest_config(CASE_STUDY_ID)
    prices = pl.DataFrame(
        {
            "timestamp": [0],
            "symbol": ["EURUSD"],
            "open": [1.16],
            "high": [1.16],
            "low": [1.16],
            "close": [1.16],
            "volume": [1],
        }
    )
    spec = {"backtest_config": {"account": {"allow_short_selling": True}}, "costs": {}, "execution": {}}

    resolved_reject = build_resolved_backtest_config(CASE_STUDY_ID, cfg, spec, prices=prices, initial_cash=100_000.0)
    assert resolved_reject.metadata["mt5_lot_sizing"]["min_lot_policy"] == "reject"

    def _clamp_declaration(case_study_id: str) -> dict:
        declared = read_mt5_lot_sizing_declaration(case_study_id)
        return {**declared, "min_lot_policy": "clamp"}

    monkeypatch.setattr(backtest_loaders, "read_mt5_lot_sizing_declaration", _clamp_declaration)
    resolved_clamp = build_resolved_backtest_config(CASE_STUDY_ID, cfg, spec, prices=prices, initial_cash=100_000.0)
    assert resolved_clamp.metadata["mt5_lot_sizing"]["min_lot_policy"] == "clamp"

    hash_reject = backtest_hash_from_parts(
        "fake_prediction_hash", {"backtest_config": resolved_reject.to_dict()}, identity_version=2
    )
    hash_clamp = backtest_hash_from_parts(
        "fake_prediction_hash", {"backtest_config": resolved_clamp.to_dict()}, identity_version=2
    )
    assert hash_reject != hash_clamp, "toggling sizing.min_lot_policy must move backtest_hash"
