"""Research-to-live parity for exness_fx_d1 — the gate of ``25_live_trading/08``.

Notebook 08 drives the same policy through a backtest pipeline and a live pipeline and then
asserts, stage by stage, that the two produced the same thing: same features, same
predictions, same signals, same sizes, same orders. Its own summary is that parity is a
**regression test**, not a one-off inspection, and that the four stages have to be compared
separately or a compensating pair of bugs passes.

The five stages here, in the order the deployment loop runs them:

1. **Data**: the live loop and the research stage read the same session panel, because both
   call ``case_studies/exness_fx_d1/_features.session_panel``. Checked by rebuilding the panel
   from only the bars that had closed at the decision instant.
2. **Features**: the row the live loop scores equals the row the batch build produces for the
   same session, on every column.
3. **Predictions**: in replay mode the loop's prediction tape is the registered validation
   prediction set, row for row - no refit, no rescoring.
4. **Sizing**: the units the live leg computes equal ``weight x equity / price``, and the lots
   are ``normalize_lot(units / trade_contract_size)`` read from ``symbol_info``.
5. **Orders**: the basket staged for the decision date equals the basket the offline
   ``ml4t.backtest.Engine`` replay chose on that same date.

Plus the property notebook 08 spends most of its length on: **determinism**. Two cycles over
the same inputs produce the same tape.

These tests read the experiment's registry, labels and MT5 parquet; without them they skip.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("ml4t.backtest", reason="parity needs the ml4t research environment")

from bots._shared.mt5_broker import normalize_lot  # noqa: E402
from bots._shared.mt5_loader import load_mt5_bars  # noqa: E402
from bots.exness_fx_d1.deploy import deployment_loop as loop  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    config = loop.load_deploy_config()
    if not loop.registry_path(config).exists():
        pytest.skip(f"no registry at {loop.registry_path(config)}; set ML4T_OUTPUT_DIR")
    if not (config.case_dir / "labels").exists():
        pytest.skip("no labels in the experiment; run 02_labels first")
    root = os.environ.get("ML4T_DATA_PATH")
    if root and not (Path(root) / "mt5" / "4h.parquet").exists():
        pytest.skip("no MT5 four-hour parquet")
    return config


@pytest.fixture(scope="module")
def viable_cfg(cfg, tmp_path_factory):
    """The shipped config, funded at `capital.min_viable_allocated`.

    `capital.allocated` is 900 USD, the equity measured on the demo login on 2026-09-07, and
    at that size every k=1 leg rounds below `volume_min` - there would be no sizing to compare.
    Only the allocation changes; the decision rule, the costs and the engine spec are the
    shipped ones, which is what parity is about.
    """
    import yaml

    raw = yaml.safe_load(loop.DEFAULT_RISK_CONFIG.read_text())
    raw["capital"]["allocated"] = float(raw["capital"]["min_viable_allocated"])
    path = tmp_path_factory.mktemp("parity_config") / "risk_config_viable.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return loop.load_deploy_config(path)


@pytest.fixture(scope="module")
def panel(cfg):
    return loop.recompute_features(cfg)


@pytest.fixture(scope="module")
def daily_prediction_hash(cfg):
    """A registered validation prediction set on the one-day label.

    ``rebalance_step[fwd_ret_1d] == 1``, so every session is a scheduled rebalance and the
    staged basket and the offline tape are comparable on the decision date itself. On the 5-
    and 21-day labels the last scheduled rebalance is usually an earlier session, which is
    correct behaviour and useless as a parity test.
    """
    db = sqlite3.connect(str(loop.registry_path(cfg)))
    try:
        row = db.execute(
            "SELECT p.prediction_hash FROM prediction_sets p "
            "JOIN training_runs t ON t.training_hash = p.training_hash "
            "WHERE p.split = 'validation' AND t.label = 'fwd_ret_1d' AND t.family = 'linear' "
            "ORDER BY p.created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        db.close()
    if row is None:
        pytest.skip("no registered validation prediction set on fwd_ret_1d")
    return row[0]


@pytest.fixture(scope="module")
def daily_run(viable_cfg, daily_prediction_hash, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity_cycle")
    return loop.run_cycle(
        model_selector="latest-complete",
        prediction_selector=daily_prediction_hash,
        mt5=loop.make_fake_terminal(viable_cfg),
        state_dir=tmp / "state",
        halt_file=tmp / "halt.json",
        risk_config=viable_cfg.risk_path,
    )


# =======================================================================================
# 1. Data parity — one panel, one rule, two consumers
# =======================================================================================
class TestStageOneData:
    def test_the_live_loop_and_the_research_stage_call_the_same_function(self):
        import inspect

        import case_studies.exness_fx_d1._features as features

        source = inspect.getsource(loop.recompute_features)
        assert "load_session_panel" in source and "build_features" in source
        assert features.session_panel.__module__ == "case_studies.exness_fx_d1._features"

    def test_the_panel_rebuilt_from_bars_closed_by_the_decision_instant_matches(self, cfg, panel):
        """The point-in-time claim: nothing in the decision row needs a bar that had not closed."""
        from case_studies.exness_fx_d1._features import GOLD_SYMBOL, bars_closed_by, session_panel

        decision_ts = panel.decision_ts
        bars = load_mt5_bars(
            "4h",
            symbols=[*cfg.universe, GOLD_SYMBOL],
            start_date=str(cfg.setup["universe"]["history_start"]),
        )
        truncated = session_panel(
            bars_closed_by(bars.filter(pl.col("symbol").is_in(cfg.universe)), decision_ts),
            calendar=cfg.setup["decision"]["session_calendar"],
            tolerance_minutes=cfg.tolerance_minutes,
            verbose=False,
        )
        live_row = truncated.filter(pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date)).sort("symbol")
        batch_row = (
            panel.prices.filter(pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date))
            .drop("decision_ts")
            .sort("symbol")
        )
        assert live_row.height == batch_row.height == len(cfg.universe)
        for column in ("open", "high", "low", "close", "volume"):
            assert live_row[column].to_list() == batch_row[column].to_list(), column

    def test_no_decision_bar_closes_after_the_declared_snapshot(self, cfg, panel):
        check = loop.decision_bar_check(cfg, panel)
        assert all(entry["lag_minutes"] >= 0 for entry in check["per_symbol"])


# =======================================================================================
# 2. Feature parity — the row the model sees
# =======================================================================================
class TestStageTwoFeatures:
    def test_the_decision_row_equals_the_batch_row_on_every_column(self, cfg, panel):
        from case_studies.exness_fx_d1._features import GOLD_SYMBOL, features_as_of

        bars = load_mt5_bars(
            "4h", symbols=cfg.universe, start_date=str(cfg.setup["universe"]["history_start"])
        )
        gold = load_mt5_bars(
            "4h", symbols=[GOLD_SYMBOL], start_date=str(cfg.setup["universe"]["history_start"])
        )
        as_of = features_as_of(bars, panel.decision_ts, cfg.setup, gold_bars=gold)
        live = as_of.filter(pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date)).sort("symbol")
        batch = panel.features.filter(
            pl.col("timestamp") == pl.lit(panel.decision_date).cast(pl.Date)
        ).sort("symbol")
        assert live.height == batch.height == len(cfg.universe)
        worst = 0.0
        for column in panel.feature_columns:
            a = live[column].to_numpy().astype(float)
            b = batch[column].to_numpy().astype(float)
            both_null = np.isnan(a) & np.isnan(b)
            diff = np.where(both_null, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
            worst = max(worst, float(diff.max()))
        assert worst == 0.0, f"features differ by {worst} between the live and the batch build"

    def test_the_feature_column_set_is_the_one_the_research_register_declares(self, cfg, panel):
        # 9 families in setup.yaml::features.families; the count is the register's, not a guess.
        assert len(panel.feature_columns) == len(set(panel.feature_columns))
        assert "usd_corr_63d" in panel.feature_columns
        assert "gold_corr_63d" in panel.feature_columns
        assert not any(c in panel.feature_columns for c in ("open", "high", "low", "close", "volume"))


# =======================================================================================
# 3. Prediction parity — the replay is the registry, not a rescoring
# =======================================================================================
class TestStageThreePredictions:
    def test_the_replayed_tape_is_the_registered_prediction_set_row_for_row(self, cfg, daily_prediction_hash):
        from case_studies.utils.registry import read_predictions

        choice = loop.resolve_model(cfg, "latest-complete", prediction_hash=daily_prediction_hash)
        replayed = loop.read_registered_predictions(cfg, choice)
        registered = read_predictions("exness_fx_d1", daily_prediction_hash, case_dir=cfg.case_dir)
        # read_predictions normalises the legacy column names, so accept either spelling.
        score_column = next(c for c in ("y_score", "prediction", "score") if c in registered.columns)
        registered = (
            registered.select(
                pl.col("timestamp").cast(pl.Date),
                "symbol",
                pl.col(score_column).cast(pl.Float64).alias("score"),
            )
            .sort(["timestamp", "symbol"])
        )
        assert replayed.height == registered.height
        assert replayed["score"].to_list() == registered["score"].to_list()
        assert replayed["symbol"].to_list() == registered["symbol"].to_list()

    def test_the_run_records_where_the_predictions_came_from(self, daily_run, daily_prediction_hash):
        assert daily_run["steps"]["5_predict"]["source"] == f"registry:{daily_prediction_hash}"
        assert daily_run["steps"]["3_retrain"]["skipped"] is True


# =======================================================================================
# 4. Sizing parity — units, then lots from symbol_info
# =======================================================================================
class TestStageFourSizing:
    def test_units_follow_weight_times_equity_over_the_unit_value(self, daily_run, viable_cfg):
        from bots._shared.mt5_broker import MT5Broker

        stage = daily_run["steps"]["7_stage"]
        prices = {
            row["symbol"]: float(row["ref_price"]) for row in stage["legs"] if row["ref_price"]
        }
        broker = MT5Broker(magic=viable_cfg.magic, mt5=loop.make_fake_terminal(viable_cfg))
        broker.account_snapshot = {"currency": "USD"}
        expected = loop.target_units(
            stage["intended_weights"], loop.unit_values(broker, prices), viable_cfg.allocated
        )
        for leg in stage["legs"]:
            assert leg["units"] == pytest.approx(expected[leg["symbol"]])

    def test_the_live_quantity_of_a_jpy_quoted_pair_differs_from_the_offline_tape(self, viable_cfg):
        """The two "quantities" are different quantities, and this pins the difference.

        Both conventions are correct inside their own system, so parity is on baskets and
        sides, never on raw unit counts:

        * the **engine** treats ``price`` as the account-currency value of one unit on both
          sides - it sizes ``target_notional / price`` (``ml4t/backtest/preopen.py:800``) and
          values ``quantity * price`` (``broker.py:2087``). A 2,400 USD weight-1 ``USDJPY``
          leg is ~16 "units" that are worth 2,400 USD and earn the pair's percentage move.
        * the **live leg** must speak MT5, where a lot is 100,000 units of the **base**
          currency, so 2,400 USD of ``USDJPY`` exposure is 2,400 units (0.024 lots).

        Sanity check on the registry rather than on theory: the generation-2 winner's
        ``fills.parquet`` has a median leg notional of 110,277 USD on ``USDJPY`` against
        100,028 on ``EURUSD`` - same order of magnitude, so the engine is holding the exposure
        the weights asked for.

        What would be a real bug is using the engine's arithmetic to compute MT5 lots, which
        is what ``unit_values`` prevents.
        """
        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=viable_cfg.magic, mt5=loop.make_fake_terminal(viable_cfg))
        broker.account_snapshot = {"currency": "USD"}
        quote = 154.304
        notional = viable_cfg.allocated

        live_units = loop.target_units(
            {"USDJPY": 1.0}, loop.unit_values(broker, {"USDJPY": quote}), notional
        )["USDJPY"]
        engine_units = round(notional * 1.0 / quote)  # the engine's own unit convention

        assert live_units == pytest.approx(notional)
        assert engine_units == pytest.approx(notional / quote, abs=1.0)
        assert live_units > 100 * engine_units

        # ...and the two describe the SAME exposure: the engine values its units at `price`,
        # so its leg is worth the notional it was asked for - to within ONE integer unit,
        # which is the granularity caveat and is worth seeing rather than hiding. At this
        # 2,400 USD allocation a USDJPY leg is only ~15.6 units, so one unit is 6.4 % of the
        # leg; at the research `initial_cash` of 100,000 the same leg is ~650 units and one
        # unit is 0.15 %. The live leg has no such rounding because its unit is one USD.
        assert engine_units * quote == pytest.approx(notional, abs=quote)
        assert live_units * 1.0 == pytest.approx(notional)
        assert abs(engine_units * quote - notional) / notional < 0.07

        # The same comparison on a USD-quoted pair must agree, so the divergence is provably
        # about the quote currency and not about the sizing code in general.
        eur_quote = 1.1629
        eur_live = loop.target_units(
            {"EURUSD": 1.0}, loop.unit_values(broker, {"EURUSD": eur_quote}), notional
        )["EURUSD"]
        assert eur_live == pytest.approx(round(notional / eur_quote), abs=1.0)

    def test_lots_are_normalize_lot_of_units_over_the_contract_size(self, daily_run, viable_cfg):
        fake = loop.make_fake_terminal(viable_cfg)
        from bots._shared.mt5_broker import MT5Broker

        broker = MT5Broker(magic=viable_cfg.magic, mt5=fake)
        for leg in daily_run["steps"]["7_stage"]["legs"]:
            info = broker.symbol_info(leg["symbol"])
            assert float(info.trade_contract_size) == leg["contract_size"]
            assert leg["lots"] == normalize_lot(
                abs(leg["units"]) / leg["contract_size"], info, min_lot_policy="reject"
            )

    def test_the_book_is_dollar_neutral_in_weight_terms(self, daily_run, viable_cfg):
        weights = daily_run["steps"]["7_stage"]["intended_weights"]
        assert viable_cfg.model_block["long_short"] is True
        assert sum(weights.values()) == pytest.approx(0.0)
        assert sorted(np.sign(list(weights.values()))) == [-1.0, 1.0]


# =======================================================================================
# 5. Order parity — the staged basket against the offline tape
# =======================================================================================
class TestStageFiveOrders:
    def test_the_staged_basket_equals_the_offline_replays_basket_on_the_same_date(self, daily_run):
        stage = daily_run["steps"]["7_stage"]
        assert stage["parity_agrees_with_last_rebalance"] is True, (
            "the basket staged for the decision date differs from the basket the offline "
            f"replay chose: staged {stage['intended_weights']}, "
            f"offline {daily_run['steps']['6_parity'].get('last_rebalance')}"
        )

    def test_the_offline_tape_is_non_empty_and_dated(self, daily_run):
        parity = daily_run["steps"]["6_parity"]
        assert parity["n_signals"] > 0
        assert parity["n_rebalances"] > 0
        assert parity["last_rebalance"]["date"] == daily_run["decision_date"]

    def test_every_staged_leg_is_in_the_declared_universe(self, viable_cfg, daily_run):
        allowed = set(viable_cfg.live_risk_block["allowed_assets"])
        assert set(daily_run["steps"]["7_stage"]["intended_basket"]) <= allowed

    def test_nothing_reached_the_broker(self, daily_run):
        stage = daily_run["steps"]["7_stage"]
        assert stage["orders_can_reach_broker"] is False
        assert stage["accepted_basket"] == []
        assert all(leg["status"] == "dry_run" for leg in stage["legs"])


# =======================================================================================
# Determinism — parity is a regression test, not an inspection
# =======================================================================================
class TestDeterminism:
    def test_two_cycles_over_the_same_inputs_produce_the_same_tape(self, viable_cfg, daily_prediction_hash, tmp_path):
        runs = []
        for i in range(2):
            runs.append(
                loop.run_cycle(
                    model_selector="latest-complete",
                    prediction_selector=daily_prediction_hash,
                    mt5=loop.make_fake_terminal(viable_cfg),
                    state_dir=tmp_path / f"state{i}",
                    halt_file=tmp_path / f"halt{i}.json",
                    risk_config=viable_cfg.risk_path,
                )
            )
        first, second = (r["steps"]["6_parity"] for r in runs)
        assert first["tape"] == second["tape"]
        assert first["last_rebalance"] == second["last_rebalance"]
        assert runs[0]["steps"]["7_stage"]["intended_weights"] == runs[1]["steps"]["7_stage"]["intended_weights"]
        assert runs[0]["decision_date"] == runs[1]["decision_date"]

    def test_the_engine_configuration_is_the_research_one_and_not_retyped(self, daily_run, viable_cfg):
        from case_studies.utils.backtest_loaders import get_backtest_config

        research = get_backtest_config("exness_fx_d1")
        engine = daily_run["steps"]["6_parity"]["engine"]
        assert engine["commission_bps"] == research.commission_bps
        assert engine["slippage_bps"] == research.slippage_bps
        # Generation 3 (2026-09-10, GEN3_DECLARATION_2026-09-10.md, OQ 32(b)): research.share_type
        # is "mt5_lot" (case_studies/utils/backtest_runner.py::_apply_lot_floor_if_declared's
        # gate, a weights-level guard); ml4t.backtest.ShareType has no "mt5_lot" member, so the
        # parity engine resolves to "fractional" (mentor fleet gate item B, 2026-09-10 -- not
        # "integer": a second engine-level truncation on top of the lot grid is an avoidable
        # residual discretisation, GEN3_DECLARATION_2026-09-10.md addendum item 1), the same
        # translation case_studies/utils/backtest_presets.py applies for the research sweep.
        expected_engine_share_type = "fractional" if research.share_type == "mt5_lot" else research.share_type
        assert engine["share_type"] == expected_engine_share_type
        assert engine["initial_cash"] == viable_cfg.allocated


class TestTheEngineHoldsTheNotionalTheWeightsAskedFor:
    """Read from the registry, so the claim rests on the run and not on the docstring.

    This exists because an earlier version of this file asserted that the research engine
    under-sized JPY- and CAD-quoted legs by the quote. It does not: the median leg notional on
    the generation-2 winner is the same order of magnitude on every pair. What IS uneven is
    the integer-unit granularity, and that is what the second test measures.
    """

    @pytest.fixture(scope="class")
    def fills(self, request):
        cfg = loop.load_deploy_config()
        path = cfg.case_dir / "run_log" / "backtest" / "fecd7db8d885" / "fills.parquet"
        if not path.exists():
            pytest.skip(f"no fills for the generation-2 winner at {path}")
        return (
            pl.read_parquet(path)
            .with_columns((pl.col("quantity").abs() * pl.col("price")).alias("notional"))
            .group_by("asset")
            .agg(
                pl.col("quantity").abs().median().alias("med_qty"),
                pl.col("notional").median().alias("med_notional"),
            )
            .sort("asset")
        )

    def test_a_jpy_quoted_leg_carries_the_same_order_of_notional_as_a_usd_quoted_one(self, fills):
        by_asset = {r["asset"]: r for r in fills.to_dicts()}
        jpy = by_asset["USDJPY"]["med_notional"]
        eur = by_asset["EURUSD"]["med_notional"]
        assert jpy > 50_000, f"USDJPY median leg notional is only {jpy:,.0f}"
        assert 0.2 < jpy / eur < 5.0, f"USDJPY/EURUSD notional ratio {jpy / eur:.2f}"

    def test_integer_units_make_the_rounding_granularity_uneven_across_pairs(self, fills):
        """The one real caveat, and it is small: 0.14 % on USDJPY, 0.001 % on EURUSD.

        `execution.share_type: integer` means a leg is a whole number of the engine's units.
        Because those units are worth `price` each, a pair quoted at 142 holds ~722 of them
        and a pair quoted at 1.11 holds ~87,000, so one unit of rounding is a systematically
        larger share of the JPY leg. At `initial_cash` 100,000 it is far below the 2.6 bp
        round-trip cost, but it is not zero and it is not symmetric across pairs.
        """
        granularity = {r["asset"]: 100.0 / r["med_qty"] for r in fills.to_dicts()}
        assert granularity["USDJPY"] > 10 * granularity["EURUSD"]
        assert granularity["USDJPY"] < 1.0, "a whole unit should still be well under 1 % of a leg"
