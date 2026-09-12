"""Point-in-time test for the exness_btc_8h feature panel (roadmap phase 2 gate).

Every feature must be recomputable at the decision timestamp from the bars that had closed by
then, and the result must equal the batch panel `03_financial_features` writes. The test rebuilds
the decision grid and the whole feature matrix through the same functions the stage calls
(`case_studies/exness_btc_8h/_features.py`), from raw bars truncated at a sample of decision
instants, and compares row by row. It also checks the decision close against the labels
`02_labels` sealed, so the price the features are built on is the price the labels are built on.

Three things make this bot's version different from `bots/exness_fx_d1/tests/test_lookahead.py`
and `bots/exness_gold_sess/tests/`, and each has its own test below:

* **two grids.** The long-window and cross-asset families are read from D1 bars and joined
  asof-backward on the D1 bar's CLOSE instant. `test_d1_join_is_backward` asserts the join and
  measures the age of the bar each row read - 0 hours at the 00:00 decision, 8 at 08:00, 16 at
  16:00 - which is the number a date-keyed join would silently turn negative.
* **swap.** Two feature columns claim to know which holds pay the broker's overnight financing.
  `test_swap_flags_match_the_broker_rule` checks them against
  `bots/_shared/costs_mt5.holding_cost_points` on the account's real `symbol_info` record rather
  than against the rule they were written from.
* **one instrument.** `test_no_cross_sectional_column` asserts that nothing in the matrix is a
  percentile, a rank or a cross-sectional z-score, because on one name every one of those is the
  constant 0.5 or undefined.

Needs the MT5 history (``ML4T_DATA_PATH/mt5/4h.parquet`` and ``daily.parquet``) and the full
project environment; both are skipped when absent. Reads only the development history: the
holdout window (``evaluation.holdout_start`` onward) is never loaded.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_btc_8h/tests/test_lookahead.py -q
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

pytest.importorskip("ml4t.engineer", reason="the feature construction needs the full project environment")

from bots._shared.costs_mt5 import holding_cost_points  # noqa: E402
from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir  # noqa: E402
from bots._shared.sessions import ServerClock  # noqa: E402
from case_studies.exness_btc_8h._features import (  # noqa: E402
    SLOT_MINUTES,
    bars_closed_by,
    build_features,
    decision_grid,
    feature_columns,
    features_as_of,
    load_d1,
    warmup_expectations,
)
from case_studies.utils.feature_engineering import assert_values_agree, warmup_audit  # noqa: E402
from utils.paths import REPO_ROOT, get_case_study_dir  # noqa: E402

CASE_STUDY_ID = "exness_btc_8h"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
N_INSTANTS = 6  # decision instants sampled across the dense development history
WITHHOLD_CUT = datetime(2023, 1, 1)


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_end(setup: dict) -> date:
    """The last calendar day this test may read: the day before the holdout opens."""
    return date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1)


@pytest.fixture(scope="module")
def bars(setup: dict, dev_end: date) -> pl.DataFrame:
    """The 8-hour bars, keyed on their OPEN - what ``load_mt5_bars`` returns."""
    if not (mt5_data_dir() / "4h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/4h.parquet)")
    return load_mt5_bars(
        "8h",
        symbols=sorted(setup["universe"]["symbols"]),
        start_date=str(setup["universe"]["history_start"]),
        end_date=str(dev_end),
    ).with_columns(pl.col("timestamp").dt.replace_time_zone(None))


@pytest.fixture(scope="module")
def d1(setup: dict, dev_end: date) -> pl.DataFrame:
    if not (mt5_data_dir() / "daily.parquet").exists():
        pytest.skip("MT5 daily history not downloaded")
    return load_d1(setup, sorted(setup["universe"]["symbols"]), end_date=str(dev_end))


@pytest.fixture(scope="module")
def reference(setup: dict, dev_end: date) -> pl.DataFrame:
    return load_d1(setup, list(setup["features"]["reference_symbols"]), end_date=str(dev_end))


@pytest.fixture(scope="module")
def panel(bars: pl.DataFrame) -> pl.DataFrame:
    return decision_grid(bars, verbose=False)


def _build(panel: pl.DataFrame, d1: pl.DataFrame, reference: pl.DataFrame, setup: dict) -> pl.DataFrame:
    return build_features(
        panel,
        d1,
        setup["features"]["windows"],
        reference=reference,
        slots_per_year=float(setup["decision"]["slots_per_year"]),
        rollover3days_dow=_rollover_dow(setup),
    )


def _rollover_dow(setup: dict) -> int:
    """``symbol_info.swap_rollover3days`` as an MT5 day of week, from the declared name."""
    names = {"sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
             "friday": 5, "saturday": 6}
    return names[str(setup["costs"]["swap"]["rollover3days"]).lower()]


@pytest.fixture(scope="module")
def batch(panel: pl.DataFrame, d1: pl.DataFrame, reference: pl.DataFrame, setup: dict) -> pl.DataFrame:
    return _build(panel, d1, reference, setup)


# ---------------------------------------------------------------------------
# The decision grid
# ---------------------------------------------------------------------------
def test_decision_grid_is_the_declared_one(panel: pl.DataFrame, setup: dict) -> None:
    """Hours, spacing and uniqueness - the three things a re-download could break silently."""
    declared = sorted(int(t.split(":")[0]) for t in setup["decision"]["snapshots_utc"])
    hours = sorted(panel["timestamp"].dt.hour().unique().to_list())
    assert hours == declared, f"decisions land on {hours}, setup.yaml declares {declared}"
    assert panel.select(["symbol", "timestamp"]).is_duplicated().sum() == 0
    deltas = (
        panel.sort(["symbol", "timestamp"])
        .with_columns((pl.col("timestamp") - pl.col("timestamp").shift(1).over("symbol")).alias("_d"))
        .drop_nulls("_d")["_d"]
        .unique()
        .to_list()
    )
    assert [int(d.total_seconds() // 60) for d in deltas] == [SLOT_MINUTES], (
        f"consecutive decisions are {sorted(int(d.total_seconds() // 60) for d in deltas)} minutes "
        f"apart, not a uniform {SLOT_MINUTES}"
    )
    print(f"\n{panel.height:,} decision slots on hours {hours}, all {SLOT_MINUTES} minutes apart")


def test_decision_close_matches_sealed_labels(panel: pl.DataFrame, setup: dict) -> None:
    """close(t+1)/close(t) - 1 on the shared panel is the fwd_ret_8h 02_labels sealed, exactly."""
    primary = setup["labels"]["primary"]
    label_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{primary}.parquet"
    if not label_path.exists():
        pytest.skip(f"labels not built yet: {label_path}")
    labels = pl.read_parquet(label_path)
    implied = panel.with_columns(
        (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("_implied")
    ).drop_nulls("_implied")
    joined = implied.join(labels, on=["symbol", "timestamp"], how="inner").drop_nulls(primary)
    assert joined.height > 0, "no development slot has a sealed label"
    gap = (joined["_implied"] - joined[primary]).abs().max()
    print(f"\n{joined.height:,} slots compared with the sealed labels, max |diff| {gap:.3e}")
    assert gap == 0.0


# ---------------------------------------------------------------------------
# The features
# ---------------------------------------------------------------------------
def test_warmup_matches_the_register(
    batch: pl.DataFrame, d1: pl.DataFrame, reference: pl.DataFrame, setup: dict
) -> None:
    """Each grid's columns are audited on the grid they count their warmup on.

    Auditing a D1 column on the 8-hour panel is meaningless and would raise: the D1 frame is read
    from 2014 for the cross-asset partners, so ``gold_ret_63d`` is already dense at panel row 1
    and an audit expecting 63 rows of warmup would call it "populated from fewer bars than its
    window spans".
    """
    from case_studies.exness_btc_8h._features import cross_asset, d1_state

    expected = warmup_expectations(setup["features"]["windows"])
    census_panel = warmup_audit(batch, expected["decision slots"], entity="symbol")
    d1_built = cross_asset(d1_state(d1, setup["features"]["windows"]), reference, setup["features"]["windows"])
    census_d1 = warmup_audit(d1_built, expected["D1 bars"], entity="symbol")
    print(
        f"\n{census_panel.height} columns audited on the decision grid, "
        f"{census_d1.height} on the D1 grid"
    )


def test_withholding_later_dates_changes_nothing(
    panel: pl.DataFrame, d1: pl.DataFrame, reference: pl.DataFrame, batch: pl.DataFrame, setup: dict
) -> None:
    """A transform fitted across the sample would move when the last years are removed."""
    cut = pl.lit(WITHHOLD_CUT)
    withheld = _build(
        panel.filter(pl.col("timestamp") < cut),
        d1.filter(pl.col("timestamp") < WITHHOLD_CUT.date()),
        reference.filter(pl.col("timestamp") < WITHHOLD_CUT.date()),
        setup,
    )
    census = assert_values_agree(
        batch.filter(pl.col("timestamp") < cut),
        withheld,
        columns=feature_columns(batch),
        keys=["timestamp", "symbol"],
    )
    print(
        f"\n{census['rows compared'].max():,} rows x {census.height} columns unchanged when "
        f"everything from {WITHHOLD_CUT.date()} is withheld"
    )


def _sampled_instants(batch: pl.DataFrame, setup: dict) -> list[datetime]:
    carrier = setup["features"]["null_policy_carrier"]
    dense = batch.filter(pl.col(carrier).is_not_null())["timestamp"].unique().sort().to_list()
    picks = np.linspace(0, len(dense) - 1, N_INSTANTS + 2)[1:-1].round().astype(int)
    return [dense[i] for i in picks]


def test_features_recomputed_at_decision_time(
    bars: pl.DataFrame,
    d1: pl.DataFrame,
    reference: pl.DataFrame,
    batch: pl.DataFrame,
    setup: dict,
) -> None:
    """Rebuilding from the bars that had closed at the decision instant reproduces the batch row.

    The truncation is on the raw bars of BOTH grids, so it tests the D1 asof join as well as the
    trailing windows: a column that read a D1 bar closing after the decision would differ here.
    """
    cols = feature_columns(batch)
    rows = []
    for instant in _sampled_instants(batch, setup):
        live = features_as_of(
            instant,
            bars,
            d1,
            setup["features"]["windows"],
            reference=reference,
            slots_per_year=float(setup["decision"]["slots_per_year"]),
            rollover3days_dow=_rollover_dow(setup),
        )
        assert live.height == len(setup["universe"]["symbols"]), (
            f"features_as_of returned {live.height} rows at {instant}; a comparison of two empty "
            "frames passes and proves nothing"
        )
        ref = batch.filter(pl.col("timestamp") == instant).sort("symbol")
        census = assert_values_agree(
            ref, live.sort("symbol"), columns=cols, keys=["timestamp", "symbol"]
        )
        n_seen = bars_closed_by(bars, instant).height
        rows.append((str(instant), n_seen, float(census["max abs difference"].max())))
    print("\nrecomputed at the decision instant (instant UTC, bars seen, max |diff|):")
    for row in rows:
        print("  ", *row)


def test_d1_join_is_backward(batch: pl.DataFrame) -> None:
    """Every row read a D1 bar that had already closed, and the age is the grid's own arithmetic.

    A join on the D1 bar's calendar DATE instead of its close instant would make this age
    negative at every decision - the single most common lookahead in a bot that mixes grids, and
    invisible in the feature values themselves.
    """
    assert (batch["d1_close_ts"] <= batch["timestamp"]).all()
    age = (
        batch.with_columns(
            ((pl.col("timestamp") - pl.col("d1_close_ts")).dt.total_hours()).alias("_age"),
            pl.col("timestamp").dt.hour().alias("_hour"),
        )
        .group_by("_hour")
        .agg(pl.col("_age").min().alias("min"), pl.col("_age").max().alias("max"))
        .sort("_hour")
    )
    observed = {int(r["_hour"]): (int(r["min"]), int(r["max"])) for r in age.iter_rows(named=True)}
    assert observed == {0: (0, 0), 8: (8, 8), 16: (16, 16)}, (
        f"the D1 bar's age at each decision hour is {observed}, not the grid's own arithmetic"
    )
    print(f"\nD1 bar age at the decision, by slot (hours): {observed}")


def test_swap_flags_match_the_broker_rule(batch: pl.DataFrame, setup: dict) -> None:
    """`pays_swap_night` and `pays_triple_swap` agree with costs_mt5 on the account's own record.

    The flags are built in ``_features.swap_and_calendar_state`` from the decision timestamp
    alone. Here they are checked against the shared cost module, driven by the swap parameters
    ``setup.yaml`` records from ``symbol_info`` - so an error in either implementation shows up
    rather than being confirmed by its own copy.
    """
    swap_block = setup["costs"]["swap"]
    symbol = sorted(setup["universe"]["symbols"])[0]
    params = {
        symbol: {
            "swap_long": float(swap_block["points_per_lot_per_night"][symbol]["long"]),
            "swap_short": float(swap_block["points_per_lot_per_night"][symbol]["short"]),
            "swap_mode": 1,
            "swap_rollover3days": _rollover_dow(setup),
            "charges_weekends": bool(swap_block["charges_weekends"]),
            "point": float(setup["costs"]["contract"]["point"]),
            "trade_contract_size": float(setup["costs"]["contract"]["contract_size"]),
        }
    }
    clock = ServerClock.from_dict(setup["decision"]["server_clock"])
    per_night = params[symbol]["swap_long"]
    sample = batch.sort("timestamp").head(3000)
    mismatched = 0
    for ts, flag, triple in sample.select(
        "timestamp", "pays_swap_night", "pays_triple_swap"
    ).iter_rows():
        entry = ts.replace(tzinfo=UTC)
        points = holding_cost_points(
            symbol, "long", entry, entry + timedelta(minutes=SLOT_MINUTES), swaps=params, clock=clock
        )
        expected_nights = 0.0 if points == 0 else (3.0 if triple else 1.0)
        if abs(points - expected_nights * per_night) > 1e-9 or (points != 0) != bool(flag):
            mismatched += 1
    assert mismatched == 0, f"{mismatched} rows disagree with costs_mt5.holding_cost_points"
    shares = (
        batch.group_by(pl.col("timestamp").dt.hour().alias("hour"))
        .agg(pl.col("pays_swap_night").mean(), pl.col("pays_triple_swap").mean())
        .sort("hour")
    )
    print(f"\n{sample.height:,} holds checked against costs_mt5; share paying swap by slot:")
    print(shares)
    paying = {int(r["hour"]) for r in shares.iter_rows(named=True) if r["pays_swap_night"] > 0}
    assert paying == {16}, f"slots paying overnight swap: {paying}, expected only the 16:00 one"


def test_no_cross_sectional_column(batch: pl.DataFrame, setup: dict) -> None:
    """One instrument: nothing in the matrix may be a rank, a percentile or a cross-sectional z.

    ``setup.yaml::features.ranked`` is empty by design, and this asserts the design reached the
    data. A percentile over one name is the constant 0.5 at every instant, so such a column would
    not merely be stale - it would be a constant regressor that a linear model cannot invert.
    """
    assert setup["features"]["ranked"] == [], "features.ranked is not empty on a one-name panel"
    banned = ("rank_", "xs_", "_pctile", "_percentile", "vs_median")
    offenders = [c for c in feature_columns(batch) if any(b in c for b in banned)]
    assert not offenders, f"cross-sectional columns on a one-instrument panel: {offenders}"
    # And no column may be constant over the whole development window: a constant carries nothing
    # and inflates the trial count of anything fitted on it.
    constant = [
        c
        for c in feature_columns(batch)
        if batch[c].drop_nulls().n_unique() <= 1
    ]
    assert not constant, f"columns constant over the development window: {constant}"
    print(f"\n{len(feature_columns(batch))} feature columns, none cross-sectional, none constant")


def test_batch_panel_matches_stage_artifact(batch: pl.DataFrame, setup: dict) -> None:
    """The batch build reproduces features/financial.parquet on every development row it holds."""
    path = get_case_study_dir(CASE_STUDY_ID) / "features" / "financial.parquet"
    if not path.exists():
        pytest.skip(f"stage artifact not built yet: {path}")
    artifact = pl.read_parquet(path)
    holdout_start = datetime.combine(
        date.fromisoformat(str(setup["evaluation"]["holdout_start"])), datetime.min.time()
    )
    artifact = artifact.filter(pl.col("timestamp") < holdout_start)
    cols = [c for c in artifact.columns if c not in {"timestamp", "symbol"}]
    assert set(cols) == set(feature_columns(batch)), "the artifact carries a different column set"
    rebuilt = batch.select("timestamp", "symbol", *cols).join(
        artifact.select("timestamp", "symbol"), on=["timestamp", "symbol"], how="inner"
    )
    assert rebuilt.height == artifact.height, "the artifact holds a development row the batch build does not"
    assert_values_agree(
        artifact.select(rebuilt.columns), rebuilt, columns=cols, keys=["timestamp", "symbol"]
    )
    print(f"\n{artifact.height:,} artifact rows x {len(cols)} columns reproduced from the shared code path")


def test_artifact_never_precedes_its_inputs(setup: dict) -> None:
    """The sidecar names the price digests the matrix was built from (no artifact, no claim)."""
    sidecar = Path(get_case_study_dir(CASE_STUDY_ID) / "features" / "financial.parquet.digest.json")
    if not sidecar.exists():
        pytest.skip("stage artifact not built yet")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "load_mt5_bars:8h" in record["inputs"]
    assert "load_mt5_bars:daily" in record["inputs"]
    assert "load_mt5_bars:daily:reference" in record["inputs"]
    assert record["written_by"].endswith("03_financial_features.py")
