"""Point-in-time test for the exness_fx_d1 feature panel (roadmap phase 2 gate).

Every feature must be recomputable at the decision timestamp from the bars that had closed
by then, and the result must equal the batch panel `03_financial_features` writes. The test
rebuilds the session panel and the whole feature matrix through the same functions the stage
calls (`case_studies/exness_fx_d1/_features.py`), from the raw four-hour bars truncated at a
sample of decision instants, and compares row by row. It also checks the session close against
the labels `02_labels` sealed, so the price the features are built on is the price the labels
are built on.

Needs the MT5 history (``ML4T_DATA_PATH/mt5/4h.parquet``) and the full project environment
(``ml4t.diagnostic`` for the session calendar); both are skipped when absent. Reads only the
development history: the holdout window (``evaluation.holdout_start`` onward) is never loaded.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_fx_d1/tests/test_lookahead.py -q
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

pytest.importorskip("ml4t.diagnostic", reason="the session calendar needs the full project environment")

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir  # noqa: E402
from case_studies.exness_fx_d1._features import (  # noqa: E402
    GOLD_SYMBOL,
    MACRO_CONFIG_PATH,
    VINTAGE_COLUMNS,
    bars_closed_by,
    build_features,
    carry_factor,
    feature_columns,
    features_as_of,
    load_carry_vintages,
    load_macro_config,
    published_rates,
    session_panel,
    vintages_published_by,
    warmup_expectations,
)
from case_studies.utils.feature_engineering import assert_values_agree, warmup_audit  # noqa: E402
from utils.paths import REPO_ROOT, get_case_study_dir  # noqa: E402

CASE_STUDY_ID = "exness_fx_d1"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
N_INSTANTS = 6  # decision instants sampled across the dense development history
# Windows for the carry family. The register in `setup.yaml` does not declare them yet - the
# family is blocked on the FRED key (`BOT.md` open question 10) - so the tests below declare
# them, which is also what turns the family on in `03_financial_features` once the data lands.
CARRY_WINDOWS = {"carry": [21, 63], "carry_zscore": 252}
# A publication lag far larger than any real one, so that a construction dating a value at the
# month it describes instead of the month it was published cannot pass by accident.
SYNTHETIC_PUBLICATION_LAG_DAYS = 45


def _naive_utc(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("timestamp").dt.replace_time_zone(None))


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_end(setup: dict) -> date:
    """The last calendar day this test may read: the day before the holdout opens."""
    return date.fromisoformat(str(setup["evaluation"]["holdout_start"])) - timedelta(days=1)


@pytest.fixture(scope="module")
def bars(setup: dict, dev_end: date) -> pl.DataFrame:
    if not (mt5_data_dir() / "4h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/4h.parquet)")
    return _naive_utc(
        load_mt5_bars(
            "4h",
            symbols=sorted(setup["universe"]["symbols"]),
            start_date=str(setup["universe"]["history_start"]),
            end_date=str(dev_end),
        )
    )


@pytest.fixture(scope="module")
def gold_bars(setup: dict, dev_end: date, bars: pl.DataFrame) -> pl.DataFrame:
    return _naive_utc(
        load_mt5_bars(
            "4h", symbols=[GOLD_SYMBOL], start_date=str(setup["universe"]["history_start"]), end_date=str(dev_end)
        )
    )


def _panel(frame: pl.DataFrame, setup: dict, *, keep_decision_ts: bool = False) -> pl.DataFrame:
    return session_panel(
        frame,
        calendar=setup["decision"]["session_calendar"],
        tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
        keep_decision_ts=keep_decision_ts,
        verbose=False,
    )


@pytest.fixture(scope="module")
def prices(bars: pl.DataFrame, setup: dict) -> pl.DataFrame:
    return _panel(bars, setup, keep_decision_ts=True)


@pytest.fixture(scope="module")
def gold(gold_bars: pl.DataFrame, setup: dict) -> pl.DataFrame:
    return _panel(gold_bars, setup)


def _build(prices: pl.DataFrame, gold: pl.DataFrame, setup: dict) -> pl.DataFrame:
    features = setup["features"]
    return build_features(
        prices.drop("decision_ts") if "decision_ts" in prices.columns else prices,
        windows=features["windows"],
        ranked=features["ranked"],
        periods_per_year=setup["evaluation"]["periods_per_year"],
        gold=gold,
    )


@pytest.fixture(scope="module")
def batch(prices: pl.DataFrame, gold: pl.DataFrame, setup: dict) -> pl.DataFrame:
    return _build(prices, gold, setup)


# ---------------------------------------------------------------------------
# The decision bar
# ---------------------------------------------------------------------------
def test_every_session_closes_on_its_decision_bar(prices: pl.DataFrame, setup: dict) -> None:
    """The decision instant is on the session's own date and never after the declared snapshot."""
    snapshot = datetime.strptime(setup["decision"]["snapshot_utc"], "%H:%M").time()
    assert (prices["decision_ts"].dt.date() == prices["timestamp"]).all()
    assert (prices["decision_ts"].dt.time() <= snapshot).all()
    assert prices.select(["symbol", "timestamp"]).is_duplicated().sum() == 0
    hours = prices["decision_ts"].dt.hour().value_counts().sort("decision_ts")
    print("\ndecision bar closes at (UTC hour -> sessions):", dict(zip(hours["decision_ts"], hours["count"], strict=True)))


def test_session_close_matches_sealed_labels(prices: pl.DataFrame, setup: dict) -> None:
    """close(t+1)/close(t) - 1 on the shared panel is the fwd_ret_1d 02_labels sealed, exactly."""
    label_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{setup['labels']['primary']}.parquet"
    if not label_path.exists():
        pytest.skip(f"labels not built yet: {label_path}")
    labels = pl.read_parquet(label_path)
    implied = prices.with_columns(
        (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("_implied")
    ).drop_nulls("_implied")
    joined = implied.join(labels, on=["symbol", "timestamp"], how="inner")
    assert joined.height == implied.height, "a development pair-session has no sealed label"
    gap = (joined["_implied"] - joined[setup["labels"]["primary"]]).abs().max()
    print(f"\n{joined.height:,} pair-sessions compared with the sealed labels, max |diff| {gap:.3e}")
    assert gap == 0.0


# ---------------------------------------------------------------------------
# The features
# ---------------------------------------------------------------------------
def test_warmup_matches_the_register(batch: pl.DataFrame, setup: dict) -> None:
    warmup_audit(batch, warmup_expectations(setup["features"]["windows"]), entity="symbol")


def test_withholding_later_dates_changes_nothing(prices: pl.DataFrame, gold: pl.DataFrame, batch: pl.DataFrame, setup: dict) -> None:
    """A transform fitted across the sample would move when the last years are removed."""
    cut = date(2022, 1, 1)
    withheld = _build(prices.filter(pl.col("timestamp") < cut), gold.filter(pl.col("timestamp") < cut), setup)
    assert_values_agree(
        batch.filter(pl.col("timestamp") < cut), withheld, columns=feature_columns(batch), keys=["timestamp", "symbol"]
    )


def _sampled_sessions(batch: pl.DataFrame, setup: dict) -> list[date]:
    carrier = setup["features"]["null_policy_carrier"]
    dense = batch.filter(pl.col(carrier).is_not_null())["timestamp"].unique().sort().to_list()
    picks = np.linspace(0, len(dense) - 1, N_INSTANTS + 2)[1:-1].round().astype(int)
    return [dense[i] for i in picks]


def test_features_recomputed_at_decision_time(
    bars: pl.DataFrame, gold_bars: pl.DataFrame, prices: pl.DataFrame, batch: pl.DataFrame, setup: dict
) -> None:
    """Rebuilding from the bars that had closed at the decision instant reproduces the batch row.

    The truncation is on the raw four-hour bars, so it tests the session aggregation as well as
    the features: a session that absorbed a bar closing after its decision would fail here.
    """
    cols = feature_columns(batch)
    instants = prices.select("timestamp", "decision_ts").unique().sort("timestamp")
    rows = []
    for session in _sampled_sessions(batch, setup):
        instant = instants.filter(pl.col("timestamp") == session)["decision_ts"][0]
        as_of = features_as_of(bars, instant, setup, gold_bars=gold_bars)
        assert as_of["timestamp"].max() == session, "the truncated panel does not end on the decided session"
        live = as_of.filter(pl.col("timestamp") == session).sort("symbol")
        ref = batch.filter(pl.col("timestamp") == session).sort("symbol")
        assert live.height == ref.height == len(setup["universe"]["symbols"])
        census = assert_values_agree(ref, live, columns=cols, keys=["timestamp", "symbol"])
        n_seen = bars_closed_by(bars, instant).height
        rows.append((session, instant, n_seen, float(census["max abs difference"].max())))
    print("\nrecomputed at the decision instant (session, instant UTC, bars seen, max |diff|):")
    for row in rows:
        print("  ", *row)


def test_batch_panel_matches_stage_artifact(batch: pl.DataFrame, setup: dict) -> None:
    """The batch build reproduces features/financial.parquet on every development row it holds."""
    path = get_case_study_dir(CASE_STUDY_ID) / "features" / "financial.parquet"
    if not path.exists():
        pytest.skip(f"stage artifact not built yet: {path}")
    artifact = pl.read_parquet(path)
    holdout_start = date.fromisoformat(str(setup["evaluation"]["holdout_start"]))
    artifact = artifact.filter(pl.col("timestamp") < holdout_start)
    cols = [c for c in artifact.columns if c not in {"timestamp", "symbol"}]
    assert set(cols) == set(feature_columns(batch)), "the artifact carries a different column set"
    rebuilt = batch.select("timestamp", "symbol", *cols).join(
        artifact.select("timestamp", "symbol"), on=["timestamp", "symbol"], how="inner"
    )
    assert rebuilt.height == artifact.height, "the artifact holds a development row the batch build does not"
    assert_values_agree(artifact.select(rebuilt.columns), rebuilt, columns=cols, keys=["timestamp", "symbol"])
    print(f"\n{artifact.height:,} artifact rows x {len(cols)} columns reproduced from the shared code path")


def test_artifact_never_precedes_its_inputs(setup: dict) -> None:
    """The sidecar names the price digests the matrix was built from (no artifact, no claim)."""
    sidecar = Path(get_case_study_dir(CASE_STUDY_ID) / "features" / "financial.parquet.digest.json")
    if not sidecar.exists():
        pytest.skip("stage artifact not built yet")
    import json

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "load_mt5_bars:4h" in record["inputs"]
    assert f"load_mt5_bars:4h:{GOLD_SYMBOL}" in record["inputs"]
    assert record["written_by"].endswith("03_financial_features.py")


# ---------------------------------------------------------------------------
# The carry family: the one input that is not a bar
# ---------------------------------------------------------------------------
# A price is knowable the moment it prints. A policy rate is knowable only when it is
# published, and for the four monthly legs of this universe that is weeks after the month the
# rate describes. The tests above cannot see that failure at all: they truncate BARS at a
# decision instant, and a macro column joined on its observation date would be reproduced
# identically by the truncated build and by the batch build, both of them wrong. So the carry
# family is tested on its own terms - against a synthetic vintage table whose publication lag is
# 45 days, large enough that observation-dating and publication-dating cannot agree by accident.
#
# The synthetic table is what makes these tests run without the FRED download, which is blocked
# (`BOT.md` open question 10). They test the join, the sign convention and the truncation rule,
# which is everything in the code path; only the values are stand-ins.
CURRENCY_LEVELS = {"USD": 2.0, "EUR": 0.5, "GBP": 1.5, "JPY": -0.1, "AUD": 3.0, "CAD": 2.5}


def _synthetic_vintages(first: date, last: date, *, lag_days: int = SYNTHETIC_PUBLICATION_LAG_DAYS) -> pl.DataFrame:
    """Monthly policy rates for six currencies, each published `lag_days` after its month.

    Shaped exactly like the ALFRED raw table `load_carry_vintages` returns: one row per series
    and observation date, with the vintage the observation was first published in, plus the
    `available_at` column the join reads.
    """
    legs = load_macro_config(MACRO_CONFIG_PATH)["legs"]
    months = pl.date_range(
        date(first.year - 2, 1, 1), date(last.year + 1, 1, 1), interval="1mo", eager=True
    )
    rows = []
    for offset, (currency, series_id) in enumerate(legs.items()):
        for step, observation in enumerate(months):
            rows.append(
                {
                    "series": str(series_id).lower(),
                    "timestamp": observation,
                    "vintage_date": observation + timedelta(days=lag_days),
                    # A level per currency plus a sawtooth whose STEP SIZE differs per
                    # currency. A ramp shared by every leg cancels in the differential, and so
                    # does a shared ramp merely shifted in phase: both leave the month-to-month
                    # change identical on the two legs, so a publication-date test on the
                    # differential could not fail. Different increments make the differential
                    # move on every publication.
                    "value": CURRENCY_LEVELS[currency] + 0.25 * ((step * (offset + 1)) % 9),
                }
            )
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("timestamp").cast(pl.Date),
            pl.col("vintage_date").cast(pl.Date),
            (pl.col("vintage_date") + pl.duration(days=1)).alias("available_at"),
        )
        .sort(["series", "available_at", "timestamp"])
    )


@pytest.fixture(scope="module")
def vintages(prices: pl.DataFrame) -> pl.DataFrame:
    return _synthetic_vintages(prices["timestamp"].min(), prices["timestamp"].max())


@pytest.fixture(scope="module")
def carry_legs() -> dict:
    return load_macro_config(MACRO_CONFIG_PATH)["legs"]


@pytest.fixture(scope="module")
def carry_panel(prices: pl.DataFrame, vintages: pl.DataFrame, carry_legs: dict) -> pl.DataFrame:
    return carry_factor(
        prices.drop("decision_ts"), vintages, legs=carry_legs, windows=CARRY_WINDOWS
    )


def test_macro_config_declares_a_leg_for_every_currency_in_the_universe(setup: dict, carry_legs: dict) -> None:
    """A pair whose leg is not declared must raise, not silently produce a null column."""
    needed = {s[:3] for s in setup["universe"]["symbols"]} | {
        s[3:6] for s in setup["universe"]["symbols"]
    }
    assert needed <= set(carry_legs), sorted(needed - set(carry_legs))
    print(f"\ncarry legs declared: {carry_legs}")


def test_availability_lag_is_at_least_one_day(tmp_path: Path) -> None:
    """A same-day read would be a claim about the release hour the data does not support."""
    config = yaml.safe_load(MACRO_CONFIG_PATH.read_text(encoding="utf-8"))
    assert int(config["carry"]["availability_lag_days"]) >= 1
    config["carry"]["availability_lag_days"] = 0
    bad = tmp_path / "macro_config.yaml"
    bad.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="availability_lag_days"):
        load_macro_config(bad)


def test_the_aligned_macro_panel_is_refused(tmp_path: Path, vintages: pl.DataFrame) -> None:
    """Only the raw table carries `vintage_date`; the aligned one is not point-in-time."""
    carry = dict(load_macro_config(MACRO_CONFIG_PATH))
    carry["raw_file"] = "aligned_like.parquet"
    # What `download_alfred.py::build_aligned_panel` writes: pivoted on the observation date,
    # no vintage column at all.
    vintages.pivot(on="series", index="timestamp", values="value", aggregate_function="first").write_parquet(
        tmp_path / "aligned_like.parquet"
    )
    with pytest.raises(ValueError, match="not the ALFRED raw table"):
        load_carry_vintages(carry, storage_dir=tmp_path)
    # And the raw table itself round-trips through the reader.
    vintages.select(VINTAGE_COLUMNS).write_parquet(tmp_path / "raw_like.parquet")
    carry["raw_file"] = "raw_like.parquet"
    read = load_carry_vintages(carry, storage_dir=tmp_path)
    assert "available_at" in read.columns
    assert (read["available_at"] - read["vintage_date"]).dt.total_days().unique().to_list() == [
        int(carry["availability_lag_days"])
    ]


def test_carry_sign_convention_on_one_pair_of_each_group(carry_panel: pl.DataFrame, vintages: pl.DataFrame, carry_legs: dict) -> None:
    """carry(XXXYYY) = rate(XXX) - rate(YYY), with the dollar on either leg.

    `EURUSD` quotes the dollar second and `USDJPY` quotes it first, so "foreign minus United
    States" would be right for one and backwards for the other. Both are checked against rates
    read independently of `carry_factor`.
    """
    grid = published_rates(vintages, carry_legs, carry_panel["timestamp"])
    for symbol, base, quote in (("EURUSD", "EUR", "USD"), ("USDJPY", "USD", "JPY")):
        got = carry_panel.filter(pl.col("symbol") == symbol).select("timestamp", "carry_diff")
        want = grid.select(
            "timestamp", (pl.col(f"_rate_{base}") - pl.col(f"_rate_{quote}")).alias("_want")
        )
        joined = got.join(want, on="timestamp", how="inner").drop_nulls()
        assert joined.height > 1000, joined.height
        gap = (joined["carry_diff"] - joined["_want"]).abs().max()
        print(f"\n{symbol}: {joined.height:,} sessions, carry = {base} - {quote}, max |diff| {gap:.3e}")
        assert gap == 0.0
    # And the two are not the same series, so the check is not vacuous.
    eur = carry_panel.filter(pl.col("symbol") == "EURUSD")["carry_diff"].drop_nulls()
    jpy = carry_panel.filter(pl.col("symbol") == "USDJPY")["carry_diff"].drop_nulls()
    assert float(eur.mean()) < 0.0 < float(jpy.mean())


def test_carry_reads_the_publication_date_and_not_the_observation_date(prices: pl.DataFrame, vintages: pl.DataFrame, carry_legs: dict) -> None:
    """The value a session reads is the last one PUBLISHED by then, never the one it describes.

    Built twice: once with the real vintage dates, once with the vintage set equal to the
    observation date - which is what reading the aligned macro panel amounts to. If the two
    agreed, the join would be dating a monthly rate at the month it describes and handing the
    decision weeks of information it could not have had.
    """
    naive = vintages.with_columns(
        pl.col("timestamp").alias("vintage_date"),
        (pl.col("timestamp") + pl.duration(days=1)).alias("available_at"),
    ).sort(["series", "available_at", "timestamp"])
    honest = carry_factor(prices.drop("decision_ts"), vintages, legs=carry_legs, windows=CARRY_WINDOWS)
    leaky = carry_factor(prices.drop("decision_ts"), naive, legs=carry_legs, windows=CARRY_WINDOWS)
    joined = honest.select("timestamp", "symbol", "carry_diff").join(
        leaky.select("timestamp", "symbol", pl.col("carry_diff").alias("_leaky")),
        on=["timestamp", "symbol"],
        how="inner",
    )
    differing = joined.filter(
        (pl.col("carry_diff") - pl.col("_leaky")).abs() > 1e-12
    )
    share = differing.height / joined.height
    print(
        f"\nobservation-dated vs publication-dated carry: {differing.height:,} of "
        f"{joined.height:,} pair-sessions differ ({share:.1%})"
    )
    assert share > 0.5, "the publication lag is not being applied at all"
    # A published value is never in the future of the session that reads it.
    for currency, series_id in carry_legs.items():
        published = vintages.filter(pl.col("series") == str(series_id).lower())
        first_available = published["available_at"].min()
        early = honest.filter(pl.col("timestamp") < first_available)
        assert early.height == 0 or early["carry_diff"].null_count() == early.height, currency


def test_no_unpublished_vintage_reaches_a_decision(prices: pl.DataFrame, vintages: pl.DataFrame, carry_legs: dict) -> None:
    """Withholding every vintage published after a session leaves that session's carry alone.

    This is the carry analogue of the bar truncation above: the panel is rebuilt from the macro
    values a trader at the decision could actually read, and every column has to match. A join
    that reached forward - on the observation date, or with a zero lag on a same-day release -
    would move here.
    """
    sessions = prices["timestamp"].unique().sort().to_list()
    picks = np.linspace(0, len(sessions) - 1, N_INSTANTS + 2)[1:-1].round().astype(int)
    full = carry_factor(prices.drop("decision_ts"), vintages, legs=carry_legs, windows=CARRY_WINDOWS)
    cols = ["carry_diff", "carry_chg_21d", "carry_chg_63d", "carry_z_252d"]
    rows = []
    for index in picks:
        session = sessions[index]
        visible = vintages_published_by(vintages, session)
        truncated = carry_factor(
            prices.filter(pl.col("timestamp") <= session).drop("decision_ts"),
            visible,
            legs=carry_legs,
            windows=CARRY_WINDOWS,
        )
        live = truncated.filter(pl.col("timestamp") == session).sort("symbol")
        batch = full.filter(pl.col("timestamp") == session).sort("symbol")
        census = assert_values_agree(batch, live, columns=cols, keys=["timestamp", "symbol"])
        rows.append(
            (
                str(session),
                int(visible.height),
                int(vintages.height - visible.height),
                float(census["max abs difference"].max()),
            )
        )
    print("\ncarry rebuilt from the vintages published by then (session, visible, withheld, max |diff|):")
    for row in rows:
        print("  ", *row)
    assert all(row[2] > 0 for row in rows), "nothing was withheld, so nothing was tested"


def test_carry_columns_are_recomputed_at_the_decision_instant(
    bars: pl.DataFrame, gold_bars: pl.DataFrame, prices: pl.DataFrame, vintages: pl.DataFrame, carry_legs: dict, setup: dict
) -> None:
    """The whole matrix, carry included, through the one `features_as_of` code path.

    The bars are truncated at the decision instant and the vintages at the same instant's
    publication cut-off, so this exercises both truncation rules at once - the shape the
    deployment loop will run in.
    """
    windows = {**setup["features"]["windows"], **CARRY_WINDOWS}
    batch = build_features(
        prices.drop("decision_ts"),
        windows=windows,
        ranked=setup["features"]["ranked"],
        periods_per_year=setup["evaluation"]["periods_per_year"],
        gold=session_panel(
            gold_bars,
            calendar=setup["decision"]["session_calendar"],
            tolerance_minutes=int(setup["decision"]["session_close_tolerance_minutes"]),
            verbose=False,
        ),
        carry_vintages=vintages,
        carry_legs=carry_legs,
    )
    assert {"carry_diff", "carry_chg_21d", "carry_chg_63d", "carry_z_252d"} <= set(batch.columns)
    with_carry_setup = {**setup, "features": {**setup["features"], "windows": windows}}
    cols = feature_columns(batch)
    instants = prices.select("timestamp", "decision_ts").unique().sort("timestamp")
    dense = batch.filter(pl.col(setup["features"]["null_policy_carrier"]).is_not_null())[
        "timestamp"
    ].unique().sort().to_list()
    picks = np.linspace(0, len(dense) - 1, 4)[1:-1].round().astype(int)
    rows = []
    for index in picks:
        session = dense[index]
        instant = instants.filter(pl.col("timestamp") == session)["decision_ts"][0]
        as_of = features_as_of(
            bars,
            instant,
            with_carry_setup,
            gold_bars=gold_bars,
            carry_vintages=vintages,
            carry_legs=carry_legs,
        )
        live = as_of.filter(pl.col("timestamp") == session).sort("symbol")
        ref = batch.filter(pl.col("timestamp") == session).sort("symbol")
        census = assert_values_agree(ref, live, columns=cols, keys=["timestamp", "symbol"])
        rows.append((str(session), str(instant), len(cols), float(census["max abs difference"].max())))
    print("\nfull matrix with carry, rebuilt at the decision instant (session, instant, columns, max |diff|):")
    for row in rows:
        print("  ", *row)


def test_warmup_expectations_cover_the_carry_columns(carry_panel: pl.DataFrame, setup: dict) -> None:
    """The two derived carry columns are audited; the level itself has no warmup."""
    windows = {**setup["features"]["windows"], **CARRY_WINDOWS}
    expected = warmup_expectations(windows)
    assert expected["carry_chg_63d"] == 63
    assert expected["carry_z_252d"] == 252
    warmup_audit(
        carry_panel,
        {k: v for k, v in expected.items() if k.startswith("carry_")},
        entity="symbol",
    )
