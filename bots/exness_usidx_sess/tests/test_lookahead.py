"""Point-in-time test for the exness_usidx_sess feature panels (roadmap phase 2 gate).

Every feature must be recomputable at the decision instant from the bars that had closed by
then, and the result must equal the batch panel. The test rebuilds both session panels and both
feature matrices through the same functions the stages call
(``case_studies/exness_usidx_sess/_features.py``), from the raw one-hour bars truncated at a
sample of decision instants, and compares row by row.

Two things here that ``exness_fx_d1``'s version of this test does not have to check, because it
has one panel and this bot has two:

* **the two specs are checked separately.** The ``intraday`` and ``overnight`` panels sit on the
  same sessions but decide six hours apart, and the second one may read a column the first one
  may not - ``intraday_ret`` finishes at the cash close. A leak there is invisible in the batch
  matrix and shows up here as a null on the truncated rebuild.
* **the labels are checked against the panel that sealed them**, and each panel's ``exec_open``
  and ``label_close`` are checked to be strictly after the decision instant they belong to.

Needs the MT5 history (``ML4T_DATA_PATH/mt5/1h.parquet``) and the full project environment
(``ml4t.diagnostic`` for the NYSE calendar); both are skipped when absent. Reads only the
development history: the holdout window (``evaluation.holdout_start`` onward) is never loaded.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_usidx_sess/tests/test_lookahead.py -q
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import yaml

pytest.importorskip(
    "ml4t.diagnostic", reason="the NYSE calendar needs the full project environment"
)

from bots._shared.mt5_loader import load_mt5_bars, mt5_data_dir  # noqa: E402
from case_studies.exness_usidx_sess._features import (  # noqa: E402
    SPECS,
    bars_closed_by,
    build_features,
    feature_columns,
    features_as_of,
    load_daily_bars,
    session_panel,
    warmup_expectations,
)
from case_studies.utils.feature_engineering import assert_values_agree, warmup_audit  # noqa: E402
from utils.paths import REPO_ROOT, get_case_study_dir  # noqa: E402

CASE_STUDY_ID = "exness_usidx_sess"
SETUP_PATH = REPO_ROOT / "case_studies" / CASE_STUDY_ID / "config" / "setup.yaml"
N_INSTANTS = 6  # decision instants sampled across the dense development history


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
    if not (mt5_data_dir() / "1h.parquet").exists():
        pytest.skip("MT5 history not downloaded (ML4T_DATA_PATH/mt5/1h.parquet)")
    return _naive_utc(
        load_mt5_bars(
            "1h",
            symbols=sorted(setup["universe"]["symbols"]),
            start_date=str(setup["universe"]["history_start"]),
            end_date=str(dev_end),
        )
    )


def _panel(frame: pl.DataFrame, setup: dict, spec: str) -> pl.DataFrame:
    decision = setup["decision"]
    return session_panel(
        frame,
        spec=spec,
        calendar=decision["session_calendar"],
        tolerance_minutes=int(decision["session_close_tolerance_minutes"]),
        open_delay_minutes=int(decision["open_delay_minutes"]),
        verbose=False,
    )


@pytest.fixture(scope="module")
def panels(bars: pl.DataFrame, setup: dict) -> dict[str, pl.DataFrame]:
    return {spec: _panel(bars, setup, spec) for spec in SPECS}


@pytest.fixture(scope="module")
def daily(setup: dict, dev_end: date) -> pl.DataFrame:
    """The D1 bars the long-window families are built on, truncated at the holdout.

    The H1 history cannot warm a window longer than 63 sessions, so the momentum, volatility and
    trend families at a quarter, a half year and a year come from data/mt5/daily.parquet instead,
    joined at the last bar whose UTC day had ended at the decision. That join is the one place in
    the construction where two frequencies meet, so it is the one most able to hide a lookahead,
    and the point-in-time test below truncates BOTH files rather than only the hourly one.
    """
    if not (mt5_data_dir() / "daily.parquet").exists():
        pytest.skip("MT5 daily history not downloaded (ML4T_DATA_PATH/mt5/daily.parquet)")
    return load_daily_bars(setup, end_date=str(dev_end))

def _build(
    prices: pl.DataFrame, setup: dict, spec: str, daily: pl.DataFrame
) -> pl.DataFrame:
    features = setup["features"]
    return build_features(
        prices,
        spec=spec,
        windows=features["windows"],
        daily_windows=features["daily_windows"],
        ranked=features.get("ranked") or [],
        periods_per_year=setup["evaluation"]["periods_per_year"],
        daily=daily,
    )


@pytest.fixture(scope="module")
def batches(
    panels: dict[str, pl.DataFrame], setup: dict, daily: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    return {spec: _build(panel, setup, spec, daily) for spec, panel in panels.items()}


# ---------------------------------------------------------------------------
# The decision instants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", SPECS)
def test_decision_precedes_execution_which_precedes_the_label(
    panels: dict[str, pl.DataFrame], spec: str
) -> None:
    """decision_ts <= exec_ts < label_ts on every row that carries a trade."""
    panel = panels[spec]
    assert panel.select(["symbol", "timestamp"]).is_duplicated().sum() == 0
    traded = panel.drop_nulls(["exec_ts", "label_ts"])
    assert (traded["exec_ts"] >= traded["decision_ts"]).all(), "a fill precedes its decision"
    assert (traded["label_ts"] > traded["exec_ts"]).all(), "an exit precedes its entry"
    assert (panel["decision_ts"].dt.date() == panel["timestamp"]).all(), (
        "a decision instant is not on the session it is filed under"
    )
    hours = panel["decision_ts"].dt.hour().value_counts().sort("decision_ts")
    print(
        f"\n{spec}: decision instants (UTC hour -> index-sessions):",
        dict(zip(hours["decision_ts"], hours["count"], strict=True)),
    )


def test_the_two_specs_do_not_overlap_in_time(panels: dict[str, pl.DataFrame]) -> None:
    """The intraday window ends where the overnight window begins, on every shared session.

    If they overlapped, the two labels would be measuring the same hours twice and a book that
    ran both specs would be doubling its exposure without saying so.
    """
    joined = (
        panels["intraday"]
        .select("symbol", "timestamp", "exec_ts", "label_ts")
        .join(
            panels["overnight"].select(
                "symbol",
                "timestamp",
                pl.col("exec_ts").alias("on_exec_ts"),
                pl.col("label_ts").alias("on_label_ts"),
            ),
            on=["symbol", "timestamp"],
            how="inner",
        )
        .drop_nulls()
    )
    assert joined.height > 0
    assert (joined["on_exec_ts"] >= joined["label_ts"]).all(), (
        "the overnight position is entered before the intraday one is closed"
    )
    print(f"\n{joined.height:,} index-sessions carry both specs and none of them overlap")


@pytest.mark.parametrize("spec", SPECS)
def test_panel_prices_match_the_sealed_labels(
    panels: dict[str, pl.DataFrame], setup: dict, spec: str
) -> None:
    """label_close / exec_open - 1 on the shared panel is exactly what 02_labels sealed."""
    name = f"fwd_ret_{spec}"
    label_path = get_case_study_dir(CASE_STUDY_ID) / "labels" / f"{name}.parquet"
    if not label_path.exists():
        pytest.skip(f"labels not built yet: {label_path}")
    labels = pl.read_parquet(label_path)
    implied = (
        panels[spec]
        .drop_nulls(["exec_open", "label_close"])
        .with_columns((pl.col("label_close") / pl.col("exec_open") - 1).alias("_implied"))
    )
    joined = implied.join(labels, on=["symbol", "timestamp"], how="inner")
    assert joined.height == implied.height, "a development index-session has no sealed label"
    gap = (joined["_implied"] - joined[name]).abs().max()
    print(f"\n{spec}: {joined.height:,} index-sessions compared, max |diff| {gap:.3e}")
    assert gap == 0.0


# ---------------------------------------------------------------------------
# The features
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", SPECS)
def test_warmup_matches_the_register(
    batches: dict[str, pl.DataFrame], setup: dict, spec: str
) -> None:
    warmup_audit(
        batches[spec], warmup_expectations(setup["features"]["windows"], spec), entity="symbol"
    )


@pytest.mark.parametrize("spec", SPECS)
def test_withholding_later_dates_changes_nothing(
    panels: dict[str, pl.DataFrame],
    batches: dict[str, pl.DataFrame],
    setup: dict,
    daily: pl.DataFrame,
    spec: str,
) -> None:
    """A transform fitted across the sample would move when the last years are removed.

    BOTH price files are truncated, not only the hourly one. A daily statistic fitted across its
    own sample would otherwise survive this test untouched, because the frame it is fitted on was
    never cut - and the long-window families are exactly the ones a whole-sample transform would
    hide in.
    """
    cut = date(2025, 1, 1)
    withheld = _build(
        panels[spec].filter(pl.col("timestamp") < cut),
        setup,
        spec,
        daily.filter(pl.col("timestamp") < cut),
    )
    assert_values_agree(
        batches[spec].filter(pl.col("timestamp") < cut),
        withheld,
        columns=feature_columns(batches[spec]),
        keys=["timestamp", "symbol"],
    )


def _sampled_sessions(batch: pl.DataFrame, setup: dict) -> list[date]:
    carrier = setup["features"]["null_policy_carrier"]
    dense = batch.filter(pl.col(carrier).is_not_null())["timestamp"].unique().sort().to_list()
    picks = np.linspace(0, len(dense) - 1, N_INSTANTS + 2)[1:-1].round().astype(int)
    return [dense[i] for i in picks]


@pytest.mark.parametrize("spec", SPECS)
def test_features_recomputed_at_decision_time(
    bars: pl.DataFrame,
    daily: pl.DataFrame,
    panels: dict[str, pl.DataFrame],
    batches: dict[str, pl.DataFrame],
    setup: dict,
    spec: str,
) -> None:
    """Rebuilding from the bars that had closed at the decision instant reproduces the batch row.

    The truncation is on the raw one-hour bars, so this tests the session aggregation as well as
    the features: a window that absorbed a bar closing after its decision, or an overnight /
    intraday column read one session too early, fails here and nowhere else.
    """
    batch = batches[spec]
    cols = feature_columns(batch)
    instants = panels[spec].select("timestamp", "decision_ts").unique().sort("timestamp")
    rows = []
    for session in _sampled_sessions(batch, setup):
        instant = instants.filter(pl.col("timestamp") == session)["decision_ts"][0]
        as_of = features_as_of(bars, instant, setup, spec=spec, daily_bars=daily)
        assert as_of["timestamp"].max() == session, (
            "the truncated panel does not end on the decided session"
        )
        live = as_of.filter(pl.col("timestamp") == session).sort("symbol")
        ref = batch.filter(pl.col("timestamp") == session).sort("symbol")
        assert live.height == ref.height == len(setup["universe"]["symbols"])
        census = assert_values_agree(ref, live, columns=cols, keys=["timestamp", "symbol"])
        rows.append(
            (
                str(session),
                str(instant),
                bars_closed_by(bars, instant).height,
                float(census["max abs difference"].max()),
            )
        )
    print(f"\n{spec}: recomputed at the decision instant (session, instant, bars seen, max |diff|):")
    for row in rows:
        print("  ", *row)


@pytest.mark.parametrize("spec", SPECS)
def test_no_feature_column_is_a_traded_price(
    batches: dict[str, pl.DataFrame], setup: dict, spec: str
) -> None:
    """The fill price and the label endpoint never reach the feature matrix.

    ``exec_open`` and ``label_close`` are dated after the decision by construction, so a build
    that let either through would be handing the model its own answer. ``EXCLUDED`` is what
    keeps them out and this is the assertion that it does.
    """
    cols = set(feature_columns(batches[spec]))
    forbidden = {"exec_open", "label_close", "exec_ts", "label_ts", "decision_ts"}
    assert not (cols & forbidden), f"{sorted(cols & forbidden)} reached the feature matrix"
    assert cols, "the feature matrix is empty"
    print(f"\n{spec}: {len(cols)} feature columns, none of them a traded price")
