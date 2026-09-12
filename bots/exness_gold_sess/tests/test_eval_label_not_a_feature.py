"""No copy of the evaluation label may reach the feature matrix - under any name.

Written 2026-09-08 for ``bots/exness_gold_sess/RULINGS_2026-09-08.md``, addendum ruling (a)
item 1: *"a test that needs no run log: for every case study declaring
``labels.classification_eval_label``, build the feature-matrix schema and assert that
``label``, ``eval_label`` and every ``<name>_right`` variant are NOT in ``feature_names``.
It runs today because it needs no artifact."*

The defect it locks out
-----------------------
``case_studies/research/labels.py:151-158`` **requires** a classification label's own parquet
to carry the continuous eval column it was cut from. ``utils/modeling.py`` then joins the eval
label's own parquet on top of that frame. Polars does not overwrite on a join: it **suffixes**,
so the incoming column landed as ``<eval>_right``, which is not equal to ``eval_label_col`` and
therefore survived the ``feature_names`` refresh **as a feature**. Measured on
``exness_gold_sess/dir_tb_8h`` on 2026-08-08: 43 features against 42 on the two regression
labels, and feature 43 was ``fwd_ret_8h_right`` - the forward return the classification label
is a sign of, sitting in the design matrix.

This is the **third** appearance of the same class of error in this bot's file (after
``open_right`` / ``close_right`` in phase 2), which is why the check here is a **set comparison
on ``feature_names``** and not a filter on a name prefix: a prefix filter is exactly what let
the suffixed variant through the first two times.

Why it needs no run log
-----------------------
It builds a synthetic case-study tree under ``ML4T_OUTPUT_DIR`` - two columns of features, a
label parquet that carries its eval column exactly as the label contract requires, and the eval
label's own parquet - and drives the **real** shared loader over it. The case study contributes
its declaration (``config/setup.yaml``) and nothing else, so the test runs against a repository
whose eight book run logs are empty.

What it does NOT claim
----------------------
It says nothing about run logs built **before** the 2026-09-08 fix. Those are **unaudited on
this axis** - neither broken nor trustworthy - for the four sibling case studies that declare a
classification eval label: ``crypto_perps_funding``, ``nasdaq100_microstructure``,
``us_firm_characteristics`` and ``sp500_equity_option_analytics``. Re-fitting them would spend
another author's trials and is not done here; the note is the deliverable.

    cd ~/ml4t && uv run --with pytest==9.0.3 python -m pytest \
      bots/exness_gold_sess/tests/test_eval_label_not_a_feature.py -q -s
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
import yaml

from utils.paths import REPO_ROOT

#: Every case study in the tree. Which of them declare a classification eval label is
#: DISCOVERED rather than listed, so a new declaration is covered the day it is written.
CANDIDATE_CASE_STUDIES = (
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
    "us_firm_characteristics",
    "sp500_equity_option_analytics",
    "exness_gold_sess",
    "exness_fx_d1",
    "exness_btc_8h",
    "exness_usidx_sess",
    "xau_fx_mt5",
    "xau_fx_mt5_d1",
)

#: The declarations that MUST be found. If a case study drops out of the discovery below - a
#: renamed label, an unparsable `setup.yaml`, a deleted mapping - the test fails instead of
#: quietly covering less than it did yesterday.
REQUIRED_COVERAGE = {
    ("crypto_perps_funding", "fwd_dir_8h", "fwd_ret_8h"),
    ("crypto_perps_funding", "fwd_dir_8h_3c", "fwd_ret_8h"),
    ("nasdaq100_microstructure", "fwd_dir_15m", "fwd_ret_15m"),
    ("us_firm_characteristics", "fwd_class_1m", "fwd_ret_1m"),
    ("sp500_equity_option_analytics", "fwd_dir_5d", "fwd_ret_5d"),
    ("sp500_equity_option_analytics", "fwd_dir_10d", "fwd_ret_10d"),
    ("exness_gold_sess", "dir_tb_8h", "fwd_ret_8h"),
}

#: Two synthetic features, and the assertion is that the feature matrix holds EXACTLY these.
SYNTHETIC_FEATURES = ("feat_a", "feat_b")

#: Long enough for the widest declared fold geometry in the tree: `us_firm_characteristics`
#: asks for 10 splits of a `10YE` train window, so a short calendar produces an empty training
#: index and the load fails for a reason that has nothing to do with the leak. The window ENDS
#: the day before the case study's own declared `holdout_start`, so no synthetic row is ever
#: written on a holdout date - the evidence boundary holds even in a temporary directory.
SYNTHETIC_DAYS = 30 * 365
SYNTHETIC_SYMBOLS = ("AAA", "BBB")


#: EMPTY, and asserted to stay empty: a `setup.yaml` that cannot be read cannot be audited for
#: this leak, so an unreadable one narrows this test's coverage without saying so.
#:
#: Observed 2026-09-08 while writing this file: `case_studies/exness_usidx_sess/config/setup.yaml`
#: failed to parse twice (`while parsing a block mapping ... line 311 top_n_predictions ...
#: expected <block end>, but found <block mapping start> ... line 354
#: per_symbol_rolling_percentile`) and parsed cleanly before and after, on both the Windows
#: worktree and the WSL clone. That is a file being WRITTEN while this test read it - another
#: bot's stage editing its own declaration - not a broken declaration, so nothing is excused
#: here. If this assertion fails on a rerun, read it as "someone is mid-write" first and as a
#: real defect second; either way it is reported rather than skipped.
KNOWN_UNPARSABLE_SETUPS: set[str] = set()


def _scan_declarations() -> tuple[list[tuple[str, str, str]], set[str]]:
    """Return every declared (case study, classification label, eval label) and the files that
    could not be read, rather than letting an unreadable file decide the coverage silently."""
    pairs: list[tuple[str, str, str]] = []
    unparsable: set[str] = set()
    for case_study_id in CANDIDATE_CASE_STUDIES:
        path = REPO_ROOT / "case_studies" / case_study_id / "config" / "setup.yaml"
        if not path.is_file():
            continue
        try:
            setup = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            unparsable.add(case_study_id)
            continue
        mapping = ((setup.get("labels") or {}).get("classification_eval_label")) or {}
        for label, eval_label in mapping.items():
            pairs.append((case_study_id, str(label), str(eval_label)))
    return sorted(pairs), unparsable


def _declared_pairs() -> list[tuple[str, str, str]]:
    """(case_study_id, classification label, eval label) for every declaration in the tree."""
    return _scan_declarations()[0]


def test_the_set_of_unreadable_declarations_has_not_grown() -> None:
    """An unparsable `setup.yaml` silently narrows what this test covers, so it is named."""
    unparsable = _scan_declarations()[1]
    assert unparsable == KNOWN_UNPARSABLE_SETUPS, (
        f"the set of case studies whose config/setup.yaml does not parse is {sorted(unparsable)}, "
        f"not the named {sorted(KNOWN_UNPARSABLE_SETUPS)}. A file that cannot be read cannot be "
        "audited for this leak, and which files those are is a fact that has to be declared "
        "rather than discovered by a test quietly covering less."
    )


def _synthetic_case_study(
    root: Path,
    case_study_id: str,
    label: str,
    eval_label: str,
    *,
    disagree: bool = False,
) -> None:
    """Write the smallest tree the shared loader accepts, with the leak's shape built in.

    The label parquet carries its own copy of the eval column - which is what the label
    contract requires and what makes the join a duplicate rather than an addition. That is the
    condition under which the suffix appeared; a test that omitted it would pass on a broken
    loader.
    """
    config_src = REPO_ROOT / "case_studies" / case_study_id / "config"
    config_dst = root / case_study_id / "config"
    config_dst.mkdir(parents=True, exist_ok=True)
    for item in config_src.iterdir():
        if item.is_file():
            shutil.copy2(item, config_dst / item.name)
        else:
            shutil.copytree(item, config_dst / item.name, dirs_exist_ok=True)

    setup = yaml.safe_load((config_dst / "setup.yaml").read_text(encoding="utf-8"))
    evaluation = setup.get("evaluation") or {}
    end = datetime.fromisoformat(str(evaluation.get("holdout_start", "2020-01-01"))[:19])
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=SYNTHETIC_DAYS)

    rows = []
    for day in range(SYNTHETIC_DAYS):
        stamp = start + timedelta(days=day)
        for symbol in SYNTHETIC_SYMBOLS:
            rows.append(
                {
                    "timestamp": stamp,
                    "symbol": symbol,
                    SYNTHETIC_FEATURES[0]: float(day),
                    SYNTHETIC_FEATURES[1]: float(day % 7),
                }
            )
    features = pl.DataFrame(rows)
    (root / case_study_id / "features").mkdir(parents=True, exist_ok=True)
    (root / case_study_id / "labels").mkdir(parents=True, exist_ok=True)
    features.write_parquet(root / case_study_id / "features" / "financial.parquet")

    eval_values = [0.001 * (i % 11) for i in range(features.height)]
    labels = features.select("timestamp", "symbol").with_columns(
        pl.Series(label, [float((i % 3) - 1) for i in range(features.height)]),
        pl.Series(eval_label, eval_values),
    )
    labels.write_parquet(root / case_study_id / "labels" / f"{label}.parquet")

    published = [v + (1.0 if disagree else 0.0) for v in eval_values]
    features.select("timestamp", "symbol").with_columns(
        pl.Series(eval_label, published)
    ).write_parquet(root / case_study_id / "labels" / f"{eval_label}.parquet")


def _load(root: Path, monkeypatch: pytest.MonkeyPatch, case_study_id: str, label: str):
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(root))
    from utils.modeling import load_modeling_dataset

    return load_modeling_dataset(case_study_id, label)


def test_every_declaration_in_the_tree_is_covered() -> None:
    """Discovery may grow, never shrink: the known declarations must all still be found."""
    found = set(_declared_pairs())
    missing = REQUIRED_COVERAGE - found
    assert not missing, (
        f"declarations that used to exist are no longer discovered: {sorted(missing)}. "
        "A test that covers less than it did yesterday is not a test that passed."
    )


@pytest.mark.parametrize(("case_study_id", "label", "eval_label"), _declared_pairs())
def test_no_copy_of_the_eval_label_reaches_the_feature_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_study_id: str,
    label: str,
    eval_label: str,
) -> None:
    """The feature matrix is EXACTLY the features - checked as a set, not as a prefix filter."""
    _synthetic_case_study(tmp_path, case_study_id, label, eval_label)
    mds = _load(tmp_path, monkeypatch, case_study_id, label)

    assert mds.task_type == "classification", (
        f"{case_study_id}:{label} is declared with an eval label and must be typed as "
        f"classification, not {mds.task_type}"
    )
    assert mds.eval_label_col == eval_label

    features = set(mds.feature_names)
    forbidden = {
        label,
        eval_label,
        f"{label}_right",
        f"{eval_label}_right",
    }
    assert features.isdisjoint(forbidden), (
        f"{case_study_id}:{label}: the feature matrix contains its own target or a copy of its "
        f"evaluation label: {sorted(features & forbidden)}"
    )
    suffixed = sorted(c for c in features if c.endswith("_right"))
    assert not suffixed, (
        f"{case_study_id}:{label}: join-suffixed columns reached the feature matrix: {suffixed}. "
        "A suffixed duplicate is the same column wearing another name."
    )
    assert features == set(SYNTHETIC_FEATURES), (
        f"{case_study_id}:{label}: the feature matrix is {sorted(features)}, not exactly "
        f"{sorted(SYNTHETIC_FEATURES)}. The set comparison is the point of this test: anything "
        "extra came out of a join and has to be named before it may be fitted on."
    )


@pytest.mark.parametrize(("case_study_id", "label", "eval_label"), sorted(REQUIRED_COVERAGE))
def test_two_readings_of_one_eval_series_must_agree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_study_id: str,
    label: str,
    eval_label: str,
) -> None:
    """The copy inside the label parquet and the eval parquet are one published series.

    The loader resolves the duplicate by renaming the incumbent and comparing, so the pair is
    checked rather than silently coexisting. If the two artifacts disagree, no model fitted on
    them means what its spec says it means, and the load must fail.
    """
    _synthetic_case_study(tmp_path, case_study_id, label, eval_label, disagree=True)
    with pytest.raises(ValueError, match="disagrees"):
        _load(tmp_path, monkeypatch, case_study_id, label)
