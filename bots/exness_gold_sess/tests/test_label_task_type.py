"""One source of truth for "is this label a classification target".

Written 2026-09-08 for ``bots/exness_gold_sess/RULINGS_2026-09-08.md`` ruling 1, execution
steps 1 and 2. It is the **falsifier for a whole class of error**, not a test of this bot:
it runs in a second and it fails the moment any case study declares a classification label
whose name the heuristic would not recognise.

The defect it locks out
-----------------------
``case_studies/research/labels.py:111-117`` derives ``LabelDefinition.task_type`` from the
**declaration** ``labels.classification_eval_label``. ``utils/modeling.py`` derived the same
fact a second time, from a weaker source: a name prefix plus an integer dtype. Two answers to
one question, and the weak one won exactly where it did harm - this bot's ``dir_tb_8h`` is
declared with a continuous eval label and stored as ``Float64``, so the heuristic called it a
regression target and three declared conventions broke at once (the eval-label join never ran,
so ``fwd_ret_8h`` stayed in the feature matrix as feature 44 of 44; ``eval_actual`` was never
written, so the pooled panel IC scored ``Spearman(class, class)``; and ``predict`` was called
on a ``LogisticRegression``, so ``probability_to_score: p_up_minus_p_down`` never executed).

What the three tests assert
---------------------------
1. **The declaration and the detector agree, on every declared label of every case study.**
   This is the permanent version of the pre-fix measurement the ruling required.
2. **The fix is additive only.** No label that the old prefix/dtype heuristic called
   classification is called regression now - which is what makes the change a no-op for the
   sealed run logs of the eight book case studies.
3. **This bot's own label**, on its own sealed artifact: ``dir_tb_8h`` is classification with
   exactly three classes, and both regression labels stay regression.

    ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess \
      uv run --with pytest==9.0.3 python -m pytest \
      bots/exness_gold_sess/tests/test_label_task_type.py -q -s
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml

from utils.modeling import _CLASSIFICATION_PREFIXES, detect_label_type
from utils.paths import REPO_ROOT, get_case_study_dir

#: The nine case studies of the book. Their run logs are sealed (and, in this checkout, empty
#: for eight of them), so the point of the equivalence is that the fix cannot move them.
BOOK_CASE_STUDIES = (
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
    "us_firm_characteristics",
    "sp500_equity_option_analytics",
)

#: The bot case studies that live in this branch. They are included because the error class is
#: cross-bot: a bot author who names a label outside the prefix set gets the same silent break.
BOT_CASE_STUDIES = (
    "exness_gold_sess",
    "exness_fx_d1",
    "exness_btc_8h",
    "exness_usidx_sess",
    "xau_fx_mt5",
    "xau_fx_mt5_d1",
)

CASE_STUDY_ID = "exness_gold_sess"


def _declared(case_study_id: str) -> tuple[list[str], dict[str, str]]:
    """Declared label names and the classification mapping, read from the repository tree."""
    path = REPO_ROOT / "case_studies" / case_study_id / "config" / "setup.yaml"
    setup = yaml.safe_load(path.read_text()) or {}
    labels = setup.get("labels") or {}
    names = [labels.get("primary"), *(labels.get("variants") or [])]
    mapping = labels.get("classification_eval_label") or {}
    return [str(n) for n in names if n], {str(k): str(v) for k, v in mapping.items()}


def _all_declared() -> list[tuple[str, str, str]]:
    """(case_study_id, label, declared task_type) for every declared label."""
    rows: list[tuple[str, str, str]] = []
    for case_study_id in BOOK_CASE_STUDIES + BOT_CASE_STUDIES:
        names, mapping = _declared(case_study_id)
        for name in names:
            rows.append(
                (case_study_id, name, "classification" if name in mapping else "regression")
            )
    return rows


def _dummy_series(task_type: str) -> pl.Series:
    """A Float64 series that carries no information the detector could use.

    Deliberately float: the dtype fallback fires only on integers, so a float series isolates
    the declaration branch from the heuristic. Three distinct values so a classification answer
    can report a class count.
    """
    return pl.Series("label", [-1.0, 0.0, 1.0] * 4)


def test_the_declaration_and_the_detector_agree_on_every_declared_label() -> None:
    """`detect_label_type(...)[0] == LabelDefinition.task_type`, all case studies, all labels."""
    rows = _all_declared()
    assert len(rows) >= 27, f"expected at least the 27 book labels, got {len(rows)}"

    disagreements = []
    for case_study_id, label, declared in rows:
        detected = detect_label_type(label, _dummy_series(declared), case_study_id)[0]
        if detected != declared:
            disagreements.append(f"{case_study_id}:{label} declared={declared} detected={detected}")
    assert not disagreements, "declaration and detector disagree: " + "; ".join(disagreements)


def test_the_declaration_branch_only_ever_adds_classification() -> None:
    """Nothing the old prefix/dtype heuristic called classification is regression now.

    This is the proposition that makes the change a no-op for the eight sealed run logs: the
    detector's answer can only move regression -> classification, never the other way, so a
    label that was fitted as a classification target before is fitted as one now.
    """
    regressions = []
    for case_study_id, label, _declared in _all_declared():
        series = _dummy_series("regression")
        old_says_classification = any(label.startswith(p) for p in _CLASSIFICATION_PREFIXES)
        new_says_classification = (
            detect_label_type(label, series, case_study_id)[0] == "classification"
        )
        if old_says_classification and not new_says_classification:
            regressions.append(f"{case_study_id}:{label}")
        # And without a case_study_id the function is exactly what it was.
        assert (
            detect_label_type(label, series)[0] == "classification"
        ) is old_says_classification, f"heuristic-only path moved for {case_study_id}:{label}"
    assert not regressions, "classification was REMOVED for: " + ", ".join(regressions)


def test_this_bots_sealed_labels_are_typed_from_the_declaration() -> None:
    """On the sealed parquets: `dir_tb_8h` is 3-class classification, the two returns are not."""
    labels_dir = get_case_study_dir(CASE_STUDY_ID, create=False) / "labels"
    if not (labels_dir / "dir_tb_8h.parquet").is_file():
        pytest.skip(f"no sealed label artifact under {labels_dir} (set ML4T_OUTPUT_DIR)")

    expected = {"fwd_ret_8h": "regression", "fwd_ret_24h": "regression", "dir_tb_8h": "classification"}
    for label, declared in expected.items():
        series = pl.read_parquet(labels_dir / f"{label}.parquet", columns=[label])[label]
        task_type, num_classes, class_values = detect_label_type(label, series, CASE_STUDY_ID)
        assert task_type == declared, f"{label}: declared {declared}, detected {task_type}"
        if label == "dir_tb_8h":
            assert num_classes == 3, f"dir_tb_8h has {num_classes} classes, expected 3"
            assert class_values == [-1.0, 0.0, 1.0], class_values
            # The dtype is exactly what defeated the heuristic; keep the fact in the record.
            assert series.dtype == pl.Float64, series.dtype
