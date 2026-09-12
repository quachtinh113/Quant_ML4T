"""The three phase-5 declarations of `exness_usidx_sess`, tested against `setup.yaml` itself.

`case_studies/exness_usidx_sess/_sweep.py` is the single source of the declared sweep: the signal
dicts `13_backtest` sends the backtest entry point and `_report_phase5` scores, the per-session
spread charge, and the minimum-legs floor. Each of them was got wrong once, and each of the three
bugs was invisible until a number came out strange:

* the signal dicts were built twice, in two files, and agreed only by inspection;
* the spread was one number over a window that straddles a fourfold change in the tape;
* the floor did not exist, so a Sharpe ratio resting on 31 legs was reported with n = 302.

So the checks below are on the declaration, not on a result. Nothing here reads a return.

    uv run --with pytest==9.0.3 python -m pytest bots/exness_usidx_sess/tests/test_sweep_declarations.py -q
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
import yaml

from utils.paths import REPO_ROOT

SETUP_PATH = REPO_ROOT / "case_studies" / "exness_usidx_sess" / "config" / "setup.yaml"


@pytest.fixture(scope="module")
def setup() -> dict:
    return yaml.safe_load(SETUP_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------- the signal dicts
def test_declared_settings_are_the_five_the_grid_declares(setup) -> None:
    from case_studies.exness_usidx_sess._sweep import declared_settings

    settings = declared_settings(setup)
    grid = setup["backtest"]["sweep"]["threshold_grid"]
    assert len(settings) == len(grid["fixed_threshold"]) + len(
        grid["per_symbol_rolling_percentile"]["lookback_days"]
    )
    # 5 settings x 178 prediction sets x 2 workspaces = 1,780, the K of this generation.
    assert len(settings) == 5


def test_fixed_threshold_settings_carry_the_signed_convention(setup) -> None:
    """The whole of the phase-5 correction is this key reaching the library."""
    from case_studies.exness_usidx_sess._sweep import declared_settings

    fixed = declared_settings(setup, family="fixed_threshold")
    assert fixed, "the fixed_threshold family may not be empty"
    for setting in fixed:
        assert setting["long_short"] is True
        assert setting["threshold_convention"] == "signed"


def test_a_signed_book_is_flat_inside_its_own_dead_band(setup) -> None:
    """End to end through the dispatcher, on the widest declared threshold.

    Under the mirrored convention the same frame has a position on every row, which is the book
    generation 1 tested. This is the regression test for that.
    """
    from case_studies.exness_usidx_sess._sweep import declared_settings
    from case_studies.utils.signals import build_target_weights_from_config

    widest = max(
        declared_settings(setup, family="fixed_threshold"), key=lambda s: s["threshold"]
    )
    assert widest["threshold"] > 0.0
    inside = widest["threshold"] / 2.0
    frame = pl.DataFrame(
        {
            "timestamp": [date(2025, 1, 2), date(2025, 1, 2)],
            "symbol": ["US500", "USTEC"],
            "y_score": [inside, -inside],
        }
    )
    assert build_target_weights_from_config(frame, widest).height == 0
    mirrored = dict(widest)
    mirrored.pop("threshold_convention")
    assert build_target_weights_from_config(frame, mirrored).height == 2


def test_percentile_settings_are_long_only_and_one_bar_a_day(setup) -> None:
    from case_studies.exness_usidx_sess._sweep import declared_settings

    for setting in declared_settings(setup, family="per_symbol_rolling_percentile"):
        assert setting["bars_per_day"] == 1
        assert setting["long_short"] is False
        assert setting["direction"] == "long_only"


def test_an_unknown_family_is_refused(setup) -> None:
    from case_studies.exness_usidx_sess._sweep import declared_settings

    with pytest.raises(ValueError, match="unknown signal family"):
        declared_settings(setup, family="equal_weight_top_k")


# --------------------------------------------------------------- the cost regime
def test_the_charge_rule_uses_only_numbers_declared_elsewhere_in_setup(setup) -> None:
    """The rule may fix a boundary; it may not introduce a rate."""
    costs = setup["costs"]
    rule = costs["spread_bps_charge_rule"]
    assert rule["pre_break_bps"] == "from_spread_bps_indices_p90"
    assert rule["post_break_bps"] == "from_recent_regime_p90"
    assert rule["unknown_session"] == "refuse"
    # The boundary is the one the measurement block already declared.
    assert rule["post_break_from"] == costs["spread_bps_by_year"]["recent_regime_from"]
    # And it is after the measured break month, not inside it.
    break_month = str(costs["spread_bps_by_year"]["regime_break"])
    assert date.fromisoformat(str(rule["post_break_from"])) > date.fromisoformat(
        f"{break_month}-01"
    )


def test_each_session_is_charged_its_own_regime(setup) -> None:
    from case_studies.exness_usidx_sess._sweep import with_spread_bps

    conservative = float(setup["costs"]["spread_bps"]["indices"][-1])
    recent = setup["costs"]["spread_bps_by_year"]["recent_regime_p90"]
    cutoff = date.fromisoformat(str(setup["costs"]["spread_bps_charge_rule"]["post_break_from"]))
    frame = pl.DataFrame(
        {
            "timestamp": [
                date(2023, 6, 1),          # pre-break
                date(2024, 10, 15),        # inside the break month
                cutoff,                    # the first post-break session
                date(2026, 2, 26),         # the last validation session
                date(2026, 2, 26),
            ],
            "symbol": ["US500", "US500", "US500", "US500", "USTEC"],
        }
    )
    charged = with_spread_bps(setup, frame)["spread_bps"].to_list()
    assert charged[0] == pytest.approx(conservative)
    assert charged[1] == pytest.approx(conservative), "the break month keeps the conservative rate"
    assert charged[2] == pytest.approx(float(recent["US500"]))
    assert charged[3] == pytest.approx(float(recent["US500"]))
    assert charged[4] == pytest.approx(float(recent["USTEC"]))
    # The correction is a real one, not a rounding: the validation window is charged between a
    # quarter and a fifth of what the single declared number charged it (4.05 / 1.05 = 3.9 on
    # US500, 4.05 / 0.87 = 4.7 on USTEC), which is the 4 to 4.9 times the mentor measured.
    assert 3.8 < conservative / charged[3] < 4.0
    assert 4.5 < conservative / charged[4] < 4.9


def test_a_symbol_with_no_declared_post_break_rate_stops_the_run(setup) -> None:
    from case_studies.exness_usidx_sess._sweep import with_spread_bps

    frame = pl.DataFrame({"timestamp": [date(2026, 1, 5)], "symbol": ["US30"]})
    with pytest.raises(RuntimeError, match="no declared post-break spread"):
        with_spread_bps(setup, frame)


# --------------------------------------------------------------- the legs floor
def test_the_minimum_traded_sessions_floor_is_declared(setup) -> None:
    from case_studies.exness_usidx_sess._sweep import min_traded_sessions

    floor = min_traded_sessions(setup)
    assert floor == 60, (
        "the floor is the same 60 the phase-5 report already required of the number of "
        "observations in a window; changing it is a declaration, not a tuning knob"
    )
