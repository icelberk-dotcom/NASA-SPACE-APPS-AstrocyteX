import numpy as np
import pytest

from utils.reference_ranges import (DAILY_KEYS, PARAMETERS, PARAM_BY_KEY, SEQ_KEYS, aggregate_risk, level_color,
                                    risk_level, risk_percent, risk_percent_array)


@pytest.mark.parametrize("pct,level", [(0, "gray"), (14.9, "gray"), (15, "green"), (40, "green"), (40.1, "yellow"),
                                       (70, "yellow"), (70.1, "red"), (90, "red"), (90.1, "black"), (100, "black")])
def test_level_boundaries_follow_the_brief(pct, level):
    assert risk_level(pct) == level


def test_missing_value_is_inactive_gray():
    assert risk_level(None) == "gray" and risk_percent("heart_rate", None) is None


def test_colours():
    assert level_color("black") == "#000000" and level_color("green") == "#2ecc71"


@pytest.mark.parametrize("p", PARAMETERS, ids=lambda p: p.key)
def test_ideal_is_gray_and_limits_hit_their_anchor_risks(p):
    assert risk_level(risk_percent(p.key, p.ideal)) == "gray"
    for side, idx in (("hi", 1), ("lo", 0)):
        n, w, d = p.normal[idx], p.warn[idx], p.danger[idx]
        if n is None:
            continue
        assert abs(risk_percent(p.key, n) - 40) < 1e-6  # normal edge -> 40 % (still green)
        assert abs(risk_percent(p.key, w) - 70) < 1e-6  # caution edge -> 70 % (still yellow)
        assert abs(risk_percent(p.key, d) - 90) < 1e-6  # danger edge -> 90 % (still red)
        beyond = d + (d - p.ideal) * 0.1
        assert risk_level(risk_percent(p.key, beyond)) == "black"


@pytest.mark.parametrize("p", PARAMETERS, ids=lambda p: p.key)
def test_risk_is_monotone_away_from_ideal(p):
    for idx, sign in ((1, 1), (0, -1)):
        if p.normal[idx] is None:
            continue
        xs = p.ideal + sign * np.linspace(0, abs(p.danger[idx] - p.ideal) * 1.6, 60)
        r = risk_percent_array(p.key, xs)
        assert np.all(np.diff(r) >= -1e-9) and r.max() <= 100.0


def test_scalar_and_vector_versions_agree():
    for p in PARAMETERS:
        xs = np.linspace(p.bounds[0], p.bounds[1], 25)
        vec = risk_percent_array(p.key, xs)
        assert np.allclose(vec, [risk_percent(p.key, float(x)) for x in xs])


def test_one_sided_parameters():
    assert risk_level(risk_percent("crp_mgl", 0.0)) == "gray"
    assert risk_level(risk_percent("crp_mgl", 80.0)) == "black"
    assert risk_level(risk_percent("spo2", 100.0)) == "gray" and risk_level(risk_percent("spo2", 85.0)) == "black"


def test_registry_layout():
    assert len(PARAMETERS) == 31 and len(SEQ_KEYS) == 7 and len(DAILY_KEYS) == 24
    for key in ("sleep_hours", "fluid_intake_ml", "urine_sg", "saliva_cortisol", "exercise_min", "gsr", "body_temp",
                "spo2", "resp_rate", "hemoglobin_gdl", "glucose_mgdl", "protein_g", "ecg_qtc_ms", "heart_rate", "systolic_bp"):
        assert key in PARAM_BY_KEY


def test_aggregate_risk_dominated_by_worst_value():
    assert aggregate_risk([10, 10, 10]) < aggregate_risk([95, 10, 10]) < 100
    assert aggregate_risk([]) == 0.0
