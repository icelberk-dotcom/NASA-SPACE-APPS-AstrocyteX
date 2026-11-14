import pytest

from backend.schemas import ReadingIn
from utils.reference_ranges import DEFAULT_VALUES, PARAM_BY_KEY, risk_level, risk_percent
from utils.synthetic_data import scenario_preset


def _reading(values=None, **kw):
    return ReadingIn(values={**DEFAULT_VALUES, **(values or {})}, **kw)


def _p(a, key):
    return next(p for p in a.parameters if p.key == key)


def test_healthy_crew_has_no_tasks(service):
    a = service.assess("h1", _reading())
    assert a.overall["level"] in ("gray", "green") and a.new_tasks == []
    assert len(a.parameters) == 31 and len(a.forecast) == 7
    assert abs(sum(a.modality_attention.values()) - 1.0) < 1e-3
    assert a.physics["projections"]["+180d"]["bone_density_change_pct"] <= 0


def test_dehydration_scenario_generates_actionable_tasks(service):
    a = service.assess("d1", _reading(scenario_preset("dehydration"), subjective_log="very thirsty and dizzy"))
    assert a.overall["n_at_risk"] >= 3
    keys = {t.parameter_key for t in a.new_tasks}
    assert "fluid_intake_ml" in keys
    titles = " ".join(t.title.lower() for t in a.new_tasks)
    assert "electrolyte" in titles
    for t in a.new_tasks:  # each task is attached to a parameter that really is in a risky state
        assert _p(a, t.parameter_key).level in ("yellow", "red", "black") and t.points > 0 and t.status == "pending"
    assert "thirst" in a.subjective_symptoms


def test_tasks_are_not_duplicated_on_repeated_assessments(service):
    r = _reading(scenario_preset("dehydration"))
    first = service.assess("d2", r)
    second = service.assess("d2", r)
    assert first.new_tasks and second.new_tasks == []
    assert len(service.list_tasks("d2", "pending")) == len(first.new_tasks)


def test_worse_severity_creates_new_tasks(service):
    service.assess("w1", _reading({"heart_rate": 108}))  # yellow
    n1 = len(service.list_tasks("w1", "pending"))
    a = service.assess("w1", _reading({"heart_rate": 165}))  # black
    assert _p(a, "heart_rate").level == "black" and len(service.list_tasks("w1", "pending")) > n1
    assert any("flight surgeon" in t.title.lower() for t in a.new_tasks)
    assert a.advisor["escalate"] is True


def test_completing_tasks_updates_xp_and_badges(service):
    a = service.assess("c1", _reading(scenario_preset("sleep_loss")))
    assert a.new_tasks
    res = service.complete_task("c1", a.new_tasks[0].id)
    assert res.points_awarded == a.new_tasks[0].points and res.state.xp == res.points_awarded
    assert any(b.id == "first_step" for b in res.new_badges)
    assert service.complete_task("c1", a.new_tasks[0].id).points_awarded == 0
    assert len(service.list_tasks("c1", "done")) == 1


def test_safety_floor_never_silences_a_red_range_alert(service):
    a = service.assess("s1", _reading({"spo2": 85.0}))
    p = _p(a, "spo2")
    assert p.rule_risk_pct > 90 and p.risk_pct >= p.rule_risk_pct - 0.1 and p.level == "black"


def test_blend_uses_both_rule_and_model(service):
    p = _p(service.assess("b1", _reading({"heart_rate": 100})), "heart_rate")
    assert p.model_risk_pct is not None and p.ci_low <= p.risk_pct <= p.ci_high
    assert p.risk_pct == pytest.approx(0.65 * p.rule_risk_pct + 0.35 * p.model_risk_pct, abs=0.2)


def test_rules_only_mode_works_without_a_model(rules_only_service):
    a = rules_only_service.assess("r1", _reading({"heart_rate": 120}))
    p = _p(a, "heart_rate")
    assert p.model_risk_pct is None and p.risk_pct == p.rule_risk_pct and a.model_version == "rules-only"
    assert risk_level(risk_percent("heart_rate", 120)) == p.level


def test_missing_parameters_are_inactive(service):
    a = service.assess("m1", ReadingIn(values={"heart_rate": 72.0}))
    assert _p(a, "heart_rate").risk_pct is not None
    assert _p(a, "glucose_mgdl").risk_pct is None and _p(a, "glucose_mgdl").level == "gray"


def test_unknown_parameter_rejected_and_values_clipped(service):
    with pytest.raises(ValueError):
        service.assess("e1", ReadingIn(values={"not_a_param": 1.0}))
    a = service.assess("e1", ReadingIn(values={"heart_rate": 9999.0}))
    assert _p(a, "heart_rate").value == PARAM_BY_KEY["heart_rate"].bounds[1]


def test_history_is_persisted_and_used(service):
    for hr in (70, 75, 80):
        service.assess("p1", _reading({"heart_rate": hr}))
    assert len(service.risk_history("p1")) == 3
    assert service.latest("p1")["astronaut_id"] == "p1"


def test_ecg_values_flow_into_assessment(service):
    ecg = {"abnormal_probability_pct": 82.0, "qtc_ms": 515.0, "heart_rate": 130.0, "class_code": "AFIB", "class_name": "AFib"}
    a = service.assess("ecg1", ReadingIn(values={"heart_rate": 80.0}, ecg=ecg))
    assert _p(a, "ecg_abnormal_pct").value == 82.0 and _p(a, "ecg_abnormal_pct").level in ("red", "black")
    assert _p(a, "ecg_qtc_ms").level == "black"
    assert _p(a, "heart_rate").value == 80.0  # wearable value wins over the ECG estimate
    assert a.ecg["class_code"] == "AFIB"
