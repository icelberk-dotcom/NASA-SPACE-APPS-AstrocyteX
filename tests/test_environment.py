"""Tests for the space-environment extension (NASA RIDGE hazards) and the evaluation metrics."""
import numpy as np
import pytest
import torch

from backend.gemini_agent import LocalReasoner
from backend.schemas import ReadingIn
from models.astroguard import AstroGuardNet, ModelConfig
from models.fusion import MODALITIES
from models.metrics import band_metrics, bands, bootstrap_ci, confusion, per_sample_confusions
from utils.environment import (ENV_EVENTS, EVENT_NAMES, MISSION_PROFILES, PROFILE_NAMES, apply_event,
                               derived_environment, environment_preset)
from utils.features import HIST_LEN, N_DAILY, N_PARAMS, N_SEQ, SEQ_LEN
from utils.reference_ranges import (CATEGORIES, DEFAULT_VALUES, ENV_CATEGORIES, ENV_IDX, ENV_KEYS, HEALTH_KEYS,
                                    PARAM_BY_KEY, PARAM_KEYS, PARAMETERS, environment_hazard_index, health_overall_risk,
                                    risk_level, risk_percent)
from utils.subjective import SUBJ_DIM
from utils.synthetic_data import IDX, coupled_param_risks, generate_dataset


# ------------------------------------------------------------------------------ registry
def test_environment_registry_covers_the_five_ridge_hazards():
    assert len(ENV_KEYS) == 12 and len(HEALTH_KEYS) == 31 and len(PARAMETERS) == 43
    assert set(ENV_CATEGORIES) == {"radiation", "isolation", "distance", "gravity", "habitat"}
    assert PARAM_KEYS[-12:] == ENV_KEYS  # contiguous block at the end (fusion splits by position)
    for cat in ENV_CATEGORIES:
        assert CATEGORIES[cat]["group"] == "environment" and any(PARAM_BY_KEY[k].category == cat for k in ENV_KEYS)


@pytest.mark.parametrize("key,value,level", [
    ("gravity_g", 0.0, "red"), ("gravity_g", 0.38, "yellow"), ("gravity_g", 1.0, "gray"),
    ("earth_distance_mkm", 0.0004, "gray"), ("earth_distance_mkm", 0.384, "green"), ("earth_distance_mkm", 225, "red"),
    ("radiation_dose_msv_d", 0.6, "green"), ("radiation_dose_msv_d", 1.8, "yellow"), ("radiation_dose_msv_d", 50, "black"),
    ("cabin_co2_mmhg", 2.5, "green"), ("cabin_co2_mmhg", 7.0, "red"), ("cabin_o2_kpa", 15.0, "red"),
    ("g_transition_days", 2, "red"), ("g_transition_days", 200, "gray"), ("isolation_days", 365, "red"),
])
def test_environment_risk_mapping(key, value, level):
    assert risk_level(risk_percent(key, value)) == level


def test_danger_limits_are_inside_the_network_input_range():
    """The z-scale rule keeps every danger limit within |z| <= 5 so clipping never hides it."""
    for p in PARAMETERS:
        for d in (p.danger[0], p.danger[1]):
            if d is not None:
                assert abs(d - p.ideal) / p.scale <= 5.0 + 1e-9, p.key


# ----------------------------------------------------------------------------- profiles
def test_mission_profiles_and_events():
    iss, mars = environment_preset("iss_leo", 100), environment_preset("mars_transit", 100)
    assert iss["gravity_g"] == 0.0 and iss["earth_distance_mkm"] < 0.001 and iss["isolation_days"] == 114
    assert mars["earth_distance_mkm"] > 50 and mars["radiation_dose_msv_d"] > iss["radiation_dose_msv_d"]
    assert derived_environment(mars)["comm_delay_s"] > 600 and derived_environment(iss)["comm_delay_s"] < 0.01
    spe = apply_event(iss, "solar_particle_event")
    assert risk_level(risk_percent("radiation_dose_msv_d", spe["radiation_dose_msv_d"])) == "black"
    landed = apply_event(iss, "post_landing")
    assert landed["g_transition_days"] <= 4
    assert set(EVENT_NAMES) == set(ENV_EVENTS) and set(PROFILE_NAMES) == set(MISSION_PROFILES)


# ------------------------------------------------------------------------ interaction prior
def _rule_vector(values):
    return np.array([risk_percent(k, values[k]) for k in PARAM_KEYS])


def _z(values):
    return np.array([(values[k] - PARAM_BY_KEY[k].ideal) / PARAM_BY_KEY[k].scale for k in PARAM_KEYS])


def test_environment_coupling_raises_physiological_risk():
    subj = np.zeros(SUBJ_DIM, dtype=np.float32)
    base = dict(DEFAULT_VALUES)
    co2 = {**base, "cabin_co2_mmhg": 7.0}
    r0 = coupled_param_risks(_rule_vector(base), subj, _z(base))
    r1 = coupled_param_risks(_rule_vector(co2), subj, _z(co2))
    assert r1[IDX["resp_rate"]] > r0[IDX["resp_rate"]] + 5
    assert r1[IDX["cabin_co2_mmhg"]] == pytest.approx(risk_percent("cabin_co2_mmhg", 7.0))  # env risk stays its own
    # side-specific: a HIGH cabin O2 is a fire hazard, it must not raise the SpO2 risk
    hi_o2 = {**base, "cabin_o2_kpa": 27.0}
    lo_o2 = {**base, "cabin_o2_kpa": 15.0}
    assert coupled_param_risks(_rule_vector(hi_o2), subj, _z(hi_o2))[IDX["spo2"]] == pytest.approx(r0[IDX["spo2"]])
    assert coupled_param_risks(_rule_vector(lo_o2), subj, _z(lo_o2))[IDX["spo2"]] > r0[IDX["spo2"]] + 10


def test_distance_amplifies_only_already_abnormal_values():
    subj = np.zeros(SUBJ_DIM, dtype=np.float32)
    sick = {**DEFAULT_VALUES, "heart_rate": 125.0}
    far = {**sick, "earth_distance_mkm": 300.0}
    r_near = coupled_param_risks(_rule_vector(sick), subj, _z(sick))
    r_far = coupled_param_risks(_rule_vector(far), subj, _z(far))
    assert r_far[IDX["heart_rate"]] > r_near[IDX["heart_rate"]] + 3
    assert r_far[IDX["glucose_mgdl"]] == pytest.approx(r_near[IDX["glucose_mgdl"]])  # normal values untouched


def test_health_overall_excludes_environment():
    risks = {k: 8.0 for k in HEALTH_KEYS}
    risks.update({k: 95.0 for k in ENV_KEYS})
    assert health_overall_risk(risks) < 15 and environment_hazard_index(risks) > 90


def test_simulator_produces_environment_aware_samples():
    d = generate_dataset(40, seed=11)
    assert d["daily"].shape == (40, N_DAILY) and d["history"].shape == (40, HIST_LEN, N_PARAMS)
    grav = d["controls"][:, 0]
    assert set(np.round(grav, 3)) & {0.0} and len(set(np.round(grav, 3))) >= 2  # several gravity fields
    assert d["profile"].shape == (40, 1) and d["event"].max() <= len(EVENT_NAMES)


# --------------------------------------------------------------------------------- model
def test_model_has_environment_modality_and_identity_film_at_init():
    model = AstroGuardNet(ModelConfig(d_model=32, head_hidden=48, d_state=16)).eval()
    b = {"seq": torch.randn(2, SEQ_LEN, N_SEQ), "daily": torch.randn(2, N_DAILY), "history": torch.randn(2, HIST_LEN, N_PARAMS),
         "subj": torch.rand(2, SUBJ_DIM), "mission": torch.rand(2, 1), "controls": torch.rand(2, 4)}
    bb = model.backbone(b)
    assert MODALITIES[-1] == "environment" and bb["fusion"]["modality_attn"].shape == (2, 5)
    assert torch.allclose(bb["env"]["gamma"], torch.zeros_like(bb["env"]["gamma"]))
    pred = model.predict(b, n_mc=4)
    assert pred["hazard_weights"].shape == (2, len(ENV_KEYS)) and np.allclose(pred["hazard_weights"].sum(1), 1, atol=1e-5)


# ------------------------------------------------------------------------------- service
def test_service_reports_environment_and_uplift(service):
    vals = {**DEFAULT_VALUES, **environment_preset("mars_transit", 120)}
    a = service.assess("env1", ReadingIn(values=vals, mission_day=120))
    env = a.environment
    assert set(env["themes"]) == set(ENV_CATEGORIES) and env["derived"]["comm_delay_s"] > 100
    assert "uplift" in env and "uplift_pts" in env["uplift"] and len(env["hazard_attention"]) == 12
    assert a.physics["gravity"] == 0.0  # gravity_g drives the physiology ODE
    gravity_tasks = [t for t in a.new_tasks if t.parameter_key == "gravity_g"]
    assert gravity_tasks and all(t.severity_level == "red" for t in gravity_tasks)
    assert a.overall["n_at_risk"] == 0  # healthy body: environment hazards are not counted as health findings
    assert a.advisor["escalate"] is False  # red environment hazards alone do not page the flight surgeon


def test_environment_black_hazard_triggers_hazard_procedure_not_flight_surgeon(service):
    vals = {**DEFAULT_VALUES, **apply_event(environment_preset("iss_leo", 60), "solar_particle_event")}
    a = service.assess("spe1", ReadingIn(values=vals, mission_day=60))
    rad = [t for t in a.new_tasks if t.parameter_key == "radiation_dose_msv_d"]
    titles = " ".join(t.title.lower() for t in rad)
    assert "shelter" in titles and "mission control" in titles and all(t.severity_level == "black" for t in rad)


def test_legacy_gravity_field_still_works(service):
    a = service.assess("g1", ReadingIn(values={"heart_rate": 72.0}, gravity=0.38))
    g = next(p for p in a.parameters if p.key == "gravity_g")
    assert g.value == 0.38 and g.level == "yellow" and a.physics["gravity"] == 0.38


def test_local_reasoner_environment_only_does_not_escalate():
    ctx = {"overall_risk_pct": 10, "at_risk": [
        {"key": "gravity_g", "level": "red", "direction": "low", "value": 0.0, "risk_pct": 82, "kind": "environment"},
        {"key": "earth_distance_mkm", "level": "red", "direction": "high", "value": 225, "risk_pct": 80, "kind": "environment"},
        {"key": "radiation_cumulative_msv", "level": "red", "direction": "high", "value": 420, "risk_pct": 76, "kind": "environment"}],
        "environment": {"hazard_index_pct": 85, "themes": {"gravity": {"risk_pct": 82, "level": "red"}},
                        "one_way_comm_delay_s": 750, "environment_uplift_of_health_risk_pts": 6.0}}
    out = LocalReasoner.reason(ctx, "en")
    assert out.escalate_to_flight_surgeon is False and out.tasks
    assert "communication delay" in out.summary and "+6" in out.summary
    assert "iletişim gecikmesi" in LocalReasoner.reason(ctx, "tr").summary


# ------------------------------------------------------------------------------- metrics
def test_band_function_matches_risk_level():
    r = np.array([0, 14.99, 15, 40, 40.01, 70, 70.01, 90, 90.01, 100])
    assert [["gray", "green", "yellow", "red", "black"][b] for b in bands(r)] == [risk_level(v) for v in r]


def test_numpy_metrics_match_scikit_learn():
    from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef

    rng = np.random.default_rng(0)
    t = rng.choice(5, 3000, p=[0.3, 0.45, 0.13, 0.09, 0.03])
    p = np.where(rng.random(3000) < 0.8, t, rng.integers(0, 5, 3000))
    m = band_metrics(t, p)
    assert m["mcc"] == pytest.approx(matthews_corrcoef(t, p), abs=1e-9)
    assert m["macro_f1"] == pytest.approx(f1_score(t, p, average="macro"), abs=1e-9)
    assert m["qwk"] == pytest.approx(cohen_kappa_score(t, p, weights="quadratic"), abs=1e-9)
    assert confusion(t, p).sum() == 3000


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    t = rng.choice(5, (200, 43))
    p = np.where(rng.random((200, 43)) < 0.85, t, rng.integers(0, 5, (200, 43)))
    psc = per_sample_confusions(t, p)
    assert psc.sum() == t.size
    ci = bootstrap_ci(psc, ("mcc",), n_boot=200)
    point = band_metrics(t, p)["mcc"]
    assert ci["mcc"]["lo"] <= point <= ci["mcc"]["hi"]
