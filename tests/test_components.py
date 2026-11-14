import numpy as np
import pytest

from backend.ecg_adapter import ECGAdapter
from backend.gemini_agent import GeminiAdvisor, LocalReasoner
from backend.schemas import AdvisorOutput, TaskDraft
from backend.task_library import LIBRARY, templates_for
from utils.ecg_synth import generate_ecg
from utils.reference_ranges import PARAMETERS
from utils.subjective import SUBJ_DIM, SYMPTOM_KEYS, SubjectiveEncoder
from utils.synthetic_data import SCENARIO_NAMES, generate_dataset


# ------------------------------------------------------------------------- ECG adapter
def test_builtin_ecg_regular_vs_irregular():
    reg = ECGAdapter.builtin_analyze(generate_ecg(hr=72, seed=2))
    irr = ECGAdapter.builtin_analyze(generate_ecg(hr=110, irregular=True, seed=3))
    assert reg.rr_regularity == "regular" and 66 <= reg.heart_rate <= 78 and reg.abnormal_probability_pct < 25
    assert irr.rr_regularity == "irregular" and irr.abnormal_probability_pct > 50


def test_builtin_ecg_handles_garbage_and_orientation():
    assert ECGAdapter.builtin_analyze(np.zeros((12, 100))).class_code == "UNKNOWN"
    sig = generate_ecg(hr=60, seed=4).T  # (samples, leads) is transposed automatically
    assert 55 <= ECGAdapter.builtin_analyze(sig).heart_rate <= 66


def test_teknofest_payload_is_normalised():
    payload = {"record_name": "x.npy", "model_version": "V23.0 OmniNet Apex-Titan",
               "prediction": {"class_code": "AFIB", "class_name": "Atrial Fibrillation", "confidence_pct": 91.2,
                              "medical_desc": "d", "severity": {"level": 3, "label": "High", "color": "#e74c3c"}},
               "probabilities": [{"class_code": "NORMAL", "class_name": "Normal", "percentage": 6.0},
                                 {"class_code": "AFIB", "class_name": "AF", "percentage": 91.2}],
               "vitals": {"heart_rate": 118, "qtc_interval": 452, "pr_interval": 0, "qrs_duration": 90, "rr_regularity": "Kaotik"},
               "v22_biomarkers": {"uncertainty_pct": 8.5}, "sgarbossa_matrix": {"stemi_suspected": False},
               "leads_preview": {"lead_ii": [0.1, 0.2]}}
    r = ECGAdapter._from_teknofest(payload)
    assert r.source == "teknofest-ecg" and r.abnormal_probability_pct == pytest.approx(94.0)
    assert r.parameter_values() == {"ecg_abnormal_pct": 94.0, "ecg_qtc_ms": 452.0, "heart_rate": 118.0}


def test_package_discovery(tmp_path, settings):
    ad = ECGAdapter(settings)
    assert ad.package_path() is None and ad.status()["mode"] == "builtin-screening"
    (tmp_path / "TEKNOFEST_EKG_Yapay_Zeka_Sistemi.py").write_text("# stub")
    settings.ecg_package_path = tmp_path
    assert ECGAdapter(settings).package_path() == tmp_path


# --------------------------------------------------------------------- offline reasoner
def test_every_parameter_has_tasks_for_its_risky_direction():
    for p in PARAMETERS:
        for direction, side in (("low", 0), ("high", 1)):
            if p.normal[side] is None:
                continue
            assert templates_for(p.key, direction), (p.key, direction)
    assert set(LIBRARY) <= {p.key for p in PARAMETERS}


def test_local_reasoner_escalates_on_black_and_localises():
    ctx = {"overall_risk_pct": 80, "at_risk": [{"key": "spo2", "level": "black", "direction": "low", "value": 84, "risk_pct": 96}]}
    out = LocalReasoner.reason(ctx, "en")
    assert out.escalate_to_flight_surgeon and any("flight surgeon" in t.title.lower() for t in out.tasks)
    assert "Uyku" not in out.summary and "Genel" in LocalReasoner.reason(ctx, "tr").summary
    assert LocalReasoner.reason({"overall_risk_pct": 5, "at_risk": []}, "en").tasks == []


def test_gemini_output_is_sanitised(settings):
    ctx = {"at_risk": [{"key": "heart_rate", "level": "red", "direction": "high", "value": 130}]}
    raw = AdvisorOutput(summary="x" * 5000, recommendations=["a"] * 9, tasks=[
        TaskDraft(parameter_key="heart_rate", title="Breathe", instructions="ok", duration_min=90, tag="breathing"),
        TaskDraft(parameter_key="glucose_mgdl", title="Not at risk - must be dropped", duration_min=5),
        TaskDraft(parameter_key="heart_rate", title="breathe", duration_min=5)])  # duplicate title
    clean = GeminiAdvisor._sanitize(raw, ctx)
    assert len(clean.summary) <= 1200 and len(clean.recommendations) == 4
    assert [t.parameter_key for t in clean.tasks] == ["heart_rate"] and clean.tasks[0].duration_min == 60


def test_advisor_without_key_is_offline(settings):
    adv = GeminiAdvisor(settings)
    assert not adv.online and adv.status()["configured"] is False
    _, source = adv.reason({"overall_risk_pct": 0, "at_risk": []})
    assert source == "offline"


def test_gemini_failure_trips_circuit_breaker(settings):
    settings.gemini_api_key = "fake-key"
    adv = GeminiAdvisor(settings)

    class Boom:
        class models:  # noqa: N801
            @staticmethod
            def generate_content(**_):
                raise RuntimeError("network down")

    adv._client = Boom()
    adv._types = type("T", (), {"GenerateContentConfig": staticmethod(lambda **kw: kw)})
    out, source = adv.reason({"overall_risk_pct": 50, "at_risk": [{"key": "spo2", "level": "red", "direction": "low", "value": 90, "risk_pct": 80}]})
    assert source == "offline" and out.tasks and not adv.online and "network down" in adv.last_error


# ---------------------------------------------------------------------- subjective log
def test_subjective_encoder_bilingual_with_negation():
    enc = SubjectiveEncoder("lexicon")
    v = enc.encode("Very tired, dizzy and thirsty. No headache.")
    s = dict(zip(SYMPTOM_KEYS, v))
    assert s["fatigue"] > 0 and s["dizziness"] > 0 and s["thirst"] > 0 and s["headache"] == 0
    tr = dict(zip(SYMPTOM_KEYS, enc.encode("Başım dönüyor ve çok yorgunum, baş ağrım yok")))
    assert tr["dizziness"] > 0 and tr["fatigue"] > 0 and tr["headache"] == 0
    assert enc.encode("").shape == (SUBJ_DIM,) and enc.encode(None)[-2] == 0.5


# ------------------------------------------------------------------------ synthetic data
def test_synthetic_dataset_shapes_and_determinism():
    a = generate_dataset(24, seed=5)
    b = generate_dataset(24, seed=5)
    assert a["seq"].shape == (24, 96, 7) and a["history"].shape == (24, 14, 43) and a["risk"].shape == (24, 44)
    assert a["phys"].shape == (24, 15, 4) and a["forecast"].shape == (24, 7)
    assert np.allclose(a["risk"], b["risk"]) and np.all((a["risk"] >= 0) & (a["risk"] <= 1))
    assert a["phys"][..., 2].max() <= 1.0 + 1e-6  # bone density never above baseline


def test_scenarios_raise_overall_risk_vs_nominal():
    # Same benign environment for all groups, so only the physiological scenario differs.
    env = {"profile": "earth_analog", "event": "none"}
    nominal = generate_dataset(40, seed=1, scenario="nominal", **env)["risk"][:, -1].mean()
    for sc in ("infection", "hypoxia_co2", "cardiac_stress"):
        assert generate_dataset(40, seed=1, scenario=sc, **env)["risk"][:, -1].mean() > nominal + 0.1


def test_environment_events_raise_overall_risk_vs_no_event():
    base = generate_dataset(40, seed=2, scenario="nominal", profile="iss_leo", event="none")["risk"][:, -1].mean()
    for ev in ("solar_particle_event", "co2_scrubber_failure", "post_landing"):
        assert generate_dataset(40, seed=2, scenario="nominal", profile="iss_leo", event=ev)["risk"][:, -1].mean() > base + 0.02
    assert "nominal" in SCENARIO_NAMES


def test_llm_tags_are_normalised_to_canonical_badge_tags():
    from backend.task_library import ALLOWED_TAGS, LIBRARY, normalize_tag

    assert normalize_tag("beslenme") == "nutrition" and normalize_tag("Dinlenme") == "recovery"
    assert normalize_tag("HYDRATION") == "hydration" and normalize_tag("???") == "general"
    for entry in LIBRARY.values():  # every library template already uses a canonical tag
        for templates in entry.values():
            assert all(t[3] in ALLOWED_TAGS for t in templates)
    ctx = {"at_risk": [{"key": "heart_rate", "level": "red", "direction": "high", "value": 130}]}
    raw = AdvisorOutput(summary="s", tasks=[TaskDraft(parameter_key="heart_rate", title="Nefes", duration_min=10, tag="nefes egzersizi")])
    assert GeminiAdvisor._sanitize(raw, ctx).tasks[0].tag == "breathing"
