import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from utils.reference_ranges import DEFAULT_VALUES
from utils.synthetic_data import scenario_preset


@pytest.fixture()
def client(service, monkeypatch):
    monkeypatch.setattr(main, "get_service", lambda: service)
    with TestClient(main.app) as c:
        yield c


def test_health_and_parameter_registry(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"]["loaded"] and h["ecg"]["mode"] == "builtin-screening"
    reg = client.get("/parameters").json()
    assert len(reg["parameters"]) == 43 and [l["key"] for l in reg["levels"]] == ["gray", "green", "yellow", "red", "black"]


def test_full_task_and_reward_flow_over_http(client):
    body = {"values": {**DEFAULT_VALUES, **scenario_preset("dehydration")}, "subjective_log": "thirsty", "mission_day": 60}
    r = client.post("/astronauts/crew1/assess", json=body)
    assert r.status_code == 200
    a = r.json()
    assert a["new_tasks"], "risky parameters must produce tasks"
    tasks = client.get("/astronauts/crew1/tasks", params={"status": "pending"}).json()
    assert len(tasks) == len(a["new_tasks"])

    g0 = client.get("/astronauts/crew1/gamification").json()
    assert g0["xp"] == 0 and g0["level"] == 1
    done = client.post(f"/astronauts/crew1/tasks/{tasks[0]['id']}/complete").json()
    assert done["points_awarded"] == tasks[0]["points"] and done["state"]["xp"] > 0
    assert any(b["earned"] for b in done["state"]["badges"])
    again = client.post(f"/astronauts/crew1/tasks/{tasks[0]['id']}/complete").json()
    assert again["already_completed"] and again["points_awarded"] == 0
    assert client.post("/astronauts/crew1/tasks/99999/complete").status_code == 404
    assert client.get("/astronauts/crew1/latest").json()["astronaut_id"] == "crew1"
    assert len(client.get("/astronauts/crew1/history").json()) == 1


def test_validation_errors(client):
    assert client.post("/astronauts/x/assess", json={"values": {"bogus": 1}}).status_code == 422
    assert client.post("/astronauts/x/assess?lang=de", json={"values": {}}).status_code == 422


def test_turkish_briefing(client):
    body = {"values": {**DEFAULT_VALUES, **scenario_preset("cardiac_stress")}}
    a = client.post("/astronauts/tr1/assess?lang=tr", json=body).json()
    assert "Genel sağlık riski" in a["advisor"]["summary"]


def test_chat_endpoint_offline(client):
    client.post("/astronauts/chat1/assess", json={"values": {**DEFAULT_VALUES, **scenario_preset("infection")}})
    r = client.post("/astronauts/chat1/chat", json={"question": "what should I do?"}).json()
    assert r["source"] == "offline" and r["answer"]


def test_ecg_endpoints_use_builtin_fallback(client):
    from utils.ecg_synth import generate_ecg

    buf = io.BytesIO()
    np.save(buf, generate_ecg(hr=72, seed=1))
    r = client.post("/ecg/analyze", files={"file": ("rec.npy", buf.getvalue())})
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "builtin-screening" and 65 <= j["heart_rate"] <= 80 and "ecg_abnormal_pct" in j["parameter_values"]
    assert client.get("/ecg/sample/AFIB").json()["rr_regularity"] == "irregular"
    assert client.post("/ecg/analyze", files={"file": ("rec.xyz", b"123")}).status_code == 422
    assert client.get("/ecg/status").json()["mode"] == "builtin-screening"


def test_web_ui_and_preset_endpoints(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "AstrocyteX" in home.text and "astrocyteX" not in home.text
    assert "overall-health" in home.text and "Quick Guide" in home.text
    assert "Slot 01" in home.text and "Cmdr." not in home.text and "Dr. Kaya" not in home.text
    assert client.get("/static/styles.css").status_code == 200
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "AstrocyteX" in js.text and "astrocyteX" not in js.text
    assert "Cmdr." not in js.text and "Dr. Kaya" not in js.text
    assert client.get("/static/diet_experiments.js").status_code == 200
    assert client.get("/static/milky_way_earth.jpg").status_code == 200

    p = client.get("/presets").json()
    assert "dehydration" in p["scenarios"] and "iss_leo" in p["profiles"] and len(p["initial_sensor_series"]["heart_rate"]) == 96

    sb = client.get("/presets/scenario/dehydration?lang=tr").json()
    assert sb["values"]["fluid_intake_ml"] < 1500 and len(sb["sensor_series"]["spo2"]) == 96

    ev = client.post("/environment/apply-event", json={"values": p["default_values"], "event": "solar_particle_event", "lang": "tr"}).json()
    assert ev["values"]["radiation_dose_msv_d"] > 20.0 and ev["log"]

    ts = client.post("/telemetry/simulate", json={"values": {"heart_rate": 85.0}}).json()
    assert len(ts["sensor_series"]["heart_rate"]) == 96 and "heart_rate" in ts["latest"]


def test_voice_and_role_endpoints_over_http(client):
    from utils.voice_synth import encode_wav_bytes, synthesize_astronaut_voice

    st = client.get("/voice/status").json()
    assert "version" in st and len(st["states"]) == 5

    smp = client.get("/voice/sample/acute_stress?sex=female&lang=tr&role=doctor").json()
    assert smp["state_key"] in {s["key"] for s in st["states"]}
    assert "mood" in smp and "biomarkers" in smp and smp.get("audio_wav_base64")

    wav, _ = synthesize_astronaut_voice("cognitive_fatigue", seed=21, duration_s=3.0)
    wav_bytes = encode_wav_bytes(wav, 16000)
    up = client.post("/voice/analyze?lang=tr&role=non_doctor", files={"file": ("crew_voice.wav", wav_bytes, "audio/wav")})
    assert up.status_code == 200
    up_json = up.json()
    assert 0.0 <= up_json["voice_risk_pct"] <= 100.0 and len(up_json["probabilities"]) == 5

    body_doc = {
        "values": {**DEFAULT_VALUES, **scenario_preset("cardiac_stress")},
        "voice": {k: v for k, v in smp.items() if not k.startswith("audio_")},
        "user_role": "doctor",
    }
    body_simple = {**body_doc, "user_role": "non_doctor"}
    r_doc = client.post("/astronauts/vcrew/assess?lang=tr", json=body_doc).json()
    r_simple = client.post("/astronauts/vcrew2/assess?lang=tr", json=body_simple).json()
    assert r_doc["overall"]["voice_risk_pct"] == pytest.approx(smp["voice_risk_pct"], abs=0.2)
    assert r_doc["advisor"]["user_role"] == "doctor" and r_simple["advisor"]["user_role"] == "non_doctor"
    assert r_doc["advisor"]["summary"] != r_simple["advisor"]["summary"]

    chat_doc = client.post("/astronauts/vcrew/chat?lang=tr", json={"question": "Ses ve kalp bulgularım ne gösteriyor?", "role": "doctor"}).json()
    chat_simple = client.post("/astronauts/vcrew/chat?lang=tr", json={"question": "Ses ve kalp bulgularım ne gösteriyor?", "role": "non_doctor"}).json()
    assert chat_doc["role"] == "doctor" and chat_simple["role"] == "non_doctor"
    assert chat_doc["answer"] != chat_simple["answer"]

    en_tasks = client.get("/astronauts/vcrew/tasks?lang=en").json()
    assert len(en_tasks) >= 1
    from backend.gemini_agent import is_turkish_text
    assert all(not is_turkish_text(t["title"]) and not is_turkish_text(t["description"]) for t in en_tasks)


def test_zero_emojis_across_application_codebase():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u23F0-\u23FA]")
    checked = 0
    offenders = []
    targets = [
        *(root / folder for folder in ("web", "ui", "backend", "models", "utils", "docs", "scripts")),
        root / "README.md",
        root / "THIRD_PARTY_NOTICES.md",
        root / "run.py",
    ]
    for item in targets:
        files = item.rglob("*") if item.is_dir() else [item]
        for p in files:
            if p.is_file() and p.suffix in (".py", ".html", ".css", ".js", ".md"):
                checked += 1
                hits = emoji_re.findall(p.read_text(encoding="utf-8", errors="ignore"))
                if hits:
                    offenders.append((str(p.relative_to(root)), hits[:5]))
    assert checked >= 30 and offenders == [], f"Found forbidden emojis: {offenders}"

