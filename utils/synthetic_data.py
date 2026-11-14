"""
Synthetic astronaut data generator (physiology-informed) used to train and test AstroGuard-X.

There is no public, labelled dataset of continuous astronaut physiology that matches the
parameter set of this project, so the networks are trained on *simulated* crews:

  1. a latent health scenario (nominal, dehydration, sleep loss, hypoxia/CO2, cardiac
     stress, bone & muscle loss, anaemia, infection, hypertension, thermal stress, mixed);
  2. 21 simulated days of all 31 parameters (14 days of history + 7 days to forecast) in
     which the scenario ramps up gradually, on top of person-specific offsets and AR(1)
     day-to-day noise;
  3. a high-rate sensor window for the most recent day (circadian rhythm + noise);
  4. a subjective log sentence matching the scenario;
  5. a patient-specific space-physiology trajectory (bone, muscle, plasma volume,
     fluid shift) with hidden rate multipliers that leak into observable biomarkers
     (CTX, urine calcium, creatinine, haematocrit) - so the Neural ODE has a signal to learn.

Training targets are the transparent reference-range risks *plus* a documented
coupling prior (``COUPLING`` / ``SYMPTOM_COUPLING``) that encodes physiological
interactions, e.g. poor sleep amplifying stress markers.  This is what lets the
network add value on top of the plain range check.

THIS IS SIMULATED DATA.  Model outputs are for demonstration and research only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from models.physics import (F_MAX, K_B, K_M, PV_LOSS, TAU_F, TAU_PV)
from utils.features import HIST_LEN, MAX_MISSION_DAYS, MC_FORECAST_DAYS, N_PARAMS, SEQ_LEN
from utils.reference_ranges import (DAILY_KEYS, PARAM_BY_KEY, PARAM_KEYS, PARAMETERS, SEQ_KEYS,
                                    aggregate_risk, risk_percent_array)
from utils.subjective import SUBJ_DIM, SYMPTOM_KEYS, SubjectiveEncoder

TOTAL_DAYS = HIST_LEN + MC_FORECAST_DAYS  # 21
NOW = HIST_LEN - 1  # index of "today" inside the 21-day simulation
IDX = {k: i for i, k in enumerate(PARAM_KEYS)}

# Scenario -> {parameter: z-shift at full strength}, in units of half-normal-widths.
SCENARIOS: Dict[str, Dict[str, float]] = {
    "nominal": {},
    "dehydration": {"fluid_intake_ml": -2.6, "urine_sg": 2.3, "sodium_mmol": 1.9, "creatinine_mgdl": 1.5,
                    "hematocrit_pct": 1.1, "heart_rate": 1.6, "systolic_bp": -1.3, "saliva_cortisol": 0.8},
    "sleep_loss": {"sleep_hours": -3.2, "heart_rate": 0.9, "gsr": 1.6, "saliva_cortisol": 2.1,
                   "systolic_bp": 0.9, "body_temp": -0.5},
    "hypoxia_co2": {"spo2": -2.7, "resp_rate": 2.6, "heart_rate": 1.6, "gsr": 1.0},
    "cardiac_stress": {"heart_rate": 2.8, "systolic_bp": 1.8, "diastolic_bp": 1.5, "ecg_abnormal_pct": 3.0,
                       "ecg_qtc_ms": 2.0, "gsr": 1.2},
    "bone_muscle_loss": {"exercise_min": -2.3, "urine_calcium": 2.6, "ctx_ngml": 2.6, "vitamin_d_ngml": -1.9,
                         "calcium_mgdl": 0.5, "protein_g": -1.6, "creatinine_mgdl": -0.8},
    "anemia": {"hemoglobin_gdl": -2.9, "hematocrit_pct": -2.5, "rbc_mil": -2.5, "heart_rate": 1.3,
               "spo2": -0.4, "resp_rate": 0.8},
    "infection": {"body_temp": 3.1, "wbc_k": 2.6, "crp_mgl": 3.6, "heart_rate": 1.9, "resp_rate": 1.1,
                  "sleep_hours": -1.0},
    "hypertension": {"systolic_bp": 2.6, "diastolic_bp": 2.3, "heart_rate": 0.6, "sodium_mmol": 0.9},
    "thermal_stress": {"body_temp": 2.1, "gsr": 2.6, "heart_rate": 1.5, "fluid_intake_ml": -1.1},
    "metabolic": {"glucose_mgdl": 4.2, "calories_kcal": -1.5, "potassium_mmol": 1.2, "saliva_cortisol": 1.0},
}
SCENARIO_NAMES = list(SCENARIOS.keys())

# Subjective sentences per scenario (English + Turkish) fed to the symptom encoder.
SCENARIO_TEXT: Dict[str, List[str]] = {
    "nominal": ["Feeling good and well rested today.", "Great session, no complaints.", "İyiyim, enerjik hissediyorum.", ""],
    "dehydration": ["I am very thirsty, dry mouth and a headache.", "Feeling dizzy and thirsty.", "Çok susadım, başım ağrıyor.", "dehydrated, tired"],
    "sleep_loss": ["Couldn't sleep, exhausted and irritable.", "Insomnia again, very tired and anxious.", "Uyuyamadım, çok yorgunum.", "tired"],
    "hypoxia_co2": ["Short of breath and a bad headache, feeling dizzy.", "Breathless, nauseous.", "Nefes darlığı ve baş dönmesi var."],
    "cardiac_stress": ["My heart is racing, palpitations and anxious.", "Heart pounding, feeling dizzy.", "Çarpıntı var, kalbim hızlı atıyor."],
    "bone_muscle_loss": ["Back pain and sore muscles and joints.", "Muscle cramps, severe back pain.", "Sırt ağrısı ve kas ağrısı var."],
    "anemia": ["Very tired, dizzy and short of breath on exertion.", "Exhausted and lightheaded.", "Halsizim, başım dönüyor."],
    "infection": ["Feeling sick, chills, tired and a bad headache.", "Congestion, runny nose and fatigue.", "Hastayım, burnum tıkanık ve yorgunum."],
    "hypertension": ["Headache and blurry vision, feeling nervous.", "Pounding headache.", "Başım çok ağrıyor, görme bulanık."],
    "thermal_stress": ["Hot and sweaty, thirsty, irritable.", "Feeling overheated and dizzy.", "Çok sıcak, susadım, sinirliyim."],
    "metabolic": ["Tired, thirsty, no appetite.", "Nauseous and tired.", "İştahım yok, bitkinim."],
}

# Interaction prior: source parameter -> {target parameter: percentage points added per
# risk point the source has above 40 %}.
COUPLING: Dict[str, Dict[str, float]] = {
    "sleep_hours": {"heart_rate": 0.15, "gsr": 0.20, "saliva_cortisol": 0.20, "systolic_bp": 0.10},
    "fluid_intake_ml": {"urine_sg": 0.15, "creatinine_mgdl": 0.15, "sodium_mmol": 0.10, "systolic_bp": 0.10, "heart_rate": 0.10},
    "spo2": {"heart_rate": 0.20, "resp_rate": 0.10},
    "resp_rate": {"spo2": 0.10},
    "heart_rate": {"ecg_abnormal_pct": 0.15, "systolic_bp": 0.05},
    "exercise_min": {"urine_calcium": 0.15, "ctx_ngml": 0.15},
    "hemoglobin_gdl": {"heart_rate": 0.15, "spo2": 0.05},
    "crp_mgl": {"wbc_k": 0.10, "body_temp": 0.10},
    "body_temp": {"heart_rate": 0.10, "resp_rate": 0.10},
    "saliva_cortisol": {"glucose_mgdl": 0.10, "sleep_hours": 0.10, "systolic_bp": 0.10},
    "potassium_mmol": {"ecg_qtc_ms": 0.20, "ecg_abnormal_pct": 0.15},
    "calcium_mgdl": {"ecg_qtc_ms": 0.15},
}
# Symptom intensity (0..1) -> percentage points added to the parameter's risk.
SYMPTOM_COUPLING: Dict[str, Dict[str, float]] = {
    "dizziness": {"systolic_bp": 10, "heart_rate": 5},
    "palpitations": {"heart_rate": 10, "ecg_abnormal_pct": 10},
    "breathless": {"spo2": 10, "resp_rate": 10},
    "thirst": {"fluid_intake_ml": 10, "urine_sg": 5},
    "insomnia": {"sleep_hours": 10},
    "anxiety": {"gsr": 10, "saliva_cortisol": 8},
    "fatigue": {"sleep_hours": 5, "hemoglobin_gdl": 5},
    "msk_pain": {"exercise_min": 5},
    "back_pain": {"urine_calcium": 5, "ctx_ngml": 5},
    "headache": {"systolic_bp": 4, "spo2": 4},
}


def coupled_param_risks(rule_risk: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """Add the interaction prior to plain range-based risks.

    rule_risk: (31,) risks in percent;  subj: (SUBJ_DIM,) encoded subjective log.
    Works on a single sample (use ``batch_coupled`` for arrays).
    """
    out = rule_risk.copy()
    for src, targets in COUPLING.items():
        excess = max(0.0, rule_risk[IDX[src]] - 40.0)
        for tgt, w in targets.items():
            out[IDX[tgt]] += w * excess
    for sym, targets in SYMPTOM_COUPLING.items():
        s = float(subj[SYMPTOM_KEYS.index(sym)])
        for tgt, pts in targets.items():
            out[IDX[tgt]] += pts * s
    return np.clip(out, 0.0, 99.5)


# ---------------------------------------------------------------------------- physics
def true_physics_trajectory(day0: float, controls: np.ndarray, mults: Dict[str, float], days: int = 14) -> np.ndarray:
    """Closed-form physics with patient-specific rate multipliers, sampled at day0 .. day0+days."""
    g, e, p, h = controls
    t = day0 + np.arange(days + 1, dtype=float)
    f_inf = F_MAX * mults["F"] * (1 - g) * (1 - 0.25 * e)
    pv_inf = 1 - PV_LOSS * mults["PV"] * (1 - g) * (1 - 0.5 * h)
    F = f_inf * (1 - np.exp(-t / TAU_F))
    PV = 1 + (pv_inf - 1) * (1 - np.exp(-t / TAU_PV))
    B = np.exp(-K_B * mults["B"] * (1 - g) * (1 - 0.7 * e) * t)
    M = np.exp(-K_M * mults["M"] * (1 - g) * (1 - 0.6 * e - 0.2 * p) * t)
    return np.stack([F, PV, B, M], axis=-1).astype(np.float32)


# ------------------------------------------------------------------------- generation
def _scale_vec() -> Tuple[np.ndarray, np.ndarray]:
    ideal = np.array([p.ideal for p in PARAMETERS])
    scale = np.array([p.scale for p in PARAMETERS])
    return ideal, scale


def _bounds_vec() -> Tuple[np.ndarray, np.ndarray]:
    return (np.array([p.bounds[0] for p in PARAMETERS]), np.array([p.bounds[1] for p in PARAMETERS]))


def generate_sample(rng: np.random.Generator, encoder: SubjectiveEncoder, scenario: Optional[str] = None) -> Dict[str, np.ndarray]:
    ideal, scale = _scale_vec()
    lo, hi = _bounds_vec()

    # ---- pick scenario(s) ------------------------------------------------------------
    if scenario is None:
        r = rng.random()
        if r < 0.22:
            scenario = "nominal"
        elif r < 0.40:
            scenario = "mixed"
        else:
            scenario = str(rng.choice(SCENARIO_NAMES[1:]))
    if scenario == "mixed":
        parts = list(rng.choice(SCENARIO_NAMES[1:], size=int(rng.integers(2, 4)), replace=False))
    else:
        parts = [scenario]
    strength = float(rng.uniform(0.25, 1.25))

    # ---- latent physiology multipliers (hidden) ------------------------------------
    mults = {k: float(np.exp(rng.normal(0, 0.25))) for k in ("F", "PV", "B", "M")}
    if "bone_muscle_loss" in parts:
        mults["B"] *= 1.5
        mults["M"] *= 1.4

    # ---- 21 days of z-scores -----------------------------------------------------------
    person = rng.normal(0, 0.30, N_PARAMS)  # persistent person-specific offset (healthy crews stay in range)
    z = np.zeros((TOTAL_DAYS, N_PARAMS))
    noise = rng.normal(0, 0.22, N_PARAMS)
    for d in range(TOTAL_DAYS):
        noise = 0.55 * noise + rng.normal(0, 0.18, N_PARAMS)  # AR(1) day-to-day variability
        z[d] = person + noise
    effect = np.zeros(N_PARAMS)
    for part in parts:
        for key, shift in SCENARIOS[part].items():
            effect[IDX[key]] += shift
    ramp = (np.arange(TOTAL_DAYS) / (TOTAL_DAYS - 1)) ** 1.4
    # The scenario is already visible "now" (0.65 of the way), and keeps growing afterwards.
    ramp = 0.15 + 0.85 * ramp
    z += strength * ramp[:, None] * effect[None, :]
    # Hidden physiology leaks into observable biomarkers.
    z[:, IDX["ctx_ngml"]] += 2.0 * np.log(mults["B"])
    z[:, IDX["urine_calcium"]] += 2.0 * np.log(mults["B"])
    z[:, IDX["creatinine_mgdl"]] -= 1.5 * np.log(mults["M"])
    z[:, IDX["hematocrit_pct"]] += 2.0 * np.log(mults["PV"])

    values = np.clip(ideal + z * scale, lo, hi)  # (21, 31) raw units
    values[:, IDX["spo2"]] = np.minimum(values[:, IDX["spo2"]], 100.0)

    # ---- continuous sensor window for "today" -----------------------------------------
    t = np.arange(SEQ_LEN)
    seq_raw = np.zeros((SEQ_LEN, len(SEQ_KEYS)))
    for j, key in enumerate(SEQ_KEYS):
        i = IDX[key]
        prev, cur = values[NOW - 1, i], values[NOW, i]
        sc = scale[i]
        base = prev + (cur - prev) * (t / (SEQ_LEN - 1))
        circadian = 0.15 * sc * np.sin(2 * np.pi * t / SEQ_LEN + rng.uniform(0, 6.28))
        ar, ar_series = 0.0, np.zeros(SEQ_LEN)
        for k in range(SEQ_LEN):
            ar = 0.8 * ar + rng.normal(0, 0.12 * sc)
            ar_series[k] = ar
        series = base + circadian + ar_series
        series[-1] = cur
        seq_raw[:, j] = np.clip(series, lo[i], hi[i])

    # ---- subjective log -------------------------------------------------------------------
    texts = [str(rng.choice(SCENARIO_TEXT[p])) for p in parts]
    subj = encoder.encode(" ".join(texts))

    # ---- controls & physics target ---------------------------------------------------------
    gravity = float(rng.choice([0.0, 0.166, 0.38, 1.0], p=[0.62, 0.10, 0.10, 0.18]))
    cur_vals = values[NOW]
    from utils.features import controls_from_values  # local import to avoid cycles

    controls = controls_from_values({k: cur_vals[IDX[k]] for k in PARAM_KEYS}, gravity)
    mission_day = float(rng.uniform(0, MAX_MISSION_DAYS))
    phys = true_physics_trajectory(mission_day, controls, mults)

    # ---- targets --------------------------------------------------------------------------------
    # One vectorised call per parameter covers "today" and all forecast days at once.
    risk_matrix = np.stack([risk_percent_array(k, values[NOW:, IDX[k]]) for k in PARAM_KEYS], axis=1)  # (8, 31)
    risk = coupled_param_risks(risk_matrix[0], subj)
    overall = aggregate_risk(list(risk))
    forecast = np.array([aggregate_risk(list(coupled_param_risks(risk_matrix[1 + f], subj)))
                         for f in range(MC_FORECAST_DAYS)])

    # ---- normalise inputs --------------------------------------------------------------------
    zc = lambda arr, cols: np.clip((arr - ideal[cols]) / scale[cols], -6, 6)  # noqa: E731
    all_cols = np.arange(N_PARAMS)
    seq_cols = np.array([IDX[k] for k in SEQ_KEYS])
    daily_cols = np.array([IDX[k] for k in DAILY_KEYS])
    return {
        "seq": zc(seq_raw, seq_cols).astype(np.float32),
        "daily": zc(values[NOW, daily_cols], daily_cols).astype(np.float32),
        "history": zc(values[:HIST_LEN], all_cols).astype(np.float32),
        "subj": subj.astype(np.float32),
        "mission": np.array([mission_day / MAX_MISSION_DAYS], dtype=np.float32),
        "controls": controls.astype(np.float32),
        "risk": (np.concatenate([risk, [overall]]) / 100.0).astype(np.float32),
        "forecast": (forecast / 100.0).astype(np.float32),
        "phys": phys,
        "scenario": np.array([SCENARIO_NAMES.index(parts[0]) if len(parts) == 1 else len(SCENARIO_NAMES)]),
    }


def generate_dataset(n: int, seed: int = 0, scenario: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Generate `n` independent simulated crew-member days (fixed seed => reproducible)."""
    rng = np.random.default_rng(seed)
    encoder = SubjectiveEncoder("lexicon")
    samples = [generate_sample(rng, encoder, scenario) for _ in range(n)]
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


# ------------------------------------------------------------------------- UI helpers
def scenario_preset(name: str, strength: float = 0.9) -> Dict[str, float]:
    """Deterministic parameter values for a named scenario (used by the dashboard 'demo presets')."""
    ideal, scale = _scale_vec()
    lo, hi = _bounds_vec()
    z = np.zeros(N_PARAMS)
    for key, shift in SCENARIOS[name].items():
        z[IDX[key]] += strength * shift
    vals = np.clip(ideal + z * scale, lo, hi)
    out = {k: float(vals[IDX[k]]) for k in PARAM_KEYS}
    out["spo2"] = min(out["spo2"], 100.0)
    return {k: round(v, PARAM_BY_KEY[k].decimals + 1) for k, v in out.items()}


def scenario_log(name: str, lang: str = "en") -> str:
    """Example subjective log that matches a scenario (English or Turkish sentence)."""
    texts = SCENARIO_TEXT[name]
    turkish = [t for t in texts if any(ch in t for ch in "çğıöşüÇĞİÖŞÜ")]
    return (turkish[0] if lang == "tr" and turkish else texts[0])
