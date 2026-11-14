"""
Synthetic astronaut data generator (physiology- and environment-informed) used to train and
test AstroGuard-X.

There is no public, labelled dataset of continuous astronaut physiology that matches the
parameter set of this project, so the networks are trained on *simulated* crews:

  1. a latent health scenario (nominal, dehydration, sleep loss, hypoxia/CO2, cardiac
     stress, bone & muscle loss, anaemia, infection, hypertension, thermal stress, metabolic,
     mixed);
  2. a **mission profile** (Earth analog, ISS, Lunar Gateway, lunar surface, Mars transit,
     Mars surface) that sets the 12 space-environment hazards (radiation, isolation &
     confinement, distance from Earth, gravity fields, closed-habitat atmosphere), and an
     optional **off-nominal event** (solar particle event, CO2 scrubber failure, thermal
     failure, O2 drop, noisy habitat, prolonged isolation, post-landing transition);
  3. 21 simulated days of all 43 parameters (14 days of history + 7 days to forecast): the
     scenario ramps up gradually on top of person-specific offsets and AR(1) noise, and the
     environment *physically* shifts physiology (e.g. high CO2 raises the respiratory rate,
     a radiation storm lowers blood counts, a recent landing lowers blood pressure);
  4. a high-rate sensor window for the most recent day (circadian rhythm + noise);
  5. a subjective log matching the scenario and the event;
  6. a patient-specific space-physiology trajectory (bone, muscle, plasma volume, fluid
     shift) with hidden rate multipliers that leak into observable biomarkers.

Training targets are the transparent reference-range risks *plus* documented interaction
priors that a per-parameter range check cannot see:

  * ``COUPLING``          physiological interactions (poor sleep -> stress markers ...);
  * ``SYMPTOM_COUPLING``  what the astronaut reports in the subjective log;
  * ``ENV_COUPLING``      how each environment hazard raises physiological risk;
  * ``DIST_AMPLIFIER``    distance from Earth escalates anything already abnormal, because
                          evacuation and real-time ground support become impossible.

This is what lets the network add value on top of the plain range check - and what makes
the environment inputs matter for the risk estimate.

THIS IS SIMULATED DATA.  Model outputs are for demonstration and research only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from models.physics import (F_MAX, K_B, K_M, PV_LOSS, TAU_F, TAU_PV)
from utils.environment import (ENV_EVENTS, EVENT_NAMES, MISSION_PROFILES, PROFILE_NAMES, apply_event, clip_env,
                               profile_values)
from utils.features import HIST_LEN, MAX_MISSION_DAYS, MC_FORECAST_DAYS, N_PARAMS, SEQ_LEN
from utils.reference_ranges import (DAILY_KEYS, ENV_IDX, ENV_KEYS, HEALTH_IDX, HEALTH_KEYS, PARAM_BY_KEY,
                                    PARAM_KEYS, PARAMETERS, SEQ_KEYS, aggregate_risk, risk_percent_array)
from utils.subjective import SUBJ_DIM, SYMPTOM_KEYS, SubjectiveEncoder

TOTAL_DAYS = HIST_LEN + MC_FORECAST_DAYS  # 21
NOW = HIST_LEN - 1  # index of "today" inside the 21-day simulation
IDX = {k: i for i, k in enumerate(PARAM_KEYS)}
N_ENV = len(ENV_KEYS)

# Scenario -> {parameter: z-shift at full strength}, in units of the parameter's z-scale.
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

# Environment hazard -> physiological risk (percentage points per hazard-risk point above 40 %).
# The side ("low"/"high"/"any") says which deviation of the hazard parameter matters, e.g. only
# a LOW cabin O2 threatens oxygen saturation (a high one is a fire hazard instead).
ENV_COUPLING: Dict[Tuple[str, str], Dict[str, float]] = {
    ("radiation_dose_msv_d", "any"): {"wbc_k": 0.22, "platelets_k": 0.22, "rbc_mil": 0.08, "crp_mgl": 0.06},
    ("radiation_cumulative_msv", "any"): {"ecg_abnormal_pct": 0.08, "wbc_k": 0.08, "platelets_k": 0.06},
    ("isolation_days", "any"): {"saliva_cortisol": 0.16, "sleep_hours": 0.14, "gsr": 0.10},
    ("habitable_volume_m3", "any"): {"saliva_cortisol": 0.12, "sleep_hours": 0.10, "gsr": 0.08},
    ("gravity_g", "any"): {"urine_calcium": 0.15, "ctx_ngml": 0.15, "exercise_min": 0.10, "vitamin_d_ngml": 0.05},
    ("g_transition_days", "any"): {"systolic_bp": 0.22, "heart_rate": 0.15, "diastolic_bp": 0.10},
    ("cabin_co2_mmhg", "any"): {"resp_rate": 0.22, "heart_rate": 0.08, "sleep_hours": 0.08},
    ("cabin_o2_kpa", "low"): {"spo2": 0.30, "resp_rate": 0.10, "heart_rate": 0.10},
    ("cabin_temp_c", "high"): {"body_temp": 0.20, "gsr": 0.10, "fluid_intake_ml": 0.10},
    ("cabin_temp_c", "low"): {"body_temp": 0.15},
    ("cabin_humidity_pct", "low"): {"fluid_intake_ml": 0.08, "resp_rate": 0.05},
    ("cabin_humidity_pct", "high"): {"body_temp": 0.05, "gsr": 0.05},
    ("cabin_noise_dba", "any"): {"sleep_hours": 0.16, "saliva_cortisol": 0.10, "gsr": 0.05},
}
# Distance from Earth escalates any physiological risk already above 40 %: at the danger
# distance (~Mars) a parameter at 70 % gains ~+7 points (no evacuation, delayed ground support).
DIST_AMPLIFIER = 0.25

# Environment hazard -> *physical* shift of physiological values (z units per unit of
# hazard excess, excess = (hazard risk - 40) / 50).  This is what the sensors would see.
ENV_PHYSIO: Dict[Tuple[str, str], Dict[str, float]] = {
    ("radiation_dose_msv_d", "any"): {"wbc_k": -1.6, "platelets_k": -1.3, "rbc_mil": -0.3},
    ("isolation_days", "any"): {"saliva_cortisol": 0.8, "sleep_hours": -0.8, "gsr": 0.4},
    ("habitable_volume_m3", "any"): {"saliva_cortisol": 0.6, "sleep_hours": -0.4},
    ("gravity_g", "any"): {"urine_calcium": 0.8, "ctx_ngml": 0.8, "systolic_bp": -0.3},
    ("g_transition_days", "any"): {"systolic_bp": -1.6, "heart_rate": 1.3, "diastolic_bp": -0.6},
    ("cabin_co2_mmhg", "any"): {"resp_rate": 1.3, "heart_rate": 0.5},
    ("cabin_o2_kpa", "low"): {"spo2": -2.0, "resp_rate": 0.8, "heart_rate": 0.8},
    ("cabin_temp_c", "high"): {"body_temp": 1.0, "gsr": 1.2, "heart_rate": 0.5},
    ("cabin_temp_c", "low"): {"body_temp": -0.8},
    ("cabin_noise_dba", "any"): {"sleep_hours": -1.0, "saliva_cortisol": 0.6, "gsr": 0.4},
}

# Mission-profile mix of the simulated population and event probability.
PROFILE_P = np.array([0.12, 0.30, 0.12, 0.14, 0.18, 0.14])
EVENT_PROB = 0.35
GRAVITY_AFTER_LANDING = {"earth_analog": 1.0, "iss_leo": 1.0, "lunar_gateway": 1.0, "lunar_surface": 0.166,
                         "mars_transit": 0.38, "mars_surface": 0.38}


def _side_ok(side: str, z_value: float) -> bool:
    return side == "any" or (side == "low" and z_value < 0) or (side == "high" and z_value >= 0)


def coupled_param_risks(rule_risk: np.ndarray, subj: np.ndarray, z_now: Optional[np.ndarray] = None) -> np.ndarray:
    """Add the interaction priors to plain range-based risks.

    rule_risk: (43,) risks in percent;  subj: (SUBJ_DIM,) encoded subjective log;
    z_now: (43,) signed z-scores (needed for side-specific environment effects; if omitted
    side-specific couplings are applied for any deviation).
    Only physiological parameters are modified - the environment risks stay their own
    hazard level.
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
    for (src, side), targets in ENV_COUPLING.items():
        if z_now is not None and not _side_ok(side, float(z_now[IDX[src]])):
            continue
        excess = max(0.0, rule_risk[IDX[src]] - 40.0)
        for tgt, w in targets.items():
            out[IDX[tgt]] += w * excess
    # Distance from Earth: escalate whatever is already abnormal.
    amp = DIST_AMPLIFIER * float(np.clip((rule_risk[IDX["earth_distance_mkm"]] - 15.0) / 75.0, 0.0, 1.0))
    h = np.array(HEALTH_IDX)
    out[h] += amp * np.maximum(0.0, out[h] - 40.0)
    out[h] = np.clip(out[h], 0.0, 99.5)
    out[ENV_IDX] = rule_risk[ENV_IDX]
    return out


def health_aggregate(risks: np.ndarray) -> float:
    """Overall health risk (environment hazards excluded, see reference_ranges.health_overall_risk)."""
    return aggregate_risk(list(risks[HEALTH_IDX]))


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


def _environment_days(rng: np.random.Generator, profile: str, event: str, mission_day: float) -> Tuple[np.ndarray, float]:
    """21 days of raw environment values (TOTAL_DAYS, N_ENV) for a profile (+ optional event)."""
    base = profile_values(profile, mission_day, rng)  # values "today"
    sds = {k: sd for k, (_mu, sd) in MISSION_PROFILES[profile]["env"].items()}  # type: ignore[union-attr]
    env = np.zeros((TOTAL_DAYS, N_ENV))
    noise = {k: 0.0 for k in ENV_KEYS}
    transition = bool(MISSION_PROFILES[profile]["transition"])
    for d in range(TOTAL_DAYS):
        offset = d - NOW
        row = dict(base)
        for k in ENV_KEYS:  # small AR(1) day-to-day drift of the static hazards
            sd = sds.get(k) or 0.0
            if sd:
                noise[k] = 0.6 * noise[k] + rng.normal(0, 0.35 * sd)
                row[k] = base[k] + noise[k]
        row["isolation_days"] = max(0.0, base["isolation_days"] + offset)
        if transition:
            row["g_transition_days"] = max(0.0, base["g_transition_days"] + offset)
        env[d] = [row[k] for k in ENV_KEYS]

    strength = 0.0
    if event != "none":
        strength = float(rng.uniform(0.4, 1.2))
        onset = int(rng.integers(NOW - 9, NOW + 1))  # the event is already visible "today"
        ev_seed = int(rng.integers(0, 2**31 - 1))  # same event magnitude on every day
        landing_offset = float(rng.uniform(0.0, 1.0))
        for d in range(max(onset, 0), TOTAL_DAYS):
            s = strength * (1.0 - np.exp(-(d - onset + 1) / 1.5))  # fast onset, then plateau
            row = dict(zip(ENV_KEYS, env[d]))
            if event == "post_landing":
                row["g_transition_days"] = float(d - onset) + landing_offset
                row["gravity_g"] = GRAVITY_AFTER_LANDING[profile]
            else:
                row = apply_event(row, event, s, np.random.default_rng(ev_seed))
            env[d] = [row[k] for k in ENV_KEYS]
    # Cumulative dose follows the (possibly event-boosted) daily dose.
    i_rate, i_cum = ENV_KEYS.index("radiation_dose_msv_d"), ENV_KEYS.index("radiation_cumulative_msv")
    cum_start = max(0.0, base["radiation_cumulative_msv"] - base["radiation_dose_msv_d"] * NOW)
    env[:, i_cum] = cum_start + np.cumsum(env[:, i_rate])
    for d in range(TOTAL_DAYS):
        env[d] = list(clip_env(dict(zip(ENV_KEYS, env[d]))).values())
    return env, strength


def _env_physio_shift(env_days: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """Physical z-shifts of physiological parameters caused by the environment, per day."""
    shift = np.zeros((env_days.shape[0], N_PARAMS))
    for (src, side), targets in ENV_PHYSIO.items():
        j = ENV_KEYS.index(src)
        vals = env_days[:, j]
        risk = risk_percent_array(src, vals)
        excess = np.clip((risk - 40.0) / 50.0, 0.0, 1.2)
        if side == "low":
            excess = excess * (vals < ideal[IDX[src]])
        elif side == "high":
            excess = excess * (vals >= ideal[IDX[src]])
        for tgt, k in targets.items():
            shift[:, IDX[tgt]] += k * excess
    return shift


def generate_sample(rng: np.random.Generator, encoder: SubjectiveEncoder, scenario: Optional[str] = None,
                    profile: Optional[str] = None, event: Optional[str] = None) -> Dict[str, np.ndarray]:
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

    # ---- environment: mission profile + optional off-nominal event -------------------
    if profile is None:
        profile = str(rng.choice(PROFILE_NAMES, p=PROFILE_P))
    if event is None:
        event = str(rng.choice(EVENT_NAMES)) if rng.random() < EVENT_PROB else "none"
    lo_day, hi_day = MISSION_PROFILES[profile]["mission_days"]  # type: ignore[misc]
    mission_day = float(rng.uniform(lo_day, hi_day))
    env_days, _ev_strength = _environment_days(rng, profile, event, mission_day)

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
    ramp = 0.15 + 0.85 * ramp  # the scenario is already visible "now" and keeps growing afterwards
    z += strength * ramp[:, None] * effect[None, :]
    z += _env_physio_shift(env_days, ideal)  # the environment physically moves physiology
    # Hidden physiology leaks into observable biomarkers.
    z[:, IDX["ctx_ngml"]] += 2.0 * np.log(mults["B"])
    z[:, IDX["urine_calcium"]] += 2.0 * np.log(mults["B"])
    z[:, IDX["creatinine_mgdl"]] -= 1.5 * np.log(mults["M"])
    z[:, IDX["hematocrit_pct"]] += 2.0 * np.log(mults["PV"])

    values = np.clip(ideal + z * scale, lo, hi)  # (21, 43) raw units
    values[:, ENV_IDX] = env_days  # environment columns are measured directly
    values[:, IDX["spo2"]] = np.minimum(values[:, IDX["spo2"]], 100.0)

    # ---- continuous sensor window for "today" (vectorised over the 7 channels) ---------
    seq_cols = np.array([IDX[k] for k in SEQ_KEYS])
    t = np.arange(SEQ_LEN)[:, None]
    prev, cur, sc = values[NOW - 1, seq_cols], values[NOW, seq_cols], scale[seq_cols]
    base = prev + (cur - prev) * (t / (SEQ_LEN - 1))
    circadian = 0.15 * sc * np.sin(2 * np.pi * t / SEQ_LEN + rng.uniform(0, 6.28, len(seq_cols)))
    eps = rng.normal(0, 1.0, (SEQ_LEN, len(seq_cols))) * (0.12 * sc)
    ar = np.zeros_like(eps)
    acc = np.zeros(len(seq_cols))
    for k in range(SEQ_LEN):
        acc = 0.8 * acc + eps[k]
        ar[k] = acc
    seq_raw = base + circadian + ar
    seq_raw[-1] = cur
    seq_raw = np.clip(seq_raw, lo[seq_cols], hi[seq_cols])

    # ---- subjective log -------------------------------------------------------------------
    texts = [str(rng.choice(SCENARIO_TEXT[p])) for p in parts]
    if event != "none" and rng.random() < 0.8:
        texts.append(str(rng.choice(ENV_EVENTS[event]["text"])))  # type: ignore[arg-type]
    subj = encoder.encode(" ".join(texts))

    # ---- controls & physics target ---------------------------------------------------------
    cur_vals = values[NOW]
    from utils.features import controls_from_values  # local import to avoid cycles

    gravity = float(cur_vals[IDX["gravity_g"]])
    controls = controls_from_values({k: cur_vals[IDX[k]] for k in PARAM_KEYS}, gravity)
    phys = true_physics_trajectory(mission_day, controls, mults)

    # ---- targets --------------------------------------------------------------------------------
    risk_matrix = np.stack([risk_percent_array(k, values[NOW:, IDX[k]]) for k in PARAM_KEYS], axis=1)  # (8, 43)
    z_days = (values - ideal) / scale
    risk = coupled_param_risks(risk_matrix[0], subj, z_days[NOW])
    overall = health_aggregate(risk)
    forecast = np.array([health_aggregate(coupled_param_risks(risk_matrix[1 + f], subj, z_days[NOW + 1 + f]))
                         for f in range(MC_FORECAST_DAYS)])

    # ---- normalise inputs --------------------------------------------------------------------
    zc = lambda arr, cols: np.clip((arr - ideal[cols]) / scale[cols], -6, 6)  # noqa: E731
    all_cols = np.arange(N_PARAMS)
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
        # bookkeeping for slice analysis (not model inputs)
        "scenario": np.array([SCENARIO_NAMES.index(parts[0]) if len(parts) == 1 else len(SCENARIO_NAMES)]),
        "profile": np.array([PROFILE_NAMES.index(profile)]),
        "event": np.array([EVENT_NAMES.index(event) if event != "none" else len(EVENT_NAMES)]),
    }


META_KEYS = ("scenario", "profile", "event")


def _generate_chunk(args: Tuple[int, int, Optional[str], Optional[str], Optional[str]]) -> Dict[str, np.ndarray]:
    n, seed, scenario, profile, event = args
    rng = np.random.default_rng(seed)
    encoder = SubjectiveEncoder("lexicon")
    samples = [generate_sample(rng, encoder, scenario, profile, event) for _ in range(n)]
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


def generate_dataset(n: int, seed: int = 0, scenario: Optional[str] = None, profile: Optional[str] = None,
                     event: Optional[str] = None, workers: int = 1, chunk: int = 1000) -> Dict[str, np.ndarray]:
    """Generate `n` independent simulated crew-member days (fixed seed => reproducible).

    With ``workers > 1`` the dataset is produced in parallel chunks (each chunk has its own
    deterministic seed derived from ``seed``), which makes large training sets fast.
    """
    if workers <= 1 or n <= chunk:
        return _generate_chunk((n, seed, scenario, profile, event))
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for i, start in enumerate(range(0, n, chunk)):
        jobs.append((min(chunk, n - start), seed * 7919 + 104729 * (i + 1), scenario, profile, event))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_generate_chunk, jobs))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ------------------------------------------------------------------------- UI helpers
def scenario_preset(name: str, strength: float = 0.9) -> Dict[str, float]:
    """Deterministic *physiological* values for a named scenario (dashboard 'demo presets').

    Environment parameters are not touched - they come from the mission profile.
    """
    ideal, scale = _scale_vec()
    lo, hi = _bounds_vec()
    z = np.zeros(N_PARAMS)
    for key, shift in SCENARIOS[name].items():
        z[IDX[key]] += strength * shift
    vals = np.clip(ideal + z * scale, lo, hi)
    out = {k: float(vals[IDX[k]]) for k in HEALTH_KEYS}
    out["spo2"] = min(out["spo2"], 100.0)
    return {k: round(v, PARAM_BY_KEY[k].decimals + 1) for k, v in out.items()}


def scenario_log(name: str, lang: str = "en") -> str:
    """Example subjective log that matches a scenario (English or Turkish sentence)."""
    texts = SCENARIO_TEXT[name]
    turkish = [t for t in texts if any(ch in t for ch in "çğıöşüÇĞİÖŞÜ")]
    return (turkish[0] if lang == "tr" and turkish else texts[0])
