"""
Offline ("edge") knowledge base of mitigation tasks.

When Gemini is unreachable - the normal situation for a crew far from Earth - the task
engine falls back to this library.  Every parameter has task templates for the *low* and
*high* direction.  A template is ``(title, instructions, minutes, tag)``.

The tags drive the gamification badges (hydration, exercise, breathing, ...).

These suggestions are general wellness / operational actions that mirror standard
spaceflight countermeasures (ARED resistance exercise, T2 treadmill, CEVIS cycle,
hydration protocols, sleep hygiene).  They are **not medical advice** - anything beyond
simple lifestyle steps is routed to the flight surgeon.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

Template = Tuple[str, str, int, str]

_HYDRATE: Template = ("Drink 500 ml electrolyte water",
                      "Sip 500 ml of electrolyte-enriched water over 15 minutes, then log your intake.", 15, "hydration")
_BREATHE: Template = ("Take a 20-minute guided breathing session",
                      "Follow paced breathing at about 6 breaths per minute (inhale 5 s, exhale 5 s) in a quiet module.", 20, "breathing")
_ARED: Template = ("Perform 15 min ARED resistance exercise",
                   "Do squats, deadlifts and heel raises on the ARED at moderate load; keep form strict and log the sets.", 15, "exercise")
_TREADMILL: Template = ("Run 20 min on the T2 treadmill with harness loading",
                        "Use a comfortable pace with the harness loading set to the planned level.", 20, "exercise")
_REMEASURE: Template = ("Re-measure and log the value in 15 minutes",
                        "Rest quietly, repeat the measurement with correct sensor placement and log the result.", 5, "monitoring")
_SURGEON: Template = ("Report to the flight surgeon",
                      "Send the latest readings and symptoms via the medical conference channel.", 5, "medical")
_SLEEP_WIND: Template = ("Wind-down routine: dim lights 60 min before sleep",
                         "Dim the module lights, avoid screens and caffeine, and keep the sleep-station temperature cool.", 30, "sleep")
_NAP: Template = ("Take a 20-minute power nap",
                  "Schedule a short nap (max 30 min) in the crew quarters, set an alarm and avoid caffeine afterwards.", 20, "sleep")
_VITD: Template = ("Take the scheduled vitamin D supplement and log it",
                   "Take the supplement exactly as prescribed in the crew protocol - no extra doses.", 2, "nutrition")
_ECG_REPEAT: Template = ("Record a repeat 12-lead ECG",
                         "Sit still for 10 s, attach all leads, record and upload the ECG to AstroGuard-X for analysis.", 10, "ecg")
_COOL: Template = ("Cool down: lower the module temperature and drink cool water",
                   "Reduce layers, increase ventilation, drink 500 ml cool water and re-check the temperature in 30 minutes.", 20, "cooling")
_MIND: Template = ("Take a 15-minute wellbeing break",
                   "Call family or a friend, listen to music or look out of the Cupola - anything that lowers tension.", 15, "wellbeing")


def _nutrition(title: str, text: str, minutes: int = 10) -> Template:
    return (title, text, minutes, "nutrition")


LIBRARY: Dict[str, Dict[str, List[Template]]] = {
    "heart_rate": {
        "high": [_BREATHE, ("Rest and re-measure your pulse", "Float calmly for 10 minutes, avoid caffeine, then re-measure your pulse.", 10, "monitoring")],
        "low": [("Do 10 minutes of easy cycling on CEVIS", "Light cycling to raise the heart rate gently; stop if you feel dizzy.", 10, "exercise"), _REMEASURE],
    },
    "spo2": {
        "low": [("Check cabin ppO2 / CO2 readings and scrubber status", "Read the environmental panel, confirm CO2 scrubber operation and report out-of-range values.", 10, "environment"),
                ("Do 10 minutes of slow deep breathing", "Sit upright, breathe slowly and deeply, then re-measure SpO2 with the sensor properly seated.", 10, "breathing")],
        "high": [],
    },
    "systolic_bp": {
        "high": [("Rest 10 minutes, then re-measure blood pressure", "Relax with feet in restraints, no talking, cuff at heart level; repeat measurement twice.", 10, "monitoring"), _BREATHE,
                 _nutrition("Choose a low-sodium meal", "Pick low-sodium items from the food system for your next meal.", 5)],
        "low": [_HYDRATE, ("Move slowly and re-measure blood pressure", "Avoid sudden movements, rest 10 minutes and re-measure.", 10, "monitoring")],
    },
    "diastolic_bp": {
        "high": [("Rest 10 minutes, then re-measure blood pressure", "Relax with feet in restraints, no talking, cuff at heart level; repeat measurement twice.", 10, "monitoring"), _BREATHE],
        "low": [_HYDRATE, _REMEASURE],
    },
    "resp_rate": {
        "high": [_BREATHE, ("Check CO2 levels in your module", "Compare the CO2 reading to limits and move to a well-ventilated module if elevated.", 10, "environment")],
        "low": [("Sit upright and take 10 deliberate deep breaths", "Stay alert, avoid sedating activities and report persistent slow breathing.", 5, "breathing"), _SURGEON],
    },
    "body_temp": {
        "high": [_COOL, _REMEASURE],
        "low": [("Warm up: add a layer and drink a warm beverage", "Put on an extra layer, drink 300 ml of a warm beverage and re-measure in 30 minutes.", 15, "cooling"), _REMEASURE],
    },
    "gsr": {
        "high": [_BREATHE, _MIND],
        "low": [("Check the GSR sensor contact", "Clean the skin, re-seat the electrodes and confirm the signal.", 5, "monitoring")],
    },
    "sleep_hours": {
        "low": [_SLEEP_WIND, _NAP, ("Avoid caffeine for the next 8 hours", "Skip coffee/tea and energy drinks until after your next sleep period.", 1, "nutrition")],
        "high": [("Do 15 minutes of light activity", "Light exercise and bright module lighting help reset the circadian rhythm.", 15, "exercise"), _SURGEON],
    },
    "fluid_intake_ml": {
        "low": [_HYDRATE, ("Set an hourly hydration reminder", "Drink 150-200 ml every hour until the daily target is reached.", 2, "hydration")],
        "high": [("Pause extra fluids and monitor sodium", "Skip additional fluids for a few hours; report swelling or headache to the flight surgeon.", 5, "hydration"), _SURGEON],
    },
    "exercise_min": {
        "low": [_ARED, _TREADMILL],
        "high": [("Schedule a recovery block with 20 min stretching", "Reduce load today; do gentle stretching and mobility work.", 20, "recovery")],
    },
    "urine_sg": {
        "high": [_HYDRATE, ("Re-test urine specific gravity later today", "Repeat the strip test with a fresh sample after rehydrating.", 5, "monitoring")],
        "low": [("Review electrolyte intake and repeat the urine test", "Diluted urine can follow excessive fluid intake; repeat the test and log your drinks.", 5, "monitoring")],
    },
    "urine_ph": {
        "high": [("Repeat urine pH with a first-morning sample", "Use a fresh first-void sample for a reliable reading.", 5, "monitoring"),
                 _nutrition("Balance your meal", "Add fruit and vegetables and limit very high-protein items today.", 10)],
        "low": [("Repeat urine pH with a first-morning sample", "Use a fresh first-void sample for a reliable reading.", 5, "monitoring"), _HYDRATE],
    },
    "urine_calcium": {
        "high": [("Do 20 min ARED heavy-load squats and deadlifts", "Weight-bearing resistance exercise slows bone resorption; follow the crew exercise prescription.", 20, "exercise"),
                 ("Drink 500 ml water to reduce kidney-stone risk", "High urinary calcium raises stone risk; keep urine dilute.", 10, "hydration")],
        "low": [("Review calcium and vitamin D intake with the flight surgeon", "Share your recent diet log.", 5, "medical")],
    },
    "saliva_cortisol": {
        "high": [_BREATHE, ("Protect your sleep window tonight", "Lower workload in the evening and keep a fixed sleep schedule.", 15, "sleep")],
        "low": [("Get bright light exposure and a protein-rich breakfast", "Use the bright-light panel for 20 minutes after waking.", 20, "wellbeing")],
    },
    "calories_kcal": {
        "low": [_nutrition("Eat a ~400 kcal high-energy snack", "Choose nuts, a bar or nut butter from the food system and log it.", 10)],
        "high": [_nutrition("Swap one heavy item for a lighter meal", "Reduce energy-dense items at the next meal and keep drinking water.", 5)],
    },
    "protein_g": {
        "low": [_nutrition("Add 25 g of protein to your next meal", "Tuna pouch, whey shake or chicken - protein protects muscle in microgravity.", 10)],
        "high": [_nutrition("Balance protein with vegetables and fluids", "Moderate protein intake and increase water to support the kidneys.", 10)],
    },
    "ecg_qtc_ms": {
        "high": [_ECG_REPEAT, ("Review medications and electrolytes with the flight surgeon", "QT-prolonging drugs and low K/Mg should be checked.", 10, "medical")],
        "low": [_ECG_REPEAT],
    },
    "ecg_abnormal_pct": {
        "high": [("Rest, then record a repeat 12-lead ECG", "Stay calm for 5 minutes, then record and upload another ECG.", 10, "ecg"), _SURGEON],
        "low": [],
    },
    "glucose_mgdl": {
        "low": [_nutrition("Eat 15 g of fast carbohydrates and re-check in 15 min", "Juice or glucose tablets, then repeat the measurement.", 15)],
        "high": [("Do 15 minutes of treadmill walking and avoid sugary drinks", "Light activity improves glucose uptake; re-check afterwards.", 15, "exercise"), _REMEASURE],
    },
    "creatinine_mgdl": {
        "high": [_HYDRATE, ("Review protein load and pain-medication use with the flight surgeon", "Kidney function should be checked.", 5, "medical")],
        "low": [("Add resistance exercise to protect muscle mass", "Low creatinine can reflect muscle loss; do the ARED session.", 15, "exercise")],
    },
    "sodium_mmol": {
        "high": [("Drink 500 ml plain water and avoid salty foods", "Rehydrate slowly and choose low-sodium items.", 15, "hydration")],
        "low": [("Take an electrolyte drink and avoid excess plain water", "Sip 300-500 ml of an electrolyte drink; report confusion or nausea.", 15, "hydration"), _SURGEON],
    },
    "potassium_mmol": {
        "high": [("Avoid potassium-rich foods and supplements", "Skip dried fruit, juices and supplements until reviewed.", 2, "nutrition"), _SURGEON],
        "low": [_nutrition("Eat potassium-rich foods", "Choose dried apricots, nuts or tomato-based items from the food system.", 10), _ECG_REPEAT],
    },
    "calcium_mgdl": {
        "high": [_HYDRATE, _SURGEON],
        "low": [("Check calcium and vitamin D intake", "Review your supplement log and discuss with the flight surgeon.", 5, "medical"), _VITD],
    },
    "crp_mgl": {
        "high": [_SURGEON, ("Rest, hydrate and sleep", "Elevated CRP suggests inflammation; reduce workload and drink water.", 15, "recovery")],
        "low": [],
    },
    "ctx_ngml": {
        "high": [("Do 20 min ARED heavy-load resistance exercise", "Mechanical loading counteracts bone resorption.", 20, "exercise"), _VITD],
        "low": [],
    },
    "vitamin_d_ngml": {
        "low": [_VITD, _nutrition("Eat vitamin D-fortified foods", "Choose fortified items such as fish or fortified drinks.", 5)],
        "high": [("Pause additional vitamin D supplements", "Do not take extra doses; discuss the level with the flight surgeon.", 3, "medical")],
    },
    "hemoglobin_gdl": {
        "low": [_nutrition("Eat an iron- and B12-rich meal", "Choose meat, fish, eggs or fortified cereal.", 10), ("Limit strenuous exercise until reviewed", "Low haemoglobin ('space anaemia') reduces oxygen carrying capacity; keep activity light.", 5, "recovery"), _SURGEON],
        "high": [_HYDRATE],
    },
    "hematocrit_pct": {
        "high": [_HYDRATE, _REMEASURE],
        "low": [_nutrition("Eat an iron- and B12-rich meal", "Choose meat, fish, eggs or fortified cereal.", 10), _SURGEON],
    },
    "rbc_mil": {
        "low": [_nutrition("Eat an iron- and B12-rich meal", "Choose meat, fish, eggs or fortified cereal.", 10), _SURGEON],
        "high": [_HYDRATE],
    },
    "wbc_k": {
        "high": [_SURGEON, ("Rest, hydrate and check your temperature", "Elevated white cells may signal infection.", 10, "monitoring")],
        "low": [("Follow infection-prevention hygiene", "Wipe shared surfaces, wash hands and report fever or sore throat.", 10, "hygiene"), _SURGEON],
    },
    "platelets_k": {
        "high": [_HYDRATE, _SURGEON],
        "low": [("Avoid activities with bruising or bleeding risk", "Handle tools carefully and report any bleeding.", 2, "medical"), _SURGEON],
    },
    # ======================= space-environment hazards (RIDGE) =======================
    "radiation_dose_msv_d": {
        "high": [("Move to the radiation storm shelter", "Go to the most shielded area (e.g. crew quarters lined with water/stowage "
                  "bags) and stay there until the dose rate drops; postpone EVAs.", 30, "radiation"),
                 ("Log your personal dosimeter reading", "Read the active dosimeter, log the value and report it to the ground "
                  "or the on-board medical officer.", 5, "radiation")],
        "low": [],
    },
    "radiation_cumulative_msv": {
        "high": [("Review your radiation dose budget with the flight surgeon", "Compare the cumulative dose with the mission "
                  "and career limits and agree on EVA / shielding plans.", 10, "radiation"),
                 ("Sleep in the shielded crew quarters", "Use the best-shielded sleep station to reduce the daily dose.", 5, "radiation")],
        "low": [],
    },
    "isolation_days": {
        "high": [("Schedule a private family conference", "Book a private video or voice call with family or friends.", 20, "wellbeing"),
                 ("Plan a shared crew meal or movie night", "Social time with the crew counters isolation-related stress.", 30, "wellbeing")],
        "low": [],
    },
    "habitable_volume_m3": {
        "low": [("Take a 15-minute privacy break in your crew quarters", "Close the curtain/door, put on music and decompress "
                 "alone.", 15, "wellbeing"),
                ("Re-organise stowage to free up personal space", "Move non-essential items out of the work and sleep areas.", 20, "environment")],
        "high": [],
    },
    "earth_distance_mkm": {
        "high": [("Review the autonomous medical checklist", "Refresh the on-board medical procedures you may need to perform "
                  "without real-time ground support.", 15, "medical"),
                 ("Run an offline medical-kit inventory check", "Verify supplies and expiry dates; communication delays mean the "
                  "crew must be self-sufficient.", 15, "monitoring")],
        "low": [],
    },
    "gravity_g": {
        "low": [_ARED, _TREADMILL,
                ("Wear the lower-body loading / thigh-cuff garment as scheduled", "Use the prescribed countermeasure garment to "
                 "limit fluid shift and deconditioning.", 30, "exercise")],
        "high": [],
    },
    "g_transition_days": {
        "low": [("Fluid-load before standing: drink 1 L of electrolyte water", "Rehydrate with an isotonic drink to support blood "
                 "pressure after the gravity transition.", 20, "hydration"),
                ("Move slowly; sit or lie down if you feel dizzy", "Rise in stages and avoid sudden head movements while your "
                 "balance and blood pressure re-adapt.", 5, "recovery")],
        "high": [],
    },
    "cabin_co2_mmhg": {
        "high": [("Check the CO2 scrubber and increase ventilation", "Confirm scrubber (CDRA) status, position a fan near your "
                  "work/sleep station and move to a well-ventilated module.", 10, "environment"),
                 ("Take a 10-minute break in a well-ventilated module", "CO2 pockets form without convection in microgravity.", 10, "environment")],
        "low": [],
    },
    "cabin_o2_kpa": {
        "low": [("Check ppO2 and follow the low-oxygen protocol", "Verify the O2 sensor, report to the ground and don an oxygen "
                 "mask if the protocol requires it.", 5, "environment"), _SURGEON],
        "high": [("Verify O2 partial pressure and remove ignition sources", "High ppO2 raises flammability - check the reading and "
                  "secure ignition sources.", 10, "environment")],
    },
    "cabin_temp_c": {
        "high": [("Lower the module temperature setpoint", "Adjust the thermal control setpoint, increase airflow and drink cool "
                  "water.", 10, "environment"), _COOL],
        "low": [("Raise the module temperature setpoint and add a layer", "Increase the heater setpoint and put on warm clothing.", 10, "environment")],
    },
    "cabin_humidity_pct": {
        "high": [("Check the humidity separator and wipe condensation", "Excess humidity promotes microbial growth; clean wet "
                  "surfaces and report the reading.", 15, "environment")],
        "low": [("Hydrate and protect your airways from dry air", "Drink water regularly and use the moisturising kit for eyes and "
                 "nose.", 5, "hydration")],
    },
    "cabin_noise_dba": {
        "high": [("Wear hearing protection during sleep and noisy work", "Use earplugs or noise-cancelling headphones and log the "
                  "noise dosimeter reading.", 5, "environment"),
                 _SLEEP_WIND],
        "low": [],
    },
}

# Canonical tags (drive the badges). Anything else coming from an LLM is normalised to one of these.
ALLOWED_TAGS = {"hydration", "exercise", "breathing", "sleep", "nutrition", "wellbeing", "ecg", "medical", "monitoring",
                "environment", "radiation", "cooling", "recovery", "hygiene", "general"}
_TAG_SYNONYMS = {"beslenme": "nutrition", "diyet": "nutrition", "sıvı": "hydration", "su": "hydration", "nefes": "breathing",
                 "egzersiz": "exercise", "spor": "exercise", "uyku": "sleep", "dinlenme": "recovery", "iyilik": "wellbeing",
                 "ekg": "ecg", "tıbbi": "medical", "doktor": "medical", "ölçüm": "monitoring", "izleme": "monitoring",
                 "çevre": "environment", "ortam": "environment", "habitat": "environment", "kabin": "environment",
                 "radyasyon": "radiation", "radiat": "radiation", "dosim": "radiation", "shelter": "radiation",
                 "soğutma": "cooling", "hijyen": "hygiene", "stres": "wellbeing", "rest": "recovery",
                 "diet": "nutrition", "water": "hydration", "fluid": "hydration", "breath": "breathing", "workout": "exercise",
                 "mental": "wellbeing", "stretching": "recovery", "measurement": "monitoring", "isolation": "wellbeing",
                 "izolasyon": "wellbeing", "ventilation": "environment"}


def normalize_tag(tag: str, fallback: str = "general") -> str:
    t = (tag or "").strip().lower()
    if t in ALLOWED_TAGS:
        return t
    for key, canon in _TAG_SYNONYMS.items():
        if key in t:
            return canon
    return fallback


CRITICAL_TASK: Template = (
    "Contact the flight surgeon NOW and follow the emergency protocol",
    "This value is in the immediate-intervention zone. Stop non-essential work, open the medical channel, "
    "report the reading and follow instructions.", 5, "medical")

# Black-level *environment* hazard (e.g. a severe solar particle event or a cabin-air emergency).
ENV_CRITICAL_TASK: Template = (
    "Alert mission control and start the hazard emergency procedure",
    "This environment hazard is in the immediate-intervention zone. Follow the emergency procedure for it "
    "(radiation shelter, mask / ventilation, thermal or pressure response) and report to mission control.", 5, "environment")


def templates_for(param_key: str, direction: str) -> List[Template]:
    entry = LIBRARY.get(param_key, {})
    t = entry.get(direction) or entry.get("low" if direction == "high" else "high") or []
    return list(t) if t else [_REMEASURE]
