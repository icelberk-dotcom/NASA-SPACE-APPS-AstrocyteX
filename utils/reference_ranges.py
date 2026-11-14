"""
Reference ranges and the percentage-based risk scale for every monitored parameter.

This module is the single source of truth for:
  * which parameters AstroGuard-X monitors (31 of them),
  * their normal / caution / danger limits,
  * how a raw value is converted into a 0-100 % risk score,
  * how a risk score maps to the coloured indicator (gray/green/yellow/red/black).

How the risk score works
------------------------
Every parameter has an *ideal* value (the astronaut's physiological baseline) and
three pairs of limits on each side of it::

        danger_lo  warn_lo  normal_lo   ideal   normal_hi  warn_hi  danger_hi
   risk:    90        70        40        8         40        70       90

The score is a piece-wise linear interpolation through those anchor points, so it
is monotone, easy to explain to a flight surgeon and trivially unit-testable.
Beyond a danger limit the score keeps growing asymptotically towards 100 %.
A side whose limit is ``None`` is "one-sided": e.g. a *low* C-reactive protein is
never a problem, so that side always returns a small baseline risk.

Indicator colours (as requested in the project brief)
-----------------------------------------------------
    < 15 %        gray    normal / baseline / inactive
    15 - 40 %     green   low risk / normal
    40 - 70 %     yellow  minor deviation
    70 - 90 %     red     high risk
    > 90 %        black   immediate intervention needed

The numeric reference limits are *illustrative* values compiled from common adult
clinical reference intervals and published spaceflight-medicine ranges (NASA
Human Research Program, NASA-STD-3001).  They must be reviewed by a qualified
flight surgeon before any operational use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Risk levels (colour indicator)
# --------------------------------------------------------------------------- #

LEVELS: List[Tuple[str, str, str, str]] = [
    # key,     label,                     hex colour, short meaning
    ("gray", "Normal / baseline", "#9aa0a6", "Within baseline"),
    ("green", "Low risk", "#2ecc71", "Normal range"),
    ("yellow", "Minor deviation", "#f1c40f", "Watch closely"),
    ("red", "High risk", "#e74c3c", "Act now"),
    ("black", "Immediate intervention", "#000000", "Emergency protocol"),
]
LEVEL_INFO: Dict[str, Dict[str, str]] = {
    k: {"label": label, "color": color, "meaning": meaning} for k, label, color, meaning in LEVELS
}
RISKY_LEVELS = ("yellow", "red", "black")  # levels that trigger AI task generation
LEVEL_ORDER = {k: i for i, (k, *_rest) in enumerate(LEVELS)}


def risk_level(risk_pct: Optional[float]) -> str:
    """Map a 0-100 % risk score to its indicator colour key.

    Boundaries follow the brief literally: <15 gray, 15-40 green, 40-70 yellow,
    70-90 red, >90 black.  ``None`` (no data yet) is treated as inactive/gray.
    """
    if risk_pct is None or (isinstance(risk_pct, float) and math.isnan(risk_pct)):
        return "gray"
    if risk_pct < 15.0:
        return "gray"
    if risk_pct <= 40.0:
        return "green"
    if risk_pct <= 70.0:
        return "yellow"
    if risk_pct <= 90.0:
        return "red"
    return "black"


def level_color(level: str) -> str:
    return LEVEL_INFO[level]["color"]


# --------------------------------------------------------------------------- #
# Parameter registry
# --------------------------------------------------------------------------- #

# Categories group parameters in the UI and act as "physiological systems" when
# the pipeline aggregates risk per system.
CATEGORIES: Dict[str, Dict[str, str]] = {
    "cardiovascular": {"en": "Cardiovascular", "tr": "Kardiyovasküler", "icon": "🫀"},
    "respiratory": {"en": "Respiratory", "tr": "Solunum", "icon": "🫁"},
    "stress_thermal": {"en": "Stress & Thermoregulation", "tr": "Stres ve Isı Düzenleme", "icon": "🌡️"},
    "sleep_activity": {"en": "Sleep & Exercise", "tr": "Uyku ve Egzersiz", "icon": "😴"},
    "hydration_renal": {"en": "Hydration & Renal", "tr": "Sıvı ve Böbrek", "icon": "💧"},
    "bone_muscle": {"en": "Bone & Muscle", "tr": "Kemik ve Kas", "icon": "🦴"},
    "hemogram": {"en": "Hemogram (CBC)", "tr": "Hemogram", "icon": "🩸"},
    "biochemistry": {"en": "Blood Biochemistry", "tr": "Kan Biyokimyası", "icon": "🧪"},
    "nutrition": {"en": "Nutrition & Diet", "tr": "Beslenme ve Diyet", "icon": "🥗"},
    "ecg": {"en": "ECG", "tr": "EKG", "icon": "📈"},
}


@dataclass(frozen=True)
class Parameter:
    key: str
    name_en: str
    name_tr: str
    unit: str
    category: str
    kind: str  # "continuous" (sensor stream) or "daily" (daily entry / lab test)
    ideal: float
    normal: Tuple[Optional[float], Optional[float]]
    warn: Tuple[Optional[float], Optional[float]]
    danger: Tuple[Optional[float], Optional[float]]
    # Plausible entry bounds for UI widgets: (min, max, step)
    bounds: Tuple[float, float, float] = (0.0, 100.0, 1.0)
    decimals: int = 0
    notes: str = ""
    # Filled automatically: normalisation scale used by the neural network inputs
    scale: float = field(default=1.0, compare=False)

    # ----- convenience helpers ------------------------------------------------
    def name(self, lang: str = "en") -> str:
        return self.name_tr if lang == "tr" else self.name_en

    def normal_range_text(self) -> str:
        lo, hi = self.normal
        if lo is None and hi is not None:
            return f"≤ {_fmt(hi, self.decimals)} {self.unit}".strip()
        if hi is None and lo is not None:
            return f"≥ {_fmt(lo, self.decimals)} {self.unit}".strip()
        return f"{_fmt(lo, self.decimals)} – {_fmt(hi, self.decimals)} {self.unit}".strip()


def _fmt(v: Optional[float], decimals: int) -> str:
    return "–" if v is None else f"{v:.{decimals}f}"


def _P(key, en, tr, unit, cat, kind, ideal, normal, warn, danger, bounds, decimals=0, notes=""):
    return Parameter(key, en, tr, unit, cat, kind, ideal, normal, warn, danger, bounds, decimals, notes)


# NOTE: order matters - it defines tensor column order for the neural networks.
# The first 7 entries are the continuous sensor channels (the Mamba-2 input).
_RAW_PARAMETERS: List[Parameter] = [
    # ---- continuous sensor channels -------------------------------------------------
    _P("heart_rate", "Heart rate / pulse", "Nabız / Kalp Ritmi", "bpm", "cardiovascular", "continuous",
       70, (55, 95), (45, 110), (35, 140), (25, 220, 1)),
    _P("spo2", "Oxygen saturation (SpO2)", "Oksijen Satürasyonu", "%", "respiratory", "continuous",
       98, (95, None), (92, None), (88, None), (60, 100, 1),
       0, "One-sided: a high SpO2 is never a problem (physically capped at 100 %)."),
    _P("systolic_bp", "Systolic blood pressure", "Sistolik Kan Basıncı", "mmHg", "cardiovascular", "continuous",
       115, (100, 130), (90, 145), (80, 180), (50, 260, 1)),
    _P("diastolic_bp", "Diastolic blood pressure", "Diyastolik Kan Basıncı", "mmHg", "cardiovascular", "continuous",
       75, (60, 85), (50, 92), (40, 110), (30, 160, 1)),
    _P("resp_rate", "Respiratory rate", "Solunum Hızı", "br/min", "respiratory", "continuous",
       14, (12, 18), (9, 23), (6, 30), (4, 60, 1)),
    _P("body_temp", "Body temperature", "Vücut Sıcaklığı", "°C", "stress_thermal", "continuous",
       36.7, (36.1, 37.2), (35.5, 38.0), (34.5, 39.5), (32.0, 42.0, 0.1), 1),
    _P("gsr", "Galvanic skin response (GSR)", "Galvanik Deri Yanıtı (GSR)", "µS", "stress_thermal", "continuous",
       4.0, (1.0, 8.0), (0.5, 14.0), (0.2, 25.0), (0.0, 40.0, 0.1), 1,
       "High values indicate sympathetic arousal (stress)."),
    # ---- sleep, hydration, activity -------------------------------------------------
    _P("sleep_hours", "Sleep duration", "Uyku Saatleri", "h", "sleep_activity", "continuous",
       8.0, (7.0, 9.0), (5.5, 10.5), (4.0, 12.0), (0.0, 16.0, 0.25), 2,
       "Nightly total from the wearable (continuous sleep tracking)."),
    _P("fluid_intake_ml", "Fluid intake", "Sıvı Tüketimi", "mL/day", "hydration_renal", "daily",
       2500, (2000, 3200), (1500, 4000), (800, 6000), (0, 8000, 50)),
    _P("exercise_min", "Exercise duration", "Egzersiz Süresi", "min/day", "sleep_activity", "daily",
       120, (90, 180), (45, 240), (10, 360), (0, 480, 5),
       notes="ISS crews typically exercise ~2 h/day (ARED, treadmill, cycle)."),
    # ---- urine & saliva tests --------------------------------------------------------
    _P("urine_sg", "Urine specific gravity", "İdrar Dansitesi", "", "hydration_renal", "daily",
       1.015, (1.005, 1.025), (1.002, 1.030), (1.000, 1.040), (1.000, 1.050, 0.001), 3),
    _P("urine_ph", "Urine pH", "İdrar pH", "", "hydration_renal", "daily",
       6.2, (5.0, 7.5), (4.6, 8.0), (4.2, 8.8), (4.0, 9.0, 0.1), 1),
    _P("urine_calcium", "Urine calcium (24 h)", "İdrar Kalsiyumu (24 s)", "mg/24h", "bone_muscle", "daily",
       150, (50, 250), (25, 320), (5, 450), (0, 700, 5), 0,
       "Elevated excretion reflects bone resorption in microgravity."),
    _P("saliva_cortisol", "Saliva cortisol", "Tükürük Kortizolü", "nmol/L", "stress_thermal", "daily",
       8.0, (3.0, 15.0), (1.5, 22.0), (0.8, 35.0), (0.0, 60.0, 0.5), 1),
    # ---- nutrition & diet ------------------------------------------------------------
    _P("calories_kcal", "Energy intake", "Enerji Alımı", "kcal/day", "nutrition", "daily",
       2800, (2300, 3300), (1800, 3800), (1200, 4800), (0, 7000, 50)),
    _P("protein_g", "Protein intake", "Protein Alımı", "g/day", "nutrition", "daily",
       100, (80, 140), (55, 180), (30, 260), (0, 400, 5)),
    # ---- ECG derived (filled by the ECG analyser adapter) ----------------------------
    _P("ecg_qtc_ms", "ECG QTc interval", "EKG QTc Aralığı", "ms", "ecg", "daily",
       400, (350, 440), (330, 470), (300, 500), (250, 700, 1), 0,
       "Bazett-corrected QT; prolongation raises arrhythmia risk."),
    _P("ecg_abnormal_pct", "ECG abnormality probability", "EKG Anormallik Olasılığı", "%", "ecg", "daily",
       5, (None, 25), (None, 50), (None, 80), (0, 100, 1), 0,
       "100 - P(normal sinus rhythm) from the ECG analyser (Teknofest V23 model)."),
    # ---- blood biochemistry ----------------------------------------------------------
    _P("glucose_mgdl", "Glucose", "Glukoz", "mg/dL", "biochemistry", "daily",
       90, (70, 100), (60, 126), (45, 200), (20, 600, 1)),
    _P("creatinine_mgdl", "Creatinine", "Kreatinin", "mg/dL", "biochemistry", "daily",
       0.9, (0.6, 1.2), (0.5, 1.5), (0.4, 3.0), (0.2, 8.0, 0.05), 2),
    _P("sodium_mmol", "Sodium", "Sodyum", "mmol/L", "biochemistry", "daily",
       140, (136, 145), (132, 149), (125, 158), (100, 180, 1)),
    _P("potassium_mmol", "Potassium", "Potasyum", "mmol/L", "biochemistry", "daily",
       4.2, (3.5, 5.0), (3.2, 5.4), (2.8, 6.2), (2.0, 8.0, 0.1), 1),
    _P("calcium_mgdl", "Calcium", "Kalsiyum", "mg/dL", "biochemistry", "daily",
       9.5, (8.6, 10.2), (8.2, 10.8), (7.0, 12.5), (5.0, 15.0, 0.1), 1),
    _P("crp_mgl", "C-reactive protein (CRP)", "C-Reaktif Protein (CRP)", "mg/L", "biochemistry", "daily",
       1.0, (None, 5.0), (None, 10.0), (None, 50.0), (0.0, 300.0, 0.5), 1),
    _P("ctx_ngml", "Bone resorption marker (CTX)", "Kemik Yıkım Belirteci (CTX)", "ng/mL", "bone_muscle", "daily",
       0.30, (None, 0.60), (None, 0.90), (None, 1.50), (0.0, 3.0, 0.05), 2,
       "Serum C-terminal telopeptide, rises with bone loss."),
    _P("vitamin_d_ngml", "Vitamin D (25-OH)", "D Vitamini (25-OH)", "ng/mL", "bone_muscle", "daily",
       40, (30, 60), (20, 80), (10, 100), (0, 150, 1)),
    # ---- hemogram (CBC) --------------------------------------------------------------
    _P("hemoglobin_gdl", "Hemoglobin", "Hemoglobin", "g/dL", "hemogram", "daily",
       14.5, (13.0, 17.0), (11.5, 18.0), (8.5, 20.0), (5.0, 24.0, 0.1), 1,
       "Space anaemia (neocytolysis) lowers red-cell mass during flight."),
    _P("hematocrit_pct", "Hematocrit", "Hematokrit", "%", "hemogram", "daily",
       43, (38, 50), (34, 54), (27, 60), (15, 70, 0.5), 1),
    _P("rbc_mil", "Red blood cells", "Eritrosit", "10⁶/µL", "hemogram", "daily",
       4.9, (4.3, 5.8), (3.8, 6.3), (3.0, 7.0), (2.0, 9.0, 0.05), 2),
    _P("wbc_k", "White blood cells", "Lökosit", "10³/µL", "hemogram", "daily",
       6.5, (4.0, 10.5), (3.0, 13.0), (1.5, 25.0), (0.5, 50.0, 0.1), 1),
    _P("platelets_k", "Platelets", "Trombosit", "10³/µL", "hemogram", "daily",
       250, (150, 400), (110, 480), (50, 800), (10, 1200, 5)),
]


def _validate_and_finalise(p: Parameter) -> Parameter:
    """Check that the limits are ordered correctly and compute the NN scale."""
    lo_n, hi_n = p.normal
    lo_w, hi_w = p.warn
    lo_d, hi_d = p.danger
    if hi_n is not None:
        assert p.ideal < hi_n, f"{p.key}: ideal must be below normal_hi"
        assert hi_w is not None and hi_d is not None
        assert hi_n < hi_w < hi_d, f"{p.key}: high-side limits must be strictly ordered"
    if lo_n is not None:
        assert p.ideal > lo_n, f"{p.key}: ideal must be above normal_lo"
        assert lo_w is not None and lo_d is not None
        assert lo_n > lo_w > lo_d, f"{p.key}: low-side limits must be strictly ordered"
    # Scale = half of the normal band, used to z-score inputs for the networks.
    if lo_n is not None and hi_n is not None:
        scale = (hi_n - lo_n) / 2.0
    elif hi_n is not None:
        scale = hi_n - p.ideal
    else:
        scale = p.ideal - lo_n  # type: ignore[operator]
    object.__setattr__(p, "scale", float(max(scale, 1e-6)))
    return p


PARAMETERS: List[Parameter] = [_validate_and_finalise(p) for p in _RAW_PARAMETERS]
PARAM_BY_KEY: Dict[str, Parameter] = {p.key: p for p in PARAMETERS}
PARAM_KEYS: List[str] = [p.key for p in PARAMETERS]
SEQ_KEYS: List[str] = [p.key for p in PARAMETERS[:7]]  # continuous sensor channels
DAILY_KEYS: List[str] = [p.key for p in PARAMETERS if p.key not in SEQ_KEYS]
DEFAULT_VALUES: Dict[str, float] = {p.key: p.ideal for p in PARAMETERS}
CATEGORY_KEYS: List[str] = list(CATEGORIES.keys())

assert PARAM_KEYS[:7] == ["heart_rate", "spo2", "systolic_bp", "diastolic_bp", "resp_rate", "body_temp", "gsr"]


# --------------------------------------------------------------------------- #
# Risk scoring
# --------------------------------------------------------------------------- #

BASELINE_RISK = 8.0  # risk at the ideal value
ONE_SIDED_FLOOR = 6.0  # risk on a side that has no limit (e.g. low CRP)


def _side_risk_arr(distance, d_norm: float, d_warn: float, d_danger: float) -> np.ndarray:
    """Vectorised risk for values `distance` away from ideal, towards one side.

    Anchors: 0 -> 8 %, d_norm -> 40 %, d_warn -> 70 %, d_danger -> 90 %.
    Past the danger limit the score approaches 100 % asymptotically.
    """
    d = np.asarray(distance, dtype=float)
    xs = [0.0, d_norm, d_warn, d_danger]
    ys = [BASELINE_RISK, 40.0, 70.0, 90.0]
    # Collapse duplicate anchors (a warn limit equal to the normal limit keeps the higher level).
    cx, cy = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        if x > cx[-1] + 1e-9:
            cx.append(x)
            cy.append(y)
        else:
            cy[-1] = y
    inside = np.interp(np.maximum(d, 0.0), cx, cy)
    span = max(d_danger - d_warn, d_danger * 0.25, 1e-6)
    beyond = 90.0 + 10.0 * (1.0 - np.exp(-(d - cx[-1]) / span)) * 0.999 + 1e-6
    return np.where(d <= cx[-1], inside, beyond)


def risk_percent_array(key: str, values) -> np.ndarray:
    """Vectorised :func:`risk_percent` for a NumPy array of measurements (no NaN handling)."""
    p = PARAM_BY_KEY[key]
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, ONE_SIDED_FLOOR)
    hi_n, hi_w, hi_d = p.normal[1], p.warn[1], p.danger[1]
    lo_n, lo_w, lo_d = p.normal[0], p.warn[0], p.danger[0]
    if hi_n is not None:
        hi = _side_risk_arr(v - p.ideal, hi_n - p.ideal, hi_w - p.ideal, hi_d - p.ideal)
        out = np.where(v >= p.ideal, hi, out)
    if lo_n is not None:
        lo = _side_risk_arr(p.ideal - v, p.ideal - lo_n, p.ideal - lo_w, p.ideal - lo_d)
        out = np.where(v < p.ideal, lo, out)
    return out


def risk_percent(key: str, value: Optional[float]) -> Optional[float]:
    """Convert a raw measurement of parameter `key` into a 0-100 % risk score."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(risk_percent_array(key, np.asarray(float(value))))

def deviation_direction(key: str, value: Optional[float]) -> str:
    """'low', 'high' or 'ok' - which side of the normal band a value is on."""
    if value is None:
        return "ok"
    p = PARAM_BY_KEY[key]
    lo, hi = p.normal
    if lo is not None and value < lo:
        return "low"
    if hi is not None and value > hi:
        return "high"
    # Inside the normal band but already drifting: use the ideal as a tie-breaker.
    return "ok"


def risk_direction(key: str, value: Optional[float]) -> str:
    """Direction used for task generation: which way the value is *drifting*.

    Unlike :func:`deviation_direction` this also reports a side for values that are
    still inside the normal band but have a risk score in the yellow zone (which can
    happen when the neural network adds uncertainty / cross-parameter coupling).
    """
    d = deviation_direction(key, value)
    if d != "ok" or value is None:
        return d
    return "high" if value >= PARAM_BY_KEY[key].ideal else "low"


def normalize_value(key: str, value: float) -> float:
    """z-score a raw value against its reference range (network input space)."""
    p = PARAM_BY_KEY[key]
    z = (float(value) - p.ideal) / p.scale
    return float(np.clip(z, -6.0, 6.0))


def denormalize_value(key: str, z: float) -> float:
    p = PARAM_BY_KEY[key]
    return float(p.ideal + z * p.scale)


def clip_to_bounds(key: str, value: float) -> float:
    lo, hi, _ = PARAM_BY_KEY[key].bounds
    return float(min(max(value, lo), hi))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate_risk(risks: List[float]) -> float:
    """Combine several 0-100 % risks into one score.

    Medicine is dominated by the worst finding, but several mild deviations
    together are also meaningful.  We therefore blend the worst value, the mean of
    the three worst values and the overall mean.  Works for NumPy arrays too.
    """
    if len(risks) == 0:
        return 0.0
    arr = np.sort(np.asarray(risks, dtype=float))[::-1]
    return float(0.60 * arr[0] + 0.25 * arr[:3].mean() + 0.15 * arr.mean())


def system_risks(param_risks: Dict[str, float]) -> Dict[str, float]:
    """Aggregate parameter risks into one risk per physiological system."""
    out: Dict[str, float] = {}
    for cat in CATEGORY_KEYS:
        vals = [param_risks[p.key] for p in PARAMETERS if p.category == cat and p.key in param_risks
                and param_risks[p.key] is not None]
        if vals:
            out[cat] = aggregate_risk(vals)
    return out


def params_in_category(cat: str) -> List[Parameter]:
    return [p for p in PARAMETERS if p.category == cat]
