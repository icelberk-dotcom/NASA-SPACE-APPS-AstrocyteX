"""
Space-environment knowledge: mission profiles, hazard events and derived quantities.

NASA's Human Research Program groups the hazards of human spaceflight into five
categories ("RIDGE"): **R**adiation, **I**solation & confinement, **D**istance from Earth,
**G**ravity fields and hostile/closed **E**nvironments.  AstroGuard-X represents each of
them with measurable parameters (see the environment block in ``reference_ranges.py``).

This module provides
  * ``MISSION_PROFILES`` - typical environment values for ISS, Gateway, the lunar surface,
    a Mars transit, the Martian surface and an Earth analog (HERA / Antarctic station),
    used to pre-fill the dashboard and to simulate training crews;
  * ``ENV_EVENTS`` - off-nominal hazard events (solar particle event, CO2 scrubber
    failure, thermal-control failure, O2 drop, noisy habitat, prolonged isolation,
    post-landing gravity transition) with the symptoms crews typically report;
  * ``derived_environment`` - quantities the crew and the advisor reason about but that do
    not need their own input field (one-way communication delay, time to definitive care).

All numbers are illustrative literature values (NASA-STD-3001, ISS environmental data,
MSL/RAD radiation measurements); they are NOT operational limits.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from utils.reference_ranges import DEFAULT_VALUES, ENV_KEYS, PARAM_BY_KEY

SPEED_OF_LIGHT_KM_S = 299_792.458

# ------------------------------------------------------------------------------ profiles
# Each entry: parameter -> (mean, standard deviation) for the simulator.  Mission-time
# dependent parameters (isolation days, cumulative dose, days since gravity transition) are
# derived from the mission day by ``profile_values``.
MISSION_PROFILES: Dict[str, Dict[str, object]] = {
    "earth_analog": {
        "en": "Earth analog (HERA / Antarctic station)", "tr": "Dünya analoğu (HERA / Antarktika istasyonu)",
        "env": {"radiation_dose_msv_d": (0.01, 0.004), "habitable_volume_m3": (35, 8), "earth_distance_mkm": (0.0, 0.0),
                "gravity_g": (1.0, 0.0), "cabin_co2_mmhg": (2.0, 0.6), "cabin_o2_kpa": (21.0, 0.3),
                "cabin_temp_c": (22.0, 1.2), "cabin_humidity_pct": (40, 8), "cabin_noise_dba": (50, 5)},
        "transition": False, "mission_days": (0, 400),
    },
    "iss_leo": {
        "en": "ISS - low Earth orbit", "tr": "ISS - alçak Dünya yörüngesi",
        "env": {"radiation_dose_msv_d": (0.55, 0.15), "habitable_volume_m3": (55, 10), "earth_distance_mkm": (0.0004, 0.00003),
                "gravity_g": (0.0, 0.0), "cabin_co2_mmhg": (2.6, 0.6), "cabin_o2_kpa": (21.2, 0.4),
                "cabin_temp_c": (22.5, 1.0), "cabin_humidity_pct": (45, 7), "cabin_noise_dba": (57, 4)},
        "transition": True, "mission_days": (0, 365),
    },
    "lunar_gateway": {
        "en": "Lunar Gateway (cislunar orbit)", "tr": "Ay Geçidi (Ay yörüngesi)",
        "env": {"radiation_dose_msv_d": (1.2, 0.3), "habitable_volume_m3": (30, 6), "earth_distance_mkm": (0.384, 0.02),
                "gravity_g": (0.0, 0.0), "cabin_co2_mmhg": (2.3, 0.5), "cabin_o2_kpa": (21.0, 0.4),
                "cabin_temp_c": (22.0, 1.0), "cabin_humidity_pct": (45, 7), "cabin_noise_dba": (53, 4)},
        "transition": True, "mission_days": (0, 90),
    },
    "lunar_surface": {
        "en": "Lunar surface habitat", "tr": "Ay yüzeyi habitatı",
        "env": {"radiation_dose_msv_d": (1.35, 0.3), "habitable_volume_m3": (20, 5), "earth_distance_mkm": (0.384, 0.02),
                "gravity_g": (0.166, 0.0), "cabin_co2_mmhg": (2.8, 0.6), "cabin_o2_kpa": (21.0, 0.5),
                "cabin_temp_c": (22.0, 1.5), "cabin_humidity_pct": (40, 8), "cabin_noise_dba": (50, 4)},
        "transition": True, "mission_days": (0, 60),
    },
    "mars_transit": {
        "en": "Mars transit (deep space)", "tr": "Mars transferi (derin uzay)",
        "env": {"radiation_dose_msv_d": (1.8, 0.3), "habitable_volume_m3": (25, 6), "earth_distance_mkm": (None, None),
                "gravity_g": (0.0, 0.0), "cabin_co2_mmhg": (2.6, 0.6), "cabin_o2_kpa": (21.0, 0.4),
                "cabin_temp_c": (22.0, 1.0), "cabin_humidity_pct": (45, 7), "cabin_noise_dba": (54, 4)},
        "transition": True, "mission_days": (0, 270),
    },
    "mars_surface": {
        "en": "Mars surface habitat", "tr": "Mars yüzeyi habitatı",
        "env": {"radiation_dose_msv_d": (0.67, 0.12), "habitable_volume_m3": (35, 8), "earth_distance_mkm": (None, None),
                "gravity_g": (0.38, 0.0), "cabin_co2_mmhg": (2.7, 0.6), "cabin_o2_kpa": (21.0, 0.5),
                "cabin_temp_c": (21.5, 1.5), "cabin_humidity_pct": (40, 8), "cabin_noise_dba": (50, 4)},
        "transition": True, "mission_days": (180, 900),
    },
}
PROFILE_NAMES: List[str] = list(MISSION_PROFILES.keys())

# --------------------------------------------------------------------------------- events
# Off-nominal hazard events.  ``shift`` is an *absolute* change in physical units at full
# strength (sampled uniformly from the range); ``set`` overrides a parameter.  ``text`` holds
# typical crew reports (EN / TR) for the subjective-log encoder.
ENV_EVENTS: Dict[str, Dict[str, object]] = {
    "solar_particle_event": {
        "en": "Solar particle event (radiation storm)", "tr": "Güneş parçacık olayı (radyasyon fırtınası)",
        "shift": {"radiation_dose_msv_d": (8.0, 80.0)},
        "text": ["Feeling nauseous and very tired.", "Midem bulanıyor ve çok yorgunum."],
    },
    "co2_scrubber_failure": {
        "en": "CO₂ scrubber failure", "tr": "CO₂ temizleyici arızası",
        "shift": {"cabin_co2_mmhg": (3.0, 7.0)},
        "text": ["Bad headache, feeling foggy and short of breath.", "Başım ağrıyor, nefes darlığı var."],
    },
    "thermal_control_failure": {
        "en": "Thermal-control failure", "tr": "Isı kontrol sistemi arızası",
        "shift": {"cabin_temp_c": (5.0, 11.0), "cabin_humidity_pct": (10.0, 30.0)},
        "text": ["Hot and sweaty, thirsty.", "Çok sıcak, terliyorum ve susadım."],
    },
    "cabin_o2_drop": {
        "en": "Cabin O₂ drop (leak / generator fault)", "tr": "Kabin O₂ düşüşü (sızıntı / jeneratör arızası)",
        "shift": {"cabin_o2_kpa": (-7.0, -3.0)},
        "text": ["Short of breath and dizzy.", "Nefes darlığı ve baş dönmesi var."],
    },
    "noisy_habitat": {
        "en": "Noisy habitat (fan / pump fault)", "tr": "Gürültülü habitat (fan / pompa arızası)",
        "shift": {"cabin_noise_dba": (10.0, 26.0)},
        "text": ["Couldn't sleep because of the noise, irritable.", "Gürültüden uyuyamadım, sinirliyim."],
    },
    "prolonged_isolation": {
        "en": "Prolonged isolation & crowding", "tr": "Uzun süreli izolasyon ve kalabalık",
        "shift": {"isolation_days": (150.0, 500.0)}, "scale": {"habitable_volume_m3": (0.35, 0.7)},
        "text": ["Feeling lonely, irritable and down.", "Yalnız ve mutsuz hissediyorum, sinirliyim."],
    },
    "post_landing": {
        "en": "Recent gravity transition (landing / arrival)", "tr": "Yakın yerçekimi geçişi (iniş / varış)",
        "set": {"g_transition_days": (0.0, 4.0)},
        "text": ["Dizzy and lightheaded when I stand up.", "Ayağa kalkınca başım dönüyor."],
    },
}
EVENT_NAMES: List[str] = list(ENV_EVENTS.keys())


# ------------------------------------------------------------------------------- helpers
def profile_values(name: str, mission_day: float, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Environment values for a mission profile at a given mission day.

    With ``rng`` the values are sampled (simulator); without it the profile means are returned
    (deterministic dashboard preset).
    """
    prof = MISSION_PROFILES[name]
    out: Dict[str, float] = {}
    for key, (mu, sd) in prof["env"].items():  # type: ignore[union-attr]
        if key == "earth_distance_mkm" and mu is None:  # Mars: synodic geometry, 55 - 400 million km
            out[key] = float(rng.uniform(55.0, 400.0)) if rng is not None else 225.0
            continue
        v = mu if rng is None or not sd else rng.normal(mu, sd)
        out[key] = float(v)
    day = float(mission_day)
    quarantine = 14.0
    out["isolation_days"] = day + quarantine
    prior_career = float(rng.uniform(0.0, 120.0)) if rng is not None else 30.0
    out["radiation_cumulative_msv"] = prior_career + out["radiation_dose_msv_d"] * day
    if prof["transition"]:
        out["g_transition_days"] = day
    else:
        out["g_transition_days"] = float(rng.uniform(300.0, 1500.0)) if rng is not None else 1000.0
    return clip_env(out)


def clip_env(values: Mapping[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in values.items():
        lo, hi, _ = PARAM_BY_KEY[k].bounds
        out[k] = float(min(max(v, lo), hi))
    return out


def apply_event(values: Mapping[str, float], event: str, strength: float = 1.0,
                rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Apply an off-nominal event to environment values (deterministic mid-range without ``rng``)."""
    ev = ENV_EVENTS[event]
    out = dict(values)
    pick = (lambda lo, hi: float(rng.uniform(lo, hi))) if rng is not None else (lambda lo, hi: (lo + hi) / 2.0)
    for key, (lo, hi) in ev.get("shift", {}).items():  # type: ignore[union-attr]
        out[key] = out.get(key, DEFAULT_VALUES[key]) + strength * pick(lo, hi)
    for key, (lo, hi) in ev.get("scale", {}).items():  # type: ignore[union-attr]
        f = pick(lo, hi)
        out[key] = out.get(key, DEFAULT_VALUES[key]) * (1.0 - strength * (1.0 - f))
    for key, (lo, hi) in ev.get("set", {}).items():  # type: ignore[union-attr]
        out[key] = pick(lo, hi)
    return clip_env(out)


def environment_preset(profile: str, mission_day: float = 90.0, event: Optional[str] = None) -> Dict[str, float]:
    """Deterministic environment values for the dashboard (profile means + optional event)."""
    vals = profile_values(profile, mission_day)
    if event:
        vals = apply_event(vals, event)
    return {k: round(v, PARAM_BY_KEY[k].decimals + 2) for k, v in vals.items()}


def event_log(event: str, lang: str = "en") -> str:
    texts = ENV_EVENTS[event]["text"]  # type: ignore[index]
    return texts[1] if lang == "tr" and len(texts) > 1 else texts[0]


def derived_environment(values: Mapping[str, float]) -> Dict[str, Optional[float]]:
    """Quantities derived from the environment inputs (for the dashboard and the advisor).

    * one-way light-time communication delay (s)
    * a rough "time to definitive medical care" (days): ~0.25 d from LEO, ~4 d from the Moon,
      and for Mars the return is constrained by orbital mechanics (modelled as >= 180 d).
    """
    d = values.get("earth_distance_mkm")
    if d is None:
        return {"comm_delay_s": None, "time_to_care_days": None}
    delay = float(d) * 1e6 / SPEED_OF_LIGHT_KM_S
    if d < 0.01:
        care = 0.25
    elif d < 2.0:
        care = 4.0
    else:
        care = 180.0
    return {"comm_delay_s": round(delay, 2), "time_to_care_days": care}


def env_vector(values: Mapping[str, float]) -> Tuple[List[str], List[float]]:
    return ENV_KEYS, [float(values.get(k, DEFAULT_VALUES[k])) for k in ENV_KEYS]
