"""
AstrocyteX service layer - the single place where all components meet.

    reading -> feature builder -> AstrocyteX-Net v3.0 (Mamba-2 | TCN+TFT | fusion | ODE/PINN | Bayesian)
                                         |
    reference ranges (transparent) ------+--> blended risk % + uncertainty --> colour indicators
                                         |
    ECG adapter + AstroVoiceNet v1.0 ----'                    |
                                                              v
         Gemini / offline reasoner (doctor / non_doctor) --> briefing + mitigation tasks --> gamification
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from backend.ecg_adapter import ECGAdapter, ECGResult
from backend.gamification import Gamification, task_points
from backend.gemini_agent import GeminiAdvisor, is_turkish_text
from backend.schemas import (Assessment, CompletionResult, GamificationState, ParameterAssessment,
                             ReadingIn, TaskDraft, TaskOut)
from backend.storage import Storage, utcnow
from backend.task_library import templates_for
from backend.voice_analyzer import VoiceAnalyzer
from models.astroguard import AstroGuardNet, MODEL_VERSION, load_checkpoint
from models.fusion import MODALITIES
from models.physics import STATE_NAMES
from utils.config import Settings, get_settings
from utils.environment import derived_environment
from utils.features import build_sample, stack_samples
from utils.reference_ranges import (CATEGORIES, DEFAULT_VALUES, ENV_KEYS, HEALTH_IDX, LEVEL_ORDER, PARAM_BY_KEY,
                                    PARAM_KEYS, PARAMETERS, RISKY_LEVELS, clip_to_bounds, environment_hazard_index,
                                    health_overall_risk, level_color, risk_direction, risk_level, risk_percent,
                                    system_risks)
from utils.subjective import SubjectiveEncoder

log = logging.getLogger("astroguard.service")
MAX_NEW_TASKS_PER_ASSESSMENT = 12
PROJECTION_DAYS = 180
MC_SEED = 1234  # common random numbers: actual vs. counterfactual predictions use identical MC draws


class AstroGuardService:
    def __init__(self, settings: Optional[Settings] = None, model: Optional[AstroGuardNet] = None,
                 load_model: bool = True, advisor: Optional[GeminiAdvisor] = None,
                 ecg: Optional[ECGAdapter] = None, voice: Optional[VoiceAnalyzer] = None,
                 storage: Optional[Storage] = None):
        self.settings = settings or get_settings()
        self.storage = storage or Storage(self.settings.db_path)
        self.gamification = Gamification(self.storage)
        self.advisor = advisor or GeminiAdvisor(self.settings)
        self.ecg = ecg or ECGAdapter(self.settings)
        self.voice = voice or VoiceAnalyzer(self.settings, load_model=load_model)
        self.encoder = SubjectiveEncoder(self.settings.subjective_backend)
        self._lock = threading.Lock()
        self.model: Optional[AstroGuardNet] = model
        if model is None and load_model:
            self.model = self._load_or_train_model()

    # ---------------------------------------------------------------- model mgmt
    def _load_or_train_model(self) -> Optional[AstroGuardNet]:
        path = self.settings.checkpoint_path
        if path.exists():
            try:
                log.info("Loading AstroGuardNet checkpoint %s", path)
                return load_checkpoint(path)
            except Exception as exc:
                log.error("Could not load checkpoint (%s) - retraining", exc)
        if self.settings.autotrain:
            from models.train import train

            log.warning("No checkpoint found - training a compact model on simulated crews (one-off, a few minutes)...")
            train(samples=1600, epochs=6, out_path=path, verbose=True)
            return load_checkpoint(path)
        log.warning("No model available: running in transparent reference-range mode only.")
        return None

    def model_info(self) -> Dict[str, Any]:
        meta = getattr(self.model, "meta", {}) if self.model else {}
        return {"loaded": self.model is not None, "version": MODEL_VERSION if self.model else None,
                "trained_on": meta.get("data", "n/a"), "validation": meta.get("metrics", {}),
                "device": "cpu", "rule_blend": self.settings.rule_blend, "environment_aware": self.model is not None,
                "official_mamba_ssm": bool(self.model and self.model.cfg.prefer_official_mamba and torch.cuda.is_available())}

    def _predict(self, sample: Dict[str, np.ndarray], n_mc: int, project_days: Optional[int] = None) -> Dict[str, Any]:
        """Monte-Carlo prediction with a fixed, isolated RNG stream (reproducible, paired comparisons)."""
        with self._lock, torch.random.fork_rng(devices=[]):
            torch.manual_seed(MC_SEED)
            return self.model.predict(stack_samples([sample]), n_mc=n_mc, project_days=project_days)

    def _environment_uplift(self, effective: Dict[str, float], reading: ReadingIn, history: List[Dict[str, float]],
                            subj_vec: np.ndarray, pred: Dict[str, Any]) -> Dict[str, Any]:
        """Counterfactual: how much does the current space environment raise the model's health risk?

        The same astronaut data are re-evaluated with every environment hazard set to its benign
        Earth-like ideal (1 g, no radiation, no isolation, nominal cabin air, on Earth).  The
        difference is the *environmental risk uplift* - the part of the risk that exists only
        because of radiation, isolation, distance, gravity and the closed habitat.
        """
        neutral = {**DEFAULT_VALUES, **effective, **{k: DEFAULT_VALUES[k] for k in ENV_KEYS}}
        neutral_hist = [{**row, **{k: DEFAULT_VALUES[k] for k in ENV_KEYS}} for row in history]
        sample = build_sample(neutral, reading.sensor_series, neutral_hist, subj_vec, reading.mission_day, 1.0)
        pn = self._predict(sample, n_mc=self.settings.mc_samples)
        actual, base = float(pred["risk"][0, -1]), float(pn["risk"][0, -1])
        diffs = [(PARAM_KEYS[j], float(pred["risk"][0, j] - pn["risk"][0, j])) for j in HEALTH_IDX]
        top = [{"key": k, "name": PARAM_BY_KEY[k].name_en, "name_tr": PARAM_BY_KEY[k].name_tr, "uplift_pts": round(d, 1)}
               for k, d in sorted(diffs, key=lambda x: -x[1])[:5] if d > 0.5]
        return {"overall_model_pct": round(actual, 1), "overall_neutral_env_pct": round(base, 1),
                "uplift_pts": round(actual - base, 1), "top_parameters": top}

    # ------------------------------------------------------------------- helpers
    def _effective_values(self, astronaut_id: str, provided: Dict[str, float]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Provided values + forward-filled previous ones. Returns (values, history rows)."""
        history = self.storage.recent_values(astronaut_id, 13)
        effective: Dict[str, float] = {}
        for row in history:  # oldest -> newest => newest wins
            effective.update(row)
        effective.update(provided)
        return effective, history

    @staticmethod
    def _validate_keys(values: Dict[str, float]) -> Dict[str, float]:
        unknown = [k for k in values if k not in PARAM_BY_KEY]
        if unknown:
            raise ValueError(f"Unknown parameter keys: {unknown}")
        return {k: clip_to_bounds(k, float(v)) for k, v in values.items() if v is not None and np.isfinite(v)}

    # ---------------------------------------------------------------------- core
    def assess(self, astronaut_id: str, reading: ReadingIn, lang: str = "en", name: Optional[str] = None) -> Assessment:
        self.storage.ensure_astronaut(astronaut_id, name)
        provided = self._validate_keys(reading.values)

        ecg_dict: Optional[Dict[str, Any]] = None
        if reading.ecg:
            ecg_dict = reading.ecg
            for k, v in _ecg_parameter_values(reading.ecg).items():
                if k == "heart_rate" and "heart_rate" in provided:
                    continue  # a wearable HR reading wins over the 10-second ECG estimate
                provided[k] = clip_to_bounds(k, v)
        # Gravity: the "gravity_g" environment parameter is the source of truth; the legacy
        # ReadingIn.gravity field is used when the parameter itself was not supplied.
        if "gravity_g" not in provided and "gravity" in reading.model_fields_set:
            provided["gravity_g"] = clip_to_bounds("gravity_g", reading.gravity)

        effective, history = self._effective_values(astronaut_id, provided)
        known = {k: effective[k] for k in PARAM_KEYS if k in effective}  # parameters we have *any* data for
        gravity = float(known.get("gravity_g", reading.gravity))

        subj_vec = self.encoder.encode(reading.subjective_log)
        symptoms = self.encoder.describe(subj_vec)

        # ---- neural prediction -----------------------------------------------------
        pred: Optional[Dict[str, Any]] = None
        if self.model is not None:
            sample = build_sample({**DEFAULT_VALUES, **effective}, reading.sensor_series, history, subj_vec,
                                  reading.mission_day, gravity)
            pred = self._predict(sample, n_mc=self.settings.mc_samples, project_days=PROJECTION_DAYS)

        a = float(np.clip(self.settings.rule_blend, 0.0, 1.0)) if pred is not None else 1.0
        params_out: List[ParameterAssessment] = []
        final_risks: Dict[str, float] = {}
        for i, p in enumerate(PARAMETERS):
            value = known.get(p.key)
            rule = risk_percent(p.key, value)
            model_r = ci_lo = ci_hi = epi = alea = None
            final: Optional[float] = rule
            if rule is not None and pred is not None:
                model_r = float(pred["risk"][0, i])
                epi, alea = float(pred["epistemic"][0, i]), float(pred["aleatoric"][0, i])
                final = a * rule + (1 - a) * model_r
                if rule >= 70.0:  # safety floor: the AI may escalate, never silence a red/black range alert
                    final = max(final, rule)
                half = (1 - a) * float(pred["ci_high"][0, i] - pred["ci_low"][0, i]) / 2.0
                ci_lo, ci_hi = max(0.0, final - half), min(100.0, final + half)
            if final is not None:
                final = float(np.clip(final, 0.0, 100.0))
                final_risks[p.key] = final
            level = risk_level(final)
            params_out.append(ParameterAssessment(
                key=p.key, name=p.name_en, name_tr=p.name_tr, category=p.category, unit=p.unit, kind=p.kind,
                value=None if value is None else float(value), normal_range=p.normal_range_text(),
                risk_pct=None if final is None else round(final, 1), rule_risk_pct=None if rule is None else round(rule, 1),
                model_risk_pct=None if model_r is None else round(model_r, 1),
                ci_low=None if ci_lo is None else round(ci_lo, 1), ci_high=None if ci_hi is None else round(ci_hi, 1),
                epistemic_pct=None if epi is None else round(epi, 2), aleatoric_pct=None if alea is None else round(alea, 2),
                level=level, color=level_color(level), direction=risk_direction(p.key, value)))

        # ---- aggregates ----------------------------------------------------------------
        # Overall HEALTH risk = physiological parameters fused with vocal prosody risk (if voice provided).
        phys_overall_pct = health_overall_risk(final_risks)
        overall_pct = phys_overall_pct
        o_lo = o_hi = overall_pct
        if pred is not None:
            m, lo, hi = float(pred["risk"][0, -1]), float(pred["ci_low"][0, -1]), float(pred["ci_high"][0, -1])
            o_lo, o_hi = max(0.0, overall_pct - (1 - a) * (m - lo)), min(100.0, overall_pct + (1 - a) * (hi - m))

        voice_dict: Optional[Dict[str, Any]] = reading.voice if reading.voice else None
        if voice_dict and voice_dict.get("voice_risk_pct") is not None:
            v_risk = float(np.clip(voice_dict["voice_risk_pct"], 0.0, 100.0))
            fused = 0.78 * phys_overall_pct + 0.22 * v_risk
            if v_risk >= 75.0:
                fused = max(fused, 0.62 * v_risk + 0.38 * phys_overall_pct)
            shift = fused - overall_pct
            overall_pct = float(np.clip(fused, 0.0, 100.0))
            o_lo = float(np.clip(o_lo + shift, 0.0, 100.0))
            o_hi = float(np.clip(o_hi + shift, 0.0, 100.0))

        overall_level = risk_level(overall_pct)
        overall: Dict[str, Any] = {
            "risk_pct": round(overall_pct, 1),
            "physiological_risk_pct": round(phys_overall_pct, 1),
            "level": overall_level,
            "color": level_color(overall_level),
            "ci_low": round(o_lo, 1),
            "ci_high": round(o_hi, 1),
            "n_at_risk": sum(1 for p in params_out if p.level in RISKY_LEVELS and p.kind != "environment"),
        }
        if voice_dict and voice_dict.get("voice_risk_pct") is not None:
            v_r = float(voice_dict["voice_risk_pct"])
            overall["voice_risk_pct"] = round(v_r, 1)
            overall["voice_level"] = str(voice_dict.get("level") or risk_level(v_r))
            overall["voice_state"] = voice_dict.get("state_key")

        def _cat_card(cat: str, r: float) -> Dict[str, Any]:
            lv = risk_level(r)
            return {"risk_pct": round(r, 1), "level": lv, "color": level_color(lv), "label": CATEGORIES[cat]["en"],
                    "label_tr": CATEGORIES[cat]["tr"], "icon": CATEGORIES[cat]["icon"]}

        systems = {cat: _cat_card(cat, r) for cat, r in system_risks(final_risks, "health").items()}

        # ---- space environment (radiation, isolation, distance, gravity, habitat) ----------
        env_present = any(k in final_risks for k in ENV_KEYS)
        environment: Dict[str, Any] = {}
        if env_present:
            hz = environment_hazard_index(final_risks)
            hz_level = risk_level(hz)
            environment = {
                "hazard_index_pct": round(hz, 1), "level": hz_level, "color": level_color(hz_level),
                "themes": {cat: _cat_card(cat, r) for cat, r in system_risks(final_risks, "environment").items()},
                "n_hazards": sum(1 for p in params_out if p.level in RISKY_LEVELS and p.kind == "environment"),
                "derived": derived_environment(known),
            }
            if pred is not None:
                environment["uplift"] = self._environment_uplift(effective, reading, history, subj_vec, pred)
                environment["hazard_attention"] = {k: round(float(w), 3) for k, w in zip(ENV_KEYS, pred["hazard_weights"][0])}

        forecast = [round(float(x), 1) for x in pred["forecast"][0]] if pred is not None else []
        physics = self._physics_summary(pred, reading, gravity) if pred is not None else {}
        modality = ({m: round(float(w), 3) for m, w in zip(MODALITIES, pred["modality_attention"][0])} if pred is not None else {})
        drivers: List[Dict[str, Any]] = []
        if pred is not None:
            w = pred["trend_variable_weights"][0]
            for j in np.argsort(w)[::-1][:6]:
                drivers.append({"key": PARAM_KEYS[j], "name": PARAMETERS[j].name_en, "weight": round(float(w[j]), 3)})

        # ---- Gemini / offline reasoning + tasks -------------------------------------
        role = reading.user_role if reading.user_role in ("doctor", "non_doctor") else "non_doctor"
        at_risk = [{"key": p.key, "name": p.name, "value": p.value, "unit": p.unit, "risk_pct": p.risk_pct,
                    "level": p.level, "direction": p.direction, "normal_range": p.normal_range,
                    "kind": "environment" if p.kind == "environment" else "health"}
                   for p in sorted(params_out, key=lambda x: -(x.risk_pct or 0)) if p.level in RISKY_LEVELS]
        context = {"astronaut_id": astronaut_id, "overall_health_risk_pct": overall["risk_pct"],
                   "overall_risk_pct": overall["risk_pct"], "at_risk": at_risk,
                   "symptoms_in_log": symptoms, "subjective_log": reading.subjective_log[:600],
                   "mission_day": reading.mission_day, "gravity_fraction": gravity, "forecast_7d_risk_pct": forecast,
                   "space_physiology_projection": physics.get("projections", {}), "ecg": _ecg_brief(ecg_dict),
                   "voice": _voice_brief(voice_dict), "environment": _env_brief(environment, known)}
        advice, source = self.advisor.reason(context, lang, role=role)
        new_tasks = self._create_tasks(astronaut_id, advice.tasks, params_out, source)
        if voice_dict:
            new_tasks.extend(self._create_voice_tasks(astronaut_id, voice_dict, lang, source))

        meta = getattr(self.model, "meta", {}) if self.model else {}
        assessment = Assessment(
            astronaut_id=astronaut_id, timestamp=utcnow(), model_version=MODEL_VERSION if self.model else "rules-only",
            model_trained_on=str(meta.get("data", "n/a")), overall=overall, systems=systems, parameters=params_out,
            forecast=forecast, physics=physics, modality_attention=modality, trend_drivers=drivers, ecg=ecg_dict,
            voice=voice_dict, environment=environment,
            advisor={"summary": advice.summary, "recommendations": advice.recommendations, "source": source,
                     "user_role": role, "escalate": advice.escalate_to_flight_surgeon, "gemini": self.advisor.status()},
            new_tasks=new_tasks, subjective_symptoms=symptoms)
        if reading.save:
            stored_assessment = assessment.model_dump(mode="json")
            stored_assessment["mission_day"] = float(reading.mission_day)
            stored_assessment["subjective_log"] = reading.subjective_log[:300] if reading.subjective_log else ""
            self.storage.add_reading(astronaut_id, {k: float(v) for k, v in known.items()}, stored_assessment)
        return assessment

    # ----------------------------------------------------------------- physics
    @staticmethod
    def _physics_summary(pred: Dict[str, Any], reading: ReadingIn, gravity: float) -> Dict[str, Any]:
        z0 = pred["physics_z0"][0]
        proj = pred["physics_projection"][0]  # (181, 4) starting from the current state

        def pack(z: np.ndarray) -> Dict[str, float]:
            return {"fluid_shift_L": round(float(z[0]), 2), "plasma_volume_change_pct": round(float((z[1] - 1) * 100), 1),
                    "bone_density_change_pct": round(float((z[2] - 1) * 100), 2),
                    "muscle_mass_change_pct": round(float((z[3] - 1) * 100), 2)}

        curve = [{"day": reading.mission_day + d, **pack(proj[d])} for d in range(0, PROJECTION_DAYS + 1, 6)]
        return {"state_names": STATE_NAMES, "current": pack(z0),
                "projections": {f"+{d}d": pack(proj[d]) for d in (30, 90, 180)}, "curve": curve,
                "gravity": gravity, "mission_day": reading.mission_day}

    # -------------------------------------------------------------------- tasks
    def _create_tasks(self, astronaut_id: str, drafts: List[TaskDraft], params: List[ParameterAssessment], source: str) -> List[TaskOut]:
        by_key = {p.key: p for p in params}
        pending = self.storage.list_tasks(astronaut_id, "pending")
        pending_titles = {t["title"].strip().lower() for t in pending}
        worst_pending: Dict[str, int] = {}
        for t in pending:
            worst_pending[t["parameter_key"]] = max(worst_pending.get(t["parameter_key"], -1), LEVEL_ORDER[t["severity_level"]])
        created: List[TaskOut] = []
        for d in drafts:
            p = by_key.get(d.parameter_key)
            if p is None or p.level not in RISKY_LEVELS:
                continue
            # Don't re-create tasks for a parameter that already has open ones, unless it got worse.
            if worst_pending.get(p.key, -1) >= LEVEL_ORDER[p.level]:
                continue
            if d.title.strip().lower() in pending_titles:
                continue
            if len(created) >= MAX_NEW_TASKS_PER_ASSESSMENT:
                break
            row = {"parameter_key": p.key, "title": d.title, "description": d.instructions, "severity_level": p.level,
                   "risk_pct": float(p.risk_pct or 0), "duration_min": d.duration_min,
                   "points": task_points(p.level, d.duration_min), "tag": d.tag, "source": source}
            tid = self.storage.add_task(astronaut_id, row)
            pending_titles.add(d.title.strip().lower())
            created.append(TaskOut(**self.storage.get_task(tid)))
        return created

    def _create_voice_tasks(self, astronaut_id: str, voice: Dict[str, Any], lang: str, source: str) -> List[TaskOut]:
        v_risk = float(voice.get("voice_risk_pct") or 0.0)
        v_lvl = str(voice.get("level") or risk_level(v_risk))
        if v_lvl not in RISKY_LEVELS:
            return []
        pending = self.storage.list_tasks(astronaut_id, "pending")
        pending_titles = {t["title"].strip().lower() for t in pending}
        worst_voice = max((LEVEL_ORDER.get(t["severity_level"], -1) for t in pending if t["parameter_key"] == "voice_prosody"), default=-1)
        if worst_voice >= LEVEL_ORDER.get(v_lvl, 0):
            return []

        state_key = str(voice.get("state_key") or "acute_stress")
        templates_tr = {
            "cognitive_fatigue": ("20 dk kontrollü dinlenme ve vokal yeniden kontrol", "Bilişsel yükü düşürmek için loş kabinde 20 dakika gözleri dinlendirin, 300 ml su için ve ses kaydını tekrarlayın.", 20, "sleep"),
            "acute_stress": ("12 dk rezonans solunumu ve vokal stres regülasyonu", "Dakikada 6 nefes (4 sn al, 6 sn ver) protokolüyle sempatik tonusu düşürün, ardından vokal kontrol kaydı alın.", 12, "breathing"),
            "low_mood_isolation": ("15 dk yapılandırılmış sosyal bağlantı ve günlük kaydı", "Mürettebat arkadaşı veya yer destek ekibiyle kısa sesli/görüntülü temas kurun ve ruh hali notu ekleyin.", 15, "wellbeing"),
            "dyspnea_hypoxic": ("Solunum kontrolü, SpO2 ve kabin CO2 doğrulaması", "Oturur pozisyona geçin, nabız oksimetresi ile SpO2 ölçün, kabin CO2 seviyesini kontrol edin ve uçuş cerrahına bildirin.", 10, "medical"),
        }
        templates_en = {
            "cognitive_fatigue": ("20 min cognitive rest & vocal re-check", "Rest in low cabin lighting for 20 minutes, drink 300 ml water, and record a follow-up voice check.", 20, "sleep"),
            "acute_stress": ("12 min paced resonance breathing & voice re-check", "Perform 6 breaths/min paced breathing (4 s inhale, 6 s exhale) to lower vocal tension, then re-record.", 12, "breathing"),
            "low_mood_isolation": ("15 min crew/ground connection & wellbeing log", "Hold a brief supportive check-in with a crewmate or ground support and log your mood.", 15, "wellbeing"),
            "dyspnea_hypoxic": ("Verify SpO2, cabin CO2 & contact flight surgeon", "Sit upright, check pulse oximetry SpO2 and cabin pCO2 immediately, and notify the flight surgeon.", 10, "medical"),
        }
        table = templates_tr if lang == "tr" else templates_en
        title, desc, mins, tag = table.get(state_key, table["acute_stress"])
        if title.strip().lower() in pending_titles:
            return []
        row = {
            "parameter_key": "voice_prosody",
            "title": title,
            "description": desc,
            "severity_level": v_lvl,
            "risk_pct": round(v_risk, 1),
            "duration_min": mins,
            "points": task_points(v_lvl, mins),
            "tag": tag,
            "source": source,
        }
        tid = self.storage.add_task(astronaut_id, row)
        return [TaskOut(**self.storage.get_task(tid))]

    def list_tasks(self, astronaut_id: str, status: Optional[str] = None, lang: str = "en") -> List[TaskOut]:
        rows = self.storage.list_tasks(astronaut_id, status)
        if lang != "tr":
            out_rows: List[Dict[str, Any]] = []
            for r in rows:
                if is_turkish_text(r.get("title", "")) or is_turkish_text(r.get("description", "")):
                    r = self._translate_task_row_to_en(r)
                out_rows.append(r)
            rows = out_rows
        return [TaskOut(**t) for t in rows]

    def _translate_task_row_to_en(self, row: Dict[str, Any]) -> Dict[str, Any]:
        key = row.get("parameter_key", "")
        tag = row.get("tag", "")
        if key == "voice_prosody":
            voice_by_tag = {
                "sleep": ("20 min cognitive rest & vocal re-check", "Rest in low cabin lighting for 20 minutes, drink 300 ml water, and record a follow-up voice check.", "sleep"),
                "breathing": ("12 min paced resonance breathing & voice re-check", "Perform 6 breaths/min paced breathing (4 s inhale, 6 s exhale) to lower vocal tension, then re-record.", "breathing"),
                "wellbeing": ("15 min crew/ground connection & wellbeing log", "Hold a brief supportive check-in with a crewmate or ground support and log your mood.", "wellbeing"),
                "medical": ("Verify SpO2, cabin CO2 & contact flight surgeon", "Sit upright, check pulse oximetry SpO2 and cabin pCO2 immediately, and notify the flight surgeon.", "medical"),
            }
            title, desc, new_tag = voice_by_tag.get(tag, voice_by_tag["breathing"])
        else:
            candidates = [*templates_for(key, "high"), *templates_for(key, "low")]
            matched = next((c for c in candidates if c[3] == tag), candidates[0])
            title, desc, _, new_tag = matched
        updated = {**row, "title": title, "description": desc, "tag": new_tag}
        if row.get("id") is not None:
            self.storage.update_task_text(int(row["id"]), title, desc, new_tag)
        return updated

    def complete_task(self, astronaut_id: str, task_id: int) -> CompletionResult:
        return self.gamification.complete_task(astronaut_id, task_id)

    def gamification_state(self, astronaut_id: str) -> GamificationState:
        self.storage.ensure_astronaut(astronaut_id)
        return self.gamification.state(astronaut_id)

    # -------------------------------------------------------------------- misc
    def latest(self, astronaut_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.latest_assessment(astronaut_id)

    def risk_history(self, astronaut_id: str, n: int = 60) -> List[Dict[str, Any]]:
        return self.storage.risk_history(astronaut_id, n)

    def delete_history_entry(self, astronaut_id: str, reading_id: int) -> bool:
        return self.storage.delete_reading(astronaut_id, reading_id)

    def clear_history(self, astronaut_id: str) -> int:
        return self.storage.clear_readings(astronaut_id)

    def chat(self, astronaut_id: str, question: str, lang: str = "en", role: str = "non_doctor") -> Dict[str, str]:
        role_norm = "doctor" if role == "doctor" else "non_doctor"
        latest = self.latest(astronaut_id) or {}
        at_risk = [{"key": p["key"], "name": p["name"], "value": p["value"], "unit": p.get("unit", ""),
                    "risk_pct": p["risk_pct"], "level": p["level"], "direction": p["direction"]}
                   for p in latest.get("parameters", []) if p["level"] in RISKY_LEVELS]
        env_raw = latest.get("environment") or {}
        env_ctx = {
            "hazard_index_pct": env_raw.get("hazard_index_pct", 0),
            "one_way_comm_delay_s": (env_raw.get("derived") or {}).get("comm_delay_s"),
            "environment_uplift_of_health_risk_pts": (env_raw.get("uplift") or {}).get("uplift_pts"),
        } if env_raw else None
        ctx = {"overall_risk_pct": latest.get("overall", {}).get("risk_pct", 0), "at_risk": at_risk,
               "symptoms_in_log": latest.get("subjective_symptoms", []), "forecast_7d_risk_pct": latest.get("forecast", []),
               "voice": _voice_brief(latest.get("voice")), "environment": env_ctx}
        answer, source = self.advisor.chat(question, ctx, lang, role=role_norm)
        return {"answer": answer, "source": source, "role": role_norm}

    # ECG ----------------------------------------------------------------------
    def ecg_status(self) -> Dict[str, Any]:
        return self.ecg.status()

    def analyze_ecg(self, content: bytes, filename: str) -> Dict[str, Any]:
        res = self.ecg.analyze_file(content, filename)
        return {**res.to_dict(), "parameter_values": res.parameter_values()}

    def ecg_sample(self, class_code: str = "NORMAL") -> Dict[str, Any]:
        res = self.ecg.sample(class_code)
        return {**res.to_dict(), "parameter_values": res.parameter_values()}

    # Voice --------------------------------------------------------------------
    def voice_status(self) -> Dict[str, Any]:
        return self.voice.status()

    def analyze_voice(self, content: bytes, filename: str, lang: str = "en", role: str = "non_doctor") -> Dict[str, Any]:
        return self.voice.analyze_bytes(content, filename, lang=lang, role=role)

    def voice_sample(self, state_name: str = "calm_regulated", sex: str = "female",
                     lang: str = "en", role: str = "non_doctor") -> Dict[str, Any]:
        return self.voice.sample(state_name=state_name, sex=sex, lang=lang, role=role)


# ------------------------------------------------------------------ module helpers
def _ecg_parameter_values(ecg: Dict[str, Any]) -> Dict[str, float]:
    """Parameters carried by an ECG result dict (supports the raw ECGResult dict form)."""
    if "parameter_values" in ecg and ecg["parameter_values"]:
        return {k: float(v) for k, v in ecg["parameter_values"].items() if k in PARAM_BY_KEY}
    out: Dict[str, float] = {}
    if ecg.get("abnormal_probability_pct") is not None:
        out["ecg_abnormal_pct"] = float(ecg["abnormal_probability_pct"])
    if ecg.get("qtc_ms"):
        out["ecg_qtc_ms"] = float(ecg["qtc_ms"])
    if ecg.get("heart_rate"):
        out["heart_rate"] = float(ecg["heart_rate"])
    return out


def _ecg_brief(ecg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not ecg:
        return None
    keys = ("class_code", "class_name", "confidence_pct", "heart_rate", "qtc_ms", "rr_regularity", "stemi_suspected", "source")
    return {k: ecg.get(k) for k in keys if k in ecg}


def _voice_brief(voice: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not voice:
        return None
    keys = ("state_key", "state_en", "state_tr", "voice_risk_pct", "ci_low", "ci_high", "level", "mood", "biomarkers", "source")
    return {k: voice.get(k) for k in keys if k in voice}


def _env_brief(environment: Dict[str, Any], known: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Compact environment summary for the advisor prompt (values, hazard levels, uplift)."""
    if not environment:
        return None
    return {
        "hazard_index_pct": environment["hazard_index_pct"],
        "themes": {cat: {"risk_pct": t["risk_pct"], "level": t["level"]} for cat, t in environment["themes"].items()},
        "values": {k: round(float(known[k]), 4) for k in ENV_KEYS if k in known},
        "one_way_comm_delay_s": environment["derived"].get("comm_delay_s"),
        "time_to_definitive_care_days": environment["derived"].get("time_to_care_days"),
        "environment_uplift_of_health_risk_pts": (environment.get("uplift") or {}).get("uplift_pts"),
        "most_affected_by_environment": [p["key"] for p in (environment.get("uplift") or {}).get("top_parameters", [])],
    }


_SERVICE: Optional[AstroGuardService] = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> AstroGuardService:
    """Process-wide singleton (lazy so imports stay cheap)."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = AstroGuardService()
        return _SERVICE
