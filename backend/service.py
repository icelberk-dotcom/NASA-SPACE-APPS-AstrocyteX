"""
AstroGuard-X service layer - the single place where all components meet.

    reading â”€â”€â–º feature builder â”€â”€â–º AstroGuardNet (Mamba-2 | TCN+TFT | fusion | ODE/PINN | Bayesian)
                                         â”‚
    reference ranges (transparent) â”€â”€â”€â”€â”€â”€â”¼â”€â”€â–º blended risk % + uncertainty â”€â”€â–º colour indicators
                                         â”‚
    ECG adapter (Teknofest V23) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                    â”‚
                                                              â–¼
              Gemini / offline reasoner â”€â”€â–º briefing + mitigation tasks â”€â”€â–º gamification

Both the FastAPI app and the Streamlit UI call this class, so behaviour is identical whether
the UI runs in-process or talks to the REST API.

How the displayed risk is computed
----------------------------------
For every parameter ``risk = a * rule + (1 - a) * model`` where ``rule`` is the transparent
reference-range score and ``model`` the Monte-Carlo mean of the neural network
(``a = ASTROGUARD_RULE_BLEND``, default 0.65).  The network can therefore *raise* concern for
interactions (poor sleep + high stress markers, symptoms in the log ...) but a clearly
abnormal value can never be hidden: if the rule score is already red/black (>= 70 %) the final
score is never below it (safety floor).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from backend.ecg_adapter import ECGAdapter, ECGResult
from backend.gamification import Gamification, task_points
from backend.gemini_agent import GeminiAdvisor
from backend.schemas import (Assessment, CompletionResult, GamificationState, ParameterAssessment,
                             ReadingIn, TaskDraft, TaskOut)
from backend.storage import Storage, utcnow
from models.astroguard import AstroGuardNet, MODEL_VERSION, load_checkpoint
from models.fusion import MODALITIES
from models.physics import STATE_NAMES
from utils.config import Settings, get_settings
from utils.features import build_sample, stack_samples
from utils.reference_ranges import (CATEGORIES, DEFAULT_VALUES, LEVEL_ORDER, PARAM_BY_KEY, PARAM_KEYS,
                                    PARAMETERS, RISKY_LEVELS, aggregate_risk, clip_to_bounds, level_color,
                                    risk_direction, risk_level, risk_percent, system_risks)
from utils.subjective import SubjectiveEncoder

log = logging.getLogger("astroguard.service")
MAX_NEW_TASKS_PER_ASSESSMENT = 10
PROJECTION_DAYS = 180


class AstroGuardService:
    def __init__(self, settings: Optional[Settings] = None, model: Optional[AstroGuardNet] = None,
                 load_model: bool = True, advisor: Optional[GeminiAdvisor] = None,
                 ecg: Optional[ECGAdapter] = None, storage: Optional[Storage] = None):
        self.settings = settings or get_settings()
        self.storage = storage or Storage(self.settings.db_path)
        self.gamification = Gamification(self.storage)
        self.advisor = advisor or GeminiAdvisor(self.settings)
        self.ecg = ecg or ECGAdapter(self.settings)
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
                "device": "cpu", "rule_blend": self.settings.rule_blend,
                "official_mamba_ssm": bool(self.model and self.model.cfg.prefer_official_mamba and torch.cuda.is_available())}

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

        effective, history = self._effective_values(astronaut_id, provided)
        known = {k: effective[k] for k in PARAM_KEYS if k in effective}  # parameters we have *any* data for

        subj_vec = self.encoder.encode(reading.subjective_log)
        symptoms = self.encoder.describe(subj_vec)

        # ---- neural prediction -----------------------------------------------------
        pred: Optional[Dict[str, Any]] = None
        if self.model is not None:
            sample = build_sample({**DEFAULT_VALUES, **effective}, reading.sensor_series, history, subj_vec,
                                  reading.mission_day, reading.gravity)
            batch = stack_samples([sample])
            with self._lock:
                pred = self.model.predict(batch, n_mc=self.settings.mc_samples, project_days=PROJECTION_DAYS)

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
        overall_pct = aggregate_risk(list(final_risks.values())) if final_risks else 0.0
        o_lo = o_hi = overall_pct
        if pred is not None:
            m, lo, hi = float(pred["risk"][0, -1]), float(pred["ci_low"][0, -1]), float(pred["ci_high"][0, -1])
            o_lo, o_hi = max(0.0, overall_pct - (1 - a) * (m - lo)), min(100.0, overall_pct + (1 - a) * (hi - m))
        overall_level = risk_level(overall_pct)
        overall = {"risk_pct": round(overall_pct, 1), "level": overall_level, "color": level_color(overall_level),
                   "ci_low": round(o_lo, 1), "ci_high": round(o_hi, 1),
                   "n_at_risk": sum(1 for p in params_out if p.level in RISKY_LEVELS)}
        systems = {}
        for cat, r in system_risks(final_risks).items():
            lv = risk_level(r)
            systems[cat] = {"risk_pct": round(r, 1), "level": lv, "color": level_color(lv),
                            "label": CATEGORIES[cat]["en"], "label_tr": CATEGORIES[cat]["tr"], "icon": CATEGORIES[cat]["icon"]}

        forecast = [round(float(x), 1) for x in pred["forecast"][0]] if pred is not None else []
        physics = self._physics_summary(pred, reading) if pred is not None else {}
        modality = ({m: round(float(w), 3) for m, w in zip(MODALITIES, pred["modality_attention"][0])} if pred is not None else {})
        drivers: List[Dict[str, Any]] = []
        if pred is not None:
            w = pred["trend_variable_weights"][0]
            for j in np.argsort(w)[::-1][:6]:
                drivers.append({"key": PARAM_KEYS[j], "name": PARAMETERS[j].name_en, "weight": round(float(w[j]), 3)})

        # ---- Gemini / offline reasoning + tasks -------------------------------------
        at_risk = [{"key": p.key, "name": p.name, "value": p.value, "unit": p.unit, "risk_pct": p.risk_pct,
                    "level": p.level, "direction": p.direction, "normal_range": p.normal_range}
                   for p in sorted(params_out, key=lambda x: -(x.risk_pct or 0)) if p.level in RISKY_LEVELS]
        context = {"astronaut_id": astronaut_id, "overall_risk_pct": overall["risk_pct"], "at_risk": at_risk,
                   "symptoms_in_log": symptoms, "subjective_log": reading.subjective_log[:600],
                   "mission_day": reading.mission_day, "gravity_fraction": reading.gravity, "forecast_7d_risk_pct": forecast,
                   "space_physiology_projection": physics.get("projections", {}), "ecg": _ecg_brief(ecg_dict)}
        advice, source = self.advisor.reason(context, lang)
        new_tasks = self._create_tasks(astronaut_id, advice.tasks, params_out, source)

        meta = getattr(self.model, "meta", {}) if self.model else {}
        assessment = Assessment(
            astronaut_id=astronaut_id, timestamp=utcnow(), model_version=MODEL_VERSION if self.model else "rules-only",
            model_trained_on=str(meta.get("data", "n/a")), overall=overall, systems=systems, parameters=params_out,
            forecast=forecast, physics=physics, modality_attention=modality, trend_drivers=drivers, ecg=ecg_dict,
            advisor={"summary": advice.summary, "recommendations": advice.recommendations, "source": source,
                     "escalate": advice.escalate_to_flight_surgeon, "gemini": self.advisor.status()},
            new_tasks=new_tasks, subjective_symptoms=symptoms)
        if reading.save:
            self.storage.add_reading(astronaut_id, {k: float(v) for k, v in known.items()}, assessment.model_dump(mode="json"))
        return assessment

    # ----------------------------------------------------------------- physics
    @staticmethod
    def _physics_summary(pred: Dict[str, Any], reading: ReadingIn) -> Dict[str, Any]:
        z0 = pred["physics_z0"][0]
        proj = pred["physics_projection"][0]  # (181, 4) starting from the current state

        def pack(z: np.ndarray) -> Dict[str, float]:
            return {"fluid_shift_L": round(float(z[0]), 2), "plasma_volume_change_pct": round(float((z[1] - 1) * 100), 1),
                    "bone_density_change_pct": round(float((z[2] - 1) * 100), 2),
                    "muscle_mass_change_pct": round(float((z[3] - 1) * 100), 2)}

        curve = [{"day": reading.mission_day + d, **pack(proj[d])} for d in range(0, PROJECTION_DAYS + 1, 6)]
        return {"state_names": STATE_NAMES, "current": pack(z0),
                "projections": {f"+{d}d": pack(proj[d]) for d in (30, 90, 180)}, "curve": curve,
                "gravity": reading.gravity, "mission_day": reading.mission_day}

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

    def list_tasks(self, astronaut_id: str, status: Optional[str] = None) -> List[TaskOut]:
        return [TaskOut(**t) for t in self.storage.list_tasks(astronaut_id, status)]

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

    def chat(self, astronaut_id: str, question: str, lang: str = "en") -> Dict[str, str]:
        latest = self.latest(astronaut_id) or {}
        at_risk = [{"key": p["key"], "name": p["name"], "value": p["value"], "risk_pct": p["risk_pct"], "level": p["level"],
                    "direction": p["direction"]} for p in latest.get("parameters", []) if p["level"] in RISKY_LEVELS]
        ctx = {"overall_risk_pct": latest.get("overall", {}).get("risk_pct", 0), "at_risk": at_risk,
               "symptoms_in_log": latest.get("subjective_symptoms", []), "forecast_7d_risk_pct": latest.get("forecast", [])}
        answer, source = self.advisor.chat(question, ctx, lang)
        return {"answer": answer, "source": source}

    # ECG ----------------------------------------------------------------------
    def ecg_status(self) -> Dict[str, Any]:
        return self.ecg.status()

    def analyze_ecg(self, content: bytes, filename: str) -> Dict[str, Any]:
        res = self.ecg.analyze_file(content, filename)
        return {**res.to_dict(), "parameter_values": res.parameter_values()}

    def ecg_sample(self, class_code: str = "NORMAL") -> Dict[str, Any]:
        res = self.ecg.sample(class_code)
        return {**res.to_dict(), "parameter_values": res.parameter_values()}


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


_SERVICE: Optional[AstroGuardService] = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> AstroGuardService:
    """Process-wide singleton (lazy so imports stay cheap)."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = AstroGuardService()
        return _SERVICE
