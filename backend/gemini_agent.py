"""
Gemini-based reasoning for contextual medical recommendations and task generation.

Uses the official **Google GenAI SDK** (``google-genai``, Apache-2.0)::

    from google import genai
    client = genai.Client(api_key=...)
    client.models.generate_content(model="gemini-3.6-flash", contents=..., config=...)

Design goals
------------
* **Structured output** - the model must answer with JSON matching :class:`AdvisorOutput`
  (``response_schema``), and every field is validated again before it is used.
* **Edge-first** - a deep-space crew cannot rely on the internet.  If there is no API key,
  no network, a timeout or an invalid answer, the deterministic *offline reasoner*
  (``LocalReasoner``) produces an equivalent briefing + tasks from the on-board task
  library.  A short circuit-breaker avoids repeated timeouts.
* **Safety** - the system prompt forbids drug dosing and diagnoses; anything serious is
  escalated to the flight surgeon.

The text of the briefing is written in the language requested by the UI (English / Turkish).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from backend.schemas import AdvisorOutput, TaskDraft
from backend.task_library import ALLOWED_TAGS, CRITICAL_TASK, ENV_CRITICAL_TASK, normalize_tag, templates_for
from utils.config import Settings, get_settings
from utils.reference_ranges import CATEGORIES, PARAM_BY_KEY, RISKY_LEVELS

log = logging.getLogger("astroguard.gemini")

SYSTEM_PROMPT = """You are AstroGuard-X, an on-board AI health assistant for astronauts on long-duration missions.
You receive risk scores (0-100 %) produced by an uncertainty-aware multimodal model together with raw values,
normal ranges, the astronaut's subjective log, space-physiology projections and the SPACE ENVIRONMENT:
NASA's five hazards (RIDGE) - Radiation, Isolation & confinement, Distance from Earth, Gravity fields and the
hostile/closed habitat (cabin CO2, O2, temperature, humidity, noise). Items with kind='environment' are hazards
of the surroundings, not of the body; 'environment_uplift_of_health_risk_pts' says how many points of the health
risk exist only because of the current environment.

Rules:
- Give short, calm, practical guidance. Never diagnose and NEVER recommend drug names or doses.
- Mitigation tasks must be concrete, safe, doable on a spacecraft in <= 30 minutes (e.g. 'Drink 500 ml electrolyte water',
  'Perform 15 min ARED resistance exercise', 'Take a 20-minute guided breathing session', 'Move to the radiation storm
  shelter', 'Check the CO2 scrubber and increase ventilation', 'Re-measure and log').
- Create 1-2 tasks per at-risk parameter (use parameter_key exactly as given). For physiological parameters at level
  'black' include a task to contact the flight surgeon immediately; for environment hazards at 'black' include a task to
  alert mission control / start the hazard emergency procedure.
- Explain how the environment changes the picture (e.g. a long communication delay or distance from Earth means the
  crew must act autonomously; high CO2 explains headache and fast breathing; a recent landing explains dizziness).
- "tag" must be exactly one of: hydration, exercise, breathing, sleep, nutrition, wellbeing, ecg, medical, monitoring,
  environment, radiation, cooling, recovery, hygiene, general (English keywords, even when the text is Turkish).
- Consider interactions (e.g. dehydration + high heart rate + dizziness) and avoid redundant tasks.
- Set escalate_to_flight_surgeon=true when any physiological parameter is black or several are red.
- Reply in the requested language. Output JSON only, matching the schema."""


def _is_env(item: Dict[str, Any]) -> bool:
    return item.get("kind") == "environment" or PARAM_BY_KEY.get(item["key"]) is not None and PARAM_BY_KEY[item["key"]].kind == "environment"


class LocalReasoner:
    """Deterministic offline (edge) reasoner built on the task library."""

    @staticmethod
    def reason(context: Dict[str, Any], lang: str = "en") -> AdvisorOutput:
        risky: List[Dict[str, Any]] = context.get("at_risk", [])
        tasks: List[TaskDraft] = []
        for item in risky:
            key, level, direction = item["key"], item["level"], item["direction"]
            templates = templates_for(key, direction)
            # yellow: 1 task; red: 2; black: 1 + the critical escalation task (avoids duplicate escalations)
            n = 2 if level == "red" else 1
            for title, text, minutes, tag in templates[:n]:
                tasks.append(TaskDraft(parameter_key=key, title=title, instructions=text, duration_min=minutes, tag=tag))
            if level == "black":
                title, text, minutes, tag = ENV_CRITICAL_TASK if _is_env(item) else CRITICAL_TASK
                tasks.append(TaskDraft(parameter_key=key, title=title, instructions=text, duration_min=minutes, tag=tag))

        health = [i for i in risky if not _is_env(i)]
        hazards = [i for i in risky if _is_env(i)]
        n_black = sum(1 for i in health if i["level"] == "black")
        n_red = sum(1 for i in health if i["level"] == "red")
        escalate = n_black > 0 or n_red >= 3
        overall = context.get("overall_risk_pct", 0.0)
        env = context.get("environment") or {}
        env_sentence = ""
        if env and (env.get("hazard_index_pct", 0) >= 15 or (env.get("environment_uplift_of_health_risk_pts") or 0) >= 1):
            uplift = env.get("environment_uplift_of_health_risk_pts")
            themes = sorted(env.get("themes", {}).items(), key=lambda kv: -kv[1]["risk_pct"])
            top = ", ".join(CATEGORIES[c]["tr" if lang == "tr" else "en"] for c, t in themes[:2] if t["risk_pct"] >= 40)
            if lang == "tr":
                env_sentence = f" Uzay ortamı tehlike indeksi %{env.get('hazard_index_pct', 0):.0f}" + (f" (öne çıkan: {top})" if top else "") + "."
                if uplift is not None and uplift >= 1:
                    env_sentence += f" Mevcut ortam sağlık riskinizi yaklaşık +{uplift:.0f} puan artırıyor."
                if (env.get("one_way_comm_delay_s") or 0) > 60:
                    env_sentence += f" Tek yönlü iletişim gecikmesi ~{env['one_way_comm_delay_s'] / 60:.0f} dk: özerk hareket edin."
            else:
                env_sentence = f" Space-environment hazard index is {env.get('hazard_index_pct', 0):.0f}%" + (f" (mainly {top})" if top else "") + "."
                if uplift is not None and uplift >= 1:
                    env_sentence += f" The current environment adds about +{uplift:.0f} points to your health risk."
                if (env.get("one_way_comm_delay_s") or 0) > 60:
                    env_sentence += f" One-way communication delay is ~{env['one_way_comm_delay_s'] / 60:.0f} min: act autonomously."

        if not risky:
            summary = ("All monitored parameters are within acceptable limits. Keep up your exercise, hydration and sleep routine."
                       if lang != "tr" else
                       "İzlenen tüm parametreler kabul edilebilir sınırlar içinde. Egzersiz, sıvı ve uyku düzeninizi sürdürün.") + env_sentence
            recs = ["Maintain your daily ARED / treadmill session.", "Keep fluid intake at ~2.5 L/day.", "Log how you feel each evening."]
            if lang == "tr":
                recs = ["Günlük ARED / koşu bandı seansınızı sürdürün.", "Günlük sıvı alımını ~2,5 L'de tutun.", "Her akşam kendinizi nasıl hissettiğinizi kaydedin."]
        else:
            names = ", ".join(PARAM_BY_KEY[i["key"]].name(lang) for i in health[:4]) or "-"
            hz_names = ", ".join(PARAM_BY_KEY[i["key"]].name(lang) for i in hazards[:3])
            if lang == "tr":
                summary = (f"Genel sağlık riski %{overall:.0f}. Dikkat gerektiren parametreler: {names}."
                           + (f" Ortam tehlikeleri: {hz_names}." if hz_names else "") + env_sentence
                           + f" {len(tasks)} hafifletme görevi oluşturuldu; tamamlayarak puan kazanın. "
                           + ("Acil olarak uçuş cerrahına başvurun." if escalate else "Değerler kötüleşirse uçuş cerrahına bildirin."))
            else:
                summary = (f"Overall health risk is {overall:.0f}%. Parameters needing attention: {names}."
                           + (f" Environment hazards: {hz_names}." if hz_names else "") + env_sentence
                           + f" {len(tasks)} mitigation tasks were generated - complete them to earn points. "
                           + ("Contact the flight surgeon immediately." if escalate else "Notify the flight surgeon if values worsen."))
            recs = []
            seen = set()
            for t in tasks:
                if t.title not in seen:
                    seen.add(t.title)
                    recs.append(t.title)
                if len(recs) == 4:
                    break
        return AdvisorOutput(summary=summary, recommendations=recs, tasks=tasks, escalate_to_flight_surgeon=escalate)

    @staticmethod
    def answer(question: str, context: Dict[str, Any], lang: str = "en") -> str:
        risky = context.get("at_risk", [])
        if not risky:
            return ("Everything looks nominal right now. I can't reach Gemini, so this answer comes from the on-board rules."
                    if lang != "tr" else "Şu an her şey normal görünüyor. Gemini'ye ulaşamıyorum; bu yanıt yerel kurallardan geliyor.")
        parts = [f"{PARAM_BY_KEY[i['key']].name(lang)}: {i['value']} ({i['risk_pct']:.0f}%, {i['level']})" for i in risky[:5]]
        head = "Offline advisor (Gemini unavailable). Current concerns: " if lang != "tr" else "Çevrimdışı danışman (Gemini yok). Mevcut sorunlar: "
        return head + "; ".join(parts) + "."


class GeminiAdvisor:
    """Gemini-first advisor with an offline fallback."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        self._types = None
        self._blocked_until = 0.0
        self.last_error: Optional[str] = None
        self.active_model: Optional[str] = None
        if self.settings.gemini_api_key:
            try:
                from google import genai
                from google.genai import types

                self._types = types
                self._client = genai.Client(
                    api_key=self.settings.gemini_api_key,
                    http_options=types.HttpOptions(timeout=int(self.settings.gemini_timeout_s * 1000)),
                )
            except Exception as exc:  # SDK missing or bad config
                self.last_error = f"Gemini init failed: {exc}"
                log.warning(self.last_error)

    # ------------------------------------------------------------------ status
    @property
    def online(self) -> bool:
        return self._client is not None and time.time() >= self._blocked_until

    def status(self) -> Dict[str, Any]:
        return {"configured": self._client is not None, "model": self.active_model or self.settings.gemini_model,
                "online": self.online, "last_error": self.last_error}

    def _trip(self, exc: Exception) -> None:
        self.last_error = str(exc)[:300]
        self._blocked_until = time.time() + 60  # circuit breaker: 1 minute offline
        log.warning("Gemini call failed, falling back to the offline reasoner: %s", self.last_error)

    def _generate(self, prompt: str, cfg: Any):
        """Call the primary model, then each fallback model if it is overloaded (503), slow (504) or retired (404)."""
        last: Optional[Exception] = None
        for model in [self.settings.gemini_model, *self.settings.gemini_fallback_models]:
            try:
                resp = self._client.models.generate_content(model=model, contents=prompt, config=cfg)
                self.active_model = model
                return resp
            except Exception as exc:
                last = exc
                log.warning("Gemini model %s failed (%s) - trying next", model, str(exc)[:80])
        raise last  # type: ignore[misc]

    # ----------------------------------------------------------------- reasoning
    def reason(self, context: Dict[str, Any], lang: str = "en") -> tuple[AdvisorOutput, str]:
        """Return (advice, source) where source is 'gemini' or 'offline'."""
        if self.online:
            try:
                prompt = (f"Language: {'Turkish' if lang == 'tr' else 'English'}\n"
                          f"Astronaut context (JSON):\n{json.dumps(context, ensure_ascii=False, default=float)}")
                cfg = self._types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, temperature=0.3, response_mime_type="application/json",
                    response_schema=AdvisorOutput)
                resp = self._generate(prompt, cfg)
                out = resp.parsed if isinstance(getattr(resp, "parsed", None), AdvisorOutput) else AdvisorOutput.model_validate_json(resp.text)
                return self._sanitize(out, context), "gemini"
            except Exception as exc:
                self._trip(exc)
        return LocalReasoner.reason(context, lang), "offline"

    def chat(self, question: str, context: Dict[str, Any], lang: str = "en") -> tuple[str, str]:
        if self.online:
            try:
                prompt = (f"Language: {'Turkish' if lang == 'tr' else 'English'}\nContext (JSON):\n"
                          f"{json.dumps(context, ensure_ascii=False, default=float)}\n\nCrew question: {question}\n"
                          "Answer in at most 6 sentences. No drug names or doses.")
                cfg = self._types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT.split("Output JSON only")[0],
                                                        temperature=0.4)
                resp = self._generate(prompt, cfg)
                return (resp.text or "").strip(), "gemini"
            except Exception as exc:
                self._trip(exc)
        return LocalReasoner.answer(question, context, lang), "offline"

    # -------------------------------------------------------------------- safety
    @staticmethod
    def _sanitize(out: AdvisorOutput, context: Dict[str, Any]) -> AdvisorOutput:
        """Keep only tasks that reference at-risk parameters, with sane durations and lengths."""
        allowed = {i["key"] for i in context.get("at_risk", [])}
        clean: List[TaskDraft] = []
        seen = set()
        for t in out.tasks:
            if t.parameter_key not in allowed or t.title.strip().lower() in seen:
                continue
            seen.add(t.title.strip().lower())
            clean.append(TaskDraft(parameter_key=t.parameter_key, title=t.title.strip()[:120],
                                   instructions=t.instructions.strip()[:600],
                                   duration_min=int(min(max(t.duration_min, 1), 60)), tag=normalize_tag(t.tag)))
        # Make sure every at-risk parameter has at least one task (fill gaps from the library).
        covered = {t.parameter_key for t in clean}
        for item in context.get("at_risk", []):
            if item["key"] not in covered:
                title, text, minutes, tag = templates_for(item["key"], item["direction"])[0]
                clean.append(TaskDraft(parameter_key=item["key"], title=title, instructions=text, duration_min=minutes, tag=tag))
        return AdvisorOutput(summary=out.summary[:1200], recommendations=[r[:200] for r in out.recommendations[:4]],
                             tasks=clean, escalate_to_flight_surgeon=out.escalate_to_flight_surgeon
                             or any(i["level"] == "black" and not _is_env(i) for i in context.get("at_risk", [])))
