"""
Gemini-based reasoning for contextual medical recommendations, dual-role clinical/crew
communication ('doctor' vs 'non_doctor'), and voice-aware task generation.

Uses the official **Google GenAI SDK** (``google-genai``, Apache-2.0)::

    from google import genai
    client = genai.Client(api_key=...)
    client.models.generate_content(model="gemini-3.6-flash", contents=..., config=...)

Design goals
------------
* **Structured output** - the model must answer with JSON matching :class:`AdvisorOutput`
  (``response_schema``), and every field is validated again before it is used.
* **Role-adaptive register** - ``role="doctor"`` ("Ben doktorum") delivers flight-surgeon
  depth (aerospace pathophysiology, quantitative biomarker shifts, vocal prosody metrics,
  NASA-STD-3001 thresholds, serial monitoring plans), whereas ``role="non_doctor"``
  ("Doktor degilim") uses calm, everyday, jargon-free language for crew members.
* **Edge-first** - a deep-space crew cannot rely on the internet. If there is no API key,
  no network, a timeout or an invalid answer, the deterministic *offline reasoner*
  (``LocalReasoner``) produces a role-adapted briefing + tasks from the on-board task
  library.
* **Safety & zero-emoji policy** - never diagnose definitively, never prescribe drug names
  or doses, escalate critical findings to the flight surgeon, and never output emojis.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from backend.schemas import AdvisorOutput, TaskDraft
from backend.task_library import ALLOWED_TAGS, CRITICAL_TASK, ENV_CRITICAL_TASK, normalize_tag, templates_for
from utils.config import Settings, get_settings
from utils.reference_ranges import CATEGORIES, PARAM_BY_KEY, RISKY_LEVELS

log = logging.getLogger("astrocytex.gemini")

_TR_CHARS_RE = re.compile(r"[ğĞşŞıİçÇöÖüÜ]")
_TR_WORDS_RE = re.compile(
    r"\b(?:egzersiz|egzersizi|nefes|dinlenme|kontrolu|olcun|yapin|yapiniz|uyku|izolasyon|vucut|seansi)\b",
    re.IGNORECASE,
)


def is_turkish_text(text: str) -> bool:
    s = text or ""
    return bool(_TR_CHARS_RE.search(s) or _TR_WORDS_RE.search(s))

SYSTEM_PROMPT = """You are AstrocyteX, an on-board aerospace medical AI assistant for astronauts and flight surgeons on long-duration space missions.
You receive risk scores (0-100 %) produced by an uncertainty-aware multimodal neural network (AstrocyteX-Net v3.0) and vocal prosody analyzer (AstroVoiceNet v1.0), together with raw values, normal ranges, the astronaut's subjective log, space-physiology projections, vocal acoustic biomarkers, and the SPACE ENVIRONMENT:
NASA's five hazards (RIDGE) - Radiation, Isolation & confinement, Distance from Earth, Gravity fields and the hostile/closed habitat (cabin CO2, O2, temperature, humidity, noise). Items with kind='environment' are hazards of the surroundings, not of the body; 'environment_uplift_of_health_risk_pts' says how many points of the health risk exist only because of the current environment.

Rules:
- NEVER use emojis, emoticons, or decorative symbols anywhere in your output.
- Never make a definitive clinical diagnosis and NEVER recommend specific pharmaceutical drug names or doses.
- Mitigation tasks must be concrete, safe, doable on a spacecraft in <= 30 minutes (e.g. 'Drink 500 ml electrolyte water',
  'Perform 15 min ARED resistance exercise', 'Take a 20-minute guided breathing session', 'Move to the radiation storm
  shelter', 'Check the CO2 scrubber and increase ventilation', 'Re-measure and log').
- Create 1-2 tasks per at-risk parameter (use parameter_key exactly as given). For physiological parameters at level
  'black' include a task to contact the flight surgeon immediately; for environment hazards at 'black' include a task to
  alert mission control / start the hazard emergency procedure.
- Explain how the space environment and vocal prosody findings interact with physiology.
- "tag" must be exactly one of: hydration, exercise, breathing, sleep, nutrition, wellbeing, ecg, medical, monitoring,
  environment, radiation, cooling, recovery, hygiene, general (English keywords, even when the text is Turkish).
- Consider cross-system interactions (e.g. dehydration + tachycardia + orthostatic dizziness, or hypercapnia + dyspneaprosody) and avoid redundant tasks.
- Set escalate_to_flight_surgeon=true when any physiological parameter is black or several are red.
- Reply strictly in the requested language. Output JSON only, matching the schema."""

ROLE_INSTRUCTIONS: Dict[str, str] = {
    "doctor": (
        "AUDIENCE ROLE: FLIGHT SURGEON / PHYSICIAN ('I am a doctor'). "
        "Use precise aerospace-medicine and clinical terminology: cite exact biomarker values, reference intervals, "
        "autonomic/HPA-axis dynamics, cephalad fluid shift, baroreflex sensitivity, cardiac repolarization (QTc), "
        "chemoreceptor CO2 drive, vocal acoustic biomarkers (F0 CV, jitter, shimmer, HNR, alpha ratio), NASA-STD-3001 "
        "operational limits, counterfactual environmental uplift, non-pharmacological differential considerations, "
        "and structured serial re-assessment intervals."
    ),
    "non_doctor": (
        "AUDIENCE ROLE: NON-PHYSICIAN CREW MEMBER ('I am not a doctor'). "
        "Use calm, clear, supportive everyday language without heavy medical jargon or complex Latin/clinical abbreviations. "
        "Explain what the body or cabin is experiencing in plain words (e.g. heart beating faster due to low fluids, "
        "feeling tired from cabin air or poor sleep, tension heard in voice tone) and give simple, practical next steps."
    ),
}


def _norm_role(role: Optional[str]) -> str:
    return "doctor" if role == "doctor" else "non_doctor"


def _is_env(item: Dict[str, Any]) -> bool:
    return item.get("kind") == "environment" or (
        PARAM_BY_KEY.get(item["key"]) is not None and PARAM_BY_KEY[item["key"]].kind == "environment"
    )


class LocalReasoner:
    """Deterministic offline (edge) reasoner built on the task library, supporting doctor & non_doctor modes."""

    @staticmethod
    def reason(context: Dict[str, Any], lang: str = "en", role: str = "non_doctor") -> AdvisorOutput:
        role = _norm_role(role)
        risky: List[Dict[str, Any]] = context.get("at_risk", [])
        tasks: List[TaskDraft] = []
        for item in risky:
            key, level, direction = item["key"], item["level"], item["direction"]
            if key not in PARAM_BY_KEY:
                continue
            templates = templates_for(key, direction)
            n = 2 if level == "red" else 1
            for title, text, minutes, tag in templates[:n]:
                tasks.append(TaskDraft(parameter_key=key, title=title, instructions=text, duration_min=minutes, tag=tag))
            if level == "black":
                title, text, minutes, tag = ENV_CRITICAL_TASK if _is_env(item) else CRITICAL_TASK
                tasks.append(TaskDraft(parameter_key=key, title=title, instructions=text, duration_min=minutes, tag=tag))

        health = [i for i in risky if not _is_env(i) and i["key"] in PARAM_BY_KEY]
        hazards = [i for i in risky if _is_env(i) and i["key"] in PARAM_BY_KEY]
        n_black = sum(1 for i in health if i["level"] == "black")
        n_red = sum(1 for i in health if i["level"] == "red")
        voice = context.get("voice") or {}
        voice_risk = float(voice.get("voice_risk_pct") or 0.0)
        voice_lvl = str(voice.get("level") or "gray")
        escalate = n_black > 0 or n_red >= 3 or voice_lvl == "black"
        overall = float(context.get("overall_risk_pct", 0.0))
        env = context.get("environment") or {}

        # Environment narrative adapted to role
        env_sentence = ""
        if env and (env.get("hazard_index_pct", 0) >= 15 or (env.get("environment_uplift_of_health_risk_pts") or 0) >= 1):
            uplift = env.get("environment_uplift_of_health_risk_pts")
            themes = sorted(env.get("themes", {}).items(), key=lambda kv: -kv[1]["risk_pct"])
            top = ", ".join(CATEGORIES[c]["tr" if lang == "tr" else "en"] for c, t in themes[:2] if t["risk_pct"] >= 40)
            if lang == "tr":
                if role == "doctor":
                    env_sentence = (
                        f" NASA-RIDGE maruziyet indeksi %{env.get('hazard_index_pct', 0):.0f}"
                        + (f" (baskın eksen: {top})" if top else "") + "."
                    )
                    if uplift is not None and uplift >= 1:
                        env_sentence += f" Karşı-olgusal ortam katkısı fizyolojik riski +{uplift:.1f} puan yükseltiyor."
                    if (env.get("one_way_comm_delay_s") or 0) > 60:
                        env_sentence += f" Tek yönlü telemetri gecikmesi {env['one_way_comm_delay_s'] / 60:.1f} dk; otonom klinik protokol öncelikli."
                else:
                    env_sentence = (
                        f" Kabin ve uzay ortamı yükü %{env.get('hazard_index_pct', 0):.0f}"
                        + (f" ({top} kaynaklı)" if top else "") + "."
                    )
                    if uplift is not None and uplift >= 1:
                        env_sentence += f" İçinde bulunduğunuz ortam vücut yorgunluğunu yaklaşık +{uplift:.0f} puan artırıyor."
                    if (env.get("one_way_comm_delay_s") or 0) > 60:
                        env_sentence += f" Dünya ile iletişim gecikmesi ~{env['one_way_comm_delay_s'] / 60:.0f} dk; adımları doğrudan uygulayın."
            else:
                if role == "doctor":
                    env_sentence = (
                        f" NASA-RIDGE environmental hazard index is {env.get('hazard_index_pct', 0):.0f}%"
                        + (f" (dominant vector: {top})" if top else "") + "."
                    )
                    if uplift is not None and uplift >= 1:
                        env_sentence += f" Counterfactual environmental uplift accounts for +{uplift:.1f} pts of physiological risk."
                    if (env.get("one_way_comm_delay_s") or 0) > 60:
                        env_sentence += f" One-way telemetry latency is {env['one_way_comm_delay_s'] / 60:.1f} min; autonomous crew medical protocol applies."
                else:
                    env_sentence = (
                        f" Cabin and space-environment load is {env.get('hazard_index_pct', 0):.0f}%"
                        + (f" (mainly {top})" if top else "") + "."
                    )
                    if uplift is not None and uplift >= 1:
                        env_sentence += f" Current surroundings add about +{uplift:.0f} points to your body stress."
                    if (env.get("one_way_comm_delay_s") or 0) > 60:
                        env_sentence += f" Earth communication delay is ~{env['one_way_comm_delay_s'] / 60:.0f} min: follow onboard steps directly."

        # Voice acoustic narrative adapted to role
        voice_sentence = ""
        if voice and voice_risk > 0:
            st_label = voice.get("state_tr" if lang == "tr" else "state_en") or voice.get("state_key", "")
            bm = voice.get("biomarkers") or {}
            if lang == "tr":
                if role == "doctor":
                    voice_sentence = (
                        f" Vokal akustik biyobelirteç analizi: {st_label} (vokal risk %{voice_risk:.1f}, "
                        f"F0={bm.get('f0_mean_hz', 0):.0f} Hz, Jitter=%{bm.get('jitter_pct', 0):.2f}, "
                        f"HNR={bm.get('hnr_db', 0):.1f} dB)."
                    )
                else:
                    voice_sentence = f" Ses analiziniz '{st_label}' görünümü veriyor (ses riski %{voice_risk:.0f})."
            else:
                if role == "doctor":
                    voice_sentence = (
                        f" Vocal prosodic telemetry indicates {st_label} (voice risk {voice_risk:.1f}%, "
                        f"F0={bm.get('f0_mean_hz', 0):.0f} Hz, jitter={bm.get('jitter_pct', 0):.2f}%, "
                        f"HNR={bm.get('hnr_db', 0):.1f} dB)."
                    )
                else:
                    voice_sentence = f" Your voice check points to '{st_label}' (voice risk {voice_risk:.0f}%)."

        if not risky and voice_lvl not in RISKY_LEVELS:
            if lang == "tr":
                if role == "doctor":
                    summary = (
                        f"Genel sağlık riski %{overall:.1f}; tüm kardiyorespiratuvar, metabolik ve hematolojik biyobelirteçler "
                        f"referans fizyolojik aralıkta seyrediyor." + env_sentence + voice_sentence
                    )
                    recs = [
                        "Mikroyerçekimi kas-iskelet dekondisyonunu önlemek için günlük ARED + koşu bandı protokolünü sürdürün.",
                        "İntravasküler plazma hacmini korumak için izotonik sıvı dengesini ~2,5 L/gün düzeyinde izleyin.",
                        "Sirkadiyen faz stabilitesi için 24 saatlik otonom HRV ve aktigrafi takibine devam edin.",
                    ]
                else:
                    summary = (
                        "İzlenen tüm sağlık değerleriniz normal sınırlar içinde görünüyor. Günlük egzersiz, su tüketimi ve uyku düzeninizi koruyun."
                        + env_sentence + voice_sentence
                    )
                    recs = [
                        "Günlük egzersiz seansınızı aksatmadan sürdürün.",
                        "Gün boyu düzenli aralıklarla yaklaşık 2,5 litre su için.",
                        "Akşamları uyku öncesi ekran ışığını azaltıp dinlenme notunuzu girin.",
                    ]
            else:
                if role == "doctor":
                    summary = (
                        f"Overall health risk is {overall:.1f}%; all cardiorespiratory, metabolic, and hematologic biomarkers "
                        f"remain within nominal physiological bounds." + env_sentence + voice_sentence
                    )
                    recs = [
                        "Maintain daily ARED resistance and treadmill loading to attenuate microgravity bone/muscle loss.",
                        "Sustain ~2.5 L/day fluid-electrolyte intake to stabilize plasma volume under cephalad shift.",
                        "Continue serial autonomic HRV and circadian actigraphy logging every 24 hours.",
                    ]
                else:
                    summary = (
                        "All monitored readings are within healthy limits right now. Keep up your daily exercise, water intake, and sleep routine."
                        + env_sentence + voice_sentence
                    )
                    recs = [
                        "Keep up your daily onboard exercise session.",
                        "Drink around 2.5 L of water steadily across the day.",
                        "Log how rested you feel each evening before sleep.",
                    ]
        else:
            names = ", ".join(PARAM_BY_KEY[i["key"]].name(lang) for i in health[:4]) or "-"
            hz_names = ", ".join(PARAM_BY_KEY[i["key"]].name(lang) for i in hazards[:3])
            if lang == "tr":
                if role == "doctor":
                    detail_metrics = ", ".join(
                        f"{PARAM_BY_KEY[i['key']].name('tr')}={i['value']} {i.get('unit', '')} (risk %{i['risk_pct']:.0f})"
                        for i in health[:3]
                    ) or names
                    summary = (
                        f"Genel sağlık riski %{overall:.1f} (kritik/yüksek sapma: {n_black} siyah, {n_red} kırmızı). "
                        f"Öne çıkan klinik sapmalar: {detail_metrics}."
                        + (f" Eşlik eden çevresel maruziyetler: {hz_names}." if hz_names else "")
                        + env_sentence + voice_sentence
                        + f" Toplam {len(tasks)} kanıta dayalı karşı-önlem protokolü tanımlandı. "
                        + ("Acil uçuş cerrahı konsültasyonu ve seri vital monitörizasyon endikedir."
                           if escalate else "15-30 dk içinde seri biyobelirteç doğrulaması ve trend izlemi önerilir.")
                    )
                else:
                    summary = (
                        f"Genel sağlık riski %{overall:.0f}. Yakından ilgilenmeniz gereken değerler: {names}."
                        + (f" Kabin/ortam uyarıları: {hz_names}." if hz_names else "")
                        + env_sentence + voice_sentence
                        + f" Rahatlamanız ve değerleri dengelemeniz için {len(tasks)} kolay görev hazırlandı. "
                        + ("Lütfen vakit kaybetmeden uçuş doktoruna haber verin."
                           if escalate else "Adımları uygulayıp biraz dinlendikten sonra ölçümü tekrarlayın.")
                    )
            else:
                if role == "doctor":
                    detail_metrics = ", ".join(
                        f"{PARAM_BY_KEY[i['key']].name('en')}={i['value']} {i.get('unit', '')} ({i['risk_pct']:.0f}% risk)"
                        for i in health[:3]
                    ) or names
                    summary = (
                        f"Overall health risk is {overall:.1f}% ({n_black} black, {n_red} red physiological deviations). "
                        f"Primary biomarker excursions: {detail_metrics}."
                        + (f" Concurrent habitat/RIDGE hazards: {hz_names}." if hz_names else "")
                        + env_sentence + voice_sentence
                        + f" {len(tasks)} non-pharmacological countermeasure protocols initiated. "
                        + ("Immediate flight-surgeon consult and continuous telemetry escalation indicated."
                           if escalate else "Repeat targeted physiological telemetry within 15-30 minutes to verify trajectory.")
                    )
                else:
                    summary = (
                        f"Overall health risk is {overall:.0f}%. Readings needing attention: {names}."
                        + (f" Cabin/space alerts: {hz_names}." if hz_names else "")
                        + env_sentence + voice_sentence
                        + f" {len(tasks)} practical steps were created to help you recover steadily. "
                        + ("Please contact the flight surgeon right away."
                           if escalate else "Follow the suggested steps, rest briefly, and re-check your readings.")
                    )
            recs = []
            seen = set()
            for t in tasks:
                if t.title not in seen:
                    seen.add(t.title)
                    recs.append(t.title)
                if len(recs) == 4:
                    break
            if not recs and voice:
                recs.append(str(voice.get("summary_tr" if lang == "tr" else "summary_en", ""))[:180])

        return AdvisorOutput(summary=summary, recommendations=recs, tasks=tasks, escalate_to_flight_surgeon=escalate)

    @staticmethod
    def answer(question: str, context: Dict[str, Any], lang: str = "en", role: str = "non_doctor") -> str:
        role = _norm_role(role)
        q_low = (question or "").lower()
        risky = context.get("at_risk", [])
        overall = float(context.get("overall_risk_pct", 0.0))
        voice = context.get("voice") or {}
        env = context.get("environment") or {}
        forecast = context.get("forecast_7d_risk_pct") or []

        # Check topic focus from question
        asks_voice = any(w in q_low for w in ("ses", "voice", "mood", "psikolo", "stres", "stress", "duygu", "konuş", "prosod"))
        asks_env = any(w in q_low for w in ("ortam", "kabin", "co2", "radyasyon", "radiation", "cabin", "environment", "oksijen"))

        if lang == "tr":
            if role == "doctor":
                header = f"[Klinik Mod - Uçuş Cerrahı Asistanı | Çevrimdışı Karar Motoru] Genel sağlık riski %{overall:.1f}."
                if asks_voice and voice:
                    bm = voice.get("biomarkers") or {}
                    mood = voice.get("mood") or {}
                    return (
                        f"{header} Vokal prosodi analizi '{voice.get('state_tr', '-')}' sınıfına işaret ediyor "
                        f"(vokal risk %{voice.get('voice_risk_pct', 0):.1f}, %90 GA [{voice.get('ci_low', 0):.1f}-{voice.get('ci_high', 0):.1f}]). "
                        f"Akustik parametreler: F0={bm.get('f0_mean_hz', 0):.1f} Hz (CV={bm.get('f0_cv', 0):.3f}), "
                        f"Jitter=%{bm.get('jitter_pct', 0):.2f}, Shimmer=%{bm.get('shimmer_pct', 0):.2f}, HNR={bm.get('hnr_db', 0):.1f} dB. "
                        f"Afektif eksenler: Valans %{mood.get('valence_pct', 0):.0f}, Uyarılmışlık %{mood.get('arousal_pct', 0):.0f}, "
                        f"Vokal stres %{mood.get('stress_pct', 0):.0f}, Bilişsel yorgunluk %{mood.get('fatigue_pct', 0):.0f}. "
                        f"Öneri: HRV (RMSSD), SpO2 ve kabin pCO2 değerleriyle çapraz korelasyon yapılarak 20 dk vagal solunum sonrası ses kaydı tekrarlanmalıdır."
                    )
                if asks_env and env:
                    return (
                        f"{header} NASA-RIDGE çevresel tehlike indeksi %{env.get('hazard_index_pct', 0):.1f}; "
                        f"karşı-olgusal analiz mevcut uzay ortamının fizyolojik riske +{env.get('environment_uplift_of_health_risk_pts') or 0:.1f} puan eklediğini gösteriyor. "
                        f"Kabin havalandırma/ECLSS parametreleri, radyasyon dozimetresi ve otonom iletişim gecikmesi ({(env.get('one_way_comm_delay_s') or 0)/60:.1f} dk) doğrultusunda "
                        f"yerinde kaynak kontrolü ve seri vital izlem uygulanmalıdır."
                    )
                if not risky:
                    fc_txt = f" 7 günlük olasılıksal risk projeksiyonu %{forecast[-1]:.1f} düzeyindedir." if forecast else ""
                    return (
                        f"{header} Tüm kardiyovasküler, solunumsal ve biyokimyasal parametreler fizyolojik referans sınırlarındadır.{fc_txt} "
                        f"Mikroyerçekimi kaynaklı sefalik sıvı kayması ve kas-iskelet dekondisyonuna karşı standart ARED direnç egzersizi, "
                        f"izotonik hidrasyon (~35 ml/kg/gün) ve sirkadiyen aktigrafi takibinin sürdürülmesi yeterlidir."
                    )
                parts = [
                    f"{PARAM_BY_KEY[i['key']].name('tr') if i['key'] in PARAM_BY_KEY else i['key']}={i.get('value')} "
                    f"(risk %{i['risk_pct']:.1f}, bant={i['level']}, yön={i.get('direction', '-')})"
                    for i in risky[:5]
                ]
                v_note = (
                    f" Eş zamanlı vokal analiz: {voice.get('state_tr')} (risk %{voice.get('voice_risk_pct', 0):.1f})."
                    if voice and voice.get("voice_risk_pct") else ""
                )
                return (
                    f"{header} Aktif patofizyolojik sapmalar: {'; '.join(parts)}.{v_note} "
                    f"Ayırıcı değerlendirmede intravasküler hipovolemi, otonomik sempatovagal dengesizlik ve kabin hiperkapni/radyasyon yükü göz önünde bulundurulmalıdır. "
                    f"İlaç dışı karşı-önlem protokolleri tamamlandıktan 20-30 dakika sonra seri ölçüm ve trend kontrolü önerilir."
                )
            else:
                # non_doctor Turkish: plain, reassuring, easy to follow
                if asks_voice and voice:
                    mood = voice.get("mood") or {}
                    return (
                        f"Ses kaydınıza göre durumunuz '{voice.get('state_tr', '-')}' olarak görünüyor ve ses kaynaklı risk payı %{voice.get('voice_risk_pct', 0):.0f}. "
                        f"Sesinizdeki gerginlik düzeyi %{mood.get('stress_pct', 0):.0f}, yorgunluk belirtisi ise %{mood.get('fatigue_pct', 0):.0f} civarında. "
                        f"Kısa bir mola verip yavaş nefes egzersizi yapmanız, biraz su içmeniz ve dinlenmeniz kendinizi çok daha rahat hissettirecektir."
                    )
                if asks_env and env:
                    return (
                        f"Bulunduğunuz kabin ve uzay ortamı şu an vücudunuza yaklaşık +{env.get('environment_uplift_of_health_risk_pts') or 0:.0f} puanlık ek bir yük bindiriyor. "
                        f"Havalandırmayı kontrol etmek, gürültüyü azaltmak ve bol su içmek bu ortam etkisini hafifletmeye yardımcı olur."
                    )
                if not risky:
                    return (
                        "Şu an tüm sağlık ölçümleriniz gayet dengeli ve normal görünüyor. "
                        "Günlük su tüketiminize, egzersizinize ve uyku düzeninize aynı şekilde devam edebilirsiniz."
                    )
                simple_names = ", ".join(
                    PARAM_BY_KEY[i["key"]].name("tr") if i["key"] in PARAM_BY_KEY else i["key"] for i in risky[:4]
                )
                v_note = (
                    f" Ayrıca ses tonunuzda '{voice.get('state_tr')}' izleri var."
                    if voice and (voice.get("voice_risk_pct") or 0) >= 35 else ""
                )
                return (
                    f"Genel sağlık riskiniz şu an %{overall:.0f} seviyesinde ve özellikle {simple_names} değerleriniz biraz ilgi istiyor.{v_note} "
                    f"Endişelenmeden su içmek, kısa bir nefes/dinlenme molası vermek ve paneldeki günlük görevleri adım adım tamamlamak vücudunuzu rahatlatacaktır; "
                    f"şikayetiniz artarsa uçuş doktoruna haber verin."
                )

        # English responses
        if role == "doctor":
            header = f"[Clinical Flight-Surgeon Mode | Edge Reasoner] Overall physiological risk is {overall:.1f}%."
            if asks_voice and voice:
                bm = voice.get("biomarkers") or {}
                mood = voice.get("mood") or {}
                return (
                    f"{header} Vocal prosodic classification indicates '{voice.get('state_en', '-')}' "
                    f"(voice risk {voice.get('voice_risk_pct', 0):.1f}%, 90% CI [{voice.get('ci_low', 0):.1f}-{voice.get('ci_high', 0):.1f}]). "
                    f"Acoustic markers: F0={bm.get('f0_mean_hz', 0):.1f} Hz (CV={bm.get('f0_cv', 0):.3f}), "
                    f"jitter={bm.get('jitter_pct', 0):.2f}%, shimmer={bm.get('shimmer_pct', 0):.2f}%, HNR={bm.get('hnr_db', 0):.1f} dB; "
                    f"stress={mood.get('stress_pct', 0):.0f}%, cognitive fatigue={mood.get('fatigue_pct', 0):.0f}%. "
                    f"Correlate with autonomic HRV (RMSSD), SpO2, and cabin pCO2, and repeat voice capture 20 min post-countermeasure."
                )
            if not risky:
                return (
                    f"{header} All cardiorespiratory, renal, and hematologic parameters remain within reference bounds. "
                    f"Continue standard microgravity countermeasures (ARED resistive loading, ~35 ml/kg/day isotonic hydration, and circadian actigraphy)."
                )
            parts = [
                f"{PARAM_BY_KEY[i['key']].name('en') if i['key'] in PARAM_BY_KEY else i['key']}={i.get('value')} "
                f"({i['risk_pct']:.1f}% risk, {i['level']}, {i.get('direction', '-')})"
                for i in risky[:5]
            ]
            return (
                f"{header} Active off-nominal biomarkers: {'; '.join(parts)}. "
                f"Evaluate for cephalad fluid-shift hypovolemia, autonomic sympathovagal drift, and habitat CO2/radiation load; "
                f"re-verify serial telemetry 20-30 min after countermeasure execution."
            )
        else:
            if asks_voice and voice:
                mood = voice.get("mood") or {}
                return (
                    f"Based on your voice check, your state looks like '{voice.get('state_en', '-')}' with a {voice.get('voice_risk_pct', 0):.0f}% voice risk score "
                    f"(stress {mood.get('stress_pct', 0):.0f}%, tiredness {mood.get('fatigue_pct', 0):.0f}%). "
                    f"Taking a quiet breather, drinking water, and resting briefly will help bring your energy back into balance."
                )
            if not risky:
                return (
                    "Everything looks healthy and steady right now. Keep drinking water regularly, doing your daily exercise, and getting good rest."
                )
            simple_names = ", ".join(
                PARAM_BY_KEY[i["key"]].name("en") if i["key"] in PARAM_BY_KEY else i["key"] for i in risky[:4]
            )
            return (
                f"Your overall health risk is currently {overall:.0f}%, mainly because {simple_names} are slightly outside your usual comfort zone. "
                f"Follow the simple recovery steps on your task list, drink some water, rest for a few minutes, and reach out to the flight surgeon if you feel worse."
            )


class GeminiAdvisor:
    """Gemini-first advisor with role-aware prompting ('doctor' vs 'non_doctor') and deterministic offline fallback."""

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
            except Exception as exc:
                self.last_error = f"Gemini init failed: {exc}"
                log.warning(self.last_error)

    @property
    def online(self) -> bool:
        return self._client is not None and time.time() >= self._blocked_until

    def status(self) -> Dict[str, Any]:
        return {
            "configured": self._client is not None,
            "model": self.active_model or self.settings.gemini_model,
            "online": self.online,
            "last_error": self.last_error,
            "supported_roles": ["doctor", "non_doctor"],
        }

    def _trip(self, exc: Exception) -> None:
        self.last_error = str(exc)[:300]
        self._blocked_until = time.time() + 60
        log.warning("Gemini call failed, falling back to the offline reasoner: %s", self.last_error)

    def _generate(self, prompt: str, cfg: Any):
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

    def reason(self, context: Dict[str, Any], lang: str = "en", role: str = "non_doctor") -> tuple[AdvisorOutput, str]:
        role = _norm_role(role)
        if self.online:
            try:
                lang_rule = (
                    "Reply strictly in Turkish."
                    if lang == "tr"
                    else "CRITICAL: Write summary, recommendations, and EVERY task title and task instruction strictly in 100% English. Do NOT output Turkish words."
                )
                prompt = (
                    f"Language: {'Turkish' if lang == 'tr' else 'English'}\n"
                    f"{lang_rule}\n"
                    f"{ROLE_INSTRUCTIONS[role]}\n"
                    f"Astronaut context (JSON):\n{json.dumps(context, ensure_ascii=False, default=float)}"
                )
                sys_inst = f"{SYSTEM_PROMPT}\n\n{ROLE_INSTRUCTIONS[role]}\n{lang_rule}"
                cfg = self._types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    temperature=0.3,
                    response_mime_type="application/json",
                    response_schema=AdvisorOutput,
                )
                resp = self._generate(prompt, cfg)
                out = (
                    resp.parsed
                    if isinstance(getattr(resp, "parsed", None), AdvisorOutput)
                    else AdvisorOutput.model_validate_json(resp.text)
                )
                return self._sanitize(out, context, lang=lang), "gemini"
            except Exception as exc:
                self._trip(exc)
        return LocalReasoner.reason(context, lang, role=role), "offline"

    def chat(self, question: str, context: Dict[str, Any], lang: str = "en", role: str = "non_doctor") -> tuple[str, str]:
        role = _norm_role(role)
        if self.online:
            try:
                max_sent = "8-10 structured clinical sentences" if role == "doctor" else "4-6 simple, reassuring sentences"
                prompt = (
                    f"Language: {'Turkish' if lang == 'tr' else 'English'}\n"
                    f"{ROLE_INSTRUCTIONS[role]}\n"
                    f"Context (JSON):\n{json.dumps(context, ensure_ascii=False, default=float)}\n\n"
                    f"Question: {question}\n"
                    f"Answer in {max_sent}. Never use emojis, and never prescribe drug names or doses."
                )
                sys_inst = f"{SYSTEM_PROMPT.split('Output JSON only')[0]}\n\n{ROLE_INSTRUCTIONS[role]}"
                cfg = self._types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.35)
                resp = self._generate(prompt, cfg)
                return (resp.text or "").strip(), "gemini"
            except Exception as exc:
                self._trip(exc)
        return LocalReasoner.answer(question, context, lang, role=role), "offline"

    @staticmethod
    def _sanitize(out: AdvisorOutput, context: Dict[str, Any], lang: str = "en") -> AdvisorOutput:
        by_key = {i["key"]: i for i in context.get("at_risk", []) if i["key"] in PARAM_BY_KEY}
        allowed = set(by_key.keys())
        clean: List[TaskDraft] = []
        seen = set()
        for t in out.tasks:
            if t.parameter_key not in allowed:
                continue
            title_s = t.title.strip()[:120]
            inst_s = t.instructions.strip()[:600]
            mins_s = int(min(max(t.duration_min, 1), 60))
            tag_s = normalize_tag(t.tag)
            if lang != "tr" and (is_turkish_text(title_s) or is_turkish_text(inst_s)):
                direction = by_key[t.parameter_key].get("direction", "high")
                title_s, inst_s, mins_s, tag_s = templates_for(t.parameter_key, direction)[0]
            if title_s.lower() in seen:
                continue
            seen.add(title_s.lower())
            clean.append(TaskDraft(
                parameter_key=t.parameter_key,
                title=title_s,
                instructions=inst_s,
                duration_min=mins_s,
                tag=tag_s,
            ))
        covered = {t.parameter_key for t in clean}
        for item in context.get("at_risk", []):
            if item["key"] in PARAM_BY_KEY and item["key"] not in covered:
                title, text, minutes, tag = templates_for(item["key"], item["direction"])[0]
                clean.append(TaskDraft(parameter_key=item["key"], title=title, instructions=text, duration_min=minutes, tag=tag))
        return AdvisorOutput(
            summary=out.summary[:1200],
            recommendations=[r[:240] for r in out.recommendations[:4]],
            tasks=clean,
            escalate_to_flight_surgeon=out.escalate_to_flight_surgeon
            or any(i["level"] == "black" and not _is_env(i) for i in context.get("at_risk", [])),
        )
