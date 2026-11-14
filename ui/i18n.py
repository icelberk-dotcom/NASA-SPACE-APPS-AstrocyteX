"""Tiny i18n layer (English / Türkçe) for the dashboard."""
from __future__ import annotations

import streamlit as st

STRINGS = {
    "en": {
        "app_title": "AstroGuard-X", "tagline": "AI astronaut health monitoring · gamified mitigation",
        "sidebar_crew": "Crew member", "astronaut_id": "Astronaut ID", "language": "Language",
        "environment": "Environment", "gravity": "Gravity", "mission_day": "Mission day",
        "system_status": "System status", "model": "Risk model", "gemini": "Gemini advisor", "ecg_engine": "ECG engine",
        "online": "online", "offline": "offline (edge mode)", "not_configured": "no API key → offline reasoner",
        "tab_dashboard": "📊 Dashboard", "tab_input": "🧾 Data entry", "tab_ecg": "📈 ECG", "tab_missions": "🎯 Missions & rewards",
        "tab_trends": "🧬 Physiology & trends", "tab_advisor": "🤖 AI advisor", "tab_about": "ℹ️ About",
        "run_assessment": "▶ Run AI assessment", "no_assessment": "No assessment yet - fill in the data and press **Run AI assessment**.",
        "overall_risk": "Overall health risk", "forecast": "7-day risk forecast", "systems": "Physiological systems",
        "parameters": "Parameters", "legend": "Risk indicator", "value": "Value", "normal": "Normal range", "risk": "Risk",
        "uncertainty": "90 % interval", "demo_preset": "Demo scenario", "apply_preset": "Load scenario",
        "subjective_log": "Subjective log (how do you feel?)", "telemetry": "Live telemetry (simulated wearable stream)",
        "stream_on": "Stream live", "sync_telemetry": "Copy latest telemetry into the form",
        "xp": "XP", "level": "Level", "rank": "Rank", "streak": "Day streak", "pending_tasks": "Active tasks",
        "completed_tasks": "Completed tasks", "badges": "Badges", "no_tasks": "No active tasks - all parameters are in a safe state. 🎉",
        "task_done_toast": "+{pts} XP", "level_up": "LEVEL UP! You reached level {lvl} - {rank}", "badge_unlocked": "Badge unlocked: {name}",
        "min": "min", "pts": "pts", "from_param": "triggered by",
        "ecg_title": "ECG analyser (TEKNOFEST V23 integration)", "ecg_upload": "Upload a 12-lead ECG (.npy, .npz, .csv, .mat, .dat)",
        "ecg_sample": "…or analyse a demo recording", "ecg_analyze": "Analyse ECG", "ecg_apply": "Apply to dashboard values",
        "ecg_applied": "ECG values applied - run the assessment to include them.", "ecg_mode": "Analyser mode",
        "advisor_title": "AI advisor", "ask": "Ask the advisor about your health status…", "briefing": "Mission briefing",
        "escalate": "⚠️ Contact the flight surgeon", "physics_title": "Space-physiology projection (Neural ODE + PINN)",
        "attention": "What drove this assessment (cross-attention)", "drivers": "Most influential parameters (TFT variable selection)",
        "history": "Overall risk history", "disclaimer": "Research / demonstration software. Not a medical device. Simulated-data models.",
        "analyzing": "Analysing with Mamba-2 · TCN+TFT · cross-attention · Neural ODE · Bayesian head …",
        "level_gray": "Normal / baseline", "level_green": "Low risk", "level_yellow": "Minor deviation",
        "level_red": "High risk", "level_black": "Immediate intervention",
        "mission_profile": "Mission profile", "apply_profile": "Reset environment to mission profile",
        "env_event": "Environment event (off-nominal)", "apply_event": "Apply event", "day": "day", "days": "days",
        "env_title": "🛰️ Space environment - NASA RIDGE hazards", "env_hazard_index": "Environment hazard index",
        "env_uplift": "Environmental risk uplift",
        "env_uplift_help": "Counterfactual: the network re-evaluates the same data in a benign Earth-like environment. "
                           "Health risk there: {base} % → here: {act} %.",
        "env_affects": "Most affected by the environment", "comm_delay": "One-way comm. delay",
        "time_to_care": "Time to definitive care", "env_section": "🌌 Space environment (radiation · isolation · distance · gravity · habitat)",
        "env_section_help": "These are hazards of the surroundings, not of the body. The AI uses them to re-weight the physiological risk.",
        "hazard_attention": "Which environment hazards the network focused on",
    },
    "tr": {
        "app_title": "AstroGuard-X", "tagline": "Yapay zekâ destekli astronot sağlık izleme · oyunlaştırılmış önlemler",
        "sidebar_crew": "Mürettebat", "astronaut_id": "Astronot ID", "language": "Dil",
        "environment": "Ortam", "gravity": "Yerçekimi", "mission_day": "Görev günü",
        "system_status": "Sistem durumu", "model": "Risk modeli", "gemini": "Gemini danışman", "ecg_engine": "EKG motoru",
        "online": "çevrimiçi", "offline": "çevrimdışı (uç mod)", "not_configured": "API anahtarı yok → çevrimdışı akıl yürütücü",
        "tab_dashboard": "📊 Panel", "tab_input": "🧾 Veri girişi", "tab_ecg": "📈 EKG", "tab_missions": "🎯 Görevler ve ödüller",
        "tab_trends": "🧬 Fizyoloji ve trendler", "tab_advisor": "🤖 YZ danışman", "tab_about": "ℹ️ Hakkında",
        "run_assessment": "▶ YZ değerlendirmesini çalıştır", "no_assessment": "Henüz değerlendirme yok - verileri girin ve **YZ değerlendirmesini çalıştır**'a basın.",
        "overall_risk": "Genel sağlık riski", "forecast": "7 günlük risk tahmini", "systems": "Fizyolojik sistemler",
        "parameters": "Parametreler", "legend": "Risk göstergesi", "value": "Değer", "normal": "Normal aralık", "risk": "Risk",
        "uncertainty": "%90 aralığı", "demo_preset": "Demo senaryosu", "apply_preset": "Senaryoyu yükle",
        "subjective_log": "Öznel günlük (nasıl hissediyorsunuz?)", "telemetry": "Canlı telemetri (simüle giyilebilir akış)",
        "stream_on": "Canlı akış", "sync_telemetry": "Son telemetriyi forma kopyala",
        "xp": "XP", "level": "Seviye", "rank": "Rütbe", "streak": "Gün serisi", "pending_tasks": "Aktif görevler",
        "completed_tasks": "Tamamlanan görevler", "badges": "Rozetler", "no_tasks": "Aktif görev yok - tüm parametreler güvenli durumda. 🎉",
        "task_done_toast": "+{pts} XP", "level_up": "SEVİYE ATLADIN! Seviye {lvl} - {rank}", "badge_unlocked": "Rozet kazanıldı: {name}",
        "min": "dk", "pts": "puan", "from_param": "tetikleyen",
        "ecg_title": "EKG analizörü (TEKNOFEST V23 entegrasyonu)", "ecg_upload": "12 derivasyonlu EKG yükleyin (.npy, .npz, .csv, .mat, .dat)",
        "ecg_sample": "…veya demo kaydı analiz edin", "ecg_analyze": "EKG'yi analiz et", "ecg_apply": "Panel değerlerine uygula",
        "ecg_applied": "EKG değerleri uygulandı - dahil etmek için değerlendirmeyi çalıştırın.", "ecg_mode": "Analizör modu",
        "advisor_title": "YZ danışman", "ask": "Sağlık durumunuz hakkında danışmana sorun…", "briefing": "Görev özeti",
        "escalate": "⚠️ Uçuş cerrahıyla iletişime geçin", "physics_title": "Uzay fizyolojisi projeksiyonu (Neural ODE + PINN)",
        "attention": "Bu değerlendirmeyi ne belirledi (çapraz dikkat)", "drivers": "En etkili parametreler (TFT değişken seçimi)",
        "history": "Genel risk geçmişi", "disclaimer": "Araştırma / gösterim yazılımı. Tıbbi cihaz değildir. Simüle veriyle eğitilmiş modeller.",
        "analyzing": "Mamba-2 · TCN+TFT · çapraz dikkat · Neural ODE · Bayes başlığı ile analiz ediliyor …",
        "level_gray": "Normal / başlangıç", "level_green": "Düşük risk", "level_yellow": "Hafif sapma",
        "level_red": "Yüksek risk", "level_black": "Acil müdahale",
        "mission_profile": "Görev profili", "apply_profile": "Ortamı görev profiline sıfırla",
        "env_event": "Ortam olayı (olağan dışı)", "apply_event": "Olayı uygula", "day": "gün", "days": "gün",
        "env_title": "🛰️ Uzay ortamı - NASA RIDGE tehlikeleri", "env_hazard_index": "Ortam tehlike indeksi",
        "env_uplift": "Ortamsal risk artışı",
        "env_uplift_help": "Karşı-olgusal analiz: ağ aynı verileri Dünya benzeri zararsız bir ortamda yeniden değerlendirir. "
                           "Oradaki sağlık riski: %{base} → burada: %{act}.",
        "env_affects": "Ortamdan en çok etkilenenler", "comm_delay": "Tek yönlü iletişim gecikmesi",
        "time_to_care": "Kesin tıbbi bakıma süre", "env_section": "🌌 Uzay ortamı (radyasyon · izolasyon · uzaklık · yerçekimi · habitat)",
        "env_section_help": "Bunlar vücudun değil çevrenin tehlikeleridir. Yapay zekâ fizyolojik riski bunlara göre yeniden ağırlıklandırır.",
        "hazard_attention": "Ağın odaklandığı ortam tehlikeleri",
    },
}


def lang() -> str:
    return st.session_state.get("lang", "en")


def t(key: str, **kw) -> str:
    s = STRINGS.get(lang(), STRINGS["en"]).get(key) or STRINGS["en"].get(key, key)
    return s.format(**kw) if kw else s
