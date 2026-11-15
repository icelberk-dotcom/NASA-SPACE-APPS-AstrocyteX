"""
AstrocyteX - Streamlit Flight-Medical Telemetry Console.

Run:  streamlit run ui/app.py

Tabs: Dashboard | Data entry | Voice acoustics | 12-lead ECG | Protocols & XP | Physiology & trends | AI advisor | About.

Streamlit (Apache-2.0) renders the UI, Plotly (MIT) the charts. The console talks to the
backend either in-process or through the FastAPI REST API (set ASTROGUARD_API_URL).
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

# Make `backend`, `models`, `utils`, `ui` importable when started with `streamlit run ui/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from ui import components as C  # noqa: E402
from ui.client import make_client  # noqa: E402
from ui.i18n import lang, t  # noqa: E402
from ui.theme import dot, esc, inject_css  # noqa: E402
from utils.environment import (  # noqa: E402
    ENV_EVENTS,
    EVENT_NAMES,
    MISSION_PROFILES,
    PROFILE_NAMES,
    derived_environment,
    environment_preset,
    event_log,
)
from utils.reference_ranges import (  # noqa: E402
    CATEGORIES,
    DEFAULT_VALUES,
    ENV_CATEGORIES,
    ENV_KEYS,
    PARAM_KEYS,
    PARAMETERS,
    SEQ_KEYS,
    params_in_category,
    risk_level,
    risk_percent,
)
from utils.synthetic_data import SCENARIO_NAMES, scenario_log, scenario_preset  # noqa: E402
from utils.telemetry import TelemetrySimulator  # noqa: E402
from utils.voice_synth import VOICE_STATE_META, VOICE_STATES  # noqa: E402

st.set_page_config(page_title="AstrocyteX", layout="wide", initial_sidebar_state="expanded")
inject_css()


@st.cache_resource(show_spinner="Starting AstrocyteX (loading neural checkpoints)...")
def get_client():
    return make_client()


client = get_client()


# --------------------------------------------------------------------------- state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("lang", "en")
    ss.setdefault("aid", "slot-01")
    ss.setdefault("user_role", "non_doctor")
    ss.setdefault("advisor_role_radio", "non_doctor")
    ss.setdefault("assessment", None)
    ss.setdefault("ecg_result", None)
    ss.setdefault("voice_result", None)
    ss.setdefault("chat", [])
    ss.setdefault("events", [])
    ss.setdefault("subj_log", "")
    ss.setdefault("mission_day", 90)
    ss.setdefault("profile_name", "iss_leo")
    ss.setdefault("event_name", EVENT_NAMES[0])
    ss.setdefault("stream_on", False)
    for p in PARAMETERS:  # one widget key per parameter, pre-filled with the ideal value
        ss.setdefault(f"val_{p.key}", float(p.ideal))
    if not ss.get("env_initialised"):  # environment widgets start from the default mission profile (ISS)
        for k, v in environment_preset(ss["profile_name"], ss["mission_day"]).items():
            ss[f"val_{k}"] = float(v)
        ss["env_initialised"] = True
    if "telemetry" not in ss:
        sim = TelemetrySimulator({k: DEFAULT_VALUES[k] for k in SEQ_KEYS})
        sim.fill()
        ss["telemetry"] = sim


init_state()
ss = st.session_state


def current_values() -> dict:
    return {k: float(ss[f"val_{k}"]) for k in PARAM_KEYS}


# ------------------------------------------------------------------------ callbacks
def cb_apply_preset() -> None:
    name = ss["preset_name"]
    for k, v in scenario_preset(name).items():
        ss[f"val_{k}"] = float(v)
    ss["subj_log"] = scenario_log(name, lang())
    ss["telemetry"].set_baseline({k: ss[f"val_{k}"] for k in SEQ_KEYS})
    ss["telemetry"].fill()


def cb_apply_profile() -> None:
    """Fill the 12 environment inputs from the selected mission profile and mission day."""
    for k, v in environment_preset(ss["profile_name"], ss["mission_day"]).items():
        ss[f"val_{k}"] = float(v)


def cb_apply_event() -> None:
    """Add an off-nominal environment event (radiation storm, CO2 scrubber failure ...) to the current inputs."""
    from utils.environment import apply_event

    current = {k: float(ss[f"val_{k}"]) for k in ENV_KEYS}
    for k, v in apply_event(current, ss["event_name"]).items():
        ss[f"val_{k}"] = float(v)
    log = event_log(ss["event_name"], lang())
    ss["subj_log"] = (ss["subj_log"] + " " + log).strip() if log not in ss["subj_log"] else ss["subj_log"]


def cb_sync_telemetry() -> None:
    sim = ss["telemetry"]
    for k in SEQ_KEYS:
        if sim.buffers[k]:
            ss[f"val_{k}"] = float(sim.buffers[k][-1])


def cb_apply_ecg() -> None:
    res = ss.get("ecg_result")
    if not res:
        return
    for k, v in res.get("parameter_values", {}).items():
        if k in PARAM_KEYS:
            ss[f"val_{k}"] = float(v)
    ss["events"].append(("info", t("ecg_applied")))


def cb_complete_task(task_id: int) -> None:
    try:
        res = client.complete(ss["aid"], task_id)
    except Exception as exc:  # network / backend failure must not crash the UI
        ss["events"].append(("error", f"Could not complete task: {exc}"))
        return
    if res["points_awarded"]:
        ss["events"].append(("toast", t("task_done_toast", pts=res["points_awarded"]) + f" | {res['task']['title']}"))
    for b in res["new_badges"]:
        ss["events"].append(("toast", f"[{b['icon']}] " + t("badge_unlocked", name=b["name"])))
    if res["leveled_up"]:
        ss["events"].append(("levelup", t("level_up", lvl=res["state"]["level"], rank=res["state"]["rank"])))


def run_assessment() -> None:
    sim = ss["telemetry"]
    sim.set_baseline({k: ss[f"val_{k}"] for k in SEQ_KEYS})
    voice_payload = None
    if ss.get("voice_result"):
        _skip = {"audio_wav_b64", "audio_wav_base64", "waveform_preview", "spectrogram_preview", "mel_spectrogram_preview"}
        voice_payload = {k: v for k, v in ss["voice_result"].items() if k not in _skip}
    payload = {
        "values": current_values(),
        "sensor_series": sim.series() if len(sim.buffers["heart_rate"]) > 8 else None,
        "subjective_log": ss["subj_log"],
        "mission_day": float(ss["mission_day"]),
        "gravity": float(ss["val_gravity_g"]),
        "ecg": ss.get("ecg_result"),
        "voice": voice_payload,
        "user_role": ss.get("user_role", "non_doctor"),
        "save": True,
    }
    with st.spinner(t("analyzing")):
        ss["assessment"] = client.assess(ss["aid"], payload, lang())
    n = len(ss["assessment"]["new_tasks"])
    if n:
        ss["events"].append(("toast", f"{n} countermeasure protocol(s) generated"))


# ---------------------------------------------------------------------- one-time load
if ss["assessment"] is None:
    try:
        ss["assessment"] = client.latest(ss["aid"])
    except Exception:
        pass

# -------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        f'<div class="ag-hero"><div class="ag-kicker">FLIGHT MEDICAL TELEMETRY</div>'
        f'<h1>{esc(t("app_title"))}</h1><p>{esc(t("tagline"))}</p></div>',
        unsafe_allow_html=True,
    )
    st.selectbox(t("language"), ["en", "tr"], key="lang", format_func=lambda x: {"en": "English", "tr": "Türkçe"}[x])
    st.text_input(t("astronaut_id"), key="aid")
    st.selectbox(
        t("role_label"),
        ["non_doctor", "doctor"],
        key="user_role",
        on_change=lambda: ss.__setitem__("advisor_role_radio", ss["user_role"]),
        format_func=lambda r: t("role_doctor") if r == "doctor" else t("role_non_doctor"),
    )
    st.markdown(f"**{t('environment')}**")
    st.selectbox(
        t("mission_profile"),
        PROFILE_NAMES,
        key="profile_name",
        on_change=cb_apply_profile,
        format_func=lambda n: MISSION_PROFILES[n]["tr" if lang() == "tr" else "en"],
    )
    st.slider(t("mission_day"), 0, 900, key="mission_day", on_change=cb_apply_profile)
    der = derived_environment({"earth_distance_mkm": ss["val_earth_distance_mkm"]})
    delay = der["comm_delay_s"] or 0.0
    st.caption(
        f"GRV {ss['val_gravity_g']:.2f} g | DST {ss['val_earth_distance_mkm']:.3f} Mkm | "
        f"{t('comm_delay')}: {delay:.1f} s"
        + (f" (~{delay / 60:.0f} min)" if delay > 90 else "")
        + f" | RAD {ss['val_radiation_dose_msv_d']:.2f} mSv/{t('day')}"
    )
    try:
        gam = client.gamification(ss["aid"])
        st.markdown(C.xp_card_html(gam), unsafe_allow_html=True)
    except Exception as exc:
        gam = None
        st.warning(f"Backend unavailable: {exc}")
    st.markdown(f"**{t('system_status')}**")
    try:
        health = client.health()
        m, g, e = health["model"], health["gemini"], health["ecg"]
        v_stat = health.get("voice") or {}
        st.markdown(f"{dot('green' if m['loaded'] else 'yellow')}{esc(t('model'))}: `{esc(m['version'] or 'rules-only')}`", unsafe_allow_html=True)
        gem = t("online") if g["online"] else (t("offline") if g["configured"] else t("not_configured"))
        st.markdown(f"{dot('green' if g['online'] else 'yellow')}{esc(t('gemini'))}: {esc(gem)}", unsafe_allow_html=True)
        st.markdown(f"{dot('green' if e['service_running'] else 'yellow')}{esc(t('ecg_engine'))}: `{esc(e['mode'])}`", unsafe_allow_html=True)
        if v_stat:
            st.markdown(
                f"{dot('green' if v_stat.get('neural_loaded') else 'yellow')}{esc(t('voice_engine'))}: `{esc(v_stat.get('model_version', 'AstrocyteX-Voice-1.0'))}`",
                unsafe_allow_html=True,
            )
        st.caption(f"backend: {client.mode}")
    except Exception as exc:
        st.caption(f"status unavailable: {exc}")
    st.markdown(
        '<div class="ag-quick-guide">'
        "<strong>Quick Guide</strong>"
        "1. Select slot, mission profile, and day.<br/>"
        "2. Enter vitals, ECG, or voice clip.<br/>"
        "3. Run assessment for fused risk.<br/>"
        "4. Toggle doctor mode for clinical detail."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(t("disclaimer"))

# ------------------------------------------------------------------------ events
for kind, msg in ss["events"]:
    if kind == "toast":
        st.toast(msg)
    elif kind == "levelup":
        st.success(msg)
    elif kind == "error":
        st.error(msg)
    else:
        st.info(msg)
ss["events"] = []

st.markdown(
    f'<div class="ag-hero"><div class="ag-kicker">Flight Medical Telemetry</div>'
    f'<h1>{esc(t("app_title"))}</h1><p>{esc(t("tagline"))}</p></div>',
    unsafe_allow_html=True,
)
tabs = st.tabs(
    [
        t("tab_dashboard"),
        t("tab_input"),
        t("tab_voice"),
        t("tab_ecg"),
        t("tab_missions"),
        t("tab_trends"),
        t("tab_advisor"),
        t("tab_about"),
        t("tab_health_history"),
    ]
)
A = ss["assessment"]

# ============================================================================ DASHBOARD
with tabs[0]:
    st.button(t("run_assessment"), type="primary", key="run_top", on_click=run_assessment)
    if not A:
        st.info(t("no_assessment"))
    else:
        o = A["overall"]
        if o["level"] == "black" or A["advisor"].get("escalate"):
            st.error(t("escalate"))
        c1, c2, c3 = st.columns([1.1, 1.2, 1.3])
        with c1:
            st.markdown(f"#### {dot(o['level'], True)}{esc(t('overall_risk'))}", unsafe_allow_html=True)
            st.plotly_chart(C.gauge(o), use_container_width=True, key="gauge")
            extra_voice = f" | Vocal: {o['voice_risk_pct']:.1f}%" if o.get("voice_risk_pct") is not None else ""
            st.caption(f"{t('uncertainty')}: {o['ci_low']:.0f}-{o['ci_high']:.0f}% | {C.level_label(o['level'])}{extra_voice}")
        with c2:
            st.markdown(f"#### {t('forecast')}")
            if A["forecast"]:
                st.plotly_chart(C.forecast_fig(A["forecast"]), use_container_width=True, key="fc_dash")
        with c3:
            st.markdown(f"#### {t('briefing')}")
            st.write(A["advisor"]["summary"])
            for r in A["advisor"]["recommendations"]:
                st.markdown(f"- {r}")
            st.caption(f"source: {A['advisor']['source']} | profile: {ss.get('user_role', 'non_doctor')}")
        st.markdown(C.legend_html(), unsafe_allow_html=True)

        if A.get("voice"):
            st.markdown(C.voice_summary_html(A["voice"]), unsafe_allow_html=True)

        E = A.get("environment") or {}
        if E:
            st.markdown(f"#### {t('env_title')}")
            e1, e2 = st.columns([1.0, 2.4])
            with e1:
                st.markdown(f"{dot(E['level'], True)}**{esc(t('env_hazard_index'))}**", unsafe_allow_html=True)
                st.plotly_chart(C.gauge({"risk_pct": E["hazard_index_pct"]}, height=200), use_container_width=True, key="env_gauge")
            with e2:
                C.cards_grid(st, E["themes"])
                m1, m2, m3 = st.columns(3)
                up = E.get("uplift")
                if up:
                    m1.metric(
                        t("env_uplift"),
                        f"{up['uplift_pts']:+.1f} pts",
                        help=t("env_uplift_help", base=up["overall_neutral_env_pct"], act=up["overall_model_pct"]),
                    )
                d = E.get("derived", {})
                if d.get("comm_delay_s") is not None:
                    m2.metric(
                        t("comm_delay"),
                        f"{d['comm_delay_s']:.1f} s" if d["comm_delay_s"] < 90 else f"{d['comm_delay_s'] / 60:.1f} min",
                    )
                if d.get("time_to_care_days") is not None:
                    m3.metric(t("time_to_care"), f"{d['time_to_care_days']:g} {t('days')}")
                if up and up["top_parameters"]:
                    names = ", ".join(
                        f"{(p['name_tr'] if lang() == 'tr' else p['name'])} (+{p['uplift_pts']:.0f})" for p in up["top_parameters"]
                    )
                    st.caption(f"{t('env_affects')}: {names}")

        st.markdown(f"#### {t('systems')}")
        C.systems_grid(st, A)
        st.markdown(f"#### {t('parameters')}")
        by_cat = {}
        for p in A["parameters"]:
            by_cat.setdefault(p["category"], []).append(p)
        for cat, params in by_cat.items():
            if cat in ENV_CATEGORIES and cat == ENV_CATEGORIES[0]:
                st.markdown(f"##### {t('env_section')}")
            worst = max((p["risk_pct"] or 0) for p in params)
            label = CATEGORIES[cat]["tr" if lang() == "tr" else "en"]
            with st.expander(f"[{CATEGORIES[cat]['icon']}] {label}  |  max {worst:.0f}%", expanded=worst > 40):
                st.markdown(C.parameter_table_html(params), unsafe_allow_html=True)

# ============================================================================ DATA ENTRY
with tabs[1]:
    top = st.columns([2, 1.2, 1.2])
    top[0].selectbox(t("demo_preset"), SCENARIO_NAMES, key="preset_name")
    top[1].button(t("apply_preset"), on_click=cb_apply_preset, key="apply_preset")
    top[2].button(t("run_assessment"), type="primary", key="run_input", on_click=run_assessment)
    env_row = st.columns([2, 1.2, 1.2])
    env_row[0].selectbox(
        t("env_event"),
        EVENT_NAMES,
        key="event_name",
        format_func=lambda n: ENV_EVENTS[n]["tr" if lang() == "tr" else "en"],
    )
    env_row[1].button(t("apply_event"), on_click=cb_apply_event, key="apply_event")
    env_row[2].button(t("apply_profile"), on_click=cb_apply_profile, key="apply_profile")
    st.markdown(C.legend_html(), unsafe_allow_html=True)

    for cat in CATEGORIES:
        params = params_in_category(cat)
        if cat == ENV_CATEGORIES[0]:
            st.markdown(f"#### {t('env_section')}")
            st.caption(t("env_section_help"))
        label = CATEGORIES[cat]["tr" if lang() == "tr" else "en"]
        worst = max(risk_percent(p.key, ss[f"val_{p.key}"]) or 0 for p in params)
        with st.expander(f"[{CATEGORIES[cat]['icon']}] {label}", expanded=worst > 40 or cat in ("cardiovascular", "respiratory")):
            cols = st.columns(3)
            for i, p in enumerate(params):
                with cols[i % 3]:
                    c_dot, c_in = st.columns([1, 8])
                    lv = risk_level(risk_percent(p.key, ss[f"val_{p.key}"]))
                    c_dot.markdown(f"<div style='margin-top:34px'>{dot(lv, True)}</div>", unsafe_allow_html=True)
                    lo, hi, step = p.bounds
                    unit = f" ({p.unit})" if p.unit else ""
                    c_in.number_input(
                        f"{p.name(lang())}{unit}",
                        min_value=float(lo),
                        max_value=float(hi),
                        step=float(step),
                        format=f"%.{p.decimals}f",
                        key=f"val_{p.key}",
                        help=f"{t('normal')}: {p.normal_range_text()}. {p.notes}",
                    )
                    c_in.caption(f"{t('normal')}: {p.normal_range_text()}")

    st.text_area(
        t("subjective_log"),
        key="subj_log",
        height=90,
        placeholder="e.g. Slept badly, mild dizziness and thirst / Kötü uyudum, hafif baş dönmesi var...",
    )

    st.markdown(f"#### {t('telemetry')}")
    sim: TelemetrySimulator = ss["telemetry"]
    tcols = st.columns([1, 1, 3])
    tcols[0].toggle(t("stream_on"), key="stream_on")
    tcols[1].button(t("sync_telemetry"), on_click=cb_sync_telemetry, key="sync_tel")

    def _telemetry_body() -> None:
        s: TelemetrySimulator = st.session_state["telemetry"]
        if st.session_state.get("stream_on"):
            s.set_baseline({k: st.session_state[f"val_{k}"] for k in SEQ_KEYS})
            s.step()
        cells = []
        for k in SEQ_KEYS:
            if not s.buffers[k]:
                continue
            v = s.buffers[k][-1]
            p = next(q for q in PARAMETERS if q.key == k)
            cells.append(
                f"{dot(risk_level(risk_percent(k, v)))}<span class='ag-muted'>{esc(p.name(lang()))}</span> "
                f"<b class='ag-mono'>{v:.{p.decimals}f}</b> <span class='ag-muted'>{esc(p.unit)}</span>"
            )
        st.markdown(" &nbsp;&middot;&nbsp; ".join(cells), unsafe_allow_html=True)
        st.plotly_chart(C.telemetry_fig(s.series()), use_container_width=True, key=f"tele_{s.t}")

    st.fragment(run_every=2 if ss.get("stream_on") else None)(_telemetry_body)()

# ===================================================================== VOICE ACOUSTICS
with tabs[2]:
    st.markdown(f"### {t('voice_title')}")
    st.caption(t("voice_subtitle"))
    vc1, vc2 = st.columns(2)
    with vc1:
        rec_audio = st.audio_input(t("voice_record"), key="voice_mic") if hasattr(st, "audio_input") else None
        up_voice = st.file_uploader(t("voice_upload"), type=["wav", "npy", "csv"], key="voice_file")
        active_upload = rec_audio if rec_audio is not None else up_voice
        if st.button(t("voice_analyze"), disabled=active_upload is None, key="voice_analyze_btn"):
            with st.spinner("AstroVoiceNet v1.0 acoustic inference..."):
                try:
                    fname = getattr(active_upload, "name", None) or "recorded_checkin.wav"
                    ss["voice_result"] = client.voice_analyze(active_upload.getvalue(), fname, lang=lang())
                    ss["events"].append(("toast", t("voice_attached")))
                except Exception as exc:
                    st.error(f"Voice analysis failed: {exc}")
    with vc2:
        state_keys = list(VOICE_STATES)
        state_labels = {k: VOICE_STATE_META[k]["tr" if lang() == "tr" else "en"] for k in state_keys}
        v_sample = st.selectbox(t("voice_sample"), state_keys, format_func=lambda k: state_labels[k], key="voice_sample_sel")
        b_col1, b_col2 = st.columns(2)
        if b_col1.button(t("voice_run_sample"), key="voice_sample_btn"):
            with st.spinner("Synthesising & analysing astronaut voice profile..."):
                try:
                    ss["voice_result"] = client.voice_sample(v_sample, lang=lang())
                except Exception as exc:
                    st.error(f"Voice preset analysis failed: {exc}")
        if b_col2.button(t("voice_clear"), disabled=ss.get("voice_result") is None, key="voice_clear_btn"):
            ss["voice_result"] = None

    V = ss.get("voice_result")
    if V:
        if V.get("audio_wav_b64"):
            st.audio(base64.b64decode(V["audio_wav_b64"]), format="audio/wav")
        st.markdown(C.voice_summary_html(V), unsafe_allow_html=True)
        note_key = (
            ("summary_doctor_tr" if ss.get("user_role") == "doctor" else "summary_simple_tr")
            if lang() == "tr"
            else ("summary_doctor_en" if ss.get("user_role") == "doctor" else "summary_simple_en")
        )
        if V.get(note_key):
            st.info(V[note_key])
        vc_a, vc_b = st.columns(2)
        mood = V.get("mood_dimensions") or {}
        if mood:
            vc_a.plotly_chart(
                C.bars(
                    {
                        "Valence": float(mood.get("valence_pct", 0)),
                        "Arousal": float(mood.get("arousal_pct", 0)),
                        "Stress": float(mood.get("stress_pct", 0)),
                        "Fatigue": float(mood.get("fatigue_pct", 0)),
                    },
                    "Affective mood dimensions (%)",
                    "#b89cff",
                ),
                use_container_width=True,
                key="voice_mood_bars",
            )
        probs = V.get("state_probabilities") or []
        if probs:
            vc_b.plotly_chart(
                C.bars(
                    {(p["label_tr"] if lang() == "tr" else p["label_en"]): float(p["probability_pct"]) for p in probs},
                    "Psychological state probabilities (%)",
                    "#8c6ce6",
                ),
                use_container_width=True,
                key="voice_prob_bars",
            )
        st.button(t("run_assessment"), type="primary", key="run_voice_assess", on_click=run_assessment)

# ===================================================================================== ECG
with tabs[3]:
    st.markdown(f"### {t('ecg_title')}")
    try:
        es = client.ecg_status()
        st.caption(f"{t('ecg_mode')}: **{es['mode']}** | package: `{es['package_path'] or 'not configured (set ECG_PACKAGE_PATH)'}`")
    except Exception as exc:
        st.caption(f"ECG status unavailable: {exc}")
    c1, c2 = st.columns(2)
    up = c1.file_uploader(t("ecg_upload"), type=["npy", "npz", "csv", "mat", "dat"], key="ecg_file")
    if c1.button(t("ecg_analyze"), disabled=up is None, key="ecg_go"):
        with st.spinner("ECG analysis..."):
            try:
                ss["ecg_result"] = client.ecg_analyze(up.getvalue(), up.name)
            except Exception as exc:
                st.error(f"ECG analysis failed: {exc}")
    sample = c2.selectbox(
        t("ecg_sample"), ["NORMAL", "AFIB", "AFL", "LBBB", "RBBB", "MI", "PVC", "PAC", "AVB", "STTC"], key="ecg_sample_cls"
    )
    if c2.button(t("ecg_analyze") + " (reference)", key="ecg_sample_go"):
        with st.spinner("ECG analysis..."):
            try:
                ss["ecg_result"] = client.ecg_sample(sample)
            except Exception as exc:
                st.error(f"ECG analysis failed: {exc}")
    R = ss.get("ecg_result")
    if R:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Diagnosis", R["class_code"], f"{R['confidence_pct']:.0f}% conf.")
        k2.metric("Heart rate", "-" if R["heart_rate"] is None else f"{R['heart_rate']:.0f} bpm")
        k3.metric("QTc", "-" if not R["qtc_ms"] else f"{R['qtc_ms']:.0f} ms")
        k4.metric("PR / QRS", f"{R['pr_ms'] or '-'} / {R['qrs_ms'] or '-'} ms")
        k5.metric("Abnormal prob.", f"{R['abnormal_probability_pct']:.0f}%")
        stemi_chip = "<span class='ag-chip' style='border-color:#e05260'>STEMI SUSPECTED</span>" if R["stemi_suspected"] else ""
        st.markdown(
            f"<span class='ag-chip' style='border-color:{R['severity_color']}'>{esc(R['severity_label'])}</span>"
            f"<span class='ag-chip'>{esc(R['model_version'])}</span><span class='ag-chip'>{esc(R['source'])}</span>"
            f"{stemi_chip}",
            unsafe_allow_html=True,
        )
        st.write(f"**{R['class_name']}** - {R['medical_desc']}")
        if R["lead_ii"]:
            st.plotly_chart(C.ecg_fig(R["lead_ii"]), use_container_width=True, key="ecg_plot")
        if R["probabilities"]:
            st.plotly_chart(
                C.bars({p["name"] or p["code"]: float(p["pct"]) for p in R["probabilities"]}, "Top class probabilities (%)"),
                use_container_width=True,
                key="ecg_probs",
            )
        st.caption(R["notes"])
        st.button(t("ecg_apply"), on_click=cb_apply_ecg, type="primary", key="ecg_apply_btn")

# ===================================================================== MISSIONS & REWARDS
with tabs[4]:
    gam = client.gamification(ss["aid"])
    cA, cB = st.columns([2, 1])
    with cA:
        st.markdown(C.xp_card_html(gam), unsafe_allow_html=True)
    cB.metric(t("completed_tasks"), gam["tasks_completed"])
    pending = client.tasks(ss["aid"], "pending")
    done = client.tasks(ss["aid"], "done")
    st.markdown(f"### {t('pending_tasks')} ({len(pending)})")
    if not pending:
        st.success(t("no_tasks"))
    for tk in pending:
        c0, c1, c2 = st.columns([0.05, 0.75, 0.2])
        c0.markdown(f"<div style='margin-top:6px'>{dot(tk['severity_level'], True)}</div>", unsafe_allow_html=True)
        with c1:
            st.checkbox(f"**{tk['title']}**", key=f"task_{tk['id']}", on_change=cb_complete_task, args=(tk["id"],))
            p_obj = next((q for q in PARAMETERS if q.key == tk["parameter_key"]), None)
            p_name = p_obj.name(lang()) if p_obj else ("Ses Akustiği" if lang() == "tr" else "Voice Acoustics")
            st.markdown(
                f"<div class='ag-task-meta'>{esc(tk['description'])}<br>{tk['duration_min']} {t('min')} &middot; "
                f"{t('from_param')}: {esc(p_name)} ({tk['risk_pct']:.0f}%) &middot; "
                f"<span class='ag-chip'>{esc(tk['tag'])}</span><span class='ag-chip'>{esc(tk['source'])}</span></div>",
                unsafe_allow_html=True,
            )
        c2.markdown(f"<div class='ag-mono' style='text-align:right;margin-top:6px'><b>+{tk['points']}</b> {t('pts')}</div>", unsafe_allow_html=True)
    if done:
        with st.expander(f"{t('completed_tasks')} ({len(done)})"):
            for tk in done[:50]:
                st.markdown(f"~~{esc(tk['title'])}~~ &middot; +{tk['points']} {t('pts')} &middot; {esc(tk['completed_at'] or '')}")
    st.markdown(f"### {t('badges')} ({sum(b['earned'] for b in gam['badges'])}/{len(gam['badges'])})")
    st.markdown(C.badges_html(gam["badges"]), unsafe_allow_html=True)

# ================================================================== PHYSIOLOGY & TRENDS
with tabs[5]:
    if not A:
        st.info(t("no_assessment"))
    else:
        st.markdown(f"### {t('history')}")
        hist = client.history(ss["aid"])
        if hist:
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Scatter(
                    x=[h["ts"] for h in hist],
                    y=[h["overall_risk_pct"] for h in hist],
                    mode="lines+markers",
                    line=dict(color="#b89cff", width=2.8),
                )
            )
            fig.update_yaxes(range=[0, 100], gridcolor="rgba(196,173,255,.10)")
            fig.update_layout(
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#eee9f8"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True, key="hist")
        if A["physics"]:
            st.markdown(f"### {t('physics_title')}")
            st.plotly_chart(C.physics_fig(A["physics"]), use_container_width=True, key="phys")
            cols = st.columns(3)
            for col, (name, pr) in zip(cols, A["physics"]["projections"].items()):
                col.markdown(f"**{name}**")
                col.write(f"Bone density: **{pr['bone_density_change_pct']:+.2f}%**")
                col.write(f"Muscle mass: **{pr['muscle_mass_change_pct']:+.2f}%**")
                col.write(f"Plasma volume: **{pr['plasma_volume_change_pct']:+.1f}%**")
            st.caption("Hybrid physics + learned-correction ODE; constants are illustrative and not clinically validated.")
        c1, c2 = st.columns(2)
        if A["modality_attention"]:
            c1.plotly_chart(
                C.bars({k: round(v * 100, 1) for k, v in A["modality_attention"].items()}, t("attention") + " (%)", "#8c6ce6"),
                use_container_width=True,
                key="attn",
            )
        if A["trend_drivers"]:
            c2.plotly_chart(
                C.bars({d["name"]: round(d["weight"] * 100, 1) for d in A["trend_drivers"]}, t("drivers") + " (%)", "#3cbf88"),
                use_container_width=True,
                key="drivers",
            )
        haz = (A.get("environment") or {}).get("hazard_attention")
        if haz:
            from utils.reference_ranges import PARAM_BY_KEY

            top_h = dict(sorted(haz.items(), key=lambda kv: -kv[1])[:8])
            st.plotly_chart(
                C.bars(
                    {PARAM_BY_KEY[k].name(lang()): round(v * 100, 1) for k, v in top_h.items()},
                    t("hazard_attention") + " (%)",
                    "#d9a938",
                ),
                use_container_width=True,
                key="hazard_attn",
            )

# ============================================================================ AI ADVISOR
with tabs[6]:
    st.markdown(f"### {t('advisor_title')}")
    st.radio(
        t("role_label"),
        ["non_doctor", "doctor"],
        horizontal=True,
        format_func=lambda r: t("role_doctor") if r == "doctor" else t("role_non_doctor"),
        key="advisor_role_radio",
        on_change=lambda: ss.__setitem__("user_role", ss["advisor_role_radio"]),
    )
    if A:
        st.info(A["advisor"]["summary"])
        for r in A["advisor"]["recommendations"]:
            st.markdown(f"- {r}")
        st.caption(
            f"source: **{A['advisor']['source']}** | Gemini model: `{A['advisor']['gemini']['model']}` | role: `{ss['user_role']}`"
            + (f" | last error: {A['advisor']['gemini']['last_error']}" if A["advisor"]["gemini"].get("last_error") else "")
        )
    for who, msg in ss["chat"]:
        st.chat_message(who).write(msg)
    q = st.chat_input(t("ask"))
    if q:
        ss["chat"].append(("user", q))
        res = client.chat(ss["aid"], q, lang(), role=ss.get("user_role", "non_doctor"))
        ss["chat"].append(("assistant", f"{res['answer']}\n\n*({res['source']} | {ss.get('user_role', 'non_doctor')})*"))
        st.rerun()

# ================================================================================== ABOUT
with tabs[7]:
    about_path = Path(__file__).resolve().parent.parent.joinpath("docs", "ABOUT_UI.md")
    st.markdown(about_path.read_text(encoding="utf-8") if about_path.exists() else "AstrocyteX")

# ========================================================================= HEALTH HISTORY
with tabs[8]:
    st.markdown(f"### {t('hh_title')}")
    hh_c1, hh_c2, hh_c3 = st.columns([2.4, 1.0, 0.8])
    hh_note = hh_c1.text_input(t("hh_note_label"), key="hh_note_input")
    if hh_c2.button(t("hh_record_btn"), type="primary", key="hh_record_btn"):
        prev_subj = ss["subj_log"]
        if hh_note.strip():
            ss["subj_log"] = f"{hh_note.strip()} - {prev_subj}" if prev_subj else hh_note.strip()
        try:
            run_assessment()
        finally:
            ss["subj_log"] = prev_subj
        st.rerun()
    if hh_c3.button(t("hh_clear_btn"), key="hh_clear_btn"):
        client.clear_history(ss["aid"])
        st.rerun()

    hh_records = client.history(ss["aid"])
    if not hh_records:
        st.info(t("hh_empty"))
    else:
        import pandas as pd
        import plotly.graph_objects as go

        level_palette = {
            "gray": "#7c7a8c",
            "green": "#29b377",
            "yellow": "#df9f15",
            "red": "#e5484d",
            "black": "#e5484d",
        }
        m1, m2, m3, m4 = st.columns(4)
        risks_all = [float(r.get("overall_risk_pct") or 0.0) for r in hh_records]
        m1.metric("Records", len(hh_records))
        m2.metric("Latest Risk", f"{risks_all[-1]:.1f}%")
        m3.metric("Mean Risk", f"{sum(risks_all) / len(risks_all):.1f}%")
        m4.metric("Peak Risk", f"{max(risks_all):.1f}%")

        st.markdown(f"#### {t('hh_col_chart')}")
        x_labels = [f"#{idx + 1} ({(r.get('ts') or '')[11:19]})" for idx, r in enumerate(hh_records)]
        bar_colors = [level_palette.get(r.get("level", "gray"), "#9b87f5") for r in hh_records]
        col_fig = go.Figure(
            data=[
                go.Bar(
                    x=x_labels,
                    y=risks_all,
                    marker_color=bar_colors,
                    text=[f"{v:.1f}%" for v in risks_all],
                    textposition="outside",
                    name="Health Risk %",
                )
            ]
        )
        col_fig.add_hline(y=40, line_dash="dash", line_color="#df9f15")
        col_fig.add_hline(y=70, line_dash="dash", line_color="#e5484d")
        col_fig.update_yaxes(range=[0, 108], gridcolor="rgba(196,173,255,.10)")
        col_fig.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee9f8"),
            margin=dict(l=12, r=12, t=22, b=16),
        )
        st.plotly_chart(col_fig, use_container_width=True, key="hh_col_plot")

        st.markdown(f"#### {t('hh_table')}")
        table_rows = []
        for idx, r in enumerate(reversed(hh_records)):
            seq_no = len(hh_records) - idx
            vit = r.get("vitals") or {}
            flags = ", ".join(f"{x.get('name')} ({x.get('risk_pct')}%)" for x in (r.get("top_risks") or [])) or "Nominal"
            table_rows.append({
                "#": seq_no,
                "Timestamp (UTC)": (r.get("ts") or "").replace("T", " ")[:19],
                "Mission Day": int(round(float(r.get("mission_day") or 0))),
                "Health Risk (%)": round(float(r.get("overall_risk_pct") or 0.0), 1),
                "Band": str(r.get("level") or "gray").upper(),
                "Physio (%)": round(float(r.get("physiological_risk_pct") or 0.0), 1),
                "Voice (%)": "-" if r.get("voice_risk_pct") is None else round(float(r["voice_risk_pct"]), 1),
                "Habitat (%)": "-" if r.get("env_hazard_pct") is None else round(float(r["env_hazard_pct"]), 1),
                "HR (bpm)": vit.get("heart_rate"),
                "SpO2 (%)": vit.get("spo2"),
                "BP (mmHg)": f"{vit.get('systolic_bp', '-')}/{vit.get('diastolic_bp', '-')}" if vit.get("systolic_bp") is not None else "-",
                "Sleep (h)": vit.get("sleep_hours"),
                "Off-Nominal Biomarkers": flags,
                "Note": r.get("subjective_log") or "-",
            })
        df_hh = pd.DataFrame(table_rows)
        st.dataframe(df_hh, use_container_width=True, hide_index=True)
        st.download_button(
            "Export CSV",
            data=df_hh.to_csv(index=False).encode("utf-8"),
            file_name=f"astrocytex_health_history_{ss['aid']}.csv",
            mime="text/csv",
            key="hh_csv_dl",
        )
