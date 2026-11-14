"""
AstroGuard-X - Streamlit dashboard.

Run:  streamlit run ui/app.py

Tabs: Dashboard · Data entry · ECG · Missions & rewards · Physiology & trends · AI advisor · About.

Streamlit (Apache-2.0) renders the UI, Plotly (MIT) the charts.  The dashboard talks to the
backend either in-process or through the FastAPI REST API (set ASTROGUARD_API_URL).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `backend`, `models`, `utils`, `ui` importable when started with `streamlit run ui/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from ui import components as C  # noqa: E402
from ui.client import make_client  # noqa: E402
from ui.i18n import lang, t  # noqa: E402
from ui.theme import dot, esc, inject_css  # noqa: E402
from utils.features import GRAVITY_PRESETS  # noqa: E402
from utils.reference_ranges import (CATEGORIES, DEFAULT_VALUES, PARAM_KEYS, PARAMETERS, SEQ_KEYS,  # noqa: E402
                                    params_in_category, risk_level, risk_percent)
from utils.synthetic_data import SCENARIO_NAMES, scenario_log, scenario_preset  # noqa: E402
from utils.telemetry import TelemetrySimulator  # noqa: E402

st.set_page_config(page_title="AstroGuard-X", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")
inject_css()


@st.cache_resource(show_spinner="Starting AstroGuard-X (loading / training the risk model on first run)…")
def get_client():
    return make_client()


client = get_client()

# --------------------------------------------------------------------------- state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("lang", "en")
    ss.setdefault("aid", "astro-1")
    ss.setdefault("assessment", None)
    ss.setdefault("ecg_result", None)
    ss.setdefault("chat", [])
    ss.setdefault("events", [])
    ss.setdefault("subj_log", "")
    ss.setdefault("mission_day", 90)
    ss.setdefault("gravity_name", "Microgravity (ISS)")
    ss.setdefault("stream_on", False)
    for p in PARAMETERS:  # one widget key per parameter, pre-filled with the ideal value
        ss.setdefault(f"val_{p.key}", float(p.ideal))
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
        ss["events"].append(("toast", t("task_done_toast", pts=res["points_awarded"]) + f" · {res['task']['title']}"))
    for b in res["new_badges"]:
        ss["events"].append(("toast", f"{b['icon']} " + t("badge_unlocked", name=b["name"])))
    if res["leveled_up"]:
        ss["events"].append(("levelup", t("level_up", lvl=res["state"]["level"], rank=res["state"]["rank"])))


def run_assessment() -> None:
    sim = ss["telemetry"]
    sim.set_baseline({k: ss[f"val_{k}"] for k in SEQ_KEYS})
    payload = {
        "values": current_values(),
        "sensor_series": sim.series() if len(sim.buffers["heart_rate"]) > 8 else None,
        "subjective_log": ss["subj_log"], "mission_day": float(ss["mission_day"]),
        "gravity": GRAVITY_PRESETS[ss["gravity_name"]], "ecg": ss.get("ecg_result"), "save": True,
    }
    with st.spinner(t("analyzing")):
        ss["assessment"] = client.assess(ss["aid"], payload, lang())
    n = len(ss["assessment"]["new_tasks"])
    if n:
        ss["events"].append(("toast", f"🎯 {n} new mitigation task(s) generated"))


# ---------------------------------------------------------------------- one-time load
if ss["assessment"] is None:
    try:
        ss["assessment"] = client.latest(ss["aid"])
    except Exception:
        pass

# -------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div class="ag-hero"><h1>🛰️ {t("app_title")}</h1><p>{esc(t("tagline"))}</p></div>', unsafe_allow_html=True)
    st.selectbox(t("language"), ["en", "tr"], key="lang", format_func=lambda x: {"en": "English", "tr": "Türkçe"}[x])
    st.text_input(t("astronaut_id"), key="aid")
    st.markdown(f"**{t('environment')}**")
    st.selectbox(t("gravity"), list(GRAVITY_PRESETS), key="gravity_name")
    st.slider(t("mission_day"), 0, 180, key="mission_day")
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
        st.markdown(f"{'🟢' if m['loaded'] else '🟡'} {t('model')}: `{m['version'] or 'rules-only'}`")
        gem = (t("online") if g["online"] else (t("offline") if g["configured"] else t("not_configured")))
        st.markdown(f"{'🟢' if g['online'] else '🟠'} {t('gemini')}: {gem}")
        st.markdown(f"{'🟢' if e['service_running'] else '🟠'} {t('ecg_engine')}: `{e['mode']}`")
        st.caption(f"backend: {client.mode}")
    except Exception as exc:
        st.caption(f"status unavailable: {exc}")
    st.caption(t("disclaimer"))

# ------------------------------------------------------------------------ events
for kind, msg in ss["events"]:
    if kind == "toast":
        st.toast(msg)
    elif kind == "levelup":
        st.balloons()
        st.success(msg)
    elif kind == "error":
        st.error(msg)
    else:
        st.info(msg)
ss["events"] = []

st.markdown(f'<div class="ag-hero"><h1>🛰️ {t("app_title")}</h1><p>{esc(t("tagline"))}</p></div>', unsafe_allow_html=True)
tabs = st.tabs([t("tab_dashboard"), t("tab_input"), t("tab_ecg"), t("tab_missions"), t("tab_trends"), t("tab_advisor"), t("tab_about")])
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
            st.markdown(f"#### {dot(o['level'], True)}{t('overall_risk')}", unsafe_allow_html=True)
            st.plotly_chart(C.gauge(o), use_container_width=True, key="gauge")
            st.caption(f"{t('uncertainty')}: {o['ci_low']:.0f}–{o['ci_high']:.0f} % · {C.level_label(o['level'])}")
        with c2:
            st.markdown(f"#### {t('forecast')}")
            if A["forecast"]:
                st.plotly_chart(C.forecast_fig(A["forecast"]), use_container_width=True, key="fc_dash")
        with c3:
            st.markdown(f"#### {t('briefing')}")
            st.write(A["advisor"]["summary"])
            for r in A["advisor"]["recommendations"]:
                st.markdown(f"- {r}")
            st.caption(f"source: {A['advisor']['source']}")
        st.markdown(C.legend_html(), unsafe_allow_html=True)
        st.markdown(f"#### {t('systems')}")
        C.systems_grid(st, A)
        st.markdown(f"#### {t('parameters')}")
        by_cat = {}
        for p in A["parameters"]:
            by_cat.setdefault(p["category"], []).append(p)
        for cat, params in by_cat.items():
            worst = max((p["risk_pct"] or 0) for p in params)
            label = CATEGORIES[cat]["tr" if lang() == "tr" else "en"]
            with st.expander(f"{CATEGORIES[cat]['icon']} {label}  ·  max {worst:.0f}%", expanded=worst > 40):
                st.markdown(C.parameter_table_html(params), unsafe_allow_html=True)

# ============================================================================ DATA ENTRY
with tabs[1]:
    top = st.columns([2, 1.2, 1.2])
    top[0].selectbox(t("demo_preset"), SCENARIO_NAMES, key="preset_name")
    top[1].button(t("apply_preset"), on_click=cb_apply_preset, key="apply_preset")
    top[2].button(t("run_assessment"), type="primary", key="run_input", on_click=run_assessment)
    st.markdown(C.legend_html(), unsafe_allow_html=True)

    for cat in CATEGORIES:
        params = params_in_category(cat)
        label = CATEGORIES[cat]["tr" if lang() == "tr" else "en"]
        worst = max(risk_percent(p.key, ss[f"val_{p.key}"]) or 0 for p in params)
        with st.expander(f"{CATEGORIES[cat]['icon']} {label}", expanded=worst > 40 or cat in ("cardiovascular", "respiratory")):
            cols = st.columns(3)
            for i, p in enumerate(params):
                with cols[i % 3]:
                    c_dot, c_in = st.columns([1, 8])
                    lv = risk_level(risk_percent(p.key, ss[f"val_{p.key}"]))
                    c_dot.markdown(f"<div style='margin-top:34px'>{dot(lv, True)}</div>", unsafe_allow_html=True)
                    lo, hi, step = p.bounds
                    unit = f" ({p.unit})" if p.unit else ""
                    c_in.number_input(f"{p.name(lang())}{unit}", min_value=float(lo), max_value=float(hi), step=float(step),
                                      format=f"%.{p.decimals}f", key=f"val_{p.key}",
                                      help=f"{t('normal')}: {p.normal_range_text()}. {p.notes}")
                    c_in.caption(f"{t('normal')}: {p.normal_range_text()}")

    st.text_area(t("subjective_log"), key="subj_log", height=90,
                 placeholder="e.g. Slept badly, a bit dizzy and thirsty… / Kötü uyudum, başım dönüyor…")

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
            cells.append(f"{dot(risk_level(risk_percent(k, v)))}<span class='ag-muted'>{esc(p.name(lang()))}</span> "
                         f"<b>{v:.{p.decimals}f}</b> <span class='ag-muted'>{esc(p.unit)}</span>")
        st.markdown(" &nbsp;·&nbsp; ".join(cells), unsafe_allow_html=True)
        st.plotly_chart(C.telemetry_fig(s.series()), use_container_width=True, key=f"tele_{s.t}")

    # The fragment re-runs by itself every 2 s while streaming is on; otherwise it is static.
    st.fragment(run_every=2 if ss.get("stream_on") else None)(_telemetry_body)()

# ===================================================================================== ECG
with tabs[2]:
    st.markdown(f"### {t('ecg_title')}")
    try:
        es = client.ecg_status()
        st.caption(f"{t('ecg_mode')}: **{es['mode']}** · package: `{es['package_path'] or 'not configured (set ECG_PACKAGE_PATH)'}`")
    except Exception as exc:
        st.caption(f"ECG status unavailable: {exc}")
    c1, c2 = st.columns(2)
    up = c1.file_uploader(t("ecg_upload"), type=["npy", "npz", "csv", "mat", "dat"], key="ecg_file")
    if c1.button(t("ecg_analyze"), disabled=up is None, key="ecg_go"):
        with st.spinner("ECG analysis…"):
            try:
                ss["ecg_result"] = client.ecg_analyze(up.getvalue(), up.name)
            except Exception as exc:
                st.error(f"ECG analysis failed: {exc}")
    sample = c2.selectbox(t("ecg_sample"), ["NORMAL", "AFIB", "AFL", "LBBB", "RBBB", "MI", "PVC", "PAC", "AVB", "STTC"], key="ecg_sample_cls")
    if c2.button("▶ " + t("ecg_analyze"), key="ecg_sample_go"):
        with st.spinner("ECG analysis… (the first call may start the ECG model)"):
            try:
                ss["ecg_result"] = client.ecg_sample(sample)
            except Exception as exc:
                st.error(f"ECG analysis failed: {exc}")
    R = ss.get("ecg_result")
    if R:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Diagnosis", R["class_code"], f"{R['confidence_pct']:.0f}% conf.")
        k2.metric("Heart rate", "—" if R["heart_rate"] is None else f"{R['heart_rate']:.0f} bpm")
        k3.metric("QTc", "—" if not R["qtc_ms"] else f"{R['qtc_ms']:.0f} ms")
        k4.metric("PR / QRS", f"{R['pr_ms'] or '—'} / {R['qrs_ms'] or '—'} ms")
        k5.metric("Abnormal prob.", f"{R['abnormal_probability_pct']:.0f} %")
        st.markdown(f"<span class='ag-chip' style='border-color:{R['severity_color']}'>{esc(R['severity_label'])}</span>"
                    f"<span class='ag-chip'>{esc(R['model_version'])}</span><span class='ag-chip'>{esc(R['source'])}</span>"
                    f"{'<span class=ag-chip style=border-color:#e74c3c>⚠ STEMI suspected</span>' if R['stemi_suspected'] else ''}",
                    unsafe_allow_html=True)
        st.write(f"**{R['class_name']}** - {R['medical_desc']}")
        if R["lead_ii"]:
            st.plotly_chart(C.ecg_fig(R["lead_ii"]), use_container_width=True, key="ecg_plot")
        if R["probabilities"]:
            st.plotly_chart(C.bars({p["name"] or p["code"]: float(p["pct"]) for p in R["probabilities"]}, "Top class probabilities (%)"),
                            use_container_width=True, key="ecg_probs")
        st.caption(R["notes"])
        st.button(t("ecg_apply"), on_click=cb_apply_ecg, type="primary", key="ecg_apply_btn")

# ===================================================================== MISSIONS & REWARDS
with tabs[3]:
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
            p = next(q for q in PARAMETERS if q.key == tk["parameter_key"])
            st.markdown(f"<div class='ag-task-meta'>{esc(tk['description'])}<br>⏱ {tk['duration_min']} {t('min')} · "
                        f"{t('from_param')}: {esc(p.name(lang()))} ({tk['risk_pct']:.0f}%) · "
                        f"<span class='ag-chip'>{esc(tk['tag'])}</span><span class='ag-chip'>{esc(tk['source'])}</span></div>",
                        unsafe_allow_html=True)
        c2.markdown(f"<div style='text-align:right;margin-top:6px'><b>+{tk['points']}</b> {t('pts')}</div>", unsafe_allow_html=True)
    if done:
        with st.expander(f"✅ {t('completed_tasks')} ({len(done)})"):
            for tk in done[:50]:
                st.markdown(f"~~{esc(tk['title'])}~~ · +{tk['points']} {t('pts')} · {esc(tk['completed_at'] or '')}")
    st.markdown(f"### {t('badges')} ({sum(b['earned'] for b in gam['badges'])}/{len(gam['badges'])})")
    st.markdown(C.badges_html(gam["badges"]), unsafe_allow_html=True)

# ================================================================== PHYSIOLOGY & TRENDS
with tabs[4]:
    if not A:
        st.info(t("no_assessment"))
    else:
        st.markdown(f"### {t('history')}")
        hist = client.history(ss["aid"])
        if hist:
            import plotly.graph_objects as go

            fig = go.Figure(go.Scatter(x=[h["ts"] for h in hist], y=[h["overall_risk_pct"] for h in hist], mode="lines+markers",
                                       line=dict(color="#8ab4f8", width=3)))
            fig.update_yaxes(range=[0, 100], gridcolor="rgba(255,255,255,.08)")
            fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"),
                              margin=dict(l=10, r=10, t=10, b=10))
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
            st.caption("Hybrid physics + learned-correction model; constants are illustrative and not clinically validated.")
        c1, c2 = st.columns(2)
        if A["modality_attention"]:
            c1.plotly_chart(C.bars({k: round(v * 100, 1) for k, v in A["modality_attention"].items()}, t("attention") + " (%)", "#c58af9"),
                            use_container_width=True, key="attn")
        if A["trend_drivers"]:
            c2.plotly_chart(C.bars({d["name"]: round(d["weight"] * 100, 1) for d in A["trend_drivers"]}, t("drivers") + " (%)", "#34a853"),
                            use_container_width=True, key="drivers")

# ============================================================================ AI ADVISOR
with tabs[5]:
    st.markdown(f"### {t('advisor_title')}")
    if A:
        st.info(A["advisor"]["summary"])
        for r in A["advisor"]["recommendations"]:
            st.markdown(f"- {r}")
        st.caption(f"source: **{A['advisor']['source']}** · Gemini model: `{A['advisor']['gemini']['model']}`"
                   + (f" · last error: {A['advisor']['gemini']['last_error']}" if A["advisor"]["gemini"].get("last_error") else ""))
    for who, msg in ss["chat"]:
        st.chat_message(who).write(msg)
    q = st.chat_input(t("ask"))
    if q:
        ss["chat"].append(("user", q))
        res = client.chat(ss["aid"], q, lang())
        ss["chat"].append(("assistant", f"{res['answer']}\n\n*({res['source']})*"))
        st.rerun()

# ================================================================================== ABOUT
with tabs[6]:
    st.markdown(Path(__file__).resolve().parent.parent.joinpath("docs", "ABOUT_UI.md").read_text(encoding="utf-8")
                if Path(__file__).resolve().parent.parent.joinpath("docs", "ABOUT_UI.md").exists() else "AstroGuard-X")
