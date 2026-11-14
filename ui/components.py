"""Reusable dashboard widgets (plotly figures + HTML fragments)."""
from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.i18n import lang, t
from ui.theme import dot, esc, risk_bar
from utils.reference_ranges import CATEGORIES, LEVEL_INFO, PARAM_BY_KEY

BAND_STEPS = [
    {"range": [0, 15], "color": "#9aa0a6"}, {"range": [15, 40], "color": "#2ecc71"},
    {"range": [40, 70], "color": "#f1c40f"}, {"range": [70, 90], "color": "#e74c3c"},
    {"range": [90, 100], "color": "#000000"},
]
_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"),
               margin=dict(l=10, r=10, t=30, b=10))


def level_label(level: str) -> str:
    return t(f"level_{level}")


def legend_html() -> str:
    items = [("gray", "< 15 %"), ("green", "15–40 %"), ("yellow", "40–70 %"), ("red", "70–90 %"), ("black", "> 90 %")]
    return " &nbsp; ".join(f'{dot(lv)}<span class="ag-muted">{rng} · {esc(level_label(lv))}</span>' for lv, rng in items)


def gauge(overall: Dict[str, Any], height: int = 250) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=overall["risk_pct"], number={"suffix": " %", "font": {"size": 44 if height >= 240 else 34}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#9aa0a6"}, "bar": {"color": "#ffffff", "thickness": 0.22},
               "steps": BAND_STEPS, "borderwidth": 0,
               "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.9, "value": overall["risk_pct"]}}))
    fig.update_layout(height=height, **{**_LAYOUT, "margin": dict(l=20, r=20, t=20, b=0)})
    return fig


def forecast_fig(forecast: List[float]) -> go.Figure:
    fig = go.Figure()
    for lo, hi, c in [(0, 15, "#9aa0a6"), (15, 40, "#2ecc71"), (40, 70, "#f1c40f"), (70, 90, "#e74c3c"), (90, 100, "#222")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.12, line_width=0)
    days = [f"+{i + 1}d" for i in range(len(forecast))]
    fig.add_trace(go.Scatter(x=days, y=forecast, mode="lines+markers", line=dict(color="#8ab4f8", width=3), name="risk"))
    fig.update_yaxes(range=[0, 100], title="%", gridcolor="rgba(255,255,255,.08)")
    fig.update_layout(height=250, showlegend=False, **_LAYOUT)
    return fig


def cards_grid(st, cards: Dict[str, Dict[str, Any]], n_cols: int = 5) -> None:
    """Grid of risk cards (icon, label, coloured indicator, %) - physiological systems or environment themes."""
    cols = st.columns(n_cols)
    for i, (_key, s) in enumerate(cards.items()):
        label = s["label_tr"] if lang() == "tr" else s["label"]
        with cols[i % n_cols]:
            st.markdown(
                f'<div class="ag-card ag-sys">{s["icon"]}<br><span class="ag-muted">{esc(label)}</span><br>'
                f'{dot(s["level"], True)}<span class="big">{s["risk_pct"]:.0f}%</span></div>', unsafe_allow_html=True)


def systems_grid(st, assessment: Dict[str, Any]) -> None:
    cards_grid(st, assessment["systems"])


def parameter_table_html(params: List[Dict[str, Any]]) -> str:
    rows = [f"<tr><th></th><th>{esc(t('parameters'))}</th><th>{esc(t('value'))}</th><th>{esc(t('normal'))}</th>"
            f"<th style='min-width:140px'>{esc(t('risk'))}</th><th>{esc(t('uncertainty'))}</th></tr>"]
    for p in params:
        name = p["name_tr"] if lang() == "tr" else p["name"]
        val = "—" if p["value"] is None else f"{p['value']:.{PARAM_BY_KEY[p['key']].decimals}f} {esc(p['unit'])}"
        risk = "—" if p["risk_pct"] is None else f"{p['risk_pct']:.0f}%"
        ci = "" if p["ci_low"] is None else f"{p['ci_low']:.0f}–{p['ci_high']:.0f}%"
        title = (f"rule: {p['rule_risk_pct']}% · model: {p['model_risk_pct']}%" if p["model_risk_pct"] is not None
                 else f"rule: {p['rule_risk_pct']}%")
        arrow = {"low": "↓", "high": "↑"}.get(p["direction"], "") if p["level"] not in ("gray", "green") else ""
        rows.append(
            f"<tr title='{esc(title)}'><td style='width:28px'>{dot(p['level'], title=level_label(p['level']))}</td>"
            f"<td>{esc(name)}</td><td><b>{val}</b> {arrow}</td><td class='ag-muted'>{esc(p['normal_range'])}</td>"
            f"<td>{risk_bar(p['risk_pct'], p['level'])}<span class='ag-muted'>{risk}</span></td>"
            f"<td class='ag-muted'>{ci}</td></tr>")
    return f"<table class='ag-table'>{''.join(rows)}</table>"


def xp_card_html(g: Dict[str, Any]) -> str:
    pct = int(g["progress"] * 100)
    return (f'<div class="ag-card"><div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<b>{esc(t("level"))} {g["level"]} · {esc(g["rank"])}</b><span class="ag-muted">🔥 {g["streak_days"]} {esc(t("streak"))}</span></div>'
            f'<div class="ag-xpbar" style="margin:8px 0 4px 0"><div style="width:{pct}%"></div></div>'
            f'<span class="ag-muted">{g["xp"]} / {g["next_level_xp"]} {esc(t("xp"))}</span></div>')


def badges_html(badges: List[Dict[str, Any]]) -> str:
    cells = "".join(
        f'<div style="flex:1 1 150px;max-width:200px"><div class="ag-badge{"" if b["earned"] else " locked"}" title="{esc(b["description"])}">'
        f'<div class="ico">{b["icon"]}</div><b style="font-size:.85rem">{esc(b["name"])}</b>'
        f'<div class="ag-muted">{esc(b["description"])}</div></div></div>' for b in badges)
    return f'<div style="display:flex;flex-wrap:wrap;gap:10px">{cells}</div>'


def physics_fig(physics: Dict[str, Any]) -> go.Figure:
    curve = physics["curve"]
    x = [c["day"] for c in curve]
    specs = [("bone_density_change_pct", "Bone density Δ%", "#8ab4f8"), ("muscle_mass_change_pct", "Muscle mass Δ%", "#34a853"),
             ("plasma_volume_change_pct", "Plasma volume Δ%", "#f1c40f"), ("fluid_shift_L", "Cephalad fluid shift (L)", "#c58af9")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in specs])
    for i, (key, _title, color) in enumerate(specs):
        fig.add_trace(go.Scatter(x=x, y=[c[key] for c in curve], mode="lines", line=dict(color=color, width=3), showlegend=False),
                      row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(title_text="mission day", gridcolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    fig.update_layout(height=430, **_LAYOUT)
    return fig


def ecg_fig(lead_ii: List[float], fs: int = 500) -> go.Figure:
    x = [i / fs for i in range(len(lead_ii))]
    fig = go.Figure(go.Scatter(x=x, y=lead_ii, mode="lines", line=dict(color="#34a853", width=1.6)))
    fig.update_xaxes(title="s", gridcolor="rgba(255,80,80,.18)", dtick=0.2)
    fig.update_yaxes(title="mV", gridcolor="rgba(255,80,80,.18)")
    fig.update_layout(height=260, **_LAYOUT)
    return fig


def bars(items: Dict[str, float], title: str, color: str = "#8ab4f8") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(items.values()), y=list(items.keys()), orientation="h", marker_color=color))
    fig.update_layout(height=max(160, 38 * len(items) + 40), title=title, **{**_LAYOUT, "margin": dict(l=10, r=10, t=40, b=10)})
    fig.update_yaxes(autorange="reversed")
    return fig


def telemetry_fig(series: Dict[str, List[float]]) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Heart rate (bpm)", "SpO2 (%)", "Blood pressure (mmHg)"])
    fig.add_trace(go.Scatter(y=series["heart_rate"], line=dict(color="#e74c3c")), row=1, col=1)
    fig.add_trace(go.Scatter(y=series["spo2"], line=dict(color="#8ab4f8")), row=1, col=2)
    fig.add_trace(go.Scatter(y=series["systolic_bp"], line=dict(color="#f1c40f")), row=1, col=3)
    fig.add_trace(go.Scatter(y=series["diastolic_bp"], line=dict(color="#c58af9")), row=1, col=3)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    fig.update_layout(height=230, showlegend=False, **_LAYOUT)
    return fig
