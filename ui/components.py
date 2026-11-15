"""Reusable telemetry widgets (Plotly figures + zero-emoji HTML fragments) for AstrocyteX."""
from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.i18n import lang, t
from ui.theme import dot, esc, risk_bar
from utils.reference_ranges import PARAM_BY_KEY

BAND_STEPS = [
    {"range": [0, 15], "color": "#443b5c"},
    {"range": [15, 40], "color": "#236e51"},
    {"range": [40, 70], "color": "#87671f"},
    {"range": [70, 90], "color": "#8c2c38"},
    {"range": [90, 100], "color": "#130921"},
]
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#eee9f8", family="Inter, Segoe UI, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def level_label(level: str) -> str:
    return t(f"level_{level}")


def legend_html() -> str:
    items = [
        ("gray", "< 15%"),
        ("green", "15-40%"),
        ("yellow", "40-70%"),
        ("red", "70-90%"),
        ("black", "> 90%"),
    ]
    return " &nbsp;&middot;&nbsp; ".join(
        f'{dot(lv)}<span class="ag-muted">{rng} {esc(level_label(lv))}</span>' for lv, rng in items
    )


def gauge(overall: Dict[str, Any], height: int = 240) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall["risk_pct"],
            number={"suffix": "%", "font": {"size": 40 if height >= 230 else 32, "color": "#f4f0ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9f93ba"},
                "bar": {"color": "#b89cff", "thickness": 0.24},
                "steps": BAND_STEPS,
                "borderwidth": 1,
                "bordercolor": "rgba(196,173,255,0.20)",
                "threshold": {"line": {"color": "#f4f0ff", "width": 3}, "thickness": 0.85, "value": overall["risk_pct"]},
            },
        )
    )
    fig.update_layout(height=height, **{**_LAYOUT, "margin": dict(l=18, r=18, t=18, b=0)})
    return fig


def forecast_fig(forecast: List[float]) -> go.Figure:
    fig = go.Figure()
    for lo, hi, c in [(0, 15, "#8e86a6"), (15, 40, "#3cbf88"), (40, 70, "#d9a938"), (70, 90, "#e05260"), (90, 100, "#24113d")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.10, line_width=0)
    days = [f"+{i + 1}d" for i in range(len(forecast))]
    fig.add_trace(go.Scatter(x=days, y=forecast, mode="lines+markers", line=dict(color="#b89cff", width=2.8), name="risk"))
    fig.update_yaxes(range=[0, 100], title="%", gridcolor="rgba(196,173,255,.10)")
    fig.update_layout(height=240, showlegend=False, **_LAYOUT)
    return fig


def cards_grid(st, cards: Dict[str, Dict[str, Any]], n_cols: int = 5) -> None:
    """Grid of risk cards (3-letter code, label, coloured indicator, %)."""
    cols = st.columns(n_cols)
    for i, (_key, s) in enumerate(cards.items()):
        label = s["label_tr"] if lang() == "tr" else s["label"]
        with cols[i % n_cols]:
            st.markdown(
                f'<div class="ag-card ag-sys"><div class="code">{esc(s["icon"])}</div>'
                f'<div class="ag-muted" style="margin:2px 0 6px 0">{esc(label)}</div>'
                f'{dot(s["level"], True)}<span class="big">{s["risk_pct"]:.0f}%</span></div>',
                unsafe_allow_html=True,
            )


def systems_grid(st, assessment: Dict[str, Any]) -> None:
    cards_grid(st, assessment["systems"])


def parameter_table_html(params: List[Dict[str, Any]]) -> str:
    rows = [
        f"<tr><th></th><th>{esc(t('parameters'))}</th><th>{esc(t('value'))}</th><th>{esc(t('normal'))}</th>"
        f"<th style='min-width:140px'>{esc(t('risk'))}</th><th>{esc(t('uncertainty'))}</th></tr>"
    ]
    for p in params:
        name = p["name_tr"] if lang() == "tr" else p["name"]
        val = "-" if p["value"] is None else f"{p['value']:.{PARAM_BY_KEY[p['key']].decimals}f} {esc(p['unit'])}"
        risk = "-" if p["risk_pct"] is None else f"{p['risk_pct']:.0f}%"
        ci = "" if p["ci_low"] is None else f"{p['ci_low']:.0f}-{p['ci_high']:.0f}%"
        title = (
            f"rule: {p['rule_risk_pct']}% | model: {p['model_risk_pct']}%"
            if p["model_risk_pct"] is not None
            else f"rule: {p['rule_risk_pct']}%"
        )
        arrow = {"low": "LOW", "high": "HIGH"}.get(p["direction"], "") if p["level"] not in ("gray", "green") else ""
        dir_tag = f" <span class='ag-chip'>{arrow}</span>" if arrow else ""
        rows.append(
            f"<tr title='{esc(title)}'><td style='width:28px'>{dot(p['level'], title=level_label(p['level']))}</td>"
            f"<td>{esc(name)}</td><td class='ag-mono'><b>{val}</b>{dir_tag}</td><td class='ag-muted ag-mono'>{esc(p['normal_range'])}</td>"
            f"<td>{risk_bar(p['risk_pct'], p['level'])}<span class='ag-muted ag-mono'>{risk}</span></td>"
            f"<td class='ag-muted ag-mono'>{ci}</td></tr>"
        )
    return f"<table class='ag-table'>{''.join(rows)}</table>"


def xp_card_html(g: Dict[str, Any]) -> str:
    pct = int(g["progress"] * 100)
    return (
        f'<div class="ag-card"><div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<b>{esc(t("level"))} {g["level"]} &middot; {esc(g["rank"])}</b>'
        f'<span class="ag-muted ag-mono">{g["streak_days"]} {esc(t("streak"))}</span></div>'
        f'<div class="ag-xpbar" style="margin:8px 0 4px 0"><div style="width:{pct}%"></div></div>'
        f'<span class="ag-muted ag-mono">{g["xp"]} / {g["next_level_xp"]} {esc(t("xp"))}</span></div>'
    )


def badges_html(badges: List[Dict[str, Any]]) -> str:
    cells = "".join(
        f'<div style="flex:1 1 155px;max-width:210px"><div class="ag-badge{"" if b["earned"] else " locked"}" title="{esc(b["description"])}">'
        f'<div class="ico">{esc(b["icon"])}</div><div><b style="font-size:.84rem">{esc(b["name"])}</b></div>'
        f'<div class="ag-muted">{esc(b["description"])}</div></div></div>'
        for b in badges
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:10px">{cells}</div>'


def voice_summary_html(v: Dict[str, Any]) -> str:
    label = v.get("state_label_tr") if lang() == "tr" else v.get("state_label_en")
    label = label or v.get("psychological_state", "-")
    lv = v.get("level", "gray")
    risk = float(v.get("risk_pct", 0.0))
    ci_lo = float(v.get("ci_low", risk))
    ci_hi = float(v.get("ci_high", risk))
    mood = v.get("mood_dimensions") or {}
    ac = v.get("acoustic_metrics") or {}
    return (
        f'<div class="ag-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
        f'<div><span class="ag-chip">VOCAL BIOMARKER</span> <b>{esc(label)}</b></div>'
        f'<div class="ag-mono">{dot(lv, True)}<b>{risk:.1f}%</b> <span class="ag-muted">({ci_lo:.0f}-{ci_hi:.0f}%)</span></div>'
        f'</div>'
        f'<div class="ag-muted ag-mono">'
        f'Valence: {mood.get("valence_pct", 0):.0f}% &middot; '
        f'Arousal: {mood.get("arousal_pct", 0):.0f}% &middot; '
        f'Stress: {mood.get("stress_pct", 0):.0f}% &middot; '
        f'Fatigue: {mood.get("fatigue_pct", 0):.0f}%<br>'
        f'F0: {ac.get("f0_mean_hz", 0):.1f} Hz &middot; '
        f'Jitter: {ac.get("jitter_pct", 0):.2f}% &middot; '
        f'Shimmer: {ac.get("shimmer_pct", 0):.2f}% &middot; '
        f'HNR: {ac.get("hnr_db", 0):.1f} dB'
        f'</div></div>'
    )


def physics_fig(physics: Dict[str, Any]) -> go.Figure:
    curve = physics["curve"]
    x = [c["day"] for c in curve]
    specs = [
        ("bone_density_change_pct", "Bone density delta %", "#b89cff"),
        ("muscle_mass_change_pct", "Muscle mass delta %", "#3cbf88"),
        ("plasma_volume_change_pct", "Plasma volume delta %", "#d9a938"),
        ("fluid_shift_L", "Cephalad fluid shift (L)", "#e07be0"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in specs])
    for i, (key, _title, color) in enumerate(specs):
        fig.add_trace(
            go.Scatter(x=x, y=[c[key] for c in curve], mode="lines", line=dict(color=color, width=2.6), showlegend=False),
            row=i // 2 + 1,
            col=i % 2 + 1,
        )
    fig.update_xaxes(title_text="mission day", gridcolor="rgba(196,173,255,.08)")
    fig.update_yaxes(gridcolor="rgba(196,173,255,.08)")
    fig.update_layout(height=420, **_LAYOUT)
    return fig


def ecg_fig(lead_ii: List[float], fs: int = 500) -> go.Figure:
    x = [i / fs for i in range(len(lead_ii))]
    fig = go.Figure(go.Scatter(x=x, y=lead_ii, mode="lines", line=dict(color="#b89cff", width=1.6)))
    fig.update_xaxes(title="s", gridcolor="rgba(196,173,255,.14)", dtick=0.2)
    fig.update_yaxes(title="mV", gridcolor="rgba(196,173,255,.14)")
    fig.update_layout(height=250, **_LAYOUT)
    return fig


def bars(items: Dict[str, float], title: str, color: str = "#8c6ce6") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(items.values()), y=list(items.keys()), orientation="h", marker_color=color))
    fig.update_layout(height=max(160, 36 * len(items) + 40), title=title, **{**_LAYOUT, "margin": dict(l=10, r=10, t=40, b=10)})
    fig.update_yaxes(autorange="reversed")
    return fig


def telemetry_fig(series: Dict[str, List[float]]) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Heart rate (bpm)", "SpO2 (%)", "Blood pressure (mmHg)"])
    fig.add_trace(go.Scatter(y=series["heart_rate"], line=dict(color="#e05260")), row=1, col=1)
    fig.add_trace(go.Scatter(y=series["spo2"], line=dict(color="#b89cff")), row=1, col=2)
    fig.add_trace(go.Scatter(y=series["systolic_bp"], line=dict(color="#d9a938")), row=1, col=3)
    fig.add_trace(go.Scatter(y=series["diastolic_bp"], line=dict(color="#3cbf88")), row=1, col=3)
    fig.update_yaxes(gridcolor="rgba(196,173,255,.08)")
    fig.update_layout(height=220, showlegend=False, **_LAYOUT)
    return fig
