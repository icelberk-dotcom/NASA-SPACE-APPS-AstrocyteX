"""Visual theme: deep dark-purple space telemetry console + Milky Way backdrop for AstrocyteX."""
from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from utils.reference_ranges import LEVEL_INFO

_BG_PATH = Path(__file__).resolve().parent.parent / "web" / "milky_way_earth.jpg"


@lru_cache(maxsize=1)
def _build_css() -> str:
    bg_layer = ""
    if _BG_PATH.is_file():
        b64 = base64.b64encode(_BG_PATH.read_bytes()).decode("ascii")
        bg_layer = f", url('data:image/jpeg;base64,{b64}') center center / cover fixed no-repeat"

    return f"""
<style>
:root {{
    --ag-bg: #080411;
    --ag-card: rgba(14, 10, 24, 0.88);
    --ag-card-elev: rgba(19, 14, 32, 0.94);
    --ag-border: rgba(255, 255, 255, 0.10);
    --ag-border-strong: rgba(255, 255, 255, 0.20);
    --ag-text: #efeff4;
    --ag-muted: #9e9bae;
    --ag-accent: #dcd9ea;
}}
.stApp {{
    background:
        linear-gradient(180deg, rgba(8, 4, 17, 0.72) 0%, rgba(12, 6, 25, 0.80) 52%, rgba(7, 3, 15, 0.92) 100%){bg_layer};
    color: var(--ag-text);
}}
[data-testid="stSidebar"] {{
    background: rgba(9, 6, 17, 0.95);
    border-right: 1px solid var(--ag-border);
}}
h1, h2, h3, h4 {{
    letter-spacing: -0.01em;
    color: var(--ag-text);
    font-weight: 600;
}}
.ag-hero {{
    padding: 4px 0 10px 0;
    border-bottom: 1px solid var(--ag-border);
    margin-bottom: 10px;
}}
.ag-hero .ag-kicker {{
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ag-muted);
}}
.ag-hero h1 {{
    font-size: 1.55rem;
    margin: 2px 0;
    color: #efeff4;
    font-weight: 600;
    letter-spacing: -0.02em;
}}
.ag-hero p {{
    margin: 0;
    color: var(--ag-muted);
    font-size: 0.84rem;
}}
.ag-card {{
    background: var(--ag-card);
    border: 1px solid var(--ag-border);
    border-radius: 4px;
    padding: 12px 14px;
}}
.ag-dot {{
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    vertical-align: middle;
    margin-right: 8px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    flex: 0 0 auto;
}}
.ag-dot.gray   {{ background: #8e86a6; }}
.ag-dot.green  {{ background: #2eb87a; }}
.ag-dot.yellow {{ background: #d4a134; }}
.ag-dot.red    {{ background: #dc505c; }}
.ag-dot.black  {{ background: #090512; border: 2px solid #efeff4; }}
.ag-dot.lg {{ width: 13px; height: 13px; }}
table.ag-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
    font-variant-numeric: tabular-nums;
}}
table.ag-table th {{
    text-align: left;
    color: var(--ag-muted);
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 6px 8px;
    border-bottom: 1px solid var(--ag-border);
}}
table.ag-table td {{
    padding: 7px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    vertical-align: middle;
}}
.ag-bar {{
    position: relative;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 2px;
    min-width: 90px;
    overflow: hidden;
}}
.ag-bar > span {{
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 2px;
}}
.ag-muted {{
    color: var(--ag-muted);
    font-size: 0.80rem;
}}
.ag-mono {{
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-variant-numeric: tabular-nums;
}}
.ag-sys {{
    text-align: left;
}}
.ag-sys .code {{
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.70rem;
    letter-spacing: 0.06em;
    color: var(--ag-muted);
    text-transform: uppercase;
}}
.ag-sys .big {{
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 1.30rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}
.ag-chip {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 0.72rem;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    border: 1px solid var(--ag-border);
    background: rgba(255, 255, 255, 0.04);
    margin-right: 5px;
}}
.ag-xpbar {{
    height: 7px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 2px;
    overflow: hidden;
}}
.ag-xpbar > div {{
    height: 100%;
    background: #dcd9ea;
    border-radius: 2px;
    transition: width 0.4s ease;
}}
.ag-badge {{
    text-align: left;
    padding: 10px 12px;
    border-radius: 4px;
    border: 1px solid var(--ag-border);
    background: var(--ag-card);
    height: 100%;
}}
.ag-badge.locked {{
    opacity: 0.40;
}}
.ag-badge .ico {{
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--ag-text);
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--ag-border-strong);
    border-radius: 3px;
    padding: 2px 6px;
    margin-bottom: 5px;
}}
.ag-task-meta {{
    color: var(--ag-muted);
    font-size: 0.79rem;
}}
.ag-quick-guide {{
    margin-top: 10px;
    padding: 9px 10px;
    border: 1px solid var(--ag-border);
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.02);
    font-size: 0.74rem;
    line-height: 1.45;
    color: var(--ag-muted);
}}
.ag-quick-guide strong {{
    display: block;
    color: var(--ag-text);
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_build_css(), unsafe_allow_html=True)


def dot(level: str, large: bool = False, title: str = "") -> str:
    """HTML for a small coloured circle (gray / green / yellow / red / black)."""
    cls = f"ag-dot {level}{' lg' if large else ''}"
    return f'<span class="{cls}" title="{html.escape(title or LEVEL_INFO[level]["label"])}"></span>'


def risk_bar(risk_pct: float | None, level: str) -> str:
    color = LEVEL_INFO[level]["color"] if level != "black" else "#f4f0ff"
    w = 0 if risk_pct is None else max(2.0, min(100.0, risk_pct))
    return f'<div class="ag-bar"><span style="width:{w}%;background:{color}"></span></div>'


def esc(s: object) -> str:
    return html.escape(str(s))
