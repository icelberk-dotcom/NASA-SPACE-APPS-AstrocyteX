"""Visual theme: CSS for the space-mission look and the coloured risk indicator circles."""
from __future__ import annotations

import html

import streamlit as st

from utils.reference_ranges import LEVEL_INFO

CSS = """
<style>
:root { --ag-card: rgba(14,28,68,0.68); --ag-border: rgba(165,198,255,0.24); }
.stApp { background: radial-gradient(1100px 600px at 12% -10%, rgba(78,125,255,0.22) 0%, transparent 60%),
                      radial-gradient(900px 520px at 92% 5%, rgba(140,92,255,0.16) 0%, transparent 58%),
                      linear-gradient(to bottom, #08122e 0%, #11214d 55%, #1a3068 100%); }
[data-testid="stSidebar"] { background: rgba(8,17,44,0.92); border-right: 1px solid rgba(165,198,255,0.22); }
h1, h2, h3 { letter-spacing: .3px; text-shadow: 0 0 16px rgba(150,185,255,0.35); }
.ag-hero { padding: 6px 0 10px 0; }
.ag-hero h1 { font-size: 2.15rem; margin: 0; background: linear-gradient(90deg,#ffffff,#9ec5ff,#c58af9); -webkit-background-clip: text;
              -webkit-text-fill-color: transparent; }
.ag-hero p { margin: 0; color: #b8cde8; }
.ag-card { background: var(--ag-card); border: 1px solid var(--ag-border); border-radius: 14px; padding: 12px 14px; backdrop-filter: blur(8px); }
.ag-dot { display: inline-block; width: 14px; height: 14px; border-radius: 50%; vertical-align: middle; margin-right: 8px;
          border: 1px solid rgba(255,255,255,.25); flex: 0 0 auto; }
.ag-dot.gray   { background: #9aa0a6; }
.ag-dot.green  { background: #2ecc71; box-shadow: 0 0 8px rgba(46,204,113,.55); }
.ag-dot.yellow { background: #f1c40f; box-shadow: 0 0 8px rgba(241,196,15,.55); }
.ag-dot.red    { background: #e74c3c; box-shadow: 0 0 9px rgba(231,76,60,.7); }
.ag-dot.black  { background: #000; border: 2px solid #fff; box-shadow: 0 0 10px rgba(255,255,255,.55); animation: agpulse 1.3s infinite; }
.ag-dot.lg { width: 22px; height: 22px; }
@keyframes agpulse { 0% { box-shadow: 0 0 0 0 rgba(255,255,255,.6);} 70% { box-shadow: 0 0 0 9px rgba(255,255,255,0);} 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0);} }
table.ag-table { width: 100%; border-collapse: collapse; font-size: .92rem; }
table.ag-table th { text-align: left; color: #9aa0a6; font-weight: 500; padding: 6px 8px; border-bottom: 1px solid var(--ag-border); }
table.ag-table td { padding: 7px 8px; border-bottom: 1px solid rgba(255,255,255,.05); vertical-align: middle; }
.ag-bar { position: relative; height: 8px; background: rgba(255,255,255,.08); border-radius: 6px; min-width: 90px; overflow: hidden; }
.ag-bar > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }
.ag-muted { color: #9aa0a6; font-size: .82rem; }
.ag-sys { text-align: center; }
.ag-sys .big { font-size: 1.5rem; font-weight: 700; }
.ag-chip { display:inline-block; padding: 1px 9px; border-radius: 10px; font-size: .75rem; border: 1px solid var(--ag-border); margin-right: 4px; }
.ag-xpbar { height: 16px; background: rgba(255,255,255,.08); border-radius: 10px; overflow: hidden; border: 1px solid var(--ag-border); }
.ag-xpbar > div { height: 100%; background: linear-gradient(90deg,#34a853,#8ab4f8,#c58af9); border-radius: 10px; transition: width .6s ease; }
.ag-badge { text-align: center; padding: 10px 4px; border-radius: 12px; border: 1px solid var(--ag-border); background: var(--ag-card); height: 100%; }
.ag-badge.locked { opacity: .35; filter: grayscale(1); }
.ag-badge .ico { font-size: 1.8rem; }
.ag-task-meta { color:#9aa0a6; font-size:.8rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def dot(level: str, large: bool = False, title: str = "") -> str:
    """HTML for a small coloured circle (gray / green / yellow / red / black)."""
    cls = f"ag-dot {level}{' lg' if large else ''}"
    return f'<span class="{cls}" title="{html.escape(title or LEVEL_INFO[level]["label"])}"></span>'


def risk_bar(risk_pct: float | None, level: str) -> str:
    color = LEVEL_INFO[level]["color"] if level != "black" else "#ffffff"
    w = 0 if risk_pct is None else max(2.0, min(100.0, risk_pct))
    return f'<div class="ag-bar"><span style="width:{w}%;background:{color}"></span></div>'


def esc(s: object) -> str:
    return html.escape(str(s))
