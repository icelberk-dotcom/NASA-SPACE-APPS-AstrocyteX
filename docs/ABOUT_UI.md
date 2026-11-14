### About AstroGuard-X

**AstroGuard-X (v2)** is an AI-powered astronaut health & space-environment monitoring prototype with a built-in
gamified mitigation system, created for the **NASA Space Apps Challenge**.

**43 Parameters** — **31 physiological** biomarkers (continuous wearables + daily clinical inputs) plus **12 NASA
spaceflight hazard** inputs across the five RIDGE themes (*radiation · isolation & confinement · distance from Earth ·
gravity fields · hostile closed environments*).

**Pipeline** — Mamba-2 (continuous sensors) · TCN + Temporal Fusion Transformer (14-day 43-variable trends) ·
Environment Encoder + FiLM modulation (12 hazards) · 5-modality cross-attention fusion · Neural ODE + PINN
(fluid shift, bone & muscle dynamics driven by gravity) · Bayesian uncertainty head · Gemini reasoning (offline fallback) ·
TEKNOFEST V23 ECG analyser.

**How to read the colours** — ⚪ gray < 15 % baseline · 🟢 green 15–40 % normal · 🟡 yellow 40–70 % minor deviation ·
🔴 red 70–90 % high risk · ⚫ black > 90 % immediate intervention.
Every yellow/red/black parameter automatically gets mitigation tasks; completing them earns XP,
levels, streaks and badges.

> ⚠️ **Not a medical device.** The reference ranges are illustrative, the neural networks are trained on
> *simulated* crews and the physiology constants are not clinically validated. Never use this software for
> real medical decisions. For emergencies contact your flight surgeon.

Open-source credits: PyTorch · mamba-ssm · Hugging Face transformers · Google GenAI SDK · Streamlit · Plotly ·
scikit-learn · NumPy/SciPy/pandas · FastAPI/Pydantic/Uvicorn — see `THIRD_PARTY_NOTICES.md`.
