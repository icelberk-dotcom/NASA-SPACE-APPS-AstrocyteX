### About AstroGuard-X

**AstroGuard-X** is an AI-powered astronaut health-monitoring prototype with a built-in gamified
mitigation system, created for the **NASA Space Apps Challenge**.

**Pipeline** — Mamba-2 (continuous sensors) · TCN + Temporal Fusion Transformer (multi-day trends) ·
cross-attention multimodal fusion (sensors + daily inputs + subjective log) · Neural ODE + PINN
(fluid shift, bone & muscle dynamics) · Bayesian uncertainty head · Gemini reasoning (offline fallback) ·
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
