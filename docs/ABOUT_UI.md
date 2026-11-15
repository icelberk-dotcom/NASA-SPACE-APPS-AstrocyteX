### About AstrocyteX

**AstrocyteX (v3.0)** is an autonomous astronaut health, vocal biomarker, and space-environment telemetry platform with an operational countermeasure and XP progression system, created for the **NASA Space Apps Challenge**.

**43 Monitored Telemetry Parameters + Vocal Biomarkers** — **31 physiological** biomarkers (continuous wearables + daily clinical inputs + 12-lead ECG) plus **12 NASA spaceflight hazard** inputs across the five RIDGE themes (*radiation, isolation & confinement, distance from Earth, gravity fields, hostile closed habitats*), augmented by **AstroVoiceNet v1.0** acoustic speech analysis (*48-band log-Mel spectrogram + 42 prosodic features*).

**Neural Architecture** — `AstrocyteX-Net v3.0` (`d_model = 80`, `16` Perceiver latents, SwiGLU `GatedInteractionBlock`s): Mamba-2 SSD continuous-sensor encoder, TCN + Temporal Fusion Transformer 14-day 43-variable trend encoder, Environment Encoder + FiLM conditioning, 5-modality cross-attention fusion, Neural ODE + PINN space-physiology solver (fluid shift, plasma volume, bone mineral density, skeletal muscle mass), Bayes-by-Backprop + MC-Dropout residual risk head, `AstroVoiceNet v1.0` psychological & affective voice analyser, role-adaptive LLM advisor (`Ben doktorum` clinical mode vs `Doktor değilim` clear crew mode), and TEKNOFEST V23 12-lead ECG integration.

**Risk Bands** — `GRAY` (< 15%, nominal baseline), `GREEN` (15-40%, low risk), `YELLOW` (40-70%, minor drift), `RED` (70-90%, high risk), `BLACK` (> 90%, critical intervention). Every yellow, red, or black finding automatically generates actionable countermeasure protocols; completing protocols awards XP, rank promotions, streaks, and operational badges.

> **Disclaimer:** Research and engineering demonstration software — not a certified medical device. Reference ranges are illustrative, neural models are trained on simulated spaceflight cohorts, and physiology constants are not clinically validated. Never use this software for real clinical decisions.

Open-source stack: PyTorch, Google GenAI SDK (`google-genai`), FastAPI, Pydantic, Uvicorn, scikit-learn, NumPy, SciPy, Matplotlib, Plotly, Streamlit, python-docx, pytest — see `THIRD_PARTY_NOTICES.md`.
