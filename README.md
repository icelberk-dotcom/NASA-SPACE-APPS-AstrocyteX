# AstrocyteX

**Autonomous Astronaut Health, Vocal Biomarker & Spaceflight-Hazard AI Telemetry Console** — developed for the **NASA Space Apps Challenge**.

**AstrocyteX (v3.0)** fuses continuous wearable sensors, daily clinical/biochemical/nutrition inputs, crew subjective logs, 12-lead ECG, **astronaut voice acoustics & psychological mood biomarkers (`AstroVoiceNet v1.0`)**, and **the five NASA spaceflight hazards** (*radiation, isolation & confinement, distance from Earth, gravity fields, hostile/closed habitats*) into a unified, uncertainty-aware risk architecture (`AstrocyteX-Net v3.0`). When any physiological biomarker, vocal marker, or habitat parameter drifts into the yellow, red, or black band, a **role-adaptive LLM advisor** (`"I am a doctor"` clinical aerospace-medicine mode vs `"I am not a doctor"` clear everyday crew mode — Gemini online, deterministic edge reasoner offline) synthesises actionable English countermeasures and rewards crew adherence with XP, ranks, streaks, and operational badges.

> **Disclaimer — Research & demonstration software (not a medical device).** Reference ranges are illustrative, neural networks are trained on simulated spaceflight cohorts, and space-physiology constants are not clinically validated. Never use this software for real clinical diagnosis or treatment.

## Key Capabilities (v3.0)

| Subsystem | Technical Specification |
|---|---|
| **Dark-Purple Milky Way Telemetry UI** | Restrained two-column aerospace flight-medical workstation (`web/` & `ui/`) over a photorealistic view of the Milky Way galaxy from Earth (`web/milky_way_earth.jpg`), flat `1px` neutral-bordered obsidian panels, tabular monospace readouts, pure English web interface with bottom-left Quick Guide, and a strict **100% zero-emoji** design policy |
| **43 Monitored Telemetry Parameters** | **31 physiological biomarkers** (wearables, sleep, hydration, urine/saliva strips, ARED/treadmill exercise, blood chemistry, complete blood count, nutrition, 12-lead ECG) + **12 NASA RIDGE spaceflight hazards** |
| **Astronaut Voice Psychological & Mood Analysis** | Interactive browser microphone recorder (16 kHz PCM `.wav`), multi-format audio uploader (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`, `.npy`, `.csv`), and playable astronaut voice presets powered by **`AstroVoiceNet v1.0`** (`(48, 64)` log-Mel spectrogram + `42` GeMAPS-inspired prosodic biomarkers) predicting 5 psychological states, 4 mood dimensions (`valence`, `arousal`, `stress`, `fatigue`), and `0-100%` vocal risk fused into overall health risk |
| **Role-Adaptive LLM Communication** | Selectable **`"I am a doctor"` (`doctor`)** vs **`"I am not a doctor"` (`non_doctor`)** profile across both assessments and interactive chat — delivering quantitative aerospace pathophysiology (`F0`, `jitter`, `shimmer`, `HNR`, `QTc`, cephalad shift, counterfactual uplift) for physicians, or calm jargon-free guidance for crew members |
| **Upgraded Multimodal Neural Backbone** | **`AstrocyteX-Net v3.0`** (`d_model=80`, `16` Perceiver latents, `SwiGLU GatedInteractionBlock`s): Mamba-2 SSD + TCN/TFT + Environment Encoder & FiLM + 5-Modality Cross-Attention + Neural ODE/PINN + Bayes-by-Backprop & MC-Dropout residual risk head |
| **12-Lead ECG Integration** | 10-class 12-lead ECG analysis (`NORMAL`, `AFIB`, `AFL`, `LBBB`, `RBBB`, `MI`, `PVC`, `PAC`, `AVB`, `STTC`) reporting `QTc`, `PR`, `QRS`, `ST`, `STEMI Suspected` alerts, and abnormal probability via `OmniNet V23` or automatic offline 10-class screening fallback, auto-synced into overall telemetry risk |
| **Longitudinal Health History** | Persistent crew snapshot recording (`#health-history`) with comparative vertical bar charts and multi-column 43-parameter + vocal-risk longitudinal telemetry tables |
| **Edge-First Autonomy & Gamification** | Operates 100% offline on CPU or GPU under deep-space communication delays; awards severity-scaled XP, rank promotions (`Cadet` to `Fleet Admiral`), streaks, and 18 operational badges |

---

## Multimodal Architecture

```mermaid
flowchart LR
    subgraph Inputs
      S["Continuous Wearables (96x7)<br/>HR, SpO2, BP, RR, Temp, GSR"]
      D["Daily Labs & Logs (31 Bio)<br/>Sleep, Fluids, Diet, Blood, CBC"]
      V["NASA RIDGE Environment (12)<br/>Radiation, Isolation, Distance, Gravity, Cabin"]
      A["Astronaut Voice Audio (16 kHz)<br/>48-Band Log-Mel + 42 GeMAPS Prosody"]
      E["12-Lead ECG<br/>OmniNet V23 / 10-Class Screening"]
    end
    S --> M["Mamba-2 SSD Encoder"]
    D --> H["14-Day 43-Var History"]
    V --> H
    H --> T["TCN + Temporal Fusion Transformer<br/>+ 7-Day Forecast"]
    V --> EN["Environment Encoder + FiLM"]
    M --> F["5-Modality Cross-Attention<br/>16 Perceiver Latents"]
    T --> F
    D --> F
    EN --> F
    F --> G["SwiGLU GatedInteractionBlocks"]
    F --> P["Neural ODE + PINN<br/>Fluid, Plasma, Bone, Muscle"]
    P --> G
    E --> D
    G --> B["Bayesian Residual Risk Head<br/>Rule Prior + Safety Floor"]
    A --> VN["AstroVoiceNet v1.0<br/>2D Depthwise-SE-Attn + ProsodyGate"]
    B --> R["Fused Health & Vocal Risk %<br/>+ 90% Credible Interval"]
    VN --> R
    R --> L["Role-Adaptive LLM Advisor<br/>Doctor vs Non-Doctor (Gemini / Edge)"]
    L --> K["Countermeasure Protocols + XP"]
```

---

## Quick Start (Cross-Platform — No GPU or API Key Required)

Requires **Python 3.10+** (tested on Python 3.10–3.13 across Windows, macOS, and Linux). Pre-trained checkpoints (`checkpoints/astroguard_v2.pt` and `checkpoints/astrovoice_v1.pt`) are bundled in the repository and automatically run on **CUDA GPU** if available or **CPU** otherwise. No `.env` file or external API key is needed.

### Option A — Windows (PowerShell)
```powershell
git clone https://github.com/icelberk-dotcom/NASA-SPACE-APPS-AstrocyteX.git
cd NASA-SPACE-APPS-AstrocyteX
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py ui
```
Then open **`http://localhost:8501`** in your browser (interactive OpenAPI docs at `http://localhost:8501/docs`).

### Option B — macOS / Linux (Bash / Zsh)
```bash
git clone https://github.com/icelberk-dotcom/NASA-SPACE-APPS-AstrocyteX.git
cd NASA-SPACE-APPS-AstrocyteX
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py ui
```
Then open **`http://localhost:8501`** in your browser.

### Option C — Docker
```bash
docker compose up --build
```
Opens the unified Web Console + REST API on **`http://localhost:8501`**.

### Additional CLI Commands
```powershell
# Launch secondary Streamlit Telemetry Console
python run.py streamlit

# Retrain neural models & re-run held-out benchmarks
python run.py train-voice
python run.py train
python run.py evaluate

# Run full automated pytest suite (194 tests)
python run.py test
```

---

## Held-Out Performance Benchmarks

Detailed tables, 1,000-resample bootstrap 95% confidence intervals, subgroup slices, and confusion matrices are auto-generated in [`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md), [`docs/ASTROCYTEX_AI_OZET_RAPORU.md`](docs/ASTROCYTEX_AI_OZET_RAPORU.md), and [`docs/voice_evaluation.json`](docs/voice_evaluation.json):

- **`AstroVoiceNet v1.0` (1,200 unseen test voice clips, 4,000 training clips):**
  - 5-Class Psychological State **MCC = `0.9957`** (`95% CI [0.9914, 0.9989]`), **Macro-F1 = `0.9963`**, **Accuracy = `99.67%`**.
  - Continuous Voice Risk **MAE = `3.168` pts** (`RMSE = 3.948` pts, **QWK = `0.9674`**), 4-Dimension Mood **MAE = `3.683` pts**, GPU latency **`2.41 ms/clip`**.
- **`AstrocyteX-Net v3.0` (6,000 unseen crew-days = 258,000 parameter predictions, 42,000 training crew-days, 2,976,783 parameters):**
  - **All 43 parameters:** **MCC = `0.9894`** (`95% CI [0.9889, 0.9899]`), **Macro-F1 = `0.9899`**, **QWK = `0.9958`**, **Critical Recall = `99.53%`** (Under-Triage `0.47%` vs `11.97%` Rule-only), **MAE = `0.144` pts** (nearly halving v2.0's `0.274` MAE).
  - **31 physiological biomarkers only (environment ablation):** **MCC = `0.9850`** (`95% CI [0.9843, 0.9857]`, Crit Recall `98.95%`, MAE `0.185` pts) vs No-Environment ablation `0.9040` (`+0.0810` MCC gain from 12 NASA RIDGE hazard channels, `p < 0.001`), Ridge `0.9358`, and MLP `0.9374` (see [`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md)).

![Confusion matrices](docs/img/confusion_matrix.png)
![MCC by environment event](docs/img/mcc_by_event.png)
![AstroVoiceNet evaluation](docs/img/voice_evaluation.png)

---

## REST API Endpoints

| Method & Endpoint | Description |
|---|---|
| `POST /astronauts/{id}/assess?lang=en\|tr` | Full multimodal assessment (`values`, `sensor_series`, `subjective_log`, `ecg`, `voice`, `user_role`) |
| `POST /astronauts/{id}/chat?lang=en\|tr` | Role-adaptive (`doctor` / `non_doctor`) Q&A over latest crew telemetry and voice state |
| `POST /voice/analyze?lang=en\|tr&role=doctor\|non_doctor` | Upload `.wav` / `.npy` / `.csv` voice clip for `AstroVoiceNet v1.0` psychological & mood inference |
| `GET /voice/sample/{state_name}` | Synthesise & analyse playable reference astronaut voice clip (`calm_regulated`, `cognitive_fatigue`, `acute_stress`, `low_mood_isolation`, `dyspnea_hypoxic`) |
| `GET /voice/status` | Voice neural engine status & validation metrics |
| `POST /ecg/analyze` / `GET /ecg/sample/{CLASS}` | 12-lead ECG analysis (`NORMAL`, `AFIB`, `MI`, `PVC`, ...) |
| `GET /astronauts/{id}/history` / `DELETE .../history` | List or delete recorded longitudinal 43-parameter + vocal-risk Health History snapshots |
| `GET /astronauts/{id}/tasks` / `POST .../complete` | Active countermeasure protocols & idempotent XP/badge completion |
| `GET /parameters` / `GET /presets` / `GET /health` | 43-parameter registry, mission profiles, scenarios, and subsystem health |

---

## NASA Standards, Citations & Open-Source License

- **NASA & Clinical Reference Standards:** Physiological and environmental reference bands are grounded in **NASA-STD-3001** (*Space Flight Human-System Standard, Vols. 1 & 2*), **NASA Human Research Program (HRP) RIDGE** spaceflight hazard profiles, **NASA SMAC / SWMQG** spacecraft air & water quality limits, and peer-reviewed clinical guidelines (AHA/ACC/ESC, KDIGO, WHO, GeMAPS). Full parameter-by-parameter citations are documented in [`docs/PARAMETRE_KAYNAKCA.md`](docs/PARAMETRE_KAYNAKCA.md).
- **Software Stack, Peer-Reviewed Architectures & AI Disclosure:** Full software versions, open-source notices, AI tool/model disclosures (`AstrocyteX-Net v3.0`, `AstroVoiceNet v1.0`, optional Google Gemini API via `google-genai`), and 28 peer-reviewed references (Dao & Gu Mamba-2, Lim et al. TFT, Jaegle et al. Perceiver, Perez et al. FiLM, Shazeer SwiGLU, Chen et al. Neural ODE, Raissi et al. PINN, Blundell et al. Bayes-by-Backprop, Gal & Ghahramani MC-Dropout, Eyben et al. GeMAPS, Hu et al. SE-Net, Klatt Formant Synthesis, Russell Circumplex Affect, Cao et al. CORAL, Chicco & Jurman MCC) are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md).
- **License:** Released under the open-source **[MIT License](LICENSE)**.
