# Third-party software, libraries & academic attributions

**AstrocyteX** builds on open-source software packages and peer-reviewed machine learning, signal processing, and aerospace-medicine literature. All libraries are invoked unmodified through their public APIs. Licences listed below correspond to upstream project declarations.

## Software libraries & runtimes

| Software / Library | Role in AstrocyteX | Modules | Licence |
|---|---|---|---|
| [Python 3.13](https://www.python.org) | Core runtime, `wave` RIFF PCM audio codec, `sqlite3` persistence | entire project | PSF-2.0 |
| [PyTorch (`torch` + CUDA)](https://pytorch.org) | `AstrocyteX-Net v3.0` & `AstroVoiceNet v1.0` neural architectures, GPU training, Monte-Carlo Bayesian inference | `models/`, `utils/` | BSD-3-Clause |
| [mamba-ssm](https://github.com/state-spaces/mamba) | Optional fused CUDA Mamba-2 kernel; `models/mamba2.py` provides a pure-PyTorch chunked SSD implementation validated against exact recurrence | `models/mamba2.py` | Apache-2.0 |
| [NumPy](https://numpy.org) | Vectorised multi-modal simulation, STFT/Mel filterbanks, acoustic prosody extraction, bootstrap resampling | `utils/`, `models/` | BSD-3-Clause |
| [SciPy](https://scipy.org) | Signal filtering, Pan-Tompkins-style R-peak detection, spectral analysis | `utils/`, `backend/ecg_adapter.py` | BSD-3-Clause |
| [scikit-learn](https://scikit-learn.org) | Ridge, MLPRegressor, HistGradientBoosting, and RandomForest baselines; MCC, Macro-F1, and QWK verification | `models/evaluate.py`, `models/train_voice.py` | BSD-3-Clause |
| [pandas](https://pandas.pydata.org) | Tabular CSV ingestion and telemetry export | `utils/`, `backend/` | BSD-3-Clause |
| [FastAPI](https://fastapi.tiangolo.com) & [Starlette](https://www.starlette.io) | REST API endpoints (`/assess`, `/voice/*`, `/ecg/*`, `/chat`) and SPA static hosting | `backend/main.py` | MIT / BSD-3-Clause |
| [Pydantic v2](https://docs.pydantic.dev) | Strict request/response schema validation (`ReadingIn`, `Assessment`, `AdvisorOutput`) | `backend/schemas.py` | MIT |
| [Uvicorn](https://www.uvicorn.org) | ASGI server for web console and API | `run.py`, `backend/main.py` | BSD-3-Clause |
| [Google GenAI SDK (`google-genai`)](https://github.com/googleapis/python-genai) | Role-adaptive (`doctor` / `non_doctor`) Gemini clinical & crew reasoning with structured JSON output | `backend/gemini_agent.py` | Apache-2.0 |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Optional multilingual text embedding encoder for crew logs | `utils/subjective.py` | Apache-2.0 |
| [Streamlit](https://streamlit.io) & [Plotly](https://plotly.com/python/) | Secondary interactive telemetry dashboard and interactive gauges/charts | `ui/` | Apache-2.0 / MIT |
| [Matplotlib](https://matplotlib.org) | Publication-grade confusion matrices, training convergence, and voice evaluation figures | `models/evaluate.py`, `models/train_voice.py`, `scripts/generate_word_report.py` | PSF-based |
| [python-docx](https://python-docx.readthedocs.io) | Automated generation of the executive Turkish Word (`.docx`) technical & benchmark report | `scripts/generate_word_report.py` | MIT |
| [Requests](https://requests.readthedocs.io) & [python-dotenv](https://github.com/theskumar/python-dotenv) | REST client, ECG adapter IPC, and `.env` configuration | `ui/client.py`, `utils/config.py` | Apache-2.0 / BSD-3-Clause |
| [W3C Web Audio API & MediaRecorder](https://www.w3.org/TR/webaudio/) | Browser-side 16 kHz PCM `.wav` microphone capture, multi-codec audio decoding, and live spectrogram rendering | `web/app.js` | W3C |
| [pytest](https://pytest.org) & [HTTPX](https://www.python-httpx.org) | Automated unit, integration, API, and headless UI test suite | `tests/` | MIT / BSD-3-Clause |

## Peer-reviewed research & technical standards

1. T. Dao and A. Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)," *Proc. ICML*, PMLR vol. 235, 2024.
2. S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," *arXiv:1803.01271*, 2018.
3. B. Lim, S. O. Arik, N. Loeff, and T. Pfister, "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," *International Journal of Forecasting*, vol. 37, no. 4, pp. 1748-1764, 2021.
4. A. Jaegle et al., "Perceiver: General Perception with Iterative Attention," *Proc. ICML*, PMLR vol. 139, pp. 4651-4664, 2021.
5. A. Vaswani et al., "Attention Is All You Need," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
6. E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual Reasoning with a General Conditioning Layer," *Proc. AAAI*, vol. 32, no. 1, 2018.
7. N. Shazeer, "GLU Variants Improve Transformer (SwiGLU)," *arXiv:2002.05202*, 2020.
8. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, "Neural Ordinary Differential Equations," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.
9. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686-707, 2019.
10. C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, "Weight Uncertainty in Neural Networks (Bayes by Backprop)," *Proc. ICML*, PMLR vol. 37, pp. 1613-1622, 2015.
11. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," *Proc. ICML*, PMLR vol. 48, pp. 1050-1059, 2016.
12. A. Kendall and Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
13. F. Eyben et al., "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing," *IEEE Transactions on Affective Computing*, vol. 7, no. 2, pp. 190-202, 2016.
14. J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," *Proc. IEEE/CVF CVPR*, pp. 7132-7141, 2018.
15. D. H. Klatt, "Software for a cascade/parallel formant synthesizer," *Journal of the Acoustical Society of America*, vol. 67, no. 3, pp. 971-995, 1980.
16. J. A. Russell, "A circumplex model of affect," *Journal of Personality and Social Psychology*, vol. 39, no. 6, pp. 1161-1178, 1980.
17. P. E. McSharry, G. D. Clifford, L. Tarassenko, and L. A. Smith, "A dynamical model for generating synthetic electrocardiogram signals," *IEEE Transactions on Biomedical Engineering*, vol. 50, no. 3, pp. 289-294, 2003.
18. W. Cao, V. Mirjalili, and S. Raschka, "Rank consistent ordinal regression for neural networks with application to age estimation (CORAL)," *Pattern Recognition Letters*, vol. 140, pp. 325-331, 2020.
19. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization (AdamW)," *Proc. ICLR*, 2019; L. N. Smith and N. Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates (OneCycleLR)," *SPIE*, 2019.
20. B. T. Polyak and A. B. Juditsky, "Acceleration of stochastic approximation by averaging," *SIAM Journal on Control and Optimization*, vol. 30, no. 4, pp. 838-855, 1992.
21. B. W. Matthews, "Comparison of the predicted and observed secondary structure of T4 phage lysozyme," *Biochimica et Biophysica Acta*, vol. 405, no. 2, pp. 442-451, 1975; J. Gorodkin, "Comparing two K-category assignments by a K-category correlation coefficient," *Computational Biology and Chemistry*, vol. 28, pp. 367-374, 2004; D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy," *BMC Genomics*, vol. 21, art. 6, 2020.
22. J. Cohen, "Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit," *Psychological Bulletin*, vol. 70, no. 4, pp. 213-220, 1968.
23. NASA, *NASA Space Flight Human-System Standard, Volume 1: Crew Health & Volume 2: Human Factors, Habitability, and Environmental Health*, NASA-STD-3001, Washington, DC, 2022-2023.
24. Z. S. Patel et al., "Red risks for a journey to the red planet: The highest priority human health risks for a mission to Mars," *npj Microgravity*, vol. 6, art. 33, 2020; C. Zeitlin et al., *Science*, vol. 340, pp. 1080-1084, 2013; D. M. Hassler et al., *Science*, vol. 343, 1244797, 2014.
25. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.
26. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
27. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357-362, 2020; P. Virtanen et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods*, vol. 17, pp. 261-272, 2020; J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90-95, 2007.

## TEKNOFEST ECG analyser (external local package)

`backend/ecg_adapter.py` interfaces with the **TEKNOFEST 2026 ECG AI system (team Ayyildiz_AI, "OmniNet Apex-Titan" V23)** via `ECG_PACKAGE_PATH` without redistributing its 138 MB checkpoint weights. When absent, the built-in screening fallback (`builtin-screening`) provides offline R-peak, HR, and HRV estimation.

## NASA notice

Created as an independent research prototype for the **NASA Space Apps Challenge**. Not endorsed by NASA.
