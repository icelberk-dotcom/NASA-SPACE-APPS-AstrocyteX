# Third-party software & attribution

AstroGuard-X stands on the shoulders of open-source projects and published research. All
libraries are used unmodified through their public APIs. Licences are those published by
the respective projects at the time of writing - please check them upstream.

## Libraries

| Project | Used for | Where | Licence |
|---|---|---|---|
| [PyTorch](https://pytorch.org) | every neural network, training, Monte-Carlo inference | `models/`, `utils/` | BSD-3-Clause |
| [mamba-ssm](https://github.com/state-spaces/mamba) (Dao & Gu) | optional fused CUDA Mamba-2 kernel (`requirements-gpu.txt`); the chunked SSD algorithm in `models/mamba2.py` is a PyTorch re-implementation of the *minimal SSD* reference listing | `models/mamba2.py` | Apache-2.0 |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | optional multilingual sentence encoder for the subjective log (`requirements-nlp.txt`) | `utils/subjective.py` | Apache-2.0 |
| [Google GenAI SDK (`google-genai`)](https://github.com/googleapis/python-genai) | Gemini reasoning, structured task generation, chat | `backend/gemini_agent.py` | Apache-2.0 |
| [Streamlit](https://streamlit.io) | dashboard UI | `ui/` | Apache-2.0 |
| [Plotly](https://plotly.com/python/) | charts, gauge | `ui/components.py` | MIT |
| [scikit-learn](https://scikit-learn.org) | evaluation metrics (MAE, accuracy, macro-F1, Cohen's κ) | `models/train.py` | BSD-3-Clause |
| [NumPy](https://numpy.org) · [SciPy](https://scipy.org) · [pandas](https://pandas.pydata.org) | numerics, ECG peak detection, CSV parsing | everywhere | BSD-3-Clause |
| [FastAPI](https://fastapi.tiangolo.com) · [Starlette](https://www.starlette.io) · [Pydantic](https://docs.pydantic.dev) · [Uvicorn](https://www.uvicorn.org) | REST API, validation | `backend/` | MIT / BSD-3-Clause |
| [Requests](https://requests.readthedocs.io) | HTTP client for the ECG service and REST client | `backend/ecg_adapter.py`, `ui/client.py` | Apache-2.0 |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` configuration | `utils/config.py` | BSD-3-Clause |
| [dulwich](https://www.dulwich.io) | optional pure-Python Git bootstrap | `scripts/git_init_fallback.py` | Apache-2.0 / GPL-2.0+ (dual) |
| [pytest](https://pytest.org) · [HTTPX](https://www.python-httpx.org) | tests | `tests/` | MIT / BSD-3-Clause |

## Research the models are based on

* T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2), ICML 2024.
* S. Bai, J. Z. Kolter, V. Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (TCN), 2018.
* B. Lim, S. Ö. Arık, N. Loeff, T. Pfister. *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, Int. J. Forecasting 2021.
* A. Jaegle et al. *Perceiver: General Perception with Iterative Attention*, ICML 2021 (latent cross-attention bottleneck).
* A. Vaswani et al. *Attention Is All You Need*, NeurIPS 2017.
* R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations*, NeurIPS 2018.
* M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks*, J. Comput. Phys. 2019.
* C. Blundell et al. *Weight Uncertainty in Neural Networks* (Bayes by Backprop), ICML 2015.
* Y. Gal, Z. Ghahramani. *Dropout as a Bayesian Approximation*, ICML 2016.
* A. Kendall, Y. Gal. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*, NeurIPS 2017.
* P. E. McSharry et al. *A dynamical model for generating synthetic electrocardiogram signals*, IEEE TBME 2003 (synthetic ECG demo signal).
* NASA Human Research Program evidence reports and NASA-STD-3001 (reference ranges, bone/muscle/fluid-shift trends).

## TEKNOFEST ECG analyser (not redistributed)

The ECG integration adapter (`backend/ecg_adapter.py`) connects to the **TEKNOFEST 2026 ECG AI system
(team Ayyıldız_AI, "OmniNet Apex-Titan" V23)**, created by the project author with Antigravity.
Its source code and 138 MB model weights are **not** part of this repository (GitHub's file-size limit);
the adapter launches/uses the package from your machine (`ECG_PACKAGE_PATH`). All rights to that package
remain with its authors.

## NASA

This project was created for the **NASA Space Apps Challenge**. It is not endorsed by NASA.
"NASA" and related marks are the property of the United States government.
