#!/usr/bin/env python
"""
AstroGuard-X launcher.

    python run.py ui         Astrocyte X · AstroGuard-X cosmic web UI + API    http://localhost:8501
    python run.py streamlit  Classic Streamlit dashboard                       http://localhost:8502
    python run.py api        FastAPI REST + web backend                        http://localhost:8000/docs
    python run.py all     REST backend + dashboard talking to it
    python run.py train   (re)train the risk model on simulated crews (GPU if available)
    python run.py evaluate  held-out test (MCC, macro-F1, QWK, safety metrics) -> docs/MODEL_EVALUATION.md
    python run.py test    run the unit tests
    python run.py ecg     check the ECG-analyser integration
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _run(cmd: list[str], env: dict | None = None) -> int:
    return subprocess.call(cmd, cwd=ROOT, env={**os.environ, **(env or {})})


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ui"
    extra = sys.argv[2:]
    if cmd in ("ui", "web"):
        return _run([PY, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8501", *extra])
    if cmd == "streamlit":
        return _run([PY, "-m", "streamlit", "run", "ui/app.py", *extra])
    if cmd == "api":
        return _run([PY, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", *extra])
    if cmd == "all":
        api = subprocess.Popen([PY, "-m", "uvicorn", "backend.main:app", "--port", "8000"], cwd=ROOT)
        try:
            time.sleep(3)
            return _run([PY, "-m", "streamlit", "run", "ui/app.py", *extra], env={"ASTROGUARD_API_URL": "http://127.0.0.1:8000"})
        finally:
            api.terminate()
    if cmd == "train":
        return _run([PY, "-u", "-m", "models.train", *extra])
    if cmd == "evaluate":
        return _run([PY, "-u", "-m", "models.evaluate", *extra])
    if cmd == "test":
        return _run([PY, "-m", "pytest", "-q", *extra])
    if cmd == "ecg":
        code = ("from backend.ecg_adapter import ECGAdapter; a=ECGAdapter(); print(a.status()); "
                "r=a.sample('AFIB'); print(r.source, r.class_code, r.heart_rate, r.abnormal_probability_pct)")
        return _run([PY, "-c", code])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
