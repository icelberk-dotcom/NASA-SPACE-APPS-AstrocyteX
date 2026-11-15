"""
ECG Integration Adapter.

Connects AstroGuard-X to the 12-lead ECG analyser built for TEKNOFEST 2026 (team
Ayyildiz_AI, "OmniNet Apex-Titan" V23 - 10-class cardiac rhythm / ischaemia classifier
with a Flask front-end).

The adapter keeps that package *untouched* and talks to it through its own HTTP API::

    GET  /api/health            model / device info
    POST /api/analyze           multipart file (.npy .npz .csv .mat .dat) -> full analysis
    GET  /api/sample/<CLASS>    built-in clinical demo signal for a class (NORMAL, AFIB ...)

Resolution order for every request
----------------------------------
 1. An ECG service is already running at ``ECG_SERVICE_URL``  -> use it.
 2. ``ECG_PACKAGE_PATH`` points to the package and ``ECG_AUTOSTART=1`` -> launch
    ``TEKNOFEST_EKG_Yapay_Zeka_Sistemi.py --no-browser --port <p>`` as a managed child
    process, wait for ``/api/health`` and use it.  ``ECG_PYTHON`` may select the interpreter
    that has the package's dependencies (torch, flask, scipy).
 3. Otherwise -> a built-in, clearly-labelled *screening* analyser (R-peak detection with
    SciPy: heart rate, rhythm regularity, HRV) so the platform keeps working offline.

Whatever the source, results are normalised into :class:`ECGResult` and mapped onto the
dashboard parameters (``ecg_abnormal_pct``, ``ecg_qtc_ms`` and, if no sensor value is
available, ``heart_rate``).

Open-source attribution: *requests* (Apache-2.0) for HTTP, *SciPy* (BSD-3-Clause) for
peak detection, *NumPy* (BSD-3-Clause).
"""
from __future__ import annotations

import atexit
import io
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from scipy import signal as sp_signal

from utils.config import PROJECT_ROOT, Settings, get_settings
from utils.ecg_synth import generate_ecg

log = logging.getLogger("astroguard.ecg")

ECG_ENTRYPOINT = "TEKNOFEST_EKG_Yapay_Zeka_Sistemi.py"
ECG_CLASSES = ["NORMAL", "AFIB", "AFL", "LBBB", "RBBB", "MI", "PVC", "PAC", "AVB", "STTC"]


@dataclass
class ECGResult:
    source: str  # "teknofest-ecg" | "builtin-screening"
    model_version: str
    record_name: str
    class_code: str
    class_name: str
    confidence_pct: float
    severity_label: str
    severity_color: str
    abnormal_probability_pct: float
    heart_rate: Optional[float] = None
    qtc_ms: Optional[float] = None
    pr_ms: Optional[float] = None
    qrs_ms: Optional[float] = None
    st_deviation_mv: Optional[float] = None
    rr_regularity: Optional[str] = None
    hrv_rmssd_ms: Optional[float] = None
    uncertainty_pct: Optional[float] = None
    stemi_suspected: bool = False
    medical_desc: str = ""
    probabilities: List[Dict[str, Any]] = field(default_factory=list)
    lead_ii: List[float] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def parameter_values(self) -> Dict[str, float]:
        """Dashboard parameters that this ECG recording can fill in."""
        out: Dict[str, float] = {"ecg_abnormal_pct": round(float(self.abnormal_probability_pct), 1)}
        if self.qtc_ms:
            out["ecg_qtc_ms"] = float(self.qtc_ms)
        if self.heart_rate:
            out["heart_rate"] = float(self.heart_rate)
        return out


# ------------------------------------------------------------------------------------
class ECGAdapter:
    """Thin, thread-safe client + process manager for the ECG analyser."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._last_health: Optional[Dict[str, Any]] = None
        self._last_check = 0.0

    # ---------------------------------------------------------------- discovery
    def package_path(self) -> Optional[Path]:
        candidates = [self.settings.ecg_package_path, PROJECT_ROOT / "external" / "ecg_package",
                      PROJECT_ROOT / "ecg_package"]
        for c in candidates:
            if c and (Path(c) / ECG_ENTRYPOINT).exists():
                return Path(c)
        return None

    def _health(self, force: bool = False) -> Optional[Dict[str, Any]]:
        if not force and time.time() - self._last_check < 5:
            return self._last_health
        try:
            r = requests.get(f"{self.settings.ecg_service_url}/api/health", timeout=1.5)
            self._last_health = r.json() if r.ok else None
        except Exception:
            self._last_health = None
        self._last_check = time.time()
        return self._last_health

    def status(self) -> Dict[str, Any]:
        health = self._health(force=True)
        pkg = self.package_path()
        if health:
            mode = "teknofest-ecg"
        elif pkg and self.settings.ecg_autostart:
            mode = "teknofest-ecg (starts on first use)"
        else:
            mode = "builtin-screening"
        return {"mode": mode, "service_running": bool(health), "service_url": self.settings.ecg_service_url,
                "package_path": str(pkg) if pkg else None, "health": health,
                "autostart": self.settings.ecg_autostart}

    # ----------------------------------------------------------- process control
    def start_service(self) -> bool:
        """Launch the Teknofest ECG server as a child process and wait until it is healthy."""
        with self._lock:
            if self._health(force=True):
                return True
            pkg = self.package_path()
            if pkg is None:
                return False
            if self._proc is None or self._proc.poll() is not None:
                log_path = PROJECT_ROOT / "data" / "ecg_service.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
                py = self.settings.ecg_python or sys.executable
                cmd = [py, ECG_ENTRYPOINT, "--no-browser", "--port", str(self.settings.ecg_port)]
                log.info("Starting ECG service: %s (cwd=%s)", " ".join(cmd), pkg)
                self._proc = subprocess.Popen(cmd, cwd=str(pkg), env=env, stdout=open(log_path, "ab"),
                                              stderr=subprocess.STDOUT)
                atexit.register(self.stop_service)  # never leave the child (and its GPU memory) behind
            deadline = time.time() + self.settings.ecg_startup_timeout_s
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    log.error("ECG service exited early with code %s (see data/ecg_service.log)", self._proc.returncode)
                    return False
                if self._health(force=True):
                    return True
                time.sleep(1.0)
            return False

    def stop_service(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def _service_ready(self) -> bool:
        if self._health():
            return True
        if self.settings.ecg_autostart and self.package_path():
            return self.start_service()
        return False

    # -------------------------------------------------------------------- analysis
    def analyze_file(self, content: bytes, filename: str) -> ECGResult:
        if self._service_ready():
            try:
                r = requests.post(f"{self.settings.ecg_service_url}/api/analyze",
                                  files={"file": (filename, content)}, timeout=120)
                payload = r.json()
                if r.ok and payload.get("status") == "ok":
                    return self._from_teknofest(payload["result"])
                raise RuntimeError(payload.get("message", f"HTTP {r.status_code}"))
            except Exception as exc:  # fall through to the built-in analyser
                log.warning("ECG service failed (%s) - using built-in screening analyser", exc)
        return self.builtin_analyze(self._load_signal(content, filename), record_name=filename)

    def analyze_array(self, sig: np.ndarray, record_name: str = "array.npy", fs: int = 500) -> ECGResult:
        buf = io.BytesIO()
        np.save(buf, np.asarray(sig, dtype=np.float32))
        if self._service_ready():
            return self.analyze_file(buf.getvalue(), record_name if record_name.endswith(".npy") else record_name + ".npy")
        return self.builtin_analyze(np.asarray(sig), record_name=record_name, fs=fs)

    def sample(self, class_code: str = "NORMAL") -> ECGResult:
        """Demo recording: the package's clinical sample if available, else a simulated ECG."""
        class_code = class_code.upper()
        if self._service_ready():
            try:
                r = requests.get(f"{self.settings.ecg_service_url}/api/sample/{class_code}", timeout=120)
                payload = r.json()
                if r.ok and payload.get("status") == "ok":
                    return self._from_teknofest(payload["result"])
            except Exception as exc:
                log.warning("ECG sample request failed: %s", exc)
        irregular = class_code in {"AFIB", "AFL", "PVC", "PAC", "AVB"}
        hr = {"AFIB": 118, "AFL": 130, "AVB": 42}.get(class_code, 72)
        return self.builtin_analyze(generate_ecg(hr=hr, irregular=irregular, seed=7), record_name=f"DEMO_{class_code}")

    # ------------------------------------------------------------------- converters
    @staticmethod
    def _from_teknofest(res: Dict[str, Any]) -> ECGResult:
        pred = res.get("prediction", {})
        vit = res.get("vitals", {})
        bio = res.get("v22_biomarkers", {})
        probs = res.get("probabilities", [])
        normal_pct = next((float(p.get("percentage", 0)) for p in probs if p.get("class_code") == "NORMAL"), None)
        if normal_pct is None:  # unusual: derive from the top class
            normal_pct = float(pred.get("confidence_pct", 0)) if pred.get("class_code") == "NORMAL" else 0.0
        sev = pred.get("severity", {})
        sg = res.get("sgarbossa_matrix", {})
        return ECGResult(
            source="teknofest-ecg",
            model_version=str(res.get("model_version", "unknown")),
            record_name=str(res.get("record_name", "ecg")),
            class_code=str(pred.get("class_code", "?")),
            class_name=str(pred.get("class_name", "?")),
            confidence_pct=float(pred.get("confidence_pct", 0.0)),
            severity_label=str(sev.get("label", "")),
            severity_color=str(sev.get("color", "#9aa0a6")),
            abnormal_probability_pct=max(0.0, min(100.0, 100.0 - normal_pct)),
            heart_rate=_f(vit.get("heart_rate")),
            qtc_ms=_f(vit.get("qtc_interval")),
            pr_ms=_f(vit.get("pr_interval")),
            qrs_ms=_f(vit.get("qrs_duration")),
            st_deviation_mv=_f(vit.get("st_deviation_mv")),
            rr_regularity=vit.get("rr_regularity"),
            uncertainty_pct=_f(bio.get("uncertainty_pct")),
            stemi_suspected=bool(sg.get("stemi_suspected", False)),
            medical_desc=str(pred.get("medical_desc", "")),
            probabilities=[{"code": p.get("class_code"), "name": p.get("class_name"), "pct": p.get("percentage")}
                           for p in sorted(probs, key=lambda p: -float(p.get("percentage", 0)))[:5]],
            lead_ii=[round(float(x), 4) for x in res.get("leads_preview", {}).get("lead_ii", [])[:1000]],
            notes="Analysed by the TEKNOFEST 2026 OmniNet ECG model.",
        )

    @staticmethod
    def _load_signal(content: bytes, filename: str) -> np.ndarray:
        fn = filename.lower()
        buf = io.BytesIO(content)
        if fn.endswith(".npy"):
            return np.load(buf)
        if fn.endswith(".npz"):
            data = np.load(buf)
            return data[list(data.keys())[0]]
        if fn.endswith(".csv"):
            import pandas as pd

            return pd.read_csv(buf).select_dtypes(include=[np.number]).values.T
        raise ValueError(f"Unsupported ECG format for the built-in analyser: {filename} (use .npy/.npz/.csv)")

    # --------------------------------------------------------- built-in fallback
    @staticmethod
    def builtin_analyze(sig: np.ndarray, record_name: str = "ecg", fs: int = 500) -> ECGResult:
        """Very small screening analyser: R-peaks -> heart rate, regularity, HRV.

        NOT a diagnostic classifier.  It exists so that the rest of the platform keeps working
        when the TEKNOFEST model is unavailable, and it says so in ``notes``.
        """
        arr = np.asarray(sig, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > arr.shape[1]:  # (samples, leads) -> (leads, samples)
            arr = arr.T
        lead = arr[1] if arr.shape[0] > 1 else arr[0]
        # Band-pass 5-25 Hz emphasises the QRS complex.
        b, a = sp_signal.butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype="band")
        filt = sp_signal.filtfilt(b, a, lead)
        env = np.abs(filt)
        peaks, _ = sp_signal.find_peaks(env, distance=int(0.25 * fs), height=0.5 * np.max(env))
        hr, regularity, rmssd, cv = None, "unknown", None, 0.0
        if len(peaks) >= 3:
            rr = np.diff(peaks) / fs
            hr = float(60.0 / np.mean(rr))
            cv = float(np.std(rr) / np.mean(rr))
            rmssd = float(np.sqrt(np.mean(np.diff(rr * 1000) ** 2)))
            regularity = "irregular" if cv > 0.15 else "regular"
        if hr is None:
            code, name, sev, color, abn = "UNKNOWN", "Signal too short / noisy", "Unclassified", "#9aa0a6", 30.0
        elif regularity == "irregular":
            code, name, sev, color, abn = "IRREGULAR", "Irregular rhythm (AF screening flag)", "High", "#e74c3c", min(95.0, 55 + 200 * (cv - 0.15))
        elif hr > 100:
            code, name, sev, color, abn = "TACHY", "Sinus tachycardia (screening)", "Medium", "#f1c40f", min(70.0, 25 + (hr - 100))
        elif hr < 50:
            code, name, sev, color, abn = "BRADY", "Bradycardia (screening)", "Medium", "#f1c40f", min(70.0, 25 + (50 - hr))
        else:
            code, name, sev, color, abn = "REGULAR", "Regular rhythm (screening)", "Low", "#2ecc71", 6.0
        return ECGResult(
            source="builtin-screening", model_version="builtin-screening-1.0", record_name=record_name,
            class_code=code, class_name=name, confidence_pct=60.0, severity_label=sev, severity_color=color,
            abnormal_probability_pct=float(abn), heart_rate=None if hr is None else round(hr, 1), rr_regularity=regularity,
            hrv_rmssd_ms=None if rmssd is None else round(rmssd, 1),
            lead_ii=[round(float(x), 4) for x in lead[:1000]],
            notes="Built-in screening analyser (no deep model). Connect the TEKNOFEST ECG package for full "
                  "10-class analysis: set ECG_PACKAGE_PATH.",
        )


def _f(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None
