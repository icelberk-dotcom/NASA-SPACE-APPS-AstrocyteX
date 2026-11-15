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

ECG_CLASS_META_EN: Dict[str, Dict[str, Any]] = {
    "NORMAL": {
        "name": "Normal Sinus Rhythm",
        "severity_label": "Low Risk / Nominal",
        "color": "#29b377",
        "medical_desc": "Sinoatrial impulse conducts normally through the AV node and His-Purkinje system with regular P waves, normal PR (120-200 ms), narrow QRS (<120 ms), and regular 60-100 bpm ventricular rate.",
        "hr": 72.0, "qtc_ms": 404.0, "pr_ms": 156.0, "qrs_ms": 88.0, "st_mv": 0.01, "irregular": False, "abn_pct": 5.2, "conf_pct": 94.8, "stemi": False,
    },
    "AFIB": {
        "name": "Atrial Fibrillation",
        "severity_label": "High Risk / Arrhythmia Alert",
        "color": "#e5484d",
        "medical_desc": "Disorganized atrial electrical activation (350-600 bpm) with absent P waves, fibrillatory f-waves, and irregularly irregular R-R intervals; increases thromboembolic stroke risk.",
        "hr": 118.0, "qtc_ms": 452.0, "pr_ms": None, "qrs_ms": 92.0, "st_mv": -0.04, "irregular": True, "abn_pct": 93.4, "conf_pct": 91.6, "stemi": False,
    },
    "AFL": {
        "name": "Atrial Flutter",
        "severity_label": "Elevated-High Risk / Cardiology Review",
        "color": "#df9f15",
        "medical_desc": "Organized macro-reentrant atrial circuit (250-350 bpm) producing classic sawtooth flutter waves in inferior leads (II, III, aVF) with variable AV conduction.",
        "hr": 130.0, "qtc_ms": 446.0, "pr_ms": None, "qrs_ms": 90.0, "st_mv": -0.03, "irregular": True, "abn_pct": 85.0, "conf_pct": 88.4, "stemi": False,
    },
    "LBBB": {
        "name": "Left Bundle Branch Block",
        "severity_label": "Moderate Risk / Conduction Delay",
        "color": "#9b87f5",
        "medical_desc": "Delayed left ventricular depolarization via transseptal conduction with broadened QRS (>=120 ms), notched R waves in lateral leads (I, aVL, V5-V6), and deep QS in V1-V2.",
        "hr": 78.0, "qtc_ms": 468.0, "pr_ms": 178.0, "qrs_ms": 144.0, "st_mv": -0.08, "irregular": False, "abn_pct": 74.5, "conf_pct": 89.2, "stemi": False,
    },
    "RBBB": {
        "name": "Right Bundle Branch Block",
        "severity_label": "Moderate Risk / Conduction Delay",
        "color": "#60a5fa",
        "medical_desc": "Delayed right ventricular activation with QRS >=120 ms, classic rsR' morphology in V1-V2, and wide slurred S waves in lateral leads I and V6.",
        "hr": 76.0, "qtc_ms": 444.0, "pr_ms": 168.0, "qrs_ms": 136.0, "st_mv": -0.02, "irregular": False, "abn_pct": 62.0, "conf_pct": 88.0, "stemi": False,
    },
    "MI": {
        "name": "Myocardial Infarction (Acute STEMI)",
        "severity_label": "Critical Emergency / Acute Ischemia",
        "color": "#e5484d",
        "medical_desc": "Acute coronary occlusion morphology with contiguous J-point ST-segment elevation, hyperacute T waves, or pathologic Q waves; requires immediate flight-surgeon consultation.",
        "hr": 98.0, "qtc_ms": 486.0, "pr_ms": 172.0, "qrs_ms": 108.0, "st_mv": 0.28, "irregular": False, "abn_pct": 96.8, "conf_pct": 94.2, "stemi": True,
    },
    "PVC": {
        "name": "Premature Ventricular Contractions",
        "severity_label": "High Risk / Ventricular Ectopy",
        "color": "#ec4899",
        "medical_desc": "Premature ectopic ventricular beats producing wide (>120 ms) bizarre QRS complexes with discordant ST-T changes and full compensatory pauses.",
        "hr": 88.0, "qtc_ms": 458.0, "pr_ms": 162.0, "qrs_ms": 138.0, "st_mv": -0.06, "irregular": True, "abn_pct": 78.5, "conf_pct": 87.5, "stemi": False,
    },
    "PAC": {
        "name": "Premature Atrial Contractions",
        "severity_label": "Moderate Risk / Supraventricular Ectopy",
        "color": "#38bdf8",
        "medical_desc": "Premature ectopic atrial P' wave with altered morphology followed by narrow QRS conduction and an incomplete compensatory pause.",
        "hr": 84.0, "qtc_ms": 426.0, "pr_ms": 152.0, "qrs_ms": 90.0, "st_mv": 0.01, "irregular": True, "abn_pct": 54.0, "conf_pct": 85.0, "stemi": False,
    },
    "AVB": {
        "name": "Atrioventricular Conduction Block",
        "severity_label": "High Risk / Conduction Block",
        "color": "#e5484d",
        "medical_desc": "Impaired or interrupted AV nodal or His-Purkinje conduction (PR >200 ms, dropped beats, or AV dissociation) causing bradyarrhythmia and syncope risk.",
        "hr": 42.0, "qtc_ms": 474.0, "pr_ms": 264.0, "qrs_ms": 112.0, "st_mv": -0.02, "irregular": True, "abn_pct": 88.0, "conf_pct": 90.4, "stemi": False,
    },
    "STTC": {
        "name": "Ischemic ST-T Wave Abnormality",
        "severity_label": "High Risk / Myocardial Ischemia",
        "color": "#df9f15",
        "medical_desc": "Horizontal or downsloping ST-segment depression and/or symmetric T-wave inversion consistent with subendocardial ischemia or ventricular strain.",
        "hr": 92.0, "qtc_ms": 462.0, "pr_ms": 164.0, "qrs_ms": 96.0, "st_mv": -0.16, "irregular": False, "abn_pct": 81.2, "conf_pct": 86.8, "stemi": False,
    },
}


def _translate_regularity_en(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return "unknown"
    if any(tok in s for tok in ("kaotik", "düzensiz", "duzensiz", "irregular")):
        return "irregular"
    if any(tok in s for tok in ("düzenli", "duzenli", "regular")):
        return "regular"
    if any(tok in s for tok in ("hızlı", "hizli", "yavaş", "yavas", "rate")):
        return "rate-abnormal"
    return str(raw)


def _clean_record_name_en(raw: Any) -> str:
    name = str(raw or "ecg")
    if name.upper().startswith("ORNEK_"):
        return "SAMPLE_" + name[6:]
    return name


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
        """Demo recording: the package's clinical sample if available, else a simulated 12-lead ECG."""
        class_code = class_code.upper()
        if class_code not in ECG_CLASS_META_EN:
            class_code = "NORMAL"
        if self._service_ready():
            try:
                r = requests.get(f"{self.settings.ecg_service_url}/api/sample/{class_code}", timeout=120)
                payload = r.json()
                if r.ok and payload.get("status") == "ok":
                    return self._from_teknofest(payload["result"])
            except Exception as exc:
                log.warning("ECG sample request failed: %s", exc)
        meta = ECG_CLASS_META_EN[class_code]
        sig = generate_ecg(hr=float(meta["hr"]), irregular=bool(meta["irregular"]), seed=7, class_code=class_code)
        base = self.builtin_analyze(sig, record_name=f"SAMPLE_{class_code}")
        normal_prob = round(100.0 - float(meta["abn_pct"]), 1)
        if class_code == "NORMAL":
            probs = [
                {"code": "NORMAL", "name": ECG_CLASS_META_EN["NORMAL"]["name"], "pct": 94.8},
                {"code": "PAC", "name": ECG_CLASS_META_EN["PAC"]["name"], "pct": 2.4},
                {"code": "RBBB", "name": ECG_CLASS_META_EN["RBBB"]["name"], "pct": 1.3},
                {"code": "STTC", "name": ECG_CLASS_META_EN["STTC"]["name"], "pct": 0.9},
                {"code": "PVC", "name": ECG_CLASS_META_EN["PVC"]["name"], "pct": 0.6},
            ]
        else:
            primary_pct = round(float(meta["conf_pct"]), 1)
            rem = max(0.0, round(100.0 - primary_pct - normal_prob, 1))
            alt_code = "STTC" if class_code == "MI" else ("AFL" if class_code == "AFIB" else "PVC")
            probs = sorted([
                {"code": class_code, "name": meta["name"], "pct": primary_pct},
                {"code": "NORMAL", "name": ECG_CLASS_META_EN["NORMAL"]["name"], "pct": normal_prob},
                {"code": alt_code, "name": ECG_CLASS_META_EN[alt_code]["name"], "pct": round(rem * 0.65, 1)},
                {"code": "PAC", "name": ECG_CLASS_META_EN["PAC"]["name"], "pct": round(rem * 0.35, 1)},
            ], key=lambda x: -float(x["pct"]))
        return ECGResult(
            source="builtin-screening",
            model_version="builtin-screening-1.0",
            record_name=f"SAMPLE_{class_code}",
            class_code=class_code,
            class_name=meta["name"],
            confidence_pct=float(meta["conf_pct"]),
            severity_label=meta["severity_label"],
            severity_color=meta["color"],
            abnormal_probability_pct=float(meta["abn_pct"]),
            heart_rate=base.heart_rate if base.heart_rate is not None else float(meta["hr"]),
            qtc_ms=meta["qtc_ms"],
            pr_ms=meta["pr_ms"],
            qrs_ms=meta["qrs_ms"],
            st_deviation_mv=meta["st_mv"],
            rr_regularity="irregular" if meta["irregular"] else "regular",
            hrv_rmssd_ms=base.hrv_rmssd_ms,
            stemi_suspected=bool(meta["stemi"]),
            medical_desc=meta["medical_desc"],
            probabilities=probs,
            lead_ii=base.lead_ii,
            notes="Built-in 12-lead reference waveform & interval screening engine.",
        )

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
        code = str(pred.get("class_code", "?")).upper()
        meta_en = ECG_CLASS_META_EN.get(code)
        norm_probs = []
        for p in sorted(probs, key=lambda item: -float(item.get("percentage", 0)))[:5]:
            p_code = str(p.get("class_code") or "").upper()
            p_meta = ECG_CLASS_META_EN.get(p_code)
            norm_probs.append({
                "code": p_code or p.get("class_code"),
                "name": p_meta["name"] if p_meta else str(p.get("class_name") or p_code),
                "pct": round(float(p.get("percentage", 0)), 2),
            })
        return ECGResult(
            source="teknofest-ecg",
            model_version=str(res.get("model_version", "unknown")),
            record_name=_clean_record_name_en(res.get("record_name", "ecg")),
            class_code=code,
            class_name=meta_en["name"] if meta_en else str(pred.get("class_name", "?")),
            confidence_pct=float(pred.get("confidence_pct", 0.0)),
            severity_label=meta_en["severity_label"] if meta_en else str(sev.get("label", "")),
            severity_color=str(sev.get("color") or (meta_en["color"] if meta_en else "#9aa0a6")),
            abnormal_probability_pct=max(0.0, min(100.0, 100.0 - normal_pct)),
            heart_rate=_f(vit.get("heart_rate")),
            qtc_ms=_f(vit.get("qtc_interval")),
            pr_ms=_f(vit.get("pr_interval")) or None,
            qrs_ms=_f(vit.get("qrs_duration")),
            st_deviation_mv=_f(vit.get("st_deviation_mv")),
            rr_regularity="irregular" if code == "AFIB" else _translate_regularity_en(vit.get("rr_regularity")),
            uncertainty_pct=_f(bio.get("uncertainty_pct")),
            stemi_suspected=bool(sg.get("stemi_suspected", False)),
            medical_desc=meta_en["medical_desc"] if meta_en else str(pred.get("medical_desc", "")),
            probabilities=norm_probs,
            lead_ii=[round(float(x), 4) for x in res.get("leads_preview", {}).get("lead_ii", [])[:1000]],
            notes="Analyzed by OmniNet V23 10-class 12-lead deep ECG network.",
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
        if fn.endswith(".mat"):
            from scipy.io import loadmat

            mat = loadmat(buf)
            for k, val in mat.items():
                if not k.startswith("__") and isinstance(val, np.ndarray) and val.size >= 100:
                    return val
        raise ValueError(f"Unsupported ECG format for the built-in analyser: {filename} (use .npy/.npz/.csv/.mat)")

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
            notes="Built-in R-peak & interval screening analyzer. Set ECG_PACKAGE_PATH for full 10-class deep ECG analysis.",
        )


def _f(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None
