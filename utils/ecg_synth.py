"""
Lightweight synthetic 12-lead ECG generator.

Used (a) by the unit tests, (b) as a demo signal in the UI, and (c) by the built-in
fallback analyser when the Teknofest ECG package is not installed.  Each beat is the sum
of five Gaussian waves (P, Q, R, S, T) in the spirit of McSharry et al. (2003),
"A dynamical model for generating synthetic ECG signals".
"""
from __future__ import annotations

import numpy as np

# (centre [s relative to R], width [s], amplitude [mV]) for P, Q, R, S, T
_WAVES = [(-0.20, 0.025, 0.12), (-0.04, 0.010, -0.12), (0.0, 0.012, 1.10), (0.04, 0.012, -0.25), (0.26, 0.050, 0.30)]
# Rough lead gains (I, II, III, aVR, aVL, aVF, V1..V6) for a normal axis.
_LEAD_GAIN = np.array([0.7, 1.0, 0.45, -0.85, 0.3, 0.75, -0.4, 0.2, 0.6, 1.0, 1.0, 0.85])


def generate_ecg(hr: float = 72.0, irregular: bool = False, seconds: float = 10.0, fs: int = 500,
                 noise: float = 0.02, seed: int | None = None, class_code: str | None = None) -> np.ndarray:
    """Return an array of shape (12, seconds*fs) in millivolts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    beats, now = [], rng.uniform(0.2, 0.6)
    mean_rr = 60.0 / hr
    while now < seconds + 1:
        beats.append(now)
        jitter = rng.normal(0, 0.28 * mean_rr) if irregular else rng.normal(0, 0.015 * mean_rr)
        now += max(0.25, mean_rr + jitter)

    cls = (class_code or "").upper()
    waves = list(_WAVES)
    if cls == "AFIB":
        waves = [w for w in waves if w[0] != -0.20]  # absent P waves
    elif cls == "AFL":
        waves = [w for w in waves if w[0] != -0.20]
    elif cls in ("LBBB", "RBBB"):
        waves = [(-0.20, 0.025, 0.11), (-0.05, 0.018, -0.14), (-0.01, 0.024, 0.92), (0.045, 0.024, 0.58 if cls == "LBBB" else -0.42), (0.28, 0.055, -0.22 if cls == "LBBB" else 0.24)]
    elif cls == "MI":
        waves = [(-0.20, 0.025, 0.11), (-0.04, 0.014, -0.32), (0.0, 0.013, 0.98), (0.09, 0.065, 0.34), (0.26, 0.060, 0.46)]
    elif cls == "STTC":
        waves = [(-0.20, 0.025, 0.12), (-0.04, 0.010, -0.10), (0.0, 0.012, 1.05), (0.10, 0.060, -0.18), (0.26, 0.052, -0.28)]
    elif cls == "AVB":
        waves = [(-0.28, 0.026, 0.13), (-0.04, 0.011, -0.12), (0.0, 0.014, 1.05), (0.04, 0.013, -0.24), (0.27, 0.052, 0.28)]

    lead = np.zeros(n)
    for idx, b in enumerate(beats):
        beat_waves = waves
        if cls == "PVC" and idx % 4 == 2:
            beat_waves = [(-0.02, 0.028, 1.35), (0.05, 0.030, -0.65), (0.26, 0.065, -0.38)]
        elif cls == "PAC" and idx % 4 == 1:
            beat_waves = [(-0.14, 0.020, 0.08), (-0.04, 0.010, -0.12), (0.0, 0.012, 1.08), (0.04, 0.012, -0.24), (0.25, 0.048, 0.28)]
        for centre, width, amp in beat_waves:
            lead += amp * np.exp(-((t - (b + centre)) ** 2) / (2 * width ** 2))

    if cls == "AFIB":
        lead += 0.045 * np.sin(2 * np.pi * 6.8 * t + 0.7) + 0.03 * np.sin(2 * np.pi * 8.3 * t)
    elif cls == "AFL":
        lead += 0.09 * (2 * ((t * 4.8) - np.floor(0.5 + t * 4.8)))  # 288 bpm sawtooth flutter waves

    lead += 0.05 * np.sin(2 * np.pi * 0.25 * t)  # baseline wander
    out = np.stack([g * lead + rng.normal(0, noise, n) for g in _LEAD_GAIN])
    return out.astype(np.float32)
