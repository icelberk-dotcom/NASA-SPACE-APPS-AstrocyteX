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
                 noise: float = 0.02, seed: int | None = None) -> np.ndarray:
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
    lead = np.zeros(n)
    for b in beats:
        for centre, width, amp in _WAVES:
            lead += amp * np.exp(-((t - (b + centre)) ** 2) / (2 * width ** 2))
    lead += 0.05 * np.sin(2 * np.pi * 0.25 * t)  # baseline wander
    out = np.stack([g * lead + rng.normal(0, noise, n) for g in _LEAD_GAIN])
    return out.astype(np.float32)
