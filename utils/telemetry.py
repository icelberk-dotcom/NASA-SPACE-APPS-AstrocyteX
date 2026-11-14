"""
Live telemetry simulator for the dashboard demo.

Generates a plausible stream for the seven continuous sensors (HR, SpO2, BP, respiration,
temperature, GSR): the current values act as the baseline, an AR(1) process adds realistic
short-term variability and a slow sinusoid mimics circadian drift.  On a real mission this
class would be replaced by the wearable-sensor ingestion service.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Mapping

import numpy as np

from utils.features import SEQ_LEN
from utils.reference_ranges import PARAM_BY_KEY, SEQ_KEYS, clip_to_bounds


class TelemetrySimulator:
    def __init__(self, baseline: Mapping[str, float], seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.baseline = {k: float(baseline[k]) for k in SEQ_KEYS}
        self.noise = {k: 0.0 for k in SEQ_KEYS}
        self.t = 0
        self.buffers: Dict[str, Deque[float]] = {k: deque(maxlen=SEQ_LEN) for k in SEQ_KEYS}

    def set_baseline(self, baseline: Mapping[str, float]) -> None:
        self.baseline = {k: float(baseline[k]) for k in SEQ_KEYS}

    def step(self) -> Dict[str, float]:
        self.t += 1
        out = {}
        for k in SEQ_KEYS:
            sc = PARAM_BY_KEY[k].scale
            self.noise[k] = 0.8 * self.noise[k] + self.rng.normal(0, 0.12 * sc)
            circadian = 0.12 * sc * np.sin(2 * np.pi * self.t / 90.0)
            v = clip_to_bounds(k, self.baseline[k] + self.noise[k] + circadian)
            self.buffers[k].append(float(v))
            out[k] = float(v)
        return out

    def fill(self, n: int = SEQ_LEN) -> None:
        for _ in range(n):
            self.step()

    def series(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self.buffers.items()}
