"""
Feature engineering: raw readings -> normalised tensors for the neural networks.

Shapes (B = batch):
    seq      (B, SEQ_LEN, 7)    continuous sensor window (Mamba-2 branch)
    daily    (B, 36)            24 daily health inputs / labs + 12 space-environment hazards
                                (the fusion network splits them by position: health | environment)
    history  (B, HIST_LEN, 43)  last two weeks of *all* parameters (TCN + TFT branch)
    subj     (B, SUBJ_DIM)      encoded subjective log
    mission  (B, 1)             mission day / 180 (physics ODE initial state)
    controls (B, 4)             [gravity, exercise effect, protein adequacy, hydration] (ODE inputs)

Everything is z-scored against the reference ranges in ``reference_ranges.py`` so the
network sees "how many z-scale units away from ideal" a value is.  That makes the model
unit-independent and keeps training stable.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from utils.reference_ranges import (DAILY_HEALTH_KEYS, DAILY_KEYS, DEFAULT_VALUES, ENV_KEYS, PARAM_KEYS, SEQ_KEYS,
                                    normalize_value)
from utils.subjective import SUBJ_DIM

SEQ_LEN = 96  # e.g. 96 samples of the wearable stream (one per 5 min = 8 h)
HIST_LEN = 14  # two weeks of daily history for the trend branch
N_SEQ = len(SEQ_KEYS)
N_DAILY = len(DAILY_KEYS)
N_DAILY_HEALTH = len(DAILY_HEALTH_KEYS)
N_ENV = len(ENV_KEYS)
N_PARAMS = len(PARAM_KEYS)
N_CONTROLS = 4
MAX_MISSION_DAYS = 180.0
MC_FORECAST_DAYS = 7  # horizon of the trend forecast head

# Planetary gravity presets (fraction of 1 g) used by the physiology ODE.
GRAVITY_PRESETS = {"Microgravity (ISS)": 0.0, "Moon": 0.166, "Mars": 0.38, "Earth": 1.0}


def _norm_vec(values: Mapping[str, float], keys: Sequence[str]) -> np.ndarray:
    return np.array([normalize_value(k, values.get(k, DEFAULT_VALUES[k])) for k in keys], dtype=np.float32)


def controls_from_values(values: Mapping[str, float], gravity: float = 0.0) -> np.ndarray:
    """Physical control inputs for the physiology ODE derived from daily behaviour."""
    ex = np.clip(values.get("exercise_min", 120.0) / 120.0, 0.0, 1.2) / 1.2
    prot = np.clip(values.get("protein_g", 100.0) / 100.0, 0.0, 1.2) / 1.2
    hyd = np.clip(values.get("fluid_intake_ml", 2500.0) / 2500.0, 0.0, 1.2) / 1.2
    # Scale so that "nominal" behaviour (120 min, 100 g, 2.5 L) gives roughly 0.83 each.
    return np.array([gravity, ex, prot, hyd], dtype=np.float32)


def resample_series(series: Optional[Sequence[float]], fallback: float, length: int = SEQ_LEN) -> np.ndarray:
    """Return exactly `length` samples: keep the newest, left-pad with the oldest."""
    if series is None or len(series) == 0:
        return np.full(length, fallback, dtype=np.float32)
    arr = np.asarray(series, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fallback)
    if len(arr) >= length:
        return arr[-length:]
    pad = np.full(length - len(arr), arr[0], dtype=np.float32)
    return np.concatenate([pad, arr])


def build_sample(
    values: Mapping[str, float],
    sensor_series: Optional[Mapping[str, Sequence[float]]] = None,
    history: Optional[List[Mapping[str, float]]] = None,
    subj_vec: Optional[np.ndarray] = None,
    mission_day: float = 0.0,
    gravity: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Build the normalised arrays for one astronaut at one point in time."""
    sensor_series = sensor_series or {}
    seq = np.zeros((SEQ_LEN, N_SEQ), dtype=np.float32)
    for j, key in enumerate(SEQ_KEYS):
        raw = resample_series(sensor_series.get(key), values.get(key, DEFAULT_VALUES[key]))
        # Make sure the newest stream sample agrees with the explicitly entered value.
        if key in values and sensor_series.get(key):
            raw = raw.copy()
            raw[-1] = values[key]
        seq[:, j] = [normalize_value(key, v) for v in raw]

    daily = _norm_vec(values, DAILY_KEYS)

    hist_rows = list(history or [])[-(HIST_LEN - 1):] + [dict(values)]
    hist = np.stack([_norm_vec(r, PARAM_KEYS) for r in hist_rows])
    if len(hist) < HIST_LEN:  # left-pad with the oldest known day
        hist = np.concatenate([np.repeat(hist[:1], HIST_LEN - len(hist), axis=0), hist])

    return {
        "seq": seq,
        "daily": daily,
        "history": hist.astype(np.float32),
        "subj": (subj_vec if subj_vec is not None else np.full(SUBJ_DIM, 0.0, dtype=np.float32)).astype(np.float32),
        "mission": np.array([float(mission_day) / MAX_MISSION_DAYS], dtype=np.float32),
        "controls": controls_from_values(values, gravity),
    }


def stack_samples(samples: List[Dict[str, np.ndarray]]):
    """Stack a list of sample dicts into batched torch tensors."""
    import torch

    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}
