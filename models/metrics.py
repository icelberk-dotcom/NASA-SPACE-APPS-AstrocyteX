"""
Evaluation metrics for the 5-band risk indicator (gray / green / yellow / red / black).

Why these metrics?
------------------
Every parameter of every crew-day is assigned to one of five *ordered* colour bands, and the
classes are very imbalanced (most values are gray/green; black is rare but the most
important).  Plain accuracy would reward a model that never says "black".  We therefore
report:

* **MCC** (Matthews correlation coefficient, multi-class R_K of Gorodkin 2004) - the primary
  score.  It uses every cell of the confusion matrix, is invariant to class imbalance and
  is only high when *all* classes are predicted well (range -1..1, 0 = chance).
* **Macro-F1** - the unweighted mean of per-class F1, so the rare black band counts as much
  as the common gray band.
* **Quadratic weighted kappa (QWK)** - the bands are ordinal: confusing gray with black is
  far worse than gray with green; QWK penalises errors by the squared band distance.
* **Balanced accuracy**, per-class precision / recall and the confusion matrix.
* **Safety metrics** - the cost of errors is asymmetric in medicine:
  - *critical recall*: share of truly red/black cases that the system flags as red/black;
  - *under-triage rate*: truly red/black cases shown as gray/green/yellow (missed alarms);
  - *actionable F1 / MCC*: binary "does this value trigger mitigation tasks" (yellow or worse).

All metrics are computed from confusion matrices with NumPy so they can be bootstrapped
thousands of times quickly; they are unit-tested against scikit-learn.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

N_BANDS = 5
BAND_NAMES = ["gray", "green", "yellow", "red", "black"]


def bands(risk_pct) -> np.ndarray:
    """Vectorised colour band (0..4) exactly matching ``reference_ranges.risk_level``."""
    r = np.asarray(risk_pct, dtype=float)
    return ((r >= 15.0).astype(int) + (r > 40.0) + (r > 70.0) + (r > 90.0)).astype(int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = N_BANDS) -> np.ndarray:
    return np.bincount(np.asarray(y_true).ravel() * k + np.asarray(y_pred).ravel(), minlength=k * k).reshape(k, k)


def mcc_from_cm(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    s = cm.sum()
    c = np.trace(cm)
    t = cm.sum(axis=1)  # true counts
    p = cm.sum(axis=0)  # predicted counts
    num = c * s - (p * t).sum()
    den = np.sqrt((s ** 2 - (p ** 2).sum()) * (s ** 2 - (t ** 2).sum()))
    return float(num / den) if den > 0 else 0.0


def per_class_from_cm(cm: np.ndarray) -> Dict[str, np.ndarray]:
    cm = cm.astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    support = cm.sum(axis=1)
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0  # sklearn: labels present in y_true or y_pred
    return {"precision": precision, "recall": recall, "f1": f1, "support": support, "present": present}


def macro_f1_from_cm(cm: np.ndarray) -> float:
    pc = per_class_from_cm(cm)
    return float(pc["f1"][pc["present"]].mean()) if pc["present"].any() else 0.0


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    pc = per_class_from_cm(cm)
    has = pc["support"] > 0
    return float(pc["recall"][has].mean()) if has.any() else 0.0


def qwk_from_cm(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    k = cm.shape[0]
    s = cm.sum()
    if s == 0:
        return 0.0
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    w = (i - j) ** 2 / (k - 1) ** 2
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / s
    den = (w * expected).sum()
    return float(1.0 - (w * cm).sum() / den) if den > 0 else 0.0


def safety_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """Clinically motivated metrics (bands: 0 gray, 1 green, 2 yellow, 3 red, 4 black)."""
    cm = cm.astype(float)
    crit_true = cm[3:, :].sum()
    crit_hit = cm[3:, 3:].sum()
    under = cm[3:, :3].sum()
    black_true = cm[4, :].sum()
    # binary "actionable" = yellow or worse (this is what triggers AI mitigation tasks)
    tp = cm[2:, 2:].sum()
    fn = cm[2:, :2].sum()
    fp = cm[:2, 2:].sum()
    tn = cm[:2, :2].sum()
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "critical_recall": float(crit_hit / crit_true) if crit_true else float("nan"),
        "under_triage_rate": float(under / crit_true) if crit_true else float("nan"),
        "black_recall": float(cm[4, 4] / black_true) if black_true else float("nan"),
        "actionable_precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "actionable_recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "actionable_f1": float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan"),
        "actionable_mcc": float((tp * tn - fp * fn) / den) if den > 0 else 0.0,
    }


def band_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion(y_true, y_pred)
    out = {"accuracy": float(np.trace(cm) / max(cm.sum(), 1)), "mcc": mcc_from_cm(cm),
           "macro_f1": macro_f1_from_cm(cm), "qwk": qwk_from_cm(cm), "balanced_accuracy": balanced_accuracy_from_cm(cm)}
    out.update(safety_from_cm(cm))
    return out


_CM_METRICS = {"mcc": mcc_from_cm, "macro_f1": macro_f1_from_cm, "qwk": qwk_from_cm,
               "balanced_accuracy": balanced_accuracy_from_cm,
               "critical_recall": lambda cm: safety_from_cm(cm)["critical_recall"],
               "under_triage_rate": lambda cm: safety_from_cm(cm)["under_triage_rate"],
               "actionable_f1": lambda cm: safety_from_cm(cm)["actionable_f1"]}


def per_sample_confusions(t_bands: np.ndarray, p_bands: np.ndarray, k: int = N_BANDS) -> np.ndarray:
    """(N, k, k) confusion matrix of every crew-day (all its parameters) - for fast bootstrapping."""
    n = t_bands.shape[0]
    codes = t_bands.reshape(n, -1) * k + p_bands.reshape(n, -1)
    out = np.zeros((n, k * k), dtype=np.int64)
    rows = np.repeat(np.arange(n), codes.shape[1])
    np.add.at(out, (rows, codes.ravel()), 1)
    return out.reshape(n, k, k)


def bootstrap_ci(per_sample_cm: np.ndarray, metrics: Sequence[str] = ("mcc", "macro_f1", "qwk"), n_boot: int = 1000,
                 seed: int = 0, paired_with: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap (resampling *crew-days*, not individual cells, to respect the
    correlation between parameters of the same day).  With ``paired_with`` the CI of the
    *difference* (this - other) on identical resamples is returned as well."""
    rng = np.random.default_rng(seed)
    n = per_sample_cm.shape[0]
    res: Dict[str, list] = {m: [] for m in metrics}
    diff: Dict[str, list] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = per_sample_cm[idx].sum(axis=0)
        cm2 = paired_with[idx].sum(axis=0) if paired_with is not None else None
        for m in metrics:
            v = _CM_METRICS[m](cm)
            res[m].append(v)
            if cm2 is not None:
                diff[m].append(v - _CM_METRICS[m](cm2))
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        a = np.asarray(res[m])
        out[m] = {"lo": float(np.nanpercentile(a, 2.5)), "hi": float(np.nanpercentile(a, 97.5))}
        if paired_with is not None:
            d = np.asarray(diff[m])
            out[m].update({"diff_lo": float(np.nanpercentile(d, 2.5)), "diff_hi": float(np.nanpercentile(d, 97.5)),
                           "p_diff_le_0": float(np.mean(d <= 0))})
    return out
