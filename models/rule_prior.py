
"""
Differentiable copy of the transparent reference-range risk score (``utils.reference_ranges``).

The network does not have to *re-discover* what a normal heart rate is.  Instead the risk
head works by **residual learning on top of the rule score**::

        risk = sigmoid( logit(rule_risk) + delta(features) )

``delta`` is what the multimodal evidence (trend, sensor window, subjective log, daily
labs, physics) adds: interactions between parameters, symptoms, trends.  With ``delta = 0``
the model reproduces the plain range check exactly, which is also a great safety
property: an untrained or failing network degrades gracefully to the rule score.

``RuleRiskTorch`` evaluates the same piece-wise-linear curves as the NumPy implementation
(they are unit-tested against each other) but on batched torch tensors.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.reference_ranges import BASELINE_RISK, ONE_SIDED_FLOOR, PARAMETERS

_BIG = 1e6


class RuleRiskTorch(nn.Module):
    def __init__(self):
        super().__init__()
        n = len(PARAMETERS)
        xs_hi, xs_lo = np.zeros((n, 4)), np.zeros((n, 4))
        has_hi, has_lo = np.zeros(n), np.zeros(n)
        for i, p in enumerate(PARAMETERS):
            if p.normal[1] is not None:
                has_hi[i] = 1.0
                xs_hi[i] = [0.0, p.normal[1] - p.ideal, p.warn[1] - p.ideal, p.danger[1] - p.ideal]
            if p.normal[0] is not None:
                has_lo[i] = 1.0
                xs_lo[i] = [0.0, p.ideal - p.normal[0], p.ideal - p.warn[0], p.ideal - p.danger[0]]
        t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        self.register_buffer("ideal", t([p.ideal for p in PARAMETERS]))
        self.register_buffer("scale", t([p.scale for p in PARAMETERS]))
        self.register_buffer("xs_hi", t(xs_hi))
        self.register_buffer("xs_lo", t(xs_lo))
        self.register_buffer("has_hi", t(has_hi))
        self.register_buffer("has_lo", t(has_lo))
        self.register_buffer("ys", t([BASELINE_RISK, 40.0, 70.0, 90.0]))

    def _side(self, d: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """Piece-wise linear risk (0-100) for distances d (B, P) with anchors xs (P, 4)."""
        out = torch.full_like(d, BASELINE_RISK)
        for j in range(3):
            width = (xs[:, j + 1] - xs[:, j]).clamp(min=1e-9)
            frac = ((d - xs[:, j]) / width).clamp(0.0, 1.0)
            out = out + frac * (self.ys[j + 1] - self.ys[j])
        span = torch.maximum(xs[:, 3] - xs[:, 2], xs[:, 3] * 0.25).clamp(min=1e-6)
        beyond = (d > xs[:, 3]).float()
        out = out + beyond * (10.0 * (1 - torch.exp(-(d - xs[:, 3]).clamp(min=0) / span)) * 0.999 + 1e-6)
        return out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, P) z-scored values (as produced by ``normalize_value``). Returns risk % (B, P)."""
        v = self.ideal + z * self.scale  # back to physical units
        dist_hi, dist_lo = (v - self.ideal).clamp(min=0), (self.ideal - v).clamp(min=0)
        hi = self._side(dist_hi, self.xs_hi)
        lo = self._side(dist_lo, self.xs_lo)
        risk = torch.where(v >= self.ideal, torch.where(self.has_hi > 0, hi, torch.full_like(hi, ONE_SIDED_FLOOR)),
                           torch.where(self.has_lo > 0, lo, torch.full_like(lo, ONE_SIDED_FLOOR)))
        return risk


def aggregate_risk_torch(risk: torch.Tensor, idx=None) -> torch.Tensor:
    """Torch version of ``aggregate_risk`` over the last dimension (0.6 max + 0.25 top-3 + 0.15 mean).

    ``idx`` optionally restricts the aggregation to a subset of columns (e.g. the
    physiological parameters only - the overall *health* risk excludes environment hazards).
    """
    if idx is not None:
        risk = risk[..., idx]
    srt, _ = torch.sort(risk, dim=-1, descending=True)
    return 0.60 * srt[..., 0] + 0.25 * srt[..., :3].mean(-1) + 0.15 * risk.mean(-1)
