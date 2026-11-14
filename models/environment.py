"""
Environment-hazard encoder (radiation, isolation & confinement, distance from Earth, gravity
fields, hostile/closed habitat) and FiLM conditioning.

Why a dedicated branch?
-----------------------
The 12 environment parameters are not symptoms of the astronaut - they are the *context*
that changes how dangerous a physiological deviation is.  A heart rate of 105 bpm means
something different in a quiet Earth analog than two days after landing, during a CO2
scrubber failure or 200 million km from the nearest hospital.  The network therefore:

1. turns every hazard into a token from **both** its z-score and its transparent hazard
   risk (the rule score is a monotone, saturating transform - it resolves small but
   important differences such as ISS (0.0004) vs. Moon (0.384 million km) that a linear
   z-score cannot, and it bounds extreme values such as a 50 mSv/day radiation storm);
2. lets the hazards interact through self-attention (distance x radiation, isolation x
   cramped volume, CO2 x noise ...);
3. hands the hazard tokens to the cross-attention fusion as a fifth modality, and
4. modulates the fused physiological representation with **FiLM** (feature-wise linear
   modulation: ``h' = h * (1 + gamma(env)) + beta(env)``) - a multiplicative "the same
   physiology is riskier in this environment" pathway.

References
----------
* E. Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
* NASA Human Research Program, "Five hazards of human spaceflight" (RIDGE), 2019-2024.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class EnvironmentEncoder(nn.Module):
    def __init__(self, n_env: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_env = n_env
        # Per-hazard embedding of [z, risk]: value_i -> d_model (vectorised, like the TFT VSN).
        self.w_z = nn.Parameter(torch.randn(n_env, d_model) * 0.2)
        self.w_r = nn.Parameter(torch.randn(n_env, d_model) * 0.2)
        self.identity = nn.Parameter(torch.randn(n_env, d_model) * 0.2)  # which hazard this token is
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.mixer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)  # attention pooling over hazards
        self.film = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.film[-1].weight)  # start as the identity modulation (gamma = beta = 0)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, z_env: torch.Tensor, risk_env: torch.Tensor) -> Dict[str, torch.Tensor]:
        """z_env: (B, E) z-scores; risk_env: (B, E) hazard risk in [0, 1]."""
        tok = z_env.unsqueeze(-1) * self.w_z + (risk_env.unsqueeze(-1) - 0.25) * self.w_r + self.identity
        tok = self.norm(self.mixer(tok))  # (B, E, d)
        w = torch.softmax(self.pool_score(tok).squeeze(-1), dim=-1)  # (B, E) hazard importance
        pooled = (tok * w.unsqueeze(-1)).sum(dim=1)  # (B, d)
        gamma, beta = self.film(pooled).chunk(2, dim=-1)
        return {"tokens": tok, "pooled": pooled, "hazard_weights": w,
                "gamma": torch.tanh(gamma) * 0.5, "beta": beta}

    @staticmethod
    def modulate(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """FiLM: feature-wise affine modulation of the fused representation."""
        return h * (1.0 + gamma) + beta
