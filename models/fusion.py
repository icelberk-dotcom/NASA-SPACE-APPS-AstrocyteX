"""
Cross-attention multimodal fusion.

Five kinds of evidence describe an astronaut's state and they live on very different
time scales and formats:

    sensor tokens       continuous streams summarised by Mamba-2        (minutes - hours)
    trend tokens        multi-day physiology summarised by TCN + TFT    (days - weeks)
    daily tokens        one token *per* daily input / lab value          (once a day)
    subjective tokens   the astronaut's free-text log, encoded           (once a day)
    environment tokens  space hazards (radiation, isolation, distance,   (mission context)
                        gravity, closed habitat) from EnvironmentEncoder

Fusion happens in two attention stages:

 1. **Context stage** - daily and subjective tokens act as *queries* and attend to the
    sensor + trend + environment tokens.  A single creatinine value, for instance, can then
    be read "in the light of" the last hour of heart-rate data and of the current habitat.
 2. **Latent stage** - a small set of learned latent queries (a Perceiver-style
    bottleneck) cross-attends to *all* tokens, followed by self-attention and an MLP.
    The attention mass each modality receives is returned so the UI can show which
    kind of evidence drove the assessment.

References
----------
* A. Jaegle et al. "Perceiver: General Perception with Iterative Attention", ICML 2021.
* A. Vaswani et al. "Attention Is All You Need", NeurIPS 2017.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

MODALITIES: List[str] = ["sensors", "trend", "daily", "subjective", "environment"]


class CrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention block: queries attend to a key/value set, then an MLP."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_model * 2, d_model))

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        out, weights = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                                 need_weights=True, average_attn_weights=True)
        q = q + out
        q = q + self.ff(self.norm_ff(q))
        return q, weights  # weights: (B, n_queries, n_kv)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x):
        h = self.norm(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.norm_ff(x))


class CrossAttentionFusion(nn.Module):
    def __init__(self, n_daily: int, subj_dim: int, d_model: int = 48, n_latents: int = 8,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1, n_subj_tokens: int = 2):
        super().__init__()
        self.d_model = d_model
        # One token per daily variable: value * w_i + b_i  (+ learned identity embedding).
        self.daily_w = nn.Parameter(torch.randn(n_daily, d_model) * 0.2)
        self.daily_b = nn.Parameter(torch.zeros(n_daily, d_model))
        self.subj_mlp = nn.Sequential(nn.Linear(subj_dim, d_model * 2), nn.GELU(),
                                      nn.Linear(d_model * 2, d_model * n_subj_tokens))
        self.n_subj_tokens = n_subj_tokens
        # Modality-type embeddings tell the attention which stream a token came from.
        self.type_emb = nn.Parameter(torch.randn(len(MODALITIES), d_model) * 0.02)

        self.context_stage = CrossAttentionLayer(d_model, n_heads, dropout)
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.2)
        self.cross_layers = nn.ModuleList([CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.self_layers = nn.ModuleList([SelfAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, sensor_tokens: torch.Tensor, trend_tokens: torch.Tensor, daily: torch.Tensor,
                subj: torch.Tensor, env_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = daily.shape[0]
        t = self.type_emb
        sensors = sensor_tokens + t[0]
        trend = trend_tokens + t[1]
        daily_tok = daily.unsqueeze(-1) * self.daily_w + self.daily_b + t[2]  # (B, n_daily, d)
        subj_tok = self.subj_mlp(subj).view(B, self.n_subj_tokens, self.d_model) + t[3]
        if env_tokens is None:  # environment unknown -> a single neutral token keeps shapes fixed
            env_tokens = torch.zeros(B, 1, self.d_model, device=daily.device, dtype=daily.dtype)
        env = env_tokens + t[4]

        # Stage 1: daily / subjective tokens read the continuous context and the environment.
        context = torch.cat([sensors, trend, env], dim=1)
        slow = torch.cat([daily_tok, subj_tok], dim=1)
        slow, _ = self.context_stage(slow, context)
        daily_tok, subj_tok = slow[:, : daily_tok.shape[1]], slow[:, daily_tok.shape[1]:]

        # Stage 2: latent bottleneck attends to every modality.
        all_tokens = torch.cat([sensors, trend, daily_tok, subj_tok, env], dim=1)
        sizes = [sensors.shape[1], trend.shape[1], daily_tok.shape[1], subj_tok.shape[1], env.shape[1]]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        weights = None
        for cross, self_attn in zip(self.cross_layers, self.self_layers):
            lat, weights = cross(lat, all_tokens)
            lat = self_attn(lat)
        fused = self.out_norm(lat.mean(dim=1))  # (B, d)

        # Attention mass per modality (averaged over latents) - sums to 1 across modalities.
        per_token = weights.mean(dim=1)  # (B, n_tokens)
        parts = torch.split(per_token, sizes, dim=1)
        modality_attn = torch.stack([p.sum(dim=1) for p in parts], dim=1)  # (B, 5)
        return {"fused": fused, "modality_attn": modality_attn, "daily_attn": parts[2], "env_attn": parts[4]}
