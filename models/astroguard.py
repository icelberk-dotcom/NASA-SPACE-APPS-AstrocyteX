"""
AstroGuard-X network: the full multimodal pipeline in one ``nn.Module``.

                 sensors (HR, SpO2, BP ...)  --> Mamba-2 encoder -----------+
                 14-day history              --> TCN + TFT encoder ---------+--> Cross-attention
                 daily inputs / labs         --> per-variable tokens -------+    fusion  --> fused
                 subjective log              --> symptom encoder -----------+       |
                                                                                    v
                         mission day + controls + fused --> Neural ODE / PINN (space physiology)
                                                                                    |
                              [fused | physics summary] --> Bayesian risk head --> risk %, uncertainty
                              fused + trend              --> 7-day risk forecast

Outputs (``predict``) are plain NumPy arrays so the backend/UI never touches tensors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from models.fusion import MODALITIES, CrossAttentionFusion
from models.mamba2 import MambaSensorEncoder
from models.physics import STATE_NAMES, PhysiologyODE
from models.rule_prior import RuleRiskTorch, aggregate_risk_torch
from models.tcn_tft import TrendEncoder
from models.uncertainty import BayesianRiskHead
from utils.features import MC_FORECAST_DAYS, N_DAILY, N_PARAMS, N_SEQ
from utils.subjective import SUBJ_DIM

N_OUTPUTS = N_PARAMS + 1  # one risk per parameter + the overall risk
MODEL_VERSION = "astroguard-net-1.0"


@dataclass
class ModelConfig:
    d_model: int = 48
    mamba_layers: int = 2
    d_state: int = 32
    headdim: int = 16
    chunk_size: int = 16
    tcn_levels: int = 3
    heads: int = 4
    fusion_layers: int = 2
    n_latents: int = 8
    dropout: float = 0.1
    head_hidden: int = 96
    horizon: int = MC_FORECAST_DAYS
    physics_horizon: int = 14
    prefer_official_mamba: bool = False  # set True to use the CUDA mamba-ssm kernel


class AstroGuardNet(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.sensor_encoder = MambaSensorEncoder(N_SEQ, c.d_model, c.mamba_layers, c.d_state, c.headdim,
                                                 c.chunk_size, prefer_official=c.prefer_official_mamba)
        self.trend_encoder = TrendEncoder(N_PARAMS, c.d_model, c.horizon, c.tcn_levels, c.heads, c.dropout)
        self.fusion = CrossAttentionFusion(N_DAILY, SUBJ_DIM, c.d_model, c.n_latents, c.heads, c.fusion_layers, c.dropout)
        self.physics = PhysiologyODE(c.d_model, hidden=c.d_model, horizon_days=c.physics_horizon)
        self.rule = RuleRiskTorch()  # differentiable reference-range score (residual-learning base)
        self.head = BayesianRiskHead(c.d_model + 8 + 2 * N_PARAMS, c.head_hidden, N_OUTPUTS, dropout=0.15)
        self.forecast_adj = nn.Linear(c.d_model, c.horizon)

    # ------------------------------------------------------------------ forward passes
    def backbone(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Deterministic part (everything before the Bayesian head)."""
        _, sensor_tokens = self.sensor_encoder(batch["seq"])
        trend = self.trend_encoder(batch["history"])
        fusion = self.fusion(sensor_tokens, trend["tokens"], batch["daily"], batch["subj"])
        physics = self.physics(batch["mission"], batch["controls"], fusion["fused"])
        # Direct measurement pathway: the current z-scored values and their rule-based risk.
        z_now = torch.cat([batch["seq"][:, -1], batch["daily"]], dim=-1)  # PARAM_KEYS order
        rule_pct = self.rule(z_now)  # (B, 31) in %
        base_pct = torch.cat([rule_pct, aggregate_risk_torch(rule_pct).unsqueeze(-1)], dim=-1)
        base_logit = torch.logit((base_pct / 100.0).clamp(0.01, 0.995))
        feat = torch.cat([fusion["fused"], physics["summary"], rule_pct / 100.0, z_now / 3.0], dim=-1)
        forecast_logits = trend["forecast_logits"] + self.forecast_adj(fusion["fused"])
        return {"feat": feat, "base_logit": base_logit, "fusion": fusion, "trend": trend, "physics": physics,
                "forecast_logits": forecast_logits}

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training forward pass: one stochastic sample of the Bayesian head."""
        bb = self.backbone(batch)
        logit, logvar = self.head(bb["feat"], bb["base_logit"], sample=True, mc_dropout=True)
        bb.update({"logit": logit, "logvar": logvar, "kl": self.head.kl()})
        return bb

    # ---------------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor], n_mc: int = 24, project_days: Optional[int] = None) -> Dict[str, object]:
        """Monte-Carlo prediction with uncertainty. All outputs are NumPy (risk in %)."""
        self.eval()
        bb = self.backbone(batch)
        mc = self.head.mc_predict(bb["feat"], bb["base_logit"], n_mc)
        out: Dict[str, object] = {
            "risk": mc["mean"].numpy() * 100.0,
            "epistemic": mc["epistemic_std"].numpy() * 100.0,
            "aleatoric": mc["aleatoric_std"].numpy() * 100.0,
            "ci_low": mc["ci_low"].numpy() * 100.0,
            "ci_high": mc["ci_high"].numpy() * 100.0,
            "forecast": torch.sigmoid(bb["forecast_logits"]).numpy() * 100.0,
            "modality_attention": bb["fusion"]["modality_attn"].numpy(),
            "trend_variable_weights": bb["trend"]["var_weights"].numpy(),
            "physics_traj": bb["physics"]["traj"].numpy(),
            "physics_z0": bb["physics"]["z0"].numpy(),
        }
        if project_days:
            proj = self.physics.project(batch["mission"], batch["controls"], bb["fusion"]["fused"], project_days)
            out["physics_projection"] = proj.numpy()
        return out


# ------------------------------------------------------------------------ checkpoints
def save_checkpoint(model: AstroGuardNet, path: Path, meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(),
                "version": MODEL_VERSION, "meta": meta or {}}, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> AstroGuardNet:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = AstroGuardNet(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.meta = ckpt.get("meta", {})  # type: ignore[attr-defined]
    return model.eval()


__all__ = ["AstroGuardNet", "ModelConfig", "save_checkpoint", "load_checkpoint", "N_OUTPUTS",
           "MODALITIES", "STATE_NAMES", "MODEL_VERSION"]
