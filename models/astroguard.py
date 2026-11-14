"""
AstroGuard-X network (v2): the full multimodal, environment-aware pipeline in one ``nn.Module``.

                 sensors (HR, SpO2, BP ...)  --> Mamba-2 encoder -----------+
                 14-day history (43 vars)    --> TCN + TFT encoder ---------+
                 daily inputs / labs         --> per-variable tokens -------+--> Cross-attention fusion --> fused
                 subjective log              --> symptom encoder -----------+            |
                 space environment (12)      --> Environment encoder -------+            |
                   (radiation, isolation,           |                                    v
                    distance, gravity, habitat)     +----- FiLM (gamma, beta) ----> fused' = fused*(1+g)+b
                                                                                         |
                         mission day + controls + fused' --> Neural ODE / PINN (space physiology)
                                                                                         |
               [fused' | physics | rule risk | z | environment] --> Bayesian risk head --> risk %, uncertainty
               fused' + trend                                    --> 7-day risk forecast

The risk head learns a *residual* on top of the transparent reference-range score of each
parameter (see ``rule_prior.py``).  Output 0..42 are the per-parameter risks, output 43 is
the overall *health* risk (physiological parameters only; environment hazards act on it
through the network).

Outputs (``predict``) are plain NumPy arrays so the backend/UI never touches tensors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from models.environment import EnvironmentEncoder
from models.fusion import MODALITIES, CrossAttentionFusion
from models.mamba2 import MambaSensorEncoder
from models.physics import STATE_NAMES, PhysiologyODE
from models.rule_prior import RuleRiskTorch, aggregate_risk_torch
from models.tcn_tft import TrendEncoder
from models.uncertainty import BayesianRiskHead
from utils.features import MC_FORECAST_DAYS, N_DAILY_HEALTH, N_ENV, N_PARAMS, N_SEQ
from utils.reference_ranges import HEALTH_IDX
from utils.subjective import SUBJ_DIM

N_OUTPUTS = N_PARAMS + 1  # one risk per parameter + the overall health risk
MODEL_VERSION = "astroguard-net-2.0"


@dataclass
class ModelConfig:
    d_model: int = 64
    mamba_layers: int = 2
    d_state: int = 32
    headdim: int = 16
    chunk_size: int = 16
    tcn_levels: int = 3
    heads: int = 4
    fusion_layers: int = 2
    n_latents: int = 12
    env_layers: int = 2
    dropout: float = 0.1
    head_hidden: int = 160
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
        self.env_encoder = EnvironmentEncoder(N_ENV, c.d_model, c.heads, c.env_layers, c.dropout)
        self.fusion = CrossAttentionFusion(N_DAILY_HEALTH, SUBJ_DIM, c.d_model, c.n_latents, c.heads, c.fusion_layers, c.dropout)
        self.physics = PhysiologyODE(c.d_model, hidden=c.d_model, horizon_days=c.physics_horizon)
        self.rule = RuleRiskTorch()  # differentiable reference-range score (residual-learning base)
        self.register_buffer("health_idx", torch.tensor(HEALTH_IDX, dtype=torch.long), persistent=False)
        self.head = BayesianRiskHead(2 * c.d_model + 8 + 2 * N_PARAMS, c.head_hidden, N_OUTPUTS, dropout=0.15)
        self.forecast_adj = nn.Linear(2 * c.d_model, c.horizon)

    # ------------------------------------------------------------------ forward passes
    def backbone(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Deterministic part (everything before the Bayesian head)."""
        # Direct measurement pathway: current z-scored values and their transparent rule risk.
        z_now = torch.cat([batch["seq"][:, -1], batch["daily"]], dim=-1)  # PARAM_KEYS order
        rule_pct = self.rule(z_now)  # (B, 43) in %

        _, sensor_tokens = self.sensor_encoder(batch["seq"])
        trend = self.trend_encoder(batch["history"])
        daily_health, z_env = batch["daily"][:, :N_DAILY_HEALTH], batch["daily"][:, N_DAILY_HEALTH:]
        env = self.env_encoder(z_env, rule_pct[:, -N_ENV:] / 100.0)
        fusion = self.fusion(sensor_tokens, trend["tokens"], daily_health, batch["subj"], env["tokens"])
        fused = EnvironmentEncoder.modulate(fusion["fused"], env["gamma"], env["beta"])  # FiLM
        physics = self.physics(batch["mission"], batch["controls"], fused)

        base_pct = torch.cat([rule_pct, aggregate_risk_torch(rule_pct, self.health_idx).unsqueeze(-1)], dim=-1)
        base_logit = torch.logit((base_pct / 100.0).clamp(0.01, 0.995))
        feat = torch.cat([fused, env["pooled"], physics["summary"], rule_pct / 100.0, z_now / 3.0], dim=-1)
        forecast_logits = trend["forecast_logits"] + self.forecast_adj(torch.cat([fused, env["pooled"]], dim=-1))
        return {"feat": feat, "base_logit": base_logit, "fusion": fusion, "trend": trend, "physics": physics,
                "env": env, "forecast_logits": forecast_logits}

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
        dev = next(self.parameters()).device
        batch = {k: v.to(dev) for k, v in batch.items()}
        bb = self.backbone(batch)
        mc = self.head.mc_predict(bb["feat"], bb["base_logit"], n_mc)
        np_ = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
        out: Dict[str, object] = {
            "risk": np_(mc["mean"]) * 100.0,
            "epistemic": np_(mc["epistemic_std"]) * 100.0,
            "aleatoric": np_(mc["aleatoric_std"]) * 100.0,
            "ci_low": np_(mc["ci_low"]) * 100.0,
            "ci_high": np_(mc["ci_high"]) * 100.0,
            "forecast": np_(torch.sigmoid(bb["forecast_logits"])) * 100.0,
            "modality_attention": np_(bb["fusion"]["modality_attn"]),
            "hazard_weights": np_(bb["env"]["hazard_weights"]),
            "trend_variable_weights": np_(bb["trend"]["var_weights"]),
            "physics_traj": np_(bb["physics"]["traj"]),
            "physics_z0": np_(bb["physics"]["z0"]),
        }
        if project_days:
            fused = EnvironmentEncoder.modulate(bb["fusion"]["fused"], bb["env"]["gamma"], bb["env"]["beta"])
            proj = self.physics.project(batch["mission"], batch["controls"], fused, project_days)
            out["physics_projection"] = np_(proj)
        return out


# ------------------------------------------------------------------------ checkpoints
def save_checkpoint(model: AstroGuardNet, path: Path, meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": asdict(model.cfg), "state_dict": state, "version": MODEL_VERSION, "meta": meta or {}}, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> AstroGuardNet:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if ckpt.get("version") != MODEL_VERSION:
        raise ValueError(f"checkpoint version {ckpt.get('version')} != {MODEL_VERSION} (retrain the model)")
    known = {f.name for f in fields(ModelConfig)}
    model = AstroGuardNet(ModelConfig(**{k: v for k, v in ckpt["config"].items() if k in known}))
    model.load_state_dict(ckpt["state_dict"])
    model.meta = ckpt.get("meta", {})  # type: ignore[attr-defined]
    return model.eval()


__all__ = ["AstroGuardNet", "ModelConfig", "save_checkpoint", "load_checkpoint", "N_OUTPUTS",
           "MODALITIES", "STATE_NAMES", "MODEL_VERSION"]
