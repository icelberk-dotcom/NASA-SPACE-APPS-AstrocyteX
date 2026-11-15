"""
AstroVoiceNet (v1.0): Multimodal Spectrogram + Prosodic Biomarker Neural Network for
Astronaut Psychological State, Mood Dimensions, and Vocal Health Risk Estimation.

Architecture
------------
1. ``SpecConvAttentionEncoder``: processes the (1, 48, 64) log-Mel spectrogram with
   residual depthwise-separable 2D convolutions, squeeze-and-excitation frequency
   recalibration, and multi-head temporal self-attention.
2. ``ProsodyGateEncoder``: processes the 42 GeMAPS-inspired clinical vocal biomarkers
   (F0 statistics, CV, jitter, shimmer, HNR, speech rate, pause dynamics, alpha/Hammarberg
   spectral balance, MFCCs) with strict training-only z-score buffers and interpretable
   per-biomarker attention gates.
3. ``StochasticAcousticHead``: Monte-Carlo dropout + heteroscedastic variance head
   producing:
   - 5-class psychological state probabilities,
   - 4 continuous mood dimensions (valence, arousal, vocal stress, neurocognitive fatigue),
   - calibrated vocal-psychological risk score (0-100 %) with 90 % credible interval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.voice_synth import N_MELS, N_VOICE_FEATURES, SPEC_FRAMES, VOICE_FEATURE_NAMES, VOICE_STATE_META, VOICE_STATES

VOICE_MODEL_VERSION = "AstrocyteX-Voice-1.0"


@dataclass
class VoiceModelConfig:
    d_model: int = 80
    spec_channels: int = 48
    heads: int = 4
    dropout: float = 0.18
    n_classes: int = len(VOICE_STATES)
    n_features: int = N_VOICE_FEATURES


class DepthwiseSepBlock2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: tuple[int, int] = (2, 2)):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, c_out), c_out)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, max(8, c_out // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(8, c_out // 4), c_out, kernel_size=1),
            nn.Sigmoid(),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False), nn.GroupNorm(min(8, c_out), c_out))
            if stride != (1, 1) or c_in != c_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn(self.pw(self.dw(x))))
        h = self.gn2(self.conv2(h))
        h = h * self.se(h)
        return F.gelu(h + self.skip(x))


class SpecConvAttentionEncoder(nn.Module):
    def __init__(self, base_ch: int = 48, d_model: int = 80, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        c1 = max(24, base_ch // 2)
        c2 = base_ch
        c3 = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c1), c1),
            nn.GELU(),
        )
        self.b1 = DepthwiseSepBlock2d(c1, c2, stride=(2, 2))  # (24, 32)
        self.b2 = DepthwiseSepBlock2d(c2, c3, stride=(2, 2))  # (12, 16)
        self.b3 = DepthwiseSepBlock2d(c3, c3, stride=(2, 1))  # (6, 16)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, 16))
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim == 3:
            spec = spec.unsqueeze(1)  # (B, 1, 48, 64)
        h = self.b3(self.b2(self.b1(self.stem(spec))))
        seq = self.freq_pool(h).squeeze(2).transpose(1, 2)  # (B, 16, D)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm(seq + attn_out)
        return seq.mean(dim=1)  # (B, D)


class ProsodyGateEncoder(nn.Module):
    def __init__(self, n_features: int = N_VOICE_FEATURES, d_model: int = 80, dropout: float = 0.15):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.GELU(),
            nn.Linear(n_features * 2, n_features),
        )
        self.fc1 = nn.Linear(n_features, d_model * 2)
        self.fc2 = nn.Linear(d_model, d_model)
        self.skip = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate_net(z_feat), dim=-1)  # (B, F) sums to 1
        gated = z_feat * (1.0 + weights * z_feat.shape[-1] * 0.35)
        a, b = self.fc1(gated).chunk(2, dim=-1)
        h = F.silu(a) * torch.sigmoid(b)
        h = self.fc2(self.drop(h))
        return self.norm(h + self.skip(z_feat)), weights


class AstroVoiceNet(nn.Module):
    def __init__(self, cfg: Optional[VoiceModelConfig] = None):
        super().__init__()
        self.cfg = cfg or VoiceModelConfig()
        c = self.cfg
        self.register_buffer("feat_mean", torch.zeros(c.n_features))
        self.register_buffer("feat_std", torch.ones(c.n_features))

        self.spec_enc = SpecConvAttentionEncoder(c.spec_channels, c.d_model, c.heads, c.dropout)
        self.pros_enc = ProsodyGateEncoder(c.n_features, c.d_model, c.dropout)

        fused_dim = c.d_model * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(fused_dim, c.d_model),
            nn.LayerNorm(c.d_model),
            nn.GELU(),
        )
        self.mc_drop = nn.Dropout(c.dropout)
        self.class_head = nn.Linear(c.d_model, c.n_classes)
        self.mood_head = nn.Linear(c.d_model, 4)  # valence, arousal, stress, fatigue
        self.risk_head = nn.Linear(c.d_model, 2)  # [risk_logit, logvar]

    def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(np.maximum(std, 1e-5), dtype=torch.float32))

    def standardize(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.clamp((feat - self.feat_mean) / self.feat_std, -6.0, 6.0)

    def forward(self, feat: torch.Tensor, spec: torch.Tensor, mc_dropout: bool = False) -> Dict[str, torch.Tensor]:
        z = self.standardize(feat)
        s_vec = self.spec_enc(spec)
        p_vec, feat_w = self.pros_enc(z)
        h = self.fusion(torch.cat([s_vec, p_vec], dim=-1))
        if mc_dropout or self.training:
            h = F.dropout(h, p=self.cfg.dropout, training=True)
        class_logits = self.class_head(h)
        mood_logits = self.mood_head(h)
        risk_out = self.risk_head(h)
        risk_logit = risk_out[:, 0]
        risk_logvar = torch.clamp(risk_out[:, 1], -7.0, 1.0)
        return {
            "class_logits": class_logits,
            "mood_logits": mood_logits,
            "risk_logit": risk_logit,
            "risk_logvar": risk_logvar,
            "feature_weights": feat_w,
        }

    @torch.no_grad()
    def predict(self, feat: np.ndarray, spec: np.ndarray, n_mc: int = 20) -> Dict[str, Any]:
        """Monte-Carlo inference for batch or single sample."""
        self.eval()
        dev = next(self.parameters()).device
        f_t = torch.as_tensor(feat, dtype=torch.float32, device=dev)
        s_t = torch.as_tensor(spec, dtype=torch.float32, device=dev)
        if f_t.ndim == 1:
            f_t = f_t.unsqueeze(0)
        if s_t.ndim == 2:
            s_t = s_t.unsqueeze(0)

        probs_mc, mood_mc, risk_mc, alea_mc, fw_mc = [], [], [], [], []
        for _ in range(max(1, n_mc)):
            out = self.forward(f_t, s_t, mc_dropout=(n_mc > 1))
            probs_mc.append(torch.softmax(out["class_logits"], dim=-1))
            mood_mc.append(torch.sigmoid(out["mood_logits"]) * 100.0)
            risk_mc.append(torch.sigmoid(out["risk_logit"]) * 100.0)
            alea_mc.append(torch.exp(0.5 * out["risk_logvar"]) * 100.0)
            fw_mc.append(out["feature_weights"])

        probs = torch.stack(probs_mc, dim=0).mean(dim=0).cpu().numpy()
        mood = torch.stack(mood_mc, dim=0).mean(dim=0).cpu().numpy()
        risk_draws = torch.stack(risk_mc, dim=0)
        risk_mean = risk_draws.mean(dim=0).cpu().numpy()
        epi_std = risk_draws.std(dim=0).cpu().numpy() if n_mc > 1 else np.zeros_like(risk_mean)
        alea_std = torch.stack(alea_mc, dim=0).mean(dim=0).cpu().numpy()
        tot_std = np.sqrt(epi_std ** 2 + alea_std ** 2)
        ci_low = np.clip(risk_mean - 1.645 * tot_std, 0.0, 100.0)
        ci_high = np.clip(risk_mean + 1.645 * tot_std, 0.0, 100.0)
        fw = torch.stack(fw_mc, dim=0).mean(dim=0).cpu().numpy()

        return {
            "probs": probs,
            "mood": mood,
            "risk_pct": risk_mean,
            "epistemic_pct": epi_std,
            "aleatoric_pct": alea_std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "feature_weights": fw,
        }


def rule_voice_assessment(biomarkers: Dict[str, float]) -> Dict[str, Any]:
    """Transparent clinical-prosodic rule scorer (sex-invariant via F0 CV, jitter, HNR, cadence)."""
    f0_cv = float(biomarkers.get("f0_cv_pct", 9.0))
    jitter = float(biomarkers.get("jitter_pct", 1.0))
    shimmer = float(biomarkers.get("shimmer_pct", 4.0))
    hnr = float(biomarkers.get("hnr_db", 14.0))
    rate = float(biomarkers.get("speech_rate_syl_s", 4.2))
    pause = float(biomarkers.get("pause_ratio_pct", 20.0))
    alpha = float(biomarkers.get("alpha_ratio_db", -8.0))

    # Score prototypes
    scores = {
        "calm_regulated": 2.2,
        "cognitive_fatigue": 0.4,
        "acute_stress": 0.4,
        "low_mood_isolation": 0.4,
        "dyspnea_hypoxic": 0.3,
    }
    if rate > 5.1 or (f0_cv > 15.0 and alpha > -6.5):
        scores["acute_stress"] += 2.4 + 0.4 * max(0.0, rate - 5.0) + 0.3 * max(0.0, jitter - 1.8)
    if pause > 30.0 and rate < 3.4:
        scores["cognitive_fatigue"] += 2.1 + 0.06 * (pause - 28.0) + 0.12 * max(0.0, shimmer - 6.0)
    if f0_cv < 5.2 and rate < 3.5 and alpha < -9.5:
        scores["low_mood_isolation"] += 2.6 + 0.35 * (5.5 - f0_cv)
    if hnr < 9.5 and pause > 26.0 and shimmer > 8.0:
        scores["dyspnea_hypoxic"] += 2.5 + 0.25 * (10.0 - hnr)

    arr = np.array([scores[s] for s in VOICE_STATES], dtype=np.float64)
    probs = np.exp(arr - arr.max())
    probs = probs / probs.sum()

    risk = sum(probs[i] * VOICE_STATE_META[s]["base_risk"] for i, s in enumerate(VOICE_STATES))
    valence = sum(probs[i] * VOICE_STATE_META[s]["valence"] for i, s in enumerate(VOICE_STATES))
    arousal = sum(probs[i] * VOICE_STATE_META[s]["arousal"] for i, s in enumerate(VOICE_STATES))
    stress = sum(probs[i] * VOICE_STATE_META[s]["stress"] for i, s in enumerate(VOICE_STATES))
    fatigue = sum(probs[i] * VOICE_STATE_META[s]["fatigue"] for i, s in enumerate(VOICE_STATES))
    return {
        "probs": probs.astype(np.float32),
        "risk_pct": float(np.clip(risk, 2.0, 98.0)),
        "mood": np.array([valence, arousal, stress, fatigue], dtype=np.float32),
    }


def save_voice_checkpoint(model: AstroVoiceNet, path: Path, meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {"config": asdict(model.cfg), "state_dict": state, "version": VOICE_MODEL_VERSION, "meta": meta or {}},
        path,
    )


def load_voice_checkpoint(path: Path, map_location: str = "cpu") -> AstroVoiceNet:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    known = {f.name for f in fields(VoiceModelConfig)}
    cfg = VoiceModelConfig(**{k: v for k, v in ckpt.get("config", {}).items() if k in known})
    model = AstroVoiceNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.meta = ckpt.get("meta", {})  # type: ignore[attr-defined]
    return model.eval()


__all__ = [
    "AstroVoiceNet",
    "VoiceModelConfig",
    "VOICE_MODEL_VERSION",
    "VOICE_FEATURE_NAMES",
    "rule_voice_assessment",
    "save_voice_checkpoint",
    "load_voice_checkpoint",
]
