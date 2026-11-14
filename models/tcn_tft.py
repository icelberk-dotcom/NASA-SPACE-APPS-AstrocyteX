"""
Temporal Convolutional Network (TCN) + Temporal Fusion Transformer (TFT) encoder for
the long-term (multi-day) physiological trend branch.

Pipeline:  history (B, H, 31)
    -> Variable Selection Network      learns which parameters matter right now
    -> TCN (dilated causal convolutions) captures local multi-scale patterns
    -> LSTM                            sequential memory
    -> gated self-attention (TFT)      long-range dependencies, interpretable heads
    -> forecast head                   risk trajectory for the next days

The TFT building blocks (GLU, Gated Residual Network, Variable Selection Network,
interpretable multi-head attention) are re-implemented from the paper.

References
----------
* S. Bai, J. Z. Kolter, V. Koltun. "An Empirical Evaluation of Generic Convolutional
  and Recurrent Networks for Sequence Modeling", 2018 (TCN).
* B. Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time
  Series Forecasting", Int. J. Forecasting, 2021.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


# ------------------------------------------------------------------------------ TCN
class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with weight norm, dropout and a residual path."""

    def __init__(self, ch_in: int, ch_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation  # left padding keeps the convolution causal
        self.conv1 = weight_norm(nn.Conv1d(ch_in, ch_out, kernel, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(ch_out, ch_out, kernel, dilation=dilation))
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.down(x))


class TCN(nn.Module):
    """Stack of temporal blocks with exponentially growing dilation (1, 2, 4, ...)."""

    def __init__(self, channels: int, levels: int = 3, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(*[TemporalBlock(channels, channels, kernel, 2 ** i, dropout) for i in range(levels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C) -> (B, T, C)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


# ------------------------------------------------------------------------------ TFT
class GatedLinearUnit(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        return F.glu(self.fc(x), dim=-1)


class GatedResidualNetwork(nn.Module):
    """GRN: LayerNorm(skip(x) + GLU(W2 ELU(W1 x + W_c c)))."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.1, dim_ctx: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.ctx = nn.Linear(dim_ctx, dim_hidden, bias=False) if dim_ctx else None
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(dim_hidden, dim_out)
        self.skip = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x, ctx: Optional[torch.Tensor] = None):
        h = self.fc1(x)
        if self.ctx is not None and ctx is not None:
            h = h + self.ctx(ctx)
        h = self.fc2(F.elu(h))
        return self.norm(self.skip(x) + self.glu(self.drop(h)))


class VariableSelectionNetwork(nn.Module):
    """Embed every scalar variable separately and weight them with a learned softmax."""

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        # One small embedding per variable (value -> d_model), vectorised with einsum.
        self.emb_w = nn.Parameter(torch.randn(n_vars, d_model) * 0.2)
        self.emb_b = nn.Parameter(torch.zeros(n_vars, d_model))
        self.var_grns = nn.ModuleList([GatedResidualNetwork(d_model, d_model, d_model, dropout) for _ in range(n_vars)])
        self.weight_grn = GatedResidualNetwork(n_vars * d_model, d_model, n_vars, dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, T, V) -> embeddings (B, T, V, d)
        emb = x.unsqueeze(-1) * self.emb_w + self.emb_b
        processed = torch.stack([g(emb[..., i, :]) for i, g in enumerate(self.var_grns)], dim=-2)
        weights = torch.softmax(self.weight_grn(emb.flatten(-2)), dim=-1)  # (B, T, V)
        out = (processed * weights.unsqueeze(-1)).sum(dim=-2)
        return out, weights


class InterpretableMultiHeadAttention(nn.Module):
    """TFT attention: heads share the value projection so their average is interpretable."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, self.dk)  # shared across heads
        self.out = nn.Linear(self.dk, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).unsqueeze(1)  # (B, 1, T, dk)
        scores = q @ k.transpose(-1, -2) / self.dk ** 0.5
        if causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, -torch.inf)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).mean(dim=1)  # average heads
        return self.out(out), attn.mean(dim=1)


class TrendEncoder(nn.Module):
    """TCN + TFT encoder over the multi-day history with a multi-horizon forecast head."""

    def __init__(self, n_vars: int, d_model: int = 48, horizon: int = 7, tcn_levels: int = 3,
                 n_heads: int = 4, dropout: float = 0.1, n_tokens: int = 2):
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_vars, d_model, dropout)
        self.tcn = TCN(d_model, tcn_levels, kernel=3, dropout=dropout)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm_gate = GatedLinearUnit(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.post_attn_gate = GatedLinearUnit(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
        self.token_proj = nn.Linear(d_model, d_model * n_tokens)
        self.n_tokens, self.d_model = n_tokens, d_model
        self.forecast = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon))

    def forward(self, history: torch.Tensor):
        # history: (B, H, V)
        x, var_weights = self.vsn(history)  # which variables matter
        x = x + self.tcn(x)  # local multi-scale temporal patterns
        lstm_out, _ = self.lstm(x)
        x = self.norm1(x + self.post_lstm_gate(lstm_out))
        attn_out, attn_w = self.attn(x)
        x = self.norm2(x + self.post_attn_gate(attn_out))
        x = self.grn(x)
        last = x[:, -1]
        tokens = self.token_proj(last).view(-1, self.n_tokens, self.d_model)
        forecast_logits = self.forecast(last)  # (B, horizon): risk trajectory logits
        return {
            "tokens": tokens,
            "last": last,
            "forecast_logits": forecast_logits,
            "var_weights": var_weights.mean(dim=1),  # (B, V) averaged over days
            "attn": attn_w,
        }
