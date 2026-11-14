"""
Mamba-2 state-space encoder for the continuous sensor streams (HR, SpO2, BP, ...).

Mamba-2 ("Transformers are SSMs", Dao & Gu, 2024) restricts the state matrix of a
selective SSM to a *scalar per head*, which turns the recurrence

        h_t = exp(A * dt_t) * h_{t-1} + dt_t * B_t * x_t          (state update)
        y_t = C_t . h_t + D * x_t                                  (read-out)

into a structured-masked attention ("state space duality").  The chunked algorithm
below (``ssd_chunked``) follows the reference *minimal SSD* listing from the paper's
official repository (state-spaces/mamba, Apache-2.0) and is written in plain PyTorch
so it runs on CPU, Windows and macOS.

When the optional CUDA package ``mamba-ssm`` is installed *and* a GPU is present,
``build_mamba2_layer`` transparently swaps in the fused official ``Mamba2`` kernel.

References
----------
* T. Dao, A. Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality", ICML 2024.
* https://github.com/state-spaces/mamba  (mamba-ssm, Apache-2.0)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction), as used in Mamba."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment-sum: out[..., i, j] = sum_{k=j+1..i} x[..., k]  for i >= j, else -inf.

    ``exp(segsum(log_a))`` is the lower-triangular matrix of cumulative decays
    between time steps, i.e. the "1-semiseparable" mask of the SSD formulation.
    """
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)  # (..., T, T)
    strict_lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~strict_lower, 0)
    x_cumsum = torch.cumsum(x, dim=-2)
    lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return x_cumsum.masked_fill(~lower, -torch.inf)


def ssd_chunked(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, chunk: int):
    """Chunked state-space-duality scan.

    Args:
        X: (b, l, h, p)  inputs already multiplied by dt
        A: (b, l, h)     log-decay per step (A * dt, negative)
        B: (b, l, h, n)  input projection
        C: (b, l, h, n)  output projection
        chunk: block length (l must be divisible by it)
    Returns:
        Y (b, l, h, p) and the final SSM state (b, h, p, n).
    """
    b, l, h, p = X.shape
    assert l % chunk == 0, "sequence length must be a multiple of the chunk size"
    c = l // chunk
    X = X.reshape(b, c, chunk, h, p)
    B = B.reshape(b, c, chunk, h, -1)
    C = C.reshape(b, c, chunk, h, -1)
    A = A.reshape(b, c, chunk, h).permute(0, 3, 1, 2)  # (b, h, c, chunk)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1) Intra-chunk (diagonal blocks): quadratic "attention-like" form inside each chunk.
    L = torch.exp(segsum(A))  # (b, h, c, chunk, chunk)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2) State at the end of every chunk, assuming a zero state at the chunk start.
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (b, h, c, chunk)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3) Propagate states across chunks (a tiny recurrence over c chunks).
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))  # (b, h, c+1, c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4) Contribution of the carried-in state to every position of each chunk.
    state_decay_out = torch.exp(A_cumsum)  # (b, h, c, chunk)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = (Y_diag + Y_off).reshape(b, l, h, p)
    return Y, final_state


def ssd_reference_recurrence(X, A, B, C):
    """Slow step-by-step recurrence; used by the unit tests to validate ``ssd_chunked``."""
    b, l, h, p = X.shape
    n = B.shape[-1]
    state = X.new_zeros(b, h, p, n)
    ys = []
    for t in range(l):
        decay = torch.exp(A[:, t])[:, :, None, None]  # (b, h, 1, 1)
        state = decay * state + torch.einsum("bhn,bhp->bhpn", B[:, t], X[:, t])
        ys.append(torch.einsum("bhn,bhpn->bhp", C[:, t], state))
    return torch.stack(ys, dim=1), state


class Mamba2Block(nn.Module):
    """One Mamba-2 mixer layer (pure PyTorch).

    in_proj -> [z | x B C | dt]; causal depth-wise conv on (x, B, C); SSD scan;
    gated RMSNorm; out_proj.
    """

    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4, expand: int = 2,
                 headdim: int = 16, chunk_size: int = 16):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.chunk_size = chunk_size

        conv_dim = self.d_inner + 2 * d_state  # x, B, C go through the conv (one B/C group)
        self.in_proj = nn.Linear(d_model, self.d_inner + conv_dim + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)

        # dt bias initialised so softplus(dt_bias) is log-uniform in [1e-3, 1e-1].
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        # A is a negative scalar per head; stored in log space to stay negative.
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, _ = u.shape
        pad = (-l) % self.chunk_size  # right-pad so l is a multiple of the chunk size
        if pad:
            u = F.pad(u, (0, 0, 0, pad))
        L = u.shape[1]

        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2))  # causal conv
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)

        x = x.reshape(b, L, self.nheads, self.headdim)
        B = B.unsqueeze(2).expand(b, L, self.nheads, self.d_state)
        C = C.unsqueeze(2).expand(b, L, self.nheads, self.d_state)
        dt = F.softplus(dt + self.dt_bias)  # (b, L, h) positive step sizes
        A = -torch.exp(self.A_log)  # (h,) negative decay rates

        y, _ = ssd_chunked(x * dt.unsqueeze(-1), A * dt, B, C, self.chunk_size)
        y = y + x * self.D.view(1, 1, -1, 1)  # skip connection D
        y = y.reshape(b, L, self.d_inner)
        y = self.norm(y * F.silu(z))  # gated norm
        out = self.out_proj(y)
        return out[:, :l]


def build_mamba2_layer(d_model: int, d_state: int, headdim: int, chunk_size: int,
                       prefer_official: bool = True) -> nn.Module:
    """Return the fused official Mamba2 (mamba-ssm) if usable, else the PyTorch block."""
    if prefer_official and torch.cuda.is_available():
        try:  # pragma: no cover - needs CUDA + mamba-ssm
            from mamba_ssm import Mamba2  # type: ignore

            return Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2, headdim=headdim)
        except Exception:
            pass
    return Mamba2Block(d_model, d_state=d_state, headdim=headdim, chunk_size=chunk_size)


class MambaSensorEncoder(nn.Module):
    """Stack of residual Mamba-2 layers over the multichannel sensor window.

    Input  (B, T, C_sensors)  - z-scored HR, SpO2, BP, ...
    Output sequence (B, T, d_model) and ``n_tokens`` pooled summary tokens that the
    cross-attention fusion layer consumes.
    """

    def __init__(self, in_channels: int, d_model: int = 48, n_layers: int = 2, d_state: int = 32,
                 headdim: int = 16, chunk_size: int = 16, n_tokens: int = 4, prefer_official: bool = False):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList(
            [build_mamba2_layer(d_model, d_state, headdim, chunk_size, prefer_official) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.n_tokens = n_tokens

    def forward(self, seq: torch.Tensor):
        h = self.in_proj(seq)
        for norm, layer in zip(self.norms, self.layers):
            h = h + layer(norm(h))  # pre-norm residual block
        h = self.final_norm(h)
        # Adaptive average pooling over time -> a few tokens (early ... recent trend of the window).
        tokens = F.adaptive_avg_pool1d(h.transpose(1, 2), self.n_tokens).transpose(1, 2)
        return h, tokens
