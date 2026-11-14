"""
Neural ODE + Physics-Informed Neural Network (PINN) for space-physiology dynamics.

We track four slowly-varying physiological states (time unit = days)::

    F   cephalad fluid shift [L]        legs -> head fluid redistribution in microgravity
    PV  plasma volume [fraction of pre-flight]
    B   bone mineral density [fraction of pre-flight]
    M   skeletal-muscle mass [fraction of pre-flight]

Controls ``u = [g, e, p, h]`` (all in 0..1): gravity fraction, exercise countermeasure
effect (ARED / treadmill), protein adequacy and hydration.

Physics prior (simplified from published spaceflight trends)::

    dF/dt  = -(F  - F_inf) / tau_F ,   F_inf  = F_MAX (1-g)(1 - 0.25 e)
    dPV/dt = -(PV - PV_inf) / tau_PV,  PV_inf = 1 - PV_LOSS (1-g)(1 - 0.5 h)
    dB/dt  = -K_B (1-g)(1 - 0.7 e) B                    (~1 %/month unmitigated)
    dM/dt  = -K_M (1-g)(1 - 0.6 e - 0.2 p) M

Hybrid "universal differential equation"::

    dz/dt = f_physics(z, u)  +  s * tanh( MLP(z, u, patient_context) )

The MLP can only add a *bounded* patient-specific correction (``s`` is a few percent of
the physical rates), so trajectories stay physically plausible.  Training adds PINN
losses: (i) the rolled-out trajectory must satisfy the physics residual
``dz/dt - f_physics`` up to the allowed correction, (ii) a collocation loss at random
state/control points, (iii) hard-physics constraints (no bone/muscle gain without
gravity, plasma volume bounds, non-negative fluid shift).

The numeric constants are illustrative, NOT clinically validated.

References
----------
* R. T. Q. Chen et al. "Neural Ordinary Differential Equations", NeurIPS 2018.
* M. Raissi, P. Perdikaris, G. Karniadakis. "Physics-informed neural networks", J. Comp.
  Phys., 2019.
* NASA Human Research Program evidence reports (bone loss, muscle atrophy, fluid shifts).
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

# ---- physical constants (days) -----------------------------------------------------
F_MAX, TAU_F = 2.0, 0.7
PV_LOSS, TAU_PV = 0.12, 4.0
K_B, K_M = 4.0e-4, 1.5e-3
STATE_NAMES = ["fluid_shift_L", "plasma_volume_frac", "bone_density_frac", "muscle_mass_frac"]
# Scaling that brings every state into O(1) units for the neural network.
Z_CENTER = torch.tensor([0.0, 1.0, 1.0, 1.0])
Z_SCALE = torch.tensor([2.0, 0.10, 0.05, 0.10])
# Largest allowed neural correction of each rate (per day).
CORR_SCALE = torch.tensor([0.5, 5.0e-3, 1.5e-4, 6.0e-4])


def scale_state(z: torch.Tensor) -> torch.Tensor:
    return (z - Z_CENTER.to(z)) / Z_SCALE.to(z)


def physics_rates(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Pure-physics right-hand side f_physics(z, u). z: (B,4) physical units, u: (B,4)."""
    F, PV, B, M = z.unbind(-1)
    g, e, p, h = u.unbind(-1)
    f_inf = F_MAX * (1 - g) * (1 - 0.25 * e)
    pv_inf = 1 - PV_LOSS * (1 - g) * (1 - 0.5 * h)
    dF = -(F - f_inf) / TAU_F
    dPV = -(PV - pv_inf) / TAU_PV
    dB = -K_B * (1 - g) * (1 - 0.7 * e) * B
    dM = -K_M * (1 - g) * (1 - 0.6 * e - 0.2 * p) * M
    return torch.stack([dF, dPV, dB, dM], dim=-1)


def physics_closed_form(t_days: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Exact solution of the physics ODE for constant controls, from the Earth baseline
    state (F=0, PV=B=M=1).  Used for the initial state and to unit-test the integrator."""
    g, e, p, h = u.unbind(-1)
    t = t_days.reshape(-1)
    f_inf = F_MAX * (1 - g) * (1 - 0.25 * e)
    pv_inf = 1 - PV_LOSS * (1 - g) * (1 - 0.5 * h)
    F = f_inf * (1 - torch.exp(-t / TAU_F))
    PV = 1 + (pv_inf - 1) * (1 - torch.exp(-t / TAU_PV))
    B = torch.exp(-K_B * (1 - g) * (1 - 0.7 * e) * t)
    M = torch.exp(-K_M * (1 - g) * (1 - 0.6 * e - 0.2 * p) * t)
    return torch.stack([F, PV, B, M], dim=-1)


class PhysiologyODE(nn.Module):
    """Neural-ODE correction on top of the space-physiology prior, with PINN losses."""

    def __init__(self, ctx_dim: int, hidden: int = 48, horizon_days: int = 14, dt: float = 1.0):
        super().__init__()
        self.horizon_days, self.dt = horizon_days, dt
        self.ctx_proj = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.Tanh())
        self.f_net = nn.Sequential(nn.Linear(4 + 4 + hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                   nn.Tanh(), nn.Linear(hidden, 4))
        self.init_net = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.f_net[-1].weight)  # start as the pure physics model
        nn.init.zeros_(self.f_net[-1].bias)

    # -------------------------------------------------------------------- dynamics
    def correction(self, z: torch.Tensor, u: torch.Tensor, ctx_h: torch.Tensor) -> torch.Tensor:
        raw = self.f_net(torch.cat([scale_state(z), u, ctx_h], dim=-1))
        return torch.tanh(raw) * CORR_SCALE.to(z)

    def rhs(self, z, u, ctx_h):
        return physics_rates(z, u) + self.correction(z, u, ctx_h)

    def initial_state(self, mission_day: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor):
        """State at the current mission day: physics closed form + learned patient offset."""
        ctx_h = self.ctx_proj(ctx)
        z0 = physics_closed_form(mission_day.reshape(-1) * 180.0, u)
        offset = torch.tanh(self.init_net(ctx_h)) * Z_SCALE.to(z0) * 0.5
        z0 = z0 + offset
        return z0, ctx_h

    def rollout(self, z0: torch.Tensor, u: torch.Tensor, ctx_h: torch.Tensor, days: int, dt: float | None = None):
        """Integrate with fixed-step RK4. Returns the state at integer days: (B, days+1, 4)."""
        dt = dt or self.dt
        steps_per_day = max(1, int(round(1.0 / dt)))
        dt = 1.0 / steps_per_day
        z, out = z0, [z0]
        for _ in range(days):
            for _ in range(steps_per_day):
                k1 = self.rhs(z, u, ctx_h)
                k2 = self.rhs(z + 0.5 * dt * k1, u, ctx_h)
                k3 = self.rhs(z + 0.5 * dt * k2, u, ctx_h)
                k4 = self.rhs(z + dt * k3, u, ctx_h)
                z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(z)
        return torch.stack(out, dim=1)

    def forward(self, mission_day: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        z0, ctx_h = self.initial_state(mission_day, u, ctx)
        traj = self.rollout(z0, u, ctx_h, self.horizon_days)
        zh = traj[:, -1]
        # Compact physics summary handed to the risk head: current + end-of-horizon state (scaled).
        summary = torch.cat([scale_state(z0), scale_state(zh)], dim=-1)
        return {"z0": z0, "traj": traj, "ctx_h": ctx_h, "summary": summary}

    @torch.no_grad()
    def project(self, mission_day: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor, days: int):
        """Long-range projection (e.g. 180 days) with coarse RK4 steps, for the UI."""
        z0, ctx_h = self.initial_state(mission_day, u, ctx)
        steps = max(days, 1)
        z, out = z0, [z0]
        dt = 1.0  # RK4 is stable for dt=1 d here (tau_F = 0.7 d is handled by the closed-form init)
        for _ in range(steps):
            k1 = self.rhs(z, u, ctx_h)
            k2 = self.rhs(z + 0.5 * dt * k1, u, ctx_h)
            k3 = self.rhs(z + 0.5 * dt * k2, u, ctx_h)
            k4 = self.rhs(z + dt * k3, u, ctx_h)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(z)
        return torch.stack(out, dim=1)

    # --------------------------------------------------------------------- PINN losses
    def pinn_losses(self, traj: torch.Tensor, u: torch.Tensor, ctx_h: torch.Tensor) -> Dict[str, torch.Tensor]:
        # (1) Trajectory residual: finite-difference derivative vs the physics prior.
        dz = traj[:, 1:] - traj[:, :-1]  # per day
        z_mid = 0.5 * (traj[:, 1:] + traj[:, :-1])
        B_, T_, _ = z_mid.shape
        u_rep = u.unsqueeze(1).expand(B_, T_, 4).reshape(-1, 4)
        f_phys = physics_rates(z_mid.reshape(-1, 4), u_rep).reshape(B_, T_, 4)
        # The residual is allowed to be as large as the permitted neural correction; in scaled
        # units it should be <~ 1, so we only penalise the part beyond that.
        resid = (dz - f_phys) / CORR_SCALE.to(traj)
        residual_loss = torch.relu(resid.abs() - 1.0).pow(2).mean()

        # (2) Collocation points: random (state, controls) pairs across the physical box.
        n = 256
        zc = Z_CENTER.to(traj) + Z_SCALE.to(traj) * torch.empty(n, 4, device=traj.device).uniform_(-2.0, 2.0)
        zc[:, 0] = zc[:, 0].abs()
        uc = torch.rand(n, 4, device=traj.device)
        ctx_c = ctx_h[torch.randint(0, ctx_h.shape[0], (n,), device=traj.device)]
        corr = self.correction(zc, uc, ctx_c) / CORR_SCALE.to(traj)
        collocation_loss = corr.pow(2).mean()  # keep the learned part a modest correction

        # (3) Hard physical constraints.
        F, PV, B, M = traj.unbind(-1)
        g = u[:, 0:1]
        bounds = (torch.relu(-F).pow(2).mean() + torch.relu(PV - 1.05).pow(2).mean()
                  + torch.relu(0.6 - PV).pow(2).mean() + torch.relu(B - 1.0).pow(2).mean() * 100
                  + torch.relu(M - 1.0).pow(2).mean() * 100 + torch.relu(0.5 - B).pow(2).mean()
                  + torch.relu(0.5 - M).pow(2).mean())
        # In (partial) microgravity bone and muscle must not increase over time.
        mono = ((torch.relu(B[:, 1:] - B[:, :-1]) + torch.relu(M[:, 1:] - M[:, :-1])) * (g < 0.9).float()).mean() * 1e3
        return {"residual": residual_loss, "collocation": collocation_loss, "constraints": bounds + mono}


def project_summary(traj: torch.Tensor) -> Dict[str, float]:
    """Human-readable summary of a single (T+1, 4) trajectory (per-cent changes vs baseline)."""
    z = traj.detach().cpu()
    return {
        "fluid_shift_L": float(z[-1, 0]),
        "plasma_volume_change_pct": float((z[-1, 1] - 1.0) * 100.0),
        "bone_density_change_pct": float((z[-1, 2] - 1.0) * 100.0),
        "muscle_mass_change_pct": float((z[-1, 3] - 1.0) * 100.0),
    }
