"""Tests for the neural building blocks (Mamba-2 SSD, TCN/TFT, fusion, physics, Bayesian head)."""
import numpy as np
import pytest
import torch

from models.astroguard import AstroGuardNet, ModelConfig, N_OUTPUTS
from models.mamba2 import Mamba2Block, ssd_chunked, ssd_reference_recurrence
from models.physics import PhysiologyODE, physics_closed_form, physics_rates
from models.uncertainty import BayesianLinear, BayesianRiskHead, exceed_probability
from utils.features import HIST_LEN, N_DAILY, N_PARAMS, N_SEQ, SEQ_LEN
from utils.subjective import SUBJ_DIM

torch.manual_seed(0)


def _batch(b=3):
    return {
        "seq": torch.randn(b, SEQ_LEN, N_SEQ), "daily": torch.randn(b, N_DAILY),
        "history": torch.randn(b, HIST_LEN, N_PARAMS), "subj": torch.rand(b, SUBJ_DIM),
        "mission": torch.rand(b, 1), "controls": torch.rand(b, 4),
    }


def test_ssd_chunked_matches_recurrence():
    b, l, h, p, n = 2, 32, 3, 4, 5
    X = torch.randn(b, l, h, p)
    A = -torch.rand(b, l, h) * 0.5
    B = torch.randn(b, l, h, n)
    C = torch.randn(b, l, h, n)
    y_fast, s_fast = ssd_chunked(X, A, B, C, chunk=8)
    y_ref, s_ref = ssd_reference_recurrence(X, A, B, C)
    assert torch.allclose(y_fast, y_ref, atol=1e-4)
    assert torch.allclose(s_fast, s_ref, atol=1e-4)


def test_mamba2_block_is_causal_and_handles_odd_lengths():
    blk = Mamba2Block(32, d_state=16, headdim=16, chunk_size=16).eval()
    x = torch.randn(2, 37, 32)  # not a multiple of the chunk size
    y = blk(x)
    assert y.shape == x.shape
    x2 = x.clone()
    x2[:, 30:] += 5.0  # changing the future must not change the past
    assert torch.allclose(blk(x2)[:, :30], y[:, :30], atol=1e-4)


def test_physics_rk4_matches_closed_form():
    ode = PhysiologyODE(ctx_dim=8, hidden=16)
    u = torch.tensor([[0.0, 0.5, 0.5, 0.5]])
    z0 = physics_closed_form(torch.tensor([0.0]), u)
    traj = [z0]
    z = z0
    dt = 0.25
    for _ in range(int(20 / dt)):
        k1 = physics_rates(z, u); k2 = physics_rates(z + .5 * dt * k1, u)
        k3 = physics_rates(z + .5 * dt * k2, u); k4 = physics_rates(z + dt * k3, u)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    exact = physics_closed_form(torch.tensor([20.0]), u)
    assert torch.allclose(z, exact, atol=1e-3)


def test_physics_microgravity_loses_bone_earth_does_not():
    ode = PhysiologyODE(ctx_dim=8, hidden=16)
    ctx = torch.zeros(2, 8)
    u = torch.tensor([[0.0, 0.0, 0.5, 0.5], [1.0, 0.0, 0.5, 0.5]])
    out = ode(torch.tensor([[0.5], [0.5]]), u, ctx)
    traj = out["traj"]
    assert traj[0, -1, 2] < traj[0, 0, 2]  # bone density falls in microgravity
    assert abs(float(traj[1, -1, 2] - traj[1, 0, 2])) < 1e-6  # unchanged at 1 g
    assert float(traj[0, 0, 0]) > 1.0  # fluid shifted cephalad


def test_pinn_losses_are_finite_and_differentiable():
    ode = PhysiologyODE(ctx_dim=8, hidden=16)
    ctx = torch.randn(4, 8, requires_grad=True)
    out = ode(torch.rand(4, 1), torch.rand(4, 4), ctx)
    losses = ode.pinn_losses(out["traj"], torch.rand(4, 4), out["ctx_h"])
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total) and ctx.grad is not None


def test_bayesian_linear_kl_and_sampling():
    layer = BayesianLinear(6, 3)
    x = torch.randn(5, 6)
    assert not torch.allclose(layer(x, sample=True), layer(x, sample=True))  # stochastic
    assert torch.allclose(layer(x, sample=False), layer(x, sample=False))  # posterior mean is deterministic
    assert layer.kl() > 0


def test_bayesian_head_mc_predict_outputs():
    head = BayesianRiskHead(16, 32, 5)
    out = head.mc_predict(torch.randn(4, 16), n_samples=8)
    for k in ("mean", "epistemic_std", "aleatoric_std", "ci_low", "ci_high"):
        assert out[k].shape == (4, 5)
    assert (out["ci_low"] <= out["mean"] + 1e-6).all() and (out["ci_high"] >= out["mean"] - 1e-6).all()
    assert exceed_probability(0.9, 0.05, 0.5) > 0.99 and exceed_probability(0.1, 0.05, 0.5) < 0.01


def test_full_model_forward_backward_and_predict():
    model = AstroGuardNet(ModelConfig(d_model=32, head_hidden=48, d_state=16))
    b = _batch()
    out = model(b)
    assert out["logit"].shape == (3, N_OUTPUTS)
    assert out["forecast_logits"].shape == (3, 7)
    assert out["physics"]["traj"].shape == (3, 15, 4)
    (out["logit"].mean() + out["kl"] * 1e-6 + out["forecast_logits"].mean()).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert sum(g is not None for g in grads) > 0.9 * len(grads)  # every branch receives gradient

    pred = model.predict(b, n_mc=4, project_days=30)
    assert pred["risk"].shape == (3, N_OUTPUTS) and (pred["risk"] >= 0).all() and (pred["risk"] <= 100).all()
    assert np.allclose(pred["modality_attention"].sum(1), 1.0, atol=1e-4)
    assert pred["physics_projection"].shape == (3, 31, 4)


def test_torch_rule_prior_matches_numpy_reference():
    """The differentiable rule score must agree with the NumPy reference-range implementation."""
    from models.rule_prior import RuleRiskTorch, aggregate_risk_torch
    from utils.reference_ranges import PARAMETERS, aggregate_risk, risk_percent_array

    rng = np.random.default_rng(1)
    z = rng.uniform(-5.5, 5.5, size=(64, len(PARAMETERS))).astype(np.float32)
    got = RuleRiskTorch()(torch.from_numpy(z)).numpy()
    for j, p in enumerate(PARAMETERS):
        values = p.ideal + z[:, j].astype(float) * p.scale
        expected = risk_percent_array(p.key, values)
        assert np.allclose(got[:, j], expected, atol=0.05), p.key
    agg = aggregate_risk_torch(torch.from_numpy(got)).numpy()
    assert np.allclose(agg, [aggregate_risk(list(r)) for r in got], atol=1e-3)


def test_untrained_model_equals_rule_score():
    """Residual learning: with the delta head near zero the model reproduces the rule score."""
    from utils.reference_ranges import PARAMETERS, risk_percent_array

    model = AstroGuardNet(ModelConfig(d_model=32, head_hidden=48, d_state=16))
    b = _batch(8)
    pred = model.predict(b, n_mc=16)["risk"][:, :N_PARAMS]
    z_now = torch.cat([b["seq"][:, -1], b["daily"]], dim=-1).numpy()
    rule = np.stack([risk_percent_array(p.key, p.ideal + z_now[:, j] * p.scale) for j, p in enumerate(PARAMETERS)], axis=1)
    assert np.abs(pred - rule).mean() < 8.0  # small perturbation from the freshly-initialised delta head
