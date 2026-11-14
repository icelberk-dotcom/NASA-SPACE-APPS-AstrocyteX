"""
Train AstroGuard-X on simulated crews.

    python -m models.train --samples 4000 --epochs 12

The composite loss mirrors the architecture:

    L =  NLL(risk)                    heteroscedastic Gaussian negative log-likelihood (aleatoric)
       + beta * KL / N                Bayes-by-Backprop complexity term (epistemic)
       + w_f  * MSE(forecast)         7-day risk trajectory (TCN + TFT branch)
       + w_p  * MSE(physics traj)     Neural-ODE rollout vs. simulated space physiology
       + w_r  * PINN residual         dz/dt must follow the physics prior
       + w_c  * collocation + constraints

Evaluation reports risk MAE and the *indicator-colour agreement* (does the model put a
parameter into the same gray/green/yellow/red/black band as the target?).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from models.astroguard import AstroGuardNet, ModelConfig, save_checkpoint
from models.physics import Z_SCALE
from models.uncertainty import gaussian_nll
from utils.config import get_settings
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from utils.reference_ranges import LEVEL_ORDER, risk_level
from utils.synthetic_data import generate_dataset

KL_BETA = 5e-3  # tempered Bayesian posterior: full-strength KL would swamp the data term
BATCH_KEYS = ["seq", "daily", "history", "subj", "mission", "controls"]


def _to_tensors(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in data.items() if k != "scenario"}


def _slice(t: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in t.items()}


def compute_loss(model: AstroGuardNet, batch: Dict[str, torch.Tensor], n_train: int, kl_weight: float = 1.0):
    out = model(batch)
    pred = torch.sigmoid(out["logit"])
    nll = gaussian_nll(pred, batch["risk"], out["logvar"])
    mse_mean = ((pred - batch["risk"]) ** 2).mean()  # direct pull on the mean (stabilises early training)
    kl = out["kl"] / n_train
    forecast = ((torch.sigmoid(out["forecast_logits"]) - batch["forecast"]) ** 2).mean()

    traj = out["physics"]["traj"]  # (B, 15, 4) physical units
    target = batch["phys"]
    scale = Z_SCALE.to(traj)
    phys_data = (((traj - target) / scale) ** 2).mean()
    z0_err = (((out["physics"]["z0"] - target[:, 0]) / scale) ** 2).mean()
    pinn = model.physics.pinn_losses(traj, batch["controls"], out["physics"]["ctx_h"])

    total = (nll + 4.0 * mse_mean + kl_weight * KL_BETA * kl + 2.0 * forecast + 1.0 * phys_data + 1.0 * z0_err
             + 0.5 * pinn["residual"] + 0.05 * pinn["collocation"] + 1.0 * pinn["constraints"])
    parts = {"loss": total.item(), "nll": nll.item(), "mse": mse_mean.item(), "kl": kl.item(),
             "forecast": forecast.item(), "phys": phys_data.item(), "pinn_res": pinn["residual"].item()}
    return total, parts


@torch.no_grad()
def evaluate(model: AstroGuardNet, data: Dict[str, torch.Tensor], n_mc: int = 12, bs: int = 256) -> Dict[str, float]:
    model.eval()
    preds, ci_lo, ci_hi, fc = [], [], [], []
    n = data["seq"].shape[0]
    for i in range(0, n, bs):
        b = {k: v[i:i + bs] for k, v in data.items() if k in BATCH_KEYS}
        o = model.predict(b, n_mc=n_mc)
        preds.append(o["risk"])
        ci_lo.append(o["ci_low"])
        ci_hi.append(o["ci_high"])
        fc.append(o["forecast"])
    pred = np.concatenate(preds)
    tgt = data["risk"].numpy() * 100.0
    lo, hi = np.concatenate(ci_lo), np.concatenate(ci_hi)
    # scikit-learn metrics: regression error + agreement of the 5 colour bands (ordinal)
    mae = float(mean_absolute_error(tgt.ravel(), pred.ravel()))
    bands_p = np.vectorize(lambda v: LEVEL_ORDER[risk_level(v)])(pred).ravel()
    bands_t = np.vectorize(lambda v: LEVEL_ORDER[risk_level(v)])(tgt).ravel()
    agree = float(accuracy_score(bands_t, bands_p))
    macro_f1 = float(f1_score(bands_t, bands_p, average="macro"))
    kappa = float(cohen_kappa_score(bands_t, bands_p, weights="quadratic"))
    near = float(np.mean(np.abs(pred - tgt) < 10))
    coverage = float(np.mean((tgt >= lo - 1e-6) & (tgt <= hi + 1e-6)))
    fmae = float(np.abs(np.concatenate(fc) - data["forecast"].numpy() * 100.0).mean())
    return {"risk_mae_pct": mae, "band_agreement": agree, "band_macro_f1": macro_f1, "band_kappa": kappa,
            "within_10pct": near, "ci90_coverage": coverage,
            "forecast_mae_pct": fmae}


def train(samples: int = 4000, epochs: int = 12, batch_size: int = 64, lr: float = 2e-3, seed: int = 0,
          out_path: Path | None = None, cfg: ModelConfig | None = None, verbose: bool = True) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = get_settings()
    out_path = Path(out_path or settings.checkpoint_path)

    if verbose:
        print(f"[train] simulating {samples} + {max(200, samples // 8)} crew-days ...")
    t0 = time.time()
    # Strict split BEFORE anything is fit: different seeds -> disjoint simulated crews.
    train_np = generate_dataset(samples, seed=seed)
    val_np = generate_dataset(max(200, samples // 8), seed=seed + 1000)
    train_t, val_t = _to_tensors(train_np), _to_tensors(val_np)
    if verbose:
        print(f"[train] data ready in {time.time() - t0:.1f}s")

    model = AstroGuardNet(cfg or ModelConfig())
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[train] AstroGuardNet: {n_params / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(samples / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)

    metrics: Dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(samples)
        agg: Dict[str, float] = {}
        nb = 0
        kl_w = min(1.0, epoch / max(1, epochs // 3))  # KL annealing: avoid posterior collapse early on
        for i in range(0, samples, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 4:
                continue
            batch = _slice(train_t, idx)
            loss, parts = compute_loss(model, batch, samples, kl_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        if verbose:
            print(f"[train] epoch {epoch:02d}/{epochs}  " + "  ".join(f"{k}={v / nb:.4f}" for k, v in agg.items()))

    metrics = evaluate(model, val_t)
    metrics.update({"samples": samples, "epochs": epochs, "params": n_params})
    if verbose:
        print("[train] validation:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    save_checkpoint(model, out_path, meta={"metrics": metrics, "data": "synthetic", "seed": seed})
    if verbose:
        print(f"[train] saved -> {out_path}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AstroGuard-X on simulated crew data")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    train(args.samples, args.epochs, args.batch_size, args.lr, args.seed, args.out)


if __name__ == "__main__":
    main()
