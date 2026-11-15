"""
Train AstrocyteX-Net (v3.0) on simulated crews.

    python -m models.train --samples 42000 --epochs 50          # full training (GPU if available)
    python -m models.train --samples 2000 --epochs 3 --workers 1  # quick smoke run

The composite loss mirrors the architecture:

    L =  NLL(risk)                    heteroscedastic Gaussian negative log-likelihood (aleatoric)
       + w_m  * MSE(risk)             direct pull on the mean
       + w_o  * ordinal band loss     cumulative-link BCE at the colour thresholds 15/40/70/90 %
       + beta * KL / N                Bayes-by-Backprop complexity term (epistemic)
       + w_f  * MSE(forecast)         7-day risk trajectory (TCN + TFT branch)
       + w_p  * MSE(physics traj)     Neural-ODE rollout vs. simulated space physiology
       + w_r  * PINN residual         dz/dt must follow the physics prior
       + w_c  * collocation + constraints

Training recipe (v3.0)
----------------------
* strictly disjoint simulated crews for train / validation (different seeds); the final
  test set used in ``models.evaluate`` uses yet another seed and is never seen here;
* AdamW + OneCycle learning-rate schedule, gradient clipping, KL annealing;
* the **ordinal band loss** optimises what the dashboard shows - the colour band -
  with larger weights on the safety-critical red/black thresholds;
* an **exponential moving average (EMA)** of the weights is evaluated every epoch and the
  best EMA weights by validation MCC are kept (early stopping).
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from models.astroguard import AstroGuardNet, ModelConfig, save_checkpoint
from models.metrics import band_metrics, bands
from models.physics import Z_SCALE
from models.uncertainty import gaussian_nll
from utils.config import get_settings
from utils.synthetic_data import META_KEYS, generate_dataset

KL_BETA = 5e-3  # tempered Bayesian posterior: full-strength KL would swamp the data term
BATCH_KEYS = ["seq", "daily", "history", "subj", "mission", "controls"]
BAND_THRESHOLDS = torch.tensor([0.15, 0.40, 0.70, 0.90])
BAND_WEIGHTS = torch.tensor([1.0, 1.15, 1.85, 2.75])  # red/black boundaries matter most clinically
ORDINAL_TAU = 0.016  # sharpness of the soft band boundaries (1.6 percentage points)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensors(data: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).float().to(device) for k, v in data.items() if k not in META_KEYS}


def _slice(t: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in t.items()}


def ordinal_band_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cumulative-link loss: for each colour threshold t_k, BCE( sigmoid((pred-t_k)/tau), 1[target > t_k] )."""
    th = BAND_THRESHOLDS.to(pred)
    w = BAND_WEIGHTS.to(pred)
    logits = (pred.unsqueeze(-1) - th) / ORDINAL_TAU
    labels = (target.unsqueeze(-1) > th).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (bce * w).mean()


def compute_loss(model: AstroGuardNet, batch: Dict[str, torch.Tensor], n_train: int, kl_weight: float = 1.0):
    out = model(batch)
    pred = torch.sigmoid(out["logit"])
    nll = gaussian_nll(pred, batch["risk"], out["logvar"])
    mse_mean = ((pred - batch["risk"]) ** 2).mean()  # direct pull on the mean (stabilises early training)
    ordinal = ordinal_band_loss(pred, batch["risk"])
    kl = out["kl"] / n_train
    forecast = ((torch.sigmoid(out["forecast_logits"]) - batch["forecast"]) ** 2).mean()

    traj = out["physics"]["traj"]  # (B, 15, 4) physical units
    target = batch["phys"]
    scale = Z_SCALE.to(traj)
    phys_data = (((traj - target) / scale) ** 2).mean()
    z0_err = (((out["physics"]["z0"] - target[:, 0]) / scale) ** 2).mean()
    pinn = model.physics.pinn_losses(traj, batch["controls"], out["physics"]["ctx_h"])

    total = (nll + 4.2 * mse_mean + 0.20 * ordinal + kl_weight * KL_BETA * kl + 2.0 * forecast + 1.0 * phys_data
             + 1.0 * z0_err + 0.5 * pinn["residual"] + 0.05 * pinn["collocation"] + 1.0 * pinn["constraints"])
    parts = {"loss": total.item(), "nll": nll.item(), "mse": mse_mean.item(), "ord": ordinal.item(), "kl": kl.item(),
             "forecast": forecast.item(), "phys": phys_data.item(), "pinn_res": pinn["residual"].item()}
    return total, parts


@torch.no_grad()
def predict_dataset(model: AstroGuardNet, data: Dict[str, torch.Tensor], n_mc: int = 12, bs: int = 512) -> Dict[str, np.ndarray]:
    model.eval()
    keys = ("risk", "ci_low", "ci_high", "forecast", "epistemic", "aleatoric", "modality_attention")
    acc: Dict[str, list] = {k: [] for k in keys}
    n = data["seq"].shape[0]
    for i in range(0, n, bs):
        b = {k: v[i:i + bs] for k, v in data.items() if k in BATCH_KEYS}
        o = model.predict(b, n_mc=n_mc)
        for k in keys:
            acc[k].append(o[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


@torch.no_grad()
def evaluate(model: AstroGuardNet, data: Dict[str, torch.Tensor], n_mc: int = 12, bs: int = 512) -> Dict[str, float]:
    """Validation metrics on all 43 parameter outputs + the overall health risk."""
    p = predict_dataset(model, data, n_mc, bs)
    pred, lo, hi = p["risk"], p["ci_low"], p["ci_high"]
    tgt = data["risk"].cpu().numpy() * 100.0
    bm = band_metrics(bands(tgt), bands(pred))
    return {"risk_mae_pct": float(np.abs(pred - tgt).mean()),
            "band_accuracy": bm["accuracy"], "band_mcc": bm["mcc"], "band_macro_f1": bm["macro_f1"], "band_qwk": bm["qwk"],
            "critical_recall": bm["critical_recall"], "under_triage_rate": bm["under_triage_rate"],
            "within_10pct": float(np.mean(np.abs(pred - tgt) < 10)),
            "ci90_coverage": float(np.mean((tgt >= lo - 1e-6) & (tgt <= hi + 1e-6))),
            "forecast_mae_pct": float(np.abs(p["forecast"] - data["forecast"].cpu().numpy() * 100.0).mean())}


def train(samples: int = 30000, epochs: int = 40, batch_size: int = 128, lr: float = 2e-3, seed: int = 0,
          out_path: Path | None = None, cfg: ModelConfig | None = None, verbose: bool = True, workers: int = 1,
          val_samples: Optional[int] = None, patience: int = 10, device: Optional[str] = None,
          ema_decay: float = 0.998) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = get_settings()
    out_path = Path(out_path or settings.checkpoint_path)
    dev = torch.device(device) if device else default_device()
    n_val = val_samples or max(400, samples // 10)

    if verbose:
        print(f"[train] device={dev}  simulating {samples} train + {n_val} validation crew-days (workers={workers}) ...")
    t0 = time.time()
    # Strict split BEFORE anything is fit: different seeds -> disjoint simulated crews.
    train_np = generate_dataset(samples, seed=seed, workers=workers)
    val_np = generate_dataset(n_val, seed=seed + 1000, workers=workers)
    train_t, val_t = _to_tensors(train_np, dev), _to_tensors(val_np, dev)
    if verbose:
        print(f"[train] data ready in {time.time() - t0:.1f}s")

    model = AstroGuardNet(cfg or ModelConfig()).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[train] AstrocyteX-Net v3.0: {n_params / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(samples / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay))

    best_score, best_state, best_metrics, best_epoch, bad = -1e9, None, {}, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        te = time.time()
        perm = torch.randperm(samples, device=dev)
        agg: Dict[str, float] = {}
        nb = 0
        kl_w = min(1.0, epoch / max(1, epochs // 4))  # KL annealing: avoid posterior collapse early on
        for i in range(0, samples, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 4:
                continue
            loss, parts = compute_loss(model, _slice(train_t, idx), samples, kl_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        m = evaluate(ema.module, val_t, n_mc=8)
        score = m["band_mcc"] - 0.01 * m["risk_mae_pct"]
        history.append({"epoch": epoch, **{k: round(v / nb, 5) for k, v in agg.items()}, **m})
        improved = score > best_score + 1e-5
        if improved:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(ema.module.state_dict())
            best_metrics = m
        else:
            bad += 1
        if verbose:
            print(f"[train] epoch {epoch:02d}/{epochs} {time.time() - te:5.1f}s  "
                  + " ".join(f"{k}={v / nb:.4f}" for k, v in agg.items() if k in ("loss", "mse", "ord", "forecast", "phys"))
                  + f" | val MAE={m['risk_mae_pct']:.3f} MCC={m['band_mcc']:.4f} F1={m['band_macro_f1']:.4f}"
                  + f" QWK={m['band_qwk']:.4f} crit.recall={m['critical_recall']:.4f}" + ("  *" if improved else ""))
        if bad >= patience:
            if verbose:
                print(f"[train] early stopping (no improvement for {patience} epochs)")
            break

    final = AstroGuardNet(cfg or ModelConfig())
    final.load_state_dict({k: v.cpu() for k, v in best_state.items()})
    metrics = dict(best_metrics)
    metrics.update({"samples": samples, "val_samples": n_val, "epochs_run": len(history), "best_epoch": best_epoch,
                    "params": n_params, "device": str(dev), "train_seconds": round(time.time() - t0, 1)})
    if verbose:
        print("[train] best validation (EMA):", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    save_checkpoint(final, out_path, meta={"metrics": metrics, "data": "synthetic (environment-aware AstrocyteX v3.0)", "seed": seed,
                                           "history": history})
    if verbose:
        print(f"[train] saved -> {out_path}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AstrocyteX-Net v3.0 on simulated crew data")
    ap.add_argument("--samples", type=int, default=42000)
    ap.add_argument("--val-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    train(args.samples, args.epochs, args.batch_size, args.lr, args.seed, args.out, workers=args.workers,
          val_samples=args.val_samples, patience=args.patience, device=args.device)


if __name__ == "__main__":
    main()
