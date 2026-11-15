"""
Held-out evaluation of AstrocyteX-Net (v3.0) - rigorous out-of-sample benchmarking.

    python -m models.evaluate                         # default: 6000 unseen crew-days, 1000 bootstrap resamples
    python -m models.evaluate --test-samples 2000 --boot 300

What is evaluated
-----------------
The task the crew sees is a 5-class *ordinal* classification for every parameter of every
crew-day: which colour band (gray / green / yellow / red / black) is the risk in?  Classes are
heavily imbalanced (black ~2 %), and errors are asymmetric (missing a red/black alarm is far
worse than a false yellow).  Therefore (see ``models/metrics.py``):

* primary metric: **MCC** (multi-class Matthews correlation) with a 95 % bootstrap CI;
* complementary: **macro-F1** (rare classes count equally), **quadratic weighted kappa**
  (ordinal distance), balanced accuracy, per-class precision/recall, confusion matrix;
* safety: **critical recall** (red/black detected as red/black), **under-triage rate**,
  black recall and the binary "actionable" (task-triggering) F1 / MCC;
* regression: MAE / RMSE of the risk %, 90 % credible-interval coverage, forecast MAE.

Compared systems (identical, seed-disjoint test crews - never seen during training):

1. ``rule``      transparent reference-range score only (what a classic monitor shows)
2. ``ridge``     linear ML baseline: Ridge residual on top of the rule score (scikit-learn)
3. ``mlp``       non-linear ML baseline: 2-hidden-layer MLP residual on top of rule (scikit-learn)
4. ``model``     AstrocyteX-Net v3.0 (Monte-Carlo mean)
5. ``deployed``  what the dashboard shows: 0.20 * rule + 0.80 * model with the red/black
                 safety floor
6. ``model_no_env``  ablation - the same network with all 12 environment inputs set to
                 neutral, to measure how much the environment data contribute
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from models.astroguard import load_checkpoint
from models.metrics import (BAND_NAMES, band_metrics, bands, bootstrap_ci, confusion, per_class_from_cm,
                            per_sample_confusions)
from models.rule_prior import RuleRiskTorch
from models.train import BATCH_KEYS, _to_tensors, default_device, predict_dataset
from utils.config import PROJECT_ROOT, get_settings
from utils.environment import EVENT_NAMES, MISSION_PROFILES, PROFILE_NAMES
from utils.features import N_DAILY_HEALTH, N_SEQ
from utils.reference_ranges import ENV_IDX, HEALTH_IDX, PARAMETERS, aggregate_risk

RULE_BLEND = 0.20


def _rule_risk(data: Dict[str, torch.Tensor]) -> np.ndarray:
    z_now = torch.cat([data["seq"][:, -1], data["daily"]], dim=-1).cpu()
    return RuleRiskTorch()(z_now).numpy()


def _health_overall(param_risk: np.ndarray) -> np.ndarray:
    return np.array([aggregate_risk(list(r[HEALTH_IDX])) for r in param_risk])


def _deployed(rule: np.ndarray, model: np.ndarray) -> np.ndarray:
    final = RULE_BLEND * rule + (1 - RULE_BLEND) * model
    final = np.where(rule >= 70.0, np.maximum(final, rule), final)  # safety floor
    return np.clip(final, 0, 100)


def _ridge_features(data: Dict[str, torch.Tensor], rule: np.ndarray) -> np.ndarray:
    z_now = torch.cat([data["seq"][:, -1], data["daily"]], dim=-1).cpu().numpy()
    hist = data["history"].cpu().numpy()
    trend = hist[:, -1] - hist[:, 0]  # 2-week change of every parameter
    return np.concatenate([z_now, np.clip(z_now, -6, 6) ** 2, trend, rule / 100.0, data["subj"].cpu().numpy(),
                           data["mission"].cpu().numpy(), data["controls"].cpu().numpy()], axis=1)


def _logit(p):
    p = np.clip(p, 0.01, 0.995)
    return np.log(p / (1 - p))


def _neutralise_env(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Ablation: every environment hazard set to its ideal value (z = 0) - Earth-like, benign."""
    out = {k: v.clone() for k, v in data.items()}
    out["daily"][:, N_DAILY_HEALTH:] = 0.0
    out["history"][:, :, ENV_IDX] = 0.0
    out["controls"][:, 0] = 1.0  # Earth gravity for the physiology ODE
    return out


def _summ(t_param: np.ndarray, p_param: np.ndarray, t_overall: np.ndarray, p_overall: np.ndarray) -> Dict[str, object]:
    tb, pb = bands(t_param), bands(p_param)
    res: Dict[str, object] = {
        "all_params": band_metrics(tb, pb),
        "health_params": band_metrics(tb[:, HEALTH_IDX], pb[:, HEALTH_IDX]),
        "environment_params": band_metrics(tb[:, ENV_IDX], pb[:, ENV_IDX]),
        "health_params_overall_band": band_metrics(bands(t_overall), bands(p_overall)),
        "mae_pct": float(np.abs(p_param - t_param).mean()),
        "mae_health_pct": float(np.abs(p_param[:, HEALTH_IDX] - t_param[:, HEALTH_IDX]).mean()),
        "rmse_pct": float(np.sqrt(((p_param - t_param) ** 2).mean())),
        "within_10pct": float(np.mean(np.abs(p_param - t_param) < 10)),
        "overall_mae_pct": float(np.abs(p_overall - t_overall).mean()),
    }
    return res


def evaluate_checkpoint(checkpoint: Path, test_samples: int = 6000, seed: int = 424242, n_boot: int = 1000,
                        n_mc: int = 16, workers: int = 1, out_dir: Path | None = None, verbose: bool = True) -> Dict:
    t0 = time.time()
    dev = default_device()
    model = load_checkpoint(checkpoint).to(dev)
    from utils.synthetic_data import generate_dataset

    if verbose:
        print(f"[eval] generating {test_samples} unseen test crew-days (seed {seed}) + baseline training set ...")
    test_np = generate_dataset(test_samples, seed=seed, workers=workers)
    base_np = generate_dataset(8000, seed=seed + 17, workers=workers)  # for the ridge baseline only
    test, base = _to_tensors(test_np, dev), _to_tensors(base_np, dev)

    tgt = test["risk"].cpu().numpy() * 100.0
    t_param, t_overall = tgt[:, :-1], tgt[:, -1]

    # ---- 1. rule baseline ------------------------------------------------------------------
    rule = _rule_risk(test)
    # ---- 2. ridge + mlp residual baselines (scikit-learn) ----------------------------------
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    rule_b = _rule_risk(base)
    xb, xt = _ridge_features(base, rule_b), _ridge_features(test, rule)
    yb = _logit(base["risk"].cpu().numpy()[:, :-1]) - _logit(rule_b / 100.0)
    scaler = StandardScaler().fit(xb)  # fitted on the baseline-training data only
    xb_s, xt_s = scaler.transform(xb), scaler.transform(xt)
    ridge = Ridge(alpha=3.0).fit(xb_s, yb)
    ridge_pred = 100.0 / (1.0 + np.exp(-(_logit(rule / 100.0) + ridge.predict(xt_s))))
    mlp = MLPRegressor(hidden_layer_sizes=(192, 96), activation="relu", alpha=1e-3, max_iter=70,
                       early_stopping=True, random_state=0).fit(xb_s, yb)
    mlp_pred = 100.0 / (1.0 + np.exp(-(_logit(rule / 100.0) + mlp.predict(xt_s))))
    # ---- 3. the network ---------------------------------------------------------------------
    tm = time.time()
    p = predict_dataset(model, test, n_mc=n_mc)
    infer_ms = (time.time() - tm) / test_samples * 1000
    model_param, model_overall = p["risk"][:, :-1], p["risk"][:, -1]
    # ---- 4. deployed blend --------------------------------------------------------------------
    dep = _deployed(rule, model_param)
    # ---- 5. ablation without environment inputs ----------------------------------------------
    p_noenv = predict_dataset(model, _neutralise_env(test), n_mc=n_mc)

    systems = {
        "rule": (rule, _health_overall(rule)),
        "ridge": (ridge_pred, _health_overall(ridge_pred)),
        "mlp": (mlp_pred, _health_overall(mlp_pred)),
        "model": (model_param, model_overall),
        "deployed": (dep, _health_overall(dep)),
        "model_no_env": (p_noenv["risk"][:, :-1], p_noenv["risk"][:, -1]),
    }
    results: Dict[str, object] = {name: _summ(t_param, sp, t_overall, so) for name, (sp, so) in systems.items()}

    # ---- uncertainty & forecast quality (network) ----------------------------------------------
    lo, hi = p["ci_low"], p["ci_high"]
    results["model"]["ci90_coverage"] = float(np.mean((tgt >= lo - 1e-6) & (tgt <= hi + 1e-6)))  # type: ignore[index]
    results["model"]["forecast_mae_pct"] = float(np.abs(p["forecast"] - test["forecast"].cpu().numpy() * 100).mean())  # type: ignore[index]
    results["model"]["inference_ms_per_crew_day"] = round(infer_ms, 3)  # type: ignore[index]

    # ---- bootstrap CIs (whole crew-days) ---------------------------------------------------------
    tb = bands(t_param)
    psc = {name: per_sample_confusions(tb, bands(sp)) for name, (sp, _so) in systems.items()}
    psc_h = {name: per_sample_confusions(tb[:, HEALTH_IDX], bands(sp[:, HEALTH_IDX])) for name, (sp, _so) in systems.items()}
    boot_metrics = ("mcc", "macro_f1", "qwk", "critical_recall", "under_triage_rate", "actionable_f1")
    boot = {name: bootstrap_ci(psc[name], boot_metrics, n_boot, seed=1) for name in systems}
    boot["health"] = {name: bootstrap_ci(psc_h[name], boot_metrics, n_boot, seed=1) for name in systems}
    boot["model_vs_rule"] = bootstrap_ci(psc["model"], boot_metrics, n_boot, seed=1, paired_with=psc["rule"])
    boot["model_vs_ridge"] = bootstrap_ci(psc["model"], boot_metrics, n_boot, seed=1, paired_with=psc["ridge"])
    boot["model_vs_mlp"] = bootstrap_ci(psc["model"], boot_metrics, n_boot, seed=1, paired_with=psc["mlp"])
    boot["model_vs_no_env"] = bootstrap_ci(psc["model"], boot_metrics, n_boot, seed=1, paired_with=psc["model_no_env"])
    boot["model_vs_no_env_health"] = bootstrap_ci(psc_h["model"], boot_metrics, n_boot, seed=1, paired_with=psc_h["model_no_env"])
    boot["deployed_vs_rule"] = bootstrap_ci(psc["deployed"], boot_metrics, n_boot, seed=1, paired_with=psc["rule"])

    # ---- cross-check against scikit-learn ----------------------------------------------------------
    from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef

    yb_t, yb_p = tb.ravel(), bands(model_param).ravel()
    sk = {"mcc": float(matthews_corrcoef(yb_t, yb_p)), "macro_f1": float(f1_score(yb_t, yb_p, average="macro")),
          "qwk": float(cohen_kappa_score(yb_t, yb_p, weights="quadratic"))}
    ours = results["model"]["all_params"]  # type: ignore[index]
    assert all(abs(sk[k] - ours[k]) < 1e-6 for k in sk), (sk, ours)

    # ---- confusion matrices & per-class ---------------------------------------------------------------
    cms = {name: confusion(tb, bands(sp)).tolist() for name, (sp, _so) in systems.items()}
    pcs = {}
    for name in ("model", "rule", "deployed"):
        pc = per_class_from_cm(np.array(cms[name]))
        pcs[name] = {b: {"precision": float(pc["precision"][i]), "recall": float(pc["recall"][i]), "f1": float(pc["f1"][i]),
                         "support": int(pc["support"][i])} for i, b in enumerate(BAND_NAMES)}

    # ---- slices: mission profile & environment event (health parameters) -------------------------------
    slices: Dict[str, Dict[str, Dict[str, float]]] = {"profile": {}, "event": {}}
    for kind, names, col in (("profile", PROFILE_NAMES, test_np["profile"][:, 0]),
                             ("event", EVENT_NAMES + ["none"], test_np["event"][:, 0])):
        for i, nm in enumerate(names):
            m = col == i
            if m.sum() < 20:
                continue
            row = {"n": int(m.sum())}
            for sysname in ("rule", "ridge", "mlp", "model", "deployed", "model_no_env"):
                sp = systems[sysname][0]
                bm = band_metrics(tb[m][:, HEALTH_IDX], bands(sp[m][:, HEALTH_IDX]))
                row[f"{sysname}_mcc"] = bm["mcc"]
                row[f"{sysname}_crit_recall"] = bm["critical_recall"]
                row[f"{sysname}_mae"] = float(np.abs(sp[m][:, HEALTH_IDX] - t_param[m][:, HEALTH_IDX]).mean())
            slices[kind][nm] = row

    # ---- class distribution of the test set ---------------------------------------------------------
    dist = np.bincount(tb.ravel(), minlength=5)
    report = {
        "checkpoint": str(checkpoint), "model_meta": {k: v for k, v in getattr(model, "meta", {}).items() if k != "history"},
        "test": {"crew_days": test_samples, "cells": int(tb.size), "seed": seed, "bootstrap_resamples": n_boot,
                 "mc_samples": n_mc, "band_distribution": {b: int(c) for b, c in zip(BAND_NAMES, dist)}},
        "results": results, "bootstrap_95ci": boot, "sklearn_crosscheck_model_all_params": sk,
        "confusion_matrices": cms, "per_class": pcs, "slices": slices,
        "seconds": round(time.time() - t0, 1),
    }
    out_dir = Path(out_dir or PROJECT_ROOT / "docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plots(report, out_dir / "img")
    (out_dir / "MODEL_EVALUATION.md").write_text(render_markdown(report), encoding="utf-8")
    if verbose:
        r = results
        for name in systems:
            a = r[name]["all_params"]  # type: ignore[index]
            print(f"[eval] {name:13s} MCC={a['mcc']:.4f} macroF1={a['macro_f1']:.4f} QWK={a['qwk']:.4f} "
                  f"crit.recall={a['critical_recall']:.4f} under-triage={a['under_triage_rate']:.4f} "
                  f"MAE={r[name]['mae_pct']:.3f}")
        print(f"[eval] report -> {out_dir / 'MODEL_EVALUATION.md'}  ({report['seconds']} s)")
    return report


# ----------------------------------------------------------------------------- plotting
def _plots(report: Dict, img_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib is optional
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), facecolor="#120922")
    for ax, name, title in ((axes[0], "rule", "Rule-Only Baseline"), (axes[1], "model", "AstrocyteX-Net v3.0")):
        ax.set_facecolor("#1a0e30")
        cm = np.array(report["confusion_matrices"][name], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Purples", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.52 else "#1d1033")
        ax.set_xticks(range(5), BAND_NAMES, color="#e8dff7")
        ax.set_yticks(range(5), BAND_NAMES, color="#e8dff7")
        ax.set_xlabel("Predicted Band", color="#cbbce6")
        ax.set_ylabel("True Band", color="#cbbce6")
        mcc = report["results"][name]["all_params"]["mcc"]
        ax.set_title(f"{title}  (MCC {mcc:.3f})", color="#f3edfc", fontweight="bold")
    fig.tight_layout()
    fig.savefig(img_dir / "confusion_matrix.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    ev = report["slices"]["event"]
    names = list(ev)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11.4, 4.2), facecolor="#120922")
    ax.set_facecolor("#1a0e30")
    palette = (("rule", "#6c6184", "Rule-Only"), ("model_no_env", "#b985d6", "v3.0 No-Env"), ("model", "#8b5cf6", "AstrocyteX-Net v3.0"))
    for k, (sysname, color, lbl) in enumerate(palette):
        ax.bar(x + (k - 1) * 0.27, [ev[n][f"{sysname}_mcc"] for n in names], 0.27, label=lbl, color=color, edgecolor="#2c1a4d")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8, color="#e8dff7")
    ax.tick_params(axis="y", colors="#cbbce6")
    ax.set_ylabel("MCC (Health Parameters)", color="#cbbce6")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#311d54", linestyle="--", alpha=0.6)
    leg = ax.legend(facecolor="#1a0e30", edgecolor="#4a2e7a", labelcolor="#f3edfc")
    for text in leg.get_texts():
        text.set_color("#f3edfc")
    ax.set_title("Per Environment Event: Rule Baseline vs. Network Without / With Environment Inputs",
                 color="#f3edfc", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#3d2569")
    fig.tight_layout()
    fig.savefig(img_dir / "mcc_by_event.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------------- markdown
def _f(v, d=4):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def render_markdown(r: Dict) -> str:
    res, boot, t = r["results"], r["bootstrap_95ci"], r["test"]
    meta = r.get("model_meta", {}).get("metrics", {})
    L = ["# AstrocyteX v3.0 - Held-Out Model Evaluation", "",
         "> Generated by `python -m models.evaluate`. All data are **simulated** crews (see `utils/synthetic_data.py`); "
         "the numbers measure agreement with the simulator's targets (reference ranges + physiological, symptom and "
         "environment interaction priors), **not clinical accuracy**.", "",
         "## Test Protocol", "",
         f"* **{t['crew_days']:,} unseen crew-days** (seed `{t['seed']}`, disjoint from training and validation seeds), "
         f"{t['cells']:,} parameter-level predictions (43 parameters each).",
         f"* Band distribution of the test cells: " + ", ".join(f"{k} {v:,}" for k, v in t["band_distribution"].items())
         + " - strongly imbalanced, black is rare.",
         f"* {t['bootstrap_resamples']} bootstrap resamples of whole crew-days for 95 % confidence intervals; "
         f"{t['mc_samples']} Monte-Carlo samples per prediction.",
         f"* Model: `{r['checkpoint']}` - {meta.get('params', '?'):,} parameters, trained on {meta.get('samples', '?'):,} "
         f"simulated crew-days, best epoch {meta.get('best_epoch', '?')} ({meta.get('device', '?')}).", "",
         "## Which Metric and Why", "",
         "The dashboard assigns every parameter to one of five **ordered, imbalanced** colour bands. "
         "Accuracy alone would reward a model that never raises a black alarm, so:", "",
         "| Metric | Role | Why it fits |", "|---|---|---|",
         "| **MCC** (multi-class Matthews) | **primary** | uses the whole confusion matrix, robust to class imbalance, high only if every band is predicted well |",
         "| Macro-F1 | complementary | unweighted mean of per-band F1 - the rare black band counts as much as gray |",
         "| Quadratic weighted kappa | complementary | the bands are ordinal; confusing gray with black is penalised far more than gray with green |",
         "| Critical recall / under-triage | safety | share of true red/black cases flagged red/black, and share shown as gray/green/yellow (missed alarms) |",
         "| Actionable F1 | operational | does the value correctly trigger AI mitigation tasks (yellow or worse)? |", "",
         "## Headline Results (All 43 Parameters, Test Set)", "",
         "| System | MCC [95 % CI] | Macro-F1 [95 % CI] | QWK | Critical recall | Under-triage | Actionable F1 | MAE (pp) |",
         "|---|---|---|---|---|---|---|---|"]
    labels = {
        "rule": "Rule-only (reference ranges)",
        "ridge": "Ridge residual (scikit-learn)",
        "mlp": "MLP non-linear residual (scikit-learn)",
        "model": "**AstrocyteX-Net v3.0**",
        "deployed": "Deployed blend (0.20 rule + 0.80 model, safety floor)",
        "model_no_env": "Ablation: v3.0 without environment inputs",
    }
    for name in ("rule", "ridge", "mlp", "model_no_env", "model", "deployed"):
        a, b = res[name]["all_params"], boot[name]
        L.append(f"| {labels[name]} | {_f(a['mcc'])} [{_f(b['mcc']['lo'])}-{_f(b['mcc']['hi'])}] | "
                 f"{_f(a['macro_f1'])} [{_f(b['macro_f1']['lo'])}-{_f(b['macro_f1']['hi'])}] | {_f(a['qwk'])} | "
                 f"{_f(a['critical_recall'])} | {_f(a['under_triage_rate'])} | {_f(a['actionable_f1'])} | {_f(res[name]['mae_pct'], 3)} |")
    L += ["", "### Physiological Health Parameters Only (31 Biomarkers - Fair Environment Ablation)", "",
          "| System | MCC [95 % CI] | Macro-F1 [95 % CI] | QWK | Critical recall | Under-triage | Actionable F1 | MAE (pp) |",
          "|---|---|---|---|---|---|---|---|"]
    for name in ("rule", "model_no_env", "ridge", "mlp", "model", "deployed"):
        a, b = res[name]["health_params"], boot["health"][name]
        L.append(f"| {labels[name]} | {_f(a['mcc'])} [{_f(b['mcc']['lo'])}-{_f(b['mcc']['hi'])}] | "
                 f"{_f(a['macro_f1'])} [{_f(b['macro_f1']['lo'])}-{_f(b['macro_f1']['hi'])}] | {_f(a['qwk'])} | "
                 f"{_f(a['critical_recall'])} | {_f(a['under_triage_rate'])} | {_f(a['actionable_f1'])} | {_f(res[name]['mae_health_pct'], 3)} |")
    L += ["", "### Statistical Significance (Paired Bootstrap of Differences)", "",
          "| Comparison | Delta MCC 95 % CI | Delta Macro-F1 95 % CI | Delta critical recall 95 % CI | P(Delta MCC <= 0) |", "|---|---|---|---|---|"]
    for key, label in (("model_vs_rule", "v3.0 - rule (all 43)"),
                       ("model_vs_ridge", "v3.0 - ridge (all 43)"),
                       ("model_vs_mlp", "v3.0 - mlp (all 43)"),
                       ("model_vs_no_env_health", "v3.0 - v3.0 without environment (31 health params)"),
                       ("model_vs_no_env", "v3.0 - v3.0 without environment (all 43)"),
                       ("deployed_vs_rule", "deployed - rule (all 43)")):
        b = boot[key]
        L.append(f"| {label} | {_f(b['mcc']['diff_lo'])} .. {_f(b['mcc']['diff_hi'])} | {_f(b['macro_f1']['diff_lo'])} .. "
                 f"{_f(b['macro_f1']['diff_hi'])} | {_f(b['critical_recall']['diff_lo'])} .. {_f(b['critical_recall']['diff_hi'])} | "
                 f"{_f(b['mcc']['p_diff_le_0'], 3)} |")
    m = res["model"]
    L += ["", "## Network Details", "",
          "| Metric | Value |", "|---|---|",
          f"| Health parameters: MCC / macro-F1 / QWK | {_f(m['health_params']['mcc'])} / {_f(m['health_params']['macro_f1'])} / {_f(m['health_params']['qwk'])} |",
          f"| Environment parameters: MCC / macro-F1 | {_f(m['environment_params']['mcc'])} / {_f(m['environment_params']['macro_f1'])} |",
          f"| Overall health-risk band: MCC / macro-F1 / QWK | {_f(m['health_params_overall_band']['mcc'])} / {_f(m['health_params_overall_band']['macro_f1'])} / {_f(m['health_params_overall_band']['qwk'])} |",
          f"| Black-band recall | {_f(m['all_params']['black_recall'])} |",
          f"| Risk MAE / RMSE (percentage points) | {_f(m['mae_pct'], 3)} / {_f(m['rmse_pct'], 3)} |",
          f"| Predictions within +/-10 points | {_f(m['within_10pct'])} |",
          f"| 90 % credible-interval coverage | {_f(m['ci90_coverage'])} |",
          f"| 7-day forecast MAE (points) | {_f(m['forecast_mae_pct'], 3)} |",
          f"| Inference time per crew-day ({'GPU' if 'cuda' in str(meta.get('device', '')) else 'CPU'}, batched) | {m['inference_ms_per_crew_day']} ms |", "",
          "### Per-Band Precision / Recall (All Parameters)", "",
          "| Band | Support | Rule recall | **v3.0 precision** | **v3.0 recall** | **v3.0 F1** |", "|---|---|---|---|---|---|"]
    for b in BAND_NAMES:
        pm, pr = r["per_class"]["model"][b], r["per_class"]["rule"][b]
        L.append(f"| {b} | {pm['support']:,} | {_f(pr['recall'])} | {_f(pm['precision'])} | {_f(pm['recall'])} | {_f(pm['f1'])} |")
    L += ["", "![Confusion matrices](img/confusion_matrix.png)", "",
          "## Slice Analysis - Does the Network Use the Environment?", "",
          "MCC on the 31 health parameters per off-nominal environment event and per mission profile. "
          "`no-env` is the same network with all environment inputs neutralised.", "",
          "| Event | n | Rule MCC | v3.0 no-env MCC | **v3.0 MCC** | Rule crit. recall | **v3.0 crit. recall** |", "|---|---|---|---|---|---|---|"]
    for nm, s in r["slices"]["event"].items():
        L.append(f"| {nm} | {s['n']} | {_f(s['rule_mcc'])} | {_f(s['model_no_env_mcc'])} | **{_f(s['model_mcc'])}** | "
                 f"{_f(s['rule_crit_recall'])} | **{_f(s['model_crit_recall'])}** |")
    L += ["", "| Mission profile | n | Rule MCC | v3.0 no-env MCC | **v3.0 MCC** | v3.0 MAE |", "|---|---|---|---|---|---|"]
    for nm, s in r["slices"]["profile"].items():
        L.append(f"| {MISSION_PROFILES[nm]['en']} | {s['n']} | {_f(s['rule_mcc'])} | {_f(s['model_no_env_mcc'])} | "
                 f"**{_f(s['model_mcc'])}** | {_f(s['model_mae'], 3)} |")
    L += ["", "![MCC by event](img/mcc_by_event.png)", "",
          f"*scikit-learn cross-check (v3.0, all parameters):* MCC {_f(r['sklearn_crosscheck_model_all_params']['mcc'], 6)}, "
          f"macro-F1 {_f(r['sklearn_crosscheck_model_all_params']['macro_f1'], 6)}, "
          f"QWK {_f(r['sklearn_crosscheck_model_all_params']['qwk'], 6)} - identical to the NumPy implementation.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out evaluation of AstrocyteX-Net v3.0")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--test-samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--mc", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    evaluate_checkpoint(args.checkpoint or get_settings().checkpoint_path, args.test_samples, args.seed, args.boot,
                        args.mc, args.workers, args.out)


if __name__ == "__main__":
    main()
