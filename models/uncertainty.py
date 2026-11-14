"""
Bayesian uncertainty quantification for the risk estimates.

Two complementary kinds of uncertainty are reported for every risk score:

* **Epistemic** ("the model doesn't know") - captured with *Bayes-by-Backprop*
  linear layers whose weights are Gaussian distributions q(w) = N(mu, sigma^2).
  Predicting repeatedly with freshly sampled weights (plus MC-dropout) gives a
  spread of answers; the variance of that spread is the epistemic variance.
* **Aleatoric** ("the data is noisy") - a heteroscedastic head predicts a
  log-variance for each output, trained with a Gaussian negative log-likelihood.

The total predictive variance is epistemic + aleatoric, which yields a credible
interval and the probability that the true risk exceeds a decision threshold.

References
----------
* C. Blundell et al. "Weight Uncertainty in Neural Networks", ICML 2015.
* Y. Gal, Z. Ghahramani. "Dropout as a Bayesian Approximation", ICML 2016.
* A. Kendall, Y. Gal. "What Uncertainties Do We Need in Bayesian Deep Learning for
  Computer Vision?", NeurIPS 2017.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Linear layer with a factorised-Gaussian posterior over weights and biases."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.5, init_rho: float = -5.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.prior_sigma = prior_sigma
        bound = 1.0 / math.sqrt(in_features)
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), init_rho))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), init_rho))

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)  # keeps sigma > 0

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:  # reparameterisation trick: w = mu + sigma * eps
            w = self.w_mu + self._sigma(self.w_rho) * torch.randn_like(self.w_mu)
            b = self.b_mu + self._sigma(self.b_rho) * torch.randn_like(self.b_mu)
        else:  # posterior mean
            w, b = self.w_mu, self.b_mu
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        """KL( q(w) || N(0, prior_sigma^2) ), summed over all weights (closed form)."""
        def _kl(mu, rho):
            sigma = self._sigma(rho)
            return (torch.log(self.prior_sigma / sigma) + (sigma ** 2 + mu ** 2) / (2 * self.prior_sigma ** 2) - 0.5).sum()

        return _kl(self.w_mu, self.w_rho) + _kl(self.b_mu, self.b_rho)


class BayesianRiskHead(nn.Module):
    """Bayesian MLP head: features -> (risk logit, aleatoric log-variance) per output."""

    def __init__(self, in_dim: int, hidden: int, n_outputs: int, dropout: float = 0.15):
        super().__init__()
        self.l1 = BayesianLinear(in_dim, hidden)
        self.l2 = BayesianLinear(hidden, hidden)
        self.mu_out = BayesianLinear(hidden, n_outputs)
        with torch.no_grad():  # start close to "delta = 0", i.e. exactly the rule score
            self.mu_out.w_mu.mul_(0.1)
        self.logvar_out = nn.Linear(hidden, n_outputs)  # deterministic aleatoric head
        self.drop = nn.Dropout(dropout)
        self.n_outputs = n_outputs

    def forward(self, x: torch.Tensor, base_logit: Optional[torch.Tensor] = None, sample: bool = True,
                mc_dropout: bool = True):
        """Return (logit, aleatoric log-variance).  With ``base_logit`` the head learns a *residual*
        delta on top of it (residual learning on the transparent rule score)."""
        h = F.gelu(self.l1(x, sample))
        h = F.dropout(h, p=self.drop.p, training=mc_dropout or self.training)
        h = F.gelu(self.l2(h, sample))
        h = F.dropout(h, p=self.drop.p, training=mc_dropout or self.training)
        logit = self.mu_out(h, sample)
        if base_logit is not None:
            logit = base_logit + logit
        logvar = self.logvar_out(h).clamp(-8.0, 2.0)
        return logit, logvar

    def kl(self) -> torch.Tensor:
        return self.l1.kl() + self.l2.kl() + self.mu_out.kl()

    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, base_logit: Optional[torch.Tensor] = None, n_samples: int = 24) -> Dict[str, torch.Tensor]:
        """Monte-Carlo prediction: mean risk, epistemic / aleatoric std and a 90 % interval."""
        probs, alea = [], []
        for _ in range(max(2, n_samples)):
            logit, logvar = self.forward(x, base_logit, sample=True, mc_dropout=True)
            probs.append(torch.sigmoid(logit))
            # The NLL is applied in probability space, so exp(logvar) already is the
            # aleatoric variance of the risk score itself.
            alea.append(torch.exp(logvar))
        P = torch.stack(probs)  # (S, B, O)
        mean = P.mean(0)
        epistemic_var = P.var(0, unbiased=True)
        aleatoric_var = torch.stack(alea).mean(0)
        total_std = torch.sqrt(epistemic_var + aleatoric_var)
        z90 = 1.645
        return {
            "mean": mean,
            "epistemic_std": epistemic_var.sqrt(),
            "aleatoric_std": aleatoric_var.sqrt(),
            "total_std": total_std,
            "ci_low": (mean - z90 * total_std).clamp(0, 1),
            "ci_high": (mean + z90 * total_std).clamp(0, 1),
        }


def exceed_probability(mean: float, std: float, threshold: float) -> float:
    """P(risk > threshold) under a normal approximation N(mean, std^2)."""
    if std <= 1e-9:
        return float(mean > threshold)
    z = (threshold - mean) / (std * math.sqrt(2.0))
    return float(0.5 * math.erfc(z))


def gaussian_nll(pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL (up to a constant): 0.5*exp(-s)*(y-p)^2 + 0.5*s."""
    return (0.5 * torch.exp(-logvar) * (target - pred) ** 2 + 0.5 * logvar).mean()
