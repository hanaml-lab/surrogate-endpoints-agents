"""Validation against the surrogate paradox, on past agent changes whose outcomes have matured.

For each past change c we have the outcome effect dY_c and the effects of the index and of its two sub-indices,
dS_c, dQ_c (controlled signals) and dW_c (independent signals), each computed with indices cross-fitted on the other
half of the change's runs. From them:

* gamma, the slope of dY on dS through the origin under Huber's loss: the trial-level surrogacy model of Daniels and
  Hughes (1997) and Buyse et al. (2000). 1 / gamma is the proportion of the effect the index explains.
* the divergence of each change, gamma_Q dQ - gamma_W dW: what the controlled signals predict minus what the
  independent ones predict. It is near zero for an honest change and large for a hack or for its removal.
* tau, three robust standard deviations of the divergence: a change whose divergence is shown to lie outside
  (-tau, tau) is flagged; one shown to lie inside is licensed for transport.
* kappa, the largest leave-one-out residual |dY_c - gamma_(-c) dS_c| among the changes the screen passes: a split-
  conformal (jackknife) bound on a new licensed change's residual, at level 1 / (n + 1).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PastChange:
    """One past change with matured outcomes; effects are differences of rates (candidate minus incumbent)."""

    name: str
    dY: float
    dS: float
    dQ: float
    dW: float
    proxy: float = float("nan")


def huber_slope(x: Sequence[float], y: Sequence[float], k: float = 1.345, iters: int = 50) -> float:
    """Slope of y on x through the origin under Huber's loss, by iteratively reweighted least squares."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    g = float(np.sum(x * y) / np.sum(x * x))
    for _ in range(iters):
        r = y - g * x
        scale = 1.4826 * np.median(np.abs(r)) + 1e-12
        w = np.minimum(1.0, k * scale / np.maximum(np.abs(r), 1e-12))
        g_new = float(np.sum(w * x * y) / np.sum(w * x * x))
        if abs(g_new - g) < 1e-10:
            return g_new
        g = g_new
    return g


@dataclass
class Validation:
    gamma: float
    gamma_q: float
    gamma_w: float
    tau: float
    kappa: float
    n_pass: int
    divergence: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    passed: np.ndarray = field(repr=False)

    @property
    def pte(self) -> float:
        """Proportion of the treatment effect explained by the index (Freedman et al., 1992)."""
        return 1.0 / self.gamma

    @property
    def alpha_kappa(self) -> float:
        """The error the margin spends: a new change's residual exceeds kappa with probability at most this."""
        return 1.0 / (self.n_pass + 1)

    def alphas(self, total: float = 0.05) -> dict[str, float]:
        """The error budget: alpha_kappa for the margin, the rest split equally over the index, divergence and
        residual sequences."""
        a_k = self.alpha_kappa
        if a_k >= total:
            raise ValueError(f"too few past changes pass the screen ({self.n_pass}) for a total error of {total}")
        rest = (total - a_k) / 3
        return dict(kappa=a_k, s=rest, d=rest, o=rest)


def validate(changes: Sequence[PastChange], tau_multiple: float = 3.0) -> Validation:
    """Fit the slopes, the screen's tolerance and the margin on past changes."""
    if len(changes) < 3:
        raise ValueError("validation needs at least three past changes")
    dY = np.array([c.dY for c in changes])
    dS = np.array([c.dS for c in changes])
    dQ = np.array([c.dQ for c in changes])
    dW = np.array([c.dW for c in changes])
    gamma, gq, gw = huber_slope(dS, dY), huber_slope(dQ, dY), huber_slope(dW, dY)
    div = gq * dQ - gw * dW
    tau = float(tau_multiple * 1.4826 * np.median(np.abs(div)))
    passed = np.abs(div) <= tau
    loo = np.array([huber_slope(np.delete(dS, i), np.delete(dY, i)) for i in range(len(dY))])
    resid = dY - loo * dS
    kappa = float(np.max(np.abs(resid[passed])))
    return Validation(gamma, gq, gw, tau, kappa, int(passed.sum()), div, resid, passed)


def sign_agreement(effect: Sequence[float], outcome: Sequence[float]) -> float:
    """Share of past changes whose proxy effect has the sign of their outcome effect."""
    e, o = np.asarray(effect, float), np.asarray(outcome, float)
    return float(np.mean(np.sign(e) == np.sign(o)))


def trial_r2(effect: Sequence[float], outcome: Sequence[float]) -> float:
    """Squared correlation of proxy and outcome effects across past changes (trial-level R^2, Buyse et al. 2000)."""
    e, o = np.asarray(effect, float), np.asarray(outcome, float)
    return float(np.corrcoef(e, o)[0, 1] ** 2)
