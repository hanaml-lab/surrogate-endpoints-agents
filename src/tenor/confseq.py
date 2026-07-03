"""Confidence sequences: intervals that hold at every sample size at once, so a gate may check them as often as it
likes without inflating its error.

Two constructions are used:

* ``betting_cs``: the predictable plug-in empirical-Bernstein (PrPl-EB) sequence of Waudby-Smith and Ramdas (2024)
  for the mean of observations in [0, 1]. It is valid in finite samples for any predictable choice of bets; the bets
  here are tuned to be tightest near a target sample size.
* ``asymptotic_cs``: the Gaussian-mixture asymptotic sequence of Waudby-Smith, Arbour, Sinha, Kennedy and Ramdas
  (2024), for observations with finite variance but no known bound (the censoring-weighted rectifier terms). It is
  used from a burn-in sample size on, before which it reports nothing.

Both return arrays ``lo, hi`` of the bounds after 1, 2, ..., n observations, with running intersection, so the
bounds never widen.
"""
from __future__ import annotations

import math

import numpy as np


def betting_cs(x: np.ndarray, alpha: float, t_star: float, c: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """PrPl-EB betting confidence sequence for the mean of x in [0, 1].

    Parameters
    ----------
    x : observations in [0, 1], in the order they arrive.
    alpha : error level; the sequence covers the mean at every n with probability at least 1 - alpha.
    t_star : the sample size near which the sequence should be tightest.
    c : cap on the bets, in (0, 1).
    """
    x = np.asarray(x, dtype=float)
    if x.size and (x.min() < -1e-12 or x.max() > 1 + 1e-12):
        raise ValueError("betting_cs needs observations in [0, 1]; rescale them first")
    n = x.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    t = np.arange(1, n + 1, dtype=float)
    mu_hat = (0.5 + np.cumsum(x)) / (t + 1)
    sig2 = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1)
    sig2_prev = np.concatenate([[0.25], sig2[:-1]])
    mu_prev = np.concatenate([[0.5], mu_hat[:-1]])
    lam = np.minimum(np.sqrt(2 * math.log(2 / alpha) / (sig2_prev * max(t_star, 1.0))), c)
    v = 4 * (x - mu_prev) ** 2
    psi = (-np.log1p(-lam) - lam) / 4
    sl = np.cumsum(lam)
    center = np.cumsum(lam * x) / sl
    half = (math.log(2 / alpha) + np.cumsum(v * psi)) / sl
    lo = np.maximum.accumulate(np.clip(center - half, 0, 1))
    hi = np.minimum.accumulate(np.clip(center + half, 0, 1))
    return lo, hi


def betting_cs_range(d: np.ndarray, alpha: float, lower: float, upper: float, t_star: float) -> tuple[np.ndarray, np.ndarray]:
    """Betting CS for observations known to lie in [lower, upper]: rescaled to [0, 1] and back."""
    width = upper - lower
    lo, hi = betting_cs((np.asarray(d, float) - lower) / width, alpha, t_star)
    return lower + width * lo, lower + width * hi


def asymptotic_cs(z: np.ndarray, alpha: float, t_star: float, burn_in: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-mixture asymptotic confidence sequence for the mean of z.

    The mixture parameter is chosen to minimise the width at ``t_star`` observations. Before ``burn_in``
    observations the bounds are infinite: the variance estimate is not yet reliable enough to lean on.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    t = np.arange(1, n + 1, dtype=float)
    mean = np.cumsum(z) / t
    ss = np.cumsum(z * z) - t * mean ** 2
    var = np.where(t > 1, ss / np.maximum(t - 1, 1), np.inf)
    rho2 = (-2 * math.log(alpha) + math.log(-2 * math.log(alpha) + 1)) / max(t_star, 1.0)
    half = np.sqrt(np.maximum(var, 1e-12)) * np.sqrt(
        2 * (t * rho2 + 1) / (t * t * rho2) * np.log(np.sqrt(t * rho2 + 1) / alpha)
    )
    half[: max(burn_in, 1) - 1] = np.inf
    lo = np.maximum.accumulate(mean - half)
    hi = np.minimum.accumulate(mean + half)
    return lo, hi


def bounds_at(lo: np.ndarray, hi: np.ndarray, counts: np.ndarray, default: tuple[float, float] = (-np.inf, np.inf)):
    """The bounds after ``counts`` observations, for an array of counts; a count of 0 gives ``default``."""
    counts = np.asarray(counts, dtype=int)
    out_lo = np.full(counts.shape, default[0], dtype=float)
    out_hi = np.full(counts.shape, default[1], dtype=float)
    seen = counts > 0
    out_lo[seen] = lo[counts[seen] - 1]
    out_hi[seen] = hi[counts[seen] - 1]
    return out_lo, out_hi
