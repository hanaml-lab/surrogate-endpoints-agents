"""TENOR's release gate (Algorithm 1 of the report).

A release pairs the k-th run of the candidate B with the k-th run of the incumbent A, in order of arrival. For
each pair, three contrasts arrive at different ages:

* the index contrast d_k = S(X_Bk) - S(X_Ak), once both runs' signals are complete;
* the divergence contrast gamma_Q (S_Q,Bk - S_Q,Ak) - gamma_W (S_W,Bk - S_W,Ak), once the 24-hour signals are in;
* the rectifier contrast e_k = R_Bk - R_Ak, once both runs reach the landmark age.

The gate keeps a confidence sequence for each and combines them: the design-based interval [L_S + L_e, U_S + U_e]
and, while the divergence screen licenses transport, the transported interval [gamma L_S - kappa, gamma U_S + kappa].
It reports their intersection, falls back to the design-based interval for good if they ever disagree, ships once
the lower bound exceeds -delta and blocks once the upper bound falls below 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .confseq import asymptotic_cs, betting_cs_range, bounds_at
from .validate import Validation

SHIP, BLOCK, WAIT = 1, -1, 0


@dataclass
class Streams:
    """The three paired contrasts of a release and the times (days since release) at which each becomes available."""

    d_index: np.ndarray
    t_index: np.ndarray
    d_div: np.ndarray
    t_div: np.ndarray
    e_rect: np.ndarray
    t_rect: np.ndarray


@dataclass
class GateTrace:
    times: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    licensed: np.ndarray
    flagged: np.ndarray
    decision: int
    decided_at: float | None

    def covered(self, truth: float) -> bool:
        return bool(np.all((self.lo <= truth) & (truth <= self.hi)))


def decide(lo: np.ndarray, hi: np.ndarray, delta: float) -> tuple[int, int]:
    """The first check at which the gate decides, and the decision. Shipping takes precedence."""
    ship = lo > -delta
    block = hi < 0
    either = ship | block
    if not either.any():
        return -1, WAIT
    j = int(np.argmax(either))
    return j, (SHIP if ship[j] else BLOCK)


def combine(S: tuple[np.ndarray, np.ndarray], e: tuple[np.ndarray, np.ndarray], div: tuple[np.ndarray, np.ndarray],
            gamma: float, kappa: float, tau: float, screen: str = "two-sided"):
    """TENOR's interval at each check, from the index, residual and divergence bounds."""
    S_lo, S_hi = S
    e_lo, e_hi = e
    d_lo, d_hi = div
    if screen == "two-sided":
        licensed = (d_hi < tau) & (d_lo > -tau)
        flagged = np.logical_or.accumulate((d_lo > tau) | (d_hi < -tau))
    elif screen == "one-sided":
        licensed = d_hi < tau
        flagged = np.logical_or.accumulate(d_lo > tau)
    elif screen == "none":
        licensed = np.ones_like(d_lo, dtype=bool)
        flagged = np.zeros_like(d_lo, dtype=bool)
    else:
        raise ValueError(screen)
    D_lo, D_hi = S_lo + e_lo, S_hi + e_hi
    T_lo, T_hi = gamma * S_lo - kappa, gamma * S_hi + kappa
    falsified = np.logical_or.accumulate(licensed & ((T_lo > D_hi) | (T_hi < D_lo)))
    use = licensed & ~falsified
    lo = np.where(use, np.maximum(D_lo, T_lo), D_lo)
    hi = np.where(use, np.minimum(D_hi, T_hi), D_hi)
    return np.clip(lo, -1, 1), np.clip(hi, -1, 1), use, flagged | falsified


class ReleaseGate:
    """Run Algorithm 1 over a grid of check times."""

    def __init__(self, validation: Validation, delta: float = 0.02, total_alpha: float = 0.05,
                 t_star_index: float = 1000.0, t_star_rect: float = 3000.0, burn_in: int = 200):
        self.v = validation
        self.delta = delta
        self.a = validation.alphas(total_alpha)
        self.t_star_index, self.t_star_rect, self.burn_in = t_star_index, t_star_rect, burn_in

    def run(self, streams: Streams, times: np.ndarray, screen: str = "two-sided") -> GateTrace:
        v, a = self.v, self.a
        n_idx = np.searchsorted(streams.t_index, times, side="right")
        n_div = np.searchsorted(streams.t_div, times, side="right")
        n_rect = np.searchsorted(streams.t_rect, times, side="right")
        S = bounds_at(*betting_cs_range(streams.d_index, a["s"], -1, 1, self.t_star_index), n_idx, (-1.0, 1.0))
        lim = v.gamma_q + v.gamma_w
        div = bounds_at(*betting_cs_range(np.clip(streams.d_div, -lim, lim), a["d"], -lim, lim, self.t_star_index),
                        n_div, (-lim, lim))
        e = bounds_at(*asymptotic_cs(streams.e_rect, a["o"], self.t_star_rect, self.burn_in), n_rect, (-2.0, 2.0))
        e = (np.maximum(e[0], -2.0), np.minimum(e[1], 2.0))
        lo, hi, licensed, flagged = combine(S, e, div, v.gamma, v.kappa, v.tau, screen)
        j, dec = decide(lo, hi, self.delta)
        return GateTrace(times, lo, hi, licensed, flagged, dec, float(times[j]) if j >= 0 else None)


def outcome_only(e_const: np.ndarray, t_rect: np.ndarray, times: np.ndarray, delta: float = 0.02, alpha: float = 0.05,
                 t_star: float = 3000.0, burn_in: int = 200) -> GateTrace:
    """Waiting for the outcome: the same gate on matured outcomes alone (the rectifier with a constant index)."""
    n = np.searchsorted(t_rect, times, side="right")
    lo, hi = bounds_at(*asymptotic_cs(e_const, alpha, t_star, burn_in), n, (-1.0, 1.0))
    lo, hi = np.clip(lo, -1, 1), np.clip(hi, -1, 1)
    j, dec = decide(lo, hi, delta)
    none = np.zeros_like(lo, dtype=bool)
    return GateTrace(times, lo, hi, none, none, dec, float(times[j]) if j >= 0 else None)


def proxy_only(d_proxy: np.ndarray, t_proxy: np.ndarray, times: np.ndarray, delta: float = 0.02, alpha: float = 0.05,
               t_star: float = 1000.0) -> GateTrace:
    """Deciding on a proxy: the gate on the change in a proxy rate in [0, 1], read as if it were the outcome's."""
    n = np.searchsorted(t_proxy, times, side="right")
    lo, hi = bounds_at(*betting_cs_range(d_proxy, alpha, -1, 1, t_star), n, (-1.0, 1.0))
    j, dec = decide(lo, hi, delta)
    none = np.zeros_like(lo, dtype=bool)
    return GateTrace(times, lo, hi, none, none, dec, float(times[j]) if j >= 0 else None)
