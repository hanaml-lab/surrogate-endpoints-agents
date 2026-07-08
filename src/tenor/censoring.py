"""Censoring: the probability that a run's outcome has matured by a landmark age, and the rectifier terms.

A run's outcome becomes known after a delay. Delays may depend on the early signals (a review queue that moves
faster for changes that draw fewer comments) but are assumed independent of the outcome given them. The maturation
probability at a landmark age ``ell`` is then estimated from past runs, whose delays are all observed, and each run
that reaches age ``ell`` contributes the inverse-probability-of-censoring rectifier term

    R = 1[D <= ell] / pi_ell(X) * (Y - S(X)),

whose mean is E[Y - S(X)] for any index S (Robins and Rotnitzky, 1992; Bang and Tsiatis, 2000).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

Runs = Mapping[str, np.ndarray]

MAX_COMMENTS = 6


def cells(runs: Runs) -> np.ndarray:
    """The cell of a run for the maturation model: comments (capped) by rollback."""
    comments = np.minimum(np.asarray(runs["comments"]), MAX_COMMENTS).astype(int)
    return comments * 2 + np.asarray(runs["rollback"]).astype(int)


@dataclass
class MaturationModel:
    """pi_ell(x) = P(delay <= ell | x), by cell, shrunk towards the overall rate in small cells."""

    landmark: float
    overall: float
    table: np.ndarray = field(repr=False)

    @classmethod
    def fit(cls, runs: Runs, landmark: float, min_cell: int = 50, prior: float = 5.0, floor: float = 0.02) -> MaturationModel:
        matured = (np.asarray(runs["delay"]) <= landmark).astype(float)
        overall = float(matured.mean())
        c = cells(runs)
        table = np.full(2 * (MAX_COMMENTS + 1), overall)
        for k in range(table.size):
            sel = c == k
            if sel.sum() >= min_cell:
                table[k] = (matured[sel].sum() + overall * prior) / (sel.sum() + prior)
        return cls(landmark, overall, np.clip(table, floor, 1.0))

    def __call__(self, runs: Runs) -> np.ndarray:
        return self.table[cells(runs)]


def rectifier_terms(runs: Runs, s: np.ndarray, pi: np.ndarray, landmark: float, weighted: bool = True) -> np.ndarray:
    """R_i = M_i / pi_i * (Y_i - S_i) for runs at the landmark. With ``weighted=False`` every run gets the same
    weight 1 / mean(pi), which ignores how delays depend on the signals (the ablation in the report)."""
    matured = (np.asarray(runs["delay"]) <= landmark).astype(float)
    w = 1.0 / pi if weighted else np.full(len(pi), 1.0 / float(np.mean(pi)))
    return matured * w * (np.asarray(runs["y"], float) - s)
