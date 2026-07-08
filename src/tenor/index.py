"""The surrogate index: P(outcome | early signals), fitted on runs whose outcomes have matured.

Signals are named; each maps to one or more features (logits for scores in (0, 1), log(1 + count) for counts, an
interaction between visible tests and failed perturbed re-checks). The index is a logistic regression fitted by
Newton-Raphson with a small ridge, so every fit is deterministic.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

Runs = Mapping[str, np.ndarray]

SIGNALS = ("judge", "tests", "ccv", "comments", "rollback", "perturb")
CONTROLLED = ("judge", "tests", "ccv")             # from the agent's own artifact and the checks it can see
INDEPENDENT = ("comments", "rollback", "perturb")  # from people, production or checks it cannot see
SCREEN_INDEPENDENT = ("comments", "perturb")       # the independent signals that are in within 24 hours


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


FEATURES: dict[str, Callable[[Runs], list[np.ndarray]]] = {
    "judge": lambda r: [logit(r["judge"])],
    "tests": lambda r: [np.asarray(r["tests"], float)],
    "ccv": lambda r: [logit(r["ccv"])],
    "comments": lambda r: [np.log1p(np.asarray(r["comments"], float))],
    "rollback": lambda r: [np.asarray(r["rollback"], float)],
    "perturb": lambda r: [np.asarray(r["perturb"], float),
                          np.asarray(r["tests"], float) * (1.0 - np.asarray(r["perturb"], float))],
}


def design(runs: Runs, signals: Sequence[str]) -> np.ndarray:
    """The design matrix of an index on the given signals, with an intercept."""
    n = len(runs["y"]) if "y" in runs else len(next(iter(runs.values())))
    cols = [np.ones(n)]
    for s in signals:
        if s not in FEATURES:
            raise KeyError(f"unknown signal {s!r}; known: {sorted(FEATURES)}")
        cols += FEATURES[s](runs)
    return np.stack(cols, 1)


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 60) -> np.ndarray:
    """Ridge-stabilised logistic regression by Newton-Raphson."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        grad = X.T @ (y - p) - ridge * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


@dataclass
class SurrogateIndex:
    """S(x) = P(Y = 1 | signals x). An index on no signals is the historical outcome rate."""

    signals: tuple[str, ...]
    weights: np.ndarray = field(repr=False)

    @classmethod
    def fit(cls, runs: Runs, signals: Sequence[str] = SIGNALS) -> SurrogateIndex:
        y = np.asarray(runs["y"], float)
        if not signals:
            return cls((), np.array([float(logit(np.array([y.mean()]))[0])]))
        return cls(tuple(signals), fit_logistic(design(runs, signals), y))

    def __call__(self, runs: Runs) -> np.ndarray:
        if not self.signals:
            n = len(runs["y"]) if "y" in runs else len(next(iter(runs.values())))
            return np.full(n, float(sigmoid(self.weights[0])))
        return sigmoid(design(runs, self.signals) @ self.weights)


def outcome_r2(y: np.ndarray, s: np.ndarray) -> float:
    """1 - Brier score / outcome variance: the share of the outcome's variance the index explains."""
    y = np.asarray(y, float)
    return float(1 - np.mean((y - s) ** 2) / np.var(y))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Binned expected calibration error."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)
