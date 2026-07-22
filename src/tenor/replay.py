"""Fit TENOR on a history of past changes and replay a release, as the report's experiments do.

``fit_history`` simulates the matured runs of past changes, fits the surrogate index, its two sub-indices and the
maturation model, and validates the index on the changes (cross-fitted, so a change's index effect is measured with
an index that did not see its runs). ``replay`` simulates one release and runs TENOR, waiting for the outcome and
deciding on a proxy over the same runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .censoring import MaturationModel, rectifier_terms
from .gate import GateTrace, ReleaseGate, Streams, outcome_only, proxy_only
from .index import CONTROLLED, SCREEN_INDEPENDENT, SIGNALS, SurrogateIndex
from .simulate import Environment, Version, concat, outcome_rate, proxy_success, simulate_runs
from .validate import PastChange, Validation, validate

H_SIGNALS = 2.0   # days until all six signals are in
H_SCREEN = 1.0    # days until the screen's signals are in


@dataclass
class Fitted:
    index: SurrogateIndex
    controlled: SurrogateIndex
    independent: SurrogateIndex
    constant: SurrogateIndex
    maturation: MaturationModel
    validation: Validation
    changes: list[PastChange]


def _fit_indices(runs):
    return (SurrogateIndex.fit(runs, SIGNALS), SurrogateIndex.fit(runs, CONTROLLED),
            SurrogateIndex.fit(runs, SCREEN_INDEPENDENT))


def fit_history(env: Environment, versions: list[Version], difficulty: list[float], changes: range,
                runs_per_version: int, seed: int) -> Fitted:
    rng = np.random.default_rng(seed)
    pairs = [(simulate_runs(rng, env, versions[c - 1], runs_per_version, difficulty[c]),
              simulate_runs(rng, env, versions[c], runs_per_version, difficulty[c])) for c in changes]
    allruns = concat([r for p in pairs for r in p])
    index, controlled, independent = _fit_indices(allruns)
    constant = SurrogateIndex.fit(allruns, ())
    maturation = MaturationModel.fit(allruns, env.landmark)
    half = runs_per_version // 2
    folds = [[({k: a[k][:half] for k in a}, {k: b[k][:half] for k in b}) for a, b in pairs],
             [({k: a[k][half:] for k in a}, {k: b[k][half:] for k in b}) for a, b in pairs]]
    fits = [_fit_indices(concat([r for p in folds[1 - f] for r in p])) for f in (0, 1)]
    past = []
    for i, c in enumerate(changes):
        effects = np.zeros(3)
        for f in (0, 1):
            a, b = folds[f][i]
            effects += [float(m(b).mean() - m(a).mean()) for m in fits[f]]
        effects /= 2
        a, b = pairs[i]
        past.append(PastChange(f"change {c}", float(b["y"].mean() - a["y"].mean()), *effects,
                               proxy=float(proxy_success(b).mean() - proxy_success(a).mean())))
    return Fitted(index, controlled, independent, constant, maturation, validate(past), past)


@dataclass
class ReleaseResult:
    truth: float
    tenor: GateTrace
    wait: GateTrace
    proxy: GateTrace


def replay(env: Environment, versions: list[Version], difficulty: list[float], c: int, fitted: Fitted, seed: int,
           delta: float = 0.02, step_hours: float = 1.0) -> ReleaseResult:
    """Replay release c (candidate c against incumbent c - 1) and decide with each method."""
    rng = np.random.default_rng(seed)
    n_a, n_b = rng.poisson(env.rate * env.horizon), rng.poisson(env.rate * env.horizon)
    a_t, b_t = np.sort(rng.uniform(0, env.horizon, n_a)), np.sort(rng.uniform(0, env.horizon, n_b))
    A = simulate_runs(rng, env, versions[c - 1], n_a, difficulty[c])
    B = simulate_runs(rng, env, versions[c], n_b, difficulty[c])
    k = min(n_a, n_b)
    paired = np.maximum(a_t[:k], b_t[:k])
    f, v = fitted, fitted.validation
    sA, sB = f.index(A), f.index(B)
    piA, piB = f.maturation(A), f.maturation(B)
    def screen_prediction(runs):
        return v.gamma_q * f.controlled(runs) - v.gamma_w * f.independent(runs)

    d_div = screen_prediction(B)[:k] - screen_prediction(A)[:k]
    streams = Streams(
        d_index=sB[:k] - sA[:k], t_index=paired + H_SIGNALS,
        d_div=d_div, t_div=paired + H_SCREEN,
        e_rect=rectifier_terms(B, sB, piB, env.landmark)[:k] - rectifier_terms(A, sA, piA, env.landmark)[:k],
        t_rect=paired + env.landmark,
    )
    times = np.arange(1, int(round(env.horizon * 24 / step_hours)) + 1) * step_hours / 24
    gate = ReleaseGate(v, delta=delta, t_star_index=env.rate * 0.75, t_star_rect=env.rate * 3.0)
    tenor = gate.run(streams, times)
    e_const = (rectifier_terms(B, f.constant(B), piB, env.landmark)[:k]
               - rectifier_terms(A, f.constant(A), piA, env.landmark)[:k])
    wait = outcome_only(e_const, paired + env.landmark, times, delta=delta, t_star=env.rate * 3.0)
    proxy = proxy_only(proxy_success(B)[:k] - proxy_success(A)[:k], paired, times, delta=delta, t_star=env.rate * 0.75)
    truth = outcome_rate(env, versions[c], difficulty[c]) - outcome_rate(env, versions[c - 1], difficulty[c])
    return ReleaseResult(truth, tenor, wait, proxy)
