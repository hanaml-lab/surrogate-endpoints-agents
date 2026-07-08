"""A delayed-outcome simulator for agent releases.

Each run has a latent quality q (relative to the difficulty of the tasks it faces). Its outcome is Bernoulli in q
plus a version-specific direct effect the signals cannot see; its six early signals are noisy functions of q. A
version may be

* honest: a shift of q;
* presentation: a small loss of q and a judge that is flattered;
* hacked: a small loss of q, plus artifacts that pass the agent-visible checks without doing the work (they raise
  tests passed, the judge's score and the completion verifier's estimate, draw more reviewer comments and
  rollbacks, fail perturbed re-checks and almost never succeed);
* latent: a modest visible shift of q and a larger hidden shift of the outcome, through a pathway none of the
  signals measures.

Both versions of a comparison run on the same concurrent tasks; an agent that improves is given harder work, so the
incumbent's outcome rate stays near the environment's base rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .index import logit, sigmoid

P_HACK_Y = 0.06
K_PERTURB = 8


@dataclass
class Environment:
    name: str
    base: float                      # outcome rate of the first version
    rate: float                      # runs per day per version
    landmark: float                  # age (days) at which a run enters the rectifier
    horizon: float                   # days a release is monitored
    delay: tuple[str, dict]          # ("fixed", {days}), ("calendar", {lo, hi}) or ("grading", {med, sd, per_rerun, per_comment})
    signal: dict = field(default_factory=lambda: dict(
        aJ=0.42, cJ=-0.4, sJ=0.80, aT=0.50, cT=-1.6, aC=0.55, cC=0.0, sC=0.70, m0=0.10, aM=0.27,
        r0=-2.7, aR=0.38, aP=0.47, cP=-0.9))
    hack: dict = field(default_factory=lambda: dict(bJ=1.4, bT=3.2, bC=1.1, bM=0.55, bR=0.9))
    spread: float = 2.6              # standard deviation of latent quality across runs


@dataclass
class Version:
    mu: float          # latent mean
    eta: float = 0.0   # direct effect on the outcome logit the signals do not see
    g: float = 0.01    # hacking intensity
    bP: float = 1.5    # how strongly a hacked artifact fails perturbed checks
    wW: float = 1.0    # how strongly people and production react to a hacked artifact
    jb: float = 0.0    # judge bias (a version can please the judge more or less without being better)
    tb: float = 0.0    # shift of the visible tests' logit (more or fewer tests written)
    mb: float = 0.0    # shift of the reviewers' comment rate
    kind: str = "honest"


def make_history(env: Environment, types: str, seed: int, shift_mean: float = 0.30, shift_sd: float = 0.80,
                 latent: tuple[float, float] = (0.45, -0.85)) -> tuple[list[Version], list[float]]:
    """Versions 0..len(types) and the task difficulty of each comparison c = 1..len(types).

    ``types`` is a string over H (honest), P (presentation), X (hacked) and L (latent). The change after a hacked
    version removes the hack and is labelled "unhack". Each change draws from its own random stream.
    """
    m0 = float(logit(np.array([env.base]))[0])
    versions, difficulty = [Version(mu=m0, kind="base")], [0.0]
    for i, t in enumerate(types, start=1):
        rng = np.random.default_rng([seed, i])
        z_shift, z_eta, z_p, u_g, u_bp, u_w, z_d = (rng.normal(), rng.normal(), rng.normal(), rng.random(),
                                                     rng.random(), rng.random(), rng.normal())
        z_j, z_t, z_m = rng.normal(), rng.normal(), rng.normal()
        prev = versions[-1]
        difficulty.append(prev.mu - m0 + 0.25 * z_d)
        eta = prev.eta + 0.02 * z_eta
        # honest changes also move the signals a little on their own
        drift = dict(jb=prev.jb + 0.10 * z_j, tb=prev.tb + 0.10 * z_t, mb=prev.mb + 0.05 * z_m)
        if t in "HL":
            shift = shift_mean + shift_sd * z_shift
            if t == "L":
                shift = latent[0] + 0.04 * z_p
                eta += latent[1]
            v = Version(mu=prev.mu + shift, eta=eta, kind="latent" if t == "L" else "honest", **drift)
        elif t == "P":
            drift["jb"] += 1.0
            v = Version(mu=prev.mu - 0.18 + 0.05 * z_p, eta=eta, kind="presentation", **drift)
        elif t == "X":
            v = Version(mu=prev.mu - 0.12 + 0.08 * z_p, eta=eta, g=0.16 + 0.39 * u_g, bP=0.5 + 1.7 * u_bp,
                        wW=0.3 + 0.7 * u_w, kind="hacked", **drift)
        else:
            raise ValueError(f"unknown change type {t!r}")
        if prev.kind == "hacked" and v.kind == "honest":
            v.kind = "unhack"
        versions.append(v)
    return versions, difficulty


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def outcome_rate(env: Environment, v: Version, difficulty: float) -> float:
    """Population outcome rate of version v on tasks of the given difficulty (Gauss-Hermite quadrature)."""
    q = v.mu - difficulty + env.spread * _GH_X
    h = v.g * (1 - sigmoid(q))
    return float(np.sum(_GH_W * ((1 - h) * sigmoid(q + v.eta) + h * P_HACK_Y)))


def simulate_runs(rng: np.random.Generator, env: Environment, v: Version, n: int, difficulty: float) -> dict[str, np.ndarray]:
    """n runs of version v: latent quality, hack flag, outcome, the six signals and the outcome's delay (days)."""
    s, hk = env.signal, env.hack
    q = v.mu - difficulty + env.spread * rng.standard_normal(n)
    hack = rng.random(n) < v.g * (1 - sigmoid(q))
    y = np.where(hack, rng.random(n) < P_HACK_Y, rng.random(n) < sigmoid(q + v.eta)).astype(np.int8)
    hf = hack.astype(float)
    judge = sigmoid(s["aJ"] * (q - s["cJ"]) + hk["bJ"] * hf + v.jb + s["sJ"] * rng.standard_normal(n))
    tests = ((s["aT"] * (q - s["cT"]) + hk["bT"] * hf + v.tb + rng.logistic(size=n)) > 0).astype(np.int8)
    ccv = sigmoid(s["aC"] * (q - s["cC"]) + hk["bC"] * hf + s["sC"] * rng.standard_normal(n))
    comments = rng.poisson(np.exp(s["m0"] - s["aM"] * q + v.wW * hk["bM"] * hf + v.mb)).astype(np.int16)
    rollback = (rng.random(n) < sigmoid(s["r0"] - s["aR"] * q + v.wW * hk["bR"] * hf)).astype(np.int8)
    perturb = rng.binomial(K_PERTURB, sigmoid(s["aP"] * (q - s["cP"]) - v.bP * hf)) / K_PERTURB
    kind, p = env.delay
    if kind == "fixed":
        delay = np.full(n, float(p["days"]))
    elif kind == "calendar":
        delay = rng.uniform(p["lo"], p["hi"], n)
    elif kind == "grading":
        delay = np.exp(math.log(p["med"]) + p["sd"] * rng.standard_normal(n) + p["per_rerun"] * rollback
                       + p["per_comment"] * np.minimum(comments, 6))
    else:
        raise ValueError(f"unknown delay model {kind!r}")
    return dict(q=q, hack=hack, y=y, judge=judge, tests=tests, ccv=ccv, comments=comments, rollback=rollback,
                perturb=perturb, delay=delay)


def concat(runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([r[k] for r in runs]) for k in runs[0]}


def proxy_success(runs: dict[str, np.ndarray]) -> np.ndarray:
    """The proxy most release decisions use: tests pass and the judge approves at submission."""
    return (np.asarray(runs["tests"]) * (np.asarray(runs["judge"]) >= 0.5)).astype(float)
