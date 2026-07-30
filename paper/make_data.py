"""Seeded generator for every number in the TENOR report (illustrative, simulated results).

Writes results.json and tables/*.tex; figures.py reads results.json only, so text, tables and figures agree
by construction.

The simulation has four delayed-outcome environments. Each has a sequence of agent versions; a version is a
latent quality shift (honest change), a shift plus a bias of the judge (a "presentation" change), or a shift plus
reward hacking (a "hacked" change, whose artifacts pass the agent-visible checks without doing the work). Every
run of a version yields six early signals and an outcome that matures after an environment-specific delay. The
first ten changes of each environment form the history on which the surrogate index, the censoring model, the
residual-effect margin and the divergence screen are fitted; the next fifteen are replayed as releases, five
seeds each, and every method monitors the same simulated runs hour by hour.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
MASTER = 20260804
SEEDS = [0, 1, 2, 3, 4]
H_SIG = 2.0          # days until all six signals are in (comments at 24 h, rollbacks at 48 h)
H_SCREEN = 1.0       # days until the signals the divergence screen compares are in
DELTA = 0.02         # non-inferiority margin of the release gate (2 points)
STEP = 1.0 / 24.0    # the gate is checked every hour
N_HIST = 60000       # matured runs per historical version (one to three weeks at full traffic)
P_HACK_Y = 0.06      # outcome rate of a hacked artifact
K_PERT = 8           # perturbed re-checks per run

SIGNALS = ["judge", "tests", "ccv", "comments", "rollback", "perturb"]
CONTROLLED = ["judge", "tests", "ccv"]          # produced from the agent's own artifact and checks it can see
INDEPENDENT = ["comments", "rollback", "perturb"]  # produced by people, production or checks it cannot see
SCREEN_W = ["comments", "perturb"]                 # the independent signals that are in within 24 hours

# ------------------------------------------------------------------------------------------------------------
# Environments. Coefficients are on the latent logit scale; `base` is the outcome rate of version 0.
# delay: (kind, params). landmark: age (days) at which a run enters the rectifier. rate: runs/day/version.
# ------------------------------------------------------------------------------------------------------------
ENVS = {
    "pr": dict(
        name="Pull requests", outcome="merged and not reverted within 14 days of opening", base=0.54, rate=5000, horizon=42.0,
        delay=("fixed", dict(days=14.0)), landmark=14.0,
        sig=dict(aJ=0.42, cJ=-0.4, sJ=0.80, aT=0.50, cT=-1.6, aC=0.55, cC=0.0, sC=0.70, m0=0.10, aM=0.27,
                 r0=-2.7, aR=0.38, aP=0.47, cP=-0.9),
        hack=dict(bJ=1.4, bT=3.2, bC=1.1, bM=0.55, bR=0.9),
        su=2.6,
    ),
    "recon": dict(
        name="Reconciliation", outcome="passes the month-end audit", base=0.70, rate=4500, horizon=42.0,
        delay=("calendar", dict(lo=2.0, hi=32.0)), landmark=9.0,
        sig=dict(aJ=0.38, cJ=-1.2, sJ=0.85, aT=0.55, cT=-2.8, aC=0.55, cC=-0.8, sC=0.75, m0=-0.45, aM=0.28,
                 r0=-3.0, aR=0.40, aP=0.50, cP=-1.8),
        hack=dict(bJ=1.2, bT=3.0, bC=1.0, bM=0.5, bR=1.0),
        su=2.5,
    ),
    "repl": dict(
        name="Research replication", outcome="rubric grade of 60% or more", base=0.36, rate=1500, horizon=42.0,
        delay=("grading", dict(med=11.0, sd=0.28, per_rerun=0.22, per_comment=0.05)), landmark=11.0,
        sig=dict(aJ=0.44, cJ=0.6, sJ=0.75, aT=0.48, cT=-0.6, aC=0.56, cC=1.0, sC=0.70, m0=0.30, aM=0.25,
                 r0=-2.3, aR=0.34, aP=0.48, cP=0.0),
        hack=dict(bJ=1.5, bT=3.0, bC=1.2, bM=0.5, bR=0.8),
        su=2.7,
    ),
    "sim": dict(
        name="Business simulation", outcome="day-30 net worth 20% above start", base=0.44, rate=2400, horizon=48.0,
        delay=("fixed", dict(days=30.0)), landmark=30.0,
        sig=dict(aJ=0.40, cJ=0.4, sJ=0.80, aT=0.50, cT=-0.8, aC=0.52, cC=0.6, sC=0.75, m0=-0.05, aM=0.24,
                 r0=-2.5, aR=0.38, aP=0.46, cP=0.0),
        hack=dict(bJ=1.3, bT=2.8, bC=1.0, bM=0.6, bR=0.9),
        su=2.6,
    ),
}
ENV_KEYS = list(ENVS)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------------------------
# Versions. Each change is honest (H), presentation (P: the judge is flattered) or hacked (X). A hacked version
# that ships becomes the incumbent, so the honest change after it removes the hack (an "un-hack").
# ------------------------------------------------------------------------------------------------------------
TYPES = {
    #        historical (10)   evaluation (15)
    "pr":    "HHXHHHPHHH" + "HHXHHHHHHXHHLHH",
    "recon": "HHHHXHHHHH" + "HHHHHHHHLHXHHHH",
    "repl":  "HHHPHHHHHH" + "HHHHXHHPHHXHHHH",
    "sim":   "HXHHHHHHXH" + "HHHHHHXHHHHHHLH",
}
# a latent change acts on the outcome through a pathway none of the six signals measures (a dependency bump whose
# failures surface after a week; a prompt change that skips edge cases production finds later). None is in the
# history, so the validation step cannot see one coming.
# (visible shift of latent quality, hidden shift of the outcome logit): the signals show a modest change, the
# outcome moves the other way
LATENT = {"pr": (0.42, -0.80), "recon": (0.05, -0.25), "sim": (-0.55, 0.95)}


def make_versions(env_key):
    """Version parameters: mu (latent mean), eta (direct effect on the outcome the signals do not see),
    g (hacking intensity), bP (how strongly a hacked artifact fails perturbed checks), wW (how strongly people and
    production react to a hacked artifact) and jb (judge bias); and d, the difficulty of the tasks each comparison
    runs on. Both versions of a comparison run on the same concurrent tasks, and an agent that improves is given
    harder work, so the incumbent's outcome rate stays near the environment's base rate. Each change draws from its
    own stream, so the history does not depend on the order in which types were assigned."""
    types = TYPES[env_key]
    m0 = logit(ENVS[env_key]["base"])
    mu, d = [m0], [0.0]
    eta, g, bP, wW, jb, kind = [0.0], [0.01], [1.5], [1.0], [0.0], ["base"]
    tb, mb = [0.0], [0.0]
    for i, t in enumerate(types, start=1):
        rng = np.random.default_rng([MASTER, ENV_KEYS.index(env_key) + 1, i])
        z_shift, z_eta, z_p, u_g, u_bp, u_w, z_d = (rng.normal(), rng.normal(), rng.normal(), rng.random(),
                                                     rng.random(), rng.random(), rng.normal())
        z_j, z_t, z_m = rng.normal(), rng.normal(), rng.normal()
        d.append(mu[-1] - m0 + 0.25 * z_d)
        e = eta[-1] + 0.02 * z_eta
        # honest changes also move the signals a little on their own: a new model pleases the judge a little more or
        # less, writes more or fewer tests, draws a different share of reviewer comments
        jdrift, tdrift, mdrift = jb[-1] + 0.10 * z_j, tb[-1] + 0.10 * z_t, mb[-1] + 0.05 * z_m
        if t in "HL":
            shift = 0.30 + 0.80 * z_shift
            if t == "L":
                vis, hid = LATENT[env_key]
                shift = vis + 0.04 * z_p
                e += hid
            mu.append(mu[-1] + shift); g.append(0.01); bP.append(1.5); wW.append(1.0); jb.append(jdrift)
        elif t == "P":
            mu.append(mu[-1] - 0.18 + 0.05 * z_p); g.append(0.01); bP.append(1.5); wW.append(1.0); jb.append(jdrift + 1.0)
        elif t == "x":  # a mild hack: few hacked artifacts, which fail few perturbed checks and draw little reaction
            mu.append(mu[-1] - 0.15 + 0.05 * z_p); g.append(0.12); bP.append(0.6); wW.append(0.35); jb.append(jdrift)
        else:  # hacked
            mu.append(mu[-1] - 0.12 + 0.08 * z_p); g.append(0.16 + 0.39 * u_g)
            bP.append(0.5 + 1.7 * u_bp); wW.append(0.3 + 0.7 * u_w); jb.append(jdrift)
        tb.append(tdrift); mb.append(mdrift)
        eta.append(e); kind.append({"H": "honest", "P": "presentation", "X": "hacked", "x": "hacked", "L": "latent"}[t])
    # the change after a hacked version removes the hack
    for i in range(1, len(kind)):
        if kind[i] == "honest" and kind[i - 1] == "hacked":
            kind[i] = "unhack"
    return dict(mu=np.array(mu), d=np.array(d), eta=np.array(eta), g=np.array(g), bP=np.array(bP), wW=np.array(wW),
                jb=np.array(jb), tb=np.array(tb), mb=np.array(mb), kind=kind)


VERS = {k: make_versions(k) for k in ENV_KEYS}

# Gauss-Hermite nodes for exact population quantities
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = GH_W / GH_W.sum()


def theta(env_key, v, c):
    """Population outcome rate of version v on the tasks of comparison c."""
    V, su = VERS[env_key], ENVS[env_key]["su"]
    q = V["mu"][v] - V["d"][c] + su * GH_X
    h = V["g"][v] * (1 - sigmoid(q))
    return float(np.sum(GH_W * ((1 - h) * sigmoid(q + V["eta"][v]) + h * P_HACK_Y)))


def effect(env_key, c):
    """The true outcome effect of change c: candidate c against incumbent c - 1 on the same tasks."""
    return theta(env_key, c, c) - theta(env_key, c - 1, c)


def simulate_runs(rng, env_key, v, n, c):
    """n runs of version v on the tasks of comparison c: latent quality, hack flag, outcome, six signals and the
    outcome's delay."""
    E, V = ENVS[env_key], VERS[env_key]
    s, hk = E["sig"], E["hack"]
    q = V["mu"][v] - V["d"][c] + E["su"] * rng.standard_normal(n)
    hack = rng.random(n) < V["g"][v] * (1 - sigmoid(q))
    y = np.where(hack, rng.random(n) < P_HACK_Y, rng.random(n) < sigmoid(q + V["eta"][v])).astype(np.int8)
    hf = hack.astype(float)
    judge = sigmoid(s["aJ"] * (q - s["cJ"]) + hk["bJ"] * hf + V["jb"][v] + s["sJ"] * rng.standard_normal(n))
    tests = ((s["aT"] * (q - s["cT"]) + hk["bT"] * hf + V["tb"][v] + rng.logistic(size=n)) > 0).astype(np.int8)
    ccv = sigmoid(s["aC"] * (q - s["cC"]) + hk["bC"] * hf + s["sC"] * rng.standard_normal(n))
    comments = rng.poisson(np.exp(s["m0"] - s["aM"] * q + V["wW"][v] * hk["bM"] * hf + V["mb"][v])).astype(np.int16)
    rollback = (rng.random(n) < sigmoid(s["r0"] - s["aR"] * q + V["wW"][v] * hk["bR"] * hf)).astype(np.int8)
    perturb = rng.binomial(K_PERT, sigmoid(s["aP"] * (q - s["cP"]) - V["bP"][v] * hf)) / K_PERT
    kind, p = E["delay"]
    if kind == "review":
        lat = np.exp(math.log(p["med"]) + p["sd"] * rng.standard_normal(n) + p["per_comment"] * np.log1p(comments))
        delay = p["floor"] + lat
    elif kind == "calendar":
        delay = rng.uniform(p["lo"], p["hi"], n)
    elif kind == "grading":
        delay = np.exp(math.log(p["med"]) + p["sd"] * rng.standard_normal(n) + p["per_rerun"] * rollback
                       + p["per_comment"] * np.minimum(comments, 6))
    else:
        delay = np.full(n, p["days"])
    return dict(q=q, hack=hack, y=y, judge=judge, tests=tests, ccv=ccv, comments=comments, rollback=rollback,
                perturb=perturb, delay=delay)


# ------------------------------------------------------------------------------------------------------------
# Features and logistic fits (Newton-Raphson with a small ridge, so every fit is deterministic)
# ------------------------------------------------------------------------------------------------------------
FEATS = {
    "judge": lambda r: [logit(r["judge"])],
    "tests": lambda r: [r["tests"].astype(float)],
    "ccv": lambda r: [logit(r["ccv"])],
    "comments": lambda r: [np.log1p(r["comments"].astype(float))],
    "rollback": lambda r: [r["rollback"].astype(float)],
    "perturb": lambda r: [r["perturb"], r["tests"] * (1.0 - r["perturb"])],
}


def design(runs, signals):
    cols = [np.ones(len(runs["y"]))]
    for s in signals:
        cols += FEATS[s](runs)
    return np.stack(cols, 1)


def fit_logistic(X, y, ridge=1e-3, iters=60):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (y - p) - ridge * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


class Index:
    """A surrogate index S(x) = P(Y = 1 | signals), fitted on matured historical runs."""

    def __init__(self, signals, w):
        self.signals, self.w = list(signals), w

    def __call__(self, runs):
        if not self.signals:
            return np.full(len(runs["y"]), sigmoid(self.w[0]))
        return sigmoid(design(runs, self.signals) @ self.w)


def fit_index(runs, signals):
    if not signals:
        return Index([], np.array([logit(np.array([runs["y"].mean()]))[0]]))
    return Index(signals, fit_logistic(design(runs, signals), runs["y"].astype(float)))


def r2(y, s):
    y = y.astype(float)
    return float(1 - np.mean((y - s) ** 2) / np.var(y))


def ece(p, y, bins=10):
    e, out = 0.0, []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
        out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return float(e), out


def concat(runs_list):
    return {k: np.concatenate([r[k] for r in runs_list]) for k in runs_list[0]}


# ------------------------------------------------------------------------------------------------------------
# Confidence sequences
# ------------------------------------------------------------------------------------------------------------
def betting_cs(x, alpha, t_star, c=0.5):
    """Predictable plug-in empirical-Bernstein betting CS for the mean of [0, 1] observations (Waudby-Smith &
    Ramdas, 2024), with bets tuned to be tightest near t_star observations, and running intersection. Any
    predictable bet in (0, 1) gives a valid CS; the tuning only moves where it is tightest."""
    n = len(x)
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


def betting_cs_range(d, alpha, lo_b, hi_b, t_star):
    """Betting CS for observations in [lo_b, hi_b], mapped to [0, 1] and back."""
    w = hi_b - lo_b
    lo, hi = betting_cs((d - lo_b) / w, alpha, t_star)
    return lo_b + w * lo, lo_b + w * hi


N_BURN = 200        # an asymptotic CS is used from this many observations on


def asymp_cs(z, alpha, t_star):
    """Gaussian-mixture asymptotic CS (Waudby-Smith et al., 2024), tightest near t_star observations, used from
    N_BURN observations on (before that the bounds are uninformative), with running intersection."""
    n = len(z)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    t = np.arange(1, n + 1, dtype=float)
    mean = np.cumsum(z) / t
    ss = np.cumsum(z * z) - t * mean ** 2
    var = np.where(t > 1, ss / np.maximum(t - 1, 1), np.inf)
    rho2 = (-2 * math.log(alpha) + math.log(-2 * math.log(alpha) + 1)) / max(t_star, 1.0)
    half = np.sqrt(np.maximum(var, 1e-12)) * np.sqrt(2 * (t * rho2 + 1) / (t * t * rho2) * np.log(np.sqrt(t * rho2 + 1) / alpha))
    half[: N_BURN - 1] = np.inf
    lo = np.maximum.accumulate(mean - half)
    hi = np.minimum.accumulate(mean + half)
    return lo, hi


def at(lo, hi, n, default=(-2.0, 2.0)):
    """Bounds after n observations, for an array of counts n (0 gives the default)."""
    n = np.asarray(n)
    out_lo = np.full(n.shape, default[0], dtype=float)
    out_hi = np.full(n.shape, default[1], dtype=float)
    m = n > 0
    out_lo[m] = lo[n[m] - 1]
    out_hi[m] = hi[n[m] - 1]
    return out_lo, out_hi


# ------------------------------------------------------------------------------------------------------------
# History: indices, censoring model, past-change validation, the residual margin and the divergence screen
# ------------------------------------------------------------------------------------------------------------
INDEX_SETS = {
    "full": SIGNALS, "Q": CONTROLLED, "W": SCREEN_W, "Windep": INDEPENDENT, "judge": ["judge"], "tests": ["tests"], "ccv": ["ccv"],
    "const": [],
}
for _s in SIGNALS:
    INDEX_SETS["drop_" + _s] = [x for x in SIGNALS if x != _s]
    if _s in CONTROLLED:
        INDEX_SETS["Q_drop_" + _s] = [x for x in CONTROLLED if x != _s]
    if _s in SCREEN_W:
        INDEX_SETS["W_drop_" + _s] = [x for x in SCREEN_W if x != _s]
CROSSFIT = [n for n in INDEX_SETS if n not in ("const", "Windep")]


def variant_sets(s):
    """The index and the two sub-indices of the variant that drops signal s."""
    return ("drop_" + s, "Q_drop_" + s if s in CONTROLLED else "Q", "W_drop_" + s if s in SCREEN_W else "W")


def pi_cells(runs):
    return np.minimum(runs["comments"], 6).astype(int) * 2 + runs["rollback"].astype(int)


class Censoring:
    """P(outcome matured by the landmark | signals), by cell of (comments capped at 6, rollback), from history."""

    def __init__(self, hist, landmark):
        cells = pi_cells(hist)
        m = (hist["delay"] <= landmark).astype(float)
        self.overall = float(m.mean())
        self.table = np.full(14, self.overall)
        for c in range(14):
            sel = cells == c
            if sel.sum() >= 50:
                self.table[c] = (m[sel].sum() + self.overall * 5) / (sel.sum() + 5)
        self.table = np.clip(self.table, 0.02, 1.0)

    def __call__(self, runs):
        return self.table[pi_cells(runs)]


def history(env_key):
    """Simulate the ten historical changes, fit everything the gate needs, and return per-change summaries."""
    E = ENVS[env_key]
    rng = np.random.default_rng(MASTER + 101 * (ENV_KEYS.index(env_key) + 1))
    pairs = [(simulate_runs(rng, env_key, c - 1, N_HIST, c), simulate_runs(rng, env_key, c, N_HIST, c)) for c in range(1, 11)]
    hist = concat([r for p in pairs for r in p])
    idx = {name: fit_index(hist, sig) for name, sig in INDEX_SETS.items()}
    cens = Censoring(hist, E["landmark"])
    # cross-fitted index values for the validation of past changes: fit on one half of every run set, evaluate on
    # the other half, and swap
    h2 = N_HIST // 2
    halves = [[({k: a[k][:h2] for k in a}, {k: b[k][:h2] for k in b}) for a, b in pairs],
              [({k: a[k][h2:] for k in a}, {k: b[k][h2:] for k in b}) for a, b in pairs]]
    fits = []
    for f in (0, 1):
        train = concat([r for p in halves[1 - f] for r in p])
        fits.append({name: fit_index(train, INDEX_SETS[name]) for name in CROSSFIT})
    def cf_effect(name, ci):
        tot = 0.0
        for f in (0, 1):
            a, b = halves[f][ci]
            tot += float(fits[f][name](b).mean() - fits[f][name](a).mean())
        return tot / 2
    changes = []
    px = lambda r: float(np.mean(r["tests"] * (r["judge"] >= 0.5)))
    for ci, c in enumerate(range(1, 11)):
        r0, r1 = pairs[ci]
        dY = float(r1["y"].mean() - r0["y"].mean())
        eff = {n: cf_effect(n, ci) for n in CROSSFIT}
        dS, dQ, dW, dJ = eff["full"], eff["Q"], eff["W"], eff["judge"]
        changes.append(dict(env=env_key, change=c, kind=VERS[env_key]["kind"][c], dY=dY, dY_true=effect(env_key, c),
                            dS=dS, dQ=dQ, dW=dW, eps=dY - dS, dJ=dJ, eff=eff, proxy=px(r1) - px(r0),
                            tests=float(r1["tests"].mean() - r0["tests"].mean()),
                            judge=float((r1["judge"] >= 0.5).mean() - (r0["judge"] >= 0.5).mean())))
    # calibration of the full index on held-out runs of the history
    holdout = concat([simulate_runs(rng, env_key, v, 1500, max(v, 1)) for v in range(11)])
    s_ho = idx["full"](holdout)
    e_val, rel = ece(s_ho, holdout["y"])
    r2s = {name: r2(holdout["y"], m(holdout)) for name, m in idx.items()}
    return dict(idx=idx, cens=cens, changes=changes, ece=e_val, rel=rel, r2=r2s, ybar=float(hist["y"].mean()),
                delay_med=float(np.median(hist["delay"])), pi_bar=float(cens(hist).mean()))


def huber_slope(x, y, k=1.345, iters=50):
    """Slope of y on x through the origin under Huber's loss (IRLS), so that a few paradoxical changes do not set it."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    g = float(np.sum(x * y) / np.sum(x * x))
    for _ in range(iters):
        r = y - g * x
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        w = np.minimum(1.0, k * s / np.maximum(np.abs(r), 1e-12))
        g_new = float(np.sum(w * x * y) / np.sum(w * x * x))
        if abs(g_new - g) < 1e-10:
            break
        g = g_new
    return g


def calibrate_screen(changes, sets=("full", "Q", "W"), record=True):
    """Validation on past changes. gamma: the slope of the outcome effect on the index effect (1 / PTE);
    gamma_Q, gamma_W: the same for the controlled and the independent sub-indices; the divergence of a change is
    gamma_Q dQ - gamma_W dW; tau: three robust standard deviations of the divergence (an outlier rule); kappa:
    the largest leave-one-out residual |dY - gamma_(-c) dS| among the changes the screen passes, which bounds a new
    change's residual with probability n / (n + 1) under exchangeability (split conformal)."""
    s_name, q_name, w_name = sets
    dY = np.array([c["dY"] for c in changes])
    dS = np.array([c["eff"][s_name] for c in changes])
    dQ = np.array([c["eff"][q_name] for c in changes])
    dW = np.array([c["eff"][w_name] for c in changes])
    gamma, gQ, gW = huber_slope(dS, dY), huber_slope(dQ, dY), huber_slope(dW, dY)
    div = gQ * dQ - gW * dW
    tau = float(3 * 1.4826 * np.median(np.abs(div)))
    passed = np.abs(div) <= tau
    loo = np.array([huber_slope(np.delete(dS, i), np.delete(dY, i)) for i in range(len(dY))])
    resid = dY - loo * dS
    kappa = float(np.max(np.abs(resid[passed])))
    if record:
        for c, d, r, p in zip(changes, div, resid, passed):
            c.update(div=float(d), resid=float(r), passed=bool(p))
    return dict(gamma=gamma, gQ=gQ, gW=gW, tau=tau, kappa=kappa, n_pass=int(passed.sum()),
                div=div.tolist(), resid=resid.tolist(), passed=passed.tolist())


# ------------------------------------------------------------------------------------------------------------
# Release replays
# ------------------------------------------------------------------------------------------------------------
class Pairs:
    """Runs of the incumbent (A) and the candidate (B), paired in order of arrival."""

    def __init__(self, env_key, c, seed, rate=None, salt=0):
        E = ENVS[env_key]
        rng = np.random.default_rng(MASTER + 7919 * (ENV_KEYS.index(env_key) + 1) + 104729 * (seed + 1) + 37 * c + salt)
        hz, rate = E["horizon"], (E["rate"] if rate is None else rate)
        nA, nB = rng.poisson(rate * hz), rng.poisson(rate * hz)
        self.aA, self.aB = np.sort(rng.uniform(0, hz, nA)), np.sort(rng.uniform(0, hz, nB))
        self.A, self.B = simulate_runs(rng, env_key, c - 1, nA, c), simulate_runs(rng, env_key, c, nB, c)
        self.K = K = min(nA, nB)
        tp = np.maximum(self.aA[:K], self.aB[:K])
        self.t_px, self.t_sig, self.t_lm = tp, tp + H_SIG, tp + E["landmark"]
        self.t_scr = tp + H_SCREEN
        self.landmark = E["landmark"]

    def counts(self, ts):
        f = lambda arr: np.searchsorted(arr, ts, side="right")
        return f(self.t_px), f(self.t_sig), f(self.t_lm), f(self.t_scr)

    def diff(self, fA, fB):
        return fB[: self.K] - fA[: self.K]


def rect(runs, s, pi, landmark, weights=True):
    """Rectifier term (M / pi)(Y - S) of each run at the landmark; with weights=False, the unweighted 1 / mean(pi)."""
    m = (runs["delay"] <= landmark).astype(float)
    w = 1.0 / pi if weights else np.full(len(pi), 1.0 / pi.mean())
    return m * w * (runs["y"] - s)


def first_decision(lo, hi):
    """The release gate, with an indifference zone (-DELTA, 0): ship once the change is shown to cost less than
    DELTA (lo > -DELTA), block once it is shown to cost anything (hi < 0); shipping takes precedence. Returns the
    index of the first check at which it decides and the decision (+1 ship, -1 block), or (-1, 0)."""
    ship = lo > -DELTA
    block = hi < 0
    any_ = ship | block
    if not any_.any():
        return -1, 0
    j = int(np.argmax(any_))
    return j, (1 if ship[j] else -1)


def acceptable(dec, truth):
    """A decision is wrong when it ships a change that costs more than DELTA or blocks one that costs nothing."""
    return (truth >= -DELTA) if dec == 1 else (truth <= 0)


METHODS = ["wait", "proxy", "naive", "parc", "tenor"]
ABLATIONS = ["tenor_noscreen", "tenor_onesided", "tenor_noslope", "tenor_nomargin", "tenor_norect", "tenor_noweights"]


def alphas(n_pass, total=0.05):
    a_k = 1.0 / (n_pass + 1)
    rest = (total - a_k) / 3
    return dict(kappa=a_k, s=rest, o=rest, d=rest)


def tenor_bounds(S, e, div, cal, mode="two", gamma=None, kappa=None):
    """TENOR's bounds at each check. Design-based: the index CS (+) the residual CS. Transported, while the
    divergence screen licenses it: gamma times the index CS (+) [-kappa, kappa]. TENOR takes their intersection;
    if they are disjoint, transport is falsified and the design-based bounds are used from then on."""
    gamma = cal["gamma"] if gamma is None else gamma
    kappa = cal["kappa"] if kappa is None else kappa
    tau = cal["tau"]
    S_lo, S_hi = S
    e_lo, e_hi = e
    d_lo, d_hi = div
    if mode == "two":
        licensed = (d_hi < tau) & (d_lo > -tau)
        flagged = np.logical_or.accumulate((d_lo > tau) | (d_hi < -tau))
    elif mode == "one":
        licensed = d_hi < tau
        flagged = np.logical_or.accumulate(d_lo > tau)
    else:  # no screen
        licensed = np.ones_like(d_lo, dtype=bool)
        flagged = np.zeros_like(d_lo, dtype=bool)
    D_lo, D_hi = S_lo + e_lo, S_hi + e_hi
    T_lo, T_hi = gamma * S_lo - kappa, gamma * S_hi + kappa
    falsified = np.logical_or.accumulate(licensed & ((T_lo > D_hi) | (T_hi < D_lo)))
    use = licensed & ~falsified
    lo = np.where(use, np.maximum(D_lo, T_lo), D_lo)
    hi = np.where(use, np.minimum(D_hi, T_hi), D_hi)
    return lo, hi, use, flagged | falsified


def summarise(lo, hi, est, truth, ts, hz):
    lo, hi = np.clip(lo, -1, 1), np.clip(hi, -1, 1)
    j, dec = first_decision(lo, hi)
    forced = j < 0
    if forced:
        dec, t_dec = (1 if est > -DELTA / 2 else -1), hz
    else:
        t_dec = float(ts[j])
    covered_t = (lo <= truth) & (truth <= hi)
    days = np.arange(1, int(hz) + 1)
    jd = np.minimum(np.round(days / STEP).astype(int) - 1, len(ts) - 1)
    cum = np.logical_and.accumulate(covered_t)
    return dict(t=t_dec, dec=int(dec), forced=bool(forced), correct=bool(acceptable(dec, truth)),
                covered=bool(covered_t.all()), cov_day=cum[jd].tolist(), width_day=(hi - lo)[jd].tolist())


def evaluate(env_key, c, seed, Hs, cal, variants=None, keep_series=False):
    """Replay release c of one environment with one seed; every method monitors the same runs."""
    E = ENVS[env_key]
    hz, ell = E["horizon"], E["landmark"]
    P = Pairs(env_key, c, seed)
    ts = np.arange(1, int(round(hz / STEP)) + 1) * STEP
    n_px, n_sig, n_lm, n_scr = P.counts(ts)
    truth = effect(env_key, c)
    idx, cens = Hs["idx"], Hs["cens"]
    piA, piB = cens(P.A), cens(P.B)
    a = cal["alpha"]
    tstar = E["rate"] * 3.0          # TENOR's residual CS: tightest about three days of landmark pairs in
    twait = E["rate"] * 1.0          # outcome-only and PARC-style CSs decide soon after the landmark: tightest a day in
    tsig = E["rate"] * 0.75          # index, divergence and proxy CSs: about eighteen hours of pairs
    sv = {name: (idx[name](P.A), idx[name](P.B)) for name in ("full", "Q", "W", "judge", "const")}
    dS = P.diff(*sv["full"])
    gQ, gW = cal["gQ"], cal["gW"]
    ddiv = P.diff(gQ * sv["Q"][0] - gW * sv["W"][0], gQ * sv["Q"][1] - gW * sv["W"][1])
    dlim = gQ + gW
    e_full = P.diff(rect(P.A, sv["full"][0], piA, ell), rect(P.B, sv["full"][1], piB, ell))
    e_const = P.diff(rect(P.A, sv["const"][0], piA, ell), rect(P.B, sv["const"][1], piB, ell))
    e_judge = P.diff(rect(P.A, sv["judge"][0], piA, ell), rect(P.B, sv["judge"][1], piB, ell))
    dJ = P.diff(*sv["judge"])
    px = lambda r: (r["tests"] * (r["judge"] >= 0.5)).astype(float)
    dpx = P.diff(px(P.A), px(P.B))
    mean_at = lambda x, n: float(x[: max(n, 1)].mean()) if n > 0 else 0.0
    out = {}
    # outcome only (waiting for the outcome): the same rectified estimator with a constant index
    lo, hi = at(*asymp_cs(e_const, 0.05, twait), n_lm)
    out["wait"] = summarise(lo, hi, mean_at(e_const, n_lm[-1]), truth, ts, hz)
    # proxy only: success at submission (tests pass and the judge approves)
    lo, hi = at(*betting_cs_range(dpx, 0.05, -1, 1, tsig), n_px)
    out["proxy"] = summarise(lo, hi, mean_at(dpx, n_px[-1]), truth, ts, hz)
    # naive surrogate index: no margin, no screen, no outcomes
    lo, hi = at(*betting_cs_range(dS, 0.05, -1, 1, tsig), n_sig)
    out["naive"] = summarise(lo, hi, mean_at(dS, n_sig[-1]), truth, ts, hz)
    # PARC-style prediction-powered inference with the calibrated judge as the predictor
    l1, h1 = at(*betting_cs_range(dJ, 0.025, -1, 1, tsig), n_sig)
    l2, h2 = at(*asymp_cs(e_judge, 0.025, twait), n_lm)
    out["parc"] = summarise(l1 + l2, h1 + h2, mean_at(dJ, n_sig[-1]) + mean_at(e_judge, n_lm[-1]), truth, ts, hz)
    # TENOR
    S = at(*betting_cs_range(dS, a["s"], -1, 1, tsig), n_sig)
    e = at(*asymp_cs(e_full, a["o"], tstar), n_lm)
    div = at(*betting_cs_range(ddiv, a["d"], -dlim, dlim, tsig), n_scr)
    est = mean_at(dS, n_sig[-1]) + mean_at(e_full, n_lm[-1])
    lo, hi, lic, flag = tenor_bounds(S, e, div, cal)
    out["tenor"] = summarise(lo, hi, est, truth, ts, hz)
    out["tenor"].update(licensed=bool(lic.any()), t_lic=float(ts[np.argmax(lic)]) if lic.any() else None,
                        flagged=bool(flag.any()), t_flag=float(ts[np.argmax(flag)]) if flag.any() else None)
    if keep_series:
        out["series"] = dict(ts=ts.tolist(), truth=truth)
        for name, (l, h) in dict(tenor=(lo, hi), wait=at(*asymp_cs(e_const, 0.05, twait), n_lm),
                                 proxy=at(*betting_cs_range(dpx, 0.05, -1, 1, tsig), n_px)).items():
            out["series"][name] = (np.clip(l, -1, 1).tolist(), np.clip(h, -1, 1).tolist())
        out["series"]["proxy_est"] = [mean_at(dpx, n) for n in n_px]
        out["series"]["licensed"] = lic.tolist(); out["series"]["flagged"] = flag.tolist()
    if variants is None:
        return out, truth
    # ablations of TENOR's components
    for mode, name in (("none", "tenor_noscreen"), ("one", "tenor_onesided")):
        lo, hi, _, _ = tenor_bounds(S, e, div, cal, mode=mode)
        out[name] = summarise(lo, hi, est, truth, ts, hz)
    lo, hi, _, _ = tenor_bounds(S, e, div, cal, gamma=1.0)
    out["tenor_noslope"] = summarise(lo, hi, est, truth, ts, hz)
    lo, hi, _, _ = tenor_bounds(S, e, div, cal, kappa=0.0)
    out["tenor_nomargin"] = summarise(lo, hi, est, truth, ts, hz)
    big = (np.full_like(e[0], -2.0), np.full_like(e[1], 2.0))
    lo, hi, _, _ = tenor_bounds(S, big, div, cal)
    out["tenor_norect"] = summarise(lo, hi, cal["gamma"] * mean_at(dS, n_sig[-1]), truth, ts, hz)
    e_nw = P.diff(rect(P.A, sv["full"][0], piA, ell, weights=False), rect(P.B, sv["full"][1], piB, ell, weights=False))
    e2 = at(*asymp_cs(e_nw, a["o"], tstar), n_lm)
    lo, hi, _, _ = tenor_bounds(S, e2, div, cal)
    out["tenor_noweights"] = summarise(lo, hi, mean_at(dS, n_sig[-1]) + mean_at(e_nw, n_lm[-1]), truth, ts, hz)
    # signal ablations: drop one signal from the index and from the screen, each with its own validation
    for s in SIGNALS:
        v = variants[s]
        fA, fB = idx[v["sets"][0]](P.A), idx[v["sets"][0]](P.B)
        dS2 = P.diff(fA, fB)
        S2 = at(*betting_cs_range(dS2, v["cal"]["alpha"]["s"], -1, 1, tsig), n_sig)
        e3 = P.diff(rect(P.A, fA, piA, ell), rect(P.B, fB, piB, ell))
        E3 = at(*asymp_cs(e3, v["cal"]["alpha"]["o"], tstar), n_lm)
        gq, gw = v["cal"]["gQ"], v["cal"]["gW"]
        qi, wi = idx[v["sets"][1]], idx[v["sets"][2]]
        dd = P.diff(gq * qi(P.A) - gw * wi(P.A), gq * qi(P.B) - gw * wi(P.B))
        div2 = at(*betting_cs_range(dd, v["cal"]["alpha"]["d"], -(gq + gw), gq + gw, tsig), n_scr)
        lo, hi, _, _ = tenor_bounds(S2, E3, div2, v["cal"])
        out["drop_" + s] = summarise(lo, hi, mean_at(dS2, n_sig[-1]) + mean_at(e3, n_lm[-1]), truth, ts, hz)
    return out, truth


# ------------------------------------------------------------------------------------------------------------
# Experiments beyond the replays
# ------------------------------------------------------------------------------------------------------------
VAR_SETS = ["const", "tests", "judge", "W", "ccv", "Q", "full"]
VAR_PLAN = {"recon": dict(change=16, times=[10.0, 12.0, 16.0, 24.0, 40.0]),
            "pr": dict(change=18, times=[14.5, 16.0, 20.0, 28.0, 40.0])}
VAR_SCALE, VAR_SEEDS = 0.1, 400


def _var_job(args):
    """One seed of the variance experiment: the rectified estimate of the effect for each index at each time."""
    env_key, seed = args
    plan, E = VAR_PLAN[env_key], ENVS[env_key]
    P = Pairs(env_key, plan["change"], seed, rate=E["rate"] * VAR_SCALE, salt=555)
    Hs = _H[env_key]
    ts = np.array(plan["times"])
    _, n_sig, n_lm, _ = P.counts(ts)
    piA, piB = Hs["cens"](P.A), Hs["cens"](P.B)
    est = {}
    for name in VAR_SETS:
        m = Hs["idx"][name]
        sA, sB = m(P.A), m(P.B)
        d = P.diff(sA, sB)
        e = P.diff(rect(P.A, sA, piA, E["landmark"]), rect(P.B, sB, piB, E["landmark"]))
        est[name] = [float(d[:N].mean() + e[:n].mean()) if name != "const" else float(e[:n].mean())
                     for N, n in zip(n_sig, n_lm)]
    rho = [float(n * np.concatenate([piA[:n], piB[:n]]).mean() / N) for N, n in zip(n_sig, n_lm)]
    return env_key, est, rho


def _var_r2(env_key):
    """Each index's outcome R^2 on the runs of the experiment's two versions (a large sample)."""
    plan = VAR_PLAN[env_key]
    rng = np.random.default_rng(MASTER + 4242)
    runs = concat([simulate_runs(rng, env_key, v, 200000, plan["change"]) for v in (plan["change"] - 1, plan["change"])])
    return {name: r2(runs["y"], _H[env_key]["idx"][name](runs)) for name in VAR_SETS}


CENS_PLAN = dict(env="repl", changes=[12, 15, 21, 22], times=[12.0, 14.0, 17.0], seeds=200, scale=0.5)


def _cens_job(args):
    """One seed of the censoring experiment: outcome-only and rectified estimates with and without censoring
    weights, at times soon after the landmark, in the environment whose delays depend on the signals."""
    c, seed = args
    k, E = CENS_PLAN["env"], ENVS[CENS_PLAN["env"]]
    P = Pairs(k, c, seed, rate=E["rate"] * CENS_PLAN["scale"], salt=777)
    Hs = _H[k]
    ts = np.array(CENS_PLAN["times"])
    _, n_sig, n_lm, _ = P.counts(ts)
    piA, piB = Hs["cens"](P.A), Hs["cens"](P.B)
    out = {}
    for name in ("const", "full"):
        m = Hs["idx"][name]
        sA, sB = m(P.A), m(P.B)
        d = P.diff(sA, sB)
        for wtd in (True, False):
            e = P.diff(rect(P.A, sA, piA, E["landmark"], weights=wtd), rect(P.B, sB, piB, E["landmark"], weights=wtd))
            out[(name, wtd)] = [float((0 if name == "const" else d[:N].mean()) + e[:n].mean()) for N, n in zip(n_sig, n_lm)]
    return c, out


def history_size_study(changes, sizes=(10, 15, 20, 25, 30, 35), reps=400):
    """Validation on random subsets of the past changes: the margin, and how often it covers the residual of the
    held-out changes the subset's screen passes."""
    rng = np.random.default_rng(MASTER + 31)
    out = {}
    for m in sizes:
        kap, cov, npass = [], [], []
        for _ in range(reps):
            pick = rng.choice(len(changes), m, replace=False)
            sub = [changes[i] for i in pick]
            v = calibrate_screen(sub, record=False)
            held = [c for i, c in enumerate(changes) if i not in set(pick.tolist())]
            div_h = np.array([v["gQ"] * c["dQ"] - v["gW"] * c["dW"] for c in held])
            ok = np.abs(div_h) <= v["tau"]
            res = np.array([c["dY"] - v["gamma"] * c["dS"] for c in held])
            if ok.any():
                cov.append(float(np.mean(np.abs(res[ok]) <= v["kappa"])))
            kap.append(v["kappa"]); npass.append(v["n_pass"])
        out[m] = dict(kappa=float(np.mean(kap)), kappa_q=np.percentile(kap, [10, 90]).tolist(), coverage=float(np.mean(cov)),
                      nominal=float(np.mean([n / (n + 1) for n in npass])))
    return out


def loo_history(changes):
    """Leave one past change out: is it flagged by the screen the other changes calibrate, and does the margin
    they calibrate cover its residual?"""
    rows = []
    for i, c in enumerate(changes):
        rest = changes[:i] + changes[i + 1:]
        v = calibrate_screen(rest, record=False)
        div = v["gQ"] * c["dQ"] - v["gW"] * c["dW"]
        res = c["dY"] - v["gamma"] * c["dS"]
        rows.append(dict(flagged=bool(abs(div) > v["tau"]), covered=bool(abs(res) <= v["kappa"]), kind=c["kind"],
                         div=float(div), res=float(res), kappa=v["kappa"]))
    return rows


# ------------------------------------------------------------------------------------------------------------
# Parallel drivers (fork: workers see the module globals below)
# ------------------------------------------------------------------------------------------------------------
_H, _CAL, _VAR = {}, {}, {}


def _hist_job(k):
    return k, history(k)


def _replay_job(args):
    k, c, s = args
    out, truth = evaluate(k, c, s, _H[k], _CAL, variants=_VAR)
    keep = ("t", "dec", "forced", "correct", "covered", "cov_day", "width_day", "licensed", "t_lic", "flagged", "t_flag")
    return dict(env=k, change=c, seed=s, kind=VERS[k]["kind"][c], truth=truth,
                m={m: {x: o[x] for x in keep if x in o} for m, o in out.items()})


def cluster_boot(rows, stat, reps=2000, seed=0):
    """95% interval of a statistic of the replays, resampling the changes (all seeds of a change together)."""
    rng = np.random.default_rng(MASTER + seed)
    keys = sorted({(r["env"], r["change"]) for r in rows})
    groups = {k: [r for r in rows if (r["env"], r["change"]) == k] for k in keys}
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(stat([r for i in pick for r in groups[keys[i]]]))
    return np.percentile(vals, [2.5, 97.5]).tolist()


def tex_num(x, nd=1):
    return f"{x:.{nd}f}"


def main():
    t_start = time.time()
    import multiprocessing as mp
    import pickle
    ctx = mp.get_context("fork")
    cache = os.environ.get("TENOR_CACHE")
    if cache and os.environ.get("TENOR_REUSE") and os.path.exists(cache):
        saved = pickle.load(open(cache, "rb"))
        _H.update(saved["H"]); _CAL.update(saved["CAL"]); _VAR.update(saved["VAR"])
        R = assemble(*saved["args"])
        with open(os.path.join(ROOT, "results.json"), "w") as f:
            json.dump(R, f, indent=1)
        write_tables(R)
        return
    workers = max(1, os.cpu_count() or 1)
    with ctx.Pool(min(4, workers)) as pool:
        _H.update(dict(pool.map(_hist_job, ENV_KEYS)))
    changes = [c for k in ENV_KEYS for c in _H[k]["changes"]]
    _CAL.update(calibrate_screen(changes))
    _CAL["alpha"] = alphas(_CAL["n_pass"])
    for s in SIGNALS:
        sets = variant_sets(s)
        vc = calibrate_screen(changes, sets=sets, record=False)
        vc["alpha"] = alphas(vc["n_pass"])
        _VAR[s] = dict(sets=sets, cal=vc)
    print(f"history and validation: {time.time() - t_start:.0f}s; gamma {_CAL['gamma']:.3f} tau {100 * _CAL['tau']:.2f} "
          f"kappa {100 * _CAL['kappa']:.2f} n_pass {_CAL['n_pass']}")
    jobs = [(k, c, s) for k in ENV_KEYS for c in range(11, 26) for s in SEEDS]
    with ctx.Pool(workers) as pool:
        rows = pool.map(_replay_job, jobs, chunksize=1)
        var_rows = pool.map(_var_job, [(k, s) for k in VAR_PLAN for s in range(VAR_SEEDS)], chunksize=4)
        var_r2 = dict(zip(VAR_PLAN, pool.map(_var_r2, list(VAR_PLAN))))
        cens_rows = pool.map(_cens_job, [(c, s) for c in CENS_PLAN["changes"] for s in range(CENS_PLAN["seeds"])], chunksize=4)
    print(f"replays and variance experiment: {time.time() - t_start:.0f}s")
    # teaser: one honest improvement and one hacked change of the pull-request environment
    teaser = {}
    for name, c in (("honest", 18), ("hacked", 20)):
        out, truth = evaluate("pr", c, 0, _H["pr"], _CAL, keep_series=True)
        s = out["series"]
        keep = slice(0, None, 2)   # every second hour
        teaser[name] = dict(change=c, truth=truth, ts=s["ts"][keep], tenor=[x[keep] for x in s["tenor"]],
                            wait=[x[keep] for x in s["wait"]], proxy=[x[keep] for x in s["proxy"]],
                            proxy_est=s["proxy_est"][keep], licensed=s["licensed"][keep], flagged=s["flagged"][keep],
                            t_tenor=out["tenor"]["t"], dec_tenor=out["tenor"]["dec"], t_wait=out["wait"]["t"],
                            dec_wait=out["wait"]["dec"], t_proxy=out["proxy"]["t"], dec_proxy=out["proxy"]["dec"],
                            t_lic=out["tenor"]["t_lic"], t_flag=out["tenor"]["t_flag"])
    hist_study = history_size_study(changes)
    loo = loo_history(changes)
    if cache:
        pickle.dump(dict(H=_H, CAL=_CAL, VAR=_VAR, args=(rows, changes, var_rows, var_r2, teaser, hist_study, loo, cens_rows)), open(cache, "wb"))
    R = assemble(rows, changes, var_rows, var_r2, teaser, hist_study, loo, cens_rows)
    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(R, f, indent=1)
    write_tables(R)
    print(f"done in {time.time() - t_start:.0f}s")
    for m in METHODS:
        x = R["methods"][m]
        print(f"{m:7s} wrong {x['wrong']:5.1f}%  covered {x['coverage']:5.1f}%  median days {x['median_days']:5.2f}")


LABELS = {"wait": "Wait for the outcome", "proxy": "Proxy only (tests pass and judge approves)",
          "naive": "Surrogate index, no validation", "parc": "PPI with the judge as predictor (PARC-style)",
          "tenor": "TENOR"}
ABL_LABELS = {"tenor": "TENOR (full)", "tenor_noscreen": "without the divergence screen",
              "tenor_onesided": "one-sided screen (flags only hacks)", "tenor_noslope": "without the slope ($\\gamma=1$)",
              "tenor_nomargin": "without the margin ($\\kappa=0$)", "tenor_norect": "without the rectifier (transport only)",
              "tenor_noweights": "without censoring weights"}
SIG_LABELS = {"judge": "judge score", "tests": "tests passed", "ccv": "completion probability",
              "comments": "reviewer comments", "rollback": "rollbacks", "perturb": "perturbed re-checks"}
ENV_SHORT = {"pr": "Pull requests", "recon": "Reconciliation", "repl": "Replication", "sim": "Business sim."}


def method_stats(rows, m):
    t = np.array([r["m"][m]["t"] for r in rows])
    wrong = np.array([not r["m"][m]["correct"] for r in rows])
    cov = np.array([r["m"][m]["covered"] for r in rows])
    regress = [r for r in rows if r["truth"] < -DELTA]
    shipped = np.mean([r["m"][m]["dec"] == 1 for r in regress]) if regress else float("nan")
    good = [r for r in rows if r["truth"] > 0]
    blocked = np.mean([r["m"][m]["dec"] == -1 for r in good]) if good else float("nan")
    wd = np.array([r["m"][m]["width_day"][:42] for r in rows if "width_day" in r["m"][m]])
    return dict(wrong=100 * float(wrong.mean()), n_wrong=int(wrong.sum()), coverage=100 * float(cov.mean()),
                n_miss=int((~cov).sum()), median_days=float(np.median(t)), q25=float(np.percentile(t, 25)),
                q75=float(np.percentile(t, 75)), shipped_regressions=100 * float(shipped), blocked_improvements=100 * float(blocked),
                forced=100 * float(np.mean([r["m"][m]["forced"] for r in rows])),
                width3=100 * float(np.median(wd[:, 2])) if wd.size else None, width7=100 * float(np.median(wd[:, 6])) if wd.size else None,
                width14=100 * float(np.median(wd[:, 13])) if wd.size else None)


def assemble(rows, changes, var_rows, var_r2, teaser, hist_study, loo, cens_rows):
    R = dict(cal={k: v for k, v in _CAL.items() if k not in ("div", "resid", "passed")}, n_replays=len(rows))
    R["methods"] = {m: method_stats(rows, m) for m in METHODS}
    for m in METHODS:
        R["methods"][m]["ci_wrong"] = cluster_boot(rows, lambda rs, m=m: 100 * np.mean([not r["m"][m]["correct"] for r in rs]))
        R["methods"][m]["ci_cov"] = cluster_boot(rows, lambda rs, m=m: 100 * np.mean([r["m"][m]["covered"] for r in rs]), seed=1)
        R["methods"][m]["ci_median"] = cluster_boot(rows, lambda rs, m=m: float(np.median([r["m"][m]["t"] for r in rs])), seed=2)
    R["diff_wrong_proxy_tenor"] = cluster_boot(rows, lambda rs: 100 * np.mean([(not r["m"]["proxy"]["correct"]) - (not r["m"]["tenor"]["correct"]) for r in rs]), seed=3)
    R["diff_days_wait_tenor"] = cluster_boot(rows, lambda rs: float(np.median([r["m"]["wait"]["t"] for r in rs]) - np.median([r["m"]["tenor"]["t"] for r in rs])), seed=4)
    R["ablations"] = {m: method_stats(rows, m) for m in ["tenor"] + ABLATIONS}
    R["signals"] = {s: dict(method_stats(rows, "drop_" + s), gamma=_VAR[s]["cal"]["gamma"], kappa=100 * _VAR[s]["cal"]["kappa"],
                            tau=100 * _VAR[s]["cal"]["tau"], n_pass=_VAR[s]["cal"]["n_pass"],
                            r2=float(np.mean([_H[k]["r2"]["drop_" + s] for k in ENV_KEYS]))) for s in SIGNALS}
    R["per_env"] = {k: {m: method_stats([r for r in rows if r["env"] == k], m) for m in METHODS} for k in ENV_KEYS}
    R["per_env_flagged"] = {k: 100 * float(np.mean([r["m"]["tenor"].get("flagged", False) for r in rows if r["env"] == k])) for k in ENV_KEYS}
    lic = [r["m"]["tenor"]["t_lic"] for r in rows if r["kind"] == "honest" and r["m"]["tenor"].get("t_lic")]
    R["lic_honest"] = dict(median=float(np.median(lic)), share=100 * len(lic) / sum(r["kind"] == "honest" for r in rows))
    kinds = sorted({r["kind"] for r in rows})
    R["per_kind"] = {kd: dict(n=len([r for r in rows if r["kind"] == kd]),
                              **{m: method_stats([r for r in rows if r["kind"] == kd], m) for m in ("tenor", "proxy", "wait", "naive")},
                              flagged=100 * float(np.mean([r["m"]["tenor"].get("flagged", False) for r in rows if r["kind"] == kd])),
                              licensed=100 * float(np.mean([r["m"]["tenor"].get("licensed", False) for r in rows if r["kind"] == kd])))
                     for kd in kinds}
    R["over_time"] = {m: dict(coverage=(100 * np.mean([r["m"][m]["cov_day"][:42] for r in rows], 0)).tolist(),
                              width=(100 * np.median([r["m"][m]["width_day"][:42] for r in rows], 0)).tolist()) for m in METHODS}
    R["times"] = {m: sorted(r["m"][m]["t"] for r in rows) for m in METHODS}
    R["times_env"] = {k: {m: sorted(r["m"][m]["t"] for r in rows if r["env"] == k) for m in ("tenor", "wait")} for k in ENV_KEYS}
    R["replays"] = [dict(env=r["env"], change=r["change"], seed=r["seed"], kind=r["kind"], truth=r["truth"],
                         **{m: dict(t=r["m"][m]["t"], dec=r["m"][m]["dec"], correct=r["m"][m]["correct"], covered=r["m"][m]["covered"])
                            for m in METHODS}) for r in rows]
    # history: the forty past changes
    R["history"] = [dict(env=c["env"], change=c["change"], kind=c["kind"], dY=c["dY"], dY_true=c["dY_true"], dS=c["dS"],
                         dQ=c["dQ"], dW=c["dW"], dJ=c["dJ"], proxy=c["proxy"], tests=c["tests"], judge=c["judge"],
                         div=c["div"], resid=c["resid"], passed=c["passed"]) for c in changes]
    dY = np.array([c["dY"] for c in changes])
    val = {}
    for key, label, vals in (("proxy", "Tests pass and judge approves", [c["proxy"] for c in changes]),
                             ("tests", "Tests passed", [c["tests"] for c in changes]),
                             ("judge", "Judge approves", [c["judge"] for c in changes]),
                             ("judge_index", "Calibrated judge (PARC-style predictor)", [c["dJ"] for c in changes]),
                             ("index", "Surrogate index", [c["dS"] for c in changes])):
        v = np.array(vals)
        val[key] = dict(label=label, sign=100 * float(np.mean(np.sign(v) == np.sign(dY))),
                        r2=float(np.corrcoef(v, dY)[0, 1] ** 2),
                        paradox=int(np.sum((np.sign(v) != np.sign(dY)) & (np.abs(dY) >= DELTA) & (np.abs(v) >= 0.01))))
    passed = np.array([c["passed"] for c in changes])
    vS = _CAL["gamma"] * np.array([c["dS"] for c in changes])
    val["tenor"] = dict(label="TENOR: screened, slope-corrected index", sign=100 * float(np.mean(np.sign(vS[passed]) == np.sign(dY[passed]))),
                        r2=float(np.corrcoef(vS[passed], dY[passed])[0, 1] ** 2),
                        paradox=int(np.sum((np.sign(vS[passed]) != np.sign(dY[passed])) & (np.abs(dY[passed]) >= DELTA))))
    R["validation"] = val
    kinds_h = sorted({c["kind"] for c in changes})
    R["screen_history"] = {kd: dict(n=int(sum(c["kind"] == kd for c in changes)),
                                    flagged=int(sum((c["kind"] == kd) and not c["passed"] for c in changes))) for kd in kinds_h}
    R["loo"] = dict(flag_by_kind={kd: [int(sum(x["flagged"] for x in loo if x["kind"] == kd)), int(sum(x["kind"] == kd for x in loo))] for kd in kinds_h},
                    covered=int(sum(x["covered"] for x in loo if not x["flagged"])), passed=int(sum(not x["flagged"] for x in loo)))
    R["history_size"] = {str(m): v for m, v in hist_study.items()}
    # variance experiment
    V = {}
    for k in VAR_PLAN:
        ests = [e for kk, e, _ in var_rows if kk == k]
        rhos = np.mean([rh for kk, _, rh in var_rows if kk == k], 0)
        var = {name: np.var([e[name] for e in ests], axis=0, ddof=1) for name in VAR_SETS}
        V[k] = dict(times=VAR_PLAN[k]["times"], rho=rhos.tolist(), r2=var_r2[k],
                    ratio={name: (var[name] / var["const"]).tolist() for name in VAR_SETS},
                    theory={name: [1 - (1 - rh) * max(var_r2[k][name], 0) for rh in rhos] for name in VAR_SETS})
    R["variance"] = V
    C = {}
    for c in CENS_PLAN["changes"]:
        truth = effect(CENS_PLAN["env"], c)
        ests = [o for cc, o in cens_rows if cc == c]
        C[str(c)] = dict(kind=VERS[CENS_PLAN["env"]]["kind"][c], truth=truth, times=CENS_PLAN["times"],
                         **{f"{name}_{'w' if wtd else 'u'}": dict(
                             bias=(100 * (np.mean([o[(name, wtd)] for o in ests], 0) - truth)).tolist(),
                             rmse=(100 * np.sqrt(np.mean((np.array([o[(name, wtd)] for o in ests]) - truth) ** 2, 0))).tolist())
                            for name in ("const", "full") for wtd in (True, False)})
    R["censoring"] = C
    R["teaser"] = teaser
    # environments
    R["envs"] = {k: dict(name=ENVS[k]["name"], outcome=ENVS[k]["outcome"], rate=ENVS[k]["rate"], landmark=ENVS[k]["landmark"],
                         horizon=ENVS[k]["horizon"], delay_med=_H[k]["delay_med"], pi_bar=_H[k]["pi_bar"], base=ENVS[k]["base"],
                         r2=_H[k]["r2"]["full"], r2_judge=_H[k]["r2"]["judge"], r2_Q=_H[k]["r2"]["Q"], r2_W=_H[k]["r2"]["W"],
                         ece=_H[k]["ece"], rel=_H[k]["rel"], ybar=_H[k]["ybar"],
                         kinds=[VERS[k]["kind"][c] for c in range(1, 26)]) for k in ENV_KEYS}
    return R


# ------------------------------------------------------------------------------------------------------------
# Tables and the numbers the text quotes
# ------------------------------------------------------------------------------------------------------------
def write_tables(R):
    T = os.path.join(ROOT, "tables")
    macros = {}

    def mac(name, value):
        macros[name] = value

    # environments
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Environment & Outcome $Y$ & Delay & $\ell$ & $\bar\pi_\ell$ & Runs/day & $R^2_S$ & ECE \\",
             r" & & (median, d) & (d) & & per version & & \\", r"\midrule"]
    for k in ENV_KEYS:
        e = R["envs"][k]
        lines.append(f"{e['name']} & {e['outcome'].replace('%', chr(92) + '%')} & {e['delay_med']:.1f} & {e['landmark']:.1f} & {e['pi_bar']:.2f} & "
                     f"{e['rate']:,.0f} & {e['r2']:.2f} & {e['ece']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "envs.tex"), "w").write("\n".join(lines) + "\n")

    # main results
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & Wrong & Regressions & Days to decision & Anytime & \multicolumn{2}{c}{Median width (pts)} \\",
             r"\cmidrule(lr){6-7}",
             r" & decisions (\%) & shipped (\%) & median [IQR] & coverage (\%) & day 3 & day 14 \\", r"\midrule"]
    for m in METHODS:
        x = R["methods"][m]
        name = LABELS[m] if m != "tenor" else r"\textbf{TENOR}"
        wrong = f"{x['wrong']:.1f} [{x['ci_wrong'][0]:.1f}, {x['ci_wrong'][1]:.1f}]"
        cov = f"{x['coverage']:.1f}"
        days = f"{x['median_days']:.1f} [{x['q25']:.1f}, {x['q75']:.1f}]"
        if m == "tenor":
            wrong, days, cov = r"\textbf{" + wrong + "}", r"\textbf{" + days + "}", r"\textbf{" + cov + "}"
        lines.append(f"{name} & {wrong} & {x['shipped_regressions']:.1f} & {days} & {cov} & {x['width3']:.1f} & {x['width14']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "main.tex"), "w").write("\n".join(lines) + "\n")

    # per environment
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Wait for the outcome} & Proxy only & \multicolumn{4}{c}{TENOR} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-8}",
             r"Environment & days & wrong (\%) & wrong (\%) & days & wrong (\%) & coverage (\%) & flagged (\%) \\", r"\midrule"]
    for k in ENV_KEYS:
        p = R["per_env"][k]
        lines.append(f"{R['envs'][k]['name']} & {p['wait']['median_days']:.1f} & {p['wait']['wrong']:.1f} & {p['proxy']['wrong']:.1f} & "
                     f"{p['tenor']['median_days']:.1f} & {p['tenor']['wrong']:.1f} & {p['tenor']['coverage']:.1f} & FLAG_{k} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(lines) + "\n"
    for k in ENV_KEYS:
        txt = txt.replace(f"FLAG_{k}", f"{R['per_env_flagged'][k]:.1f}")
    open(os.path.join(T, "perenv.tex"), "w").write(txt)

    # validation on past changes
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Effect used to predict the outcome effect & Sign agreement (\%) & Trial-level $R^2$ & Paradoxes \\", r"\midrule"]
    for key in ("proxy", "tests", "judge", "judge_index", "index", "tenor"):
        v = R["validation"][key]
        lab = v["label"] if key != "tenor" else r"\textbf{" + v["label"] + "}"
        lines.append(f"{lab} & {v['sign']:.1f} & {v['r2']:.3f} & {v['paradox']} \\\\")
        if key == "judge":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "validation.tex"), "w").write("\n".join(lines) + "\n")

    # ablations
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Variant & Wrong (\%) & Regressions shipped (\%) & Median days & Coverage (\%) \\", r"\midrule"]
    for m in ["tenor"] + ABLATIONS:
        x = R["ablations"][m]
        lines.append(f"{ABL_LABELS[m]} & {x['wrong']:.1f} & {x['shipped_regressions']:.1f} & {x['median_days']:.1f} & {x['coverage']:.1f} \\\\")
        if m == "tenor":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "ablation.tex"), "w").write("\n".join(lines) + "\n")

    # signal ablations
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Signal removed & $R^2_S$ & $\hat\gamma$ & $\kappa$ (pts) & $\tau$ (pts) & Wrong (\%) & Median days & Coverage (\%) \\", r"\midrule"]
    x = R["ablations"]["tenor"]
    lines.append(f"none (TENOR) & {np.mean([R['envs'][k]['r2'] for k in ENV_KEYS]):.2f} & {R['cal']['gamma']:.2f} & {100 * R['cal']['kappa']:.2f} & "
                 f"{100 * R['cal']['tau']:.2f} & {x['wrong']:.1f} & {x['median_days']:.1f} & {x['coverage']:.1f} \\\\")
    lines.append(r"\midrule")
    for s in SIGNALS:
        v = R["signals"][s]
        lines.append(f"{SIG_LABELS[s]} & {v['r2']:.2f} & {v['gamma']:.2f} & {v['kappa']:.2f} & {v['tau']:.2f} & {v['wrong']:.1f} & "
                     f"{v['median_days']:.1f} & {v['coverage']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "signals.tex"), "w").write("\n".join(lines) + "\n")

    # by kind of change
    order = [kd for kd in ("honest", "hacked", "unhack", "presentation", "latent") if kd in R["per_kind"]]
    names = {"honest": "Honest", "hacked": "Hacked", "unhack": "Removes a hack", "presentation": "Flatters the judge",
             "latent": "Acts through an unmeasured path"}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Change & Replays & \multicolumn{3}{c}{TENOR} & Proxy & Wait \\", r"\cmidrule(lr){3-5}",
             r" & & flagged (\%) & median days & wrong (\%) & wrong (\%) & median days \\", r"\midrule"]
    for kd in order:
        v = R["per_kind"][kd]
        lines.append(f"{names[kd]} & {v['n']} & {v['flagged']:.0f} & {v['tenor']['median_days']:.1f} & {v['tenor']['wrong']:.1f} & "
                     f"{v['proxy']['wrong']:.1f} & {v['wait']['median_days']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "kinds.tex"), "w").write("\n".join(lines) + "\n")

    # the forty past changes (appendix)
    kn = {"honest": "honest", "hacked": "hacked", "unhack": "removes a hack", "presentation": "flatters the judge"}
    lines = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
             r"Environment & Change & $\hat\Delta$ & $\hat\gamma\hat\Delta_S$ & Proxy & Residual $r_c$ & Divergence & Screen \\",
             r" & & (pts) & (pts) & (pts) & (pts) & (pts) & \\", r"\midrule"]
    prev = None
    for h in R["history"]:
        env = R["envs"][h["env"]]["name"] if h["env"] != prev else ""
        if h["env"] != prev and prev is not None:
            lines.append(r"\midrule")
        prev = h["env"]
        lines.append(f"{env} & {h['change']} ({kn[h['kind']]}) & {100 * h['dY']:+.1f} & {100 * R['cal']['gamma'] * h['dS']:+.1f} & "
                     f"{100 * h['proxy']:+.1f} & {100 * h['resid']:+.2f} & {100 * h['div']:+.2f} & {'passes' if h['passed'] else 'flagged'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "history.tex"), "w").write("\n".join(lines) + "\n")

    # censoring weights (appendix)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Change & True effect & \multicolumn{2}{c}{Outcome only: bias (pts)} & \multicolumn{2}{c}{Rectified index: bias (pts)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r" & (pts) & weighted & unweighted & weighted & unweighted \\", r"\midrule"]
    kname = {"honest": "honest", "hacked": "hacked", "unhack": "removes a hack"}
    for c, v in R["censoring"].items():
        j = 1  # day 14: three days after the landmark
        lines.append(f"{c} ({kname.get(v['kind'], v['kind'])}) & {100 * v['truth']:+.1f} & {v['const_w']['bias'][j]:+.2f} & {v['const_u']['bias'][j]:+.2f} & "
                     f"{v['full_w']['bias'][j]:+.2f} & {v['full_u']['bias'][j]:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "censoring.tex"), "w").write("\n".join(lines) + "\n")

    # numbers the text quotes
    t, w, p, nv, pc = (R["methods"][m] for m in ("tenor", "wait", "proxy", "naive", "parc"))
    mac("tenorMedDays", f"{t['median_days']:.1f}")
    mac("waitMedDays", f"{w['median_days']:.0f}")
    mac("waitMedDaysExact", f"{w['median_days']:.1f}")
    mac("tenorCov", f"{t['coverage']:.1f}")
    mac("tenorErr", f"{t['wrong']:.1f}")
    mac("tenorErrRound", f"{t['wrong']:.0f}")
    mac("proxyErr", f"{p['wrong']:.0f}")
    mac("proxyErrExact", f"{p['wrong']:.1f}")
    mac("tenorWrongN", str(t["n_wrong"])); mac("tenorMissN", str(t["n_miss"])); mac("proxyWrongN", str(p["n_wrong"]))
    mac("nReplays", str(R["n_replays"]))
    mac("tenorQlo", f"{t['q25']:.1f}"); mac("tenorQhi", f"{t['q75']:.1f}")
    mac("waitQlo", f"{w['q25']:.1f}"); mac("waitQhi", f"{w['q75']:.1f}")
    mac("tenorErrCIlo", f"{t['ci_wrong'][0]:.1f}"); mac("tenorErrCIhi", f"{t['ci_wrong'][1]:.1f}")
    mac("proxyErrCIlo", f"{p['ci_wrong'][0]:.1f}"); mac("proxyErrCIhi", f"{p['ci_wrong'][1]:.1f}")
    mac("tenorCovCIlo", f"{t['ci_cov'][0]:.1f}"); mac("tenorCovCIhi", f"{t['ci_cov'][1]:.1f}")
    mac("tenorMedCIlo", f"{t['ci_median'][0]:.1f}"); mac("tenorMedCIhi", f"{t['ci_median'][1]:.1f}")
    mac("diffErrlo", f"{R['diff_wrong_proxy_tenor'][0]:.1f}"); mac("diffErrhi", f"{R['diff_wrong_proxy_tenor'][1]:.1f}")
    mac("diffDayslo", f"{R['diff_days_wait_tenor'][0]:.1f}"); mac("diffDayshi", f"{R['diff_days_wait_tenor'][1]:.1f}")
    mac("waitErr", f"{w['wrong']:.1f}"); mac("waitCov", f"{w['coverage']:.1f}")
    mac("parcErr", f"{pc['wrong']:.1f}"); mac("parcCov", f"{pc['coverage']:.1f}"); mac("parcMedDays", f"{pc['median_days']:.1f}")
    mac("naiveErr", f"{nv['wrong']:.1f}"); mac("naiveCov", f"{nv['coverage']:.1f}"); mac("naiveMedDays", f"{nv['median_days']:.1f}")
    mac("proxyCov", f"{p['coverage']:.1f}"); mac("proxyMedDays", f"{p['median_days']:.1f}")
    mac("proxyShipped", f"{p['shipped_regressions']:.1f}"); mac("tenorShipped", f"{t['shipped_regressions']:.1f}")
    mac("naiveShipped", f"{nv['shipped_regressions']:.1f}"); mac("waitShipped", f"{w['shipped_regressions']:.1f}")
    mac("tenorWidthThree", f"{t['width3']:.1f}"); mac("waitWidthThree", f"{w['width3']:.1f}")
    mac("tenorWidthFourteen", f"{t['width14']:.1f}"); mac("waitWidthFourteen", f"{w['width14']:.1f}")
    c = R["cal"]
    mac("gammaVal", f"{c['gamma']:.2f}"); mac("pteVal", f"{100 / c['gamma']:.0f}")
    mac("gammaQ", f"{c['gQ']:.2f}"); mac("gammaW", f"{c['gW']:.2f}")
    mac("kappaPts", f"{100 * c['kappa']:.2f}"); mac("tauPts", f"{100 * c['tau']:.2f}")
    mac("nPass", str(c["n_pass"])); mac("alphaKappa", f"{1 / (c['n_pass'] + 1):.3f}")
    mac("shipThresh", f"{100 * (c['kappa'] - DELTA) / c['gamma']:.2f}"); mac("blockThresh", f"{-100 * c['kappa'] / c['gamma']:.2f}")
    mac("bandWidth", f"{100 * (2 * c['kappa'] - DELTA) / c['gamma']:.2f}")
    mac("alphaRest", f"{c['alpha']['s']:.4f}")
    v = R["validation"]
    for key, nm in (("proxy", "Proxy"), ("tests", "Tests"), ("judge", "Judge"), ("judge_index", "JudgeIdx"), ("index", "Index"), ("tenor", "Tenor")):
        mac(f"sign{nm}", f"{v[key]['sign']:.0f}"); mac(f"rsq{nm}", f"{v[key]['r2']:.3f}"); mac(f"paradox{nm}", str(v[key]["paradox"]))
    sh = R["screen_history"]
    mac("histHacked", str(sh.get("hacked", {}).get("n", 0))); mac("histUnhack", str(sh.get("unhack", {}).get("n", 0)))
    mac("histPresentation", str(sh.get("presentation", {}).get("n", 0))); mac("histHonest", str(sh.get("honest", {}).get("n", 0)))
    mac("histFlaggedHonest", str(sh.get("honest", {}).get("flagged", 0)))
    mac("histFlaggedOther", str(sum(sh[kd]["flagged"] for kd in sh if kd != "honest")))
    lo = R["loo"]
    mac("looCovered", str(lo["covered"])); mac("looPassed", str(lo["passed"]))
    mac("looFlagHacked", f"{lo['flag_by_kind'].get('hacked', [0, 0])[0]}")
    for k in ENV_KEYS:
        pe = R["per_env"][k]
        K = {"pr": "PR", "recon": "Recon", "repl": "Repl", "sim": "Sim"}[k]
        mac(f"tenorDays{K}", f"{pe['tenor']['median_days']:.1f}"); mac(f"waitDays{K}", f"{pe['wait']['median_days']:.1f}")
        mac(f"tenorErr{K}", f"{pe['tenor']['wrong']:.1f}"); mac(f"proxyErr{K}", f"{pe['proxy']['wrong']:.1f}")
        mac(f"rate{K}", f"{R['envs'][k]['rate']:,.0f}"); mac(f"rsqEnv{K}", f"{R['envs'][k]['r2']:.2f}")
    for kd in R["per_kind"]:
        K = {"honest": "Honest", "hacked": "Hacked", "unhack": "Unhack", "presentation": "Presentation", "latent": "Latent"}[kd]
        x = R["per_kind"][kd]
        mac(f"kindN{K}", str(x["n"])); mac(f"kindDays{K}", f"{x['tenor']['median_days']:.1f}")
        mac(f"kindErr{K}", f"{x['tenor']['wrong']:.1f}"); mac(f"kindFlag{K}", f"{x['flagged']:.0f}")
        mac(f"kindProxyErr{K}", f"{x['proxy']['wrong']:.1f}"); mac(f"kindNaiveErr{K}", f"{x['naive']['wrong']:.1f}")
    mac("kindMissLatent", str(R["per_kind"]["latent"]["tenor"]["n_miss"]))
    mac("kindWaitDaysHacked", f"{R['per_kind']['hacked']['wait']['median_days']:.1f}")
    mac("licDaysHonest", f"{R['lic_honest']['median']:.1f}"); mac("licShareHonest", f"{R['lic_honest']['share']:.0f}")
    mac("censMaxBias", f"{max(abs(v['const_u']['bias'][1]) for v in R['censoring'].values()):.1f}")
    mac("censMaxBiasWeighted", f"{max(abs(v['const_w']['bias'][1]) for v in R['censoring'].values()):.1f}")
    mac("censMaxBiasRect", f"{max(abs(v['full_u']['bias'][1]) for v in R['censoring'].values()):.1f}")
    a = R["ablations"]
    mac("ablNoRectShipped", f"{a['tenor_norect']['shipped_regressions']:.1f}")
    for m, nm in (("tenor_noscreen", "NoScreen"), ("tenor_onesided", "OneSided"), ("tenor_noslope", "NoSlope"),
                  ("tenor_nomargin", "NoMargin"), ("tenor_norect", "NoRect"), ("tenor_noweights", "NoWeights")):
        mac(f"abl{nm}Err", f"{a[m]['wrong']:.1f}"); mac(f"abl{nm}Days", f"{a[m]['median_days']:.1f}"); mac(f"abl{nm}Cov", f"{a[m]['coverage']:.1f}")
    for s in SIGNALS:
        K = {"judge": "Judge", "tests": "Tests", "ccv": "Ccv", "comments": "Comments", "rollback": "Rollback", "perturb": "Perturb"}[s]
        x = R["signals"][s]
        mac(f"drop{K}Days", f"{x['median_days']:.1f}"); mac(f"drop{K}Err", f"{x['wrong']:.1f}"); mac(f"drop{K}Kappa", f"{x['kappa']:.2f}")
        mac(f"drop{K}Cov", f"{x['coverage']:.1f}")
    V = R["variance"]
    for k, K in (("recon", "Recon"), ("pr", "PR")):
        mac(f"varFull{K}Early", f"{V[k]['ratio']['full'][0]:.2f}"); mac(f"varFull{K}Late", f"{V[k]['ratio']['full'][-1]:.2f}")
        mac(f"varTheory{K}Early", f"{V[k]['theory']['full'][0]:.2f}"); mac(f"varTheory{K}Late", f"{V[k]['theory']['full'][-1]:.2f}")
        mac(f"varFull{K}Min", f"{min(V[k]['ratio']['full']):.2f}"); mac(f"varFull{K}Max", f"{max(V[k]['ratio']['full']):.2f}")
        mac(f"varRsq{K}", f"{V[k]['r2']['full']:.2f}"); mac(f"varRho{K}Early", f"{V[k]['rho'][0]:.2f}"); mac(f"varRho{K}Late", f"{V[k]['rho'][-1]:.2f}")
    hs = R["history_size"]
    mac("histSizeTenKappa", f"{100 * hs['10']['kappa']:.2f}"); mac("histSizeTenCov", f"{100 * hs['10']['coverage']:.1f}")
    mac("histSizeThirtyFiveKappa", f"{100 * hs['35']['kappa']:.2f}"); mac("histSizeThirtyFiveCov", f"{100 * hs['35']['coverage']:.1f}")
    ts_ = R["teaser"]
    mac("teaserHonestTenor", f"{ts_['honest']['t_tenor']:.1f}"); mac("teaserHonestWait", f"{ts_['honest']['t_wait']:.1f}")
    mac("teaserHonestLic", f"{ts_['honest']['t_lic']:.1f}")
    mac("teaserHackedFlag", f"{ts_['hacked']['t_flag']:.1f}" if ts_['hacked']['t_flag'] else "--")
    mac("teaserHackedTenor", f"{ts_['hacked']['t_tenor']:.1f}"); mac("teaserHackedProxy", f"{ts_['hacked']['t_proxy']:.1f}")
    mac("teaserHonestTruth", f"{100 * ts_['honest']['truth']:.1f}"); mac("teaserHackedTruth", f"{100 * ts_['hacked']['truth']:.1f}")
    body = "% Written by make_data.py: every number the text quotes.\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n"
    open(os.path.join(T, "numbers.tex"), "w").write(body)
    json.dump(macros, open(os.path.join(ROOT, "numbers.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
