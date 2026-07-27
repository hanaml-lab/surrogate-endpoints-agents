"""CPU toy: fit TENOR on a simulated history of agent changes and replay six releases.

    python -m tenor.demo                # seconds on a laptop
    python -m tenor.demo --config configs/toy.yaml

The history has thirty past changes of a pull-request-like environment, a few of them reward-hacked. TENOR fits the
surrogate index, validates it on the past changes (slope, margin, divergence screen) and then decides six releases
hour by hour: two honest improvements, an honest regression, a hacked change, the change that removes the hack, and
a change whose effect runs through a pathway none of the signals measures. Waiting for the outcome and deciding on a
proxy (tests pass and the judge approves) run on the same simulated runs.
"""
from __future__ import annotations

import argparse
import time

import yaml

from .replay import fit_history, replay
from .simulate import Environment, make_history

DEFAULT = {
    "environment": {"name": "pull requests (toy)", "base": 0.54, "rate": 1500.0, "landmark": 14.0, "horizon": 28.0,
                    "delay": {"kind": "fixed", "days": 14.0}},
    "history": {"types": "HHXHHHHHPHHHHHHHHXHHHHHHHHHHHH", "runs_per_version": 6000, "seed": 7},
    "releases": {"types": "HHHXHHL", "seed": 11},
    "gate": {"delta": 0.02},
}

LABEL = {1: "ship", -1: "block", 0: "escalate"}


def load(path: str | None) -> dict:
    if path is None:
        return DEFAULT
    with open(path) as fh:
        return yaml.safe_load(fh)


def _fmt(trace, truth: float, delta: float) -> str:
    """A decision, the day it was made and whether it was acceptable. Undecided at the horizon: escalated."""
    dec = trace.decision
    if dec == 0:
        return f"{'escalate':<8} {'':>5}  {'':<5}"
    ok = (truth >= -delta) if dec == 1 else (truth <= 0)
    return f"{LABEL[dec]:<8} {trace.decided_at:4.1f}d  {'ok' if ok else 'WRONG':<5}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="YAML config (default: the built-in toy, configs/toy.yaml)")
    args = ap.parse_args(argv)
    cfg = load(args.config)
    t0 = time.time()
    e = cfg["environment"]
    delay = dict(e["delay"])
    kind = delay.pop("kind")
    env = Environment(e["name"], e["base"], e["rate"], e["landmark"], e["horizon"], (kind, delay))
    h, r = cfg["history"], cfg["releases"]
    types = h["types"] + r["types"]
    versions, difficulty = make_history(env, types, seed=h["seed"])
    n_hist = len(h["types"])
    fitted = fit_history(env, versions, difficulty, range(1, n_hist + 1), h["runs_per_version"], seed=h["seed"])
    v = fitted.validation
    print(f"TENOR toy: {env.name}, {n_hist} past changes, {env.rate:.0f} runs/day/version, landmark {env.landmark:.0f} days")
    print(f"slope gamma {v.gamma:.3f}: the index explains {100 * v.pte:.0f}% of an effect")
    print(f"margin kappa {100 * v.kappa:.2f} pts, screen tolerance tau {100 * v.tau:.2f} pts; "
          f"{v.n_pass} past changes pass the screen")
    flagged = [c.name.replace("change ", "") for c, ok in zip(fitted.changes, v.passed, strict=False) if not ok]
    print(f"past changes the screen flags: {', '.join(flagged) if flagged else 'none'}")
    delta = cfg["gate"]["delta"]
    print()
    print(f"{'release':<9}{'kind':<9}{'effect':>8}   {'TENOR':<22}{'wait for the outcome':<22}{'proxy only'}")
    wrong = {"tenor": 0, "wait": 0, "proxy": 0}
    for i in range(len(r["types"])):
        c = n_hist + 1 + i
        res = replay(env, versions, difficulty, c, fitted, seed=r["seed"] + i, delta=delta)
        cells = []
        for name in ("tenor", "wait", "proxy"):
            tr = getattr(res, name)
            s = _fmt(tr, res.truth, delta)
            wrong[name] += s.endswith("WRONG")
            cells.append(s)
        print(f"{c:<9}{versions[c].kind:<9}{100 * res.truth:>+7.1f}    {cells[0]:<22}{cells[1]:<22}{cells[2]}".rstrip())
    print()
    print(f"wrong decisions: TENOR {wrong['tenor']}, waiting {wrong['wait']}, proxy {wrong['proxy']} "
          f"(of {len(r['types'])}); {time.time() - t0:.1f}s on CPU")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
