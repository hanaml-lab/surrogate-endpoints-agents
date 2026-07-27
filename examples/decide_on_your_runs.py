"""Apply TENOR's gate to your own runs.

You need, for past changes whose outcomes have matured, the runs of both versions with their six signals and
outcomes; and for the release you are deciding, the runs so far with their signals, arrival times and, where
matured, outcomes and delays. Each run is a dict of equal-length arrays with these keys:

    judge     the judge's score at submission, in (0, 1)
    tests     1 if the visible tests or validation rules passed
    ccv       a calibrated completion probability, in (0, 1)
    comments  reviewer change requests within 24 hours
    rollback  1 if the change was rolled back within 48 hours
    perturb   pass rate of the artifact under perturbed re-checks, in [0, 1]
    y         the outcome (only needed where it has matured; NaN elsewhere is fine for the release)
    delay     days from the run to its outcome (np.inf if not yet matured)

This script fabricates such data with the simulator so that it runs as is; replace ``load_*`` with your own.
"""
from __future__ import annotations

import numpy as np

from tenor import (
    CONTROLLED,
    SCREEN_INDEPENDENT,
    SIGNALS,
    MaturationModel,
    PastChange,
    ReleaseGate,
    Streams,
    SurrogateIndex,
    rectifier_terms,
    validate,
)
from tenor.simulate import Environment, concat, make_history, simulate_runs

ENV = Environment("my agent", base=0.55, rate=4000, landmark=14.0, horizon=21.0, delay=("fixed", {"days": 14.0}))


def load_history():
    """Past changes: pairs (incumbent runs, candidate runs), outcomes all matured."""
    versions, difficulty = make_history(ENV, "HHXHHHHHHHHHHHHHHHHHHHHH", seed=1)
    rng = np.random.default_rng(1)
    return [(simulate_runs(rng, ENV, versions[c - 1], 5000, difficulty[c]),
             simulate_runs(rng, ENV, versions[c], 5000, difficulty[c])) for c in range(1, 25)]


def load_release():
    """The release being decided: incumbent and candidate runs with arrival times (days since release)."""
    versions, difficulty = make_history(ENV, "HHXHHHHHHHHHHHHHHHHHHHHHH", seed=1)
    rng = np.random.default_rng(2)
    n = int(ENV.rate * 5)
    A = simulate_runs(rng, ENV, versions[24], n, difficulty[25])
    B = simulate_runs(rng, ENV, versions[25], n, difficulty[25])
    return A, B, np.sort(rng.uniform(0, 5, n)), np.sort(rng.uniform(0, 5, n))


def main():
    history = load_history()
    runs = concat([r for pair in history for r in pair])
    index, q_index, w_index = (SurrogateIndex.fit(runs, s) for s in (SIGNALS, CONTROLLED, SCREEN_INDEPENDENT))
    maturation = MaturationModel.fit(runs, ENV.landmark)
    # For brevity the index is not cross-fitted here; the report fits it on the other half of each change's runs.
    past = [PastChange(f"change {i}", float(b["y"].mean() - a["y"].mean()), float(index(b).mean() - index(a).mean()),
                       float(q_index(b).mean() - q_index(a).mean()), float(w_index(b).mean() - w_index(a).mean()))
            for i, (a, b) in enumerate(history, start=1)]
    v = validate(past)
    A, B, tA, tB = load_release()
    k = min(len(tA), len(tB))
    paired = np.maximum(tA[:k], tB[:k])
    sA, sB = index(A), index(B)
    def screen(r):
        return v.gamma_q * q_index(r) - v.gamma_w * w_index(r)

    streams = Streams(d_index=sB[:k] - sA[:k], t_index=paired + 2.0, d_div=screen(B)[:k] - screen(A)[:k], t_div=paired + 1.0,
                      e_rect=rectifier_terms(B, sB, maturation(B), ENV.landmark)[:k]
                      - rectifier_terms(A, sA, maturation(A), ENV.landmark)[:k],
                      t_rect=paired + ENV.landmark)
    times = np.arange(1, 5 * 24 + 1) / 24
    trace = ReleaseGate(v, t_star_index=ENV.rate * 0.75, t_star_rect=ENV.rate * 3).run(streams, times)
    i = -1
    print(f"slope {v.gamma:.2f}, margin {100 * v.kappa:.2f} pts, tolerance {100 * v.tau:.2f} pts")
    print(f"after {times[i]:.1f} days: effect in [{100 * trace.lo[i]:+.1f}, {100 * trace.hi[i]:+.1f}] points; "
          f"licensed: {bool(trace.licensed[i])}, flagged: {bool(trace.flagged[i])}")
    if trace.decision == 0:
        print("no decision yet: keep collecting runs")
    else:
        print(f"{'ship' if trace.decision == 1 else 'block'} at day {trace.decided_at:.1f}")


if __name__ == "__main__":
    main()
