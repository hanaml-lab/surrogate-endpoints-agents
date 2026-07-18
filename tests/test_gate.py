import numpy as np

from tenor.gate import BLOCK, SHIP, WAIT, ReleaseGate, Streams, combine, decide
from tenor.validate import PastChange, validate


def test_decide_ships_blocks_and_waits():
    assert decide(np.array([-0.05, -0.01]), np.array([0.05, 0.03]), 0.02) == (1, SHIP)
    assert decide(np.array([-0.05, -0.04]), np.array([0.05, -0.001]), 0.02) == (1, BLOCK)
    assert decide(np.array([-0.05]), np.array([0.05]), 0.02) == (-1, WAIT)


def test_combine_intersects_when_licensed_and_falls_back_when_flagged():
    S = (np.array([0.04]), np.array([0.06]))
    e = (np.array([-0.5]), np.array([0.5]))
    lo, hi, use, flag = combine(S, e, (np.array([-0.01]), np.array([0.01])), gamma=1.0, kappa=0.01, tau=0.03)
    assert use[0] and not flag[0] and np.isclose(lo[0], 0.03) and np.isclose(hi[0], 0.07)
    lo, hi, use, flag = combine(S, e, (np.array([0.05]), np.array([0.08])), gamma=1.0, kappa=0.01, tau=0.03)
    assert not use[0] and flag[0] and np.isclose(lo[0], -0.46)


def test_combine_falsifies_transport_that_disagrees_with_outcomes():
    S = (np.array([0.04, 0.04]), np.array([0.06, 0.06]))
    e = (np.array([-0.5, -0.20]), np.array([0.5, -0.12]))       # outcomes arrive and contradict the index
    lo, hi, use, flag = combine(S, e, (np.array([-0.01, -0.01]), np.array([0.01, 0.01])), 1.0, 0.01, 0.03)
    assert use[0] and not use[1] and flag[1]
    assert hi[1] < 0


def _validation():
    """Thirty honest past changes whose controlled and independent signals disagree by a point or two."""
    rng = np.random.default_rng(0)
    ch = []
    for i in range(30):
        dS = rng.normal(0, 0.08)
        ch.append(PastChange(f"c{i}", 1.1 * dS + rng.normal(0, 0.004), dS, dS + rng.normal(0, 0.012),
                             0.85 * dS + rng.normal(0, 0.010)))
    return validate(ch)


def test_gate_ships_a_clear_improvement_before_any_outcome():
    rng = np.random.default_rng(1)
    k = 6000
    t = np.sort(rng.uniform(0, 3, k))
    sB, sA = rng.beta(4, 3, k), rng.beta(3, 4, k)
    streams = Streams(d_index=sB - sA, t_index=t + 2.0, d_div=rng.normal(0, 0.35, k), t_div=t + 1.0,
                      e_rect=rng.normal(0, 0.5, k), t_rect=t + 14.0)
    trace = ReleaseGate(_validation(), t_star_index=2000, t_star_rect=6000).run(streams, np.arange(1, 24 * 6) / 24)
    assert trace.decision == SHIP and trace.decided_at < 4
