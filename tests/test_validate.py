import numpy as np
import pytest

from tenor.validate import PastChange, huber_slope, sign_agreement, trial_r2, validate


def test_huber_slope_ignores_a_few_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.08, 40)
    y = 1.2 * x + rng.normal(0, 0.003, 40)
    y[:3] = -2 * x[:3]          # three paradoxical changes
    assert abs(huber_slope(x, y) - 1.2) < 0.05


def _changes(rng, n_honest=30, n_hacked=4):
    out = []
    for i in range(n_honest):
        dS = rng.normal(0, 0.08)
        out.append(PastChange(f"h{i}", dY=1.1 * dS + rng.normal(0, 0.005), dS=dS, dQ=dS + rng.normal(0, 0.004),
                              dW=0.85 * dS + rng.normal(0, 0.004)))
    for i in range(n_hacked):
        out.append(PastChange(f"x{i}", dY=-0.06, dS=-0.01, dQ=0.04, dW=-0.03))
    return out


def test_screen_flags_hacks_and_margin_bounds_honest_residuals():
    v = validate(_changes(np.random.default_rng(1)))
    assert not v.passed[-4:].any()          # every hack is flagged
    assert v.passed[:30].mean() > 0.9       # honest changes pass
    assert abs(v.gamma - 1.1) < 0.05
    assert v.kappa < 0.02


def test_error_budget_adds_up():
    v = validate(_changes(np.random.default_rng(2)))
    a = v.alphas(0.05)
    assert abs(sum(a.values()) - 0.05) < 1e-12
    assert a["kappa"] == pytest.approx(1 / (v.n_pass + 1))


def test_margin_covers_a_new_exchangeable_change():
    """Split-conformal guarantee: a new honest change's residual is within kappa with probability >= n / (n + 1)."""
    rng = np.random.default_rng(3)
    hits = []
    for _ in range(300):
        v = validate(_changes(rng, n_hacked=0))
        new = _changes(rng, n_honest=1, n_hacked=0)[0]
        hits.append(abs(new.dY - v.gamma * new.dS) <= v.kappa)
    assert np.mean(hits) >= 30 / 31 - 0.04


def test_agreement_metrics():
    assert sign_agreement([1, -1, 2], [3, -1, -2]) == pytest.approx(2 / 3)
    assert trial_r2([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
