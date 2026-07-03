import numpy as np
import pytest

from tenor.confseq import asymptotic_cs, betting_cs, betting_cs_range, bounds_at


def test_betting_cs_covers_at_every_time():
    """Anytime coverage: the true mean stays inside the sequence at every n in at least 1 - alpha of runs."""
    rng = np.random.default_rng(0)
    alpha, misses = 0.1, 0
    for _ in range(200):
        x = (rng.random(800) < 0.3).astype(float)
        lo, hi = betting_cs(x, alpha, t_star=300)
        misses += not np.all((lo <= 0.3) & (0.3 <= hi))
    assert misses / 200 <= alpha


def test_betting_cs_bounds_never_widen():
    rng = np.random.default_rng(1)
    lo, hi = betting_cs(rng.random(500), 0.05, t_star=200)
    assert np.all(np.diff(lo) >= -1e-12) and np.all(np.diff(hi) <= 1e-12)
    assert np.all(lo <= hi)


def test_betting_cs_rejects_values_outside_unit_interval():
    with pytest.raises(ValueError):
        betting_cs(np.array([0.2, 1.5]), 0.05, t_star=10)


def test_betting_cs_range_maps_back():
    rng = np.random.default_rng(2)
    d = rng.uniform(-1, 1, 2000) * 0.5 + 0.1
    lo, hi = betting_cs_range(np.clip(d, -1, 1), 0.05, -1, 1, t_star=1000)
    assert lo[-1] < 0.1 < hi[-1]
    assert hi[-1] - lo[-1] < 0.1


def test_asymptotic_cs_is_silent_before_burn_in_and_covers_after():
    rng = np.random.default_rng(3)
    misses = 0
    for _ in range(200):
        z = rng.normal(0.5, 2.0, 3000)
        lo, hi = asymptotic_cs(z, 0.05, t_star=1000, burn_in=200)
        assert np.isinf(lo[:199]).all() and np.isinf(hi[:199]).all()
        misses += not np.all((lo[199:] <= 0.5) & (0.5 <= hi[199:]))
    assert misses / 200 <= 0.08


def test_bounds_at_counts():
    lo, hi = np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.8, 0.7])
    low, high = bounds_at(lo, hi, np.array([0, 1, 3]), default=(-1.0, 1.0))
    assert low.tolist() == [-1.0, 0.1, 0.3] and high.tolist() == [1.0, 0.9, 0.7]
