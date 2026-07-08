import numpy as np

from tenor.censoring import MaturationModel, rectifier_terms
from tenor.index import SIGNALS, SurrogateIndex, expected_calibration_error, outcome_r2
from tenor.simulate import Environment, Version, concat, simulate_runs

ENV = Environment("toy", base=0.5, rate=1000, landmark=11.0, horizon=30.0,
                  delay=("grading", dict(med=11.0, sd=0.28, per_rerun=0.22, per_comment=0.05)))


def _runs(n=40000, seed=0, mu=0.0):
    rng = np.random.default_rng(seed)
    return simulate_runs(rng, ENV, Version(mu=mu), n, 0.0)


def test_index_is_calibrated_and_informative():
    train, test = _runs(seed=0), _runs(seed=1)
    s = SurrogateIndex.fit(train, SIGNALS)(test)
    assert expected_calibration_error(s, test["y"]) < 0.02
    assert outcome_r2(test["y"], s) > 0.3


def test_index_on_no_signals_is_the_outcome_rate():
    runs = _runs(seed=2)
    s = SurrogateIndex.fit(runs, ())(runs)
    assert np.allclose(s, runs["y"].mean(), atol=1e-6)


def test_rectifier_is_unbiased_when_delays_depend_on_signals():
    """E[(M / pi)(Y - S)] = E[Y - S] for any S: here S is a deliberately poor constant."""
    hist, now = _runs(seed=3), _runs(seed=4, mu=0.3)
    model = MaturationModel.fit(hist, ENV.landmark)
    s = np.full(len(now["y"]), 0.2)
    pi = model(now)
    target = float(np.mean(now["y"] - s))
    weighted = rectifier_terms(now, s, pi, ENV.landmark).mean()
    unweighted = rectifier_terms(now, s, pi, ENV.landmark, weighted=False).mean()
    assert abs(weighted - target) < 0.01
    assert abs(unweighted - target) > abs(weighted - target)


def test_maturation_probabilities_are_valid():
    model = MaturationModel.fit(_runs(seed=5), ENV.landmark)
    assert np.all(model.table >= 0.02) and np.all(model.table <= 1.0)


def test_concat_keeps_fields():
    a, b = _runs(n=10, seed=6), _runs(n=5, seed=7)
    c = concat([a, b])
    assert len(c["y"]) == 15 and set(c) == set(a)
