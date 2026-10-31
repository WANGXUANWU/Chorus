import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import chorus.model as cm
from chorus import Chorus, InMemoryDesign, SplineDesign
from chorus.ledger import MemoryBudgetError


def make(n=400, nv=200, p=300, q=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n + nv, p))
    y = np.zeros((n + nv, q))
    for r in range(q):
        for j in rng.choice(p, 5, replace=False):
            y[:, r] += rng.normal(1, .3) * np.sin(2 * x[:, j] + rng.uniform(0, 3))
    y += .7 * rng.normal(size=y.shape)
    sd = SplineDesign().fit(x[:n])
    X = sd.transform(x)
    y = y - y[:n].mean(0)
    return InMemoryDesign(X[:n], sd.groups), y[:n], InMemoryDesign(X[n:], sd.groups), y[n:]


def test_forced_splits_match_unsplit(monkeypatch):
    calls = []
    orig = cm._split
    monkeypatch.setattr(cm, '_split', lambda c, s: calls.append(1) or orig(c, s))
    d, y, dv, yv = make()
    base = Chorus(tol=1e-9).fit(d, y, dv, yv)
    small = Chorus(tol=1e-9, max_width=1200).fit(d, y, dv, yv)   # union 1500 > 1200 > single response
    assert len(calls) > 0
    assert np.array_equal(base.support_, small.support_)
    assert np.array_equal(base.pick_, small.pick_)
    assert np.allclose(base.coef_, small.coef_, atol=1e-7)


def test_single_response_overflow_is_reported():
    d, y, dv, yv = make(q=3)
    try:
        Chorus(max_width=100, stage2=False).fit(d, y, dv, yv)
    except MemoryBudgetError:
        return
    raise AssertionError('expected MemoryBudgetError')


def test_stage2_without_shared_statistics_is_identical():
    d, y, dv, yv = make(p=100, q=8)
    a = Chorus(tol=1e-9).fit(d, y, dv, yv)
    b = Chorus(tol=1e-9, stage2_shared=False, batch_size=1).fit(d, y, dv, yv)
    assert np.array_equal(a.pick_, b.pick_)
    assert np.allclose(a.coef_, b.coef_, atol=1e-6)
