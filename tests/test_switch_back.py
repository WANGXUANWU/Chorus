import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from chorus import InMemoryDesign, SplineDesign
from chorus.engine import fit_batch, WorkingSet
from chorus.validation import make_folds


def make_wide(n=80, nv=60, p=300, q=6, seed=3):
    """Short and wide design (n well below D) with correlated predictors and weak signals."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n + nv, 1))
    x = np.tanh(.6 * z + rng.standard_normal((n + nv, p)))
    y = np.zeros((n + nv, q))
    for r in range(q):
        for j in rng.choice(p, 8, replace=False):
            y[:, r] += rng.normal(.6, .2) * np.sin(2 * x[:, j] + rng.uniform(0, 3))
    y += rng.normal(size=y.shape)
    sd = SplineDesign(n_knots=2).fit(x[:n])
    X = sd.transform(x)
    y = y - y[:n].mean(0)
    return X[:n], y[:n], X[n:], y[n:], sd.groups


def test_to_residual_matches_columns():
    X, y, Xv, yv, groups = make_wide()
    d = InMemoryDesign(X, groups)
    ws = WorkingSet(d, 2, 'gram')
    ws.extend([3, 7, 11])
    ws.to_residual()
    assert ws.mode == 'residual'
    assert np.allclose(ws.xwt, d.cols([3, 7, 11]).T)


def test_switch_back_keeps_the_certified_solutions():
    X, y, Xv, yv, groups = make_wide()
    sizes = groups[:, 1] - groups[:, 0]
    pen = np.repeat(np.sqrt(sizes)[:, None], y.shape[1], 1).astype(float)
    f = np.geomspace(1, .02, 30)
    fold, cluster = make_folds(len(yv), 0)
    d, dv = InMemoryDesign(X, groups), InMemoryDesign(Xv, groups)
    kw = dict(tol=1e-10, val=(dv, yv), patience=None, val_fold=fold, val_cluster=cluster, mode='auto')
    a = fit_batch(d, y, f, pen, switch_back=False, **kw)
    b = fit_batch(d, y, f, pen, switch_back=True, **kw)
    assert (a['best_k'] == b['best_k']).all()
    assert (a['entry'] == b['entry']).all()
    sa = np.sqrt(np.add.reduceat(a['best_coef'] ** 2, groups[:, 0], axis=0)) > 0
    sb = np.sqrt(np.add.reduceat(b['best_coef'] ** 2, groups[:, 0], axis=0)) > 0
    assert (sa == sb).all()
    assert np.abs(a['best_coef'] - b['best_coef']).max() <= 1e-6 * max(1., np.abs(a['best_coef']).max())
    assert b['max_gap'] <= 1e-10 if 'max_gap' in b else True
