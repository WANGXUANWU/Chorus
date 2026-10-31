import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pytest
from chorus import SplineDesign, InMemoryDesign, DiskDesign, write_disk_design, Chorus
from chorus.engine import fit_batch, group_norms


def make(n=300, p=40, q=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, p))
    f = [np.sin(np.pi * x[:, 0]), x[:, 1] ** 2 - 1 / 3, x[:, 2], np.exp(x[:, 3]) - np.sinh(1)]
    y = np.stack([sum(f[k] * rng.normal(1, .3) for k in range(4) if rng.random() < .8) + .5 * rng.normal(size=n)
                  for _ in range(q)], axis=1)
    sd = SplineDesign().fit(x)
    X = sd.transform(x)
    return X, y - y.mean(0), sd.groups


def objective(X, y, b, groups, lam):
    w = np.sqrt(groups[:, 1] - groups[:, 0])
    pen = sum(wj * np.linalg.norm(b[a:c]) for (a, c), wj in zip(groups, w))
    return ((y - X @ b) ** 2).sum() / (2 * len(y)) + lam * pen


def test_matches_skglm():
    from skglm import GroupLasso
    X, y, groups = make()
    sizes = groups[:, 1] - groups[:, 0]
    d = InMemoryDesign(X, groups)
    pen = np.repeat(np.sqrt(sizes)[:, None], y.shape[1], 1)
    res = fit_batch(d, y, np.array([1., .5, .2, .1]), pen, tol=1e-9, keep_path=True)
    for r in range(y.shape[1]):
        lam = .1 * res['lam_max'][r]
        m = GroupLasso(groups=[int(s) for s in sizes], weights=np.sqrt(sizes), alpha=lam, fit_intercept=False,
                       tol=1e-12, max_iter=1000, max_epochs=100000).fit(X, y[:, r])
        o1 = objective(X, y[:, r], res['path'][-1][:, r], groups, lam)
        o2 = objective(X, y[:, r], m.coef_, groups, lam)
        assert abs(o1 - o2) <= 1e-8 * max(1, abs(o2)), (o1, o2)


def test_batch_equals_single_and_disk(tmp_path):
    X, y, groups = make(seed=1)
    sizes = groups[:, 1] - groups[:, 0]
    pen = np.repeat(np.sqrt(sizes)[:, None], y.shape[1], 1)
    f = np.geomspace(1, .05, 8)
    d = InMemoryDesign(X, groups)
    rb = fit_batch(d, y, f, pen, tol=1e-10, keep_path=True)
    for r in range(y.shape[1]):
        rs = fit_batch(d, y[:, [r]], f, pen[:, [r]], tol=1e-10, keep_path=True)
        assert np.allclose(rb['path'][-1][:, r], rs['path'][-1][:, 0], atol=1e-6)
    path = str(tmp_path / 'x.f64')
    write_disk_design(path, len(X), groups, lambda mm: mm.__setitem__(slice(None), X), dtype='float64')
    dd = DiskDesign(path, chunk_bytes=1 << 16)
    rd = fit_batch(dd, y, f, pen, tol=1e-10, keep_path=True)
    assert np.allclose(rb['path'][-1], rd['path'][-1], atol=1e-8)
    assert rd['read_bytes'] > 0


def test_chorus_fit_runs():
    X, y, groups = make(n=400, seed=2)
    Xtr, Xva = X[:300], X[300:]
    ytr, yva = y[:300] - y[:300].mean(0), y[300:] - y[:300].mean(0)
    m = Chorus(n_lambda=15, alphas=(0, .5, 1), gammas=(1,)).fit(
        InMemoryDesign(Xtr - Xtr.mean(0), groups), ytr, InMemoryDesign(Xva - Xtr.mean(0), groups), yva)
    assert m.stats_['max_gap'] <= 1e-6
    assert m.coef_.shape == (X.shape[1], y.shape[1])


def test_modes_agree():
    X, y, groups = make(n=200, p=60, seed=3)
    sizes = groups[:, 1] - groups[:, 0]
    pen = np.repeat(np.sqrt(sizes)[:, None], y.shape[1], 1)
    f = np.geomspace(1, .02, 10)
    d = InMemoryDesign(X, groups)
    a = fit_batch(d, y, f, pen, tol=1e-10, keep_path=True, mode='gram')
    b = fit_batch(d, y, f, pen, tol=1e-10, keep_path=True, mode='residual')
    assert np.allclose(a['path'][-1], b['path'][-1], atol=1e-6)
    assert a['gaps'].max() <= 1e-10 and b['gaps'].max() <= 1e-10


def test_patience_keeps_selection():
    X, y, groups = make(n=400, p=50, q=8, seed=4)
    Xtr, Xva, ytr, yva = X[:300], X[300:], y[:300], y[300:]
    sizes = groups[:, 1] - groups[:, 0]
    pen = np.repeat(np.sqrt(sizes)[:, None], y.shape[1], 1)
    f = np.geomspace(1, .02, 20)
    d, dv = InMemoryDesign(Xtr - Xtr.mean(0), groups), InMemoryDesign(Xva - Xtr.mean(0), groups)
    a = fit_batch(d, ytr - ytr.mean(0), f, pen, tol=1e-8, val=(dv, yva - ytr.mean(0)))
    b = fit_batch(d, ytr - ytr.mean(0), f, pen, tol=1e-8, val=(dv, yva - ytr.mean(0)), patience=5)
    same = (a['best_k'] == b['best_k'])
    assert same.mean() >= .75
    assert np.allclose(a['best_coef'][:, same], b['best_coef'][:, same], atol=1e-6)
