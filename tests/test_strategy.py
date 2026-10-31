import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pytest
from chorus import Chorus, InMemoryDesign, DiskDesign, SplineDesign, write_disk_design
from chorus.engine import fit_batch
from chorus.gram_path import fit_batch_gram
from chorus.validation import make_folds
from chorus.ledger import Ledger
import chorus.model as cm


def make(n=400, nv=200, p=120, q=10, seed=0):
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
    return X[:n], y[:n], X[n:], y[n:], sd.groups


def test_gram_path_reproduces_engine_stage1():
    X, y, Xv, yv, groups = make()
    sizes = groups[:, 1] - groups[:, 0]
    pen = np.repeat(np.sqrt(sizes)[:, None], y.shape[1], 1).astype(float)
    f = np.geomspace(1, .02, 30)
    fold, cluster = make_folds(len(yv), 0)
    d, dv = InMemoryDesign(X, groups), InMemoryDesign(Xv, groups)
    # covariance updates (the engine's mode on disk); 'auto' switches modes by measured machine
    # constants, which can move the entry of a group tied at lambda_max by one level
    a = fit_batch(d, y, f, pen, tol=1e-10, val=(dv, yv), patience=5, val_fold=fold, val_cluster=cluster,
                  mode='gram')
    g, c, _ = d.stats_scan(y)
    b = fit_batch_gram(g, c, np.einsum('ij,ij->j', y, y) / len(y), sizes, f, pen, tol=1e-10,
                       val=(dv.x, yv), max_rounds=200, patience=5, val_fold=fold, val_cluster=cluster,
                       track_norms=True, ws_cap=True, tighten=True, chunked=True)
    assert np.array_equal(a['val']['best_k'], b['val']['best_k'])
    assert np.array_equal(a['entry'], b['entry'])
    assert np.array_equal(a['stopped_at'], b['stopped_at'])
    assert np.array_equal(a['best_norm'] > 0, b['best_norm'] > 0)
    assert np.allclose(a['best_coef'], b['best_coef'], atol=1e-6)
    assert np.allclose(a['val']['closs'], b['val']['closs'], rtol=1e-6, atol=1e-12)
    live = b['gaps'] > 0
    assert b['gaps'][live].max() <= 1e-10


def test_gaps_masked_after_exit():
    X, y, Xv, yv, groups = make(seed=1)
    sizes = groups[:, 1] - groups[:, 0]
    pen = np.repeat(np.sqrt(sizes)[:, None], y.shape[1], 1).astype(float)
    f = np.geomspace(1, .02, 20)
    d = InMemoryDesign(X, groups)
    g, c, _ = d.stats_scan(y)
    b = fit_batch_gram(g, c, np.einsum('ij,ij->j', y, y) / len(y), sizes, f, pen, tol=1e-8,
                       val=(Xv, yv), patience=1, track_norms=True)
    for r in range(y.shape[1]):
        assert np.all(b['gaps'][b['stopped_at'][r] + 1:, r] == 0)


def test_chorus_strategies_identical():
    X, y, Xv, yv, groups = make(p=80, q=8, seed=2)
    d, dv = InMemoryDesign(X, groups), InMemoryDesign(Xv, groups)
    a = Chorus(tol=1e-9, strategy='working_set').fit(d, y, dv, yv)
    b = Chorus(tol=1e-9, strategy='stats').fit(d, y, dv, yv)
    assert b.stats_['strategy_stats'] == 1 and a.stats_['strategy_stats'] == 0
    assert np.array_equal(a.pick_, b.pick_)
    assert np.array_equal(a.support_, b.support_)
    # entry orders agree except for ties at lambda_max (rounding of the two Gram products)
    diff = a.entry_ != b.entry_
    assert diff.mean() <= .01 and np.all(np.minimum(a.entry_, b.entry_)[diff] % len(a.factors) <= 1)
    assert np.allclose(a.phi_, b.phi_)
    assert np.allclose(a.coef_, b.coef_, atol=1e-6)
    assert b.stats_['stage2_read_bytes'] == 0
    assert b.stats_['max_gap'] <= 1e-9


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_disk_stats_scan_matches_memory(tmp_path, dtype):
    X, y, Xv, yv, groups = make(n=1001, p=30, q=4, seed=3)
    path = str(tmp_path / 'x.bin')
    write_disk_design(path, len(X), groups, lambda mm: mm.__setitem__(slice(None), X), dtype=dtype)
    Xs = X.astype(dtype).astype(np.float64)
    dd = DiskDesign(path, chunk_bytes=1 << 16)
    g, c, tm = dd.stats_scan(y, panel_bytes=97 * 8 * X.shape[1])
    assert np.allclose(g, Xs.T @ Xs / len(X), atol=1e-12)
    assert np.allclose(c, Xs.T @ y / len(X), atol=1e-12)
    assert g.flags.c_contiguous and np.array_equal(g, g.T)
    assert dd.read_bytes == len(X) * X.shape[1] * np.dtype(dtype).itemsize
    if dtype == 'float64':
        dv = InMemoryDesign(Xv, groups)
        a = Chorus(tol=1e-9, strategy='stats').fit(InMemoryDesign(X, groups), y, dv, yv)
        b = Chorus(tol=1e-9, strategy='stats').fit(DiskDesign(path), y, dv, yv)
        assert np.array_equal(a.pick_, b.pick_) and np.array_equal(a.support_, b.support_)
        assert b.stats_['passes'] == 1 and b.stats_['stage2_read_bytes'] == 0
    dd.close()


def test_planner_rules():
    led = Ledger.__new__(Ledger)
    led.M, led.delta, led.eps, led.batch_share, led.M0, led.retained = 2 << 30, 64 << 20, .1, .5, 400 << 20, 0
    assert led.plan_stats(400_000, 2500, 500, 64, 5000, 5000, disk=True) is not None
    assert led.plan_stats(50_000, 25_000, 5000, 64, 5000, 5000, disk=True) is None
    X, y, Xv, yv, groups = make(p=40, q=4, seed=4)
    for kw in ({'mode': 'gram'}, {'max_width': 500}, {'stage2_shared': False}, {'strong': False}):
        m = Chorus(**kw)
        m.ledger_, m._stats = None, {'pred_stats_seconds': 0., 'pred_ws_seconds': 0.}
        assert m._choose_strategy(InMemoryDesign(X, groups), y, yv, None)[0] == 'working_set'
    m = Chorus(stats_max_gram_bytes=100)
    m.ledger_, m._stats = None, {'pred_stats_seconds': 0., 'pred_ws_seconds': 0.}
    assert m._choose_strategy(InMemoryDesign(X, groups), y, yv, None)[0] == 'working_set'


def test_width_overflow_falls_back(monkeypatch):
    X, y, Xv, yv, groups = make(p=60, q=6, seed=5)
    d, dv = InMemoryDesign(X, groups), InMemoryDesign(Xv, groups)
    a = Chorus(tol=1e-9, strategy='working_set').fit(d, y, dv, yv)
    tiny = {'B': 6, 'gram_bytes': 8 * X.shape[1] ** 2, 'panel_bytes': 0, 'm_sub': 5}
    monkeypatch.setattr(cm.Chorus, '_choose_strategy', lambda self, *a_: ('stats', tiny))
    b = Chorus(tol=1e-9).fit(d, y, dv, yv)
    assert b.stats_['strategy_fallback'] == 1 and b.stats_['strategy_stats'] == 0
    assert np.array_equal(a.pick_, b.pick_)
    assert np.allclose(a.coef_, b.coef_, atol=1e-8)
