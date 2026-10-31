import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from chorus import InMemoryDesign, SplineDesign
from chorus.engine import fit_batch


def _case(n, p, q=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, p))
    y = np.stack([sum(np.sin(2 * x[:, j]) for j in rng.choice(p, 4, replace=False)) + .5 * rng.normal(size=n)
                  for _ in range(q)], axis=1)
    sd = SplineDesign().fit(x)
    return InMemoryDesign(sd.transform(x), sd.groups), y - y.mean(0), sd.groups


def test_modes_give_same_solutions():
    for n, p in ((600, 40), (120, 80)):
        d, y, groups = _case(n, p)
        pen = np.repeat(np.sqrt(groups[:, 1] - groups[:, 0])[:, None].astype(float), y.shape[1], 1)
        f = np.geomspace(1, .05, 10)
        out = {m: fit_batch(d, y, f, pen, tol=1e-10, keep_path=True, mode=m) for m in ('gram', 'residual', 'auto')}
        for m in ('residual', 'auto'):
            assert np.allclose(out['gram']['path'][-1], out[m]['path'][-1], atol=1e-6), m
        assert out['auto']['gaps'].max() <= 1e-10 * 10
