"""Materialize a large simulated case whose training design lives on disk.

Training rows are generated in chunks. Spline knots come from the first chunk; the
centering and orthonormalizing transforms are then computed exactly from all training rows
in a streamed first pass, and a second pass writes the column-major float64 design. The
same file serves Chorus (memory map) and biglasso (file-backed big.matrix descriptor), so
both read identical bytes. Validation and test rows are small and stored as usual.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from chorus import SplineDesign
from data.simulate import ar1_uniform, component, SHAPES
from bench.caseio import _write


def make_disk_case(root, n_train, p=500, q=64, s=6, kappa=.5, rho=.5, snr=2., seed=0,
                   n_val=5000, n_test=10000, chunk=20000, n_lambda=30, lambda_min_ratio=0.02):
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(SHAPES)
    n_core = int(round(kappa * s))
    core = rng.choice(p, n_core, replace=False)
    rest = np.setdiff1d(np.arange(p), core)
    support = np.zeros((p, q), bool)
    spec = []
    for r in range(q):
        feats = np.concatenate([core, rng.choice(rest, s - n_core, replace=False)])
        support[feats, r] = True
        for j in feats:
            spec.append((r, int(j), names[rng.integers(len(names))],
                         float(rng.uniform(.5, 1.5) * rng.choice([-1, 1]))))

    def gen(n_rows, seed_rows):
        g = np.random.default_rng(seed_rows)
        x = ar1_uniform(g, n_rows, p, rho)
        mean = np.zeros((n_rows, q))
        for r, j, shape, amp in spec:
            mean[:, r] += amp * component(shape, x[:, j])
        return x, mean, g

    chunks = [(c0, min(n_train, c0 + chunk)) for c0 in range(0, n_train, chunk)]
    x0, m0, _ = gen(chunks[0][1] - chunks[0][0], seed * 1000 + 1)
    sig = m0.std(axis=0)
    noise_sd = sig / np.sqrt(snr)
    sd = SplineDesign().fit(x0)
    groups = sd.groups
    D = int(groups[-1, 1])
    raw_terms = [(t.spline, t.feature) for t in sd.terms]
    # pass 1: exact means and per-group second moments of the raw spline columns
    sums = [np.zeros(t.spline.n_features_out_) for t in sd.terms]
    cross = [np.zeros((t.spline.n_features_out_,) * 2) for t in sd.terms]
    for k, (c0, c1) in enumerate(chunks):
        x, _, _ = gen(c1 - c0, seed * 1000 + 1 + k)
        for i, (sp, f) in enumerate(raw_terms):
            z = sp.transform(x[:, [f]])
            sums[i] += z.sum(0)
            cross[i] += z.T @ z
    for i, t in enumerate(sd.terms):
        mu = sums[i] / n_train
        cov = cross[i] / n_train - np.outer(mu, mu)
        w, v = np.linalg.eigh(cov)
        keep = w > 1e-10 * w.max()
        t.center = mu
        t.transform = v[:, keep] / np.sqrt(w[keep])
    sd.group_sizes_ = [t.transform.shape[1] for t in sd.terms]
    groups = sd.groups
    D = int(groups[-1, 1])
    # pass 2: write the design and the responses
    path = os.path.join(root, 'X_train.f64')
    mm = np.memmap(path, dtype=np.float64, mode='w+', shape=(n_train, D), order='F')
    ytr = np.empty((n_train, q))
    for k, (c0, c1) in enumerate(chunks):
        x, mean, g = gen(c1 - c0, seed * 1000 + 1 + k)
        sd.transform_into(x, mm, c0)
        ytr[c0:c1] = mean + g.standard_normal((c1 - c0, q)) * noise_sd
    mm.flush()
    del mm
    ymean = ytr.mean(0)
    _write(os.path.join(root, 'y_train.f64'), ytr - ymean)
    xv, mv, gv = gen(n_val + n_test, seed * 1000 + 999)
    yv = mv + gv.standard_normal(mv.shape) * noise_sd
    for name, sl in (('val', slice(0, n_val)), ('test', slice(n_val, n_val + n_test))):
        _write(os.path.join(root, f'X_{name}.f64'), sd.transform(xv[sl]))
        _write(os.path.join(root, f'y_{name}.f64'), yv[sl] - ymean)
        _write(os.path.join(root, f'xraw_{name}.f64'), xv[sl])
    meta = {'groups': groups.tolist(), 'D': D, 'q': q, 'p_raw': p, 'p_sam': p,
            'term_features': sd.term_features.tolist(), 'term_kinds': ['additive'] * len(sd.terms),
            'n_train': n_train, 'n_val': n_val, 'n_test': n_test, 'n_lambda': n_lambda,
            'lambda_min_ratio': lambda_min_ratio, 'kind': 'disk_simulation', 'disk': True,
            'params': dict(n_train=n_train, p=p, q=q, s=s, kappa=kappa, rho=rho, snr=snr, seed=seed),
            'bytes_train_design': n_train * D * 8}
    with open(os.path.join(root, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(root, 'X_train.f64.json'), 'w') as fh:
        json.dump({'n': n_train, 'D': D, 'dtype': 'float64', 'groups': groups.tolist()}, fh)
    # biglasso descriptor over the same bytes
    desc = ("new('big.matrix.descriptor', description = list(sharedType = 'FileBacked', "
            f"filename = 'X_train.f64', dirname = '{os.path.abspath(root).replace(os.sep, '/')}/', "
            f"totalRows = {n_train}L, totalCols = {D}L, rowOffset = c(0, {n_train}), colOffset = c(0, {D}), "
            f"nrow = {n_train}, ncol = {D}, rowNames = NULL, colNames = NULL, type = 'double', separated = FALSE))")
    with open(os.path.join(root, 'X_train_big.desc'), 'w') as fh:
        fh.write(desc + '\n')
    np.savez_compressed(os.path.join(root, 'truth.npz'), support=support,
                        mean_test_centered=mv[n_val:] - ymean)
    return meta


if __name__ == '__main__':
    print(make_disk_case(sys.argv[1], **json.loads(sys.argv[2]))['D'])
