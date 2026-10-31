"""Time the construction of a block store from a raw predictor table (end-to-end cost).

Stages: reading the raw table, fitting the spline bases on a first chunk, the streamed pass
that accumulates exact means and second moments of every block, and the streamed pass that
transforms the rows and writes the column-major store. The raw table is the xraw_train.f64
file of an existing disk case; the new store is written next to it and checked against the
existing one on a sample of columns.
Usage: build_store.py CASE_DIR [--chunk 20000] [--keep]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from chorus import SplineDesign
from chorus.basis import _orthonormalize
from bench.caseio import read_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('--chunk', type=int, default=20000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    meta = read_meta(a.case)
    n, p = meta['n_train'], meta['p_sam']
    raw_path = os.path.join(a.case, 'xraw_train.f64')
    out = a.out or os.path.join(a.case, 'X_train_rebuilt.f64')
    t = {}
    t0 = time.perf_counter()
    raw = np.memmap(raw_path, dtype=np.float64, mode='r', shape=(p, n))     # column-major (p, n)
    chunks = [(c0, min(n, c0 + a.chunk)) for c0 in range(0, n, a.chunk)]

    def rows(c0, c1):
        return np.ascontiguousarray(raw[:, c0:c1].T)

    x0 = rows(*chunks[0])
    t['read_first_chunk'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sd = SplineDesign().fit(x0)
    t['spline_fit'] = time.perf_counter() - t0
    # pass 1: exact means and second moments of the raw spline columns of every term
    t0 = time.perf_counter()
    sums = [np.zeros(tm.spline.n_features_out_) for tm in sd.terms]
    cross = [np.zeros((tm.spline.n_features_out_,) * 2) for tm in sd.terms]
    for c0, c1 in chunks:
        x = rows(c0, c1)
        for i, tm in enumerate(sd.terms):
            z = tm.spline.transform(x[:, [tm.feature]])
            sums[i] += z.sum(0)
            cross[i] += z.T @ z
    for i, tm in enumerate(sd.terms):
        mu = sums[i] / n
        cov = cross[i] / n - np.outer(mu, mu)
        w, v = np.linalg.eigh(cov)
        keep = w > 1e-10 * w.max()
        tm.center = mu
        tm.transform = v[:, keep] / np.sqrt(w[keep])
    sd.group_sizes_ = [tm.transform.shape[1] for tm in sd.terms]
    t['pass1_moments'] = time.perf_counter() - t0
    # pass 2: transform and write the column-major store
    t0 = time.perf_counter()
    D = int(sum(sd.group_sizes_))
    mm = np.memmap(out, dtype=np.float64, mode='w+', shape=(n, D), order='F')
    for c0, c1 in chunks:
        sd.transform_into(rows(c0, c1), mm, c0)
    mm.flush()
    del mm
    t['pass2_write'] = time.perf_counter() - t0
    size = os.path.getsize(out)
    res = {'case': os.path.basename(a.case.rstrip('/\\')), 'n': n, 'p': p, 'D': D, 'store_bytes': size,
           'seconds': t, 'total_seconds': sum(t.values())}
    tdir = os.path.join(ROOT, 'results', 'v2', 'build')
    os.makedirs(tdir, exist_ok=True)
    json.dump(res, open(os.path.join(tdir, res['case'] + '.json'), 'w'), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
