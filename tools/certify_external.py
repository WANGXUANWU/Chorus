"""Post hoc certificates for every solver of the group lasso estimator (Table 2).

For each timing run of skglm, Celer, adelie and Chorus on an in-memory case, the selected fit
of every response is checked against the group lasso objective at its own level with the
duality gap that Chorus uses (relative to the null objective). Output: results/v2/certificates.csv
with the largest relative gap per run.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta, load


def rel_gaps(X, Y, B, groups, lam):
    """Relative duality gaps of coefficients B (D x q) at levels lam (q,)."""
    n = X.shape[0]
    w = np.sqrt(groups[:, 1] - groups[:, 0]).astype(float)
    U = Y - X @ B
    G = X.T @ U / n
    gn = np.sqrt(np.add.reduceat(G * G, groups[:, 0], axis=0))
    bn = np.sqrt(np.add.reduceat(B * B, groups[:, 0], axis=0))
    s = np.maximum(1., (gn / (lam[None, :] * w[:, None])).max(axis=0))
    rss = (U * U).sum(0) / n
    primal = .5 * rss + lam * (w[:, None] * bn).sum(0)
    dual = (Y * U).sum(0) / (s * n) - rss / (2 * s * s)
    yy = (Y * Y).sum(0) / n
    return np.maximum(primal - dual, 0.) / np.maximum(.5 * yy, 1e-300)


def run_dirs():
    for root, exps in ((os.path.join(ROOT, 'results', 'timing'), ('q_scaling', 'real')),
                       (os.path.join(ROOT, 'results', 'timing2'), ('e1_gl',))):
        f = os.path.join(root, 'timing_log.jsonl')
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get('experiment') not in exps or r.get('status') != 'ok':
                continue
            if r['method'] not in ('skglm', 'Celer', 'adelie', 'Chorus'):
                continue
            yield root, r


def main():
    rows = []
    plans = {}
    for p in ('timing_plan.json', 'timing_plan2.json', 'timing_plan2_adelie_tight.json'):
        f = os.path.join(ROOT, 'configs', p)
        if os.path.exists(f):
            for it in json.load(open(f)):
                plans[(it['tag'], it['method'])] = it
    cache = {}
    for root, r in run_dirs():
        it = plans.get((r['tag'], r['method']))
        if it is None:
            continue
        opts = it.get('options', {})
        if r['method'] == 'Chorus' and opts.get('stage2', True):
            continue                                   # only the group lasso estimator
        d = os.path.join(root, r['tag'], r['method'], f"rep{r.get('rep', 0)}")
        if not os.path.exists(os.path.join(d, 'output.npz')):
            continue
        case = it['case']
        if case not in cache:
            m = read_meta(case)
            cache[case] = (m, np.asarray(load(case, 'X', 'train', m)), np.asarray(load(case, 'y', 'train', m)))
        m, X, Y = cache[case]
        g = np.asarray(m['groups'])
        rr = opts.get('resp_range') or [0, Y.shape[1]]
        Yb = Y[:, rr[0]:rr[1]]
        z = np.load(os.path.join(d, 'output.npz'))
        B = z['coef']
        n = X.shape[0]
        K = int(m.get('n_lambda', 30))
        f = np.geomspace(1, float(m.get('lambda_min_ratio', .02)), K)
        corr = X.T @ Yb / n
        lmax = (np.sqrt(np.add.reduceat(corr ** 2, g[:, 0], axis=0)) / np.sqrt(g[:, 1] - g[:, 0])[:, None]).max(axis=0)
        if 'best_k' in z.files:
            k = z['best_k']
        elif 'val_path' in z.files:
            vp = z['val_path']
            k = np.array([int(np.flatnonzero(vp[:, j] <= np.nanmin(vp[:, j]) * (1 + 1e-12))[0]) for j in range(vp.shape[1])])
        else:
            k = z['s1_best_k'][2]
        lam = f[k] * lmax
        gaps = rel_gaps(X, Yb, B, g, lam)
        rows.append({'experiment': r['experiment'], 'tag': r['tag'], 'method': r['method'],
                     'variant': r.get('variant', r['method']), 'max_gap': float(gaps.max()), 'median_gap': float(np.median(gaps))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'v2', 'certificates.csv'), index=False)


if __name__ == '__main__':
    main()
