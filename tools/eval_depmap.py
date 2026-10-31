"""Evaluate every method on the DepMap comparison case.

Per response: test R2, number of selected predictors (genes), whether a silver-standard
predictor (the response gene's own expression or a close paralog) is selected, and the rank
of the best silver predictor in the path-entry order. Writes a per-response CSV and recall@k.
Usage: eval_depmap.py CASE_DIR RESULTS_ROOT OUT_PREFIX
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta, load
from bench.evaluate import feature_level

METHODS = ['Chorus', 'grpreg', 'SAM', 'sparsegl', 'Celer', 'skglm', 'biglasso']


def load_any(mdir, meta):
    """Read a joint run (output.npz or R binaries) or a merged chunked run."""
    for d in (os.path.join(mdir, 'merged'), mdir):
        f = os.path.join(d, 'output.npz')
        if os.path.exists(f):
            z = np.load(f)
            out = {k: z[k] for k in z.files}
            if 'support' not in out:
                g = np.asarray(meta['groups'])
                out['support'] = np.sqrt(np.add.reduceat(out['coef'] ** 2, g[:, 0], axis=0)) > 0
            return out
    q, G, nte = meta['q'], len(meta['groups']), meta['n_test']
    rd = lambda f, k: np.fromfile(os.path.join(mdir, f), dtype=np.float64).reshape((q, k)).T
    return {'pred': rd('pred.f64', nte), 'entry': rd('entry.f64', G).astype(int), 'support': rd('support.f64', G) > 0}


def main(case, root, prefix):
    meta = read_meta(case)
    info = json.load(open(os.path.join(case, 'info.json')))
    silver = info['silver']
    yte = load(case, 'y', 'test', meta)
    rows, curves = [], {}
    K = 60
    for m in METHODS:
        mdir = os.path.join(root, m)
        if not os.path.exists(mdir):
            continue
        try:
            o = load_any(mdir, meta)
        except Exception as e:
            print('skip', m, e)
            continue
        pred = o['pred']
        r2 = 1 - ((yte - pred) ** 2).sum(0) / ((yte - yte.mean(0)) ** 2).sum(0)
        sel = feature_level(np.asarray(o['support']).astype(bool), meta, 'any')
        ent = feature_level(np.asarray(o['entry']).astype(np.int64), meta, 'min')
        ranks = np.full(meta['q'], np.inf)
        for r in range(meta['q']):
            # every response counts for R2 and size; silver metrics only where a silver gene exists
            row = {'method': m, 'response': info['responses'][r], 'r2': r2[r], 'size': int(sel[:, r].sum()),
                   'has_silver': bool(silver[r]), 'silver_selected': np.nan, 'silver_rank': np.nan}
            if silver[r]:
                score = -ent[:, r].astype(float) + 1e-9 * np.arange(ent.shape[0])[::-1]
                order = np.argsort(-score)
                pos = {f: i for i, f in enumerate(order)}
                ranks[r] = min(pos[f] for f in silver[r]) + 1
                row['silver_selected'] = float(any(sel[f, r] for f in silver[r]))
                row['silver_rank'] = ranks[r]
            rows.append(row)
        has = np.array([bool(s) for s in silver])
        curves[m] = [float(np.mean(ranks[has] <= k)) for k in range(1, K + 1)]
    df = pd.DataFrame(rows)
    df.to_csv(prefix + '_per_response.csv', index=False)
    json.dump(curves, open(prefix + '_recall.json', 'w'))
    print(df.groupby('method')[['r2', 'size', 'silver_selected', 'silver_rank']].median().round(3))
    print(df.groupby('method')[['r2', 'silver_selected']].mean().round(3))


if __name__ == '__main__':
    main(*sys.argv[1:4])
