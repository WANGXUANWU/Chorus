"""Per-response path agreement of every sweep run with the most precise run of its workload.

For each workload of the precision sweep, the reference is the run (repetition 0) with the
smallest largest certified gap. Every other run is compared with it response by response over the
levels both visit, level 0 excluded: share of levels with the same support, smallest Jaccard
similarity of the supports, and largest test-prediction difference relative to the spread of the
test response. Writes results/v2/sweep_agreement.csv (one row per run and response).
Usage: sweep_agreement.py [TIMING_ROOT]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta, load
from certify_sweep import path_of


def main(troot):
    plan = {it['tag']: it for it in json.load(open(os.path.join(ROOT, 'configs', 'timing_plan3_sweep.json')))}
    cert = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sweep_certificates.csv'))
    rows = []
    for wl, grp in cert.groupby('workload'):
        ref_tag = grp.sort_values('max_gap').iloc[0].tag
        it = plan[ref_tag]
        m = read_meta(it['case'])
        rr = it['options'].get('resp_range') or [0, int(m['q'])]
        Xt = np.asarray(load(it['case'], 'X', 'test', m))
        Yt = np.asarray(load(it['case'], 'y', 'test', m))[:, rr[0]:rr[1]]
        g = np.asarray(m['groups'])
        ysd = np.sqrt(((Yt - Yt.mean(0)) ** 2).sum(0))

        def paths(tag):
            r = grp[grp.tag == tag].iloc[0]
            z = np.load(os.path.join(troot, tag, r.method, 'rep0', 'output.npz'))
            return path_of(z, r.method)

        rcp, rstop, _ = paths(ref_tag)
        for _, r in grp.iterrows():
            cp, stop, _ = paths(r.tag)
            q = Yt.shape[1]
            same, tot = np.zeros(q), np.zeros(q)
            jac, dmax = np.ones(q), np.zeros(q)
            for k in range(1, len(cp)):
                vis = (stop >= k) & (rstop >= k)
                if not vis.any():
                    continue
                sa = np.sqrt(np.add.reduceat(cp[k][:, vis] ** 2, g[:, 0], axis=0)) > 0
                sr = np.sqrt(np.add.reduceat(rcp[k][:, vis] ** 2, g[:, 0], axis=0)) > 0
                same[vis] += (sa == sr).all(axis=0)
                tot[vis] += 1
                inter, union = (sa & sr).sum(0), (sa | sr).sum(0)
                jac[vis] = np.minimum(jac[vis], np.where(union > 0, inter / np.maximum(union, 1), 1.))
                d = np.linalg.norm(Xt @ (cp[k][:, vis] - rcp[k][:, vis]), axis=0) / np.maximum(ysd[vis], 1e-12)
                dmax[vis] = np.maximum(dmax[vis], d)
            for j in range(q):
                rows.append({'workload': wl, 'method': r.method, 'tol': r.tol, 'tag': r.tag, 'reference': ref_tag,
                             'response': j, 'levels': int(tot[j]), 'share_same_support': same[j] / max(tot[j], 1),
                             'jaccard_min': jac[j], 'pred_diff_max': dmax[j]})
        print(wl, ref_tag, flush=True)
    out = os.path.join(ROOT, 'results', 'v2', 'sweep_agreement.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out, len(rows))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'timing3'))
