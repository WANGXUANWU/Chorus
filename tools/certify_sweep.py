"""Certificates and path agreement of the precision sweep (amendment 21, E2).

For every sweep run, the saved coefficient path is checked level by level with the relative
duality gap of Chorus (relative to the null objective) at every visited level, and compared
with the reference path of the same workload (the run with the smallest largest gap): largest
relative difference of the test predictions over the visited levels, smallest support Jaccard
similarity, share of responses with the same selected level, and the change of the test R2 of
the selected fit. Usage: certify_sweep.py [TIMING_ROOT]
Writes results/v2/sweep_runs.csv (one row per run) and results/v2/sweep_certificates.csv
(one row per workload, method and tolerance: median time over repetitions, largest gap).
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
from certify_external import rel_gaps


def path_of(z, method):
    cp = z['coef_path']
    stop = z['s1_stopped_at'] if method == 'Chorus' else z['stopped_at']
    best = z['s1_best_k'][2] if method == 'Chorus' else z['best_k']
    return cp, np.asarray(stop), np.asarray(best)


def main(troot, outdir=os.path.join(ROOT, 'results', 'v2')):
    plan = {it['tag']: it for it in json.load(open(os.path.join(ROOT, 'configs', 'timing_plan3_sweep.json')))}
    rows, paths, data = [], {}, {}
    for line in open(os.path.join(troot, 'timing_log.jsonl')):
        r = json.loads(line)
        it = plan.get(r['tag'])
        if it is None or r['status'] != 'ok':
            continue
        d = os.path.join(troot, r['tag'], r['method'], f"rep{r['rep']}", 'output.npz')
        if not os.path.exists(d):
            continue
        wl = it['workload']
        if wl not in data:
            m = read_meta(it['case'])
            rr = it['options'].get('resp_range') or [0, int(m['q'])]
            X = np.asarray(load(it['case'], 'X', 'train', m))
            Y = np.asarray(load(it['case'], 'y', 'train', m))[:, rr[0]:rr[1]]
            Xt = np.asarray(load(it['case'], 'X', 'test', m))
            Yt = np.asarray(load(it['case'], 'y', 'test', m))[:, rr[0]:rr[1]]
            g = np.asarray(m['groups'])
            K = int(m.get('n_lambda', 30))
            f = np.geomspace(1, float(m.get('lambda_min_ratio', .02)), K)
            corr = X.T @ Y / X.shape[0]
            lmax = (np.sqrt(np.add.reduceat(corr ** 2, g[:, 0], axis=0)) / np.sqrt(g[:, 1] - g[:, 0])[:, None]).max(axis=0)
            data[wl] = (X, Y, Xt, Yt, g, f, lmax)
        X, Y, Xt, Yt, g, f, lmax = data[wl]
        z = np.load(d)
        cp, stop, best = path_of(z, r['method'])
        K = len(f)
        gaps = np.full((K, Y.shape[1]), np.nan)
        for k in range(K):
            vis = stop >= k
            if vis.any():
                gaps[k, vis] = rel_gaps(X, Y[:, vis], cp[k][:, vis], g, f[k] * lmax[vis])
        sel = np.array([gaps[best[j], j] for j in range(Y.shape[1])])
        rows.append({'workload': wl, 'method': r['method'], 'tol': it['tol'], 'rep': r['rep'], 'tag': r['tag'],
                     'fit_seconds': r['fit_seconds'], 'max_gap': float(np.nanmax(gaps)),
                     'median_gap': float(np.nanmedian(gaps)), 'max_gap_selected': float(sel.max())})
        if r['rep'] == 0:
            paths[(wl, r['tag'])] = (cp, stop, best)
        print(json.dumps(rows[-1]), flush=True)
    runs = pd.DataFrame(rows)
    # agreement with the reference path of each workload
    agree = []
    for wl, grp in runs[runs.rep == 0].groupby('workload'):
        ref_tag = grp.sort_values('max_gap').iloc[0].tag
        X, Y, Xt, Yt, g, f, lmax = data[wl]
        rcp, rstop, rbest = paths[(wl, ref_tag)]
        sst = ((Yt - Yt.mean(0)) ** 2).sum(0)
        r2_ref = 1 - ((Yt - Xt @ np.stack([rcp[rbest[j]][:, j] for j in range(Y.shape[1])], 1)) ** 2).sum(0) / sst
        ysd = np.sqrt(sst)                                 # test response spread, the scale of a prediction difference
        for tag in grp.tag:
            cp, stop, best = paths[(wl, tag)]
            dmax, same, tot = 0., 0, 0
            for k in range(1, len(f)):                     # level 0 is lambda_max, where every fit is zero
                vis = (stop >= k) & (rstop >= k)
                if not vis.any():
                    continue
                pa, pr = Xt @ cp[k][:, vis], Xt @ rcp[k][:, vis]
                dmax = max(dmax, float((np.linalg.norm(pa - pr, axis=0) / np.maximum(ysd[vis], 1e-12)).max()))
                sa = np.sqrt(np.add.reduceat(cp[k][:, vis] ** 2, g[:, 0], axis=0)) > 0
                sr = np.sqrt(np.add.reduceat(rcp[k][:, vis] ** 2, g[:, 0], axis=0)) > 0
                same += int((sa == sr).all(axis=0).sum())
                tot += int(vis.sum())
            bsel = np.stack([cp[best[j]][:, j] for j in range(Y.shape[1])], 1)
            rsel = np.stack([rcp[rbest[j]][:, j] for j in range(Y.shape[1])], 1)
            sa = np.sqrt(np.add.reduceat(bsel ** 2, g[:, 0], axis=0)) > 0
            sr = np.sqrt(np.add.reduceat(rsel ** 2, g[:, 0], axis=0)) > 0
            inter, union = (sa & sr).sum(0), (sa | sr).sum(0)
            jac = np.where(union > 0, inter / np.maximum(union, 1), 1.)
            r2 = 1 - ((Yt - Xt @ bsel) ** 2).sum(0) / sst
            agree.append({'tag': tag, 'reference': ref_tag, 'rel_pred_diff_max': dmax,
                          'share_same_support_levels': same / max(tot, 1), 'support_jaccard_selected_min': float(jac.min()),
                          'same_selected_level': float((best == rbest).mean()),
                          'test_r2_change_max': float(np.abs(r2 - r2_ref).max())})
    runs.to_csv(os.path.join(outdir, 'sweep_runs.csv'), index=False)
    agg = runs.groupby(['workload', 'method', 'tol', 'tag']).agg(
        median_seconds=('fit_seconds', 'median'), min_seconds=('fit_seconds', 'min'),
        max_seconds=('fit_seconds', 'max'), reps=('rep', 'count'), max_gap=('max_gap', 'max'),
        median_gap=('median_gap', 'median'), max_gap_selected=('max_gap_selected', 'max')).reset_index()
    agg = agg.merge(pd.DataFrame(agree), on='tag', how='left')
    agg.to_csv(os.path.join(outdir, 'sweep_certificates.csv'), index=False)
    print(agg.to_string())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'timing3'),
         *(sys.argv[2:3]))
