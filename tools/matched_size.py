"""Matched-size comparison on the real workloads (amendment 22 (b)).

For every response and method, the compared model is the path model with the smallest validation
error among the models of both basis sizes (SpAM: three and five functions) whose size, in selected
predictors, does not exceed the size of the Chorus model of that response; its test R2 is compared
with the test R2 of Chorus. When the selected model of a basis is no larger than the Chorus model,
it is the best model of that basis under the cap, so GMCP needs paths only for the other responses.
  matched_size.py --lists FILE      write those GMCP responses (input of run_real_a22.py --grpreg-lists)
  matched_size.py                   compare; writes results/v2/matched_size.csv and *_per_response.csv
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta, load
from bench.evaluate import feature_level

CASES = ['depmap_c200'] + [f'depmap_c200_s{s}' for s in range(3101, 3106)] + ['beijing', 'beijing_o2']
LABEL = {**{c: f'D{i}' for i, c in enumerate(CASES[:6])}, 'beijing': 'B1', 'beijing_o2': 'B2'}
A22, R2D = os.path.join(ROOT, 'results', 'real_a22'), os.path.join(ROOT, 'results', 'real2')
R1D = os.path.join(ROOT, 'results', 'real')


def sel_dir(case, run):
    """Folder of a selected fit: the amendment 14 runs (real2) or, for the d = 5 runs of D0 and B1, the first runs."""
    d = os.path.join(R2D, case, run)
    return d if os.path.exists(os.path.join(d, 'merged')) else os.path.join(R1D, case, run)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


class Path:
    """Per-level test predictions P (K, nte, q), selected predictors S (K, q) and validation MSE
    V (K, q) of one method and basis; responses without a path keep NaN validation errors."""

    def __init__(self, q, K, nte):
        self.P = np.full((K, nte, q), np.nan)
        self.S = np.zeros((K, q), int)
        self.V = np.full((K, q), np.nan)

    def put(self, idx, P, A, V, meta):
        self.P[:, :, idx] = P
        self.S[:, idx] = np.stack([feature_level(a.astype(bool), meta).sum(0) for a in A])
        self.V[:, idx] = V
        return self


def load_py(d):
    z = np.load(os.path.join(d, 'output.npz'))
    return z['pred_path'].astype(float), z['active_path'], z['val_path']


def load_r(d, q, K, nte, G):
    P = np.fromfile(os.path.join(d, 'test_path.f32'), np.float32).reshape(q, K, nte).transpose(1, 2, 0)
    A = np.fromfile(os.path.join(d, 'active_path.u8'), np.uint8).reshape(q, K, G).transpose(1, 2, 0)
    V = np.fromfile(os.path.join(d, 'val_path.f64')).reshape(q, K).T
    return P.astype(float), A, V


def gather(d, meta, q, K, nte):
    """Path of one run folder: a single run, chunk_r0_r1 runs or list_i runs (resp_list)."""
    path, G = Path(q, K, nte), len(meta['groups'])
    if os.path.exists(os.path.join(d, 'output.npz')):
        return path.put(np.arange(q), *load_py(d), meta)
    if os.path.exists(os.path.join(d, 'test_path.f32')):
        return path.put(np.arange(q), *load_r(d, q, K, nte, G), meta)
    for c in sorted(glob.glob(os.path.join(d, 'chunk_*')) + glob.glob(os.path.join(d, 'list_*'))):
        res = json.load(open(os.path.join(c, 'result.json')))
        if res.get('status') != 'ok':
            continue
        if os.path.basename(c).startswith('chunk_'):
            r0, r1 = map(int, os.path.basename(c).split('_')[1:])
            idx = np.arange(r0, r1)
        else:
            idx = np.asarray(json.load(open(os.path.join(c, 'run.json')))['options']['resp_list'], int)
        data = load_py(c) if os.path.exists(os.path.join(c, 'output.npz')) else load_r(c, len(idx), K, nte, G)
        path.put(idx, *data, meta)
    return path


def sel_val(d, K):
    """Validation MSE path (K, q) of a selected run: merged/val.npz, or the chunks' val_path.f64."""
    f = os.path.join(d, 'merged', 'val.npz')
    if os.path.exists(f):
        return np.load(f)['val_path']
    parts = []
    for c in sorted(glob.glob(os.path.join(d, 'chunk_*')), key=lambda c: int(os.path.basename(c).split('_')[1])):
        v = np.fromfile(os.path.join(c, 'val_path.f64'))
        parts.append(v.reshape(-1, K).T)
    return np.concatenate(parts, axis=1)

def best_under_cap(path, cap):
    """Prediction (nte, q) and validation MSE (q,) of the best path model with size <= cap."""
    Vm = np.where((path.S <= cap[None, :]) & np.isfinite(path.V), path.V, np.inf)
    k = np.argmin(Vm, axis=0)
    r = np.arange(path.P.shape[2])
    return path.P[k, :, r].T, Vm[k, r]


def gmcp_basis(case, run, meta, q, K, nte, cap):
    """GMCP under the cap for one basis: full paths when present, otherwise the selected fit
    (best under the cap when no larger than the cap) and paths of the listed responses."""
    d = os.path.join(A22, case, run)
    full = sorted(glob.glob(os.path.join(d, 'chunk_*')))
    if full:
        return best_under_cap(gather(d, meta, q, K, nte), cap)
    z = np.load(os.path.join(sel_dir(case, run), 'merged', 'output.npz'))
    vp = sel_val(sel_dir(case, run), K)
    pred, val = z['pred'].astype(float).copy(), np.nanmin(vp, axis=0)
    size = feature_level(z['support'].astype(bool), meta).sum(0)
    over = size > cap
    if over.any():
        lp = gather(d, meta, q, K, nte)
        p2, v2 = best_under_cap(lp, cap)
        miss = over & ~np.isfinite(lp.V).any(axis=0)
        if miss.any():
            raise RuntimeError(f'{case} {run}: GMCP paths missing for {int(miss.sum())} responses')
        pred[:, over], val[over] = p2[:, over], v2[over]
    return pred, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chorus', default='Chorus_a22')
    ap.add_argument('--lists')
    ap.add_argument('--cases', nargs='*')
    a = ap.parse_args()
    rows, per, need = [], [], {}
    for case in (a.cases or CASES):
        c5 = os.path.join(ROOT, 'results', 'cases', case)
        meta, meta3 = read_meta(c5), read_meta(c5 + '_d3')
        yte = load(c5, 'y', 'test', meta)
        q, nte, K = meta['q'], meta['n_test'], int(meta.get('n_lambda', 30))
        fc = os.path.join(A22, case, a.chorus, 'output.npz')
        if a.lists and not os.path.exists(fc):
            continue                    # Chorus still running on this case; its list comes later
        zc = np.load(fc)
        cap = feature_level(zc['support'].astype(bool), meta).sum(0)
        r2_ch = r2(yte, zc['pred'])
        need[case] = {}
        for run, m_ in (('grpreg', meta), ('grpreg_d3', meta3)):
            z = np.load(os.path.join(sel_dir(case, run), 'merged', 'output.npz'))
            need[case][run] = [int(r) for r in np.flatnonzero(feature_level(z['support'].astype(bool), m_).sum(0) > cap)]
        if a.lists:
            continue
        spec = {'GL': [(os.path.join(R2D, case, 'adelieP'), meta), (os.path.join(R2D, case, 'adelieP_d3'), meta3)],
                'Adaptive GL': [(os.path.join(A22, case, 'adaptive'), meta), (os.path.join(A22, case, 'adaptive_d3'), meta3)],
                'SGL': [(os.path.join(A22, case, 'sparsegl'), meta), (os.path.join(A22, case, 'sparsegl_d3'), meta3)],
                'SpAM': [(os.path.join(A22, case, 'SAM'), meta), (os.path.join(A22, case, 'SAM_p5'), meta)],
                'MTL': [(os.path.join(A22, case, 'biglasso'), meta), (os.path.join(A22, case, 'biglasso_d3'), meta3)]}
        res = {}
        for lab, runs in spec.items():
            if not all(os.path.exists(d) for d, _ in runs):
                print('skip', case, lab, '(runs missing)')
                continue
            cands = [best_under_cap(gather(d, m_, q, K, nte), cap) for d, m_ in runs]
            res[lab] = cands
        res['GMCP'] = [gmcp_basis(case, 'grpreg', meta, q, K, nte, cap),
                       gmcp_basis(case, 'grpreg_d3', meta3, q, K, nte, cap)]
        for lab, cands in res.items():
            pick = np.argmin(np.stack([v for _, v in cands]), axis=0)
            pm = np.stack([p for p, _ in cands])[pick, :, np.arange(q)].T
            rr = r2(yte, pm)
            gain = r2_ch - rr
            rows.append({'case': LABEL[case], 'method': lab, 'r2_chorus': float(np.mean(r2_ch)),
                         'r2_matched': float(np.mean(rr)), 'gain': float(np.mean(gain)),
                         'share_chorus_better': float(np.mean(gain > 0)), 'n': q})
            per += [{'case': LABEL[case], 'method': lab, 'response': r, 'size_chorus': int(cap[r]),
                     'r2_chorus': float(r2_ch[r]), 'r2_matched': float(rr[r])} for r in range(q)]
    if a.lists:
        json.dump(need, open(a.lists, 'w'), indent=1)
        print({c: {k: len(v) for k, v in d.items()} for c, d in need.items()})
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'matched_size.csv'), index=False)
    pd.DataFrame(per).to_csv(os.path.join(ROOT, 'results', 'v2', 'matched_size_per_response.csv'), index=False)
    print(df.pivot(index='method', columns='case', values='r2_matched').round(4).to_string())
    print(df.pivot(index='method', columns='case', values='share_chorus_better').round(3).to_string())


if __name__ == '__main__':
    main()
