"""Data of the Figure 3 selection panel (amendment 22 (d)): one setting fixed in advance
(kappa = 0.5, rho = 0.7, SNR 1, seeds 1001-1005, 80 responses), every method under the common protocol
(basis size by the validation error at the selected level). For every method: the recall of each of the
six true predictors of a response, sorted from the most to the least often selected, and the share of
responses with at least k false positives (k = 1..20), plus the mean numbers of true and false positives.
Writes results/v2/fig3_selection.json.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from bench.caseio import read_meta
from bench.evaluate import load_output, feature_level

MAIN = os.path.join(ROOT, 'results', 'formal', 'acc')
SET = 'n_train400_n_val200_n_test1000_p200_q16_s6_kappa0.5_rho0.7_snr1.0_seed{}'
RUNS = {'Chorus': [('Chorus_a22', 'case')], 'Adaptive GL': [('adaptive', 'case'), ('adaptive_d3', 'case_d3')],
        'GMCP': [('grpreg', 'case'), ('grpreg_d3', 'case_d3')], 'SGL': [('sparsegl', 'case'), ('sparsegl_d3', 'case_d3')],
        'SpAM': [('SAM', 'case'), ('SAM_p5', 'case')], 'GL': [('adelie', 'case'), ('adelie_d3', 'case_d3')],
        'MTL': [('biglasso', 'case'), ('biglasso_d3', 'case_d3')]}
KMAX = 20


def val_min(out, q):
    if os.path.exists(os.path.join(out, 'output.npz')):
        z = np.load(os.path.join(out, 'output.npz'))
        if 'val_path' in z.files:
            return np.nanmin(z['val_path'], axis=0)
    f = os.path.join(out, 'val_path.f64')
    K = os.path.getsize(f) // 8 // q
    return np.nanmin(np.fromfile(f).reshape(q, K).T, axis=0)


def main():
    res = {}
    for m, runs in RUNS.items():
        hits, fps = [], []
        for seed in range(1001, 1006):
            d = os.path.join(MAIN, SET.format(seed))
            truth = np.load(os.path.join(d, 'case', 'truth.npz'))['support'].astype(bool)
            sels, vals = [], []
            for run, sub in runs:
                meta = read_meta(os.path.join(d, sub))
                o = load_output(os.path.join(d, run), meta)
                sels.append(feature_level(np.asarray(o['support']).astype(bool), meta, 'any'))
                vals.append(val_min(os.path.join(d, run), truth.shape[1]) if len(runs) > 1 else np.zeros(truth.shape[1]))
            pick = np.argmin(np.stack(vals), axis=0)
            sel = np.stack(sels)[pick, :, np.arange(truth.shape[1])].T          # (p, q)
            for r in range(truth.shape[1]):
                t = np.flatnonzero(truth[:, r])
                hits.append(sel[t, r])
                fps.append(int((sel[:, r] & ~truth[:, r]).sum()))
        hits = np.asarray(hits, float)
        fps = np.asarray(fps)
        res[m] = {'recall_sorted': sorted(hits.mean(axis=0).tolist(), reverse=True),
                  'fp_at_least': [float((fps >= k).mean()) for k in range(1, KMAX + 1)],
                  'tp_mean': float(hits.sum(axis=1).mean()), 'fp_mean': float(fps.mean()), 'n': int(len(fps)),
                  'fp_counts': fps.tolist()}             # per response, for the survival curve of Figure 3 v10
        print(m, round(res[m]['tp_mean'], 2), round(res[m]['fp_mean'], 2))
    json.dump(res, open(os.path.join(ROOT, 'results', 'v2', 'fig3_selection.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
