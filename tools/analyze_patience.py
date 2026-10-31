"""Early termination against the full path (reviewer comment 6): results/v2/patience.csv.

For every case and patience value: fit time relative to the full path, share of responses whose
final selection differs (candidate or level), mean absolute difference in validation error of the
selected stage-1 fit, Jaccard similarity of the selected supports, and the change in mean test R2
(and FEE when the truth is known).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta, load
from bench.evaluate import metrics

base = os.path.join(ROOT, 'results', 'v2', 'patience')
rows = []
for case in sorted(os.listdir(base)):
    full = os.path.join(base, case, 'pNone')
    if not os.path.exists(os.path.join(full, 'output.npz')):
        continue
    cdir = (os.path.join(ROOT, 'results', 'formal', 'acc', case, 'case') if case.startswith('n_train')
            else os.path.join(ROOT, 'results', 'cases', case))
    zf = np.load(os.path.join(full, 'output.npz'))
    rf = json.load(open(os.path.join(full, 'run.json')))
    mf = metrics(cdir, full)
    for p in sorted(os.listdir(os.path.join(base, case))):
        d = os.path.join(base, case, p)
        if not os.path.exists(os.path.join(d, 'output.npz')):
            continue
        z = np.load(os.path.join(d, 'output.npz'))
        r = json.load(open(os.path.join(d, 'run.json')))
        m = metrics(cdir, d)
        same_pick = z['pick'] == zf['pick']
        same_level = np.all(z['s1_best_k'] == zf['s1_best_k'], axis=0)
        sa, sb = z['support'].astype(bool), zf['support'].astype(bool)
        jac = np.where((sa | sb).sum(0) > 0, (sa & sb).sum(0) / np.maximum((sa | sb).sum(0), 1), 1.)
        vf = np.nanmin(zf['s1_val_mse'], axis=0)
        vp = np.nanmin(z['s1_val_mse'], axis=0)
        row = {'case': case, 'patience': p[1:], 'time_ratio': r['fit_seconds'] / rf['fit_seconds'],
               'fit_seconds': r['fit_seconds'], 'changed_selection': float(1 - np.mean(same_pick & same_level)),
               'changed_level': float(1 - np.mean(same_level)), 'val_gap': float(np.mean((vp - vf) / np.maximum(vf, 1e-300))),
               'support_jaccard': float(np.mean(jac)), 'd_r2': float(np.mean(m['r2']) - np.mean(mf['r2']))}
        if 'fee' in m:
            row['d_fee'] = float(np.mean(m['fee']) - np.mean(mf['fee']))
            row['d_auprc'] = float(np.mean(m['auprc']) - np.mean(mf['auprc']))
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, 'results', 'v2', 'patience.csv'), index=False)
print(df.groupby('patience')[['time_ratio', 'changed_selection', 'changed_level', 'val_gap', 'support_jaccard', 'd_r2']].mean().round(4))
