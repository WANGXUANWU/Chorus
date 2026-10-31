"""Per-response cost of validation-aware exit (amendment 21, reviewer issue 1).

For every case of the patience study and every patience value, each response is compared with the
full path of the same pipeline: change in test R2, and in simulation in F1 and FEE, plus the
relative change of the smallest stage-1 validation error. DepMap uses the 10-hour full-path run
(pNone_10h) when it exists. Writes results/v2/exit_losses.csv (one row per response and patience)
and prints the median, 5th percentile, worst value and share below -0.01 in test R2.
Usage: exit_losses.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from bench.evaluate import metrics

base = os.path.join(ROOT, 'results', 'v2', 'patience')
rows = []
for case in sorted(os.listdir(base)):
    full = None
    for name in ('pNone_10h', 'pNone'):
        d = os.path.join(base, case, name)
        if os.path.exists(os.path.join(d, 'output.npz')):
            full = d
            break
    if full is None:
        continue
    cdir = (os.path.join(ROOT, 'results', 'formal', 'acc', case, 'case') if case.startswith('n_train')
            else os.path.join(ROOT, 'results', 'cases', case))
    group = 'Simulation' if case.startswith('n_train') else ('Beijing' if case.startswith('beijing') else 'DepMap')
    mf = metrics(cdir, full)
    vf = np.nanmin(np.load(os.path.join(full, 'output.npz'))['s1_val_mse'], axis=0)
    for p in ('p1', 'p2', 'p3', 'p5', 'p8'):
        d = os.path.join(base, case, p)
        if not os.path.exists(os.path.join(d, 'output.npz')):
            continue
        m = metrics(cdir, d)
        vp = np.nanmin(np.load(os.path.join(d, 'output.npz'))['s1_val_mse'], axis=0)
        for r in range(len(m['r2'])):
            row = {'group': group, 'case': case, 'patience': int(p[1:]), 'response': r,
                   'd_r2': m['r2'][r] - mf['r2'][r], 'd_val': (vp[r] - vf[r]) / max(vf[r], 1e-300)}
            if 'fee' in m:
                row['d_f1'] = m['f1'][r] - mf['f1'][r]
                row['d_fee'] = (m['fee'][r] - mf['fee'][r]) / max(mf['fee'][r], 1e-300)
            rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, 'results', 'v2', 'exit_losses.csv'), index=False)
summ = df[df.patience == 5].groupby('group').agg(
    responses=('d_r2', 'size'), median=('d_r2', 'median'), p05=('d_r2', lambda v: np.percentile(v, 5)),
    worst=('d_r2', 'min'), share_below=('d_r2', lambda v: np.mean(v < -.01)), changed=('d_r2', lambda v: np.mean(v != 0)),
    val_median=('d_val', 'median'), val_p95=('d_val', lambda v: np.percentile(v, 95)))
print(summ.to_string())
if 'd_f1' in df:
    s = df[(df.patience == 5) & df.d_f1.notna()]
    print('simulation F1: median %.4f p05 %.4f worst %.4f | FEE rel: median %.4f p95 %.4f worst %.4f' % (
        s.d_f1.median(), np.percentile(s.d_f1, 5), s.d_f1.min(), s.d_fee.median(), np.percentile(s.d_fee, 95), s.d_fee.max()))
