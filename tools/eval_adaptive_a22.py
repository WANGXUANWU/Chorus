"""Common-protocol metrics of adaptive GL (adelie, amendment 22) on the reported simulated cases.

For every response the basis size (d = 5 or 3) with the smaller validation error at its selected
level is chosen, as for every other method. Writes results/v2/sim_adaptive_a22.csv (per case) and
results/v2/sim_adaptive_a22_per_response.csv.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from bench.evaluate import metrics
from rerun_a22 import FAMILIES


def main():
    rows, per = [], []
    for fam, root, seeds in FAMILIES:
        for name in sorted(os.listdir(root)):
            if 'seed' not in name or int(name.rsplit('seed', 1)[1]) not in seeds:
                continue
            outs = [os.path.join(root, name, o) for o in ('adaptive', 'adaptive_d3')]
            if not all(os.path.exists(os.path.join(o, 'output.npz')) for o in outs):
                continue
            z5, z3 = (np.load(os.path.join(o, 'output.npz')) for o in outs)
            v5, v3 = np.nanmin(z5['val_path'], axis=0), np.nanmin(z3['val_path'], axis=0)
            use3 = v3 < v5
            m5 = metrics(os.path.join(root, name, 'case'), outs[0])
            m3 = metrics(os.path.join(root, name, 'case_d3'), outs[1])
            m = {k: np.where(use3, m3[k], m5[k]) for k in m5}
            rows.append({'family': fam, 'case': name, **{k: float(np.nanmean(v)) for k, v in m.items()},
                         'share_d3': float(use3.mean())})
            q = len(use3)
            per += [{'family': fam, 'case': name, 'response': r, **{k: float(v[r]) for k, v in m.items()}} for r in range(q)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'sim_adaptive_a22.csv'), index=False)
    pd.DataFrame(per).to_csv(os.path.join(ROOT, 'results', 'v2', 'sim_adaptive_a22_per_response.csv'), index=False)
    print(df.groupby('family')[['f1', 'mcc', 'fee', 'r2', 'size', 'share_d3']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
