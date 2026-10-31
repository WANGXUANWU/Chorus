"""Per-response accuracy of the amendment 22 comparison on the reported simulated cases.

Chorus (<case>/Chorus_a22) is scored here; adaptive GL comes from results/v2/sim_adaptive_a22_per_response.csv
and the other baselines from the common-protocol rows of results/v2/sim_*_per_response.csv.
Writes results/v2/sim_a22_per_response.csv.
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
from collect_sim_a22 import KEEP

PR = {'formal': 'sim_formal_per_response.csv', 'robust': 'sim_robust_per_response.csv',
      'sharing': 'sim_sharing_per_response.csv'}
MET = ('r2', 'fee', 'f1', 'mcc', 'tpr', 'fpr', 'size', 'auprc')


def main():
    parts = []
    for fam, root, seeds in FAMILIES:
        old = pd.read_csv(os.path.join(ROOT, 'results', 'v2', PR[fam]))
        old = old[(old['protocol'] == 'aligned') & old['method'].isin(KEEP)].copy()
        old['method'] = old['method'].map(KEEP)
        old['family'] = fam
        parts.append(old[['family', 'case', 'method', 'response', *MET]])
        rows = []
        for name in sorted(old['case'].unique()):
            out = os.path.join(root, name, 'Chorus_a22')
            if not os.path.exists(os.path.join(out, 'output.npz')):
                continue
            m = metrics(os.path.join(root, name, 'case'), out)
            q = len(m['r2'])
            rows += [{'family': fam, 'case': name, 'method': 'Chorus', 'response': r, **{k: float(m[k][r]) for k in MET}}
                     for r in range(q)]
        parts.append(pd.DataFrame(rows))
    ada = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sim_adaptive_a22_per_response.csv'))
    ada['method'] = 'Adaptive GL'
    parts.append(ada[['family', 'case', 'method', 'response', *MET]])
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'sim_a22_per_response.csv'), index=False)
    print(df.groupby(['family', 'method']).size().unstack().to_string())


if __name__ == '__main__':
    main()
