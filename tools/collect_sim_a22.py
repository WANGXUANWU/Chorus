"""Per-case accuracy of the amendment 22 comparison on the reported simulated cases.

Chorus is the frozen configuration (<case>/Chorus_a22), adaptive GL the adelie two-stage fit
(results/v2/sim_adaptive_a22.csv); GL (adelie), GMCP, SGL, SpAM and MTL keep their common-protocol
runs (results/v2/sim_*.csv, protocol 'aligned'). Writes results/v2/sim_a22.csv (one row per case and
method, with the family and the setting columns) and prints the family means.
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

KEEP = {'GL (adelie)': 'GL', 'gMCP (grpreg)': 'GMCP', 'SGL (sparsegl)': 'SGL', 'SpAM (SAM)': 'SpAM',
        'MTL (biglasso)': 'MTL'}
FILES = {'formal': 'sim_formal.csv', 'robust': 'sim_robust.csv', 'sharing': 'sim_sharing.csv'}


def main():
    rows = []
    for fam, root, seeds in FAMILIES:
        old = pd.read_csv(os.path.join(ROOT, 'results', 'v2', FILES[fam]))
        old = old[(old['protocol'] == 'aligned') & old['method'].isin(KEEP)]
        setting_cols = [c for c in old.columns if c not in ('method', 'protocol', 'r2', 'fee', 'f1', 'mcc', 'tpr', 'fpr',
                                                           'size', 'auprc', 'share_d3', 'share_gl', 'share_a0',
                                                           'share_mid', 'share_a1')]
        settings = old.drop_duplicates('case').set_index('case')[[c for c in setting_cols if c != 'case']]
        for _, r in old.iterrows():
            rows.append({'family': fam, 'case': r['case'], 'method': KEEP[r['method']],
                         **{k: r[k] for k in ('r2', 'fee', 'f1', 'mcc', 'tpr', 'fpr', 'size', 'auprc')}})
        for name in settings.index:
            out = os.path.join(root, name, 'Chorus_a22')
            if not os.path.exists(os.path.join(out, 'output.npz')):
                continue
            m = metrics(os.path.join(root, name, 'case'), out)
            rows.append({'family': fam, 'case': name, 'method': 'Chorus', **{k: float(np.nanmean(v)) for k, v in m.items()}})
    ada = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sim_adaptive_a22.csv'))
    for _, r in ada.iterrows():
        rows.append({'family': r['family'], 'case': r['case'], 'method': 'Adaptive GL',
                     **{k: r[k] for k in ('r2', 'fee', 'f1', 'mcc', 'tpr', 'fpr', 'size', 'auprc')}})
    df = pd.DataFrame(rows)
    allset = []
    for fam, _, _ in FAMILIES:
        old = pd.read_csv(os.path.join(ROOT, 'results', 'v2', FILES[fam]))
        keep = [c for c in ('case', 'n_train', 'p', 'q', 's', 'kappa', 'rho', 'snr', 'seed', 'noise',
                            'n_clusters', 'scale_range', 'snr_range') if c in old.columns]
        allset.append(old.drop_duplicates('case')[keep])
    df = df.merge(pd.concat(allset).drop_duplicates('case'), on='case', how='left')
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'sim_a22.csv'), index=False)
    order = ['Chorus', 'Adaptive GL', 'GMCP', 'SGL', 'SpAM', 'GL', 'MTL']
    with pd.option_context('display.width', 200):
        for fam, d in df.groupby('family'):
            print(fam, d['case'].nunique(), 'cases')
            print(d.groupby('method')[['f1', 'mcc', 'fee', 'r2', 'fpr', 'size', 'auprc']].mean().reindex(order).round(4).to_string())


if __name__ == '__main__':
    main()
