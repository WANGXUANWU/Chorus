"""Raw real-data metrics of the amendment 22 comparison (Table 3).

Chorus is the frozen configuration (results/real_a22/C/Chorus_a22) and adaptive GL the adelie two-stage
fit (results/real_a22/C/adaptive and adaptive_d3, basis chosen per response by validation); GL (adelie),
GMCP, SGL, SpAM and MTL keep their common-protocol rows of results/v2/real_metrics.csv and
real_per_response.csv. Writes results/v2/real_a22_metrics.csv and real_a22_per_response.csv.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta
from collect_real_v2 import read_run, score, summarize

CASES = ['depmap_c200'] + [f'depmap_c200_s{s}' for s in range(3101, 3106)] + ['beijing', 'beijing_o2']
A22 = os.path.join(ROOT, 'results', 'real_a22')
KEEP = {'GL (adelie)': 'GL', 'gMCP (grpreg)': 'GMCP', 'SGL (sparsegl)': 'SGL', 'SpAM (SAM)': 'SpAM',
        'MTL (biglasso)': 'MTL'}


def main():
    old = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'real_metrics.csv'))
    oldp = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'real_per_response.csv'))
    old = old[(old.protocol == 'aligned') & old.method.isin(KEEP)].copy()
    oldp = oldp[(oldp.protocol == 'aligned') & oldp.method.isin(KEEP)].copy()
    old['method'], oldp['method'] = old.method.map(KEEP), oldp.method.map(KEEP)
    rows, per = [old], [oldp]
    for case in CASES:
        c5, c3 = os.path.join(ROOT, 'results', 'cases', case), os.path.join(ROOT, 'results', 'cases', case + '_d3')
        meta = read_meta(c5)
        kind = 'depmap' if case.startswith('depmap') else 'beijing'
        info = json.load(open(os.path.join(c5, 'info.json'))) if kind == 'depmap' else None
        got = {}
        f = os.path.join(A22, case, 'Chorus_a22', 'output.npz')
        if os.path.exists(f):
            z = np.load(f)
            got['Chorus'] = score(c5, z['pred'], z['support'], z['entry'], kind, info)
        r5, r3 = read_run(os.path.join(A22, case, 'adaptive'), c5), read_run(os.path.join(A22, case, 'adaptive_d3'), c3)
        if r5 is not None and r3 is not None:
            use3 = r3['vmin'] < r5['vmin']
            pick = lambda k: np.where(use3[None, :], r3[k], r5[k])
            got['Adaptive GL'] = score(c5, pick('pred'), pick('support'), pick('entry'), kind, info)
        for m, s in got.items():
            rows.append(pd.DataFrame([summarize(case, m, 'aligned', s, kind, meta)]))
            per.append(pd.DataFrame([{'case': case, 'method': m, 'protocol': 'aligned', 'response': r,
                                      'r2': float(s['r2'][r]), 'size': int(s['size'][r]),
                                      'silver_rank': float(s['silver_rank'][r]) if 'silver_rank' in s else np.nan}
                                     for r in range(len(s['r2']))]))
    df, pr = pd.concat(rows, ignore_index=True), pd.concat(per, ignore_index=True)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'real_a22_metrics.csv'), index=False)
    pr.to_csv(os.path.join(ROOT, 'results', 'v2', 'real_a22_per_response.csv'), index=False)
    cols = [c for c in ('case', 'method', 'r2_mean', 'size_median', 'rec1', 'rec10') if c in df]
    print(df[cols].round(4).to_string())


if __name__ == '__main__':
    main()
