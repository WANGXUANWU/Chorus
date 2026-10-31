"""Confirmation cases that played no part in any design decision.

DepMap: the 200 dependencies and 2,464 expression predictors of depmap_c200, resplit 60/20/20
within each lineage with seeds never used before (3101 to 3105); the spline basis is refit on
each new training split. Beijing: a second time origin (train to February 2014, validation to
February 2015, test to February 2016). Each case is written with five (d=5) and three (d=3)
basis functions per predictor.
Usage: make_confirm_cases.py [--seeds 3101 3102 3103 3104 3105] [--beijing]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from chorus import SplineDesign
from bench.caseio import read_meta, load, write_case

CASES = os.path.join(ROOT, 'results', 'cases')
KNOTS = {5: 4, 3: 2}


def depmap_resplit(seed):
    src = os.path.join(CASES, 'depmap_c200')
    meta = read_meta(src)
    info = json.load(open(os.path.join(src, 'info.json'), encoding='utf-8'))
    x = np.vstack([np.asarray(load(src, 'xraw', s, meta)) for s in ('train', 'val', 'test')])
    y = np.vstack([np.asarray(load(src, 'y', s, meta)) for s in ('train', 'val', 'test')])
    lineage = np.asarray(info['lineage'])
    rng = np.random.default_rng(seed)
    split = np.array(['train'] * len(lineage), dtype=object)
    for lin in sorted(set(lineage)):
        idx = np.flatnonzero(lineage == lin)
        rng.shuffle(idx)
        nv = nt = int(round(.2 * len(idx)))
        split[idx[:nv]] = 'val'
        split[idx[nv:nv + nt]] = 'test'
    order = np.concatenate([np.flatnonzero(split == k) for k in ('train', 'val', 'test')])
    x, y = x[order], y[order]
    n = {k: int((split == k).sum()) for k in ('train', 'val', 'test')}
    sl = {'train': slice(0, n['train']), 'val': slice(n['train'], n['train'] + n['val']),
          'test': slice(n['train'] + n['val'], len(order))}
    sub = {**info, 'lines': [info['lines'][i] for i in order], 'lineage': list(lineage[order])}
    for d, k in KNOTS.items():
        dst = os.path.join(CASES, f'depmap_c200_s{seed}' + ('' if d == 5 else '_d3'))
        if os.path.exists(os.path.join(dst, 'meta.json')):
            continue
        sd = SplineDesign(n_knots=k).fit(x[sl['train']])
        write_case(dst, sd, x, y, sl, extra={'n_lambda': 30, 'lambda_min_ratio': 0.02, 'kind': 'depmap',
                                             'split_seed': seed, 'basis_d': d,
                                             'response_index_in_all': meta.get('response_index_in_all')})
        with open(os.path.join(dst, 'info.json'), 'w', encoding='utf-8') as fh:
            json.dump(sub, fh)
        print(dst, n, flush=True)


def beijing_origin2():
    from data.beijing import make_case
    src = os.path.join(ROOT, 'data', 'processed', 'beijing_base.csv.gz')
    for d, k in KNOTS.items():
        dst = os.path.join(CASES, 'beijing_o2' + ('' if d == 5 else '_d3'))
        if os.path.exists(os.path.join(dst, 'meta.json')):
            continue
        m = make_case(dst, src, origin=2, n_knots=k)
        print(dst, {s: m[f'n_{s}'] for s in ('train', 'val', 'test')}, 'D', m['D'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='*', default=[3101, 3102, 3103, 3104, 3105])
    ap.add_argument('--beijing', action='store_true')
    a = ap.parse_args()
    for s in a.seeds:
        depmap_resplit(s)
    if a.beijing:
        beijing_origin2()
