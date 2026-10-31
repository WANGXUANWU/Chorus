"""Invariance of the selection under the two stage-1 strategies (amendment 21).

Runs the worker twice per case, once with strategy 'working_set' (the frozen engine) and once
with strategy 'stats' (full sufficient statistics), with the options of the recorded Chorus2
run, and compares stage-1 selected levels, picks, supports, coefficients and test predictions.
Cases: the 36 tuning cases (seeds 101-103), Beijing, and the first 16 DepMap dependencies.
Usage: check_strategy_invariance.py [--workers 3]
Writes results/v2/invariance/<case>/<strategy>/ and results/v2/strategy_invariance.csv.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
WORKER = os.path.join(ROOT, 'src', 'bench', 'py_worker.py')
OUT = os.path.join(ROOT, 'results', 'v2', 'invariance')


def cases():
    out = []
    for cdir in sorted(glob.glob(os.path.join(ROOT, 'results', 'pilot', 'acc', '*'))):
        rj = os.path.join(cdir, 'Chorus2', 'run.json')
        if os.path.exists(rj):
            out.append((os.path.basename(cdir), os.path.join(cdir, 'case'), json.load(open(rj))['options']))
    for name, extra in (('beijing', {}), ('depmap_c200', {'resp_range': [0, 16]})):
        rj = os.path.join(ROOT, 'results', 'real2', name, 'Chorus2', 'run.json')
        opts = json.load(open(rj))['options']
        out.append((name, os.path.join(ROOT, 'results', 'cases', name), {**opts, **extra}))
    return out


def run(job):
    name, case, opts, strategy = job
    out = os.path.join(OUT, name, strategy)
    if os.path.exists(os.path.join(out, 'output.npz')):
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    o = dict(opts, strategy=strategy)
    o.pop('budget', None)
    r = subprocess.run([PY, WORKER, case, 'Chorus', out, json.dumps(o)], env=env, capture_output=True, text=True)
    print(json.dumps({'case': name, 'strategy': strategy, 'rc': r.returncode,
                      'tail': (r.stdout.strip().splitlines() or [''])[-1][:120]}), flush=True)


def compare(name, case):
    a = np.load(os.path.join(OUT, name, 'working_set', 'output.npz'))
    b = np.load(os.path.join(OUT, name, 'stats', 'output.npz'))
    ra = json.load(open(os.path.join(OUT, name, 'working_set', 'result.json')))
    rb = json.load(open(os.path.join(OUT, name, 'stats', 'result.json')))
    from bench.caseio import read_meta, load
    meta = read_meta(case)
    yte = load(case, 'y', 'test', meta)
    q = a['pick'].shape[0]
    yte = yte[:, :q] if yte.shape[1] >= q else yte
    sst = ((yte - yte.mean(0)) ** 2).sum(0)
    r2a = 1 - ((yte - a['pred']) ** 2).sum(0) / sst
    r2b = 1 - ((yte - b['pred']) ** 2).sum(0) / sst
    return {'case': name, 'q': q,
            'strategy_stats_used': rb['chorus'].get('strategy_stats', 0),
            'same_s1_level': float((a['s1_best_k'] == b['s1_best_k']).mean()),
            'same_pick': float((a['pick'] == b['pick']).mean()),
            'same_support': float((a['support'] == b['support']).all(axis=0).mean()),
            'max_abs_coef': float(np.abs(a['coef'] - b['coef']).max()),
            'max_abs_r2': float(np.abs(r2a - r2b).max()),
            'mean_r2_ws': float(r2a.mean()), 'mean_r2_stats': float(r2b.mean()),
            'max_gap_ws': ra['chorus']['max_gap'], 'max_gap_stats': rb['chorus']['max_gap'],
            'fit_ws': ra['fit_seconds'], 'fit_stats': rb['fit_seconds']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    cs = cases()
    jobs = [(n, c, o, s) for n, c, o in cs for s in ('working_set', 'stats')]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    import pandas as pd
    rows = [compare(n, c) for n, c, _ in cs
            if all(os.path.exists(os.path.join(OUT, n, s, 'output.npz')) for s in ('working_set', 'stats'))]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'strategy_invariance.csv'), index=False)
    print(df[['case', 'same_s1_level', 'same_pick', 'same_support', 'max_abs_coef', 'max_abs_r2']].to_string())
    print('responses:', int(df.q.sum()), 'identical picks:', int(round((df.same_pick * df.q).sum())),
          'identical supports:', int(round((df.same_support * df.q).sum())))


if __name__ == '__main__':
    main()
