"""Real-data accuracy runs for the revised protocol (amendment 14).

For every case name C (with sibling C_d3) this runs Chorus with both bases and saved
candidates, and every comparison method on the d=5 and d=3 designs (SAM with 3 and 5 basis
functions), saving validation residuals. Per-response methods run in response chunks in
parallel and are merged; Chorus and biglasso always run on all responses. Results go to
results/real2/C/<run name>.
Usage: run_real2.py CASE [CASE ...] [--workers 5] [--chorus-only] [--skip M ...] [--max-chorus 3]
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
from bench.runner import run
from bench.caseio import read_meta
from run_real import merge

CASES = os.path.join(ROOT, 'results', 'cases')
VP = {'save_val_pred': True}
PER_RESPONSE = {'grpreg', 'skglm', 'Celer', 'SAM', 'sparsegl', 'adelie'}
COST = {'grpreg': 0, 'Chorus': 1, 'skglm': 2, 'Celer': 3, 'SAM': 4, 'biglasso': 5, 'sparsegl': 6, 'adelie': 7}


def merge_val(mdir, chunks):
    """Concatenate validation paths and residuals of chunk runs into merged/val.npz."""
    vps, vrs = [], []
    for r0, r1 in chunks:
        d = os.path.join(mdir, f'chunk_{r0}_{r1}')
        res = json.load(open(os.path.join(d, 'result.json')))
        if os.path.exists(os.path.join(d, 'output.npz')):
            z = np.load(os.path.join(d, 'output.npz'))
            vps.append(z['val_path'])
            vrs.append(z['val_resid'] if 'val_resid' in z.files else None)
        else:
            q, K, nv = res['q'], res['K'], res['n_val']
            vps.append(np.fromfile(os.path.join(d, 'val_path.f64')).reshape(q, K).T)
            f = os.path.join(d, 'val_resid.f32')
            vrs.append(np.fromfile(f, dtype=np.float32).reshape(q, K, nv).transpose(1, 2, 0)
                       if os.path.exists(f) else None)
    out = {'val_path': np.concatenate(vps, axis=1)}
    if all(v is not None for v in vrs):
        out['val_resid'] = np.concatenate(vrs, axis=2)
    np.savez_compressed(os.path.join(mdir, 'merged', 'val.npz'), **out)


def jobs_for(name, skip, chunk):
    c5, c3 = os.path.join(CASES, name), os.path.join(CASES, name + '_d3')
    meta = read_meta(c5)
    q = meta['q']
    root = os.path.join(ROOT, 'results', 'real2', name)
    chorus_opts = {'bases': ['../' + name + '_d3'], 'save_variants': True}
    if meta.get('kind') == 'beijing':
        chorus_opts['val_cluster'] = 'week'
    spec = [('Chorus2', 'Chorus', c5, chorus_opts)]
    for m in ('adelie', 'grpreg', 'skglm', 'Celer', 'sparsegl', 'biglasso'):
        o = {**VP, 'save_path': True} if m == 'adelie' else VP
        spec.append((m, m, c5, o))
        spec.append((m + '_d3', m, c3, o))
    spec.append(('SAM', 'SAM', c5, VP))
    spec.append(('SAM_p5', 'SAM', c5, {**VP, 'sam_p': 5}))
    jobs, merges = [], []
    for out, m, case, opts in spec:
        if out in skip:
            continue
        mdir = os.path.join(root, out)
        if m in PER_RESPONSE or (m == 'biglasso' and meta.get('kind') == 'depmap'):   # biglasso in blocks of 40 on DepMap, as before
            chunks = [(r0, min(q, r0 + chunk)) for r0 in range(0, q, chunk)]
            for r0, r1 in chunks:
                jobs.append((case, m, os.path.join(mdir, f'chunk_{r0}_{r1}'), {**opts, 'resp_range': [r0, r1]}))
            merges.append((case, mdir, chunks, read_meta(case)))
        else:
            jobs.append((case, m, mdir, opts))
    return jobs, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='+')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=40)
    ap.add_argument('--skip', nargs='*', default=[])
    ap.add_argument('--max-chorus', type=int, default=3)
    a = ap.parse_args()
    jobs, merges = [], []
    for name in a.cases:
        j, m = jobs_for(name, set(a.skip), a.chunk)
        jobs += j
        merges += m
    jobs.sort(key=lambda j: COST[j[1]])
    sem = threading.Semaphore(a.max_chorus)       # Chorus on DepMap holds about 3 GB

    def work(j):
        case, m, out, opts = j
        if os.path.exists(os.path.join(out, 'run.json')):
            return
        if m == 'Chorus':
            with sem:
                r = run(case, m, out, opts, cap_seconds=8 * 3600)
        else:
            r = run(case, m, out, opts, cap_seconds=8 * 3600)
        print(json.dumps({'out': out, 'status': r['run_status'], 'fit_seconds': r.get('fit_seconds'),
                          'error': r.get('error')}), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    for case, mdir, chunks, meta in merges:
        try:
            merge(case, mdir, chunks, meta)
            merge_val(mdir, chunks)
            print('merged', mdir, flush=True)
        except Exception as e:
            print('merge failed', mdir, repr(e), flush=True)


if __name__ == '__main__':
    main()
