"""Accuracy runs on the real-data cases with every method, then merge response chunks.

Per-response methods (skglm, Celer, grpreg, sparsegl, SAM) are split into response chunks
that run in parallel; Chorus and biglasso share information across responses and always run
on all responses at once. Usage: run_real.py [--after-pid PID] [--workers 5] CASE...
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import psutil
from bench.runner import run
from bench.caseio import read_meta
from bench.evaluate import load_output

PER_RESPONSE = ['grpreg', 'skglm', 'Celer', 'SAM', 'sparsegl']
JOINT = ['Chorus', 'biglasso']


def merge(case, mdir, chunks, meta):
    """Concatenate chunk outputs along responses into mdir/merged (npz)."""
    parts = []
    for (r0, r1) in chunks:
        sub = dict(meta)
        sub['q'] = r1 - r0
        parts.append(load_output(os.path.join(mdir, f'chunk_{r0}_{r1}'), sub))
    out = {k: np.concatenate([p[k] for p in parts], axis=1) for k in ('pred', 'entry', 'support')}
    if all(p.get('coef') is not None for p in parts):
        out['coef'] = np.concatenate([p['coef'] for p in parts], axis=1)
    else:
        out['coef'] = None
    secs = sum(json.load(open(os.path.join(mdir, f'chunk_{a}_{b}', 'run.json'))).get('fit_seconds', 0) or 0
               for a, b in chunks)
    os.makedirs(os.path.join(mdir, 'merged'), exist_ok=True)
    np.savez_compressed(os.path.join(mdir, 'merged', 'output.npz'),
                        **{k: v for k, v in out.items() if v is not None},
                        **({} if out['coef'] is not None else {'support_only': np.array(1)}))
    json.dump({'run_status': 'ok', 'fit_seconds_sum': secs, 'chunks': chunks},
              open(os.path.join(mdir, 'merged', 'run.json'), 'w'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='+')
    ap.add_argument('--after-pid', type=int)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=40)
    a = ap.parse_args()
    if a.after_pid:
        while psutil.pid_exists(a.after_pid):
            time.sleep(60)
    jobs, merges = [], []
    for case in a.cases:
        meta = read_meta(case)
        q = meta['q']
        root = os.path.join(ROOT, 'results', 'real', os.path.basename(case.rstrip('/')))
        for m in JOINT:
            opts = {'save_variants': True} if m == 'Chorus' else {}
            jobs.append((case, m, os.path.join(root, m), opts))
        for m in PER_RESPONSE:
            chunks = [(r0, min(q, r0 + a.chunk)) for r0 in range(0, q, a.chunk)]
            for r0, r1 in chunks:
                jobs.append((case, m, os.path.join(root, m, f'chunk_{r0}_{r1}'), {'resp_range': [r0, r1]}))
            merges.append((case, os.path.join(root, m), chunks, meta))
    # longest jobs first
    order = {'grpreg': 0, 'skglm': 1, 'Celer': 2, 'Chorus': 3, 'biglasso': 3, 'SAM': 4, 'sparsegl': 5}
    jobs.sort(key=lambda j: order[j[1]])

    def work(j):
        case, m, out, opts = j
        if os.path.exists(os.path.join(out, 'run.json')):
            return
        r = run(case, m, out, opts, cap_seconds=6 * 3600)
        print(json.dumps({'case': case, 'method': m, 'out': out, 'status': r['run_status'],
                          'fit_seconds': r.get('fit_seconds')}), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    for case, mdir, chunks, meta in merges:
        try:
            merge(case, mdir, chunks, meta)
            print('merged', mdir, flush=True)
        except Exception as e:
            print('merge failed', mdir, repr(e), flush=True)


if __name__ == '__main__':
    main()
