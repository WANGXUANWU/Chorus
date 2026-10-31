"""Build simulated cases on a parameter grid and run all methods on each (parallel).

Usage: run_sim_grid.py OUT_ROOT GRID_JSON [--workers 5] [--methods ...] [--opts JSON]
GRID_JSON maps parameter names to lists; 'seed' lists the seeds. Accuracy runs may run in
parallel; timing comparisons use tools/run_timing.py, which runs one process at a time.
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from bench.make_sim_case import make_sim_case
from bench.runner import run
from bench.evaluate import metrics

METHODS = ['Chorus', 'skglm', 'Celer', 'grpreg', 'sparsegl', 'SAM', 'biglasso']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('grid')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--methods', nargs='*', default=METHODS)
    ap.add_argument('--chorus_opts', default='{"save_variants": true}')
    ap.add_argument('--n_lambda', type=int, default=30)
    ap.add_argument('--ratio', type=float, default=0.02)
    a = ap.parse_args()
    grid = json.loads(open(a.grid[1:]).read() if a.grid.startswith('@') else a.grid)
    # a grid is a dict of lists (full product) or a list of such dicts (union of products)
    combos = []
    for g in (grid if isinstance(grid, list) else [grid]):
        ks = list(g)
        combos += [dict(zip(ks, v)) for v in itertools.product(*[g[k] for k in ks])]
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for c in combos:
        name = '_'.join(f"{k}{'-'.join(map(str, c[k])) if isinstance(c[k], list) else c[k]}" for k in c)
        cdir = os.path.join(a.out, name, 'case')
        if not os.path.exists(os.path.join(cdir, 'meta.json')):
            make_sim_case(cdir, n_lambda=a.n_lambda, lambda_min_ratio=a.ratio, **c)
        for m in a.methods:
            jobs.append((name, cdir, m, c.get('seed', 0)))
    # seed-major order: every setting completes for a seed before the next seed starts
    jobs.sort(key=lambda j: j[3])
    jobs = [j[:3] for j in jobs]

    def work(job):
        name, cdir, m = job
        out = os.path.join(a.out, name, m)
        if os.path.exists(os.path.join(out, 'run.json')):
            with open(os.path.join(out, 'run.json')) as fh:
                r = json.load(fh)
        else:
            opts = json.loads(a.chorus_opts) if m == 'Chorus' else {}
            r = run(cdir, m, out, opts, cap_seconds=4 * 3600)
        row = {'case': name, 'method': m, 'status': r.get('run_status'),
               'fit_seconds': r.get('fit_seconds'), 'wall_seconds': r.get('wall_seconds'),
               'peak_rss': r.get('peak_rss')}
        if r.get('run_status') == 'ok':
            row.update({k: float(np.nanmean(v)) for k, v in metrics(cdir, out).items()})
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(work, jobs))
    with open(os.path.join(a.out, 'summary.jsonl'), 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
