"""Parallel execution under the same total CPU (amendment 21, E9).

Each plan item is split into `workers` contiguous response ranges, and one fresh
single-threaded worker per range is started at the same time through the harness (each
process tree sampled for memory). The wall time runs from the first start to the last end;
the memory of the parallel execution is the sum of the per-process peaks.
Usage: run_parallel.py PLAN_JSON OUT_ROOT
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.runner import run
from bench.quiet import wait_quiet, Background

plan = json.load(open(sys.argv[1]))
out_root = sys.argv[2]
os.makedirs(out_root, exist_ok=True)
log = open(os.path.join(out_root, 'parallel_log.jsonl'), 'a')
max_reps = max(it.get('reps', 1) for it in plan)
for rep in range(max_reps):
    for it in plan:
        if rep >= it.get('reps', 1):
            continue
        base = os.path.join(out_root, it['tag'], it['method'], f'rep{rep}')
        if os.path.exists(os.path.join(base, 'done.json')):
            continue
        r0, r1 = it['options'].get('resp_range', [0, it['q']])
        w = it['workers']
        cuts = [r0 + (r1 - r0) * i // w for i in range(w + 1)]
        jobs = [(i, {**it['options'], 'resp_range': [cuts[i], cuts[i + 1]]}) for i in range(w) if cuts[i + 1] > cuts[i]]
        def one(job):
            i, opts = job
            return run(it['case'], it['method'], os.path.join(base, f'part{i}'), opts,
                       cap_seconds=it.get('cap_seconds', 3600))

        for attempt in range(5):                       # quiet machine, repeat when disturbed (amendment 21)
            wait_quiet(log=lambda s: print(s, flush=True))
            t0 = time.time()
            with Background() as bg, ThreadPoolExecutor(len(jobs)) as ex:
                res = list(ex.map(one, jobs))
            if bg.mean <= .3:
                break
        row = {k: it.get(k) for k in ('experiment', 'tag', 'method', 'variant', 'q', 'workers')}
        row.update({'rep': rep, 'wall_seconds': time.time() - t0,
                    'status': 'ok' if all(r['run_status'] == 'ok' for r in res) else
                              ','.join(sorted({r['run_status'] for r in res})),
                    'max_fit_seconds': max(r.get('fit_seconds') or 0 for r in res),
                    'sum_peak_wset': sum(r.get('peak_wset') or 0 for r in res),
                    'max_peak_wset': max(r.get('peak_wset') or 0 for r in res), 'bg_cores_mean': bg.mean,
                    'time': time.time()})
        json.dump(row, open(os.path.join(base, 'done.json'), 'w'))
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps(row), flush=True)
