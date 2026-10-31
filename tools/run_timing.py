"""Sequential timing runs: one fresh single-threaded process at a time.

Repetitions are interleaved (all items once, then all items again), so slow drifts in
background load affect every method alike; the minimum over repetitions is reported.
Once a method is stopped (OOM or time cap) in an experiment, larger sizes of the same
experiment are not run for it and are recorded as stopped, unless --no-skip is given (used to
measure where every skipped setting stops; amendment 13).
Usage: run_timing.py PLAN_JSON OUT_ROOT [--no-skip]
"""
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.runner import run
from bench.quiet import wait_quiet, Background

BG_LIMIT = 0.3        # mean background load (cores) above which a run is repeated (amendment 21 addendum)
MAX_TRIES = 5

plan = json.load(open(sys.argv[1]))
out_root = sys.argv[2]
no_skip = '--no-skip' in sys.argv[3:]
os.makedirs(out_root, exist_ok=True)
logf = os.path.join(out_root, 'timing_log.jsonl')
failed = {}
if os.path.exists(logf):
    for line in open(logf):
        r = json.loads(line)
        if r['status'] in ('OOM', 'TIMEOUT'):
            failed[(r['experiment'], r['method'])] = min(failed.get((r['experiment'], r['method']), 1e18), r['size'])
log = open(logf, 'a')
max_reps = max(it.get('reps', 1) for it in plan)
for rep in range(max_reps):
    for it in plan:
        if rep >= it.get('reps', 1):
            continue
        key = (it['experiment'], it['method'])
        out = os.path.join(out_root, it['tag'], it['method'], f'rep{rep}')
        if os.path.exists(os.path.join(out, 'run.json')):
            continue
        # larger sizes are harder only for response and row scaling; a larger batch size or memory
        # budget makes a run easier, so a stop there says nothing about the other settings
        if not no_skip and key in failed and it['size'] > failed[key] and it['experiment'] in ('q_scaling', 'n_scaling'):
            row = {**{k: it.get(k) for k in ('experiment', 'tag', 'method', 'size', 'q', 'n', 'B', 'M')},
                   'rep': rep, 'status': 'SKIPPED_AFTER_STOP'}
            log.write(json.dumps(row) + '\n'); log.flush()
            continue
        budget = int(it['budget_gb'] * (1 << 30)) if it.get('budget_gb') else None
        for attempt in range(MAX_TRIES):
            wait_quiet(log=lambda s: print(s, flush=True))
            with Background() as bg:
                r = run(it['case'], it['method'], out, it.get('options', {}), budget_bytes=budget,
                        cap_seconds=it.get('cap_seconds', 3600))
            if bg.mean <= BG_LIMIT or attempt == MAX_TRIES - 1:
                break
            # disturbed by another workload: keep the evidence aside and repeat the run
            aside = os.path.join(os.path.dirname(out_root.rstrip('\\/')), 'timing3_disturbed', it['tag'], it['method'],
                                 f'rep{rep}_try{attempt}')
            os.makedirs(os.path.dirname(aside), exist_ok=True)
            shutil.move(out, aside)
            log.write(json.dumps({**{k: it.get(k) for k in ('experiment', 'tag', 'method', 'variant', 'size')},
                                  'rep': rep, 'status': 'DISTURBED', 'fit_seconds': r.get('fit_seconds'),
                                  'bg_cores_mean': bg.mean, 'bg_cores_peak': bg.peak, 'time': time.time()}) + '\n')
            log.flush()
        row = {**{k: it.get(k) for k in ('experiment', 'tag', 'method', 'variant', 'size', 'q', 'n', 'B', 'M')},
               'rep': rep, 'status': r['run_status'], 'fit_seconds': r.get('fit_seconds'),
               'bg_cores_mean': bg.mean, 'bg_cores_peak': bg.peak, 'attempts': attempt + 1,
               'peak_wset': r.get('peak_wset'), 'peak_rss_sampled': r.get('peak_rss_sampled'),
               'device_read_bytes': r.get('device_read_bytes'), 'chorus': r.get('chorus'), 'error': r.get('error'),
               'wall_seconds': r['wall_seconds'], 'load_seconds': r.get('load_seconds'),
               'peak_rss': r['peak_rss'], 'read_bytes': r.get('chorus', {}).get('read_bytes'),
               'batch_size': r.get('chorus', {}).get('batch_size'),
               'pass_seconds': r.get('chorus', {}).get('pass_seconds'),
               'gram_seconds': r.get('chorus', {}).get('gram_seconds'),
               'stage2_seconds': r.get('chorus', {}).get('stage2_seconds'), 'time': time.time()}
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps(row), flush=True)
        if r['run_status'] in ('OOM', 'TIMEOUT'):
            failed[key] = min(failed.get(key, 1e18), it['size'])
