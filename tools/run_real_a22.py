"""Real-data baseline runs with test predictions at every level (amendment 22).

For every case name C (with sibling C_d3) this runs the published baselines on the d=5 and d=3
designs (SpAM by samQL with three and five functions) and saves, besides the selected fit, the
test predictions and active groups at every path level, so every method can be compared with
Chorus at the same model size (tools/matched_size.py). GL comes from the existing adelie path
runs (results/real2/C/adelieP*), and the selected GMCP fits from results/real2. Because the
matched-size model of a response equals its selected model whenever that model is no larger than
the model of Chorus, grpreg paths (about 90 s per DepMap response) run only for the responses
listed in --grpreg-lists (JSON {case: {run name: [response indices]}}). Per-response methods run
in chunks of responses; the selected fits are merged as in run_real2.py.
Results go to results/real_a22/C/<run name>.
Usage: run_real_a22.py [CASE ...] [--workers 1] [--only RUN ...] [--grpreg-lists FILE]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from bench.caseio import read_meta
from run_real import merge
from run_real2 import merge_val
from tune_a22 import run_retry, done_ok

CASES = os.path.join(ROOT, 'results', 'cases')
ALL = ['depmap_c200', 'beijing', 'beijing_o2'] + [f'depmap_c200_s{s}' for s in range(3101, 3106)]
R_PATH = {'save_val_pred': True, 'save_test_path': True}
PY_PATH = {'save_val_pred': True, 'save_path': True}
COST = {'adaptive': 0, 'sparsegl': 1, 'biglasso': 2, 'SAM': 3, 'grpreg': 4}


def jobs_for(name, only, chunk, grpreg_lists):
    c5, c3 = os.path.join(CASES, name), os.path.join(CASES, name + '_d3')
    meta = read_meta(c5)
    q = meta['q']
    root = os.path.join(ROOT, 'results', 'real_a22', name)
    spec = [('adaptive', 'adaptive', c5, PY_PATH), ('adaptive_d3', 'adaptive', c3, PY_PATH)]
    for m in ('sparsegl', 'biglasso'):
        spec += [(m, m, c5, R_PATH), (m + '_d3', m, c3, R_PATH)]
    spec += [('SAM', 'SAM', c5, R_PATH), ('SAM_p5', 'SAM', c5, {**R_PATH, 'sam_p': 5})]
    jobs, merges = [], []
    for out, m, case, opts in spec:
        if only and out not in only:
            continue
        mdir = os.path.join(root, out)
        if m == 'biglasso' and meta.get('kind') != 'depmap':     # the joint model sees all responses
            jobs.append((case, m, mdir, opts))
            continue
        chunks = [(r0, min(q, r0 + chunk)) for r0 in range(0, q, chunk)]
        for r0, r1 in chunks:
            jobs.append((case, m, os.path.join(mdir, f'chunk_{r0}_{r1}'), {**opts, 'resp_range': [r0, r1]}))
        merges.append((case, mdir, chunks, read_meta(case)))     # the d = 3 sibling has its own groups
    for out, case in (('grpreg', c5), ('grpreg_d3', c3)):
        rl = sorted(int(r) for r in (grpreg_lists or {}).get(name, {}).get(out, []))
        for i in range(0, len(rl), chunk):
            jobs.append((case, 'grpreg', os.path.join(root, out, f'list_{i}'), {**R_PATH, 'resp_list': rl[i:i + chunk]}))
    return jobs, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*', default=ALL)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=40)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--grpreg-lists')
    ap.add_argument('--reverse', action='store_true', help='take the jobs from the end (a second driver)')
    ap.add_argument('--claim', help='tag of an exclusive claim file per job, shared by cooperating drivers')
    a = ap.parse_args()
    lists = json.load(open(a.grpreg_lists)) if a.grpreg_lists else None
    jobs, merges = [], []
    for name in a.cases:
        j, m = jobs_for(name, a.only, a.chunk, lists)
        jobs += j
        merges += m
    jobs.sort(key=lambda j: COST[j[1]])
    if a.reverse:
        jobs = jobs[::-1]

    def work(j):
        case, m, out, opts = j
        if done_ok(out):
            return
        if a.claim:                     # several drivers on one job list: a job runs once
            os.makedirs(out, exist_ok=True)
            try:
                os.close(os.open(os.path.join(out, '.claim_' + a.claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                return
        r = run_retry(case, m, out, opts, cap_seconds=10 * 3600)
        print(json.dumps({'out': out, 'status': r['run_status'], 'fit_seconds': r.get('fit_seconds'),
                          'error': r.get('error')}), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    for case, mdir, chunks, meta in merges:
        try:
            for r0, r1 in chunks:
                done_ok(os.path.join(mdir, f'chunk_{r0}_{r1}'))
            merge(case, mdir, chunks, meta)
            merge_val(mdir, chunks)
            print('merged', mdir, flush=True)
        except Exception as e:
            print('merge failed', mdir, repr(e), flush=True)
    print('real_a22 done', len(jobs), flush=True)


if __name__ == '__main__':
    main()
