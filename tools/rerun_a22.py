"""Formal reruns of amendment 22 with the frozen configuration (one worker by default).

  rerun_a22.py sim  [--workers 1]   Chorus (frozen options) and adaptive GL on the formal, robustness
                                    and sharing cases that the paper reports (seeds 1001-1005,
                                    3001-3005, 4001-4005)
  rerun_a22.py real [--workers 1]   Chorus (frozen options) on the eight real cases
Chorus runs go to <case>/Chorus_a22, adaptive GL to <case>/adaptive and <case>/adaptive_d3.
The frozen options come from configs/chorus_frozen_a22.json ("worker_options").
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
from tune_a22 import run_retry, done_ok

FAMILIES = [('formal', os.path.join(ROOT, 'results', 'formal', 'acc'), range(1001, 1006)),
            ('robust', os.path.join(ROOT, 'results', 'robust'), range(3001, 3006)),
            ('sharing', os.path.join(ROOT, 'results', 'sharing'), range(4001, 4006))]
REAL = ['depmap_c200', 'beijing', 'beijing_o2'] + [f'depmap_c200_s{s}' for s in range(3101, 3106)]
PY_PATH = {'save_val_pred': True, 'save_path': True}


def frozen():
    return json.load(open(os.path.join(ROOT, 'configs', 'chorus_frozen_a22.json')))['worker_options']


def sim_jobs():
    opts = frozen()
    jobs = []
    for fam, root, seeds in FAMILIES:
        for name in sorted(os.listdir(root)):
            if 'seed' not in name or int(name.rsplit('seed', 1)[1]) not in seeds:
                continue
            c5, c3 = os.path.join(root, name, 'case'), os.path.join(root, name, 'case_d3')
            if not os.path.exists(os.path.join(c5, 'meta.json')):
                continue
            jobs.append((c5, 'Chorus', os.path.join(root, name, 'Chorus_a22'), {**opts, 'save_variants': True}))
            jobs.append((c5, 'adaptive', os.path.join(root, name, 'adaptive'), PY_PATH))
            if os.path.exists(os.path.join(c3, 'meta.json')):
                jobs.append((c3, 'adaptive', os.path.join(root, name, 'adaptive_d3'), PY_PATH))
    return jobs


def sim_jobs_adaptive():
    """Adaptive GL on both bases of every reported simulated case (independent of Chorus)."""
    jobs = []
    for fam, root, seeds in FAMILIES:
        for name in sorted(os.listdir(root)):
            if 'seed' not in name or int(name.rsplit('seed', 1)[1]) not in seeds:
                continue
            for sub, out in (('case', 'adaptive'), ('case_d3', 'adaptive_d3')):
                c = os.path.join(root, name, sub)
                if os.path.exists(os.path.join(c, 'meta.json')):
                    jobs.append((c, 'adaptive', os.path.join(root, name, out), PY_PATH))
    return jobs


def real_jobs(cases=None):
    opts = frozen()
    jobs = []
    for name in (cases or REAL):
        case = os.path.join(ROOT, 'results', 'cases', name)
        o = {**opts, 'bases': [b.replace('case', name) for b in opts.get('bases', ['../case_d3'])],
             'save_variants': True}
        if read_meta(case).get('kind') == 'beijing':
            o['val_cluster'] = 'week'
        jobs.append((case, 'Chorus', os.path.join(ROOT, 'results', 'real_a22', name, 'Chorus_a22'), o))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['sim', 'real'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--method', help='run only this method (e.g. adaptive, which needs no frozen configuration)')
    ap.add_argument('--cases', nargs='*', help='real cases to run (default: all eight)')
    a = ap.parse_args()
    if a.method == 'adaptive':
        jobs = [j for j in sim_jobs_adaptive()]
    else:
        jobs = sim_jobs() if a.what == 'sim' else real_jobs(a.cases)
        if a.method:
            jobs = [j for j in jobs if j[1] == a.method]

    def work(j):
        case, m, out, opts = j
        if done_ok(out):
            return
        r = run_retry(case, m, out, opts, cap_seconds=8 * 3600)
        print(json.dumps({'out': out, 'status': r['run_status'], 'fit_seconds': r.get('fit_seconds'),
                          'error': r.get('error')}), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    print('rerun_a22', a.what, 'done', len(jobs), flush=True)


if __name__ == '__main__':
    main()
