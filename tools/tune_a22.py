"""Tuning study of amendment 22 (tuning data only).

Cases: the 36 tuning cases of the main grid (results/pilot/acc, seeds 101-103) and 18 new tuning
cases with seeds 201-202 in the settings of the robustness and sharing grids (results/pilot/tune_a22).
Every case gets d=3 and d=7 siblings (case_d3, case_d7).
  tune_a22.py cases                 generate the new cases and the siblings
  tune_a22.py fit [--workers 3]     run the fit-level variants F0..F8 with saved candidates
  tune_a22.py score                 score fit variants, then selection variants on the chosen fit
Outputs: results/pilot/tune_a22/<case>/<variant>, results/pilot/tune_a22_fit.csv,
results/pilot/tune_a22_select.csv and a printed decision under the rule of amendment 22.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from bench.runner import run, PY
from bench.evaluate import metrics

OLD = os.path.join(ROOT, 'results', 'pilot', 'acc')
NEW = os.path.join(ROOT, 'results', 'pilot', 'tune_a22')
BASE = dict(n_train=400, n_val=200, n_test=1000, p=200, q=16, s=6, kappa=0.5, rho=0.5, snr=2.0)
SETTINGS = [{}, {'n_train': 200, 'n_val': 100}, {'s': 12}, {'noise': 't3'}, {'p': 500},
            {'snr_range': [0.25, 8.0]}, {'scale_range': 3.0}, {'n_clusters': 2}, {'q': 64, 'kappa': 0.0}]
SEEDS = (201, 202)
FIT = {   # fit-level variants: Chorus options on top of the frozen configuration
    'F0': {},
    'F1': {'gammas': [0.5, 1.0, 2.0]},
    'F2': {'bases': ['../case_d3', '../case_d7']},
    'F3': {'relax_per_response': True},
    'F4': {'relax_grid': [i / 8 for i in range(9)]},
    'F5': {'alphas': [0.0, 0.5, 1.0]},
    'F6': {'scale_norm': False},
    'F7': {'patience': 3},
    'F8': {'patience': 8},
    'F9': {'bases': ['../case_d3', '../case_d7'], 'relax_per_response': True},
    'F10': {'bases': ['../case_d3', '../case_d7'], 'relax_per_response': True, 'scale_norm': False},
}
METRICS = (('f1', False), ('mcc', False), ('fee', True), ('r2', False))


def run_retry(*args, tries=20, **kw):
    """bench.runner.run that waits and retries when the machine runs out of commit memory
    (the shared PC raises WinError 1455 in psutil); the run itself is repeated from scratch."""
    import time
    for t in range(tries):
        try:
            return run(*args, **kw)
        except (OSError, MemoryError) as e:
            print(json.dumps({'retry': args[2], 'error': repr(e)}), flush=True)
            time.sleep(120)
    return {'run_status': 'error', 'error': 'retries exhausted'}


def done_ok(out):
    """True when out holds a finished run; failed runs (for example out of memory) run again."""
    f = os.path.join(out, 'run.json')
    if not os.path.exists(f):
        # a worker whose parent runner died (out of memory) still finishes and writes result.json;
        # its outputs are complete, so a run.json is written from it (memory samples are absent)
        g = os.path.join(out, 'result.json')
        try:
            res = json.load(open(g)) if os.path.exists(g) else {}
        except ValueError:
            res = {}
        if res.get('status') != 'ok':
            return False
        json.dump({**res, 'run_status': 'ok', 'note': 'run.json written from result.json (parent runner lost)'},
                  open(f, 'w'), indent=1)
        return True
    try:
        return json.load(open(f)).get('run_status') == 'ok'
    except ValueError:
        return False


def name_of(c):
    return '_'.join(f"{k}{'-'.join(map(str, v)) if isinstance(v, list) else v}" for k, v in c.items())


def case_dirs():
    """(name, case folder, output root) of every tuning case."""
    out = []
    for name in sorted(os.listdir(OLD)):
        if os.path.exists(os.path.join(OLD, name, 'case', 'meta.json')):
            out.append((name, os.path.join(OLD, name, 'case'), os.path.join(NEW, name)))
    for s in SETTINGS:
        for seed in SEEDS:
            c = {**BASE, **s, 'seed': seed}
            name = name_of(c)
            out.append((name, os.path.join(NEW, name, 'case'), os.path.join(NEW, name)))
    return out


def make_cases():
    from bench.make_sim_case import make_sim_case
    for s in SETTINGS:
        for seed in SEEDS:
            c = {**BASE, **s, 'seed': seed}
            cdir = os.path.join(NEW, name_of(c), 'case')
            if not os.path.exists(os.path.join(cdir, 'meta.json')):
                make_sim_case(cdir, n_lambda=30, lambda_min_ratio=0.02, **c)

    def sib(job):
        cdir, tag, knots = job
        dst = os.path.join(os.path.dirname(cdir), tag)
        if not os.path.exists(os.path.join(dst, 'meta.json')):
            subprocess.run([PY, os.path.join(ROOT, 'tools', 'make_basis_case.py'), cdir, dst, str(knots)],
                           check=True, capture_output=True)
    jobs = [(c, t, k) for _, c, _ in case_dirs() for t, k in (('case_d3', 2), ('case_d7', 6))]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(sib, jobs))
    print('cases ready', len(case_dirs()), flush=True)


def fit(workers):
    jobs = []
    for name, cdir, oroot in case_dirs():
        for v, extra in FIT.items():
            opts = {'bases': ['../case_d3'], 'save_variants': True, **extra}
            # bases are relative to the case folder; the siblings live next to it
            jobs.append((cdir, os.path.join(oroot, v), opts))

    def work(j):
        cdir, out, opts = j
        if done_ok(out):
            return
        r = run_retry(cdir, 'Chorus', out, opts, cap_seconds=3600)
        print(json.dumps({'out': out, 'status': r['run_status'], 'fit_seconds': r.get('fit_seconds')}), flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, jobs))
    print('fit done', len(jobs), flush=True)


def mean_rank(df, col='variant'):
    ranks = []
    for m, asc in METRICS:
        ranks.append(df.pivot(index='case', columns=col, values=m).rank(axis=1, ascending=asc))
    return (sum(ranks) / len(ranks)).mean().sort_values()


def score():
    from tune_selection import apply_rule, score as sel_score
    rows = []
    for name, cdir, oroot in case_dirs():
        for v in FIT:
            out = os.path.join(oroot, v)
            if not os.path.exists(os.path.join(out, 'output.npz')):
                continue
            m = metrics(cdir, out)
            secs = json.load(open(os.path.join(out, 'run.json'))).get('fit_seconds')
            rows.append({'case': name, 'variant': v, 'family': 'main' if 'seed10' in name else 'robust_sharing',
                         'seconds': secs, **{k: float(np.nanmean(x)) for k, x in m.items()}})
    fdf = pd.DataFrame(rows)
    full = fdf.groupby('case')['variant'].count()
    fdf = fdf[fdf['case'].isin(full[full == len(FIT)].index)]
    fdf.to_csv(os.path.join(ROOT, 'results', 'pilot', 'tune_a22_fit.csv'), index=False)
    cols = ['f1', 'mcc', 'fee', 'r2', 'size', 'seconds']
    print('cases scored', fdf['case'].nunique())
    print(fdf.groupby('variant')[cols].mean().round(4).to_string())
    mr = mean_rank(fdf)
    print('mean rank, all tuning cases'); print(mr.round(3).to_string())
    for fam, d in fdf.groupby('family'):
        print('mean rank,', fam); print(mean_rank(d).round(3).to_string())
    # selection variants on the saved candidates of the frozen fit and of every fit variant
    srows = []
    base = dict(parsimony='groups', reference='first')
    for name, cdir, oroot in case_dirs():
        for v in FIT:
            f = os.path.join(oroot, v, 'output.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            for zeta in (0.25, 0.5, 0.75, 1.0):
                for shrink in ('eb', 'none', 'pool'):
                    for loss in ('closs', 'vloss'):
                        rule = {**base, 'kappa': zeta, 'shrink': shrink}
                        pick = apply_rule(z, rule, None, loss)
                        srows.append({'case': name, 'fit': v, 'variant': f'z{zeta}_{shrink}_{loss}',
                                      **sel_score(cdir, z, pick)})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(ROOT, 'results', 'pilot', 'tune_a22_select.csv'), index=False)
    for v, d in sdf.groupby('fit'):
        print('selection variants under', v)
        print(mean_rank(d).head(8).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['cases', 'fit', 'score'])
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(NEW, exist_ok=True)
    {'cases': make_cases, 'fit': lambda: fit(a.workers), 'score': score}[a.step]()


if __name__ == '__main__':
    main()
