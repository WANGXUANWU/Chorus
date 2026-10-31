"""Sensitivity of the frozen configuration on the 100 cases of the main grid (amendment 22 (d)).

Reported only; nothing here changes the method.
  sensitivity_a22.py fit [--workers 1]   refits with patience 3, 8 and the full path (<case>/Chorus_a22_p*)
  sensitivity_a22.py score               offline sweeps on the saved candidates of Chorus_a22: zeta, the
                                         alpha grid, the gamma set, the basis sizes and the relaxed refit;
                                         patience from the refits (F1, FEE and fit time against patience 5)
Writes results/v2/sensitivity_a22.csv with one row per (parameter, value) and the mean F1, MCC, FEE and
test R2 over cases, with the standard error over cases.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from bench.evaluate import metrics
from tune_selection import apply_rule, score
from tune_a22 import run_retry, done_ok
from rerun_a22 import frozen

MAIN = os.path.join(ROOT, 'results', 'formal', 'acc')
PATIENCE = {'Chorus_a22_p3': 3, 'Chorus_a22_p8': 8, 'Chorus_a22_pfull': None}
KEEP = ('f1', 'mcc', 'fee', 'r2', 'tpr', 'fpr', 'size', 'auprc')        # metrics kept per case


def cases():
    return [n for n in sorted(os.listdir(MAIN)) if 'seed' in n and int(n.rsplit('seed', 1)[1]) <= 1005
            and os.path.exists(os.path.join(MAIN, n, 'Chorus_a22', 'output.npz'))]


def fit(workers):
    opts = frozen()
    jobs = [(os.path.join(MAIN, n, 'case'), os.path.join(MAIN, n, out), {**opts, 'patience': p})
            for n in cases() for out, p in PATIENCE.items()]

    def work(j):
        case, out, o = j
        if done_ok(out):
            return
        r = run_retry(case, 'Chorus', out, o, cap_seconds=3600)
        print(json.dumps({'out': out, 'status': r['run_status'], 'fit_seconds': r.get('fit_seconds')}), flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, jobs))
    print('sensitivity fit done', len(jobs), flush=True)


def offline_rules():
    eb = dict(parsimony='groups', reference='first', shrink='eb')
    a_all = lambda c: True
    out = []
    for z in (0.25, 0.5, 0.75, 1.0):
        out.append(('zeta', z, {**eb, 'kappa': z}, a_all, 'cand_pred'))
    grids = {'2': (0., 1.), '3': (0., .5, 1.), '5': (0., .25, .5, .75, 1.)}
    for k, g in grids.items():
        out.append(('alpha grid', k, {**eb, 'kappa': .5}, lambda c, g=g: c['kind'] == 'GL' or c['alpha'] in g, 'cand_pred'))
    for k, g in {'1': (1.,), '2': (2.,), '1, 2': (1., 2.)}.items():
        out.append(('gamma', k, {**eb, 'kappa': .5}, lambda c, g=g: c['kind'] == 'GL' or c['gamma'] in g, 'cand_pred'))
    for k, b in {'5': (0,), '3': (1,), '3, 5': (0, 1)}.items():
        out.append(('basis', k, {**eb, 'kappa': .5}, lambda c, b=b: c['basis'] in b, 'cand_pred'))
    out.append(('relaxed refit', 'off', {**eb, 'kappa': .5}, a_all, 'cand_pred_pen'))
    out.append(('relaxed refit', 'on', {**eb, 'kappa': .5}, a_all, 'cand_pred'))
    return out


def summarize(df, keys):
    g = df.groupby(keys)
    m = g[['f1', 'mcc', 'fee', 'r2']].mean()
    se = g[['f1', 'fee']].sem().rename(columns={'f1': 'f1_se', 'fee': 'fee_se'})
    return m.join(se).reset_index()


def main_score():
    rows = []
    rules = offline_rules()
    for n in cases():
        case = os.path.join(MAIN, n, 'case')
        z = np.load(os.path.join(MAIN, n, 'Chorus_a22', 'output.npz'))
        for par, val, rule, subset, key in rules:
            m = score(case, z, apply_rule(z, rule, subset, 'closs'), pred_key=key)
            rows.append({'case': n, 'parameter': par, 'value': str(val), **{k: m[k] for k in KEEP if k in m}})
        t5 = json.load(open(os.path.join(MAIN, n, 'Chorus_a22', 'run.json'))).get('fit_seconds')
        m5 = metrics(case, os.path.join(MAIN, n, 'Chorus_a22'))
        rows.append({'case': n, 'parameter': 'patience', 'value': '5', 'time_ratio': 1.,
                     **{k: float(np.nanmean(m5[k])) for k in KEEP if k in m5}})
        for out, p in PATIENCE.items():
            d = os.path.join(MAIN, n, out)
            if not os.path.exists(os.path.join(d, 'output.npz')):
                continue
            m = metrics(case, d)
            t = json.load(open(os.path.join(d, 'run.json'))).get('fit_seconds')
            rows.append({'case': n, 'parameter': 'patience', 'value': 'full' if p is None else str(p),
                         'time_ratio': t / t5 if t and t5 else np.nan,
                         **{k: float(np.nanmean(m[k])) for k in KEEP if k in m}})
    df = pd.DataFrame(rows)
    # per case, for the dense sensitivity table of the 29 Oct review (tools/table_sensitivity_a23.py)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'sensitivity_a22_per_case.csv'), index=False)
    res = summarize(df, ['parameter', 'value'])
    tr = df.groupby(['parameter', 'value'])['time_ratio'].median().reset_index()
    res = res.merge(tr, on=['parameter', 'value'], how='left')
    res.to_csv(os.path.join(ROOT, 'results', 'v2', 'sensitivity_a22.csv'), index=False)
    with pd.option_context('display.width', 200):
        print(res.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['fit', 'score'])
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    fit(a.workers) if a.step == 'fit' else main_score()


if __name__ == '__main__':
    main()
