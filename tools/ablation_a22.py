"""Cumulative ablation of amendment 22 (d), offline on the saved candidates of the formal runs.

Rows add one component at a time to the group lasso: per-response reweighting (alpha = 0), borrowing
across responses (all alpha), the cross-fitted empirical Bayes selection, the choice of basis size
and the relaxed refit (the full method). A further row replaces the selection by the 1-SE rule
(fewest terms within one standard error of the smallest validation loss, no shrinkage, no
cross-fitting). Rows without the relaxed refit score the penalized predictions. For every case the
F1 score and FEE of each row are computed; the table reports the change from the previous row
(F1 in points, FEE in percent of the previous value, positive when the component helps), with the
standard error over cases and a paired t-test at 5%.
Usage: ablation_a22.py [--run Chorus_a22] [--kappa 0.5]
Writes results/v2/ablation_a22_cases.csv and results/v2/ablation_a22.csv.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from scipy import stats
from tune_selection import apply_rule, score, labels

FAMILIES = [('formal', os.path.join(ROOT, 'results', 'formal', 'acc'), range(1001, 1006)),
            ('sharing', os.path.join(ROOT, 'results', 'sharing'), range(4001, 4006))]


def pick_min(z, subset, loss='vloss'):
    """Candidate with the smallest summed validation loss per response within the subset."""
    lab = labels(z)
    idx = np.asarray([i for i in range(len(lab)) if subset(lab[i])])
    return idx[np.argmin(z[f'cand_{loss}'][idx].sum(axis=1), axis=0)]


def rows_for(z, kappa, extra, order='A'):
    d5 = lambda c: c['basis'] == 0
    gl5 = lambda c: c['kind'] == 'GL' and c['basis'] == 0
    rw5 = lambda c: d5(c) and (c['kind'] == 'GL' or c['alpha'] == 0.)
    eb = dict(kappa=kappa, parsimony='groups', reference='first', shrink='eb', **extra)
    one_se = dict(kappa=1.0, parsimony='groups', reference='first', shrink='none')
    if order == 'B':          # the selection rule before borrowing, which it governs
        return [
            ('Group lasso', pick_min(z, gl5), 'cand_pred_pen'),
            ('+ Reweighting', pick_min(z, rw5), 'cand_pred_pen'),
            ('+ Empirical Bayes selection', apply_rule(z, eb, rw5, 'closs'), 'cand_pred_pen'),
            ('+ Borrowing across responses', apply_rule(z, eb, d5, 'closs'), 'cand_pred_pen'),
            ('+ Choice of basis size', apply_rule(z, eb, None, 'closs'), 'cand_pred_pen'),
            ('+ Relaxed refit', apply_rule(z, eb, None, 'closs'), 'cand_pred'),
            ('1-SE rule instead of the selection', apply_rule(z, one_se, None, 'vloss'), 'cand_pred'),
        ]
    if order == 'R':          # every row relaxed except the first, the refit counted first
        return [
            ('Group lasso', pick_min(z, gl5), 'cand_pred_pen'),
            ('+ Relaxed refit', pick_min(z, gl5), 'cand_pred'),
            ('+ Reweighting', pick_min(z, rw5), 'cand_pred'),
            ('+ Borrowing with EB selection', apply_rule(z, eb, d5, 'closs'), 'cand_pred'),
            ('+ Choice of basis size', apply_rule(z, eb, None, 'closs'), 'cand_pred'),
            ('1-SE rule instead of the selection', apply_rule(z, one_se, None, 'vloss'), 'cand_pred'),
        ]
    if order == 'L':          # leave one out of the full method (cardiac_FM Table 5 layout)
        full = apply_rule(z, eb, None, 'closs')
        return [
            ('Full', full, 'cand_pred'),
            ('Without reweighting', apply_rule(z, eb, lambda c: c['kind'] == 'GL', 'closs'), 'cand_pred'),
            ('Without borrowing', apply_rule(z, eb, lambda c: c['kind'] == 'GL' or c['alpha'] == 0., 'closs'), 'cand_pred'),
            ('Without the EB selection', pick_min(z, lambda c: True), 'cand_pred'),
            ('Without the choice of basis size', apply_rule(z, eb, d5, 'closs'), 'cand_pred'),
            ('Without the relaxed refit', full, 'cand_pred_pen'),
            ('1-SE rule instead of the selection', apply_rule(z, one_se, None, 'vloss'), 'cand_pred'),
        ]
    if order == 'G':          # borrowing together with the selection that decides how much to borrow
        return [
            ('Group lasso', pick_min(z, gl5), 'cand_pred_pen'),
            ('+ Reweighting', pick_min(z, rw5), 'cand_pred_pen'),
            ('+ Borrowing with EB selection', apply_rule(z, eb, d5, 'closs'), 'cand_pred_pen'),
            ('+ Choice of basis size', apply_rule(z, eb, None, 'closs'), 'cand_pred_pen'),
            ('+ Relaxed refit', apply_rule(z, eb, None, 'closs'), 'cand_pred'),
            ('1-SE rule instead of the selection', apply_rule(z, one_se, None, 'vloss'), 'cand_pred'),
        ]
    return [
        ('Group lasso', pick_min(z, gl5), 'cand_pred_pen'),
        ('+ Reweighting', pick_min(z, rw5), 'cand_pred_pen'),
        ('+ Borrowing across responses', pick_min(z, d5), 'cand_pred_pen'),
        ('+ Empirical Bayes selection', apply_rule(z, eb, d5, 'closs'), 'cand_pred_pen'),
        ('+ Choice of basis size', apply_rule(z, eb, None, 'closs'), 'cand_pred_pen'),
        ('+ Relaxed refit', apply_rule(z, eb, None, 'closs'), 'cand_pred'),
        ('1-SE rule instead of the selection', apply_rule(z, one_se, None, 'vloss'), 'cand_pred'),
    ]


def column(name):
    if '_q64_' in name and 'kappa0.0' in name:
        return 'q=64'
    if 'snr_range' in name:
        return 'Mixed'
    if 'scale_range' in name:
        return 'Rescaled'
    if 'n_clusters' in name:
        return 'Groups'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='Chorus_a22')
    ap.add_argument('--kappa', type=float, default=0.5)
    ap.add_argument('--shrink', default='eb')
    ap.add_argument('--order', default='A', choices=['A', 'B', 'G', 'R', 'L'])
    a = ap.parse_args()
    out = []
    for fam, root, seeds in FAMILIES:
        for name in sorted(os.listdir(root)):
            if 'seed' not in name or int(name.rsplit('seed', 1)[1]) not in seeds:
                continue
            f = os.path.join(root, name, a.run, 'output.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            case = os.path.join(root, name, 'case')
            cols = []
            if fam == 'formal':
                cols.append('All')
                if 'kappa0.0_' in name:
                    cols.append('kappa=0')
                if 'kappa1.0_' in name:
                    cols.append('kappa=1')
            else:
                c = column(name)
                if c:
                    cols.append(c)
            if not cols:
                continue
            for row, pick, key in rows_for(z, a.kappa, {}, a.order):
                m = score(case, z, pick, pred_key=key)
                for c in cols:
                    out.append({'case': name, 'column': c, 'row': row, 'f1': m['f1'], 'fee': m['fee']})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', f'ablation_a22_{a.order}_cases.csv'), index=False)
    order = [r for r, _, _ in rows_for(np.load(f), a.kappa, {}, a.order)] if out else []
    ncum = max(1, sum(1 for r in order if r == 'Group lasso' or r.startswith('+ ')))
    res = []
    for col, d in df.groupby('column'):
        f1 = d.pivot(index='case', columns='row', values='f1')
        fee = d.pivot(index='case', columns='row', values='fee')
        for i, row in enumerate(order):
            ref = order[i - 1] if 0 < i < ncum else (order[ncum - 1] if i >= ncum else None)
            if ref is None:
                res.append({'column': col, 'row': row, 'f1': f1[row].mean(), 'fee': fee[row].mean()})
                continue
            df1 = 100 * (f1[row] - f1[ref]) if i < ncum else 100 * (f1[ref] - f1[row])
            dfe = 100 * (fee[ref] - fee[row]) / fee[ref] if i < ncum else 100 * (fee[row] - fee[ref]) / fee[ref]
            res.append({'column': col, 'row': row, 'n': len(df1),
                        'f1_gain': df1.mean(), 'f1_se': df1.std(ddof=1) / np.sqrt(len(df1)),
                        'f1_p': stats.ttest_1samp(df1, 0.).pvalue if df1.std() > 0 else 1.,
                        'fee_gain': dfe.mean(), 'fee_se': dfe.std(ddof=1) / np.sqrt(len(dfe)),
                        'fee_p': stats.ttest_1samp(dfe, 0.).pvalue if dfe.std() > 0 else 1.})
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(ROOT, 'results', 'v2', f'ablation_a22_{a.order}.csv'), index=False)
    with pd.option_context('display.width', 200):
        print(r.pivot(index='row', columns='column', values='f1_gain').reindex(order).round(2).to_string())
        print(r.pivot(index='row', columns='column', values='fee_gain').reindex(order).round(2).to_string())
        print(r.pivot(index='row', columns='column', values='f1_p').reindex(order).round(3).to_string())
        print(r.pivot(index='row', columns='column', values='fee_p').reindex(order).round(3).to_string())


if __name__ == '__main__':
    main()
