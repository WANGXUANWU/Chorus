"""Offline evaluation of selection rules on saved Chorus candidates (amendment 14).

For every case with a Chorus run that saved its candidates, every selection variant is
applied to the same candidates; the chosen candidate's saved test predictions, supports
and entry orders are scored with bench.evaluate.metrics.
Usage: tune_selection.py ROOT [--run Chorus2] [--out CSV] [--variants tune|ablation]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from bench.evaluate import metrics
from chorus.selection import eb_select

TUNE = [dict(kappa=k, parsimony=p, reference=r, shrink='eb')
        for k in (1.0, 0.5) for p in ('groups', 'params') for r in ('first', 'best_gl')]


def labels(z):
    out = []
    for s in z['cand_label']:
        kind, a, g, b = str(s).split('|')
        out.append({'kind': kind, 'alpha': None if a == 'None' else float(a),
                    'gamma': None if g == 'None' else float(g), 'basis': int(b)})
    return out


def apply_rule(z, rule, subset=None, loss='closs'):
    """Return pick (q,) as candidate indices into the full candidate list."""
    lab = labels(z)
    C = len(lab)
    idx = np.arange(C) if subset is None else np.asarray([i for i in range(C) if subset(lab[i])])
    closs = z[f'cand_{loss}'][idx]
    groups, params = z['cand_groups'][idx], z['cand_params'][idx]
    q = closs.shape[2]
    if len(idx) == 1:
        return np.full(q, idx[0])
    gl = [k for k, i in enumerate(idx) if lab[i]['kind'] == 'GL']
    if not gl:
        gl = [0]
    if rule.get('reference', 'first') == 'best_gl':
        ref = np.asarray(gl)[np.argmin(closs[gl].sum(axis=1), axis=0)]
    else:
        ref = np.full(q, gl[0])
    pick, _ = eb_select(closs, groups, params, z['val_var'], int(z['n_val']), ref,
                        rule.get('kappa', 1.0), rule.get('parsimony', 'groups'), rule.get('shrink', 'eb'))
    return idx[pick]


def score(case, z, pick, pred_key='cand_pred'):
    q = len(pick)
    r = np.arange(q)
    pred = z[pred_key][pick, :, r].T.astype(float)
    sup = z['cand_support'][pick, :, r].T
    ent = z['cand_entry'][pick, :, r].T.astype(np.int64)
    m = metrics(case, None, pred=pred, support=sup, entry=ent)
    score.last = m
    lab = labels(z)
    kinds = [lab[i] for i in pick]
    share = {'share_gl': np.mean([k['kind'] == 'GL' for k in kinds]),
             'share_a0': np.mean([k['alpha'] == 0. for k in kinds]),
             'share_mid': np.mean([k['alpha'] is not None and 0 < k['alpha'] < 1 for k in kinds]),
             'share_a1': np.mean([k['alpha'] == 1. for k in kinds]),
             'share_d3': np.mean([k['basis'] == 1 for k in kinds])}
    return {**{k: float(np.nanmean(v)) for k, v in m.items()}, **share}


def textbook_adaptive(z, basis=None):
    """Two-stage adaptive group lasso as published (Huang et al. 2010): weights from the
    validation-selected group-lasso fit, no borrowing (alpha = 0), gamma in {1, 2}, the stage-2 level
    and gamma (and the basis when basis is None) chosen by the smallest validation error; penalized
    coefficients (no refit). Returns pick (q,)."""
    lab = labels(z)
    idx = np.asarray([i for i, c in enumerate(lab) if c['kind'] != 'GL' and c['alpha'] == 0.
                      and (basis is None or c['basis'] == basis)])
    v = z['cand_vloss'][idx].sum(axis=1)
    return idx[np.argmin(v, axis=0)]


def ablation_rules():
    best = dict(kappa=0.5, parsimony='groups', reference='first', shrink='eb')
    return [
        ('Chorus', best, None, 'closs'),
        ('no shrinkage', {**best, 'shrink': 'none'}, None, 'closs'),
        ('full pooling', {**best, 'shrink': 'pool'}, None, 'closs'),
        ('no cross-fitting', best, None, 'vloss'),
        ('GL only', best, lambda c: c['kind'] == 'GL', 'closs'),
        ('GL d=5 only', best, lambda c: c['kind'] == 'GL' and c['basis'] == 0, 'closs'),
        ('no borrowing', best, lambda c: c['kind'] == 'GL' or c['alpha'] == 0., 'closs'),
        ('pooled weights', best, lambda c: c['kind'] == 'GL' or c['alpha'] == 1., 'closs'),
        ('fixed alpha 0.5', best, lambda c: c['kind'] == 'GL' or c['alpha'] == .5, 'closs'),
        ('d = 5 only', best, lambda c: c['basis'] == 0, 'closs'),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--run', default='Chorus2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--variants', default='tune')
    a = ap.parse_args()
    rows = []
    for root in a.roots:
        for name in sorted(os.listdir(root)):
            f = os.path.join(root, name, a.run, 'output.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            case = os.path.join(root, name, 'case')
            if a.variants == 'tune':
                rules = [(f"k{r['kappa']}_{r['parsimony']}_{r['reference']}", r, None, 'closs') for r in TUNE]
            else:
                rules = ablation_rules()
            for vname, rule, subset, loss in rules:
                pick = apply_rule(z, rule, subset, loss)
                rows.append({'case': name, 'variant': vname, **score(case, z, pick)})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    if a.variants == 'tune':
        ranks = []
        for metric, asc in (('f1', False), ('mcc', False), ('fee', True), ('r2', False)):
            rk = df.pivot(index='case', columns='variant', values=metric).rank(axis=1, ascending=asc)
            ranks.append(rk)
        mr = sum(ranks) / len(ranks)
        print(df.groupby('variant')[['f1', 'mcc', 'fee', 'r2', 'auprc', 'size', 'share_gl', 'share_a0', 'share_mid', 'share_a1', 'share_d3']].mean().round(4).to_string())
        print('mean rank'); print(mr.mean().sort_values().round(3).to_string())
    else:
        print(df.groupby('variant')[['f1', 'mcc', 'fee', 'r2', 'auprc', 'size', 'share_gl', 'share_a0', 'share_mid', 'share_a1', 'share_d3']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
