"""Uniform evaluation of every method's output on a case.

Metrics per response, then averaged: test coefficient of determination (R2), relative
function estimation error (FEE, mean squared deviation from the noiseless regression
function divided by its variance), and, when the true support is known, F1, Matthews
correlation coefficient (MCC), true and false positive rates (TPR, FPR) and the area
under the precision-recall curve (AUPRC) of the path-entry ranking.
"""
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score
from .caseio import read_meta, load


def load_output(out, meta):
    q, G, D, nte = meta['q'], len(meta['groups']), meta['D'], meta['n_test']
    if os.path.exists(os.path.join(out, 'output.npz')):
        z = np.load(os.path.join(out, 'output.npz'))
        coef, entry, pred = z['coef'], z['entry'], z['pred']
        groups = np.asarray(meta['groups'])
        support = (z['support'].astype(bool) if 'support' in z.files
                   else np.sqrt(np.add.reduceat(coef * coef, groups[:, 0], axis=0)) > 0)
        return {'pred': pred, 'entry': entry, 'support': support, 'coef': coef, 'npz': z}
    rd = lambda f, k: np.fromfile(os.path.join(out, f), dtype=np.float64).reshape((q, k)).T
    o = {'pred': rd('pred.f64', nte), 'entry': rd('entry.f64', G).astype(int),
         'support': rd('support.f64', G) > 0}
    if os.path.exists(os.path.join(out, 'coef.f64')):
        o['coef'] = rd('coef.f64', D)
    return o


def feature_level(mat_g, meta, how='any'):
    """Map group-level (G, q) arrays to raw features via term_features."""
    tf = np.asarray(meta['term_features'])
    p = meta['p_raw']
    out = np.zeros((p, mat_g.shape[1]), dtype=mat_g.dtype)
    if how == 'any':
        for g, f in enumerate(tf):
            out[f] |= mat_g[g]
    else:  # 'min' for entry indices
        out[:] = np.iinfo(np.int32).max
        for g, f in enumerate(tf):
            out[f] = np.minimum(out[f], mat_g[g])
    return out


def metrics(case, out, method=None, pred=None, support=None, entry=None, truth=None):
    meta = read_meta(case)
    o = load_output(out, meta) if pred is None else {'pred': pred, 'support': support, 'entry': entry}
    yte = load(case, 'y', 'test', meta)
    p = o['pred']
    sst = ((yte - yte.mean(0)) ** 2).sum(0)
    r2 = 1 - ((yte - p) ** 2).sum(0) / sst
    res = {'r2': r2}
    tpath = os.path.join(case, 'truth.npz')
    if truth is None and os.path.exists(tpath):
        truth = np.load(tpath)
    if truth is not None:
        m = truth['mean_test_centered']
        res['fee'] = ((p - m) ** 2).mean(0) / m.var(0)
        S = truth['support'].astype(bool)                       # (p_raw, q)
        sel = feature_level(np.asarray(o['support']).astype(bool), meta, 'any')
        ent = feature_level(np.asarray(o['entry']).astype(np.int64), meta, 'min')
        tp = (sel & S).sum(0); fp = (sel & ~S).sum(0); fn = (~sel & S).sum(0); tn = (~sel & ~S).sum(0)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.)
        rec = tp / np.maximum(tp + fn, 1)
        res['f1'] = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0.)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        res['mcc'] = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-300), 0.)
        res['tpr'], res['fpr'] = rec, fp / np.maximum(fp + tn, 1)
        res['size'] = sel.sum(0)
        K = int(ent[ent < np.iinfo(np.int32).max].max()) + 1 if (ent < np.iinfo(np.int32).max).any() else 1
        score = (K + 1 - np.minimum(ent, K + 1)).astype(float)   # earlier entry ranks higher
        res['auprc'] = np.array([average_precision_score(S[:, r], score[:, r] + 1e-9 * np.random.default_rng(r).random(S.shape[0]))
                                 for r in range(S.shape[1])])
    return res


def summarize(res):
    return {k: float(np.nanmean(v)) for k, v in res.items()}
