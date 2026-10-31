"""Collect real-data metrics of the revised study (DepMap, Beijing and the confirmation cases).

Each comparison method is scored under the default protocol (d = 5, SAM with 3 functions) and
the aligned protocol (basis size chosen per response by validation error; biglasso jointly per
fitted block of responses). Chorus and its selection ablations come from the saved candidates.
DepMap metrics: test R2, selected genes, and the rank of the best silver-standard predictor
(the response gene or a close paralog) in the path entry order. Beijing metrics: test R2 per
pollutant and selected predictors.
Usage: collect_real_v2.py CASE [CASE ...]   -> results/v2/real_metrics.csv, real_per_response.csv
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta, load
from bench.evaluate import feature_level
from tune_selection import apply_rule, ablation_rules, textbook_adaptive

CASES = os.path.join(ROOT, 'results', 'cases')
OLD, NEW = os.path.join(ROOT, 'results', 'real'), os.path.join(ROOT, 'results', 'real2')
METHODS = {'GL (skglm)': 'skglm', 'GL (Celer)': 'Celer', 'GL (adelie)': 'adelie', 'gMCP (grpreg)': 'grpreg',
           'SGL (sparsegl)': 'sparsegl', 'SpAM (SAM)': 'SAM', 'MTL (biglasso)': 'biglasso'}


def _ok(d):
    f = os.path.join(d, 'run.json')
    return os.path.exists(f) and json.load(open(f)).get('run_status') == 'ok'


def _read_one(d, meta):
    """pred (n_test, q), support (G, q), entry (G, q), val_path (K, q) of one (chunk) run."""
    q, G, nte = meta['q'], len(meta['groups']), meta['n_test']
    f = os.path.join(d, 'output.npz')
    if os.path.exists(f):
        z = np.load(f)
        g = np.asarray(meta['groups'])
        sup = z['support'].astype(bool) if 'support' in z.files else \
            np.sqrt(np.add.reduceat(z['coef'] ** 2, g[:, 0], axis=0)) > 0
        return z['pred'], sup, z['entry'], (z['val_path'] if 'val_path' in z.files else None)
    res = json.load(open(os.path.join(d, 'result.json')))
    q = res.get('q', q)
    rd = lambda name, k: np.fromfile(os.path.join(d, name)).reshape(q, k).T
    vp = rd('val_path.f64', res.get('K', 30)) if os.path.exists(os.path.join(d, 'val_path.f64')) else None
    return rd('pred.f64', nte), rd('support.f64', G) > 0, rd('entry.f64', G).astype(int), vp


def read_run(mdir, case_dir):
    """Concatenate chunks if present. Returns pred, support, entry, val_min (q,), blocks (list of column ranges)."""
    meta = read_meta(case_dir)
    if not os.path.isdir(mdir):
        return None
    chunks = sorted([c for c in os.listdir(mdir) if c.startswith('chunk_')], key=lambda c: int(c.split('_')[1]))
    parts = []
    if chunks:
        for c in chunks:
            a, b = map(int, c.split('_')[1:])
            if not _ok(os.path.join(mdir, c)):
                return None
            sub = dict(meta, q=b - a)
            parts.append(((a, b), _read_one(os.path.join(mdir, c), sub)))
    else:
        if not _ok(mdir):
            return None
        parts.append(((0, meta['q']), _read_one(mdir, meta)))
    pred = np.concatenate([p[1][0] for p in parts], axis=1)
    sup = np.concatenate([p[1][1] for p in parts], axis=1)
    ent = np.concatenate([p[1][2] for p in parts], axis=1)
    vmin = np.concatenate([np.nanmin(p[1][3], axis=0) if p[1][3] is not None else np.full(p[0][1] - p[0][0], np.nan)
                           for p in parts])
    return {'pred': pred, 'support': sup, 'entry': ent, 'vmin': vmin, 'blocks': [p[0] for p in parts]}


def score(case_dir, pred, sup, ent, kind, info=None):
    meta = read_meta(case_dir)
    yte = np.asarray(load(case_dir, 'y', 'test', meta))
    r2 = 1 - ((yte - pred) ** 2).sum(0) / ((yte - yte.mean(0)) ** 2).sum(0)
    selF = feature_level(np.asarray(sup).astype(bool), meta, 'any')
    out = {'r2': r2, 'size': selF.sum(0)}
    if kind == 'depmap':
        entF = feature_level(np.asarray(ent).astype(np.int64), meta, 'min')
        ranks = np.full(meta['q'], np.nan)
        for r, sv in enumerate(info['silver']):
            if sv:
                score_ = -entF[:, r].astype(float) + 1e-9 * np.arange(entF.shape[0])[::-1]
                order = np.argsort(-score_)
                pos = np.empty(len(order), int)
                pos[order] = np.arange(len(order))
                ranks[r] = min(pos[f] for f in sv) + 1
        out['silver_rank'] = ranks
    return out


def summarize(case, method, protocol, s, kind, meta):
    row = {'case': case, 'method': method, 'protocol': protocol, 'r2_mean': float(np.mean(s['r2'])),
           'r2_median': float(np.median(s['r2'])), 'r2_gt01': float(np.mean(s['r2'] > .1)),
           'size_median': float(np.median(s['size'])), 'size_mean': float(np.mean(s['size']))}
    if kind == 'depmap':
        rk = s['silver_rank'][np.isfinite(s['silver_rank'])]
        for k in (1, 5, 10):
            row[f'rec{k}'] = float(np.mean(rk <= k))
    else:
        pol = np.array([c.split('|')[1] for c in meta['ycols']])
        for p in dict.fromkeys(pol):
            row[f'r2_{p}'] = float(np.mean(s['r2'][pol == p]))
    return row


def collect(case):
    c5, c3 = os.path.join(CASES, case), os.path.join(CASES, case + '_d3')
    meta = read_meta(c5)
    kind = meta.get('kind', 'depmap' if case.startswith('depmap') else 'beijing')
    info = json.load(open(os.path.join(c5, 'info.json'))) if kind == 'depmap' else None
    rows, per = [], []

    def add(method, protocol, s, extra=None):
        rows.append({**summarize(case, method, protocol, s, kind, meta), **(extra or {})})
        for r in range(len(s['r2'])):
            per.append({'case': case, 'method': method, 'protocol': protocol, 'response': r,
                        'r2': float(s['r2'][r]), 'size': int(s['size'][r]),
                        'silver_rank': float(s['silver_rank'][r]) if 'silver_rank' in s else np.nan})

    for label, m in METHODS.items():
        r5 = read_run(os.path.join(NEW, case, m), c5) or read_run(os.path.join(OLD, case, m), c5)
        if r5 is None:
            continue
        add(label, 'default', score(c5, r5['pred'], r5['support'], r5['entry'], kind, info))
        alt, alt_case = (m + '_d3', c3) if m != 'SAM' else ('SAM_p5', c5)
        r3 = read_run(os.path.join(NEW, case, alt), alt_case)
        if r3 is None:
            continue
        if m == 'biglasso':
            use3 = np.zeros(meta['q'], bool)
            for a, b in r5['blocks']:
                use3[a:b] = np.nanmean(r3['vmin'][a:b]) < np.nanmean(r5['vmin'][a:b])
        else:
            use3 = r3['vmin'] < r5['vmin']
        pick = lambda k: np.where(use3[None, :], r3[k], r5[k])
        add(label, 'aligned', score(c5, pick('pred'), pick('support'), pick('entry'), kind, info),
            {'share_alt_basis': float(use3.mean())})
    p5, p3 = os.path.join(NEW, case, 'adelieP'), os.path.join(NEW, case, 'adelieP_d3')
    if _ok(p5) and _ok(p3):
        from collect_v2 import one_se
        e5, pr5, s5, en5 = one_se(c5, p5)
        e3, pr3, s3, en3 = one_se(c3, p3)
        use3 = e3 < e5
        add('GL + 1-SE', 'aligned', score(c5, np.where(use3[None], pr3, pr5), np.where(use3[None], s3, s5),
                                          np.where(use3[None], en3, en5), kind, info))
        add('GL + 1-SE', 'default', score(c5, pr5, s5, en5, kind, info))
    f = os.path.join(NEW, case, 'Chorus2', 'output.npz')
    if os.path.exists(f):
        z = np.load(f)
        for vname, rule, subset, loss in ablation_rules():
            pk = apply_rule(z, rule, subset, loss)
            r = np.arange(len(pk))
            s = score(c5, z['cand_pred'][pk, :, r].T.astype(float), z['cand_support'][pk, :, r].T,
                      z['cand_entry'][pk, :, r].T.astype(np.int64), kind, info)
            labs = [str(z['cand_label'][i]) for i in pk]
            add(vname, 'aligned', s, {'share_d3': float(np.mean([l.endswith('|1') for l in labs])),
                                      'share_gl': float(np.mean([l.startswith('GL') for l in labs]))})
            if vname == 'Chorus' and 'cand_pred_pen' in z.files:
                for proto, b_ in (('default', 0), ('aligned', None)):
                    pa = textbook_adaptive(z, b_)
                    add('adaptive GL', proto, score(c5, z['cand_pred_pen'][pa, :, r].T.astype(float), z['cand_support'][pa, :, r].T,
                                                    z['cand_entry'][pa, :, r].T.astype(np.int64), kind, info))
                add('no refit', 'aligned', score(c5, z['cand_pred_pen'][pk, :, r].T.astype(float), z['cand_support'][pk, :, r].T,
                                                 z['cand_entry'][pk, :, r].T.astype(np.int64), kind, info))
    f = os.path.join(NEW, case, 'Chorus2_noscale', 'output.npz')
    if os.path.exists(f):
        z = np.load(f)
        vname, rule, subset, loss = ablation_rules()[0]
        pk = apply_rule(z, rule, subset, loss)
        r = np.arange(len(pk))
        add('Chorus, no scale normalization', 'aligned',
            score(c5, z['cand_pred'][pk, :, r].T.astype(float), z['cand_support'][pk, :, r].T,
                  z['cand_entry'][pk, :, r].T.astype(np.int64), kind, info))
    old = read_run(os.path.join(OLD, case, 'Chorus'), c5)
    if old is not None:
        add('Chorus (previous)', 'default', score(c5, old['pred'], old['support'], old['entry'], kind, info))
    return rows, per


def main(cases):
    rows, per = [], []
    for c in cases:
        try:
            r, p = collect(c)
            rows += r
            per += p
        except Exception as e:
            print('skip', c, repr(e)[:200], flush=True)
    os.makedirs(os.path.join(ROOT, 'results', 'v2'), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'results', 'v2', 'real_metrics.csv'), index=False)
    pd.DataFrame(per).to_csv(os.path.join(ROOT, 'results', 'v2', 'real_per_response.csv'), index=False)
    cols = [c for c in ('case', 'method', 'protocol', 'r2_mean', 'r2_median', 'size_median', 'rec1', 'rec10') if c in df]
    print(df[cols].round(4).to_string())


if __name__ == '__main__':
    main(sys.argv[1:])
