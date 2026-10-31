"""Collect simulation metrics of the revised study (amendment 14).

For every case it scores each comparison method under two protocols:
  default  the d=5 design and the level with the smallest validation error (as published);
  aligned  for each response, the basis size (d in {3, 5}; SAM: 3 or 5 functions) with the
           smaller validation error at its selected level (biglasso: jointly over responses).
It adds GL + 1-SE (adelie paths: the sparsest level whose validation error is within one
standard error of the minimum) and Chorus with the selection ablations of tune_selection.py.
Usage: collect_v2.py OUT_CSV ROOT [ROOT ...]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from bench.caseio import read_meta
from bench.evaluate import metrics, load_output
from tune_selection import apply_rule, score, ablation_rules, textbook_adaptive

SUFFIX = os.environ.get('CHORUS_SUFFIX', '')     # e.g. _a16 to read archived Chorus runs

# label -> (d=5 run, d=3 run, joint selection of d)
BASELINES = {
    'GL (skglm)': ('skglm', 'skglm_d3', False),
    'GL (Celer)': ('Celer', 'Celer_d3', False),
    'GL (adelie)': ('adelie', 'adelie_d3', False),
    'gMCP (grpreg)': ('grpreg', 'grpreg_d3', False),
    'SGL (sparsegl)': ('sparsegl', 'sparsegl_d3', False),
    'SpAM (SAM)': ('SAM', 'SAM_p5', False),
    'MTL (biglasso)': ('biglasso', 'biglasso_d3', True),
}


def val_path(mdir, meta):
    """(K, q) validation MSE along the path of a run."""
    f = os.path.join(mdir, 'output.npz')
    if os.path.exists(f):
        z = np.load(f)
        if 'val_path' in z.files:
            return z['val_path']
    res = json.load(open(os.path.join(mdir, 'result.json')))
    q, K = res.get('q', meta['q']), res.get('K', 30)
    return np.fromfile(os.path.join(mdir, 'val_path.f64')).reshape(q, K).T


def load_run(cdir_case, mdir):
    meta = read_meta(cdir_case)
    o = load_output(mdir, meta)
    vp = val_path(mdir, meta)
    return o, vp


def ok(mdir):
    f = os.path.join(mdir, 'run.json')
    return os.path.exists(f) and json.load(open(f)).get('run_status') == 'ok'


def one_se(cdir, mdir):
    """GL + 1-SE from an adelie run with saved paths: (val-selected mse, pred, support, entry)."""
    z = np.load(os.path.join(mdir, 'output.npz'))
    vr = z['val_resid'].astype(float)                  # (K, n_val, q)
    sq = vr * vr
    mse = sq.mean(axis=1)                              # (K, q)
    q = mse.shape[1]
    kmin = np.nanargmin(mse, axis=0)
    se = sq[kmin, :, np.arange(q)].std(axis=1, ddof=1) / np.sqrt(sq.shape[1])
    k1 = np.array([int(np.flatnonzero(mse[:, r] <= mse[kmin[r], r] + se[r])[0]) for r in range(q)])
    r_ = np.arange(q)
    return (mse[kmin, r_], z['pred_path'][k1, :, r_].T.astype(float), z['active_path'][k1, :, r_].T, z['entry'])


LAST = {}


def score_cols(cdir, pred, sup, ent):
    m = metrics(cdir, None, pred=pred, support=sup, entry=ent)
    LAST['m'] = m
    return {k: float(np.nanmean(v)) for k, v in m.items()}


class Rows(list):
    """Case rows; each append also keeps the per-response metrics of the scoring call."""
    def __init__(self):
        super().__init__()
        self.per = []

    def append(self, r):
        super().append(r)
        m = LAST.pop('m', None)
        if m is None and getattr(score, 'last', None) is not None:
            m, score.last = score.last, None
        if m is not None:
            self.per.append((r['method'], r['protocol'], m))


def collect_case(case_dir):
    cdir = os.path.join(case_dir, 'case')
    rows = Rows()
    for label, (r5, r3, joint) in BASELINES.items():
        m5, m3 = os.path.join(case_dir, r5), os.path.join(case_dir, r3)
        if r5 == 'grpreg' and not ok(m5):
            m5 = os.path.join(case_dir, 'grpreg_vp')
        if not ok(m5):
            continue
        o5, v5 = load_run(cdir, m5)
        rows.append({'method': label, 'protocol': 'default', **score_cols(cdir, o5['pred'], o5['support'], o5['entry']),
                     'share_d3': 0.})
        if not ok(m3):
            continue
        o3, v3 = load_run(cdir if r3 == 'SAM_p5' else os.path.join(case_dir, 'case_d3'), m3)
        e5, e3 = np.nanmin(v5, axis=0), np.nanmin(v3, axis=0)
        if joint:
            use3 = np.full(len(e5), np.nanmean(e3) < np.nanmean(e5))
        else:
            use3 = e3 < e5
        pred = np.where(use3[None, :], o3['pred'], o5['pred'])
        sup = np.where(use3[None, :], o3['support'], o5['support'])
        ent = np.where(use3[None, :], o3['entry'], o5['entry'])
        rows.append({'method': label, 'protocol': 'aligned', **score_cols(cdir, pred, sup, ent),
                     'share_d3': float(use3.mean())})
    # GL + 1-SE (adelie paths), d chosen by validation error at the minimum
    p5, p3 = os.path.join(case_dir, 'adelieP'), os.path.join(case_dir, 'adelieP_d3')
    if ok(p5):
        e5, pr5, s5, en5 = one_se(cdir, p5)
        rows.append({'method': 'GL + 1-SE', 'protocol': 'default', **score_cols(cdir, pr5, s5, en5), 'share_d3': 0.})
        if ok(p3):
            e3, pr3, s3, en3 = one_se(cdir, p3)
            use3 = e3 < e5
            rows.append({'method': 'GL + 1-SE', 'protocol': 'aligned',
                         **score_cols(cdir, np.where(use3[None], pr3, pr5), np.where(use3[None], s3, s5),
                                      np.where(use3[None], en3, en5)), 'share_d3': float(use3.mean())})
    # Chorus and selection ablations from the saved candidates
    for run in ('Chorus2', 'Chorus2_noscale'):
        f = os.path.join(case_dir, run + SUFFIX, 'output.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        rules = ablation_rules() if run == 'Chorus2' else ablation_rules()[:1]
        for vname, rule, subset, loss in rules:
            pick = apply_rule(z, rule, subset, loss)
            name = vname if run == 'Chorus2' else 'Chorus, no scale normalization'
            rows.append({'method': name, 'protocol': 'aligned' if 'd = 5' not in vname else 'default',
                         **score(cdir, z, pick)})
            if run == 'Chorus2' and vname == 'Chorus' and 'cand_pred_pen' in z.files:
                rows.append({'method': 'no refit', 'protocol': 'aligned', **score(cdir, z, pick, 'cand_pred_pen')})
        if run == 'Chorus2' and 'cand_pred_pen' in z.files:
            rows.append({'method': 'adaptive GL', 'protocol': 'default',
                         **score(cdir, z, textbook_adaptive(z, 0), 'cand_pred_pen')})
            rows.append({'method': 'adaptive GL', 'protocol': 'aligned',
                         **score(cdir, z, textbook_adaptive(z), 'cand_pred_pen')})
    # previous version of Chorus (amendments 1 to 13), for reference only
    if ok(os.path.join(case_dir, 'Chorus')):
        o, _ = load_output(os.path.join(case_dir, 'Chorus'), read_meta(cdir)), None
        rows.append({'method': 'Chorus (previous)', 'protocol': 'default',
                     **score_cols(cdir, o['pred'], o['support'], o['entry'])})
    params = dict(read_meta(cdir).get('params', {}))
    name = os.path.basename(case_dir)          # sharing settings are also encoded in the case name
    if 'snr_range' in name and 'snr_range' not in params:
        params['snr_range'] = [float(v) for v in name.split('snr_range')[1].split('_')[0].split('-')]
    if 'scale_range' in name and 'scale_range' not in params:
        params['scale_range'] = float(name.split('scale_range')[1].split('_')[0])
    if 'n_clusters' in name and 'n_clusters' not in params:
        params['n_clusters'] = int(name.split('n_clusters')[1].split('_')[0])
    for r in rows:
        r.update({'case': os.path.basename(case_dir), **{k: (json.dumps(v) if isinstance(v, list) else v)
                                                         for k, v in params.items()}})
    per = []
    for method, protocol, m in rows.per:
        q = len(m['r2'])
        for j in range(q):
            per.append({'case': os.path.basename(case_dir), 'method': method, 'protocol': protocol, 'response': j,
                        **{k: float(v[j]) for k, v in m.items() if np.ndim(v) == 1 and len(v) == q}})
    return list(rows), per


def main(out, roots):
    rows, per = [], []
    for root in roots:
        fam = os.path.basename(os.path.normpath(root)) if os.path.basename(os.path.normpath(root)) != 'acc' else 'formal'
        for name in sorted(os.listdir(root)):
            cd = os.path.join(root, name)
            if not os.path.exists(os.path.join(cd, 'Chorus2' + SUFFIX, 'run.json')):
                continue
            try:
                rr, pp = collect_case(cd)
                for r in rr:
                    r['family'] = fam
                    rows.append(r)
                per += pp
            except Exception as e:
                print('skip', name, repr(e)[:200], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.DataFrame(per).to_csv(out.replace('.csv', '_per_response.csv'), index=False)
    if df.empty:
        return
    print(df.groupby(['family', 'protocol', 'method'])[['f1', 'mcc', 'fee', 'r2', 'auprc', 'size']].mean().round(4).to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
