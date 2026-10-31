"""Numbers of the amendment 22 text (simulation part), written to manuscript/numbers_a22.tex.

Existing macros whose value changes are redefined with \\renewcommand; new statements get new macros.
Sources: results/v2/sim_a22.csv, ablation_a22_L.csv, the saved candidates of <case>/Chorus_a22.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from tune_selection import apply_rule, score, labels
from rerun_a22 import FAMILIES

OUT = os.path.join(ROOT, 'manuscript', 'numbers_a22.tex')
NEW, RE = {}, {}


def f(v, d=2):
    s = f'{v:.{d}f}'
    return s


def pct(v, d=1):
    return f'{100 * v:.{d}f}\\%'


def setting_key(r):
    if r.family == 'formal':
        return ('formal', r.kappa, r.rho, r.snr)
    if r.family == 'robust':
        return ('robust', r.n_train, r.p, r.s, r.q, str(r.noise))
    return ('sharing', r.kappa, r.q, str(r.snr_range), str(r.scale_range), str(r.n_clusters))


def main():
    d = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sim_a22.csv'))
    d['key'] = d.apply(setting_key, axis=1)
    F = d[d.family == 'formal']
    g = F.groupby('method').mean(numeric_only=True)
    RE['NumFOneChorus'] = f(g.loc['Chorus', 'f1'])
    RE['NumFOneAGL'] = f(g.loc['Adaptive GL', 'f1'])
    RE['NumFOneGMCP'] = f(g.loc['GMCP', 'f1'])
    RE['NumFEEChorus'] = f(g.loc['Chorus', 'fee'], 3)
    RE['NumFEEAGL'] = f(g.loc['Adaptive GL', 'fee'], 3)
    RE['NumFEEGMCP'] = f(g.loc['GMCP', 'fee'], 3)
    RE['NumRtwoChorus'] = f(g.loc['Chorus', 'r2'], 3)
    RE['NumRtwoBest'] = f(g.drop('Chorus')['r2'].max(), 3)
    RE['NumSizeGL'] = f(g.loc['GL', 'size'], 0)
    RE['NumSizeSAM'] = f(g.loc['SpAM', 'size'], 0)
    RE['NumSizeMTL'] = f(g.loc['MTL', 'size'], 0)
    RE['NumSizeGMCP'] = f(g.loc['GMCP', 'size'], 1)
    RE['NumSizeAGL'] = f(g.loc['Adaptive GL', 'size'], 1)
    RE['NumSizeChorus'] = f(g.loc['Chorus', 'size'], 1)
    RE['NumFPRGL'] = f(g.loc['GL', 'fpr'], 3)
    RE['NumFPRChorus'] = f(g.loc['Chorus', 'fpr'], 4)
    RE['NumTPRChorus'] = f(g.loc['Chorus', 'tpr'], 3)
    s1 = F[F.snr == 1].groupby(['method', 'kappa'])['f1'].mean()
    RE['NumFOneSnrOneKZero'] = f(s1[('Chorus', 0.0)])
    RE['NumFOneSnrOneKOne'] = f(s1[('Chorus', 1.0)])
    RE['NumFOneSnrOneAGLMin'] = f(s1['Adaptive GL'].min())
    RE['NumFOneSnrOneAGLMax'] = f(s1['Adaptive GL'].max())
    # wins over (setting, metric) combinations: best mean, and significance against every other method
    win = tie = lose = sig = 0
    for key, sub in d.groupby('key'):
        for met, low in (('f1', False), ('mcc', False), ('fee', True), ('r2', False), ('auprc', False)):
            w = sub.pivot_table(index='case', columns='method', values=met)
            m = w.mean()
            best = m.idxmin() if low else m.idxmax()
            if best == 'Chorus':
                win += 1
                if all(ttest_rel(w['Chorus'], w[o]).pvalue < .05 for o in w.columns if o != 'Chorus'):
                    sig += 1
            elif ttest_rel(w['Chorus'], w[best]).pvalue >= .05 or np.allclose(w['Chorus'], w[best]):
                tie += 1
            else:
                lose += 1
    cells = win + tie + lose
    RE['NumWinSim'], RE['NumCellsSim'] = str(win), str(cells)
    RE['NumWinSimPct'] = f'{100 * win / cells:.0f}\\%'
    RE['NumBestTiedAll'] = str(win + tie)
    NEW['NumSigSim'] = str(sig)
    NEW['NumLoseSim'] = str(lose)
    # robustness
    R = d[d.family == 'robust']
    RE['NumRobFOneMin'] = f(R[R.method == 'Chorus'].groupby('key')['f1'].mean().min())
    feebest = 0
    for key, sub in R.groupby('key'):
        w = sub.pivot_table(index='case', columns='method', values='fee').mean()
        feebest += int(w.idxmin() == 'Chorus')
    NEW['NumRobFEEBest'] = str(feebest)
    # ablation (leave one out)
    ab = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'ablation_a22_L.csv')).set_index(['row', 'column'])
    RE['NumAblReweightFOneAll'] = f(ab.loc[('Without reweighting', 'All'), 'f1_gain'], 1)
    RE['NumAblReweightFEEAll'] = f(ab.loc[('Without reweighting', 'All'), 'fee_gain'], 1) + '\\%'
    for col, tag in (('kappa=1', 'KOne'), ('kappa=0', 'KZero')):
        RE[f'NumAblBorrowFOne{tag}'] = f(ab.loc[('Without borrowing', col), 'f1_gain'], 1)
        RE[f'NumAblBorrowFEE{tag}'] = f(ab.loc[('Without borrowing', col), 'fee_gain'], 1) + '\\%'
    RE['NumAblBorrowFEEMixed'] = f(ab.loc[('Without borrowing', 'Mixed'), 'fee_gain'], 1) + '\\%'
    NEW['NumAblSelFOneAll'] = f(ab.loc[('Without the EB selection', 'All'), 'f1_gain'], 1)
    NEW['NumAblSelFOneMax'] = f(max(ab.loc[('Without the EB selection', c), 'f1_gain'] for c in ('kappa=0', 'Groups', 'q=64')), 1)
    NEW['NumAblRefitFEEAll'] = f(ab.loc[('Without the relaxed refit', 'All'), 'fee_gain'], 1) + '\\%'
    NEW['NumAblBasisFEEAll'] = f(ab.loc[('Without the choice of basis size', 'All'), 'fee_gain'], 1) + '\\%'
    NEW['NumOneSEFOneAll'] = f(-ab.loc[('1-SE rule instead of the selection', 'All'), 'f1_gain'], 1)
    NEW['NumOneSEFEEAll'] = f(ab.loc[('1-SE rule instead of the selection', 'All'), 'fee_gain'], 1) + '\\%'
    NEW['NumOneSEFEEMin'] = f(min(ab.loc[('1-SE rule instead of the selection', c), 'fee_gain']
                                  for c in ('All', 'kappa=0', 'kappa=1', 'Mixed', 'Rescaled', 'Groups', 'q=64')), 1) + '\\%'
    # candidate picks and pooled weights on the saved candidates
    a0k0, a1k1, f1pool, f1ch = [], [], [], []
    for fam, root, seeds in FAMILIES:
        for name in sorted(os.listdir(root)):
            if 'seed' not in name or int(name.rsplit('seed', 1)[1]) not in seeds:
                continue
            p = os.path.join(root, name, 'Chorus_a22', 'output.npz')
            if not os.path.exists(p):
                continue
            z = np.load(p)
            lab = labels(z)
            al = np.array([lab[i]['alpha'] if lab[i]['alpha'] is not None else -1 for i in z['pick']])
            if fam == 'formal' and 'kappa0.0_' in name:
                a0k0 += list(al == 0.)
            if fam == 'formal' and 'kappa1.0_' in name:
                a1k1 += list(al == 1.)
            if fam == 'sharing' and 'n_clusters' in name:
                rule = dict(kappa=.5, parsimony='groups', reference='first', shrink='eb')
                pk = apply_rule(z, rule, lambda c: c['kind'] == 'GL' or c['alpha'] == 1., 'closs')
                f1pool.append(score(os.path.join(root, name, 'case'), z, pk)['f1'])
                f1ch.append(score(os.path.join(root, name, 'case'), z, z['pick'])['f1'])
    RE['NumShareAlphaZeroAtKappaZero'] = f'{100 * np.mean(a0k0):.0f}\\%'
    RE['NumShareAlphaOneAtKappaOne'] = f'{100 * np.mean(a1k1):.0f}\\%'
    RE['NumShareGroupsFOnePool'] = f(np.mean(f1pool))
    RE['NumShareGroupsFOneChorus'] = f(np.mean(f1ch))
    # mean alpha of the selected fits without and with full overlap (main grid)
    import json
    for k, tag in ((0.0, 'KZero'), (1.0, 'KOne')):
        al = []
        for name in sorted(os.listdir(FAMILIES[0][1])):
            if f'kappa{k}_' in name and 'seed' in name and int(name.rsplit('seed', 1)[1]) <= 1005:
                z = np.load(os.path.join(FAMILIES[0][1], name, 'Chorus_a22', 'output.npz'))
                lab = labels(z)
                al += [lab[i]['alpha'] for i in z['pick'] if lab[i]['alpha'] is not None]
        NEW[f'NumAlpha{tag}'] = f(np.mean(al))
    # selections of Figure 3e
    sel = json.load(open(os.path.join(ROOT, 'results', 'v2', 'fig3_selection.json')))
    NEW['NumSelFPChorus'] = f(sel['Chorus']['fp_mean'], 1)
    NEW['NumSelFPAGL'] = f(sel['Adaptive GL']['fp_mean'], 1)
    NEW['NumSelFPMTL'] = f(sel['MTL']['fp_mean'], 0)
    pr = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sim_a22_per_response.csv'))
    cs = [c for c in pr.case.unique() if 'kappa0.5_rho0.7_snr1.0' in c and 'q16' in c]
    sf = pr[pr.case.isin(cs)].groupby('method')['f1'].mean()
    NEW['NumSelFOneChorus'] = f(sf['Chorus'])
    NEW['NumSelFOneGap'] = f(sf['Chorus'] - sf['Adaptive GL'])
    # robustness: largest FEE lead of another method
    gaps = []
    for key, sub in R.groupby('key'):
        w = sub.pivot_table(index='case', columns='method', values='fee').mean()
        gaps.append(w['Chorus'] - w.drop('Chorus').min())
    NEW['NumRobFEEGapMax'] = f(max(gaps), 4)
    # passes per response in a shared working set of 64 or more responses (timing3)
    import glob
    ps = []
    for q in (64, 256):
        for fl in glob.glob(os.path.join(ROOT, 'results', 'timing3', f'q{q}_chorus_batched_ws', 'Chorus', 'rep*', 'result.json')):
            ps.append(json.load(open(fl))['chorus']['passes'] / q)
    NEW['NumPassesShared'] = f(min(ps))
    # median per-response gains in the sharing scenarios (claim check)
    S = pr[pr.family == 'sharing']
    meds = []
    for case_set in S.groupby('case'):
        pass
    piv1 = S.pivot_table(index=['case', 'response'], columns='method', values='f1')
    piv2 = S.pivot_table(index=['case', 'response'], columns='method', values='fee')
    worst = min(min(np.median(piv1['Chorus'] - piv1[m]), np.median(1 - piv2['Chorus'] / piv2[m])) for m in piv1.columns if m != 'Chorus')
    print('smallest median gain over the sharing family (all scenarios pooled):', worst)
    lines = ['% amendment 22 numbers (tools/numbers_a22.py); simulation part']
    lines += [f'\\renewcommand{{\\{k}}}{{{v}}}' for k, v in RE.items()]
    for k, v in NEW.items():          # names that may exist in older number files
        lines += [f'\\providecommand{{\\{k}}}{{}}', f'\\renewcommand{{\\{k}}}{{{v}}}']
    real = os.path.join(ROOT, 'manuscript', 'numbers_a22_real.tex')
    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
