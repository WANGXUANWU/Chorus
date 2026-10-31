"""Numbers of the amendment 22 text, real-data part, written to manuscript/numbers_a22_real.tex.

Sources: results/v2/real_a22_metrics.csv (raw, Table 3), results/v2/matched_size.csv and
matched_size_per_response.csv (Figure 4), and the saved candidates of results/real_a22/C/Chorus_a22.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from bench.caseio import read_meta, load

DEP = ['depmap_c200'] + [f'depmap_c200_s{s}' for s in range(3101, 3106)]
OUT = os.path.join(ROOT, 'manuscript', 'numbers_a22_real.tex')
N = {}


def f(v, d=3):
    return f'{v:.{d}f}'


def main():
    raw = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'real_a22_metrics.csv'))
    raw = raw[raw.protocol == 'aligned']
    dep = raw[raw.case.isin(DEP)].groupby('method')[['r2_mean', 'size_median', 'rec10']].mean()
    b1, b2 = raw[raw.case == 'beijing'].set_index('method'), raw[raw.case == 'beijing_o2'].set_index('method')
    # model sizes as in Table 3: medians over all responses of the DepMap splits and of the two Beijing origins
    pr_raw = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'real_a22_per_response.csv'))
    pr_raw = pr_raw[pr_raw.protocol == 'aligned']
    dsz = pr_raw[pr_raw.case.isin(DEP)].groupby('method')['size'].median()
    bsz = pr_raw[pr_raw.case.isin(['beijing', 'beijing_o2'])].groupby('method')['size'].median()
    N['NumRawDepRtwoMTL'], N['NumRawDepSizeMTL'] = f(dep.loc['MTL', 'r2_mean']), f(dsz['MTL'], 0)
    dense = dep.loc[['SGL', 'SpAM', 'GL']]
    N['NumRawDepRtwoDense'] = f(dense.r2_mean.mean())
    N['NumRawDepSizeDenseMin'], N['NumRawDepSizeDenseMax'] = f(dsz[dense.index].min(), 0), f(dsz[dense.index].max(), 0)
    N['NumRawDepRtwoChorus'], N['NumRawDepSizeChorus'] = f(dep.loc['Chorus', 'r2_mean']), f(dsz['Chorus'], 0)
    N['NumRawDepRtwoAGL'], N['NumRawDepSizeAGL'] = f(dep.loc['Adaptive GL', 'r2_mean']), f(dsz['Adaptive GL'], 0)
    N['NumRawBjOneSpAM'], N['NumRawBjTwoSpAM'] = f(b1.loc['SpAM', 'r2_mean']), f(b2.loc['SpAM', 'r2_mean'])
    N['NumRawBjSizeSpAM'] = f(bsz['SpAM'], 0)
    N['NumRawBjOneChorus'], N['NumRawBjTwoChorus'] = f(b1.loc['Chorus', 'r2_mean']), f(b2.loc['Chorus', 'r2_mean'])
    N['NumRawBjSizeChorus'] = f(bsz['Chorus'], 0)
    N['NumRecTenChorus'] = f'{100 * dep.loc["Chorus", "rec10"]:.0f}\\%'
    N['NumRecTenBest'] = f'{100 * dep.drop("Chorus").rec10.max():.0f}\\%'
    N['NumRecTenBestName'] = dep.drop('Chorus').rec10.idxmax()
    # matched size
    m = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'matched_size.csv'))
    pr = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'matched_size_per_response.csv'))
    sig = 0
    for c, d in pr.groupby('case'):
        w = d.pivot_table(index='response', columns='method', values='r2_matched')
        ch = d.drop_duplicates('response').set_index('response')['r2_chorus']
        sig += int(all(ttest_rel(ch, w[k]).pvalue < .05 and ch.mean() > w[k].mean() for k in w.columns))
    words = dict(enumerate(['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'all eight']))
    N['NumMatchSigCases'] = words[sig]
    best = m.groupby('case').apply(lambda g: bool((g.r2_chorus >= g.r2_matched).all()))
    N['NumMatchBestCount'] = words[int(best.sum())]
    N['NumMatchDepBestCount'] = words[int(best[best.index.str.startswith('D')].sum())]
    # Figure 4a, bottom: cases in which Chorus is within 0.01 of the most accurate method for the largest share
    near_best = 0
    for c, d in pr.groupby('case'):
        w = d.pivot_table(index='response', columns='method', values='r2_matched')
        w['Chorus'] = d.drop_duplicates('response').set_index('response')['r2_chorus']
        share = w.ge(w.max(axis=1) - .01, axis=0).mean()
        near_best += int(share['Chorus'] > share.drop('Chorus').max())
    N['NumNearBestCount'] = words[near_best]
    # DepMap: the largest lead of a baseline in a split, and the six splits pooled per dependency
    dpm = m[m.case.str.startswith('D')]
    N['NumMatchDepLossMax'] = f'{max(0., -dpm.gain.min()):.4f}'
    dd = pr[pr.case.str.startswith('D')]
    per = dd.groupby(['method', 'response'])[['r2_chorus', 'r2_matched']].mean()
    pool = per.groupby(level=0).mean()
    near_m = ['GL', 'SGL', 'SpAM']
    N['NumMatchDepPoolChorus'] = f(pool.r2_chorus.iloc[0])
    N['NumMatchDepPoolNear'] = f(pool.loc[near_m, 'r2_matched'].max())
    pmax = max(ttest_rel(per.loc[k].r2_chorus, per.loc[k].r2_matched).pvalue for k in near_m)
    N['NumMatchDepPoolP'] = f'{np.ceil(pmax * 100) / 100:.2f}'
    bj = m[m.case.str.startswith('B')]
    N['NumMatchBjGainMin'], N['NumMatchBjGainMax'] = f(bj.gain.min()), f(bj.gain.max())
    N['NumMatchBjWinMin'], N['NumMatchBjWinMax'] = f'{100 * bj.share_chorus_better.min():.0f}\\%', f'{100 * bj.share_chorus_better.max():.0f}\\%'
    dp = m[m.case.str.startswith('D')]
    near = dp[dp.method.isin(['GL', 'SGL', 'SpAM'])]
    far = dp[dp.method.isin(['Adaptive GL', 'GMCP', 'MTL'])]
    N['NumMatchDepNearMin'], N['NumMatchDepNearMax'] = f(near.gain.min()), f(near.gain.max())
    N['NumMatchDepFarMin'], N['NumMatchDepFarMax'] = f(far.gain.min(), 2), f(far.gain.max(), 2)
    # the relaxed refit on Beijing: mean test R2 of the selected candidates, penalized and relaxed
    for case, tag in (('beijing', 'One'), ('beijing_o2', 'Two')):
        z = np.load(os.path.join(ROOT, 'results', 'real_a22', case, 'Chorus_a22', 'output.npz'))
        c5 = os.path.join(ROOT, 'results', 'cases', case)
        yte = load(c5, 'y', 'test', read_meta(c5))
        pen = z['pred_pen']
        r2p = 1 - ((yte - pen) ** 2).sum(0) / ((yte - yte.mean(0)) ** 2).sum(0)
        r2r = 1 - ((yte - z['pred']) ** 2).sum(0) / ((yte - yte.mean(0)) ** 2).sum(0)
        N[f'NumBjRefitPen{tag}'], N[f'NumBjRefitRel{tag}'] = f(r2p.mean()), f(r2r.mean())
    lines = ['% amendment 22 numbers, real-data part (tools/numbers_a22_real.py)']
    for k, v in N.items():
        lines += [f'\\providecommand{{\\{k}}}{{}}', f'\\renewcommand{{\\{k}}}{{{v}}}']
    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(l for l in lines if l.startswith('\\renew')))


if __name__ == '__main__':
    main()
