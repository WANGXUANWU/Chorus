"""Additional number macros of the final revision (imported by make_numbers_v2.py through put()).

Ablation effects (Table 5), best-or-tied counts over all simulation cells, the sharing families
against the variant without borrowing, and the real-data frontier and matched-size results."""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

PCT = chr(92) + '%'


def extra(ROOT, put, ACC):
    v2 = os.path.join(ROOT, 'results', 'v2')
    # best or statistically tied over all simulation cells (paired t-test over seeds, 5%)
    tot = tie = strict = 0
    for f in ('sim_formal', 'sim_robust', 'sim_sharing'):
        d = pd.read_csv(os.path.join(v2, f + '.csv'))
        d = d[(d.protocol == 'aligned') & d.method.isin(ACC)].copy()
        d['sit'] = d.case.str.replace(r'_seed\d+', '', regex=True)
        for _, g in d.groupby('sit'):
            for m, low in (('f1', 0), ('mcc', 0), ('fee', 1), ('r2', 0), ('auprc', 0)):
                W = g.pivot_table(index='seed', columns='method', values=m)
                mu = W.mean().round(3)
                best = mu.idxmin() if low else mu.idxmax()
                tot += 1
                strict += int(mu['Chorus'] == mu[best])
                ok = mu['Chorus'] == mu[best]
                if not ok:
                    dd = (W['Chorus'] - W[best]).dropna()
                    ok = len(dd) > 1 and ttest_rel(W['Chorus'], W[best], nan_policy='omit').pvalue >= .05
                tie += int(ok)
    put('NumCellsAll', str(tot)); put('NumBestAll', str(strict)); put('NumBestTiedAll', str(tie))
    put('NumBestAllPct', f'{100 * strict / tot:.0f}' + PCT); put('NumBestTiedAllPct', f'{100 * tie / tot:.0f}' + PCT)

    # ablation effects (Table 5): F1 loss in points and FEE increase in percent
    F = pd.read_csv(os.path.join(v2, 'sim_formal.csv'))
    S = pd.read_csv(os.path.join(v2, 'sim_sharing.csv'))
    keep = lambda d: d[(d.protocol == 'aligned') | (d.method == 'd = 5 only')]
    F, S = keep(F), keep(S)
    groups = {'All': F, 'KZero': F[F.kappa == 0], 'KOne': F[F.kappa == 1], 'Mixed': S[S.snr_range.notna()],
              'Rescaled': S[S.scale_range.notna()], 'Groups': S[S.n_clusters.notna()], 'NoShare': S[S.q == 64]}
    tags = {'no borrowing': 'Borrow', 'GL only': 'Reweight', 'pooled weights': 'Pooled', 'fixed alpha 0.5': 'Fixed',
            'Chorus, no scale normalization': 'Scale', 'd = 5 only': 'Basis', 'no cross-fitting': 'Crossfit',
            'no shrinkage': 'Shrink', 'full pooling': 'FullPool', 'no refit': 'Refit'}
    for meth, tag in tags.items():
        for gk, d in groups.items():
            c = d[d.method == 'Chorus'].set_index('case')
            v = d[d.method == meth].set_index('case')
            j = c.index.intersection(v.index)
            if not len(j):
                continue
            put(f'NumAbl{tag}FOne{gk}', f"{100 * (c.loc[j, 'f1'].mean() - v.loc[j, 'f1'].mean()):.1f}")
            put(f'NumAbl{tag}FEE{gk}', f"{100 * (v.loc[j, 'fee'].mean() / c.loc[j, 'fee'].mean() - 1):.1f}" + PCT)

    # sharing families: Chorus against the variant without borrowing
    Sall = pd.read_csv(os.path.join(v2, 'sim_sharing.csv'))
    Sall = Sall[Sall.protocol == 'aligned']
    fam = {'Mixed': Sall.snr_range.notna(), 'Rescaled': Sall.scale_range.notna(), 'Groups': Sall.n_clusters.notna(),
           'NoShare': Sall.q == 64, 'Overlap': (Sall.kappa == 1) & (Sall.q == 16)}
    for k, sel in fam.items():
        g = Sall[sel].groupby('method')[['f1', 'fee']].mean()
        for m, tag in (('no borrowing', 'NoBorrow'), ('adaptive GL', 'AGL')):
            if m in g.index:
                put(f'NumShare{k}FOne{tag}', f'{g.loc[m, "f1"]:.2f}')
                put(f'NumShare{k}FEE{tag}', f'{g.loc[m, "fee"]:.3f}')

    # real data: frontier and matched size
    R = pd.read_csv(os.path.join(v2, 'real_metrics.csv'))
    R = R[R.protocol == 'aligned']
    FR = json.load(open(os.path.join(v2, 'real_frontier.json')))
    cases = ['depmap_c200'] + [f'depmap_c200_s{s}' for s in range(3101, 3106)] + ['beijing', 'beijing_o2']
    mp = {m: ('GL (adelie)' if m == 'GL (skglm)' else m) for m in ACC}
    beat = dom = n = 0
    for case in cases:
        c = R[(R.case == case) & (R.method == 'Chorus')].iloc[0]
        for m in ACC[1:]:
            d = R[(R.case == case) & (R.method == mp[m])]
            if not len(d):
                continue
            d = d.iloc[0]
            n += 1
            beat += int(d.size_median <= c.size_median and d.r2_mean >= c.r2_mean)
            dom += int(d.size_median >= c.size_median and d.r2_mean <= c.r2_mean)
    put('NumRealPairs', str(n)); put('NumRealBeatChorus', str(beat)); put('NumRealDominated', str(dom))
    gains, wins = {}, {}
    for case in cases:
        mm = FR[case]['matched']
        g = np.array(mm['chorus_r2']) - np.array(mm['gl_r2'])
        gains[case], wins[case] = g.mean(), 100 * np.mean(g > 0)
    dep = [c for c in cases if c.startswith('depmap')]
    put('NumMatchBjOne', f"{gains['beijing']:.3f}"); put('NumMatchBjTwo', f"{gains['beijing_o2']:.3f}")
    put('NumMatchBjOneWin', f"{wins['beijing']:.0f}" + PCT); put('NumMatchBjTwoWin', f"{wins['beijing_o2']:.0f}" + PCT)
    put('NumMatchDepMin', f'{min(gains[c] for c in dep):.3f}'); put('NumMatchDepMax', f'{max(gains[c] for c in dep):.3f}')
    put('NumMatchDepWinMin', f'{min(wins[c] for c in dep):.0f}' + PCT); put('NumMatchDepWinMax', f'{max(wins[c] for c in dep):.0f}' + PCT)
    g = lambda case, meth, col: R[(R.case == case) & (R.method == meth)][col].iloc[0]
    put('NumDepRtwoMTL', f"{g('depmap_c200', 'MTL (biglasso)', 'r2_mean'):.3f}")
    put('NumDepSizeMTL', f"{g('depmap_c200', 'MTL (biglasso)', 'size_median'):.0f}")
    put('NumDepSizeGL', f"{g('depmap_c200', 'GL (adelie)', 'size_median'):.0f}")
    put('NumDepRtwoGL', f"{g('depmap_c200', 'GL (adelie)', 'r2_mean'):.3f}")
    put('NumBjRtwoSpAM', f"{g('beijing', 'SpAM (SAM)', 'r2_mean'):.3f}")
    put('NumBjSizeSpAM', f"{g('beijing', 'SpAM (SAM)', 'size_median'):.0f}")
    put('NumBjRtwoNoRefit', f"{g('beijing', 'no refit', 'r2_mean'):.3f}")
    put('NumDepRtwoNoRefit', f"{g('depmap_c200', 'no refit', 'r2_mean'):.3f}")
    put('NumBjTwoRtwoChorus', f"{g('beijing_o2', 'Chorus', 'r2_mean'):.3f}")
    put('NumBjTwoRtwoSpAM', f"{g('beijing_o2', 'SpAM (SAM)', 'r2_mean'):.3f}")
    conf = [c for c in dep if c != 'depmap_c200']
    cr = R[R.case.isin(conf) & (R.method == 'Chorus')].r2_mean
    put('NumConfRtwoChorus', f'{cr.mean():.3f}'); put('NumConfRtwoChorusSD', f'{cr.std(ddof=1):.3f}')
    put('NumConfSizeChorus', f"{R[R.case.isin(conf) & (R.method == 'Chorus')].size_median.mean():.0f}")
    put('NumConfSizeGL', f"{R[R.case.isin(conf) & (R.method == 'GL (adelie)')].size_median.mean():.0f}")
    put('NumConfRtwoGL', f"{R[R.case.isin(conf) & (R.method == 'GL (adelie)')].r2_mean.mean():.3f}")
    rec = R[(R.case == 'depmap_c200')].set_index('method')
    put('NumDepRecTenChorus', f"{100 * rec.loc['Chorus', 'rec10']:.0f}" + PCT)
    put('NumDepRecTenBest', f"{100 * rec.loc[[mp[m] for m in ACC[1:]], 'rec10'].max():.0f}" + PCT)
