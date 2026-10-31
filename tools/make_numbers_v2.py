"""Number macros of the revised paper (manuscript/numbers.tex), computed from results/v2 and the
timing logs. Every number quoted in the text comes from here.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

ACC = ['Chorus', 'adaptive GL', 'gMCP (grpreg)', 'GL + 1-SE', 'SGL (sparsegl)', 'SpAM (SAM)', 'GL (skglm)', 'MTL (biglasso)']
NAME = {'GL (adelie)': 'GL', 'adaptive GL': 'adaptive GL', 'gMCP (grpreg)': 'gMCP', 'GL + 1-SE': 'GL with the 1-SE rule',
        'SGL (sparsegl)': 'SGL', 'SpAM (SAM)': 'SAM', 'GL (skglm)': 'GL', 'MTL (biglasso)': 'MTL'}
MET = [('f1', False), ('mcc', False), ('fee', True), ('r2', False), ('auprc', False)]
out = {}


def put(k, v):
    out[k] = v


def f2(x):
    return f'{x:.2f}'


def f3(x):
    return f'{x:.3f}'


def load_sim(f):
    d = pd.read_csv(f)
    keep = (d.protocol == 'aligned') | d.method.isin(['Chorus'])
    d = d[keep & d.method.isin(ACC)].copy()
    d['sit'] = d.case.str.replace(r'_seed\d+', '', regex=True)
    return d


def wins(d):
    tot = win = 0
    for _, g in d.groupby('sit'):
        for m, low in MET:
            v = g.groupby('method')[m].mean().round(3)
            tot += 1
            win += int(v.get('Chorus') == (v.min() if low else v.max()))
    return win, tot


F = load_sim(os.path.join(ROOT, 'results', 'v2', 'sim_formal.csv'))
Rb = load_sim(os.path.join(ROOT, 'results', 'v2', 'sim_robust.csv'))
S = load_sim(os.path.join(ROOT, 'results', 'v2', 'sim_sharing.csv'))
put('NumSimCases', '100')
mF = F.groupby('method')[['f1', 'mcc', 'fee', 'r2', 'auprc', 'size', 'fpr', 'tpr']].mean()
others = [m for m in ACC if m != 'Chorus']
best_f1 = mF.loc[others, 'f1'].idxmax()
put('NumFOneChorus', f2(mF.loc['Chorus', 'f1'])); put('NumFOneBest', f2(mF.loc[best_f1, 'f1']))
put('NumFOneBestMethod', NAME[best_f1])
put('NumMCCChorus', f2(mF.loc['Chorus', 'mcc'])); put('NumMCCBest', f2(mF.loc[others, 'mcc'].max()))
best_fee = mF.loc[others, 'fee'].idxmin()
put('NumFEEChorus', f3(mF.loc['Chorus', 'fee'])); put('NumFEEBest', f3(mF.loc[best_fee, 'fee']))
put('NumFEEBestMethod', NAME[best_fee])
put('NumFEEGain', f"{100 * (1 - mF.loc['Chorus', 'fee'] / mF.loc[best_fee, 'fee']):.0f}\\%")
put('NumRtwoChorus', f3(mF.loc['Chorus', 'r2'])); put('NumRtwoBest', f3(mF.loc[others, 'r2'].max()))
put('NumAUPRCChorus', f3(mF.loc['Chorus', 'auprc'])); put('NumAUPRCBest', f3(mF.loc[others, 'auprc'].max()))
put('NumSizeChorus', f"{mF.loc['Chorus', 'size']:.1f}"); put('NumSizeAGL', f"{mF.loc['adaptive GL', 'size']:.1f}")
put('NumSizeGL', f"{mF.loc['GL (skglm)', 'size']:.0f}"); put('NumSizeGMCP', f"{mF.loc['gMCP (grpreg)', 'size']:.1f}")
put('NumSizeSAM', f"{mF.loc['SpAM (SAM)', 'size']:.0f}"); put('NumSizeMTL', f"{mF.loc['MTL (biglasso)', 'size']:.0f}")
put('NumFPRChorus', f3(mF.loc['Chorus', 'fpr'])); put('NumFPRGL', f3(mF.loc['GL (skglm)', 'fpr']))
put('NumTPRChorus', f3(mF.loc['Chorus', 'tpr'])); put('NumTPRGL', f3(mF.loc['GL (skglm)', 'tpr']))
put('NumFOneOneSE', f2(mF.loc['GL + 1-SE', 'f1'])); put('NumFEEOneSE', f3(mF.loc['GL + 1-SE', 'fee']))
put('NumFOneGMCP', f2(mF.loc['gMCP (grpreg)', 'f1'])); put('NumFEEGMCP', f3(mF.loc['gMCP (grpreg)', 'fee']))
put('NumFOneAGL', f2(mF.loc['adaptive GL', 'f1'])); put('NumFEEAGL', f3(mF.loc['adaptive GL', 'fee']))
wf, tf = wins(F); wr, tr = wins(Rb); ws, ts = wins(S)
put('NumWinFormal', str(wf)); put('NumCellsFormal', str(tf))
put('NumWinRobust', str(wr)); put('NumCellsRobust', str(tr))
put('NumWinSharing', str(ws)); put('NumCellsSharing', str(ts))
put('NumWinSim', str(wf + wr + ws)); put('NumCellsSim', str(tf + tr + ts))
put('NumWinSimPct', f"{100 * (wf + wr + ws) / (tf + tr + ts):.0f}\\%")
# per-metric wins on the formal grid
for m, low in MET:
    w = 0
    for _, g in F.groupby('sit'):
        v = g.groupby('method')[m].mean().round(3)
        w += int(v.get('Chorus') == (v.min() if low else v.max()))
    put('NumWinFormal' + {'f1': 'FOne', 'mcc': 'MCC', 'fee': 'FEE', 'r2': 'Rtwo', 'auprc': 'AUPRC'}[m], str(w))
g1 = F[F.snr == 1.].groupby(['method', 'kappa']).f1.mean()
put('NumFOneSnrOneKZero', f2(g1.loc[('Chorus', 0.)])); put('NumFOneSnrOneKOne', f2(g1.loc[('Chorus', 1.)]))
put('NumFOneSnrOneAGLMin', f2(g1.loc['adaptive GL'].min())); put('NumFOneSnrOneAGLMax', f2(g1.loc['adaptive GL'].max()))
C = json.load(open(os.path.join(ROOT, 'results', 'v2', 'components_v2.json')))
for shp in ('sine', 'step'):
    put('NumComp' + shp.capitalize() + 'Chorus', f2(C['err']['Chorus'][shp]))
    put('NumComp' + shp.capitalize() + 'GL', f2(C['err']['GL (skglm)'][shp]))
mR = Rb.groupby('method')[['f1', 'fee']].mean()
put('NumRobFOneChorus', f2(mR.loc['Chorus', 'f1'])); put('NumRobFOneAGL', f2(mR.loc['adaptive GL', 'f1']))
put('NumRobFOneMin', f2(Rb[Rb.method == 'Chorus'].groupby('sit').f1.mean().min()))
# sharing families
Sall = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sim_sharing.csv'))
fam = {'Mixed': Sall.snr_range.notna(), 'Rescaled': Sall.scale_range.notna(), 'Groups': Sall.n_clusters.notna(),
       'NoShare': Sall.q == 64, 'Overlap': (Sall.kappa == 1) & (Sall.q == 16)}
for k, sel in fam.items():
    d = Sall[sel]
    g = d.groupby('method')[['f1', 'fee']].mean()
    for m, tag in (('Chorus', 'Chorus'), ('adaptive GL', 'AGL'), ('pooled weights', 'Pool'),
                   ('Chorus, no scale normalization', 'NoScale')):
        if m in g.index:
            put(f'NumShare{k}FOne{tag}', f2(g.loc[m, 'f1']))
            put(f'NumShare{k}FEE{tag}', f3(g.loc[m, 'fee']))
ch = Sall[Sall.method == 'Chorus']
put('NumShareAlphaZeroAtKappaZero', f"{100 * ch[(ch.kappa == 0) & (ch.q == 16)].share_a0.mean():.0f}\\%")
put('NumShareAlphaOneAtKappaOne', f"{100 * ch[(ch.kappa == 1) & (ch.q == 16)].share_a1.mean():.0f}\\%")
put('NumShareDThree', f"{100 * ch.share_d3.mean():.0f}\\%")

# fixed descriptive numbers
put('NumOOCMaxN', '400{,}000')
put('NumDepLines', '1{,}103')
put('NumBjResp', '72')

# real data (when available)
rp = os.path.join(ROOT, 'results', 'v2', 'real_metrics.csv')
if os.path.exists(rp):
    Rl = pd.read_csv(rp)
    Rl = Rl[(Rl.protocol == 'aligned') | Rl.method.isin(['Chorus'])]
    for case, tag in (('depmap_c200', 'Dep'), ('beijing', 'Bj')):
        d = Rl[(Rl.case == case) & Rl.method.isin(ACC)].set_index('method')
        if 'Chorus' not in d.index:
            continue
        put(f'Num{tag}RtwoChorus', f3(d.loc['Chorus', 'r2_mean']))
        b = d.drop('Chorus').r2_mean.idxmax()
        put(f'Num{tag}RtwoBest', f3(d.loc[b, 'r2_mean'])); put(f'Num{tag}RtwoBestMethod', NAME[b])
        put(f'Num{tag}SizeChorus', f"{d.loc['Chorus', 'size_median']:.0f}"); put(f'Num{tag}SizeBest', f"{d.loc[b, 'size_median']:.0f}")

sys.path.insert(0, os.path.join(ROOT, 'tools'))
from numbers_extra_v3 import extra
extra(ROOT, put, ACC)
from numbers_efficiency_v3 import efficiency
efficiency(ROOT, put)

lines = ['% Auto-generated by tools/make_numbers_v2.py. Do not edit by hand.']
lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in out.items()]
old = os.path.join(ROOT, 'manuscript', 'numbers.tex')
keep = []
if os.path.exists(old):                     # keep macros that are not regenerated (efficiency until rerun)
    for l in open(old, encoding='utf-8'):
        if l.startswith('\\newcommand{\\'):
            k = l.split('{\\')[1].split('}')[0]
            if k not in out:
                keep.append(l.rstrip('\n'))
open(old, 'w', encoding='utf-8').write('\n'.join(lines + keep) + '\n')
print(len(out), 'macros;', len(keep), 'kept from the previous file')
