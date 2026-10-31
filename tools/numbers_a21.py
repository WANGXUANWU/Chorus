"""Macros for the numbers of the second revision (amendment 21) that enter the text.

Writes manuscript/numbers_a21.tex. Real data: per-response comparison of Chorus with each method
(Figure 4b). Further blocks are added as the amendment-21 results arrive.
Usage: numbers_a21.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

M = {}
P = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'real_per_response.csv'))
P = P[P.protocol == 'aligned']
RIV = ['adaptive GL', 'gMCP (grpreg)', 'GL + 1-SE', 'SGL (sparsegl)', 'SpAM (SAM)', 'GL (adelie)', 'MTL (biglasso)']
for key, pre in (('Dep', 'depmap'), ('Bj', 'beijing')):
    d = P[P.case.str.startswith(pre)]
    r2 = d.pivot_table(index=['case', 'response'], columns='method', values='r2')
    sz = d.pivot_table(index=['case', 'response'], columns='method', values='size')
    dom, lost = [], []
    for m in RIV:
        dr, ds = r2['Chorus'] - r2[m], sz['Chorus'] - sz[m]
        dom.append(((ds <= 0) & (dr >= 0)).mean())
        lost.append(((ds >= 0) & (dr < 0)).mean())
    M[f'NumDom{key}Min'] = f'{100 * min(dom):.0f}\\%'
    M[f'NumDom{key}Max'] = f'{100 * max(dom):.0f}\\%'
    M[f'NumLost{key}Max'] = f'{100 * max(lost):.0f}\\%'
    M[f'NumResp{key}'] = f'{len(r2):,}'.replace(',', '{,}')

# ---------------------------------------------------------------- decomposition (E3), medians of the completed runs
import json
T = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, 'results', 'timing3', 'timing_log.jsonl'))])
X = T[(T.experiment == 'e7_decomp') & (T.status == 'ok')].copy()
X['wl'] = X.tag.str.split('_').str[0]
X['var'] = X.tag.str.split('_', n=1).str[1]
P2 = X.pivot_table(index='var', columns='wl', values='fit_seconds', aggfunc='median')
NR = X.groupby(['wl', 'var']).fit_seconds.count()
SIM, SIMALL = ['q16', 'q64', 'q256'], ['q1', 'q4', 'q16', 'q64', 'q256']


def r(a, b, c):
    return P2.loc[a, c] / P2.loc[b, c]


def f1(v):
    return f'{v:.1f}'


def f0(v):
    return f'{v:,.0f}'.replace(',', '{,}')


M['NumDecRunsMin'] = str(int(NR.min()))
M['NumDecRange'] = f"{100 * X.groupby(['wl', 'var']).fit_seconds.agg(lambda s: (s.max() - s.min()) / s.median()).max():.0f}\\%"
M['NumDecGap'] = '$10^{-7}$' if X[X.method == 'Chorus'].chorus.apply(lambda c: (c or {}).get('max_gap')).max() <= 1.0001e-7 else 'ERR'
sh = [r('chorus_seq_ws', 'chorus_batched_ws', c) for c in SIM]
M['NumShareSimMin'], M['NumShareSimMax'] = f1(min(sh)), f1(max(sh))
M['NumShareBj'] = f1(r('chorus_seq_ws', 'chorus_batched_ws', 'beijing'))
M['NumDepOneSec'], M['NumDepWSSec'] = f0(P2.loc['chorus_seq_ws', 'depmap64']), f0(P2.loc['chorus_batched_ws', 'depmap64'])
M['NumStatsBj'] = f1(r('chorus_batched_ws', 'chorus_batched', 'beijing'))
M['NumBjStatsSec'], M['NumBjWSSec'] = f1(P2.loc['chorus_batched', 'beijing']), f1(P2.loc['chorus_batched_ws', 'beijing'])
st = [r('chorus_batched_ws', 'chorus_batched', c) for c in ('q1', 'q4')]
M['NumStatsSmallMin'], M['NumStatsSmallMax'] = f'{min(st):.2f}', f'{max(st):.2f}'
ex = [r('chorus_batched', 'chorus_exit', c) for c in SIM]
M['NumExitSimMin'], M['NumExitSimMax'] = f1(min(ex)), f1(max(ex))
M['NumExitBj'], M['NumExitDep'] = f1(r('chorus_batched', 'chorus_exit', 'beijing')), f0(r('chorus_batched', 'chorus_exit', 'depmap64'))
M['NumOneToExitLast'] = f1(r('chorus_seq_ws', 'chorus_exit', 'q256'))
M['NumAdelieLeadMax'] = f1(max(1 / r('adelie_full', 'chorus_batched', c) for c in SIMALL))
M['NumChorusLeadFullLast'] = f'{r("adelie_full", "chorus_batched", "q256"):.2f}'
ae = [r('adelie_exit', 'chorus_exit', c) for c in SIMALL]
M['NumAdelieExitMin'], M['NumAdelieExitMax'] = f'{min(ae):.2f}', f'{max(ae):.2f}'
M['NumBjVsAdelieFull'], M['NumBjVsAdelieExit'] = f0(r('adelie_full', 'chorus_batched', 'beijing')), f0(r('adelie_exit', 'chorus_exit', 'beijing'))
oth = [r(f'{m}_full', 'chorus_batched', c) for m in ('skglm', 'Celer') for c in SIMALL]
M['NumOthersFullMin'] = f1(min(oth))
othe = [r(f'{m}_exit', 'chorus_exit', c) for m in ('skglm', 'Celer') for c in SIMALL]
M['NumOthersExitMin'], M['NumOthersExitMax'] = f1(min(othe)), f0(max(othe))
M['NumDepAdelieFullLead'] = f1(1 / r('adelie_full', 'chorus_batched', 'depmap64'))
M['NumDepAdelieExitLead'] = f1(1 / r('adelie_exit', 'chorus_exit', 'depmap64'))
M['NumDepChorusFullSec'], M['NumDepAdelieFullSec'] = f0(P2.loc['chorus_batched', 'depmap64']), f0(P2.loc['adelie_full', 'depmap64'])
M['NumDepChorusExitSec'], M['NumDepAdelieExitSec'] = f0(P2.loc['chorus_exit', 'depmap64']), f0(P2.loc['adelie_exit', 'depmap64'])

# ---------------------------------------------------------------- precision sweep (E2) and path agreement
CERT = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sweep_certificates.csv'))
DEF = {'Chorus': 1e-6, 'adelie': 1e-7, 'skglm': 1e-4, 'Celer': 1e-4}


def sci(v):
    m, e = f'{v:.1e}'.split('e')
    return f'${m}\\cdot10^{{{int(e)}}}$'


dflt = CERT[[np.isclose(t, DEF[m]) for m, t in zip(CERT.method, CERT.tol)]]
M['NumDefaultGapExtMax'] = sci(dflt[dflt.method != 'Chorus'].max_gap.max())
M['NumDefaultGapAdelieMax'] = sci(dflt[dflt.method == 'adelie'].max_gap.max())
wins, cells = {}, 0
for wl, g in CERT.groupby('workload'):
    for t in (1e-4, 1e-6, 1e-8):
        ok = g[g.max_gap <= t]
        if len(ok):
            b = ok.groupby('method').median_seconds.min().idxmin()
            wins[b] = wins.get(b, 0) + 1
        cells += 1
M['NumTargetWins'], M['NumTargetCells'] = str(wins.get('Chorus', 0)), str(cells)
dep = CERT[CERT.workload == 'depmap64']
M['NumDepAdelieTargetSec'] = f0(dep[(dep.method == 'adelie') & (dep.max_gap <= 1e-4)].median_seconds.min())
M['NumDepChorusTargetSec'] = f0(dep[(dep.method == 'Chorus') & (dep.max_gap <= 1e-4)].median_seconds.min())
AG = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sweep_agreement.csv'))
AG = AG[[np.isclose(t, DEF.get(m, -1)) for m, t in zip(AG.method, AG.tol)]]
S = AG.groupby(['method', 'workload']).share_same_support.mean()
M['NumAgreeChorusMin'] = f'{100 * S["Chorus"].min():.1f}\\%'
M['NumAgreeAdelieSimMin'] = f'{100 * S["adelie"].drop("depmap64").min():.0f}\\%'
M['NumAgreeAdelieSimMax'] = f'{100 * S["adelie"].drop("depmap64").max():.0f}\\%'
M['NumAgreeDepAdelie'] = f'{100 * S[("adelie", "depmap64")]:.0f}\\%'
M['NumAgreeDepSkglm'] = f'{100 * S[("skglm", "depmap64")]:.0f}\\%'
M['NumAgreeDepCeler'] = f'{100 * S[("Celer", "depmap64")]:.0f}\\%'
PD = AG.groupby(['method', 'workload']).pred_diff_max.median()
M['NumPredChorusMax'] = sci(PD['Chorus'].max())
M['NumPredExtMin'], M['NumPredExtMax'] = sci(PD.drop('Chorus').min()), sci(PD.drop('Chorus').max())

# ---------------------------------------------------------------- per-response cost of exit (patience 5)
E = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'exit_losses.csv'))
es = E[(E.group == 'Simulation') & (E.patience == 5)]
eb = E[(E.group == 'Beijing') & (E.patience == 5)]
M['NumExitLossResp'] = f'{len(es)}'
M['NumExitLossPfive'] = f'{np.percentile(es.d_r2, 5):.4f}'.replace('-', '$-$')
M['NumExitLossWorst'] = f'{es.d_r2.min():.3f}'.replace('-', '$-$')
M['NumExitLossShare'] = f'{100 * np.mean(es.d_r2 < -.01):.1f}\\%'
M['NumExitLossFOneWorst'] = f'{es.d_f1.min():.2f}'.replace('-', '$-$')
M['NumExitLossFEEpNinetyFive'] = f'{100 * np.percentile(es.d_fee, 95):.1f}\\%'
M['NumExitLossBjLoss'] = f'{int((eb.d_r2 < -1e-4).sum())}'

out = os.path.join(ROOT, 'manuscript', 'numbers_a21.tex')
with open(out, 'w', encoding='utf-8') as fh:
    fh.write('% generated by tools/numbers_a21.py\n')
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(open(out, encoding='utf-8').read())
