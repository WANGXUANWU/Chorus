"""Number macros of the efficiency section (called by make_numbers_v2.py through put()).

Everything comes from the timing logs (results/timing for skglm and Celer of the first version,
results/timing2 for the revision), the certificates, the store builds, the early-termination study
and the update-strategy grid."""
import json
import os

import numpy as np
import pandas as pd

TIMES = chr(92) + 'times'


def _log(path):
    return pd.DataFrame([json.loads(l) for l in open(path)]) if os.path.exists(path) else pd.DataFrame()


def _tmin(d):
    d = d[d.status == 'ok']
    return float(d.fit_seconds.min()) if len(d) else np.nan


def efficiency(ROOT, put):
    old = _log(os.path.join(ROOT, 'results', 'timing', 'timing_log.jsonl'))
    new = _log(os.path.join(ROOT, 'results', 'timing2', 'timing_log.jsonl'))
    once = new.experiment.isin(['e3_budget', 'e3_stress'])
    new = pd.concat([new[~once], new[once].drop_duplicates(subset=['experiment', 'tag'], keep='last')])
    cert = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'certificates.csv'))
    g = lambda tag, v, exp='e1_gl': _tmin(new[(new.experiment == exp) & (new.tag == f'{tag}_{v}')])
    o = lambda tag, m, exp: _tmin(old[(old.experiment == exp) & (old.tag == tag) & (old.method == m)])
    gap = lambda suffix: cert[cert.tag.str.endswith(suffix)].max_gap.max()
    ext_gap = {'skglm': cert[cert.method == 'skglm'].max_gap.max(), 'Celer': cert[cert.method == 'Celer'].max_gap.max(),
               'adelie': gap('_adelie'), 'adelie_tight': gap('_adelie_tight') if cert.tag.str.endswith('_adelie_tight').any() else np.nan}
    precise = [k for k, v in ext_gap.items() if np.isfinite(v) and v <= 1e-4]

    def sci(x):
        e = int(np.floor(np.log10(x))); m = round(x / 10 ** e)
        if m >= 10:
            m, e = 1, e + 1
        return f'{m}' + chr(92) + 'cdot10^{' + str(e) + '}'
    put('NumGapChorus', '$' + sci(max(gap('_chorus_seq'), gap('_chorus_batched'), gap('_chorus_batched_stop'))) + '$')
    put('NumGapAdelie', '$' + sci(ext_gap['adelie']) + '$')
    put('NumGapAdelieDep', '$' + sci(cert[cert.tag == 'depmap64_adelie'].max_gap.max()) + '$')
    if np.isfinite(ext_gap['adelie_tight']):
        put('NumGapAdelieTight', '$' + sci(ext_gap['adelie_tight']) + '$')
    put('NumGapExternalPrecise', '$' + sci(max(ext_gap[k] for k in precise)) + '$')
    sp = {}
    for tag, oexp in [(f'q{q}', 'q_scaling') for q in (1, 4, 16, 64, 256)] + [('depmap64', 'real'), ('beijing', 'real')]:
        ext = {'skglm': o(tag, 'skglm', oexp), 'Celer': o(tag, 'Celer', oexp), 'adelie': g(tag, 'adelie'),
               'adelie_tight': g(tag, 'adelie_tight')}
        cx = g(tag, 'chorus_batched_stop')
        sp[tag] = (np.nanmin([ext[k] for k in precise]) / cx, ext['adelie'] / cx, ext, cx)
    simq = [f'q{q}' for q in (1, 4, 16, 64, 256)]
    put('NumSpeedPreciseMin', f"{min(sp[t][0] for t in simq):.1f}"); put('NumSpeedPreciseMax', f"{max(sp[t][0] for t in simq):.1f}")
    put('NumSpeedPreciseQLast', f"{sp['q256'][0]:.1f}")
    put('NumSpeedAdelieQOne', f"{sp['q1'][1]:.1f}"); put('NumSpeedAdelieQLast', f"{sp['q256'][1]:.1f}")
    put('NumSpeedAdelieQSixteen', f"{sp['q16'][1]:.1f}")
    put('NumDepExit', f"{sp['depmap64'][3]:.0f}"); put('NumDepAdelie', f"{sp['depmap64'][2]['adelie']:.0f}")
    put('NumDepSkglm', f"{sp['depmap64'][2]['skglm']:,.0f}".replace(',', '{,}'))
    if np.isfinite(sp['depmap64'][2]['adelie_tight']):
        put('NumDepAdelieTight', f"{sp['depmap64'][2]['adelie_tight']:.0f}")
    put('NumDepSpeedPrecise', f"{sp['depmap64'][0]:.1f}")
    put('NumBjExit', f"{sp['beijing'][3]:.1f}"); put('NumBjAdelie', f"{sp['beijing'][2]['adelie']:.0f}")
    put('NumBjSpeedAdelie', f"{sp['beijing'][1]:.0f}")
    put('NumDepFullPath', f"{g('depmap64', 'chorus_seq'):,.0f}".replace(',', '{,}'))
    # two-stage estimator
    ts = {t: (g(t, 'full_seq', 'e1_full'), g(t, 'full_batched', 'e1_full'), g(t, 'full_shared', 'e1_full')) for t in sp}
    put('NumTwoStageSpeedMin', f"{min(ts[t][0] / ts[t][2] for t in simq):.1f}")
    put('NumTwoStageSpeedMax', f"{max(ts[t][0] / ts[t][2] for t in simq):.1f}")
    put('NumTwoStageBj', f"{ts['beijing'][0] / ts['beijing'][2]:.0f}")
    put('NumTwoStageDepShared', f"{ts['depmap64'][2]:.0f}"); put('NumTwoStageDepSeq', f"{ts['depmap64'][0]:.0f}")
    # out of core
    e2 = new[new.experiment == 'e2_rows']
    ch = e2[(e2.method == 'Chorus') & (e2.status == 'ok')].sort_values('size')
    put('NumOOCTimeMax', f"{ch.fit_seconds.iloc[-1] / 60:.0f}")
    put('NumOOCPeakMax', f"{ch.peak_wset.max() / 2 ** 30:.2f}")
    put('NumOOCPeakMin', f"{ch.peak_wset.min() / 2 ** 30:.2f}")
    put('NumOOCRatio', f"{8.0e9 / 2 ** 31:.1f}" + '$' + TIMES + '$')
    ad = e2[e2.method == 'adelie']
    put('NumOOCAdelieStop', f"{int(ad[ad.status != 'ok']['size'].min()):,}".replace(',', '{,}'))
    req = [c.get('read_bytes', 0) + c.get('stage2_read_bytes', 0) for c in ch.chorus]
    dev = [max(v.values()) for v in ch.device_read_bytes]
    put('NumDeviceRatioMin', f"{min(d / r for d, r in zip(dev, req)):.2f}"); put('NumDeviceRatioMax', f"{max(d / r for d, r in zip(dev, req)):.2f}")
    b = json.load(open(os.path.join(ROOT, 'results', 'v2', 'build', 'disk_n400000.json')))
    put('NumBuildMin', f"{b['total_seconds'] / 60:.1f}")
    put('NumBuildShare', f"{100 * b['total_seconds'] / (b['total_seconds'] + ch.fit_seconds.iloc[-1]):.0f}" + chr(92) + '%')
    # batch size
    e4 = new[(new.experiment == 'e4_batch') & (new.status == 'ok')].sort_values('size')
    put('NumBatchSpeed', f"{e4.fit_seconds.iloc[0] / e4.fit_seconds.iloc[-1]:.1f}")
    put('NumBatchPassesOne', f"{e4.chorus.iloc[0]['passes']:,}".replace(',', '{,}')); put('NumBatchPassesB', str(e4.chorus.iloc[-1]['passes']))
    # budgets
    bud = new[new.experiment.isin(['e3_budget', 'e3_stress'])]
    put('NumBudgetRuns', str(len(bud)))
    ok = bud[bud.peak_wset.notna()]
    put('NumBudgetMaxFrac', f"{(ok.peak_wset / ok.M).max():.2f}")
    m1 = bud[bud.tag == 'M1.0_full'].iloc[-1]
    put('NumBudgetOneMin', f"{m1.fit_seconds / 60:.0f}"); put('NumBudgetOneB', str(m1.chorus['batch_size']))
    m2 = bud[bud.tag == 'M2.0_full'].iloc[-1]
    put('NumBudgetTwoMin', f"{m2.fit_seconds / 60:.0f}"); put('NumBudgetTwoB', str(m2.chorus['batch_size']))
    # early termination
    pc = os.path.join(ROOT, 'results', 'v2', 'patience.csv')
    if os.path.exists(pc):
        P = pd.read_csv(pc)
        sim = P[P.case.str.startswith('n_train')]
        s5 = sim[sim.patience.astype(str).isin(['5', '5.0'])]
        put('NumPatTime', f"{s5.time_ratio.mean():.2f}"); put('NumPatChanged', f"{100 * s5.changed_selection.mean():.0f}" + chr(92) + '%')
        put('NumPatJaccard', f"{s5.support_jaccard.mean():.3f}"); put('NumPatDRtwo', f"{abs(s5.d_r2.mean()):.4f}")
        s8 = sim[sim.patience.astype(str).isin(['8', '8.0'])]
        put('NumPatEightTime', f"{s8.time_ratio.mean():.2f}"); put('NumPatEightChanged', f"{100 * s8.changed_selection.mean():.0f}" + chr(92) + '%')
        dp = os.path.join(ROOT, 'results', 'v2', 'patience', 'depmap_c200', 'p5', 'run.json')
        if os.path.exists(dp):
            put('NumPatDepMin', f"{json.load(open(dp))['fit_seconds'] / 60:.0f}")
    # update strategy
    pg = os.path.join(ROOT, 'results', 'v2', 'planner_grid.csv')
    if os.path.exists(pg):
        G = pd.read_csv(pg)
        G['fast'] = G[['t_gram', 't_residual']].min(axis=1)
        X = G[~G.B.isin([1, 16])]                        # cells left from the registered grid (B = 64 at n = 500)
        G = G[G.B.isin([1, 16])]                         # the reduced grid shown in Figure 2i
        r = G.t_auto / G.fast
        put('NumPlannerCells', str(len(G))); put('NumPlannerRegretMax', f"{r.max():.2f}"); put('NumPlannerRegretMean', f"{r.mean():.2f}")
        put('NumPlannerThresholdMax', f"{(G.t_threshold / G.fast).max():.2f}")
        put('NumPlannerExtraMax', f"{(X.t_auto / X.fast).max():.2f}")
        slow = np.maximum(G.t_gram, G.t_residual) / G.fast
        put('NumPlannerFixedWorst', f"{slow.max():.1f}")
