"""Figure 3 (pre-submission revision): statistical accuracy under the common protocol.

Half-page layout in the style of cardiac_FM Figs. 4-5 (author comments of 28 Oct): the five radars of
v8 become one radar (a) whose spokes are the (family, metric) pairs, keeping only the spokes on which
Chorus has rank 1 (all ranks in the technical report); (b, c) paired per-response gains of Chorus in
the six sharing scenarios; (d) the robustness heat table with the support metrics F1 and MCC, framed
cell best, asterisk significantly better than every other method (paired t-test over seeds, 5%);
(e) selections in one setting fixed in advance (kappa = 0.5, rho = 0.7, SNR 1), as frequencies over
its 80 responses. Six published baselines with the names of the paper.
Usage: fig_accuracy_v9.py SIM_CSV SIM_PER_RESPONSE_CSV SELECTION_JSON OUT
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
import style
import style_v7 as sv

sv.setup()
sim_csv, pr_csv, sel_json, OUT = sys.argv[1:5]
ACC = ['Chorus', 'Adaptive GL', 'GMCP', 'SGL', 'SpAM', 'GL', 'MTL']
OLD = {'Chorus': 'Chorus', 'Adaptive GL': 'adaptive GL', 'GMCP': 'gMCP (grpreg)', 'SGL': 'SGL (sparsegl)',
       'SpAM': 'SpAM (SAM)', 'GL': 'GL (skglm)', 'MTL': 'MTL (biglasso)'}
COLOR = {m: sv.COLOR[OLD[m]] for m in ACC}
MARKER = {m: sv.MARKER[OLD[m]] for m in ACC}
EDGE = {m: sv.EDGE[OLD[m]] for m in ACC}
RIVALS = ACC[1:]
A = sv.ANNOT
HIGHER = {'f1': True, 'mcc': True, 'auprc': True, 'r2': True, 'fpr': False, 'size_err': False, 'fee': False}

D = pd.read_csv(sim_csv)
D = D[D.method.isin(ACC)].copy()
D['size_err'] = (D['size'] - D['s']).abs()
P = pd.read_csv(pr_csv)
P = P[P.method.isin(ACC)].merge(D[['case', 'kappa', 'q', 'snr_range', 'scale_range', 'n_clusters']].drop_duplicates('case'),
                                on='case')
SEL = json.load(open(sel_json))

fig = plt.figure(figsize=(style.TEXT_W, 4.85))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.06], hspace=.40, left=.055, right=.995, top=.865, bottom=.075)
row1 = gs[0].subgridspec(1, 2, wspace=.30, width_ratios=[.34, 1.0])
row2 = gs[1].subgridspec(1, 2, wspace=.22, width_ratios=[1.0, .80])
fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], mec=EDGE[m], lw=1.6 if m == 'Chorus' else 1.0,
                           ms=3.8 if m == 'Chorus' else 3.1, label=m) for m in ACC],
           loc='upper center', ncol=7, bbox_to_anchor=(.5, .999), handlelength=1.6, columnspacing=1.2,
           handletextpad=.35, fontsize=6.6)


def mix(c, t, base):
    r = np.array(mc.to_rgb(c)); b = np.array(mc.to_rgb(base))
    return tuple(r + (b - r) * t)


# ---------------------------------------------------------------- a: one radar of all families
FAMS = [('formal', 'Main Grid'), ('robust', 'Robustness'), ('sharing', 'Sharing')]
METS = [('f1', 'F1'), ('mcc', 'MCC'), ('auprc', 'AUPRC'), ('fpr', 'FPR'), ('size_err', 'Size'), ('fee', 'FEE'),
        ('r2', '$R^2$')]
spokes, fam_of, rank = [], [], {m: [] for m in ACC}
dropped = []
for fam, _ in FAMS:
    d = D[D.family == fam]
    for met, lab in METS:
        mean = d.groupby('method')[met].mean().reindex(ACC)
        r = (-mean if HIGHER[met] else mean).rank(method='min')
        if r['Chorus'] != 1 or (r == 1).sum() > 1:
            dropped.append((fam, met))
            continue
        spokes.append(lab); fam_of.append(fam)
        for m in ACC:
            rank[m].append(r[m])
print('spokes', len(spokes), 'dropped', dropped)
ax = fig.add_subplot(row1[0, 0], projection='polar')
n = len(spokes)
th = np.linspace(0, 2 * np.pi, n, endpoint=False)
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
rad = lambda r: 1 - (np.asarray(r, float) - 1) / 6 * .80
ring = np.linspace(0, 2 * np.pi, 240)
SECT = {'formal': '#F7ECF0', 'robust': '#EAF0F8', 'sharing': '#E9F4EE'}
ARC = {'formal': '#C98AA0', 'robust': '#7FA3CF', 'sharing': '#6FB596'}
step = 2 * np.pi / n
for fam, _ in FAMS:                                   # one soft sector per family
    idx = [i for i, f in enumerate(fam_of) if f == fam]
    if idx:
        t0, t1 = th[idx[0]] - step / 2, th[idx[-1]] + step / 2
        tt = np.linspace(t0, t1, 60)
        ax.fill(np.r_[tt, tt[::-1]], np.r_[np.full(60, 1.0), np.full(60, rad(7))], color=SECT[fam], lw=0, zorder=0)
        ax.plot(tt[1:-1], np.full(58, 1.035), color=ARC[fam], lw=2.4, solid_capstyle='butt', zorder=2, clip_on=False)
for r_, lw_ in ((1, .6), (4, .35), (7, .35)):
    ax.plot(ring, np.full_like(ring, rad(r_)), color='#C9CDD3', lw=lw_, zorder=1)
for t in th:
    ax.plot([t, t], [rad(7), 1], color='#D5D8DD', lw=.35, zorder=1)
for r_ in (1, 4, 7):
    ax.text(th[0] + step / 2, rad(r_), str(r_), fontsize=A, color=sv.MUTED, ha='center', va='center', zorder=6,
            bbox=dict(fc='white', ec='none', pad=.1, alpha=.85))
for m in sorted(RIVALS, key=lambda m: -np.mean(rank[m])) + ['Chorus']:
    r = rad(rank[m]); tt, rr = np.r_[th, th[0]], np.r_[r, r[0]]
    if m == 'Chorus':
        ax.fill(tt, rr, color=COLOR[m], alpha=.18, lw=0, zorder=7)
        ax.plot(tt, rr, color=COLOR[m], lw=1.5, zorder=8)
        ax.scatter(th, r, s=8, color=COLOR[m], edgecolor='white', lw=.4, zorder=9)
    else:
        ax.plot(tt, rr, color=COLOR[m], lw=.7, alpha=.9, zorder=5)
        ax.scatter(th, r, s=4, marker=MARKER[m], color=COLOR[m], edgecolor=EDGE[m], lw=.25, zorder=6)
ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.spines['polar'].set_visible(False)
for t, lab in zip(th, spokes):
    s, c = np.sin(t), np.cos(t)
    ax.text(t, 1.12, lab, fontsize=A, ha='center' if abs(s) < .3 else ('left' if s > 0 else 'right'),
            va='center' if abs(c) < .3 else ('bottom' if c > 0 else 'top'), color=style.INK, clip_on=False)
for fam, name in FAMS:                                # family names outside their sectors
    idx = [i for i, f in enumerate(fam_of) if f == fam]
    if idx:
        tm = np.mean(th[idx])
        pass
mean = {m: float(np.mean(rank[m])) for m in ACC}
best = min(RIVALS, key=lambda m: mean[m])
ax.set_title('Rank on Every Family and Metric', fontsize=7.0, pad=30)
ax.legend(handles=[Patch(fc=ARC[f], ec='none', label=nm) for f, nm in FAMS], loc='upper center', bbox_to_anchor=(.5, -.13),
          ncol=3, fontsize=A, handlelength=.9, handleheight=.6, handletextpad=.3, columnspacing=.8, frameon=False)
ax.text(.5, 1.165, f'Mean Rank: Chorus {mean["Chorus"]:.1f}, {best} {mean[best]:.1f}', transform=ax.transAxes,
        fontsize=A, ha='center', va='bottom', color=sv.MUTED)
style.panel_label(ax, 'a', x=-.20, y=1.27)

# ---------------------------------------------------------------- b, c: paired gains, sharing scenarios
FAM = [('Overlap\n$\\kappa$=0', lambda d: (d.kappa == 0) & (d.q == 16) & d.snr_range.isna() & d.scale_range.isna() & d.n_clusters.isna()),
       ('Overlap\n$\\kappa$=1', lambda d: (d.kappa == 1) & (d.q == 16)),
       ('Mixed\nSNR', lambda d: d.snr_range.notna()),
       ('Rescaled\nResponses', lambda d: d.scale_range.notna()),
       ('Two\nGroups', lambda d: d.n_clusters.notna()),
       ('No Sharing\n$q$=64', lambda d: d.q == 64)]
S = P[P.family == 'sharing']
gain = {}
for fi, (_, sel) in enumerate(FAM):
    d = S[sel(S)]
    f1 = d.pivot_table(index=['case', 'response'], columns='method', values='f1')
    fe = d.pivot_table(index=['case', 'response'], columns='method', values='fee')
    gain[fi] = {m: ((f1['Chorus'] - f1[m]).values, 100 * (1 - fe['Chorus'] / fe[m]).values) for m in RIVALS}


def boxes(ax, k, ylim):
    ax.set_ylim(*ylim)
    nm = len(RIVALS); w = .86 / nm
    rng = np.random.default_rng(7)
    for fi in range(len(FAM)):
        if fi:
            ax.axvline(fi - .5, color='#E2E2E2', lw=.5, zorder=0)
        for mi, m in enumerate(RIVALS):
            v = gain[fi][m][k]; v = v[np.isfinite(v)]
            x = fi + (mi - (nm - 1) / 2) * w
            c = COLOR[m]; dark, light = mix(c, .28, 'black'), mix(c, .18, 'white')
            bw = w * .76
            vs = np.clip(v, *ylim)
            ax.scatter(x + (rng.random(len(vs)) - .5) * bw * .9, vs, s=.5, color=c, alpha=.22, lw=0, zorder=2,
                       rasterized=True)
            q1, med, q3 = np.percentile(v, [25, 50, 75]); lo, hi = np.percentile(v, [10, 90])
            for a_, b_ in ((lo, q1), (q3, hi)):
                ax.plot([x, x], [a_, b_], color=dark, lw=.5, zorder=3, solid_capstyle='butt')
            for e_ in (lo, hi):
                ax.plot([x - bw * .22, x + bw * .22], [e_, e_], color=dark, lw=.5, zorder=3, solid_capstyle='butt')
            h = max(q3 - q1, (ylim[1] - ylim[0]) * .014)
            ax.add_patch(Rectangle((x - bw / 2, (q1 + q3) / 2 - h / 2), bw, h, fc=light, ec=dark, lw=.45, zorder=4))
            ax.plot([x - bw / 2, x + bw / 2], [med, med], color='white', lw=.9, zorder=5, solid_capstyle='butt')
    ax.axhline(0, color='#3A3D42', lw=.5, zorder=1)
    ax.set_xlim(-.5, len(FAM) - .5); ax.set_xticks(range(len(FAM))); ax.grid(axis='x', visible=False)


right = row1[0, 1].subgridspec(2, 1, hspace=.10)
ax = fig.add_subplot(right[0])
boxes(ax, 0, (-.12, 1.02))
ax.set_xticklabels([]); ax.set_ylabel('F1 Gain')
ax.set_title('Gain of Chorus per Response in the Sharing Scenarios, by Compared Method', loc='left', x=0, pad=12)
ax.legend(handles=[Patch(fc=mix(COLOR[m], .18, 'white'), ec=mix(COLOR[m], .28, 'black'), lw=.45, label=m) for m in RIVALS],
          loc='lower right', bbox_to_anchor=(1.0, .99), ncol=6, fontsize=A, handlelength=.9, handleheight=.7,
          handletextpad=.3, columnspacing=.7, borderaxespad=0, frameon=False)
style.panel_label(ax, 'b', x=-.075, y=1.12)
ax = fig.add_subplot(right[1])
boxes(ax, 1, (-30, 100))
ax.set_xticklabels([f[0] for f in FAM], fontsize=6.2, linespacing=1.0)
ax.set_ylabel('FEE Reduction (%)')
style.panel_label(ax, 'c', x=-.075, y=1.06)

# ---------------------------------------------------------------- d: robustness heat table, F1 and MCC
R = D[D.family == 'robust'].copy()
SIT = [('Base', {}), (r'$n$=200', {'n_train': 200}), (r'$n$=800', {'n_train': 800}), (r'$p$=500', {'p': 500}),
       (r'$p$=1000', {'p': 1000}), ('3 Active', {'s': 3}), ('12 Active', {'s': 12}), (r'$t_3$ Noise', {'noise': 't3'}),
       (r'$q$=4', {'q': 4}), (r'$q$=64', {'q': 64})]
BASE = {'n_train': 400, 'p': 200, 's': 6, 'q': 16, 'noise': np.nan}
cmap = mc.LinearSegmentedColormap.from_list('cardiac', sv.HEAT_ACC)
hd = row2[0, 0].subgridspec(2, 2, hspace=.20, wspace=.04, width_ratios=[1, .22])
for k, (metric, title) in enumerate((('f1', 'F1'), ('mcc', 'MCC'))):
    mat = np.full((len(ACC), len(SIT)), np.nan)
    star = np.zeros(len(SIT), bool)
    for si, (_, ch) in enumerate(SIT):
        cond = np.ones(len(R), bool)
        for kk, v in {**BASE, **ch}.items():
            cond &= (R[kk].isna().values if (isinstance(v, float) and np.isnan(v)) else (R[kk] == v).values)
        w = R[cond].pivot_table(index='seed', columns='method', values=metric)
        mat[:, si] = [w[m].mean() for m in ACC]
        star[si] = all(ttest_rel(w['Chorus'], w[m]).pvalue < .05 and w['Chorus'].mean() > w[m].mean() for m in RIVALS)
    lo, hi = np.nanmin(mat), np.nanmax(mat)
    ax = fig.add_subplot(hd[k, 0])
    ax.imshow((mat - lo) / (hi - lo), cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            t = (mat[i, j] - lo) / (hi - lo)
            s = f'{mat[i, j]:.2f}'.lstrip('0') if mat[i, j] < .995 else '1.00'
            if ACC[i] == 'Chorus' and star[j]:
                s += '*'
            ax.text(j, i, s, ha='center', va='center', fontsize=A, color='white' if t > .62 else style.INK,
                    fontweight='bold' if ACC[i] == 'Chorus' else 'normal')
        j_best = None
    for j in range(mat.shape[1]):
        i = int(np.nanargmax(mat[:, j]))
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=style.INK, lw=.7, zorder=5))
    ax.set_yticks(range(len(ACC))); ax.set_yticklabels(ACC, fontsize=6.2)
    for tl, m in zip(ax.get_yticklabels(), ACC):
        if m == 'Chorus':
            tl.set_fontweight('bold')
    ax.set_xticks(range(len(SIT)))
    ax.set_xticklabels([s_ for s_, _ in SIT] if k == 1 else [], fontsize=A, rotation=30, ha='right', rotation_mode='anchor')
    ax.tick_params(length=0, pad=1.5); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-.5, -.58, f'{title}, Higher Is Better', fontsize=6.3, va='bottom', ha='left')
    if k == 0:
        ax.set_title('Robustness in Ten Situations, Framed Is Best', loc='left', x=-.14, pad=10)
        style.panel_label(ax, 'd', x=-.22, y=1.16)
    bx = fig.add_subplot(hd[k, 1])
    mm = np.nanmean(mat, axis=1)
    bx.barh(np.arange(len(ACC)), mm, color=[COLOR[m] for m in ACC], height=.62, lw=0)
    for y_, v_, m in zip(range(len(ACC)), mm, ACC):
        bx.text(v_ + .03, y_, f'{v_:.2f}'.lstrip('0'), va='center', fontsize=A, fontweight='bold' if m == 'Chorus' else 'normal')
    bx.set_ylim(len(ACC) - .5, -.5); bx.set_xlim(0, 1.55)
    bx.set_yticks([]); bx.grid(False); bx.spines['left'].set_visible(False)
    bx.tick_params(axis='x', labelsize=A, length=1.5, pad=1); bx.set_xticks([0, 1])
    if k == 1:
        bx.set_xlabel('Mean', fontsize=6.3, labelpad=1)
    else:
        bx.set_xticklabels([])

# ---------------------------------------------------------------- e: selection frequencies in one setting
ax = fig.add_subplot(row2[0, 1])
setting = P[(P.family == 'formal') & (P.kappa == .5)]
cases = [c for c in setting.case.unique() if 'rho0.7_snr1.0' in c]
f1s = setting[setting.case.isin(cases)].groupby('method')['f1'].mean()
NT, NF = 6, 12
for i, m in enumerate(ACC):
    y = len(ACC) - 1 - i
    rec, fpl = SEL[m]['recall_sorted'], SEL[m]['fp_at_least'][:NF]
    for k_, v in enumerate(rec):
        ax.add_patch(Rectangle((k_, y), .9, .84, fc=mix(COLOR[m], 1 - v, 'white') if v > 0 else '#F3F4F6', lw=0))
    for k_, v in enumerate(fpl):
        ax.add_patch(Rectangle((NT + .6 + k_, y), .9, .84, fc=mix(COLOR[m], 1 - .85 * v, 'white') if v > 0 else '#F3F4F6',
                               lw=0))
    ax.text(NT + NF + 2.6, y + .42, f'{SEL[m]["tp_mean"]:.1f}', va='center', ha='right', fontsize=A)
    ax.text(NT + NF + 5.6, y + .42, f'{SEL[m]["fp_mean"]:.1f}', va='center', ha='right', fontsize=A)
    ax.text(NT + NF + 8.2, y + .42, f'{f1s[m]:.2f}'.lstrip('0'), va='center', ha='right', fontsize=A,
            fontweight='bold' if m == 'Chorus' else 'normal')
ax.axvline(NT + .3, color=style.INK, lw=.7, ls='--')
for x_, lab in ((NT + NF + 2.6, 'TP'), (NT + NF + 5.6, 'FP'), (NT + NF + 8.2, 'F1')):
    ax.text(x_, len(ACC) + .05, lab, fontsize=A, ha='right', va='bottom')
ax.text(NT / 2, -.55, 'True, by Recall', fontsize=A, ha='center', va='top')
ax.text(NT + .6 + NF / 2, -.55, r'Share with $\geq k$ False, $k$=1 to 12', fontsize=A, ha='center', va='top')
ax.set_xlim(-.1, NT + NF + 8.4); ax.set_ylim(-.15, len(ACC) + .05)
ax.set_yticks(np.arange(len(ACC)) + .42); ax.set_yticklabels(ACC[::-1], fontsize=6.2)
for tl in ax.get_yticklabels():
    if tl.get_text() == 'Chorus':
        tl.set_fontweight('bold')
ax.set_xticks([]); ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0, pad=1.5)
ax.set_title(r'Selections, 80 Responses at $\kappa$=0.5, $\rho$=0.7, SNR 1', loc='left', x=0, pad=10)
style.panel_label(ax, 'e', x=-.20, y=1.10)
style.save(fig, OUT)
print('saved', OUT)
