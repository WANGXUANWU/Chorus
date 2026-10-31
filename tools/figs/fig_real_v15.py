"""Figure 4 (pre-submission revision, v15): real many-response workloads at matched model size, in the
visual form of v11 (approved 10-22).

(a) Annotated heat tables in the style of cardiac_FM Fig. 4d. Top: mean test R2 of Chorus and of every
baseline at the model size of Chorus (for each response, the baseline path model with the smallest
validation error among the models of both basis sizes that are no larger than the Chorus model; amendment
22 (b)), in the six DepMap splits and at the two Beijing time origins; darker is better within each case,
framed cells are the best, an asterisk marks a significant lead of Chorus over every baseline (paired
t-test over responses, 5% level), side bars give the mean rank. Bottom: share of responses on which each method
is within 0.01 of the most accurate method at the model size of Chorus (a performance profile), framed and
ranked in the same way (author comment of 29 Oct: both tables hold the same rows). Strip: mean gain of
Chorus over GL (the estimator of its first stage) at equal size. (b) uses thinner bars than v13 and, in v15, a one-row key (author comment of 29 Oct: the key was too tall).
(b) Per-response comparison with each baseline at the model size of Chorus in the style of PARALLAX
Fig. 5b: share of responses on which Chorus is ahead by more than 0.01 in test R2, ahead by up to 0.01,
behind or tied within 0.01, or behind by more than 0.01; right, the mean gain.
(c) Four DepMap dependencies with a known biomarker. (d) Seasonal effects of three predictors on every
pollutant, one curve per station. Raw test R2 and model sizes are in Table 3.
Usage: fig_real_v15.py MATCHED_PER_RESPONSE_CSV OUT
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
import style
import style_v7 as sv
from chorus import SplineDesign
from bench.caseio import read_meta, load

sv.setup()
pr_csv, OUT = sys.argv[1:3]
ACC = ['Chorus', 'Adaptive GL', 'GMCP', 'SGL', 'SpAM', 'GL', 'MTL']
OLD = {'Chorus': 'Chorus', 'Adaptive GL': 'adaptive GL', 'GMCP': 'gMCP (grpreg)', 'SGL': 'SGL (sparsegl)',
       'SpAM': 'SpAM (SAM)', 'GL': 'GL (skglm)', 'MTL': 'MTL (biglasso)'}
COLOR = {m: sv.COLOR[OLD[m]] for m in ACC}
EDGE = {m: sv.EDGE[OLD[m]] for m in ACC}
MARKER = {m: sv.MARKER[OLD[m]] for m in ACC}
RIV, A = ACC[1:], sv.ANNOT
LAB = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'B1', 'B2']
CR = COLOR['Chorus']
POLL = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
PLAB = {'PM2.5': r'PM$_{2.5}$', 'PM10': r'PM$_{10}$', 'SO2': r'SO$_2$', 'NO2': r'NO$_2$', 'CO': 'CO', 'O3': r'O$_3$'}
PCOL = dict(zip(POLL, ['#3A8DDE', '#9B59B6', '#EDA100', '#2A9D84', '#56606E', '#C8522B']))
HACC = mc.LinearSegmentedColormap.from_list('acc', sv.HEAT_ACC)
HWIN = mc.LinearSegmentedColormap.from_list('win', sv.HEAT_TIME)
A22 = os.path.join(ROOT, 'results', 'real_a22')


def mix(c, t, base):
    a, b = np.array(mc.to_rgb(c)), np.array(mc.to_rgb(base))
    return tuple(a + (b - a) * t)


M = pd.read_csv(pr_csv)
fig = plt.figure(figsize=(style.TEXT_W, 5.05))
gs = fig.add_gridspec(2, 1, hspace=.36, height_ratios=[1.22, 1], left=.085, right=.99, top=.915, bottom=.07)
r1 = gs[0].subgridspec(1, 2, wspace=.40, width_ratios=[1.32, 1])
r2 = gs[1].subgridspec(1, 2, wspace=.16, width_ratios=[1.0, 1.32])
fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], mec=EDGE[m], lw=1.6 if m == 'Chorus' else 1.0,
                           ms=3.8 if m == 'Chorus' else 3.1, label=m) for m in ACC],
           loc='upper center', ncol=7, bbox_to_anchor=(.5, .999), handlelength=1.6, columnspacing=1.2,
           handletextpad=.35, fontsize=6.6)

# ---------------------------------------------------------------- a: heat tables at the model size of Chorus
TAU = .01                                      # near-best margin in test R2, as in panel b
tab = np.full((len(ACC), len(LAB)), np.nan)
near = np.full((len(ACC), len(LAB)), np.nan)
star = np.zeros(len(LAB), bool)
gain_gl = np.zeros(len(LAB))
for j, c in enumerate(LAB):
    d = M[M.case == c]
    w = d.pivot_table(index='response', columns='method', values='r2_matched')
    w['Chorus'] = d.drop_duplicates('response').set_index('response')['r2_chorus']
    w = w[ACC]
    tab[:, j] = w.mean().values
    near[:, j] = 100 * w.ge(w.max(axis=1) - TAU, axis=0).mean().values
    star[j] = all(ttest_rel(w.Chorus, w[m]).pvalue < .05 and w.Chorus.mean() > w[m].mean() for m in RIV)
    gain_gl[j] = (w.Chorus - w.GL).mean()

sa = r1[0, 0].subgridspec(3, 2, height_ratios=[7, 7, 1.75], hspace=.16, wspace=.04, width_ratios=[1, .20])
# top: mean test R2 at the model size of Chorus
ax = fig.add_subplot(sa[0, 0])
lo, hi = np.nanmin(tab, axis=0), np.nanmax(tab, axis=0)
norm = (tab - lo) / np.maximum(hi - lo, 1e-12)                  # within each case
ax.imshow(norm, cmap=HACC, vmin=-.05, vmax=1, aspect='auto', interpolation='nearest')
for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
        s = f'{tab[i, j]:.3f}'.replace('0.', '.', 1)
        if i == 0 and star[j]:
            s += '*'
        ax.text(j, i, s, ha='center', va='center', fontsize=A, color='white' if norm[i, j] > .62 else style.INK,
                fontweight='bold' if ACC[i] == 'Chorus' else 'normal')
for j in range(tab.shape[1]):
    i = int(np.nanargmax(tab[:, j]))
    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=style.INK, lw=.75, zorder=5))
ax.axvline(5.5, color='white', lw=2.2)
ax.set_yticks(range(len(ACC))); ax.set_yticklabels(ACC, fontsize=6.3)
ax.get_yticklabels()[0].set_fontweight('bold')
ax.set_xticks(range(len(LAB))); ax.set_xticklabels([])
ax.tick_params(length=0, pad=1.5); ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.text(-.5, -.62, r'Mean Test $R^2$ at the Model Size of Chorus, Higher Is Better', fontsize=6.5, va='bottom', ha='left')
ax.text(2.5, -1.55, 'DepMap, Six Splits', ha='center', va='bottom', fontsize=6.8)
ax.text(6.5, -1.55, 'Beijing, Two Origins', ha='center', va='bottom', fontsize=6.8)
style.panel_label(ax, 'a', x=-.20, y=1.16)
rank = pd.DataFrame(-tab).rank(axis=0, method='average').values.mean(axis=1)
bx = fig.add_subplot(sa[0, 1])
y = np.arange(len(ACC))
bx.barh(y, 8 - rank, color=[COLOR[m] for m in ACC], edgecolor=[EDGE[m] for m in ACC], height=.62, lw=.3)
for y_, r_, m in zip(y, rank, ACC):
    bx.text(8 - r_ + .25, y_, f'{r_:.1f}', va='center', fontsize=A, color=style.INK,
            fontweight='bold' if m == 'Chorus' else 'normal')
bx.set_ylim(len(ACC) - .5, -.5); bx.set_xlim(0, 10.2)
bx.set_yticks([]); bx.set_xticks([]); bx.grid(False)
for sp in bx.spines.values():
    sp.set_visible(False)
bx.set_title('Mean Rank', fontsize=6.4, pad=2)
# bottom: share of responses on which each method is within TAU of the best method (same rows as the top)
ax = fig.add_subplot(sa[1, 0])
lo, hi = np.nanmin(near, axis=0), np.nanmax(near, axis=0)
nn = (near - lo) / np.maximum(hi - lo, 1e-12)                   # within each case
ax.imshow(nn, cmap=HWIN, vmin=-.05, vmax=1, aspect='auto', interpolation='nearest')
for i in range(near.shape[0]):
    for j in range(near.shape[1]):
        ax.text(j, i, f'{near[i, j]:.0f}', ha='center', va='center', fontsize=A,
                color='white' if nn[i, j] > .62 else style.INK, fontweight='bold' if ACC[i] == 'Chorus' else 'normal')
for j in range(near.shape[1]):
    i = int(np.nanargmax(near[:, j]))
    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=style.INK, lw=.75, zorder=5))
ax.axvline(5.5, color='white', lw=2.2)
ax.set_yticks(range(len(ACC))); ax.set_yticklabels(ACC, fontsize=6.3)
ax.get_yticklabels()[0].set_fontweight('bold')
ax.set_xticks(range(len(LAB))); ax.set_xticklabels([])
ax.tick_params(length=0, pad=1.5); ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.text(-.5, -.62, 'Responses Within 0.01 of the Best Method at the Size of Chorus (%)', fontsize=6.5, va='bottom',
        ha='left')
rank_n = pd.DataFrame(-near).rank(axis=0, method='average').values.mean(axis=1)
bx = fig.add_subplot(sa[1, 1])
y = np.arange(len(ACC))
bx.barh(y, 8 - rank_n, color=[COLOR[m] for m in ACC], edgecolor=[EDGE[m] for m in ACC], height=.62, lw=.3)
for y_, r_, m in zip(y, rank_n, ACC):
    bx.text(8 - r_ + .25, y_, f'{r_:.1f}', va='center', fontsize=A, color=style.INK,
            fontweight='bold' if m == 'Chorus' else 'normal')
bx.set_ylim(len(ACC) - .5, -.5); bx.set_xlim(0, 10.2)
bx.set_yticks([]); bx.set_xticks([]); bx.grid(False)
for sp in bx.spines.values():
    sp.set_visible(False)
# strip: gain over GL at equal size
gx = fig.add_subplot(sa[2, 0])
gx.imshow(np.clip(gain_gl, 0, None)[None, :] / gain_gl.max(),
          cmap=mc.LinearSegmentedColormap.from_list('g', ['#FFFFFF', '#E7A6B4', CR]), vmin=-.1, vmax=1, aspect='auto')
for j, g in enumerate(gain_gl):
    gx.text(j, 0, f'{g:+.3f}'.replace('0.', '.', 1), ha='center', va='center', fontsize=A,
            color='white' if g / gain_gl.max() > .55 else style.INK)
gx.axvline(5.5, color='white', lw=2.2)
gx.set_yticks([0]); gx.set_yticklabels(['Gain over GL\nat Equal Size'], fontsize=6.0, linespacing=.95)
gx.set_xticks(range(len(LAB))); gx.set_xticklabels(LAB, fontsize=6.4)
gx.tick_params(length=0, pad=1.5); gx.grid(False)
for sp in gx.spines.values():
    sp.set_visible(False)

# ---------------------------------------------------------------- b: per-response comparison at matched size
CAT = [(r'Ahead $>$0.01', CR), (r'Ahead $\leq$0.01', '#E7A6B4'), (r'Behind $\leq$0.01', '#C9D0DA'),
       (r'Behind $>$0.01', None)]
sb = r1[0, 1].subgridspec(3, 2, hspace=.30, wspace=.05, width_ratios=[1, .17], height_ratios=[.12, 1, 1])
lg = fig.add_subplot(sb[0, :])
lg.axis('off')
lg.text(-.01, .5, 'Chorus:', transform=lg.transAxes, ha='right', va='center', fontsize=A, fontweight='bold', color=CR)
lg.legend(handles=[Patch(fc=(c or '#8A9099'), ec='white', label=n) for n, c in CAT], loc='center left',
          bbox_to_anchor=(0., .5), ncol=4, fontsize=A, handlelength=.9, handleheight=.8, columnspacing=.75,
          handletextpad=.3, borderaxespad=0, frameon=False)
style.panel_label(lg, 'b', x=-.30, y=1.6)
for k, (pre, title) in enumerate((('D', 'DepMap, 1,200 Responses'), ('B', 'Beijing, 144 Responses'))):
    d = M[M.case.str.startswith(pre)]
    ax = fig.add_subplot(sb[k + 1, 0])
    rx = fig.add_subplot(sb[k + 1, 1])
    for i, m in enumerate(RIV):
        g = (d[d.method == m].r2_chorus - d[d.method == m].r2_matched).values
        share = [np.mean(g > .01), np.mean((g > 0) & (g <= .01)), np.mean((g <= 0) & (g >= -.01)), np.mean(g < -.01)]
        left = 0.
        for (name, col), s in zip(CAT, share):
            c = COLOR[m] if col is None else col
            ax.barh(i, s, left=left, height=.52, color=c, edgecolor='white', lw=.5, zorder=3)
            if s >= .12:
                ink = 'white' if col in (CR, None) and m != 'MTL' else style.INK
                ax.text(left + s / 2, i, f'{100 * s:.0f}', ha='center', va='center', fontsize=A, color=ink, zorder=4)
            left += s
        rx.text(.05, i, f'{g.mean():+.3f}'.replace('0.', '.', 1), va='center', fontsize=A)
    for a_ in (ax, rx):
        a_.set_ylim(len(RIV) - .5, -.5)
        a_.grid(False)
    ax.axvline(.5, color='white', lw=.8, ls=(0, (2, 2)), zorder=3.5)
    ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(['0', '50%', '100%'] if k == 1 else [])
    ax.set_yticks(range(len(RIV))); ax.set_yticklabels(RIV, fontsize=6.3)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_title(title, fontsize=6.8, loc='left', pad=2)
    rx.set_xlim(0, 1); rx.set_xticks([]); rx.set_yticks([])
    for sp in rx.spines.values():
        sp.set_visible(False)
    if k == 0:
        rx.set_title('Mean Gain', fontsize=6.2, pad=2, loc='left')
    else:
        ax.set_xlabel('Share of Responses', labelpad=1)

# ---------------------------------------------------------------- c: DepMap dependencies with known biomarkers
dcase = os.path.join(ROOT, 'results', 'cases', 'depmap_c200')
dm = read_meta(dcase)
info = json.load(open(os.path.join(dcase, 'info.json')))
sym = lambda c: re.sub(r'\s*\(\d+\)$', '', c)
feats, resp = [sym(f) for f in info['features']], [sym(r) for r in info['responses']]
z = np.load(os.path.join(A22, 'depmap_c200', 'Chorus_a22', 'output.npz'))
xd, yd = load(dcase, 'xraw', 'train', dm), load(dcase, 'y', 'train', dm)
yte = load(dcase, 'y', 'test', dm)
r2c = 1 - ((yte - z['pred']) ** 2).sum(0) / ((yte - yte.mean(0)) ** 2).sum(0)
sds = {0: SplineDesign(n_knots=4).fit(xd), 1: SplineDesign(n_knots=2).fit(xd)}
coefs = {0: z['coef'], 1: z['coef_b1']}
cand = []
for r, sv_ in enumerate(info['silver']):
    b = int(z['basis_of'][r]); sd = sds[b]
    tk = {t.feature: k for k, t in enumerate(sd.terms)}
    for f in sv_:
        k = tk.get(f)
        if k is None:
            continue
        a_, b_ = sd.groups[k]
        nrm = np.linalg.norm(coefs[b][a_:b_, r])
        if nrm > 0:
            cand.append((nrm, r, f, k, b))
cand.sort(reverse=True)
seen, four = set(), []
for c in cand:
    if c[1] not in seen and len(four) < 4:
        four.append(c); seen.add(c[1])
row = r2[0, 0].subgridspec(2, 2, wspace=.30, hspace=.62)
for i, (nrm, r, f, k, b) in enumerate(four):
    ax = fig.add_subplot(row[i // 2, i % 2])
    sd = sds[b]
    Xall = sd.transform(xd)
    a_, b_ = sd.groups[k]
    x = xd[:, f]
    grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 100)
    Xg = np.tile(np.median(xd, axis=0), (len(grid), 1)); Xg[:, f] = grid
    fx = sd._term_columns(sd.terms[k], Xg) @ coefs[b][a_:b_, r]
    part = yd[:, r] - (Xall @ coefs[b][:, r] - sd._term_columns(sd.terms[k], xd) @ coefs[b][a_:b_, r])
    self_ = feats[f] == resp[r]
    col = '#3A8DDE' if self_ else CR
    ax.scatter(x, part, s=1.0, color='#9A9EA6', alpha=.35, lw=0, rasterized=True)
    ax.plot(grid, fx, color=col, lw=1.6)
    how = 'Own Expression' if self_ else f'Paralog {feats[f]}'
    ax.set_title(f'{resp[r]} on {how}', fontsize=6.4, pad=2)
    ax.text(.97, .05, f'$R^2$ {r2c[r]:.2f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=A)
    if i // 2 == 1:
        ax.set_xlabel('Expression (log TPM)', labelpad=1)
    if i % 2 == 0:
        ax.set_ylabel('Gene Effect', labelpad=1)
    if i == 0:
        style.panel_label(ax, 'c', x=-.36, y=1.10)
    ax.grid(axis='x', visible=False)

# ---------------------------------------------------------------- d: seasonal effects, all stations
bcase = os.path.join(ROOT, 'results', 'cases', 'beijing')
bm = read_meta(bcase)
pol_of = np.array([c.split('|')[1] for c in bm['ycols']])
zb = np.load(os.path.join(A22, 'beijing', 'Chorus_a22', 'output.npz'))
xb = load(bcase, 'xraw', 'train', bm)
sdb = {0: SplineDesign(n_knots=4).fit(xb, varying=bm['varying_features'], index=bm['index_feature']),
       1: SplineDesign(n_knots=2).fit(xb, varying=bm['varying_features'], index=bm['index_feature'])}
cb_ = {0: zb['coef'], 1: zb['coef_b1']}
ui = bm['index_feature']
u = np.linspace(np.percentile(xb[:, ui], 1), np.percentile(xb[:, ui], 99), 60)
tf = np.asarray(bm['term_features'])
kinds = np.asarray(bm['term_kinds'])
sup = zb['support'].astype(bool)
TERMS = [('PRES_lag0', 'Pressure'), ('DEWP_lag0', 'Dew Point'), ('TEMP_lag0', 'Temperature')]


def effect(r, j):
    b = int(zb['basis_of'][r]); sd = sdb[b]
    lo_, hi_ = np.percentile(xb[:, j], [10, 90])
    E = []
    for xv in (lo_, hi_):
        Xg = np.tile(np.median(xb, axis=0), (len(u), 1)); Xg[:, ui] = u; Xg[:, j] = xv
        e = np.zeros(len(u))
        for k, t in enumerate(sd.terms):
            if t.feature == j:
                a_, b_ = sd.groups[k]
                e += sd._term_columns(t, Xg) @ cb_[b][a_:b_, r]
        E.append(e)
    return E[1] - E[0]


sd_ = r2[0, 1].subgridspec(len(TERMS), len(POLL), wspace=.10, hspace=.16)
for ti, (var, name) in enumerate(TERMS):
    j = bm['xcols'].index(var)
    vk = [k for k in range(len(tf)) if kinds[k] == 'varying' and tf[k] == j]
    effs = {pl: np.array([effect(r, j) for r in np.flatnonzero(pol_of == pl)]) for pl in POLL}
    lim = max(np.abs(v).max() for v in effs.values()) * 1.12
    for pi, pl in enumerate(POLL):
        ax = fig.add_subplot(sd_[ti, pi])
        v = effs[pl]
        for e in v:
            ax.plot(u, e, color=PCOL[pl], lw=.5, alpha=.45)
        ax.plot(u, np.median(v, axis=0), color=mix(PCOL[pl], .25, 'black'), lw=1.4)
        ax.axhline(0, color='#3A3D42', lw=.45)
        ax.set_ylim(-lim, lim); ax.set_xlim(u[0], u[-1])
        n_sel = int(sup[vk[0], pol_of == pl].sum()) if vk else 0
        ax.text(.96, .93, f'{n_sel}/12', transform=ax.transAxes, fontsize=A, va='top', ha='right', color=style.INK,
                bbox=dict(fc='white', ec='none', pad=.2, alpha=.85), zorder=6)
        ax.set_xticks([105, 288]); ax.grid(axis='x', visible=False)
        ax.set_xticklabels(['Apr', 'Oct'] if ti == len(TERMS) - 1 else [], fontsize=A)
        if pi == 0:
            ax.set_ylabel(name, fontsize=6.4, labelpad=1)
            ax.tick_params(axis='y', labelsize=A, pad=1)
        else:
            ax.set_yticklabels([])
        if ti == 0:
            ax.set_title(PLAB[pl], fontsize=6.6, pad=2, color=mix(PCOL[pl], .25, 'black'))
        if ti == 0 and pi == 0:
            style.panel_label(ax, 'd', x=-.75, y=1.22)
            ax.text(-.2, 1.42, 'Effect of a High Against a Low Value on log Concentration, One Curve per Station',
                    transform=ax.transAxes, fontsize=6.6, ha='left', va='bottom')
style.save(fig, OUT)
print('saved', OUT, 'stars', star.tolist())
