"""Figure 2 (revision of 30 Oct, v23): efficiency and memory in the layout of cardiac_FM Figs 4 and 5.

Author comments on v20: panel (c) carried too little information, and the row of boxes did not use its width.
v21 makes (c) two rows of small multiples over the tolerance sweep (largest duality gap and largest change of
the test R2 against seconds) and splits the bottom row into two cardiac_FM Fig 5 box panels with faint
per-response dots: (e) the prediction difference from the exact path at the default and the matched
tolerances, (f) the share of path levels whose support equals that of the exact path at the default
tolerances. Panels (a), (b) and (d) are those of v20. Every panel title names the stage of Chorus it times.
Data: results/timing3 (amendment 21), results/timing2 (budget runs), results/v2/sweep_*.
v22 (author comments on v21): the keys sit next to the panels they explain instead of one row at the top: a
one-column key beside (a), a one-row key of the solvers on the title line of (c) and a one-row key of box
colours on the title line of (e); (f) uses a marked broken axis, 80 to 100% over most of its height, because
most boxes lie above 80% and only DepMap falls below.
v23 (author comments on v22): (f) shows the share of path levels whose support differs from that of the exact
path on a square-root axis, so zero sits at the bottom and the boxes spread over the height instead of
crowding at 100%; (a) drops the full-path line of Chorus (its times stay in (d)); (b) drops the batch sizes and
the refused budgets (both in the text); (c) uses thinner lines, smaller markers and fewer tick labels.
Usage: fig_efficiency_v23.py OUT
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import style
import style_v7 as sv

sv.setup()
OUT = sys.argv[1]
A, GIB = 6.0, 2 ** 30
INK, MUTED, SEP = style.INK, sv.MUTED, '#E3E5E8'
V2 = os.path.join(ROOT, 'results', 'v2')
C = {'Chorus': '#B4304A', 'adelie': '#2F6DB5', 'skglm': '#21A08C', 'Celer': '#BE8A1E', 'one': '#9A80C8'}
MK = {'Chorus': 'o', 'adelie': 's', 'skglm': 'D', 'Celer': '^', 'one': 's'}
GLS = ['Chorus', 'adelie', 'skglm', 'Celer']
EXT = ['adelie', 'skglm', 'Celer']
DEF_TOL = {'Chorus': 1e-6, 'adelie': 1e-7, 'skglm': 1e-4, 'Celer': 1e-4}
WL4 = [('q16', r'$q$ = 16'), ('q64', r'$q$ = 64'), ('beijing', 'Beijing'), ('depmap64', 'DepMap')]
HEAT = mc.LinearSegmentedColormap.from_list('pinkmagenta', ['#FCF1F5', '#F5CEDC', '#E89AB6', '#D2668F', '#A93A6C',
                                                            '#7E2150'])
QS = [1, 4, 16, 64, 256]
qa = np.array(QS, float)


def log(path):
    d = pd.DataFrame([json.loads(line) for line in open(path)])
    return d[d.status != 'DISTURBED']


T2 = log(os.path.join(ROOT, 'results', 'timing2', 'timing_log.jsonl'))
T3 = log(os.path.join(ROOT, 'results', 'timing3', 'timing_log.jsonl'))


def med(tag):
    ok = T3[(T3.tag == tag) & (T3.status == 'ok')].fit_seconds.astype(float)
    return float(ok.median()) if len(ok) else np.nan


def pexp(v):
    return r'$10^{%d}$' % int(round(np.log10(v)))


def fmt_s(v):
    return f'{v:.2f}' if v < 10 else (f'{v:.1f}' if v < 100 else f'{v:.0f}')


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=A + .3, pad=1.5)


def title(ax, stage, text, pad=.045):
    """Panel title that names the stage of Chorus it times: bold stage, then the subject."""
    y = 1 + pad / (ax.get_position().height * H)
    t1 = ax.text(0, y, stage + ':', transform=ax.transAxes, fontsize=7.0, fontweight='bold', ha='left', va='bottom')
    bb = t1.get_window_extent(renderer=fig.canvas.get_renderer())
    x1 = ax.transAxes.inverted().transform((bb.x1 + 2.6 * fig.dpi / 72, bb.y0))[0]
    ax.text(x1, y, text, transform=ax.transAxes, fontsize=7.0, ha='left', va='bottom')


def label(s, x, y):
    fig.text(x, y, s, fontsize=8.5, fontweight='bold', va='bottom', ha='left')


def cbox(ax, x, v, c, w):
    """cardiac_FM Fig 5 box: solid fill, white median, coloured whiskers with caps (5th to 95th percentile)."""
    v = np.asarray(v, float)
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    lo, hi = np.percentile(v, [5, 95])
    for a, b in ((lo, q1), (q3, hi)):
        ax.plot([x, x], [a, b], color=c, lw=.75, zorder=3, solid_capstyle='butt')
    for e in (lo, hi):
        ax.plot([x - w * .28, x + w * .28], [e, e], color=c, lw=.75, zorder=3)
    if ax.get_yscale() == 'log':
        q3 = max(q3, q1 * 1.25)
    elif ax.get_yscale() == 'function' and q3 - q1 < .5:   # square-root axis: a thin bar at the value
        q1, q3 = md, md + .12
    elif q3 - q1 < 1.2:
        q1, q3 = md - 1.6, md + .3
    ax.add_patch(Rectangle((x - w / 2, q1), w, q3 - q1, fc=c, ec=c, lw=.5, zorder=4))
    tall = abs(ax.transData.transform((x, q3))[1] - ax.transData.transform((x, q1))[1]) * 72 / ax.figure.dpi
    if tall > 3.2:                # a white median only where the box is tall enough to show it
        ax.plot([x - w / 2 + w * .1, x + w / 2 - w * .1], [md, md], color='white', lw=1.0, zorder=5, solid_capstyle='butt')


def dots(ax, x, v, c, w, rng):
    """Faint per-response dots behind a box, jittered within its width."""
    ax.scatter(x + (rng.random(len(v)) - .5) * w * .8, v, s=1.6, color=c, alpha=.3, lw=0, zorder=2, rasterized=True)


H = 4.75
fig = plt.figure(figsize=(style.TEXT_W, H))
rng = np.random.default_rng(5)

LX, LW = .064, .512               # left block: (a), (b), (c)
RX = .602                         # right block: (d)
TOP = .955                        # top edge of (a), (b) and (d)
ROW1 = (.765, TOP - .765)         # (a) and (b)
C1 = (.522, .123)                 # (c), first row of small multiples
C2 = (.386, .123)                 # (c), second row
ROW3 = (.062, .205)               # (e) and (f)
AW = .168                         # width of (a); its key fills the gap to (b)

# ------------------------------------------------------------------ (a) time per response against q
ax = fig.add_axes([LX, ROW1[0], AW, ROW1[1]])
cur = {k: np.array([med(f'q{q}_{t}') for q in QS]) / qa for k, t in
       (('one', 'chorus_seq_ws'), ('full', 'chorus_batched'), ('exit', 'chorus_exit'))}
ext = {m: np.array([med(f'q{q}_{m}_exit') for q in QS]) / qa for m in EXT}
for m in EXT[::-1]:
    ax.plot(qa, ext[m], color=C[m], marker=MK[m], lw=1.0, ms=2.9, zorder=4)
ax.plot(qa, cur['one'], color=C['one'], ls=(0, (2.5, 1.5)), marker='s', mfc='white', mec=C['one'], mew=.8, lw=1.1,
        ms=3.2, zorder=5)
ax.plot(qa, cur['exit'], color=C['Chorus'], marker='o', lw=1.8, ms=3.6, zorder=6)
for yv, c in ((cur['one'][-1], C['one']), (ext['adelie'][-1], C['adelie'])):
    ax.text(qa[-1] * 1.3, yv, f'{yv / cur["exit"][-1]:.1f}' + r'$\times$', va='center', ha='left', fontsize=A + .3,
            color=c, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(.75, 256 * 3.6)
ax.set_xticks(QS)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_ylim(.14, 24)
ax.set_yticks([.2, .5, 1, 2, 5, 10, 20])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel(r'Responses $q$', labelpad=1, fontsize=6.6)
ax.set_ylabel('Seconds per Response', labelpad=1, fontsize=6.6)
clean(ax)
title(ax, 'Stage 1', 'Time per Response')
label('a', .004, TOP + .01)
KEYA = [Line2D([], [], color=C['Chorus'], marker='o', lw=1.7, ms=3.6, label='Chorus'),
        Line2D([], [], color=C['one'], ls=(0, (2.5, 1.5)), marker='s', mfc='white', mec=C['one'], mew=.8, lw=1.1, ms=3.2,
               label='One per Batch')]
KEYA += [Line2D([], [], color=C[m], marker=MK[m], lw=1.1, ms=3.3, label=m) for m in EXT]
ax.legend(handles=KEYA, loc='center left', bbox_to_anchor=(1.03, .5), ncol=1, fontsize=6.3, handlelength=2.4,
          handletextpad=.4, labelspacing=.8, borderaxespad=0, frameon=False)

# ------------------------------------------------------------------ (b) peak working set against the budget
BX = LX + AW + .162
ax = fig.add_axes([BX, ROW1[0], LX + LW - BX, ROW1[1]])
ax.plot([0, 2.95], [0, 2.95], color='#9AA0A6', lw=.8, ls=(0, (1.5, 1.5)), zorder=1)
for key, name, ls in (('M', 'Tall', '-'), ('S', 'Stress', (0, (3, 1.4)))):
    pts = []
    for M in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        d = T2[T2.tag.isin([f'{key}{M}_full', f'{key}{M:.1f}_full', f'{key}{M:g}_full'])]
        if not len(d):
            continue
        r = d.iloc[-1]
        err = str(r.get('error') or '')
        peak = float(r.get('peak_wset') or 0) / GIB
        c = r.chorus if isinstance(r.chorus, dict) else {}
        if r.status == 'ok':
            pts.append((M, peak, 'ok', c.get('batch_size')))
        elif r.status == 'TIMEOUT':
            pts.append((M, peak, 'cap', None))
        elif 'start-up requirement' in err:
            m_ = re.search(r'requirement ([0-9.]+) GiB', err)
            pts.append((M, float(m_.group(1)), 'refused', None))
        elif 'single response' in err:
            pts.append((M, peak, 'infeasible', None))
    run = [p for p in pts if p[2] != 'refused']
    ax.plot([p[0] for p in run], [p[1] for p in run], color=C['Chorus'], ls=ls, lw=1.3, zorder=3)
    for M, yv, st, B in pts:
        if st == 'ok':
            ax.scatter([M], [yv], marker='o', s=15, color=C['Chorus'], edgecolor='white', lw=.4, zorder=5)
        elif st == 'cap':
            ax.scatter([M], [yv], marker='o', s=15, color='white', edgecolor=C['Chorus'], lw=.9, zorder=5)
        elif st == 'infeasible':
            ax.scatter([M], [yv], marker='x', s=15, color=C['Chorus'], lw=.9, zorder=5)
    lx, ly = run[-1][0], run[-1][1]
    ax.text(lx + .1, ly + (.2 if key == 'M' else .22), name, fontsize=A + .3, color=C['Chorus'], ha='right',
            va='bottom', fontweight='bold')
ax.text(2.25, 2.33, 'Budget', fontsize=A, color=MUTED, ha='center', va='bottom', rotation=31.5, rotation_mode='anchor')
ax.set_xlim(.5, 4.4)
ax.set_ylim(0, 3.75)
ax.set_xticks([1, 2, 3, 4])
ax.set_yticks([0, 1, 2, 3])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
ax.set_xlabel(r'Budget $M$ (GiB)', labelpad=1, fontsize=6.6)
ax.set_ylabel('Peak (GiB)', labelpad=1, fontsize=6.6)
ax.legend(handles=[Line2D([], [], marker='o', color='white', mec=C['Chorus'], ls='', ms=3.4, mew=.9, label='Time Cap'),
                   Line2D([], [], marker='x', color=C['Chorus'], ls='', ms=3.6, mew=.9, label='Infeasible')],
          loc='upper left', ncol=3, fontsize=A, handletextpad=.15, columnspacing=.7, borderaxespad=.15,
          handlelength=.9, frameon=False)
clean(ax)
title(ax, 'Both Stages', 'Peak Memory')
label('b', BX - .05, TOP + .01)

# ------------------------------------------------------------------ (c) two rows of small multiples over the sweep
sc = pd.read_csv(os.path.join(V2, 'sweep_certificates.csv'))
G = .016
W = (LW - 3 * G) / 4
ROWS_C = [(C1, 'max_gap', 'Gap', (3e-13, .8), [1e-12, 1e-6, 1], 1e-4),
          (C2, 'test_r2_change_max', r'$R^2$ Change', (3e-12, .2), [1e-10, 1e-6, 1e-2], None)]
for i, (wl, wlab) in enumerate(WL4):
    lo, hi = sc[sc.workload == wl].median_seconds.min(), sc[sc.workload == wl].median_seconds.max()
    for ri, (pos, col, ylab, ylim, yt, ref) in enumerate(ROWS_C):
        ax = fig.add_axes([LX + i * (W + G), pos[0], W, pos[1]])
        if ref:
            ax.axhline(ref, color='#B9BEC4', lw=.6, ls=(0, (1.5, 1.5)), zorder=1)
        for m in GLS[::-1]:
            d = sc[(sc.workload == wl) & (sc.method == m) & (sc[col] > 0)].sort_values('median_seconds')
            ax.plot(d.median_seconds, d[col], color=C[m], marker=MK[m], lw=1.4 if m == 'Chorus' else .85,
                    ms=2.9 if m == 'Chorus' else 2.4, zorder=6 if m == 'Chorus' else 4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pexp(v)))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(lo / 1.8, hi * 1.8)
        dec = range(int(np.floor(np.log10(lo / 1.8))) + 1, int(np.floor(np.log10(hi * 1.8))) + 1)
        ax.set_xticks([10 ** e for e in dec])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        if ri == 0:
            ax.set_xticklabels([])
            ax.text(.5, 1.04, wlab, transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5)
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(ylab, labelpad=1, fontsize=6.4)
        clean(ax)
        if i == 0 and ri == 0:
            title(ax, 'Stage 1', 'Precision Against Time', pad=.19)
fig.text(LX + LW / 2, C2[0] - .052, 'Seconds', ha='center', va='center', fontsize=6.6)
KEYS = [Line2D([], [], color=C[m], marker=MK[m], lw=1.4 if m == 'Chorus' else .85, ms=2.9 if m == 'Chorus' else 2.4,
               label=m) for m in GLS]
fig.legend(handles=KEYS, loc='lower right', bbox_to_anchor=(LX + LW + .004, C1[0] + C1[1] + .041), ncol=4,
           fontsize=6.3, handlelength=2.0, handletextpad=.35, columnspacing=1.0, borderaxespad=0, frameon=False)
label('c', .004, C1[0] + C1[1] + .045)

# ------------------------------------------------------------------ (d) one heat table of median seconds
DB = .33                          # bottom of (d): its colour bar label sits at the height of the seconds of (c)
ax = fig.add_axes([RX, DB, .997 - RX, TOP - DB])
COLS = [('q1', '1'), ('q4', '4'), ('q16', '16'), ('q64', '64'), ('q256', '256'), ('beijing', 'Beijing'),
        ('depmap64', 'DepMap')]
CW = [1, 1, 1, 1, 1, 1.62, 1.62]
CX = np.concatenate([[0], np.cumsum(CW)])
BLOCKS = [('Full Path', [('Chorus', 'chorus_batched', 'Chorus'), ('One per Batch', 'chorus_seq_ws', 'one'),
                         ('adelie', 'adelie_full', 'adelie'), ('skglm', 'skglm_full', 'skglm'),
                         ('Celer', 'Celer_full', 'Celer')]),
          ('Same Exit', [('Chorus', 'chorus_exit', 'Chorus'), ('adelie', 'adelie_exit', 'adelie'),
                         ('skglm', 'skglm_exit', 'skglm'), ('Celer', 'Celer_exit', 'Celer')])]
LAB0 = -2.62
BX0, BW, GMAX = CX[-1] + .45, 1.85, 16.0
y = 0.0
gm_all = {}
for bi, (btitle, rows) in enumerate(BLOCKS):
    ax.text(LAB0, y + .42, btitle, ha='left', va='center', fontsize=6.8, fontweight='bold')
    y += .85
    V = np.array([[med(f'{wl}_{key}') for wl, _ in COLS] for _, key, _ in rows])
    R = V / np.nanmin(V, axis=0)
    for i, (lab, key, ck) in enumerate(rows):
        for j in range(len(COLS)):
            t = min(np.log10(R[i, j]) / np.log10(30), 1.)
            ax.add_patch(Rectangle((CX[j], y + i), CW[j], 1, fc=HEAT(.03 + .94 * t), ec='white', lw=1.1, zorder=2))
            ax.text(CX[j] + CW[j] / 2, y + i + .53, fmt_s(V[i, j]), ha='center', va='center', fontsize=A, zorder=3,
                    color='white' if t > .55 else INK, fontweight='bold' if R[i, j] == 1 else 'normal')
        ax.text(-.14, y + i + .53, lab, ha='right', va='center', fontsize=6.3, color=C[ck],
                fontweight='bold' if ck == 'Chorus' else 'normal')
        g = float(np.exp(np.log(R[i]).mean()))
        gm_all[(btitle, lab)] = g
        ax.add_patch(Rectangle((BX0, y + i + .2), BW * min(g, GMAX) / GMAX, .62, fc=C[ck], ec='none', zorder=2))
        ax.text(BX0 + BW * min(g, GMAX) / GMAX + .1, y + i + .53, f'{g:.1f}' + r'$\times$', ha='left', va='center',
                fontsize=A, color=C[ck], fontweight='bold' if ck == 'Chorus' else 'normal')
    y += len(rows) + .35
y -= .35
for j, (_, lab) in enumerate(COLS):
    ax.text(CX[j] + CW[j] / 2, y + .2, lab, ha='center', va='top', fontsize=6.2)
ax.text(CX[5] / 2, y + 1.0, r'Responses $q$', ha='center', va='top', fontsize=6.4)
for tv in (1, 8, 16):
    xt = BX0 + BW * tv / GMAX
    ax.plot([xt, xt], [y + .05, y + .2], color=INK, lw=.5)
    ax.text(xt, y + .25, f'{tv}', ha='center', va='top', fontsize=6.0)
ax.plot([BX0, BX0 + BW], [y + .05, y + .05], color=INK, lw=.5)
ax.text(BX0 + BW / 2, y + 1.0, 'Geometric Mean', ha='center', va='top', fontsize=6.2)
cy = y + 1.75                     # colour bar under the cells
for k in range(60):
    ax.add_patch(Rectangle((CX[1] + k * (CX[6] - CX[1]) / 60, cy), (CX[6] - CX[1]) / 60 + .002, .42,
                           fc=HEAT(.03 + .94 * k / 59), ec='none', zorder=2))
ax.add_patch(Rectangle((CX[1], cy), CX[6] - CX[1], .42, fc='none', ec='#B8BCC2', lw=.4, zorder=3))
for tv in (1, 3, 10, 30):
    xt = CX[1] + (CX[6] - CX[1]) * np.log10(tv) / np.log10(30)
    ax.text(xt, cy + .55, f'{tv}', ha='center', va='top', fontsize=6.0)
ax.text((CX[1] + CX[6]) / 2, cy + 1.3, 'Times the Fastest in the Column', ha='center', va='top', fontsize=6.2)
ax.set_xlim(LAB0 - .05, BX0 + BW + 1.0)
ax.set_ylim(cy + 2.0, -.05)
ax.axis('off')
title(ax, 'Stage 1', 'Median Seconds in Memory')
label('d', RX - .03, TOP + .01)
print('geometric means', {k: round(v, 2) for k, v in gm_all.items()})

# ------------------------------------------------------------------ (e) and (f) cardiac Fig 5 boxes with response dots
ag = pd.read_csv(os.path.join(V2, 'sweep_agreement.csv'))
SLOT, BOXW = .225, .19
EW = .555                         # width of (e); (f) fills the rest of the row
ax = fig.add_axes([LX, ROW3[0], EW, ROW3[1]])
ax.set_yscale('log')
ax.set_ylim(3e-9, .6)
groups = [(wl, wlab, mode) for wl, wlab in WL4 for mode in ('Default', 'Matched')]
for gi, (wl, wlab, mode) in enumerate(groups):
    if gi:
        ax.axvline(gi - .5, color='#CDD1D6' if gi % 2 == 0 else SEP, lw=.8 if gi % 2 == 0 else .5, zorder=1)
    for j, m in enumerate(GLS):
        if mode == 'Default' or m == 'Chorus':
            tol = DEF_TOL[m]
        else:
            tol = sc[(sc.workload == wl) & (sc.method == m) & (sc.max_gap <= 1e-4)].tol.max()
        v = ag[(ag.workload == wl) & (ag.method == m) & np.isclose(ag.tol, tol, rtol=1e-6)].pred_diff_max.values
        if len(v):
            v = np.clip(v, 4e-9, None)
            dots(ax, gi + (j - 1.5) * SLOT, v, C[m], BOXW, rng)
            cbox(ax, gi + (j - 1.5) * SLOT, v, C[m], w=BOXW)
ax.set_xlim(-.5, len(groups) - .5)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[2] for g in groups], fontsize=6.2)
ax.tick_params(axis='x', length=0, pad=2)
for k, (wl, wlab) in enumerate(WL4):
    ax.text(2 * k + .5, -.21, wlab, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.6,
            fontweight='bold')
ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pexp(v)))
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_ylabel('Prediction Difference', labelpad=1, fontsize=6.6)
ax.grid(axis='x', visible=False)
clean(ax)
title(ax, 'Stage 1', 'Distance From the Exact Path')
fig.legend(handles=[Patch(fc=C[m], ec=C[m], label=m) for m in GLS], loc='lower right',
           bbox_to_anchor=(LX + EW + .004, ROW3[0] + ROW3[1] + .009), ncol=4, fontsize=6.3, handlelength=1.1,
           handleheight=.8, handletextpad=.35, columnspacing=1.0, borderaxespad=0, frameon=False)
label('e', .004, ROW3[0] + ROW3[1] + .012)

FX = LX + EW + .075
ax = fig.add_axes([FX, ROW3[0], .997 - FX, ROW3[1]])
ax.set_yscale('function', functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square))
ax.set_ylim(0, 92)
for gi, (wl, wlab) in enumerate(WL4):
    if gi:
        ax.axvline(gi - .5, color='#CDD1D6', lw=.8, zorder=1)
    for j, m in enumerate(GLS):
        v = 100 * (1 - ag[(ag.workload == wl) & (ag.method == m) &
                          np.isclose(ag.tol, DEF_TOL[m], rtol=1e-6)].share_same_support.values)
        if len(v):
            dots(ax, gi + (j - 1.5) * SLOT, v, C[m], BOXW, rng)
            cbox(ax, gi + (j - 1.5) * SLOT, v, C[m], w=BOXW)
ax.set_xlim(-.5, len(WL4) - .5)
ax.set_yticks([0, 5, 10, 20, 40, 60, 80])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_xticks(range(len(WL4)))
ax.set_xticklabels([wlab for _, wlab in WL4], fontsize=6.6)
for t in ax.get_xticklabels():
    t.set_fontweight('bold')
ax.tick_params(axis='x', length=0, pad=2)
ax.set_ylabel('Levels (%)', labelpad=1, fontsize=6.6)
ax.grid(axis='x', visible=False)
clean(ax)
title(ax, 'Stage 1', 'Levels With a Different Support')
label('f', FX - .045, ROW3[0] + ROW3[1] + .012)

style.save(fig, OUT)
print('saved', OUT)
