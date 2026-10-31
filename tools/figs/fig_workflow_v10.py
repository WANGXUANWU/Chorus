"""Figure 1 (revision of 31 Oct): v8 with stronger text, every position and icon unchanged.

Author comment of 31 Oct: some words look too thin, adjust them without touching the layout. Labels that name
a result or a decision are bold (Fits?, Reads Overlap Products, Exact Gaps, No Pass per Level, Violation, Exit
after pi, Next lambda). The captions inside the four step boxes stay regular in a darker tone, because bold
'Gram or Residual' is 2 pt wider than its box, and gray notes use a darker gray. Text sizes stay.
Left: block store and memory budget. Middle: stage 1, either the statistics strategy or the
working-set loop of one level. Right: stage 2, reweighting, 22 candidates, cross-fitted empirical
Bayes selection and the relaxed refit. Blue marks objects shared by the batch, amber objects of one
response, slate the storage device, teal certified results.
Usage: fig_workflow_v8.py OUT
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Ellipse, Polygon
import matplotlib.colors as mc
import style
import style_v7 as sv

sv.setup()
OUT = sys.argv[1] if len(sys.argv) > 1 else 'manuscript/figures/F1_workflow_v10'
SH, PR, DV, OK, ST2, CR = '#2F6FB3', '#E08A00', '#56606E', '#2A9D84', '#8E44AD', sv.COLOR['Chorus']
INK, MUTED = sv.INK, sv.MUTED
NOTE = '#40464F'                          # gray notes, darker than MUTED, which stays for thin lines
DSH, DOK = '#235389', '#1F7563'           # captions inside the step boxes, darker SH and OK
FIG_W, FIG_H = style.TEXT_W, 2.38
ASP = FIG_W / FIG_H                       # x units per y unit of equal length
T = 6.0                                   # body text size (print points)


def tint(c, t):
    a = np.array(mc.to_rgb(c))
    return tuple(a + (1 - a) * t)


def rbox(x, y, w, h, color, fill=.07, lw=.7, r=.010, ls='-', z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', lw=lw, ec=color,
                                fc=tint(color, 1 - fill), ls=ls, zorder=z))


def arrow(p0, p1, color=INK, lw=.65, ms=5.5, cs='arc3,rad=0', ls='-', z=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color,
                                 connectionstyle=cs, zorder=z, shrinkA=0, shrinkB=0, linestyle=ls))


def line(xs, ys, color=INK, lw=.65, ls='-', z=8):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='round')


def txt(x, y, s, size=T, color=INK, ha='center', va='center', w='normal', z=9, bg=False):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, fontweight=w, zorder=z, linespacing=.95,
            bbox=dict(fc='white', ec='none', pad=.5) if bg else None)


def check(x, y, s, color, lw=1.3):
    """Check mark of half-width s centred at (x, y)."""
    line([x - s, x - s * .25, x + s], [y, y - s * ASP * .55, y + s * ASP * .75], color, lw=lw, z=10)


def badge(x, y, n, color, d=.0150):
    """Small numbered disc of diameter d (x units) whose left edge is at x."""
    ax.add_patch(Ellipse((x + d / 2, y), d, d * ASP, color=color, zorder=10, lw=0))
    txt(x + d / 2, y - .002, str(n), 6.0, 'white', w='bold', z=11)
    return x + d + .004                   # where the title text starts


def grid(x, y, w, h, n, cols, rng, z=5):
    """Mini heat grid (a Gram matrix) with a dark diagonal."""
    cw, ch = w / n, h / n
    for i in range(n):
        for j in range(n):
            v = rng.random() if i != j else 1.
            c = cols[0] if v < .45 else cols[1] if v < .8 else cols[2]
            ax.add_patch(Rectangle((x + j * cw, y + (n - 1 - i) * ch), cw * .88, ch * .88, fc=c, ec='none', zorder=z))


def columns(x, y, w, h, ncol, colors, gap=.0022, z=5):
    cw = (w - gap * (ncol - 1)) / ncol
    for i in range(ncol):
        ax.add_patch(Rectangle((x + i * (cw + gap), y), cw, h, fc=colors[i % len(colors)], ec='none', zorder=z))


fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
rng = np.random.default_rng(4)
BLUES = [tint(SH, .80), tint(SH, .45), SH]
SLATES = ['#A7B0BD', '#7D8899']

# ======================================================================== A: store and budget
A0, AW = .006, .190
rbox(A0, .085, AW, .845, DV, fill=.035, lw=.7)
txt(A0 + .010, .895, 'Store and Budget', 6.8, DV, ha='left', w='bold')
cx, cy, cw, chh = A0 + .050, .600, .070, .190                    # store cylinder with column blocks
ax.add_patch(Rectangle((cx - cw / 2, cy), cw, chh, fc='#EEF0F3', ec=DV, lw=.65, zorder=3))
ax.add_patch(Ellipse((cx, cy + chh), cw, .050, fc='#F8F9FA', ec=DV, lw=.65, zorder=4))
ax.add_patch(Ellipse((cx, cy), cw, .050, fc='#EEF0F3', ec=DV, lw=.65, zorder=2))
bx = cx - cw / 2 + .007
for j in range(4):                                                # blocks X_1..X_p of d_j columns
    columns(bx + j * .0145, cy + .066, .0125, chh - .102, 3, SLATES, gap=.0011)
txt(bx + .006, cy + .040, r'$\mathbf{X}_1$', 6.0, DV)
txt(bx + .0285, cy + .040, r'$\cdots$', 6.0, DV)
txt(bx + .050, cy + .040, r'$\mathbf{X}_p$', 6.0, DV)
txt(cx, cy - .048, r'Block Store $\mathbf{X}$', 6.2, DV, w='bold')
txt(cx, cy - .086, r'$n\times D$ on Disk', T, NOTE)
px, pw = A0 + .110, .072                                          # prefetch timeline: reads overlap products
txt(px + pw / 2, .790, 'Prefetch', 6.2, SH, w='bold')
for row, (lab, col, off) in enumerate((('Read', DV, 0.), ('GEMM', SH, .012))):
    yy = .730 - row * .052
    line([px, px + pw], [yy - .006, yy - .006], '#C9CDD3', lw=.4, z=3)
    for k in range(3):
        ax.add_patch(Rectangle((px + off + k * .021, yy), .017, .026, fc=tint(col, .25 if row else .35),
                               ec=col, lw=.4, zorder=5))
txt(px + pw / 2, .640, 'Reads Overlap', T, SH, w='bold')
txt(px + pw / 2, .607, 'Products', T, SH, w='bold')
arrow((cx + cw / 2 + .003, .710), (px - .004, .743), DV, lw=.65, ms=5)
txt(px + pw / 2, .552, '64 MiB Ranges', T, NOTE)
txt(px + pw / 2, .514, 'No File Cache', T, NOTE)
txt(A0 + .010, .425, r'Memory $\leq M$', 6.4, INK, ha='left', w='bold')
segs = [(r'$M_0$', .17, '#B9BEC6', INK), ('Fixed', .21, '#8E9AAA', 'white'), ('Batch', .23, PR, 'white'),
        ('Gram', .29, SH, 'white'), (r'$\delta$', .10, '#E6E8EC', INK)]
x0, y0, W, H = A0 + .010, .318, AW - .020, .066
cur = x0
for lab, frac, col, ink in segs:
    ax.add_patch(Rectangle((cur, y0), frac * W, H, fc=col, ec='white', lw=.9, zorder=5))
    txt(cur + frac * W / 2, y0 + H / 2, lab, T, ink, w='bold')
    cur += frac * W
txt(A0 + AW / 2, .272, 'When the Gram Would Overflow', T, NOTE)
cxs = [x0 + .028, x0 + .086, x0 + .146]
for (lab, col), xx in zip((('Evict', PR), ('Split', PR), ('Refuse', CR)), cxs):
    rbox(xx - .025, .140, .050, .066, col, fill=.10, lw=.6, r=.008)
    txt(xx, .173, lab, T, col, w='bold')
for a_, b_ in zip(cxs[:-1], cxs[1:]):
    arrow((a_ + .026, .173), (b_ - .026, .173), MUTED, lw=.55, ms=4.5)

# ======================================================================== B: stage 1
B0, BW = .205, .505
rbox(B0, .085, BW, .845, SH, fill=.035, lw=.75)
txt(B0 + .010, .895, 'Stage 1: Group Lasso Path of Every Response', 6.8, SH, ha='left', w='bold')
dx, dy, dr = B0 + .042, .505, .032                 # strategy decision
ax.add_patch(Polygon([(dx - dr, dy), (dx, dy + dr * ASP), (dx + dr, dy), (dx, dy - dr * ASP)], closed=True,
                     fc='white', ec=SH, lw=.75, zorder=6))
txt(dx, dy + .020, r'$8D^2$', T, SH, w='bold')
txt(dx, dy - .027, 'Fits?', T, SH, w='bold')
arrow((A0 + AW, .505), (dx - dr - .002, dy), DV, lw=.65)
line([dx, dx], [dy + dr * ASP, .742], SH)
arrow((dx, .742), (B0 + .095, .742), SH, lw=.65)
txt(dx - .007, .665, 'Yes', T, OK, ha='right', w='bold')
line([dx, dx], [dy - dr * ASP, .300], SH)
arrow((dx, .300), (B0 + .095, .300), SH, lw=.65)
txt(dx - .007, .350, 'No', T, CR, ha='right', w='bold')
# --- statistics lane
L1y, L1h, LX, LW = .650, .185, B0 + .095, .400
rbox(LX, L1y, LW, L1h, OK, fill=.06, lw=.65)
txt(LX + .010, L1y + L1h - .030, 'Statistics: One Pass', 6.2, OK, ha='left', w='bold')
gh = .088
gx0 = LX + .020
grid(gx0, L1y + .045, gh / ASP * 1.25, gh, 6, BLUES, rng)      # 0.009 higher: the labels cleared the lane border
txt(gx0 + gh / ASP * .62, L1y + .026, r'$\mathbf{X}^{\top}\mathbf{X}$', T, SH)
columns(gx0 + .062, L1y + .045, .014, gh, 1, [tint(PR, .15)])
txt(gx0 + .069, L1y + .026, r'$\mathbf{X}^{\top}\mathbf{Y}$', T, PR)
arrow((gx0 + .090, L1y + .080), (gx0 + .116, L1y + .080), INK, lw=.6, ms=5)
ppx, ppy, ppw, pph = gx0 + .124, L1y + .028, .100, .105
ax.add_patch(Rectangle((ppx, ppy), ppw, pph, fc='white', ec='#C9CDD3', lw=.45, zorder=4))
lam = np.linspace(0, 1, 40)
for k, (start, top) in enumerate(((.10, .92), (.30, .70), (.50, .48), (.68, .28))):
    yv = np.clip((lam - start) / (1 - start), 0, None) ** .8 * top
    ax.plot(ppx + lam * ppw, ppy + .010 + yv * (pph - .02), color=[PR, '#C7792B', tint(PR, .35), tint(PR, .55)][k],
            lw=.8, zorder=6)
txt(ppx + ppw / 2, ppy + pph + .022, 'Gram Path', T, INK, w='bold')
ex = ppx + ppw + .034
check(ex, L1y + .118, .010, OK)
txt(ex, L1y + .062, 'Exact Gaps', T, OK, w='bold')
txt(LX + LW - .036, L1y + .104, 'No Pass', T, NOTE, w='bold')
txt(LX + LW - .036, L1y + .068, 'per Level', T, NOTE, w='bold')
# --- working-set lane: one penalty level
L2y, L2h = .110, .505
rbox(LX, L2y, LW, L2h, CR, fill=.035, lw=.65)
txt(LX + .010, L2y + L2h - .030, r'Working Sets: Level $\lambda_{r,k}$, Batch $\mathcal{B}$', 6.2, CR, ha='left', w='bold')
sw, shh, sy = .093, .214, L2y + .176
sx = [LX + .006 + i * .0985 for i in range(4)]
titles = ['Candidates', 'Solve', 'Shared Scan', 'Certify']
cols = [PR, SH, SH, OK]
for i, (x, tl, c) in enumerate(zip(sx, titles, cols)):
    rbox(x, sy, sw, shh, c, fill=.08, lw=.65, r=.009, z=2)
    t0 = badge(x + .006, sy + shh - .028, i + 1, c)
    txt(t0, sy + shh - .029, tl, T, c, ha='left', w='bold')
for r_ in range(3):                                # 1: candidate sets of three responses, their union
    yy = sy + .136 - r_ * .031
    ax.add_patch(Rectangle((sx[0] + .012, yy), .069, .021, fc='white', ec='#C9CDD3', lw=.35, zorder=4))
    for j in np.flatnonzero(rng.random(9) < .45):
        ax.add_patch(Rectangle((sx[0] + .012 + j * .069 / 9, yy), .069 / 9 * .9, .021, fc=PR, ec='none', zorder=5))
ax.add_patch(Rectangle((sx[0] + .012, sy + .047), .069, .019, fc=SH, ec='none', zorder=5))
txt(sx[0] + .0465, sy + .024, r'Union $\mathcal{W}$', T, DSH)
gw = .040                                          # 2: Gram or residual updates chosen by cost
grid(sx[1] + .010, sy + .058, gw, gw * ASP * .72, 6, BLUES, rng)
ax.add_patch(Rectangle((sx[1] + .010 + gw / 6, sy + .058 + gw * ASP * .72 / 2), gw / 3, gw * ASP * .72 / 3,
                       fill=False, ec=PR, lw=.9, zorder=7))
txt(sx[1] + .057, sy + .101, 'or', T, NOTE)
columns(sx[1] + .067, sy + .058, .010, gw * ASP * .72, 1, [tint(PR, .2)])
columns(sx[1] + .079, sy + .058, .006, gw * ASP * .72, 1, ['#9AA5B4'])
txt(sx[1] + .0465, sy + .024, 'Gram or Residual', T, DSH)
columns(sx[2] + .012, sy + .058, .030, .094, 5, SLATES)     # 3: X^T R_B in one read
txt(sx[2] + .049, sy + .105, r'$\times$', 6.6, INK)
columns(sx[2] + .057, sy + .058, .022, .094, 4, [PR, tint(PR, .3)])
txt(sx[2] + .0465, sy + .024, r'$\mathbf{X}^{\top}\mathbf{R}_{\mathcal{B}}$, One Read', T, DSH)
qx, qy, qw, qh = sx[3] + .012, sy + .058, .068, .096         # 4: primal and dual close to a gap below eps
ax.add_patch(Rectangle((qx, qy), qw, qh, fc='white', ec='#C9CDD3', lw=.4, zorder=4))
tt = np.linspace(0, 1, 40)
prim = qy + qh * (.50 + .40 * np.exp(-4 * tt))
dual = qy + qh * (.44 - .34 * np.exp(-4 * tt))
ax.plot(qx + .004 + tt * (qw - .016), prim, color=CR, lw=.85, zorder=6)
ax.plot(qx + .004 + tt * (qw - .016), dual, color=SH, lw=.85, zorder=6)
gxp = qx + qw - .010
line([gxp, gxp], [dual[-1], prim[-1]], OK, lw=1.6, z=7)
line([qx + .003, qx + qw - .003], [qy + qh * .47, qy + qh * .47], MUTED, lw=.4, ls=':', z=5)
txt(qx + qw - .004, qy + qh - .016, 'Primal', T, CR, ha='right')
txt(qx + qw - .004, qy + .016, 'Dual', T, SH, ha='right')
txt(sx[3] + .0465, sy + .024, r'KKT, gap$_r\leq\epsilon$', T, DOK)
for a_, b_ in zip(sx[:-1], sx[1:]):
    arrow((a_ + sw + .001, sy + shh / 2), (b_ - .001, sy + shh / 2), INK, lw=.6, ms=5)
vt = sy + shh + .022                               # violation: back to the candidates
line([sx[3] + sw / 2, sx[3] + sw / 2], [sy + shh, vt], CR, lw=.65)
line([sx[3] + sw / 2, sx[0] + sw / 2], [vt, vt], CR, lw=.65)
arrow((sx[0] + sw / 2, vt), (sx[0] + sw / 2, sy + shh + .002), CR, lw=.65, ms=5)
txt(sx[2] + .010, vt, r'Violation: Enlarge $\mathcal{W}_r$', T, CR, w='bold', bg=True)
vy = L2y + .015                                    # 5: validation-aware exit
t0 = badge(sx[0] + .006, vy + .108, 5, OK)
txt(t0, vy + .107, 'Exit', T, OK, ha='left', w='bold')
txt(sx[0] + .046, vy + .045, 'Validation\nError', T, NOTE)
cvx, cvw = sx[1] + .008, .175
uu = np.linspace(0, 1, 50)
ax.plot(cvx + uu * cvw, vy + .018 + .068 * (uu - .42) ** 2 / .34, color=OK, lw=.95, zorder=6)
line([cvx + .42 * cvw] * 2, [vy + .004, vy + .125], MUTED, lw=.5, ls=':')
line([cvx + .72 * cvw] * 2, [vy + .004, vy + .125], CR, lw=.65, ls='--')
txt(cvx + .42 * cvw - .006, vy + .110, 'Best', T, NOTE, ha='right')
txt(cvx + .72 * cvw + .005, vy + .110, r'Exit after $\pi$', T, CR, ha='left', w='bold')
arrow((sx[3] + .0465, sy - .002), (sx[3] + .0465, vy + .080), INK, lw=.6, ms=5)
txt(sx[3] + .0465, vy + .048, r'Next $\lambda_{r,k+1}$', T, INK, w='bold')

# ======================================================================== C: stage 2
C0, CWd = .720, .274
rbox(C0, .085, CWd, .845, ST2, fill=.035, lw=.75)
txt(C0 + .010, .895, 'Stage 2: Reweight and Select', 6.8, ST2, ha='left', w='bold')
arrow((LX + LW + .002, .742), (C0 - .002, .742), SH, lw=.65)
arrow((LX + LW + .002, .360), (C0 - .002, .360), CR, lw=.65)
txt(C0 + .137, .815, r'$w_{jr}=\sqrt{d_j}\,\{(1-\alpha)a_{jr}+\alpha\bar a_j\}^{-\gamma}$', 6.3, ST2)
txt(C0 + .137, .765, r'Scale-Free $a_{jr}$, Gram of $\mathcal{U}$', T, NOTE)
txt(C0 + .012, .700, '22 Candidates', 6.2, ST2, ha='left', w='bold')
for row, dlab in enumerate((r'$d$=5', r'$d$=3')):
    yy = .630 - row * .054
    txt(C0 + .028, yy + .018, dlab, T, INK)
    for j in range(11):
        ax.add_patch(Rectangle((C0 + .050 + j * .019, yy), .016, .036, fc=tint(ST2, .86 - .075 * j), ec='none', zorder=5))
txt(C0 + .137, .535, r'GL and 10 Pairs $(\alpha,\gamma)$', T, NOTE)
txt(C0 + .012, .470, 'Cross-Fitted EB', 6.2, ST2, ha='left', w='bold')
ly, mid = .392, C0 + .137
line([C0 + .020, C0 + .254], [ly, ly], '#C9CDD3', lw=.5, z=4)
line([mid, mid], [ly - .040, ly + .045], ST2, lw=.65, ls='--', z=5)
for xv in (.032, .066, .098, .178, .208, .242):
    x_ = C0 + xv
    tgt = mid + (x_ - mid) * .42
    ax.scatter(x_, ly, s=9, color=tint(ST2, .55), edgecolor=ST2, lw=.35, zorder=6)
    ax.scatter(tgt, ly, s=9, color=ST2, edgecolor='white', lw=.3, zorder=7)
    arrow((x_, ly + .024), (tgt, ly + .024), MUTED, lw=.45, ms=3.8)
txt(mid, ly - .065, r'Score on $V_2$ the Fit Chosen on $V_1$', T, NOTE)
arrow((mid, ly - .095), (mid, .248), ST2, lw=.6, ms=5)
rbox(C0 + .012, .105, .250, .138, CR, fill=.10, lw=.75)
xx = np.linspace(0, 1, 40)
ax.plot(C0 + .025 + xx * .075, .135 + .062 * np.sin(np.pi * xx) ** 2, color=CR, lw=1.1, zorder=6)
ax.plot(C0 + .025 + xx * .075, .135 + .036 * np.sin(np.pi * xx) ** 2, color=tint(CR, .5), lw=.8, ls='--', zorder=6)
txt(C0 + .178, .198, 'Relaxed Refit', 6.2, CR, w='bold')
txt(C0 + .178, .152, 'Selected Fit per Response', T, INK)

# ======================================================================== legend
items = [('Shared by the Batch', SH), ('One Response', PR), ('Storage Device', DV), ('Certified', OK)]
lx = .300
for name, col in items:
    ax.add_patch(Rectangle((lx, .028), .013, .026, fc=col, ec='none', zorder=5))
    txt(lx + .018, .041, name, 6.1, INK, ha='left')
    lx += .018 + .0058 * len(name) + .040
style.save(fig, OUT)
print('saved', OUT)
