"""Shared figure style: Times New Roman text, deep saturated palette, PVLDB widths.

PVLDB two-column layout: text width 7.0 in, column width 3.33 in. Every method keeps one
color and marker in every figure; Chorus is drawn last, thickest and in crimson.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

TEXT_W, COL_W = 7.0, 3.33

METHODS = ['Chorus', 'grpreg', 'SAM', 'sparsegl', 'Celer', 'skglm', 'biglasso']
LABEL = {'Chorus': 'Chorus', 'grpreg': 'grpreg (gMCP)', 'SAM': 'SAM (SpAM)', 'sparsegl': 'sparsegl (SGL)',
         'Celer': 'Celer (GL)', 'skglm': 'skglm (GL)', 'biglasso': 'biglasso (MTL)'}
SHORT = {'Chorus': 'Chorus', 'grpreg': 'grpreg', 'SAM': 'SAM', 'sparsegl': 'sparsegl', 'Celer': 'Celer',
         'skglm': 'skglm', 'biglasso': 'biglasso'}
COLOR = {'Chorus': '#D1243F',      # crimson rose
         'grpreg': '#2166AC',      # blue
         'SAM': '#12957F',         # teal green
         'sparsegl': '#EE9A1F',    # amber
         'Celer': '#8A5BC4',       # lavender purple
         'skglm': '#5B6E8C',       # slate
         'biglasso': '#7E7E7E'}    # grey
MARKER = {'Chorus': 'o', 'grpreg': 's', 'SAM': 'D', 'sparsegl': '^', 'Celer': 'v', 'skglm': 'P',
          'biglasso': 'X'}
INK = '#1A1A1A'
GRID = '#D9D9D9'
TINT = {'navy': '#1F3B73', 'teal': '#0B7A75', 'crimson': '#C8102E', 'amber': '#D97A00',
        'purple': '#6A3D9A', 'green': '#2E7D32', 'slate': '#44546A'}


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    serif = 'Times New Roman' if 'Times New Roman' in names else 'DejaVu Serif'
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': [serif], 'mathtext.fontset': 'stix',
        'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.3,
        'ytick.labelsize': 6.3, 'legend.fontsize': 6.5, 'axes.linewidth': .6,
        'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.2,
        'ytick.major.size': 2.2, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .4, 'grid.linestyle': '-',
        'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.1,
        'lines.markersize': 3.2, 'savefig.dpi': 400, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.dpi': 150})
    return serif


def panel_label(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom', ha='left')


def method_handles(methods=METHODS, lw=1.1):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=lw * (1.6 if m == 'Chorus' else 1),
                   ms=3.6 if m == 'Chorus' else 3.0, label=LABEL[m]) for m in methods]


def save(fig, path):
    fig.savefig(path + '.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close(fig)


def human(v, _=None):
    """Compact tick labels: 0.5, 2, 30, 1k, 25k, 1M."""
    if v >= 1e6:
        s = f'{v / 1e6:g}M'
    elif v >= 1e3:
        s = f'{v / 1e3:g}k'
    else:
        s = f'{v:g}'
    return s


def logaxis(ax, axis='x', ticks=None, base=10):
    from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator, NullLocator
    a = ax.xaxis if axis == 'x' else ax.yaxis
    (ax.set_xscale if axis == 'x' else ax.set_yscale)('log', base=base)
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
        a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(human))
    a.set_minor_formatter(NullFormatter())


def dodge_box(ax, data, groups, methods, slot=.86, fill=.82, shade=True, log=False, mean=False):
    """Grouped box plots in the style of cardiac_FM Fig. 5: one group per x position, one narrow
    box per method with a gap between boxes, light shading of alternate groups, white median.

    data[g][m] is the array of values of method m in group g."""
    import numpy as np
    nm = len(methods)
    w = slot / nm
    for gi, g in enumerate(groups):
        if shade and gi % 2 == 0:
            ax.axvspan(gi - .5, gi + .5, color='#F3F4F7', zorder=0, lw=0)
        for mi, m in enumerate(methods):
            v = np.asarray(data[gi][mi], dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            x = gi + (mi - (nm - 1) / 2) * w
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            lo = v[v >= q1 - 1.5 * iqr].min()
            hi = v[v <= q3 + 1.5 * iqr].max()
            c = COLOR[m]
            ax.plot([x, x], [lo, q1], color=c, lw=.55, zorder=2, solid_capstyle='butt')
            ax.plot([x, x], [q3, hi], color=c, lw=.55, zorder=2, solid_capstyle='butt')
            bw = w * fill
            span = (q3 - q1) if not log else (np.log10(max(q3, 1e-12)) - np.log10(max(q1, 1e-12)))
            lim = ax.get_ylim()
            rel = (q3 - q1) / max(lim[1] - lim[0], 1e-12) if not log else span / max(np.log10(lim[1] / max(lim[0], 1e-12)), 1e-12)
            if rel < .03:
                ax.plot([x - bw / 2, x + bw / 2], [med, med], color=c, lw=2.4, zorder=3, solid_capstyle='butt')
            else:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((x - bw / 2, q1), bw, q3 - q1, fc=c, ec=c, lw=.4, zorder=3))
                ax.plot([x - bw / 2, x + bw / 2], [med, med], color='white', lw=.8, zorder=4, solid_capstyle='butt')
            if mean:
                ax.scatter([x], [v.mean()], s=4, color='white', edgecolor=c, lw=.4, zorder=5)
    ax.set_xlim(-.5, len(groups) - .5)
    ax.set_xticks(range(len(groups)))
    ax.grid(axis='x', visible=False)
