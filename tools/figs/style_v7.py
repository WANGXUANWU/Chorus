"""Method palette of the second revision (amendment 21 figures).

The cardiac_FM hues of style_v6, re-stepped so that every pair of the eight methods passes the
normal-vision floor (OKLab Delta E >= 15) and the worst colour-blind pair stays in the 6 to 8 band
that markers make legal (tools/figs/validate_palette.py, all pairs). SAM is labelled by its
package, because Chorus also fits sparse additive models. Annotations never go below 6 pt.
"""
import style

ACC = ['Chorus', 'adaptive GL', 'gMCP (grpreg)', 'GL + 1-SE', 'SGL (sparsegl)', 'SpAM (SAM)', 'GL (skglm)', 'MTL (biglasso)']
LABEL = {'Chorus': 'Chorus', 'adaptive GL': 'Adaptive GL', 'gMCP (grpreg)': 'gMCP (grpreg)', 'GL + 1-SE': 'GL, 1-SE Rule',
         'SGL (sparsegl)': 'SGL (sparsegl)', 'SpAM (SAM)': 'SAM', 'GL (skglm)': 'GL (skglm, Celer, adelie)',
         'MTL (biglasso)': 'MTL (biglasso)'}
SHORT = {'Chorus': 'Chorus', 'adaptive GL': 'Adaptive GL', 'gMCP (grpreg)': 'gMCP', 'GL + 1-SE': 'GL 1-SE',
         'SGL (sparsegl)': 'SGL', 'SpAM (SAM)': 'SAM', 'GL (skglm)': 'GL', 'MTL (biglasso)': 'MTL'}
COLOR = {'Chorus': '#B4304A', 'adaptive GL': '#9B59B6', 'gMCP (grpreg)': '#3A8DDE', 'GL + 1-SE': '#EDA100',
         'SGL (sparsegl)': '#A57A55', 'SpAM (SAM)': '#2A9D84', 'GL (skglm)': '#56606E', 'MTL (biglasso)': '#B0B0B0'}
MARKER = {'Chorus': 'o', 'adaptive GL': 'h', 'gMCP (grpreg)': 's', 'GL + 1-SE': '<', 'SGL (sparsegl)': '^',
          'SpAM (SAM)': 'D', 'GL (skglm)': 'P', 'MTL (biglasso)': 'X'}
EDGE = {m: ('#7A7A7A' if m == 'MTL (biglasso)' else c) for m, c in COLOR.items()}   # light grey needs an edge
HEAT_ACC = ['#FBF0F3', '#F3C9D5', '#E490A8', '#C9557F', '#9B2C5F']    # accuracy heat cells (cardiac_FM)
HEAT_TIME = ['#F2F7FC', '#CFE1F2', '#93C0E3', '#4A8FCB', '#1F5A9A']   # time and outcome cells (one blue hue)
INK, MUTED, GRID = '#1A1A1A', '#5F6368', '#DADCE0'
ANNOT = 6.0          # smallest text size at print scale


def setup():
    serif = style.setup()
    import matplotlib as mpl
    mpl.rcParams.update({'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4, 'legend.fontsize': 6.6,
                         'axes.labelsize': 7, 'axes.titlesize': 7.4})
    return serif


def handles(methods=ACC, lw=1.1):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLOR[m], marker=MARKER[m], mec=EDGE[m], lw=lw * (1.6 if m == 'Chorus' else 1),
                   ms=3.8 if m == 'Chorus' else 3.1, label=LABEL[m]) for m in methods]
