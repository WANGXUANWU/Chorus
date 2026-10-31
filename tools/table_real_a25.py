"""Table 3 of the 30 Oct revision: real workloads at the model size of Chorus, with the raw results beside them.

Author comment of 30 Oct: the table should show the comparison that matters with the best values marked. The
primary columns give the test R2 at the model size of Chorus (for each response and baseline, the path model of
either basis size with the smallest validation error among those no larger than the model of Chorus, as in
Figure 4a), DepMap pooled over its six splits and Beijing at both time origins. The raw test R2 of every method
at its own validation-selected size and the median model size stay beside them, so the trade between accuracy
and size remains visible. DepMap values average each dependency over the six splits first, so means, standard
errors and paired t-tests run over the 200 dependencies; Beijing runs over its 72 responses. Bold on a shaded
cell: best of a column; underline: second best; dagger: not significantly different from the best (5%).
Model sizes are reported, not ranked.
Reads results/v2/matched_size_per_response.csv and real_a22_per_response.csv; writes
manuscript/tables/T4_real_a25.tex.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ORDER = ['Chorus', 'Adaptive GL', 'GMCP', 'SGL', 'SpAM', 'GL', 'MTL']
DEP = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
RAWCASE = {'D0': 'depmap_c200', 'D1': 'depmap_c200_s3101', 'D2': 'depmap_c200_s3102', 'D3': 'depmap_c200_s3103',
           'D4': 'depmap_c200_s3104', 'D5': 'depmap_c200_s3105', 'B1': 'beijing', 'B2': 'beijing_o2'}
SHADE = '\\cellcolor{best}'


def f3(v):
    return f'{v:.3f}'.lstrip('0') if v >= 0 else '$-$' + f'{abs(v):.3f}'.lstrip('0')


def matched():
    """Per response and method: test R2 at the model size of Chorus (Chorus itself at its own size)."""
    m = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'matched_size_per_response.csv'))
    rows = [m[['case', 'response', 'r2_chorus']].drop_duplicates(['case', 'response'])
            .rename(columns={'r2_chorus': 'r2'}).assign(method='Chorus')]
    rows.append(m[['case', 'response', 'method', 'r2_matched']].rename(columns={'r2_matched': 'r2'}))
    return pd.concat(rows, ignore_index=True)


def raw():
    r = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'real_a22_per_response.csv'))
    inv = {v: k for k, v in RAWCASE.items()}
    r = r[r.protocol == 'aligned'].copy()
    r['case'] = r.case.map(inv)
    return r


def column(d, cases):
    """Mean, standard error and the per-unit matrix (DepMap: dependencies averaged over the splits)."""
    sub = d[d.case.isin(cases)]
    w = sub.groupby(['method', 'response']).r2.mean().unstack(0)[ORDER]
    return w.mean(), w.std(ddof=1) / np.sqrt(len(w)), w


def marks(mean, w):
    shown = mean.round(3)
    levels = sorted(shown.unique(), reverse=True)
    best = shown.idxmax()
    out = {}
    for k in ORDER:
        s = f3(mean[k])
        if shown[k] == levels[0]:
            s = SHADE + '\\textbf{' + s + '}'
        elif shown[k] == levels[1]:
            s = '\\underline{' + s + '}'
        if shown[k] != levels[0] and ttest_rel(w[k], w[best]).pvalue >= .05:
            s += '$^\\dagger$'
        out[k] = s
    return out


def main():
    mt, rw = matched(), raw()
    cols = []
    for d, se_on in ((mt, True), (rw, False)):
        for cases in (DEP, ['B1'], ['B2']):
            mean, se, w = column(d, cases)
            mk = marks(mean, w)
            cols.append({k: mk[k] + ('{\\tiny$\\,\\pm$' + f3(se[k]) + '}' if se_on else '') for k in ORDER})
            if d is mt and cases == DEP:
                p = {k: ttest_rel(w['Chorus'], w[k]).pvalue for k in ORDER if k != 'Chorus'}
                print('DepMap pooled at equal size', mean.round(4).to_dict(), 'p', {k: round(v, 4) for k, v in p.items()})
            if d is mt and cases != DEP:
                print(cases, 'equal size', mean.round(4).to_dict())
    size = {}
    for k in ORDER:
        dep = rw[(rw.method == k) & rw.case.isin(DEP)]['size'].median()
        bj = rw[(rw.method == k) & rw.case.isin(['B1', 'B2'])]['size'].median()
        size[k] = (f'{dep:.0f}', f'{bj:.0f}')
    lines = ['\\begin{table}[t]',
             '\\caption{Test $R^2$ on the real workloads at the model size of \\sys{} and at the size each method '
             'selects. Bold on a shaded cell: best. Underline: second best. $\\dagger$: not significantly different '
             'from the best.}',
             '\\label{tab:real}', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{1.3pt}',
             '\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrrrr@{}}', '\\toprule',
             '& \\multicolumn{3}{c}{At the size of \\sys{} $\\uparrow$} & \\multicolumn{3}{c}{Own size $\\uparrow$} & '
             '\\multicolumn{2}{c}{Size}\\\\',
             '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-9}',
             'Method & DepMap & B1 & B2 & DepMap & B1 & B2 & Dep. & Bj.\\\\', '\\midrule']
    for k in ORDER:
        lines.append(('\\textbf{\\sys{}}' if k == 'Chorus' else k) + ' & ' + ' & '.join(c[k] for c in cols) + ' & '
                     + ' & '.join(size[k]) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular*}', '\\end{table}']
    out = os.path.join(ROOT, 'manuscript', 'tables', 'T4_real_a25.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(out)


if __name__ == '__main__':
    main()
