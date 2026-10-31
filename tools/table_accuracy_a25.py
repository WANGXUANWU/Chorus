"""Table 2 of the 30 Oct revision: accuracy of every method in all three simulation families.

Author comments of 30 Oct: mark the best value of every row plainly and give the comparison that carries the
story. Layout of cardiac_FM Table 3 at full width: one column per method, rows grouped by family (main grid,
robustness, sharing) with the seven metrics of the rank radar of Figure 3a (F1, MCC, AUPRC, FPR, size error,
FEE, test R2), mean and standard error over cases in every cell. The best mean of a row is bold on a shaded
cell, the second best is underlined, and a dagger marks a mean not significantly different from the best
(paired t-test over cases, 5%). TPR is left out as in Figure 3a: it rewards dense supports, and F1 and MCC
already weigh it against the false positives.
Writes manuscript/tables/T3_accuracy_a25.tex.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ORDER = ['Chorus', 'Adaptive GL', 'GMCP', 'SGL', 'SpAM', 'GL', 'MTL']
FAMILIES = [('formal', 'Main grid'), ('robust', 'Robustness'), ('sharing', 'Sharing')]
METRICS = [('f1', 'F1', False, 3), ('mcc', 'MCC', False, 3), ('auprc', 'AUPRC', False, 3),
           ('fpr', 'FPR', True, 4), ('size_err', 'Size error', True, 2), ('fee', 'FEE', True, 4),
           ('r2', 'Test $R^2$', False, 3)]
SHADE = '\\cellcolor{best}'


def num(v, dig):
    s = f'{v:.{dig}f}'
    return s.lstrip('0') if abs(v) < 1 and not s.startswith('-') else s


def row(d, met, low, dig):
    w = d.pivot(index='case', columns='method', values=met)[ORDER]
    mean, se = w.mean(), w.std(ddof=1) / np.sqrt(len(w))
    shown = mean.round(dig)
    levels = sorted(shown.unique(), reverse=not low)
    best = shown.idxmin() if low else shown.idxmax()
    out = []
    for m in ORDER:
        s = num(mean[m], dig)
        if shown[m] == levels[0]:
            s = SHADE + '\\textbf{' + s + '}'
        elif len(levels) > 1 and shown[m] == levels[1]:
            s = '\\underline{' + s + '}'
        if shown[m] != levels[0] and ttest_rel(w[m], w[best]).pvalue >= .05:
            s += '$^\\dagger$'
        out.append(s + '{\\tiny$\\,\\pm$' + num(se[m], dig if dig >= 3 else 2) + '}')
    return out, best


def compact(d):
    """Variant with one row per method and the three families as column blocks, means only (the full table is
    about 3.6 in tall and cannot share a page with Figure 3 or Figure 4)."""
    cells = {m: [] for m in ORDER}
    for fam, _ in FAMILIES:
        sub = d[d.family == fam]
        for met, lab, low, dig in METRICS:
            out, _ = row(sub, met, low, min(dig, 3) if met != 'fpr' else 4)
            for m, c in zip(ORDER, out):
                cells[m].append(c.split('{\\tiny')[0])
    nm = len(METRICS)
    head = ' & '.join(f'{lab} {"$\\downarrow$" if low else "$\\uparrow$"}' for _, lab, low, _ in METRICS)
    lines = ['\\begin{table*}[t]',
             '\\caption{Accuracy of all methods in the three simulation families, means over cases. Bold on a shaded '
             'cell: best. Underline: second best. $\\dagger$: not significantly different from the best.}',
             '\\label{tab:accuracy}', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{1.0pt}',
             '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'r' * (nm * 3) + '@{}}', '\\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{{nm}}}{{c}}{{{name}, {d[d.family == fam].case.nunique()} cases}}'
                                for fam, name in FAMILIES) + '\\\\',
             ''.join(f'\\cmidrule(lr){{{2 + k * nm}-{1 + (k + 1) * nm}}}' for k in range(3)),
             'Method & ' + ' & '.join([head.replace('Size error', 'Size').replace('Test $R^2$', '$R^2$')] * 3) + '\\\\',
             '\\midrule']
    for m in ORDER:
        lines.append(('\\textbf{\\sys{}}' if m == 'Chorus' else m) + ' & ' + ' & '.join(cells[m]) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular*}', '\\end{table*}']
    return lines


def main():
    d = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sim_a22.csv'))
    d = d[d.method.isin(ORDER)].copy()
    d['size_err'] = (d['size'] - d['s']).abs()
    out = os.path.join(ROOT, 'manuscript', 'tables', 'T3_accuracy_a25_compact.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(compact(d)) + '\n')
    print(out)
    lines = ['\\begin{table*}[t]',
             '\\caption{Accuracy of all methods in the three simulation families, mean $\\pm$ standard error over '
             'cases. Bold on a shaded cell: best. Underline: second best. $\\dagger$: not significantly different '
             'from the best.}',
             '\\label{tab:accuracy}', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{2.0pt}',
             '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'r' * len(ORDER) + '@{}}', '\\toprule',
             'Metric & ' + ' & '.join('\\textbf{\\sys{}}' if m == 'Chorus' else m for m in ORDER) + '\\\\', '\\midrule']
    wins = 0
    for fi, (fam, fname) in enumerate(FAMILIES):
        sub = d[d.family == fam]
        if fi:
            lines.append('\\midrule')
        lines.append(f'\\multicolumn{{{len(ORDER) + 1}}}{{@{{}}l}}{{\\textit{{{fname}, '
                     f'{sub.case.nunique()} cases}}}}\\\\')
        for met, lab, low, dig in METRICS:
            cells, best = row(sub, met, low, dig)
            wins += best == 'Chorus'
            lines.append(f'\\quad {lab} {"$\\downarrow$" if low else "$\\uparrow$"} & ' + ' & '.join(cells) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular*}', '\\end{table*}']
    out = os.path.join(ROOT, 'manuscript', 'tables', 'T3_accuracy_a25.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(out, 'Chorus best in', wins, 'of', len(FAMILIES) * len(METRICS), 'rows')


if __name__ == '__main__':
    main()
