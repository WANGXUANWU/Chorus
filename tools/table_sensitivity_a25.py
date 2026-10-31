"""Table 5 of the 30 Oct revision: the sensitivity table of tools/table_sensitivity_a24.py cut to one column.

Author comment of 30 Oct: the sensitivity analysis took too much of the evaluation. One row per varied choice
(the group is named in the row), five columns over the 100 cases of the main grid (F1, MCC, FPR, FEE and test
R2, standard errors after F1 and FEE) and adaptive GL, the best baseline, as an unranked reference row. Bold on a
shaded cell: best setting of a column; underline: second best (as printed).
Reads results/v2/sensitivity_a22_per_case.csv and sim_a22.csv; writes manuscript/tables/T6_sensitivity_a25.tex.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

ROWS = [('zeta', '0.5', '\\textbf{Frozen configuration}'),
        ('zeta', '0.25', '$\\zeta=0.25$'), ('zeta', '0.75', '$\\zeta=0.75$'), ('zeta', '1.0', '$\\zeta=1$'),
        ('alpha grid', '2', '$\\alpha\\in\\{0,1\\}$'), ('alpha grid', '3', '$\\alpha\\in\\{0,0.5,1\\}$'),
        ('gamma', '1', '$\\gamma=1$ only'), ('gamma', '2', '$\\gamma=2$ only'),
        ('basis', '3', '$d=3$ only'), ('basis', '5', '$d=5$ only'),
        ('patience', '3', 'Patience $\\pi=3$'), ('patience', '8', 'Patience $\\pi=8$'),
        ('patience', 'full', 'Full path, no exit'), ('relaxed refit', 'off', 'Without the refit')]
COLS = [('f1', 3, False, True), ('mcc', 3, False, False), ('fpr', 4, True, False), ('fee', 4, True, True),
        ('r2', 3, False, False)]
SHADE = '\\cellcolor{best}'


def num(v, dig):
    return f'{v:.{dig}f}'.lstrip('0') if abs(v) < 1 else f'{v:.{dig}f}'


def stats(d):
    return [(float(d[m].mean()), float(d[m].std(ddof=1) / np.sqrt(len(d)))) for m, *_ in COLS]


def main():
    P = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sensitivity_a22_per_case.csv'), dtype={'value': str})
    D = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sim_a22.csv'))
    agl = D[(D.family == 'formal') & (D.method == 'Adaptive GL')]
    data = [(lab, stats(P[(P.parameter == par) & (P.value == val)])) for par, val, lab in ROWS]
    assert all(len(P[(P.parameter == par) & (P.value == val)]) == 100 for par, val, _ in ROWS)
    marks = []
    for j, (met, dig, low, _) in enumerate(COLS):
        shown = sorted({round(s[j][0], dig) for _, s in data}, reverse=not low)
        marks.append((shown[0], shown[1]))

    def cell(j, mean, se, ranked=True):
        met, dig, low, with_se = COLS[j]
        s = num(mean, dig)
        if ranked and round(mean, dig) == marks[j][0]:
            s = SHADE + '\\textbf{' + s + '}'
        elif ranked and round(mean, dig) == marks[j][1]:
            s = '\\underline{' + s + '}'
        return s + ('{\\tiny$\\,\\pm$' + num(se, dig) + '}' if with_se else '')

    lines = ['\\begin{table}[t]',
             '\\caption{Sensitivity of \\sys{} on the main grid, one choice varied at a time. Bold on a shaded cell: '
             'best setting. Underline: second best.}',
             '\\label{tab:sensitivity}', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{1.5pt}',
             '\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}', '\\toprule',
             'Setting & F1 $\\uparrow$ & MCC $\\uparrow$ & FPR $\\downarrow$ & FEE $\\downarrow$ & '
             'Test $R^2$ $\\uparrow$\\\\', '\\midrule']
    for lab, s in data:
        lines.append(lab + ' & ' + ' & '.join(cell(j, m, e) for j, (m, e) in enumerate(s)) + '\\\\')
    lines.append('\\midrule')
    lines.append('\\textit{Adaptive GL} & ' + ' & '.join(cell(j, m, e, ranked=False)
                                                         for j, (m, e) in enumerate(stats(agl))) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular*}', '\\end{table}']
    out = os.path.join(ROOT, 'manuscript', 'tables', 'T6_sensitivity_a25.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(out)
    ref = stats(agl)
    print('min F1 lead over adaptive GL', round(min(s[0][0] for _, s in data) - ref[0][0], 4),
          'settings with FEE above adaptive GL', [lab for lab, s in data if s[3][0] > ref[3][0]])


if __name__ == '__main__':
    main()
