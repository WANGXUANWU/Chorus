"""Table 4 (amendment 22): leave-one-out ablation of Chorus on the simulation cases (layout of cardiac_FM
Table 5). Each row removes one component from the frozen configuration; values are the drop in F1 (points)
and the relative rise in FEE (percent), positive when the component helps. Gray: not significant
(paired t-test over cases, 5%). Reads results/v2/ablation_a22_L.csv; writes
manuscript/tables/T5_attribution_a22.tex.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

ROWS = [('Without reweighting', 'Without reweighting (group lasso)'),
        ('Without borrowing', 'Without borrowing ($\\alpha=0$ only)'),
        ('Without the EB selection', 'Without the empirical Bayes selection'),
        ('Without the choice of basis size', 'Without the choice of basis size ($d=5$)'),
        ('Without the relaxed refit', 'Without the relaxed refit')]
COLS = [('All', 'All'), ('kappa=0', '$\\kappa=0$'), ('kappa=1', '$\\kappa=1$'), ('Mixed', 'Mixed'),
        ('Rescaled', 'Rescaled'), ('Groups', 'Groups'), ('q=64', '$q=64$')]


def cell(v, p, pct):
    s = f'{v:+.1f}' + ('\\%' if pct else '')
    s = s.replace('-', '$-$')
    if abs(v) < .05:
        s = '0.0' + ('\\%' if pct else '')
    return s if p < .05 else '\\textcolor{gray}{' + s + '}'


def main():
    r = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'ablation_a22_L.csv'))
    lines = ['\\begin{table*}[t]',
             '\\caption{Change in accuracy when one component is removed from \\sys{}: drop in F1 (points) and relative '
             'rise in FEE, positive when the component helps. Columns: the 100 cases of the main grid, its cases with '
             '$\\kappa=0$ and $\\kappa=1$, and the sharing scenarios (mixed SNR, rescaled responses, two groups and 64 '
             'responses without shared predictors). The refit keeps every support, so its F1 entries are zero. '
             'Gray: not significant (paired $t$-test over cases, 5\\% level).}',
             '\\label{tab:attribution}', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{2.4pt}',
             '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'r' * (2 * len(COLS)) + '@{}}', '\\toprule',
             f'& \\multicolumn{{{len(COLS)}}}{{c}}{{F1 loss (points)}} & \\multicolumn{{{len(COLS)}}}{{c}}{{FEE increase}}\\\\',
             f'\\cmidrule(lr){{2-{1 + len(COLS)}}}\\cmidrule(lr){{{2 + len(COLS)}-{1 + 2 * len(COLS)}}}',
             'Configuration & ' + ' & '.join(l for _, l in COLS) + ' & ' + ' & '.join(l for _, l in COLS) + '\\\\', '\\midrule']
    for key, lab in ROWS:
        d = r[r.row == key].set_index('column')
        f1 = [cell(d.loc[c, 'f1_gain'], d.loc[c, 'f1_p'], False) for c, _ in COLS]
        fe = [cell(d.loc[c, 'fee_gain'], d.loc[c, 'fee_p'], True) for c, _ in COLS]
        lines.append(lab + ' & ' + ' & '.join(f1 + fe) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular*}', '\\end{table*}']
    out = os.path.join(ROOT, 'manuscript', 'tables', 'T5_attribution_a22.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
