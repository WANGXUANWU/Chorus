"""Table 1 of the 30 Oct revision: every solver with the design on disk under a 2 GiB budget.

Author comments of 30 Oct: the memory ledger (old Table 1) stood in Section 3 with measured values and symbols
defined later, so it was removed; the on-disk comparison of all solvers left Figure 2, which keeps one heat
table. One column: minutes and peak working set (GiB) of the eight solvers at 25,000 to 400,000 training rows
(p = 500, q = 64, design of 0.5 to 8 GB). Chorus runs both stages and selection; the other solvers fit their own
estimator. A run stopped at the budget or at the time cap of one hour keeps the time and the peak measured at
the stop, marked with a dagger (no empty cells). The same runs as panel (b) of Figure 2 v19.
Data: results/timing (n_scaling), results/timing2 (e2_rows), results/timing3 (tall runs).
Writes manuscript/tables/T1_ooc_a25.tex.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

GIB = 2 ** 30
TALL = [25000, 50000, 100000, 200000, 400000]
ROWS = [('Chorus', '\\textbf{\\sys{}}'), ('adelie', 'GL, adelie'), ('skglm', 'GL, skglm'), ('Celer', 'GL, Celer'),
        ('grpreg', 'GMCP, grpreg'), ('sparsegl', 'SGL, sparsegl'), ('SAM', 'SpAM, SAM'), ('biglasso', 'MTL, biglasso')]


def log(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['experiment', 'tag', 'method', 'status', 'size'])
    d = pd.DataFrame([json.loads(line) for line in open(path)])
    return d[d.status != 'DISTURBED']


T1 = log(os.path.join(ROOT, 'results', 'timing', 'timing_log.jsonl'))
T2 = log(os.path.join(ROOT, 'results', 'timing2', 'timing_log.jsonl'))
T3 = log(os.path.join(ROOT, 'results', 'timing3', 'timing_log.jsonl'))


def ooc(r, n):
    """The run of solver r at n rows, as selected for Figure 2 v19 (b)."""
    if r == 'Chorus':
        cand = [T2[(T2.experiment == 'e2_rows') & (T2.tag == f'n{n}_full')], T3[T3.tag == f'tall{n}_chorus']]
    elif r == 'adelie':
        cand = [T3[T3.tag == f'tall{n}_adelie'], T2[(T2.experiment == 'e2_rows') & (T2.tag == f'n{n}_adelie')]]
    else:
        cand = [T3[T3.tag == f'tall{n}_{r}'], T1[(T1.experiment == 'n_scaling') & (T1.method == r) & (T1['size'] == n)]]
    for d in cand:
        d = d[d.status.isin(['ok', 'OOM', 'TIMEOUT'])] if len(d) else d
        if len(d):
            ok = d[d.status == 'ok']
            row = (ok if len(ok) else d).iloc[-1].to_dict()
            if len(ok):
                row['fit_seconds'] = float(ok.fit_seconds.median())
            return row
    return None


def main():
    lines = ['\\begin{table}[t]',
             '\\caption{Fits with the design on disk and a budget of 2~GiB: minutes and peak working set (GiB) at '
             '25{,}000 to 400{,}000 rows. $\\dagger$: stopped at the budget or at the time cap, measured at the stop.}',
             '\\label{tab:ooc}', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{1.2pt}',
             '\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l' + 'r' * 10 + '@{}}', '\\toprule',
             '& \\multicolumn{5}{c}{Minutes} & \\multicolumn{5}{c}{Peak GiB}\\\\',
             '\\cmidrule(lr){2-6}\\cmidrule(l){7-11}',
             'Thousand rows & ' + ' & '.join([f'{n // 1000}' for n in TALL] * 2) + '\\\\',
             '\\midrule']
    done = {}
    for r, lab in ROWS:
        mins, peaks = [], []
        for n in TALL:
            v = ooc(r, n)
            ok = v['status'] == 'ok'
            m = (v['fit_seconds'] if ok else float(v.get('wall_seconds') or np.nan)) / 60
            p = float(v.get('peak_wset') or v.get('peak_rss') or v.get('peak_rss_sampled') or np.nan) / GIB
            ms = (f'{m:.1f}' if m < 100 else f'{m:.0f}')
            ps = f'{p:.2f}' if ok else f'{p:.1f}'
            if not ok:
                ms, ps = ms + '$^\\dagger$', ps + '$^\\dagger$'
            elif r == 'Chorus':
                ms, ps = '\\cellcolor{best}\\textbf{' + ms + '}', '\\textbf{' + ps + '}'
            mins.append(ms)
            peaks.append(ps)
            done[(r, n)] = ok
        lines.append(lab + ' & ' + ' & '.join(mins + peaks) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular*}', '\\end{table}']
    out = os.path.join(ROOT, 'manuscript', 'tables', 'T1_ooc_a25.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(out)
    print({r: [n for n in TALL if done[(r, n)]] for r, _ in ROWS})


if __name__ == '__main__':
    main()
