"""Update-strategy grid (reviewer comment 7): fixed covariance, fixed residual, the earlier
threshold rule and the cost model, on in-memory group lasso paths of one batch.

Cells: n in {500, 2000, 8000, 32000} x p in {250, 1000} x B in {1, 16, 64} x s in {3, 12}
(kappa = 0.5, rho = 0.5, SNR 2, d = 5). Every cell is generated once and each strategy is
timed in the same process (fresh working set), interleaved over two repetitions; the minimum
is kept. Usage: run_planner_grid.py OUT_CSV
"""
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
from chorus import SplineDesign, InMemoryDesign
from chorus.engine import fit_batch
from chorus.costmodel import constants
from data.simulate import simulate


def main(out, ns=(500, 2000, 8000, 32000), Bs=(1, 16, 64), reps=2):
    constants()
    rows = []
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        rows = prev.to_dict('records')
        done = {(r['n'], r['p'], r['B'], r['s']) for r in rows}
    for n, p, B, s in itertools.product(ns, (250, 1000), Bs, (3, 12)):
        if (n, p, B, s) in done:
            continue
        sim = simulate(n, 10, 10, p, B, s=s, kappa=.5, rho=.5, snr=2., seed=5000 + n + p + B + s)
        x = sim['x'][:n]
        sd = SplineDesign().fit(x)
        X = sd.transform(x)
        y = sim['y'][:n] - sim['y'][:n].mean(0)
        d = InMemoryDesign(X, sd.groups)
        pen = np.repeat(np.sqrt(sd.groups[:, 1] - sd.groups[:, 0])[:, None].astype(float), B, 1)
        f = np.geomspace(1, .02, 30)
        res = {}
        for rep in range(reps):
            for mode in ('gram', 'residual', 'threshold', 'auto'):
                t = time.perf_counter()
                r = fit_batch(d, y, f, pen, tol=1e-6, mode=mode)
                el = time.perf_counter() - t
                key = mode
                if key not in res or el < res[key][0]:
                    res[key] = (el, r['mode_levels'].count('gram'), len(r['working_set']))
        row = {'n': n, 'p': p, 'D': d.D, 'B': B, 's': s}
        for mode, (el, gl, w) in res.items():
            row[f't_{mode}'] = el
            row[f'gram_levels_{mode}'] = gl
            row[f'width_{mode}'] = w * 5
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    # optional reduced grid: run_planner_grid.py OUT N_LIST B_LIST REPS, e.g. 500,2000,8000 1,16 1
    if len(sys.argv) > 2:
        main(sys.argv[1], tuple(int(v) for v in sys.argv[2].split(',')), tuple(int(v) for v in sys.argv[3].split(',')),
             int(sys.argv[4]))
    else:
        main(sys.argv[1])
