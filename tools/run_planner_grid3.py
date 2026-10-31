"""Strategy grid with the statistics strategy (amendment 21, E8).

The 24 cells of the reduced grid (amendment 20: n in {500, 2000, 8000}, p in {250, 1000},
B in {1, 16}, s in {3, 12}) are timed with fixed covariance updates, fixed residual updates,
the earlier threshold rule, the engine's cost model ('auto') and the statistics strategy (one
Gram product, then the certified Gram path), interleaved in one process with one repetition.
The planner of Chorus chooses statistics or the engine from predict_strategy; its time is
the time of the chosen execution. Usage: run_planner_grid3.py OUT_CSV
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
from chorus.gram_path import fit_batch_gram
from chorus.costmodel import constants, predict_strategy
from chorus.model import _screen_widths
from data.simulate import simulate
from bench.quiet import wait_quiet, Background


def stats_fit(d, y, f, pen):
    g, c, _ = d.stats_scan(y)
    sizes = d.groups[:, 1] - d.groups[:, 0]
    return fit_batch_gram(g, c, np.einsum('ij,ij->j', y, y) / d.n, sizes, f, pen, tol=1e-6, max_rounds=200,
                          ws_cap=True, tighten=True, chunked=True)


def main(out, ns=(500, 2000, 8000), Bs=(1, 16)):
    constants()
    rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
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
        row = {'n': n, 'p': p, 'D': d.D, 'B': B, 's': s}
        for attempt in range(5):                       # quiet machine, repeat when disturbed (amendment 21)
            wait_quiet()
            with Background() as bg:
                for mode in ('gram', 'residual', 'threshold', 'auto', 'stats'):
                    t = time.perf_counter()
                    r = stats_fit(d, y, f, pen) if mode == 'stats' else fit_batch(d, y, f, pen, tol=1e-6, mode=mode)
                    row[f't_{mode}'] = time.perf_counter() - t
                    row[f'max_gap_{mode}'] = float(r['gaps'].max())
            if bg.mean <= .3:
                break
        row['bg_cores_mean'] = bg.mean
        m1, m2 = _screen_widths(d.xt_dot(y) / n, d.groups, pen, f)
        t_s, t_w = predict_strategy(n, d.D, B, len(f), B, False, None, m1, m2)
        row['pred_stats'], row['pred_ws'] = t_s, t_w
        row['choice'] = 'stats' if 1.25 * t_s < t_w else 'auto'
        row['t_planner'] = row[f"t_{row['choice']}"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
