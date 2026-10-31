"""Add the raw training predictors to the disk-resident simulation cases.

The disk cases were written without the raw training predictors, which SAM needs because it
builds its own basis. The rows are regenerated exactly from the per-chunk seeds of
src/bench/make_disk_case.py and written column-major chunk by chunk, so memory stays small.
The procedure is first checked on the validation rows, whose raw predictors were stored.
Usage: add_disk_xraw.py CASE_DIR [CASE_DIR ...]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from data.simulate import ar1_uniform
from bench.caseio import load

CHUNK = 20000          # chunk size used by tools/make_disk_cases.py
args = sys.argv[1:]
if args and args[0].startswith('--chunk='):        # the wide cases use 5,000-row chunks
    CHUNK = int(args.pop(0).split('=')[1])

for root in args:
    meta = json.load(open(os.path.join(root, 'meta.json')))
    par = meta['params']
    n, p, rho, seed = meta['n_train'], par['p'], par['rho'], par['seed']
    # check: the validation rows come from seed * 1000 + 999 and must match the stored file
    xv = ar1_uniform(np.random.default_rng(seed * 1000 + 999), meta['n_val'] + meta['n_test'], p, rho)
    stored = np.asarray(load(root, 'xraw', 'val', meta))
    err = np.abs(xv[:meta['n_val']] - stored).max()
    assert err == 0, (root, err)
    path = os.path.join(root, 'xraw_train.f64')
    mm = np.memmap(path, dtype=np.float64, mode='w+', shape=(n, p), order='F')
    for k, c0 in enumerate(range(0, n, CHUNK)):
        c1 = min(n, c0 + CHUNK)
        mm[c0:c1] = ar1_uniform(np.random.default_rng(seed * 1000 + 1 + k), c1 - c0, p, rho)
    mm.flush()
    del mm
    print(root, 'xraw_train written', os.path.getsize(path), 'bytes; validation check exact', flush=True)
