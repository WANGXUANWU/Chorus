"""Copy a real-data case with a different spline basis size (diagnostic for basis sensitivity).

The raw predictors, responses and splits are reused; only the spline expansion changes.
Usage: make_basis_case.py SRC_CASE DST_CASE N_KNOTS [DEGREE]
(n_knots=4, degree=3 gives the default d=5; n_knots=2 gives d=3, the size of SAM's basis.)
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from chorus import SplineDesign
from bench.caseio import read_meta, load, write_case

src, dst, n_knots = sys.argv[1], sys.argv[2], int(sys.argv[3])
degree = int(sys.argv[4]) if len(sys.argv) > 4 else 3
meta = read_meta(src)
xs = [np.asarray(load(src, 'xraw', s, meta)) for s in ('train', 'val', 'test')]
ys = [np.asarray(load(src, 'y', s, meta)) for s in ('train', 'val', 'test')]
n = [len(a) for a in xs]
sl = {'train': slice(0, n[0]), 'val': slice(n[0], n[0] + n[1]), 'test': slice(n[0] + n[1], sum(n))}
x, y = np.vstack(xs), np.vstack(ys)
sd = SplineDesign(n_knots=n_knots, degree=degree).fit(x[sl['train']], varying=meta.get('varying_features'),
                                                       index=meta.get('index_feature'))
keep = ('groups', 'D', 'q', 'p_raw', 'term_features', 'term_kinds', 'p_sam') + tuple(f'n_{s}' for s in sl)
extra = {k: v for k, v in meta.items() if k not in keep}
extra.update({'n_knots': n_knots, 'degree': degree, 'basis_source': os.path.basename(os.path.normpath(src))})
m = write_case(dst, sd, x, y, sl, extra=extra)
for f in ('info.json', 'timestamps.npy', 'truth.npz', 'spec.json'):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
sizes = np.diff(np.asarray(m['groups']), axis=1).ravel()
print(dst, 'D', m['D'], 'groups', len(sizes), 'sizes', sorted(set(sizes.tolist())))
