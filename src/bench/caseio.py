"""Case directories shared by every method: identical designs, responses and splits.

A case stores the training-centered spline design of each split as column-major float64
binaries (readable by numpy memory maps and by R readBin), the raw predictors for methods
that build their own basis (SAM), centered responses, the group layout and the common
regularization-path settings. Ground truth, when known, is stored for evaluation only.
"""
import json
import os
import numpy as np

SPLITS = ('train', 'val', 'test')


def _write(path, a):
    np.asfortranarray(a, dtype=np.float64).T.tofile(path)   # column-major bytes


def write_case(root, design, x_raw, y, split, extra=None, truth=None, raw_cols=None):
    """design: fitted SplineDesign (fit on training rows); x_raw (n, p) raw predictors
    (the matrix from which the design is computed); y (n, q); split: dict of slices."""
    os.makedirs(root, exist_ok=True)
    tr = split['train']
    ymean = y[tr].mean(axis=0)
    meta = {'groups': design.groups.tolist(), 'D': int(design.groups[-1, 1]),
            'q': int(y.shape[1]), 'p_raw': int(x_raw.shape[1]),
            'term_features': design.term_features.tolist(),
            'term_kinds': [t.kind for t in design.terms]}
    xc = None
    for name in SPLITS:
        sl = split[name]
        X = design.transform(x_raw[sl])
        if name == 'train':
            xc = X.mean(axis=0)                                   # zero up to rounding
        _write(os.path.join(root, f'X_{name}.f64'), X - xc)
        _write(os.path.join(root, f'y_{name}.f64'), y[sl] - ymean)
        xr = x_raw[sl] if raw_cols is None else x_raw[sl][:, raw_cols]
        _write(os.path.join(root, f'xraw_{name}.f64'), xr)
        meta[f'n_{name}'] = int(X.shape[0])
    meta['p_sam'] = int(x_raw.shape[1] if raw_cols is None else len(raw_cols))
    meta.update(extra or {})
    with open(os.path.join(root, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)
    if truth is not None:
        np.savez_compressed(os.path.join(root, 'truth.npz'), **truth)
    return meta


def read_meta(root):
    with open(os.path.join(root, 'meta.json'), encoding='utf-8') as fh:
        return json.load(fh)


def load(root, kind, name, meta=None, mmap=False):
    """kind in {'X', 'y', 'xraw'}; returns an (n, k) Fortran float64 array (or memmap)."""
    meta = meta or read_meta(root)
    n = meta[f'n_{name}']
    k = {'X': meta['D'], 'y': meta['q'], 'xraw': meta['p_sam']}[kind]
    path = os.path.join(root, f'{kind}_{name}.f64')
    if mmap:
        return np.memmap(path, dtype=np.float64, mode='r', shape=(n, k), order='F')
    return np.fromfile(path, dtype=np.float64).reshape((k, n)).T
