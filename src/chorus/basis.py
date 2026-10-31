"""Training-only spline bases shared by Chorus and every comparison method.

Each predictor j is expanded by a cubic B-spline basis with quantile knots, centered
on the training rows and orthonormalized so that X_j^T X_j / n = I. Varying-coefficient
terms x_j * beta_j(u) use the tensor columns x_j * B(u), orthonormalized the same way.
"""
from dataclasses import dataclass, field
import numpy as np
from sklearn.preprocessing import SplineTransformer


@dataclass
class _Term:
    kind: str            # 'additive' or 'varying'
    feature: int
    spline: object
    center: np.ndarray
    transform: np.ndarray
    index: int = -1      # index variable column for varying terms
    base: object = None  # additive term of the same feature (varying terms only)
    proj: np.ndarray = None
    raw_center: np.ndarray = None


def _spline(n_knots, degree, col):
    """Cubic B-spline with quantile knots, or uniform knots when ties make quantile
    knots coincide (for example zero-inflated expression)."""
    qs = np.quantile(col, np.linspace(0, 1, n_knots))
    knots = 'quantile' if np.all(np.diff(qs) > 1e-8 * max(np.ptp(col), 1e-12)) else 'uniform'
    return SplineTransformer(n_knots=n_knots, degree=degree, knots=knots, include_bias=True,
                             extrapolation='constant').fit(col[:, None])


def _orthonormalize(z, rank_tol=1e-10):
    center = z.mean(axis=0)
    zc = z - center
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    keep = s > rank_tol * s[0]
    t = vt[keep].T * (np.sqrt(len(z)) / s[keep])
    return center, t


@dataclass
class SplineDesign:
    """Additive (and optional varying-coefficient) group design."""
    n_knots: int = 4
    degree: int = 3
    terms: list = field(default_factory=list)

    def fit(self, x, varying=None, index=None, additive=None):
        """x: (n, p) training predictors. Columns in `additive` (default: all except the
        index) enter as smooth terms f_j(x_j); columns in `varying` additionally (or only)
        enter as varying-coefficient terms x_j * beta_j(u) with u = x[:, index]. A feature
        may carry both term types, as in varying-coefficient additive models (VCAM)."""
        x = np.asarray(x, dtype=np.float64)
        p = x.shape[1]
        varying = list(varying or [])
        additive = [j for j in range(p) if j != index] if additive is None else list(additive)
        self.terms, self.group_sizes_ = [], []
        for j in additive:
            if np.ptp(x[:, j]) <= 1e-12:
                continue
            sp = _spline(self.n_knots, self.degree, x[:, j])
            center, t = _orthonormalize(sp.transform(x[:, [j]]))
            self.terms.append(_Term('additive', j, sp, center, t))
            self.group_sizes_.append(t.shape[1])
        by_feature = {t.feature: t for t in self.terms}
        for j in varying:
            sp = _spline(self.n_knots, self.degree, x[:, index])
            raw = sp.transform(x[:, [index]]) * x[:, [j]]
            rc = raw.mean(axis=0)
            raw = raw - rc
            base, proj = by_feature.get(j), None
            if base is not None:
                # remove the additive span of x_j so the term carries only the variation in u
                za = (sp_base := base.spline.transform(x[:, [j]]) - base.center) @ base.transform
                proj = za.T @ raw / len(x)
                raw = raw - za @ proj
            center, t = _orthonormalize(raw)
            term = _Term('varying', j, sp, center, t, index, base, proj, rc)
            self.terms.append(term)
            self.group_sizes_.append(t.shape[1])
        self.n_features_in_ = p
        return self

    def _term_columns(self, term, x):
        if term.kind == 'varying':
            raw = term.spline.transform(x[:, [term.index]]) * x[:, [term.feature]] - term.raw_center
            if term.base is not None:
                za = (term.base.spline.transform(x[:, [term.feature]]) - term.base.center) @ term.base.transform
                raw = raw - za @ term.proj
        else:
            raw = term.spline.transform(x[:, [term.feature]])
        return (raw - term.center) @ term.transform

    def transform(self, x, dtype=np.float64):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty((len(x), sum(self.group_sizes_)), dtype=dtype, order='F')
        offset = 0
        for term in self.terms:
            z = self._term_columns(term, x)
            out[:, offset:offset + z.shape[1]] = z
            offset += z.shape[1]
        return out

    def transform_into(self, x, out, row_start=0):
        """Write the expanded rows of x into a preallocated (possibly memory-mapped)
        Fortran array, one term at a time, so the full float64 design never resides."""
        offset = 0
        for term in self.terms:
            z = self._term_columns(term, np.asarray(x, dtype=np.float64))
            out[row_start:row_start + len(x), offset:offset + z.shape[1]] = z
            offset += z.shape[1]

    @property
    def groups(self):
        ends = np.cumsum(self.group_sizes_)
        return np.stack([ends - np.asarray(self.group_sizes_), ends], axis=1)

    @property
    def term_features(self):
        return np.array([t.feature for t in self.terms])
