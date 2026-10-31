"""Validation tracking along regularization paths with cross-fitting folds.

The validation rows are split into two folds V1 and V2 and grouped into clusters (single
rows for exchangeable data, calendar weeks for hourly series). Every fitted column of a
path is tracked under three selectors: the level with the smallest error on V1 (S=0), on
V2 (S=1) and on all validation rows (S=2). For S=0 and S=1 the tracker keeps the
per-cluster squared-error sums of the selected fit on all clusters, so the cross-fitted
error of a candidate uses, for each cluster, the fit that was selected on the other fold.
"""
import numpy as np

REL = 1e-10       # relative improvement required to move the selected level


def make_folds(n_val, seed=0, clusters=None, fold=None):
    """Default folds: seeded random halves of the clusters (single rows when none given)."""
    clusters = np.arange(n_val) if clusters is None else np.asarray(clusters)
    if fold is None:
        ids = np.unique(clusters)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ids))
        f_of = np.zeros(len(ids), dtype=np.int64)
        f_of[perm[len(ids) // 2:]] = 1
        fold = f_of[np.searchsorted(ids, clusters)]
    return np.asarray(fold, dtype=np.int64), np.unique(clusters, return_inverse=True)[1]


class ValTracker:
    def __init__(self, n_val, B, K, fold=None, cluster=None, serve=None):
        self.n_val, self.B, self.K = n_val, B, K
        # serve (3, B) bool: selectors whose choice is needed for each column (patience only)
        self.serve = np.ones((3, B), bool) if serve is None else np.asarray(serve, bool)
        self.folds = fold is not None
        cluster = np.arange(n_val) if cluster is None else np.asarray(cluster)
        self.order = np.argsort(cluster, kind='stable')
        cs = cluster[self.order]
        self.starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        self.C = len(self.starts)
        if self.folds:
            fold = np.asarray(fold)
            self.cl_fold = fold[self.order][self.starts]
            for c in range(self.C):     # clusters must be nested in folds
                end = self.starts[c + 1] if c + 1 < self.C else n_val
                assert np.all(fold[self.order][self.starts[c]:end] == self.cl_fold[c])
            self.n_fold = np.array([(fold == 0).sum(), (fold == 1).sum(), n_val], dtype=float)
        else:
            self.cl_fold = np.zeros(self.C, dtype=np.int64)
            self.n_fold = np.array([np.nan, np.nan, n_val], dtype=float)
        self.sel = (0, 1, 2) if self.folds else (2,)
        self.best_val = np.full((3, B), np.inf)
        self.best_k = np.zeros((3, B), dtype=np.int32)
        self.closs = np.zeros((3, self.C, B))      # per-cluster loss sums of the selected fit
        self.val_mse = np.full((3, K, B), np.nan)

    def update(self, k, L, resid):
        """resid (n_val, |L|): validation residuals of live columns L at level k.
        Returns a dict S -> boolean mask over L of columns whose selected level moved to k."""
        sq = resid * resid
        cs = np.add.reduceat(sq[self.order], self.starts, axis=0)        # (C, |L|)
        out = {}
        for S in self.sel:
            if S == 2:
                mse = cs.sum(axis=0) / self.n_val
            else:
                mse = cs[self.cl_fold == S].sum(axis=0) / self.n_fold[S]
            self.val_mse[S, k, L] = mse
            better = mse < self.best_val[S, L] * (1 - REL)
            bl = L[better]
            self.best_val[S, bl] = mse[better]
            self.best_k[S, bl] = k
            self.closs[S][:, bl] = cs[:, better]
            out[S] = better
        return out

    def exhausted(self, k, L, patience):
        """Columns of L whose every selector has seen no improvement for `patience` levels."""
        done = np.ones(len(L), bool)
        for S in self.sel:
            done &= ((k - self.best_k[S, L]) >= patience) | ~self.serve[S, L]
        return done

    def crossfit(self):
        """(C, B) per-cluster loss of the fit selected on the other fold."""
        if not self.folds:
            return self.closs[2]
        use_s1 = (self.cl_fold == 0)[:, None]          # clusters of V1 are scored by the V2 fit
        return np.where(use_s1, self.closs[1], self.closs[0])

    def summary(self):
        return {'best_k': self.best_k.copy(), 'best_val': self.best_val.copy(),
                'closs': self.closs.copy(), 'crossfit': self.crossfit(), 'val_mse': self.val_mse.copy(),
                'cl_fold': self.cl_fold.copy()}
