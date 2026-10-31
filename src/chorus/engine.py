"""Batched working-set block coordinate descent (BCD) for many group-lasso paths.

For a batch of B responses sharing one design, every pass over the data is a matrix
product with B right-hand sides, so one read of each predictor block serves the whole
batch. Inside the union working set W the solver uses one of two update modes, chosen by
a cost rule: covariance updates on the shared Gram matrix G_WW = X_W^T X_W / n, whose
cost does not depend on n (tall or disk-resident data), or block-major residual updates
that reuse each block for all B responses (n small relative to |W|). Exactness is
certified per response by a duality gap computed on the full design.

Objective for response r (columns of X centered and group-orthonormal, y centered):
    P_r(b) = ||y_r - X b||^2 / (2n) + lam_r * sum_j w_jr ||b_j||_2 ,  w_jr in (0, inf].
"""
import time
import numpy as np
from .kernels import solve_batch, residual_bcd, compact_gram
from .ledger import BatchOverflow
from .costmodel import prefer_gram, prefer_residual
from .validation import ValTracker


def group_norms(m, groups):
    """(D, B) -> (G, B) Euclidean norms of contiguous row groups."""
    return np.sqrt(np.add.reduceat(m * m, groups[:, 0], axis=0))


class _Full(Exception):
    pass


class WorkingSet:
    """Union working set W of a batch: ids, coefficients and the cached quantities of the
    active update mode (Gram matrix, or transposed columns), grown incrementally. With a
    width cap m_max the Gram buffer never exceeds m_max x m_max."""

    def __init__(self, design, B, mode, m_max=None):
        self.design, self.mode, self.m_max = design, mode, m_max
        self.ids, self.pos, self.off = [], {}, [0]
        self.coef = np.zeros((0, B))
        self.gram = np.zeros((0, 0))
        self._cap = np.zeros((0, 0))
        self.xwt = np.zeros((0, design.n))
        self.grow_seconds = 0.
        self.evicted = 0

    @property
    def m(self):
        return self.off[-1]

    def width(self, groups):
        return int(sum(self.design.groups[j, 1] - self.design.groups[j, 0] for j in groups))

    def extend(self, new):
        new = [int(j) for j in new if int(j) not in self.pos]
        if not new:
            return
        d = self.design
        m0 = self.m
        m1 = m0 + self.width(new)
        if self.m_max is not None and m1 > self.m_max:
            raise _Full()
        t0 = time.perf_counter()
        if self.mode == 'gram':
            g_nn = d.cross(new, new)
            g_wn = d.cross(self.ids, new)
            if m1 > self._cap.shape[0]:
                cap = max(m1, int(1.5 * self._cap.shape[0]), 256)
                if self.m_max is not None:
                    cap = min(cap, self.m_max)
                buf = np.empty((cap, cap))
                buf[:m0, :m0] = self._cap[:m0, :m0]
                self._cap = buf
            self._cap[:m0, m0:m1], self._cap[m0:m1, :m0] = g_wn, g_wn.T
            self._cap[m0:m1, m0:m1] = g_nn
            self.gram = self._cap[:m1, :m1]
        else:
            xn = d.cols(new)
            self.xwt = np.ascontiguousarray(np.vstack([self.xwt, xn.T]))
        width = m1 - m0
        self.coef = np.vstack([self.coef, np.zeros((width, self.coef.shape[1]))])
        for j in new:
            a, b = d.groups[j]
            self.pos[j] = len(self.ids)
            self.ids.append(j)
            self.off.append(self.off[-1] + int(b - a))
        self.grow_seconds += time.perf_counter() - t0

    def to_gram(self):
        """Switch from residual to covariance updates: the Gram matrix of the resident columns."""
        m, n = self.m, self.design.n
        cap = max(m, 256) if self.m_max is None else min(max(m, 256), self.m_max)
        self._cap = np.empty((cap, cap))
        if m:
            # the leading block of a wider buffer is not C-contiguous, so no out= product here
            self._cap[:m, :m] = self.xwt[:m] @ self.xwt[:m].T
            self._cap[:m, :m] /= n
        self.gram = self._cap[:m, :m]
        self.xwt = np.zeros((0, n))
        self.mode = 'gram'

    def to_residual(self):
        """Switch from covariance back to residual updates (resident designs only): the
        transposed columns of W replace the Gram matrix; the caller recomputes the residuals.
        On short and wide designs the candidate sets near the dense end of a path grow past
        n columns, where a covariance epoch costs more than a residual epoch (amendment 22)."""
        n = self.design.n
        self.xwt = np.ascontiguousarray(self.design.cols(self.ids).T) if self.ids else np.zeros((0, n))
        self._cap = np.zeros((0, 0))
        self.gram = self._cap
        self.mode = 'residual'

    def evict(self, keep_groups):
        """Drop every block of W not in keep_groups; returns the column map for the batch
        coefficients (rows to keep), so callers can compact their own arrays."""
        keep_groups = set(int(j) for j in keep_groups)
        kept = [j for j in self.ids if j in keep_groups]
        if len(kept) == len(self.ids):
            return None
        rows = np.concatenate([np.arange(self.off[self.pos[j]], self.off[self.pos[j] + 1]) for j in kept])             if kept else np.zeros(0, np.int64)
        rows = rows.astype(np.int64)
        if self.mode == 'gram' and len(rows):
            compact_gram(self._cap, rows)
        elif self.mode != 'gram':
            self.xwt = np.ascontiguousarray(self.xwt[rows])
        self.coef = self.coef[rows]
        self.evicted += len(self.ids) - len(kept)
        self.ids, self.pos, self.off = [], {}, [0]
        for j in kept:
            a, b = self.design.groups[j]
            self.pos[j] = len(self.ids)
            self.ids.append(j)
            self.off.append(self.off[-1] + int(b - a))
        self.gram = self._cap[:self.m, :self.m] if self.mode == 'gram' else self.gram
        return rows

    def bounds(self):
        return np.stack([self.off[:-1], self.off[1:]], axis=1).astype(np.int64) if self.ids \
            else np.zeros((0, 2), np.int64)

    def columns(self):
        return np.concatenate([np.arange(*self.design.groups[j]) for j in self.ids]) if self.ids \
            else np.zeros(0, np.int64)


def block_norms(m, bounds):
    """(sum d, B) -> (|W|, B) norms of contiguous blocks given (|W|, 2) bounds."""
    if len(bounds) == 0:
        return np.zeros((0, m.shape[1]))
    return np.sqrt(np.add.reduceat(m * m, np.asarray(bounds)[:, 0], axis=0))


def _penalty_sum(coef, bounds, pen_w, lam):
    if len(bounds) == 0:
        return np.zeros(coef.shape[1])
    w = np.where(np.isfinite(pen_w), pen_w, 0.)
    return lam * (w * block_norms(coef, bounds)).sum(axis=0)


def restricted_gap(coef, h, cw, bounds, pen_w, allowed, lam, yy):
    """Relative duality gap of the problem restricted to the allowed groups of W, from
    Gram quantities only (H = X_W^T R / n, C_W = X_W^T Y / n)."""
    btc = np.einsum('ij,ij->j', coef, cw)
    bth = np.einsum('ij,ij->j', coef, h)
    rss = np.maximum(yy - btc - bth, 0.)
    primal = 0.5 * rss + _penalty_sum(coef, bounds, pen_w, lam)
    scale = np.ones(coef.shape[1])
    if len(bounds):
        w = np.where(allowed & np.isfinite(pen_w), pen_w, np.inf)
        scale = np.maximum(scale, (block_norms(h, bounds) / (lam[None, :] * w)).max(axis=0))
    dual = (yy - btc) / scale - rss / (2 * scale ** 2)
    return np.maximum(primal - dual, 0.) / np.maximum(0.5 * yy, 1e-300)


def full_check(design, ws, yb, lam, pen, yy, resid=None):
    """One streamed pass: residual, full gradient, KKT violations and exact gaps."""
    n = design.n
    if resid is None:
        resid = yb - design.x_dot_cols(ws.ids, ws.coef)
    grad = design.xt_dot(resid) / n
    gn = group_norms(grad, design.groups)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.isfinite(pen), gn / (lam[None, :] * pen), 0.)
    scale = np.maximum(1., ratio.max(axis=0))
    rss = np.einsum('ij,ij->j', resid, resid) / n
    yr = np.einsum('ij,ij->j', yb, resid) / n
    bnd = ws.bounds()
    pw = pen[ws.ids] if ws.ids else np.zeros((0, yb.shape[1]))
    primal = 0.5 * rss + _penalty_sum(ws.coef, bnd, pw, lam)
    dual = yr / scale - rss / (2 * scale ** 2)
    gap = np.maximum(primal - dual, 0.) / np.maximum(0.5 * yy, 1e-300)
    active = np.zeros(gn.shape, bool)
    if len(bnd):
        active[ws.ids] = block_norms(ws.coef, bnd) > 0
    viol = (~active) & (ratio > 1. + 1e-10)
    return gn, gap, viol


def _residual_solve(ws, resid, yb, ok, thr, cw, yy, tol, n, max_epochs=20000, k_aa=5,
                    check_every=5):
    """Block-major residual BCD with batched Anderson acceleration (accepted per response
    when it lowers the objective) and restricted-gap stopping. Modifies ws.coef, resid."""
    coef = np.ascontiguousarray(ws.coef)
    bnd = ws.bounds()
    hist = [coef.copy()]
    epochs = 0
    ok0 = ok.copy()

    def primal(c, r):
        pen = (thr * block_norms(c, bnd)).sum(axis=0) if len(bnd) else 0.
        return 0.5 * np.einsum('ij,ij->j', r, r) / n + pen

    while epochs < max_epochs:
        residual_bcd(ws.xwt, resid, coef, bnd, ok, thr, float(n), 1, 0., 10 ** 9)
        epochs += 1
        hist.append(coef.copy())
        if len(hist) == k_aa + 1:
            U = np.diff(np.stack(hist), axis=0)                      # (k, m, B)
            M = np.einsum('kmc,lmc->ckl', U, U)
            tr = np.trace(M, axis1=1, axis2=2)
            M += (1e-10 * np.maximum(tr, 1e-300))[:, None, None] * np.eye(k_aa)[None]
            try:
                z = np.linalg.solve(M, np.ones((M.shape[0], k_aa, 1)))[..., 0]   # (B, k)
                w = z / z.sum(axis=1, keepdims=True)
                xa = np.einsum('ck,kmc->mc', w, np.stack(hist[1:]))
                if np.isfinite(xa).all():
                    ra = yb - ws.xwt.T @ xa
                    better = primal(xa, ra) < primal(coef, resid)
                    if better.any():
                        coef[:, better] = xa[:, better]
                        resid[:, better] = ra[:, better]
            except np.linalg.LinAlgError:
                pass
            hist = [coef.copy()]
        if epochs % check_every == 0:
            h = ws.xwt @ resid / n
            rel = restricted_gap(coef, h, cw, bnd, np.where(ok0, thr, np.inf), ok0,
                                 np.ones(coef.shape[1]), yy)
            done = rel <= tol
            if done.all():
                break
            # converged responses stop updating; the rest keep iterating
            ok = np.ascontiguousarray(ok0 & ~done[None, :])
    ws.coef = coef
    return epochs


def choose_mode(design, mode):
    """Initial update strategy. Disk-resident designs always use covariance updates, because
    residual sweeps would rescan the device; resident designs start with residual updates
    (no setup) and switch to covariance updates when the cost model favors them; 'threshold'
    reproduces the earlier fixed rule (residual when n <= 0.25 D)."""
    if mode in ('gram', 'residual'):
        return mode
    if design.nbytes_resident() == 0:
        return 'gram'
    if mode == 'threshold':
        return 'residual' if design.n <= 0.25 * design.D else 'gram'
    return 'residual'


def _cap(mask, score, active, min_k=10, factor=2.):
    """Prioritized working sets: keep active blocks plus at most max(min_k, factor*|active|)
    further candidates per response, ranked by their dual violation score."""
    out = mask & active
    na = active.sum(axis=0)
    for c in range(mask.shape[1]):
        cand = np.flatnonzero(mask[:, c] & ~active[:, c])
        k = int(max(min_k, factor * na[c]))
        if len(cand) > k:
            cand = cand[np.argsort(-score[cand, c], kind='stable')[:k]]
        out[cand, c] = True
    return out


class _View:
    """Working set restricted to the live (not yet terminated) batch columns."""

    def __init__(self, ws, coef):
        self.ids, self.coef, self.xwt, self.design = ws.ids, coef, ws.xwt, ws.design
        self._ws = ws

    def bounds(self):
        return self._ws.bounds()

    def columns(self):
        return self._ws.columns()


def fit_batch(design, yb, factors, pen, tol=1e-6, val=None, keep_path=False,
              max_rounds=200, strong=True, mode='auto', patience=None, ws_cap=True,
              val_fold=None, val_cluster=None, serve=None, m_max=None, ledger=None, switch_back=False):
    """Group-lasso paths for the batch columns of yb on lam_k = factors[k] * lam_max.

    pen (G, B): penalty weights (np.inf removes a group for that response).
    val: optional (design_val, y_val) used to track validation MSE along the path.
    val_fold, val_cluster: optional cross-fitting folds (0/1) and clusters of the validation
    rows; the selected level is then tracked on each fold and on the whole split.
    patience: with validation data, a response leaves the batch once no selector has
    improved for `patience` consecutive levels (validation-aware termination).
    switch_back: a resident design in covariance mode returns to residual updates when the
    cost model predicts them cheaper for the coming level (amendment 22); False reproduces
    the one-way switch of the frozen engine.
    Returns per-response best coefficients (dense, D x B), entry indices and statistics."""
    t_start = time.perf_counter()
    n, B, G = design.n, yb.shape[1], design.G
    adaptive = mode == 'auto' and design.nbytes_resident() > 0
    mode = choose_mode(design, mode)
    mode_levels, level_epochs = [], 10.
    switches = 0
    max_width = 0
    yb = np.ascontiguousarray(yb, dtype=np.float64)
    yy = np.einsum('ij,ij->j', yb, yb) / n
    read0 = design.read_bytes
    t_pass = time.perf_counter()
    c_full = design.xt_dot(yb) / n
    pass_seconds = time.perf_counter() - t_pass
    passes = 1
    gn = group_norms(c_full, design.groups)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio0 = np.where(np.isfinite(pen), gn / pen, 0.)
    lam_max = np.maximum(ratio0.max(axis=0), 1e-300)
    ws = WorkingSet(design, B, mode, m_max)

    def grow(new, needed, coef):
        """Extend W by `new`; on overflow evict blocks outside `needed`, then retry."""
        try:
            ws.extend(new)
            return coef
        except _Full:
            rows = ws.evict(needed)
            if rows is not None:
                coef = np.ascontiguousarray(coef[rows])
            try:
                ws.extend(new)
                return coef
            except _Full:
                sig = np.zeros((G, B), bool)
                sig[:, L] = allowed
                raise BatchOverflow(k, sig)
    resid = yb.copy() if mode == 'residual' else None
    K = len(factors)
    entry = np.full((G, B), K, dtype=np.int32)
    tracker = ValTracker(val[1].shape[0], B, K, val_fold, val_cluster, serve) if val is not None else None
    best_coef = np.zeros((design.D, B))
    best_norm = np.zeros((3, G, B))
    gaps = np.zeros((K, B))
    live = np.ones(B, bool)
    stopped_at = np.full(B, K, dtype=np.int32)
    path = [] if keep_path else None
    inner_epochs = 0
    prev_lam = lam_max.copy()
    for k, f in enumerate(factors):
        L = np.flatnonzero(live)
        if len(L) == 0:
            break
        lam_all = f * lam_max
        lam, yl, pl, yyl = lam_all[L], yb[:, L], pen[:, L], yy[L]
        coef = np.ascontiguousarray(ws.coef[:, L])
        active = np.zeros((G, len(L)), bool)
        if ws.ids:
            active[ws.ids] = block_norms(coef, ws.bounds()) > 0
        if strong:
            allowed = ((gn[:, L] >= pl * (2 * lam - prev_lam[L])[None, :]) & np.isfinite(pl)) | active
        else:
            allowed = np.isfinite(pl).copy()
        if ws_cap:
            with np.errstate(divide='ignore', invalid='ignore'):
                score = np.where(np.isfinite(pl), gn[:, L] / (pl * lam[None, :]), 0.)
            allowed = _cap(allowed, score, active)
        if ledger is not None and ledger.guard():
            rows = ws.evict(np.flatnonzero(allowed.any(axis=1)))
            if rows is not None:
                coef = np.ascontiguousarray(coef[rows])
            if ledger.guard():
                sig = np.zeros((G, B), bool)
                sig[:, L] = allowed
                raise BatchOverflow(k, sig)
        if adaptive and mode == 'residual':
            new_g = np.flatnonzero(allowed.any(axis=1))
            m_after = ws.width(set(ws.ids) | set(int(j) for j in new_g))
            widths = (allowed * (design.groups[:, 1] - design.groups[:, 0])[:, None]).sum(axis=0)
            if prefer_gram(n, widths, m_after, m_after, level_epochs)[0]:
                ws.to_gram()
                mode, resid = 'gram', None
                switches += 1
        elif adaptive and mode == 'gram' and switch_back:
            new_g = np.flatnonzero(allowed.any(axis=1))
            m_after = ws.width(set(ws.ids) | set(int(j) for j in new_g))
            widths = (allowed * (design.groups[:, 1] - design.groups[:, 0])[:, None]).sum(axis=0)
            if prefer_residual(n, widths, m_after, m_after - ws.m, ws.m, B, level_epochs):
                ws.to_residual()
                mode = 'residual'
                resid = np.ascontiguousarray(yb - design.x_dot_cols(ws.ids, ws.coef))
                switches += 1
        mode_levels.append(mode)
        ep0 = inner_epochs
        coef = grow(np.flatnonzero(allowed.any(axis=1)), np.flatnonzero(allowed.any(axis=1)), coef)
        if coef.shape[0] < ws.m:
            coef = np.vstack([coef, np.zeros((ws.m - coef.shape[0], len(L)))])
        rl = np.ascontiguousarray(resid[:, L]) if mode == 'residual' else None
        inner_tol = tol * 0.1
        need = np.ones(len(L), bool)     # columns with a violation or an open gap
        for _ in range(max_rounds):
            ids = np.asarray(ws.ids, dtype=np.int64)
            pw = pl[ids]
            fin = np.isfinite(pw)
            ok = np.ascontiguousarray(allowed[ids] & fin & (need[None, :] if mode == 'gram' else True))
            thr = np.ascontiguousarray(np.where(fin, lam[None, :] * pw, 0.))
            cw = np.ascontiguousarray(c_full[ws.columns()][:, L])
            if mode == 'gram':
                coef = np.ascontiguousarray(coef)
                ep, _ = solve_batch(ws._cap, cw, coef, ws.bounds(), ok, thr,
                                    np.ascontiguousarray(yyl), inner_tol, 100000, 10)
                inner_epochs += int(ep.max()) if len(ep) else 0
                view = _View(ws, coef)
                t_pass = time.perf_counter()
                gnl, gap, viol = full_check(design, view, yl, lam, pl, yyl)
                pass_seconds += time.perf_counter() - t_pass
            else:
                view = _View(ws, np.ascontiguousarray(coef))
                inner_epochs += _residual_solve(view, rl, yl, ok, thr, cw, yyl, inner_tol, n)
                coef = view.coef
                t_pass = time.perf_counter()
                rl = np.ascontiguousarray(yl - design.x_dot_cols(ws.ids, coef))
                gnl, gap, viol = full_check(design, _View(ws, coef), yl, lam, pl, yyl, rl)
                pass_seconds += time.perf_counter() - t_pass
            passes += 1
            need = viol.any(axis=0) | (gap > tol)
            if not need.any():
                break
            if not viol.any():
                inner_tol *= 0.1
            if ws_cap and viol.any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    vs = np.where(np.isfinite(pl), gnl / (pl * lam[None, :]), 0.)
                act = np.zeros_like(viol)
                act[ws.ids] = block_norms(coef, ws.bounds()) > 0
                viol = _cap(viol, vs, act)
            allowed |= viol
            coef = grow(np.flatnonzero(viol.any(axis=1)), np.flatnonzero(allowed.any(axis=1)), coef)
            if coef.shape[0] < ws.m:
                coef = np.vstack([coef, np.zeros((ws.m - coef.shape[0], len(L)))])
        if ws.coef.shape[0] < ws.m:
            ws.coef = np.vstack([ws.coef, np.zeros((ws.m - ws.coef.shape[0], B))])
        ws.coef[:, L] = coef
        level_epochs = max(inner_epochs - ep0, 1)
        max_width = max(max_width, ws.m)
        if mode == 'residual':
            resid[:, L] = rl
        gn[:, L] = gnl
        gaps[k, L] = gap
        coef_full = np.zeros((design.D, len(L)))
        cols = ws.columns()
        if len(cols):
            coef_full[cols] = coef
        nz = group_norms(coef_full, design.groups) > 0
        el = entry[:, L]
        el[(el == K) & nz] = k
        entry[:, L] = el
        if val is not None:
            dv, yv = val
            pred = dv.x_dot_cols(ws.ids, coef)
            moved = tracker.update(k, L, yv[:, L] - pred)
            gnorm = group_norms(coef_full, design.groups)
            for S, better in moved.items():
                best_norm[S][:, L[better]] = gnorm[:, better]
                if S == 2:
                    best_coef[:, L[better]] = coef_full[:, better]
            if patience is not None:
                done = tracker.exhausted(k, L, patience)
                live[L[done]] = False
                stopped_at[L[done]] = k
        if keep_path:
            full = np.zeros((design.D, B))
            full[:, L] = coef_full
            path.append(full)
        prev_lam = lam_all
    vs = tracker.summary() if tracker is not None else None
    return {'best_coef': best_coef, 'best_k': vs['best_k'][2] if vs else None,
            'val_mse': vs['val_mse'][2] if vs else None, 'val': vs, 'best_norm': best_norm, 'entry': entry,
            'lam_max': lam_max, 'gaps': gaps, 'c_full': c_full, 'path': path, 'mode': mode,
            'working_set': list(ws.ids), 'passes': passes, 'inner_epochs': inner_epochs,
            'read_bytes': design.read_bytes - read0, 'gram_seconds': ws.grow_seconds, 'evicted': ws.evicted, 'mode_levels': mode_levels, 'switches': switches, 'max_width': max_width,
            'stopped_at': stopped_at, 'pass_seconds': pass_seconds, 'seconds': time.perf_counter() - t_start}
