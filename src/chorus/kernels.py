"""Compiled per-response covariance-update kernels on a shared Gram matrix.

The batch shares the Gram matrix of the union working set W and all data passes; each
response then runs block coordinate descent (BCD) on its own candidate blocks with a
contiguous copy of its sub-Gram, plus Anderson acceleration of the iterates (accepted only
when it lowers the objective), so the inner cost matches a single-response solver.
"""
import numpy as np
from numba import njit

K_AA = 5


@njit(cache=True)
def _primal(gs, c, x, starts, tb, yy):
    tot = len(x)
    gx = np.dot(gs, x)
    q = 0.
    lin = 0.
    for i in range(tot):
        q += x[i] * gx[i]
        lin += x[i] * c[i]
    pen = 0.
    for t in range(len(tb)):
        s = 0.
        for k in range(starts[t], starts[t + 1]):
            s += x[k] * x[k]
        pen += tb[t] * np.sqrt(s)
    return 0.5 * (yy - 2. * lin + q) + pen


@njit(cache=True)
def _gap(gs, c, x, h, starts, tb, yy):
    tot = len(x)
    btc = 0.
    bth = 0.
    for i in range(tot):
        btc += x[i] * c[i]
        bth += x[i] * h[i]
    rss = max(yy - btc - bth, 0.)
    pen = 0.
    scale = 1.
    for t in range(len(tb)):
        nb2 = 0.
        nh2 = 0.
        for k in range(starts[t], starts[t + 1]):
            nb2 += x[k] * x[k]
            nh2 += h[k] * h[k]
        pen += tb[t] * np.sqrt(nb2)
        if tb[t] > 0:
            scale = max(scale, np.sqrt(nh2) / tb[t])
    primal = 0.5 * rss + pen
    dual = (yy - btc) / scale - rss / (2. * scale * scale)
    return max(primal - dual, 0.) / max(0.5 * yy, 1e-300)


@njit(cache=True)
def _solve_one(gram, cw, coef, bounds, blocks, thr, yy, tol, max_epochs, check_every):
    """Solve one response restricted to `blocks` (local block ids into bounds).
    gram (m, m), cw (m,), coef (m,) modified in place, thr (|W|,) = lam * w (finite).
    Returns (epochs, relative restricted gap)."""
    nb = len(blocks)
    starts = np.empty(nb + 1, np.int64)
    starts[0] = 0
    for t in range(nb):
        starts[t + 1] = starts[t] + bounds[blocks[t], 1] - bounds[blocks[t], 0]
    tot = starts[nb]
    rows = np.empty(tot, np.int64)
    tb = np.empty(nb)
    for t in range(nb):
        tb[t] = thr[blocks[t]]
        for k in range(starts[t + 1] - starts[t]):
            rows[starts[t] + k] = bounds[blocks[t], 0] + k
    if tot == 0:
        return 0, 0.
    gs = np.empty((tot, tot))
    for i in range(tot):
        for k in range(tot):
            gs[i, k] = gram[rows[i], rows[k]]
    c = np.empty(tot)
    x = np.empty(tot)
    for i in range(tot):
        c[i] = cw[rows[i]]
        x[i] = coef[rows[i]]
    h = c - np.dot(gs, x)
    hist = np.empty((K_AA + 1, tot))
    nhist = 0
    rel = np.inf
    epochs = 0
    z = np.empty(16)
    full = True              # alternate full sweeps with sweeps over nonzero blocks only
    since_full = 0
    for epoch in range(1, max_epochs + 1):
        epochs = epoch
        maxd = 0.
        maxx = 0.
        did_full = full
        for t in range(nb):
            a = starts[t]
            d = starts[t + 1] - a
            if not full:
                nz = False
                for k in range(d):
                    if x[a + k] != 0.:
                        nz = True
                        break
                if not nz:
                    continue
            nrm = 0.
            for k in range(d):
                z[k] = x[a + k] + h[a + k]
                nrm += z[k] * z[k]
            nrm = np.sqrt(nrm)
            shrink = 0.
            if nrm > tb[t]:
                shrink = 1. - tb[t] / nrm
            for k in range(d):
                dk = z[k] * shrink - x[a + k]
                if dk != 0.:
                    x[a + k] += dk
                    if abs(dk) > maxd:
                        maxd = abs(dk)
                    row = gs[a + k]
                    for i in range(tot):
                        h[i] -= dk * row[i]
                if abs(x[a + k]) > maxx:
                    maxx = abs(x[a + k])
        if full:
            full = False
            since_full = 0
        else:
            since_full += 1
            if maxd <= 1e-7 * max(maxx, 1e-12) or since_full >= 25:
                full = True
        for i in range(tot):
            hist[nhist, i] = x[i]
        nhist += 1
        if nhist == K_AA + 1:
            nhist = 0
            M = np.empty((K_AA, K_AA))
            for u in range(K_AA):
                for v in range(u, K_AA):
                    acc = 0.
                    for i in range(tot):
                        acc += (hist[u + 1, i] - hist[u, i]) * (hist[v + 1, i] - hist[v, i])
                    M[u, v] = acc
                    M[v, u] = acc
            tr = 0.
            for u in range(K_AA):
                tr += M[u, u]
            if tr > 0.:
                for u in range(K_AA):
                    M[u, u] += 1e-10 * tr
                zz = np.linalg.solve(M, np.ones(K_AA))
                sz = zz.sum()
                if sz != 0. and np.isfinite(sz):
                    xa = np.zeros(tot)
                    for u in range(K_AA):
                        w = zz[u] / sz
                        for i in range(tot):
                            xa[i] += w * hist[u + 1, i]
                    if _primal(gs, c, xa, starts, tb, yy) < _primal(gs, c, x, starts, tb, yy):
                        x[:] = xa
                        h = c - np.dot(gs, x)
        if did_full or epoch == max_epochs:
            h = c - np.dot(gs, x)
            rel = _gap(gs, c, x, h, starts, tb, yy)
            if rel <= tol:
                break
    for i in range(tot):
        coef[rows[i]] = x[i]
    return epochs, rel


@njit(cache=True)
def solve_batch(gram, cw, coef, bounds, allowed, thr, yy, tol, max_epochs, check_every):
    """Loop _solve_one over batch columns. allowed (|W|, B) bool, thr (|W|, B)."""
    B = cw.shape[1]
    nW = bounds.shape[0]
    epochs = np.zeros(B, np.int64)
    gaps = np.zeros(B)
    for c in range(B):
        cnt = 0
        for t in range(nW):
            if allowed[t, c]:
                cnt += 1
        blocks = np.empty(cnt, np.int64)
        p = 0
        for t in range(nW):
            if allowed[t, c]:
                blocks[p] = t
                p += 1
        col = np.ascontiguousarray(coef[:, c])
        e, g = _solve_one(gram, np.ascontiguousarray(cw[:, c]), col, bounds, blocks,
                          np.ascontiguousarray(thr[:, c]), yy[c], tol, max_epochs, check_every)
        coef[:, c] = col
        epochs[c] = e
        gaps[c] = g
    return epochs, gaps


@njit(cache=True)
def residual_bcd(xwt, resid, coef, bounds, allowed, thr, n, max_epochs, tol, check_every):
    """Block-major BCD over the batch with explicit residuals (for n small relative to |W|).

    xwt (m, n) C-order: transposed working-set columns; resid (n, B) C-order, modified;
    coef (m, B) modified; allowed (|W|, B); thr (|W|, B) = lam * w. Every visit of a block
    updates all batch columns with one block-times-residual product (shared block read).
    Stops when the largest relative coefficient change falls below tol, checked every
    `check_every` epochs. Returns epochs used."""
    m, nn = xwt.shape
    B = resid.shape[1]
    nW = bounds.shape[0]
    g = np.empty((16, B))
    delta = np.empty((16, B))
    epochs = 0
    for epoch in range(1, max_epochs + 1):
        epochs = epoch
        maxchg = 0.
        maxc = 0.
        for t in range(nW):
            any_ok = False
            for c in range(B):
                if allowed[t, c]:
                    any_ok = True
                    break
            if not any_ok:
                continue
            a = bounds[t, 0]
            d = bounds[t, 1] - a
            for k in range(d):
                for c in range(B):
                    g[k, c] = 0.
            cols = np.empty(B, np.int64)
            ncols = 0
            for c in range(B):
                if allowed[t, c]:
                    cols[ncols] = c
                    ncols += 1
            for k in range(d):
                row = xwt[a + k]
                for i in range(nn):
                    xv = row[i]
                    for cc in range(ncols):
                        c = cols[cc]
                        g[k, c] += xv * resid[i, c]
            moved = False
            for c in range(B):
                if not allowed[t, c]:
                    for k in range(d):
                        delta[k, c] = 0.
                    continue
                nrm = 0.
                for k in range(d):
                    zk = coef[a + k, c] + g[k, c] / n
                    g[k, c] = zk
                    nrm += zk * zk
                nrm = np.sqrt(nrm)
                shrink = 0.
                if nrm > thr[t, c]:
                    shrink = 1. - thr[t, c] / nrm
                for k in range(d):
                    new = g[k, c] * shrink
                    dk = new - coef[a + k, c]
                    delta[k, c] = dk
                    if dk != 0.:
                        moved = True
                        coef[a + k, c] = new
                        if abs(dk) > maxchg:
                            maxchg = abs(dk)
                    if abs(new) > maxc:
                        maxc = abs(new)
            if moved:
                for k in range(d):
                    row = xwt[a + k]
                    for i in range(nn):
                        xv = row[i]
                        for cc in range(ncols):
                            c = cols[cc]
                            resid[i, c] -= xv * delta[k, c]
        if epoch % check_every == 0 and maxchg <= tol * max(maxc, 1e-12):
            break
    return epochs


@njit(cache=True)
def compact_gram(cap, keep):
    """In-place compaction of a Gram buffer to the rows and columns in `keep` (increasing).
    Every destination index is at most its source index, so row-major traversal never
    overwrites an entry before it is read."""
    m = len(keep)
    for i in range(m):
        ri = keep[i]
        for j in range(m):
            cap[i, j] = cap[ri, keep[j]]


@njit(cache=True)
def symmetrize_lower(g):
    """Copy the lower triangle of a square matrix to its upper triangle, in place."""
    m = g.shape[0]
    for j in range(m):
        for i in range(j + 1, m):
            g[j, i] = g[i, j]
