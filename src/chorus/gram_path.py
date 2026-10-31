"""Working-set group-lasso paths solved entirely from sufficient statistics.

Given G_UU = X_U^T X_U / n, C_U = X_U^T Y / n and yy = ||y||^2 / n on a group set U, and
penalty weights that are infinite outside U, the full problem equals the problem on U, so
the duality gap computed here is an exact certificate and no pass over the data is needed.
Used by the reweighting stage of Chorus, and by stage 1 under the statistics strategy
(U = all groups). Each response works on its own strong set inside the shared Gram matrix;
gradients are refreshed from the nonzero coefficients only.
"""
import numpy as np
from .kernels import solve_batch
from .validation import ValTracker
from .ledger import BatchOverflow, WidthOverflow

CHUNK_BYTES = 32 << 20     # bound on the transients of gradient and validation products


def _cap(mask, score, active, min_k=10, factor=2.):
    """Prioritized candidate sets, as in the engine: active blocks plus at most
    max(min_k, factor * |active|) further blocks per response, ranked by dual violation."""
    out = mask & active
    na = active.sum(axis=0)
    for c in range(mask.shape[1]):
        cand = np.flatnonzero(mask[:, c] & ~active[:, c])
        k = int(max(min_k, factor * na[c]))
        if len(cand) > k:
            cand = cand[np.argsort(-score[cand, c], kind='stable')[:k]]
        out[cand, c] = True
    return out


def _h_update(gram, cu, coef, nzr):
    """h = cu - G[:, nzr] coef[nzr], gathering rows of the symmetric G in bounded chunks."""
    h = cu.copy()
    if len(nzr) == 0:
        return h
    step = max(1, CHUNK_BYTES // (8 * gram.shape[0]))
    for i in range(0, len(nzr), step):
        idx = nzr[i:i + step]
        h -= gram[idx].T @ coef[idx]
    return h


def _predict(xv, coef, nzr):
    """xv[:, nzr] coef[nzr] in bounded column chunks."""
    out = np.zeros((xv.shape[0], coef.shape[1]))
    step = max(1, CHUNK_BYTES // (8 * max(xv.shape[0], 1)))
    for i in range(0, len(nzr), step):
        idx = nzr[i:i + step]
        out += xv[:, idx] @ coef[idx]
    return out


def _gap_contig(coef, h, cu, starts, pen, finite, lam, yy):
    """Exact relative duality gap over all finite-weight groups of U (contiguous layout)."""
    btc = np.einsum('ij,ij->j', coef, cu)
    bth = np.einsum('ij,ij->j', coef, h)
    rss = np.maximum(yy - btc - bth, 0.)
    bn = np.sqrt(np.add.reduceat(coef * coef, starts, axis=0))
    hn = np.sqrt(np.add.reduceat(h * h, starts, axis=0))
    pw = np.where(finite, pen, 0.)
    primal = 0.5 * rss + lam * (pw * bn).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(finite, hn / (lam[None, :] * pen), 0.)
    scale = np.maximum(1., ratio.max(axis=0))
    dual = (yy - btc) / scale - rss / (2 * scale ** 2)
    return np.maximum(primal - dual, 0.) / np.maximum(0.5 * yy, 1e-300), hn, bn


def fit_batch_gram(gram, cu, yy, sizes, factors, pen, tol=1e-6, val=None, max_rounds=50,
                   patience=None, val_fold=None, val_cluster=None, serve=None,
                   track_norms=False, ws_cap=False, tighten=False, chunked=False,
                   ledger=None, m_sub=None, keep_path=False):
    """gram (m, m), cu (m, B), sizes: group sizes of U in order, pen (|U|, B).
    val: (xv_U (n_val, m), y_val (n_val, B)); val_fold and val_cluster as in fit_batch.
    Stage-1 options (all off for stage 2, whose results they leave unchanged):
    track_norms returns the group norms of the fit selected by each selector; ws_cap and
    tighten follow the engine (prioritized candidate sets, a tighter inner tolerance when a
    gap stays open without violations); chunked bounds the gradient and validation transients;
    ledger checks the resident set before every level (BatchOverflow) and m_sub bounds the
    candidate width of one response (WidthOverflow); keep_path returns the coefficients of every
    visited level (columns retired by the exit keep their last fit).
    Returns dense best coefficients on U (whole-split selector) and the validation summary."""
    sizes = np.asarray(sizes, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    bounds = np.ascontiguousarray(np.stack([starts, starts + sizes], axis=1))
    nU, B = len(sizes), cu.shape[1]
    gram = np.ascontiguousarray(gram)
    cu = np.ascontiguousarray(cu)
    yy = np.ascontiguousarray(yy)
    finite = np.isfinite(pen)
    gn = np.sqrt(np.add.reduceat(cu * cu, starts, axis=0)) if nU else np.zeros((0, B))
    with np.errstate(divide='ignore', invalid='ignore'):
        lam_max = np.maximum(np.where(finite, gn / pen, 0.).max(axis=0) if nU else np.zeros(B), 1e-300)
    coef = np.zeros((cu.shape[0], B))
    K = len(factors)
    entry = np.full((nU, B), K, dtype=np.int32)
    best_coef = np.zeros_like(coef)
    best_norm = np.zeros((3, nU, B)) if track_norms else None
    tracker = ValTracker(val[1].shape[0], B, K, val_fold, val_cluster, serve) if val is not None else None
    gaps = np.zeros((K, B))
    stopped_at = np.full(B, K, dtype=np.int32)
    path = [] if keep_path else None
    prev = lam_max.copy()
    epochs = 0
    max_cand = 0
    live = np.ones(B, bool)
    hn = gn.copy()
    for k, f in enumerate(factors):
        if not live.any():
            break
        lam = f * lam_max
        bn = np.sqrt(np.add.reduceat(coef * coef, starts, axis=0)) if nU else np.zeros((0, B))
        allowed = ((hn >= pen * (2 * lam - prev)[None, :]) & finite) | (bn > 0)
        if ws_cap:
            with np.errstate(divide='ignore', invalid='ignore'):
                score = np.where(finite, hn / (pen * lam[None, :]), 0.)
            allowed = _cap(allowed, score, bn > 0)
        allowed[:, ~live] = False
        if ledger is not None and ledger.guard():
            raise BatchOverflow(k, allowed.copy())
        thr = np.ascontiguousarray(np.where(finite, lam[None, :] * pen, 0.))
        gap = np.zeros(B)
        need = live.copy()            # columns with a violation or an open gap
        inner_tol = tol * 0.1
        for _ in range(max_rounds):
            ok = np.ascontiguousarray(allowed & finite & need[None, :])
            if m_sub is not None:
                width = int((ok * sizes[:, None]).sum(axis=0).max()) if nU else 0
                max_cand = max(max_cand, width)
                if width > m_sub:
                    raise WidthOverflow(k, width)
            ep, _ = solve_batch(gram, cu, coef, bounds, ok, thr, yy, inner_tol, 100000, 10)
            epochs += int(ep.max()) if len(ep) else 0
            nzr = np.flatnonzero(np.abs(coef).max(axis=1) > 0)
            if chunked:
                h_full = _h_update(gram, cu, coef, nzr)
            else:
                h_full = cu - gram[:, nzr] @ coef[nzr] if len(nzr) else cu.copy()
            gap, hn, bn = _gap_contig(coef, h_full, cu, starts, pen, finite, lam, yy)
            with np.errstate(divide='ignore', invalid='ignore'):
                viol = (bn == 0) & finite & (hn > lam[None, :] * pen * (1 + 1e-10))
            viol[:, ~live] = False
            need = live & (viol.any(axis=0) | (gap > tol))
            if not need.any():
                break
            if tighten and not viol.any():
                inner_tol *= 0.1
            if ws_cap and viol.any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    vs_ = np.where(finite, hn / (pen * lam[None, :]), 0.)
                viol = _cap(viol, vs_, bn > 0)
            allowed |= viol
        gaps[k, live] = gap[live]     # retired columns keep their last certified level
        nz = bn > 0
        entry[(entry == K) & nz & live[None, :]] = k
        if val is not None:
            xv, yv = val
            L = np.flatnonzero(live)
            nzr = np.flatnonzero(np.abs(coef[:, L]).max(axis=1) > 0) if len(L) else np.zeros(0, int)
            if chunked:
                pred = _predict(xv, coef[:, L], nzr)
            else:
                pred = xv[:, nzr] @ coef[np.ix_(nzr, L)] if len(nzr) else np.zeros((yv.shape[0], len(L)))
            moved = tracker.update(k, L, yv[:, L] - pred)
            if track_norms:
                for S, better in moved.items():
                    best_norm[S][:, L[better]] = bn[:, L[better]]
            b2 = L[moved[2]]
            best_coef[:, b2] = coef[:, b2]
            if patience is not None:
                done = tracker.exhausted(k, L, patience)
                live[L[done]] = False
                stopped_at[L[done]] = k
        if keep_path:
            path.append(coef.copy())
        prev = lam
    vs = tracker.summary() if tracker is not None else None
    return {'best_coef': best_coef, 'best_k': vs['best_k'][2] if vs else None,
            'val_mse': vs['val_mse'][2] if vs else None, 'val': vs, 'entry': entry,
            'best_norm': best_norm, 'stopped_at': stopped_at, 'max_cand_width': max_cand, 'path': path,
            'lam_max': lam_max, 'gaps': gaps, 'epochs': epochs}
