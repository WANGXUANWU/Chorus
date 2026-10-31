"""Pooled relaxed refit of selected supports (protocol amendment 17).

For a response with selected columns S and penalized coefficients b_p, the least-squares
coefficients on S are b_l = (G_SS + eps tr(G_SS)/|S| I)^(-1) X_S^T y / n. They come from the
Gram matrix and the correlations that stage 2 already holds, so the refit reads no training
row. The relaxed fit is b(phi) = phi b_l + (1 - phi) b_p. One weight phi in PHI is chosen per
candidate for all responses together, by the validation error summed over responses in units
of each response's validation variance, so responses pool their evidence as in the
reweighting. phi = 0 keeps the penalized fit. The support never changes.
"""
import numpy as np

PHI = (0., .25, .5, .75, 1.)


def relax_block(gram, xy, xv, y_val, coef, starts, sizes, eps=1e-6, phis=PHI, per_response=False):
    """Relax every response of one candidate. gram, xy (|cols| x q), xv (n_val x |cols|) and
    coef (|cols| x q) are indexed by the columns of a group set with the given starts and sizes.
    Returns phi (q,) and the relaxed coefficients of the responses with phi > 0 as
    {r: (positions, values)}. per_response=True lets every response choose its own phi by its
    own validation error (a tuning variant of amendment 22)."""
    q = coef.shape[1]
    loss = np.zeros(len(phis))
    own = {}
    fits = {}
    for r in range(q):
        nz = [t for t in range(len(starts)) if np.any(coef[starts[t]:starts[t] + sizes[t], r])]
        if not nz:
            continue
        pos = np.concatenate([np.arange(starts[t], starts[t] + sizes[t]) for t in nz])
        G = gram[np.ix_(pos, pos)]
        k = len(pos)
        try:
            bl = np.linalg.solve(G + eps * max(np.trace(G) / k, 1e-300) * np.eye(k), xy[pos, r])
        except np.linalg.LinAlgError:
            continue
        bp = coef[pos, r]
        pp, pl = xv[:, pos] @ bp, xv[:, pos] @ bl
        v = max(y_val[:, r].var(), 1e-300)
        lr = np.array([np.mean((y_val[:, r] - (ph * pl + (1 - ph) * pp)) ** 2) / v for ph in phis])
        loss += lr
        own[r] = phis[int(np.argmin(lr))]
        fits[r] = (pos, bl, bp)
    phi = np.zeros(q)
    if not fits:
        return phi, {}
    ph = phis[int(np.argmin(loss))]
    out = {}
    for r, (pos, bl, bp) in fits.items():
        pr = own[r] if per_response else ph
        if pr == 0:
            continue
        phi[r] = pr
        out[r] = (pos, pr * bl + (1 - pr) * bp)
    return phi, out
