"""Cross-fitted empirical-Bayes selection among candidate fits of many responses.

Input, for C candidates and q responses: the cross-fitted per-cluster squared-error sums
of every candidate (each cluster scored by the fit selected on the other validation fold),
the number of selected groups and parameters of every candidate's final fit, and the
validation variance of every response.

For candidate c and response r the paired difference to a reference candidate is
d_rc = sum_b D_rcb with D_rcb the normalized per-cluster loss difference; its sampling
variance s2_rc is estimated robustly over clusters. Across responses the differences of a
candidate follow a normal random-effects model d_rc ~ N(theta_rc, s2_rc),
theta_rc ~ N(mu_c, tau2_c), fitted by the DerSimonian-Laird moment estimator. Each response
then receives the posterior mean of theta_rc, which shrinks its own noisy difference toward
the cross-response mean in proportion to its sampling variance. Among the candidates whose
posterior mean is within kappa posterior standard errors of the best one, the sparsest is
chosen. With tau2 = 0 every response shrinks to the shared mean (one shared choice), and
with tau2 large every response keeps its own difference (per-response choice).
"""
import numpy as np


def _dl(d, s2):
    """DerSimonian-Laird mean and between-response variance (d, s2 of length q)."""
    w = 1. / s2
    mu = (w * d).sum() / w.sum()
    Q = (w * (d - mu) ** 2).sum()
    den = w.sum() - (w * w).sum() / w.sum()
    tau2 = max(0., (Q - (len(d) - 1)) / den) if den > 0 else 0.
    return mu, tau2


def eb_select(closs, groups, params, yvar, n_val, ref, kappa=1.0, parsimony='groups',
              shrink='eb', order=None):
    """closs (C, Nb, q) cross-fitted per-cluster losses; groups, params (C, q);
    yvar (q,); ref (q,) index of the reference candidate of each response.
    shrink: 'eb' (random effects), 'none' (own difference only) or 'pool' (shared mean).
    Returns pick (q,) and a dict of diagnostics."""
    C, Nb, q = closs.shape
    order = np.arange(C) if order is None else np.asarray(order)
    norm = n_val * np.maximum(yvar, 1e-300)
    refl = closs[ref, :, np.arange(q)].T                     # (Nb, q)
    D = (closs - refl[None]) / norm[None, None, :]           # (C, Nb, q)
    d = D.sum(axis=1)                                        # (C, q)
    Dm = D - D.mean(axis=1, keepdims=True)
    s2 = Nb / max(Nb - 1, 1) * (Dm * Dm).sum(axis=1)         # (C, q)
    pos = s2[s2 > 0]
    floor = 1e-12 * (np.median(pos) if len(pos) else 1.)
    theta = np.zeros((C, q))
    var = np.zeros((C, q))
    mu_c, tau2_c = np.zeros(C), np.zeros(C)
    for c in range(C):
        same = s2[c] <= floor                                # identical to the reference
        if same.all():
            continue
        ok = ~same
        s2c = np.maximum(s2[c], floor)
        if shrink == 'none':
            theta[c], var[c] = d[c], s2c
        else:
            mu, tau2 = _dl(d[c, ok], s2c[ok])
            if shrink == 'pool':
                tau2 = 0.
            om = np.where(ok, 1. / (s2c + tau2), 0.)
            om /= om.sum()
            mu = (om * d[c]).sum()
            M = (D[c] * om[None, :]).sum(axis=1)             # (Nb,) cluster contributions to mu
            vmu = Nb / max(Nb - 1, 1) * ((M - M.mean()) ** 2).sum()
            Bf = s2c / (s2c + tau2) if tau2 > 0 else np.ones(q)
            theta[c] = mu + (1 - Bf) * (d[c] - mu)
            var[c] = (1 - Bf) * s2c + Bf ** 2 * vmu
            mu_c[c], tau2_c[c] = mu, tau2
        theta[c, same] = 0.
        var[c, same] = 0.
    theta[ref, np.arange(q)] = 0.
    var[ref, np.arange(q)] = 0.
    pick = np.empty(q, dtype=np.int64)
    for r in range(q):
        best = int(np.argmin(theta[:, r]))
        ok = theta[:, r] <= theta[best, r] + kappa * np.sqrt(var[:, r] + var[best, r]) + 1e-15
        cand = np.flatnonzero(ok)
        if parsimony == 'groups':
            key = (order[cand], theta[cand, r], params[cand, r], groups[cand, r])
        else:
            key = (order[cand], theta[cand, r], groups[cand, r], params[cand, r])
        pick[r] = cand[np.lexsort(key)[0]]
    return pick, {'theta': theta, 'var': var, 'd': d, 's2': s2, 'mu': mu_c, 'tau2': tau2_c}
