"""Cost model for the choice between covariance (Gram) and residual updates.

Two machine constants are measured once per process by a short benchmark: the time per
floating-point operation of a dense matrix product (Gram construction) and of the compiled
coordinate kernels (covariance epochs and residual epochs). For a level with live widths
m_r (columns of each response's candidate set), union width m and E expected epochs:
    residual epoch cost   T_R = t_res * 4 n sum_r m_r          (block product and update)
    covariance cost       T_G = t_gemm * 2 n m_new (m + m_new) + t_cd * 2 E sum_r m_r^2
The engine starts a resident design with residual updates (no setup) and switches to the
Gram matrix when T_G, including the conversion, is below T_R by a 25% margin; with
switch_back it returns to residual updates when T_R, including its conversion, is below T_G
by the same margin, which happens near the dense end of paths on short and wide designs.
"""
import time
import numpy as np

_CONST = {}


def constants():
    if _CONST:
        return _CONST
    from .kernels import residual_bcd, solve_batch
    rng = np.random.default_rng(0)
    # dense products (Gram construction)
    a = rng.standard_normal((4000, 256))
    t = time.perf_counter()
    for _ in range(3):
        a.T @ a
    _CONST['gemm'] = (time.perf_counter() - t) / (3 * 2 * 4000 * 256 * 256)
    # residual epochs
    n, m, B = 4000, 100, 8
    xwt = np.ascontiguousarray(rng.standard_normal((m, n)) / np.sqrt(n))
    bounds = np.stack([np.arange(0, m, 5), np.arange(5, m + 1, 5)], axis=1).astype(np.int64)
    resid = np.ascontiguousarray(rng.standard_normal((n, B)))
    coef = np.zeros((m, B))
    ok = np.ones((len(bounds), B), bool)
    thr = np.full((len(bounds), B), 1e-3)
    residual_bcd(xwt, resid.copy(), coef.copy(), bounds, ok, thr, float(n), 1, 0., 10 ** 9)
    t = time.perf_counter()
    residual_bcd(xwt, resid, coef, bounds, ok, thr, float(n), 5, 0., 10 ** 9)
    _CONST['res'] = (time.perf_counter() - t) / (5 * 4 * n * m * B)
    # covariance epochs
    g = xwt @ xwt.T
    cw = np.ascontiguousarray(rng.standard_normal((m, B)))
    yy = np.full(B, 10.)
    solve_batch(g, cw, np.zeros((m, B)), bounds, ok, thr, yy, 0., 3, 10)
    t = time.perf_counter()
    solve_batch(g, cw, np.zeros((m, B)), bounds, ok, thr, yy, 0., 20, 10)
    _CONST['cd'] = (time.perf_counter() - t) / (20 * 2 * m * m * B)
    # products X^T R with a batch of right-hand sides (passes over the design)
    r = rng.standard_normal((4000, 16))
    a.T @ r
    t = time.perf_counter()
    for _ in range(5):
        a.T @ r
    _CONST['xtr'] = (time.perf_counter() - t) / (5 * 2 * 4000 * 256 * 16)
    return _CONST


def predict_strategy(n, D, q, K, B_ws, disk, bw=None, m1=None, m2=None, epochs=10., resid_frac=0.5):
    """Predicted stage-1 times of the statistics strategy and of the working-set engine.

    Statistics: one pass forming X^T X (2 n D^2 flops, overlapped with the read on disk) and
    X^T Y, then covariance solves. Working sets: ceil(q / B) (1 + K) passes, each a product
    X^T R with B right-hand sides plus residual panel reads, the growth of the union Gram
    matrix up to D columns, and the cheaper of covariance and residual solves (covariance
    only on disk). m1, m2: mean and mean-square candidate widths per response and level."""
    c = constants()
    passes = -(-q // max(B_ws, 1)) * (1 + K)
    m1 = float(D if m1 is None else m1)
    m2 = float(D * D if m2 is None else m2)
    s_cov = c['cd'] * 2 * epochs * K * q * m2
    s_res = c['res'] * 4 * n * epochs * K * q * m1
    gram = c['gemm'] * 2 * n * D * D
    xty = c['xtr'] * 2 * n * D * q
    if disk:
        bw = float(bw or 2.5e8)
        t_stats = max(8. * n * D / bw, gram) + xty + s_cov
        per_pass = max(8. * n * D * (1 + resid_frac) / bw, c['xtr'] * 2 * n * D * B_ws)
        t_ws = passes * per_pass + gram + 8. * n * D / bw + s_cov
    else:
        t_stats = gram + xty + s_cov
        per_pass = c['xtr'] * 2 * n * D * B_ws * (1 + resid_frac)
        t_ws = passes * per_pass + min(gram + s_cov, s_res)
    return t_stats, t_ws


def prefer_gram(n, widths, m_union, m_new, epochs, margin=1.25):
    """True when covariance updates are predicted to be cheaper for the coming level.
    widths: candidate widths of the live responses; m_new: width added to the Gram matrix
    (the whole union when converting from residual updates)."""
    c = constants()
    w = np.asarray(widths, dtype=float)
    E = max(float(epochs), 5.)
    t_res = c['res'] * 4 * n * w.sum() * E
    t_gram = c['gemm'] * 2 * n * m_new * (m_union + m_new) + c['cd'] * 2 * E * (w * w).sum()
    return t_gram * margin < t_res, t_gram, t_res


def prefer_residual(n, widths, m_union, m_new, m_held, B, epochs, margin=1.25):
    """True when a batch in covariance mode should return to residual updates for the coming
    level (resident designs, amendment 22): the covariance cost of the level (growth of the
    held Gram matrix by m_new columns and the epochs) exceeds, by the margin, the residual
    epochs plus the conversion (transposed columns of the union and one residual product)."""
    c = constants()
    w = np.asarray(widths, dtype=float)
    E = max(float(epochs), 5.)
    t_gram = c['gemm'] * 2 * n * m_new * (m_held + m_new) + c['cd'] * 2 * E * (w * w).sum()
    t_res = c['res'] * 4 * n * w.sum() * E + c['xtr'] * 2 * n * m_union * B
    return t_res * margin < t_gram
