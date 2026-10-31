"""Chorus estimator: batched group-lasso paths, cross-response reweighting, cross-fitted
empirical-Bayes selection.

Stage 1 fits the sparse additive group lasso (GL) path of every response, either from the
full sufficient statistics X^T X and X^T Y formed in one pass (statistics strategy, when the
Gram matrix fits the memory budget and the cost model predicts it to be cheaper) or with the
working-set engine in batches that share each pass over the design. Stage 2 refits with weights
    w_jr = sqrt(d_j) * ((1 - alpha) a_jr + alpha * mean_r a_jr) ** (-gamma),
where a_jr = ||f_jr|| / sd(y_r) is the stage-1 component norm in units of the response's
standard deviation, so the weights do not depend on the scale of any response. Stage 2
uses sufficient statistics only. Every candidate (GL and each (alpha, gamma), for each
supplied basis) is scored by cross-fitting over two validation folds, and a response's
candidate is chosen by the empirical-Bayes rule of selection.py. Every candidate is then
relaxed on its own support by the pooled refit of relax.py, which reuses the stage-2 Gram
matrix and keeps the support.
"""
import time
import numpy as np
from .engine import fit_batch, group_norms
from .gram_path import fit_batch_gram
from .selection import eb_select
from .validation import make_folds
from .relax import relax_block
from .ledger import Ledger, BatchOverflow, MemoryBudgetError, WidthOverflow
from .costmodel import predict_strategy


def _split(cols, signature):
    """Split a batch into two halves along the leading principal direction of the
    candidate-block pattern of its responses, so responses with similar supports stay together."""
    S = signature[:, :len(cols)].T.astype(float)
    S -= S.mean(axis=0, keepdims=True)
    if len(cols) > 2 and np.any(S):
        u = np.linalg.svd(S, full_matrices=False)[0][:, 0]
        order = np.argsort(u, kind='stable')
    else:
        order = np.arange(len(cols))
    h = len(cols) // 2
    return [cols[np.sort(order[:h])], cols[np.sort(order[h:])]]


def _run_batches(fn, q, B, ledger):
    """Apply fn(cols) to batches of B columns; on a working-set overflow split the batch and
    rerun both halves from the first level (exact, at the cost of recomputed levels)."""
    queue = [np.arange(b0, min(q, b0 + B)) for b0 in range(0, q, B)]
    while queue:
        cols = queue.pop(0)
        try:
            fn(cols)
        except BatchOverflow as e:
            if len(cols) == 1:
                raise MemoryBudgetError('the working set of a single response exceeds the budget')
            if ledger is not None:
                ledger.events['splits'] += 1
            queue[:0] = _split(cols, e.signature)


def lambda_factors(n_lambda=30, ratio=0.02):
    return np.geomspace(1.0, ratio, n_lambda)


def plan_batch(n, D, q, budget_bytes=None, resident_bytes=0, overhead=200 << 20,
               chunk_bytes=64 << 20, max_batch=64):
    """Largest batch B whose per-batch arrays fit the numerical budget (legacy planner;
    the memory ledger replaces it when a budget is enforced)."""
    if budget_bytes is None:
        return min(q, max_batch)
    free = budget_bytes - resident_bytes - overhead - 2 * chunk_bytes
    per = 8 * (2 * n + 4 * D)
    return int(max(1, min(q, max_batch, free // per)))


def _cols_of(design, U):
    return np.concatenate([np.arange(*design.groups[j]) for j in U]) if len(U) else np.zeros(0, np.int64)


class _GramHeld:
    """Training design whose Gram blocks are answered from the full Gram matrix kept by the
    statistics strategy, so stage 2 and the relaxed refit read no training row."""

    def __init__(self, design, gram):
        self.design, self.gram = design, gram

    def __getattr__(self, k):
        return getattr(self.design, k)

    def cross(self, idx_a, idx_b):
        ca, cb = _cols_of(self.design, list(idx_a)), _cols_of(self.design, list(idx_b))
        if len(ca) == self.design.D and len(cb) == self.design.D:
            return self.gram
        return self.gram[np.ix_(ca, cb)]


def _screen_widths(c_full, groups, pen, factors):
    """Mean and mean-square candidate width per response and level under the sequential
    strong rule evaluated at the null fit (used by the cost model only)."""
    sizes = (groups[:, 1] - groups[:, 0]).astype(float)
    gn = group_norms(c_full, groups)
    lam_max = np.maximum((gn / pen).max(axis=0), 1e-300)
    w = []
    for k in range(1, len(factors)):
        thr = pen * ((2 * factors[k] - factors[k - 1]) * lam_max)[None, :]
        w.append(((gn >= thr) * sizes[:, None]).sum(axis=0))
    w = np.asarray(w, dtype=float)
    return float(w.mean()), float((w * w).mean())


class Chorus:
    def __init__(self, n_lambda=30, lambda_min_ratio=0.02, alphas=(0., .25, .5, .75, 1.),
                 gammas=(1., 2.), batch_size=None, memory_budget=None, tol=1e-6,
                 stage2=True, strong=True, stage2_batch=128, patience=5, mode='auto', ws_cap=True,
                 kappa=0.5, parsimony='groups', shrink='eb', reference='first', scale_norm=True,
                 crossfit=True, fold_seed=0, keep_candidates=False, max_width=None,
                 stage2_shared=True, relax=True, strategy='auto', stats_max_gram_bytes=2 << 30,
                 stats_panel_bytes=256 << 20, keep_path=False, relax_grid=None, relax_per_response=False,
                 switch_back=False):
        self.factors = lambda_factors(n_lambda, lambda_min_ratio)
        self.alphas, self.gammas = tuple(alphas), tuple(gammas)
        self.batch_size, self.memory_budget = batch_size, memory_budget
        self.tol, self.stage2, self.strong = tol, stage2, strong
        self.stage2_batch = stage2_batch
        self.patience = patience
        self.mode, self.ws_cap = mode, ws_cap
        self.kappa, self.parsimony, self.shrink, self.reference = kappa, parsimony, shrink, reference
        self.scale_norm, self.crossfit, self.fold_seed = scale_norm, crossfit, fold_seed
        self.keep_candidates = keep_candidates
        self.max_width = max_width
        self.stage2_shared = stage2_shared     # False: stage 2 rescans the design (no shared statistics)
        self.relax = relax
        # relaxation weights tried and whether each response picks its own (amendment 22 variants)
        self.relax_grid = tuple(relax_grid) if relax_grid is not None else None
        self.relax_per_response = relax_per_response
        self.switch_back = switch_back       # covariance back to residual updates when cheaper (amendment 22)
        # 'stats': full sufficient statistics in one pass; 'working_set': the batched engine;
        # 'auto': statistics when the Gram matrix fits and the cost model predicts it cheaper
        self.strategy = strategy
        self.stats_max_gram_bytes = stats_max_gram_bytes    # Gram limit without a memory budget
        self.stats_panel_bytes = stats_panel_bytes
        self.keep_path = keep_path        # stage-1 coefficients of every visited level (K, D, q)

    # ------------------------------------------------------------------ stage 1
    def _choose_strategy(self, design, y, y_val, cluster):
        """Return ('stats', plan) or ('working_set', None) for one basis."""
        if self.strategy == 'working_set':
            return 'working_set', None
        forced = self.mode != 'auto' or self.max_width is not None or not self.stage2_shared or not self.strong
        if self.strategy == 'auto' and forced:
            return 'working_set', None
        n, D, G, q, K = design.n, design.D, design.G, y.shape[1], len(self.factors)
        disk = design.nbytes_resident() == 0
        C = len(np.unique(cluster)) if cluster is not None else y_val.shape[0]
        if self.ledger_ is not None:
            bb = design.buffer_bytes() if hasattr(design, 'buffer_bytes') else 0
            plan = self.ledger_.plan_stats(n, D, G, q, y_val.shape[0], C, bb, design.nbytes_resident(),
                                           self.batch_size or 64, disk=disk, panel_cap=self.stats_panel_bytes)
        elif 8 * D * D <= self.stats_max_gram_bytes:
            plan = {'B': min(q, self.batch_size or 64), 'gram_bytes': 8 * D * D,
                    'panel_bytes': self.stats_panel_bytes, 'm_sub': None}
        else:
            plan = None
        if self.strategy == 'stats':
            if plan is None:
                raise MemoryBudgetError('the full Gram matrix and its transients do not fit the budget')
            return 'stats', plan
        if plan is None:
            return 'working_set', None
        if self.ledger_ is not None:
            try:
                B_ws = self.ledger_.plan(n, D, G, q, y_val.shape[0], C,
                                         design.buffer_bytes() if hasattr(design, 'buffer_bytes') else 0,
                                         design.nbytes_resident(), self.batch_size or 64)[0]
            except MemoryBudgetError:
                return 'stats', plan
        else:
            B_ws = self.batch_size or plan_batch(n, D, q, None, design.nbytes_resident())
        m1 = m2 = None
        if not disk:
            sizes = (design.groups[:, 1] - design.groups[:, 0]).astype(float)
            pen = np.repeat(np.sqrt(sizes)[:, None], q, axis=1)
            m1, m2 = _screen_widths(design.xt_dot(y) / n, design.groups, pen, self.factors)
        t_stats, t_ws = predict_strategy(n, D, q, K, B_ws, disk, design.probe() if disk else None, m1, m2)
        self._stats['pred_stats_seconds'] += t_stats
        self._stats['pred_ws_seconds'] += t_ws
        return ('stats', plan) if 1.25 * t_stats < t_ws else ('working_set', None)

    def _stage1(self, design, y, design_val, y_val, fold, cluster):
        strategy, plan = self._choose_strategy(design, y, y_val, cluster)
        if strategy == 'stats':
            try:
                return self._stage1_stats(design, y, design_val, y_val, fold, cluster, plan)
            except WidthOverflow:
                if self.ledger_ is not None:
                    self.ledger_.events['strategy_fallback'] += 1
                self._stats['strategy_fallback'] += 1
        return self._stage1_ws(design, y, design_val, y_val, fold, cluster)

    def _stage1_stats(self, design, y, design_val, y_val, fold, cluster, plan):
        """Stage 1 from the full sufficient statistics: one pass, then every response's path
        solved from X^T X / n and X^T Y / n with exact duality gaps (gram_path)."""
        n, q, G, K = design.n, y.shape[1], design.G, len(self.factors)
        sizes = design.groups[:, 1] - design.groups[:, 0]
        base = np.sqrt(sizes.astype(float))
        r0 = design.read_bytes
        gram, c_full, tm = design.stats_scan(y, plan['panel_bytes'] or self.stats_panel_bytes)
        if self.ledger_ is not None:
            self.ledger_.rss()
        yy = np.einsum('ij,ij->j', y, y) / n
        xv = design_val.x if hasattr(design_val, 'x') else design_val.cols(list(range(G)))
        out = {'B': plan['B'], 'm_max': 0, 'strategy': 'stats', 'gram_bytes': plan['gram_bytes'],
               'coef': np.zeros((design.D, q)), 'entry': np.zeros((G, q), np.int32),
               'norm': np.zeros((3, G, q)), 'c_full': c_full, 'closs': None,
               'best_k': np.zeros((3, q), np.int32), 'val_mse': np.full((K, q), np.nan),
               'stopped_at': np.zeros(q, np.int32), 'gaps': np.zeros(q)}
        st = {'passes': 1, 'read_bytes': 0, 'inner_epochs': 0,
              'pass_seconds': tm['scan_seconds'] - tm['syrk_seconds'], 'gram_seconds': tm['syrk_seconds'],
              'max_gap': 0., 'evicted': 0, 'max_width': 0}
        if self.keep_path:
            out['path'] = np.zeros((K, design.D, q))

        def one(cols):
            pen = np.repeat(base[:, None], len(cols), axis=1)
            res = fit_batch_gram(gram, c_full[:, cols], yy[cols], sizes, self.factors, pen, tol=self.tol,
                                 val=(xv, y_val[:, cols]), max_rounds=200, patience=self.patience,
                                 val_fold=fold, val_cluster=cluster, track_norms=True, ws_cap=self.ws_cap,
                                 tighten=True, chunked=True, ledger=self.ledger_, m_sub=plan['m_sub'],
                                 keep_path=self.keep_path)
            if self.keep_path:
                for k, c in enumerate(res['path']):
                    out['path'][k][:, cols] = c
            v = res['val']
            if out['closs'] is None:
                out['closs'] = np.zeros((3, v['closs'].shape[1], q))
                out['cl_fold'] = v['cl_fold']
            out['coef'][:, cols] = res['best_coef']
            out['entry'][:, cols] = res['entry']
            out['norm'][:, :, cols] = res['best_norm']
            out['closs'][:, :, cols] = v['closs']
            out['best_k'][:, cols] = v['best_k']
            out['val_mse'][:, cols] = v['val_mse'][2]
            out['stopped_at'][cols] = res['stopped_at']
            out['gaps'][cols] = res['gaps'].max(axis=0)
            st['inner_epochs'] += res['epochs']
            st['max_gap'] = max(st['max_gap'], float(res['gaps'].max()))
            st['max_width'] = max(st['max_width'], int(res['max_cand_width']))

        _run_batches(one, q, plan['B'], self.ledger_)
        st['read_bytes'] = design.read_bytes - r0
        out['stats'] = st
        out['gram'] = gram          # kept for stage 2 and the relaxed refit, released after the fit
        return out

    def _stage1_ws(self, design, y, design_val, y_val, fold, cluster):
        n, q, G, K = design.n, y.shape[1], design.G, len(self.factors)
        base = np.sqrt((design.groups[:, 1] - design.groups[:, 0]).astype(float))
        m_max = None
        if self.ledger_ is not None:
            C = len(np.unique(cluster)) if cluster is not None else y_val.shape[0]
            bb = design.buffer_bytes() if hasattr(design, 'buffer_bytes') else 0
            B, m_max = self.ledger_.plan(n, design.D, G, q, y_val.shape[0], C, bb,
                                         design.nbytes_resident(), self.batch_size or 64)
        else:
            B = self.batch_size or plan_batch(n, design.D, q, None, design.nbytes_resident())
            m_max = self.max_width
        out = {'B': B, 'm_max': m_max, 'coef': np.zeros((design.D, q)), 'entry': np.zeros((G, q), np.int32),
               'norm': np.zeros((3, G, q)), 'c_full': np.zeros((design.D, q)),
               'closs': None, 'best_k': np.zeros((3, q), np.int32), 'val_mse': np.full((K, q), np.nan),
               'stopped_at': np.zeros(q, np.int32), 'gaps': np.zeros(q)}
        st = {'passes': 0, 'read_bytes': 0, 'inner_epochs': 0, 'pass_seconds': 0., 'gram_seconds': 0.,
              'max_gap': 0., 'evicted': 0, 'max_width': 0}
        if self.keep_path:
            out['path'] = np.zeros((K, design.D, q))

        def one(cols):
            pen = np.repeat(base[:, None], len(cols), axis=1)
            res = fit_batch(design, y[:, cols], self.factors, pen, tol=self.tol,
                            val=(design_val, y_val[:, cols]), strong=self.strong,
                            patience=self.patience, mode=self.mode, ws_cap=self.ws_cap,
                            val_fold=fold, val_cluster=cluster, switch_back=self.switch_back,
                            m_max=self.ledger_.m_max_for(len(cols)) if self.ledger_ is not None else m_max,
                            ledger=self.ledger_, keep_path=self.keep_path)
            if self.keep_path:
                for k, c in enumerate(res['path']):
                    out['path'][k][:, cols] = c
            v = res['val']
            if out['closs'] is None:
                out['closs'] = np.zeros((3, v['closs'].shape[1], q))
                out['cl_fold'] = v['cl_fold']
            out['coef'][:, cols] = res['best_coef']
            out['entry'][:, cols] = res['entry']
            out['norm'][:, :, cols] = res['best_norm']
            out['c_full'][:, cols] = res['c_full']
            out['closs'][:, :, cols] = v['closs']
            out['best_k'][:, cols] = v['best_k']
            out['val_mse'][:, cols] = v['val_mse'][2]
            out['stopped_at'][cols] = res['stopped_at']
            out['gaps'][cols] = res['gaps'].max(axis=0)
            for k in ('passes', 'read_bytes', 'inner_epochs', 'pass_seconds', 'gram_seconds', 'evicted'):
                st[k] += res[k]
            st['max_gap'] = max(st['max_gap'], float(res['gaps'].max()))
            st['max_width'] = max(st['max_width'], res['max_width'])

        _run_batches(one, q, B, self.ledger_)
        out['stats'] = st
        return out

    # ------------------------------------------------------------------ stage 2
    def _stage2(self, design, y, design_val, y_val, s1, fold, cluster, scale, bidx, gl_cand):
        n, q, K = design.n, y.shape[1], len(self.factors)
        sizes = design.groups[:, 1] - design.groups[:, 0]
        base = np.sqrt(sizes.astype(float))
        sel = (0, 1, 2) if fold is not None else (2,)
        a = {S: s1['norm'][S] / scale[None, :] for S in sel}
        U = np.flatnonzero(np.any([a[S].mean(axis=1) > 0 for S in sel], axis=0))
        cands = []
        if len(U) == 0:
            return cands, U
        cols = _cols_of(design, U)
        mU = len(cols)
        streamed = not self.stage2_shared
        if self.ledger_ is not None and not streamed:
            g_copies = 1 if isinstance(design, _GramHeld) else 2     # the full Gram is already charged
            need = 8 * (g_copies * mU * mU + y_val.shape[0] * mU + mU * (q + 4 * min(3 * q, self.stage2_batch)))
            streamed = (1 + self.ledger_.eps) * need > self.ledger_.free(self.ledger_.fixed)
        if streamed:
            if self.ledger_ is not None and self.stage2_shared:
                self.ledger_.events['stage2_fallback'] += 1
            gram = xv = None
        else:
            t0 = time.perf_counter()
            r0 = design.read_bytes
            gram = design.cross(list(U), list(U))
            xv = design_val.cols(list(U))
            self._stats['stage2_read_bytes'] += design.read_bytes - r0
            self._stats['stage2_gram_seconds'] += time.perf_counter() - t0
        yy = np.einsum('ij,ij->j', y, y) / n
        cu = s1['c_full'][cols]
        su = sizes[U]
        starts = np.concatenate([[0], np.cumsum(su)[:-1]])
        cl_fold = s1['cl_fold']
        for alpha in self.alphas:
            for gamma in self.gammas:
                pens = {}
                for S in sel:
                    mix = (1 - alpha) * a[S][U] + alpha * a[S][U].mean(axis=1, keepdims=True)
                    with np.errstate(divide='ignore'):
                        pens[S] = base[U][:, None] * np.where(mix > 0, mix ** (-gamma), np.inf)
                col_resp, col_pen, cmap = [], [], {}
                for r in range(q):
                    for S in sel:
                        hit = None
                        for S2 in sel:
                            if S2 < S and np.array_equal(pens[S2][:, r], pens[S][:, r]):
                                hit = cmap[(r, S2)]
                                break
                        if hit is None:
                            hit = len(col_resp)
                            col_resp.append(r)
                            col_pen.append(pens[S][:, r])
                        cmap[(r, S)] = hit
                col_resp = np.asarray(col_resp)
                P = np.stack(col_pen, axis=1)
                nc = len(col_resp)
                serve = np.zeros((3, nc), bool)
                for (r, S), ci in cmap.items():
                    serve[S, ci] = True
                bc = np.zeros((len(cols), nc))
                ent = np.zeros((len(U), nc), np.int32)
                closs = np.zeros((3, len(cl_fold), nc))
                if not streamed:
                    for c0 in range(0, nc, self.stage2_batch):
                        c1 = min(nc, c0 + self.stage2_batch)
                        rr = col_resp[c0:c1]
                        res = fit_batch_gram(gram, cu[:, rr], yy[rr], su, self.factors, P[:, c0:c1],
                                             tol=self.tol, val=(xv, y_val[:, rr]), patience=self.patience,
                                             val_fold=fold, val_cluster=cluster, serve=serve[:, c0:c1])
                        bc[:, c0:c1] = res['best_coef']
                        ent[:, c0:c1] = res['entry']
                        closs[:, :, c0:c1] = res['val']['closs']
                        self._stats['stage2_epochs'] += res['epochs']
                else:
                    # the Gram matrix of U does not fit the budget: solve the same weighted
                    # problems with the streaming engine (weights are infinite outside U)
                    Pfull = np.full((design.G, nc), np.inf)
                    Pfull[U] = P
                    B2 = self.stage1_[bidx]['B']

                    def one(cc):
                        rr = col_resp[cc]
                        res = fit_batch(design, y[:, rr], self.factors, Pfull[:, cc], tol=self.tol,
                                        val=(design_val, y_val[:, rr]), strong=self.strong,
                                        patience=self.patience, mode=self.mode, ws_cap=self.ws_cap,
                                        val_fold=fold, val_cluster=cluster, serve=serve[:, cc], switch_back=self.switch_back,
                                        m_max=self.ledger_.m_max_for(len(cc)) if self.ledger_ is not None else None,
                                        ledger=self.ledger_)
                        bc[:, cc] = res['best_coef'][cols]
                        ent[:, cc] = res['entry'][U]
                        closs[:, :, cc] = res['val']['closs']
                        self._stats['stage2_epochs'] += res['inner_epochs']
                        self._stats['stage2_read_bytes'] += res['read_bytes']

                    _run_batches(one, nc, B2, self.ledger_)
                rr = np.arange(q)
                c2 = np.array([cmap[(r, 2)] for r in rr])
                coef_u = bc[:, c2]
                if fold is not None:
                    c0_ = np.array([cmap[(r, 0)] for r in rr])
                    c1_ = np.array([cmap[(r, 1)] for r in rr])
                    cross = np.where((cl_fold == 0)[:, None], closs[1][:, c1_], closs[0][:, c0_])
                else:
                    cross = closs[2][:, c2]
                nzg = np.sqrt(np.add.reduceat(coef_u * coef_u, starts, axis=0)) > 0      # (|U|, q)
                cands.append({'basis': bidx, 'kind': 'reweighted', 'alpha': alpha, 'gamma': gamma,
                              'closs': cross, 'vloss': closs[2][:, c2],
                              'groups': nzg.sum(axis=0), 'params': (nzg * su[:, None]).sum(axis=0),
                              'coef_u': coef_u, 'entry_u': ent[:, c2], 'U': U, 'cols': cols})
        if self.relax:
            self._relax(design, design_val, y_val, s1, [gl_cand] + cands, U, gram, xv, fold, cluster)
        return cands, U

    # ------------------------------------------------------------------ relaxed refit
    def _relax(self, design, design_val, y_val, s1, cl, U, gram, xv, fold, cluster):
        """Pooled relaxed refit of every candidate of one basis (cl[0] is its GL candidate).
        With the stage-2 Gram matrix of U in memory, no training row is read; otherwise the
        Gram matrix of the union of the candidates' supports is formed in one shared pass."""
        q = y_val.shape[1]
        sizes = design.groups[:, 1] - design.groups[:, 0]
        cols = _cols_of(design, U)
        su = sizes[U]
        starts = np.concatenate([[0], np.cumsum(su)[:-1]]).astype(np.int64)
        blocks = [s1['coef'][cols]] + [c['coef_u'] for c in cl[1:]]
        for c in cl:
            c['phi'], c['relax'] = np.zeros(q), {}
        if gram is None:
            act = np.zeros(len(U), bool)
            for b_ in blocks:
                act |= np.add.reduceat(b_ * b_, starts, axis=0).max(axis=1) > 0
            V = np.flatnonzero(act)
            if len(V) == 0:
                return
            pos = np.concatenate([np.arange(starts[t], starts[t] + su[t]) for t in V])
            mV = len(pos)
            need = 8 * (mV * mV + y_val.shape[0] * mV)
            if self.ledger_ is not None and (1 + self.ledger_.eps) * need > self.ledger_.free(self.ledger_.fixed):
                self._stats['relax_skipped'] += 1
                return
            r0 = design.read_bytes
            gram = design.cross(list(U[V]), list(U[V]))
            xv = design_val.cols(list(U[V]))
            self._stats['stage2_read_bytes'] += design.read_bytes - r0
            blocks = [b_[pos] for b_ in blocks]
            xy = s1['c_full'][cols[pos]]
            su_ = su[V]
            st_ = np.concatenate([[0], np.cumsum(su_)[:-1]]).astype(np.int64)
            back = cols[pos]
        else:
            xy, su_, st_, back = s1['c_full'][cols], su, starts, cols
        outside = np.setdiff1d(np.arange(design.G), U)
        gl_out = (group_norms(s1['coef'], design.groups)[outside] > 0).any(axis=0) if len(outside) else np.zeros(q, bool)
        for i, (c, b_) in enumerate(zip(cl, blocks)):
            rk = {'per_response': self.relax_per_response}
            if self.relax_grid is not None:
                rk['phis'] = self.relax_grid
            c['phi'], out = relax_block(gram, xy, xv, y_val, b_, st_, su_, **rk)
            if i == 0:                      # a GL support outside U is never refit partially
                for r in np.flatnonzero(gl_out):
                    c['phi'][r] = 0.
                    out.pop(r, None)
            c['relax'] = {r: (back[p], v) for r, (p, v) in out.items()}

    # ------------------------------------------------------------------ fit
    def fit(self, design, y, design_val, y_val, val_fold=None, val_cluster=None):
        t0 = time.perf_counter()
        designs = list(design) if isinstance(design, (list, tuple)) else [design]
        designs_val = list(design_val) if isinstance(design_val, (list, tuple)) else [design_val]
        q, n_val, K = y.shape[1], y_val.shape[0], len(self.factors)
        scale = y.std(axis=0) if self.scale_norm else np.ones(q)
        scale = np.where(scale > 0, scale, 1.)
        if self.crossfit:
            fold, cluster = make_folds(n_val, self.fold_seed, val_cluster, val_fold)
        else:
            fold, cluster = None, (np.arange(n_val) if val_cluster is None else np.unique(val_cluster, return_inverse=True)[1])
        self._stats = {'stage2_read_bytes': 0, 'stage2_gram_seconds': 0., 'stage2_epochs': 0, 'relax_skipped': 0,
                       'strategy_stats': 0, 'strategy_fallback': 0, 'stats_gram_bytes': 0,
                       'pred_stats_seconds': 0., 'pred_ws_seconds': 0.}
        self.ledger_ = Ledger(self.memory_budget) if self.memory_budget else None
        self.stage1_, cands = [], []
        held = list(designs)          # training designs seen by stage 2 (Gram held under statistics)
        t1 = 0.
        for b, (dt, dv) in enumerate(zip(designs, designs_val)):
            ts = time.perf_counter()
            s1 = self._stage1(dt, y, dv, y_val, fold, cluster)
            t1 += time.perf_counter() - ts
            gram = s1.pop('gram', None)
            if gram is not None:
                held[b] = _GramHeld(dt, gram)
                self._stats['strategy_stats'] += 1
                self._stats['stats_gram_bytes'] += 8 * dt.D * dt.D
                if self.ledger_ is not None:
                    self.ledger_.retain(8 * dt.D * dt.D)
            self.stage1_.append(s1)
            sizes = dt.groups[:, 1] - dt.groups[:, 0]
            nzg = s1['norm'][2] > 0
            cross = (np.where((s1['cl_fold'] == 0)[:, None], s1['closs'][1], s1['closs'][0])
                     if fold is not None else s1['closs'][2])
            cands.append({'basis': b, 'kind': 'GL', 'alpha': None, 'gamma': None, 'closs': cross,
                          'vloss': s1['closs'][2], 'groups': nzg.sum(axis=0),
                          'params': (nzg * sizes[:, None]).sum(axis=0)})
        self.stage1_seconds_ = t1
        ts = time.perf_counter()
        if self.stage2:
            gl_of = {c['basis']: c for c in cands if c['kind'] == 'GL'}
            for b, (dt, dv) in enumerate(zip(held, designs_val)):
                cb, _ = self._stage2(dt, y, dv, y_val, self.stage1_[b], fold, cluster, scale, b, gl_of[b])
                cands.extend(cb)
        elif self.relax:
            for b, (dt, dv) in enumerate(zip(held, designs_val)):
                U = np.flatnonzero(group_norms(self.stage1_[b]['coef'], dt.groups).max(axis=1) > 0)
                if len(U):
                    self._relax(dt, dv, y_val, self.stage1_[b], [cands[b]], U, None, None, fold, cluster)
        for b, dt in enumerate(held):
            if isinstance(dt, _GramHeld):
                held[b] = None
                if self.ledger_ is not None:
                    self.ledger_.retain(-8 * dt.D * dt.D)
        del held
        self.stage2_seconds_ = time.perf_counter() - ts
        self.candidates_ = cands
        ts = time.perf_counter()
        self._select(designs, y_val, cands)
        self.select_seconds_ = time.perf_counter() - ts
        st = self.stage1_[0]['stats'].copy()
        for s1 in self.stage1_[1:]:
            for k in ('passes', 'read_bytes', 'inner_epochs', 'pass_seconds', 'gram_seconds', 'evicted'):
                st[k] += s1['stats'][k]
            st['max_gap'] = max(st['max_gap'], s1['stats']['max_gap'])
        if self.ledger_ is not None:
            self._stats.update({f'ledger_{k}': v for k, v in self.ledger_.events.items()})
            self._stats.update({'ledger_M0': self.ledger_.M0, 'ledger_m_max': self.stage1_[0]['m_max']})
        self.stats_ = {**st, **self._stats, 'batch_size': self.stage1_[0]['B'],
                       'stage1_seconds': self.stage1_seconds_, 'stage2_seconds': self.stage2_seconds_,
                       'select_seconds': self.select_seconds_, 'seconds': time.perf_counter() - t0}
        if not self.keep_candidates:
            for c in cands:
                c.pop('coef_u', None)
        return self

    def _select(self, designs, y_val, cands):
        q, K = y_val.shape[1], len(self.factors)
        yvar = np.maximum(y_val.var(axis=0), 1e-300)
        closs = np.stack([c['closs'] for c in cands])
        groups = np.stack([c['groups'] for c in cands])
        params = np.stack([c['params'] for c in cands])
        gl = [i for i, c in enumerate(cands) if c['kind'] == 'GL']
        if self.reference == 'best_gl':
            e = closs[gl].sum(axis=1)
            ref = np.asarray(gl)[np.argmin(e, axis=0)]
        else:
            ref = np.full(q, gl[0])
        order = np.arange(len(cands))
        if len(cands) == 1:
            pick, diag = np.zeros(q, np.int64), {}
        else:
            pick, diag = eb_select(closs, groups, params, yvar, y_val.shape[0], ref, self.kappa,
                                   self.parsimony, self.shrink, order)
        self.pick_, self.select_diag_ = pick, diag
        self.coef_b_ = [np.zeros((d.D, q)) for d in designs]
        G = designs[0].G
        self.entry_ = np.zeros((G, q), np.int32)
        for r in range(q):
            c = cands[pick[r]]
            b = c['basis']
            s1 = self.stage1_[b]
            if c['kind'] == 'GL':
                self.coef_b_[b][:, r] = s1['coef'][:, r]
                self.entry_[:, r] = s1['entry'][:, r]
            else:
                self.coef_b_[b][c['cols'], r] = c['coef_u'][:, r]
                e = K + s1['entry'][:, r].astype(np.int32)
                e[c['U']] = np.where(c['entry_u'][:, r] < K, c['entry_u'][:, r], K + s1['entry'][c['U'], r])
                self.entry_[:, r] = e
        self.support_ = np.zeros((G, q), bool)
        for b, d in enumerate(designs):
            self.support_ |= group_norms(self.coef_b_[b], d.groups) > 0
        # relaxed refit of the chosen candidate (the support is kept)
        self.coef_pen_b_ = [cb.copy() for cb in self.coef_b_]
        self.phi_ = np.zeros(q)
        for r in range(q):
            c = cands[pick[r]]
            if r in c.get('relax', {}):
                idx, v = c['relax'][r]
                self.coef_b_[c['basis']][:, r] = 0.
                self.coef_b_[c['basis']][idx, r] = v
                self.phi_[r] = c['phi'][r]
        self.basis_of_ = np.array([cands[i]['basis'] for i in pick])
        self.config_of_ = [(cands[i]['kind'], cands[i]['alpha'], cands[i]['gamma'], cands[i]['basis']) for i in pick]
        self.coef_ = self.coef_b_[0]

    def candidate_coef(self, i, design_index_D, relaxed=True):
        """Dense coefficients of candidate i on its basis (D_b x q), relaxed when relaxed=True."""
        c = self.candidates_[i]
        s1 = self.stage1_[c['basis']]
        if c['kind'] == 'GL':
            out = s1['coef'].copy()
        else:
            out = np.zeros((design_index_D, s1['coef'].shape[1]))
            out[c['cols']] = c['coef_u']
        if relaxed:
            for r, (idx, v) in c.get('relax', {}).items():
                out[:, r] = 0.
                out[idx, r] = v
        return out

    def candidate_entry(self, i):
        K = len(self.factors)
        c = self.candidates_[i]
        s1 = self.stage1_[c['basis']]
        if c['kind'] == 'GL':
            return s1['entry']
        e = K + s1['entry'].astype(np.int32)
        e[c['U']] = np.where(c['entry_u'] < K, c['entry_u'], K + s1['entry'][c['U']])
        return e

    def predict(self, designs_test, coef=None):
        dts = list(designs_test) if isinstance(designs_test, (list, tuple)) else [designs_test]
        coefs = self.coef_b_ if coef is None else (coef if isinstance(coef, (list, tuple)) else [coef])
        out = None
        for dt, cb in zip(dts, coefs):
            groups = np.flatnonzero(group_norms(cb, dt.groups).max(axis=1) > 0)
            if len(groups) == 0:
                p = np.zeros((dt.n, cb.shape[1]))
            else:
                p = dt.x_dot_cols(list(groups), cb[_cols_of(dt, groups)])
            out = p if out is None else out + p
        return out

    def support(self, groups=None, coef=None):
        return self.support_ if coef is None else group_norms(coef, groups) > 0
