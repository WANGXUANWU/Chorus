"""One isolated Python method run on a case: Chorus (and its ablations), skglm, Celer.

Usage: py_worker.py CASE_DIR METHOD OUT_DIR [JSON_OPTIONS]
Timing stages: load (reading arrays), warmup (imports and compilation), fit (paths for all
responses plus validation selection), predict. The parent process samples memory.
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from threadpoolctl import threadpool_limits
from bench.caseio import read_meta, load

case, method, out = sys.argv[1], sys.argv[2], sys.argv[3]
opts = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
os.makedirs(out, exist_ok=True)


def stage(name):
    with open(os.path.join(out, 'stage.json'), 'w') as fh:
        json.dump({'stage': name, 'time': time.time()}, fh)


t_all = time.perf_counter()
res = {'method': method, 'options': opts}
try:
    stage('load')
    meta = read_meta(case)
    groups = np.asarray(meta['groups'], dtype=np.int64)
    sizes = groups[:, 1] - groups[:, 0]
    K = int(meta.get('n_lambda', 30))
    factors = np.geomspace(1.0, float(meta.get('lambda_min_ratio', 0.02)), K)
    disk = bool(opts.get('disk', False))
    if method.startswith('Chorus') and opts.get('budget'):
        # refuse an infeasible budget before any input is loaded or any kernel is compiled
        import psutil
        rss0 = psutil.Process().memory_info().rss
        from chorus.ledger import preflight, Ledger
        n_t, n_v, D_ = int(meta['n_train']), int(meta['n_val']), int(meta['D'])
        q_ = (opts['resp_range'][1] - opts['resp_range'][0]) if opts.get('resp_range') else int(meta['q'])
        planned = 8 * (n_t * q_ + n_v * q_ + n_v * D_ + (0 if disk else n_t * D_))
        fixed = Ledger.full_q(D_, len(groups), q_, n_v) + (3 * int(opts.get('chunk_bytes', 64 << 20)) if disk else 0)
        res['preflight_bytes'] = preflight(int(opts['budget']), planned, fixed, now=rss0)
    t0 = time.perf_counter()
    yt, yv = load(case, 'y', 'train', meta), load(case, 'y', 'val', meta)
    if opts.get('resp_range'):
        r0, r1 = opts['resp_range']
        yt, yv = yt[:, r0:r1], yv[:, r0:r1]
    xv = load(case, 'X', 'val', meta)
    if method.startswith('Chorus') and disk:
        xt = None
    else:
        xt = load(case, 'X', 'train', meta)
    res['load_seconds'] = time.perf_counter() - t0
    q, G = yt.shape[1], len(groups)
    with threadpool_limits(limits=int(opts.get('threads', 1))):
        stage('warmup')
        t0 = time.perf_counter()
        if method.startswith('Chorus'):
            from chorus import Chorus, InMemoryDesign, DiskDesign
            from chorus.engine import group_norms
            from chorus.kernels import solve_batch
            # explicit compilation warmup on a tiny problem (excluded from fit time)
            from chorus.engine import fit_batch
            rng = np.random.default_rng(0)
            xw = rng.standard_normal((20, 4)); xw -= xw.mean(0)
            xw /= np.sqrt((xw ** 2).mean(0))
            gw = np.array([[0, 2], [2, 4]])
            fit_batch(InMemoryDesign(xw, gw), rng.standard_normal((20, 2)), np.array([1., .5]),
                      np.ones((2, 2)), tol=1e-4)
            # kernels and machine constants of the statistics strategy (excluded from fit time)
            from chorus.gram_path import fit_batch_gram
            from chorus.kernels import symmetrize_lower
            from chorus.costmodel import constants
            gww = xw.T @ xw / 20
            fit_batch_gram(gww, xw.T @ rng.standard_normal((20, 2)) / 20, np.ones(2), np.array([2, 2]),
                           np.array([1., .5]), np.ones((2, 2)), tol=1e-4, track_norms=True, ws_cap=True,
                           tighten=True, chunked=True, max_rounds=200)
            symmetrize_lower(np.asfortranarray(gww.copy()))
            constants()
        elif method == 'skglm':
            from skglm import GroupLasso
        elif method == 'Celer':
            from celer import GroupLasso
        elif method in ('adelie', 'adaptive'):
            import adelie as ad
        res['warmup_seconds'] = time.perf_counter() - t0
        stage('fit')
        t0 = time.perf_counter()
        if method.startswith('Chorus'):
            if disk:
                d_train = DiskDesign(os.path.join(case, 'X_train.f64'),
                                     chunk_bytes=int(opts.get('chunk_bytes', 64 << 20)))
            else:
                d_train = InMemoryDesign(xt, groups)
            # further bases (e.g. three functions per predictor) as sibling case folders
            base_dirs = [os.path.normpath(os.path.join(case, bd)) for bd in opts.get('bases', [])]
            d_trains, d_vals = [d_train], [InMemoryDesign(xv, groups)]
            for bd in base_dirs:
                mb = read_meta(bd)
                gb = np.asarray(mb['groups'], dtype=np.int64)
                d_trains.append(InMemoryDesign(load(bd, 'X', 'train', mb), gb))
                d_vals.append(InMemoryDesign(load(bd, 'X', 'val', mb), gb))
            vfold = vclu = None
            if opts.get('val_cluster') == 'week':
                ts = np.load(os.path.join(case, 'timestamps.npy'))     # hours since 1970, all rows
                nt, nv = meta['n_train'], meta['n_val']
                week = (ts[nt:nt + nv] - 96) // (24 * 7)                  # weeks start on Monday
                vclu, vfold = week - week.min(), (week % 2).astype(np.int64)
            kw = dict(n_lambda=K, lambda_min_ratio=float(meta.get('lambda_min_ratio', 0.02)),
                      alphas=tuple(opts.get('alphas', (0., .25, .5, .75, 1.))),
                      gammas=tuple(opts.get('gammas', (1., 2.))),
                      batch_size=opts.get('batch_size'), memory_budget=opts.get('budget'),
                      stage2=bool(opts.get('stage2', True)), strong=bool(opts.get('strong', True)),
                      patience=opts.get('patience', 5), mode=opts.get('mode', 'auto'),
                      ws_cap=bool(opts.get('ws_cap', True)), tol=float(opts.get('tol', 1e-6)),
                      kappa=float(opts.get('kappa', 0.5)), parsimony=opts.get('parsimony', 'groups'),
                      shrink=opts.get('shrink', 'eb'), reference=opts.get('reference', 'first'),
                      scale_norm=bool(opts.get('scale_norm', True)), crossfit=bool(opts.get('crossfit', True)),
                      fold_seed=int(opts.get('fold_seed', 0)),
                      stage2_shared=bool(opts.get('stage2_shared', True)), relax=bool(opts.get('relax', True)),
                      keep_candidates=bool(opts.get('save_variants', False)),
                      strategy=opts.get('strategy', 'auto'), keep_path=bool(opts.get('save_coef_path', False)))
            if opts.get('stats_max_gram_bytes'):
                kw['stats_max_gram_bytes'] = int(opts['stats_max_gram_bytes'])
            if opts.get('relax_grid'):
                kw['relax_grid'] = tuple(float(v) for v in opts['relax_grid'])
            if opts.get('switch_back'):
                kw['switch_back'] = True
            if opts.get('relax_per_response'):
                kw['relax_per_response'] = True
            model = Chorus(**kw).fit(d_trains, yt, d_vals, yv, val_fold=vfold, val_cluster=vclu)
            coef, entry = model.coef_, model.entry_
            res['fit_seconds'] = time.perf_counter() - t0
            cfg = {}
            for c in model.config_of_:
                cfg[str(c)] = cfg.get(str(c), 0) + 1
            res['chorus'] = {**{k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                                for k, v in model.stats_.items()}, 'config_counts': cfg,
                             'tau2': [float(t) for t in model.select_diag_.get('tau2', [])]}
            stage('predict')
            del xt
            xte = load(case, 'X', 'test', meta)
            d_tests = [InMemoryDesign(xte, groups)]
            for bd in base_dirs:
                mb = read_meta(bd)
                d_tests.append(InMemoryDesign(load(bd, 'X', 'test', mb), np.asarray(mb['groups'], dtype=np.int64)))
            t1 = time.perf_counter()
            pred = model.predict(d_tests)
            res['predict_seconds'] = time.perf_counter() - t1
            s1 = model.stage1_[0]
            extra = {'support': model.support_, 'pick': model.pick_, 'basis_of': model.basis_of_,
                     'phi': model.phi_, 'pred_pen': model.predict(d_tests, model.coef_pen_b_),
                     's1_val_mse': s1['val_mse'], 's1_best_k': s1['best_k'], 's1_stopped_at': s1['stopped_at'],
                     's1_gaps': s1['gaps']}
            if 'path' in s1:
                extra['coef_path'] = s1['path']
            for b in range(1, len(d_trains)):
                extra[f'coef_b{b}'] = model.coef_b_[b]
            if opts.get('save_variants'):
                # every candidate (basis, GL or (alpha, gamma)), for offline selection ablations
                C = len(model.candidates_)
                extra['cand_pred'] = np.zeros((C, xte.shape[0], q), np.float32)
                extra['cand_pred_pen'] = np.zeros((C, xte.shape[0], q), np.float32)
                extra['cand_phi'] = np.stack([c.get('phi', np.zeros(q)) for c in model.candidates_])
                extra['cand_support'] = np.zeros((C, G, q), bool)
                extra['cand_entry'] = np.zeros((C, G, q), np.int16)
                for i, c in enumerate(model.candidates_):
                    dtb = d_tests[c['basis']]
                    cb = model.candidate_coef(i, dtb.D)
                    extra['cand_pred'][i] = model.predict([dtb], [cb])
                    extra['cand_pred_pen'][i] = model.predict([dtb], [model.candidate_coef(i, dtb.D, relaxed=False)])
                    extra['cand_support'][i] = group_norms(cb, dtb.groups) > 0
                    extra['cand_entry'][i] = model.candidate_entry(i)
                for k2 in ('closs', 'vloss', 'groups', 'params'):
                    extra[f'cand_{k2}'] = np.stack([c[k2] for c in model.candidates_])
                extra['cand_label'] = np.array([f"{c['kind']}|{c['alpha']}|{c['gamma']}|{c['basis']}"
                                                for c in model.candidates_])
                extra['val_var'] = yv.var(axis=0)
                extra['n_val'] = np.array(yv.shape[0])
                extra['cl_fold'] = s1['cl_fold']
                for k2, v2 in model.select_diag_.items():
                    extra[f'sel_{k2}'] = v2
            val_path = None
        else:
            n = xt.shape[0]
            corr = xt.T @ yt / n
            lam_max = (np.sqrt(np.add.reduceat(corr ** 2, groups[:, 0], axis=0)) /
                       np.sqrt(sizes)[:, None]).max(axis=0)
            coef = np.zeros((meta['D'], q))
            entry = np.full((G, q), K, dtype=np.int32)
            val_path = np.full((K, q), np.nan)
            best_k = np.zeros(q, dtype=np.int32)
            save_vp = bool(opts.get('save_val_pred'))
            vres = np.zeros((K, xv.shape[0], q), dtype=np.float32) if save_vp else None
            tol = float(opts.get('tol', 1e-6))
            save_path = bool(opts.get('save_path'))
            if save_path:      # test predictions and active groups at every level (accuracy runs only)
                xte_p = load(case, 'X', 'test', meta)
                pred_path = np.zeros((K, xte_p.shape[0], q), np.float32)
                act_path = np.zeros((K, G, q), bool)
            save_cp = bool(opts.get('save_coef_path'))
            coef_path = np.zeros((K, meta['D'], q)) if save_cp else None      # float64: gaps to 1e-13
            # validation-aware exit with the rule of Chorus (whole-split selector, patience levels)
            patience = opts.get('patience')
            if patience is not None:
                from chorus.validation import ValTracker
            stopped_at = np.full(q, K, dtype=np.int32)

            def level(r, k, b, tracker):
                """Record level k of response r; True when the exit rule retires the response."""
                nzg = np.sqrt(np.add.reduceat(b * b, groups[:, 0])) > 0
                entry[(entry[:, r] == K) & nzg, r] = k
                rv = yv[:, r] - xv @ b
                val_path[k, r] = float((rv ** 2).mean())
                if save_vp:
                    vres[k, :, r] = rv
                if save_path:
                    pred_path[k, :, r] = xte_p @ b
                    act_path[k, :, r] = nzg
                if save_cp:
                    coef_path[k, :, r] = b
                if tracker is None:
                    if val_path[k, r] < level.best[r] * (1 - 1e-12):
                        level.best[r], coef[:, r], best_k[r] = val_path[k, r], b, k
                    return False
                moved = tracker.update(k, np.array([0]), rv[:, None])
                if moved[2][0]:
                    coef[:, r], best_k[r] = b, k
                if tracker.exhausted(k, np.array([0]), int(patience))[0]:
                    stopped_at[r] = k
                    return True
                return False
            level.best = np.full(q, np.inf)

            if method == 'adelie':
                xa = np.asfortranarray(xt)
                for r in range(q):
                    tracker = ValTracker(xv.shape[0], 1, K) if patience is not None else None
                    exit_cond = None
                    if tracker is not None:
                        def exit_cond(state, r=r, tracker=tracker):
                            bk = state.betas[state.betas.shape[0] - 1].toarray().ravel()
                            return level(r, state.betas.shape[0] - 1, bk, tracker)
                    st = ad.grpnet(xa, ad.glm.gaussian(np.ascontiguousarray(yt[:, r])),
                                   groups=groups[:, 0].astype(np.int64),
                                   penalty=np.sqrt(sizes).astype(np.float64),
                                   lmda_path=factors * lam_max[r], early_exit=False, intercept=False,
                                   tol=float(opts.get('adelie_tol', 1e-7)), progress_bar=False, n_threads=1,
                                   exit_cond=exit_cond)
                    if tracker is None:
                        Bp = st.betas.toarray()                      # (K, D)
                        for k in range(K):
                            level(r, k, Bp[k], None)
            elif method == 'adaptive':
                # two-stage adaptive group lasso (Huang, Horowitz and Wei 2010) solved by adelie:
                # stage 1 is the validation-selected group lasso, stage 2 refits the groups it
                # selected with penalties sqrt(d_j) ||b1_j||^(-gamma); gamma and the stage-2
                # level are chosen by the validation error (amendment 22)
                xa = np.asfortranarray(xt)
                ad_tol = float(opts.get('adelie_tol', 1e-7))
                gammas = tuple(float(g) for g in opts.get('gammas', (1., 2.)))
                gamma_of = np.zeros(q)
                for r in range(q):
                    yr = np.ascontiguousarray(yt[:, r])
                    st = ad.grpnet(xa, ad.glm.gaussian(yr), groups=groups[:, 0].astype(np.int64),
                                   penalty=np.sqrt(sizes).astype(np.float64),
                                   lmda_path=factors * lam_max[r], early_exit=False, intercept=False,
                                   tol=ad_tol, progress_bar=False, n_threads=1)
                    B1 = st.betas.toarray()
                    b1 = B1[int(np.argmin(((yv[:, r:r + 1] - xv @ B1.T) ** 2).mean(axis=0)))]
                    nrm = np.sqrt(np.add.reduceat(b1 * b1, groups[:, 0]))
                    J = np.flatnonzero(nrm > 0)
                    if len(J) == 0:             # stage 1 selects nothing: the null model
                        for k in range(K):
                            level(r, k, np.zeros(meta['D']), None)
                        continue
                    cols = np.concatenate([np.arange(groups[j, 0], groups[j, 1]) for j in J])
                    xj = np.asfortranarray(xt[:, cols])
                    gj = np.concatenate([[0], np.cumsum(sizes[J])[:-1]]).astype(np.int64)
                    cj = xj.T @ yr / n
                    best = None
                    for gam in gammas:
                        pen = (np.sqrt(sizes[J]) * nrm[J] ** (-gam)).astype(np.float64)
                        lm = float((np.sqrt(np.add.reduceat(cj * cj, gj)) / pen).max())
                        st2 = ad.grpnet(xj, ad.glm.gaussian(yr), groups=gj, penalty=pen,
                                        lmda_path=factors * lm, early_exit=False, intercept=False,
                                        tol=ad_tol, progress_bar=False, n_threads=1)
                        B2 = np.zeros((K, meta['D']))
                        b2 = st2.betas.toarray()
                        B2[:b2.shape[0], cols] = b2
                        B2[b2.shape[0]:] = B2[b2.shape[0] - 1]
                        v2 = ((yv[:, r:r + 1] - xv @ B2.T) ** 2).mean(axis=0)
                        if best is None or v2.min() < best[0]:
                            best = (v2.min(), gam, B2)
                    gamma_of[r] = best[1]
                    for k in range(K):
                        level(r, k, best[2][k], None)
            else:
                Est = GroupLasso
                for r in range(q):
                    tracker = ValTracker(xv.shape[0], 1, K) if patience is not None else None
                    m = Est(groups=[int(s) for s in sizes], weights=np.sqrt(sizes), alpha=lam_max[r],
                            fit_intercept=False, warm_start=True, tol=tol, max_iter=1000,
                            max_epochs=100000)
                    for k, f in enumerate(factors):
                        m.set_params(alpha=float(f * lam_max[r]))
                        m.fit(xt, yt[:, r])
                        if level(r, k, m.coef_.copy(), tracker):
                            break
            res['fit_seconds'] = time.perf_counter() - t0
            stage('predict')
            xte = load(case, 'X', 'test', meta)
            t1 = time.perf_counter()
            pred = xte @ coef
            res['predict_seconds'] = time.perf_counter() - t1
            extra = {'best_k': best_k, 'lam_max': lam_max, 'stopped_at': stopped_at}
            if method == 'adaptive':
                extra['gamma_of'] = gamma_of
            if save_cp:
                extra['coef_path'] = coef_path
            if save_vp:
                extra['val_resid'] = vres
            if save_path:
                extra['pred_path'] = pred_path
                extra['active_path'] = act_path
        stage('serialize')
        np.savez_compressed(os.path.join(out, 'output.npz'), coef=coef, entry=entry, pred=pred,
                            **({'val_path': val_path} if val_path is not None else {}), **extra)
    res['status'] = 'ok'
except Exception as e:  # failures stay visible
    res['status'] = 'error'
    res['error'] = repr(e)
    with open(os.path.join(out, 'traceback.txt'), 'w', encoding='utf-8') as fh:
        fh.write(traceback.format_exc())
res['worker_seconds'] = time.perf_counter() - t_all
with open(os.path.join(out, 'result.json'), 'w', encoding='utf-8') as fh:
    json.dump(res, fh, indent=1)
stage('done')
print(json.dumps({k: v for k, v in res.items() if k not in ('chorus',)}), flush=True)
