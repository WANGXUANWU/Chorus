"""Timing plans of amendment 21.

  sweep   E2  precision sweep (configs/timing_plan3_sweep.json)
  ooc     E5-E7  tall and wide out of core, budgets (configs/timing_plan3_ooc.json)
  par     E9  parallel execution (configs/parallel_plan3.json)
  decomp  E3  full path and exit at matched precision (configs/timing_plan3_decomp.json);
          needs results/v2/sweep_certificates.csv for the matched tolerances
Usage: make_timing_plan3.py sweep|ooc|par|decomp
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, 'results', 'cases')
GL_ONLY = {'stage2': False, 'crossfit': False, 'relax': False}
SIM = os.path.join(CASES, 'eff_q256')
WORKLOADS = {                       # name: (case, options, q, n)
    'q1': (SIM, {'resp_range': [0, 1]}, 1, 2000),
    'q4': (SIM, {'resp_range': [0, 4]}, 4, 2000),
    'q16': (SIM, {'resp_range': [0, 16]}, 16, 2000),
    'q64': (SIM, {'resp_range': [0, 64]}, 64, 2000),
    'q256': (SIM, {'resp_range': [0, 256]}, 256, 2000),
    'beijing': (os.path.join(CASES, 'beijing'), {}, 72, 6688),
    'depmap64': (os.path.join(CASES, 'depmap_c200'), {'resp_range': [0, 64]}, 64, 663),
}


def item(exp, tag, method, variant, wl, opts, reps, cap, **extra):
    case, wopts, q, n = WORKLOADS[wl]
    return {'experiment': exp, 'tag': tag, 'method': method, 'variant': variant, 'size': q, 'q': q, 'n': n,
            'workload': wl, 'case': case, 'options': {**wopts, **opts}, 'reps': reps, 'cap_seconds': cap, **extra}


def sweep():
    plan = []
    for wl in ('q16', 'q64', 'beijing', 'depmap64'):
        sim = wl.startswith('q')
        reps, cap = (3, 3600) if wl != 'depmap64' else (1, 7200)
        for tol in ((1e-4, 1e-6, 1e-8, 1e-10) if sim else (1e-4, 1e-6, 1e-8)):
            for m in ('skglm', 'Celer'):
                plan.append(item('e6_tol', f'{wl}_{m}_tol{tol:g}', m, f'{m}_tol', wl,
                                 {'tol': tol, 'save_coef_path': True}, reps, cap, tol=tol))
        for tol in (1e-4, 1e-7, 1e-9, 1e-11, 1e-13):
            plan.append(item('e6_tol', f'{wl}_adelie_tol{tol:g}', 'adelie', 'adelie_tol', wl,
                             {'adelie_tol': tol, 'save_coef_path': True}, reps, cap, tol=tol))
        for tol in (1e-4, 1e-6, 1e-8, 1e-10):
            plan.append(item('e6_tol', f'{wl}_chorus_tol{tol:g}', 'Chorus', 'chorus_tol', wl,
                             {**GL_ONLY, 'patience': None, 'batch_size': 64, 'tol': tol, 'save_coef_path': True},
                             reps, cap, tol=tol))
    return plan


def decomp(tstar):
    """tstar: {(workload, method): tolerance option} from the sweep certificates."""
    plan = []
    for wl in WORKLOADS:
        reps = 3
        cap = 7200
        for m, key in (('skglm', 'tol'), ('Celer', 'tol'), ('adelie', 'adelie_tol')):
            t = tstar[(wl, m)]
            for pat, v in ((None, 'full'), (5, 'exit')):
                plan.append(item('e7_decomp', f'{wl}_{m}_{v}', m, f'{m}_{v}', wl, {key: t, 'patience': pat},
                                 reps, cap, tol=t))
        for v, o in (('chorus_seq_ws', {'batch_size': 1, 'patience': None, 'strategy': 'working_set'}),
                     ('chorus_batched_ws', {'batch_size': 64, 'patience': None, 'strategy': 'working_set'}),
                     ('chorus_exit_ws', {'batch_size': 64, 'patience': 5, 'strategy': 'working_set'}),
                     ('chorus_batched', {'batch_size': 64, 'patience': None}),
                     ('chorus_exit', {'batch_size': 64, 'patience': 5})):
            plan.append(item('e7_decomp', f'{wl}_{v}', 'Chorus', v, wl, {**GL_ONLY, **o}, reps, cap))
    return plan


def ooc():
    plan = []
    M2 = 2 << 30
    tall = [25000, 50000, 100000, 200000, 400000]
    wide = [25000, 50000, 100000]
    for kind, sizes in (('tall', tall), ('wide', wide)):
        for n in sizes:
            case = os.path.join(CASES, f'disk_n{n}' if kind == 'tall' else f'wide_n{n}')
            base = {'experiment': f'e8_{kind}', 'size': n, 'q': 64, 'n': n, 'M': M2, 'case': case,
                    'reps': 1, 'budget_gb': 2.0}
            # the working-set engine on tall designs is the frozen code of the e2_rows runs (reused)
            plan.append({**base, 'tag': f'{kind}{n}_chorus', 'method': 'Chorus', 'variant': 'chorus',
                         'options': {'disk': True, 'budget': M2}, 'cap_seconds': 10800})
            plan.append({**base, 'tag': f'{kind}{n}_fullgram', 'method': 'Chorus', 'variant': 'full_gram',
                         'options': {**GL_ONLY, 'disk': True, 'strategy': 'stats', 'patience': None,
                                     'stats_max_gram_bytes': 1 << 40}, 'cap_seconds': 10800})
            plan.append({**base, 'tag': f'{kind}{n}_fixedscan', 'method': 'Chorus', 'variant': 'fixed_scan',
                         'options': {**GL_ONLY, 'disk': True, 'strategy': 'working_set', 'mode': 'gram',
                                     'batch_size': 64, 'patience': None}, 'cap_seconds': 10800})
            if kind == 'wide':
                for m in ('adelie', 'skglm', 'Celer', 'grpreg', 'sparsegl', 'SAM', 'biglasso'):
                    plan.append({**base, 'tag': f'{kind}{n}_{m}', 'method': m, 'variant': m, 'options': {},
                                 'cap_seconds': 3600})
    budget_cases = (('tall', 'disk_n200000', (0.5, 0.75, 1.0, 1.5, 2.0, 4.0)),
                    ('wide', 'wide_n50000', (1.0, 1.5, 2.0, 3.0, 4.0)),
                    ('stress', 'disk_stress', (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)))
    for kind, cname, Ms in budget_cases:
        for M in Ms:
            Mb = int(M * (1 << 30))
            plan.append({'experiment': f'e9_budget_{kind}', 'tag': f'budget_{kind}_M{M:g}', 'method': 'Chorus',
                         'variant': 'chorus', 'size': M, 'q': 64, 'M': Mb, 'case': os.path.join(CASES, cname),
                         'options': {'disk': True, 'budget': Mb}, 'reps': 1, 'budget_gb': M, 'cap_seconds': 10800})
    return plan


def par():
    plan = []
    for wl in ('q64', 'q256', 'beijing', 'depmap64'):
        for m, o in (('adelie', {}), ('skglm', {}), ('Chorus', {**GL_ONLY, 'patience': None, 'batch_size': 64})):
            plan.append(item('e10_par', f'{wl}_{m}_par6', m, f'{m}_par6', wl, o, 3, 7200, workers=6))
    return plan


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'decomp':
        import pandas as pd
        cert = pd.read_csv(os.path.join(ROOT, 'results', 'v2', 'sweep_certificates.csv'))
        tstar = {}
        for (wl, m), g in cert.groupby(['workload', 'method']):
            ok = g[g.max_gap <= 1e-4]
            tstar[(wl, m)] = float(ok.sort_values('median_seconds').iloc[0].tol) if len(ok) else float(g.tol.min())
        for wl in ('q1', 'q4', 'q256'):         # unswept sizes use the stricter of q16 and q64
            for m in ('skglm', 'Celer', 'adelie'):
                tstar[(wl, m)] = min(tstar[('q16', m)], tstar[('q64', m)])
        plan = decomp(tstar)
        json.dump({f'{k[0]}|{k[1]}': v for k, v in tstar.items()},
                  open(os.path.join(ROOT, 'configs', 'tstar_a21.json'), 'w'), indent=1)
    else:
        plan = {'sweep': sweep, 'ooc': ooc, 'par': par}[which]()
    out = os.path.join(ROOT, 'configs', {'sweep': 'timing_plan3_sweep.json', 'ooc': 'timing_plan3_ooc.json',
                                         'par': 'parallel_plan3.json', 'decomp': 'timing_plan3_decomp.json'}[which])
    json.dump(plan, open(out, 'w'), indent=1)
    print(which, len(plan), 'items ->', out)
