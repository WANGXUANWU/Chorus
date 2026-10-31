"""Timing plan of the revision (amendment 16): configs/timing_plan2.json.

E1 same estimator, different execution (in memory, n = 2,000, p = 500, d = 5):
   group lasso path with validation selection: skglm, Celer, adelie, Chorus sequential
   (B = 1), Chorus batched (B = 64), Chorus batched with validation-aware termination;
   full two-stage estimator: sequential (B = 1, stage 2 rescans the design), batched without
   shared statistics, Chorus (batched, shared statistics).
E2 rows on disk (q = 64, M = 2 GiB enforced): Chorus at n = 25k to 400k.
E3 memory budget (n = 200k on disk; and a stress case with p = 2,000 that forces splits).
E4 batch size on disk (n = 50k).
E5 real data (sequential): Chorus and adelie on DepMap (64 responses) and Beijing (72).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda name: os.path.join(ROOT, 'results', 'cases', name)
GIB = 1 << 30
GL = {'stage2': False, 'crossfit': False, 'patience': None, 'relax': False}   # the group-lasso estimator only
plan = []
for q in (1, 4, 16, 64, 256):
    reps = 3 if q <= 16 else 1
    for m in ('skglm', 'Celer', 'adelie'):
        if m in ('skglm', 'Celer'):
            continue            # reused from the first timing study (same harness, same estimator)
        plan.append(dict(experiment='e1_gl', tag=f'q{q}', method=m, variant=m, size=q, q=q, n=2000,
                         case=C('eff_q256'), options={'resp_range': [0, q]}, reps=reps, cap_seconds=3600))
    for name, o in (('seq', {**GL, 'batch_size': 1}), ('batched', {**GL, 'batch_size': 64}),
                    ('batched_stop', {**GL, 'batch_size': 64, 'patience': 5})):
        plan.append(dict(experiment='e1_gl', tag=f'q{q}', method='Chorus', variant=f'chorus_{name}', size=q, q=q,
                         n=2000, case=C('eff_q256'), options={'resp_range': [0, q], **o}, reps=reps, cap_seconds=7200))
    for name, o in (('seq', {'batch_size': 1, 'stage2_shared': False}),
                    ('batched', {'batch_size': 64, 'stage2_shared': False}),
                    ('shared', {'batch_size': 64})):
        plan.append(dict(experiment='e1_full', tag=f'q{q}', method='Chorus', variant=f'full_{name}', size=q, q=q,
                         n=2000, case=C('eff_q256'), options={'resp_range': [0, q], **o}, reps=reps,
                         cap_seconds=7200))
for Bq in (128, 256):
    plan.append(dict(experiment='e1_bcap', tag=f'q256_B{Bq}', method='Chorus', variant=f'full_shared_B{Bq}', size=Bq,
                     q=256, n=2000, case=C('eff_q256'), options={'resp_range': [0, 256], 'batch_size': Bq}, reps=1,
                     cap_seconds=7200))
for n in (25000, 50000, 100000, 200000, 400000):
    plan.append(dict(experiment='e2_rows', tag=f'n{n}', method='Chorus', variant='full', size=n, q=64, n=n,
                     M=2 * GIB, case=C(f'disk_n{n}'), options={'disk': True, 'budget': 2 * GIB}, reps=1,
                     budget_gb=2.0, cap_seconds=3 * 3600))
for n in (25000, 50000, 100000, 200000, 400000):
    plan.append(dict(experiment='e2_rows', tag=f'n{n}', method='adelie', variant='adelie', size=n, q=64, n=n,
                     M=2 * GIB, case=C(f'disk_n{n}'), options={}, reps=1, budget_gb=2.0, cap_seconds=3600))
for M in (0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
    plan.append(dict(experiment='e3_budget', tag=f'M{M}', method='Chorus', variant='full', size=M, q=64, n=200000,
                     M=M * GIB, case=C('disk_n200000'), options={'disk': True, 'budget': int(M * GIB)}, reps=1,
                     budget_gb=M, cap_seconds=3 * 3600))
    plan.append(dict(experiment='e3_stress', tag=f'S{M}', method='Chorus', variant='full', size=M, q=64, n=25000,
                     M=M * GIB, case=C('disk_stress'), options={'disk': True, 'budget': int(M * GIB)}, reps=1,
                     budget_gb=M, cap_seconds=3 * 3600))
for B in (1, 4, 16, 64):
    plan.append(dict(experiment='e4_batch', tag=f'B{B}', method='Chorus', variant='full', size=B, q=64, n=50000, B=B,
                     case=C('disk_n50000'), options={'disk': True, 'batch_size': B, 'budget': 2 * GIB}, reps=1,
                     budget_gb=2.0, cap_seconds=3 * 3600))
# same estimator on the real workloads (DepMap: first 64 dependencies; Beijing: all 72 responses)
for tag, case, rr, extra in (('depmap64', 'depmap_c200', [0, 64], {}), ('beijing', 'beijing', None, {'val_cluster': 'week'})):
    base = {'resp_range': rr} if rr else {}
    plan.append(dict(experiment='e1_gl', tag=tag, method='adelie', variant='adelie', size=0, q=64 if rr else 72,
                     case=C(case), options=base, reps=1, cap_seconds=7200))
    for name, o in (('seq', {**GL, 'batch_size': 1}), ('batched', {**GL, 'batch_size': 64}),
                    ('batched_stop', {**GL, 'batch_size': 64, 'patience': 5})):
        plan.append(dict(experiment='e1_gl', tag=tag, method='Chorus', variant=f'chorus_{name}', size=0,
                         q=64 if rr else 72, case=C(case), options={**base, **o}, reps=1, cap_seconds=7200))
    for name, o in (('seq', {'batch_size': 1, 'stage2_shared': False}), ('batched', {'batch_size': 64, 'stage2_shared': False}),
                    ('shared', {'batch_size': 64})):
        plan.append(dict(experiment='e1_full', tag=tag, method='Chorus', variant=f'full_{name}', size=0,
                         q=64 if rr else 72, case=C(case), options={**base, **extra, **o}, reps=1, cap_seconds=3 * 3600))
for m, o in (('Chorus', {}), ('adelie', {})):
    plan.append(dict(experiment='e5_real', tag='depmap64', method=m, variant=m, size=0, q=64, case=C('depmap_c200'),
                     options={'resp_range': [0, 64], **o}, reps=1, cap_seconds=7200))
    plan.append(dict(experiment='e5_real', tag='beijing', method=m, variant=m, size=0, q=72, case=C('beijing'),
                     options={**o, **({'val_cluster': 'week'} if m == 'Chorus' else {})}, reps=1, cap_seconds=7200))
for it in plan:
    it['tag'] = it['tag'] + '_' + it['variant']
json.dump(plan, open(os.path.join(ROOT, 'configs', 'timing_plan2.json'), 'w'), indent=1)
print(len(plan), 'items')
