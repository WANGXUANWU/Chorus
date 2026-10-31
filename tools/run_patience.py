"""Early-termination study (reviewer comment 6): Chorus with patience 1, 2, 3, 5, 8 and the
full path, one fresh single-threaded process at a time, on the twenty formal settings of seed
1001 and on both real cases. Outputs go to results/v2/patience/<case>/p<patience>.
Usage: run_patience.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.runner import run

PATIENCE = [None, 1, 2, 3, 5, 8]
cases = []
acc = os.path.join(ROOT, 'results', 'formal', 'acc')
for name in sorted(os.listdir(acc)):
    if name.endswith('seed1001'):
        cases.append((name, os.path.join(acc, name, 'case'), {'bases': ['../case_d3']}))
cases.append(('depmap_c200', os.path.join(ROOT, 'results', 'cases', 'depmap_c200'), {'bases': ['../depmap_c200_d3']}))
cases.append(('beijing', os.path.join(ROOT, 'results', 'cases', 'beijing'),
              {'bases': ['../beijing_d3'], 'val_cluster': 'week'}))
for name, case, base_opts in cases:
    for pat in PATIENCE:
        out = os.path.join(ROOT, 'results', 'v2', 'patience', name, f'p{pat}')
        if os.path.exists(os.path.join(out, 'run.json')):
            continue
        r = run(case, 'Chorus', out, {**base_opts, 'patience': pat}, cap_seconds=4 * 3600)
        print(json.dumps({'case': name, 'patience': pat, 'status': r['run_status'],
                          'fit_seconds': r.get('fit_seconds')}), flush=True)
