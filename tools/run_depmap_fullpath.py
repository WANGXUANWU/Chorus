"""E4 of amendment 21: DepMap full path of the whole pipeline (patience None) with a 10-hour cap,
same strategy as the existing patience runs (working set), for per-response exit losses."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.runner import run

out = os.path.join(ROOT, 'results', 'v2', 'patience', 'depmap_c200', 'pNone_10h')
if not os.path.exists(os.path.join(out, 'run.json')):
    r = run(os.path.join(ROOT, 'results', 'cases', 'depmap_c200'), 'Chorus', out,
            {'bases': ['../depmap_c200_d3'], 'patience': None, 'strategy': 'working_set'}, cap_seconds=10 * 3600)
    print(json.dumps({'status': r['run_status'], 'fit_seconds': r.get('fit_seconds')}), flush=True)
