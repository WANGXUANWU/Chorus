"""Run one method on one case (detached helper). Usage: run_one.py CASE METHOD OUT [JSON_OPTS]"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.runner import run
a4 = sys.argv[4] if len(sys.argv) > 4 else '{}'
opts = json.loads(open(a4[1:]).read() if a4.startswith('@') else a4)
r = run(sys.argv[1], sys.argv[2], sys.argv[3], opts, cap_seconds=6 * 3600)
print(r['run_status'], r.get('fit_seconds'), r.get('chorus', {}).get('config'), flush=True)
