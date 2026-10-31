"""Disk case that stresses the working set: p = 2,000 predictors (D = 10,000), no shared support,
24 active predictors per response, 64 responses, n = 25,000 training rows."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.make_disk_case import make_disk_case

root = os.path.join(ROOT, 'results', 'cases', 'disk_stress')
if not os.path.exists(os.path.join(root, 'meta.json')):
    meta = make_disk_case(root, n_train=25000, p=2000, q=64, s=24, kappa=0., rho=.5, snr=2., seed=2101,
                          n_val=5000, n_test=10000, chunk=5000)
    print('stress case', meta['D'], flush=True)
