"""Generate the out-of-core scaling cases (training design on disk)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.make_disk_case import make_disk_case

for n in (25000, 50000, 100000, 200000, 400000):
    root = os.path.join(ROOT, 'results', 'cases', f'disk_n{n}')
    if os.path.exists(os.path.join(root, 'meta.json')):
        continue
    meta = make_disk_case(root, n_train=n, p=500, q=64, s=6, kappa=.5, rho=.5, snr=2., seed=2001,
                          n_val=5000, n_test=10000, chunk=20000)
    print(n, meta['D'], meta['bytes_train_design'] / 2 ** 30, flush=True)
