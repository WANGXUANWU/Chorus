"""Generate the wide out-of-core cases of amendment 21 (p = 5,000, D = 25,000, full Gram 5 GB)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from bench.make_disk_case import make_disk_case

for n in (25000, 50000, 100000):
    root = os.path.join(ROOT, 'results', 'cases', f'wide_n{n}')
    if os.path.exists(os.path.join(root, 'meta.json')):
        continue
    meta = make_disk_case(root, n_train=n, p=5000, q=64, s=6, kappa=.5, rho=.5, snr=2., seed=2201,
                          n_val=2000, n_test=2000, chunk=5000)
    print(n, meta['D'], meta['bytes_train_design'] / 2 ** 30, flush=True)
