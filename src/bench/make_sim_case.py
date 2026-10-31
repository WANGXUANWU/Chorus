"""Materialize a simulated case directory (design, responses, splits and truth)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from chorus import SplineDesign
from data.simulate import simulate
from bench.caseio import write_case


def make_sim_case(root, n_lambda=30, lambda_min_ratio=0.02, **params):
    sim = simulate(**params)
    sl = sim['split']
    sd = SplineDesign().fit(sim['x'][sl['train']])
    ymean = sim['y'][sl['train']].mean(0)
    truth = {'support': sim['support'], 'mean_test_centered': sim['mean'][sl['test']] - ymean}
    meta = write_case(root, sd, sim['x'], sim['y'], sl, truth=truth,
                      extra={'n_lambda': n_lambda, 'lambda_min_ratio': lambda_min_ratio,
                             'kind': 'simulation', 'params': sim['params']})
    with open(os.path.join(root, 'spec.json'), 'w') as fh:
        json.dump(sim['spec'], fh)
    return meta


if __name__ == '__main__':
    root = sys.argv[1]
    params = json.loads(sys.argv[2])
    print(make_sim_case(root, **params)['D'])
