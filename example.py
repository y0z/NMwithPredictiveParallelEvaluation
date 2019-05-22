from nelder_mead import NelderMead

import argparse
import json
import numpy as np
import os
from pyDOE import lhs


def benchmark(x):
    # sphere function
    return sum(x ** 2)


if __name__ == '__main__':
    '''
        Usage:
            python3 example.py --num_dim=2 --num_parallels 10 --num_montecarlo 100 \
                    --num_speculative_iter 3 --max_gp_samples 100
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_dim', type=int, required=True)
    parser.add_argument('--num_parallels', type=int, required=True)
    parser.add_argument('--num_montecarlo', type=int, required=True)
    parser.add_argument('--num_speculative_iter', type=int, required=True)
    parser.add_argument('--max_gp_samples', type=int, required=True)
    args = parser.parse_args()

    # set config
    num_dim = args.num_dim
    num_parallels = args.num_parallels
    num_montecarlo = args.num_montecarlo
    num_speculative_iter = args.num_speculative_iter
    max_gp_samples = args.max_gp_samples

    # generate initial simplex using Latin hypercube sampling
    initial_simplex = 10 * lhs(num_dim, samples=num_dim + 1) - 5

    # optimize
    nm = NelderMead(
        benchmark,
        speculative_exec=True,
        num_montecarlo=num_montecarlo,
        num_speculative_iter=num_speculative_iter,
        num_parallels=num_parallels,
        max_gp_samples=max_gp_samples)
    x, fx, k = nm.optimize(initial_simplex, min_diam=1e-4)  # the results of NM (x, fx, # of iters)
    steps = nm.f.count  # # of eval steps
    evals = len(nm.f.keys)  # # of evaluations

    # the result of NM method
    print('x:\t%s' % (x))
    print('fx:\t%s' % (fx))
    print('k:\t%s' % (k))
    print('steps:\t%s' % (steps))
    print('evals:\t%s' % (evals))

    # the best point from all evaluated points
    best_idx = np.argmin(nm.f.values)
    best_x = nm.f.keys[best_idx]
    best_fx = nm.f.values[best_idx]
    print('best x:\t%s' % (best_x))
    print('best fx:\t%s' % (best_fx))
