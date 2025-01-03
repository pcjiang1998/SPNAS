import argparse

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.optimize import minimize

from config import pop_size, n_gen, callback, batch_size, lr, weight_decay, train_gen
from problem import EvoXBench
from utils.io_util import folder_create
from utils.log_util import Logger
from utils.seed import set_seed


def main():
    p = EvoXBench(benchmark, inner_evaluator=inner_evaluator, surrogate=surrogate)

    algorithm = GA(
        pop_size=pop_size * 10,
        n_offsprings=pop_size * 10,
        sampling=LatinHypercubeSampling(),
        crossover=SimulatedBinaryCrossover(),
        mutation=PolynomialMutation(),
        eliminate_duplicates=True)

    res = minimize(p, algorithm, ('n_gen', n_gen), verbose=False, callback=callback, save_history=True)
    best_x = res.X if len(res.X.shape) != 1 else res.X[np.newaxis, :]
    best_x = best_x.round().astype(int)
    Logger.info(f'{best_x=}')
    Logger.info(f'Val Err = {np.min(p.benchmark.evaluate(X=best_x, true_eval=False)[:, 0])}, '
                f'Test Err = {np.min(p.benchmark.evaluate(X=best_x, true_eval=True)[:, 0])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['NB101', 'NB201', 'NB201-Cifar100', 'NB201-ImageNet16-120',
                                 'NATS', 'NATS-Cifar100', 'NATS-ImageNet16-120',
                                 'DARTS', 'MNV3', 'ResNet50', 'Transformer'], default='NB101')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-log', '--log', type=str, default='EXP')
    parser.add_argument('-c', '--comment', type=str, default='w_MSE')
    args = parser.parse_args()

    if args.dataset == 'NB101':
        from core.core_nb101 import inner_evaluator, surrogate, benchmark
    elif args.dataset == 'NB201':
        from core.core_nb201 import inner_evaluator, surrogate, set_dataset

        name, benchmark = set_dataset(args.dataset)
    elif args.dataset == 'NB201-Cifar100':
        from core.core_nb201 import inner_evaluator, surrogate, set_dataset

        name, benchmark = set_dataset(args.dataset)
    elif args.dataset == 'NB201-ImageNet16-120':
        from core.core_nb201 import inner_evaluator, surrogate, set_dataset

        name, benchmark = set_dataset(args.dataset)
    elif args.dataset == 'NATS':
        from core.core_nats import inner_evaluator, surrogate, set_dataset

        name, benchmark = set_dataset(args.dataset)
    elif args.dataset == 'NATS-Cifar100':
        from core.core_nats import inner_evaluator, surrogate, set_dataset

        name, benchmark = set_dataset(args.dataset)
    elif args.dataset == 'NATS-ImageNet16-120':
        from core.core_nats import inner_evaluator, surrogate, set_dataset

        name, benchmark = set_dataset(args.dataset)
    elif args.dataset == 'DARTS':
        from core.core_darts import inner_evaluator, surrogate, benchmark
    elif args.dataset == 'MNV3':
        from core.core_mnv3 import inner_evaluator, surrogate, benchmark
    elif args.dataset == 'ResNet50':
        from core.core_resnet50 import inner_evaluator, surrogate, benchmark
    elif args.dataset == 'Transformer':
        from core.core_transformer import inner_evaluator, surrogate, benchmark
    else:
        raise Exception('Not supported dataset.')

    if args.comment:
        Logger(f'{args.comment}-{args.dataset}-{args.seed}-{pop_size}x{n_gen}({train_gen})-',
               folder=folder_create(args.log))
    else:
        Logger(f'{args.dataset}-{args.seed}-{pop_size}x{n_gen}({train_gen})-',
               folder=folder_create(args.log))

    Logger.info(f'Name: {args.dataset} start')
    Logger.info(f'Hyper-params: {pop_size=}, {n_gen=}, {batch_size=}, {lr=}, {weight_decay=}, {train_gen=}.')
    Logger.info(f'run: seed={args.seed}   ===========')

    set_seed(args.seed)
    main()
    callback.reset()
