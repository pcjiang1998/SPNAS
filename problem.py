import numpy as np
from evoxbench.modules import Benchmark

from pymoo.core.problem import Problem
from scipy.stats import kendalltau

from config import train_gen, ignore_unknown_history
from utils.log_util import Logger


def build_benchmark(name):
    assert name in [
        'NB101', 'NB201', 'NATS', 'NB201-Cifar100', 'NATS-Cifar100', 'NB201-ImageNet16-120', 'NATS-ImageNet16-120',
        'DARTS', 'MNV3', 'ResNet50', 'Transformer'
    ]
    objs = 'err&flops'
    normalized_objectives = False
    if name == 'NB101':
        from evoxbench.benchmarks import NASBench101Benchmark
        return NASBench101Benchmark(objs=objs, normalized_objectives=normalized_objectives)
    elif name == 'NB201':
        from evoxbench.benchmarks import NASBench201Benchmark
        return NASBench201Benchmark(objs=objs, normalized_objectives=normalized_objectives)
    elif name == 'NB201-Cifar100':
        from evoxbench.benchmarks import NASBench201Benchmark
        return NASBench201Benchmark(objs=objs, dataset='cifar100', normalized_objectives=normalized_objectives)
    elif name == 'NB201-ImageNet16-120':
        from evoxbench.benchmarks import NASBench201Benchmark
        return NASBench201Benchmark(objs=objs, dataset='ImageNet16-120', normalized_objectives=normalized_objectives)
    elif name == 'NATS':
        from evoxbench.benchmarks import NATSBenchmark
        return NATSBenchmark(objs=objs, normalized_objectives=normalized_objectives)
    elif name == 'NATS-Cifar100':
        from evoxbench.benchmarks import NATSBenchmark
        return NATSBenchmark(objs=objs, dataset='cifar100', normalized_objectives=normalized_objectives)
    elif name == 'NATS-ImageNet16-120':
        from evoxbench.benchmarks import NATSBenchmark
        return NATSBenchmark(objs=objs, dataset='ImageNet16-120', normalized_objectives=normalized_objectives)
    elif name == 'DARTS':
        from evoxbench.benchmarks import DARTSBenchmark
        return DARTSBenchmark(objs=objs, normalized_objectives=normalized_objectives)
    elif name == 'MNV3':
        from evoxbench.benchmarks import MobileNetV3Benchmark
        return MobileNetV3Benchmark(objs=objs, normalized_objectives=normalized_objectives)
    elif name == 'ResNet50':
        from evoxbench.benchmarks import ResNet50DBenchmark
        return ResNet50DBenchmark(objs=objs, normalized_objectives=normalized_objectives)
    elif name == 'Transformer':
        from evoxbench.benchmarks import TransformerBenchmark
        return TransformerBenchmark(objs=objs, normalized_objectives=normalized_objectives)
    else:
        raise Exception()


class EvoXBench(Problem):
    """Only 'err' is considered."""

    def __init__(self, benchmark: str or Benchmark, inner_evaluator, surrogate, ignore_unknown=ignore_unknown_history):
        if isinstance(benchmark, str):
            self.benchmark: Benchmark = build_benchmark(benchmark)
        else:
            self.benchmark: Benchmark = benchmark
        self.xu = self.benchmark.search_space.ub
        self.xl = self.benchmark.search_space.lb
        self.inner_evaluator = inner_evaluator
        self.surrogate = surrogate
        self.model = None
        self.ignore_unknown = ignore_unknown
        super().__init__(n_var=self.benchmark.search_space.n_var, n_obj=1, xl=self.xl, xu=self.xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        algorithm = kwargs['algorithm']
        n_gen = algorithm.n_gen

        # Truly query
        result1 = np.empty([x.shape[0]])
        for i in range(x.shape[0]):
            result1[i] = self.inner_evaluator(x[i])

        # Surrogate
        if n_gen <= train_gen:
            # Train
            if self.ignore_unknown:
                _, self.model = self.surrogate(x, self.model, need_train=True, true_label=result1)
            else:
                _, self.model = self.surrogate(x, self.model, need_train=True, true_label=None)
            out["F"] = result1
        else:
            # Predict
            if self.ignore_unknown:
                result2, self.model = self.surrogate(x, self.model, need_train=False, true_label=result1)
            else:
                result2, self.model = self.surrogate(x, self.model, need_train=False, true_label=None)
            out["F"] = result2
            mask = result1 != 1
            ktau = kendalltau(result1[mask], result2[mask]).correlation
            min_index = np.argmin(result1)
            min_index_pred = np.argmin(result2)
            Logger.info(f'ktau={ktau}, '
                        f'Predict/Truly {result2[min_index_pred]}/{result1[min_index]}, '
                        f'cross find {result1[min_index_pred]}/{result2[min_index]}')
