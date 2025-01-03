import copy

import numpy as np
import torch

from arch.nasbench101 import NASBench101Graph
from config import pop_size, callback
from problem import build_benchmark
from recorder import Recorder
from surrgoate_model import MLP
from train import train

name = 'NB101'
benchmark = build_benchmark(name)


def inner_evaluator(x):
    """评估单个X，可能需要处理是否已经不需要评估，处理重复"""
    if callback.count >= pop_size:
        # print('Skip')
        return 1
    else:
        x = x[np.newaxis, :].round().astype(int)
        arch = benchmark.search_space.decode(x)[0]
        try:
            graph = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])
        except TypeError:
            return 1
        if graph.fingerprint in callback.current:
            return 1
        elif graph.fingerprint in callback.history.keys():
            # print('Exists')
            callback.current[graph.fingerprint] = callback.history[graph.fingerprint]
            return callback.history[graph.fingerprint].data
        else:
            fitness = benchmark.evaluate(X=x, true_eval=False)
            if fitness[0, 0] != 1:
                # print('Right')
                callback.count += 1
                matrix, ops = NASBench101Graph.fill_graph(matrix=graph.matrix, ops=graph.ops, fill_nodes=7)
                encoding = benchmark.search_space.encode([{'matrix': matrix, 'ops': ops}])[0]
                recorder = Recorder(
                    fingerprint=graph.fingerprint,
                    original_encode=encoding,
                    data=fitness[0, 0]
                )
                callback.current[graph.fingerprint] = recorder
                callback.history[graph.fingerprint] = recorder
            else:
                # print('Error')
                pass
        return fitness[0, 0]


def surrogate(x, model, need_train, true_label=None):
    """包括预处理，训练，预测"""
    x = x.round().astype(int)
    # x_bak = copy.deepcopy(x)
    flag = np.ones(x.shape[0], dtype=bool)
    for i in range(x.shape[0]):
        arch = benchmark.search_space.decode(x[[i]])[0]
        try:
            graph = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])
            if not graph.is_valid():
                flag[i] = False
        except TypeError:
            flag[i] = False
            continue
        if len(graph.ops) < 7:
            matrix, ops = NASBench101Graph.fill_graph(matrix=graph.matrix, ops=graph.ops, fill_nodes=7)
        else:
            matrix, ops = graph.matrix, graph.ops
        encoding = benchmark.search_space.encode([{'matrix': matrix, 'ops': ops}])[0]
        if flag[i] and benchmark.evaluate([encoding])[0, 0] == 1:
            flag[i] = False
        x[i] = encoding
    if true_label is None:
        true_label = benchmark.evaluate(x)[:, 0].reshape(-1)

    x = np.concatenate([x[:, :21], np.eye(3, dtype=int)[x[:, 21:]].reshape([x.shape[0], -1])], axis=1)

    x_data = torch.from_numpy(x.astype(np.float32))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoch = 500
    # train
    if need_train:
        if model is None:
            model = MLP(x_data.shape[1])
        history_data = [
            (
                np.concatenate([
                    item.original_encode[:21],
                    np.eye(3, dtype=int)[item.original_encode[21:]].reshape([-1])], axis=0),
                item.data.item()
            )
            for item in callback.history.values()]
        model = train(device, model, num_epoch, true_label, x_data, history_data)

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        predict = model(x_data[flag.tolist()].to(device))
        result = np.ones([x_data.shape[0]])
        result[flag.tolist()] = predict.view(-1).cpu().numpy()
        return result, model
