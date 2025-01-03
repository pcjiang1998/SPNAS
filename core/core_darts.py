import numpy as np
import torch

from cal_hash_for_list import cal_hash
from config import pop_size, callback
from problem import build_benchmark
from recorder import Recorder
from surrgoate_model import MLP
from train import train

name = 'DARTS'
benchmark = build_benchmark(name)

length = 32


def inner_evaluator(x):
    """评估单个X，可能需要处理是否已经不需要评估，处理重复"""
    if callback.count >= pop_size:
        # print('Skip')
        return 1
    else:
        x = x.round().astype(int)

        arch = benchmark.search_space.decode(x[np.newaxis, :])[0]
        tmp = benchmark.search_space.encode([arch])[0]
        fingerprint = cal_hash(tmp[np.newaxis, :])
        if fingerprint in callback.current:
            return 1
        elif fingerprint in callback.history.keys():
            # print('Exists')
            callback.current[fingerprint] = callback.history[fingerprint]
            return callback.history[fingerprint].data
        else:
            fitness = benchmark.evaluate(X=x[np.newaxis, :], true_eval=False)
            if fitness[0, 0] != 1:
                # print('Right')
                callback.count += 1
                recorder = Recorder(
                    fingerprint=fingerprint,
                    original_encode=tmp,
                    data=fitness[0, 0]
                )
                callback.current[fingerprint] = recorder
                callback.history[fingerprint] = recorder
            else:
                # print('Error')
                pass
        return fitness[0, 0]


def surrogate(x, model, need_train, true_label=None):
    """包括预处理，训练，预测"""
    x = x.round().astype(int)
    flag = np.ones(x.shape[0], dtype=bool)
    for i in range(x.shape[0]):
        encoding = benchmark.search_space.encode(benchmark.search_space.decode(x[[i]]))[0]
        if flag[i] and benchmark.evaluate([encoding])[0, 0] == 1:
            flag[i] = False
        x[i] = encoding
    if true_label is None:
        true_label = benchmark.evaluate(x)[:, 0].reshape(-1)

    x = np.concatenate([(np.eye(benchmark.search_space.ub[i] + 1)[x[:, i]]) for i in range(length)], axis=1)

    x_data = torch.from_numpy(x.astype(np.float32))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoch = 500
    # train
    if need_train:
        if model is None:
            model = MLP(x_data.shape[1])
        history_data = [
            (
                np.concatenate([(np.eye(benchmark.search_space.ub[i] + 1)[item.original_encode[i]])
                                for i in range(length)], axis=0),
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
