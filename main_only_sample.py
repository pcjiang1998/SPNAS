import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from scipy.stats import kendalltau

from arch.nasbench101 import NASBench101Graph
from cal_hash_for_list import cal_hash
from config import lr, weight_decay, num_epochs, batch_size
from problem import build_benchmark
from ranker.loss import ScoreLoss, DistanceLoss
from surrgoate_model import MLP
from utils.data_util import save_obj
from utils.io_util import folder_create
from utils.log_util import Logger
from utils.seed import set_seed


def try_generate(benchmark):
    arch = benchmark.search_space.sample(1)
    X = benchmark.search_space.encode(arch)
    return X


def get_data(total_num, dataset):
    benchmark = build_benchmark(dataset)
    data = {}
    while len(data) < total_num:
        X = try_generate(benchmark)
        if benchmark.evaluate(X)[0, 0] == 1:
            continue
        if dataset == 'NB101':
            arch = benchmark.search_space.decode(X)[0]
            graph = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])
            matrix, ops = NASBench101Graph.fill_graph(matrix=graph.matrix, ops=graph.ops, fill_nodes=7)
            x = benchmark.search_space.encode([{'matrix': matrix, 'ops': ops}])
            err_val = benchmark.evaluate(x, true_eval=False)[0, 0]
            err_test = benchmark.evaluate(x, true_eval=True)[0, 0]
            # 消除随机的影响
            if len(graph.ops) != len(ops):
                x[0, len(graph.ops) - len(ops):] = 3
            ############
            eye = np.zeros([4, 3], dtype=int)
            eye[:3, :] += np.eye(3, dtype=int)
            x_onehot = np.concatenate([x[:, :21], eye[x[:, 21:]].reshape([x.shape[0], -1])], axis=1)
            unique_hash = graph.fingerprint
            data[unique_hash] = (x[0], x_onehot[0], err_val, err_test)
            continue  # 跳过查询
        elif dataset in ['NB201', 'NB201-Cifar100', 'NB201-ImageNet16-120']:
            arch = benchmark.search_space.decode(X)
            x = benchmark.search_space.encode(arch)
            x_onehot = np.eye(5)[x].reshape([x.shape[0], -1])
            unique_hash = cal_hash(x[0])
        elif dataset in ['NATS', 'NATS-Cifar100', 'NATS-ImageNet16-120']:
            arch = benchmark.search_space.decode(X)
            x = benchmark.search_space.encode(arch)
            x_onehot = np.eye(8)[x].reshape([x.shape[0], -1])
            unique_hash = cal_hash(x[0])
        elif dataset == 'DARTS':
            length = 32
            arch = benchmark.search_space.decode(X)
            x = benchmark.search_space.encode(arch)
            x_onehot = np.concatenate([(np.eye(benchmark.search_space.ub[i] + 1)[x[:, i]]) for i in range(length)],
                                      axis=1)
            unique_hash = cal_hash(x[0])
        elif dataset == 'MNV3':
            length = 21
            arch = benchmark.search_space.decode(X)
            x = benchmark.search_space.encode(arch)
            x_onehot = np.concatenate([(np.eye(benchmark.search_space.ub[i] + 1)[x[:, i]]) for i in range(length)],
                                      axis=1)
            unique_hash = cal_hash(x[0])
        elif dataset == 'ResNet50':
            length = 25
            arch = benchmark.search_space.decode(X)
            x = benchmark.search_space.encode(arch)
            x_onehot = np.concatenate([(np.eye(benchmark.search_space.ub[i] + 1)[x[:, i]]) for i in range(length)],
                                      axis=1)
            unique_hash = cal_hash(x[0])
        elif dataset == 'Transformer':
            length = 34
            arch = benchmark.search_space.decode(X)
            x = benchmark.search_space.encode(arch)
            x_onehot = np.concatenate([(np.eye(benchmark.search_space.ub[i] + 1)[x[:, i]]) for i in range(length)],
                                      axis=1)
            unique_hash = cal_hash(x[0])
        else:
            raise Exception()
        err_val = benchmark.evaluate(x, true_eval=False)[0, 0]
        err_test = benchmark.evaluate(x, true_eval=True)[0, 0]
        data[unique_hash] = (x[0], x_onehot[0], err_val, err_test)
    return data, benchmark


class SampledDataset(torch.utils.data.Dataset):
    def __init__(self, data, benchmark, true_mode=False):
        self.data = data
        self.benchmark = benchmark
        self.keys = list(data.keys())
        np.random.shuffle(self.keys)
        self.true_mode = true_mode

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[self.keys[index]][1]).float()
        # y = torch.from_numpy(self.benchmark.evaluate([self.data[self.keys[index]][0]], true_eval=False))[
        #     0, 0].float()
        # y_true = torch.from_numpy(self.benchmark.evaluate([self.data[self.keys[index]][0]], true_eval=True))[
        #     0, 0].float()
        y, y_true = self.data[self.keys[index]][2], self.data[self.keys[index]][3]
        if self.true_mode:
            return x, y, y_true
        else:
            return x, y

    def __len__(self):
        return len(self.keys)


def train(device, model, num_epoch, dataloaders, need_test):
    verbose_epoch = list(range(0, num_epoch, max(1, num_epoch // 10))) + [num_epoch - 1]

    loss_func = nn.MSELoss().to(device)
    rank_loss = ScoreLoss().to(device)
    triple_loss = DistanceLoss().to(device)
    loss_weight = model.loss_params
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    if 'val' in dataloaders.keys() and need_test:
        with torch.no_grad():
            model.eval()
            model = model.to(device)
            labels, predicts = [], []
            for data, label in dataloaders['val']:
                data = data.to(device)
                predict = model(data).cpu().numpy().tolist()
                predicts.extend(predict)
                labels.extend(label.cpu().numpy().tolist())
            mse_weight, rank_weight, triple_weight = loss_weight.get_weight()
            rank_predict, rank = np.argsort(np.argsort(predicts)), np.argsort(np.argsort(labels))
            Logger.info(f'Before Start: '
                        f'mse={mse_weight:.2f}, '
                        f'rank={rank_weight:.2f}, '
                        f'triple={triple_weight:.2f}, '
                        f'KTau={kendalltau(rank_predict, rank).correlation}')
    for epoch in range(num_epoch):
        model.train()
        for data, label in dataloaders['train']:
            data, label = data.to(device), label.to(device)
            predict = model(data)

            optimizer.zero_grad()

            loss = 0
            loss += rank_loss(predict, label)
            loss += triple_loss(predict, label) * torch.sigmoid(loss_weight.params[2])
            loss.backward()

            optimizer.step()

        if 'val' in dataloaders.keys() and need_test and epoch in verbose_epoch:
            with torch.no_grad():
                model.eval()
                model = model.to(device)
                labels, predicts = [], []
                for data, label in dataloaders['val']:
                    data = data.to(device)
                    predict = model(data).cpu().numpy().tolist()
                    predicts.extend(predict)
                    labels.extend(label.cpu().numpy().tolist())
                mse_weight, rank_weight, triple_weight = loss_weight.get_weight()
                rank_predict, rank = np.argsort(np.argsort(predicts)), np.argsort(np.argsort(labels))
                Logger.info(f'Epoch {epoch + 1}: '
                            f'mse={mse_weight:.2f}, '
                            f'rank={rank_weight:.2f}, '
                            f'triple={triple_weight:.2f}, '
                            f'KTau={kendalltau(rank_predict, rank).correlation}')
        else:
            mse_weight, rank_weight, triple_weight = loss_weight.get_weight()
            Logger.info(f'Epoch {epoch + 1}: '
                        f'mse={mse_weight:.2f}, '
                        f'rank={rank_weight:.2f}, '
                        f'triple={triple_weight:.2f}')
    return model


def test(model, dataloaders, device):
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        val_labels, test_labels, predicts = [], [], []
        for data, val_label, test_label in dataloaders['test']:
            data = data.to(device)
            predict = model(data).cpu().numpy().tolist()
            predicts.extend(predict)
            val_labels.extend(val_label.cpu().numpy().tolist())
            test_labels.extend(test_label.cpu().numpy().tolist())
    return np.array(val_labels), np.array(test_labels), np.array(predicts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_num', type=int, default=424, help='total number of training data')
    parser.add_argument('--test_num', type=float, default=5000, help='total number of test data')
    parser.add_argument('--count', type=int, default=100, help='number of architecture tested')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['NB101', 'NB201', 'NB201-Cifar100', 'NB201-ImageNet16-120',
                                 'NATS', 'NATS-Cifar100', 'NATS-ImageNet16-120',
                                 'DARTS', 'MNV3', 'ResNet50', 'Transformer'], default='NB201')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-log', '--log', type=str, default='EXP')
    parser.add_argument('-c', '--comment', type=str, default='Onle_sample_ablation')
    args = parser.parse_args()
    if args.comment:
        Logger(f'{args.comment}-{args.dataset}-{args.seed}-{args.train_num}+{args.test_num}-',
               folder=folder_create(args.log))
    else:
        Logger(f'{args.dataset}-{args.seed}-{args.train_num}+{args.test_num}-',
               folder=folder_create(args.log))
    set_seed(args.seed)
    Logger.info(f'seed:{args.seed}, random number: {torch.rand(1).numpy().tolist()}')
    ########################################

    train_num = args.train_num
    test_num = args.test_num
    data, benchmark = get_data(train_num + test_num, args.dataset)
    keys = list(data.keys())
    np.random.shuffle(keys)
    train_data = {k: data[k] for k in keys[:train_num]}
    test_data = {k: data[k] for k in keys[train_num:]}
    save_obj({'train_data': train_data, 'test_data': test_data}, os.path.join(Logger.get_folder(), 'data.pkl'))
    train_set = SampledDataset(train_data, benchmark, False)
    val_set = SampledDataset(test_data, benchmark, False)
    test_set = SampledDataset(test_data, benchmark, True)
    Logger.info(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    model = MLP(len(train_set[0][0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(device, model, num_epochs, dataloaders, True)

    val_labels, test_labels, pred_labels = test(model, dataloaders, device)
    torch.save(model.cpu(), os.path.join(Logger.get_folder(), f'model-{args.dataset}.pth'))

    s = np.argsort(np.argsort(pred_labels))

    Logger.info(f'Val search model: {val_labels[s.argmin()]}')
    Logger.info(f'Val KTau = {kendalltau(pred_labels, val_labels).correlation}')

    Logger.info(f'Test search model: {test_labels[s.argmin()]}')
    Logger.info(f'Test KTau = {kendalltau(pred_labels, test_labels).correlation}')


if __name__ == "__main__":
    main()
