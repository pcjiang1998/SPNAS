import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from torch.utils.data import TensorDataset, DataLoader

from ranker.loss import RankLoss, TripleLoss
from surrgoate_model import LossParameters
from utils.log_util import Logger
from config import batch_size, lr, weight_decay


def train(device, model, num_epoch, true_label, x_data, history_data):
    dataset = [torch.from_numpy(np.array(i).astype(np.float32)) for i in zip(*history_data)]
    dataset = TensorDataset(*dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = nn.MSELoss().to(device)
    rank_loss = RankLoss().to(device)
    triple_loss = TripleLoss().to(device)
    loss_weight = model.loss_params
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    if true_label is not None:
        with torch.no_grad():
            model.eval()
            model = model.to(device)
            mask = (true_label != 1).tolist()
            predict = model(x_data[mask].to(device))
            predict = predict.view(-1).cpu().numpy()
            mse_weight, rank_weight, triple_weight = loss_weight.get_weight()
    for epoch in range(num_epoch):
        model.train()
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            predict = model(data)

            optimizer.zero_grad()

            loss = 0
            loss += rank_loss(predict, label)
            loss += triple_loss(predict, label) * torch.sigmoid(loss_weight.params[2])
            loss.backward()
            optimizer.step()
        if true_label is not None:
            with torch.no_grad():
                model.eval()
                model = model.to(device)
                mask = (true_label != 1).tolist()
                predict = model(x_data[mask].to(device))
                predict = predict.view(-1).cpu().numpy()
                mse_weight, rank_weight, triple_weight = loss_weight.get_weight()
                Logger.info(f'Epoch {epoch + 1}: '
                            f'KTau={kendalltau(predict, true_label[mask]).correlation}')
    return model
