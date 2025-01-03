import torch
import torch.nn as nn
import torch.nn.functional as F


class LossParameters(nn.Module):
    def __init__(self):
        super(LossParameters, self).__init__()
        self.params = nn.Parameter(torch.zeros([3]))

    def mse_weight(self):
        return self.get_weight()[0]

    def rank_weight(self):
        return self.get_weight()[1]

    def triple_weight(self):
        return self.get_weight()[2]

    def get_weight(self):
        # with torch.no_grad():
        #     tmp = torch.softmax(self.params, dim=0).cpu().numpy().tolist()
        # return tmp
        return self.params


class MLP(nn.Module):
    def __init__(self, in_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 2 * in_features)
        self.fc2 = nn.Linear(2 * in_features, 2 * in_features)
        self.fc3 = nn.Linear(2 * in_features, in_features)
        self.fc4 = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(0.5)
        self.loss_params = LossParameters()

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        return out.view(-1)
