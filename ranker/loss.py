import torch
from torch import nn


class ScoreLoss(nn.Module):
    """modified from ReNAS: https://github.com/huawei-noah/Efficient-Computing.git"""

    def __init__(self, reduce='mean'):
        super(ScoreLoss, self).__init__()
        assert reduce in ['mean', 'sum', 'none'], "reduce should be 'mean','sum' or 'none'."
        self.reduce = reduce

    def forward(self, outputs, labels, reduce=None):
        if reduce is None:
            reduce = self.reduce
        else:
            assert reduce in ['mean', 'sum', 'none'], "reduce should be 'mean','sum' or 'none'."
        output = outputs.unsqueeze(1) if len(outputs.shape) == 1 else outputs
        output1 = output.repeat(1, outputs.shape[0])
        label = labels.unsqueeze(1) if len(labels.shape) == 1 else labels
        label1 = label.repeat(1, labels.shape[0])
        tmp = (output1 - output1.t()) * torch.sign(label1 - label1.t())
        tmp = torch.log(1 + torch.exp(-tmp))
        if reduce == 'none':
            mask = torch.triu(torch.ones_like(tmp), diagonal=1) == 1
            loss = tmp[mask]
        else:
            eye_tmp = tmp * torch.eye(len(tmp), device=outputs.device)
            new_tmp = tmp - eye_tmp
            if reduce == 'mean':
                loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))
            else:
                loss = torch.sum(new_tmp) / 2
        return loss


class DistanceLoss(nn.Module):
    def __init__(self, reduce='mean'):
        super(DistanceLoss, self).__init__()
        assert reduce in ['mean', 'sum', 'none'], "reduce should be 'mean','sum' or 'none'."
        self.reduce = reduce
        self.inner_loss = ScoreLoss(self.reduce)

    def forward(self, outputs, labels, reduce=None):
        if reduce is None:
            reduce = self.reduce
        else:
            assert reduce in ['mean', 'sum', 'none'], "reduce should be 'mean','sum' or 'none'."
        output = outputs.unsqueeze(1) if len(outputs.shape) == 1 else outputs
        output1 = output.repeat(1, outputs.shape[0])
        label = labels.unsqueeze(1) if len(labels.shape) == 1 else labels
        label1 = label.repeat(1, labels.shape[0])
        diff_output = output1 - output1.T
        diff_label = label1 - label1.T
        mask = torch.triu(torch.ones_like(output1), diagonal=1) == 1
        diff_output = torch.abs(diff_output[mask])
        diff_label = torch.abs(diff_label[mask])
        return self.inner_loss(diff_output, diff_label, reduce)
