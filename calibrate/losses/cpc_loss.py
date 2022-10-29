# implementation of CPC loss 
# Reference:
#   Calibrating Deep Neural Networks by Pairwise Constraints. CVPR 2022


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.torch_helper import one_hot
from ..utils.constants import EPS


class CPCLoss(nn.Module):
    def __init__(self, lambd_bdc=1.0, lambd_bec=1.0, ignore_index=-100):
        super().__init__()
        self.lambd_bdc = lambd_bdc
        self.lambd_bec = lambd_bec
        self.ignore_index = ignore_index

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_bdc", "loss_bec"

    def bdc(self, logits, targets_one_hot):
        # 1v1 Binary Discrimination Constraints (BDC)
        logits_y = logits[targets_one_hot == 1].view(logits.size(0), -1)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        loss_bdc = - F.logsigmoid(logits_y - logits_rest).sum() / (logits.size(1) - 1) / logits.size(0)

        return loss_bdc

    def bec(self, logits, targets_one_hot):
        # Binary Exclusion COnstraints (BEC)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        diff = logits_rest.unsqueeze(2) - logits_rest.unsqueeze(1)
        loss_bec = - torch.sum(
            0.5 * F.logsigmoid(diff + EPS)
            / (logits.size(1) - 1) / (logits.size(1) - 2) / logits.size(0)
        )

        return loss_bec

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)

        targets_one_hot = one_hot(targets, inputs.size(1))
        loss_bdc = self.bdc(inputs, targets_one_hot)
        loss_bec = self.bec(inputs, targets_one_hot)

        loss = loss_ce + self.lambd_bdc * loss_bdc + self.lambd_bec * loss_bec

        return loss, loss_ce, loss_bdc, loss_bec
