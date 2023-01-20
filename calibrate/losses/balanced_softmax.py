import torch
import torch.nn as nn
from torch.nn import functional as F


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, sample_per_class) -> None:
        super().__init__()
        self.sample_per_class = torch.tensor(sample_per_class)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()

        loss = self.cross_entropy(logits, labels)

        return loss
