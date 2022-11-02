import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from ..utils import reduce_tensor


class LogitMarginPlus(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 num_classes,
                 margin=10,
                 alpha=1.0,
                 ignore_index=-100,
                 gamma=1.1,
                 tao=1.1,
                 lambd_min: float = 1e-6,
                 lambd_max: float = 1e6,
                 step_size=100):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.tao = tao
        self.lambd_min = lambd_min
        self.lambd_max = lambd_max
        self.step_size = step_size

        # alpha for each class
        self.lambd = self.alpha * torch.ones(self.num_classes, requires_grad=False).cuda()
        self.prev_score, self.curr_score = (
            torch.zeros(self.num_classes, requires_grad=False).cuda(),
            torch.zeros(self.num_classes, requires_grad=False).cuda()
        )

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def reset_update_lambd(self):
        self.prev_score, self.curr_score = self.curr_score, torch.zeros(self.num_classes).cuda()
        self.count = 0

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

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

        diff = self.get_diff(inputs)
        # loss_margin = torch.clamp(diff - self.margin, min=0).mean()
        loss_margin = F.relu(diff-self.margin)
        loss_margin = torch.einsum("ik,k->ik", loss_margin, self.lambd).mean()

        # loss = loss_ce + self.alpha * loss_margin
        loss = loss_ce + loss_margin

        return loss, loss_ce, loss_margin

    def update_lambd(self, logits):
        diff = self.get_diff(logits)
        loss_margin = F.relu(diff-self.margin)
        loss_margin = torch.einsum("ik,k->ik", loss_margin, self.lambd).sum(dim=0)

        self.curr_score += loss_margin
        self.count += logits.shape[0]

    def set_lambd(self, epoch):
        self.curr_score = self.curr_score / self.count
        if dist.is_initialized():
            self.curr_score = reduce_tensor(self.curr_score, dist.get_world_size())
        if (epoch + 1) % self.step_size == 0 and self.prev_score.sum() != 0:
            self.lambd = torch.where(
                self.curr_score > self.prev_score * self.tao,
                self.lambd * self.gamma,
                self.lambd
            )
            self.lambd = torch.where(
                self.curr_score < self.prev_score / self.tao,
                self.lambd / self.gamma,
                self.lambd
            )
            self.lambd = torch.clamp(self.lambd, min=self.lambd_min, max=self.lambd_max).detach()

    def get_lambd_metric(self):
        return self.lambd.mean().item(), self.lambd.max().item()
