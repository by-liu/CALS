import logging
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from torch import distributed as dist
import torch

from .penalty import get_penalty_func
from .aug_lagrangian import AugLagrangian
from ..utils import reduce_tensor

logger = logging.getLogger(__name__)


class AugLagrangianClass(AugLagrangian):
    def __init__(
        self,
        num_classes: int = 10,
        margin: float = 10,
        penalty: str = "p2",
        lambd_min: float = 1e-6,
        lambd_max: float = 1e6,
        lambd_step: int = 1,
        rho_min: float = 1,
        rho_max: float = 10,
        # rho_update: bool = False,
        rho_step: int = -1,
        gamma: float = 1.2,
        tao: float = 0.9,
        normalize: bool = True
    ):
        assert penalty in ("p2", "p3", "phr", "relu"), f"invalid penalty: {penalty}"
        self.num_classes = num_classes
        self.margin = margin
        self.penalty_func = get_penalty_func(penalty)
        self.lambd_min = lambd_min
        self.lambd_max = lambd_max
        self.lambd_step = lambd_step
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_update = rho_step > 0
        self.rho_step = rho_step
        self.gamma = gamma
        self.tao = tao
        self.normalize = normalize

        self.lambd = self.lambd_min * torch.ones(self.num_classes, requires_grad=False).cuda()
        self.rho = self.rho_min

        self.prev_penalty = None

    def get(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits.movedim(1, -1)  # move class dimension to last

        h = self.get_constraints(logits)
        p, _ = self.penalty_func(h, self.lambd, self.rho)
        penalty = p.sum(dim=-1).mean()  # sum over classes and average over samples (and possibly pixels)
        constraint = h.mean()
        return penalty, constraint

    def reset_update_lambd(self, epoch):
        if (epoch + 1) % self.lambd_step == 0:
            self.grad_p_sum = torch.zeros_like(self.lambd)
            self.sample_num = 0

    def update_lambd(self, logits, epoch):
        """update lamdb based on the gradeint on the logits
        """
        if (epoch + 1) % self.lambd_step == 0:
            logits = logits.movedim(1, -1)  # move class dimension to last

            h = self.get_constraints(logits)
            _, grad_p = self.penalty_func(h, self.lambd, self.rho)
            grad_p = torch.clamp(grad_p, min=self.lambd_min, max=self.lambd_max)
            grad_p = grad_p.flatten(start_dim=0, end_dim=-2)
            self.grad_p_sum += grad_p.sum(dim=0)
            self.sample_num += grad_p.shape[0]

    def set_lambd(self, epoch):
        if (epoch + 1) % self.lambd_step == 0:
            grad_p_mean = self.grad_p_sum / self.sample_num
            if dist.is_initialized():
                grad_p_mean = reduce_tensor(grad_p_mean, dist.get_world_size())
            self.lambd = torch.clamp(grad_p_mean, min=self.lambd_min, max=self.lambd_max).detach()

    def update_rho_by_val(self, val_penalty, epoch):
        if self.rho_update:
            if (epoch + 1) % self.rho_step == 0:
                if (
                    self.prev_penalty is not None
                    and self.prev_penalty > 0
                    and val_penalty > 0
                    and val_penalty > self.tao * self.prev_penalty
                ):
                    self.rho = min(self.rho_max, self.rho * self.gamma)
                    logger.info("Adjusting rho in AugLagrangian to {}".format(self.rho))
            self.prev_penalty = val_penalty

    def get_lambd_metric(self):
        lambd = self.lambd

        return lambd.mean().item(), lambd.max().item()
