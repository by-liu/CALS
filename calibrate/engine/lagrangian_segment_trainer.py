from typing import Dict
import numpy as np
import os.path as osp
from shutil import copyfile
import time
import json
import logging
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
from terminaltables.ascii_table import AsciiTable
from timm.utils import dispatch_clip_grad


from .segment_trainer import SegmentTrainer


class LagrangianSegmentTrainer(SegmentTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.init_lagrangian()

    def init_lagrangian(self) -> None:
        self.lagrangian = instantiate(self.cfg.lag.object)

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # forward
            outputs = self.model(inputs)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
                # For compounding loss, make sure the first term is the overall loss
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            if self.cfg.train.get("clip_grad") is not None:
                dispatch_clip_grad(
                    self.model.parameters(),
                    value=self.cfg.train.get("clip_grad", 2.0),
                    mode=self.cfg.train.get("clip_mode", "Norm")
                )
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            self.segment_evaluator.update(
                to_numpy(pred_labels), to_numpy(labels)
            )
            # self.logits_evaluator.update(to_numpy(outputs), to_numpy(labels))
            self.lr_scheduler.step_update(epoch * max_iter + i)
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_train_epoch_info(epoch)

    @torch.no_grad()