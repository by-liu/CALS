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
from ..losses import LogitMarginL1
from ..evaluation import AverageMeter
from ..utils import (
    load_train_checkpoint, save_train_checkpoint, round_dict,
    to_numpy, get_lr
)

logger = logging.getLogger(__name__)


class LagrangianSegmentTrainer(SegmentTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.init_lagrangian()

    def build_meter(self):
        super().build_meter()
        self.penalty_meter = AverageMeter()
        self.constraint_meter = AverageMeter()

    def reset_meter(self):
        super().reset_meter()
        self.penalty_meter.reset()
        self.constraint_meter.reset()

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
            penalty, constraint = self.lagrangian.get(outputs)
            # backward
            self.optimizer.zero_grad()
            (loss_total + penalty).backward()
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
            self.evaluator.update(
                to_numpy(pred_labels), to_numpy(labels)
            )
            self.penalty_meter.update(penalty.item())
            self.constraint_meter.update(constraint.item())

            self.lr_scheduler.step_update(epoch * max_iter + i)
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_train_epoch_info(epoch)

    def log_iter_info(self, iter, max_iter, epoch, phase="train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.avg
        log_dict["batch_time"] = self.batch_time_meter.avg
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())
        log_dict["penalty"] = self.penalty_meter.val
        log_dict["constraint"] = self.constraint_meter.val
        if phase == "train":
            log_dict["lr"] = get_lr(self.optimizer)
        logger.info("{} iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable and phase.lower() == "train":
            wandb_log_dict = {"iter": epoch * max_iter + iter}
            wandb_log_dict.update(dict(
                (f"{phase}/iter/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_train_epoch_info(self, epoch):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, LogitMarginL1):
            log_dict["alpha"] = self.loss_func.alpha
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        log_dict["penalty"] = self.penalty_meter.avg
        log_dict["constraint"] = self.constraint_meter.avg
        lambd_mean, lambd_max = self.lagrangian.get_lambd_metric()
        log_dict["lambd_mean"] = lambd_mean
        log_dict["lambd_max"] = lambd_max
        log_dict["rho_mean"], log_dict["lambd_max"] = self.lagrangian.get_rho_metric()
        logger.info("train epoch[{}]\t{}".format(
            epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                (f"train/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="val"):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)

        if phase == "val":
            self.lagrangian.reset_update_lambd(epoch)

        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # forward
            outputs = self.model(inputs)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
            loss = self.loss_func(outputs, labels)
            if phase == "val":
                self.lagrangian.update_lambd(outputs, epoch)
                penalty, constraint = self.lagrangian.get(outputs)
                self.penalty_meter.update(penalty.item())
                self.constraint_meter.update(constraint.item())
            # metric
            self.loss_meter.update(loss)
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                to_numpy(pred_labels), to_numpy(labels)
            )
            if phase.lower() == "test":
                self.calibrate_evaluator.update(
                    outputs, labels
                )
            # self.logits_evaluator(
            #     np.expand_dims(to_numpy(outputs), axis=0),
            #     np.expand_dims(to_numpy(labels), axis=0)
            # )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_eval_epoch_info(epoch, phase)
        if phase == "val":
            self.lagrangian.set_lambd(epoch)
            self.lagrangian.update_rho(epoch)

        return self.loss_meter.avg(0), self.evaluator.mean_score(main=True)
