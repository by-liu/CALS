import logging
import time
import wandb
from shutil import copyfile
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from contextlib import suppress
import json
import os
import os.path as osp
from terminaltables.ascii_table import AsciiTable

import torch
from torch import distributed as dist
import torch.nn.functional as F
from timm.utils import NativeScaler, dispatch_clip_grad

from ..utils import (
    reduce_tensor, gather, build_dist_data_loader,
    set_random_seed, to_numpy, get_lr, round_dict,
    save_train_checkpoint, load_train_checkpoint
)
from .optimizer import build_optimizer
from ..evaluation import (
    AverageMeter, LossMeter, accuracy, ClassificationEvaluator,
    CalibrateEvaluator, LogitsEvaluator
)
from ..losses import LogitMarginL1
from .distributed_trainer import DistributedTrainer

logger = logging.getLogger(__name__)


class DistributedTrainerPlus(DistributedTrainer):
    def log_train_epoch_info(self, epoch) -> None:
        log_dict = {}
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_avgs())
        if hasattr(self.loss_func, "alpha"):
            log_dict["alpha"] = self.loss_func.alpha
        log_dict["acc"] = self.acc_meter.avg
        log_dict["acc5"] = self.acc5_meter.avg
        if self.rank == 0 and self.cfg.train.evaluate_logits:
            log_dict.update(self.logits_evaluator.mean_score())
        lambd_mean, lambd_max = self.loss_func.get_lambd_metric()
        log_dict["lambd_mean"] = lambd_mean
        log_dict["lambd_max"] = lambd_max
        logger.info(
            "train epoch[{}/{}]\t{}".format(epoch + 1, self.cfg.train.max_epoch, json.dumps(round_dict(log_dict)))
        )
        if self.rank == 0 and self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                (f"train/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="val"):
        self.reset_meter()
        self.model.eval()

        if phase == "val":
            self.loss_func.reset_update_lambd()
        max_iter = len(data_loader)
        end = time.time()
        for i, (samples, targets) in enumerate(data_loader):
            self.data_time_meter.update(time.time() - end)
            samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            # forward pass
            with self.amp_autocast():
                outputs = self.model(samples)
            # metric
            loss = self.loss_func(outputs, targets)
            if phase == "val":
                self.loss_func.update_lambd(outputs)

            acc, acc5 = accuracy(outputs, targets, topk=(1, 5))

            reduced_loss = self.reduce_loss(loss)
            acc = reduce_tensor(acc, self.world_size)
            acc5 = reduce_tensor(acc5, self.world_size)

            torch.cuda.synchronize()

            self.loss_meter.update(reduced_loss, targets.size(0))
            self.acc_meter.update(acc.item(), targets.size(0))
            self.acc5_meter.update(acc5.item(), targets.size(0))

            logits_list = [torch.zeros_like(outputs) for _ in range(self.world_size)]
            labels_list = [torch.zeros_like(targets) for _ in range(self.world_size)]
            if self.rank == 0:
                gather(outputs, logits_list)
                gather(targets, labels_list)
            else:
                gather(outputs)
                gather(targets)

            if self.rank == 0:
                logits = torch.cat(logits_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                self.calibrate_evaluator.update(logits, labels)
                self.logits_evaluator.update(to_numpy(logits))

                predicts = F.softmax(logits, dim=1)
                self.classification_evaluator.update(
                    to_numpy(predicts), to_numpy(labels),
                )

            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_eval_iter_info(i, max_iter, epoch=epoch, phase=phase)
            end = time.time()
        self.log_eval_epoch_info(epoch=epoch, phase=phase)
        if phase == "val":
            self.loss_func.set_lambd(epoch)

        return self.loss_meter.avg(0), self.acc_meter.avg