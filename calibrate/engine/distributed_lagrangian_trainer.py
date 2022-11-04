"""
Distributed training with lagrangian multiplier for calibration
"""
import logging
import json
import time
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from terminaltables.ascii_table import AsciiTable

import torch
from torch import distributed as dist
import torch.nn.functional as F
from timm.utils import NativeScaler, dispatch_clip_grad

from .distributed_trainer import DistributedTrainer


from ..utils import (
    reduce_tensor, gather, build_dist_data_loader,
    set_random_seed, to_numpy, get_lr, round_dict,
    save_train_checkpoint, load_train_checkpoint
)
from ..evaluation import (
    AverageMeter, LossMeter, accuracy, ClassificationEvaluator,
    CalibrateEvaluator, LogitsEvaluator
)


logger = logging.getLogger(__name__)


class DistributedLagrangianTrainer(DistributedTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.init_lagrangian()

    def init_lagrangian(self) -> None:
        self.lagrangian = instantiate(self.cfg.lag.object)

    def build_meter(self) -> None:
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        if hasattr(self.loss_func, "names"):
            self.loss_meter = LossMeter(
                num_terms=len(self.loss_func.names),
                names=self.loss_func.names
            )
        else:
            self.loss_meter = LossMeter()
        self.acc_meter = AverageMeter()
        self.acc5_meter = AverageMeter()
        self.penalty_meter = AverageMeter()
        self.constraint_meter = AverageMeter()
        if self.rank == 0:
            self.classification_evaluator = ClassificationEvaluator(self.num_classes)
            self.calibrate_evaluator = CalibrateEvaluator(
                self.num_classes,
                num_bins=self.cfg.calibrate.num_bins
            )
            self.logits_evaluator = LogitsEvaluator()

    def reset_meter(self) -> None:
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()
        self.acc_meter.reset()
        self.acc5_meter.reset()
        self.penalty_meter.reset()
        self.constraint_meter.reset()
        if self.rank == 0:
            self.classification_evaluator.reset()
            self.calibrate_evaluator.reset()
            self.logits_evaluator.reset()

    def log_train_iter_info(self, iter, epoch):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.avg
        log_dict["batch_time"] = self.batch_time_meter.avg
        log_dict.update(self.loss_meter.get_vals())
        log_dict["acc"] = self.acc_meter.val
        log_dict["penalty"] = self.penalty_meter.val
        log_dict["constraint"] = self.constraint_meter.val
        log_dict["lr"] = get_lr(self.optimizer)
        if self.rank == 0 and self.cfg.train.evaluate_logits:
            log_dict.update(self.logits_evaluator.curr_score())
        # log_dict.update(self.probs_evaluator.curr_score())
        logger.info("train iter[{}/{}][{}]\t{}".format(
            iter + 1, self.train_iter_per_epoch, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))
        if self.rank == 0 and self.cfg.wandb.enable:
            wandb_log_dict = {"iter": epoch * self.train_iter_per_epoch + iter}
            wandb_log_dict.update(dict(
                ("train/iter/{}".format(key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

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
        log_dict["penalty"] = self.penalty_meter.avg
        log_dict["constraint"] = self.constraint_meter.avg
        lambd_mean, lambd_max = self.lagrangian.get_lambd_metric()
        log_dict["lambd_mean"] = lambd_mean
        log_dict["lambd_max"] = lambd_max
        log_dict["rho_mean"], log_dict["rho_max"] = self.lagrangian.get_rho_metric()
        logger.info(
            "train epoch[{}/{}]\t{}".format(epoch + 1, self.cfg.train.max_epoch, json.dumps(round_dict(log_dict)))
        )
        if self.rank == 0 and self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("train/{}".format(key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def train_epoch(self, epoch: int) -> None:
        self.reset_meter()
        self.model.train()

        end = time.time()
        for i, (samples, targets) in enumerate(self.train_loader):
            self.data_time_meter.update(time.time() - end)
            samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            if self.mixup_fn is not None:
                mixup_samples, mixup_targets = self.mixup_fn(samples, targets)
                # forward pass
                with self.amp_autocast():
                    outputs = self.model(mixup_samples)
                    loss = self.loss_func(outputs, mixup_targets)
            else:
                # forward pass
                with self.amp_autocast():
                    outputs = self.model(samples)
                    loss = self.loss_func(outputs, targets)
            loss_total = loss[0] if isinstance(loss, tuple) else loss
            # langrangian
            penalty, constraint = self.lagrangian.get(outputs)
            # backward pass
            self.optimizer.zero_grad()
            if hasattr(self, "loss_scalar"):
                self.loss_scalar(
                    loss_total + penalty, self.optimizer,
                    clip_grad=self.cfg.train.get("clip_grad", 2.0),
                    clip_mode=self.cfg.train.get("clip_mode", "norm"),
                    parameters=self.model.parameters(),
                    create_graph=hasattr(self.optimizer, "is_second_order") and self.optimizer.is_second_order
                )
            else:
                (loss_total + penalty).backward()
                if self.cfg.train.get("clip_grad") is not None:
                    dispatch_clip_grad(
                        self.model.parameters(),
                        value=self.cfg.train.get("clip_grad", 2.0),
                        mode=self.cfg.train.get("clip_mode", "Norm")
                    )
                    self.optimizer.step()
            # metric
            reduced_loss = self.reduce_loss(loss)
            reduced_penalty = reduce_tensor(penalty, self.world_size)
            reduced_constraint = reduce_tensor(constraint, self.world_size)

            acc, acc5 = accuracy(outputs, targets, topk=(1, 5))
            acc = reduce_tensor(acc, self.world_size)
            acc5 = reduce_tensor(acc5, self.world_size)

            if self.cfg.train.evaluate_logits:
                logits_list = [
                    torch.zeros_like(outputs) for _ in range(self.world_size)
                ]
                if self.rank == 0:
                    gather(outputs, logits_list)
                else:
                    gather(outputs)
                if self.rank == 0:
                    logits = torch.cat(logits_list, dim=0)
                    self.logits_evaluator.update(to_numpy(logits))

            torch.cuda.synchronize()

            self.loss_meter.update(reduced_loss, targets.size(0))
            self.acc_meter.update(acc.item(), targets.size(0))
            self.acc5_meter.update(acc5.item(), targets.size(0))
            self.penalty_meter.update(reduced_penalty.item())
            self.constraint_meter.update(reduced_constraint.item())

            if self.lr_scheduler is not None:
                self.lr_scheduler.step_update(epoch * self.train_iter_per_epoch + i)
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_train_iter_info(i, epoch)
            end = time.time()
        self.log_train_epoch_info(epoch)

    def log_eval_epoch_info(self, epoch, phase="val") -> None:
        log_dict = {}
        log_dict.update(self.loss_meter.get_avgs())
        log_dict["ori_acc"] = self.acc_meter.avg
        log_dict["ori_acc5"] = self.acc5_meter.avg
        if phase == "val":
            log_dict["penalty"] = self.penalty_meter.avg
            log_dict["constraint"] = self.constraint_meter.avg
        if self.rank == 0:
            log_dict["samples"] = self.classification_evaluator.num_samples()
            classification_metric, classification_table_data = self.classification_evaluator.mean_score()
            logger.info("\n" + AsciiTable(classification_table_data).table)
            log_dict.update(classification_metric)
            calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score()
            logger.info("\n" + AsciiTable(calibrate_table_data).table)
            log_dict.update(calibrate_metric)
            log_dict.update(self.logits_evaluator.mean_score())

        logger.info(
            "{} epoch[{}]\t{}".format(phase, epoch + 1, json.dumps(round_dict(log_dict)))
        )

        if self.rank == 0 and self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict["{}/classification_score_table".format(phase)] = (
                wandb.Table(
                    columns=classification_table_data[0],
                    data=classification_table_data[1:]
                )
            )
            wandb_log_dict["{}/calibrate_score_table".format(phase)] = (
                wandb.Table(
                    columns=calibrate_table_data[0],
                    data=calibrate_table_data[1:]
                )
            )
            wandb.log(wandb_log_dict)

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="val"):
        self.reset_meter()
        self.model.eval()

        criterion = (
            self.loss_func if not self.cfg.mixup.enable
            else torch.nn.CrossEntropyLoss()
        )

        max_iter = len(data_loader)

        if phase == "val":
            self.lagrangian.reset_update_lambd(epoch)
        end = time.time()
        for i, (samples, targets) in enumerate(data_loader):
            self.data_time_meter.update(time.time() - end)
            samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            # forward pass
            with self.amp_autocast():
                outputs = self.model(samples)
            # metric
            loss = criterion(outputs, targets)
            if phase == "val":
                self.lagrangian.update_lambd(outputs, epoch)
                penalty, constraint = self.lagrangian.get(outputs)
                reduced_penalty = reduce_tensor(penalty, self.world_size)
                reduced_constraint = reduce_tensor(constraint, self.world_size)
                self.penalty_meter.update(reduced_penalty.item())
                self.constraint_meter.update(reduced_constraint.item())

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
            self.lagrangian.set_lambd(epoch)
            self.lagrangian.update_rho(epoch)
            # if self.lagrangian.rho_update:
            #     self.lagrangian.update_rho_by_val(self.penalty_meter.avg, epoch)

        return self.loss_meter.avg(0), self.acc_meter.avg
