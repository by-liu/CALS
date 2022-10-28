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

from .trainer import Trainer
from ..losses import LogitMarginL1
from ..evaluation import (
    AverageMeter, LossMeter, SegmentEvaluator, SegmentCalibrateEvaluator
)
from ..utils import (
    load_train_checkpoint, save_train_checkpoint, round_dict,
    to_numpy, get_lr
)

logger = logging.getLogger(__name__)


class SegmentTrainer(Trainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_meter(self):
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
        self.evaluator = SegmentEvaluator(
            self.train_loader.dataset.classes,
            ignore_index=255
        )
        self.calibrate_evaluator = SegmentCalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
            ignore_index=255,
        )

    def reset_meter(self):
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()
        self.calibrate_evaluator.reset()

    def log_iter_info(self, iter, max_iter, epoch, phase="train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.avg
        log_dict["batch_time"] = self.batch_time_meter.avg
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())
        if phase == "train":
            log_dict["lr"] = get_lr(self.optimizer)
        # log_dict.update(self.logits_evaluator.curr_score())
        # log_dict.update(self.probs_evaluator.curr_score())
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
        # log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info("train epoch[{}]\t{}".format(
            epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                (f"train/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_eval_epoch_info(self, epoch, phase="val"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict.update(self.loss_meter.get_avgs())
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        if phase.lower() == "test":
            calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(print=False)
            log_dict.update(calibrate_metric)
        # log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info("{} epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        class_table_data = self.evaluator.class_score()
        logger.info("\n" + AsciiTable(class_table_data).table)
        if phase.lower() == "test":
            logger.info("\n" + AsciiTable(calibrate_table_data).table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                (f"{phase}/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict[f"{phase}/segment_score_table"] = (
                wandb.Table(
                    columns=class_table_data[0],
                    data=class_table_data[1:]
                )
            )
            if phase.lower() == "test":
                wandb_log_dict[f"{phase}/calibrate_score_table"] = (
                    wandb.Table(
                        columns=calibrate_table_data[0],
                        data=calibrate_table_data[1:]
                    )
                )
            # if "test" in phase.lower() and self.cfg.calibrate.visualize:
            #     fig_reliab, fig_hist = self.calibrate_evaluator.plot_reliability_diagram()
            #     wandb_log_dict["{}/calibrate_reliability".format(phase)] = fig_reliab
            #     wandb_log_dict["{}/confidence_histogram".format(phase)] = fig_hist
            wandb.log(wandb_log_dict)

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
            self.evaluator.update(
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
    def eval_epoch(self, data_loader, epoch, phase="val"):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # forward
            outputs = self.model(inputs)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
            loss = self.loss_func(outputs, labels)
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

        return self.loss_meter.avg(0), self.evaluator.mean_score(main=True)

    def test(self):
        logger.info("We are almost done : final testing ...")
        self.build_test_loader()
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(f" Test at best epoch {epoch + 1}")
        logger.info("#################")
        logger.info(f"Best epoch[{epoch + 1}] :")
        load_train_checkpoint(
            osp.join(self.cfg.work_dir, "best.pth"), self.model
        )
        self.eval_epoch(self.test_loader, epoch, phase="test")
