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

from .tester import Tester
from ..losses import LogitMarginL1
from ..evaluation import (
    AverageMeter, LossMeter, SegmentEvaluator, SegmentCalibrateEvaluator
)
from ..utils import (
    load_train_checkpoint, save_train_checkpoint, round_dict,
    to_numpy, get_lr
)

logger = logging.getLogger(__name__)


class SegmentTester(Tester):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        self.evaluator = SegmentEvaluator(
            self.test_loader.dataset.classes,
            ignore_index=255
        )
        self.calibrate_evaluator = SegmentCalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
            ignore_index=255,
        )

    def reset_meter(self):
        self.batch_time_meter.reset()
        self.evaluator.reset()
        self.calibrate_evaluator.reset()

    @torch.no_grad()
    def eval_epoch(self, data_loader, phase="val"):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # forward
            outputs = self.model(inputs)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
            # metric
            self.calibrate_evaluator.update(
                outputs, labels
            )
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                to_numpy(pred_labels), to_numpy(labels)
            )
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_eval_iter_info(i, max_iter, phase)
            end = time.time()
        self.log_eval_epoch_info(phase)

    def log_eval_iter_info(self, iter, max_iter, phase="val"):
        log_dict = {"batch_time": self.batch_time_meter.avg}
        log_dict.update(self.evaluator.curr_score())
        logger.info(
            f"{phase} iter[{iter + 1}/{max_iter}]\t{json.dumps(round_dict(log_dict))}"
        )

    def log_eval_epoch_info(self, phase="val"):
        log_dict = {"samples": self.evaluator.num_samples()}
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(print=False)
        log_dict.update(calibrate_metric)
        logger.info("{} epoch\t{}".format(
            phase, json.dumps(round_dict(log_dict))
        ))
        class_table_data = self.evaluator.class_score()
        logger.info("\n" + AsciiTable(class_table_data).table)
        logger.info("\n" + AsciiTable(calibrate_table_data).table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {}
            wandb_log_dict.update(dict(
                (f"{phase}/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict[f"{phase}/segment_score_table"] = (
                wandb.Table(
                    columns=class_table_data[0],
                    data=class_table_data[1:]
                )
            )
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
