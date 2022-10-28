import logging
import time
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from contextlib import suppress
import json
import os
import os.path as osp
from terminaltables.ascii_table import AsciiTable
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch import distributed as dist
import torch.nn.functional as F
from timm.utils import NativeScaler, dispatch_clip_grad

from ..utils import (
    reduce_tensor, gather, build_dist_data_loader,
    set_random_seed, to_numpy, get_lr, round_dict,
    load_checkpoint
)
from .optimizer import build_optimizer
from ..evaluation import (
    AverageMeter, LossMeter, accuracy, ClassificationEvaluator,
    CalibrateEvaluator, LogitsEvaluator
)
from ..losses import LogitMarginL1
from .temperature_scaling import ModelWithTemperature

logger = logging.getLogger(__name__)


class Tester:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        logger.info("\n" + OmegaConf.to_yaml(self.cfg, resolve=True))
        set_random_seed(
            self.cfg.seed if self.cfg.seed is not None else None,
            deterministic=True if self.cfg.seed is not None else False
        )

        self.build_test_loader()
        self.build_model(self.cfg.test.checkpoint)
        self.build_meter()
        self.init_wandb_or_not()

    def build_test_loader(self) -> None:
        # data pipeline
        self.test_dataset = instantiate(self.cfg.data.object.test)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.test_batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

    def build_model(self, checkpoint: Optional[str] = "") -> None:
        self.model = instantiate(self.cfg.model.object)
        self.model.cuda()
        logger.info("Model initialized")
        self.checkpoint_path = osp.join(
            self.cfg.work_dir, "best.pth" if not checkpoint else checkpoint
        )
        load_checkpoint(self.checkpoint_path, self.model)

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        self.evaluator = ClassificationEvaluator(self.num_classes)
        self.calibrate_evaluator = CalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
        )

    def reset_meter(self):
        self.batch_time_meter.reset()
        self.evaluator.reset()
        self.calibrate_evaluator.reset()

    def init_wandb_or_not(self) -> None:
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=["test"],
            )
            wandb.run.name = "{}-{}-{}".format(
                wandb.run.id, self.cfg.model.name, self.cfg.loss.name
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    @torch.no_grad()
    def eval_epoch(
        self, data_loader,
        phase="val",
        temp=1.0,
        post_temp=False
    ) -> None:
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)

        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # forward
            outputs = self.model(inputs)
            if post_temp:
                outputs = outputs / temp
            # metric
            self.calibrate_evaluator.update(outputs, labels)
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels)
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.cfg.log_period == 0:
                self.log_eval_iter_info(i, max_iter, phase=phase)
        self.log_eval_epoch_info(phase)
        if self.cfg.test.save_logits:
            logits_save_path = (
                osp.splitext(self.checkpoint_path)[0]
                + "_logits"
                + ("_pt.npz" if post_temp else ".npz")
            )
            self.calibrate_evaluator.save_npz(logits_save_path)

    def log_eval_iter_info(self, iter, max_iter, phase="val"):
        log_dict = {}
        log_dict["batch_time"] = self.batch_time_meter.avg
        log_dict.update(self.evaluator.curr_score())
        logger.info(
            f"{phase} iter[{iter + 1}/{max_iter}]\t{json.dumps(round_dict(log_dict))}"
        )

    def log_eval_epoch_info(self, phase="val"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        classification_metric, classification_table_data = self.evaluator.mean_score()
        log_dict.update(classification_metric)
        calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score()
        log_dict.update(calibrate_metric)
        logger.info(f"{phase} epoch\t{json.dumps(round_dict(log_dict))}")
        logger.info("\n" + AsciiTable(classification_table_data).table)
        logger.info("\n" + AsciiTable(calibrate_table_data).table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {}
            wandb_log_dict.update(dict(
                (f"{phase}/{key}", value) for (key, value) in log_dict.items()
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
            if "test" in phase and self.cfg.calibrate.visualize:
                fig_reliab, fig_hist = self.calibrate_evaluator.plot_reliability_diagram()
                wandb_log_dict["{}/calibrate_reliability".format(phase)] = fig_reliab
                wandb_log_dict["{}/confidence_histogram".format(phase)] = fig_hist
            wandb.log(wandb_log_dict)

    def post_temperature(self):
        _, self.val_loader = instantiate(self.cfg.data.object.trainval)
        model_with_temp = ModelWithTemperature(
            self.model,
            learn=self.cfg.post_temperature.learn,
            grid_search_interval=self.cfg.post_temperature.grid_search_interval,
            cross_validate=self.cfg.post_temperature.cross_validate,
        )
        model_with_temp.set_temperature(self.val_loader)
        temp = model_with_temp.get_temperature()
        if self.cfg.wandb.enable:
            wandb.log({
                "temperature": temp
            })
        return temp

    def test(self):
        logger.info(
            "Everything is perfect so far. Let's start testing. Good luck!"
        )
        self.eval_epoch(self.test_loader, phase="test")
        if self.cfg.post_temperature.enable:
            logger.info("Test with post-temperature scaling!")
            temp = self.post_temperature()
            self.eval_epoch(self.test_loader, phase="test-p", temp=temp, post_temp=True)

    def run(self):
        self.test()
