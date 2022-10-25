"""
Distributed training class
"""
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

logger = logging.getLogger(__name__)


class DistributedTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.init_distributed()
        logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
        set_random_seed(
            self.cfg.seed + self.rank if self.cfg.seed is not None else None,
            deterministic=True if self.cfg.seed is not None else False
        )
        self.build_data_loader()
        self.build_model()
        self.build_loss()
        self.build_optimizer()
        self.build_meter()
        self.init_wandb_or_not()

    def init_wandb_or_not(self) -> None:
        if self.cfg.wandb.enable and self.rank == 0:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=["train"],
            )
            wandb.run.name = "{}-{}-{}".format(
                wandb.run.id, self.cfg.data.name, self.cfg.model.name
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    def init_distributed(self) -> None:
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()

    def build_data_loader(self) -> None:
        self.train_dataset, self.val_dataset = instantiate(self.cfg.data.object.trainval)

        self.train_loader = build_dist_data_loader(
            self.train_dataset,
            self.cfg.data.train_batch_size,
            world_size=self.world_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True
        )
        self.train_iter_per_epoch = len(self.train_loader)

        self.val_loader = build_dist_data_loader(
            self.val_dataset,
            self.cfg.data.val_batch_size,
            world_size=self.world_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=False
        )

        logger.info("Train dataset initialized : {}".format(self.train_dataset))
        logger.info("Val dataset initialized : {}".format(self.val_dataset))
        logger.info("Distributed train and val data loader initialized.")

        self.mixup_fn = (
            instantiate(self.cfg.mixup.object) if self.cfg.mixup.enable
            else None
        )

    def build_test_loader(self) -> None:
        self.test_dataset = instantiate(self.cfg.data.object.test)

        self.test_loader = build_dist_data_loader(
            self.test_dataset,
            self.cfg.data.test_batch_size,
            world_size=self.world_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=False,
        )

        logger.info("Test dataset initialized : {}".format(self.test_dataset))
        logger.info("Distributed test data loader initialized.")

    def build_model(self) -> None:
        self.model = instantiate(self.cfg.model.object)
        self.model.cuda()
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        logger.info(
            "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
            "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
        )
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.local_rank]
        )

    def build_loss(self) -> None:
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.cuda()
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        if self.cfg.train.use_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scalar = NativeScaler()
            logger.info("Using native Torch AMP. Training in mixed precision.")
        else:
            logger.info("AMP not enabled. Training in float32.")

    def build_optimizer(self) -> None:
        self.optimizer = build_optimizer(self.cfg.optim, self.model)
        self.lr_scheduler = instantiate(self.cfg.lr_scheduler.object, self.optimizer)
        logger.info("Optimizer initialized : {}".format(self.optimizer))
        logger.info("LR scheduler initialized : {}".format(self.lr_scheduler))

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
        if self.rank == 0:
            self.classification_evaluator.reset()
            self.calibrate_evaluator.reset()
            self.logits_evaluator.reset()

    def reduce_loss(self, loss):
        if isinstance(loss, (tuple, list)):
            reduced_loss = [reduce_tensor(l.data, self.world_size) for l in loss]
        else:
            reduced_loss = reduce_tensor(loss.data, self.world_size)

        return reduced_loss

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
            # backward pass
            self.optimizer.zero_grad()
            if hasattr(self, "loss_scalar"):
                self.loss_scalar(
                    loss_total, self.optimizer,
                    clip_grad=self.cfg.train.get("clip_grad", 2.0),
                    clip_mode=self.cfg.train.get("clip_mode", "norm"),
                    parameters=self.model.parameters(),
                    create_graph=hasattr(self.optimizer, "is_second_order") and self.optimizer.is_second_order
                )
            else:
                loss_total.backward()
                if self.cfg.train.get("clip_grad") is not None:
                    dispatch_clip_grad(
                        self.model.parameters(),
                        value=self.cfg.train.get("clip_grad", 2.0),
                        mode=self.cfg.train.get("clip_mode", "Norm")
                    )
                self.optimizer.step()
            # metric
            reduced_loss = self.reduce_loss(loss)

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

            if self.lr_scheduler is not None:
                self.lr_scheduler.step_update(epoch * self.train_iter_per_epoch + i)
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_train_iter_info(i, epoch)
            end = time.time()
        self.log_train_epoch_info(epoch)

    def log_train_iter_info(self, iter, epoch):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.avg
        log_dict["batch_time"] = self.batch_time_meter.avg
        log_dict.update(self.loss_meter.get_vals())
        log_dict["acc"] = self.acc_meter.val
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
        logger.info(
            "train epoch[{}/{}]\t{}".format(epoch + 1, self.cfg.train.max_epoch, json.dumps(round_dict(log_dict)))
        )
        if self.rank == 0 and self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("train/{}".format(key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_eval_iter_info(self, iter, max_iter, epoch, phase="val"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict.update(self.loss_meter.get_vals())
        log_dict["acc"] = self.acc_meter.val
        # log_dict.update(self.probs_evaluator.curr_score())
        logger.info("{} iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))

    def log_eval_epoch_info(self, epoch, phase="val") -> None:
        log_dict = {}
        log_dict.update(self.loss_meter.get_avgs())
        log_dict["ori_acc"] = self.acc_meter.avg
        log_dict["ori_acc5"] = self.acc5_meter.avg
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

        end = time.time()
        for i, (samples, targets) in enumerate(data_loader):
            self.data_time_meter.update(time.time() - end)
            samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            # forward pass
            with self.amp_autocast():
                outputs = self.model(samples)
            # metric
            loss = criterion(outputs, targets)
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
        return self.loss_meter.avg(0), self.acc_meter.avg

    def start_or_resume(self):
        if self.cfg.train.resume:
            self.start_epoch, self.best_epoch, self.best_val_score = (
                load_train_checkpoint(
                    self.cfg.work_dir,
                    # device=torch.device(self.local_rank),
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                )
            )
        else:
            self.start_epoch = 0
            self.best_epoch = -1
            self.best_val_score = -float("inf")

    def train(self):
        # self.start_or_resume()
        logger.info(
            "Everything is perfect so far. Let's start training. Good luck!"
        )

        for epoch in range(self.start_epoch, self.cfg.train.max_epoch):
            logger.info("=" * 20)
            logger.info(" Start epoch {}".format(epoch + 1))
            logger.info("=" * 20)
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            self.train_epoch(epoch)
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch, phase="val")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch + 1, val_score)

            # run alpah scheduler for LogitMargin loss
            if isinstance(self.loss_func, LogitMarginL1):
                self.loss_func.schedule_alpha(epoch)

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_epoch = epoch
                best_checkpoint = True
            else:
                best_checkpoint = False

            if self.rank == 0:
                save_train_checkpoint(
                    self.cfg.work_dir, self.model, self.optimizer, self.lr_scheduler,
                    epoch=epoch,
                    best_checkpoint=best_checkpoint,
                    val_score=val_score,
                )

                if best_checkpoint and self.cfg.wandb.enable:
                    wandb.log({
                        "epoch": epoch,
                        "val/best_epoch": self.best_epoch,
                        "val/best_acc": self.best_val_score,
                    })

            logger.info(
                "Epoch[{}]\tBest {} on Val : {:.4f} at epoch {}".format(
                    epoch + 1, "acc", self.best_val_score, self.best_epoch + 1
                )
            )
        if self.rank == 0 and self.cfg.wandb.enable:
            copyfile(
                osp.join(self.cfg.work_dir, "best.pth"),
                osp.join(self.cfg.work_dir, f"{wandb.run.name}-best.pth")
            )

    def test(self):
        logger.info("We are almost done: final evaluation on test set.")
        self.build_test_loader()
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_train_checkpoint(
            osp.join(self.cfg.work_dir, "best.pth"),
            model=self.model,
        )
        self.eval_epoch(self.test_loader, epoch, phase="test")

    def run(self):
        self.start_or_resume()
        self.train()
        self.test()
