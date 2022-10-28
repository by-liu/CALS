import os.path as osp
from shutil import copyfile
import time
import json
import logging
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
import wandb
from terminaltables.ascii_table import AsciiTable
from timm.utils import dispatch_clip_grad

from ..losses import LogitMarginL1
from ..evaluation import (
    AverageMeter, LossMeter, ClassificationEvaluator,
    CalibrateEvaluator, LogitsEvaluator, accuracy
)
from ..utils import (
    load_train_checkpoint, save_train_checkpoint, round_dict,
    to_numpy, get_lr, set_random_seed
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
        set_random_seed(
            self.cfg.seed if self.cfg.seed is not None else None,
            deterministic=True if self.cfg.seed is not None else False
        )
        self.build_data_loader()
        self.build_model()
        self.build_loss()
        self.build_optimizer()
        self.build_meter()
        self.init_wandb_or_not()

    def build_data_loader(self) -> None:
        # data pipeline
        self.train_dataset, self.val_dataset = instantiate(self.cfg.data.object.trainval)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )
        self.train_iter_per_epoch = len(self.train_loader)

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

        logger.info("Data pipeline initialized")

    def build_test_loader(self) -> None:
        self.test_dataset = instantiate(self.cfg.data.object.test)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.test_batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

    def build_model(self) -> None:
        # network
        self.model = instantiate(self.cfg.model.object)
        self.model.cuda()
        logger.info("Model initialized")

    def build_loss(self) -> None:
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.cuda()
        logger.info(self.loss_func)
        logger.info("Loss initialized")

    def build_optimizer(self) -> None:
        # build solver
        self.optimizer = instantiate(
            self.cfg.optim.object, self.model.parameters()
        )
        self.lr_scheduler = instantiate(self.cfg.lr_scheduler.object, self.optimizer)
        logger.info("Optimizer initialized : {}".format(self.optimizer))
        logger.info("LR scheduler initialized : {}".format(self.lr_scheduler))

    def init_wandb_or_not(self) -> None:
        if self.cfg.wandb.enable:
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

    def start_or_resume(self):
        if self.cfg.train.resume:
            self.start_epoch, self.best_epoch, self.best_val_score = (
                load_train_checkpoint(
                    self.cfg.work_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler
                )
            )
        else:
            self.start_epoch = 0
            self.best_epoch = -1
            self.best_val_score = -float("inf")
        self.max_epoch = self.cfg.train.max_epoch

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
        self.evaluator = ClassificationEvaluator(self.num_classes)
        self.calibrate_evaluator = CalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
        )
        self.logits_evaluator = LogitsEvaluator()
        self.acc_meter = AverageMeter()

    def reset_meter(self):
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()
        self.calibrate_evaluator.reset()
        self.logits_evaluator.reset()
        self.acc_meter.reset()

    def log_train_iter_info(self, iter, epoch):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.avg
        log_dict["batch_time"] = self.batch_time_meter.avg
        log_dict.update(self.loss_meter.get_vals())
        log_dict["acc"] = self.acc_meter.val
        log_dict["lr"] = get_lr(self.optimizer)
        if self.cfg.train.evaluate_logits:
            log_dict.update(self.logits_evaluator.curr_score())
        # log_dict.update(self.probs_evaluator.curr_score())
        logger.info("train iter[{}/{}][{}]\t{}".format(
            iter + 1, self.train_iter_per_epoch, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = {"iter": epoch * self.train_iter_per_epoch + iter}
            wandb_log_dict.update(dict(
                (f"train/iter/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_train_epoch_info(self, epoch):
        log_dict = {}
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, LogitMarginL1):
            log_dict["alpha"] = self.loss_func.alpha
        log_dict["acc"] = self.acc_meter.avg
        if self.cfg.train.evaluate_logits:
            log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info(f"train epoch[{epoch + 1}/{self.max_epoch}]\t{json.dumps(round_dict(log_dict))}")
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
        classification_metric, classification_table_data = self.evaluator.mean_score()
        log_dict.update(classification_metric)
        calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score()
        log_dict.update(calibrate_metric)
        log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info("{} epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        logger.info("\n" + AsciiTable(classification_table_data).table)
        logger.info("\n" + AsciiTable(calibrate_table_data).table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                (f"{phase}/{key}", value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict[f"{phase}/classification_score_table"] = (
                wandb.Table(
                    columns=classification_table_data[0],
                    data=classification_table_data[1:]
                )
            )
            wandb_log_dict[f"{phase}/calibrate_score_table"] = (
                wandb.Table(
                    columns=calibrate_table_data[0],
                    data=calibrate_table_data[1:]
                )
            )
            if "test" in phase and self.cfg.calibrate.visualize:
                fig_reliab, fig_hist = self.calibrate_evaluator.plot_reliability_diagram()
                wandb_log_dict["{}/calibrate_reliability".format(phase)] = fig_reliab
                wandb_log_dict["{}/confidence_histogram".format(phase)] = fig_hist
            wandb.log(wandb_log_dict)

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        end = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
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
            # predicts = F.softmax(outputs, dim=1)
            # pred_labels = torch.argmax(predicts, dim=1)
            acc  = accuracy(outputs, labels)[0]
            self.acc_meter.update(acc.item(), labels.size(0))

            if self.cfg.train.evaluate_logits:
                self.logits_evaluator.update(to_numpy(outputs))
            # self.probs_evaluator.update(to_numpy(predicts))

            self.lr_scheduler.step_update(epoch * self.train_iter_per_epoch + i)

            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_train_iter_info(i, epoch)
            end = time.time()
        self.log_train_epoch_info(epoch)

    @torch.no_grad()
    def eval_epoch(
        self, data_loader, epoch,
        phase="val",
        temp=1.0,
        post_temp=False
    ):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, samples in enumerate(data_loader):
            inputs, labels = samples[0].cuda(non_blocking=True), samples[1].cuda(non_blocking=True)
            # forward
            outputs = self.model(inputs)
            if post_temp:
                outputs = outputs / temp
            loss = self.loss_func(outputs, labels)
            # metric
            self.loss_meter.update(loss)
            self.calibrate_evaluator.update(outputs, labels)
            self.logits_evaluator.update(to_numpy(outputs))
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels)
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_eval_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_eval_epoch_info(epoch, phase)

        return (
            self.loss_meter.avg(0),
            self.evaluator.mean_score()[0][self.evaluator.main_metric()],
        )

    def log_eval_iter_info(self, iter, max_iter, epoch, phase="val"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.avg
        log_dict["batch_time"] = self.batch_time_meter.avg
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())
        if self.cfg.train.evaluate_logits:
            log_dict.update(self.logits_evaluator.curr_score())
        log_dict["lr"] = get_lr(self.optimizer)
        # log_dict.update(self.probs_evaluator.curr_score())
        logger.info("{} iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))

    def train(self):
        logger.info(
            "Everything is perfect so far. Let's start training. Good luck!"
        )
        for epoch in range(self.start_epoch, self.max_epoch):
            logger.info("=" * 20)
            logger.info(" Start epoch {}".format(epoch + 1))
            logger.info("=" * 20)
            self.train_epoch(epoch)
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch, phase="val")
            # run lr scheduler
            self.lr_scheduler.step(epoch + 1, val_score)

            if isinstance(self.loss_func, LogitMarginL1):
                self.loss_func.schedule_alpha(epoch)

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_epoch = epoch
                best_checkpoint = True
            else:
                best_checkpoint = False

            save_train_checkpoint(
                self.cfg.work_dir, self.model, self.optimizer, self.lr_scheduler,
                epoch=epoch,
                best_checkpoint=best_checkpoint,
                val_score=val_score,
            )
            # logging best performance on val so far
            logger.info(
                f"Epoch[{epoch + 1}]\tBest {self.evaluator.main_metric()} on Val : {self.best_val_score:.4f} at epoch {self.best_epoch + 1}"
            )
            if self.cfg.wandb.enable and best_checkpoint:
                wandb.log({
                    "epoch": epoch,
                    "val/best_epoch": self.best_epoch,
                    f"val/best_{self.evaluator.main_metric()}": self.best_val_score,
                })
        if self.cfg.wandb.enable:
            copyfile(
                osp.join(self.cfg.work_dir, "best.pth"),
                osp.join(self.cfg.work_dir, "{}-best.pth".format(wandb.run.name))
            )

    # def post_temperature(self):
    #     model_with_temp = ModelWithTemperature(
    #         self.model,
    #         learn=self.cfg.post_temperature.learn,
    #         grid_search_interval=self.cfg.post_temperature.grid_search_interval,
    #         cross_validate=self.cfg.post_temperature.cross_validate,
    #         device=self.device
    #     )
    #     model_with_temp.set_temperature(self.val_loader)
    #     temp = model_with_temp.get_temperature()
    #     wandb.log({
    #         "temperature": temp
    #     })
    #     return temp

    def test(self):
        logger.info("We are almost done : final testing ...")
        self.build_test_loader()
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_train_checkpoint(
            osp.join(self.cfg.work_dir, "best.pth"), self.model
        )
        self.eval_epoch(self.test_loader, epoch, phase="test")
        # if self.cfg.post_temperature.enable:
        #     logger.info("Test with post-temperature scaling!")
        #     temp = self.post_temperature()
        #     self.eval_epoch(self.test_loader, epoch, phase="TestPT", temp=temp, post_temp=True)

    def run(self):
        self.start_or_resume()
        self.train()
        self.test()
