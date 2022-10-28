import logging
from terminaltables import AsciiTable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from statistics import mean
import wandb

from .evaluator import Evaluator
from .metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from .reliability_diagram import ReliabilityDiagram
from ..utils.torch_helper import to_numpy

logger = logging.getLogger(__name__)


class SegmentCalibrateEvaluator(Evaluator):
    def __init__(self, num_classes,
        num_bins:int = 15, ignore_index:  int = -1, batch_mode: bool = False 
    ) -> None:
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.ignore_index = ignore_index
        self.batch_mode = batch_mode
        self.reset()

        self.nll_criterion = nn.CrossEntropyLoss().cuda()
        self.ece_criterion = ECELoss(self.num_bins).cuda()
        self.aece_criterion = AdaptiveECELoss(self.num_bins).cuda()
        self.cece_criterion = ClasswiseECELoss(self.num_bins).cuda()

    def reset(self) -> None:
        self.count = [] if self.batch_mode else 0
        self.nll = []
        self.ece = []
        self.aece = []
        self.cece = []

    def num_samples(self):
        return sum(self.count) if self.batch_mode else self.count

    def main_metric(self) -> None:
        return "ece"

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """update

        Args:
            logits (torch.Tensor): n x num_classes
            label (torch.Tensor): n x 1
        """
        assert logits.shape[0] == labels.shape[0]
        n, c, x, y = logits.shape

        logits = torch.einsum("ncxy->nxyc", logits)
        logits = logits.reshape(n * x * y, -1)
        labels = labels.reshape(n * x * y)

        if 0 <= self.ignore_index:
            index = torch.nonzero(labels != self.ignore_index).squeeze()
            logits = logits[index, :]
            labels = labels[index]

        if self.batch_mode:
            # dismiss background
            index = torch.nonzero(labels != 0).squeeze()
            logits = logits[index, :].cuda()
            labels = labels[index].cuda()

            n = logits.shape[0]
            self.count.append(n)
            nll = self.nll_criterion(logits, labels).item()
            ece = self.ece_criterion(logits, labels).item()
            aece = self.aece_criterion(logits, labels).item()
            cece = self.cece_criterion(logits, labels).item()

            self.nll.append(nll)
            self.ece.append(ece)
            self.aece.append(aece)
            self.cece.append(cece)
        else:
            # dismiss background
            #index = torch.nonzero(labels != 0).squeeze()
            #logits = logits[index, :].cuda()
            #labels = labels[index].cuda()

            self.count += 1
            self.nll.append(self.nll_criterion(logits, labels).item())
            self.ece.append(self.ece_criterion(logits, labels).item())
            self.aece.append(self.aece_criterion(logits, labels).item())
            self.cece.append(self.cece_criterion(logits, labels).item())

        # self.count.append(n)
        # nll = self.nll_criterion(logits, labels).item()
        # ece = self.ece_criterion(logits, labels).item()
        # aece = self.aece_criterion(logits, labels).item()
        # cece = self.cece_criterion(logits, labels).item()

        # self.nll.append(nll)
        # self.ece.append(ece)
        # self.aece.append(aece)
        # self.cece.append(cece)

    def mean_score(self, print=False, all_metric=True):
        if self.batch_mode:
            total_count = sum(self.count)
            nll, ece, aece, cece = 0, 0, 0, 0
            for i in range(len(self.nll)):
                nll += self.nll[i] * (self.count[i] / total_count)
                ece += self.ece[i] * (self.count[i] / total_count)
                aece += self.aece[i] * (self.count[i] / total_count)
                cece += self.cece[i] * (self.count[i] / total_count)
        else:
            nll = mean(self.nll)
            ece = mean(self.ece)
            aece = mean(self.aece)
            cece = mean(self.cece)

        metric = {"nll": nll, "ece": ece, "aece": aece, "cece": cece}

        columns = ["samples", "nll", "ece", "aece", "cece"]
        table_data = [columns]
        table_data.append(
            [
                self.num_samples(),
                "{:.5f}".format(nll),
                "{:.5f}".format(ece),
                "{:.5f}".format(aece),
                "{:.5f}".format(cece),
            ]
        )

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        if all_metric:
            return metric, table_data
        else:
            return metric[self.main_metric()]

    def wandb_score_table(self):
        _, table_data = self.mean_score(print=False)
        return wandb.Table(
            columns=table_data[0],
            data=table_data[1:]
        )

    def plot_reliability_diagram(self):
        diagram = ReliabilityDiagram(bins=25, style="curve")
        probs = F.softmax(self.logits, dim=1)
        fig_reliab, fig_hist = diagram.plot(to_numpy(probs), to_numpy(self.labels))
        return fig_reliab, fig_hist

    def save_npz(self, save_path):
        np.savez(
            save_path,
            logits=to_numpy(self.logits),
            labels=to_numpy(self.labels)
        )
