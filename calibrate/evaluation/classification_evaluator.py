import logging
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import top_k_accuracy_score, confusion_matrix

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 1. / batch_size for k in topk]


class ClassificationEvaluator(Evaluator):
    def __init__(
        self,
        num_classes: int
    ) -> None:
        self.num_classes = num_classes

    def reset(self) -> None:
        self.preds = None
        self.labels = None

    def main_metric(self) -> None:
        return "acc"

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def update(self, pred: np.ndarray, label: np.ndarray, accumulate=True) -> float:
        """update

        Args:
            pred (np.ndarray): n x num_classes, predicted scores
            label (np.ndarray): n x 1, ground truth labels

        Returns:
            float: acc
        """
        assert pred.shape[0] == label.shape[0]
        if accumulate:
            if self.preds is None:
                self.preds = pred
                self.labels = label
            else:
                self.preds = np.concatenate((self.preds, pred), axis=0)
                self.labels = np.concatenate((self.labels, label), axis=0)

        pred_label = np.argmax(pred, axis=1)
        acc = (pred_label == label).astype("int").sum() / label.shape[0]

        self.curr = {"acc": acc}

        return acc

    def curr_score(self) -> Dict[str, float]:
        return self.curr

    def mean_score(self) -> Tuple[Dict[str, float], List[List[str]]]:
        acc = top_k_accuracy_score(self.labels, self.preds, k=1)
        acc_5 = top_k_accuracy_score(self.labels, self.preds, k=5)
        metric = {"acc": acc, "acc_5": acc_5}

        pred_labels = np.argmax(self.preds, axis=1)
        confusion = confusion_matrix(self.labels, pred_labels, normalize="true")
        macc = np.diagonal(confusion).mean()
        metric["macc"] = macc

        # table format for printing/logging
        columns = ["samples", "acc", "acc_5", "macc"]
        table_data = [columns]
        table_data.append(
            [
                self.num_samples(),
                "{:.5f}".format(acc),
                "{:.5f}".format(acc_5),
                "{:.5f}".format(macc)
            ]
        )

        return metric, table_data

    # def wandb_score_table(self):
    #     _, table_data = self.mean_score(print=False)
    #     return wandb.Table(
    #         columns=table_data[0],
    #         data=table_data[1:]
    #     )