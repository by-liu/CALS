import torch
from typing import Dict, Optional, List


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeter:
    """A class wrapper to record the values of a loss function.
    Support loss function with mutiple returning terms [num_terms]
    """
    def __init__(self, num_terms: int = 1, names: Optional[List] = None) -> None:
        self.num_terms = num_terms
        self.names = (
            names if names is not None
            else ["loss" if i == 0 else "loss_" + str(i) for i in range(self.num_terms)]
        )
        self.meters = [AverageMeter() for _ in range(self.num_terms)]

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def avg(self, index=None):
        if index is None:
            ret = {}
            for name, meter in zip(self.names, self.meters):
                ret[name] = meter.avg
            return ret
        else:
            return self.meters[index].avg

    def update(self, val, n: int = 1):
        if not isinstance(val, (tuple, list)):
            val = [val]
        for x, meter in zip(val, self.meters):
            if isinstance(x, torch.Tensor):
                x = x.item()
            meter.update(x, n)

    def get_vals(self) -> Dict:
        ret = {}
        for name, meter in zip(self.names, self.meters):
            ret[name] = meter.val
        return ret

    def get_avgs(self) -> Dict:
        ret = {}
        for name, meter in zip(self.names, self.meters):
            ret[name] = meter.avg
        return ret

    def status(self) -> str:
        ret = []
        for name, meter in zip(self.names, self.meters):
            ret.append("{} {:.4f} ({:.4f})".format(name, meter.val, meter.avg))
        return "\t".join(ret)

    def avg_status(self) -> str:
        ret = []
        for name, meter in zip(self.names, self.meters):
            ret.append("{} {:.4f}".format(name, meter.avg))
        return "\t".join(ret)
