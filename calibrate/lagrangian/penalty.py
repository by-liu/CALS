from typing import Tuple

import torch
from torch import Tensor


def p2(h: Tensor, lambd: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor]:
    y_sup = lambd * h + lambd * rho * (h ** 2) + 1 / 6 * (rho ** 2) * (h ** 3)
    y_inf = lambd * h / (1 - rho * h.clamp(max=0))

    grad_y_sup = lambd + 2 * lambd * rho * h + 1 / 2 * (rho ** 2) * (h ** 2)
    grad_y_inf = lambd / (1 - rho * h.clamp(max=0)) ** 2

    sup = h >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def p3(h: Tensor, lambd: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor]:
    if lambd.ndim == 1:
        lambd = lambd.unsqueeze(dim=0).expand(h.shape)

    if isinstance(rho, torch.Tensor) and rho.ndim == 1:
        rho = rho.unsqueeze(dim=0).expand(h.shape)

    y_sup = lambd * h + lambd * rho * (h ** 2)
    y_inf = lambd * h / (1 - rho * h.clamp(max=0))

    grad_y_sup = lambd + 2 * lambd * rho * h
    grad_y_inf = lambd / (1 - rho * h.clamp(max=0)) ** 2

    sup = h >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def phr(h: Tensor, lambd: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor]:
    x = lambd + rho * h
    y_sup = 1 / (2 * rho) * (x ** 2 - lambd ** 2)
    y_inf = - 1 / (2 * rho) * (lambd ** 2)

    grad_y_sup = x
    grad_y_inf = torch.zeros_like(h)

    sup = x >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def relu(h: Tensor, lambd: Tensor, *arg) -> Tuple[Tensor, Tensor]:
    y_sup = lambd * h
    y_inf = torch.zeros_like(h)

    grad_y_sup = lambd
    grad_y_inf = torch.zeros_like(h)

    sup = h >= 0
    return (
        torch.where(sup, y_sup, y_inf),
        torch.where(sup, grad_y_sup, grad_y_inf)
    )


def get_penalty_func(name):
    all_penalties = {
        "p2": p2,
        "p3": p3,
        "phr": phr,
    }

    return all_penalties[name]
