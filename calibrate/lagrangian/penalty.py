import torch


def p2(h: torch.Tensor, lambd: torch.Tensor, rho: torch.Tensor):
    if lambd.ndim == 1:
        lambd = lambd.unsqueeze(dim=0).expand(h.shape)

    if isinstance(rho, torch.Tensor) and rho.ndim == 1:
        rho = rho.unsqueeze(dim=0).expand(h.shape)

    y_sup = lambd * h + lambd * rho * (h ** 2) + 1 / 6 * (rho ** 2) * (h ** 3)
    y_inf = lambd * h / (1 - rho * h.clamp(max=0))

    grad_y_sup = lambd + 2 * lambd * rho * h + 1 / 2 * (rho ** 2) * (h ** 2)
    grad_y_inf = lambd / (1 - rho * h.clamp(max=0)) ** 2

    return (
        torch.where(h >= 0, y_sup, y_inf),
        torch.where(h >= 0, grad_y_sup, grad_y_inf)
    )


def p3(h: torch.Tensor, lambd: torch.Tensor, rho: torch.Tensor):
    if lambd.ndim == 1:
        lambd = lambd.unsqueeze(dim=0).expand(h.shape)

    if isinstance(rho, torch.Tensor) and rho.ndim == 1:
        rho = rho.unsqueeze(dim=0).expand(h.shape)

    y_sup = lambd * h + lambd * rho * (h ** 2)
    y_inf = lambd * h / (1 - rho * h.clamp(max=0))

    grad_y_sup = lambd + 2 * lambd * rho * h
    grad_y_inf = lambd / (1 - rho * h.clamp(max=0)) ** 2

    return (
        torch.where(h >= 0, y_sup, y_inf),
        torch.where(h >= 0, grad_y_sup, grad_y_inf)
    )


def phr(h: torch.Tensor, lambd: torch.Tensor, rho: torch.Tensor):
    if lambd.ndim == 1:
        lambd = lambd.unsqueeze(dim=0).expand(h.shape)

    if isinstance(rho, torch.Tensor) and rho.ndim == 1:
        rho = rho.unsqueeze(dim=0).expand(h.shape)

    x = lambd + rho * h
    y_sup = 1 / (2 * rho) * (x ** 2 - lambd ** 2)
    y_inf = - 1 / (2 * rho) * (lambd ** 2)

    grad_y_sup = x
    grad_y_inf = torch.zeros_like(h)

    return (
        torch.where(x >= 0, y_sup, y_inf),
        torch.where(x >= 0, grad_y_sup, grad_y_inf)
    )


def relu(h: torch.Tensor, lambd: torch.Tensor, *arg):
    if lambd.ndim == 1:
        lambd = lambd.unsqueeze(dim=0).expand(h.shape)

    y_sup = lambd * h
    y_inf = torch.zeros_like(h)

    grad_y_sup = lambd
    grad_y_inf = torch.zeros_like(h)

    return (
        torch.where(h >= 0, y_sup, y_inf),
        torch.where(h >= 0, grad_y_sup, grad_y_inf)
    )


def get_penalty_func(name):
    all_penalties = {
        "p2": p2,
        "p3": p3,
        "phr": phr,
        "linear": linear,
    }

    return all_penalties[name]
