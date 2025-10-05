import torch
from torch import Tensor


# ===== Test functions (x: 1-D tensor) =====
def sphere(x: Tensor) -> Tensor:
    return (x * x).sum()


def rosenbrock(x: Tensor, a: float = 1.0, b: float = 100.0) -> Tensor:
    return (b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2).sum()


def rastrigin(x: Tensor, A: float = 10.0) -> Tensor:
    n = x.numel()
    return A * n + (x * x - A * torch.cos(2 * math.pi * x)).sum()


def ackley(x: Tensor) -> Tensor:
    n = x.numel()
    s1 = torch.sqrt((x * x).sum() / n)
    s2 = torch.cos(2 * math.pi * x).sum() / n
    return -20.0 * torch.exp(-0.2 * s1) - torch.exp(s2) + 20.0 + math.e


def griewank(x: Tensor) -> Tensor:
    n = x.numel()
    sum_term = (x * x).sum() / 4000.0
    i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    prod_term = torch.cos(x / torch.sqrt(i)).prod()
    return sum_term - prod_term + 1.0


def schwefel(x: Tensor) -> Tensor:
    return 418.9829 * x.numel() - (x * torch.sin(torch.sqrt(torch.abs(x)))).sum()


def zakharov(x: Tensor) -> Tensor:
    i = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    term1 = (x * x).sum()
    term2 = (0.5 * i * x).sum()
    return term1 + term2**2 + term2**4


# 2D only
def himmelblau(x: Tensor) -> Tensor:
    assert x.numel() == 2
    X, Y = x[0], x[1]
    return (X * X + Y - 11) ** 2 + (X + Y * Y - 7) ** 2


def beale(x: Tensor) -> Tensor:
    assert x.numel() == 2
    X, Y = x[0], x[1]
    return (
        (1.5 - X + X * Y) ** 2
        + (2.25 - X + X * Y**2) ** 2
        + (2.625 - X + X * Y**3) ** 2
    )
