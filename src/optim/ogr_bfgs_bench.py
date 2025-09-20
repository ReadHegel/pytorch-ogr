from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional

import torch
from torch import Tensor

from src.optim.linesearch import Linesearch


try:
    from .OGR import OGR
except Exception:
    from OGR import OGR

try:
    from .BFGS import BFGS
except Exception:
    from BFGS import BFGS

SEED = 42


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


# ===== Registry (bounds + meta) =====
@dataclass(frozen=True)
class FunMeta:
    fn: Callable[[Tensor], Tensor]
    bounds: Tuple[Tuple[float, float], ...]
    global_min_f: float
    name: str


def repeat_bounds(b: Tuple[float, float], n: int):
    return tuple([b] * n)


REG_ALL: Dict[str, FunMeta] = {
    "sphere": FunMeta(sphere, repeat_bounds((-5.0, 5.0), 2), 0.0, "Sphere"),
    "rosenbrock": FunMeta(rosenbrock, repeat_bounds((-2.0, 2.0), 2), 0.0, "Rosenbrock"),
    "rastrigin": FunMeta(rastrigin, repeat_bounds((-5.12, 5.12), 2), 0.0, "Rastrigin"),
    "ackley": FunMeta(ackley, repeat_bounds((-5.0, 5.0), 2), 0.0, "Ackley"),
    "griewank": FunMeta(griewank, repeat_bounds((-5.0, 5.0), 2), 0.0, "Griewank"),
    "schwefel": FunMeta(schwefel, repeat_bounds((-500.0, 500.0), 2), 0.0, "Schwefel"),
    "zakharov": FunMeta(zakharov, repeat_bounds((-5.0, 5.0), 2), 0.0, "Zakharov"),
    "himmelblau": FunMeta(himmelblau, ((-5.0, 5.0), (-5.0, 5.0)), 0.0, "Himmelblau"),
    "beale": FunMeta(beale, ((-4.5, 4.5), (-4.5, 4.5)), 0.0, "Beale"),
}

OGR_SETTINGS = {
    "rosenbrock": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "rastrigin": dict(lr=0.6, beta=0.2, max_step_norm=2.5),
    "schwefel": dict(lr=0.5, beta=0.2, max_step_norm=1.5),
    "ackley": dict(lr=0.5, beta=0.2, max_step_norm=1.5),
    "griewank": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "zakharov": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "himmelblau": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "beale": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "sphere": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
}


def sample_uniform(
    bounds: Tuple[Tuple[float, float], ...],
    device: torch.device,
    dtype: torch.dtype,
    seed=None,
) -> Tensor:
    lows = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    highs = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    u = torch.rand(
        lows.size(),
        dtype=lows.dtype,
        layout=lows.layout,
        device=lows.device,
        generator=generator,
    )
    return lows + u * (highs - lows)


def clamp_inplace(x: Tensor, bounds: Tuple[Tuple[float, float], ...]) -> None:
    """Project x back into box bounds (in-place)."""
    lows = torch.tensor([b[0] for b in bounds], device=x.device, dtype=x.dtype)
    highs = torch.tensor([b[1] for b in bounds], device=x.device, dtype=x.dtype)
    x.data.copy_(torch.minimum(torch.maximum(x.data, lows), highs))


@dataclass
class RunCfg:
    dim: int = 2
    restarts: int = 8
    steps: int = 1000
    tol_grad: float = 1e-9
    seed: int = 123
    device: str = "cpu"
    dtype: torch.dtype = (
        torch.float64
    )  # float64 znacznie stabilniejsze dla (quasi-)Newton
    is_linesearch: bool = False


@dataclass
class Result:
    best_f: float
    best_x: Tensor
    iters: int
    time_s: float


def minimize_with_ogr(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    steps: int,
    tol_grad: float,
    bounds: Tuple[Tuple[float, float], ...],
    ogr_cfg: Dict[str, float],
) -> Result:
    x = x0.clone().detach().requires_grad_(True)
    opt = OGR(
        [x],
        lr=ogr_cfg.get("lr", 0.5),
        beta=ogr_cfg.get("beta", 0.2),
        eps=1e-12,
        linesearch=ogr_cfg["linesearch"],
        maximize=False,
        max_step_norm=ogr_cfg.get("max_step_norm", 1.0),
    )
    t0 = time.time()
    best_f = math.inf
    best_x = x.detach().clone()
    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        f = fn(x)
        f.backward()
        opt.step()
        clamp_inplace(x, bounds)  # BOXING
        gnorm = x.grad.detach().norm().item() if x.grad is not None else float("inf")
        if f.item() < best_f:
            best_f = float(f.item())
            best_x = x.detach().clone()
        if gnorm < tol_grad:
            break
    return Result(best_f, best_x, it, time.time() - t0)


def minimize_with_bfgs(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    lr: float,
    steps: int,
    tol_grad: float,
    bounds: Tuple[Tuple[float, float], ...],
    bfgs_cfg: Dict = {},
) -> Result:
    x = x0.clone().detach().requires_grad_(True)
    opt = BFGS([x], lr=lr, linesearch=bfgs_cfg["linesearch"])
    t0 = time.time()
    best_f = math.inf
    best_x = x.detach().clone()
    it = 0
    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        f = fn(x)
        f.backward()
        opt.step()
        clamp_inplace(x, bounds)  # BOXING
        gnorm = x.grad.detach().norm().item() if x.grad is not None else float("inf")
        if f.item() < best_f:
            best_f = float(f.item())
            best_x = x.detach().clone()
        if gnorm < tol_grad:
            break
    return Result(best_f, best_x, it, time.time() - t0)


def run_suite(cfg: RunCfg, names: Optional[List[str]] = None) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = cfg.dtype

    sel = names or list(REG_ALL.keys())
    print(f"Running on: {device}, dtype={dtype}\n")
    print(
        f"Dimensions: {cfg.dim}, Restarts: {cfg.restarts}, Steps: {cfg.steps}, tol_grad: {cfg.tol_grad}\n"
    )

    for name in sel:
        meta = REG_ALL[name]
        if (
            len(meta.bounds) == 2
            and cfg.dim != 2
            and name not in ("himmelblau", "beale")
        ):
            bounds = repeat_bounds(meta.bounds[0], cfg.dim)
        else:
            bounds = meta.bounds

        def wrap_fn(z: Tensor) -> Tensor:
            return meta.fn(z)

        linesearch = Linesearch(wrap_fn) if cfg.is_linesearch else None

        print(f"=== {meta.name} ({name}) ===")

        ogr_cfg = OGR_SETTINGS.get(name, dict(lr=0.5, beta=0.2, max_step_norm=1.0))
        bfgs_cfg = {}

        ogr_cfg["linesearch"] = linesearch
        bfgs_cfg["linesearch"] = linesearch

        best_ogr = math.inf
        ogr_err = None

        local_restarts = 30 if name == "schwefel" else cfg.restarts
        local_steps = 2000 if name == "schwefel" else cfg.steps

        for i in range(local_restarts):
            x0 = sample_uniform(bounds, device, dtype, seed=SEED + i)
            try:
                res = minimize_with_ogr(
                    wrap_fn, x0, local_steps, cfg.tol_grad, bounds, ogr_cfg
                )
                best_ogr = min(best_ogr, res.best_f)
            except Exception as e:
                ogr_err = str(e)
                break
        if ogr_err is None:
            print(f"Best OGR   : {best_ogr:.6e}")
        else:
            print(f"Best OGR   : ERROR ({ogr_err})")

        best_bfgs = math.inf

        for i in range(local_restarts):
            x0 = sample_uniform(bounds, device, dtype, seed=SEED + i)
            res = minimize_with_bfgs(
                wrap_fn,
                x0,
                ogr_cfg["lr"],
                local_steps,
                cfg.tol_grad,
                bounds,
                bfgs_cfg=bfgs_cfg,
            )
            best_bfgs = min(best_bfgs, res.best_f)
        print(f"Best BFGS  : {best_bfgs:.6e}")
        print(f"Target f*  : {meta.global_min_f:.6e}\n")


if __name__ == "__main__":
    CONFIG = RunCfg(
        dim=10,
        restarts=5,
        steps=2000,
        tol_grad=1e-8,
        seed=42,
        device="cpu",
        dtype=torch.float64,
        is_linesearch=True,
    )
    run_suite(CONFIG)
