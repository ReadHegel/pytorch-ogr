from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional

import torch
from torch import Tensor

import itertools
import os
import numpy as np

import matplotlib.pyplot as plt

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
PLOT_PATH = "plots"


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
    print_trace: bool = False


@dataclass
class Result:
    best_f: float = math.inf
    best_x: Tensor = torch.tensor([])
    iters: int = -1
    time_s: float = -1
    points: List[Tensor] = None

    def __lt__(self, other: "Result") -> bool:
        return self.best_f < other.best_f


def minimize_with_ogr(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    steps: int,
    tol_grad: float,
    bounds: Tuple[Tuple[float, float], ...],
    ogr_cfg: Dict[str, float],
) -> Result:
    return minimize(
        fn,
        x0,
        steps=steps,
        tol_grad=tol_grad,
        bounds=bounds,
        cfg=ogr_cfg,
        opt_class=OGR,
    )


def minimize_with_bfgs(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    steps: int,
    tol_grad: float,
    bounds: Tuple[Tuple[float, float], ...],
    bfgs_cfg: Dict = {},
) -> Result:
    return minimize(
        fn,
        x0,
        steps=steps,
        tol_grad=tol_grad,
        bounds=bounds,
        cfg=bfgs_cfg,
        opt_class=BFGS,
    )


def _minimize_with_opt(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    opt,
    steps: int,
    tol_grad: float,
    bounds: Tuple[Tuple[float, float], ...],
):
    t0 = time.time()
    best_f = math.inf
    best_x = x.detach().clone()
    it = 0
    points = [x.detach().clone()]

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        f = fn(x)
        f.backward()
        opt.step()

        clamp_inplace(x, bounds)  # BOXING

        points.append(x.detach().clone())

        gnorm = x.grad.detach().norm().item() if x.grad is not None else float("inf")
        if f.item() < best_f:
            best_f = float(f.item())
            best_x = x.detach().clone()
        if gnorm < tol_grad:
            break

    return Result(best_f, best_x, it, time.time() - t0, points)


def minimize(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    steps: int,
    tol_grad: float,
    bounds: Tuple[Tuple[float, float], ...],
    cfg: Dict,
    opt_class,
):
    x = x0.clone().detach().requires_grad_(True)
    if opt_class == OGR:
        opt = OGR(
            [x],
            lr=cfg.get("lr", 0.5),
            beta=cfg.get("beta", 0.2),
            eps=1e-12,
            linesearch=cfg["linesearch"],
            maximize=False,
            max_step_norm=cfg.get("max_step_norm", 1.0),
        )
    else:
        opt = BFGS([x], lr=cfg["lr"], linesearch=cfg["linesearch"])

    return _minimize_with_opt(
        fn=fn, x=x, opt=opt, steps=steps, tol_grad=tol_grad, bounds=bounds
    )


TRACES_PATH = os.path.join(PLOT_PATH, "traces")


def print_optimization_path(
    f,
    points: list[Tensor],
    bounds: Tuple[Tuple, Tuple],
    name: str = "Default name",
) -> None:
    if len(points) != 0 and points[0].shape != (2,):
        print(points[0].shape)
        raise Exception(
            "Only two dimentional plots are supported by 'print_optimization_path'"
        )

    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i, j in itertools.product(range(X.shape[0]), range(Y.shape[1])):
        inp = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
        Z[i, j] = f(inp).item()

    plt.contourf(X, Y, Z, levels=50, cmap="viridis")

    points_np = torch.stack(points).numpy()
    plt.scatter(
        points_np[:, 0],
        points_np[:, 1],
        s=7,
        c=np.arange(len(points_np)),
        cmap="magma",
    )

    plt.title(name)
    plt.savefig(os.path.join(TRACES_PATH, name))
    plt.close()


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
        bfgs_cfg["lr"] = ogr_cfg["lr"]

        best_ogr_res = Result()
        ogr_err = None

        local_restarts = 30 if name == "schwefel" else cfg.restarts
        local_steps = 2000 if name == "schwefel" else cfg.steps

        # OGR optimization
        for i in range(local_restarts):
            x0 = sample_uniform(bounds, device, dtype, seed=SEED + i)
            try:
                res = minimize_with_ogr(
                    wrap_fn, x0, local_steps, cfg.tol_grad, bounds, ogr_cfg
                )
                best_ogr_res = min(best_ogr_res, res)
            except Exception as e:
                ogr_err = str(e)
                break
        if ogr_err is None:
            print(f"Best OGR   : {best_ogr_res.best_f:.6e}")
        else:
            print(f"Best OGR   : ERROR ({ogr_err})")

        if cfg.print_trace:
            try:
                print_optimization_path(
                    f=wrap_fn,
                    points=best_ogr_res.points,
                    bounds=bounds,
                    name=f"OGR {name} opt trace",
                )
            except Exception as e:
                print(f"didnt print trace because {str(e)}")

        # BFGS optmization
        best_bfgs_res = Result()

        for i in range(local_restarts):
            x0 = sample_uniform(bounds, device, dtype, seed=SEED + i)
            res = minimize_with_bfgs(
                wrap_fn,
                x0,
                local_steps,
                cfg.tol_grad,
                bounds,
                bfgs_cfg=bfgs_cfg,
            )
            best_bfgs_res = min(best_bfgs_res, res)
        print(f"Best BFGS  : {best_bfgs_res.best_f:.6e}")
        print(f"Target f*  : {meta.global_min_f:.6e}\n")

        if cfg.print_trace:
            try:
                print_optimization_path(
                    f=wrap_fn,
                    points=best_bfgs_res.points,
                    bounds=bounds,
                    name=f"BFGS {name} opt trace",
                )
            except Exception as e:
                print(f"didnt print trace because {str(e)}")


if __name__ == "__main__":
    CONFIG = RunCfg(
        dim=2,
        restarts=5,
        steps=2000,
        tol_grad=1e-8,
        seed=42,
        device="cpu",
        dtype=torch.float64,
        is_linesearch=False,
        print_trace=True,
    )
    run_suite(CONFIG)
