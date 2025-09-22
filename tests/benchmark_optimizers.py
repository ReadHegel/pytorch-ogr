from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional

import torch
from torch import Tensor

from src.optim.linesearch import Linesearch
import os


try:
    from src.optim.OGR import OGR
except Exception:
    from OGR import OGR

try:
    from src.optim.BFGS import BFGS
except Exception:
    from BFGS import BFGS

import matplotlib.pyplot as plt

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



def run_benchmarks(dim=2, steps=2000, tol_grad=1e-8, seed=42, device="cpu", dtype=torch.float64, out_dir="plots"):
    torch.manual_seed(seed)
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)

    results_all = {}

    for name, meta in REG_ALL.items():
        print(f"\n>>> Running benchmarks for {meta.name} ({name})")
        if len(meta.bounds) == 2 and dim != 2 and name not in ("himmelblau", "beale"):
            bounds = tuple([meta.bounds[0]] * dim)
        else:
            bounds = meta.bounds

        def wrap_fn(z: Tensor) -> Tensor:
            return meta.fn(z)

        ogr_cfg = OGR_SETTINGS.get(name, dict(lr=0.5, beta=0.2, max_step_norm=1.0))
        ogr_cfg_base = dict(ogr_cfg)

        losses = {"OGR": [], "OGR+LS": [], "BFGS": [], "BFGS+LS": []}

        for i in range(200):
            if i % 50 == 0 and i > 0:
                print(f"  Processed {i}/200 restarts...")
            x0 = sample_uniform(bounds, device, dtype, seed=seed + i)

            # OGR
            r = minimize_with_ogr(wrap_fn, x0, steps, tol_grad, bounds, {**ogr_cfg_base, "linesearch": None})
            losses["OGR"].append(r.best_f)

            # OGR + line search
            r = minimize_with_ogr(wrap_fn, x0, steps, tol_grad, bounds, {**ogr_cfg_base, "linesearch": Linesearch(wrap_fn)})
            losses["OGR+LS"].append(r.best_f)

            # BFGS
            r = minimize_with_bfgs(wrap_fn, x0, ogr_cfg_base["lr"], steps, tol_grad, bounds, {"linesearch": None})
            losses["BFGS"].append(r.best_f)

            # BFGS + line search
            r = minimize_with_bfgs(wrap_fn, x0, ogr_cfg_base["lr"], steps, tol_grad, bounds, {"linesearch": Linesearch(wrap_fn)})
            losses["BFGS+LS"].append(r.best_f)

        # sort each optimizer’s losses
        for k in losses:
            losses[k] = sorted(losses[k])

        results_all[name] = (meta, losses)
        print(f"  Finished {meta.name}, best results: OGR={min(losses['OGR']):.2e}, BFGS={min(losses['BFGS']):.2e}")

        # Save plot immediately after each function
        plt.figure(figsize=(10, 6))
        for method, vals in losses.items():
            plt.scatter(range(1, len(vals) + 1), vals, label=method, s=15)
        plt.title(f"{meta.name} ({name})")
        plt.xlabel("Start point index (sorted)")
        plt.ylabel("Loss (log scale)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{name}.png")
        plt.savefig(fname)
        plt.close()
        print(f"  Saved plot to {fname}")

    return results_all


if __name__ == "__main__":
    run_benchmarks(dim=10, steps=2000, tol_grad=1e-8, seed=42, device="cpu", dtype=torch.float64, out_dir="plots_2000steps")