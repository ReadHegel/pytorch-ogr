from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt



from src.optim.OGR import OGR



from src.optim.BFGS import BFGS



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
    return term1 + term2 ** 2 + term2 ** 4

def himmelblau(x: Tensor) -> Tensor:
    assert x.numel() == 2
    X, Y = x[0], x[1]
    return (X * X + Y - 11) ** 2 + (X + Y * Y - 7) ** 2

def beale(x: Tensor) -> Tensor:
    assert x.numel() == 2
    X, Y = x[0], x[1]
    return (1.5 - X + X * Y) ** 2 + (2.25 - X + X * Y ** 2) ** 2 + (2.625 - X + X * Y ** 3) ** 2


from dataclasses import dataclass

@dataclass(frozen=True)
class FunMeta:
    fn: Callable[[Tensor], Tensor]
    bounds: Tuple[Tuple[float, float], ...]
    global_min_f: float
    name: str

def repeat_bounds(b: Tuple[float, float], n: int):
    return tuple([b] * n)

REG_ALL: Dict[str, FunMeta] = {
    "sphere":     FunMeta(sphere,    repeat_bounds((-5.0,   5.0),   2), 0.0, "Sphere"),
    "rosenbrock": FunMeta(rosenbrock,repeat_bounds((-2.0,   2.0),   2), 0.0, "Rosenbrock"),
    "rastrigin":  FunMeta(rastrigin, repeat_bounds((-5.12,  5.12),  2), 0.0, "Rastrigin"),
    "ackley":     FunMeta(ackley,    repeat_bounds((-5.0,   5.0),   2), 0.0, "Ackley"),
    "griewank":   FunMeta(griewank,  repeat_bounds((-5.0,   5.0),   2), 0.0, "Griewank"),
    "schwefel":   FunMeta(schwefel,  repeat_bounds((-500.0, 500.0), 2), 0.0, "Schwefel"),
    "zakharov":   FunMeta(zakharov,  repeat_bounds((-5.0,   5.0),   2), 0.0, "Zakharov"),
    "himmelblau": FunMeta(himmelblau,((-5.0, 5.0), (-5.0, 5.0)),      0.0, "Himmelblau"),
    "beale":      FunMeta(beale,     ((-4.5, 4.5), (-4.5, 4.5)),      0.0, "Beale"),
}

OGR_SETTINGS = {
    "rosenbrock": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "rastrigin":  dict(lr=0.6, beta=0.2, max_step_norm=2.5),
    "schwefel":   dict(lr=0.5, beta=0.2, max_step_norm=1.5),
    "ackley":     dict(lr=0.5, beta=0.2, max_step_norm=1.5),
    "griewank":   dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "zakharov":   dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "himmelblau": dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "beale":      dict(lr=0.5, beta=0.2, max_step_norm=1.0),
    "sphere":     dict(lr=0.5, beta=0.2, max_step_norm=1.0),
}

def sample_uniform(bounds: Tuple[Tuple[float, float], ...],
                   device: torch.device, dtype: torch.dtype) -> Tensor:
    lows  = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    highs = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)
    u = torch.rand_like(lows)
    return lows + u * (highs - lows)

def clamp_inplace(x: Tensor, bounds: Tuple[Tuple[float, float], ...]) -> None:
    lows  = torch.tensor([b[0] for b in bounds], device=x.device, dtype=x.dtype)
    highs = torch.tensor([b[1] for b in bounds], device=x.device, dtype=x.dtype)
    x.data.copy_(torch.minimum(torch.maximum(x.data, lows), highs))

@dataclass
class RunCfg:
    dim: int = 2
    restarts: int = 343   
    steps: int = 50    
    tol_grad: float = 1e-9
    seed: int = 42
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

def run_single_start(fn: Callable[[Tensor], Tensor],
                     x0: Tensor,
                     optimizer_kind: str,
                     steps: int,
                     ogr_cfg: dict,
                     bounds,
                     device: torch.device,
                     dtype: torch.dtype) -> float:
    """
    optimizer_kind: one of "OGR", "BFGS", "OGR_ls", "BFGS_ls"
    Returns final loss (float) after `steps` iterations starting from x0.
    """
    x = x0.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)

    if optimizer_kind == "OGR":
        opt = OGR([x], lr=ogr_cfg.get("lr", 0.5), beta=ogr_cfg.get("beta", 0.2),
                  eps=1e-12, differentiable=False, max_step_norm=ogr_cfg.get("max_step_norm", 1.0))
        use_ls = False
    elif optimizer_kind == "OGR_ls":
        opt = OGR([x], lr=ogr_cfg.get("lr", 0.5), beta=ogr_cfg.get("beta", 0.2),
                  eps=1e-12, differentiable=False, max_step_norm=ogr_cfg.get("max_step_norm", 1.0))
        use_ls = True
    elif optimizer_kind == "BFGS":
        opt = BFGS([x], lr=1.0)
        use_ls = False
    elif optimizer_kind == "BFGS_ls":
        opt = BFGS([x], lr=1.0)
        use_ls = True
    else:
        raise ValueError("Unknown optimizer kind")

    def closure_bfgs():
        opt.zero_grad() if hasattr(opt, "zero_grad") else None
        x.grad = None
        f = fn(x)
        f.backward()
        return f

    def closure_ogr():
        opt.zero_grad() if hasattr(opt, "zero_grad") else None
        x.grad = None
        f = fn(x)
        return f

    closure = closure_bfgs if optimizer_kind.startswith("BFGS") else closure_ogr

    for it in range(steps):
        if use_ls:  
            opt.step(closure=closure, use_line_search=use_ls)
        else:
            opt.step(closure=closure)
        clamp_inplace(x, bounds)

    with torch.no_grad():
        final = float(fn(x).item())
    return final

def benchmark_function(meta: FunMeta, cfg: RunCfg, ogr_cfg: dict) -> Dict[str, List[float]]:
    device = torch.device(cfg.device)
    dtype = cfg.dtype
    results: Dict[str, List[float]] = {"OGR": [], "BFGS": [], "OGR_ls": [], "BFGS_ls": []}

    torch.manual_seed(cfg.seed)
    starts = [sample_uniform(meta.bounds, device, dtype) for _ in range(cfg.restarts)]

    print(f"Benchmarking {meta.name} with {cfg.restarts} restarts, {cfg.steps} steps each...")

    for idx, x0 in enumerate(starts, 1):
        if idx % 50 == 0:
            print(f"  start {idx}/{cfg.restarts}")
        for kind in list(results.keys()):
            loss_final = run_single_start(fn=meta.fn, x0=x0, optimizer_kind=kind, bounds=meta.bounds,
                                         steps=cfg.steps, ogr_cfg=ogr_cfg, device=device, dtype=dtype)
            results[kind].append(loss_final)

    return results

def plot_sorted_results(meta_name: str, results: Dict[str, List[float]], out_prefix: str = "bench4"):
    plt.figure(figsize=(9,6))

    markers = {"OGR": "o", "BFGS": "s", "OGR_ls": "^", "BFGS_ls": "x"}
    colors = {"OGR": "tab:blue", "BFGS": "tab:orange", "OGR_ls": "tab:green", "BFGS_ls": "tab:red"}

    for name, vals in results.items():
        if len(vals) == 0:
            print(f"[WARN] {name} has no results!")
            continue

        arr = torch.tensor(vals, dtype=torch.float64)
        arr_sorted, _ = torch.sort(arr)
        arr_sorted_clamped = torch.clamp(arr_sorted, min=1e-12)

        print(f"\nResults for {meta_name} — {name}")
        print(f"  count: {len(arr_sorted)}")
        print(f"  min: {arr_sorted[0].item():.4e}, max: {arr_sorted[-1].item():.4e}")
        print(f"  first 10 sorted: {arr_sorted[:10].numpy()}")
        print(f"  last 10 sorted: {arr_sorted[-10:].numpy()}")

        xs = np.arange(1, len(arr_sorted) + 1)
        plt.scatter(xs, arr_sorted_clamped.numpy(), s=8, marker=markers.get(name,"o"),
                    color=colors.get(name,"k"), label=name, alpha=0.8)

    plt.yscale("log")
    plt.xlabel("sorted start index (1 = best)")
    plt.ylabel("final loss (sorted)")
    plt.title(f"Final sorted losses — {meta_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out_file = f"{out_prefix}_{meta_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot: {out_file}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    cfg = RunCfg(dim=2, restarts=343, steps=10, tol_grad=1e-9, seed=42, device="cpu", dtype=torch.float64)

    sel = list(REG_ALL.keys()) 
    print(f"Running benchmarks for functions: {sel}\n")

    for name in sel:
        meta = REG_ALL[name]
        if len(meta.bounds) == 2 and cfg.dim != 2 and name not in ("himmelblau", "beale"):
            bounds_local = repeat_bounds(meta.bounds[0], cfg.dim)
        else:
            bounds_local = meta.bounds

        ogr_cfg = OGR_SETTINGS.get(name, dict(lr=0.5, beta=0.2, max_step_norm=1.0))

        globals()["bounds_local"] = bounds_local

        results = benchmark_function(meta, cfg, ogr_cfg)
        plot_sorted_results(meta.name, results, out_prefix="bench4")
