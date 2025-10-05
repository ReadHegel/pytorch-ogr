from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from src.optim.BFGS import BFGS
from src.optim.linesearch import Linesearch
from src.optim.OGR import OGR

from .benchmark_functions import (
    ackley,
    beale,
    griewank,
    himmelblau,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
    zakharov,
)
from .plot_functions import HessianPlot, InvHessianPlot, Plot, PlotList, TracePlot
from .utils import (
    clamp_inplace,
    get_bp_hessian_from_loss,
    repeat_bounds,
    sample_uniform,
)

SEED = 42


@dataclass
class RunCfg:
    dim: int = 2
    restarts: int = 10
    steps: int = 100
    tol_grad: float = 1e-8
    seed: int = SEED
    device: str = "cpu"
    dtype: torch.dtype = (
        torch.float64
    )  # float64 znacznie stabilniejsze dla (quasi-)Newton
    is_linesearch: bool = False
    print_trace: bool = False
    print_hessian: bool = False


@dataclass
class Result:
    best_f: float = math.inf
    best_x: Tensor = torch.tensor([])
    iters: int = -1
    time_s: float = -1
    points: List[Tensor] = None
    hessian_real: list[Tensor] = None
    hessian_est: list[Tensor] = None
    hessian_inv_est: list[Tensor] = None

    def __lt__(self, other: "Result") -> bool:
        return self.best_f < other.best_f


# ===== Registry (bounds + meta) =====
@dataclass(frozen=True)
class FunMeta:
    fn: Callable[[Tensor], Tensor]
    bounds: Tuple[Tuple[float, float], ...]
    global_min_f: float
    name: str


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
    real_hessians = []
    est_hessians = [opt.get_H()]
    est_inv_hessians = [opt.get_H_inv()]

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        f = fn(x)
        f.backward()
        opt.step()

        # after step optain real gradient
        opt.zero_grad(set_to_none=True)
        f = fn(x)
        real_hessians.append(get_bp_hessian_from_loss(f, [x]))

        clamp_inplace(x, bounds)  # BOXING

        points.append(x.detach().clone())
        est_hessians.append(opt.get_H())
        est_inv_hessians.append(opt.get_H_inv())

        gnorm = x.grad.detach().norm().item() if x.grad is not None else float("inf")

        if f.item() < best_f:
            best_f = float(f.item())
            best_x = x.detach().clone()
        if gnorm < tol_grad:
            break

    return Result(
        best_f,
        best_x,
        it,
        time.time() - t0,
        points=points,
        hessian_real=real_hessians,
        hessian_est=est_hessians,
        hessian_inv_est=est_inv_hessians,
    )


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


def perform_multiround_optimization(
    method_name: str,
    experiment_name: str,
    plot_list: PlotList,
    local_restarts: int,
    bounds,
    device,
    dtype,
    wrap_fn,
    local_steps,
    tol_grad,
    cfg,
):
    best_res = Result()
    err = None

    minimize_f = None
    if method_name == "OGR":
        minimize_f = minimize_with_ogr
    elif method_name == "BFGS":
        minimize_f = minimize_with_bfgs
    else:
        raise RuntimeError(f"No such method as: {method_name}")

    for i in range(local_restarts):
        x0 = sample_uniform(bounds, device, dtype, seed=SEED + i)
        try:
            res = minimize_with_ogr(wrap_fn, x0, local_steps, tol_grad, bounds, cfg)
            best_res = min(best_res, res)
        except Exception as e:
            err = str(e)
            traceback.print_exc()
            break

    if err is None:
        print(f"Best {method_name}   : {best_res.best_f:.6e}")
    else:
        print(f"Best {method_name}   : ERROR ({err})")

    best_res.experiment_name = method_name + " " + experiment_name
    best_res.optimized_function = wrap_fn

    plot_list.print(best_res)

    return best_res


def run(cfg: RunCfg, plot_list: PlotList) -> None:
    device = torch.device(cfg.device)
    dtype = cfg.dtype

    print(f"Running on: {device}, dtype={dtype}\n")
    print(
        f"""Dimensions: {cfg.dim},
        Restarts: {cfg.restarts},
        Steps: {cfg.steps},
        tol_grad: {cfg.tol_grad}\n"""
    )

    for name in list(REG_ALL.keys()):
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

        local_restarts = 30 if name == "schwefel" else cfg.restarts
        local_steps = 2000 if name == "schwefel" else cfg.steps

        best_ogr_res = perform_multiround_optimization(
            method_name="OGR",
            experiment_name=f"opt of {name} function",
            plot_list=plot_list,
            local_restarts=local_restarts,
            bounds=bounds,
            dtype=dtype,
            wrap_fn=wrap_fn,
            local_steps=local_steps,
            tol_grad=cfg.tol_grad,
            cfg=ogr_cfg,
        )

        print(f"Best OGR  : {best_ogr_res.best_f:.6e}")
        print(f"Target f*  : {meta.global_min_f:.6e}\n")

        best_bfgs_res = perform_multiround_optimization(
            method_name="BFGS",
            experiment_name=f"opt of {name} function",
            plot_list=plot_list,
            local_restarts=local_restarts,
            bounds=bounds,
            dtype=dtype,
            wrap_fn=wrap_fn,
            local_steps=local_steps,
            tol_grad=cfg.tol_grad,
            cfg=bfgs_cfg,
        )

        print(f"Best BFGS  : {best_bfgs_res.best_f:.6e}")
        print(f"Target f*  : {meta.global_min_f:.6e}\n")


def setup_plots(config: RunCfg):
    plot_list = []
    if config.print_trace:
        plot_list.append(TracePlot)
    if config.print_hessian:
        plot_list.append(HessianPlot)
        plot_list.append(InvHessianPlot)

    return PlotList(plot_list)


def main(config: RunCfg):
    plot_list: PlotList = setup_plots(config)
    torch.manual_seed(config.seed)
    run(
        config,
        plot_list,
    )


if __name__ == "__main__":
    CONFIG = RunCfg()
    main(CONFIG)
