"""
OGR optimizer (stabilized)
Paper: https://arxiv.org/pdf/1901.11457
"""

from typing import Union, Optional, List, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable

try:
    from .utils import restore_tensor_list, flat_tensor_list  # package-style
except Exception:
    from utils import restore_tensor_list, flat_tensor_list  # local fallback

from .linesearch import Linesearch

# FOR DEBUG
# def _safe_eigh(H: Tensor, jitter: float = 1e-8, max_tries: int = 6) -> Tuple[Tensor, Tensor]:
#     I = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
#     for k in range(max_tries):
#         Hs = 0.5 * (H + H.T)
#         try:
#             L, Q = torch.linalg.eigh(Hs)
#             return L, Q
#         except Exception:
#             H = Hs + (jitter * (10.0 ** k)) * I
#     Hs = 0.5 * (H + H.T) + (jitter * (10.0 ** (max_tries - 1))) * I
#     return torch.linalg.eigh(Hs)


def _get_hessian(
    mean_params: Tensor,
    mean_grads: Tensor,
    mean: float,
    d_params_param: Tensor,
    d_grads_param: Tensor,
    eps: float,
) -> Tensor:
    size = mean_params.shape[0]
    A = d_params_param
    B = d_grads_param + d_grads_param.T

    jitter = torch.eye(A.shape[0], device=A.device, dtype=A.dtype) * eps
    L, O = torch.linalg.eigh(A + jitter)

    Li = L.unsqueeze(0).expand(size, size)
    Lj = L.unsqueeze(1).expand(size, size)
    denom = Li + Lj

    if (denom == 0).any():
        raise RuntimeError("Zero eighen value in covariant matrix")

    BO = O.T @ B @ O
    H = O @ (BO / denom) @ O.T
    return 0.5 * (H + H.T)


def _get_H_inv_regular(H: Tensor, clip_eigh: Optional[float], eps: float) -> Tensor:
    L, Q = torch.linalg.eigh(H)

    L_abs = L.abs()
    if clip_eigh is not None:
        L_abs = torch.maximum(torch.ones_like(L_abs) * clip_eigh, L_abs)

    inv_diag = 1.0 / L_abs
    H_inv = Q @ torch.diag(inv_diag) @ Q.T
    return 0.5 * (H_inv + H_inv.T)


class OGR(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = (1 / 1.5),
        clip_eigen: Optional[float] = None,
        beta: float = 0.30,
        eps: float = 1e-12,
        linesearch: Linesearch = None,
        maximize: bool = False,
        differentiable: bool = False,
        hybrid_clipping: bool = False,
        neg_clip_val: Optional[float] = None,
        max_step_norm: Optional[float] = 1.0,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "clip_eigen": clip_eigen,
            "maximize": maximize,
            "beta": beta,
            "linesearch": linesearch,
            "eps": eps,
            "differentiable": differentiable,
            "hybrid_clipping": hybrid_clipping,
            "neg_clip_val": neg_clip_val,
            "max_step_norm": max_step_norm,
        }
        super().__init__(params, defaults)

    def _init_group(self, group):
        has_sparse_grad = False
        params: List[Tensor] = []
        grads: List[Tensor] = []

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

        params_flat = flat_tensor_list(params)
        grads_flat = flat_tensor_list(grads)
        size = params_flat.shape[0]

        if "first_time" not in group:
            group["first_time"] = True
            group["params_size"] = [param.numel() for param in params]
            group["params_shape"] = [param.shape for param in params]

            group["mean_params"] = torch.zeros_like(params_flat)
            group["mean_grads"] = torch.zeros_like(params_flat)
            group["mean"] = 0.0
            group["d_params_params"] = torch.eye(
                size, device=params_flat.device, dtype=params_flat.dtype
            )
            group["d_grads_params"] = torch.eye(
                size, device=params_flat.device, dtype=params_flat.dtype
            )
        elif group["first_time"]:
            group["first_time"] = False

        return (
            has_sparse_grad,
            params_flat,
            grads_flat,
            group["mean_params"],
            group["mean_grads"],
            group["mean"],
            group["d_params_params"],
            group["d_grads_params"],
            group["first_time"],
        )

    def _save_to_group(self, group, **kwargs):
        for key, value in kwargs.items():
            group[key] = value
        return group

    def get_H(self):
        g = self.param_groups[0]
        return _get_hessian(
            g["mean_params"],
            g["mean_grads"],
            g["mean"],
            g["d_params_params"],
            g["d_grads_params"],
            g["eps"],
        )

    def get_H_inv(self):
        return torch.inverse(self.get_H())

    @_use_grad_for_differentiable
    def step(self, closure=None) -> Union[None, float]:
        loss: Union[None, float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            raise ValueError("Only one parameter group is supported")

        for group in self.param_groups:
            (
                _,
                params,
                grads,
                mean_params,
                mean_grads,
                means,
                d_params_params,
                d_grads_params,
                first_time,
            ) = self._init_group(group=group)

            (
                update_values,
                mean_params,
                mean_grads,
                d_params_params,
                d_grads_params,
                means,
            ) = ogr(
                params,
                grads,
                first_time=first_time,
                lr=group["lr"],
                clip_eigen=group["clip_eigen"],
                beta=group["beta"],
                eps=group["eps"],
                linesearch=group["linesearch"],
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                means=means,
                maximize=group["maximize"],
                hybrid_clipping=group["hybrid_clipping"],
                neg_clip_val=group["neg_clip_val"],
                max_step_norm=group["max_step_norm"],
            )

            update_values = restore_tensor_list(
                update_values, group["params_size"], group["params_shape"]
            )

            self._save_to_group(
                group,
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                means=means,
            )

            for p, update_value in zip(group["params"], update_values):
                p.data.add_(update_value)

        return loss


def _get_new_moving_average(
    param: Tensor,
    grad: Tensor,
    first_time: bool,
    beta: float,
    mean_param: Tensor,
    mean_grad: Tensor,
    d_params_param: Tensor,
    d_grads_param: Tensor,
    mean: float,
):
    # ruchome średnie + kowariancje
    local_mean_params = mean_param + beta * (param - mean_param)
    local_mean_grads = mean_grad + beta * (grad - mean_grad)
    local_mean = mean * beta + 1.0

    local_d_params_params = d_params_param + beta * (
        torch.outer(param - local_mean_params, param - local_mean_params)
        - d_params_param
    )
    local_d_grads_params = d_grads_param + beta * (
        torch.outer(grad - local_mean_grads, param - local_mean_params) - d_grads_param
    )

    return (
        local_mean_params,
        local_mean_grads,
        local_mean,
        local_d_params_params,
        local_d_grads_params,
    )


def ogr(
    params_flat: Tensor,
    grads_flat: Tensor,
    first_time: bool,
    lr: float,
    clip_eigen: Optional[float],
    beta: float,
    eps: float,
    linesearch: Optional[Linesearch],
    mean_params: Tensor,
    mean_grads: Tensor,
    d_params_params: Tensor,
    d_grads_params: Tensor,
    means: float,
    maximize: bool,
    hybrid_clipping: bool,
    neg_clip_val: Optional[float],
    max_step_norm: Optional[float],
):
    grads_flat = grads_flat if not maximize else -grads_flat

    (
        new_mean_params,
        new_mean_grads,
        new_mean,
        new_d_params_params,
        new_d_grads_params,
    ) = _get_new_moving_average(
        param=params_flat,
        grad=grads_flat,
        first_time=first_time,
        beta=beta,
        mean_param=mean_params,
        mean_grad=mean_grads,
        d_params_param=d_params_params,
        d_grads_param=d_grads_params,
        mean=means,
    )

    mean_params = new_mean_params
    mean_grads = new_mean_grads
    d_params_params = new_d_params_params
    d_grads_params = new_d_grads_params
    means = new_mean

    H = _get_hessian(
        mean_params,
        mean_grads,
        means,
        d_params_param=d_params_params,
        d_grads_param=d_grads_params,
        eps=eps,
    )

    H_inv = _get_H_inv_regular(H, clip_eigh=clip_eigen, eps=eps)

    update_values = - (H_inv @ grads_flat)

    # perform linesearch
    if linesearch is not None:
        update_values = linesearch.perform_search(params_flat, update_values, grads_flat)
    else: 
        update_values *= lr 

    if max_step_norm is not None:
        step_norm = update_values.norm()
        if step_norm > max_step_norm:
            update_values = update_values * (max_step_norm / (step_norm + 1e-12))

    if torch.isnan(params_flat).any():
        raise RuntimeError(
            "Wykryto wartość NaN w parametrach - trening jest niestabilny."
        )

    return (
        update_values,
        mean_params,
        mean_grads,
        d_params_params,
        d_grads_params,
        means,
    )
