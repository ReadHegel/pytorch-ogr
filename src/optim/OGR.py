"""
This file contains implementation of the title OGR optimizer
optimalization algorithm from paper: https://arxiv.org/pdf/1901.11457
"""

from typing import Union, Optional

import torch
from torch.func import grad
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torch import Tensor

def flat_params(params: list[Tensor]):
    return torch.cat([
        param.flatten() for param in params
    ]) 

def param_restore_size(flat_params, params_sizes, params_shapes):
    # # print(flat_params.shape)
    # # print(params_shapes)
    # # print(params_sizes)
    splited = torch.split(flat_params, params_sizes)
    # # print(splited[0].shape)
    return [s.view(shape) for s, shape in zip(splited, params_shapes)]

class OGR(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        beta: float = 0.50,
        eps: float = 1e-12,
        maximize: bool = False,
        differentiable: bool = False,
        hybrid_clipping: bool = False,
        neg_clip_val: Optional[float] = None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "maximize": maximize,
            "beta": beta,
            "eps": eps,
            "differentiable": differentiable,
            "hybrid_clipping": hybrid_clipping,
            "neg_clip_val": neg_clip_val,
        }

        super().__init__(params, defaults)


    def _init_group(
        self,
        group,
    ):
        has_sparse_grad = False

        params = []
        grads = []

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)

            if p.grad.is_sparse:
                has_sparse_grad = True

        params_flat = flat_params(params)
        grads_flat = flat_params(grads)
        # print("__init_group: ", torch.isnan(grads_flat).any())
        size = params_flat.shape[0]

        if "first_time" not in group: 
            group["first_time"] = True 
        
            group["params_size"] = [param.numel() for param in params]
            group["params_shape"] = [param.shape for param in params]
        
            group["mean_params"] = torch.zeros_like(params_flat)
            group["mean_grads"] = torch.zeros_like(params_flat)
            group["mean"] = 0
            group["d_params_params"] = torch.zeros((size, size))
            group["d_grads_params"] = torch.zeros((size, size))

        elif group["first_time"]: 
            group["first_time"] = False

        return has_sparse_grad, params_flat, grads_flat, group["mean_params"], group["mean_grads"], group["mean"], group["d_params_params"], group["d_grads_params"], group["first_time"]

    def _save_to_group(self, group, **kwargs):
        for key, value in kwargs.items():
            group[key] = value
        return group

    @_use_grad_for_differentiable
    def step(self, closure=None) -> Union[None, float]:
        loss: Union[None, float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1: 
            raise ValueError("Only one parameter group is supported")

        for group in self.param_groups:
            _, params, \
            grads, \
            mean_params, \
            mean_grads, \
            means, \
            d_params_params, \
            d_grads_params, \
            first_time = self._init_group(
                group=group
            )

            # # print(params.shape)

            update_values, mean_params, \
            mean_grads, d_params_params, \
            d_grads_params, means \
            = ogr(
                params,
                grads,
                first_time=first_time,
                lr=group["lr"],
                beta=group["beta"],
                eps=group["eps"],
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                means=means, 
                maximize=group["maximize"],
                hybrid_clipping=group["hybrid_clipping"],
                neg_clip_val=group["neg_clip_val"],
            )

            update_values = param_restore_size(update_values, group["params_size"], group["params_shape"])

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
    if first_time: 
        beta = 1

    # Calculate means
    # In comments we include the formulas with symbols coresponing
    # to ones used in the paper https://arxiv.org/pdf/1901.11457

    # mean_theta = beta * theta + (1 - beta) mean_theta
    # print("in new arerage params", mean_param,
          # torch.isnan(mean_param).any(), param, torch.isnan(param).any())

    local_mean_params = mean_param + beta * (param - mean_param)

    # mean_g = beta * g + (1 - beta) * g
    # print("in new average", mean_grad,
          # torch.isnan(mean_grad).any(), grad, torch.isnan(grad).any())
    local_mean_grads = mean_grad + beta * (grad - mean_grad)

    local_mean = mean * beta + 1

    # d_theta_theta = (1 - beta) * theta_hat_mean**2 + beta * theta_hat**2
    local_d_params_params = d_params_param + beta * (
        torch.outer(param - local_mean_params, param - local_mean_params) - d_params_param
    )

    # d_g_theta = (1 - beta) * d_g_theta + beta * g_hat_theta_hat
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


def _get_hessian(
    mean_params: Tensor, 
    mean_grads: Tensor,
    mean: float,
    d_params_param: Tensor,
    d_grads_param: Tensor,
    eps: float,
):

    size = mean_params.shape[0]

    # print("mean_grads, mean_params", mean_grads, mean_params)
    A = d_params_param - mean_params.unsqueeze(1) @ mean_params.unsqueeze(0)
    B = d_grads_param + d_grads_param.T 
    # print("A in fu", A)
    A = A + A.T

    L, O = torch.linalg.eigh(A)

    # print("L in fu", L, torch.min(L))

    H = O @ (O.T @ B @ O) / \
        (   
            L.unsqueeze(0).expand(size,size) + 
            L.unsqueeze(1).expand(size,size) 
            # torch.maximum(
            #     L.unsqueeze(0).expand(size,size) + 
            #     L.unsqueeze(1).expand(size,size), 
            #     torch.ones((size,size)) * eps
            # )
        ) @ O.T

    return H


def ogr(
    params_flat: Tensor,
    grads_flat: Tensor,
    first_time: bool,
    lr: float,
    beta: float,
    eps: float,
    mean_params: Tensor,
    mean_grads: Tensor,
    d_params_params: Tensor,
    d_grads_params: Tensor,
    means: float,
    maximize: bool,
    hybrid_clipping: bool,
    neg_clip_val: Optional[float],
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

    # print("H", H)

    L, Q = torch.linalg.eigh(H)

    # print("L, Q", L, Q)
    #
    # print("max grad", torch.max(torch.abs(mean_grads)))
    # print("max Q", torch.max(torch.abs(Q)))
    update_values = - (1 / 1.5) * Q @ (
            torch.diag(
                1 / torch.maximum(torch.ones_like(L) * 20, torch.abs(L))
            )
        ) @ Q.T @ mean_grads 
   
    # print(
    #     "L", L,
    #     torch.maximum(torch.ones_like(L) * 10, torch.abs(L))
    # )
    # print(
    # )
    # print("max update values", torch.max(torch.abs(update_values)))
    
    # update_values = torch.clip(
    #     update_values, 
    #     min = -2,
    #     max = 2,
    # )

    # print("update_values", update_values)

    # for debug
    if torch.isnan(params_flat).any():
        raise RuntimeError(
            "Wykryto wartość NaN w parametrach - trening jest niestabilny."
        )

    return update_values, mean_params, \
            mean_grads, d_params_params, \
            d_grads_params, means
   
