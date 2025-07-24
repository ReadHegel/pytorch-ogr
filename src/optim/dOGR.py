"""
    This file contains implementation of the diagonal OGR function
    optimalization algorithm from paper: https://arxiv.org/pdf/1901.11457
"""

from typing import Union, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torch import Tensor


class dOGR(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        beta: float = 0.99,
        maximize: bool = False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "maximize": maximize,
            "beta": beta,
        }

        super().__init__(params, defaults)

    def _init_group(
        self,
        group, 
        params, 
        grads,
    ):
        has_sparse_grad = False
        
        for p in group["params"]:
            if p.grad is not None: 
                params.append(p)
                grads.append(p.grad)

            if p.grad.is_sparse:
                has_sparse_grad = True

        return has_sparse_grad

    def step(self, closure=None) -> Union[None, float]:
        loss: Union[None, float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []

            has_sparse_grad = self._init_group(
                group, params, grads, 
            )

            dogr(
                params,
                grads,
                lr=group["lr"],
                beta=group["beta"],
                maximize=group["maximize"],
            )

        return loss


def dogr(
    params: list[Tensor],
    grads: list[Tensor],
    lr: float,
    beta: float,
    mean_params: list[Tensor],
    mean_grads: list[Tensor],
    d_params_params: list[Tensor], 
    d_grads_params: list[Tensor],
    maximize: bool,
):
    for i, param in enumerate(params): 
        grad = grads[i] if not maximize else -grads[i]

        # Calculate means 
        # In comments we include the formulas with symbols coresponing 
        # to ones used in the paper https://arxiv.org/pdf/1901.11457
        
        # mean_theta = beta * theta + (1 - beta) mean_theta
        mean_params[i] += beta * (param - mean_params[i])
        
        # mean_g = beta * g + (1 - beta) * g
        mean_grads[i] += beta * (grad - mean_grads[i])
        
        # mean

        # d_theta_theta = (1 - beta) * theta_hat_mean**2 + beta * theta_hat**2
        d_params_params[i] += beta * (
            (param - mean_params[i])**2 - d_params_params[i]
        )

        # d_g_theta = (1 - beta) * d_g_theta + beta * g_hat_theta_hat 
        d_grads_params[i] += beta * (
            (grad - mean_grads[i]) * (param - mean_params[i]) - 
            d_grads_params[i]
        )

        param.add_(

        )

        # simple SGD
        param.add_(grad, alpha=-lr)
        
