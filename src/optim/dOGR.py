"""
This file contains implementation of the diagonal OGR function
optimalization algorithm from paper: https://arxiv.org/pdf/1901.11457
"""

from typing import Union, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torch import Tensor


class dOGR(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        beta: float = 0.50,
        eps: float = 1e-4,
        maximize: bool = False,
        differentiable: bool = False
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
        }

        super().__init__(params, defaults)

    def _init_group(
        self,
        group,
        params,
        grads,
        mean_params: list[Tensor],
        mean_grads: list[Tensor],
        d_params_params: list[Tensor],
        d_grads_params: list[Tensor],
        mean: list[Tensor],
    ):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)

            if p.grad.is_sparse:
                has_sparse_grad = True

            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:

                def init_as_zero(name: str):
                    state[name] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                init_as_zero("mean_params")
                init_as_zero("mean_grads")
                init_as_zero("d_params_params")
                init_as_zero("d_grads_params")
                init_as_zero("mean")

            mean_params.append(state["mean_params"])
            mean_grads.append(state["mean_grads"])
            d_params_params.append(state["d_params_params"])
            d_grads_params.append(state["d_grads_params"])
            mean.append(state["mean"])

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None) -> Union[None, float]:
        loss: Union[None, float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            mean_params: list[Tensor] = []
            mean_grads: list[Tensor] = []
            d_params_params: list[Tensor] = []
            d_grads_params: list[Tensor] = []
            mean: list[Tensor] = []

            has_sparse_grad = self._init_group(
                group=group,
                params=params,
                grads=grads,
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                mean=mean,
            )

            dogr(
                params,
                grads,
                lr=group["lr"],
                beta=group["beta"],
                eps=group["eps"],
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                mean=mean,
                maximize=group["maximize"],
            )

        return loss


def dogr(
    params: list[Tensor],
    grads: list[Tensor],
    lr: float,
    beta: float,
    eps: float,
    mean_params: list[Tensor],
    mean_grads: list[Tensor],
    d_params_params: list[Tensor],
    d_grads_params: list[Tensor],
    mean: list[Tensor],
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

        # s = beta * s + 1
        mean[i] += (beta - 1) * mean[i] + 1

        # d_theta_theta = (1 - beta) * theta_hat_mean**2 + beta * theta_hat**2
        d_params_params[i] += beta * (
            (param - mean_params[i]) ** 2 - d_params_params[i]
        )

        # d_g_theta = (1 - beta) * d_g_theta + beta * g_hat_theta_hat
        d_grads_params[i] += beta * (
            (grad - mean_grads[i]) * (param - mean_params[i]) - d_grads_params[i]
        )

        # Diagonal Hessian
        H_inv = d_params_params[i] / (
            d_grads_params[i] + torch.sign(d_grads_params[i]) * eps
        )

        # Calculate p
        p = (mean_params[i] - H_inv * mean_grads[i]) / (mean[i] + eps)

        H_sign = torch.sign(H_inv)

        # theta = theta + sing(H)(p - param)
        # param.add_(H_sign * (p - param), alpha=lr)
       # param.add_(grad, alpha=-lr)        if isinstance(lr, Tensor):
