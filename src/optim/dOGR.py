"""
This file contains implementation of the diagonal OGR function
optimalization algorithm from paper: https://arxiv.org/pdf/1901.11457
"""

from typing import Union

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
        eps: float = 1e-8,
        sec_ord_lr: float = 0.5,
        gamma: float = 0.5,
        clip_min: float = 5, 
        clip_max: float = 5, 
        maximize: bool = False,
        differentiable: bool = False,
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
            "gamma": gamma, 
            "sec_ord_lr": sec_ord_lr, 
            "clip_min": clip_min, 
            "clip_max": clip_max, 
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
        mean_grads_gamma: list[Tensor],
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
                init_as_zero("mean_grads_gamma")
                init_as_zero("d_params_params")
                init_as_zero("d_grads_params")
                init_as_zero("mean")

            mean_params.append(state["mean_params"])
            mean_grads.append(state["mean_grads"])
            mean_grads_gamma.append(state["mean_grads_gamma"])
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
            mean_grads_gamma: list[Tensor] = []
            d_params_params: list[Tensor] = []
            d_grads_params: list[Tensor] = []
            mean: list[Tensor] = []

            has_sparse_grad = self._init_group(
                group=group,
                params=params,
                grads=grads,
                mean_params=mean_params,
                mean_grads=mean_grads,
                mean_grads_gamma=mean_grads_gamma,
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
                sec_ord_lr=group["sec_ord_lr"],
                gamma=group["gamma"],
                clip_max=group["clip_max"],
                clip_min=group["clip_min"],
                mean_params=mean_params,
                mean_grads=mean_grads,
                mean_grads_gamma=mean_grads_gamma,
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
    sec_ord_lr: float,
    beta: float,
    eps: float,
    gamma: float,
    clip_min: float, 
    clip_max: float, 
    mean_params: list[Tensor],
    mean_grads: list[Tensor],
    mean_grads_gamma: list[Tensor],
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
        mean_grads_gamma[i] += gamma * (grad - mean_grads_gamma[i])

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
        H_sign = torch.sign(d_grads_params[i])
        H = d_grads_params[i] / (d_params_params[i] + eps)

        # theta = theta + sing(H)(p - param)
        cliped = torch.clip(H, min=clip_min, max=clip_max)
        param.add_(
            torch.where(
                H_sign == 0,
                - grad * lr,
                - 1 / cliped * mean_grads_gamma[i] * sec_ord_lr,
            ),
        )

        if torch.isnan(param).int().sum() > 0:
            raise ValueError("nan in parameters")
