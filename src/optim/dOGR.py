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
        eps: float = 1e-8,
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
        params,
        grads,
        mean_params: list[Tensor],
        mean_grads: list[Tensor],
        d_params_params: list[Tensor],
        d_grads_params: list[Tensor],
        means: list[Tensor],
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
                init_as_zero("means")

            mean_params.append(state["mean_params"])
            mean_grads.append(state["mean_grads"])
            d_params_params.append(state["d_params_params"])
            d_grads_params.append(state["d_grads_params"])
            means.append(state["means"])

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
            means: list[Tensor] = []

            has_sparse_grad = self._init_group(
                group=group,
                params=params,
                grads=grads,
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                means=means,
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
                means=means,
                maximize=group["maximize"],
                hybrid_clipping=group["hybrid_clipping"],
                neg_clip_val=group["neg_clip_val"],
            )

            for p, mean_param, mean_grad, d_pp, d_gp, mean in zip(
                params, mean_params, mean_grads, d_params_params, d_grads_params, means
            ):
                state = self.state[p]
                state["mean_params"] = mean_param
                state["mean_grads"] = mean_grad
                state["d_params_params"] = d_pp
                state["d_grads_params"] = d_gp
                state["means"] = mean 

        return loss

    def get_hessian(self):
        if len(self.param_groups) > 1:
            raise ValueError("Can't get the hessian with more then one group")

        group = next(iter(self.param_groups))

        params: list[Tensor] = []
        grads: list[Tensor] = []
        mean_params: list[Tensor] = []
        mean_grads: list[Tensor] = []
        d_params_params: list[Tensor] = []
        d_grads_params: list[Tensor] = []
        means: list[Tensor] = []

        _ = self._init_group(
            group=group,
            params=params,
            grads=grads,
            mean_params=mean_params,
            mean_grads=mean_grads,
            d_params_params=d_params_params,
            d_grads_params=d_grads_params,
            means=means,
        )

        Hs = []

        for i, param in enumerate(params):
            grad = grads[i] if not group["maximize"] else -grads[i]

            (
                _,
                _,
                _,
                new_d_params_params,
                new_d_grads_params,
            ) = _get_new_moving_average(
                param=param,
                grad=grad,
                beta=group["beta"],
                mean_param=mean_params[i],
                mean_grad=mean_grads[i],
                d_params_param=d_params_params[i],
                d_grads_param=d_grads_params[i],
                mean=means[i],
            )

            # Diagonal Hessian
            H, _ = _get_hessian(
                d_params_param=new_d_params_params,
                d_grads_param=new_d_grads_params,
                eps=group["eps"],
            )
            Hs.append(H)

        return Hs


def _get_new_moving_average(
    param: Tensor,
    grad: Tensor,
    beta: float,
    mean_param: Tensor,
    mean_grad: Tensor,
    d_params_param: Tensor,
    d_grads_param: Tensor,
    mean: Tensor,
):
    # Calculate means
    # In comments we include the formulas with symbols coresponing
    # to ones used in the paper https://arxiv.org/pdf/1901.11457

    # mean_theta = beta * theta + (1 - beta) mean_theta
    local_mean_params = mean_param + beta * (param - mean_param)

    # mean_g = beta * g + (1 - beta) * g
    local_mean_grads = mean_grad + beta * (grad - mean_grad)

    local_mean = mean * beta + 1

    # d_theta_theta = (1 - beta) * theta_hat_mean**2 + beta * theta_hat**2
    local_d_params_params = d_params_param + beta * (
        (param - local_mean_params) ** 2 - d_params_param
    )

    # d_g_theta = (1 - beta) * d_g_theta + beta * g_hat_theta_hat
    local_d_grads_params = d_grads_param + beta * (
        (grad - local_mean_grads) * (param - local_mean_params) - d_grads_param
    )

    return (
        local_mean_params,
        local_mean_grads,
        local_mean,
        local_d_params_params,
        local_d_grads_params,
    )


def _get_hessian(
    d_params_param: Tensor,
    d_grads_param: Tensor,
    eps: float,
):
    H_sign = torch.sign(d_grads_param)
    H = d_grads_param / (d_params_param + eps)

    return H, H_sign


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
    means: list[Tensor],
    maximize: bool,
    hybrid_clipping: bool,
    neg_clip_val: Optional[float],
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        (
            new_mean_params,
            new_mean_grads,
            new_mean,
            new_d_params_params,
            new_d_grads_params,
        ) = _get_new_moving_average(
            param=param,
            grad=grad,
            beta=beta,
            mean_param=mean_params[i],
            mean_grad=mean_grads[i],
            d_params_param=d_params_params[i],
            d_grads_param=d_grads_params[i],
            mean=means[i],
        )

        mean_params[i] = new_mean_params
        mean_grads[i] = new_mean_grads
        d_params_params[i] = new_d_params_params
        d_grads_params[i] = new_d_grads_params
        means[i] = new_mean

        # Diagonal Hessian
        H, H_sign = _get_hessian(
            d_params_param=d_params_params[i],
            d_grads_param=d_grads_params[i],
            eps=eps,
        )

        # theta = theta + sing(H)(p - param)
        if not hybrid_clipping:
            param.add_(
                torch.where(
                    H_sign == 0,
                    -grad * lr,
                    -1
                    / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H))
                    * mean_grads[i],
                ),
            )
        else:
            normal_step = (
                -(1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H)))
                * mean_grads[i]
            )
            aggressive_clip_step = (
                -(
                    1
                    / torch.maximum(
                        torch.abs(H) * 1.5, neg_clip_val * torch.ones_like(H)
                    )
                )
                * mean_grads[i]
            )

            param.add_(
                torch.where(
                    H > 0,
                    normal_step,
                    torch.where(H < 0, aggressive_clip_step, -grad * lr),
                )
            )

        # for debug
        if torch.isnan(param).any():
            raise RuntimeError(
                "Wykryto wartość NaN w parametrach - trening jest niestabilny."
            )
