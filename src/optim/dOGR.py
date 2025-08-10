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
        trust_factor: float = 1.0,
        eps: float = 1e-8,
        maximize: bool = False,
        differentiable: bool = False,
        linear_clipping: bool = False,
        nonlinear_clipping: bool = False,
        var_clipping: bool = False,
        var_fixed: float = 1.0,
        p_norm: float = 2.0,
        p_eps: float = 0.1,
        neg_clip_val: float = 10.0,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")


        defaults = {
            "lr": lr,
            "maximize": maximize,
            "beta": beta,
            "trust_factor": trust_factor,
            "eps": eps,
            "differentiable": differentiable,
            "linear_clipping": linear_clipping,
            "nonlinear_clipping": nonlinear_clipping,
            "var_clipping": var_clipping,
            "var_fixed": var_fixed,
            "p_norm": p_norm,
            "p_eps": p_eps,
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
                trust_factor=group["trust_factor"],
                eps=group["eps"],
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                mean=mean,
                maximize=group["maximize"],
                linear_clipping=group["linear_clipping"],
                nonlinear_clipping=group["nonlinear_clipping"],
                var_clipping=group["var_clipping"],
                var_fixed=group["var_fixed"],
                p_norm=group["p_norm"],
                p_eps=group["p_eps"],
                neg_clip_val=group["neg_clip_val"],
            )

        return loss


def dogr(
    params: list[Tensor],
    grads: list[Tensor],
    lr: float,
    beta: float,
    trust_factor: float,
    eps: float,
    mean_params: list[Tensor],
    mean_grads: list[Tensor],
    d_params_params: list[Tensor],
    d_grads_params: list[Tensor],
    mean: list[Tensor],
    maximize: bool,
    linear_clipping: bool,
    nonlinear_clipping: bool,
    var_clipping: bool,
    var_fixed: float,
    p_norm: float,
    p_eps: float,
    neg_clip_val: float,
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
        H_sign = torch.sign(d_grads_params[i])
        H = d_grads_params[i] / (d_params_params[i] + eps)

        
        
        if linear_clipping:
            normal_step = - (1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H))) * mean_grads[i]
            aggressive_clip_step = - (1 / torch.maximum(torch.abs(H) * 1.5, neg_clip_val * torch.ones_like(H))) * mean_grads[i]

            param.add_(
                torch.where(
                    H > 0,
                    normal_step,
                    torch.where(
                        H < 0,
                        aggressive_clip_step,
                        -grad * lr
                    )
                )
            )
        
        # 1 / (|H|^p + p_eps^p)^(1/p) * mean_grads[i]
        elif nonlinear_clipping:
            normal_step = - (1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H))) * mean_grads[i]
            
        
            abs_H = torch.abs(H)

            denominator = (abs_H.pow(p_norm) + p_eps**p_norm).pow(1.0 / p_norm)
            nonlinear_clip_step = - (1 / denominator) * mean_grads[i]

            param.add_(
                torch.where(
                    H > 0,
                    normal_step,
                    torch.where(
                        H < 0,
                        nonlinear_clip_step, 
                        -grad * lr
                    )
                ), alpha=trust_factor
            )

        elif var_clipping:
            normal_step = - (1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H))) * mean_grads[i]
            denominator = torch.abs(H) + torch.sqrt(H**2 + var_fixed)
            var_clip_step = - (2 / (denominator + eps)) * mean_grads[i]
            param.add_(
                torch.where(
                    H > 0,
                    normal_step,
                    torch.where(
                        H < 0,
                        var_clip_step, 
                        -grad * lr
                    )
                ), alpha=trust_factor
            )



        # theta = theta + sing(H)(p - param)
        else:
            param.add_(
                torch.where(
                    H_sign == 0,
                    - grad * lr,
                    - 1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H)) * mean_grads[i],
                ),
            )

        # for debug
        if torch.isnan(param).any():
            raise RuntimeError("Wykryto wartość NaN w parametrach - trening jest niestabilny.")
