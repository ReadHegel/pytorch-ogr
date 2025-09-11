"""
This file contains implementation of the title OGR optimizer
optimalization algorithm from paper: https://arxiv.org/pdf/1901.11457
"""

from typing import Union, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torch import Tensor

def flat_params(params: list[Tensor]):
    return torch.cat([
        param.flatten() for param in params
    ]) 

def param_restore_size(flat_params, params_sizes, params_shapes):
    splited = torch.split(flat_params, params_sizes)
    return [s.view(shape) for s, shape in zip(splited, params_shapes)]

class OGR(Optimizer):
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
        size = params_flat.shape[0]

        if "first_time" not in group: 
            group["first_time"] = True 
        
            group["params_size"] = [param.numel() for param in params]
            group["params_shape"] = [param.shape() for param in params]
        
            group["mean_params"] = torch.zeros_like(params_flat)
            group["mean_grads"] = torch.zeros_like(params_flat)
            group["mean"] = torch.zeros_like(params_flat)
            group["d_params_params"] = torch.zeros((size, size))
            group["d_grads_params"] = torch.zeros((size, size))

        elif group["first_time"]: 
            group["first_time"] = False

        return has_sparse_grad, params_flat, grads_flat, group["mean_params"], group["mean_grads"], group["mean"], group["d_params_params"], group["d_grads_params"], group["first_time"]

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
            mean, \
            d_params_params, \
            d_grads_params, \
            first_time = self._init_group(
                group=group
            )

            update_values = ogr(
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
                means=mean,
                maximize=group["maximize"],
                hybrid_clipping=group["hybrid_clipping"],
                neg_clip_val=group["neg_clip_val"],
            )

            update_values = param_restore_size(update_values, group["params_size"], group["params_shape"])

            for p, update_value in zip(group["params"], update_values):
                p._add(update_value)

        return loss

    # def get_hessian(self):
    #     if len(self.param_groups) > 1:
    #         raise ValueError("Can't get the hessian with more then one group")
    #
    #     group = next(iter(self.param_groups))
    #
    #     params: list[Tensor] = []
    #     grads: list[Tensor] = []
    #     mean_params: list[Tensor] = []
    #     mean_grads: list[Tensor] = []
    #     d_params_params: list[Tensor] = []
    #     d_grads_params: list[Tensor] = []
    #     means: list[Tensor] = []
    #
    #     _ = self._init_group(
    #         group=group,
    #         params=params,
    #         grads=grads,
    #         mean_params=mean_params,
    #         mean_grads=mean_grads,
    #         d_params_params=d_params_params,
    #         d_grads_params=d_grads_params,
    #         means=means,
    #     )
    #
    #     Hs = []
    #
    #     for i, param in enumerate(params):
    #         grad = grads[i] if not group["maximize"] else -grads[i]
    #
    #         (
    #             _,
    #             _,
    #             _,
    #             new_d_params_params,
    #             new_d_grads_params,
    #         ) = _get_new_moving_average(
    #             param=param,
    #             grad=grad,
    #             beta=group["beta"],
    #             mean_param=mean_params[i],
    #             mean_grad=mean_grads[i],
    #             d_params_param=d_params_params[i],
    #             d_grads_param=d_grads_params[i],
    #             mean=means[i],
    #         )
    #
    #         # Diagonal Hessian
    #         H, _ = _get_hessian(
    #             d_params_param=new_d_params_params,
    #             d_grads_param=new_d_grads_params,
    #             eps=group["eps"],
    #         )
    #         Hs.append(H)
    #
    #     return Hs


def _get_new_moving_average(
    param: Tensor,
    grad: Tensor,
    first_time: bool,
    beta: float,
    mean_param: Tensor,
    mean_grad: Tensor,
    d_params_param: Tensor,
    d_grads_param: Tensor,
    mean: Tensor,
):
    if first_time: 
        beta = 1

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
    mean: Tensor,
    d_params_param: Tensor,
    d_grads_param: Tensor,
    eps: float,
):

    A = (mean * d_grads_param).unsqueeze(1) - mean_grads.unsqueeze(1) @ mean_params.unsqueeze(0)
    B = (mean * d_params_param).unsqueeze(1) - mean_params.unsqeeze(1) @ mean_params.unsqeeze(0)

    H_inv = B @ torch.inverse(A)
    H = A @ torch.inverse(B)

    return H, H_inv


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
    means: Tensor,
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

    H, H_inv = _get_hessian(
        mean_params,
        mean_grads, 
        means,
        d_params_param=d_params_params,
        d_grads_param=d_grads_params,
        eps=eps,
    )

    # Może można tu zamiast mean_grads dać gradienty z ADAM-a 
    # TODO napisać to satabilniej bez używania inverse
    p = (mean_params - H_inv * mean_grads) / means

    L, V = torch.linalg.eig((H + H.T) / 2)

    update_values = lr * V @ torch.diag(torch.sign(L)) @ V.T (params_flat - p)
    
    # for debug
    if torch.isnan(params_flat).any():
        raise RuntimeError(
            "Wykryto wartość NaN w parametrach - trening jest niestabilny."
        )

    return update_values
   
