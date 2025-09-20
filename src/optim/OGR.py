"""
OGR optimizer (stabilized)
Paper: https://arxiv.org/pdf/1901.11457
"""
from typing import Union, Optional, List, Tuple

from sympy import group
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from scipy.optimize import line_search

try:
    from .utils import restore_tensor_list, flat_tensor_list 
except Exception:
    from utils import restore_tensor_list, flat_tensor_list  


def _safe_eigh(H: Tensor, jitter: float = 1e-8, max_tries: int = 6) -> Tuple[Tensor, Tensor]:
    I = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    for k in range(max_tries):
        Hs = 0.5 * (H + H.T)
        try:
            L, Q = torch.linalg.eigh(Hs)
            return L, Q
        except Exception:
            H = Hs + (jitter * (10.0 ** k)) * I
    Hs = 0.5 * (H + H.T) + (jitter * (10.0 ** (max_tries - 1))) * I
    return torch.linalg.eigh(Hs)


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
    B = 0.5 * (d_grads_param + d_grads_param.T)

    L, O = _safe_eigh(A, jitter=max(eps, 1e-10))

    alpha = torch.as_tensor(max(eps, 1e-10), dtype=L.dtype, device=L.device)
    Li = L.unsqueeze(0).expand(size, size)
    Lj = L.unsqueeze(1).expand(size, size)
    denom = Li + Lj
    denom = torch.where(denom.abs() < alpha, alpha, denom)

    BO = O.T @ B @ O
    H = O @ (BO / denom) @ O.T
    return 0.5 * (H + H.T)


def _get_H_inv_from_H(H: Tensor) -> Tensor:
    L, Q = _safe_eigh(H)
    floor = torch.tensor(20.0, dtype=L.dtype, device=L.device)
    inv_diag = 1.0 / torch.maximum(floor, L.abs())
    H_inv = Q @ torch.diag(inv_diag) @ Q.T
    return 0.5 * (H_inv + H_inv.T)


class OGR(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = (1 / 1.5),
        beta: float = 0.30,
        eps: float = 1e-12,
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
            "maximize": maximize,
            "beta": beta,
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
            group["d_params_params"] = torch.eye(size, device=params_flat.device, dtype=params_flat.dtype)
            group["d_grads_params"] = torch.eye(size, device=params_flat.device, dtype=params_flat.dtype)
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
        H = self.get_H()
        return _get_H_inv_from_H(H)

    @_use_grad_for_differentiable
    def step(self, closure=None, use_line_search=False) -> Union[None, float]:
        loss: Union[None, float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            raise ValueError("Only one parameter group is supported")

        for group in self.param_groups:
            if loss is not None:
                params_for_grad = [p for p in group["params"]]
                # compute_graph required if we want differentiable behavior elsewhere
                create_graph = group.get("differentiable", False)
                grads = torch.autograd.grad(loss, params_for_grad, create_graph=create_graph, allow_unused=True)
                # Attach grads to parameters (replace None with zeros)
                for p, g in zip(params_for_grad, grads):
                    if g is None:
                        p.grad = torch.zeros_like(p)
                    else:
                        p.grad = g
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
                max_step_norm=group["max_step_norm"],
                use_line_search=use_line_search,
                closure=closure,
                param_list=group["params"],
                param_sizes=group["params_size"],
                param_shapes=group["params_shape"],
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

        final_loss = closure()
        return final_loss


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
        torch.outer(param - local_mean_params, param - local_mean_params) - d_params_param
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
    max_step_norm: Optional[float],
    closure,
    param_list,
    param_sizes,
    param_shapes,
    use_line_search: bool = False,

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

    H_inv = _get_H_inv_from_H(H)
    direction = - (H_inv @ grads_flat)

    if max_step_norm is not None:
        step_norm = direction.norm()
        if step_norm > max_step_norm:
            direction = direction * (max_step_norm / (step_norm + 1e-12))

    alpha = lr
    final_updates = alpha * direction

    if use_line_search:
        current_params_np = params_flat.detach().cpu().numpy()
        grad_np = grads_flat.detach().cpu().numpy()
        direction_np = direction.detach().cpu().numpy()

        def _get_loss_and_grad(params_np):
            temp_params_tensor_list = restore_tensor_list(
                torch.from_numpy(params_np).to(param_list[0].device, dtype=param_list[0].dtype),
                param_sizes,
                param_shapes,
            )

            original_params_data = [p.data.clone() for p in param_list]

            with torch.no_grad():
                for p, t in zip(param_list, temp_params_tensor_list):
                    p.data.copy_(t)

            with torch.enable_grad():

                new_loss = closure()


                new_grads = torch.autograd.grad(new_loss, param_list, create_graph=True, allow_unused=True)

            new_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(new_grads, param_list)]

            new_grad = flat_tensor_list([g.detach() for g in new_grads]).cpu().numpy()

            with torch.no_grad():
                for p, orig in zip(param_list, original_params_data):
                    p.data.copy_(orig)

            return new_loss.item(), new_grad



        alpha, _, _, _, _, new_grad = line_search(
            f=lambda x: _get_loss_and_grad(x)[0],
            myfprime=lambda x: _get_loss_and_grad(x)[1],
            xk=current_params_np,
            pk=direction_np,
            gfk=grad_np
        )
        
        if alpha is None:
            alpha = 1.0
        
        final_updates = alpha * direction

    if torch.isnan(params_flat).any():
        raise RuntimeError("Wykryto wartość NaN w parametrach - trening jest niestabilny.")

    return (
        final_updates,
        mean_params,
        mean_grads,
        d_params_params,
        d_grads_params,
        means,
    )
