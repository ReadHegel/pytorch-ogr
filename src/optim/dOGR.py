"""
This file contains implementation of the diagonal OGR optimization algorithm
inspired by: https://arxiv.org/pdf/1901.11457

Additions:
- nn_policy: stochastic policy (squashed Gaussian) with proper log_prob for REINFORCE
- step() returns (actions, log_prob, entropy) when nn_policy=True; otherwise returns None
- init_as_zeros toggle to initialize EMA states from zeros (legacy) or from first sample (param/grad)
"""

from typing import Union, Tuple, Callable, Optional
from torch.distributions import Normal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable

from .policy_net import PolicyNet


class dOGR(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        policy_net: PolicyNet,
        policy_optimizer: Optimizer,
        net: nn.Module,
        lr: Union[float, Tensor] = 1e-3,
        beta: float = 0.50,
        trust_factor: float = 1.0,
        eps: float = 1e-8,
        maximize: bool = False,
        differentiable: bool = False,
        linear_clipping: bool = False,
        nonlinear_clipping: bool = False,
        nn_policy: bool = False,
        var_clipping: bool = False,
        var_fixed: float = 1.0,
        p_norm: float = 2.0,
        p_eps: float = 0.1,
        neg_clip_val: float = 10.0,
        # policy std for Gaussian policy (can be tuned)
        policy_std: float = 0.1,
        # control EMA init style
        init_as_zeros: bool = True,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if policy_std <= 0.0:
            raise ValueError("policy_std must be > 0")

        defaults = {
            "lr": lr,
            "maximize": maximize,
            "beta": beta,
            "trust_factor": trust_factor,
            "eps": eps,
            "differentiable": differentiable,
            "linear_clipping": linear_clipping,
            "nonlinear_clipping": nonlinear_clipping,
            "nn_policy": nn_policy,
            "var_clipping": var_clipping,
            "var_fixed": var_fixed,
            "p_norm": p_norm,
            "p_eps": p_eps,
            "neg_clip_val": neg_clip_val,
            "policy_std": policy_std,
            "init_as_zeros": init_as_zeros,
        }

        super().__init__(params, defaults)
        self.policy_net = policy_net
        self.policy_optimizer = policy_optimizer
        self.net = net
        self.loss_fn = F.cross_entropy

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
                if group.get("init_as_zeros", True):
                    # Legacy: start from zeros
                    def init_as_zero(name: str):
                        state[name] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    init_as_zero("mean_params")
                    init_as_zero("mean_grads")
                    init_as_zero("d_params_params")
                    init_as_zero("d_grads_params")
                    init_as_zero("mean")
                else:
                    # Warm-start: initialize from current param/grad sample
                    state["mean_params"] = p.detach().clone()
                    if p.grad is not None:
                        state["mean_grads"] = p.grad.detach().clone()
                    else:
                        state["mean_grads"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["d_params_params"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["d_grads_params"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # start counter from 1 to reduce cold-start bias
                    state["mean"] = torch.ones_like(p, memory_format=torch.preserve_format)

            mean_params.append(state["mean_params"])
            mean_grads.append(state["mean_grads"])
            d_params_params.append(state["d_params_params"])
            d_grads_params.append(state["d_grads_params"])
            mean.append(state["mean"])

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Returns (actions, log_prob, entropy) only when nn_policy=True; otherwise None."""
        if closure is not None:
            with torch.enable_grad():
                _ = closure()

        actions_list = []
        logprob_list = []
        entropy_list = []

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            mean_params: list[Tensor] = []
            mean_grads: list[Tensor] = []
            d_params_params: list[Tensor] = []
            d_grads_params: list[Tensor] = []
            mean: list[Tensor] = []

            self._init_group(
                group=group,
                params=params,
                grads=grads,
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
                mean=mean,
            )

            result = dogr(
                params=params,
                grads=grads,
                main_net=self.net,
                policy_net=self.policy_net,
                policy_optimizer=self.policy_optimizer,
                loss_fn=self.loss_fn,
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
                nn_policy=group["nn_policy"],
                var_clipping=group["var_clipping"],
                var_fixed=group["var_fixed"],
                p_norm=group["p_norm"],
                p_eps=group["p_eps"],
                neg_clip_val=group["neg_clip_val"],
                policy_std=group["policy_std"],
            )

            if result is not None:
                a, lp, ent = result
                actions_list.append(a)
                logprob_list.append(lp)
                entropy_list.append(ent)

        if actions_list:
            actions = torch.stack(actions_list).mean()  # diagnostic mean action over groups
            log_prob = torch.stack(logprob_list).sum()  # joint log-prob across groups/params
            entropy = torch.stack(entropy_list).sum()
            return actions, log_prob, entropy

        return None


def _safe_stack(*tensors: Tensor) -> Tensor:
    """Stack last-dim features; ensure same device/dtype."""
    dev = tensors[0].device
    tensors = [t.to(dev) for t in tensors]
    return torch.stack(tensors, dim=-1)


def dogr(
    params: list[Tensor],
    grads: list[Tensor],
    main_net: nn.Module,
    policy_net: PolicyNet,
    policy_optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
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
    nn_policy: bool,
    var_clipping: bool,
    var_fixed: float,
    p_norm: float,
    p_eps: float,
    neg_clip_val: float,
    policy_std: float,
) -> Optional[Tuple[Tensor, Tensor, Tensor]]:

    actions_accum: list[Tensor] = []
    logprob_accum: list[Tensor] = []
    entropy_accum: list[Tensor] = []

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        # --- Running means / stats ---
        mean_params[i] += beta * (param - mean_params[i])          # mean_theta
        mean_grads[i]  += beta * (grad  - mean_grads[i])           # mean_g
        mean[i]        += (beta - 1) * mean[i] + 1                 # s counter

        d_params_params[i] += beta * ((param - mean_params[i]) ** 2 - d_params_params[i])  # d_theta_theta
        d_grads_params[i]  += beta * ((grad - mean_grads[i]) * (param - mean_params[i]) - d_grads_params[i])  # d_g_theta

        # Diagonal Hessian proxy
        H_sign = torch.sign(d_grads_params[i])
        H = d_grads_params[i] / (d_params_params[i] + eps)

        if linear_clipping:
            normal_step = -(1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H))) * mean_grads[i]
            aggressive_clip_step = -(1 / torch.maximum(torch.abs(H) * 1.5, neg_clip_val * torch.ones_like(H))) * mean_grads[i]
            param.add_(torch.where(H > 0, normal_step, torch.where(H < 0, aggressive_clip_step, -grad * lr)))

        elif nonlinear_clipping:
            normal_step = -(1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H))) * mean_grads[i]
            abs_H = torch.abs(H)
            denominator = (abs_H.pow(p_norm) + p_eps**p_norm).pow(1.0 / p_norm)
            nonlinear_clip_step = -(1 / denominator) * mean_grads[i]
            param.add_(torch.where(H > 0, normal_step, torch.where(H < 0, nonlinear_clip_step, -grad * lr)), alpha=trust_factor)

        elif nn_policy:
            # --- POLICY NN: stochastic policy with squashed Gaussian ---
            # We build features per weight; detach to avoid second-order grads through stats.
            policy_input = _safe_stack(
                mean_grads[i].detach(),
                H.detach(),
                d_params_params[i].detach(),
                d_grads_params[i].detach(),
            )

            # 1) Compute policy WITH grads (even though outer step may be in no_grad)
            with torch.enable_grad():
                mu = policy_net(policy_input).squeeze(-1)  # raw mu (last layer must be linear)
                dist = Normal(loc=mu, scale=torch.as_tensor(policy_std, device=mu.device, dtype=mu.dtype))
                z = dist.rsample()
                action = 2.0 * torch.sigmoid(z)  # (0, 2)

                # Change-of-variables: y = 2*sigmoid(z); dy/dz = y * (1 - y/2)
                log_jac = torch.log(action * (1.0 - action / 2.0) + eps)
                log_prob = dist.log_prob(z) - log_jac
                entropy = dist.entropy()  # same shape as z

                # prepare scalars to accumulate (keep graph!)
                log_prob_param = log_prob.sum()
                action_param = action.mean()
                entropy_param = entropy.sum()

            # 2) Apply update to parameters WITHOUT grads (avoid in-place on leaf requiring grad)
            with torch.no_grad():
                final_update = -action * mean_grads[i]
                param.add_(final_update, alpha=trust_factor)

            # 3) Accumulate diagnostics
            actions_accum.append(action_param)     # scalar
            logprob_accum.append(log_prob_param)   # scalar
            entropy_accum.append(entropy_param)    # scalar

        elif var_clipping:
            normal_step = -(1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H))) * mean_grads[i]
            denominator = torch.abs(H) + torch.sqrt(H**2 + var_fixed)
            var_clip_step = -(2 / (denominator + eps)) * mean_grads[i]
            param.add_(torch.where(H > 0, normal_step, torch.where(H < 0, var_clip_step, -grad * lr)), alpha=trust_factor)

        else:
            # Default dOGR step
            param.add_(
                torch.where(
                    H_sign == 0,
                    -grad * lr,
                    -1 / torch.maximum(torch.abs(H) * 1.5, 5 * torch.ones_like(H)) * mean_grads[i],
                ),
            )

        # Stability check
        if torch.isnan(param).any():
            raise RuntimeError("Wykryto wartość NaN w parametrach - trening jest niestabilny.")

    # Return tuple only for nn_policy (after updating ALL params)
    if nn_policy and len(actions_accum) > 0:
        actions_agg = torch.stack(actions_accum).mean()   # mean action across all param tensors
        logprob_agg = torch.stack(logprob_accum).sum()    # sum of log-probs across all param tensors
        entropy_agg = torch.stack(entropy_accum).sum()
        return actions_agg, logprob_agg, entropy_agg

    return None
