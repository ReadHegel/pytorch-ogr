import re
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable

from .utils import restore_tensor_list, flat_tensor_list


class BFGS(Optimizer):
    def __init__(self, params, lr=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Wrong learning rate: {lr}")

        defaults = dict(lr=lr)
        super(BFGS, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("BFSG dosnt support more then one group")

        self.count_params = 0
        self.parameter_device = None
        self.param_sizes = []
        self.param_shapes = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.count_params += p.data.numel()
                    self.param_sizes.append(p.data.numel())
                    self.param_shapes.append(p.data.shape)
                    self.parameter_device = p.device

            group["step"] = 0
            group["H_inv"] = torch.eye(self.count_params, device=self.parameter_device)
            group["prev_flat_grad"] = torch.zeros(
                self.count_params, device=self.parameter_device
            )
            group["prev_p_flat"] = torch.zeros(
                self.count_params, device=self.parameter_device
            )

    def get_H_inv(self):
        group = self.param_groups[0]
        return group["H_inv"]

    def get_H(self): 
        return torch.inverse(self.get_H_inv())

    # @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            param_list = []
            grad_list = []

            for p in group["params"]:
                if p.grad is not None:
                    param_list.append(p)
                    grad_list.append(p.grad.data)

                    if p.grad.data.is_sparse:
                        raise RuntimeError("BFGS dosnt support sparse gradients.")

            p_flat = flat_tensor_list(param_list).to(device=self.parameter_device)
            flat_grad = flat_tensor_list(grad_list).to(device=self.parameter_device)

            group["step"] += 1

            if group["step"] > 1:
                y = flat_grad - group["prev_flat_grad"]
                s = p_flat - group["prev_p_flat"]

                sy = torch.dot(s, y)

                if sy > 1e-10:
                    H_inv = group["H_inv"]
                    rho = 1.0 / sy

                    I = torch.eye(self.count_params, device=self.parameter_device)

                    term1 = I - rho * torch.outer(s, y)
                    term2 = I - rho * torch.outer(y, s)

                    H_inv = torch.matmul(
                        torch.matmul(term1, H_inv), term2
                    ) + rho * torch.outer(s, s)
                    group["H_inv"] = H_inv

            group["prev_flat_grad"] = flat_grad.clone()
            group["prev_p_flat"] = p_flat.clone()

            direction = -torch.matmul(group["H_inv"], flat_grad)

            updates_list = [
                t.to(device=self.parameter_device)
                for t in restore_tensor_list(
                    direction, self.param_sizes, self.param_shapes
                )
            ]

            i = 0
            for p in group["params"]:
                if p.grad is not None:
                    update = updates_list[i]
                    i += 1

                    p.data.add_(update, alpha=lr)

        return loss
