import torch
from torch.optim.optimizer import Optimizer

try:
    from .utils import restore_tensor_list, flat_tensor_list  # package-style
except Exception:
    from utils import restore_tensor_list, flat_tensor_list  # local fallback


class BFGS(Optimizer):
    def __init__(self, params, lr: float = 1.0, linesearch=None):
        if not 0.0 <= lr:
            raise ValueError(f"Wrong learning rate: {lr}")

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("BFGS doesn't support more than one parameter group")

        self.count_params = 0
        self.parameter_device = None
        self.parameter_dtype = None
        self.param_sizes = []
        self.param_shapes = []
        
        self.linesearch=linesearch

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    if self.parameter_device is None:
                        self.parameter_device = p.device
                    if self.parameter_dtype is None:
                        self.parameter_dtype = p.data.dtype

                    if p.device != self.parameter_device:
                        raise RuntimeError("All params must be on the same device")
                    if p.data.dtype != self.parameter_dtype:
                        raise RuntimeError("All params must share the same dtype")

                    self.count_params += p.data.numel()
                    self.param_sizes.append(p.data.numel())
                    self.param_shapes.append(p.data.shape)

            group["step"] = 0
            group["H_inv"] = torch.eye(
                self.count_params,
                device=self.parameter_device,
                dtype=self.parameter_dtype,
            )
            group["prev_flat_grad"] = torch.zeros(
                self.count_params,
                device=self.parameter_device,
                dtype=self.parameter_dtype,
            )
            group["prev_p_flat"] = torch.zeros(
                self.count_params,
                device=self.parameter_device,
                dtype=self.parameter_dtype,
            )

    def get_H_inv(self):
        return self.param_groups[0]["H_inv"]

    def get_H(self):
        return torch.inverse(self.get_H_inv())

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
                    if p.grad.data.is_sparse:
                        raise RuntimeError("BFGS doesn't support sparse gradients.")
                    param_list.append(p)
                    grad_list.append(p.grad.data)

            if not param_list:
                continue
            p_flat = flat_tensor_list(param_list).to(
                device=self.parameter_device, dtype=self.parameter_dtype
            )
            flat_grad = flat_tensor_list(grad_list).to(
                device=self.parameter_device, dtype=self.parameter_dtype
            )

            group["step"] += 1

            if group["step"] > 1:
                y = flat_grad - group["prev_flat_grad"]  # Δg
                s = p_flat - group["prev_p_flat"]  # Δx
                sy = torch.dot(s, y)

                eps = torch.finfo(self.parameter_dtype).eps
                if sy > eps:
                    H_inv = group["H_inv"]
                    rho = 1.0 / sy

                    I = torch.eye(
                        self.count_params,
                        device=self.parameter_device,
                        dtype=self.parameter_dtype,
                    )

                    term1 = I - rho * torch.outer(s, y)
                    term2 = I - rho * torch.outer(y, s)
                    H_inv = term1 @ H_inv @ term2 + rho * torch.outer(s, s)
                    group["H_inv"] = 0.5 * (H_inv + H_inv.T)

            group["prev_flat_grad"] = flat_grad.clone()
            group["prev_p_flat"] = p_flat.clone()

            direction = - (group["H_inv"] @ flat_grad)
            
            if self.linesearch is not None: 
                direction = self.linesearch.perform_search(p_flat, direction, flat_grad)
            else: 
                direction *= lr

            # Reshape back update values 
            updates_list = [
                t.to(device=self.parameter_device, dtype=self.parameter_dtype)
                for t in restore_tensor_list(
                    direction, self.param_sizes, self.param_shapes
                )
            ]

            # Update parameters
            i = 0
            for p in group["params"]:
                if p.grad is not None:
                    update = updates_list[i]
                    i += 1
                    if update.dtype != p.data.dtype:
                        update = update.to(dtype=p.data.dtype)
                    p.data.add_(update, alpha=lr)

        return loss
