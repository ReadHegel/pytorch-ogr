import torch
from torch.optim.optimizer import Optimizer
from scipy.optimize import line_search

try:
    from .utils import restore_tensor_list, flat_tensor_list  # package-style
except Exception:
    from utils import restore_tensor_list, flat_tensor_list   # local fallback


class BFGS(Optimizer):
    def __init__(self, params, lr: float = 1.0):
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
                self.count_params, device=self.parameter_device, dtype=self.parameter_dtype
            )
            group["prev_flat_grad"] = torch.zeros(
                self.count_params, device=self.parameter_device, dtype=self.parameter_dtype
            )
            group["prev_p_flat"] = torch.zeros(
                self.count_params, device=self.parameter_device, dtype=self.parameter_dtype
            )
            group["prev_loss"] = float('inf')

    def get_H_inv(self):
        return self.param_groups[0]["H_inv"]

    def get_H(self):
        return torch.inverse(self.get_H_inv())

    def step(self, closure=None, use_line_search=False):
        initial_loss = None
        if closure is not None:
            with torch.enable_grad():
                initial_loss = closure()

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
                y = flat_grad - group["prev_flat_grad"]  
                s = p_flat   - group["prev_p_flat"]      
                sy = torch.dot(s, y)

                eps = torch.finfo(self.parameter_dtype).eps
                if sy > eps:
                    H_inv = group["H_inv"]
                    rho = 1.0 / sy

                    I = torch.eye(
                        self.count_params, device=self.parameter_device, dtype=self.parameter_dtype
                    )

                    term1 = I - rho * torch.outer(s, y)
                    term2 = I - rho * torch.outer(y, s)
                    H_inv = term1 @ H_inv @ term2 + rho * torch.outer(s, s)
                    group["H_inv"] = 0.5 * (H_inv + H_inv.T)

            direction = -(group["H_inv"] @ flat_grad)

            alpha = group['lr']
            new_loss = initial_loss

            if use_line_search:
                current_params_np = p_flat.detach().cpu().numpy()
                grad_np = flat_grad.detach().cpu().numpy()
                direction_np = direction.detach().cpu().numpy()

                def get_loss_and_grad(params_np):
                    updates_list = [
                        t.to(device=self.parameter_device, dtype=self.parameter_dtype)
                        for t in restore_tensor_list(torch.from_numpy(params_np), self.param_sizes, self.param_shapes)
                    ]

                    original_data = [p.data.clone() for p in param_list]
                    original_grads = [None if p.grad is None else p.grad.clone() for p in param_list]
                    with torch.no_grad():
                        for p, new_val in zip(param_list, updates_list):
                            p.data.copy_(new_val)


                    for p in param_list:
                        p.grad = None


                    new_loss = closure() 

                    new_grads = []
                    for p in param_list:
                        g = p.grad
                        if g is None:
                            new_grads.append(torch.zeros_like(p))
                        else:
                            new_grads.append(g.detach().clone())


                    new_grad_flat = flat_tensor_list(new_grads).cpu().numpy()


                    with torch.no_grad():
                        for p, orig in zip(param_list, original_data):
                            p.data.copy_(orig)


                    for p, og in zip(param_list, original_grads):
                        if og is None:
                            p.grad = None
                        else:
                            p.grad = og.clone()

                    return float(new_loss), new_grad_flat


                alpha, _, _, _, _, new_grad = line_search(
                    f=lambda x: get_loss_and_grad(x)[0],
                    myfprime=lambda x: get_loss_and_grad(x)[1],
                    xk=current_params_np,
                    pk=direction_np,
                    gfk=grad_np,
                )

                if alpha is None:
                    alpha = 1.0
            
            final_updates = alpha * direction



            updates_list = [
                t.to(device=self.parameter_device, dtype=self.parameter_dtype)
                for t in restore_tensor_list(final_updates, self.param_sizes, self.param_shapes)
            ]

            i = 0
            for p in group["params"]:
                if p.grad is not None:
                    update = updates_list[i]
                    i += 1
                    if update.dtype != p.data.dtype:
                        update = update.to(dtype=p.data.dtype)
                    p.data.add_(update)

            group["prev_flat_grad"] = flat_grad.clone()
            group["prev_p_flat"] = p_flat.clone()

        final_loss = closure()
        return final_loss
