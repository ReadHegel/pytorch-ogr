import torch
from torch import Tensor
import math
from typing import Optional

class Linesearch:
    def __init__(self, f, c: float = 1e-4, tau: float = 0.5, max_backtracks: int = 50, dtype=torch.float64, device="cpu"):
        self.f = f
        self.c = float(c)
        self.tau = float(tau)
        self.max_backtracks = int(max_backtracks)
        self.dtype = dtype
        self.device = torch.device(device)

    def _to_dtype_device(self, t: Tensor) -> Tensor:
        return t.to(dtype=self.dtype, device=self.device)

    def perform_search(self, init_param: Tensor, direction: Tensor, grad: Optional[Tensor] = None) -> Tensor:
        init_param = self._to_dtype_device(init_param)
        direction = self._to_dtype_device(direction)

        f0 = self.f(init_param)
        if not isinstance(f0, torch.Tensor):
            f0 = torch.tensor(float(f0), dtype=self.dtype, device=self.device)

        slope = None
        if grad is not None:
            slope = self._to_dtype_device(grad).dot(direction)
            if not isinstance(slope, torch.Tensor):
                slope = torch.tensor(float(slope), dtype=self.dtype, device=self.device)
        else:
            eps = 1e-8
            dir_norm = direction.norm().item()
            if dir_norm == 0:
                return torch.zeros_like(direction)
            eps_step = (eps / max(dir_norm, 1.0))
            probe = init_param + (eps_step * direction)
            f_probe = self.f(probe)
            if not isinstance(f_probe, torch.Tensor):
                f_probe = torch.tensor(float(f_probe), dtype=self.dtype, device=self.device)
            slope = (f_probe - f0) / (eps_step if eps_step != 0 else eps)
        use_armijo = True
        try:
            slope_val = float(slope)
            if slope_val >= 0:
                use_armijo = False
        except Exception:
            use_armijo = False

        alpha = 1.0
        best_step = None
        best_value = None

        for _ in range(self.max_backtracks):
            new_param = init_param + alpha * direction
            f_new = self.f(new_param)
            if not isinstance(f_new, torch.Tensor):
                f_new = torch.tensor(float(f_new), dtype=self.dtype, device=self.device)

            if use_armijo:
                rhs = f0 + self.c * alpha * slope
                if f_new <= rhs:
                    return alpha * direction
            else:
                if f_new < f0:
                    return alpha * direction

            if best_value is None or f_new < best_value:
                best_value = f_new
                best_step = alpha * direction

            alpha *= self.tau

        if best_step is not None:
            return best_step

        return 1e-12 * direction  
