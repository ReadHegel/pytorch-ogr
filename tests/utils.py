from typing import Tuple

import torch
from torch import Tensor


def sample_uniform(
    bounds: Tuple[Tuple[float, float], ...],
    device: torch.device,
    dtype: torch.dtype,
    seed=None,
) -> Tensor:
    lows = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    highs = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    u = torch.rand(
        lows.size(),
        dtype=lows.dtype,
        layout=lows.layout,
        device=lows.device,
        generator=generator,
    )
    return lows + u * (highs - lows)


def clamp_inplace(x: Tensor, bounds: Tuple[Tuple[float, float], ...]) -> None:
    """Project x back into box bounds (in-place)."""
    lows = torch.tensor([b[0] for b in bounds], device=x.device, dtype=x.dtype)
    highs = torch.tensor([b[1] for b in bounds], device=x.device, dtype=x.dtype)
    x.data.copy_(torch.minimum(torch.maximum(x.data, lows), highs))


def repeat_bounds(b: Tuple[float, float], n: int):
    return tuple([b] * n)


def get_bp_hessian_from_loss(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])
    n = grad_flat.numel()

    H_rows = []
    for i in range(n):
        g2 = torch.autograd.grad(
            grad_flat[i], params, retain_graph=True, create_graph=True
        )
        g2_flat = torch.cat([g.flatten() for g in g2])
        H_rows.append(g2_flat)

    return torch.stack(H_rows)
