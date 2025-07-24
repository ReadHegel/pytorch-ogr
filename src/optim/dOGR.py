from typing import Union, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torch import Tensor


class dOGR(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
        }

        super().__init__(params, defaults)

    def step(self, closure=None) -> Union[None, float]:
        loss: Union[None, float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.params_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []

            dogr(params, grads, lr=group["lr"])

        return loss


def dogr(
    params: list[Tensor],
    grads: list[Tensor],
    lr: float,
):
    pass
