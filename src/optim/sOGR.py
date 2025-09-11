"""
This file contains implementation of the title OGR optimizer
optimalization algorithm from paper: https://arxiv.org/pdf/1901.11457
"""

from typing import Union, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torch import Tensor

from OGR import OGR

class sOGR(Optimizer):
    def __init__(
        self,
        params_ogr: ParamsT,
        optim_other: Optimizer,
        lr: float,
        beta: float = 0.50,
        eps: float = 1e-8,
        maximize: bool = False,
        differentiable: bool = False,
        hybrid_clipping: bool = False,
        neg_clip_val: Optional[float] = None,
    ):
        self.ogr = OGR(
            params_ogr,
            lr,
            beta,
            eps,
            maximize,
            differentiable,
            hybrid_clipping,
            neg_clip_val,
        )

        self.other_optim = optim_other


    @_use_grad_for_differentiable
    def step(self) -> None:
        self.ogr.step()
        self.other_optim.step()
