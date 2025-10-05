import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .metadata import Plot, Result

PLOT_NAME = "Hessian plot"
INV_PLOT_NAME = "Inverion Hessian plot"
PLOT_FOLDER = "hessians"


def print_hessian_in_time(
    real_hessian: list[Tensor],
    est_hessian: list[Tensor],
    name: str,
):
    def clear_nones(list_of_tensors: list[Tensor]):
        i = 0
        while not torch.is_tensor(list_of_tensors[i]):
            if i >= len(list_of_tensors):
                raise Exception("None of hessians is tensor")
            i += 1

        ref = list_of_tensors[i]

        for i in range(len(list_of_tensors)):
            if not torch.is_tensor(list_of_tensors[i]):
                list_of_tensors[i] = torch.eye(ref.shape[0])

        return list_of_tensors

    # to numpy
    real_hessian = torch.stack(clear_nones(real_hessian)).detach().numpy()
    est_hessian = torch.stack(clear_nones(est_hessian)).detach().numpy()

    n = real_hessian.shape[1]

    fig, axes = plt.subplots(n, n, figsize=(10, 8))

    for i, j in itertools.product(range(n), range(n)):
        ax = axes[i, j]

        ax.plot(real_hessian[:, i, j], label="Real", color="blue")
        ax.plot(est_hessian[:, i, j], label="Estimated H", color="red")

        ax.set_title(f"Hessian coefficient ({i}, {j})")

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")

        ax.legend()

    fig.suptitle("Hessian coefficents throughout training", fontsize=16)
    plt.tight_layout()


class HessianPlot(Plot):
    def __init__(self):
        super().__init__(PLOT_NAME, PLOT_FOLDER)

    def _inner_print(self, result: Result, name: str):
        print_hessian_in_time(
            real_hessian=result.hessian_real,
            est_hessian=result.hessian_est,
            name=name,
        )


class InvHessianPlot(Plot):
    def __init__(self):
        super().__init__(INV_PLOT_NAME, PLOT_FOLDER)

    def _inner_print(self, result: Result, name: str):
        print_hessian_in_time(
            real_hessian=[torch.inverse(h) for h in result.hessian_real],
            est_hessian=result.hessian_inv_est,
            name=name,
        )
