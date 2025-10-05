import math
import os
import traceback
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor

PLOT_PATH = "plots"
TRACES_PATH = os.path.join(PLOT_PATH, "traces")
HESSIAN_PATH = os.path.join(PLOT_PATH, "hess")


@dataclass
class Result:
    best_f: float = math.inf
    best_x: Tensor = torch.tensor([])
    iters: int = -1
    time_s: float = -1
    points: List[Tensor] = None
    hessian_real: list[Tensor] = None
    hessian_est: list[Tensor] = None
    hessian_inv_est: list[Tensor] = None
    experiment_name: str = "[no experiment name provided]"
    optimized_function = None

    def __lt__(self, other: "Result") -> bool:
        return self.best_f < other.best_f


class Plot:
    def __init__(self, plot_name: str, plot_folder_name: str):
        self.plot_name = plot_name
        self.plot_folder_name = self.plot_folder_name

    def print(self, result: Result):
        name = result.experiment_name + "-" + self.plot_name

        try:
            self._inner_print(result, name)
        except Exception as e:
            print(f"PRINTING ERROR IN: {name}, ERROR: {str(e)}")
            traceback.print_exc()

        path = os.path.join(HESSIAN_PATH, self.plot_folder_name)
        path = os.path.join(path, name)

        plt.savefig(path)
        plt.close()

    def _inner_print(self, result: Result, name: str):
        raise NotImplementedError()


class PlotList:
    def __init__(self, plot_list: List[Plot]):
        self.plot_list = plot_list

    def append_plot(self, plot):
        self.plot_list.append(plot)

    def print(self, result):
        for p in self.plot_list:
            p.print(result)
