import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch

from .metadata import Plot, Result

PLOT_NAME = "Trace plot"
PLOT_FOLDER = "traces"


class TracePlot(Plot):
    def __init__(self):
        super().__init__(PLOT_NAME, PLOT_FOLDER)

    def _inner_print(self, result: Result, name: str):
        self.__print_traces(
            f=result.optimized_function,
            points=result.points,
            bounds=result.bounds,
            name=name,
        )

    def __print_traces(self, f, points, bounds, name):
        if len(points) != 0 and points[0].shape != (2,):
            print(points[0].shape)
            raise Exception(
                "Only two dimentional plots are supported by 'print_optimization_path'"
            )

        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i, j in itertools.product(range(X.shape[0]), range(Y.shape[1])):
            inp = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
            Z[i, j] = f(inp).item()

        plt.contourf(X, Y, Z, levels=50, cmap="viridis")

        points_np = torch.stack(points).numpy()
        plt.scatter(
            points_np[:, 0],
            points_np[:, 1],
            s=7,
            c=np.arange(len(points_np)),
            cmap="magma",
        )

        plt.title(name)
