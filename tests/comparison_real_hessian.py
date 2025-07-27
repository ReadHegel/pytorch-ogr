import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import grad
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.optim.dOGR import dOGR
from .nets import get_mini_FC, get_FC
from .datamodule import MNISTDataModule
from .main import run


def estimate_hessian(optim):
    Hs = []
    for group in optim.param_groups:
        params: list[Tensor] = []
        grads: list[Tensor] = []
        mean_params: list[Tensor] = []
        mean_grads: list[Tensor] = []
        d_params_params: list[Tensor] = []
        d_grads_params: list[Tensor] = []
        mean: list[Tensor] = []

        has_sparse_grad = optim._init_group(
            group=group,
            params=params,
            grads=grads,
            mean_params=mean_params,
            mean_grads=mean_grads,
            d_params_params=d_params_params,
            d_grads_params=d_grads_params,
            mean=mean,
        )

        Hs.extend(
            dOGR_hessian(
                params=params,
                grads=grads,
                beta=group["beta"],
                eps=group["eps"],
                mean_params=mean_params,
                mean_grads=mean_grads,
                d_params_params=d_params_params,
                d_grads_params=d_grads_params,
            )
        )

    return Hs


def dOGR_hessian(
    params: list[Tensor],
    grads: list[Tensor],
    beta: float,
    eps: float,
    mean_params: list[Tensor],
    mean_grads: list[Tensor],
    d_params_params: list[Tensor],
    d_grads_params: list[Tensor],
):
    Hs = []
    for i, param in enumerate(params):
        grad = grads[i]

        # Calculate means
        # In comments we include the formulas with symbols coresponing
        # to ones used in the paper https://arxiv.org/pdf/1901.11457

        # mean_theta = beta * theta + (1 - beta) mean_theta
        local_mean_params = mean_params[i] + beta * (param - mean_params[i])

        # mean_g = beta * g + (1 - beta) * g
        local_mean_grads = mean_grads[i] + beta * (grad - mean_grads[i])

        # d_theta_theta = (1 - beta) * theta_hat_mean**2 + beta * theta_hat**2
        local_d_params_params = d_params_params[i] + beta * (
            (param - local_mean_params) ** 2 - d_params_params[i]
        )

        # d_g_theta = (1 - beta) * d_g_theta + beta * g_hat_theta_hat
        local_d_grads_params = d_grads_params[i] + beta * (
            (grad - local_mean_grads) * (param - local_mean_params) - d_grads_params[i]
        )

        # # Diagonal Hessian
        # print(local_d_params_params)
        Hs.append(
            local_d_grads_params
            / local_d_params_params
            # / torch.maximum(
            #     local_d_params_params, 1e-8 * torch.ones_like(local_d_params_params)
            # )
        )

    return Hs


def get_hessian(grads, params):
    Hs = []
    for param, grd in zip(params, grads):
        snd_grad = []

        for i in range(param.numel()):
            snd_grad.append(
                grad(outputs=grd.flatten()[i], inputs=param, create_graph=True)[
                    0
                ].flatten()[i]
            )

        Hs.append(torch.stack(snd_grad).reshape_as(param))
    return Hs


def plot(Hs, estimated_Hs, title: str = ""):
    fig, ax = plt.subplots()

    count = 0
    count_bigger = 0
    minn = 100

    for H, est_H in zip(Hs, estimated_Hs):
        H = H.flatten()
        est_H = est_H.flatten()

        count_bigger += (est_H > 5).int().sum()
        count += est_H.shape[0]
        minn = min(minn, torch.min(H).item())
        # mask = abs(est_H) < 10
        #
        # H = H[mask]
        # est_H = est_H[mask]

        ax.scatter(H.numpy(force=True), est_H.numpy(force=True), color="b")

    print(count, count_bigger, count_bigger / count)
    print(minn)

    # Center plot
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    xabs_max = abs(max(ax.get_xlim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)

    # Add x = y line
    xy_abs_max = max(yabs_max, xabs_max)
    ax.plot(
        [-xy_abs_max, xy_abs_max],
        [-xy_abs_max, xy_abs_max],
        color="r",
        label="x = y",
    )

    ax.set_xlabel("Real hesssian diagonal value")
    ax.set_ylabel("Esimated hessain diagonal value (dOGR)")
    ax.set_title(title)

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    plt.grid(True)

    plt.show()


def compare_with_hessian(net, optim, batch: Tuple[Tensor, Tensor], name: str):
    optim.zero_grad()

    x, y = batch

    logits = net(x)

    loss = F.cross_entropy(logits, y)
    loss.backward(retain_graph=True)

    # In fact this is diag(Hess)
    estimated_Hs = estimate_hessian(optim)

    # print("estimated_hessian: ", estimated_Hs)

    # Computing hessian with backpropagation
    # print(f"params: {net.parameters()}")

    optim.zero_grad()

    loss_grad = grad(loss, net.parameters(), create_graph=True)
    # print(f"loss_grad: {loss_grad}")

    # print(f"params: {list(net.parameters())}")
    # In fact this is diag(Hess)
    Hs = get_hessian(loss_grad, net.parameters())
    # print(f"Hs: {Hs}")

    plot(Hs, estimated_Hs, name)


def test1():
    real_H = torch.rand((2, 2), requires_grad=True)
    print(f"real_H: {real_H}")
    h = torch.rand(1, requires_grad=True)
    print(f"h: {h}")
    p = torch.rand(2, requires_grad=True)
    print(f"p: {p}")

    params = torch.rand(2, requires_grad=True)
    print(f"params: {params}")

    loss = h + ((params - p) @ real_H @ (params - p)) / 2
    print("loss: ", loss)
    loss_grad = grad(loss, params, create_graph=True)

    print(f"loss_grad: {loss_grad}")
    print(f"loss_grad should be: {real_H @ (params - p)}")

    Hs = get_hessian(loss_grad, [params])
    print(f"Hs: {Hs}")


def test2():
    BATCH_SIZE = 2
    net = torch.nn.Linear(3, 4)
    torch.nn.init.ones_(net.weight)
    torch.nn.init.ones_(net.bias)
    optim = dOGR(net.parameters())
    batch1 = (torch.ones(BATCH_SIZE, 3), torch.zeros(BATCH_SIZE, dtype=torch.long))
    batch2 = (torch.ones(BATCH_SIZE, 3), torch.zeros(BATCH_SIZE, dtype=torch.long))

    optim.zero_grad()
    x, y = batch1
    logits = net(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optim.step()

    compare_with_hessian(net, optim, batch2, "DUPA")


def test3():
    net = get_mini_FC()
    optimizer = dOGR(net.parameters(), eps=0)
    # optimizer = torch.optim.Adam(net.parameters())

    run(
        net,
        optimizer,
        max_epochs=1,
        batch_size=32,
        version=1,
        name="real_hess",
    )

    datamodule = MNISTDataModule(32)
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))

    compare_with_hessian(net, optimizer, batch, "")


if __name__ == "__main__":
    # test2()
    test3()
