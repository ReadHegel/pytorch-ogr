import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import grad
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init

from src.optim.dOGR import dOGR
from .nets import get_mini_FC, get_FC
from .datamodule import MNISTDataModule
from .main import run


def get_dogr_hessian(optim: dOGR):
    return optim.get_hessian()


# ----- eval Hessian matrix -----
# ----- second implementation for debug -----
# def eval_hessian(loss_grad, model):
#     cnt = 0
#     for g in loss_grad:
#         g_vector = (
#             g.contiguous().view(-1)
#             if cnt == 0
#             else torch.cat([g_vector, g.contiguous().view(-1)])
#         )
#         cnt = 1
#     l = g_vector.size(0)
#     hessian = torch.zeros(l, l)
#     for idx in range(l):
#         grad2rd = grad(g_vector[idx], model.parameters(), create_graph=True)
#         cnt = 0
#         for g in grad2rd:
#             g2 = (
#                 g.contiguous().view(-1)
#                 if cnt == 0
#                 else torch.cat([g2, g.contiguous().view(-1)])
#             )
#             cnt = 1
#         hessian[idx] = g2
#     return hessian


def get_bp_hessian(grads, params):
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


def plot_compararion(Hs, estimated_Hs, title: str = ""):
    fig, ax = plt.subplots()

    for H, est_H in zip(Hs, estimated_Hs):
        H = H.flatten()
        est_H = est_H.flatten()

        ax.scatter(H.numpy(force=True), est_H.numpy(force=True), color="b")

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


additional_points_to_plot = {}
additional_grads_points = {}


def one_parameter_plot(
    net,
    optim,
    Hs,
    est_Hs,
    param_name,
    param_index,
    x,
    y,
):
    target_param = None
    target_i = None
    for i, (name, param) in enumerate(net.named_parameters()):
        print(name)
        if name == param_name:
            target_param = param
            target_i = i
            break

    if target_param is None:
        raise ValueError(f"Prameter: {param_name} not found")

    hs = Hs[target_i][param_index].item()
    est_hs = est_Hs[target_i][param_index].item()

    original_value = target_param[param_index].item()
    pred = net(x)

    original_loss = F.cross_entropy(pred, y)
    original_loss.backward()
    original_loss = original_loss.item()

    grad_value = target_param.grad[param_index].item()
    mean_grad_value = optim.state[target_param]["mean_grads"][param_index].item()

    values = torch.linspace(
        original_value - 0.6, original_value + 0.6, 50, requires_grad=False
    )
    losses = []

    for val in values:
        with torch.no_grad():
            target_param[param_index] = val

            pred = net(x)
            loss = F.cross_entropy(pred, y)
            losses.append(loss.item())

    with torch.no_grad():
        target_param[param_index] = original_value

    # ---------- PLOT ------------

    def get_parabola(x, f, df, ddf, values):
        print(x, f, df, ddf)
        a = ddf / 2
        b = df - 2 * a * x
        c = f - (a * x * x + b * x)
        return a * values * values + b * values + c

    def interpole_parabole(tab, values):
        x = np.array([p[0] for p in tab])
        y = np.array([p[1] for p in tab])

        a, b, c = np.polyfit(x, y, deg=2)
        return a * values * values + b * values + c

    plt.plot(values.numpy(), losses, label="loss ( parameter )")
    plt.plot(
        values.numpy(),
        # original_loss + hs * (values.numpy() - original_value),
        get_parabola(
            original_value,
            original_loss,
            grad_value,
            hs,
            values.numpy(),
        ),
        label="Parabola estimated with real 2nd deriv",
    )
    plt.plot(
        values.numpy(),
        mean_grad_value + est_hs * (values.numpy() - original_value),
        label="estimated line of gradients by dOGR",
    )
    plt.plot(
        values.numpy(),
        grad_value + hs * (values.numpy() - original_value),
        label="line with tangent equal 2nd deriviative",
    )
    plt.plot(
        values.numpy(),
        # original_loss + est_hs * (values.numpy() - original_value),
        get_parabola(
            original_value,
            original_loss,
            grad_value,
            est_hs,
            values.numpy(),
        ),
        label="Parabola estimated with dOGR",
    )
    plt.scatter(
        original_value, original_loss, color="red", zorder=5, label="loss value"
    )
    plt.scatter(
        original_value, grad_value, color="green", zorder=5, label="derivative value"
    )
    plt.scatter(
        -grad_value / est_hs + original_value,
        0,
        marker="x",
        color="green",
        zorder=5,
        label="root of line of dOGR gradients",
    )

    for point in additional_points_to_plot[(param_name, param_index)]:
        plt.scatter(
            point[0],
            point[1],
            color="red",
            zorder=5,
        )
    for point in additional_grads_points[(param_name, param_index)]:
        plt.scatter(
            point[0],
            point[1],
            color="green",
            zorder=5,
        )
    plt.plot(
        values.numpy(),
        interpole_parabole(
            additional_points_to_plot[(param_name, param_index)], values.numpy()
        ),
        "r--",
        label="interpolated parabola",
    )

    plt.xlabel(f"Parameter value {param_name}{param_index}")
    plt.ylabel("Loss")
    plt.title("Dependence of loss on parameter")
    plt.legend()
    plt.show()


def parameter_plot(
    net,
    optim,
    Hs,
    est_Hs,
    param_list_to_plot,
    x,
    y,
):
    for param_name, param_index in param_list_to_plot:
        one_parameter_plot(
            net,
            optim,
            Hs,
            est_Hs,
            param_name,
            param_index,
            x,
            y,
        )


def run_and_plot(
    net,
    optim,
    pretrain,
    param_list_to_plot: list[Tuple],
    batch_to_plot: Tuple,
    pretrain_args=[],
):
    loss = pretrain(net, optim, *pretrain_args)

    # In fact this is diag(Hess)
    est_Hs = get_dogr_hessian(optim)

    # Compute diagonal of hessian with backpropagation
    optim.zero_grad()
    loss_grad = grad(loss, net.parameters(), create_graph=True)
    Hs = get_bp_hessian(loss_grad, net.parameters())

    print("Hs: ", Hs, "est_Hs", est_Hs)
    plot_compararion(Hs, est_Hs)
    parameter_plot(
        net,
        optim,
        Hs=Hs,
        est_Hs=est_Hs,
        param_list_to_plot=param_list_to_plot,
        x=batch_to_plot[0],
        y=batch_to_plot[1],
    )


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

    Hs = get_bp_hessian(loss_grad, [params])
    print(f"Hs: {Hs}")


def test2():
    global additional_points_to_plot
    global additional_grads_points

    BATCH_SIZE = 256

    param_list_to_plot = [
        ("weight", (0, 0)),
        ("weight", (1, 1)),
    ]

    for param_tuple in param_list_to_plot:
        additional_grads_points[param_tuple] = []
        additional_points_to_plot[param_tuple] = []

    def pretrain(net, optim):
        loss = None
        N = 7
        for i in range(N):
            optim.zero_grad()
            batch = (
                torch.ones(BATCH_SIZE, 3),
                torch.zeros(BATCH_SIZE, dtype=torch.long),
            )
            x, y = batch
            logits = net(x)
            loss = F.cross_entropy(logits, y)

            if i == N - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            for param_tuple in param_list_to_plot:
                param_name, param_index = param_tuple
                additional_points_to_plot[(param_tuple)].append(
                    (
                        dict(net.named_parameters())[param_name][param_index].item(),
                        loss.item(),
                    )
                )
                additional_grads_points[(param_tuple)].append(
                    (
                        dict(net.named_parameters())[param_name][param_index].item(),
                        dict(net.named_parameters())[param_name]
                        .grad[param_index]
                        .item(),
                    )
                )

            optim.step()
        return loss

    net = torch.nn.Linear(3, 4)
    init.normal_(net.weight, mean=0.0, std=0.25)
    optim = dOGR(net.parameters(), beta=0.60)

    run_and_plot(
        net,
        optim,
        pretrain,
        param_list_to_plot,
        (torch.ones(BATCH_SIZE, 3), torch.zeros(BATCH_SIZE, dtype=torch.long)),
        [],
    )


# def test3():
#     net = get_mini_FC()
#     optimizer = dOGR(net.parameters(), beta=0.8)
#     pytorch_total_params = sum(p.numel() for p in net.parameters())
#     print(pytorch_total_params)
#     # optimizer = torch.optim.Adam(net.parameters())
#
#     run(
#         net,
#         optimizer,
#         max_epochs=1,
#         batch_size=128,
#         version=1,
#         name="real_hess",
#     )
#
#     datamodule = MNISTDataModule(128)
#     datamodule.prepare_data()
#     datamodule.setup()
#
#     dataloader = datamodule.train_dataloader()
#     batch = next(iter(dataloader))
#
#     compare_with_hessian(net, optimizer, batch, "")


if __name__ == "__main__":
    test2()
    # test3()
