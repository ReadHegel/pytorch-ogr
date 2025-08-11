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


# eval Hessian matrix
def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = (
            g.contiguous().view(-1)
            if cnt == 0
            else torch.cat([g_vector, g.contiguous().view(-1)])
        )
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = (
                g.contiguous().view(-1)
                if cnt == 0
                else torch.cat([g2, g.contiguous().view(-1)])
            )
            cnt = 1
        hessian[idx] = g2
    return hessian


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

    # optim.zero_grad()
    #
    # Hs = eval_hessian(loss_grad, net)
    # diag = torch.diag(Hs)
    # print("eig:")
    # eig = (torch.linalg.eig(Hs)[0]).double()
    # print(torch.sort(eig))
    # print(torch.diag(Hs))
    # print(torch.min(diag), torch.max(diag))
    # def f(*params):
    #     # Update model parameters manually
    #     with torch.no_grad():
    #         for p, new_p in zip(net.parameters(), params):
    #             p.copy_(new_p)
    #
    #     x, y = batch
    #     logits = net(x)
    #     return F.cross_entropy(logits, y)
    #
    # # Get parameters as a tuple of tensors
    # params = tuple(p.detach().clone().requires_grad_(True) for p in net.parameters())
    #
    # # Compute Hessian
    # hess = torch.autograd.functional.hessian(f, params)
    #
    # print(
    #     "DUPADUPADUPA",
    #     hess
    # )
    print("Hs: ", Hs, "est_Hs", estimated_Hs)
    plot(Hs, estimated_Hs, name)
    parameter_plot(
        net,
        Hs=Hs,
        est_Hs=estimated_Hs,
        param_name="weight",
        param_index=(0, 0),
        x=batch[0],
        y=batch[1],
    )


additional_points_to_plot = []
grads_points = []


def parameter_plot(
    net,
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

    hs = Hs[i][param_index].item()
    est_hs = est_Hs[i][param_index].item()

    original_value = target_param[param_index].item()
    pred = net(x)
    original_loss = F.cross_entropy(pred, y)
    original_loss.backward()
    original_loss = original_loss.item()

    grad_value = target_param.grad[param_index].item()

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
        interpole_parabole(additional_points_to_plot, values.numpy()),
        "r--",
        label="interpolated parabola",
    )
    plt.plot(
        values.numpy(),
        grad_value + est_hs * (values.numpy() - original_value),
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
        -grad_value / est_hs + original_value, 0,
        marker="x", color="green", zorder=5, label="root of line of dOGR gradients"
    )
    for point in additional_points_to_plot:
        plt.scatter(
            point[0],
            point[1],
            color="red",
            zorder=5,
        )
    for point in grads_points:
        plt.scatter(
            point[0],
            point[1],
            color="green",
            zorder=5,
        )
    plt.xlabel(f"Parameter value {param_name}{param_index}")
    plt.ylabel("Loss")
    plt.title("Dependence of loss on parameter")
    plt.legend()
    plt.show()


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
    global additional_points_to_plot

    BATCH_SIZE = 256
    net = torch.nn.Linear(3, 4)
    init.normal_(net.weight, mean=0.0, std=0.25)
    optim = dOGR(net.parameters(), beta=0.8)

    for i in range(4):
        optim.zero_grad()
        batch = (torch.ones(BATCH_SIZE, 3), torch.zeros(BATCH_SIZE, dtype=torch.long))
        x, y = batch
        logits = net(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        additional_points_to_plot.append(
            (
                dict(net.named_parameters())["weight"][0, 0].item(),
                loss.item(),
            )
        )
        grads_points.append(
            (
                dict(net.named_parameters())["weight"][0, 0].item(),
                dict(net.named_parameters())["weight"].grad[0, 0].item(),
            )
        )

        optim.step()

    batch = (torch.ones(BATCH_SIZE, 3), torch.zeros(BATCH_SIZE, dtype=torch.long))
    compare_with_hessian(net, optim, batch, "nazwa jakas tam")


def test3():
    net = get_mini_FC()
    optimizer = dOGR(net.parameters(), beta=0.8)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
    # optimizer = torch.optim.Adam(net.parameters())

    run(
        net,
        optimizer,
        max_epochs=1,
        batch_size=128,
        version=1,
        name="real_hess",
    )

    datamodule = MNISTDataModule(128)
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))

    compare_with_hessian(net, optimizer, batch, "")


if __name__ == "__main__":
    test2()
    # test3()
