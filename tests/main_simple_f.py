import torch
import torch.nn as nn
from src.optim.OGR import OGR
from .trainloop import get_bp_hessian_from_loss


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.ones(1))
        self.y = nn.Parameter(torch.ones(1))

    def forward(self):
        return 3 * self.x**2 + self.y**2


def use_OGR():
    model = Quadratic()
    optimizer = OGR(model.parameters())
    def closure():
        optimizer.zero_grad()
        loss = model()
        return loss



    for step in range(20):
        loss = optimizer.step(closure=closure, use_line_search=False)
        real_loss = model()

        print("H estimated with OGR", optimizer.get_H())
        print("H real", get_bp_hessian_from_loss(real_loss, list(model.parameters())))

        print(
            f"Step {step:02d} | loss = {loss.item():.6f} | x = {model.x.item():.4f}, y = {model.y.item():.4f}"
        )


if __name__ == "__main__":
    use_OGR()
