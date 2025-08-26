import torch
from torch import nn

class PolicyNet(nn.Module):
    def __init__(self, input_features: int, output_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, output_features),
            nn.Sigmoid() 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)