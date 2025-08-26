import torch
from torch import nn
import numpy as np

class FC(nn.Module):
    def __init__(self, input_dims=(1, 28, 28), num_classes=10):
        super().__init__()
        input_size = int(np.prod(input_dims))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class LeNet(nn.Module):
    def __init__(self, input_dims=(1, 28, 28), num_classes=10):
        super().__init__()
        self.input_dims = input_dims
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dims[0], out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        conv_output_size = self._get_conv_output_size(input_dims)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def _get_conv_output_size(self, shape):
        """Pomocnicza funkcja do obliczenia rozmiaru wyjścia z warstw konwolucyjnych."""
        with torch.no_grad():
            input_tensor = torch.rand(1, *shape)
            output_tensor = self.conv_layers(input_tensor)
        return int(np.prod(output_tensor.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def get_FC(input_dims=(1, 28, 28), num_classes=10):
    return FC(input_dims, num_classes)

def get_LeNet(input_dims=(1, 28, 28), num_classes=10):
    return LeNet(input_dims, num_classes)