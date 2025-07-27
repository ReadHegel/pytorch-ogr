import torch 
from torch import nn

def get_FC(): 
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64), 
        nn.ReLU(),
        nn.Linear(64, 64), 
        nn.ReLU(),
        nn.Linear(64, 10),
    )

def get_mini_FC(): 
    return nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2 * 14 * 14, 20),
        nn.ReLU(),
        nn.Linear(20, 20), 
        nn.ReLU(),
        nn.Linear(20, 10),
    )

def get_LeNet(): 
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), 
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), 
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Linear(16 * 5 * 5, 120), 
        nn.ReLU(),
        nn.Linear(120, 84), 
        nn.ReLU(),
        nn.Linear(84, 10),
    )

