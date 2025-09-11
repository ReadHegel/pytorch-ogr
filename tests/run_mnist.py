import argparse
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.optim.dOGR import dOGR
from src.optim.policy_net import PolicyNet
from tests.trainloop import TestLightningModule


PROJ = Path(__file__).resolve().parents[1]
LOGDIR = PROJ / "lightning_logs"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.fc(x)


def make_loaders(datadir="DATA", batch_size=64):
    tfm = ToTensor()
    train = MNIST(datadir, train=True, download=True, transform=tfm)
    val = MNIST(datadir, train=False, download=True, transform=tfm)
    # WINDOWS: trzymaj num_workers = 0 i persistent_workers = False
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=0, persistent_workers=False
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False,
        num_workers=0, persistent_workers=False
    )
    return train_loader, val_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_as_zeros", type=int, default=0, help="0/1: inicjalizacja zerowa polityki")
    p.add_argument("--max_steps", type=int, default=0, help="maks. liczba kroków treningu (0 = domyślnie PL)")
    p.add_argument("--logger_name", type=str, default="mnist_run", help="nazwa eksperymentu (katalog w lightning_logs)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--data_dir", type=str, default="DATA")
    return p.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = parse_args()

    L.seed_everything(42, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model i polityka
    net = Net().to(device)
    policy_net = PolicyNet(input_features=4, output_features=1, hidden=128, n_layers=2).to(device)
    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

    # dOGR z polityką NN
    opt = dOGR(
        params=net.parameters(),
        policy_net=policy_net,
        policy_optimizer=policy_opt,
        net=net,
        lr=1e-2,
        beta=0.5,
        trust_factor=1.0,
        nn_policy=True,
        policy_std=0.1,
        init_as_zeros=bool(args.init_as_zeros),
        differentiable=True,
    )

    module = TestLightningModule(net, opt)
    train_loader, val_loader = make_loaders(datadir=args.data_dir, batch_size=args.batch_size)

    # Logger -> lightning_logs/<logger_name>/version_*/
    logger = CSVLogger(save_dir=str(LOGDIR), name=args.logger_name)

    # Trainer: użyj max_steps jeśli > 0; inaczej zostaw domyślne zachowanie PL
    trainer_kwargs = dict(
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    if args.max_steps and args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(module, dataloaders=val_loader)
