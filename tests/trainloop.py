print("In module products __package__, __name__ ==", __package__, __name__)
import torch
import lightning as L
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.utils.data as data
from torch.utils.data import DataLoader

import os
import time 


SEED = 42
DATADIR = "DATA"


class TestLightningModule(L.LightningModule):
    def __init__(
        self,
        net,
        optimizer,
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self.net(x)

        loss = F.cross_entropy(pred, target)

        self.log_dict({
            "train_loss": loss,
        })
        return loss

    def on_train_epoch_start(self): 
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_time)

    def test_step(self, batch, batch_index):
        x, target = batch
        pred = self.net(x)

        loss = F.cross_entropy(pred, target)
        accuracy = (target == torch.argmax(pred, dim=1)).float().mean() * 100

        self.log_dict({"test_loss": loss, "test_accuracy": accuracy})

    def validation_step(self, batch, batch_index):
        x, target = batch
        pred = self.net(x)

        loss = F.cross_entropy(pred, target)
        accuracy = (target == torch.argmax(pred, dim=1)).float().mean() * 100

        self.log_dict({"val_loss": loss, "val_accuracy": accuracy})

    def configure_optimizers(self):
        return self.optimizer

def fit_and_test(module, trainer, datamodule):
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
