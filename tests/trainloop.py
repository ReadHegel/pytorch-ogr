print("In module products __package__, __name__ ==", __package__, __name__)
import torch
import lightning as L
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.utils.data as data
from torch.utils.data import DataLoader
from src.optim.dOGR import dOGR
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
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # Pobieramy główny optymalizator (dOGR lub inny)
        optimizer = self.optimizers()

        # Sprawdzamy, czy użyliśmy dOGR z polityką NN
        is_nn_policy = isinstance(optimizer, dOGR) and getattr(optimizer, 'nn_policy', False)

        x, target = batch
        x = x.to(self.device)
        target = target.to(self.device)

        if is_nn_policy:
            # --- PĘTLA DLA dOGR Z POLITYKĄ NN (META-UCZENIE) ---
            
            # Pobieramy sieć-politykę i jej optymalizator
            policy_net = optimizer.policy_net
            policy_optimizer = optimizer.policy_optimizer
            
            # 1. Obliczamy stratę PRZED krokiem, aby mieć punkt odniesienia
            with torch.no_grad():
                loss_before = F.cross_entropy(self.net(x), target)

            # 2. Obliczamy gradienty dla GŁÓWNEJ SIECI
            optimizer.zero_grad()
            pred = self.net(x)
            loss = F.cross_entropy(pred, target)
            self.manual_backward(loss)
            
            # 3. Wykonujemy krok GŁÓWNYM optymalizatorem. 
            #    Metoda step() zmienia wagi modelu I ZWRACA podjęte akcje
            actions = optimizer.step()
            if actions is not None:
                actions = actions.to(self.device)

            # 4. Obliczamy stratę PO kroku, żeby zobaczyć efekt
            with torch.no_grad():
                loss_after = F.cross_entropy(self.net(x), target)
            
            # 5. Obliczamy nagrodę i stratę dla SIECI-POLITYKI
            reward = (loss_before - loss_after).detach() # Odłączamy od grafu
            policy_loss = -(actions * reward).mean() # Chcemy maksymalizować (akcja * nagroda)
            
            # 6. Trenujemy sieć-politykę
            policy_optimizer.zero_grad()
            # Używamy retain_graph=True, jeśli graf jest współdzielony, dla bezpieczeństwa
            policy_loss.backward(retain_graph=True) 
            policy_optimizer.step()

        else:
            # --- STANDARDOWA PĘTLA RĘCZNEJ OPTYMALIZACJI ---
            pred = self.net(x)
            loss = F.cross_entropy(pred, target)

            optimizer.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step() # Wywołujemy standardowy krok

        # Logujemy stratę i zwracamy ją
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
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
