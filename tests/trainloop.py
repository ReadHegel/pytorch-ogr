#print("In module products __package__, __name__ ==", __package__, __name__)
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


def _has_nn_policy(opt) -> bool:
    return isinstance(opt, dOGR) and any(getattr(g, "get", lambda k, d=None: d)("nn_policy", False) if isinstance(g, dict) else g.get("nn_policy", False) for g in getattr(opt, "param_groups", []))


class TestLightningModule(L.LightningModule):
    def __init__(self, net, optimizer):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        # Ręczna optymalizacja
        self.automatic_optimization = False
        # EMA baseline do stabilizacji REINFORCE
        self.register_buffer("baseline", torch.tensor(0.0))

    def training_step(self, batch, batch_idx):
        # Pobieramy główny optymalizator (dOGR lub inny)
        optimizer = self.optimizers()

        is_nn_policy = _has_nn_policy(optimizer)

        x, target = batch
        x = x.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        if is_nn_policy:
            # --- PĘTLA DLA dOGR Z POLITYKĄ NN (REINFORCE) ---
            policy_net = optimizer.policy_net
            policy_optimizer = optimizer.policy_optimizer

            # Loss PRZED krokiem
            with torch.no_grad():
                loss_before = F.cross_entropy(self.net(x), target)

            # Backprop dla sieci głównej
            optimizer.zero_grad()
            pred = self.net(x)
            loss = F.cross_entropy(pred, target)
            self.manual_backward(loss)

            # Krok dOGR sterowany polityką → (actions, log_prob[, entropy])
            out = optimizer.step()
            if not (isinstance(out, tuple) and len(out) in (2, 3)):
                raise RuntimeError("dOGR.step() musi zwracać (actions, log_prob[, entropy]) dla gałęzi nn_policy.")

            if len(out) == 2:
                actions, log_prob = out
                entropy = None
            else:
                actions, log_prob, entropy = out

            log_prob = torch.clamp(log_prob, -10, 10)

            # Loss PO kroku
            with torch.no_grad():
                loss_after = F.cross_entropy(self.net(x), target)

            # REINFORCE: policy_loss = - log_prob * advantage  (+ opcjonalny entropy bonus)
            reward = (loss_before - loss_after).detach()
            # Aktualizacja EMA baseline
            self.baseline = 0.9 * self.baseline + 0.1 * reward.mean()
            # Standaryzacja i sprowadzenie do SKALARA
            adv = reward - self.baseline
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            advantage = adv.mean()

            # log_prob to skalar
            policy_loss = -(log_prob * advantage)
            if entropy is not None:
                policy_loss = policy_loss - 1e-2 * entropy

            # Update polityki
            policy_optimizer.zero_grad()
            self.manual_backward(policy_loss)
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            policy_optimizer.step()

            # Logi diagnostyczne
            self.log("policy_loss", policy_loss, on_step=True, prog_bar=True)
            self.log("advantage", advantage, on_step=True, prog_bar=False)
            self.log("reward", reward.mean(), on_step=True, prog_bar=False)
            self.log("loss_after", loss_after, on_step=True, prog_bar=False)

        else:
            # --- STANDARDOWA PĘTLA  OPTYMALIZACJI ---
            pred = self.net(x)
            loss = F.cross_entropy(pred, target)

            optimizer.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step()

        # Log straty
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_time)

    def test_step(self, batch, batch_index):
        x, target = batch
        x = x.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        pred = self.net(x)
        loss = F.cross_entropy(pred, target)
        accuracy = (target == torch.argmax(pred, dim=1)).float().mean() * 100

        self.log_dict({"test_loss": loss, "test_accuracy": accuracy})

    def validation_step(self, batch, batch_index):
        x, target = batch
        x = x.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        pred = self.net(x)
        loss = F.cross_entropy(pred, target)
        accuracy = (target == torch.argmax(pred, dim=1)).float().mean() * 100

        self.log_dict({"val_loss": loss, "val_accuracy": accuracy})

    def configure_optimizers(self):
        return self.optimizer

def fit_and_test(module, trainer, datamodule):
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
