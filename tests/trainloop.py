#print("In module products __package__, __name__ ==", __package__, __name__)
import torch
import lightning as L
from torch.func import grad
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.utils.data as data
from torch.utils.data import DataLoader
from src.optim.dOGR import dOGR
import os
import time
import time


SEED = 42
DATADIR = "DATA"


def get_bp_hessian_from_loss(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])
    n = grad_flat.numel()

    H_rows = []
    for i in range(n):
        g2 = torch.autograd.grad(
            grad_flat[i], params, retain_graph=True, create_graph=True
        )
        g2_flat = torch.cat([g.flatten() for g in g2])
        H_rows.append(g2_flat)

    return torch.stack(H_rows)


class TestLightningModule(L.LightningModule):
    def __init__(
        self,
        net,
        optimizer_cls,
        optimizer_kwargs=None,
    ):
        super().__init__()
        self.net = net
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}

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

        self.log_dict(
            {
                "train_loss": loss,
            }
        )
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
        with torch.enable_grad():
            pred = self.net(x)

            loss = F.cross_entropy(pred, target)
            accuracy = (target == torch.argmax(pred, dim=1)).float().mean() * 100

            log_dict = {
                "val_loss": loss,
                "val_accuracy": accuracy,
            }

            opt = self.optimizers()
            H_bp = get_bp_hessian_from_loss(loss, list(self.net.parameters()))
            H_inv = None
            H = None

            if hasattr(opt, "get_H_inv"):
                H_inv = opt.get_H_inv()
                try:
                    # pseudo-odwrotność H_bp dla stabilności
                    H_bp_inv = torch.linalg.pinv(H_bp)
                    mse_H_inv = (
                        H_inv.flatten() - H_bp_inv.flatten()
                    ) ** 2 / H_inv.numel()
                    log_dict["hessian_inv_mse"] = torch.sum(mse_H_inv)
                except RuntimeError as e:
                    print("Shape mismatch in H_inv comparison:", e)
                print("H_inv", H_inv)
                print("H_bp_inv", H_bp_inv)

            if hasattr(opt, "get_H"):
                H = opt.get_H()
                try:
                    mse_H = (H_bp.flatten() - H.flatten()) ** 2 / H.numel()
                    log_dict["hessian_mse"] = torch.sum(mse_H)
                except RuntimeError as e:
                    print("Shape mismatch in H comparison:", e)
                print("H", H)
                print("H_bp", H_bp)

        self.log_dict(log_dict)

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)


def fit_and_test(module, trainer, datamodule):
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
