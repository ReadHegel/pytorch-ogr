import lightning.pytorch as pl
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path
        
PATH_DATASETS = str(Path(__file__).parent.parent / "DATA")

class ZeroTargetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y%2

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [55000, 5000],
                generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)

class ZEROMNISTDataModule(MNISTDataModule):
    def setup(self, stage=None):
        # najpierw normalna logika z klasy bazowej
        super().setup(stage)

        # podmieniamy targety na zero
        if stage == "fit" or stage is None:
            self.mnist_train = ZeroTargetDataset(self.mnist_train)
            self.mnist_val = ZeroTargetDataset(self.mnist_val)

        if stage == "test" or stage is None:
            self.mnist_test = ZeroTargetDataset(self.mnist_test)
