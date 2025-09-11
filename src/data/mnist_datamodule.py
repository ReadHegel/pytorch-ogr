import lightning as L
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class MNISTDM(L.LightningDataModule):
    def __init__(self, datadir="DATA", batch_size=64, num_workers=4):
        super().__init__()
        self.datadir, self.batch_size, self.num_workers = datadir, batch_size, num_workers
    def setup(self, stage=None):
        self.train = MNIST(self.datadir, train=True,  download=True, transform=ToTensor())
        self.val   = MNIST(self.datadir, train=False, download=True, transform=ToTensor())
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def test_dataloader(self):
        return self.val_dataloader()
