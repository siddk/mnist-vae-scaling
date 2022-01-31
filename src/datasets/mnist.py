"""
mnist.py

Core LightningDataModule for the MNIST dataset. Builds out "flattened" MNIST images (for simple MLP-VAE reconstruction).
"""
import logging
from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# Nest overwatch under root `mnist_vae` logger, inheriting formatting!
overwatch = logging.getLogger("mnist-vae.datasets.mnist")


class MNISTDataModule(LightningDataModule):
    def __init__(self, bsz: int, data_dir: str, download: bool = True):
        super().__init__()
        self.bsz, self.data_dir, self.download = bsz, data_dir, download

    def setup(self, stage: Optional[str] = None):
        mnist_full = MNIST(self.data_dir, train=True, download=self.download, transform=ToTensor())
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_train, batch_size=self.bsz, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_val, batch_size=self.bsz, shuffle=False)
