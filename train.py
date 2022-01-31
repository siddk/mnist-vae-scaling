"""
train.py

Core training script -- loads and preprocesses, instantiates a Lightning Module, and runs training. Fill in with more
repository/project-specific training details!

Run with: `python train.py`
"""
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary

from src.datasets import MNISTDataModule
from src.models import VAE
from src.overwatch import get_overwatch


# Constants (usually passed in via YAML...)
LOG_DIR, DATA_DIR, RUN_DIR = "logs/", "data/", "runs/"
BSZ, EPOCHS, LATENT_DIM, HIDDEN_DIM, SEED = 256, 20, 2, 512, 21
GPUS = 1 if torch.cuda.is_available() else 0


def train() -> None:
    # Create unique run identifier...
    run_id = f"mnist-vae+x{SEED}"

    # Spawn overwatch & log details
    overwatch = get_overwatch(level=20, name="mnist-vae")
    overwatch.info(f"Starting run: `{run_id}`...")

    # Set randomness, ensure dataloader worker randomness
    overwatch.info(f"Setting random seed to `{SEED}`!")
    seed_everything(SEED, workers=True)

    # Get MNIST DataModule
    mnist_datamodule = MNISTDataModule(BSZ, DATA_DIR, download=True)

    # Build VAE
    vae = VAE(input_dim=784, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)

    # Create Callbacks
    callbacks = [
        RichModelSummary(),
        ModelCheckpoint(
            dirpath=os.path.join(RUN_DIR, run_id),
            filename="epoch={epoch}-val_mse={Val MSE:.3f}",
            monitor="Val MSE",
            mode="min",
            save_top_k=EPOCHS,
            auto_insert_metric_name=False,
        ),
    ]

    # Fit
    trainer = Trainer(max_epochs=EPOCHS, gpus=GPUS, log_every_n_steps=10, logger=None, callbacks=callbacks)
    trainer.fit(vae, datamodule=mnist_datamodule)


if __name__ == "__main__":
    train()
