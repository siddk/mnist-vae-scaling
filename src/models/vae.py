"""
vae.py

Core LightningModule defining a fairly simple MLP-based VAE for MNIST reconstruction.
"""
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


# Nest overwatch under root `mnist-vae` logger, inheriting formatting!
overwatch = logging.getLogger("mnist-vae.models.vae")


class VAE(LightningModule):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 2):
        """Creates a basic 2-layer MLP VAE Encoder/Decoder architecture."""
        super().__init__()
        self.input_dim, self.latent_dim, self.hidden_dim = input_dim, latent_dim, hidden_dim

        # Create Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.mu, self.log_sigma = [nn.Linear(self.hidden_dim, self.latent_dim) for _ in range(2)]

        # Create Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

    def embed(self, img: torch.Tensor) -> torch.Tensor:
        # Encode
        shared = self.encoder(img.flatten(start_dim=1))
        mu, sigma = self.mu(shared), torch.exp(self.log_sigma(shared))
        q = torch.distributions.Normal(mu, sigma)
        return q.rsample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).reshape(-1, 1, 28, 28)

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution, torch.distributions.Distribution]:
        # Encode
        shared = self.encoder(img.flatten(start_dim=1))
        mu, sigma = self.mu(shared), torch.exp(self.log_sigma(shared))

        # Create VAE Distribution & Prior
        q = torch.distributions.Normal(mu, sigma)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        z = q.rsample()

        # Decode
        x_hat = self.decoder(z).reshape(-1, 1, 28, 28)
        return x_hat, p, q

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Compute reconstruction loss
        x, _ = batch
        x_hat, p, q = self.forward(x)

        # Compute reconstruction loss
        mse_loss = F.mse_loss(x_hat, x, reduction="sum")

        # Compute KL term (q || p)
        kl_loss = torch.distributions.kl_divergence(q, p).mean()

        # Compute full loss
        loss = mse_loss + kl_loss

        # Log & return...
        self.log("Train Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # Compute reconstruction loss
        x, _ = batch
        x_hat, _, _ = self.forward(x)
        mse_loss = F.mse_loss(x_hat, x, reduction="sum")

        # Log...
        self.log("Val MSE", mse_loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters())
