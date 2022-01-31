"""
evaluate.py

Evaluation script -- loads each checkpoint (can be run async) and plots the learned latent space (across validation
examples), as well as 3 GiFs of latent space interpolations.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from src.datasets import MNISTDataModule
from src.models import VAE
from src.overwatch import get_overwatch


# Constants (usually passed in via YAML...)
RUN_ID, VIZ_PATH, DATA_DIR, BSZ, MAX_EPOCHS, SEED = "mnist-vae+x21", "visualizations/", "data/", 256, 20, 21


def evaluate() -> None:
    # Spawn overwatch & log details
    overwatch = get_overwatch(level=20, name="mnist-vae")
    overwatch.info(f"Starting evaluation of run `{RUN_ID}`...")

    # We're outside the Lightning abstraction -- manual device placement...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED, workers=True)

    # Create visualization directory
    os.makedirs(os.path.join(VIZ_PATH, RUN_ID), exist_ok=True)

    # Data handling
    mnist_datamodule = MNISTDataModule(BSZ, DATA_DIR, download=False)
    mnist_datamodule.setup()

    # While Loop on MAX_EPOCHS
    epoch, vae = 0, None
    while epoch < MAX_EPOCHS:
        overwatch.info(f"Evaluating checkpoint @ Epoch {epoch}...")

        # Find valid checkpoint...
        checkpoint = [x for x in os.listdir(f"runs/{RUN_ID}") if f"epoch={epoch}-" in x]
        assert len(checkpoint) <= 1, "Poorly constructed string match..."
        if len(checkpoint) == 0:
            continue

        # Instantiate model & datamodule
        vae = VAE.load_from_checkpoint(os.path.join(f"runs/{RUN_ID}", checkpoint[0])).to(device)

        # Iterate through validation set & visualize latent space
        with torch.no_grad():
            for i, (img, label) in enumerate(mnist_datamodule.val_dataloader()):
                z = vae.embed(img).to("cpu").detach().numpy()
                plt.scatter(z[:, 0], z[:, 1], c=label, cmap="tab10")

        # Create colorbar, and save figure
        plt.colorbar()
        plt.savefig(os.path.join(VIZ_PATH, RUN_ID, f"manifold-epoch={epoch}.png"))
        plt.clf()

        # Bump epoch
        epoch += 1

    # Drop 3 Interpolation GIFs
    overwatch.info("Generating interpolation GIFs...")
    dl = mnist_datamodule.val_dataloader()
    imgs, labels = next(iter(dl))

    # Generate GIFs for morphing 0 --> 8, 9 --> 6, 5 --> 2
    for (start, end) in [(0, 8), (9, 6), (5, 2)]:
        overwatch.info(f"Generating GIF for interpolating between `{start}` and `{end}`...")
        x1 = imgs[labels == start][7]
        x2 = imgs[labels == end][7]

        # Compute Zs
        z1, z2 = vae.embed(x1), vae.embed(x2)
        z = torch.stack([z1 + (z2 - z1) * t for t in np.linspace(0, 1, num=100)])

        # Create list of interpolated images
        interpolated = vae.decode(z).to("cpu").detach().numpy() * 255
        images = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolated]
        images += images[::-1]  # UX for looping forward/backward

        # Save GIF
        images[0].save(
            os.path.join(VIZ_PATH, RUN_ID, f"{start}-{end}.gif"), save_all=True, append_images=images[1:], loop=1
        )


if __name__ == "__main__":
    evaluate()
