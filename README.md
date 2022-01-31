# MNIST VAE (Scaling)

Simple repository for training a VAE on the MNIST dataset, for testing/developing a solution for multi-cluster (+ cloud)
job submission. Built with  [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), using
[Anaconda](https://www.anaconda.com/) for Python dependencies and sane quality defaults (Black, Flake, isort).

---

## Contributing

+ Install and activate the Conda Environment using the `QUICKSTART` instructions below.

+ On installing new dependencies (via `pip` or `conda`), please make sure to update the `environment-<ID>.yaml` files
via the following command (note that you need to separately create the `environment-cpu.yaml` file by exporting from
your local development environment!):

  `make serialize-env --arch=<cpu | gpu>`

---

## Quickstart

Clones `mnist-vae-scaling` to the working directory, then walks through dependency setup, mostly leveraging the
`environment-<arch>.yaml` files.

### Shared Environment (for Clusters w/ Centralized Conda)

TBD.

### Local Development - Linux w/ GPU & CUDA 11.3

Note: Assumes that `conda` (Miniconda or Anaconda are both fine) is installed and on your path.

Ensure that you're using the appropriate `environment-<gpu | cpu>.yaml` file --> if PyTorch doesn't build properly for
your setup, checking the CUDA Toolkit is usually a good place to start. We have `environment-<gpu>.yaml` files for CUDA
11.3 (and any additional CUDA Toolkit support can be added -- file an issue if necessary).

```bash
https://github.com/siddk/mnist-vae-scaling.git
cd mnist-vae-scaling
conda env create -f environments/environment-gpu.yaml  # Choose CUDA Kernel based on Hardware - by default used 11.3!
conda activate mnist-vae-scaling
pre-commit install  # Important!
```

### Local Development - CPU (Mac OS & Linux)

Note: Assumes that `conda` (Miniconda or Anaconda are both fine) is installed and on your path. Use the `-cpu`
environment file.

```bash
https://github.com/siddk/mnist-vae-scaling.git
cd mnist-vae-scaling
conda env create -f environments/environment-cpu.yaml
conda activate mnist-vae-scaling
pre-commit install  # Important!
```

## Usage

This repository comes with sane defaults for `black`, `isort`, and `flake8` for formatting and linting. It additionally
defines a bare-bones Makefile (to be extended for your specific build/run needs) for formatting/checking, and dumping
updated versions of the dependencies (after installing new modules).

Other repository-specific usage notes should go here (e.g., training models, running a saved model, running a
visualization, etc.).

## Repository Structure

High-level overview of repository file-tree (expand on this as you build out your project). This is meant to be brief,
more detailed implementation/architectural notes should go in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

+ `conf` - Quinine Configurations (`.yaml`) for various runs (used in lieu of `argparse` or `typed-argument-parser`)
+ `environments` - Serialized Conda Environments for both CPU and GPU (CUDA 11.0). Other architectures/CUDA toolkit
environments can be added here as necessary.
+ `src/` - Source Code - has all utilities for preprocessing, Lightning Model definitions, utilities.
    + `preprocessing/` - Preprocessing Code (fill in details for specific project).
    + `models/` - Lightning Modules (fill in details for specific project).
+ `tests/` - Tests - Please test your code... just, please (more details to come).
+ `train.py` - Top-Level (main) entry point to repository, for training and evaluating models. Can define additional
top-level scripts as necessary.
+ `Makefile` - Top-level Makefile (by default, supports `conda` serialization, and linting). Expand to your needs.
+ `.flake8` - Flake8 Configuration File (Sane Defaults).
+ `.pre-commit-config.yaml` - Pre-Commit Configuration File (Sane Defaults).
+ `pyproject.toml` - Black and isort Configuration File (Sane Defaults).
+ `ARCHITECTURE.md` - Write up of repository architecture/design choices, how to extend and re-work for different
applications.
+ `CONTRIBUTING.md` - Detailed instructions for contributing to the repository, in furtherance of the default
instructions above.
+ `README.md` - You are here!
+ `LICENSE` - By default, research code is made available under the MIT License. Change as you see fit, but think
deeply about why!

---

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary for your collaborators
, since you'll be setting things up, but I like to keep this in the README in case things break in the future).
Generally, if you're just trying to run/use this code, look at the Quickstart section above.

### GPU & Cluster Environments (CUDA 11.3)

```bash
conda create --name mnist-vae-scaling python=3.8
conda activate mnist-vae-scaling
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch   # CUDA=11.0 on most of Cluster!
conda install ipython pytorch-lightning -c conda-forge

pip install black flake8 isort matplotlib pre-commit quinine rich wandb

# Install other dependencies via pip below -- conda dependencies should be added above (always conda before pip!)
...
```

### CPU Environments (Usually for Local Development -- Geared for Mac OS & Linux)

Similar to the above, but installs the CPU-only versions of Torch and similar dependencies.

```bash
conda create --name mnist-vae-scaling python=3.8
conda activate mnist-vae-scaling
conda install pytorch torchvision torchaudio -c pytorch
conda install ipython pytorch-lightning -c conda-forge

pip install black flake8 isort matplotlib pre-commit quinine rich wandb

# Install other dependencies via pip below -- conda dependencies should be added above (always conda before pip!)
...
```

### Containerized Setup

Support for running `mnist-vae-scaling` inside of a Docker or Singularity container is TBD. If this support is urgently
required, please file an issue.
