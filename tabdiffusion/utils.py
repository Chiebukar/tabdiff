# tabdiffusion/utils.py
"""
Utility functions for TabDiffusion:
- device helpers
- reproducibility helpers
- cosine learning-rate schedule
- training/validation loss plotting
- general-purpose logging utilities
"""

import math
import numpy as np
import torch
import random
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Device Helpers
# ----------------------------------------------------------------------

def get_device(device="auto"):
    """
    Returns a torch.device object.

    Args:
        device: "auto" (default), "cpu", "cuda", or "mps"

    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

def set_seed(seed: int = 42):
    """
    Ensures deterministic behaviour across Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[TabDiffusion] Seed set to {seed}")


# ----------------------------------------------------------------------
# Cosine Schedule (for LR or beta noise)
# ----------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    Nicely shaped cosine schedule for diffusion noise betas.

    Args:
        timesteps: number of diffusion steps
        s: small offset to prevent extremely small early alphas

    Returns:
        numpy array of shape (timesteps,) containing betas
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)

    # cumulative product of alphas according to cosine curve
    alphas_cumprod = np.cos(((x / timesteps) + s) /
                            (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = []
    for t in range(1, steps):
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t - 1]
        beta = min(1 - a_t / a_prev, 0.999)
        betas.append(beta)

    return np.array(betas, dtype=np.float32)


# ----------------------------------------------------------------------
# Loss Plotting
# ----------------------------------------------------------------------

def plot_losses(train_losses, val_losses, title="Training & Validation Loss"):
    """
    Plot train/validation loss curves. Works with lists or NumPy arrays.

    Args:
        train_losses: list/array of training losses
        val_losses: list/array of validation losses
        title: plot title

    Returns:
        None
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))

    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="s")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Loss Logging
# ----------------------------------------------------------------------

def log_loss(epoch, train_loss, val_loss):
    """
    Standardized logger for training loops.
    
    Args:
        epoch: int
        train_loss: float
        val_loss: float

    Returns:
        None
    """
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
