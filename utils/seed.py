"""
Random seed utilities for reproducible experiments.
"""

import random

import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed used for Python, NumPy, and PyTorch.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior for CUDA operations where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


if __name__ == "__main__":

    set_seed(42)

    print("Seed configuration completed.")
