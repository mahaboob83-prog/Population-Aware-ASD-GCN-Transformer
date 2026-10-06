"""
Reproducibility utilities.

Provides a single function for configuring random seeds across
Python, NumPy, and PyTorch.
"""

import os
import random

import numpy as np
import torch


SEED = 42


def set_seed(seed=SEED):
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed used for Python, NumPy, and PyTorch.
    """

    if not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, got {type(seed).__name__}."
        )

    # Python random generator
    random.seed(seed)

    # NumPy random generator
    np.random.seed(seed)

    # PyTorch CPU random generator
    torch.manual_seed(seed)

    # PyTorch CUDA random generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    set_seed(SEED)

    first = torch.rand(5)

    set_seed(SEED)

    second = torch.rand(5)

    assert torch.equal(
        first,
        second
    )

    print(f"Seed: {SEED}")
    print("Seed reproducibility test passed.")
