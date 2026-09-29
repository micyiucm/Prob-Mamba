"""Backward-compatible imports for the former notebook utility module."""

from .datasets import create_sequences, make_loaders
from .metrics import quasi_likelihood
from .numerics import gamma, rho, safe_cholesky, softplus_pos


def param_count(model) -> int:
    """Return the number of trainable parameters in a PyTorch module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
