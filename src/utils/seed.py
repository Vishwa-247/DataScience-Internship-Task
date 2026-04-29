"""Deterministic seeding across `random`, `numpy`, and `torch` (CPU + CUDA)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we touch so training runs are reproducible.

    Args:
        seed: Integer seed shared across all RNGs.

    Side effects:
        Mutates `PYTHONHASHSEED`, the `random` and `numpy` global RNGs, and
        (when available) `torch` CPU/CUDA RNGs and cuDNN flags.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism > throughput for this project (43 small series, no need for cudnn autotune)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
