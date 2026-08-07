"""Reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "worker_init_fn"]


def seed_everything(seed: int = 1234, deterministic: bool = False, threads: int = 0) -> int:
    """Seed every RNG the pipeline touches.

    ``threads=0`` leaves the BLAS thread count alone. Capping it is worth doing
    only when you are timing CPU runs: the layers are dominated by many small
    ``eigh`` calls, and an oversubscribed thread pool makes wall-clock numbers
    noisy. It costs throughput otherwise.

    ``deterministic=True`` also disables the nondeterministic CUDA kernels, which
    is what you want when reproducing a published number and not what you want
    when training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if threads:
        torch.set_num_threads(int(threads))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
