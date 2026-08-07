"""Distributed helpers.

EGN is a small-matrix, high-arithmetic-intensity workload, so scaling is by
``DistributedDataParallel`` -- one process per GPU. ``nn.DataParallel`` is
supported as a fallback but is not recommended: it re-scatters the model every
step and serialises the Python side, which on a network whose layers are already
short kernel launches costs more than it saves.

Nothing here is required to use the library on a single device; every function
degrades to a no-op when the process was not launched with ``torchrun``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    "is_distributed",
    "world_size",
    "rank",
    "local_rank",
    "is_main_process",
    "init_distributed",
    "cleanup_distributed",
    "barrier",
    "all_reduce_mean",
    "unwrap",
]


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return rank() == 0


def init_distributed(backend: Optional[str] = None) -> bool:
    """Initialise the process group if ``torchrun`` set the environment.

    Returns True when this process is part of a group. Safe to call in a plain
    single-process script -- it does nothing and returns False.
    """
    if is_distributed():
        return True
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return True


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across ranks (identity when not distributed)."""
    if not is_distributed():
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / world_size()


def unwrap(model):
    """Strip ``DistributedDataParallel`` / ``DataParallel`` wrappers."""
    return getattr(model, "module", model)
