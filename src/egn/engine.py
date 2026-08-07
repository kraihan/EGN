"""Training and evaluation loops.

Written for throughput on GPU:

* one host synchronisation per epoch, not per step -- running losses are
  accumulated as tensors on the device and read once at the end;
* ``non_blocking`` transfers paired with pinned host memory;
* optional autocast. Note what autocast does and does not do here: the spectral
  operators always upcast to at least float32 internally and emit float32, so
  autocast speeds up the dense matmuls and the linear head while leaving every
  eigendecomposition at full precision. bfloat16 is the safe choice; float16 is
  accepted but its dynamic range is a poor fit for spectra spanning several
  orders of magnitude, so a ``GradScaler`` is used with it and divergence is
  still possible.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.dist import all_reduce_mean, is_distributed, unwrap

__all__ = ["train_one_epoch", "evaluate"]


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    grad_clip: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler=None,
) -> Dict[str, float]:
    model.train()
    total = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    loss_sum = torch.zeros((), device=device)

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp, amp_dtype):
            logits = model(x)
            loss = criterion(logits, y, unwrap(model)) if _takes_model(criterion) else criterion(logits, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = y.numel()
        loss_sum += loss.detach() * bs
        correct += (logits.detach().argmax(-1) == y).sum()
        total += bs

    return _reduce(loss_sum, correct, total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    model.eval()
    total = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    loss_sum = torch.zeros((), device=device)

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with _autocast(device, amp, amp_dtype):
            logits = model(x)
            if criterion is not None:
                loss = (
                    criterion(logits, y, unwrap(model))
                    if _takes_model(criterion)
                    else criterion(logits, y)
                )
                loss_sum += loss.detach() * y.numel()
        correct += (logits.argmax(-1) == y).sum()
        total += y.numel()

    return _reduce(loss_sum, correct, total)


def _reduce(loss_sum, correct, total) -> Dict[str, float]:
    if is_distributed():
        stacked = torch.stack([loss_sum, correct, total])
        torch.distributed.all_reduce(stacked)
        loss_sum, correct, total = stacked.unbind()
    total_f = total.clamp_min(1)
    return {
        "loss": (loss_sum / total_f).item(),
        "accuracy": (correct / total_f).item(),
        "count": int(total.item()),
    }


def _takes_model(criterion) -> bool:
    from .losses import PrototypeCrossEntropy

    return isinstance(criterion, PrototypeCrossEntropy)
