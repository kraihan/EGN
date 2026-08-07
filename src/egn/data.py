"""Data adapters.

The library ships no dataset. It ships the three lines that turn whatever a user
already has -- a NumPy array, a torch tensor, a ``Dataset``, a ``DataLoader``, a
list of variable-length trials -- into something the training loop can consume.
That is the whole of ``egn.data``, and it is deliberate: a classifier library
that knows about specific benchmarks is a benchmark harness, not a library.

Labels may be integers, strings or any hashable; :class:`LabelEncoder` maps them
to contiguous indices and back, so ``predict`` returns labels in the caller's own
vocabulary.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

__all__ = [
    "LabelEncoder",
    "ArrayDataset",
    "as_dataset",
    "make_loader",
    "collate_padded",
]


class LabelEncoder:
    """Map arbitrary labels to ``0..C-1`` and back."""

    def __init__(self) -> None:
        self.classes_: Optional[np.ndarray] = None

    def fit(self, y) -> "LabelEncoder":
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> Tensor:
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder is not fitted")
        y = np.asarray(y).reshape(-1)
        lookup = {c: i for i, c in enumerate(self.classes_.tolist())}
        try:
            idx = [lookup[v] for v in y.tolist()]
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"unseen label {exc.args[0]!r}") from None
        return torch.as_tensor(idx, dtype=torch.long)

    def fit_transform(self, y) -> Tensor:
        return self.fit(y).transform(y)

    def inverse_transform(self, idx) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder is not fitted")
        idx = np.asarray(idx.cpu() if isinstance(idx, Tensor) else idx).reshape(-1)
        return self.classes_[idx]

    def __len__(self) -> int:
        return 0 if self.classes_ is None else len(self.classes_)


class ArrayDataset(Dataset):
    """Zero-copy view over an in-memory array or tensor, with optional labels."""

    def __init__(self, X, y=None, dtype: torch.dtype = torch.float32):
        self.X = torch.as_tensor(np.asarray(X) if not isinstance(X, Tensor) else X)
        self.X = self.X.to(dtype)
        self.y = None if y is None else torch.as_tensor(y).long().reshape(-1)
        if self.y is not None and len(self.y) != len(self.X):
            raise ValueError(
                f"X has {len(self.X)} samples but y has {len(self.y)}"
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


def collate_padded(batch):
    """Collate variable-length ``(C, T)`` trials by truncating to the shortest.

    Truncation, not padding: zero padding biases a covariance estimate towards
    the identity by exactly the padded fraction, which is a silent accuracy loss
    that is very hard to find later.
    """
    xs, ys = zip(*batch) if isinstance(batch[0], (tuple, list)) else (batch, None)
    t = min(x.shape[-1] for x in xs)
    X = torch.stack([x[..., :t] for x in xs])
    return (X, torch.stack([torch.as_tensor(y) for y in ys])) if ys else X


def as_dataset(X, y=None, dtype: torch.dtype = torch.float32) -> Dataset:
    """Accept an array, tensor, ``TensorDataset``, ``Dataset`` or list of trials."""
    if isinstance(X, DataLoader):
        return X.dataset
    if isinstance(X, Dataset):
        return X
    if isinstance(X, (list, tuple)) and len(X) and isinstance(X[0], (np.ndarray, Tensor)):
        if len({tuple(np.shape(x)) for x in X}) > 1:
            return _RaggedDataset(X, y, dtype)
    return ArrayDataset(X, y, dtype)


class _RaggedDataset(Dataset):
    def __init__(self, X, y, dtype):
        self.X = [torch.as_tensor(np.asarray(x)).to(dtype) for x in X]
        self.y = None if y is None else torch.as_tensor(y).long().reshape(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


def make_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    sampler=None,
    ragged: bool = False,
) -> DataLoader:
    """A ``DataLoader`` with the settings that matter for GPU throughput.

    ``pin_memory`` defaults to CUDA availability and ``persistent_workers`` is on
    whenever workers are used: EGN batches are small matrices, so worker startup
    would otherwise be a visible fraction of each epoch.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_padded if ragged else None,
    )
