"""Pooling over the manifold-channel axis.

The analogue of global average pooling in a CNN, except that the average is a
Frechet mean. Two estimators are available and the choice is the main
speed/fidelity dial in the network:

``frechet``    the true Riemannian barycentre, by damped fixed-point iteration.
               Cost: ``iters`` eigendecompositions per sample.
``logeuclid``  the closed-form log-Euclidean barycentre. Cost: one
               eigendecomposition per matrix, no iteration, and no
               data-dependent control flow at all. Equal to the Frechet mean
               when the inputs commute, close to it when they nearly do.
``arithmetic`` the plain Euclidean average. Stays in ``S^n_++`` -- the cone is
               convex -- but it is the barycentre of no Riemannian metric, and
               it inflates the spectrum: the average of two matrices related by
               a congruence has a larger determinant than either. Provided as an
               ablation, not as a recommendation.

``logeuclid`` is the default because on GPU the iteration is the dominant cost
of a forward pass and the accuracy difference is typically within seed noise.
Switch to ``frechet`` when the channel spread is large.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..geometry import frechet_mean, log_euclidean_mean

__all__ = ["RiemannianPool"]


class RiemannianPool(nn.Module):
    def __init__(
        self,
        mode: str = "logeuclid",
        dim: int = 1,
        iters: int = 5,
        learn_weights: bool = False,
        channels: int = 1,
    ):
        super().__init__()
        if mode not in ("frechet", "logeuclid", "arithmetic"):
            raise ValueError(f"unknown pooling mode '{mode}'")
        self.mode, self.dim, self.iters = mode, int(dim), int(iters)
        if learn_weights and channels > 1:
            self.logit_weights: Optional[nn.Parameter] = nn.Parameter(torch.zeros(channels))
        else:
            self.logit_weights = None

    def forward(self, S: Tensor) -> Tensor:
        w = None
        if self.logit_weights is not None:
            w = torch.softmax(self.logit_weights, 0).view(
                (1, -1) + (1,) * (S.dim() - 2)
            ).to(S.dtype)
        if self.mode == "frechet":
            return frechet_mean(S, dim=self.dim, weights=w, iters=self.iters)
        if self.mode == "logeuclid":
            return log_euclidean_mean(S, dim=self.dim, weights=w)
        if w is None:
            return S.mean(dim=self.dim)
        return (S * w).sum(dim=self.dim) / w.sum(dim=self.dim).clamp_min(1e-12)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, dim={self.dim}, iters={self.iters}"
