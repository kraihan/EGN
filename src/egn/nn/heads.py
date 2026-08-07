"""Classification heads.

``GeodesicPrototypeHead``
    ``p(c | Sigma) = softmax_c( -d^2(Sigma, P_c) / tau )`` with the prototypes
    ``P_c`` themselves trainable SPD parameters. Geometry all the way to the
    logits. Cost is ``B * C`` eigenvalue computations per forward, which is the
    price of an exact geodesic distance.

``TangentHead``
    one logarithm per sample, then an ordinary linear classifier on the
    vectorised tangent matrix. Cost is ``B`` eigendecompositions, independent of
    the class count, which is what makes it the right default when ``C`` is
    large. It is not a weaker model -- it is the exact linear classifier in the
    tangent space at a learnable reference point.

Both are invariant in the joint sense: congruencing the input *and* the
reference/prototypes by the same matrix leaves the logits unchanged. Invariance
while holding the prototypes fixed is false -- congruencing only the input
changes every geodesic distance.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..config import config
from ..functional import eigvals_logsq, invsqrtm, logm, sqrtm_pair, sym
from ..geometry import random_spd
from ..manifolds import SPD, ManifoldParameter

__all__ = ["GeodesicPrototypeHead", "TangentHead", "vectorize_tangent"]


def vectorize_tangent(V: Tensor) -> Tensor:
    """Isometric vectorisation of a symmetric matrix.

    Off-diagonal entries are scaled by ``sqrt(2)`` so that the Euclidean norm of
    the vector equals the Frobenius norm of the matrix; a linear layer on top
    therefore sees an undistorted geometry.
    """
    n = V.shape[-1]
    idx = torch.triu_indices(n, n, device=V.device)
    scale = torch.where(idx[0] == idx[1], 1.0, math.sqrt(2.0)).to(V.dtype)
    return V[..., idx[0], idx[1]] * scale


class GeodesicPrototypeHead(nn.Module):
    """Softmax over negative squared geodesic distances to SPD prototypes.

    Parameters
    ----------
    dim : matrix size of the pooled feature
    num_classes : number of classes
    prototypes_per_class : more than one turns each class into a union of
        geodesic balls, at proportional cost; the class score is the softmin over
        its own prototypes
    temperature : initial ``tau``; learnable unless ``learn_temperature=False``
    chunk : cap on the number of (sample, prototype) pairs decomposed at once.
        Set it when ``B * C * n^2`` would not fit; ``0`` disables chunking.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        prototypes_per_class: int = 1,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        chunk: int = 0,
    ):
        super().__init__()
        self.dim, self.num_classes = int(dim), int(num_classes)
        self.per_class = int(prototypes_per_class)
        self.chunk = int(chunk)
        total = self.num_classes * self.per_class
        init = random_spd((total, dim, dim), condition=2.0)
        self.prototypes = ManifoldParameter(init, SPD())
        log_tau = torch.tensor(float(temperature)).log()
        self.log_temperature = (
            nn.Parameter(log_tau) if learn_temperature else nn.Parameter(log_tau, requires_grad=False)
        )

    def distances(self, S: Tensor) -> Tensor:
        """Squared geodesic distances, shape ``(B, num_classes * per_class)``.

        The prototype whitening factors are computed once per forward -- ``C``
        decompositions -- and reused for the whole batch, instead of once per
        (sample, prototype) pair.
        """
        inv = invsqrtm(self.prototypes)  # (P, n, n)
        X = S.unsqueeze(1)  # (B, 1, n, n)
        if self.chunk and inv.shape[0] > self.chunk:
            outs = [
                eigvals_logsq(c.unsqueeze(0) @ X @ c.unsqueeze(0))
                for c in inv.split(self.chunk, dim=0)
            ]
            return torch.cat(outs, dim=-1)
        W = inv.unsqueeze(0)  # (1, P, n, n)
        return eigvals_logsq(W @ X @ W)

    def forward(self, S: Tensor) -> Tensor:
        d2 = self.distances(S)
        tau = self.log_temperature.exp().clamp_min(1e-4)
        logits = -d2 / tau
        if self.per_class > 1:
            logits = logits.view(-1, self.num_classes, self.per_class).logsumexp(-1)
        return logits

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_classes={self.num_classes}, "
            f"prototypes_per_class={self.per_class}"
        )


class TangentHead(nn.Module):
    """Logarithm at a reference point, then a linear classifier.

    The reference is a trainable SPD parameter initialised at the identity, so
    the layer learns *where* on the manifold to linearise. With
    ``reference='identity'`` and ``learn_reference=False`` this reduces to the
    standard log-Euclidean readout used by SPDNet-style models.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        learn_reference: bool = True,
        batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim, self.num_classes = int(dim), int(num_classes)
        eye = torch.eye(dim)
        if learn_reference:
            self.reference = ManifoldParameter(eye.clone(), SPD())
        else:
            self.register_buffer("reference", eye.clone())
        feat = dim * (dim + 1) // 2
        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm1d(feat))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(feat, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, S: Tensor) -> Tensor:
        _, inv = sqrtm_pair(self.reference)
        V = logm(sym(inv @ S @ inv))
        return self.classifier(vectorize_tangent(V))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_classes={self.num_classes}"
