"""Objectives.

The head returns logits, so the objective is ordinary cross-entropy and autograd
produces the correct softmax derivative ``p(c) - 1[y = c]``. Composed with the
analytic derivative of the squared geodesic distance this reproduces

    grad_{P_c} L = (2 / (tau B)) sum_i ( p(c|S_i) - 1[y_i=c] ) Log_{P_c}(S_i)

which is exactly what the tests check against autograd. No custom backward is
needed anywhere, and writing one is how sign errors get in.

``separation`` optionally adds a repulsion term between prototypes of different
classes. It is off by default: with a learnable temperature the model can shrink
``tau`` instead of separating the prototypes, and the extra term is what stops
that degenerate solution when the class count is large.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .geometry import squared_distance

__all__ = ["PrototypeCrossEntropy", "build_criterion"]


class PrototypeCrossEntropy(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        separation: float = 0.0,
        class_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.separation = float(separation)
        self.register_buffer(
            "class_weight", class_weight if class_weight is not None else torch.empty(0)
        )

    def forward(self, logits: Tensor, target: Tensor, model: Optional[nn.Module] = None) -> Tensor:
        w = self.class_weight if self.class_weight.numel() else None
        loss = F.cross_entropy(
            logits, target, weight=w, label_smoothing=self.label_smoothing
        )
        if self.separation > 0 and model is not None:
            P = getattr(getattr(model, "head", None), "prototypes", None)
            if P is not None and P.shape[0] > 1:
                d2 = squared_distance(P.unsqueeze(0), P.unsqueeze(1))
                off = ~torch.eye(P.shape[0], dtype=torch.bool, device=P.device)
                loss = loss - self.separation * d2[off].clamp_max(100.0).mean()
        return loss


def build_criterion(
    label_smoothing: float = 0.0,
    separation: float = 0.0,
    class_weight: Optional[Tensor] = None,
) -> nn.Module:
    return PrototypeCrossEntropy(label_smoothing, separation, class_weight)
