"""The residual-free EGN block: the manifold analogue of ``conv -> bn -> relu``.

    Sigma -> BiMap -> SPDBatchNorm -> GeometricBias -> SpectralActivation -> dropout

Stacking blocks with decreasing ``dim`` and increasing ``channels`` reproduces
the shape schedule of a convolutional trunk, and the same intuitions transfer:
narrow early layers under-fit, an over-aggressive dimension drop in one step
loses spectrum that later layers cannot recover.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .layers import BiMap, GeodesicDropout, GeometricBias, SpectralActivation, SPDBatchNorm

__all__ = ["EGNBlock"]


class EGNBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_channels: int = 1,
        out_channels: int = 1,
        mix: bool = False,
        constrained: bool = True,
        batchnorm: bool = True,
        bias: bool = True,
        activation: str = "rect",
        floor: float = 1e-4,
        dropout: float = 0.0,
        bn_mean: str = "logeuclid",
    ):
        super().__init__()
        self.bimap = BiMap(
            in_dim, out_dim, in_channels, out_channels, mix=mix, constrained=constrained
        )
        self.norm = SPDBatchNorm(out_dim, out_channels, mean=bn_mean) if batchnorm else None
        self.bias = GeometricBias(out_dim, out_channels) if bias else None
        self.act = SpectralActivation(activation, floor=floor)
        self.drop = GeodesicDropout(dropout) if dropout > 0 else None

    def forward(self, S: Tensor) -> Tensor:
        S = self.bimap(S)
        if self.norm is not None:
            S = self.norm(S)
        if self.bias is not None:
            S = self.bias(S)
        S = self.act(S)
        if self.drop is not None:
            S = self.drop(S)
        return S
