"""Manifold layers, assembled the way ``torch.nn`` is."""

from .blocks import EGNBlock
from .heads import GeodesicPrototypeHead, TangentHead, vectorize_tangent
from .inputs import ToSPD, infer_input_kind, infer_matrix_size
from .layers import (
    BiMap,
    GeodesicDropout,
    GeometricBias,
    SPDBatchNorm,
    SpectralActivation,
)
from .pooling import RiemannianPool

__all__ = [
    "BiMap",
    "EGNBlock",
    "GeodesicDropout",
    "GeodesicPrototypeHead",
    "GeometricBias",
    "RiemannianPool",
    "SPDBatchNorm",
    "SpectralActivation",
    "TangentHead",
    "ToSPD",
    "infer_input_kind",
    "infer_matrix_size",
    "vectorize_tangent",
]
