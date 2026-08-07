"""Manifold-preserving layers.

Every layer maps ``S^d_++ -> S^m_++`` exactly. There is no projection step, no
re-symmetrisation of a matrix that had already drifted off the manifold, and no
additive Euclidean bias anywhere -- an SPD matrix plus an arbitrary symmetric
matrix is not SPD.

The correspondence with a convolutional network is deliberate:

===========================  =======================================
CNN                          EGN
===========================  =======================================
``Conv2d(c_in, c_out, k)``   ``BiMap(d_in, d_out, c_in, c_out)``
``BatchNorm2d``              ``SPDBatchNorm``
``ReLU``                     ``SpectralActivation``
``Dropout``                  ``GeodesicDropout``
``bias``                     ``GeometricBias`` (a metric isometry)
===========================  =======================================

``BiMap`` reduces the matrix size the way a strided convolution reduces spatial
resolution, and ``channels`` play the role of feature maps.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import config
from ..functional import expm, logm, powm, reeig, sqrtm_pair, sym
from ..geometry import frechet_mean, geodesic, log_euclidean_mean
from ..manifolds import SPD, ManifoldParameter, Stiefel

__all__ = [
    "BiMap",
    "GeometricBias",
    "SpectralActivation",
    "SPDBatchNorm",
    "GeodesicDropout",
]


class BiMap(nn.Module):
    """Congruence by an orthonormal frame: ``Sigma -> W^T Sigma W``.

    ``W`` lives on the Stiefel manifold ``St(d, m)`` with ``m <= d``, so the
    output is SPD whenever the input is (``W`` has full column rank), and the
    layer is dimensionality reducing exactly like a strided convolution.

    Equivariance. A congruence of the input by ``Q in O(d)`` is absorbed into a
    reparameterisation of the weight, ``W -> Q^T W``, not into a congruence of
    the output. The stronger claim -- that the output is congruent by ``Q`` --
    holds only in the square case ``d = m`` with ``W`` orthogonal.

    Parameters
    ----------
    in_dim, out_dim : matrix sizes, ``out_dim <= in_dim``
    in_channels, out_channels : manifold channels, as in a convolution
    constrained : keep ``W`` on the Stiefel manifold (default). With ``False``
        the weight is an ordinary unconstrained parameter updated by Adam. The
        congruence is then only positive *semi*-definite -- nothing stops ``W``
        from losing column rank -- so this setting relies on the spectral floor
        downstream and exists as an ablation of the manifold constraint.
    mix : if ``True`` each output channel is the sum of congruences of *all*
        input channels, ``Sigma'_o = (1/C) sum_i W_{oi}^T Sigma_i W_{oi}``, which
        is still SPD and gives the layer genuine cross-channel mixing. If
        ``False`` (default) channels are transformed independently, which is the
        form for which the reparameterisation identity above is stated.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_channels: int = 1,
        out_channels: int = 1,
        mix: bool = False,
        constrained: bool = True,
    ):
        super().__init__()
        if out_dim > in_dim:
            raise ValueError(f"BiMap cannot expand: out_dim={out_dim} > in_dim={in_dim}")
        if not mix and in_channels not in (1, out_channels):
            raise ValueError(
                f"without mixing, in_channels ({in_channels}) must be 1 or equal to "
                f"out_channels ({out_channels}); pass mix=True to combine channels"
            )
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.mix = bool(mix)
        self.constrained = bool(constrained)

        shape = (
            (out_channels, in_channels, in_dim, out_dim)
            if mix
            else (out_channels, in_dim, out_dim)
        )
        manifold = Stiefel()
        w = manifold.init(torch.empty(*shape))
        # both branches start from the same orthonormal draw, so the ablation
        # isolates the constraint during training rather than the initialisation
        self.weight = ManifoldParameter(w, manifold) if constrained else nn.Parameter(w)

    def forward(self, S: Tensor) -> Tensor:
        # S: (B, C_in, d, d)
        if self.mix:
            W = self.weight  # (C_out, C_in, d, m)
            X = S.unsqueeze(1)  # (B, 1, C_in, d, d)
            out = W.transpose(-1, -2) @ X @ W  # (B, C_out, C_in, m, m)
            out = out.mean(dim=2)
        else:
            W = self.weight  # (C_out, d, m)
            if self.in_channels == 1 and self.out_channels > 1:
                S = S.expand(-1, self.out_channels, -1, -1)
            out = W.transpose(-1, -2) @ S @ W
        return sym(out)

    def extra_repr(self) -> str:
        return (
            f"{self.in_dim}->{self.out_dim}, channels {self.in_channels}->"
            f"{self.out_channels}, mix={self.mix}, constrained={self.constrained}"
        )


class GeometricBias(nn.Module):
    """Learnable congruence by an invertible matrix: ``Sigma -> D Sigma D^T``.

    This is an isometry of the affine-invariant metric, so it moves features
    around the manifold without distorting any distance the head will measure --
    the manifold analogue of a bias term. ``D`` is parameterised by a Cholesky
    factor with a positive diagonal, so it is invertible by construction and
    never needs a post-hoc repair.
    """

    def __init__(self, dim: int, channels: int = 1):
        super().__init__()
        self.dim, self.channels = int(dim), int(channels)
        self.raw = nn.Parameter(torch.zeros(channels, dim, dim))
        self.log_diag = nn.Parameter(torch.zeros(channels, dim))

    def factor(self) -> Tensor:
        L = torch.tril(self.raw, diagonal=-1)
        return L + torch.diag_embed(self.log_diag.exp())

    def forward(self, S: Tensor) -> Tensor:
        D = self.factor()
        return sym(D @ S @ D.transpose(-1, -2))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, channels={self.channels}"


class SpectralActivation(nn.Module):
    """Isotropic nonlinearity on the spectrum.

    ``rect``   clamp eigenvalues from below (the manifold ReLU)
    ``power``  ``Sigma^t`` with learnable ``t > 0``: ``t < 1`` compresses the
               spectrum and is the cheapest available conditioner for deep stacks
    ``none``   identity

    Acting on eigenvalues only, these commute with congruence by any orthogonal
    matrix, so they leave the equivariance statement of the forward map intact.
    A matrix logarithm is *not* an activation -- it is a readout to the tangent
    space, and lives in the head.
    """

    def __init__(self, kind: str = "rect", floor: float = 1e-4, power: float = 0.5):
        super().__init__()
        if kind not in ("rect", "power", "none"):
            raise ValueError(f"unknown activation '{kind}'")
        self.kind = kind
        self.floor = float(floor)
        if kind == "power":
            self.log_power = nn.Parameter(torch.tensor(float(power)).log())

    def forward(self, S: Tensor) -> Tensor:
        if self.kind == "rect":
            return reeig(S, self.floor)
        if self.kind == "power":
            return _powm_learnable(S, self.log_power)
        return S

    def extra_repr(self) -> str:
        return f"kind={self.kind}, floor={self.floor}"


def _powm_learnable(S: Tensor, log_power: Tensor) -> Tensor:
    """``S^t`` with gradient flowing into ``t`` as well as into ``S``.

    ``d/dt S^t = S^t log S``, so one extra logarithm buys the derivative in the
    exponent; the eigendecomposition is shared through ``expm(t log S)``.
    """
    L = logm(S)
    return expm(log_power.exp() * L)


class SPDBatchNorm(nn.Module):
    """Riemannian batch normalisation on ``S^n_++``.

    Whitens by the batch Frechet mean and re-biases towards a learnable SPD
    reference ``G``::

        Sigma -> G^{1/2} ( M^{-1/2} Sigma M^{-1/2} ) G^{1/2}

    which is exactly the affine-invariant analogue of ``(x - mu)/sigma * gamma``.
    In eval mode ``M`` is the running mean, transported the same way, so the
    layer is deterministic at test time.

    This is the single most useful layer for making deep SPD stacks trainable:
    without it the spectra of successive congruences drift by orders of
    magnitude and the logarithms in the head saturate.
    """

    def __init__(
        self,
        dim: int,
        channels: int = 1,
        momentum: float = 0.1,
        mean: str = "logeuclid",
        iters: int = 3,
        learn_reference: bool = True,
    ):
        super().__init__()
        self.dim, self.channels = int(dim), int(channels)
        self.momentum = float(momentum)
        self.mean, self.iters = mean, int(iters)
        eye = torch.eye(dim).expand(channels, dim, dim).clone()
        self.register_buffer("running_mean", eye.clone())
        if learn_reference:
            self.reference = ManifoldParameter(eye.clone(), SPD())
        else:
            self.register_buffer("reference", eye.clone())

    def _batch_mean(self, S: Tensor) -> Tensor:
        if self.mean == "frechet":
            return frechet_mean(S, dim=0, iters=self.iters, keepdim=True)
        return log_euclidean_mean(S, dim=0, keepdim=True)

    def forward(self, S: Tensor) -> Tensor:
        if self.training:
            M = self._batch_mean(S)  # (1, C, n, n)
            with torch.no_grad():
                new = geodesic(self.running_mean, M.detach().squeeze(0), self.momentum)
                self.running_mean.copy_(new)
        else:
            M = self.running_mean.unsqueeze(0)
        _, inv = sqrtm_pair(M)
        white = sym(inv @ S @ inv)
        root, _ = sqrtm_pair(self.reference.unsqueeze(0))
        return sym(root @ white @ root)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, channels={self.channels}, mean={self.mean}"


class GeodesicDropout(nn.Module):
    """Stochastic interpolation towards the identity along the AIRM geodesic.

    ``Sigma -> gamma(t)`` on the geodesic from ``Sigma`` to ``I``, with a random
    ``t in [0, p]`` per sample. Unlike coordinate dropout this cannot leave the
    manifold, and unlike a convex combination with the identity it follows the
    metric the rest of the network uses.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = float(p)

    def forward(self, S: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return S
        # geodesic from S to I at parameter t is S^{1-t} in the whitened frame:
        # gamma(t) = S^{1/2} (S^{-1/2} I S^{-1/2})^t S^{1/2} = S^{1-t}
        t = torch.rand(S.shape[:-2] + (1, 1), dtype=S.dtype, device=S.device) * self.p
        L = logm(S)
        return expm((1.0 - t) * L)

    def extra_repr(self) -> str:
        return f"p={self.p}"
