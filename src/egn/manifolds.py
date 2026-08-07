"""Manifolds for constrained parameters.

Self-contained on purpose: the library depends on PyTorch and nothing else, so a
user can ``pip install egn`` without pulling a Riemannian-optimisation stack in.
Three manifolds cover every parameter EGN has.

``Stiefel``    orthonormal frames ``W in St(d, m)``, ``W^T W = I``, ``d >= m``
``SPD``        symmetric positive definite matrices, affine-invariant metric
``Euclidean``  everything else (temperatures, gains, classifier biases)

Each provides ``project`` (onto the manifold), ``egrad2rgrad`` (Euclidean to
Riemannian gradient), ``retract`` (move along a tangent direction and stay on
the manifold) and ``transport``. The optimiser in :mod:`egn.optim` reads the
manifold off each parameter, so adding a manifold needs no optimiser change.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .functional import expm, sym
from .config import config

__all__ = ["Manifold", "Euclidean", "Stiefel", "SPD", "ManifoldParameter"]


class Manifold:
    name = "manifold"

    def project(self, x: Tensor) -> Tensor:
        return x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project an ambient vector onto the tangent space at ``x``."""
        return u

    def egrad2rgrad(self, x: Tensor, g: Tensor) -> Tensor:
        return g

    def retract(self, x: Tensor, u: Tensor) -> Tensor:
        return x + u

    def transport(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        return u

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return (u * v).flatten(-2).sum(-1)

    def init(self, x: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        return x

    def __repr__(self) -> str:
        return self.name


class Euclidean(Manifold):
    name = "euclidean"


class Stiefel(Manifold):
    """``W^T W = I`` with ``W`` of shape ``(..., d, m)``, ``d >= m``.

    Retraction is by QR with a sign fix, which is the cheap second-order accurate
    choice; the canonical-metric gradient projection is used.
    """

    name = "stiefel"

    def project(self, x: Tensor) -> Tensor:
        q, r = torch.linalg.qr(x)
        sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return q * sign.unsqueeze(-2)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        xtu = x.transpose(-1, -2) @ u
        return u - x @ sym(xtu)

    def egrad2rgrad(self, x: Tensor, g: Tensor) -> Tensor:
        return self.proju(x, g)

    def retract(self, x: Tensor, u: Tensor) -> Tensor:
        return self.project(x + u)

    def transport(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        return self.proju(y, u)

    def init(self, x: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        with torch.no_grad():
            a = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            return self.project(a)


class SPD(Manifold):
    """SPD matrices under the affine-invariant metric.

    ``egrad2rgrad`` is ``X sym(G) X`` and the retraction is the exact exponential
    map, so a parameter that starts SPD can never leave the manifold, whatever
    the step size.

    ``max_step`` is a trust region on the *whitened* tangent norm. It does not
    exist to keep the parameter on the manifold -- the exponential map already
    guarantees that -- but to keep it in a numerically usable part of it. An
    unclipped Adam step early in training can push a prototype's condition number
    past what ``eigh`` will factor, and the run then dies inside a library call
    with an unhelpful convergence error. Clipping the step length is sync free
    and costs nothing once the iterates settle.
    """

    name = "spd"

    def __init__(self, max_step: float = 4.0, max_condition: float = 1e6):
        self.max_step = float(max_step)
        self.max_condition = float(max_condition)

    def project(self, x: Tensor) -> Tensor:
        """Clamp the spectrum to a *relative* floor.

        An absolute floor is not enough in float32: once the condition number
        passes ``1/eps`` the smallest eigenvalue of an exactly-SPD matrix is
        indistinguishable from zero after rounding, and the next ``eigh`` may
        return it negative. Bounding the condition number instead keeps
        positivity meaningful at the working precision.
        """
        s = sym(x)
        work = config.resolve_spectral_dtype(s)
        lam, U = torch.linalg.eigh(s.to(work))
        floor = lam.amax(-1, keepdim=True).clamp_min(config.eps(work)) / self.max_condition
        lam = torch.maximum(lam, floor).clamp_min(config.eps(work))
        return (U @ torch.diag_embed(lam) @ U.transpose(-1, -2)).to(x.dtype)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return sym(u)

    def egrad2rgrad(self, x: Tensor, g: Tensor) -> Tensor:
        return x @ sym(g) @ x

    def retract(self, x: Tensor, u: Tensor) -> Tensor:
        # exp_x(u) from a single decomposition of x:
        # x^{1/2} exp(x^{-1/2} u x^{-1/2}) x^{1/2}
        from .functional import sqrtm_pair

        root, inv = sqrtm_pair(x)
        w = sym(inv @ u @ inv)
        norm = w.flatten(-2).norm(dim=-1).clamp_min(config.eps(w.dtype))
        scale = (self.max_step / norm).clamp(max=1.0)[..., None, None]
        y = sym(root @ expm(w * scale) @ root)
        # one extra small decomposition, on parameters only, buys a hard bound on
        # the condition number and therefore a hard positivity guarantee
        return self.project(y) if self.max_condition else y

    def transport(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        return sym(u)

    def inner(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        xi = torch.linalg.inv(x)
        return torch.einsum("...ij,...ji->...", xi @ u, xi @ v)

    def init(self, x: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        from .geometry import random_spd

        with torch.no_grad():
            return random_spd(
                tuple(x.shape), condition=4.0, dtype=x.dtype, device=x.device, generator=generator
            )


class ManifoldParameter(nn.Parameter):
    """An ``nn.Parameter`` that carries the manifold it is constrained to.

    Survives ``state_dict`` round-trips, ``.to()``, DDP broadcast and
    ``deepcopy``; the optimiser dispatches on ``param.manifold``.
    """

    def __new__(cls, data: Tensor, manifold: Manifold, requires_grad: bool = True):
        obj = torch.Tensor._make_subclass(cls, data, requires_grad)
        obj.manifold = manifold
        return obj

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        out = ManifoldParameter(self.data.clone(), self.manifold, self.requires_grad)
        memo[id(self)] = out
        return out

    def __repr__(self) -> str:
        return f"ManifoldParameter({self.manifold}, shape={tuple(self.shape)})"
