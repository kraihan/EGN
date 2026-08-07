"""Affine-invariant Riemannian geometry of the SPD manifold.

The metric is ``g_S(U, V) = tr(S^-1 U S^-1 V)`` on ``S^n_++``. Under it the
manifold is Hadamard -- complete, simply connected, non-positively curved -- so
the Frechet mean exists, is unique, and the fixed-point iteration below
converges from any SPD seed.

All routines are batched and, with ``config.sync_free`` set (the default), free
of host synchronisation: the mean runs a fixed iteration budget instead of
polling a residual.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .config import config
from .functional import expm, invsqrtm, logm, powm, sqrtm_pair, sym, eigvals_logsq

__all__ = [
    "riemannian_log",
    "riemannian_exp",
    "squared_distance",
    "distance",
    "geodesic",
    "frechet_mean",
    "log_euclidean_mean",
    "parallel_transport_identity",
    "egrad2rgrad_spd",
    "random_spd",
    "is_spd",
]


def riemannian_log(S: Tensor, P: Tensor) -> Tensor:
    """``Log_S(P) = S^{1/2} log(S^{-1/2} P S^{-1/2}) S^{1/2}``."""
    root, inv = sqrtm_pair(S)
    return sym(root @ logm(sym(inv @ P @ inv)) @ root)


def riemannian_exp(S: Tensor, V: Tensor) -> Tensor:
    """``Exp_S(V) = S^{1/2} exp(S^{-1/2} V S^{-1/2}) S^{1/2}``.

    Exact for every symmetric ``V``: the output is SPD by construction, which is
    what lets the optimiser update SPD parameters without any projection step.
    """
    root, inv = sqrtm_pair(S)
    return sym(root @ expm(sym(inv @ V @ inv)) @ root)


def squared_distance(S: Tensor, P: Tensor) -> Tensor:
    """``d^2(S, P) = || log(S^{-1/2} P S^{-1/2}) ||_F^2``.

    The squared form is smooth everywhere (no cut locus) and its Riemannian
    gradient in ``S`` is exactly ``-2 Log_S(P)``, so no chain-rule correction is
    needed anywhere downstream. Shapes broadcast.
    """
    inv = invsqrtm(S)
    return eigvals_logsq(inv @ P @ inv)


def distance(S: Tensor, P: Tensor) -> Tensor:
    return squared_distance(S, P).clamp_min(0.0).sqrt()


def geodesic(A: Tensor, B: Tensor, t) -> Tensor:
    """``gamma(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}``.

    Satisfies ``gamma(0) = A``, ``gamma(1) = B`` and ``d(A, gamma(t)) = t d(A, B)``.
    This is the manifold-faithful interpolation; a convex combination stays in
    ``S^n_++`` but is not a geodesic of this metric.
    """
    root, inv = sqrtm_pair(A)
    return sym(root @ powm(sym(inv @ B @ inv), float(t)) @ root)


def frechet_mean(
    S: Tensor,
    dim: int = 1,
    weights: Optional[Tensor] = None,
    iters: int = 5,
    step: float = 1.0,
    keepdim: bool = False,
) -> Tensor:
    """Weighted Frechet (Karcher) mean under the AIRM.

        G <- G^{1/2} exp( step * sum_i w_i log(G^{-1/2} S_i G^{-1/2}) ) G^{1/2}

    Note the ``G^{1/2}`` factor on *both* sides; dropping the right-hand one
    yields an iterate that is not even symmetric.

    The iteration budget is fixed rather than residual-driven. Each residual test
    would cost a device synchronisation, and in practice the seed -- the
    arithmetic mean, which is already a first-order approximation -- puts the
    iterate within numerical noise of the fixed point in three to five steps.

    Parameters
    ----------
    S : (..., k, ..., n, n)
    dim : axis to average over
    weights : broadcastable against ``S`` along ``dim``, renormalised to sum to 1
    iters : fixed number of fixed-point iterations
    step : damping in ``(0, 1]``
    """
    k = S.shape[dim]
    if k == 1:
        return S if keepdim else S.squeeze(dim)

    if weights is None:
        w = S.new_full((1,) * (S.dim() - 2) + (1, 1), 1.0 / k)
    else:
        w = weights / weights.sum(dim=dim, keepdim=True).clamp_min(config.eps(S.dtype))

    G = (S * w).sum(dim=dim, keepdim=True)
    for _ in range(max(int(iters), 0)):
        root, inv = sqrtm_pair(G)
        tangent = logm(sym(inv @ S @ inv))
        drift = (tangent * w).sum(dim=dim, keepdim=True)
        G = sym(root @ expm(step * drift) @ root)
    return G if keepdim else G.squeeze(dim)


def log_euclidean_mean(
    S: Tensor,
    dim: int = 1,
    weights: Optional[Tensor] = None,
    keepdim: bool = False,
) -> Tensor:
    """``exp( sum_i w_i log S_i )`` -- closed form, one eigendecomposition.

    The barycentre of the flat log-Euclidean pullback metric. It stays exactly in
    ``S^n_++`` and costs O(1) decompositions instead of O(iters), but it is the
    Frechet mean of the AIRM only when the inputs commute.
    """
    if S.shape[dim] == 1:
        return S if keepdim else S.squeeze(dim)
    if weights is None:
        w = S.new_full((1,) * (S.dim() - 2) + (1, 1), 1.0 / S.shape[dim])
    else:
        w = weights / weights.sum(dim=dim, keepdim=True).clamp_min(config.eps(S.dtype))
    tangent = (logm(S) * w).sum(dim=dim, keepdim=True)
    out = expm(sym(tangent))
    return out if keepdim else out.squeeze(dim)


def parallel_transport_identity(S: Tensor, V: Tensor) -> Tensor:
    """Transport a tangent vector at the identity to ``S``: ``S^{1/2} V S^{1/2}``."""
    root, _ = sqrtm_pair(S)
    return sym(root @ V @ root)


def egrad2rgrad_spd(S: Tensor, egrad: Tensor) -> Tensor:
    """Euclidean gradient to AIRM gradient: ``S sym(dL/dS) S``.

    From ``g_S(grad, V) = <dL/dS, V>`` for all symmetric ``V``:
    ``tr(S^-1 grad S^-1 V) = tr(sym(dL/dS) V)``, hence ``grad = S sym(.) S``.
    For ``L = 1/2 d^2(S, P)`` this returns exactly ``-Log_S(P)``.
    """
    return S @ sym(egrad) @ S


def random_spd(
    *shape,
    condition: float = 10.0,
    dtype: torch.dtype = torch.float32,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Random SPD matrices with a controlled condition number.

    Eigenvalues are log-uniform in ``[1/sqrt(c), sqrt(c)]`` and eigenvectors are
    Haar-orthogonal, so samples are congruence-covariant and never near-singular.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    A = torch.randn(*shape, dtype=dtype, device=device, generator=generator)
    Q, _ = torch.linalg.qr(A)
    span = torch.tensor(float(condition), dtype=dtype, device=device).log()
    lam = (torch.rand(shape[:-1], dtype=dtype, device=device, generator=generator) - 0.5) * span
    return sym(Q @ torch.diag_embed(lam.exp()) @ Q.transpose(-1, -2))


def is_spd(S: Tensor, tol: float = 1e-6) -> Tensor:
    """Per-matrix boolean: symmetric and strictly positive definite."""
    asym = (S - S.transpose(-1, -2)).abs().amax(dim=(-2, -1))
    lam_min = torch.linalg.eigvalsh(sym(S.double())).amin(-1)
    return (asym <= tol) & (lam_min > 0)
