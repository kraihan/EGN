"""Batched spectral operators on symmetric / SPD matrices.

Every operator here is

* **batched** -- it accepts any leading shape ``(..., n, n)`` and issues exactly
  one ``eigh`` per call, never a Python loop over the batch;
* **sync free** -- no ``.item()``, no data-dependent control flow, so the CUDA
  queue is never drained mid-forward;
* **autocast safe** -- inputs are cast to the resolved spectral dtype before the
  decomposition and back afterwards, so the operators can sit inside an
  ``autocast`` region without producing half-precision eigendecompositions;
* **differentiable through repeated eigenvalues** -- the backward pass uses the
  Loewner (divided-difference) matrix, and clamped eigenvalues receive a zero
  subgradient rather than an unbounded one.

A spectral map is ``M = U diag(lam) U^T  ->  U diag(phi(lam)) U^T``. Its
differential is ``U ( L * (U^T sym(dX) U) ) U^T`` with

    L_ij = (phi(lam_i) - phi(lam_j)) / (lam_i - lam_j)   for distinct eigenvalues
    L_ii = phi'(lam_i)                                   on the diagonal / ties
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function

from .config import config

__all__ = [
    "sym",
    "spd_regularize",
    "shrinkage",
    "identity_like",
    "trace",
    "logm",
    "expm",
    "sqrtm",
    "invsqrtm",
    "sqrtm_pair",
    "powm",
    "reeig",
    "eigvals_logsq",
]


# --------------------------------------------------------------------------- #
# elementary helpers
# --------------------------------------------------------------------------- #
def sym(A: Tensor) -> Tensor:
    """Symmetrise the trailing two axes."""
    return 0.5 * (A + A.transpose(-1, -2))


def identity_like(A: Tensor) -> Tensor:
    return torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A)


def trace(A: Tensor) -> Tensor:
    """Batched trace, shape ``(...)``."""
    return torch.einsum("...ii->...", A)


def spd_regularize(Sigma: Tensor, ridge: Optional[float] = None) -> Tensor:
    """``Sigma + ridge * tr(Sigma)/n * I`` -- scale invariant ridge."""
    ridge = config.ridge if ridge is None else ridge
    if ridge <= 0:
        return Sigma
    n = Sigma.shape[-1]
    scale = trace(Sigma)[..., None, None] / n
    eye = torch.eye(n, dtype=Sigma.dtype, device=Sigma.device)
    return Sigma + ridge * scale * eye


def shrinkage(Sigma: Tensor, gamma: float = 0.1) -> Tensor:
    """Ledoit-Wolf style shrinkage towards the scaled identity."""
    if gamma <= 0:
        return Sigma
    n = Sigma.shape[-1]
    scale = trace(Sigma)[..., None, None] / n
    eye = torch.eye(n, dtype=Sigma.dtype, device=Sigma.device)
    return (1.0 - gamma) * Sigma + gamma * scale * eye


# --------------------------------------------------------------------------- #
# spectral function registry
# --------------------------------------------------------------------------- #
PhiFn = Callable[[Tensor, Optional[float]], Tensor]
_PHI: Dict[str, Tuple[PhiFn, PhiFn]] = {}


def _register(name: str, phi: PhiFn, dphi: PhiFn) -> None:
    _PHI[name] = (phi, dphi)


def _floor(lam: Tensor) -> float:
    return config.eps(lam.dtype)


_register(
    "log",
    lambda lam, a: lam.clamp_min(_floor(lam)).log(),
    lambda lam, a: torch.where(
        lam > _floor(lam), lam.clamp_min(_floor(lam)).reciprocal(), torch.zeros_like(lam)
    ),
)
_register("exp", lambda lam, a: lam.exp(), lambda lam, a: lam.exp())
_register(
    "sqrt",
    lambda lam, a: lam.clamp_min(_floor(lam)).sqrt(),
    lambda lam, a: torch.where(
        lam > _floor(lam), 0.5 * lam.clamp_min(_floor(lam)).rsqrt(), torch.zeros_like(lam)
    ),
)
_register(
    "invsqrt",
    lambda lam, a: lam.clamp_min(_floor(lam)).rsqrt(),
    lambda lam, a: torch.where(
        lam > _floor(lam),
        -0.5 * lam.clamp_min(_floor(lam)).pow(-1.5),
        torch.zeros_like(lam),
    ),
)
_register(
    "pow",
    lambda lam, a: lam.clamp_min(_floor(lam)).pow(a),
    lambda lam, a: torch.where(
        lam > _floor(lam),
        a * lam.clamp_min(_floor(lam)).pow(a - 1.0),
        torch.zeros_like(lam),
    ),
)
_register(
    "rect",
    lambda lam, a: lam.clamp_min(a),
    lambda lam, a: (lam > a).to(lam.dtype),
)


# --------------------------------------------------------------------------- #
# core: one eigh, Loewner backward
# --------------------------------------------------------------------------- #
def _out_dtype(M: Tensor) -> torch.dtype:
    """Never emit a half-precision SPD matrix.

    Under ``autocast`` the input can arrive as bfloat16 or float16. A covariance
    whose spectrum spans several orders of magnitude does not survive that, and
    the damage shows up as a NaN inside the next logarithm rather than as a
    gradual loss of accuracy. Spectral operators therefore always return at
    least float32, which also stops the low-precision dtype from propagating
    down the rest of the trunk.
    """
    return M.dtype if M.dtype in (torch.float32, torch.float64) else torch.float32


def _eigh(M: Tensor) -> Tuple[Tensor, Tensor]:
    """``eigh`` with optional chunking over the flattened batch axis."""
    chunk = config.eig_chunk
    if chunk and M.dim() > 2:
        flat = M.reshape(-1, *M.shape[-2:])
        if flat.shape[0] > chunk:
            lam, U = zip(*(torch.linalg.eigh(c) for c in flat.split(chunk, dim=0)))
            lam = torch.cat(lam, 0).reshape(*M.shape[:-1])
            U = torch.cat(U, 0).reshape(M.shape)
            return lam, U
    return torch.linalg.eigh(M)


def _loewner(lam: Tensor, lam_phi: Tensor, slope: Tensor) -> Tensor:
    if not config.loewner_guard:
        return _loewner_unguarded(lam, lam_phi, slope)
    eps = config.eps(lam.dtype)
    gap = lam.unsqueeze(-1) - lam.unsqueeze(-2)
    tied = gap.abs() < eps
    gap = gap.masked_fill(tied, 1.0)

    off = lam_phi.unsqueeze(-1) - lam_phi.unsqueeze(-2)
    off = off.masked_fill(tied, 0.0)

    diag = 0.5 * (slope.unsqueeze(-1) + slope.unsqueeze(-2))
    diag = diag.masked_fill(~tied, 0.0)
    return (off + diag) / gap


def _loewner_unguarded(lam: Tensor, lam_phi: Tensor, slope: Tensor) -> Tensor:
    """The divided-difference matrix without a tolerance on eigenvalue gaps.

    Exact equality is still handled on the diagonal, because ``0/0`` there is
    unconditional and no implementation survives it. Everything off-diagonal is
    the literal quotient. This is the form most derivations write down, and it is
    correct for a spectrum in general position; it returns non-finite values the
    moment two eigenvalues coincide, which a spectral floor arranges deliberately
    whenever it clamps more than one of them to the same value.

    Provided for ablation only.
    """
    n = lam.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=lam.device).expand(
        lam.shape[:-1] + (n, n)
    )
    gap = (lam.unsqueeze(-1) - lam.unsqueeze(-2)).masked_fill(eye, 1.0)
    off = (lam_phi.unsqueeze(-1) - lam_phi.unsqueeze(-2)).masked_fill(eye, 0.0)
    return off / gap + torch.diag_embed(slope)


class _SpectralMap(Function):
    """Applies ``phi`` to the spectrum, with the Loewner differential."""

    @staticmethod
    def forward(ctx, M: Tensor, name: str, arg, force_sym: bool, force_pos: bool):
        out_dtype = _out_dtype(M)
        work = config.resolve_spectral_dtype(M)
        Mw = M.to(work)
        if force_sym:
            Mw = sym(Mw)
        lam, U = _eigh(Mw)
        if force_pos:
            lam = lam.clamp_min(config.eps(work))
        phi, _ = _PHI[name]
        lam_phi = phi(lam, arg)
        X = U @ torch.diag_embed(lam_phi) @ U.transpose(-1, -2)
        ctx.save_for_backward(lam, lam_phi, U)
        ctx.name, ctx.arg, ctx.out_dtype = name, arg, out_dtype
        return X.to(out_dtype)

    @staticmethod
    def backward(ctx, dX: Tensor):
        lam, lam_phi, U = ctx.saved_tensors
        _, dphi = _PHI[ctx.name]
        L = _loewner(lam, lam_phi, dphi(lam, ctx.arg))
        Ut = U.transpose(-1, -2)
        dM = U @ (L * (Ut @ sym(dX.to(lam.dtype)) @ U)) @ Ut
        return dM.to(ctx.out_dtype), None, None, None, None


class _SqrtPair(Function):
    """``(Sigma^{1/2}, Sigma^{-1/2})`` from a single eigendecomposition.

    The whitening pair appears in every affine-invariant formula; sharing one
    decomposition halves the cost of Exp, Log and the geodesic distance.
    """

    @staticmethod
    def forward(ctx, M: Tensor, force_sym: bool):
        out_dtype = _out_dtype(M)
        work = config.resolve_spectral_dtype(M)
        Mw = M.to(work)
        if force_sym:
            Mw = sym(Mw)
        lam, U = _eigh(Mw)
        lam = lam.clamp_min(config.eps(work))
        s, i = lam.sqrt(), lam.rsqrt()
        Ut = U.transpose(-1, -2)
        A = U @ torch.diag_embed(s) @ Ut
        B = U @ torch.diag_embed(i) @ Ut
        ctx.save_for_backward(lam, s, i, U)
        ctx.out_dtype = out_dtype
        return A.to(out_dtype), B.to(out_dtype)

    @staticmethod
    def backward(ctx, dA: Tensor, dB: Tensor):
        lam, s, i, U = ctx.saved_tensors
        Ut = U.transpose(-1, -2)
        eps = config.eps(lam.dtype)
        ok = lam > eps
        slope_s = torch.where(ok, 0.5 * lam.clamp_min(eps).rsqrt(), torch.zeros_like(lam))
        slope_i = torch.where(ok, -0.5 * lam.clamp_min(eps).pow(-1.5), torch.zeros_like(lam))
        L1 = _loewner(lam, s, slope_s)
        L2 = _loewner(lam, i, slope_i)
        dM = U @ (L1 * (Ut @ sym(dA.to(lam.dtype)) @ U)) @ Ut
        dM = dM + U @ (L2 * (Ut @ sym(dB.to(lam.dtype)) @ U)) @ Ut
        return dM.to(ctx.out_dtype), None


# --------------------------------------------------------------------------- #
# public operators
# --------------------------------------------------------------------------- #
def logm(M: Tensor, force_sym: bool = False) -> Tensor:
    """Matrix logarithm of SPD matrices."""
    return _SpectralMap.apply(M, "log", None, force_sym, False)


def expm(M: Tensor, force_sym: bool = False) -> Tensor:
    """Matrix exponential of symmetric matrices (output is SPD)."""
    return _SpectralMap.apply(M, "exp", None, force_sym, False)


def sqrtm(M: Tensor, force_sym: bool = False) -> Tensor:
    return _SpectralMap.apply(M, "sqrt", None, force_sym, False)


def invsqrtm(M: Tensor, force_sym: bool = False) -> Tensor:
    return _SpectralMap.apply(M, "invsqrt", None, force_sym, False)


def powm(M: Tensor, t: float, force_sym: bool = False) -> Tensor:
    """Fractional power ``M^t`` (``t`` is treated as a constant)."""
    return _SpectralMap.apply(M, "pow", float(t), force_sym, False)


def reeig(M: Tensor, floor: float = 1e-4, force_sym: bool = False) -> Tensor:
    """Clamp the spectrum from below; the output stays strictly inside S^n_++."""
    return _SpectralMap.apply(M, "rect", float(floor), force_sym, False)


def sqrtm_pair(M: Tensor, force_sym: bool = False) -> Tuple[Tensor, Tensor]:
    return _SqrtPair.apply(M, force_sym)


def eigvals_logsq(M: Tensor) -> Tensor:
    """``sum_i log^2 lambda_i(M)`` for SPD ``M`` -- one ``eigvalsh``, no vectors.

    This is the whole cost of a geodesic distance once the input has been
    whitened, and ``eigvalsh`` is materially cheaper than ``eigh`` because the
    eigenvectors are never formed.
    """
    work = config.resolve_spectral_dtype(M)
    lam = torch.linalg.eigvalsh(sym(M).to(work)).clamp_min(config.eps(work))
    return lam.log().pow(2).sum(-1).to(M.dtype)
