"""Input adapters -- the layer that makes EGN data independent.

Everything downstream consumes one canonical layout::

    (B, K, n, n)      B = batch, K = manifold channels, n = matrix size

``ToSPD`` maps whatever the user has into that layout, and infers which
convention to use from the tensor rank when ``kind="auto"``:

===================  ==================================  =========================
input                interpretation                      output
===================  ==================================  =========================
``(B, n, n)``        already SPD (covariance descriptor)  ``(B, 1, n, n)``
``(B, K, n, n)``     multi-branch SPD                     ``(B, K, n, n)``
``(B, C, T)``        multichannel signal, ``T`` samples   ``(B, K, C, C)``
``(B, T, D)``        sequence of ``D``-dim features       ``(B, K, D, D)``
``(B, C, H, W)``     image / feature map                  ``(B, K, C, C)``
``(B, D)``           a single feature vector              rejected -- see note
===================  ==================================  =========================

Two conventions collide at rank 3: ``(B, C, T)`` and ``(B, T, D)`` are the same
shape. The rule is that the *shorter* trailing axis is the feature axis only if
``time_axis`` says so; the default assumes ``(B, C, T)`` with ``T > C``, which is
the usual case for signals, and falls back to transposing when it is not.
Set ``kind`` explicitly when your data is ambiguous -- guessing is a convenience,
not a contract.

A rank-2 input has no second moment to estimate, so it is rejected with an
explanatory error rather than silently turned into a rank-1 outer product, which
is singular and would only fail later inside a logarithm.

``branches`` is the manifold analogue of a CNN's channel count at the stem: the
signal is cut into ``branches`` contiguous windows and one covariance is formed
per window, so the network sees ``K`` matrices per sample instead of one.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..functional import shrinkage as _shrink
from ..functional import spd_regularize, sym

__all__ = ["ToSPD", "infer_input_kind", "infer_matrix_size"]

_KINDS = ("auto", "spd", "signal", "sequence", "image")


def infer_input_kind(shape) -> str:
    """Best-effort convention for a tensor shape (batch axis included)."""
    if len(shape) == 4:
        return "spd" if shape[-1] == shape[-2] else "image"
    if len(shape) == 3:
        return "spd" if shape[-1] == shape[-2] else "signal"
    raise ValueError(
        f"cannot infer an SPD convention from shape {tuple(shape)}. "
        "EGN needs a second moment: give it a signal (B, C, T), a sequence "
        "(B, T, D), a feature map (B, C, H, W) or a covariance (B, n, n)."
    )


def infer_matrix_size(shape, kind: str = "auto", time_axis: str = "last") -> int:
    """Matrix size ``n`` that :class:`ToSPD` will produce for this input shape."""
    kind = infer_input_kind(shape) if kind == "auto" else kind
    if kind == "spd":
        return int(shape[-1])
    if kind == "image":
        return int(shape[1])
    if kind == "sequence":
        return int(shape[-1])
    if kind == "signal":
        c, t = int(shape[-2]), int(shape[-1])
        return c if time_axis == "last" or t >= c else t
    raise ValueError(f"unknown kind '{kind}'")


class ToSPD(nn.Module):
    """Form SPD matrices from raw input.

    Parameters
    ----------
    kind : one of ``auto``, ``spd``, ``signal``, ``sequence``, ``image``
    branches : split the sample axis into this many windows, one matrix each
    ridge : scale-invariant ridge added to every output, repairing the rank
        deficiency that appears whenever the number of samples is comparable to
        the matrix size (routine for EEG epochs and for image region covariances)
    shrinkage : optional Ledoit-Wolf shrinkage towards the scaled identity,
        the better conditioner when ``n`` is large relative to the sample count
    center : subtract the sample mean before forming the second moment
    check_input : raise on NaN or Inf before anything else runs. Off by default
        because the test forces a device synchronisation on every forward pass;
        :class:`egn.EGNClassifier` runs it once on the first batch instead, which
        is where a bad array actually comes from.
    enforce_spd : clamp the spectrum from below after forming the matrix. Costs
        one extra eigendecomposition per batch and is off by default, because
        the ridge already covers the usual rank deficiency. Turn it on for
        third-party descriptors, which are routinely indefinite in the last bits
        -- otherwise the first logarithm downstream returns NaN and the failure
        surfaces far from its cause.
    """

    def __init__(
        self,
        kind: str = "auto",
        branches: int = 1,
        ridge: float = 1e-4,
        shrinkage: float = 0.0,
        center: bool = True,
        time_axis: str = "last",
        enforce_spd: bool = False,
        check_input: bool = False,
    ):
        super().__init__()
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got '{kind}'")
        self.kind = kind
        self.branches = int(branches)
        self.ridge = float(ridge)
        self.shrinkage = float(shrinkage)
        self.center = bool(center)
        self.time_axis = time_axis
        self.enforce_spd = bool(enforce_spd)
        self.check_input = bool(check_input)

    # ---------------------------------------------------------------- utils
    def _covariance(self, X: Tensor) -> Tensor:
        """``(..., d, t) -> (..., d, d)``, optionally split into branches."""
        t = X.shape[-1]
        k = max(self.branches, 1)
        if k > 1:
            usable = (t // k) * k
            X = X[..., :usable].unflatten(-1, (k, usable // k))  # (..., d, k, t')
            X = X.transpose(-3, -2)  # (..., k, d, t')
        else:
            X = X.unsqueeze(-3)
        if self.center:
            X = X - X.mean(dim=-1, keepdim=True)
        denom = max(X.shape[-1] - 1, 1)
        return (X @ X.transpose(-1, -2)) / denom

    # -------------------------------------------------------------- forward
    def forward(self, X: Tensor) -> Tensor:
        if self.check_input and not torch.isfinite(X).all():
            raise ValueError(
                "input contains NaN or Inf. EGN cannot form a covariance from it, and "
                "the failure would otherwise surface as an eigendecomposition error "
                "several layers later. Clean the input, or set check_input=False to "
                "skip this test."
            )
        kind = infer_input_kind(X.shape) if self.kind == "auto" else self.kind

        if kind == "spd":
            S = sym(X)
            if S.dim() == 3:
                S = S.unsqueeze(1)
        elif kind == "signal":
            if X.dim() != 3:
                raise ValueError(f"'signal' expects (B, C, T), got {tuple(X.shape)}")
            if self.time_axis == "first" or (self.time_axis == "auto" and X.shape[-1] < X.shape[-2]):
                X = X.transpose(-1, -2)
            S = self._covariance(X)
        elif kind == "sequence":
            if X.dim() != 3:
                raise ValueError(f"'sequence' expects (B, T, D), got {tuple(X.shape)}")
            S = self._covariance(X.transpose(-1, -2))
        elif kind == "image":
            if X.dim() != 4:
                raise ValueError(f"'image' expects (B, C, H, W), got {tuple(X.shape)}")
            S = self._covariance(X.flatten(2))
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(kind)

        if self.enforce_spd:
            from ..functional import reeig
            from ..config import config as _cfg

            S = reeig(S, _cfg.eps(S.dtype))
        if self.shrinkage > 0:
            S = _shrink(S, self.shrinkage)
        return spd_regularize(S, self.ridge)

    def extra_repr(self) -> str:
        return (
            f"kind={self.kind}, branches={self.branches}, ridge={self.ridge}, "
            f"shrinkage={self.shrinkage}, enforce_spd={self.enforce_spd}"
        )
