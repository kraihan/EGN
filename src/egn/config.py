"""Global numerical and performance policy for EGN.

Everything that trades speed against numerical margin is decided here, once, so
that no layer has to guess. The two settings that matter on GPU are

``spectral_dtype``
    the precision in which ``eigh`` is executed. The historical EGN code ran the
    whole network in ``float64``; consumer GPUs execute fp64 at 1/32 of their
    fp32 rate, which is why the GPU was slower than the CPU. The default policy
    here is ``"auto"``: ``float32`` on CUDA, ``float64`` on CPU. Set it to
    ``"float64"`` only when you are reproducing a theory check.

``loewner_guard``
    whether the backward pass detects *near*-coincident eigenvalues and replaces
    the divided difference with the derivative. With it off, only exact equality
    on the diagonal is handled -- the form most implementations write -- and a
    spectrum with tied eigenvalues (which a spectral floor produces on purpose)
    yields ``0/0`` in the off-diagonal block. Exposed so the guard can be
    ablated; leave it on for anything but that experiment.

``sync_free``
    when true, no operator calls ``.item()``, ``.cpu()`` or ``bool(tensor)`` in
    the forward pass. Iterative routines (the Karcher mean) then run a fixed
    number of iterations instead of testing a residual. Every such test costs a
    full device synchronisation, which on a deep SPD network dominates the
    runtime.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

__all__ = ["Config", "config", "set_performance_defaults", "numeric_context"]


@dataclass
class Config:
    spectral_dtype: str = "auto"  # "auto" | "float32" | "float64"
    sync_free: bool = True
    eig_chunk: int = 0  # 0 = no chunking; else max matrices per eigh call
    loewner_guard: bool = True
    eps32: float = 1e-6
    eps64: float = 1e-12
    ridge: float = 1e-4

    def eps(self, dtype: torch.dtype) -> float:
        return self.eps64 if dtype == torch.float64 else self.eps32

    def resolve_spectral_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        if self.spectral_dtype == "float32":
            return torch.float32
        if self.spectral_dtype == "float64":
            return torch.float64
        # auto
        if tensor.is_cuda:
            return torch.float32
        return torch.float64 if tensor.dtype == torch.float64 else torch.float32


config = Config()


def set_performance_defaults(tf32: bool = True, benchmark: bool = True) -> None:
    """Enable the CUDA fast paths EGN benefits from.

    Call once at process start. ``tf32`` only affects the dense matmuls that
    surround the eigendecompositions, never the eigendecompositions themselves,
    so the manifold constraints are unaffected.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = benchmark
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")


@contextlib.contextmanager
def numeric_context(**kwargs):
    """Temporarily override fields of the global :data:`config`."""
    old = {k: getattr(config, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(config, k, v)
    try:
        yield config
    finally:
        for k, v in old.items():
            setattr(config, k, v)
