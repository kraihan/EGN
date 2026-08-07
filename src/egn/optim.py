"""Riemannian optimisers.

Both optimisers accept a mixed parameter list: anything that is a
:class:`~egn.manifolds.ManifoldParameter` is updated by
``retract(x, -lr * direction)`` after its gradient has been converted with
``egrad2rgrad``; everything else follows the ordinary Euclidean rule. That means
one optimiser object for the whole network, and no bespoke update code that can
drift out of sync with the geometry.

Distributed data parallel works unchanged: DDP all-reduces the *Euclidean*
gradients in its backward hook, every rank then applies the identical
deterministic Riemannian update to identical parameters, so the replicas stay
bit-identical without any extra synchronisation.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer

from .manifolds import Euclidean, ManifoldParameter

__all__ = ["RiemannianAdam", "RiemannianSGD", "build_optimizer"]

_EUCLIDEAN = Euclidean()


def _manifold_of(p):
    return getattr(p, "manifold", _EUCLIDEAN)


class RiemannianAdam(Optimizer):
    """Adam with manifold-aware moments.

    ``amsgrad`` is on by default: on the SPD manifold the curvature of the
    distance-based loss makes the raw second-moment estimate jump when a
    prototype passes close to a sample, and the max-form removes the resulting
    step-size spikes.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = True,
        stabilize_every: int = 100,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            stabilize_every=stabilize_every,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            wd, amsgrad = group["weight_decay"], group["amsgrad"]
            stab = group["stabilize_every"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RiemannianAdam does not support sparse gradients")
                manifold = _manifold_of(p)
                euclidean = isinstance(manifold, Euclidean)

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]

                if wd:
                    grad = grad.add(p, alpha=wd) if euclidean else grad

                rgrad = grad if euclidean else manifold.egrad2rgrad(p, grad)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(rgrad, alpha=1 - b1)
                v.mul_(b2).addcmul_(rgrad, rgrad, value=1 - b2)
                vhat = v
                if amsgrad:
                    torch.maximum(state["max_exp_avg_sq"], v, out=state["max_exp_avg_sq"])
                    vhat = state["max_exp_avg_sq"]

                bias1 = 1 - b1**t
                bias2 = 1 - b2**t
                direction = (m / bias1) / ((vhat / bias2).sqrt().add_(eps))

                if euclidean:
                    p.add_(direction, alpha=-lr)
                else:
                    direction = manifold.proju(p, direction)
                    new_p = manifold.retract(p, -lr * direction)
                    m.copy_(manifold.transport(p, new_p, m))
                    p.copy_(new_p)
                    if stab and t % stab == 0:
                        p.copy_(manifold.project(p))
        return loss


class RiemannianSGD(Optimizer):
    """SGD with momentum, manifold aware."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        stabilize_every: int = 100,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stabilize_every=stabilize_every,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            wd, nesterov, stab = group["weight_decay"], group["nesterov"], group["stabilize_every"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                manifold = _manifold_of(p)
                euclidean = isinstance(manifold, Euclidean)
                grad = p.grad
                if wd and euclidean:
                    grad = grad.add(p, alpha=wd)
                rgrad = grad if euclidean else manifold.egrad2rgrad(p, grad)

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                state["step"] += 1
                buf = state["momentum_buffer"]
                if mom:
                    buf.mul_(mom).add_(rgrad)
                    rgrad = rgrad.add(buf, alpha=mom) if nesterov else buf

                if euclidean:
                    p.add_(rgrad, alpha=-lr)
                else:
                    d = manifold.proju(p, rgrad)
                    new_p = manifold.retract(p, -lr * d)
                    buf.copy_(manifold.transport(p, new_p, buf))
                    p.copy_(new_p)
                    if stab and state["step"] % stab == 0:
                        p.copy_(manifold.project(p))
        return loss


def build_optimizer(
    model,
    lr: float = 1e-3,
    kind: str = "adam",
    weight_decay: float = 0.0,
    manifold_lr_scale: float = 1.0,
    **kwargs,
):
    """Split parameters into manifold / Euclidean groups and build the optimiser.

    Weight decay is applied only to Euclidean parameters: shrinking a Stiefel
    frame or an SPD prototype towards zero is meaningless -- zero is not on
    either manifold.
    """
    manifold_params, euclidean_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (manifold_params if isinstance(p, ManifoldParameter) else euclidean_params).append(p)

    groups = [
        {"params": manifold_params, "weight_decay": 0.0, "lr": lr * manifold_lr_scale},
        {"params": euclidean_params, "weight_decay": weight_decay, "lr": lr},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    kind = kind.lower()
    if kind in ("adam", "riemannianadam", "amsgrad"):
        return RiemannianAdam(groups, lr=lr, amsgrad=kind != "adam" or True, **kwargs)
    if kind in ("sgd", "riemanniansgd"):
        return RiemannianSGD(groups, lr=lr, **kwargs)
    raise ValueError(f"unknown optimizer '{kind}'")
