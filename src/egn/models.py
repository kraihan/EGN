"""The EGN model.

    x --ToSPD--> (B, K, d, d) --EGNBlock xL--> (B, K', m, m) --pool--> (B, m, m) --head--> logits

Data independence is the point: ``EGN(num_classes=10)`` is a complete model. The
matrix size, the number of manifold channels and the layer widths are inferred
from the first batch it sees, exactly like ``nn.LazyLinear``, so the same object
trains on 22-channel EEG, on 93-dimensional skeleton covariances and on a
401x401 image descriptor without any change.

Materialisation happens on the first forward pass. Under DDP you must build
before wrapping -- call :meth:`EGN.build_from_example` (or one forward on a
single rank's batch) first, otherwise the replicas have nothing to broadcast.
:class:`egn.EGNClassifier` does this for you.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .nn.blocks import EGNBlock
from .nn.heads import GeodesicPrototypeHead, TangentHead
from .nn.inputs import ToSPD, infer_matrix_size
from .nn.pooling import RiemannianPool

__all__ = ["EGN", "egn_tiny", "egn_small", "egn_base", "default_dims"]


def default_dims(in_dim: int, depth: int = 2, min_dim: int = 8, ratio: float = 0.5) -> list:
    """Geometric width schedule, the SPD analogue of halving spatial resolution.

    Never drops below ``min_dim`` and never proposes a step that would discard
    more than ``1 - ratio`` of the spectrum at once.
    """
    dims = [int(in_dim)]
    for _ in range(max(depth, 0)):
        nxt = max(int(round(dims[-1] * ratio)), min_dim)
        if nxt >= dims[-1]:
            break
        dims.append(nxt)
    return dims


class EGN(nn.Module):
    """Universal SPD-manifold classifier.

    Parameters
    ----------
    num_classes : inferred from the labels by :class:`egn.EGNClassifier` when None
    in_dim : matrix size after ``ToSPD``; inferred from the first batch when None
    input_kind : ``auto`` / ``spd`` / ``signal`` / ``sequence`` / ``image``
    branches : stem channels, i.e. how many windows the input is split into
    dims : explicit width schedule, e.g. ``(64, 32, 16)``. The first entry must
        equal ``in_dim``; when None a geometric schedule of ``depth`` blocks is used
    channels : manifold channels per block, e.g. ``(4, 4)``; scalar broadcasts
    depth : number of blocks when ``dims`` is None
    head : ``tangent`` (fast, linear in the tangent space) or ``geodesic``
        (prototype distances -- fully geometric, cost grows with class count)
    pool : ``logeuclid`` (closed form) or ``frechet`` (true barycentre)
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        in_dim: Optional[int] = None,
        input_kind: str = "auto",
        branches: int = 1,
        dims: Optional[Sequence[int]] = None,
        channels: Optional[Sequence[int] | int] = None,
        depth: int = 2,
        min_dim: int = 8,
        head: str = "tangent",
        pool: str = "logeuclid",
        constrained: bool = True,
        pool_iters: int = 5,
        batchnorm: bool = True,
        bias: bool = True,
        activation: str = "rect",
        dropout: float = 0.0,
        mix: bool = False,
        ridge: float = 1e-4,
        shrinkage: float = 0.0,
        prototypes_per_class: int = 1,
        temperature: float = 1.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.hparams = dict(
            num_classes=num_classes,
            in_dim=in_dim,
            in_channels=None,
            input_kind=input_kind,
            branches=branches,
            dims=list(dims) if dims is not None else None,
            channels=channels,
            depth=depth,
            min_dim=min_dim,
            head=head,
            pool=pool,
            constrained=constrained,
            pool_iters=pool_iters,
            batchnorm=batchnorm,
            bias=bias,
            activation=activation,
            dropout=dropout,
            mix=mix,
            ridge=ridge,
            shrinkage=shrinkage,
            prototypes_per_class=prototypes_per_class,
            temperature=temperature,
            head_dropout=head_dropout,
        )
        self.num_classes = num_classes
        self.to_spd = ToSPD(
            kind=input_kind, branches=branches, ridge=ridge, shrinkage=shrinkage
        )
        self.blocks: Optional[nn.Sequential] = None
        self.pool: Optional[RiemannianPool] = None
        self.head: Optional[nn.Module] = None
        if in_dim is not None and num_classes is not None:
            self.build(in_dim, num_classes)

    # ------------------------------------------------------------------ build
    @property
    def is_built(self) -> bool:
        return self.head is not None

    def build(
        self,
        in_dim: int,
        num_classes: Optional[int] = None,
        in_channels: Optional[int] = None,
    ) -> "EGN":
        """Materialise the trunk and head.

        ``in_channels`` is the number of manifold channels the stem produces. It
        is normally ``branches``, but an input that already carries a branch axis
        -- ``(B, K, n, n)`` -- sets it to ``K``, which is why it is a parameter
        rather than a constant.
        """
        h = self.hparams
        stem_channels = int(in_channels or h["branches"])
        num_classes = num_classes or h["num_classes"] or self.num_classes
        if num_classes is None:
            raise ValueError("num_classes is required to build the model")
        self.num_classes = int(num_classes)

        dims = h["dims"] or default_dims(in_dim, depth=h["depth"], min_dim=h["min_dim"])
        if dims[0] != in_dim:
            dims = [int(in_dim)] + [d for d in dims if d < in_dim]
        n_blocks = max(len(dims) - 1, 0)

        ch = h["channels"]
        if ch is None:
            ch_list = [stem_channels] * n_blocks
        elif isinstance(ch, int):
            ch_list = [ch] * n_blocks
        else:
            ch_list = list(ch)
            if len(ch_list) != n_blocks:
                raise ValueError(
                    f"channels has {len(ch_list)} entries but there are {n_blocks} blocks"
                )

        blocks, c_in = [], stem_channels
        for i in range(n_blocks):
            c_out = int(ch_list[i])
            blocks.append(
                EGNBlock(
                    dims[i],
                    dims[i + 1],
                    in_channels=c_in,
                    out_channels=c_out,
                    mix=h["mix"] or (c_in not in (1, c_out)),
                    constrained=h["constrained"],
                    batchnorm=h["batchnorm"],
                    bias=h["bias"],
                    activation=h["activation"],
                    dropout=h["dropout"],
                )
            )
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.feature_dim = dims[-1]
        self.feature_channels = c_in

        self.pool = RiemannianPool(mode=h["pool"], dim=1, iters=h["pool_iters"])

        if h["head"] == "geodesic":
            self.head = GeodesicPrototypeHead(
                self.feature_dim,
                self.num_classes,
                prototypes_per_class=h["prototypes_per_class"],
                temperature=h["temperature"],
            )
        elif h["head"] in ("tangent", "logeig"):
            self.head = TangentHead(
                self.feature_dim,
                self.num_classes,
                learn_reference=h["head"] == "tangent",
                dropout=h["head_dropout"],
            )
        else:
            raise ValueError(f"unknown head '{h['head']}'")
        # submodules are created in training mode by default; a model that was put
        # in eval() before its first forward would otherwise build a head whose
        # BatchNorm is still training, and fail on a single-sample batch
        self.train(self.training)
        self.hparams["in_dim"] = int(in_dim)
        self.hparams["in_channels"] = stem_channels
        self.hparams["num_classes"] = self.num_classes
        return self

    def build_from_example(self, x: Tensor, num_classes: Optional[int] = None) -> "EGN":
        """Infer the matrix size and channel count from one batch, then build.

        The stem is run on a single sample rather than reasoning about the shape,
        so any convention ``ToSPD`` supports is handled without a second code
        path that could disagree with it.
        """
        with torch.no_grad():
            S = self.to_spd(x[:1])
        return self.build(S.shape[-1], num_classes, in_channels=S.shape[1])

    # ---------------------------------------------------------------- forward
    def features(self, x: Tensor) -> Tensor:
        """Pooled SPD feature, shape ``(B, m, m)``. Useful as a descriptor."""
        S = self.to_spd(x)
        if not self.is_built:
            self.build(S.shape[-1], in_channels=S.shape[1])
        S = self.blocks(S)
        return self.pool(S)

    def forward(self, x: Tensor) -> Tensor:
        # the feature call may materialise the head, so resolve it afterwards
        features = self.features(x)
        return self.head(features)

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        return torch.softmax(self.forward(x), dim=-1)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x).argmax(dim=-1)

    # ------------------------------------------------------------ bookkeeping
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        if not self.is_built:
            return "not built (waiting for the first batch)"
        return f"feature_dim={self.feature_dim}, channels={self.feature_channels}"


# ----------------------------------------------------------------- factories
def egn_tiny(num_classes: Optional[int] = None, **kw) -> EGN:
    """One block, tangent head. The baseline to beat before adding depth."""
    kw.setdefault("depth", 1)
    kw.setdefault("branches", 1)
    kw.setdefault("head", "tangent")
    return EGN(num_classes=num_classes, **kw)


def egn_small(num_classes: Optional[int] = None, **kw) -> EGN:
    kw.setdefault("depth", 2)
    kw.setdefault("branches", 2)
    kw.setdefault("channels", 2)
    kw.setdefault("head", "tangent")
    return EGN(num_classes=num_classes, **kw)


def egn_base(num_classes: Optional[int] = None, **kw) -> EGN:
    """Deeper trunk, mixed channels, geodesic prototype head."""
    kw.setdefault("depth", 3)
    kw.setdefault("branches", 4)
    kw.setdefault("channels", 4)
    kw.setdefault("mix", True)
    kw.setdefault("head", "geodesic")
    kw.setdefault("pool", "frechet")
    return EGN(num_classes=num_classes, **kw)
