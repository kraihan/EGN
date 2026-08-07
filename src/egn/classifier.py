"""``EGNClassifier`` -- the one-object API.

    from egn import EGNClassifier
    clf = EGNClassifier(epochs=30).fit(X_train, y_train)
    print(clf.score(X_test, y_test))

``X`` may be a NumPy array, a torch tensor, a ``Dataset`` or a ``DataLoader``;
``y`` may be integers, strings, or anything hashable. The matrix size, the class
count and the label vocabulary are inferred at ``fit`` time, which is what makes
one object usable across EEG, skeleton, radar and image-descriptor data without
a per-dataset subclass.

Scaling out is a flag, not a rewrite. Launch with ``torchrun --nproc_per_node=N``
and the classifier detects the process group, wraps the model in
``DistributedDataParallel``, installs a ``DistributedSampler`` and reduces
metrics across ranks. Nothing else changes. Single-node multi-GPU without
``torchrun`` is available as ``data_parallel=True``, with the caveat in
``egn.utils.dist``.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import config, set_performance_defaults
from .data import LabelEncoder, as_dataset, make_loader
from .engine import evaluate, train_one_epoch
from .losses import build_criterion
from .models import EGN
from .optim import build_optimizer
from .utils.dist import (
    barrier,
    init_distributed,
    is_distributed,
    is_main_process,
    local_rank,
    unwrap,
    world_size,
)
from .utils.seed import seed_everything

__all__ = ["EGNClassifier"]


def _device_is_usable(device: torch.device) -> bool:
    """Actually execute a kernel on the device before committing to it.

    ``torch.cuda.is_available()`` only reports that a driver and a device exist.
    It returns True on a card whose compute capability the installed wheel was
    not built for -- Kaggle's P100 (sm_60) against a PyTorch built for sm_70 and
    up is the common case -- and the run then dies with
    ``cudaErrorNoKernelImageForDevice`` on the first allocation, several frames
    away from the cause. One tiny kernel is a cheap way to find out.
    """
    try:
        torch.zeros(1, device=device).add_(1).cpu()
        return True
    except Exception:
        return False


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        # an explicit request is honoured as given: if it fails, the user asked
        # for it and wants the real error, not a silent downgrade
        return torch.device(device)
    if torch.cuda.is_available():
        candidate = torch.device("cuda", local_rank())
        if _device_is_usable(candidate):
            return candidate
        warnings.warn(
            "a CUDA device is present but cannot run kernels from this PyTorch "
            "build (usually a compute-capability mismatch -- check the sm_XX list "
            "in the warning PyTorch printed at import). Falling back to CPU. "
            "Install a PyTorch build matching your GPU, or pass device='cuda' to "
            "see the underlying error.",
            RuntimeWarning,
            stacklevel=2,
        )
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EGNClassifier:
    """Universal SPD-manifold classifier with a scikit-learn surface.

    Parameters
    ----------
    Model
        ``dims``, ``depth``, ``channels``, ``branches``, ``head``, ``pool``,
        ``activation``, ``dropout``, ``batchnorm``, ``mix``, ``input_kind``,
        ``ridge``, ``shrinkage`` -- passed straight to :class:`egn.EGN`.
    Optimisation
        ``epochs``, ``batch_size``, ``lr``, ``optimizer``, ``weight_decay``,
        ``grad_clip``, ``label_smoothing``, ``scheduler`` (``none`` or ``cosine``).
    Runtime
        ``device`` (``auto``), ``amp`` and ``amp_dtype``, ``num_workers``,
        ``data_parallel``, ``spectral_dtype`` (``auto`` uses float32 on CUDA and
        float64 on CPU), ``seed``, ``verbose``.
    """

    def __init__(
        self,
        # model
        dims: Optional[List[int]] = None,
        depth: int = 2,
        channels: Optional[Any] = None,
        branches: int = 1,
        head: str = "tangent",
        pool: str = "logeuclid",
        pool_iters: int = 5,
        activation: str = "rect",
        dropout: float = 0.0,
        batchnorm: bool = True,
        bias: bool = True,
        constrained: bool = True,
        mix: bool = False,
        input_kind: str = "auto",
        ridge: float = 1e-4,
        shrinkage: float = 0.0,
        min_dim: int = 8,
        prototypes_per_class: int = 1,
        # optimisation
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        label_smoothing: float = 0.0,
        separation: float = 0.0,
        scheduler: str = "none",
        # runtime
        device: str = "auto",
        amp: bool = False,
        amp_dtype: str = "bfloat16",
        num_workers: int = 0,
        data_parallel: bool = False,
        spectral_dtype: str = "auto",
        seed: Optional[int] = 1234,
        verbose: int = 1,
    ):
        self.params: Dict[str, Any] = dict(
            dims=dims,
            depth=depth,
            channels=channels,
            branches=branches,
            head=head,
            pool=pool,
            pool_iters=pool_iters,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
            bias=bias,
            constrained=constrained,
            mix=mix,
            input_kind=input_kind,
            ridge=ridge,
            shrinkage=shrinkage,
            min_dim=min_dim,
            prototypes_per_class=prototypes_per_class,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            label_smoothing=label_smoothing,
            separation=separation,
            scheduler=scheduler,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
            num_workers=num_workers,
            data_parallel=data_parallel,
            spectral_dtype=spectral_dtype,
            seed=seed,
            verbose=verbose,
        )
        for k, v in self.params.items():
            setattr(self, k, v)

        self.model: Optional[EGN] = None
        self.label_encoder = LabelEncoder()
        self.history: List[Dict[str, float]] = []
        self.device_: Optional[torch.device] = None

    # -------------------------------------------------------- sklearn surface
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self.params)

    def set_params(self, **kw) -> "EGNClassifier":
        for k, v in kw.items():
            if k not in self.params:
                raise ValueError(f"unknown parameter '{k}'")
            self.params[k] = v
            setattr(self, k, v)
        return self

    @property
    def classes_(self) -> Optional[np.ndarray]:
        return self.label_encoder.classes_

    # ------------------------------------------------------------------ build
    def _make_model(self, num_classes: int) -> EGN:
        return EGN(
            num_classes=num_classes,
            input_kind=self.input_kind,
            branches=self.branches,
            dims=self.dims,
            channels=self.channels,
            depth=self.depth,
            min_dim=self.min_dim,
            head=self.head,
            pool=self.pool,
            pool_iters=self.pool_iters,
            batchnorm=self.batchnorm,
            bias=self.bias,
            constrained=self.constrained,
            activation=self.activation,
            dropout=self.dropout,
            mix=self.mix,
            ridge=self.ridge,
            shrinkage=self.shrinkage,
            prototypes_per_class=self.prototypes_per_class,
        )

    def _prepare(self, X, y) -> Tuple[Dataset, int]:
        if isinstance(X, DataLoader):
            dataset = X.dataset
            ys = _labels_of(dataset)
        elif isinstance(X, Dataset) and y is None:
            dataset = X
            ys = _labels_of(dataset)
        else:
            if y is None:
                raise ValueError("y is required unless X is a Dataset carrying labels")
            ys = np.asarray(y).reshape(-1)
            dataset = as_dataset(X, self.label_encoder.fit_transform(ys))
            return dataset, len(self.label_encoder)
        if ys is None:
            raise ValueError("could not read labels off the dataset; pass y explicitly")
        self.label_encoder.fit(ys)
        encoded = self.label_encoder.transform(ys)
        if not torch.equal(encoded, torch.as_tensor(ys).long().reshape(-1)):
            # the dataset's own labels are not 0..C-1; remap on the fly rather
            # than silently training against the wrong indices
            dataset = _RemapLabels(dataset, encoded)
        return dataset, len(self.label_encoder)

    # -------------------------------------------------------------------- fit
    def fit(self, X, y=None, eval_set: Optional[Tuple] = None) -> "EGNClassifier":
        if self.seed is not None:
            seed_everything(self.seed)
        config.spectral_dtype = self.spectral_dtype
        set_performance_defaults()
        init_distributed()

        self.device_ = _resolve_device(self.device)
        dataset, num_classes = self._prepare(X, y)

        sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
        loader = (
            X
            if isinstance(X, DataLoader) and sampler is None
            else make_loader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                sampler=sampler,
                num_workers=self.num_workers,
                drop_last=self.batchnorm and len(dataset) > self.batch_size,
            )
        )

        # --- materialise before any wrapper, so DDP has parameters to broadcast
        model = self._make_model(num_classes)
        example = _first_input(loader)
        if not torch.isfinite(example).all():
            raise ValueError(
                "the first training batch contains NaN or Inf. EGN cannot form a "
                "covariance from it; clean the data before fitting."
            )
        model.build_from_example(example, num_classes)
        model.to(self.device_)
        self.model = model

        net: nn.Module = model
        if is_distributed():
            net = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device_.index] if self.device_.type == "cuda" else None,
                broadcast_buffers=True,
            )
        elif self.data_parallel and torch.cuda.device_count() > 1:
            net = nn.DataParallel(model)

        criterion = build_criterion(
            label_smoothing=self.label_smoothing, separation=self.separation
        ).to(self.device_)
        optimizer = build_optimizer(
            unwrap(net), lr=self.lr, kind=self.optimizer, weight_decay=self.weight_decay
        )
        amp_dtype = torch.float16 if self.amp_dtype == "float16" else torch.bfloat16
        scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp and amp_dtype is torch.float16 and self.device_.type == "cuda"
        )
        sched = None
        if self.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.epochs * max(len(loader), 1), 1)
            )

        val_loader = None
        if eval_set is not None:
            Xv, yv = eval_set
            val_loader = make_loader(
                as_dataset(Xv, self.label_encoder.transform(yv)),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

        self.history = []
        for epoch in range(self.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            stats = train_one_epoch(
                net,
                loader,
                optimizer,
                criterion,
                self.device_,
                amp=self.amp,
                amp_dtype=amp_dtype,
                grad_clip=self.grad_clip,
                scaler=scaler,
                scheduler=sched,
            )
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in stats.items()}}
            if val_loader is not None:
                vstats = evaluate(
                    net, val_loader, criterion, self.device_, amp=self.amp, amp_dtype=amp_dtype
                )
                record.update({f"val_{k}": v for k, v in vstats.items()})
            self.history.append(record)
            if self.verbose and is_main_process():
                msg = f"epoch {epoch + 1:3d}/{self.epochs}  loss {record['train_loss']:.4f}  acc {record['train_accuracy']:.4f}"
                if val_loader is not None:
                    msg += f"  | val loss {record['val_loss']:.4f}  val acc {record['val_accuracy']:.4f}"
                print(msg, flush=True)
        barrier()
        return self

    # ---------------------------------------------------------------- predict
    def _inference_loader(self, X) -> DataLoader:
        if isinstance(X, DataLoader):
            return X
        return make_loader(
            as_dataset(X), batch_size=self.batch_size, num_workers=self.num_workers
        )

    @torch.no_grad()
    def decision_function(self, X) -> np.ndarray:
        self._check_fitted()
        self.model.eval()
        outs = []
        for batch in self._inference_loader(X):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            outs.append(self.model(x.to(self.device_, non_blocking=True)).float().cpu())
        return torch.cat(outs).numpy()

    def predict_proba(self, X) -> np.ndarray:
        return torch.softmax(torch.from_numpy(self.decision_function(X)), dim=-1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.label_encoder.inverse_transform(self.decision_function(X).argmax(-1))

    def score(self, X, y) -> float:
        pred = self.predict(X)
        return float((pred == np.asarray(y).reshape(-1)).mean())

    @torch.no_grad()
    def transform(self, X) -> np.ndarray:
        """Pooled SPD features -- use EGN as a descriptor for another model."""
        self._check_fitted()
        self.model.eval()
        outs = []
        for batch in self._inference_loader(X):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            outs.append(self.model.features(x.to(self.device_)).float().cpu())
        return torch.cat(outs).numpy()

    # --------------------------------------------------------------- persist
    def save(self, path: str) -> None:
        self._check_fitted()
        torch.save(
            {
                "params": self.params,
                "hparams": self.model.hparams,
                "state_dict": self.model.state_dict(),
                "classes": self.label_encoder.classes_,
                "history": self.history,
                "format": 1,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "auto", weights_only: bool = False) -> "EGNClassifier":
        blob = torch.load(path, map_location="cpu", weights_only=weights_only)
        clf = cls(**blob["params"])
        clf.device_ = _resolve_device(device)
        h = blob["hparams"]
        model = EGN(
            **{k: v for k, v in h.items() if k not in ("in_dim", "num_classes", "in_channels")}
        )
        model.build(h["in_dim"], h["num_classes"], in_channels=h.get("in_channels"))
        model.load_state_dict(blob["state_dict"])
        clf.model = model.to(clf.device_)
        clf.label_encoder.classes_ = blob["classes"]
        clf.history = blob.get("history", [])
        return clf

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("call fit() before predict/score/transform")
        if self.device_ is None:
            self.device_ = _resolve_device(self.device)

    def __repr__(self) -> str:
        state = "fitted" if self.model is not None else "unfitted"
        return f"EGNClassifier({state}, head={self.head}, depth={self.depth}, device={self.device})"


# ------------------------------------------------------------------ helpers
class _RemapLabels(Dataset):
    """Wrap a labelled dataset, replacing its labels with contiguous indices."""

    def __init__(self, base: Dataset, labels: torch.Tensor):
        self.base, self.labels = base, labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        x = item[0] if isinstance(item, (list, tuple)) else item
        return x, self.labels[i]


def _first_input(loader: DataLoader) -> torch.Tensor:
    for batch in loader:
        return batch[0] if isinstance(batch, (list, tuple)) else batch
    raise ValueError("the training loader is empty")


def _labels_of(dataset) -> Optional[np.ndarray]:
    for attr in ("y", "labels", "targets"):
        v = getattr(dataset, attr, None)
        if v is not None:
            return np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v).reshape(-1)
    if isinstance(dataset, torch.utils.data.TensorDataset) and len(dataset.tensors) > 1:
        return dataset.tensors[-1].cpu().numpy().reshape(-1)
    return None
