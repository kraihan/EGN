"""Building an EGN trunk by hand and training it with the Riemannian optimiser.

Use this when the EGNClassifier constructor does not express what you want: a
non-uniform width schedule, a custom head, a shared trunk across two tasks, or
an EGN block inside a larger network.

Run: python examples/03_custom_architecture.py
"""

import numpy as np
import torch
import torch.nn as nn

import egn
from egn.geometry import is_spd
from egn.nn import (
    BiMap,
    EGNBlock,
    GeodesicPrototypeHead,
    GeometricBias,
    RiemannianPool,
    SpectralActivation,
    SPDBatchNorm,
    ToSPD,
)


def synthetic(n=600, classes=3, channels=16, samples=128, seed=0):
    """Signals whose covariance carries the label, not their marginals."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    mixing = rng.normal(size=(classes, channels, channels))
    X = np.stack([mixing[y[i]] @ rng.normal(size=(channels, samples)) for i in range(n)])
    return X.astype(np.float32), y


X, y = synthetic()
split = len(y) * 3 // 4
Xtr, ytr = torch.from_numpy(X[:split]), torch.from_numpy(y[:split]).long()
Xte, yte = torch.from_numpy(X[split:]), torch.from_numpy(y[split:]).long()

# --------------------------------------------------------------- architecture
# Composed with nn.Sequential like any PyTorch model. The shape schedule mirrors
# a convolutional trunk: matrix size falls, manifold channels rise.
model = nn.Sequential(
    ToSPD(kind="signal", branches=4, ridge=1e-3),   # (B, 4, 16, 16)
    EGNBlock(16, 12, in_channels=4, out_channels=8, mix=True, dropout=0.1),
    EGNBlock(12, 8, in_channels=8, out_channels=8),
    RiemannianPool("frechet", iters=5),             # (B, 8, 8)
    GeodesicPrototypeHead(8, num_classes=3, prototypes_per_class=2),
)

# The same trunk written out layer by layer, if you want to vary one piece:
_explicit = nn.Sequential(
    ToSPD(kind="signal", branches=4, ridge=1e-3),
    BiMap(16, 12, in_channels=4, out_channels=8, mix=True),
    SPDBatchNorm(12, 8),
    GeometricBias(12, 8),
    SpectralActivation("rect", floor=1e-4),
    RiemannianPool("logeuclid"),
    GeodesicPrototypeHead(12, num_classes=3),
)

print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------------------------ training
# build_optimizer splits manifold from Euclidean parameters automatically and
# applies weight decay only to the Euclidean group -- shrinking a Stiefel frame
# towards zero is meaningless, since zero is not on the manifold.
opt = egn.build_optimizer(model, lr=5e-3, kind="adam", weight_decay=1e-4)
criterion = egn.build_criterion(label_smoothing=0.05, separation=0.01)

for epoch in range(30):
    model.train()
    perm = torch.randperm(len(Xtr))
    for s in range(0, len(Xtr) - 31, 32):
        idx = perm[s : s + 32]
        opt.zero_grad(set_to_none=True)
        criterion(model(Xtr[idx]), ytr[idx], model).backward()
        opt.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(-1) == yte).float().mean().item()
        print(f"epoch {epoch + 1:3d}  test accuracy {acc:.3f}")

# ------------------------------------------------------- checking the invariants
model.eval()
with torch.no_grad():
    features = nn.Sequential(*list(model)[:-1])(Xte[:16])
    W = model[1].bimap.weight
    orthogonality = (W.transpose(-1, -2) @ W - torch.eye(W.shape[-1])).abs().max()
    prototypes = model[-1].prototypes

print(f"\npooled features are SPD:      {bool(is_spd(features.double()).all())}")
print(f"Stiefel orthogonality error: {orthogonality:.2e}")
print(f"prototypes stayed SPD:       {bool(is_spd(prototypes.detach().double()).all())}")
