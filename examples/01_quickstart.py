"""Five ways to use EGN, in increasing order of control.

Run: python examples/quickstart.py
"""

import numpy as np
import torch

import egn
from egn import EGN, EGNClassifier
from egn.nn import EGNBlock, GeodesicPrototypeHead, RiemannianPool, ToSPD


def synthetic(n=400, classes=3, channels=16, samples=128, seed=0):
    """Class-conditional signals whose *covariance* carries the label.

    Each class gets a fixed mixing matrix, so the classes are separable on the
    manifold and not separable from the marginal statistics of a single channel.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    mixing = rng.normal(size=(classes, channels, channels))
    X = np.stack([mixing[y[i]] @ rng.normal(size=(channels, samples)) for i in range(n)])
    return X.astype(np.float32), y


X, y = synthetic()
split = len(y) * 3 // 4
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

# 1 --------------------------------------------------------------- the one-liner
clf = EGNClassifier(epochs=25, batch_size=64, lr=5e-3, verbose=0).fit(Xtr, ytr)
print(f"1. EGNClassifier            test accuracy {clf.score(Xte, yte):.3f}")

# 2 ------------------------------------------------- a different model, same API
clf = EGNClassifier(
    epochs=25, batch_size=64, lr=5e-3, verbose=0,
    branches=4, depth=2, channels=4, mix=True, head="geodesic", pool="frechet",
).fit(Xtr, ytr, eval_set=(Xte, yte))
print(f"2. geodesic head            test accuracy {clf.score(Xte, yte):.3f}")

# 3 ------------------------------------------------------- EGN as an nn.Module
model = EGN(num_classes=3, branches=2, depth=2, head="tangent")
logits = model(torch.from_numpy(Xtr[:8]))          # shapes inferred on first call
print(f"3. nn.Module                logits {tuple(logits.shape)}, "
      f"{model.num_parameters()} parameters")

# 4 ------------------------------------------------------ hand-built architecture
custom = torch.nn.Sequential(
    ToSPD(kind="signal", branches=4),
    EGNBlock(16, 12, in_channels=4, out_channels=8, mix=True),
    EGNBlock(12, 8, in_channels=8, out_channels=8),
    RiemannianPool("logeuclid"),
    GeodesicPrototypeHead(8, num_classes=3, prototypes_per_class=2),
)
print(f"4. custom Sequential        logits {tuple(custom(torch.from_numpy(Xtr[:8])).shape)}")

# 5 ------------------------------------------- geometry on its own, no model at all
A = egn.geometry.random_spd((4, 6, 6))
B = egn.geometry.random_spd((4, 6, 6))
print(f"5. geodesic distance        {egn.geometry.distance(A, B).mean():.4f}, "
      f"midpoint is SPD: {bool(egn.geometry.is_spd(egn.geometry.geodesic(A, B, 0.5)).all())}")

# EGN as a feature extractor for anything else
features = clf.transform(Xte)
print(f"   pooled descriptors        {features.shape}")
