"""Four input conventions, one model class.

The point of the library is that none of these need a different model, a
different loader or a different config -- only a different array.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from egn import EGNClassifier

rng = np.random.default_rng(0)


def report(name, X, y, **kw):
    clf = EGNClassifier(epochs=8, batch_size=32, lr=5e-3, verbose=0, **kw).fit(X, y)
    print(f"{name:<34} train accuracy {clf.score(X, y):.3f}")


# 1. multichannel signal: (N, C, T) -- EEG epochs, radar returns, IMU windows
X = rng.normal(size=(200, 12, 256)).astype(np.float32)
y = rng.integers(0, 3, 200)
report("signal (N, C, T)", X, y)

# 2. precomputed covariances: (N, n, n) -- descriptors from any other pipeline
A = rng.normal(size=(200, 10, 10))
X = (A @ A.transpose(0, 2, 1) + np.eye(10)).astype(np.float32)
report("covariance (N, n, n)", X, y)

# 3. feature maps: (N, C, H, W) -- region covariance of a CNN's activations
X = rng.normal(size=(200, 16, 8, 8)).astype(np.float32)
report("feature map (N, C, H, W)", X, y)

# 4. sequences of features: (N, T, D) -- word/frame embeddings, sensor tables
X = rng.normal(size=(200, 120, 14)).astype(np.float32)
report("sequence (N, T, D)", X, y, input_kind="sequence")

# 5. a torch Dataset that already carries labels
ds = TensorDataset(torch.randn(200, 12, 256), torch.randint(0, 3, (200,)))
clf = EGNClassifier(epochs=5, verbose=0).fit(ds)
print(f"{'torch Dataset':<34} fitted, classes {clf.classes_}")

# 6. string labels come back as string labels
X = rng.normal(size=(120, 8, 200)).astype(np.float32)
labels = rng.choice(["rest", "left", "right"], 120)
clf = EGNClassifier(epochs=5, verbose=0).fit(X, labels)
print(f"{'string labels':<34} predicts {set(clf.predict(X))}")
