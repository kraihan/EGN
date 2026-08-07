# Getting started

## Install

```bash
pip install egnlib
```

`import egn` — the distribution is `egnlib`, the package is `egn`. PyTorch ≥ 2.0 and NumPy are the
only dependencies.

## The shortest useful program

```python
import numpy as np
from egn import EGNClassifier

X = np.random.randn(512, 22, 256)      # trials, channels, samples
y = np.random.randint(0, 4, 512)

clf = EGNClassifier(epochs=30).fit(X, y)
print(clf.score(X, y))
```

Nothing about the dataset was declared. The matrix size, the number of manifold channels, the class
count and the label vocabulary were all inferred at `fit` time, the same way `nn.LazyLinear` infers
its input width.

## What EGN actually does with your data

The first thing that happens is that your input becomes a **covariance matrix** — a point on the SPD
manifold. Everything after that is a map from SPD matrices to SPD matrices, ending in a classifier
that measures distances in the manifold's own metric.

This is the right model when the *second-order structure* of your data carries the label: which
channels co-vary with which, rather than what any single channel does on its own. EEG, radar,
skeleton motion and diffusion imaging all have that character. Data where the mean carries the
signal is better served by an ordinary network.

## Input conventions

| Input | Meaning | `input_kind` |
|---|---|---|
| `(B, n, n)` | covariance already computed | `"spd"` |
| `(B, K, n, n)` | multi-branch SPD, e.g. one per frequency band | `"spd"` |
| `(B, C, T)` | multichannel signal | `"signal"` |
| `(B, T, D)` | sequence of `D`-dimensional features | `"sequence"` |
| `(B, C, H, W)` | feature map or image | `"image"` |

`input_kind="auto"` (the default) infers from the tensor rank. Two conventions collide at rank 3:
`(B, C, T)` and `(B, T, D)` are the same shape, and the default assumes the signal reading. **State
`input_kind` explicitly when your data is ambiguous** — inference is a convenience, not a contract.

A rank-2 input is rejected. There is no second moment in a single vector, and turning it into a
rank-one outer product would produce a singular matrix that fails later, inside a logarithm, far
from the cause.

## Manifold channels

`branches=K` splits the sample axis into `K` windows and forms one covariance per window — the
manifold equivalent of a multi-channel stem. `channels=K` lets the first `BiMap` expand a single
input matrix into `K` feature maps.

**This matters more than it looks.** With one channel, the pooling layer has nothing to pool and
becomes the identity. If you are comparing pooling strategies, make sure `branches` or `channels`
exceeds one, or every variant will produce the same number.

## Labels

Integers, strings, or anything hashable. `predict` returns labels in your own vocabulary:

```python
clf = EGNClassifier(epochs=20).fit(X, ["rest", "left", "right", ...])
clf.predict(X_test)          # array(['left', 'rest', ...], dtype='<U5')
```

## Persisting a model

```python
clf.save("model.pt")
clf = EGNClassifier.load("model.pt")
```

The architecture, weights, label vocabulary and training history travel together, so a reloaded
model needs no reconstruction code.

## Next

- Designing an architecture by hand: [layers.md](layers.md)
- What the metric guarantees: [geometry.md](geometry.md)
- Making it fast: [performance.md](performance.md)
