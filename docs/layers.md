# Layers

Every layer maps `S^d_++ → S^m_++` exactly. There is no projection step, no re-symmetrisation of a
matrix that had already drifted, and no additive bias anywhere — an SPD matrix plus an arbitrary
symmetric matrix is not SPD.

## The correspondence with a CNN

| CNN | EGN | Preserves |
|---|---|---|
| `Conv2d(c_in, c_out, k)` | `BiMap(d_in, d_out, c_in, c_out)` | SPD, reduces size |
| `BatchNorm2d` | `SPDBatchNorm` | SPD |
| `ReLU` | `SpectralActivation` | SPD, commutes with congruence |
| `Dropout` | `GeodesicDropout` | SPD |
| bias | `GeometricBias` | SPD **and** every distance (isometry) |
| global average pool | `RiemannianPool` | SPD |

## BiMap

```python
BiMap(in_dim, out_dim, in_channels=1, out_channels=1, mix=False, constrained=True)
```

`Σ → Wᵀ Σ W` with `W` on the Stiefel manifold `St(d, m)`, `m ≤ d`. Full column rank means the output
is SPD whenever the input is, and `out_dim < in_dim` reduces size the way a strided convolution
reduces resolution.

`mix=True` makes each output channel the average of congruences of *all* input channels,
`Σ′_o = (1/C) Σ_i W_{oi}ᵀ Σ_i W_{oi}`. Still SPD, and it gives genuine cross-channel mixing. The
default `mix=False` transforms channels independently, which is the form for which the
reparameterisation identity in [geometry.md](geometry.md) is stated.

`constrained=False` replaces the manifold parameter with an ordinary one. **Ablation only** — the
congruence is then merely positive *semi*-definite, since nothing stops `W` from losing rank.

## GeometricBias

`Σ → D Σ Dᵀ` with `D` a Cholesky factor with positive diagonal, so it is invertible by construction
and never needs repair. This is an **isometry** of the affine-invariant metric: it moves features
around the manifold without distorting any distance the head will later measure. That is what makes
it the right analogue of a bias term, and why an additive bias would be wrong.

## SpectralActivation

| `kind` | Effect |
|---|---|
| `"rect"` | clamp eigenvalues from below — the manifold ReLU |
| `"power"` | `Σ^t` with learnable `t > 0`; `t < 1` compresses the spectrum, the cheapest conditioner for deep stacks |
| `"none"` | identity |

All act on eigenvalues only, so they commute with congruence by any orthogonal matrix and leave the
equivariance statement intact. **A matrix logarithm is not an activation** — it is a readout to the
tangent space, and it belongs in the head.

## SPDBatchNorm

```
Σ → G^{1/2} (M^{-1/2} Σ M^{-1/2}) G^{1/2}
```

Whiten by the batch Fréchet mean `M`, re-bias towards a learnable SPD reference `G` — the
affine-invariant analogue of `(x − μ)/σ · γ`. In eval mode `M` is a running mean transported the
same way, so the layer is deterministic at test time.

This is the single most useful layer for making deep SPD stacks trainable. Without it the spectra
of successive congruences drift by orders of magnitude and the logarithms in the head saturate.

## GeodesicDropout

Stochastic interpolation towards the identity **along the geodesic**, with a random `t ∈ [0, p]` per
sample. Unlike coordinate dropout this cannot leave the manifold, and unlike a convex combination
with the identity it follows the metric the rest of the network uses.

## RiemannianPool

| `mode` | Cost | Notes |
|---|---|---|
| `"logeuclid"` | one decomposition per matrix | closed form, no iteration, **default** |
| `"frechet"` | `iters` decompositions per sample | the true barycentre |
| `"arithmetic"` | none | **ablation only** — the barycentre of no Riemannian metric; inflates the spectrum |

`logeuclid` is the default because on GPU the iteration dominates a forward pass and the accuracy
difference is typically within seed noise. Switch to `frechet` when the channel spread is large.

The two agree exactly when the inputs commute.

## Heads

**`TangentHead(dim, num_classes, learn_reference=True)`** — one logarithm per sample, then a linear
classifier on the isometrically vectorised tangent matrix (off-diagonals scaled by `√2` so the
vector norm equals the Frobenius norm). Cost is independent of the class count. With
`learn_reference=False` this is exactly the classic LogEig+FC readout.

**`GeodesicPrototypeHead(dim, num_classes, prototypes_per_class=1)`** —
`p(c | Σ) = softmax(−d²(Σ, P_c)/τ)` with trainable SPD prototypes. Geometry all the way to the
logits, and the prototypes are inspectable objects on the manifold. Cost is `B × C` eigenvalue
computations; prototype whitening factors are computed once per forward and reused across the batch.

`prototypes_per_class > 1` turns each class into a union of geodesic balls.

## EGNBlock

`BiMap → SPDBatchNorm → GeometricBias → SpectralActivation → GeodesicDropout` — the `conv → bn →
relu` analogue. Stacking blocks with decreasing `dim` and increasing `channels` reproduces the shape
schedule of a convolutional trunk, and the same intuitions apply: narrow early layers under-fit, and
an over-aggressive size drop in one step loses spectrum that later layers cannot recover.

## Building by hand

```python
import torch.nn as nn
from egn.nn import EGNBlock, GeodesicPrototypeHead, RiemannianPool, ToSPD

model = nn.Sequential(
    ToSPD(kind="signal", branches=4),
    EGNBlock(22, 16, in_channels=4, out_channels=8, mix=True),
    EGNBlock(16,  8, in_channels=8, out_channels=8),
    RiemannianPool("frechet", iters=5),
    GeodesicPrototypeHead(8, num_classes=4),
)
```

Train it with `egn.build_optimizer(model, lr=1e-3)`, which splits parameters into manifold and
Euclidean groups automatically. Weight decay applies only to the Euclidean group — shrinking a
Stiefel frame towards zero is meaningless, since zero is not on the manifold.
