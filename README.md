<div align="center">

# EGN — Equivariant Geodesic Networks

**Implementation of	Equivariant Geodesic Networks: End-to-End Classification on the SPD Manifold. (Under review at AAAI 27)**

[![PyPI](https://img.shields.io/pypi/v/egnlib.svg)](https://pypi.org/project/egnlib/)
[![Python](https://img.shields.io/pypi/pyversions/egnlib.svg)](https://pypi.org/project/egnlib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/kraihan/EGN/actions/workflows/tests.yml/badge.svg)](https://github.com/kraihan/EGN/actions/workflows/tests.yml)

[Installation](#installation) · [Quickstart](#quickstart) · [Documentation](docs/) · [Examples](examples/) · [Experiments](experiments/) · [Citation](#citation)

</div>

---

## What this is

Many kinds of data are naturally described not by a vector but by a **covariance matrix**: EEG
epochs, radar returns, skeleton trajectories, diffusion tensors, functional-connectivity maps.
Those matrices live on a curved space — the SPD manifold — where the straight line between two
points leaves the space entirely. Treating them as flat vectors discards the geometry that makes
them informative.

EGN is a deep classifier that stays on that manifold from input to logits. Every intermediate
representation is an exact SPD matrix under the affine-invariant metric; there is no projection
step anywhere in the forward pass, and constrained parameters are updated by retraction, so a
Stiefel frame stays orthonormal and an SPD prototype stays positive definite regardless of step
size.

```
x ──ToSPD──► (B, K, d, d) ──EGNBlock ×L──► (B, K′, m, m) ──pool──► (B, m, m) ──head──► logits
```

**Design goal: one object, any dataset.** `EGNClassifier(epochs=30).fit(X, y)` infers the matrix
size, the channel count, the class count and the label vocabulary from the data itself. The same
code trains on 22-channel EEG, 93×93 skeleton covariances, complex radar returns and image
descriptors without a per-dataset subclass.

---

## Installation

```bash
pip install egnlib
```

Requires Python ≥ 3.9. The only dependencies are **PyTorch ≥ 2.0 and NumPy** — no `geoopt`, no
Riemannian-optimisation stack. The manifolds and optimisers are implemented in this package
(`src/egn/manifolds.py`, `src/egn/optim.py`).

From source:

```bash
git clone https://github.com/kraihan/EGN.git
cd EGN
pip install -e ".[dev]"
pytest -q
```

---

## Quickstart

```python
import numpy as np
from egn import EGNClassifier

X = np.random.randn(512, 22, 256)      # 512 trials, 22 channels, 256 samples
y = np.random.randint(0, 4, 512)

clf = EGNClassifier(epochs=30).fit(X, y)
print(clf.score(X, y))
```

`X` may be a NumPy array, a torch tensor, a `Dataset` or a `DataLoader`; `y` may be integers,
strings, or anything hashable. `predict` returns labels in your own vocabulary.

### Input conventions

`ToSPD` infers the convention from the tensor rank when `input_kind="auto"`:

| Input | Meaning | Output |
|---|---|---|
| `(B, n, n)` | covariance already on the manifold | `(B, 1, n, n)` |
| `(B, K, n, n)` | multi-branch SPD (e.g. per frequency band) | `(B, K, n, n)` |
| `(B, C, T)` | multichannel signal | `(B, K, C, C)` |
| `(B, T, D)` | sequence of features (`input_kind="sequence"`) | `(B, K, D, D)` |
| `(B, C, H, W)` | feature map / image | `(B, K, C, C)` |

Rank-2 input is rejected with an explanatory error rather than silently becoming a singular
rank-one outer product that would fail later inside a logarithm.

---

## The CNN correspondence

The layer vocabulary mirrors a convolutional network on purpose, so that architectural intuition
transfers:

| CNN | EGN | What it does |
|---|---|---|
| `Conv2d(c_in, c_out, k)` | `BiMap(d_in, d_out, c_in, c_out)` | congruence by a Stiefel frame, `Σ → WᵀΣW` |
| channels | manifold channels | independent, or mixed via `mix=True` |
| stride / downsampling | matrix-size reduction `d_in → d_out` | |
| `BatchNorm2d` | `SPDBatchNorm` | whitens by a Fréchet mean, re-biases to a learnable SPD reference |
| `ReLU` | `SpectralActivation` | acts on eigenvalues only, so it commutes with congruence |
| `Dropout` | `GeodesicDropout` | interpolates towards the identity **along a geodesic** |
| bias | `GeometricBias` | congruence by an invertible Cholesky factor — a metric isometry |
| global average pool | `RiemannianPool` | Fréchet, log-Euclidean, or (for ablation) arithmetic |
| linear classifier | `TangentHead` / `GeodesicPrototypeHead` | |

Building an architecture by hand looks like any other PyTorch model:

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

Or use the factories, following the `torchvision` convention: `egn_tiny`, `egn_small`, `egn_base`.

### Choosing a head

**`TangentHead`** (default) takes one logarithm per sample and applies a linear classifier in the
tangent space at a learnable reference point. Cost is independent of the class count.

**`GeodesicPrototypeHead`** scores by squared geodesic distance to trainable SPD prototypes,
`p(c | Σ) = softmax(−d²(Σ, P_c)/τ)`. Fully geometric, inspectable prototypes, cost grows with the
number of classes.

Both are invariant in the *joint* sense: congruencing the input **and** the reference by the same
matrix leaves the logits unchanged. Invariance while holding the prototypes fixed is false —
congruencing only the input changes every geodesic distance. See [docs/geometry.md](docs/geometry.md).

---

## Performance

An earlier research prototype of this method ran in `float64` and was **slower on GPU than on
CPU**. Six causes, all addressed here:

1. **`float64` everywhere.** Consumer and inference-class GPUs execute double precision at 1/32 of
   their `float32` rate. The policy is now `config.spectral_dtype = "auto"` — `float32` on CUDA,
   `float64` on CPU.
2. **Host synchronisation inside the forward pass.** The old Fréchet mean polled a residual with
   `.item()` every iteration, draining the CUDA queue several times per layer. Nothing in the
   forward pass calls `.item()`; the mean runs a fixed iteration budget.
3. **Repeated eigendecompositions.** `sqrtm_pair` returns `Σ^{1/2}` and `Σ^{-1/2}` from one
   decomposition; the prototype head whitens once per prototype per forward, not once per
   (sample, prototype) pair; distances use `eigvalsh`, which never forms eigenvectors.
4. **Batching.** Every operator accepts an arbitrary leading shape and issues exactly one `eigh`
   per call. `config.eig_chunk` caps the batch when memory is the constraint.
5. **Per-step metric readback.** Training statistics accumulate on-device and are read once per
   epoch.
6. **Pooling cost.** The default pool is the closed-form log-Euclidean barycentre; the iterative
   Fréchet mean is one flag away.

Measure your own hardware rather than trusting any of this:

```bash
python -m egn.benchmark --sizes 16 32 64 --batch 256 1024 4096
```

**Honest caveat.** These are small-matrix, decomposition-heavy workloads. A GPU wins decisively at
large batch and `float32`; at batch 32 with 8×8 matrices it may still lose to a CPU because the
kernels never fill the device. That is a property of the operation, not of this implementation.

### Scaling out

```bash
torchrun --nproc_per_node=4 examples/04_train_ddp.py
```

`EGNClassifier` detects the process group, wraps the model in `DistributedDataParallel`, installs a
`DistributedSampler` and reduces metrics across ranks. DDP works with the Riemannian optimiser
without special handling: DDP all-reduces the Euclidean gradients, and every rank then applies the
same deterministic retraction to the same parameters, so replicas stay identical.

---

## Geometry as a public API

`egn.geometry` and `egn.functional` are usable standalone, with no model involved:

```python
from egn.geometry import distance, frechet_mean, geodesic, riemannian_log

d   = distance(A, B)                 # affine-invariant geodesic distance
M   = frechet_mean(batch, dim=1)     # Riemannian barycentre
mid = geodesic(A, B, 0.5)            # midpoint on the manifold
```

Everything is differentiable, **including through repeated eigenvalues**: the backward pass uses the
Löwner divided-difference matrix, and clamped eigenvalues receive a zero subgradient rather than an
unbounded one. This is not a detail — a spectral floor deliberately creates exact eigenvalue ties,
and the naive divided-difference formula returns `NaN` the moment it does.

---

## Repository layout

```
EGN/
├── src/egn/              the library
│   ├── functional.py     batched spectral operators, Löwner backward
│   ├── geometry.py       AIRM: exp, log, distance, geodesic, Fréchet mean
│   ├── manifolds.py      Stiefel, SPD, ManifoldParameter  (no geoopt)
│   ├── optim.py          RiemannianAdam / RiemannianSGD
│   ├── nn/               ToSPD, BiMap, SPDBatchNorm, heads, blocks, pooling
│   ├── models.py         EGN + egn_tiny / egn_small / egn_base
│   ├── classifier.py     EGNClassifier — the scikit-learn surface
│   ├── data.py           adapters for arrays, tensors, Datasets, DataLoaders
│   ├── engine.py         training / evaluation loops, AMP, DDP reduction
│   └── benchmark.py      python -m egn.benchmark
├── tests/                46 tests — geometry identities, not just shapes
├── docs/                 conceptual documentation
├── examples/             runnable scripts, smallest first
├── experiments/          reproduction notebooks for the paper's tables
└── benchmarks/           hardware measurement scripts
```

---

## Tests

```bash
pytest -q
```

The suite asserts the **geometry**, not just the shapes: exp/log inverse to machine precision,
affine invariance of the distance, geodesic constant speed, the Fréchet mean as a fixed point and
its equivariance, `gradcheck` on every spectral operator including an exact eigenvalue tie, and the
closed-form prototype gradient against autograd.

It also asserts the **counter-examples** — that a convex combination is *not* a geodesic, that a
rectangular `BiMap` is not output-congruent, and that the unconstrained ablation genuinely leaves
the Stiefel manifold — because those are the claims that are easiest to overstate.

Independently verified against `pyriemann`: the Fréchet mean agrees to `4.7e-10` and the
affine-invariant distance exactly. See [experiments/](experiments/).

---

## Documentation

| Page | Contents |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | install, first model, input conventions |
| [docs/geometry.md](docs/geometry.md) | the metric, what is and is not invariant |
| [docs/layers.md](docs/layers.md) | every layer, what it preserves, when to use it |
| [docs/performance.md](docs/performance.md) | precision policy, GPU behaviour, DDP |
| [docs/api.md](docs/api.md) | full API reference |
| [docs/faq.md](docs/faq.md) | common errors and what they mean |
| [MIGRATION.md](MIGRATION.md) | moving from the earlier research codebase |

---

## Related work

This package sits alongside, not on top of, the existing SPD ecosystem:

- **[pyRiemann](https://github.com/pyRiemann/pyRiemann)** — Riemannian statistics and classical
  classifiers (MDM, tangent-space LDA) for biosignals. Not a deep-learning library.
- **[geomstats](https://github.com/geomstats/geomstats)** — general-purpose differential geometry.
  Has SPD geometry; no deep classifier.
- **[SPDLearn](https://spdlearn.org/)** — SPD deep learning via trivialization, with reference
  implementations of SPDNet-family models.
- **SPDNet** (Huang & Van Gool, AAAI 2017), **SPDNetBN** (Brooks et al., NeurIPS 2019) — the
  architectures this line of work builds on.

EGN's distinguishing choices are the geodesic prototype head, the geometric bias as a metric
isometry, geodesic dropout, the guarded Löwner backward, and the GPU-oriented numerical policy.

---

## Citation

```bibtex
@software{khan2026egn,
  author  = {Khan, Md Raihan},
  title   = {EGN: Equivariant Geodesic Networks on the SPD manifold},
  year    = {2026},
  url     = {https://github.com/kraihan/EGN},
  version = {0.2.2}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable form. If a peer-reviewed reference becomes
available it will be listed in [PAPER.md](PAPER.md).

---

## License

MIT — see [LICENSE](LICENSE). This package contains **no third-party research code**. Earlier
prototypes of this work were developed inside a fork of an SPD-network repository; that fork is not
redistributed here, which is why this package can carry a permissive licence. See
[MIGRATION.md](MIGRATION.md).

## Author

**Md Raihan Khan** — Lecturer, Department of EEE, North Western University, Bangladesh; M.Sc.
candidate, Khulna University of Engineering and Technology.
[Website](https://kraihan.github.io) · [Google Scholar](https://scholar.google.com/citations?user=E3iFEuUAAAAJ&hl=en)

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
