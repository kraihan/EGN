# Migrating from the research repository

The old repository was a fork of SPDMLR with an `egn/` package added, driven by hydra
configs and shell scripts. This package is a library: no configs, no entry-point scripts, no
datasets, no upstream code.

## The licensing reason to switch

The old checkout redistributed `spdnets/`, `spd/`, `library/`, `datasets/spdnet/`,
`datasets/eeg/`, `Hyperplane/` and the `conf/SPDNet`, `conf/TSMNet` trees from a repository
that ships **no LICENSE file**. Publishing that as a package is not clearly permitted. This
package contains none of it, which is why it can carry an MIT licence.

If you still need the SPDNet / TSMNet baselines, keep them in a separate private checkout
and import `egn` into it, rather than shipping them inside your release.

## File map

| Old | New |
|---|---|
| `egn/geometry.py` | `egn/functional.py` (spectral operators) + `egn/geometry.py` (AIRM) |
| `egn/layers.py` | `egn/nn/layers.py`, `egn/nn/inputs.py`, `egn/nn/blocks.py` |
| `egn/heads.py` | `egn/nn/heads.py` |
| `egn/losses.py` | `egn/losses.py` |
| `egn/models/egn.py` | `egn/models.py` |
| `egn/utils/pipeline.py`, `egn/training/egn_training.py` | `egn/engine.py`, `egn/classifier.py` |
| `egn/utils/registry.py`, `datasets/egn/*` | removed — `egn/data.py` adapts whatever you have |
| `conf/EGN/*.yaml`, `exp_egn.sh` | constructor arguments |
| `verify_egn.py` | `tests/test_egn.py` |
| geoopt dependency | `egn/manifolds.py`, `egn/optim.py` |

## API map

| Old | New |
|---|---|
| `SPDFormation(mode="gram")` | `ToSPD(kind="signal")` |
| `SPDFormation(mode="ridge")` | `ToSPD(kind="spd")` |
| `EquivariantBilinearMap` | `nn.BiMap` |
| `GeometricBias` | `nn.GeometricBias` |
| `SpectralActivation` | `nn.SpectralActivation` |
| `RiemannianMeanPool` | `nn.RiemannianPool` |
| `GeodesicSoftDropout` | `nn.GeodesicDropout` |
| `TangentVectorization` | `nn.vectorize_tangent` / `nn.TangentHead` |
| `karcher_mean` | `geometry.frechet_mean` |
| `log_euclidean_mean` | `geometry.log_euclidean_mean` |
| `airm_gradient` | `geometry.egrad2rgrad_spd` |
| `MatrixLog.apply(X)` | `functional.logm(X)` (same for exp / sqrt / invsqrt / pow) |
| `EigRectify` | `functional.reeig` |
| `variant="geodesic"` | `pool="frechet"`, `head="geodesic"` |
| `variant="fixed"` | `pool="logeuclid"`, `head="tangent"` |

## Behavioural changes worth knowing before you re-run experiments

* **Default precision is `float32` on CUDA.** Set `spectral_dtype="float64"` if you are
  reproducing a number that was produced in double precision.
* **The Fréchet mean runs a fixed iteration count** (`pool_iters`, default 5) rather than
  polling a residual. Raise it if you need the tighter fixed point; the old adaptive
  damping is gone because every residual test cost a device synchronisation.
* **`SPDBatchNorm` is new and on by default.** It changes results, usually for the better,
  and it is what makes deeper trunks trainable. Pass `batchnorm=False` for the old
  behaviour.
* **SPD parameters are condition-number bounded** (`SPD(max_condition=1e6)`) and steps are
  clipped to a trust region. Without this an aggressive learning rate could drive a
  prototype to the boundary and crash inside `eigh`.
* **The prototype head uses the squared distance and plain cross-entropy.** No custom
  backward. `tests/test_egn.py::test_prototype_gradient_matches_closed_form` checks the
  resulting gradient against the analytic expression.

## Reproducing an old experiment

```python
# old: python EGN-Geodesic.py dataset=HDM05 nnet.variant=geodesic fit.epochs=200 seed=1024
from egn import EGNClassifier

clf = EGNClassifier(
    depth=2, branches=1, head="geodesic", pool="frechet", pool_iters=20,
    epochs=200, batch_size=30, lr=1e-2, seed=1024,
    spectral_dtype="float64",
)
clf.fit(X_train, y_train, eval_set=(X_test, y_test))
```

Loading the data is now your code, not the library's — a `numpy` array of shape
`(N, n, n)` or `(N, C, T)` and a label vector is all it wants.
