# API reference

## `egn.EGNClassifier`

The scikit-learn surface. Constructor arguments, grouped.

### Model

| Argument | Default | Meaning |
|---|---|---|
| `dims` | `None` | explicit width schedule, e.g. `[93, 50, 30]`; first entry must be the input size |
| `depth` | `2` | number of blocks when `dims` is `None` |
| `channels` | `None` | manifold channels per block; scalar broadcasts |
| `branches` | `1` | stem channels — windows the input is split into |
| `head` | `"tangent"` | `"tangent"`, `"logeig"`, `"geodesic"` |
| `pool` | `"logeuclid"` | `"logeuclid"`, `"frechet"`, `"arithmetic"` (ablation) |
| `pool_iters` | `5` | Fréchet iterations |
| `activation` | `"rect"` | `"rect"`, `"power"`, `"none"` |
| `dropout` | `0.0` | geodesic dropout probability |
| `batchnorm` | `True` | insert `SPDBatchNorm` in each block |
| `bias` | `True` | insert `GeometricBias` in each block |
| `constrained` | `True` | keep `BiMap` weights on Stiefel (ablation: `False`) |
| `mix` | `False` | cross-channel mixing in `BiMap` |
| `input_kind` | `"auto"` | `"spd"`, `"signal"`, `"sequence"`, `"image"` |
| `ridge` | `1e-4` | scale-invariant ridge on the formed covariance |
| `shrinkage` | `0.0` | Ledoit-Wolf shrinkage towards the scaled identity |
| `min_dim` | `8` | floor for the automatic width schedule |
| `prototypes_per_class` | `1` | geodesic head only |

### Optimisation

`epochs=30`, `batch_size=64`, `lr=1e-3`, `optimizer="adam"`, `weight_decay=0.0`, `grad_clip=0.0`,
`label_smoothing=0.0`, `separation=0.0` (prototype repulsion), `scheduler="none"` or `"cosine"`.

### Runtime

`device="auto"`, `amp=False`, `amp_dtype="bfloat16"`, `num_workers=0`, `data_parallel=False`,
`spectral_dtype="auto"`, `seed=1234`, `verbose=1`.

### Methods

| Method | Returns |
|---|---|
| `fit(X, y=None, eval_set=None)` | `self` |
| `predict(X)` | labels in the caller's vocabulary |
| `predict_proba(X)` | `(N, C)` |
| `decision_function(X)` | logits |
| `score(X, y)` | accuracy |
| `transform(X)` | pooled SPD descriptors — use EGN as a feature extractor |
| `save(path)` / `EGNClassifier.load(path)` | architecture, weights, labels, history |
| `get_params()` / `set_params(**kw)` | scikit-learn compatibility |

`classes_` holds the label vocabulary; `history` holds the per-epoch metrics.

---

## `egn.EGN`

An `nn.Module`. Same model arguments as above, plus `num_classes` and `in_dim`. Shapes are inferred
on the first forward pass; call `build_from_example(x, num_classes)` to materialise eagerly, which
is required before wrapping in `DistributedDataParallel`.

`features(x)` returns the pooled SPD descriptor, `num_parameters()` the trainable count.

Factories: `egn_tiny`, `egn_small`, `egn_base`.

---

## `egn.nn`

`ToSPD`, `BiMap`, `GeometricBias`, `SpectralActivation`, `SPDBatchNorm`, `GeodesicDropout`,
`RiemannianPool`, `TangentHead`, `GeodesicPrototypeHead`, `EGNBlock`, `vectorize_tangent`,
`infer_input_kind`, `infer_matrix_size`. See [layers.md](layers.md).

---

## `egn.geometry`

| Function | Meaning |
|---|---|
| `riemannian_log(S, P)` / `riemannian_exp(S, V)` | logarithm and exponential maps |
| `squared_distance(S, P)` / `distance(S, P)` | affine-invariant distance |
| `geodesic(A, B, t)` | constant-speed interpolation |
| `frechet_mean(S, dim, weights, iters)` | Riemannian barycentre |
| `log_euclidean_mean(S, dim, weights)` | closed-form barycentre |
| `parallel_transport_identity(S, V)` | transport a tangent vector from the identity |
| `egrad2rgrad_spd(S, egrad)` | Euclidean → Riemannian gradient |
| `random_spd(*shape, condition=...)` | Haar eigenvectors, log-uniform spectrum |
| `is_spd(S, tol)` | per-matrix symmetry and positivity check |

---

## `egn.functional`

`sym`, `spd_regularize`, `shrinkage`, `trace`, `identity_like`, `logm`, `expm`, `sqrtm`, `invsqrtm`,
`sqrtm_pair`, `powm`, `reeig`, `eigvals_logsq`.

All batched over arbitrary leading shapes, one `eigh` per call, differentiable through repeated
eigenvalues.

---

## `egn.manifolds`

`Manifold`, `Euclidean`, `Stiefel`, `SPD`, `ManifoldParameter`.

Each manifold provides `project`, `proju`, `egrad2rgrad`, `retract`, `transport`, `inner`, `init`.
`SPD(max_step=4.0, max_condition=1e6)` — see [geometry.md](geometry.md) for why those bounds exist.

---

## `egn.optim`

`RiemannianAdam`, `RiemannianSGD`, `build_optimizer(model, lr, kind, weight_decay)`.

Both accept a mixed parameter list; anything that is a `ManifoldParameter` is updated by retraction,
everything else follows the ordinary Euclidean rule.

---

## `egn.data`

`LabelEncoder`, `ArrayDataset`, `as_dataset`, `make_loader`, `collate_padded`.

`collate_padded` truncates variable-length trials to the shortest rather than zero-padding: padding
biases a covariance estimate towards the identity by exactly the padded fraction.

---

## `egn.engine`, `egn.utils`, `egn.config`

`train_one_epoch`, `evaluate` · `seed_everything`, `init_distributed`, `is_main_process`,
`all_reduce_mean`, `unwrap` · `config`, `numeric_context`, `set_performance_defaults`.
