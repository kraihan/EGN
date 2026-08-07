# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are tagged so a reader can tell library quality from experiment scaffolding:

- **fix** / **feature** — affects every user
- **ablation** — a switch that exists to disable part of the method for a paper table; no ordinary
  user should set it

## [0.2.2] — 2026-08-08

### Fixed
- **fix**: a NaN or Inf input surfaced as `_LinAlgError: the algorithm failed to converge` from
  inside `torch.linalg.eigh`, several layers from the cause. `EGNClassifier.fit` now validates the
  first batch, and `ToSPD(check_input=True)` validates every batch. The check is off in the forward
  path by default because a finiteness test forces a device synchronisation.
- **fix**: `EGN(...).eval()` followed by a single-sample forward crashed. The lazy build created
  submodules in training mode regardless of the parent's state, leaving the head's `BatchNorm1d`
  expecting a batch larger than one. `build()` now propagates the parent's train/eval state.

## [0.2.1] — 2026-08-08

### Added
- **ablation**: `config.loewner_guard`. Disabling it removes the tolerance on eigenvalue gaps in the
  divided-difference backward, reproducing the form most derivations write. It returns non-finite
  values whenever two eigenvalues coincide, which is exactly what a spectral floor arranges. For
  ablation only.

## [0.2.0] — 2026-08-08

### Added
- **ablation**: `pool="arithmetic"` — the Euclidean average, the barycentre of no Riemannian metric.
- **ablation**: `constrained=False` on `BiMap` — an unconstrained weight, ablating the Stiefel
  constraint while keeping the same initialisation.
- **ablation**: `head="logeig"` — `TangentHead` with the reference pinned at the identity, i.e. the
  classic LogEig+FC readout.
- **ablation**: `separation` in `build_criterion` / `EGNClassifier` — prototype repulsion.
- **feature**: `bias` and `constrained` exposed on `EGNClassifier`.

## [0.1.1] — 2026-08-08

### Fixed
- **fix**: `torch.cuda.is_available()` returns True on a GPU whose compute capability the installed
  PyTorch was not built for (a P100 against an sm_70+ wheel), and the first allocation then died
  with `cudaErrorNoKernelImageForDevice`. Device selection now runs one small kernel before
  committing, warns, and falls back to CPU. An explicit `device="cuda"` is still honoured as given.

## [0.1.0] — 2026-08-07

### Added
- Initial release. Batched spectral operators with a guarded Löwner backward; affine-invariant
  geometry; Stiefel and SPD manifolds with retraction-based optimisers; `ToSPD` input adapters;
  `BiMap`, `GeometricBias`, `SpectralActivation`, `SPDBatchNorm`, `GeodesicDropout`,
  `RiemannianPool`; `TangentHead` and `GeodesicPrototypeHead`; `EGN` with lazy shape inference;
  `EGNClassifier`; DDP support; `python -m egn.benchmark`.
- Rewritten from a research prototype that ran in `float64` with host synchronisation inside the
  forward pass and was consequently slower on GPU than on CPU.
