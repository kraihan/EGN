# Paper

<!--
  ------------------------------------------------------------------------------
  TEMPLATE — fill in after a decision is public.

  Do not add a link to a manuscript under double-blind review, and do not name a
  venue in a public repository while review is ongoing. A public repository that
  identifies itself as the implementation of a submission is a de-anonymisation
  route regardless of intent: a reviewer who searches a distinctive term from the
  paper can find it. Keep this file as a stub until the decision is out.
  ------------------------------------------------------------------------------
-->

A manuscript describing the method implemented in this repository is in preparation. This page will
carry the reference, the BibTeX entry, and the mapping from the paper's sections to the code once it
is publicly available.

## Method-to-code map

| Component | Implementation |
|---|---|
| Affine-invariant geometry | [`src/egn/geometry.py`](src/egn/geometry.py) |
| Spectral operators, Löwner backward | [`src/egn/functional.py`](src/egn/functional.py) |
| Equivariant bilinear map | `BiMap` in [`src/egn/nn/layers.py`](src/egn/nn/layers.py) |
| Geometric bias (metric isometry) | `GeometricBias`, same file |
| Spectral activation | `SpectralActivation`, same file |
| Riemannian batch normalisation | `SPDBatchNorm`, same file |
| Geodesic soft dropout | `GeodesicDropout`, same file |
| Fréchet / log-Euclidean pooling | [`src/egn/nn/pooling.py`](src/egn/nn/pooling.py) |
| Geodesic prototype head | `GeodesicPrototypeHead` in [`src/egn/nn/heads.py`](src/egn/nn/heads.py) |
| Prototype repulsion term | [`src/egn/losses.py`](src/egn/losses.py) |
| Stiefel / SPD retraction | [`src/egn/manifolds.py`](src/egn/manifolds.py), [`src/egn/optim.py`](src/egn/optim.py) |

## Claims and where they are tested

| Claim | Test |
|---|---|
| Exp and Log are mutually inverse | `test_exp_log_are_inverse` |
| The distance is affine invariant | `test_distance_is_affine_invariant` |
| Geodesics have constant speed | `test_geodesic_endpoints_and_speed` |
| A convex combination is **not** a geodesic | `test_convex_combination_is_not_a_geodesic` |
| The Fréchet mean is a fixed point, and equivariant | `test_frechet_mean_is_a_fixed_point`, `test_frechet_mean_is_affine_equivariant` |
| `∇ ½d²(S,P) = −Log_S(P)` | `test_egrad2rgrad_reproduces_minus_log` |
| Spectral gradients are correct, including at an exact eigenvalue tie | `test_gradcheck_spectral`, `test_gradcheck_with_repeated_eigenvalues` |
| The prototype gradient matches the closed form | `test_prototype_gradient_matches_closed_form` |
| A rectangular `BiMap` is **not** output-congruent | `test_bimap_reparameterisation_identity` |
| Constrained weights stay on their manifolds; the ablation does not | `test_stiefel_and_spd_parameters_stay_on_their_manifolds`, `test_unconstrained_bimap_leaves_the_stiefel_manifold` |

## Reproducing the tables

See [`experiments/`](experiments/).
