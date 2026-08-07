# The geometry

## Why not just flatten the matrix

An SPD matrix has `n(n+1)/2` free numbers, so it is tempting to vectorise it and use an ordinary
network. The problem is that the set of SPD matrices is an open cone, not a vector space, and the
operations a network performs leave it immediately:

- the average of two covariances is SPD, but it is not their barycentre under any metric the data
  respects — it systematically inflates the determinant;
- a gradient step of any size can push a matrix out of the cone entirely;
- the straight line between two covariances is not the path of intermediate covariances.

EGN uses the **affine-invariant Riemannian metric** (AIRM),

```
g_S(U, V) = tr(S⁻¹ U S⁻¹ V)
```

Under it, `S^n_++` is a Hadamard manifold — complete, simply connected, non-positively curved.
Three consequences the implementation relies on: the Fréchet mean exists and is unique, the
fixed-point iteration that finds it converges from any SPD seed, and there is no cut locus, so the
squared distance is smooth everywhere.

## The operations

| Concept | Formula | In code |
|---|---|---|
| Logarithm | `Log_S(P) = S^{1/2} log(S^{-1/2} P S^{-1/2}) S^{1/2}` | `geometry.riemannian_log` |
| Exponential | `Exp_S(V) = S^{1/2} exp(S^{-1/2} V S^{-1/2}) S^{1/2}` | `geometry.riemannian_exp` |
| Distance | `d²(S, P) = ‖log(S^{-1/2} P S^{-1/2})‖²_F` | `geometry.squared_distance` |
| Geodesic | `γ(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}` | `geometry.geodesic` |
| Gradient conversion | `grad = S sym(∂L/∂S) S` | `geometry.egrad2rgrad_spd` |

Two identities worth internalising, both asserted in the test suite:

`d(A, γ(t)) = t · d(A, B)` — the geodesic has constant speed, so `t` means what you expect.

`∇_S ½ d²(S, P) = −Log_S(P)` — the Riemannian gradient of the squared distance is exactly the
negative logarithm. No chain-rule correction is needed anywhere downstream, which is why the
prototype head needs no custom backward.

## What is invariant, and what is not

Affine invariance means

```
d(A S Aᵀ, A P Aᵀ) = d(S, P)     for every invertible A
```

This is the property that makes the metric appropriate for covariance data: a change of sensor
gains, a re-referencing of EEG channels, or a linear mixing of sources is a congruence, and the
metric is blind to it.

**Two claims that are commonly overstated, and are false:**

*"The network output is congruent to the input's congruence."* For a rectangular `BiMap` with
`W ∈ St(d, m)`, `m < d`, congruencing the input by `Q` is absorbed into a **reparameterisation of
the weight**, `W → QᵀW`, not into a congruence of the output. The strong form holds only in the
square case with `W` orthogonal. `tests/test_egn.py::test_bimap_reparameterisation_identity`
asserts the correct version and the counter-example.

*"The classifier is invariant to congruence of the input."* Only jointly. Congruencing the input
**and** the prototypes by the same matrix leaves the logits unchanged. Congruencing only the input
changes every geodesic distance, and it should — otherwise the prototypes would carry no
information about where the data lives.

## Differentiating through eigenvalues

A spectral map `M = U diag(λ) Uᵀ ↦ U diag(φ(λ)) Uᵀ` has differential
`U (L ∘ (Uᵀ sym(dX) U)) Uᵀ`, where `L` is the Löwner divided-difference matrix

```
L_ij = (φ(λ_i) − φ(λ_j)) / (λ_i − λ_j)      i ≠ j
L_ii = φ′(λ_i)
```

The off-diagonal expression is `0/0` when two eigenvalues coincide. This is not a rare edge case in
this architecture — `SpectralActivation("rect")` clamps eigenvalues to a floor, which produces
**exactly equal** floating-point values on purpose. A naive implementation returns `NaN` the first
time it fires.

`functional._loewner` detects near-coincident eigenvalues within a tolerance and substitutes the
derivative. Eigenvalues that were clamped receive a zero subgradient rather than an unbounded one.
The guard can be disabled with `config.loewner_guard = False` **for ablation only**; the
consequence is measurable and is one row of the paper's stability table.

## Keeping parameters on their manifolds

Constrained parameters are not projected after the fact — they are updated by retraction:

- **Stiefel** (`BiMap` weights): gradient projected to the tangent space, then QR retraction with a
  sign fix. Orthonormality holds to `1e-5` after hundreds of steps.
- **SPD** (prototypes, batch-norm references): `egrad2rgrad` is `X sym(G) X`, and the retraction is
  the exact exponential map, so positive definiteness is guaranteed for any step size.

Two safeguards on the SPD manifold exist for numerical, not mathematical, reasons. A trust region
caps the whitened tangent norm, and the retraction bounds the condition number. Without them an
aggressive Adam step early in training can drive a prototype to the boundary and the run dies inside
`eigh` with an unhelpful convergence error. In `float32`, a matrix whose condition number exceeds
`1/ε` is SPD in exact arithmetic but not after rounding — bounding conditioning is what makes
"positive definite" mean something at the working precision.

## Standalone use

None of this requires a model:

```python
import torch
from egn.geometry import distance, frechet_mean, geodesic, random_spd

S = random_spd((64, 10, 10), condition=20.0)
M = frechet_mean(S.unsqueeze(0), dim=1, iters=20)   # Riemannian barycentre
d = distance(S[0], S[1])
mid = geodesic(S[0], S[1], 0.5)
```

Verified against `pyriemann`: the Fréchet mean agrees to `4.7e-10`, the distance exactly.
