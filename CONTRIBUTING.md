# Contributing

Thanks for considering a contribution.

## Development setup

```bash
git clone https://github.com/kraihan/EGN.git
cd EGN
pip install -e ".[dev]"
pre-commit install
pytest -q
```

## The one rule that matters

**A change to the geometry needs a test that states an identity, not a shape.**

`assert out.shape == (8, 4)` proves nothing about a Riemannian operator. The existing suite checks
things like `Exp_S(Log_S(P)) == P` to machine precision, `d(A, QAQᵀ) == d(B, QBQᵀ)` for the affine
invariance, `d(A, γ(t)) == t·d(A, B)` for constant speed, and `gradcheck` on every spectral operator
including one with an exact eigenvalue tie. New operators are held to that standard.

If a claim is easy to overstate, add the counter-example too. The suite contains
`test_convex_combination_is_not_a_geodesic` and `test_bimap_reparameterisation_identity` precisely
because both statements are commonly asserted in a stronger form than is true.

## Performance constraints

The forward pass is on a hot path with small matrices, where a single host synchronisation can cost
more than the arithmetic. In `src/egn/functional.py`, `geometry.py` and `nn/`:

- no `.item()`, `.cpu()`, `bool(tensor)`, or data-dependent Python branching;
- one `eigh` per call — batch over the leading axes, never loop;
- iterative routines run a fixed budget rather than polling a residual;
- validation that requires reading a tensor belongs at the boundary (`fit`, `ToSPD(check_input=)`),
  not in the layer.

## Numerical policy

Precision is decided once, in `src/egn/config.py`, not per layer. If you need double precision for a
derivation, use `with egn.numeric_context(spectral_dtype="float64"):` rather than casting inside an
operator.

## Style

`ruff` with a 100-character line length, enforced by pre-commit. Docstrings explain *why* a choice
was made where the choice is not obvious — a reader who knows the mathematics should not have to
guess why the implementation deviates from the textbook form.

## Pull requests

1. One logical change per PR.
2. Tests pass, and new behaviour has a test.
3. Add a `CHANGELOG.md` entry tagged **fix**, **feature**, or **ablation**.
4. Ablation switches — flags whose purpose is to disable part of the method — are documented as
   such, so they are never mistaken for tuning knobs.
