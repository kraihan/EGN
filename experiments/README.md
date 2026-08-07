# Experiments

Reproduction material for the empirical claims in the README and the accompanying manuscript.
Nothing here is imported by the library — `src/egn` ships no dataset code, by design.

| File | What it produces |
|---|---|
| [`ablation_study.ipynb`](ablation_study.ipynb) | eight architectural variants × five datasets, emitted as a LaTeX table |
| [`radar_stability.py`](radar_stability.py) | accuracy retained under input perturbation and precision, plus non-finite-gradient counts |
| [`validation_suite.ipynb`](validation_suite.ipynb) | fifteen checks: parity, timing, parameter counts, independent correctness, failure modes |

## Datasets

All are public and none are redistributed here. Each notebook states the download step.

| Dataset | Shape after loading | Task |
|---|---|---|
| HDM05 | `(2086, 93, 93)` | 117 skeleton actions |
| Radar | `(3000, 40, 99)` after real embedding | 3 classes, complex returns |
| SEED | `(675, 5, 62, 62)` per-band connectivity | 3 emotions |
| DEAP | `(1280, 32, 1920)` | binary valence |
| Psychiatric EEG | `(212, 6, 19, 19)` band coherence | binary |

## Protocol notes

**Subject-independent splits** for SEED and DEAP. A within-subject split leaks identity through the
covariance structure and inflates every row by roughly the same amount, which hides exactly the
differences an ablation is meant to expose. `validation_suite.ipynb` asserts that no participant
appears on both sides.

**More than one manifold channel** in every configuration. With a single channel the pooling layer
is the identity and the two barycentre ablations would silently equal the full model — the easiest
way to produce an ablation table that looks complete and means nothing.

**Seed variance is reported.** `validation_suite.ipynb` checks whether any ablation gap exceeds the
seed standard deviation. If none does, there is no ablation result, and it is better to find that
here than in review.

## Independent verification

The Fréchet mean and affine-invariant distance are checked against `pyriemann`, which is an
independent implementation of the same mathematics:

```
pyriemann  mean 4.651e-10   distance 0.000e+00
```

The library's own test suite proves internal consistency; this proves agreement with the field.
