# Performance

## The precision policy

Everything that trades speed against numerical margin is decided once, in `egn.config`.

```python
egn.config.spectral_dtype   # "auto" (default) | "float32" | "float64"
egn.config.sync_free        # True — no host synchronisation in the forward pass
egn.config.eig_chunk        # 0 — cap matrices per eigh call when memory is tight
egn.config.loewner_guard    # True — ablation switch, see geometry.md
```

`"auto"` means **`float32` on CUDA, `float64` on CPU**. This is the single most important setting.
Consumer and inference-class GPUs execute double precision at 1/32 of their single-precision rate; a
`float64` SPD network on a T4 is genuinely slower than the same network on a decent CPU. Datacentre
cards with a 1:2 ratio do not have this problem, so measure before assuming.

Set `float64` explicitly when reproducing a derivation:

```python
with egn.numeric_context(spectral_dtype="float64"):
    ...
```

## Why the forward pass has no `.item()`

An SPD network is a long chain of short kernel launches. A single `.item()`, `.cpu()` or
`bool(tensor)` drains the CUDA queue and costs more than the arithmetic it guards. Consequences for
the implementation:

- the Fréchet mean runs a **fixed iteration budget** (`pool_iters`, default 5) instead of polling a
  residual — the arithmetic-mean seed is already a first-order approximation, so three to five steps
  put the iterate within numerical noise of the fixed point;
- `SpectralActivation` compares against a tensor, never a Python float read from the device;
- training statistics accumulate on-device and are read once per epoch, not per step;
- input validation lives at the boundary (`fit`, `ToSPD(check_input=True)`), not in the layer.

## Where the time actually goes

```bash
python -m egn.benchmark --sizes 16 32 64 --batch 256 1024 4096
```

The output separates `eigh` throughput per device and dtype from end-to-end model throughput. Two
readings to look for:

**If the `float64` rows are far slower than `float32` on CUDA**, leave `spectral_dtype` at `"auto"`.

**If throughput is flat in the batch size**, the GPU is latency bound — the kernels are too small to
fill the device. Raise the batch before changing anything else.

**Honest caveat.** These are small-matrix, decomposition-heavy workloads. A GPU wins decisively at
large batch and `float32`; at batch 32 with 8×8 matrices it may still lose to a CPU. That is a
property of the operation, not of this implementation, and no amount of engineering changes it.

## Cost of the architectural choices

| Choice | Cost per forward |
|---|---|
| `pool="logeuclid"` | one decomposition per matrix |
| `pool="frechet"` | `iters` × that |
| `head="tangent"` | `B` decompositions, independent of class count |
| `head="geodesic"` | `B × C` eigenvalue computations |
| `SPDBatchNorm` | one decomposition per matrix, plus the batch mean |

With many classes, the geodesic head dominates. `TangentHead` is not a weaker model — it is the
exact linear classifier in the tangent space at a learnable reference point.

## Automatic mixed precision

`amp=True` is supported and defaults to `bfloat16`. The spectral operators **always upcast to at
least `float32` internally and emit `float32`**, so autocast accelerates the dense matmuls and the
linear head while leaving every eigendecomposition at full precision.

`float16` is accepted with a `GradScaler`, but its dynamic range is a poor fit for spectra spanning
several orders of magnitude, and divergence is still possible. `bfloat16` is the safe choice.

## Distributed training

```bash
torchrun --nproc_per_node=4 examples/04_train_ddp.py
```

`EGNClassifier` detects the process group, wraps the model in `DistributedDataParallel`, installs a
`DistributedSampler` and reduces metrics across ranks. The model is materialised **before** wrapping,
so the replicas have parameters to broadcast — if you build a model manually, call
`build_from_example` before `DistributedDataParallel`.

DDP needs no special handling for the Riemannian optimiser. DDP all-reduces the *Euclidean*
gradients in its backward hook; every rank then applies the same deterministic `egrad2rgrad` and
retraction to identical parameters, so replicas stay identical without extra synchronisation.

`data_parallel=True` uses `nn.DataParallel` for a quick single-process run. **Not recommended**: it
re-scatters the model every step, serialises the Python side, and discards replica buffer updates,
which corrupts `SPDBatchNorm`'s running Fréchet mean.
