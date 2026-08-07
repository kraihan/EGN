# Benchmarks

```bash
python -m egn.benchmark --sizes 16 32 64 --batch 256 1024 4096
```

Three measurements, in increasing order of usefulness:

1. **`eigh` throughput per device and dtype.** This is the number that explains a slow GPU.
2. **Forward and forward+backward throughput** of a real model, per configuration.
3. **A sweep over batch size** — the most effective lever, because these kernels are small and the
   GPU stays latency bound until the batch is large.

## Reading the output

**`float64` far slower than `float32` on CUDA** → leave `config.spectral_dtype` at `"auto"`. Expect
this on consumer and inference-class cards, which run double precision at 1/32 rate. Datacentre
cards with a 1:2 ratio show a much smaller gap.

**Throughput flat in batch size** → latency bound. Raise the batch before changing anything else.

**GPU slower than CPU at small batch** → expected, and not a bug. Small-matrix decomposition
workloads do not fill a GPU. The crossover point is hardware-specific; find yours here rather than
assuming one.

## Reporting

When quoting numbers, state the card, the PyTorch version, the batch size, the matrix size and the
dtype policy. A speedup figure without those is not reproducible, and the spread across hardware for
this workload is large enough that it matters.
