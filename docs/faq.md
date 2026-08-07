# FAQ and error messages

## "cannot infer an SPD convention from shape (N, D)"

A rank-2 input has no second moment. EGN needs a signal `(B, C, T)`, a sequence `(B, T, D)`, a
feature map `(B, C, H, W)`, or a covariance `(B, n, n)`.

If each row of your table is one sample, you need a way to form a matrix from it. Two options that
work: group the features into a `(channels, features_per_channel)` matrix and take its covariance;
or, if the columns are already pairwise quantities — coherences, correlations — reassemble them into
the matrix they came from. The second is much better when it is available.

## "input contains NaN or Inf"

Raised at the boundary rather than three layers down. Without the check this surfaces as
`_LinAlgError: the algorithm failed to converge` from inside `eigh`, which tells you nothing about
the cause.

Check for missing values in the source data. `ToSPD(check_input=True)` validates every batch; the
default validates only the first batch in `fit`, because a finiteness test forces a device
synchronisation.

## "CUDA error: no kernel image is available for execution on the device"

Your GPU's compute capability is not in the installed PyTorch build. Common on Kaggle when a P100
(sm_60) is allocated against a wheel built for sm_70 and up.

With `device="auto"` (the default) EGN runs one small kernel first, warns, and falls back to CPU.
If you passed `device="cuda"` explicitly it is honoured as given, and you get the real error.
Install a PyTorch build matching your card, or switch accelerators.

## "Expected more than 1 value per channel when training"

A batch of one reached a `BatchNorm1d` in training mode. Either call `.eval()` before single-sample
inference, or pass `batchnorm=False`. Note `EGNClassifier` sets `drop_last=True` when batch-norm is
on and the dataset is larger than one batch, so this normally only appears in manual loops.

## Why is my GPU slower than my CPU?

Almost always precision. Check `egn.config.spectral_dtype` — `"auto"` gives `float32` on CUDA, which
is what you want. If you set `"float64"`, a consumer GPU runs it at 1/32 speed.

If precision is right, you are probably latency bound: these kernels are small, and the device stays
underused until the batch is large. Run `python -m egn.benchmark` and look at whether throughput
changes with batch size.

## Which head should I use?

`tangent` unless you have a reason. It is the exact linear classifier in the tangent space at a
learnable reference and its cost does not grow with the class count.

Use `geodesic` when you want prototypes you can inspect as points on the manifold, or when classes
are naturally described as regions rather than half-spaces. On a 117-class problem the prototypes
are two thirds of the model's parameters, so budget for it.

## Which pooling?

`logeuclid` unless the channel spread is large. It is closed-form, and it equals the Fréchet mean
exactly when the inputs commute. `frechet` costs `iters` decompositions per sample.

`arithmetic` exists for ablation and should not be used in a real model.

## My ablation shows no difference between pooling modes

Check your channel count. With one manifold channel there is nothing to pool and every mode is the
identity. Set `branches > 1` or `channels > 1`.

## Does it work on complex-valued data?

Yes, through the real embedding. A complex vector `z` maps to `[Re z; Im z]`, and the covariance of
the embedded vectors is real SPD of twice the size. This is the standard isometry between Hermitian
and real SPD problems. `experiments/` has a worked radar example.

## Can I use EGN as a feature extractor?

`clf.transform(X)` returns pooled SPD descriptors. `model.features(x)` does the same inside a larger
network.

## How do I reproduce a number in double precision?

```python
with egn.numeric_context(spectral_dtype="float64"):
    ...
```

Or pass `spectral_dtype="float64"` to `EGNClassifier`.
