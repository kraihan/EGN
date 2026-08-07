"""``python -m egn.benchmark`` -- find out where the time actually goes.

Three measurements, in increasing order of usefulness:

1. ``eigh`` throughput per device and dtype. This is the number that explains a
   slow GPU. Consumer cards (GeForce, and the T4) execute float64 at 1/32 of
   their float32 rate, so a float64 network can easily be slower on GPU than on
   CPU. Datacentre cards with a 1:2 ratio do not have this problem.
2. Forward and forward+backward throughput of a real model, per configuration.
3. A sweep over batch size, which is the single most effective lever: the
   kernels are tiny, so the GPU is latency bound until the batch is large.

Example::

    python -m egn.benchmark --sizes 16 32 64 --batch 256 1024 4096
"""

from __future__ import annotations

import argparse
import time
from typing import List

import torch

from .config import config, set_performance_defaults
from .geometry import random_spd
from .models import EGN


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, reps: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / reps


def bench_eigh(devices: List[str], sizes: List[int], batch: int) -> None:
    print(f"\n== eigh throughput ({batch} matrices per call) ==")
    print(f"{'device':>8} {'dtype':>9} {'n':>5} {'ms/call':>10} {'matrices/s':>14}")
    for dev in devices:
        device = torch.device(dev)
        for dtype in (torch.float32, torch.float64):
            for n in sizes:
                X = random_spd((batch, n, n), dtype=dtype, device=device)
                try:
                    ms = _time(lambda: torch.linalg.eigh(X), device) * 1e3
                except RuntimeError as exc:  # pragma: no cover
                    print(f"{dev:>8} {str(dtype)[6:]:>9} {n:>5}   failed: {exc}")
                    continue
                print(
                    f"{dev:>8} {str(dtype)[6:]:>9} {n:>5} {ms:>10.2f} {batch / (ms / 1e3):>14,.0f}"
                )


def bench_model(devices: List[str], n: int, batch_sizes: List[int], num_classes: int = 4) -> None:
    configs = {
        "tangent/logeuclid": dict(head="tangent", pool="logeuclid", depth=2, branches=1),
        "geodesic/logeuclid": dict(head="geodesic", pool="logeuclid", depth=2, branches=1),
        "geodesic/frechet": dict(head="geodesic", pool="frechet", depth=2, branches=4, channels=4),
    }
    print(f"\n== model throughput (matrix size {n}) ==")
    print(f"{'device':>8} {'config':>20} {'batch':>7} {'fwd ms':>9} {'fwd+bwd ms':>12} {'samples/s':>12}")
    for dev in devices:
        device = torch.device(dev)
        for name, kw in configs.items():
            for bs in batch_sizes:
                model = EGN(num_classes=num_classes, input_kind="spd", **kw).to(device)
                X = random_spd((bs, n, n), dtype=torch.float32, device=device)
                y = torch.randint(0, num_classes, (bs,), device=device)
                model.build_from_example(X, num_classes)
                model.to(device)

                def fwd():
                    with torch.no_grad():
                        model(X)

                def fwd_bwd():
                    model.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(X), y)
                    loss.backward()

                try:
                    f = _time(fwd, device, reps=5) * 1e3
                    fb = _time(fwd_bwd, device, reps=5) * 1e3
                except RuntimeError as exc:  # pragma: no cover
                    print(f"{dev:>8} {name:>20} {bs:>7}   failed: {exc}")
                    continue
                print(
                    f"{dev:>8} {name:>20} {bs:>7} {f:>9.2f} {fb:>12.2f} {bs / (fb / 1e3):>12,.0f}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="EGN performance benchmark")
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--batch", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--eigh-batch", type=int, default=4096)
    ap.add_argument("--model-size", type=int, default=32)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    set_performance_defaults()
    devices = ["cpu"]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.append("cuda")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__} | spectral dtype policy: {config.spectral_dtype}")

    bench_eigh(devices, args.sizes, args.eigh_batch)
    bench_model(devices, args.model_size, args.batch)
    print(
        "\nIf the float64 rows are far slower than float32 on CUDA, leave "
        "config.spectral_dtype at 'auto' (float32 on GPU). If throughput is flat "
        "in the batch size, the GPU is latency bound: raise the batch."
    )


if __name__ == "__main__":
    main()
