"""Multi-GPU training.

    torchrun --nproc_per_node=4 examples/train_ddp.py --epochs 50

Nothing in this file is DDP-specific except the launcher: EGNClassifier detects
the process group that torchrun sets up, wraps the model, installs a
DistributedSampler and reduces metrics across ranks. Running it without torchrun
trains on one device.

The two settings that matter for scaling are the batch size -- these kernels are
small, so the device stays latency bound until the batch is large -- and
num_workers, since covariance formation happens on the CPU side of the loader.
"""

import argparse

import numpy as np

from egn import EGNClassifier
from egn.utils import cleanup_distributed, is_main_process, world_size


def synthetic(n, classes, channels, samples, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    mixing = rng.normal(size=(classes, channels, channels))
    X = np.stack([mixing[y[i]] @ rng.normal(size=(channels, samples)) for i in range(n)])
    return X.astype(np.float32), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the dense parts")
    args = ap.parse_args()

    X, y = synthetic(8192, classes=4, channels=32, samples=256)
    split = len(y) * 4 // 5

    clf = EGNClassifier(
        depth=2,
        branches=2,
        channels=4,
        mix=True,
        head="tangent",
        pool="logeuclid",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        amp=args.amp,
        scheduler="cosine",
        verbose=1,
    )
    clf.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))

    if is_main_process():
        print(f"world size {world_size()}  |  test accuracy {clf.score(X[split:], y[split:]):.4f}")
        clf.save("egn_ddp.pt")
    cleanup_distributed()


if __name__ == "__main__":
    main()
