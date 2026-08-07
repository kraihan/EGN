"""Stability under input perturbation and precision -- Radar.

Fills Table `tab:stability`. Five settings, each trained ``N_RUNS`` times from
scratch on clean data, then evaluated on inputs perturbed at three noise levels.

Two numbers per cell:

*accuracy retained* = accuracy on perturbed test inputs / accuracy on clean test
inputs, in percent. A ratio rather than an absolute accuracy, because the five
settings do not have the same clean accuracy and the question here is
sensitivity, not performance.

*NaN runs* = how many of the ``N_RUNS`` training runs produced at least one
non-finite gradient. Non-finite steps are **counted and skipped** rather than
allowed to poison the weights, so a fragile setting still yields a measurable
accuracy row instead of a column of zeros. That is a deliberate choice: it makes
the failure visible in one column instead of destroying three others.

The two baselines are re-implementations from the published formulas, written
here rather than vendored, so this file carries no third-party code. Both are
heads on the *same* trunk as EGN, so the comparison isolates the readout
geometry.

Usage
-----
    python radar_stability.py                      # full table
    N_RUNS=3 EPOCHS=10 python radar_stability.py   # quick pass
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

import egn
from egn import EGN, build_criterion, build_optimizer
from egn.functional import invsqrtm, logm, powm, sym
from egn.manifolds import SPD, ManifoldParameter

# --------------------------------------------------------------------- config
RADAR_DIR = os.environ.get("RADAR_DIR", "/kaggle/working/data/radar_npy/radar")
RESULTS_PATH = "/kaggle/working/radar_stability.json"
N_RUNS = int(os.environ.get("N_RUNS", 10))        # the "out of XXXX" in the caption
EPOCHS = int(os.environ.get("EPOCHS", 30))
BATCH = 128
LR = 5e-3
EPSILONS = (1e-3, 1e-2, 1e-1)
TRAIN_FRACTION = 0.7

# Trunk shared by every setting, so only the readout / precision / guard varies.
TRUNK = dict(
    input_kind="signal",
    branches=4,
    channels=4,
    dims=[40, 24, 12],
    pool="frechet",
    dropout=0.1,
    ridge=1e-3,
)


def device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda").add_(1).cpu()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


DEVICE = device()


# ---------------------------------------------------------------------- data
def load_radar(root: str = RADAR_DIR, window: int = 20, hop: int = 10):
    """Complex returns -> real embedded windows, ``(N, 2*window, n_windows)``."""
    files = sorted(f for f in os.listdir(root) if f.endswith(".npy"))
    X, y = [], []
    for f in files:
        m = re.search(r"_(\d+)\.npy$", f) or re.search(r"(\d+)\.npy$", f)
        z = np.load(os.path.join(root, f)).ravel()
        W = np.stack([z[s:s + window] for s in range(0, len(z) - window + 1, hop)], axis=1)
        X.append(np.concatenate([W.real, W.imag], axis=0))
        y.append(int(m.group(1)))
    return np.stack(X).astype(np.float32), np.asarray(y)


def stratified_split(y, fraction=TRAIN_FRACTION, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cut = max(int(round(len(idx) * fraction)), 1)
        tr.append(idx[:cut]); te.append(idx[cut:])
    return np.concatenate(tr), np.concatenate(te)


def perturb(X: torch.Tensor, eps: float, generator: torch.Generator) -> torch.Tensor:
    """Additive Gaussian noise at relative RMS level ``eps``.

    Scaled per sample by that sample's own RMS, so ``eps`` means the same thing
    for a loud return and a quiet one. This is measurement noise on the raw
    signal, not a perturbation of the covariance: perturbing the descriptor
    directly would let a method look stable simply by being insensitive to the
    part of the input the descriptor discards.
    """
    if eps <= 0:
        return X
    rms = X.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
    noise = torch.randn(X.shape, dtype=X.dtype, device=X.device, generator=generator)
    return X + eps * rms * noise


# ------------------------------------------------------------------ baselines
class LCMHead(nn.Module):
    r"""SPD MLR under the :math:`\theta`-deformed Log-Cholesky metric.

    The Log-Cholesky map -- strictly lower triangle of the Cholesky factor,
    stacked with the log of its diagonal -- is an isometry onto a Euclidean
    space, so a multinomial logistic regression under this metric *is* a linear
    layer on that vectorisation. The deformation is the power map
    :math:`S \mapsto S^{\theta}` applied first; the customary :math:`1/\theta`
    factor is a fixed scale that the linear layer absorbs, so it is omitted.

    Reference: Chen et al., "RMLR: Extending Multinomial Logistic Regression into
    General Geometries" / the SPD MLR line of work.
    """

    def __init__(self, dim: int, num_classes: int, theta: float = 0.5):
        super().__init__()
        self.dim, self.theta = int(dim), float(theta)
        self.linear = nn.Linear(dim * (dim + 1) // 2, num_classes)
        idx = torch.tril_indices(dim, dim, offset=-1)
        self.register_buffer("tril_idx", idx)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        D = powm(S, self.theta)
        L = torch.linalg.cholesky(sym(D))
        strict = L[..., self.tril_idx[0], self.tril_idx[1]]
        logdiag = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12).log()
        return self.linear(torch.cat([strict, logdiag], dim=-1))


class GyroAIMHead(nn.Module):
    r"""Gyro-structured MLR under the affine-invariant metric.

    The signed margin to the hyperplane through :math:`P` with normal :math:`A`
    is :math:`\langle \mathrm{Log}_P(S), A\rangle_P / \|A\|_P`. Whitening by
    :math:`P^{-1/2}` turns the metric inner product into a Frobenius one, so the
    logit reduces to
    :math:`\langle \log(P^{-1/2} S P^{-1/2}), \tilde A\rangle_F / \|\tilde A\|_F`
    with :math:`\tilde A` symmetric and free.

    Reference: Nguyen, "Building neural networks on matrix manifolds: a
    gyrovector space approach", and the gyro-SPD MLR that follows from it.
    """

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        eye = torch.eye(dim).expand(num_classes, dim, dim).clone()
        self.points = ManifoldParameter(eye, SPD())
        self.normals = nn.Parameter(torch.randn(num_classes, dim, dim) / math.sqrt(dim))

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        inv = invsqrtm(self.points).unsqueeze(0)          # (1, C, n, n)
        V = logm(sym(inv @ S.unsqueeze(1) @ inv))          # (B, C, n, n)
        A = sym(self.normals).unsqueeze(0)
        num = (V * A).sum(dim=(-2, -1))
        den = A.flatten(-2).norm(dim=-1).clamp_min(1e-12)
        return num / den


# --------------------------------------------------------------------- models
SETTINGS = {
    "EGN, float64 (default)": dict(dtype=torch.float64, spectral="float64", guard=True, head="geodesic"),
    "EGN, float32":           dict(dtype=torch.float32, spectral="float32", guard=True, head="geodesic"),
    "EGN, no Loewner guard":  dict(dtype=torch.float64, spectral="float64", guard=False, head="geodesic"),
    "SPD MLR (0.5)-LCM":      dict(dtype=torch.float64, spectral="float64", guard=True, head="lcm"),
    "Gyro-AIM MLR":           dict(dtype=torch.float64, spectral="float64", guard=True, head="gyro"),
}


def build_model(setting: dict, example: torch.Tensor, num_classes: int) -> EGN:
    head = setting["head"]
    model = EGN(num_classes=num_classes, head="geodesic" if head == "geodesic" else "tangent",
                **TRUNK)
    model.build_from_example(example, num_classes)
    if head == "lcm":
        model.head = LCMHead(model.feature_dim, num_classes, theta=0.5)
    elif head == "gyro":
        model.head = GyroAIMHead(model.feature_dim, num_classes)
    return model.to(dtype=setting["dtype"], device=DEVICE)


# ------------------------------------------------------------------- training
def train_once(setting, Xtr, ytr, seed):
    """Train from scratch; return the model and the non-finite gradient count."""
    egn.seed_everything(seed)
    dtype = setting["dtype"]
    Xtr = Xtr.to(DEVICE, dtype)
    ytr = ytr.to(DEVICE)

    model = build_model(setting, Xtr[:2], int(ytr.max().item()) + 1)
    opt = build_optimizer(model, lr=LR, kind="adam")
    criterion = build_criterion(separation=0.01 if setting["head"] == "geodesic" else 0.0)

    n = len(Xtr)
    nonfinite = 0
    with egn.numeric_context(spectral_dtype=setting["spectral"], loewner_guard=setting["guard"]):
        model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n, device=DEVICE)
            for s in range(0, n - BATCH + 1, BATCH):
                idx = perm[s:s + BATCH]
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(Xtr[idx]), ytr[idx], model)
                if not torch.isfinite(loss):
                    nonfinite += 1
                    continue
                loss.backward()
                bad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if bad:
                    # count it and skip; letting NaN into the weights would make
                    # the accuracy columns unreadable for this row
                    nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
    return model, nonfinite


@torch.no_grad()
def accuracy(model, setting, X, y):
    model.eval()
    correct = 0
    with egn.numeric_context(spectral_dtype=setting["spectral"], loewner_guard=setting["guard"]):
        for s in range(0, len(X), 256):
            xb = X[s:s + 256].to(DEVICE, setting["dtype"])
            out = model(xb)
            out = torch.nan_to_num(out, nan=-1e9)
            correct += (out.argmax(-1).cpu() == y[s:s + 256]).sum().item()
    return correct / len(X)


# ----------------------------------------------------------------------- main
def main():
    X, y = load_radar()
    print(f"radar {X.shape} | classes {np.unique(y).tolist()} | device {DEVICE}")
    X = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()

    results = {}
    if os.path.exists(RESULTS_PATH):
        results = json.load(open(RESULTS_PATH))

    for name, setting in SETTINGS.items():
        rows = results.setdefault(name, {"retention": {str(e): [] for e in EPSILONS},
                                         "clean": [], "nan_runs": 0, "runs": 0})
        while rows["runs"] < N_RUNS:
            seed = 1000 + rows["runs"]
            tr, te = stratified_split(y, seed=seed)
            t0 = time.perf_counter()
            model, nonfinite = train_once(setting, X[tr], y_t[tr], seed)

            gen = torch.Generator().manual_seed(seed)
            clean = accuracy(model, setting, X[te], y_t[te])
            for eps in EPSILONS:
                Xp = perturb(X[te], eps, gen)
                acc = accuracy(model, setting, Xp, y_t[te])
                rows["retention"][str(eps)].append(
                    100.0 * acc / clean if clean > 0 else float("nan")
                )
            rows["clean"].append(clean)
            rows["nan_runs"] += int(nonfinite > 0)
            rows["runs"] += 1
            json.dump(results, open(RESULTS_PATH, "w"), indent=1)
            print(f"  {name:<24} run {rows['runs']}/{N_RUNS}  clean {clean * 100:5.2f}  "
                  f"nonfinite steps {nonfinite:4d}  ({time.perf_counter() - t0:5.1f}s)")

    print_table(results)


def print_table(results):
    print(f"\n{'Setting':<24}" + "".join(f"{f'eps={e:g}':>12}" for e in EPSILONS)
          + f"{'NaN runs':>10}{'clean':>9}")
    for name in SETTINGS:
        r = results.get(name)
        if not r or not r["runs"]:
            continue
        cells = [f"{np.nanmean(r['retention'][str(e)]):>11.2f}" for e in EPSILONS]
        print(f"{name:<24}" + "".join(cells)
              + f"{r['nan_runs']:>6}/{r['runs']:<4}{np.mean(r['clean']) * 100:>8.2f}")

    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Setting} & $\varepsilon{=}10^{-3}$ & $10^{-2}$ & $10^{-1}$ "
             r"& \textbf{NaN runs} \\", r"\midrule"]
    labels = {
        "EGN, float64 (default)": "EGN, float64 (default)",
        "EGN, float32": "EGN, float32",
        "EGN, no Loewner guard": "EGN, no L\\\"owner guard",
        "SPD MLR (0.5)-LCM": "SPD MLR $(0.5)$-LCM",
        "Gyro-AIM MLR": "Gyro-AIM MLR",
    }
    for name in SETTINGS:
        r = results.get(name)
        if not r or not r["runs"]:
            continue
        vals = [f"{np.nanmean(r['retention'][str(e)]):.2f}" for e in EPSILONS]
        lines.append(f"{labels[name]:<24}& " + " & ".join(vals)
                     + f" & {r['nan_runs']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    print("\n" + latex)
    print(f"\ncaption: out of {N_RUNS} runs per setting")
    with open("/kaggle/working/stability_table.tex", "w") as fh:
        fh.write(latex)


if __name__ == "__main__":
    main()
