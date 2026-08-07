"""Test suite.

Split in two halves. The first asserts the *geometry* -- these are the claims
the method rests on, and each one is stated as an identity that must hold to
numerical precision, in float64, including the counter-examples that motivated
the current formulation. The second asserts the *engineering*: shapes, manifold
constraints surviving optimisation, save/load round-trips, and that a fit on
separable data actually separates it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import egn
from egn import functional as F
from egn import geometry as G
from egn.manifolds import SPD, ManifoldParameter, Stiefel
from egn.nn import (
    BiMap,
    EGNBlock,
    GeodesicDropout,
    GeodesicPrototypeHead,
    GeometricBias,
    RiemannianPool,
    SPDBatchNorm,
    SpectralActivation,
    TangentHead,
    ToSPD,
)

torch.manual_seed(0)
DT = torch.float64


@pytest.fixture(autouse=True)
def _f64():
    with egn.numeric_context(spectral_dtype="float64"):
        yield


def spd(*shape, cond=8.0):
    return G.random_spd(shape, condition=cond, dtype=DT)


def close(a, b, tol=1e-8):
    return torch.allclose(a, b, atol=tol, rtol=tol)


# ---------------------------------------------------------------- geometry
def test_spectral_roundtrip():
    S = spd(6, 5, 5)
    assert close(F.expm(F.logm(S)), S)
    assert close(F.sqrtm(S) @ F.sqrtm(S), S)
    A, B = F.sqrtm_pair(S)
    assert close(A @ B, torch.eye(5, dtype=DT).expand_as(S))


def test_exp_log_are_inverse():
    S, P = spd(4, 5, 5), spd(4, 5, 5)
    V = G.riemannian_log(S, P)
    assert close(G.riemannian_exp(S, V), P, 1e-7)


def test_distance_is_affine_invariant():
    S, P = spd(3, 6, 6), spd(3, 6, 6)
    A = torch.randn(6, 6, dtype=DT)
    lhs = G.distance(S, P)
    rhs = G.distance(A @ S @ A.T, A @ P @ A.T)
    assert close(lhs, rhs, 1e-6)


def test_distance_matches_frobenius_of_log():
    S, P = spd(4, 5, 5), spd(4, 5, 5)
    inv = F.invsqrtm(S)
    ref = torch.linalg.matrix_norm(F.logm(F.sym(inv @ P @ inv)))
    assert close(G.distance(S, P), ref, 1e-8)


def test_geodesic_endpoints_and_speed():
    A, B = spd(3, 5, 5), spd(3, 5, 5)
    assert close(G.geodesic(A, B, 0.0), A, 1e-7)
    assert close(G.geodesic(A, B, 1.0), B, 1e-7)
    for t in (0.25, 0.5, 0.9):
        assert close(G.distance(A, G.geodesic(A, B, t)), t * G.distance(A, B), 1e-7)


def test_convex_combination_is_not_a_geodesic():
    """The reason interpolation is geodesic and not convex."""
    A, B = spd(1, 5, 5), spd(1, 5, 5)
    t = 0.5
    convex = (1 - t) * A + t * B
    assert not close(convex, G.geodesic(A, B, t), 1e-3)


def test_frechet_mean_is_a_fixed_point():
    S = spd(1, 12, 6, 6)
    M = G.frechet_mean(S, dim=1, iters=40)
    drift = G.riemannian_log(M.unsqueeze(1), S).mean(1)
    assert drift.abs().max() < 1e-6


def test_frechet_mean_is_affine_equivariant():
    S = spd(1, 8, 5, 5)
    A = torch.randn(5, 5, dtype=DT)
    M = G.frechet_mean(S, dim=1, iters=40)
    M2 = G.frechet_mean(A @ S @ A.T, dim=1, iters=40)
    assert close(A @ M @ A.T, M2, 1e-5)


def test_means_agree_when_inputs_commute():
    Q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=DT))
    lam = torch.rand(1, 7, 5, dtype=DT) + 0.5
    S = Q @ torch.diag_embed(lam) @ Q.T
    assert close(
        G.frechet_mean(S, dim=1, iters=30), G.log_euclidean_mean(S, dim=1), 1e-7
    )


def test_egrad2rgrad_reproduces_minus_log():
    """``grad_S (1/2 d^2(S, P)) = -Log_S(P)``."""
    S, P = spd(2, 5, 5), spd(2, 5, 5)
    S = S.clone().requires_grad_(True)
    loss = 0.5 * G.squared_distance(S, P).sum()
    (egrad,) = torch.autograd.grad(loss, S)
    assert close(G.egrad2rgrad_spd(S.detach(), egrad), -G.riemannian_log(S.detach(), P), 1e-6)


@pytest.mark.parametrize("op", [F.logm, F.expm, F.sqrtm, F.invsqrtm])
def test_gradcheck_spectral(op):
    S = spd(2, 4, 4).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda X: op(F.sym(X)).sum(), (S,), eps=1e-6, atol=1e-5)


def test_gradcheck_with_repeated_eigenvalues():
    """The Loewner backward must survive an exact eigenvalue tie."""
    base = torch.eye(4, dtype=DT) * 2.0
    S = (base + 1e-9 * torch.randn(4, 4, dtype=DT)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda X: F.logm(F.sym(X)).sum(), (S,), eps=1e-6, atol=1e-4)


def test_prototype_gradient_matches_closed_form():
    """``grad_P L = (2/(tau B)) sum_i (p - 1[y=c]) Log_P(S_i)``."""
    torch.manual_seed(3)
    B, C, n = 6, 3, 4
    head = GeodesicPrototypeHead(n, C, temperature=1.0).to(DT)
    S = spd(B, n, n)
    y = torch.randint(0, C, (B,))
    logits = head(S)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    P = head.prototypes.detach()
    p = torch.softmax(logits.detach(), -1)
    onehot = torch.nn.functional.one_hot(y, C).to(DT)
    coeff = (p - onehot) / B  # (B, C)
    logs = G.riemannian_log(P.unsqueeze(0), S.unsqueeze(1))  # (B, C, n, n)
    expected = 2.0 * (coeff[..., None, None] * logs).sum(0)

    got = G.egrad2rgrad_spd(P, head.prototypes.grad)
    assert close(got, expected, 1e-6)


# ------------------------------------------------------------------ layers
def assert_spd(S, tol=1e-9):
    assert torch.isfinite(S).all()
    lam = torch.linalg.eigvalsh(F.sym(S.double()))
    assert lam.min() > 0, f"min eigenvalue {lam.min().item()}"
    assert (S - S.transpose(-1, -2)).abs().max() < 1e-6


@pytest.mark.parametrize("mix", [False, True])
def test_bimap_outputs_spd(mix):
    layer = BiMap(8, 4, in_channels=2, out_channels=3 if mix else 2, mix=mix).to(DT)
    out = layer(spd(5, 2, 8, 8))
    assert out.shape[-1] == 4
    assert_spd(out)


def test_bimap_weight_is_orthonormal():
    layer = BiMap(9, 5, out_channels=3)
    W = layer.weight
    eye = torch.eye(5).expand(3, 5, 5)
    assert close(W.transpose(-1, -2) @ W, eye, 1e-5)


def test_bimap_reparameterisation_identity():
    """Congruence of the input is absorbed by ``W -> Q^T W``, not by the output.

    This is the correct statement of the layer's symmetry. The stronger claim --
    that the output is itself congruent by ``Q`` -- is false for a rectangular
    frame, and the second half of this test is the counter-example.
    """
    layer = BiMap(6, 3).to(DT)
    S = spd(4, 1, 6, 6)
    Q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=DT))
    W = layer.weight.detach()

    congruent_input = F.sym(Q @ S @ Q.transpose(-1, -2))
    lhs = F.sym(W.transpose(-1, -2) @ congruent_input @ W)
    Wq = Q.transpose(-1, -2) @ W
    rhs = F.sym(Wq.transpose(-1, -2) @ S @ Wq)
    assert close(lhs, rhs, 1e-9)

    # and the strong claim does not hold: no 3x3 congruence relates the two
    plain = layer(S)
    assert not close(lhs, plain, 1e-3)


def test_layers_preserve_the_manifold():
    S = spd(6, 2, 8, 8)
    for layer in [
        GeometricBias(8, 2),
        SpectralActivation("rect"),
        SpectralActivation("power"),
        SPDBatchNorm(8, 2),
        GeodesicDropout(0.5),
    ]:
        assert_spd(layer.to(DT)(S))


def test_block_and_pool_shapes():
    block = EGNBlock(10, 6, in_channels=1, out_channels=4, mix=True).to(DT)
    out = block(spd(7, 1, 10, 10))
    assert out.shape == (7, 4, 6, 6)
    assert_spd(out)
    pooled = RiemannianPool("frechet", iters=3)(out)
    assert pooled.shape == (7, 6, 6)
    assert_spd(pooled)


def test_batchnorm_running_stats_used_in_eval():
    bn = SPDBatchNorm(5, 1).to(DT)
    S = spd(16, 1, 5, 5)
    bn.train()
    bn(S)
    bn.eval()
    a, b = bn(S[:4]), bn(S[:4])
    assert close(a, b)


@pytest.mark.parametrize(
    "shape,kind,expect_n",
    [
        ((8, 6, 6), "auto", 6),
        ((8, 3, 6, 6), "auto", 6),
        ((8, 12, 100), "signal", 12),
        ((8, 100, 12), "sequence", 12),
        ((8, 5, 9, 9), "image", 5),
    ],
)
def test_tospd_conventions(shape, kind, expect_n):
    x = spd(*shape) if shape[-1] == shape[-2] and kind in ("auto", "spd") else torch.randn(*shape, dtype=DT)
    out = ToSPD(kind=kind)(x)
    assert out.shape[-1] == expect_n and out.dim() == 4
    assert_spd(out)


def test_tospd_branches():
    out = ToSPD(kind="signal", branches=4)(torch.randn(8, 12, 200, dtype=DT))
    assert out.shape == (8, 4, 12, 12)


def test_rank_two_input_is_rejected():
    with pytest.raises(ValueError, match="second moment"):
        ToSPD()(torch.randn(8, 12))


# --------------------------------------------------------------- optimisation
def test_stiefel_and_spd_parameters_stay_on_their_manifolds():
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        BiMap(8, 5, out_channels=2), RiemannianPool(), GeodesicPrototypeHead(5, 3)
    )
    opt = egn.build_optimizer(model, lr=0.5)
    S = G.random_spd((32, 1, 8, 8), condition=6.0)
    y = torch.randint(0, 3, (32,))
    for _ in range(25):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(S), y).backward()
        opt.step()

    W = model[0].weight.detach().double()
    eye = torch.eye(5).double().expand_as(W.transpose(-1, -2) @ W)
    assert (W.transpose(-1, -2) @ W - eye).abs().max() < 1e-5
    lam = torch.linalg.eigvalsh(model[2].prototypes.detach().double())
    assert lam.min() > 0


def test_optimizer_decreases_the_loss():
    torch.manual_seed(2)
    model = torch.nn.Sequential(BiMap(6, 4), RiemannianPool(), TangentHead(4, 2))
    opt = egn.build_optimizer(model, lr=1e-2)
    S = G.random_spd((64, 1, 6, 6))
    y = torch.randint(0, 2, (64,))
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(S), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# --------------------------------------------------------------------- model
def test_model_lazy_build_and_forward():
    model = egn.EGN(num_classes=4)
    x = torch.randn(16, 9, 128)
    assert not model.is_built
    out = model(x)
    assert model.is_built and out.shape == (16, 4)


@pytest.mark.parametrize("factory", [egn.egn_tiny, egn.egn_small, egn.egn_base])
def test_factories_run(factory):
    model = factory(num_classes=3)
    out = model(torch.randn(12, 16, 64))
    assert out.shape == (12, 3)
    assert torch.isfinite(out).all()


def test_same_model_handles_different_data_shapes():
    for x in [
        torch.randn(8, 20, 300),          # signal
        torch.randn(8, 16, 16),           # covariance
        torch.randn(8, 4, 12, 12),        # multi-branch SPD
        torch.randn(8, 10, 7, 7),         # feature map
    ]:
        out = egn.EGN(num_classes=5)(x)
        assert out.shape == (8, 5)


def test_features_are_spd():
    model = egn.egn_small(num_classes=3)
    assert_spd(model.features(torch.randn(10, 12, 64)))


# ---------------------------------------------------------------- classifier
def _separable(n=240, c=3, ch=8, t=96, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    mix = rng.normal(size=(c, ch, ch))
    X = np.stack([mix[y[i]] @ rng.normal(size=(ch, t)) for i in range(n)])
    return X.astype(np.float32), y


def test_classifier_learns_and_round_trips(tmp_path):
    X, y = _separable()
    clf = egn.EGNClassifier(epochs=25, batch_size=64, lr=5e-3, depth=1, verbose=0)
    clf.fit(X, y, eval_set=(X, y))
    acc = clf.score(X, y)
    assert acc > 0.6, f"accuracy {acc}"
    assert clf.predict_proba(X).shape == (len(y), 3)

    path = tmp_path / "model.pt"
    clf.save(str(path))
    reloaded = egn.EGNClassifier.load(str(path))
    assert np.array_equal(reloaded.predict(X), clf.predict(X))


def test_classifier_accepts_string_labels():
    X, y = _separable(n=60, c=2)
    labels = np.array(["left", "right"])[y]
    clf = egn.EGNClassifier(epochs=2, verbose=0).fit(X, labels)
    assert set(clf.predict(X)).issubset({"left", "right"})


def test_classifier_geodesic_head():
    X, y = _separable(n=90, c=3)
    clf = egn.EGNClassifier(epochs=3, head="geodesic", pool="frechet", verbose=0).fit(X, y)
    assert clf.predict(X).shape == (90,)


def test_transform_gives_spd_descriptors():
    X, y = _separable(n=40, c=2)
    clf = egn.EGNClassifier(epochs=2, verbose=0).fit(X, y)
    feats = clf.transform(X)
    assert feats.shape[0] == 40 and feats.shape[-1] == feats.shape[-2]


# ---------------------------------------------------------------- ablations
def test_pool_modes_differ_and_stay_spd():
    S = spd(6, 5, 4, 4)
    means = {m: RiemannianPool(m, iters=20)(S) for m in ("frechet", "logeuclid", "arithmetic")}
    for M in means.values():
        assert_spd(M)
    # the arithmetic average inflates the spectrum relative to the barycentre
    assert torch.linalg.det(means["arithmetic"]).mean() > torch.linalg.det(means["frechet"]).mean()
    assert not close(means["arithmetic"], means["frechet"], 1e-3)


def test_unconstrained_bimap_leaves_the_stiefel_manifold():
    """The ablation must actually ablate: same init, different training."""
    torch.manual_seed(5)
    S = G.random_spd((64, 1, 8, 8))
    y = torch.randint(0, 2, (64,))
    errors = {}
    for constrained in (True, False):
        torch.manual_seed(5)
        model = torch.nn.Sequential(
            BiMap(8, 5, out_channels=2, constrained=constrained),
            RiemannianPool(),
            TangentHead(5, 2),
        )
        opt = egn.build_optimizer(model, lr=1e-2)
        for _ in range(30):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(S), y).backward()
            opt.step()
        W = model[0].weight.detach()
        eye = torch.eye(5).expand_as(W.transpose(-1, -2) @ W)
        errors[constrained] = (W.transpose(-1, -2) @ W - eye).abs().max().item()
    assert errors[True] < 1e-4 < errors[False]


def test_logeig_head_has_a_fixed_reference():
    from egn.manifolds import ManifoldParameter

    tangent = egn.EGN(num_classes=3, head="tangent", in_dim=8)
    logeig = egn.EGN(num_classes=3, head="logeig", in_dim=8)
    assert isinstance(tangent.head.reference, ManifoldParameter)
    assert not isinstance(logeig.head.reference, ManifoldParameter)


def test_separation_term_changes_the_loss():
    head = GeodesicPrototypeHead(5, 3).to(DT)
    logits, y = head(spd(8, 5, 5)), torch.randint(0, 3, (8,))
    plain = egn.build_criterion(separation=0.0)
    repel = egn.build_criterion(separation=0.1)

    class _M(torch.nn.Module):
        pass

    m = _M()
    m.head = head
    assert not torch.isclose(plain(logits, y, m), repel(logits, y, m))
