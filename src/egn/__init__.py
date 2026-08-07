"""EGN -- Equivariant Geodesic Networks.

A universal classifier on the SPD manifold, packaged the way a convolutional
network is: layers in :mod:`egn.nn`, models in :mod:`egn.models`, one high-level
estimator in :class:`egn.EGNClassifier`, and no dataset-specific code anywhere.

    import numpy as np
    from egn import EGNClassifier

    X = np.random.randn(512, 22, 256)        # 512 trials, 22 channels, 256 samples
    y = np.random.randint(0, 4, 512)

    clf = EGNClassifier(epochs=20).fit(X, y)
    clf.score(X, y)

Every intermediate representation is an exact SPD matrix under the
affine-invariant metric; there is no projection step anywhere in the forward
pass, and constrained parameters are updated by retraction, so they cannot leave
their manifolds regardless of step size.
"""

from .config import Config, config, numeric_context, set_performance_defaults
from .classifier import EGNClassifier
from .losses import PrototypeCrossEntropy, build_criterion
from .models import EGN, default_dims, egn_base, egn_small, egn_tiny
from .optim import RiemannianAdam, RiemannianSGD, build_optimizer
from .utils.seed import seed_everything
from . import functional, geometry, manifolds, nn, data, engine, utils

__version__ = "0.2.2"

__all__ = [
    "Config",
    "EGN",
    "EGNClassifier",
    "PrototypeCrossEntropy",
    "RiemannianAdam",
    "RiemannianSGD",
    "build_criterion",
    "build_optimizer",
    "config",
    "data",
    "default_dims",
    "egn_base",
    "egn_small",
    "egn_tiny",
    "engine",
    "functional",
    "geometry",
    "manifolds",
    "nn",
    "numeric_context",
    "seed_everything",
    "set_performance_defaults",
    "utils",
    "__version__",
]
