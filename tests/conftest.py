"""Shared fixtures.

Most tests here deliberately need NO dolfinx and NO FEA. The properties that
were actually wrong in this codebase -- a convergence test that could never
fire, a gradient reduction, a biased statistic, a config that did not describe
its run -- are all checkable against closed-form answers on small arrays. Tests
that need a solver are marked and skipped when dolfinx is absent, so the suite
runs anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "dolfinx: requires dolfinx/PETSc")
    config.addinivalue_line("markers", "mpi: must be run under mpirun with >1 rank")


@pytest.fixture
def rng():
    return np.random.default_rng(20240727)


@pytest.fixture
def small_kl():
    """A tiny hand-built KLExpansionResult: 12 nodes, 4 modes, non-uniform
    pointwise variance so the normalization is actually exercised."""
    from src.random_fields.kernel import KernelParams
    from src.random_fields.kl_expansion import KLExpansionResult

    rng = np.random.default_rng(7)
    n_nodes, n_kl = 12, 4
    modes, _ = np.linalg.qr(rng.standard_normal((n_nodes, n_kl)))
    eigenvalues = np.array([1.0, 0.5, 0.25, 0.125])
    return KLExpansionResult(
        eigenvalues=eigenvalues,
        modes=modes,
        mean_field=np.zeros(n_nodes),
        variance_explained=0.95,
        n_kl=n_kl,
        node_coordinates=rng.random((n_nodes, 3)),
        kernel_params=KernelParams(sigma=1.0, length_scale=1.0, spatial_dim=3),
    )
