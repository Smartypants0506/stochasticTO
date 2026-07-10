"""Standalone smoke test for src/random_fields/* -- no FEniCSx/PETSc required.

Run: python scripts/test_random_field_standalone.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from src.random_fields.kernel import KernelParams, build_squared_exponential
from src.random_fields.kl_expansion import (
    compute_kl_expansion,
    sample_gaussian_field,
    verify_sample_covariance,
)
from src.random_fields.threshold_transform import (
    MarginalTransformParams,
    ThresholdMarginalTransform,
)
import openturns as ot


def make_synthetic_grid_mesh(nx: int = 12, ny: int = 8, lx: float = 1.0, ly: float = 0.5):
    """Build a plain rectangular grid + Delaunay triangulation.

    Stands in for a real FEniTop mesh export -- same [N x 2] node array and
    [M x 3] simplex array shape that FEniCSx would give you, so this module
    is fully decoupled from dolfinx for testing purposes.
    """
    xs = np.linspace(0.0, lx, nx)
    ys = np.linspace(0.0, ly, ny)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    simplices = Delaunay(nodes).simplices
    return nodes, simplices


def main() -> None:
    print([m for m in dir(ot.KarhunenLoeveResult) if "odes" in m or "igen" in m])
    print("=== Step 1: Build synthetic mesh (stand-in for FEniTop mesh) ===")
    nodes, simplices = make_synthetic_grid_mesh()
    print(f"nodes: {nodes.shape}, simplices: {simplices.shape}")

    print("\n=== Step 2: Build squared-exponential kernel ===")
    kernel_params = KernelParams(sigma=1.0, length_scale=0.15, spatial_dim=2)
    cov_model = build_squared_exponential(kernel_params)
    print(f"Kernel built: {cov_model}")

    print("\n=== Step 3: Compute KL expansion (>=95% variance) ===")
    kl_result = compute_kl_expansion(nodes, simplices, kernel_params)
    print(f"N_kl={kl_result.n_kl}, variance_explained={kl_result.variance_explained:.4f}")
    assert kl_result.variance_explained >= 0.95, "Variance threshold gate FAILED"
    print("PASS: variance_explained >= 0.95")

    print("\n=== Step 4: Verify sample covariance matches theoretical kernel ===")
    cov_check = verify_sample_covariance(kl_result, n_samples=3000, seed=42)
    print(cov_check)
    assert cov_check["passed"], "Sample covariance verification gate FAILED"
    print("PASS: sample covariance matches theoretical kernel within tolerance")

    print("\n=== Step 5: Sample Gaussian field realizations G(x) ===")
    g_samples = sample_gaussian_field(kl_result, n_samples=5, seed=1)
    print(f"G(x) samples shape: {g_samples.shape}, mean={g_samples.mean():.4f}, "
          f"std={g_samples.std():.4f}")

    print("\n=== Step 6: Apply memoryless marginal transform eta(x) = T(G(x)) ===")
    transform_params = MarginalTransformParams(eta_min=0.3, eta_max=0.7, alpha=2.0, beta=2.0)
    transform = ThresholdMarginalTransform(transform_params)
    eta_samples = transform.transform(g_samples)
    print(f"eta(x) samples shape: {eta_samples.shape}, "
          f"min={eta_samples.min():.4f}, max={eta_samples.max():.4f}")

    within_bounds = transform.validate_bounds(eta_samples)
    assert within_bounds, "Bounds validation gate FAILED"
    print(f"PASS: all eta(x) values within [{transform_params.eta_min}, {transform_params.eta_max}]")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()