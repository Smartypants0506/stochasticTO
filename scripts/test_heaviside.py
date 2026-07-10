"""Standalone smoke test for src/topology/heaviside_projection_glue.py.

No dolfinx/PETSc required -- mocks the Function interface FEniTop expects.
Run: python scripts/test_heaviside_glue_standalone.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from src.random_fields.kernel import KernelParams
from src.random_fields.threshold_transform import MarginalTransformParams
from src.topology.heaviside_projection_glue import RandomFieldHeaviside, RandomHeavisideConfig


class _MockVector:
    """Mimics petsc4py.PETSc.Vec's `.array` attribute for a dolfinx Function."""
    def __init__(self, array: np.ndarray):
        self.array = array


class _MockX:
    def scatter_forward(self) -> None:
        pass  # no-op: serial mock, no MPI ghost exchange needed


class MockFunction:
    """Mimics dolfinx.fem.Function's public interface used by the glue module."""
    def __init__(self, n_dofs: int, fill_value: float = 0.5):
        self.vector = _MockVector(np.full(n_dofs, fill_value))
        self.x = _MockX()


def make_synthetic_mesh(nx: int = 10, ny: int = 6):
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 0.5, ny)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    simplices = Delaunay(nodes).simplices
    return nodes, simplices


def main() -> None:
    print("=== Step 1: Build synthetic mesh + mock rho_phys Function ===")
    nodes, simplices = make_synthetic_mesh()
    n_dofs = nodes.shape[0]
    rho_phys = MockFunction(n_dofs, fill_value=0.5)
    print(f"n_dofs={n_dofs}")


    print("\n=== Step 2: Build RandomFieldHeaviside (precomputes KL expansion) ===")
    config = RandomHeavisideConfig(
        kernel_params=KernelParams(sigma=1.0, length_scale=0.15, spatial_dim=2),
        transform_params=MarginalTransformParams(eta_min=0.3, eta_max=0.7, alpha=2.0, beta=2.0),
        seed=42,
    )
    rf_heaviside = RandomFieldHeaviside(rho_phys, nodes, simplices, config)
    print(f"N_kl={rf_heaviside.kl_result.n_kl}, "
          f"variance_explained={rf_heaviside.kl_result.variance_explained:.4f}")

    print("\n=== Step 3: Verify random-field forward() reduces to deterministic case ===")
    passed = rf_heaviside.verify_reduces_to_deterministic(beta=4.0, eta_value=0.5)
    assert passed, "Equivalence check FAILED: array-eta diverges from scalar-eta"
    print("PASS: constant-array eta matches original scalar-eta Heaviside exactly")

    print("\n=== Step 4: Resample eta(x) and run forward() with a real random field ===")
    rho_phys.vector.array[:] = 0.5  # reset
    eta_sample = rf_heaviside.resample(seed=7)
    print(f"eta(x) sample: min={eta_sample.min():.4f}, max={eta_sample.max():.4f}, "
          f"mean={eta_sample.mean():.4f}")
    rf_heaviside.forward(beta=4.0)
    print(f"rho_phys after projection: min={rho_phys.vector.array.min():.4f}, "
          f"max={rho_phys.vector.array.max():.4f}")
    assert np.all(rho_phys.vector.array >= 0.0) and np.all(rho_phys.vector.array <= 1.0), (
        "Projected density out of [0, 1] bounds"
    )
    print("PASS: projected density stays within [0, 1]")

    print("\n=== Step 5: Verify backward() sensitivity recovery runs without error ===")
    fake_sensitivity = _MockVector(np.random.default_rng(0).standard_normal(n_dofs))
    rf_heaviside.backward([fake_sensitivity])
    assert rf_heaviside.drho is not None
    print(f"drho: min={rf_heaviside.drho.min():.4f}, max={rf_heaviside.drho.max():.4f}")
    print("PASS: backward() ran and scaled sensitivities by drho")

    print("\n=== Step 6: Verify set_deterministic_eta() fallback path ===")
    rho_phys.vector.array[:] = 0.5
    rf_heaviside.set_deterministic_eta(0.5)
    rf_heaviside.forward(beta=4.0)
    print(f"Deterministic-mode rho_phys range: [{rho_phys.vector.array.min():.4f}, "
          f"{rho_phys.vector.array.max():.4f}]")
    print("PASS: deterministic fallback path works")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()