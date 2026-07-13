"""
src/validation/sensitivity_fd_check.py

Stage 2 verification gate #2 (masterContext.md Section 3.1):
"Computes adjoint sensitivities dC/drho_e ... must still pass
finite-difference verification to relative error < 1e-5".

Perturbs each element's density by a small step and compares the resulting
change in compliance against FEniTop's own adjoint-computed sensitivity.
Uses a small mesh (few hundred elements) since this is an O(N) cost check,
not meant to scale to production mesh sizes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter, Heaviside
from src.fenitop.sensitivity import Sensitivity

logger = logging.getLogger(__name__)

FD_STEP = 1e-6
REL_ERROR_TOL = 1e-5


@dataclass
class SensitivityCheckResult:
    max_relative_error: float
    n_elements_checked: int
    passed: bool


def run_sensitivity_fd_check(
    n_check_elements: int = 20, nx: int = 30, ny: int = 10,
    seed: int = 0,
) -> SensitivityCheckResult:
    """Compare adjoint dC/drho_e against central-difference dC/drho_e on a
    random subset of elements (checking all elements on a real mesh is
    prohibitively expensive; n_check_elements is a representative sample)."""
    comm = MPI.COMM_WORLD
    mesh = create_rectangle(comm, [[0, 0], [30, 10]], [nx, ny], CellType.quadrilateral)

    fem_dict = {
        "mesh": mesh, "young's modulus": 100.0, "poisson's ratio": 0.3,
        "disp_bc": lambda x: np.isclose(x[0], 0),
        "traction_bcs": [[(0.0, -1.0), lambda x: np.isclose(x[0], 30) & (x[1] > 4) & (x[1] < 6)]],
        "body_force": (0.0, 0.0), "quadrature_degree": 2,
        "petsc_options": {"ksp_type": "cg", "pc_type": "gamg"},
    }
    opt_dict = {
        "penalty": 3.0, "epsilon": 1e-6, "opt_compliance": True,
        "filter_radius": 1.2,
    }

    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem_dict, opt_dict)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                    opt_dict["filter_radius"], fem_dict["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt_dict, linear_problem, u_field, lambda_field, rho_phys_field)

    rng = np.random.default_rng(seed)
    n_elems = rho_field.x.petsc_vec.array.size
    rho_field.x.petsc_vec.array[:] = 0.5

    def compute_compliance() -> float:
        density_filter.forward()
        heaviside.forward(beta=1.0)
        linear_problem.solve_fem()
        [C_value, _, _], _ = sens_problem.evaluate()
        return C_value

    C0 = compute_compliance()
    density_filter.forward()
    heaviside.forward(beta=1.0)
    linear_problem.solve_fem()
    [_, _, _], sensitivities = sens_problem.evaluate()
    heaviside.backward(sensitivities)
    [dCdrho, _, _] = density_filter.backward(sensitivities)

    check_indices = rng.choice(n_elems, size=min(n_check_elements, n_elems), replace=False)
    rel_errors = []
    for idx in check_indices:
        base = rho_field.x.petsc_vec.array[idx]
        rho_field.x.petsc_vec.array[idx] = base + FD_STEP
        C_plus = compute_compliance()
        rho_field.x.petsc_vec.array[idx] = base - FD_STEP
        C_minus = compute_compliance()
        rho_field.x.petsc_vec.array[idx] = base

        fd_grad = (C_plus - C_minus) / (2 * FD_STEP)
        adjoint_grad = dCdrho[idx]
        rel_err = abs(fd_grad - adjoint_grad) / (abs(adjoint_grad) + 1e-12)
        rel_errors.append(rel_err)

    max_rel_error = float(np.max(rel_errors))
    passed = max_rel_error < REL_ERROR_TOL

    logger.info(
        "Sensitivity FD check: max_rel_error=%.3e over %d elements, passed=%s",
        max_rel_error, len(check_indices), passed,
    )
    return SensitivityCheckResult(
        max_relative_error=max_rel_error,
        n_elements_checked=len(check_indices),
        passed=passed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_sensitivity_fd_check()
    if not result.passed:
        raise SystemExit(
            f"FAILED sensitivity FD check: max_rel_error={result.max_relative_error:.3e} "
            f"exceeds tolerance {REL_ERROR_TOL:.0e}. Do not proceed until fixed."
        )
    print("PASSED: adjoint sensitivities verified against finite differences.")