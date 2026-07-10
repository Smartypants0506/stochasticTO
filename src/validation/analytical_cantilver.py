"""
src/validation/analytical_cantilever.py

Stage 2 verification gate #1 (masterContext.md Section 3.1):
"Verification required: Cantilever beam analytical solution
delta = P*L^3/(3*E*I)".

This module does NOT run topology optimization -- it runs a single FEniTop
linear-elastic FEA solve (vol_frac=1, i.e. fully solid domain) on a simple
rectangular cantilever and compares tip displacement against Euler-Bernoulli
beam theory. Must be run and passed before Stage 2's FEniTop core is
considered verified for this project's use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

from src.fenitop.fem import form_fem

logger = logging.getLogger(__name__)


@dataclass
class CantileverCheckResult:
    tip_displacement_fem: float
    tip_displacement_analytical: float
    relative_error: float
    passed: bool


def analytical_tip_deflection(P: float, L: float, E: float, I: float) -> float:
    """Euler-Bernoulli cantilever tip deflection: delta = P*L^3/(3*E*I)."""
    return P * L**3 / (3 * E * I)


def run_cantilever_check(
    length: float = 60.0, height: float = 4.0,
    nx: int = 240, ny: int = 16,
    E: float = 200e3, nu: float = 0.3,
    tip_force: float = -100.0,
    rel_error_tol: float = 0.05,
) -> CantileverCheckResult:
    """Run a fully-solid FEniTop FEA solve and compare to beam theory.

    A generous rel_error_tol (5%) is used because 2D plane-strain elasticity
    (what form_fem() implements) is not identical to 1D Euler-Bernoulli beam
    theory, especially for a moderately stubby height/length ratio; this
    tolerance itself must be documented and justified in the verification
    report, not silently widened if the check fails.
    """
    comm = MPI.COMM_WORLD
    mesh = create_rectangle(comm, [[0, 0], [length, height]], [nx, ny], CellType.quadrilateral)

    fem_dict = {
        "mesh": mesh,
        "young's modulus": E,
        "poisson's ratio": nu,
        "disp_bc": lambda x: np.isclose(x[0], 0),
        "traction_bcs": [[(0.0, tip_force),
                            lambda x: np.isclose(x[0], length) & (x[1] > height * 0.4) & (x[1] < height * 0.6)]],
        "body_force": (0.0, 0.0),
        "quadrature_degree": 2,
        "petsc_options": {"ksp_type": "cg", "pc_type": "gamg"},
    }
    opt_dict = {
        "penalty": 3.0, "epsilon": 1e-6, "opt_compliance": True,
    }

    linear_problem, u_field, _, rho_field, rho_phys_field = form_fem(fem_dict, opt_dict)
    rho_field.vector.array[:] = 1.0       # fully solid: no SIMP penalization effect
    rho_phys_field.vector.array[:] = 1.0
    linear_problem.solve_fem()

    tip_dofs = np.isclose(mesh.geometry.x[:, 0], length) & \
               (mesh.geometry.x[:, 1] > height * 0.4) & (mesh.geometry.x[:, 1] < height * 0.6)
    tip_disp_fem = float(np.mean(u_field.x.array.reshape(-1, 2)[tip_dofs, 1])) if tip_dofs.any() else float("nan")

    I = height**3 / 12.0  # per-unit-width second moment of area (2D plane strain)
    tip_disp_analytical = analytical_tip_deflection(tip_force, length, E, I)

    rel_error = abs(tip_disp_fem - tip_disp_analytical) / abs(tip_disp_analytical)
    passed = rel_error < rel_error_tol

    logger.info(
        "Cantilever check: FEM=%.6g, analytical=%.6g, rel_error=%.4f, passed=%s",
        tip_disp_fem, tip_disp_analytical, rel_error, passed,
    )
    return CantileverCheckResult(
        tip_displacement_fem=tip_disp_fem,
        tip_displacement_analytical=tip_disp_analytical,
        relative_error=rel_error,
        passed=passed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_cantilever_check()
    if not result.passed:
        raise SystemExit(
            f"FAILED cantilever verification: rel_error={result.relative_error:.4f} "
            f"exceeds tolerance. Do not proceed to Stage 3 until this passes."
        )
    print("PASSED: FEniTop core verified against Euler-Bernoulli cantilever solution.")