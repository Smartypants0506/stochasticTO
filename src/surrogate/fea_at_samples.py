"""
src/surrogate/fea_at_samples.py

Stage 5 (PCE Surrogate Construction) -- implementation-modules.md Item 11.

For each training-sample KL coefficient vector xi, perturbs the PHYSICAL
DENSITY FIELD (not the mesh) via RandomFieldHeaviside.set_eta_from_xi(),
then solves the fixed-density FEA problem through FEniTop's own form_fem()
linear solve. The nominal design density rho_e is held constant across all
samples -- only the random Heaviside threshold eta(x) varies, per
master-context Section 3.1/3.2's requirement that geometric manufacturing
error is modeled as a random field perturbing rho_phys, not as mesh warping.

Collects BOTH compliance and volume (plus their adjoint gradients) in the
same FEA-solve loop, since both are needed downstream as separate PCE
surrogates (compliance for the robust objective, volume for the mean-volume
constraint E[V] <= Vfrac) and collecting both costs zero extra FEA solves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.topology.heaviside_projection_glue import RandomFieldHeaviside

logger = logging.getLogger(__name__)


@dataclass
class SurrogateTrainingData:
    """Per-sample FEA outputs collected for PCE training (compliance AND volume).

    Attributes:
        compliance_samples: [n_train] compliance C_i at each xi_train row.
        volume_samples: [n_train] volume fraction V_i at each xi_train row
            (V varies with eta(x) because rho_phys depends on eta through
            the Heaviside projection).
        dC_drho_samples: [n_train x n_elems] adjoint dC_i/drho, chained
            through Heaviside.backward() and DensityFilter.backward().
        dV_drho_samples: [n_train x n_elems] adjoint dV_i/drho, same chaining.
    """
    compliance_samples: np.ndarray
    volume_samples: np.ndarray
    dC_drho_samples: np.ndarray
    dV_drho_samples: np.ndarray


def run_fea_at_samples(
    fem_dict: dict,
    opt_dict: dict,
    rho_nominal: np.ndarray,
    density_filter,
    heaviside: RandomFieldHeaviside,
    sens_problem,
    xi_train: np.ndarray,
    beta: float,
    linear_problem,
    rho_field,
) -> SurrogateTrainingData:
    """Evaluate compliance and volume at each training sample's perturbed density field.

    Args:
        fem_dict: FEniTop fem dict from fenitop_adapter.build_fem_dict.
        opt_dict: FEniTop opt dict from fenitop_adapter.build_opt_dict.
        rho_nominal: [n_elems] nominal (pre-projection) design density,
            held fixed across all samples -- only eta(x) varies per sample.
        density_filter: FEniTop's DensityFilter instance (from topopt.py's
            initialization), applied identically (deterministically) each
            sample since the Helmholtz PDE filter has no randomness.
        heaviside: A RandomFieldHeaviside already built against this
            problem's rho_phys/mesh (via build_random_heaviside_from_function_space).
        sens_problem: FEniTop's Sensitivity instance (from topopt.py's
            initialization), used to read [C, V, U] and their adjoint
            sensitivities in a single call per sample.
        xi_train: [n_train x n_kl] KL coefficient matrix from sampler.py.
        beta: Heaviside sharpness parameter. Must come from the same
            beta_max/continuation schedule used in the converged nominal
            SIMP run (config.optimization.beta_max), so training compliance
            is consistent with the design the robust loop will optimize.

    Returns:
        A SurrogateTrainingData with compliance/volume samples and their
        per-sample adjoint gradients w.r.t. the unfiltered design variable rho.

    Raises:
        RuntimeError: If any sample produces a non-finite compliance or
            volume value.
    """

    n_train = xi_train.shape[0]
    n_elems = rho_nominal.size
    compliance_samples = np.empty(n_train)
    volume_samples = np.empty(n_train)
    dC_drho_samples = np.empty((n_train, n_elems))
    dV_drho_samples = np.empty((n_train, n_elems))

    for j in range(n_train):
        rho_field.x.petsc_vec.array[:] = rho_nominal
        density_filter.forward()  # deterministic Helmholtz filter: rho -> rho_tilde

        heaviside.set_eta_from_xi(xi_train[j])
        heaviside.forward(beta)  # perturbs rho_phys via eta(x), not the mesh

        linear_problem.solve_fem()

        func_values, sensitivities = sens_problem.evaluate()
        C_value, V_value, _ = func_values
        dCdrho_vec, dVdrho_vec, _ = sensitivities

        if not np.isfinite(C_value) or not np.isfinite(V_value):
            raise RuntimeError(
                f"Non-finite compliance/volume at training sample {j} "
                f"(C={C_value}, V={V_value}). Likely a near-disconnected "
                "structure under this eta(x) draw -- investigate before "
                "trusting the PCE surrogate."
            )

        heaviside.backward(sensitivities)  # chain rule through Heaviside, in-place
        dCdrho, dVdrho, _ = density_filter.backward(sensitivities)

        compliance_samples[j] = C_value
        volume_samples[j] = V_value
        dC_drho_samples[j, :] = dCdrho
        dV_drho_samples[j, :] = dVdrho

        if (j + 1) % 50 == 0 or j == n_train - 1:
            logger.info("FEA-at-samples: completed %d/%d solves", j + 1, n_train)

    return SurrogateTrainingData(
        compliance_samples=compliance_samples,
        volume_samples=volume_samples,
        dC_drho_samples=dC_drho_samples,
        dV_drho_samples=dV_drho_samples,
    )