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


MPI note: this loop is WORLD-COLLECTIVE, not sample-parallel. Every rank
must call run_fea_at_samples() together with an identical xi_train, and
every one of the n_train solves is itself a full distributed parallel solve
across all ranks (FEniTop's mesh/KSP are already domain-decomposed). This
is the correct parallelism model for `-n 64`: each individual FEA solve
gets faster as ranks increase, rather than different ranks solving
different samples independently (which the current single shared mesh/KSP
setup cannot support without a separate per-group mesh -- out of scope
here). rho_nominal, and every per-sample gradient row stored here, must
therefore be THIS RANK's LOCAL dof/element slice, not a global array.
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
        dC_drho_samples: [n_train x n_elems_local] adjoint dC_i/drho, chained
            through Heaviside.backward() and DensityFilter.backward().
            n_elems_local is THIS RANK's local element count under MPI.
        dV_drho_samples: [n_train x n_elems_local] adjoint dV_i/drho, same
            chaining and same local-sizing convention.
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
        rho_nominal: [n_elems_local] nominal (pre-projection) design density,
            held fixed across all samples -- only eta(x) varies per sample.
            Must be THIS RANK's local slice, matching rho_field's local dof
            array size exactly (see raises below).
        density_filter: FEniTop's DensityFilter instance (from topopt.py's
            initialization), applied identically (deterministically) each
            sample since the Helmholtz PDE filter has no randomness.
        heaviside: A RandomFieldHeaviside already built against this
            problem's rho_phys/mesh (via build_random_heaviside_from_function_space).
            Must be called with the same xi on every rank -- see its own
            MPI design note.
        sens_problem: FEniTop's Sensitivity instance (from topopt.py's
            initialization), used to read [C, V, U] and their adjoint
            sensitivities in a single call per sample.
        xi_train: [n_train x n_kl] KL coefficient matrix from sampler.py.
            Must be IDENTICAL on every rank (this is a collective loop).
        beta: Heaviside sharpness parameter. Must come from the same
            beta_max/continuation schedule used in the converged nominal
            SIMP run (config.optimization.beta_max), so training compliance
            is consistent with the design the robust loop will optimize.


    Returns:
        A SurrogateTrainingData with compliance/volume samples (global,
        identical on every rank via FEniTop's internal allreduce) and their
        per-sample adjoint gradients w.r.t. the unfiltered design variable
        rho, restricted to this rank's local elements.


    Raises:
        ValueError: If rho_nominal's size does not match rho_field's local
            dof array size, which would otherwise corrupt the density field
            silently under MPI.
        RuntimeError: If any sample produces a non-finite compliance or
            volume value.
    """
    expected_local_shape = rho_field.x.petsc_vec.array.shape
    if rho_nominal.shape != expected_local_shape:
        raise ValueError(
            f"rho_nominal shape {rho_nominal.shape} does not match rho_field's "
            f"local dof shape {expected_local_shape}. Under MPI, rho_nominal "
            "must be THIS RANK's local slice, not a global array -- passing "
            "a global-sized array here would silently corrupt the density "
            "field on every rank but rank 0."
        )

    comm = fem_dict["mesh"].comm
    is_root = comm.rank == 0

    n_train = xi_train.shape[0]
    n_elems_local = rho_nominal.size
    compliance_samples = np.empty(n_train)
    volume_samples = np.empty(n_train)
    dC_drho_samples = np.empty((n_train, n_elems_local))
    dV_drho_samples = np.empty((n_train, n_elems_local))


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


        if is_root and ((j + 1) % 50 == 0 or j == n_train - 1):
            logger.info("FEA-at-samples: completed %d/%d solves", j + 1, n_train)


    return SurrogateTrainingData(
        compliance_samples=compliance_samples,
        volume_samples=volume_samples,
        dC_drho_samples=dC_drho_samples,
        dV_drho_samples=dV_drho_samples,
    )