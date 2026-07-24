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

import time


logger = logging.getLogger(__name__)

# Rebuild the (frozen) GAMG preconditioner every this many samples within a
# batch, if PC reuse is engaged at all. Set to 1 (rebuild every sample --
# effectively OFF): for this project's SIMP contrast (epsilon=1e-6) combined
# with a sharp Heaviside projection (beta=8), even a small eta(x) shift between
# samples can flip which near-threshold elements are solid vs. void, changing
# the connectivity/contrast pattern the GAMG hierarchy depends on -- the
# "mildly-varying matrix" assumption reuse relies on does not hold here.
# Empirically this made reuse fail (DIVERGED_INDEFINITE_PC) on nearly every
# solve, paying for a failed CG attempt AND a fresh rebuild -- strictly worse
# than always rebuilding. LinearProblem.solve_fem()'s retry-on-failure fallback
# still protects correctness if this is ever raised again for a problem where
# reuse genuinely helps; it just won't fire while this stays at 1.
PC_REBUILD_INTERVAL = 1



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
    raise_on_nonfinite: bool = True,
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

    # rho_nominal is fixed across all samples -- only eta(x) varies -- so the
    # density write + Helmholtz filter solve only needs to happen ONCE, not
    # once per sample. Also fixes the missing ghost sync: writing directly
    # to .petsc_vec.array only touches OWNED dofs, so scatter_forward() is
    # required before density_filter.forward() reads ghost values during
    # its PDE assembly (matters for -n ranks > 1).
    rho_field.x.petsc_vec.array[:] = rho_nominal
    rho_field.x.scatter_forward()
    density_filter.forward()  # deterministic Helmholtz filter: rho -> rho_tilde
    rho_tilde_cached = heaviside.rho_phys.x.petsc_vec.array.copy()

    # --- solver warm-start + GAMG hierarchy reuse across this batch ----------
    # rho_nominal is fixed and only eta(x) varies mildly between samples, so the
    # stiffness matrices are close: (1) warm-start CG from the previous sample's
    # solution, and (2) build the GAMG hierarchy once and reuse it, rebuilding
    # only every PC_REBUILD_INTERVAL samples to bound CG-iteration growth as
    # eta walks away. Both are math-exact -- solve_fem() still assembles the true
    # matrix and CG converges to the same tolerance; only setup cost/iteration
    # count change. See LinearProblem.enable_warm_start / set_reuse_preconditioner.
    linear_problem.enable_warm_start(True)

    batch_t0 = time.time()
    for j in range(n_train):
        _s0 = time.time()
        heaviside.rho_phys.x.petsc_vec.array[:] = rho_tilde_cached
        heaviside.set_eta_from_xi(xi_train[j])
        _s1 = time.time()

        heaviside.forward(beta)  # perturbs rho_phys via eta(x), not the mesh
        _s2 = time.time()

        # reuse the frozen PC except on periodic rebuild steps (j==0 always
        # rebuilds, so the hierarchy is set up once before any reuse).
        rebuild_pc = (j % PC_REBUILD_INTERVAL == 0)
        linear_problem.set_reuse_preconditioner(not rebuild_pc)
        linear_problem.solve_fem()
        _s3 = time.time()

        func_values, sensitivities = sens_problem.evaluate()
        _s4 = time.time()

        C_value, V_value, _ = func_values

        if is_root and j < 3:
            logger.info(
                "eta=%.3f proj=%.3f solve=%.3f sens=%.3f",
                _s1 - _s0, _s2 - _s1, _s3 - _s2, _s4 - _s3,
            )

        if not np.isfinite(C_value) or not np.isfinite(V_value):
            if raise_on_nonfinite:
                raise RuntimeError(
                    f"Non-finite compliance/volume at training sample {j} "
                    f"(C={C_value}, V={V_value}). Likely a near-disconnected "
                    "structure under this eta(x) draw -- investigate before "
                    "trusting the PCE surrogate."
                )
            # Deferred mode (used by the sub-communicator grouped runner): record
            # the non-finite value and continue, so a single bad sample in one
            # group cannot raise mid-loop while other groups sit in a collective
            # (that would deadlock COMM_WORLD). The grouped runner performs a
            # single collective finiteness check after reassembling all samples.
            logger.warning(
                "Non-finite compliance/volume at sample %d (C=%s, V=%s); "
                "recording and continuing (deferred check).", j, C_value, V_value,
            )

        heaviside.backward(sensitivities)  # chain rule through Heaviside, in-place
        dCdrho, dVdrho, _ = density_filter.backward(sensitivities)

        compliance_samples[j] = C_value
        volume_samples[j] = V_value
        dC_drho_samples[j, :] = dCdrho
        dV_drho_samples[j, :] = dVdrho

        if is_root and ((j + 1) % 50 == 0 or j == n_train - 1):
            logger.info("FEA-at-samples: completed %d/%d solves (%.2fs for last batch)",
                        j + 1, n_train, time.time() - batch_t0)
            batch_t0 = time.time()

    # Unfreeze the PC so the frozen hierarchy from THIS batch's design cannot be
    # silently reused by a later solve at a different design (e.g. the next
    # refresh, or the MMA world solve sharing this LinearProblem).
    linear_problem.set_reuse_preconditioner(False)

    return SurrogateTrainingData(
        compliance_samples=compliance_samples,
        volume_samples=volume_samples,
        dC_drho_samples=dC_drho_samples,
        dV_drho_samples=dV_drho_samples,
    )