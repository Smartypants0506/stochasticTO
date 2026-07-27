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
            None in ACCUMULATE mode -- see below.
        dV_drho_samples: [n_train x n_elems_local] adjoint dV_i/drho, same
            chaining and same local-sizing convention. None in accumulate mode.

    ACCUMULATE MODE (accumulate_gradients=True)
    -------------------------------------------
    The SAA robust objective never needs the individual gradient rows. It needs
    exactly two reductions of them:

        dmu/drho    = (1/N)  sum_i dC_i
        dsigma/drho = (1/((N-1) sigma)) sum_i (C_i - mu) dC_i
        dE[V]/drho  = (1/N)  sum_i dV_i

    all of which accumulate in place as the batch runs. Materializing the full
    [N x n_elems] matrices instead cost, per objective evaluation, an
    [N x n_elems_local] buffer AND -- in the sample-parallel path -- 2N
    world-broadcasts of a full global array, one pair per sample. At N=512 and
    400 iterations that was the dominant communication cost of the whole solve.

    The accumulator fields below are algebraically identical to reducing the
    stored rows; the only difference is that nothing is stored.

        dC_sum:          sum_i dC_i/drho
        dC_centered_sum: sum_i (C_i - C_reference) dC_i/drho
        dV_sum:          sum_i dV_i/drho
        C_reference:     the shift used in dC_centered_sum, and the reason it
            exists: accumulating sum_i C_i dC_i and subtracting mu*sum_i dC_i
            afterwards is mathematically the same but numerically poor here,
            because C_i ~ 0.16 with a spread of ~0.008 -- a 20:1 ratio that
            loses over a digit to cancellation. Shifting by a value already
            inside the sample range keeps every accumulated term the size of
            the spread rather than the size of the mean. The exact centered sum
            is recovered as
                sum_i (C_i - mu) dC_i = dC_centered_sum + (C_reference - mu) * dC_sum
            which is an identity, not an approximation.
    """
    compliance_samples: np.ndarray
    volume_samples: np.ndarray
    dC_drho_samples: np.ndarray | None
    dV_drho_samples: np.ndarray | None
    dC_sum: np.ndarray | None = None
    dC_centered_sum: np.ndarray | None = None
    dV_sum: np.ndarray | None = None
    C_reference: float | None = None



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
    accumulate_gradients: bool = False,
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

    # In accumulate mode the per-sample gradient rows are never materialized --
    # only the three reductions the SAA objective actually consumes. See
    # SurrogateTrainingData's docstring.
    if accumulate_gradients:
        dC_drho_samples = dV_drho_samples = None
        dC_sum = np.zeros(n_elems_local)
        dC_centered_sum = np.zeros(n_elems_local)
        dV_sum = np.zeros(n_elems_local)
        C_reference = None
    else:
        dC_drho_samples = np.empty((n_train, n_elems_local))
        dV_drho_samples = np.empty((n_train, n_elems_local))
        dC_sum = dC_centered_sum = dV_sum = None
        C_reference = None

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
        if accumulate_gradients:
            if C_reference is None:
                # First finite compliance in this batch: any value inside the
                # sample range works as the shift, and using one from the batch
                # itself needs no extra information.
                C_reference = float(C_value)
            dC_sum += dCdrho
            dC_centered_sum += (C_value - C_reference) * dCdrho
            dV_sum += dVdrho
        else:
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
        dC_sum=dC_sum,
        dC_centered_sum=dC_centered_sum,
        dV_sum=dV_sum,
        C_reference=C_reference,
    )