"""
src/optimization/saa_robust_driver.py

Surrogate-free robust topology optimization via Sample Average Approximation
(SAA) -- masterContext Section 3.5 (robust objective J = mu_C + lambda*sigma_C,
mean-volume constraint E[V] <= vol_frac), but WITHOUT the PCE middleman.

Instead of fitting a PCE over the KL coordinates xi and evaluating mu_C/sigma_C
analytically (with a linear rho-extrapolation that had to be periodically
refreshed and was the source of the volume-collapse divergence), this driver
fixes ONE large sample set xi_saa [N x n_kl] and, at EVERY MMA iteration,
evaluates the EXACT sample-average robust objective and its EXACT gradient via
N full FEA solves at the current design rho:

    mu_C(rho)    = mean_i C_i(rho)
    sigma_C(rho) = std_i  C_i(rho)                 (ddof=1)
    E[V](rho)    = mean_i V_i(rho)
    J(rho)       = mu_C + lambda*sigma_C
    dJ/drho      = mean_i dC_i/drho + lambda * (centered-sample dsigma/drho)
    dE[V]/drho   = mean_i dV_i/drho

Because xi_saa is FIXED across all iterations (and all lambda), J_N(rho) is a
smooth DETERMINISTIC function of rho (common random numbers / SAA), so the MMA
converges cleanly to its exact minimizer -> a real KKT residual (not NaN). The
only approximation is the finite sample count N, controlled by making N large
(accuracy-for-compute).

Reuses, unchanged:
  * setup_robust_problem / run_fea_at_samples_grouped / GroupFEAContext
    (dolfiny_mma_driver.py) -- FEA machinery + MPI sample-parallel batch solve.
  * run_fea_at_samples (fea_at_samples.py) -- serial fallback.
  * robust_objective.py / robust_gradient.py -- the direct sample estimators
    (the pre-PCE MVP math, now the primary path).
  * fenitop/mma.py -- the dolfiny TAO MMA (incl. its graceful subsolver-MAXITS
    handling).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.fenitop.utility import Communicator, Plotter, save_xdmf
from src.fenitop.mma import MMA

from src.optimization.dolfiny_mma_driver import (
    RobustProblemContext,
    run_fea_at_samples_grouped,
)
from src.surrogate.fea_at_samples import run_fea_at_samples
from src.optimization.robust_objective import (
    RobustObjectiveConfig,
    RobustEvaluationResult,
    compute_robust_objective_value,
)
from src.optimization.robust_gradient import (
    compute_robust_gradient,
    compute_mean_volume_gradient,
)

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

# E[V] <= vol_frac*(1+tol) counts as feasible for the best-design checkpoint.
_VOL_FEAS_TOL = 0.02


@dataclass
class SAALoopState:
    """Mutable state threaded through the TAO callbacks (which are stateless
    functions). Holds the per-rho FEA-batch cache (so the objective, constraint,
    and Jacobian callbacks -- all called by MMA at the same rho within one outer
    iteration -- share ONE batch of N FEA solves) and the best-feasible-design
    checkpoint returned at exit."""
    true_outer_iteration: int = 0
    outer_iteration: int = 0
    n_batch_evals: int = 0

    cached_rho: np.ndarray | None = None
    cached_result: RobustEvaluationResult | None = None

    best_J: float | None = None
    best_rho_local: np.ndarray | None = None
    best_mu_C: float | None = None
    best_sigma_C: float | None = None
    best_mean_volume: float | None = None


def _evaluate_saa(
    ctx: RobustProblemContext, opt: dict, rho_local: np.ndarray,
    xi_saa: np.ndarray, beta: float,
) -> RobustEvaluationResult:
    """Run the fixed SAA batch of N FEA solves at rho_local and package the
    per-sample outputs + sample-average stats as a RobustEvaluationResult.

    World-collective: every rank must call this together (the batch solve is
    MPI-collective, sample-parallel when ctx.group is set).
    """
    if ctx.group is not None:
        data = run_fea_at_samples_grouped(ctx, ctx.group, opt, rho_local, xi_saa, beta)
    else:
        data = run_fea_at_samples(
            ctx.fem, opt, rho_local, ctx.density_filter, ctx.rf_heaviside,
            ctx.sens_problem, xi_saa, beta, ctx.linear_problem, ctx.rho_field,
        )

    C = data.compliance_samples
    V = data.volume_samples
    return RobustEvaluationResult(
        compliance_samples=C,
        volume_samples=V,
        dC_drho_samples=data.dC_drho_samples,
        dV_drho_samples=data.dV_drho_samples,
        mu_C=float(C.mean()),
        sigma_C=float(C.std(ddof=1)),
        mean_volume=float(V.mean()),
    )


def _get_result(
    state: SAALoopState, ctx: RobustProblemContext, opt: dict,
    rho_local: np.ndarray, xi_saa: np.ndarray, beta: float,
) -> RobustEvaluationResult:
    """Return the SAA batch result at rho_local, reusing the cache when the
    design is unchanged since the last evaluation. The match decision is
    reduced across ALL ranks (MPI.MIN) so every rank agrees whether to run the
    collective batch -- otherwise a partial call would deadlock."""
    local_match = (
        state.cached_rho is not None
        and state.cached_rho.shape == rho_local.shape
        and np.array_equal(state.cached_rho, rho_local)
    )
    all_match = bool(comm.allreduce(1 if local_match else 0, op=MPI.MIN))
    if all_match:
        return state.cached_result

    result = _evaluate_saa(ctx, opt, rho_local, xi_saa, beta)
    state.cached_rho = rho_local.copy()
    state.cached_result = result
    state.n_batch_evals += 1
    return result


def run_saa_robust_topopt(
    ctx: RobustProblemContext,
    opt: dict,
    lambda_tradeoff: float,
    xi_saa: np.ndarray,
    beta: float | None = None,
    x0_local: np.ndarray | None = None,
) -> dict:
    """Run the TAO MMA outer loop for ONE lambda against the EXACT sample-average
    robust objective (no surrogate).

    Args:
        ctx: RobustProblemContext from setup_robust_problem() (FEA machinery +
            optional sample-parallel GroupFEAContext + warm-start).
        opt: FEniTop opt dict (needs vol_frac, move, opt_tol, max_iter, ...).
        lambda_tradeoff: mean-variance weight in J = mu_C + lambda*sigma_C.
        xi_saa: [N x n_kl] FIXED KL-coefficient sample set, identical on every
            rank, reused at every iteration (SAA / common random numbers).
        beta: Heaviside sharpness for the batch solves (default opt["saa_beta"]
            or 8.0).
        x0_local: THIS RANK's local slice of the starting design (defaults to
            ctx.rho_warm_start_local). Pass the previous lambda's converged
            design for lambda-continuation across a Pareto sweep.

    Returns:
        dict with rho_robust (global), mu_C, sigma_C, mean_volume, kkt_residual,
        tao_converged_reason, n_fea_batches, iteration_log.
    """
    if beta is None:
        beta = float(opt.get("saa_beta", 8.0))
    if x0_local is None:
        x0_local = ctx.rho_warm_start_local

    rho_field = ctx.rho_field
    rho_phys_field = ctx.rho_phys_field
    density_filter = ctx.density_filter
    warm_start_comm = ctx.warm_start_comm
    n_elems_local = ctx.n_elems_local
    n_elems_global = ctx.n_elems_global
    col_start = ctx.col_start

    robust_config = RobustObjectiveConfig(lambda_tradeoff=lambda_tradeoff)
    state = SAALoopState()
    iteration_log: list[dict] = []

    def objective_gradient_callback(tao: PETSc.TAO, x: PETSc.Vec, g: PETSc.Vec) -> float:
        """J = mu_C + lambda*sigma_C and dJ/drho, both EXACT on the fixed SAA set."""
        rho_current = x.getArray(readonly=True).copy()
        result = _get_result(state, ctx, opt, rho_current, xi_saa, beta)

        J_value = compute_robust_objective_value(result, robust_config)
        dJ_drho = compute_robust_gradient(result, robust_config)
        g.setArray(dJ_drho)

        # Best-feasible-design checkpoint (deterministic J decreases as MMA
        # converges, so the best feasible J is the converged design).
        if result.mean_volume <= opt["vol_frac"] * (1.0 + _VOL_FEAS_TOL):
            if state.best_J is None or J_value < state.best_J:
                state.best_J = J_value
                state.best_rho_local = rho_current.copy()
                state.best_mu_C = result.mu_C
                state.best_sigma_C = result.sigma_C
                state.best_mean_volume = result.mean_volume

        iteration_log.append({
            "outer_iteration": state.outer_iteration,
            "true_outer_iteration": state.true_outer_iteration,
            "J": J_value,
            "mu_C": result.mu_C,
            "sigma_C": result.sigma_C,
            "mean_volume": result.mean_volume,
        })
        state.outer_iteration += 1
        return J_value

    def inequality_constraint_callback(tao: PETSc.TAO, x: PETSc.Vec, c: PETSc.Vec) -> None:
        """h(rho) = vol_frac - E[V] >= 0 (mma.py sign-flips to its <=0 form)."""
        rho_current = x.getArray(readonly=True).copy()
        result = _get_result(state, ctx, opt, rho_current, xi_saa, beta)
        h_value = opt["vol_frac"] - result.mean_volume

        if comm.rank == 0:
            logger.info(
                "constraint check: vol_frac=%.6g E[V]=%.6g h_value=%.6g",
                opt["vol_frac"], result.mean_volume, h_value,
            )
            c.setValue(0, h_value)
        c.assemble()

    def jacobian_inequality_callback(
        tao: PETSc.TAO, x: PETSc.Vec, J: PETSc.Mat, Jp: PETSc.Mat,
    ) -> None:
        """dh/drho = -dE[V]/drho = -mean_i(dV_i/drho)."""
        rho_current = x.getArray(readonly=True).copy()
        result = _get_result(state, ctx, opt, rho_current, xi_saa, beta)
        dh_drho = -compute_mean_volume_gradient(result)
        global_cols = np.arange(col_start, col_start + n_elems_local, dtype=PETSc.IntType)
        J.setValues([0], global_cols, dh_drho.reshape(1, -1))
        J.assemble()

    def mma_iteration_monitor(tao: PETSc.TAO) -> None:
        state.true_outer_iteration += 1
        if comm.rank == 0 and state.cached_result is not None:
            r = state.cached_result
            logger.info(
                "[SAA] outer_iter=%d: mu_C=%.6g sigma_C=%.6g E[V]=%.6g "
                "J=%.6g (fea_batches=%d)",
                state.true_outer_iteration, r.mu_C, r.sigma_C, r.mean_volume,
                r.mu_C + lambda_tradeoff * r.sigma_C, state.n_batch_evals,
            )

    # --- TAO MMA scaffolding (mirrors run_mma_with_pce) ----------------------
    tao = PETSc.TAO().create(comm)
    tao.setType(PETSc.TAO.Type.PYTHON)
    tao.setPythonContext(MMA())

    x0 = rho_field.x.petsc_vec.copy()
    x0.setArray(x0_local)
    tao.setSolution(x0)

    lb = x0.copy(); lb.set(0.0)
    ub = x0.copy(); ub.set(1.0)
    tao.setVariableBounds((lb, ub))

    grad_vec = x0.copy()
    tao.setObjectiveGradient(objective_gradient_callback, grad_vec)

    constraint_vec = PETSc.Vec().createMPI(1, comm=comm)
    tao.setInequalityConstraints(inequality_constraint_callback, constraint_vec)

    local_rows = 1 if comm.rank == 0 else 0
    jacobian_mat = PETSc.Mat().createDense(
        ((local_rows, 1), (n_elems_local, n_elems_global)), comm=comm
    )
    jacobian_mat.setUp()
    tao.setJacobianInequality(jacobian_inequality_callback, jacobian_mat, jacobian_mat)

    tao.setTolerances(gatol=opt["opt_tol"])
    tao.setMaximumIterations(opt["max_iter"])

    prefix = tao.getOptionsPrefix() or ""
    opts = PETSc.Options()
    opts[f"{prefix}tao_mma_move_limit"] = opt["move"]
    opts[f"{prefix}tao_mma_asymptote_init"] = opt.get("asymptote_init", 0.5)
    opts[f"{prefix}tao_mma_asymptote_min"] = opt.get("asymptote_min", 0.01)
    opts[f"{prefix}tao_mma_asymptote_max"] = opt.get("asymptote_max", 10.0)
    opts[f"{prefix}tao_mma_subsolver_tao_type"] = "bqnls"
    opts[f"{prefix}tao_mma_subsolver_tao_ls_type"] = "armijo"
    opts[f"{prefix}tao_mma_subsolver_tao_max_it"] = 500
    opts[f"{prefix}tao_mma_subsolver_tao_gatol"] = 1e-4
    opts[f"{prefix}tao_mma_subsolver_tao_grtol"] = 1e-4
    opts[f"{prefix}tao_mma_subsolver_tao_gttol"] = 1e-4
    tao.setFromOptions()
    tao.setMonitor(mma_iteration_monitor)

    tao.solve()

    converged_reason = tao.getConvergedReason()
    if comm.rank == 0 and converged_reason < 0:
        logger.warning(
            "TAO MMA reached reason=%d (e.g. max-iterations) for lambda=%.3g "
            "without a strict KKT stop; returning the best feasible design "
            "evaluated. Consider raising max_iter if the objective was still "
            "decreasing.", converged_reason, lambda_tradeoff,
        )

    rho_robust_local = tao.getSolution().getArray(readonly=True).copy()

    # Return the best FEASIBLE design actually evaluated. With a deterministic
    # SAA objective this is the converged iterate; it also cleanly handles a
    # max-iter stop (returns the best design instead of a possibly-infeasible
    # final step). Falls back to the raw final iterate only if no feasible
    # design was ever seen.
    if state.best_rho_local is not None:
        rho_robust_local = state.best_rho_local.copy()
        result_mu_C = state.best_mu_C
        result_sigma_C = state.best_sigma_C
        result_mean_volume = state.best_mean_volume
    else:
        final = _get_result(state, ctx, opt, rho_robust_local, xi_saa, beta)
        result_mu_C = final.mu_C
        result_sigma_C = final.sigma_C
        result_mean_volume = final.mean_volume

    # Sync rho_field/rho_phys_field to the returned design so the saved
    # XDMF/plot reflect it (density_filter.forward writes the filtered field).
    rho_field.x.petsc_vec.array[:] = rho_robust_local
    rho_field.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    density_filter.forward()

    rho_robust_global = warm_start_comm.gather(rho_robust_local)
    rho_robust_global = comm.bcast(rho_robust_global, root=0)

    S_comm = Communicator(rho_phys_field.function_space, ctx.fem["mesh_serial"])
    values = S_comm.gather(rho_phys_field)
    if comm.rank == 0:
        plotter = Plotter(ctx.fem["mesh_serial"])
        plotter.plot(values)
    save_xdmf(ctx.fem["mesh"], rho_phys_field)

    if comm.rank == 0:
        logger.info(
            "SAA lambda=%.3g done: mu_C=%.6g sigma_C=%.6g E[V]=%.6g "
            "kkt=%.4g reason=%d (%d FEA batches of N=%d)",
            lambda_tradeoff, result_mu_C, result_sigma_C, result_mean_volume,
            float(grad_vec.norm()), converged_reason, state.n_batch_evals,
            xi_saa.shape[0],
        )

    return {
        "rho_robust": rho_robust_global,
        "mu_C": result_mu_C,
        "sigma_C": result_sigma_C,
        "mean_volume": result_mean_volume,
        "kkt_residual": float(grad_vec.norm()),
        "tao_converged_reason": converged_reason,
        "n_fea_batches": state.n_batch_evals,
        "iteration_log": iteration_log,
    }
