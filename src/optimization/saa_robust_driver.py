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
# NOTE: this is a checkpointing convenience, not a licence to return an
# infeasible design silently. The realized E[V] and its violation are reported
# alongside every returned design so any slack that was actually used is visible.
_VOL_FEAS_TOL = 0.02


def build_beta_schedule(
    beta_start: float, beta_max: float, continuation: bool = True
) -> list[float]:
    """Doubling Heaviside-sharpness schedule [beta_start, 2*beta_start, ..., beta_max].

    WHY CONTINUATION IS REQUIRED HERE
    ---------------------------------
    The robust solve previously ran at a FIXED beta = 8 while the nominal SIMP
    warm-start it began from had been continued all the way to beta = 128. Two
    consequences, both fatal to the physical interpretation:

      * beta = 8 leaves a wide band of intermediate density. The reported design
        is substantially gray, so its compliance is not the compliance of any
        manufacturable structure.
      * eta only means "the threshold at which material is deemed present"
        in the sharp-projection limit. At beta = 8 the projection is a soft
        ramp and a shift in eta is not equivalent to a boundary offset, which
        is the entire physical premise of the uncertainty model.

    Running one converged solve per beta stage (rather than bumping beta inside
    a single solve) keeps each stage a well-posed optimization with its own
    convergence test, and makes the final stage -- the only one at beta_max --
    the one whose optimality is reported.

    Args:
        beta_start: Sharpness for the first stage.
        beta_max: Final sharpness; the last stage always lands exactly here.
        continuation: When False, returns [beta_max] (single stage). Use only
            for deliberate ablation -- starting cold at a large beta is a
            well-known way to lock a SIMP design into a poor local minimum.

    Returns:
        Increasing list of beta values, ending exactly at beta_max.
    """
    if beta_start <= 0 or beta_max <= 0:
        raise ValueError(
            f"beta_start and beta_max must be > 0, got {beta_start}, {beta_max}"
        )
    if not continuation:
        return [float(beta_max)]
    if beta_start >= beta_max:
        return [float(beta_max)]

    schedule = []
    beta = float(beta_start)
    while beta < beta_max:
        schedule.append(beta)
        beta *= 2.0
    schedule.append(float(beta_max))
    return schedule


def measure_non_discreteness(rho_phys_field) -> float:
    """Sigmund's measure of non-discreteness, M_nd = sum 4*rho*(1-rho) / n, in %.

    Zero for a perfectly black-and-white design, 100% for an all-gray one. A
    robust-TO paper has to report this: without it, a compliance improvement can
    come from nothing more than leaving material at intermediate density, which
    is not a manufacturable structure and is exactly what a fixed low beta
    encourages.

    Computed over the CG1 physical-density dofs (which is where the projection
    acts in this code), reduced across ranks so the result is world-identical.
    """
    local = np.asarray(rho_phys_field.x.petsc_vec.array, dtype=float)
    local_sum = float(np.sum(4.0 * local * (1.0 - local)))
    local_count = int(local.size)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    return 100.0 * global_sum / global_count if global_count else float("nan")


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
    xi_saa: np.ndarray, beta: float, accumulate_gradients: bool = True,
) -> RobustEvaluationResult:
    """Run the fixed SAA batch of N FEA solves at rho_local and package the
    sample-average stats + gradient reductions as a RobustEvaluationResult.

    accumulate_gradients defaults to True: the robust objective only ever needs
    three reductions of the per-sample gradients, so building the
    [N x n_elems_local] matrices -- and, in the sample-parallel path,
    broadcasting a full global array twice per sample -- was pure overhead. The
    values produced are identical either way; pass False to materialize the
    rows (used by tests comparing the two paths).

    World-collective: every rank must call this together (the batch solve is
    MPI-collective, sample-parallel when ctx.group is set).

    NOTE ON THE ESTIMATOR. sigma_C below is the ddof=1 sample standard
    deviation. It is the EXACT standard deviation of the fixed sample set, and
    the gradient chained from it is the exact gradient of that deterministic
    quantity -- that part is not an approximation.

    When xi_saa comes from LHS (the default) the samples are negatively
    correlated, so this is not an unbiased estimator of the underlying sigma.
    The direction and size of that bias are both known. For any design whose
    strata reproduce the marginal exactly,

        E[s^2] = (N/(N-1)) ( sigma^2 - Var(xbar) )

    LHS exists to make Var(xbar) smaller than the iid sigma^2/N, so less is
    subtracted and the estimator is biased UPWARD, bounded by the
    perfect-stratification limit:

        1 <= E[s^2]/sigma^2 <= N/(N-1)

    At N=512 that caps the variance bias at 0.196% and the sigma bias at 0.098%
    -- far below any difference this pipeline reports, so the LHS default is
    fine at production N. It is NOT negligible at small N (6.7% in variance at
    N=16), which is why the study configs use monte_carlo sampling. See
    tests/test_robust_statistics.py for the derivation as an executable check.
    """
    if ctx.group is not None:
        data = run_fea_at_samples_grouped(
            ctx, ctx.group, opt, rho_local, xi_saa, beta,
            accumulate_gradients=accumulate_gradients,
        )
    else:
        data = run_fea_at_samples(
            ctx.fem, opt, rho_local, ctx.density_filter, ctx.rf_heaviside,
            ctx.sens_problem, xi_saa, beta, ctx.linear_problem, ctx.rho_field,
            accumulate_gradients=accumulate_gradients,
        )

    C = data.compliance_samples
    V = data.volume_samples
    mu_C = float(C.mean())

    # The serial path centers on its own in-batch reference; re-center on the
    # true batch mean here using the same exact identity the grouped path uses.
    dC_centered_sum = data.dC_centered_sum
    if dC_centered_sum is not None and data.C_reference is not None:
        dC_centered_sum = dC_centered_sum + (data.C_reference - mu_C) * data.dC_sum

    return RobustEvaluationResult(
        compliance_samples=C,
        volume_samples=V,
        dC_drho_samples=data.dC_drho_samples,
        dV_drho_samples=data.dV_drho_samples,
        dC_sum=data.dC_sum,
        dC_centered_sum=dC_centered_sum,
        dV_sum=data.dV_sum,
        mu_C=mu_C,
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


def _run_saa_stage(
    ctx: RobustProblemContext,
    opt: dict,
    lambda_tradeoff: float,
    xi_saa: np.ndarray,
    beta: float,
    x0_local: np.ndarray,
    max_iter: int,
    move_limit: float | None = None,
) -> dict:
    """One TAO MMA solve for ONE lambda at ONE fixed Heaviside sharpness beta,
    against the EXACT sample-average robust objective (no surrogate).

    Called once per beta stage by run_saa_robust_topopt(); see
    build_beta_schedule() for why the continuation runs as separate solves
    rather than as a beta bump inside a single solve.

    Args:
        ctx: RobustProblemContext from setup_robust_problem() (FEA machinery +
            optional sample-parallel GroupFEAContext + warm-start).
        opt: FEniTop opt dict (needs vol_frac, move, opt_tol, ...).
        lambda_tradeoff: mean-variance weight in J = mu_C + lambda*sigma_C.
        xi_saa: [N x n_kl] FIXED KL-coefficient sample set, identical on every
            rank, reused at every iteration (SAA / common random numbers).
        beta: Heaviside sharpness for this stage's batch solves.
        x0_local: THIS RANK's local slice of the starting design.
        max_iter: Iteration budget for this stage.

    Returns:
        dict with rho_robust (global), rho_robust_local, mu_C, sigma_C,
        mean_volume, converged, optimality, grad_norm, tao_converged_reason,
        n_fea_batches, iteration_log, beta, M_nd, volume_violation.
    """
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
    mma_context = MMA()
    # The volume constraint is normalized by vol_frac, so the reported
    # feasibility/complementarity residuals read as a FRACTION of the volume
    # budget rather than an absolute volume fraction.
    mma_context.set_constraint_scales((opt["vol_frac"],))
    mma_context.set_constraint_tolerance(float(opt.get("constraint_tol", 1e-4)))
    tao.setPythonContext(mma_context)

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

    # gatol now governs the RELATIVE stationarity residual (dimensionless), not
    # the raw objective-gradient norm -- see src/optimization/optimality.py.
    # robust_opt_tol, NOT opt_tol: the latter is the nominal Stage-2 OC loop's
    # design-CHANGE threshold and means something entirely different.
    tao.setTolerances(gatol=float(opt.get("robust_opt_tol", 1.0e-3)))
    tao.setMaximumIterations(max_iter)

    prefix = tao.getOptionsPrefix() or ""
    opts = PETSc.Options()
    opts[f"{prefix}tao_mma_move_limit"] = (
        opt["move"] if move_limit is None else float(move_limit)
    )
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
    optimality = mma_context.optimality
    # reason > 0 is a genuine TAO convergence code. reason == 0
    # (TAO_CONTINUE_ITERATING) means the solve stopped mid-flight and is NOT a
    # success -- mma.py now converts an exhausted budget into DIVERGED_MAXITS,
    # but any other path leaving reason == 0 must still be treated as failure.
    converged = converged_reason > 0
    if comm.rank == 0 and not converged:
        logger.warning(
            "TAO MMA did NOT reach first-order optimality for lambda=%.3g "
            "(reason=%d). Returning the best feasible design evaluated; it is "
            "NOT a converged optimum and must not be reported as one. "
            "Final optimality: %s",
            lambda_tradeoff, converged_reason,
            optimality.summary() if optimality is not None else "unavailable",
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

    # Discreteness of the design actually being returned. rho_phys_field was
    # just refreshed from rho_robust_local by density_filter.forward(), but the
    # Heaviside projection has NOT been re-applied to it, so project at this
    # stage's beta before measuring -- M_nd of the unprojected filtered field
    # would understate how black-and-white the design really is.
    rho_tilde_snapshot = rho_phys_field.x.petsc_vec.array.copy()
    ctx.rf_heaviside.forward(beta, eta=0.5)
    m_nd = measure_non_discreteness(rho_phys_field)
    rho_phys_field.x.petsc_vec.array[:] = rho_tilde_snapshot
    rho_phys_field.x.scatter_forward()

    volume_violation = max(0.0, result_mean_volume - opt["vol_frac"]) / opt["vol_frac"]

    if comm.rank == 0:
        logger.info(
            "SAA lambda=%.3g beta=%.4g done: mu_C=%.6g sigma_C=%.6g E[V]=%.6g "
            "(violation %.3g%%) M_nd=%.3g%% converged=%s reason=%d "
            "(%d FEA batches of N=%d) | %s",
            lambda_tradeoff, beta, result_mu_C, result_sigma_C, result_mean_volume,
            100.0 * volume_violation, m_nd, converged, converged_reason,
            state.n_batch_evals, xi_saa.shape[0],
            optimality.summary() if optimality is not None else "no optimality record",
        )

    return {
        "rho_robust": rho_robust_global,
        "rho_robust_local": rho_robust_local,
        "mu_C": result_mu_C,
        "sigma_C": result_sigma_C,
        "mean_volume": result_mean_volume,
        # Relative amount of the E[V] <= vol_frac budget the returned design
        # actually overshoots. The best-design checkpoint tolerates a small
        # overshoot (_VOL_FEAS_TOL); reporting the realized number means that
        # slack can never pass unnoticed as "feasible".
        "volume_violation": volume_violation,
        "beta": float(beta),
        "M_nd_percent": m_nd,
        # Real first-order optimality evidence. `grad_norm` is the quantity the
        # previous version of this driver reported as "kkt_residual"; it is kept
        # only so old runs stay comparable, and is NOT an optimality measure for
        # this constrained problem (see src/optimization/optimality.py).
        "converged": converged,
        "optimality": optimality.as_dict() if optimality is not None else None,
        "grad_norm": float(grad_vec.norm()),
        "tao_converged_reason": converged_reason,
        "n_fea_batches": state.n_batch_evals,
        "iteration_log": iteration_log,
    }


def run_saa_robust_topopt(
    ctx: RobustProblemContext,
    opt: dict,
    lambda_tradeoff: float,
    xi_saa: np.ndarray,
    beta: float | None = None,
    x0_local: np.ndarray | None = None,
) -> dict:
    """Robust SAA solve for ONE lambda, with Heaviside continuation.

    Runs one converged _run_saa_stage() per beta in build_beta_schedule(),
    warm-starting each stage from the previous one and splitting the total
    iteration budget across stages (so continuation costs no extra FEA relative
    to the previous fixed-beta behaviour). The result reported is the FINAL
    stage's -- the only one at beta_max, and therefore the only one whose
    design is near-discrete enough for the eta-as-boundary-offset
    interpretation to hold.

    Args:
        ctx: RobustProblemContext from setup_robust_problem().
        opt: FEniTop opt dict. Reads saa_beta (schedule start), saa_beta_max
            (schedule end, defaults to beta_max), saa_beta_continuation,
            and max_iter (TOTAL budget across all stages).
        lambda_tradeoff: mean-variance weight in J = mu_C + lambda*sigma_C.
        xi_saa: [N x n_kl] FIXED sample set (common random numbers).
        beta: Overrides the schedule with a single fixed beta. Use only for
            deliberate ablation.
        x0_local: THIS RANK's local slice of the starting design (defaults to
            ctx.rho_warm_start_local).

    Returns:
        The final stage's result dict, extended with `beta_schedule` and
        `stage_results` (one entry per stage) so the continuation history is
        recoverable from the artifacts.
    """
    if x0_local is None:
        x0_local = ctx.rho_warm_start_local

    if beta is not None:
        schedule = [float(beta)]
    else:
        schedule = build_beta_schedule(
            beta_start=float(opt.get("saa_beta", 8.0)),
            beta_max=float(opt.get("saa_beta_max", opt.get("beta_max", 128.0))),
            continuation=bool(opt.get("saa_beta_continuation", True)),
        )

    n_stages = len(schedule)

    # MOVE-LIMIT CONTINUATION.
    #
    # Measured problem this exists to fix: with a FIXED move limit the design
    # change dx sits at exactly the move limit on every single iteration of
    # every stage, i.e. the MMA subproblem solution is permanently on the
    # trust-region boundary. The relative stationarity residual then falls for
    # ~6 iterations and plateaus, oscillating (0.080 -> 0.134 -> 0.124 in the
    # beta=128 stage of a study-mesh run) instead of decaying. It never
    # approaches robust_opt_tol = 1e-3; the gap study's seven field designs land
    # at 0.038-0.106. More iterations do not help -- the iterate is not
    # converging, it is stepping the maximum distance forever.
    #
    # Shrinking the move limit as beta sharpens lets the iterate come off that
    # boundary. Tying the reduction to the beta schedule is the principled
    # choice: the projection's responsive band is ~19/beta wide, so it halves
    # every time beta doubles, and a step that was appropriate at beta=8 is
    # far too large at beta=128.
    #
    # DEFAULT IS 1.0 = INERT. Enabling this changes the designs every run
    # produces, so it stays off until a controlled comparison justifies it.
    move_reduction = float(opt.get("move_reduction", 1.0))
    if not 0.0 < move_reduction <= 1.0:
        raise ValueError(
            f"move_reduction must be in (0, 1], got {move_reduction}. Values "
            "above 1 would GROW the move limit as beta sharpens, which is the "
            "wrong direction."
        )
    base_move = float(opt["move"])
    move_floor = float(opt.get("move_min", 1.0e-3))
    move_schedule = [
        max(base_move * (move_reduction ** k), move_floor) for k in range(n_stages)
    ]

    total_iter = int(opt["max_iter"])
    # Split the budget evenly; the final stage absorbs the remainder because it
    # is the one whose converged design is reported.
    per_stage = max(1, total_iter // n_stages)
    budgets = [per_stage] * n_stages
    budgets[-1] = max(1, total_iter - per_stage * (n_stages - 1))

    if comm.rank == 0:
        logger.info(
            "SAA lambda=%.3g: beta continuation over %d stage(s) %s with "
            "iteration budgets %s (total %d)",
            lambda_tradeoff, n_stages, schedule, budgets, total_iter,
        )

    stage_results = []
    current_x0 = np.asarray(x0_local, dtype=float).copy()
    for stage_index, (stage_beta, stage_budget, stage_move) in enumerate(
        zip(schedule, budgets, move_schedule)
    ):
        if comm.rank == 0:
            logger.info(
                "SAA lambda=%.3g stage %d/%d: beta=%.4g, max_iter=%d, move=%.4g",
                lambda_tradeoff, stage_index + 1, n_stages, stage_beta, stage_budget,
                stage_move,
            )
        result = _run_saa_stage(
            ctx, opt, lambda_tradeoff, xi_saa, stage_beta, current_x0, stage_budget,
            move_limit=stage_move,
        )
        stage_results.append(
            {
                "stage": stage_index,
                "beta": result["beta"],
                "max_iter": stage_budget,
                "mu_C": result["mu_C"],
                "sigma_C": result["sigma_C"],
                "mean_volume": result["mean_volume"],
                "volume_violation": result["volume_violation"],
                "M_nd_percent": result["M_nd_percent"],
                "move_limit": stage_move,
                "converged": result["converged"],
                "optimality": result["optimality"],
                "n_fea_batches": result["n_fea_batches"],
            }
        )
        current_x0 = np.asarray(result["rho_robust_local"], dtype=float).copy()

    # `result` is the final stage's -- the only one solved at beta_max.
    final = dict(result)
    final["beta_schedule"] = schedule
    final["move_schedule"] = move_schedule
    final["move_reduction"] = move_reduction
    final["stage_results"] = stage_results
    final["n_fea_batches_total"] = int(sum(s["n_fea_batches"] for s in stage_results))
    return final


def save_design_artifacts(ctx: RobustProblemContext, path_prefix: str) -> None:
    """Write the current design's XDMF + preview image under an EXPLICIT prefix.

    This used to happen inside the solve, via save_xdmf()/Plotter.plot() called
    with their default path="" -- which writes optimized_design.xdmf/.jpg into
    the CURRENT WORKING DIRECTORY. Every lambda in a sweep therefore overwrote
    the previous lambda's artifacts, and the results landed in the repository
    root rather than under the run's own output directory. Callers now pass a
    per-lambda, per-run prefix so each design's artifacts survive.

    Collective: the gather runs on every rank; only rank 0 writes.

    Args:
        ctx: RobustProblemContext whose rho_phys_field holds the design to save.
        path_prefix: Prefix for the output files, e.g.
            "output/stage5_optimization/<run_id>/lambda_0.0_". The parent
            directory must already exist.
    """
    phys_comm = Communicator(ctx.rho_phys_field.function_space, ctx.fem["mesh_serial"])
    values = phys_comm.gather(ctx.rho_phys_field)
    if comm.rank == 0:
        Plotter(ctx.fem["mesh_serial"]).plot(values, path=path_prefix)
    save_xdmf(ctx.fem["mesh"], ctx.rho_phys_field, path=path_prefix)
