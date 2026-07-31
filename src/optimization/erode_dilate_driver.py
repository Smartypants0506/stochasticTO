"""Three-field erode/dilate robust TO — the established baseline.

WHY THIS EXISTS
---------------
The project's contribution is a random-field threshold eta(x) optimized by
sample-average approximation at 512 FEA solves per iteration. The established
method for threshold uncertainty is Wang, Lazarov & Sigmund (SMO 2011): project
at three DETERMINISTIC thresholds -- eroded, intermediate, dilated -- and
minimize the worst of the three. Three FEA solves per iteration.

Nothing in this codebase compared against it. Without that comparison the SAA
results are unreviewable: a 171x cost multiplier has to buy something, and the
only way to show it does is to run both and report robustness AND cost side by
side. That cost ratio is itself a headline number for the paper.

FORMULATION
-----------
    min_rho   max{ C(eta_lo), C(eta_mid), C(eta_hi) }
    s.t.      V(eta_hi) <= vol_frac          (volume on the DILATED design)
              0 <= rho <= 1

Notes on the two choices that matter:

  * The volume constraint goes on the DILATED (largest) realization, following
    Wang/Lazarov/Sigmund. Constraining the intermediate design instead would let
    the dilated one exceed the budget, which is exactly the failure mode the
    formulation exists to prevent. (The SAA driver constrains E[V], a genuinely
    different -- weaker -- requirement. That difference is a real distinction
    between the methods and belongs in the comparison, not hidden by quietly
    matching the constraints up.)

  * max{} is nonsmooth. It is handled by the standard epigraph reformulation --
    minimize t subject to C_k - t <= 0 for each k -- which is smooth and is what
    MMA is built for. Taking the gradient of the active branch instead (the
    common shortcut) makes the objective discontinuous exactly where the
    branches cross, which is where the optimizer spends its time.

WHAT IS REUSED, UNCHANGED
-------------------------
  * RandomFieldHeaviside.forward(beta, eta=<float>) -- already accepts a scalar
    threshold, so no new projection code is needed.
  * setup_robust_problem / RobustProblemContext -- the same FEA machinery.
  * build_beta_schedule / measure_non_discreteness / save_design_artifacts.
  * optimality.py via the shared MMA context, so "converged" means the same
    thing here as it does for the SAA driver and the two are comparable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.fenitop.mma import MMA
from src.optimization.dolfiny_mma_driver import RobustProblemContext
from src.optimization.saa_robust_driver import (
    build_beta_schedule, measure_non_discreteness,
)

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

# Epigraph variable t is appended to the design vector. It is unbounded above in
# principle; this bound just keeps MMA's asymptote initialization well posed and
# is checked against at exit.
_T_UPPER_BOUND_FACTOR = 2.0

# Damping and cadence for the adaptive dilated-volume bound. The ratio between
# the dilated and intermediate volumes drifts as beta sharpens, so the bound has
# to follow it -- but feeding MMA a constraint bound that moves every iteration
# destabilizes its asymptote update, which is the oscillation this mechanism
# exists to remove. Update every few iterations, and damp when we do.
_VOLUME_TARGET_DAMPING = 0.3
_VOLUME_TARGET_EVERY = 5


@dataclass
class ErodeDilateState:
    """Per-rho cache shared by the objective, constraint and Jacobian callbacks,
    which MMA calls at the same design within one outer iteration."""

    outer_iteration: int = 0
    n_fea_solves: int = 0
    cached_rho: np.ndarray | None = None
    cached: dict | None = None
    history: list = field(default_factory=list)
    # Adaptive bound on the DILATED realization's volume. Initialized per stage
    # and rescaled by _update_volume_target; see that function for the rationale.
    volume_target: float = 0.0
    volume_target_history: list = field(default_factory=list)


def _update_volume_target(state: "ErodeDilateState", data: dict, vol_frac: float) -> None:
    """Rescale the dilated-volume bound so the INTERMEDIATE design meets vol_frac.

    WHY THIS IS NEEDED (Wang, Lazarov & Sigmund, SMO 2011, sec. 3.2)
    ----------------------------------------------------------------
    The volume constraint belongs on the dilated realization -- constraining the
    intermediate design would let the dilated one overrun the budget, which is
    the failure mode the three-field formulation exists to prevent. But applying
    `V_dilated <= vol_frac` LITERALLY is far more restrictive than intended: the
    dilated design is systematically larger than the intermediate one, so at
    eta_lo = 0.25 a design whose intermediate volume is the target 0.08 has a
    dilated volume of roughly 0.20. Demanding V_dilated <= 0.08 therefore
    implicitly demands an intermediate design of about 0.03 -- a structure less
    than half the intended mass.

    Measured consequence before this fix: V_dilated sat at 0.19-0.24 across every
    beta stage against a 0.08 target (violation 1.52), never satisfied, with the
    volume multiplier diverging (complementarity ~2e9).

    The standard remedy is to make the bound adaptive. Track the ratio between
    the intermediate and dilated volumes at the current design and set

        V*_dilated = vol_frac * (V_dilated / V_intermediate)

    so that satisfying the dilated constraint drives the INTERMEDIATE design to
    vol_frac -- which is the quantity the volume budget actually refers to, and
    the quantity the SAA path's E[V] constraint is comparable with.

    The update is deliberately damped and applied only every few iterations: the
    ratio moves as the design sharpens, and feeding a constraint bound that
    jumps every iteration into MMA's asymptote update produces exactly the
    oscillation this is meant to remove.
    """
    volumes = data["volumes"]
    v_dilated = float(volumes[-1])
    v_intermediate = float(volumes[1])
    if v_intermediate <= 0.0:
        return
    proposed = vol_frac * (v_dilated / v_intermediate)
    # Damped update; first call takes the proposal outright.
    if state.volume_target <= 0.0:
        state.volume_target = proposed
    else:
        state.volume_target = (
            (1.0 - _VOLUME_TARGET_DAMPING) * state.volume_target
            + _VOLUME_TARGET_DAMPING * proposed
        )
    state.volume_target_history.append(
        {"iteration": state.outer_iteration, "target": state.volume_target,
         "v_dilated": v_dilated, "v_intermediate": v_intermediate}
    )


def _evaluate_three_fields(
    ctx: RobustProblemContext, opt: dict, rho_local: np.ndarray,
    thresholds: tuple[float, float, float], beta: float,
) -> dict:
    """Solve at each of the three deterministic thresholds; return compliances,
    volumes and their gradients w.r.t. the unfiltered design variable.

    World-collective. Exactly 3 FEA solves -- contrast with the SAA driver's N.
    """
    ctx.rho_field.x.petsc_vec.array[:] = rho_local
    ctx.rho_field.x.scatter_forward()
    ctx.density_filter.forward()
    rho_tilde = ctx.rho_phys_field.x.petsc_vec.array.copy()

    compliances, volumes, dC, dV = [], [], [], []
    for eta in thresholds:
        # Reset to the filtered field: forward() overwrites rho_phys in place.
        ctx.rho_phys_field.x.petsc_vec.array[:] = rho_tilde
        ctx.rf_heaviside.forward(beta, eta=eta)
        ctx.linear_problem.solve_fem()

        (C_value, V_value, _), sensitivities = ctx.sens_problem.evaluate()
        if not np.isfinite(C_value):
            raise RuntimeError(
                f"Non-finite compliance at eta={eta:.4g}. At the eroded "
                "threshold this usually means that realization of the "
                "structure does not carry load -- a result about the design, "
                "not a solver glitch."
            )
        ctx.rf_heaviside.backward(sensitivities)
        dC_drho, dV_drho, _ = ctx.density_filter.backward(sensitivities)

        compliances.append(float(C_value))
        volumes.append(float(V_value))
        dC.append(np.asarray(dC_drho).copy())
        dV.append(np.asarray(dV_drho).copy())

    return {
        "thresholds": thresholds,
        "compliances": np.array(compliances),
        "volumes": np.array(volumes),
        "dC_drho": dC,
        "dV_drho": dV,
    }


def _get(state: ErodeDilateState, ctx, opt, rho_local, thresholds, beta) -> dict:
    """Cached evaluation. The match decision is reduced with MPI.MIN so every
    rank agrees whether to run the collective solve -- a partial call deadlocks."""
    local_match = (
        state.cached_rho is not None
        and state.cached_rho.shape == rho_local.shape
        and np.array_equal(state.cached_rho, rho_local)
    )
    if bool(comm.allreduce(1 if local_match else 0, op=MPI.MIN)):
        return state.cached

    result = _evaluate_three_fields(ctx, opt, rho_local, thresholds, beta)
    state.cached_rho = rho_local.copy()
    state.cached = result
    state.n_fea_solves += len(thresholds)
    return result


def _run_stage(
    ctx: RobustProblemContext, opt: dict, thresholds: tuple[float, float, float],
    beta: float, x0_local: np.ndarray, max_iter: int,
) -> dict:
    """One MMA solve of the epigraph problem at fixed beta."""
    n_local = ctx.n_elems_local
    n_global = ctx.n_elems_global
    col_start = ctx.col_start
    n_constraints = len(thresholds) + 1        # 3 epigraph rows + 1 volume row

    state = ErodeDilateState()

    # The epigraph variable t lives on rank 0 only, appended after the design
    # variables. Every callback therefore splits x into (design, t).
    t_local_size = 1 if comm.rank == 0 else 0

    def split(x: PETSc.Vec) -> tuple[np.ndarray, float]:
        array = x.getArray(readonly=True)
        design = np.array(array[:n_local], copy=True)
        t_value = float(array[n_local]) if t_local_size else 0.0
        return design, comm.bcast(t_value, root=0)

    def objective_gradient(tao, x, g) -> float:
        """min t~ -- the objective is the (normalized) epigraph variable itself,
        so its gradient is zero in every design direction and one in t~. All the
        physics is in the constraints."""
        _, t_value = split(x)
        grad = np.zeros(n_local + t_local_size)
        if t_local_size:
            grad[n_local] = 1.0
        g.setArray(grad)
        state.outer_iteration += 1
        return t_value

    def constraints(tao, x, c) -> None:
        """h >= 0 form (mma.py sign-flips): t~ - C_k/C_ref >= 0,
        V*_dilated - V_dilated >= 0. Compliances are divided by C_ref so the
        epigraph rows are O(1) and commensurate with the volume row.

        V*_dilated is the ADAPTIVE target (state.volume_target), not vol_frac --
        see _update_volume_target for why."""
        design, t_value = split(x)
        data = _get(state, ctx, opt, design, thresholds, beta)
        values = [t_value - C / c_ref for C in data["compliances"]]
        values.append(state.volume_target - data["volumes"][-1])  # dilated
        if comm.rank == 0:
            for i, value in enumerate(values):
                c.setValue(i, value)
        c.assemble()

    def jacobian(tao, x, J, Jp) -> None:
        design, _ = split(x)
        data = _get(state, ctx, opt, design, thresholds, beta)
        design_cols = np.arange(col_start, col_start + n_local, dtype=PETSc.IntType)
        t_col = np.array([n_global], dtype=PETSc.IntType)

        for i in range(len(thresholds)):
            # d/d_design (t~ - C_i/C_ref) = -dC_i/C_ref ; d/dt~ = 1
            J.setValues([i], design_cols,
                        (-data["dC_drho"][i] / c_ref).reshape(1, -1))
            if t_local_size:
                J.setValues([i], t_col, np.array([[1.0]]))
        volume_row = len(thresholds)
        J.setValues([volume_row], design_cols, (-data["dV_drho"][-1]).reshape(1, -1))
        if t_local_size:
            J.setValues([volume_row], t_col, np.array([[0.0]]))
        J.assemble()

    def monitor(tao) -> None:
        # Refresh the adaptive dilated-volume bound. Done here, in the monitor,
        # because it must happen between MMA outer iterations -- never inside a
        # constraint or Jacobian callback, which MMA may call several times at
        # the same design while building its subproblem. Moving the bound
        # mid-subproblem would make the constraint inconsistent with its own
        # Jacobian. Every rank runs this: state.cached is world-consistent and
        # the target must stay identical across ranks.
        iteration = tao.getIterationNumber()
        if state.cached is not None and iteration % _VOLUME_TARGET_EVERY == 0:
            _update_volume_target(state, state.cached, opt["vol_frac"])

        if comm.rank == 0 and state.cached is not None:
            C = state.cached["compliances"]
            V = state.cached["volumes"]
            logger.info(
                "[erode/dilate] iter=%d C=[%s] (worst=%.6g) V_dilated=%.6g "
                "(target %.6g, V_mid=%.6g) (%d FEA solves so far)",
                iteration, ", ".join(f"{c:.6g}" for c in C),
                C.max(), V[-1], state.volume_target, V[1], state.n_fea_solves,
            )
            state.history.append({
                "iteration": int(tao.getIterationNumber()),
                "compliances": C.tolist(),
                "worst_compliance": float(C.max()),
                "volume_dilated": float(V[-1]),
            })

    tao = PETSc.TAO().create(comm)
    tao.setType(PETSc.TAO.Type.PYTHON)
    mma_context = MMA()
    mma_context.set_constraint_scales(
        tuple([1.0] * len(thresholds) + [opt["vol_frac"]])
    )
    mma_context.set_constraint_tolerance(float(opt.get("constraint_tol", 1e-4)))
    tao.setPythonContext(mma_context)

    # Start t at the worst compliance of the initial design, so the epigraph
    # constraints t - C_k >= 0 are all feasible at iteration 0.
    # _evaluate_three_fields is WORLD-COLLECTIVE, so it is called unconditionally
    # on every rank -- guarding it behind `if rank == 0` would deadlock.
    initial_data = _evaluate_three_fields(
        ctx, opt, np.asarray(x0_local, dtype=float), thresholds, beta
    )
    state.n_fea_solves += len(thresholds)
    t_init = float(initial_data["compliances"].max())

    # Seed the adaptive dilated-volume bound from the starting design, so the
    # very first subproblem already sees a reachable constraint. Without this
    # the bound would be 0.0 on iteration 0 and every design would look grossly
    # infeasible, which is what drove the multiplier blow-up.
    _update_volume_target(state, initial_data, opt["vol_frac"])

    # WHY THE EPIGRAPH VARIABLE IS NORMALIZED.
    #
    # MMA applies its move limit as a fraction of each variable's RANGE
    # (mma.py: alpha = max(alpha, x - move_limit * x_range)). The design
    # variables live in [0,1], so move=0.02 lets them move 0.02 per iteration.
    # An un-normalized t bounded by [0, 100*t_init] has a range of 100*t_init,
    # so the SAME move limit lets it jump 0.02 * 100 * t_init = 2 * t_init per
    # iteration -- on the smoke mesh, 322 compliance units per step while the
    # compliance being minimized is only ~161. Measured dx was 322.8, matching
    # that arithmetic exactly. t then oscillates wildly, the epigraph rows
    # t - C_k >= 0 are alternately trivially slack or grossly violated, the
    # design receives no usable gradient signal, and the worst-case compliance
    # never decreases.
    #
    # Dividing t and every compliance by C_ref makes t~ = t/C_ref an O(1)
    # quantity on the same scale as the design variables, so one shared move
    # limit is meaningful for both. This is a change of variables, not of the
    # problem: the optimum is identical, and t is converted back to physical
    # units before it is reported.
    c_ref = max(t_init, 1e-30)

    # Design vector extended by the NORMALIZED epigraph variable t~ (rank 0).
    x0 = PETSc.Vec().createMPI((n_local + t_local_size, n_global + 1), comm=comm)
    initial = np.empty(n_local + t_local_size)
    initial[:n_local] = x0_local
    if t_local_size:
        initial[n_local] = t_init / c_ref          # = 1.0 by construction
    x0.setArray(initial)
    tao.setSolution(x0)

    lb = x0.copy(); ub = x0.copy()
    lb_array = np.zeros(n_local + t_local_size)
    ub_array = np.ones(n_local + t_local_size)
    if t_local_size:
        lb_array[n_local] = 0.0
        # t~ starts at 1.0 and should only ever decrease, so a modest headroom
        # factor is enough. Keeping the range O(1) is the whole point -- a large
        # factor here silently restores the runaway-step behaviour above.
        ub_array[n_local] = _T_UPPER_BOUND_FACTOR
    lb.setArray(lb_array); ub.setArray(ub_array)
    tao.setVariableBounds((lb, ub))

    grad_vec = x0.copy()
    tao.setObjectiveGradient(objective_gradient, grad_vec)

    # local_rows MUST be computed once and reused for both the constraint
    # vector and the Jacobian's row layout. createMPI(n_constraints, comm)
    # (a bare global size) lets PETSc auto-partition across every rank -- for
    # n_constraints=1 (the SAA path) that trivially lands on rank 0 and
    # happens to match the Jacobian's hardcoded "all rows on rank 0" layout,
    # but for n_constraints=4 (this driver's 3 epigraph rows + 1 volume row)
    # PETSc spreads the 4 entries across every rank, producing a LOCAL size
    # that disagrees with the Jacobian's local_rows -- e.g. rank 0 sees
    # multipliers with local dim 1 against a Jacobian with local dim 4, and
    # Mat.multTranspose in optimality.py aborts with "Nonconforming object
    # sizes". Forcing the SAME explicit (local_rows, n_constraints) layout on
    # both objects makes them agree by construction instead of by coincidence.
    local_rows = n_constraints if comm.rank == 0 else 0
    constraint_vec = PETSc.Vec().createMPI((local_rows, n_constraints), comm=comm)
    tao.setInequalityConstraints(constraints, constraint_vec)

    jacobian_mat = PETSc.Mat().createDense(
        ((local_rows, n_constraints), (n_local + t_local_size, n_global + 1)), comm=comm
    )
    jacobian_mat.setUp()
    tao.setJacobianInequality(jacobian, jacobian_mat, jacobian_mat)

    tao.setTolerances(gatol=float(opt.get("robust_opt_tol", 1e-3)))
    tao.setMaximumIterations(max_iter)

    prefix = tao.getOptionsPrefix() or ""
    opts = PETSc.Options()
    opts[f"{prefix}tao_mma_move_limit"] = opt["move"]
    opts[f"{prefix}tao_mma_subsolver_tao_type"] = "bqnls"
    opts[f"{prefix}tao_mma_subsolver_tao_ls_type"] = "armijo"
    opts[f"{prefix}tao_mma_subsolver_tao_max_it"] = 500
    tao.setFromOptions()
    tao.setMonitor(monitor)

    tao.solve()

    converged_reason = tao.getConvergedReason()
    converged = converged_reason > 0
    optimality = mma_context.optimality
    if comm.rank == 0 and not converged:
        logger.warning(
            "Erode/dilate stage (beta=%.4g) did NOT reach first-order "
            "optimality (reason=%d): %s", beta, converged_reason,
            optimality.summary() if optimality else "unavailable",
        )

    design_local, t_value = split(tao.getSolution())
    final = _get(state, ctx, opt, design_local, thresholds, beta)

    ctx.rho_field.x.petsc_vec.array[:] = design_local
    ctx.rho_field.x.scatter_forward()
    ctx.density_filter.forward()
    rho_tilde_snapshot = ctx.rho_phys_field.x.petsc_vec.array.copy()
    ctx.rf_heaviside.forward(beta, eta=0.5)
    m_nd = measure_non_discreteness(ctx.rho_phys_field)
    ctx.rho_phys_field.x.petsc_vec.array[:] = rho_tilde_snapshot
    ctx.rho_phys_field.x.scatter_forward()

    rho_global = ctx.warm_start_comm.gather(design_local)
    rho_global = comm.bcast(rho_global, root=0)

    return {
        "rho_robust": rho_global,
        "rho_robust_local": design_local,
        # t is solved for in normalized units (t~ = t/C_ref); convert back so
        # the artifact is directly comparable with `compliances` and with the
        # SAA driver's objective, neither of which is normalized.
        "epigraph_t": t_value * c_ref,
        "epigraph_t_normalized": t_value,
        "epigraph_c_ref": c_ref,
        "compliances": final["compliances"].tolist(),
        "worst_compliance": float(final["compliances"].max()),
        "volumes": final["volumes"].tolist(),
        "volume_dilated": float(final["volumes"][-1]),
        "volume_intermediate": float(final["volumes"][1]),
        "volume_target_dilated": float(state.volume_target),
        "volume_target_history": state.volume_target_history,
        # Violation is measured against the ADAPTIVE bound the solver was
        # actually given. The physically meaningful budget check is the
        # INTERMEDIATE volume against vol_frac -- reported separately, and it is
        # the number comparable with the SAA path's E[V].
        "volume_violation": max(
            0.0, float(final["volumes"][-1]) - state.volume_target
        ) / max(state.volume_target, 1e-30),
        "volume_violation_intermediate_vs_vol_frac": max(
            0.0, float(final["volumes"][1]) - opt["vol_frac"]
        ) / opt["vol_frac"],
        "beta": float(beta),
        "M_nd_percent": m_nd,
        "converged": converged,
        "optimality": optimality.as_dict() if optimality is not None else None,
        "tao_converged_reason": converged_reason,
        "n_fea_solves": state.n_fea_solves,
        "iteration_log": state.history,
    }


def run_erode_dilate_topopt(
    ctx: RobustProblemContext,
    opt: dict,
    eta_lo: float,
    eta_hi: float,
    eta_mid: float = 0.5,
    x0_local: np.ndarray | None = None,
) -> dict:
    """Erode/dilate robust design with the same beta continuation as the SAA path.

    Args:
        ctx: RobustProblemContext from setup_robust_problem().
        opt: Effective FEniTop opt dict.
        eta_lo, eta_hi: The band ends. Pass the SAME band the SAA run used
            (random_field.eta_min / eta_max) or the comparison is between two
            different problems.
        eta_mid: Intermediate threshold, conventionally 0.5.
        x0_local: Starting design, defaults to ctx.rho_warm_start_local.

    Returns:
        The final stage's result dict, plus beta_schedule, stage_results and
        n_fea_solves_total. n_fea_solves_total is the number to put next to the
        SAA run's n_fea_batches_total * N in the paper's cost comparison.
    """
    if x0_local is None:
        x0_local = ctx.rho_warm_start_local

    # eta_hi = eroded (higher threshold removes material), eta_lo = dilated.
    # Ordered lo -> mid -> hi so that index -1 is the ERODED case... but the
    # volume constraint must go on the LARGEST design, which is the DILATED one
    # at the LOWEST threshold. Order thresholds descending so index -1 is the
    # dilated (largest-volume) realization and the constraint code reads plainly.
    thresholds = (float(eta_hi), float(eta_mid), float(eta_lo))

    schedule = build_beta_schedule(
        beta_start=float(opt.get("saa_beta", 8.0)),
        beta_max=float(opt.get("saa_beta_max", opt.get("beta_max", 128.0))),
        continuation=bool(opt.get("saa_beta_continuation", True)),
    )
    total_iter = int(opt["max_iter"])
    per_stage = max(1, total_iter // len(schedule))
    budgets = [per_stage] * len(schedule)
    budgets[-1] = max(1, total_iter - per_stage * (len(schedule) - 1))

    if comm.rank == 0:
        logger.info(
            "Erode/dilate baseline: thresholds (eroded, mid, dilated) = %s, "
            "beta schedule %s, budgets %s. 3 FEA solves per iteration vs the "
            "SAA path's N -- the cost ratio is a reported result.",
            thresholds, schedule, budgets,
        )

    stage_results = []
    current_x0 = np.asarray(x0_local, dtype=float).copy()
    for index, (beta, budget) in enumerate(zip(schedule, budgets)):
        result = _run_stage(ctx, opt, thresholds, beta, current_x0, budget)
        stage_results.append({
            "stage": index,
            "beta": result["beta"],
            "worst_compliance": result["worst_compliance"],
            "volume_dilated": result["volume_dilated"],
            "M_nd_percent": result["M_nd_percent"],
            "converged": result["converged"],
            "n_fea_solves": result["n_fea_solves"],
        })
        current_x0 = np.asarray(result["rho_robust_local"], dtype=float).copy()

    final = dict(result)
    final["thresholds"] = list(thresholds)
    final["beta_schedule"] = schedule
    final["stage_results"] = stage_results
    final["n_fea_solves_total"] = int(sum(s["n_fea_solves"] for s in stage_results))
    return final
