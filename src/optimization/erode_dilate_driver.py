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
_T_UPPER_BOUND_FACTOR = 100.0


@dataclass
class ErodeDilateState:
    """Per-rho cache shared by the objective, constraint and Jacobian callbacks,
    which MMA calls at the same design within one outer iteration."""

    outer_iteration: int = 0
    n_fea_solves: int = 0
    cached_rho: np.ndarray | None = None
    cached: dict | None = None
    history: list = field(default_factory=list)


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
        """min t -- the objective is the epigraph variable itself, so its
        gradient is zero in every design direction and one in t. All the
        physics is in the constraints."""
        _, t_value = split(x)
        grad = np.zeros(n_local + t_local_size)
        if t_local_size:
            grad[n_local] = 1.0
        g.setArray(grad)
        state.outer_iteration += 1
        return t_value

    def constraints(tao, x, c) -> None:
        """h >= 0 form (mma.py sign-flips): t - C_k >= 0, vol_frac - V_dilated >= 0."""
        design, t_value = split(x)
        data = _get(state, ctx, opt, design, thresholds, beta)
        values = [t_value - C for C in data["compliances"]]
        values.append(opt["vol_frac"] - data["volumes"][-1])   # dilated realization
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
            # d/d_design (t - C_i) = -dC_i ; d/dt (t - C_i) = 1
            J.setValues([i], design_cols, (-data["dC_drho"][i]).reshape(1, -1))
            if t_local_size:
                J.setValues([i], t_col, np.array([[1.0]]))
        volume_row = len(thresholds)
        J.setValues([volume_row], design_cols, (-data["dV_drho"][-1]).reshape(1, -1))
        if t_local_size:
            J.setValues([volume_row], t_col, np.array([[0.0]]))
        J.assemble()

    def monitor(tao) -> None:
        if comm.rank == 0 and state.cached is not None:
            C = state.cached["compliances"]
            V = state.cached["volumes"]
            logger.info(
                "[erode/dilate] iter=%d C=[%s] (worst=%.6g) V_dilated=%.6g "
                "(%d FEA solves so far)",
                tao.getIterationNumber(), ", ".join(f"{c:.6g}" for c in C),
                C.max(), V[-1], state.n_fea_solves,
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

    # Design vector extended by the epigraph variable t (rank 0 owns it).
    x0 = PETSc.Vec().createMPI((n_local + t_local_size, n_global + 1), comm=comm)
    initial = np.empty(n_local + t_local_size)
    initial[:n_local] = x0_local
    if t_local_size:
        initial[n_local] = t_init
    x0.setArray(initial)
    tao.setSolution(x0)

    lb = x0.copy(); ub = x0.copy()
    lb_array = np.zeros(n_local + t_local_size)
    ub_array = np.ones(n_local + t_local_size)
    if t_local_size:
        lb_array[n_local] = 0.0
        ub_array[n_local] = _T_UPPER_BOUND_FACTOR * max(t_init, 1e-30)
    lb.setArray(lb_array); ub.setArray(ub_array)
    tao.setVariableBounds((lb, ub))

    grad_vec = x0.copy()
    tao.setObjectiveGradient(objective_gradient, grad_vec)

    constraint_vec = PETSc.Vec().createMPI(n_constraints, comm=comm)
    tao.setInequalityConstraints(constraints, constraint_vec)

    local_rows = n_constraints if comm.rank == 0 else 0
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
        "epigraph_t": t_value,
        "compliances": final["compliances"].tolist(),
        "worst_compliance": float(final["compliances"].max()),
        "volumes": final["volumes"].tolist(),
        "volume_dilated": float(final["volumes"][-1]),
        "volume_violation": max(
            0.0, float(final["volumes"][-1]) - opt["vol_frac"]
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
