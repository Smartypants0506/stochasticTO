"""
src/optimization/dolfiny_mma_driver.py


Stage 5 (Robust Topology Optimization Loop) -- masterContext Section 3.5.


Wires FEniTop's fixed-mesh FEA/sensitivity machinery, the randomized
Heaviside projection (topology/heaviside_projection_glue.py), and the
compliance+volume PCE surrogate pair (surrogate/pce_model.py,
optimization/pce_evaluation.py) into dolfiny's native PETSc TAO MMA
algorithm (mma-13.py's MMA class), per the "never reimplement MMA" rule.


This module does NOT reuse FEniTop's own topopt.py while-loop or its
mma_optimizer()/optimality_criteria() functions -- those are the Stage-2
nominal-SIMP driver, retained only as the warm-start source. This is a
distinct, TAO-native outer loop for the robust (PCE-driven, mean-variance
scalarized) objective, per masterContext's replacement of the OpenMDAO/
PyOptSparse/ParOpt stack with dolfiny+TAO.


MPI note: this loop is world-collective. rho_field/x/g and every per-rank
array (rho_current, dJ_drho, dh_drho) are THIS RANK's LOCAL dof/element
slice, consistent with fea_at_samples.py's local-sizing convention. The
single global inequality constraint (Vfrac - E[V] >= 0) and its Jacobian
row, however, are GLOBAL, size-1 objects spanning the whole communicator --
they must be built with parallel-aware PETSc constructors (createMPI /
createDense with explicit local/global sizes), never createSeq, since
createSeq requires a size-1 communicator and will raise immediately under
`mpirun -n 64`.


Warm-start note: rho_warm_start, as produced by main.py, is now a GLOBAL
array (topopt.py's own final save was fixed to gather rho_field to a true
global array before np.save, and main.py broadcasts that identical global
array to every rank via comm.bcast). This module must therefore SCATTER it
down to each rank's local dof slice before use -- via the same
Communicator.bcast() pattern utility.py already uses for gathering -- not
assign it directly into a local PETSc array.
"""
from __future__ import annotations


import logging
from dataclasses import dataclass, field


import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter
from src.fenitop.sensitivity import Sensitivity
from src.fenitop.utility import Communicator, Plotter, save_xdmf
from src.fenitop.mma import MMA  # dolfiny-style native TAO MMA (mma-13.py)


from src.random_fields.kernel import KernelParams
from src.random_fields.threshold_transform import MarginalTransformParams
from src.random_fields.kl_expansion import KLExpansionResult
from src.topology.heaviside_projection_glue import (
    RandomFieldHeaviside,
    RandomHeavisideConfig,
    build_random_heaviside_from_function_space,
)
from src.sampling.sampler import generate_train_test_samples
from src.surrogate.fea_at_samples import run_fea_at_samples
from src.surrogate.pce_builder import build_pce_surrogate
from src.surrogate.pce_model import build_pce_gradient_model
from src.optimization.pce_evaluation import (
    PCERefreshPolicy,
    evaluate_from_pce,
    get_pce_robust_gradient,
    get_pce_volume_gradient,
)
from src.optimization.robust_objective import RobustObjectiveConfig


logger = logging.getLogger(__name__)



@dataclass
class RobustLoopState:
    """Mutable state threaded through TAO callbacks (TAO callbacks are stateless
    functions; this class holds everything that must persist across calls).


    Attributes:
        outer_iteration: Number of objective/gradient evaluations so far.
            Drives both the beta continuation schedule and the PCE
            refresh policy -- NOT the same as TAO's internal MMA
            iteration count, since TAO may call the objective/gradient
            evaluator more than once per outer MMA step (e.g. during
            line search in the dual subsolve).
        beta: Current Heaviside sharpness parameter.
        compliance_pce: Currently valid compliance PCEGradientModel, or
            None before the first training call.
        volume_pce: Currently valid volume PCEGradientModel, or None
            before the first training call.
        refresh_policy: Governs when compliance_pce/volume_pce are rebuilt.
    """
    outer_iteration: int = 0
    beta: float = 1.0
    compliance_pce: object = None
    volume_pce: object = None
    refresh_policy: PCERefreshPolicy = field(default_factory=PCERefreshPolicy)



def _retrain_pce_pair(
    fem: dict, 
    opt: dict, 
    rho_current: np.ndarray, 
    density_filter: DensityFilter,
    rf_heaviside: RandomFieldHeaviside, 
    sens_problem: Sensitivity, 
    beta: float,
    kl_result: KLExpansionResult, 
    linear_problem, 
    rho_field,
) -> tuple:
    """Retrain the compliance+volume PCE pair at the current design iterate.


    This is the expensive path (opt["pce_n_train"] FEA solves). Called only
    when RobustLoopState.refresh_policy.needs_refresh() is True.


    Args:
        fem: FEniTop fem dict.
        opt: FEniTop opt dict; must contain "pce_n_train", "n_kl",
            "pce_hyperbolic_q", "pce_max_degree", "pce_q2_threshold",
            "kl_model" (fitted KLModel from Stage 3), and "vol_frac".
        rho_current: [n_elems_local] design density at which to train (the
            optimizer's current iterate -- held fixed during training).
            THIS RANK's local slice, matching rho_field's local dof array.
        density_filter: FEniTop's DensityFilter, reused for training solves.
        rf_heaviside: RandomFieldHeaviside instance, reused for training solves.
        sens_problem: FEniTop's Sensitivity instance, reused for training solves.
        beta: Current Heaviside sharpness parameter -- training solves must
            use the SAME beta as the outer loop's current continuation
            stage, or the PCE will not reflect the optimizer's actual
            current projection sharpness.


    Returns:
        (compliance_pce_model, volume_pce_model) tuple of PCEGradientModel.


    Raises:
        RuntimeError: If either PCE fails its Q^2 >= threshold gate --
            this is a hard verification gate per masterContext Section 7
            and must never be bypassed, even mid-optimization.
    """


    n_train = opt["pce_n_train"]


    logger.info(
        "Retraining PCE pair at outer_iteration checkpoint: n_train=%d, beta=%.3g",
        n_train, beta,
    )


    train_set, test_set = generate_train_test_samples(
    kl_result, n_train=n_train, n_test=opt["pce_n_test"],
    seed=opt.get("pce_seed", 0),
)
    xi_train, xi_test = train_set.xi, test_set.xi


    training_data = run_fea_at_samples(
    fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi_train, beta,
    linear_problem, rho_field,
)
    test_data = run_fea_at_samples(
    fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi_test, beta,
    linear_problem, rho_field,
)


    compliance_pce_result = build_pce_surrogate(
        xi_train, training_data.compliance_samples,
        xi_test, test_data.compliance_samples,
        hyperbolic_q=opt["pce_hyperbolic_q"],
        max_degree_attempts=opt["pce_max_degree_attempts"],
    )


    if compliance_pce_result.q2 < opt["pce_q2_threshold"]:
        raise RuntimeError(
            f"Compliance PCE failed Q^2 gate: Q^2={compliance_pce_result.q2:.4f} "
            f"< threshold={opt['pce_q2_threshold']}."
        )


    volume_pce_result = build_pce_surrogate(
        xi_train, training_data.volume_samples,
        xi_test, test_data.volume_samples,
        hyperbolic_q=opt["pce_hyperbolic_q"],
        max_degree_attempts=opt["pce_max_degree_attempts"],
    )
    if volume_pce_result.q2 < opt["pce_q2_threshold"]:
        raise RuntimeError(
            f"Volume PCE failed Q^2 gate: Q^2={volume_pce_result.q2:.4f} "
            f"< threshold={opt['pce_q2_threshold']}."
        )


    compliance_pce_model = build_pce_gradient_model(
        compliance_pce_result, xi_train, training_data.dC_drho_samples,
    )
    volume_pce_model = build_pce_gradient_model(
        volume_pce_result, xi_train, training_data.dV_drho_samples,
    )


    logger.info(
        "PCE pair retrained: compliance Q^2=%.4f, volume Q^2=%.4f",
        compliance_pce_result.q2, volume_pce_result.q2,
    )
    return compliance_pce_model, volume_pce_model



def run_robust_topopt(
    fem: dict,
    opt: dict,
    rho_warm_start: np.ndarray,
    lambda_tradeoff: float,
    kl_result: "KLExpansionResult",
) -> dict:
    """Run the dolfiny-MMA-driven robust topology optimization loop for one lambda.


    Args:
        fem: FEniTop fem dict (mesh, material, BCs, load cases).
        opt: FEniTop opt dict, extended with PCE/robust-loop keys:
            "pce_n_train", "n_kl", "pce_hyperbolic_q", "pce_max_degree",
            "pce_q2_threshold", "pce_refresh_interval", "beta_interval",
            "beta_max", "vol_frac", "opt_tol", "max_iter", "filter_radius".
        rho_warm_start: [n_elems_GLOBAL] converged nominal SIMP design from
            FEniTop's own topopt.py run -- required warm start, per
            masterContext Section 3.5's "starts from nominal FEniTop SIMP
            solution as warm start". Must be a full GLOBAL array, IDENTICAL
            on every rank (e.g. loaded on rank 0 and comm.bcast'd by the
            caller) -- this function scatters it internally to each rank's
            local dof slice via Communicator.bcast().
        lambda_tradeoff: Mean-variance scalarization weight for this run
            (one point on the Pareto sweep).


    Returns:
        dict with "rho_robust" (converged density field), "mu_C", "sigma_C",
        "mean_volume", "kkt_residual", "tao_converged_reason",
        "iteration_log" (list of per-outer-iteration dicts for CSV export).


    Raises:
        RuntimeError: If TAO reports a non-converged reason at the end of
            the solve, or if any PCE retrain fails its Q^2 gate.
    """
    comm = MPI.COMM_WORLD
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(
        comm, rho_field, rho_phys_field, opt["filter_radius"], fem["petsc_options"]
    )
    random_heaviside_config = RandomHeavisideConfig(
        kernel_params=opt["kernel_params"],
        transform_params=opt["transform_params"],
        variance_threshold=opt.get("kl_variance_threshold", 0.95),
        seed=opt.get("random_field_seed"),
    )
    rf_heaviside = build_random_heaviside_from_function_space(
    rho_phys_field, kl_result, random_heaviside_config,
    )
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)


    n_elems_local = rho_field.x.petsc_vec.array.size
    index_map = rho_field.function_space.dofmap.index_map
    n_elems_global = index_map.size_global
    col_start = index_map.local_range[0]

    if rho_warm_start.size != n_elems_global:
        raise ValueError(
            f"rho_warm_start has {rho_warm_start.size} entries but the design "
            f"space has {n_elems_global} GLOBAL dofs. rho_warm_start must be "
            "a full global array (identical on every rank), not a local slice "
            "-- see this function's docstring."
        )

    # Scatter the global warm-start array into this rank's local dof slice,
    # using the same Communicator.bcast() pattern utility.py's Communicator
    # already uses (mirror image of its gather() used at the end below).
    warm_start_comm = Communicator(rho_field.function_space, fem["mesh_serial"])
    warm_start_comm.bcast(rho_field, rho_warm_start)
    rho_warm_start_local = rho_field.x.petsc_vec.array.copy()


    robust_config = RobustObjectiveConfig(lambda_tradeoff=lambda_tradeoff)
    state = RobustLoopState(
        beta=1.0,
        refresh_policy=PCERefreshPolicy(refresh_interval=opt["pce_refresh_interval"]),
    )
    iteration_log: list[dict] = []


    def objective_gradient_callback(tao: PETSc.TAO, x: PETSc.Vec, g: PETSc.Vec) -> float:
        """TAO objective+gradient callback for the robust scalarized objective J."""
        rho_current = x.getArray(readonly=True).copy()


        if (state.outer_iteration % opt["beta_interval"] == 0
                and state.beta < opt["beta_max"] and state.outer_iteration > 0):
            state.beta = min(state.beta * 2, opt["beta_max"])
            logger.info("Beta continuation: increased to %.3g at outer_iteration=%d",
                         state.beta, state.outer_iteration)


        density_filter.forward()  # rho -> rho_tilde (deterministic Helmholtz filter)
        # NOTE: rho_phys itself is set INSIDE run_fea_at_samples/robust evaluation
        # via rf_heaviside per-sample; the design variable driving the filter here
        # is rho_current, consistent with the trained PCE's rho_nominal only if
        # a refresh has just occurred -- enforced below.


        if state.refresh_policy.needs_refresh(state.outer_iteration):
            state.compliance_pce, state.volume_pce = _retrain_pce_pair(
    fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, state.beta, kl_result,
    linear_problem, rho_field)
            state.refresh_policy.last_refresh_iteration = state.outer_iteration


        result = evaluate_from_pce(state.compliance_pce, state.volume_pce, n_elems_local)
        J_value = result.mu_C + lambda_tradeoff * result.sigma_C
        dJ_drho = get_pce_robust_gradient(state.compliance_pce, robust_config)
        g.setArray(dJ_drho)


        iteration_log.append({
            "outer_iteration": state.outer_iteration,
            "beta": state.beta,
            "J": J_value,
            "mu_C": result.mu_C,
            "sigma_C": result.sigma_C,
            "mean_volume": result.mean_volume,
        })
        state.outer_iteration += 1
        return J_value


    def inequality_constraint_callback(tao: PETSc.TAO, x: PETSc.Vec, c: PETSc.Vec) -> None:
        """TAO inequality constraint callback: h(rho) = Vfrac - E[V] >= 0."""
        if state.volume_pce is None:
            raise RuntimeError(
                "Volume PCE not yet trained; objective_gradient_callback must "
                "run at least once before the constraint callback."
            )
        h_value = opt["vol_frac"] - state.volume_pce.mu_C
        if comm.rank == 0:
            c.setValue(0, h_value)
        c.assemble()


    def jacobian_inequality_callback(
        tao: PETSc.TAO, x: PETSc.Vec, J: PETSc.Mat, Jp: PETSc.Mat,
    ) -> None:
        """TAO constraint Jacobian callback: dh/drho = -dE[V]/drho.


        dh_drho is THIS RANK's local slice of the gradient (matching
        rho_field's local dof partitioning). It is written into the single
        global constraint row (global row index 0) at this rank's owned
        GLOBAL column range [col_start, col_start + n_elems_local) -- using
        LOCAL indices here (as the original code did) would make every
        rank overwrite the same columns instead of each rank contributing
        its own distinct slice of the design vector.
        """
        dh_drho = -get_pce_volume_gradient(state.volume_pce)
        global_cols = np.arange(col_start, col_start + n_elems_local, dtype=PETSc.IntType)
        J.setValues([0], global_cols, dh_drho.reshape(1, -1))
        J.assemble()


    tao = PETSc.TAO().create(comm)
    tao.setType(PETSc.TAO.Type.PYTHON)
    tao.setPythonContext(MMA())


    x0 = rho_field.x.petsc_vec.copy()
    x0.setArray(rho_warm_start_local)
    tao.setSolution(x0)


    lb = x0.copy(); lb.set(0.0)
    ub = x0.copy(); ub.set(1.0)
    tao.setVariableBounds((lb, ub))


    grad_vec = x0.copy()
    tao.setObjectiveGradient(objective_gradient_callback, grad_vec)


    # NOTE: createSeq requires a size-1 communicator and will raise under
    # `mpirun -n 64`; the single global constraint must instead be a proper
    # distributed (global size 1) PETSc vector, matching jacobian_mat's row.
    constraint_vec = PETSc.Vec().createMPI(1, comm=comm)
    tao.setInequalityConstraints(inequality_constraint_callback, constraint_vec)


    # NOTE: (1, n_elems) previously used n_elems_local as if it were the
    # GLOBAL matrix size, and every rank passed a DIFFERENT value for what
    # PETSc requires to be a collectively-agreed global size -- this would
    # either error immediately or silently build a mis-sized matrix.
    # createDense's size tuple below is ((local_rows, global_rows),
    # (local_cols, global_cols)); only rank 0 owns the single global row.
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
    opts[f"{prefix}tao_mma_move_limit"] = opt["move"]          # was silently defaulting to 0.5
    opts[f"{prefix}tao_mma_asymptote_init"] = opt.get("asymptote_init", 0.5)
    opts[f"{prefix}tao_mma_asymptote_min"] = opt.get("asymptote_min", 0.01)
    opts[f"{prefix}tao_mma_asymptote_max"] = opt.get("asymptote_max", 10.0)

    tao.setTolerances(gatol=opt["opt_tol"])
    tao.setMaximumIterations(opt["max_iter"])
    
    opts[f"{prefix}tao_mma_subsolver_tao_type"] = "bqnls"          # keep as-is, or "bnls"
    opts[f"{prefix}tao_mma_subsolver_tao_ls_type"] = "armijo"      # replace fragile morethuente


    tao.setFromOptions()


    state.compliance_pce, state.volume_pce = _retrain_pce_pair(
    fem, opt, rho_warm_start_local, density_filter, rf_heaviside, sens_problem, state.beta, kl_result,
    linear_problem, rho_field)
    state.refresh_policy.last_refresh_iteration = 0
    tao.solve()


    converged_reason = tao.getConvergedReason()
    if converged_reason < 0:
        raise RuntimeError(
            f"TAO MMA did not converge: reason code={converged_reason}. "
            "Check iteration_log for divergence pattern before trusting rho_robust."
        )


    rho_robust = tao.getSolution().getArray(readonly=True).copy()


    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
        values = S_comm.gather(rho_phys_field)
        plotter.plot(values)
        save_xdmf(fem["mesh"], rho_phys_field)
        np.save(f"output/rho_robust_lambda{lambda_tradeoff:.4g}.npy", rho_robust)


    return {
            "rho_robust": rho_robust,
            "mu_C": state.compliance_pce.mu_C,
            "sigma_C": state.compliance_pce.sigma_C,
            "mean_volume": state.volume_pce.mu_C,
            "kkt_residual": float(grad_vec.norm()),
            "tao_converged_reason": converged_reason,
            "iteration_log": iteration_log,
            "compliance_pce": state.compliance_pce,
            "volume_pce": state.volume_pce,
        }