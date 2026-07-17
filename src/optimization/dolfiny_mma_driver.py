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

from pathlib import Path
import pickle

comm = MPI.COMM_WORLD

USE_FEA_CACHE = True   # <-- flip this to False to force fresh FEA solves

_FEA_CACHE_DIR = Path("output/cache/fea_at_samples")


logger = logging.getLogger(__name__)

def _cached_fea_at_samples(
    fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi, beta,
    linear_problem, rho_field, tag: str,
):
    # Key on sample count + a content hash of rho_current so switching
    # n_train/n_test or moving to a different design iterate can't silently
    # return a stale, wrong-shaped cached result.
    import hashlib
    rho_hash = hashlib.sha256(rho_current.tobytes()).hexdigest()[:12]
    cache_file = _FEA_CACHE_DIR / f"{tag}_n{xi.shape[0]}_{rho_hash}.pkl"

    if USE_FEA_CACHE:
        cache_hit = cache_file.exists() if comm.rank == 0 else False
        cache_hit = comm.bcast(cache_hit, root=0)
        if cache_hit:
            if comm.rank == 0:
                logger.info("FEA-at-samples cache HIT (%s): %s", tag, cache_file)
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
            else:
                data = None
            return comm.bcast(data, root=0)

    data = run_fea_at_samples(
        fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi, beta,
        linear_problem, rho_field,
    )
    if USE_FEA_CACHE and comm.rank == 0:
        _FEA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info("FEA-at-samples cache WRITTEN (%s): %s", tag, cache_file)
    return data


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
    true_outer_iteration: int = 0   # ADD THIS — increments once per real MMA step, via tao.setMonitor
    beta: float = 1.0
    compliance_pce: object = None
    volume_pce: object = None
    refresh_policy: PCERefreshPolicy = field(default_factory=PCERefreshPolicy)
    rho_trained_local: np.ndarray | None = None  # rho at which compliance_pce/volume_pce were fit

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


    training_data = _cached_fea_at_samples(
        fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi_train, beta,
        linear_problem, rho_field, tag="train",
    )
    test_data = _cached_fea_at_samples(
        fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi_test, beta,
        linear_problem, rho_field, tag="test",
    )


    # Dimension reduction: a cheap degree-1 fit (n_kl+1 coefficients, never
    # sample-starved) ranks which of the n_kl KL modes actually drive
    # compliance variance. Fitting the PRODUCTION PCE only on the active
    # subset avoids burning the sample budget on cross-terms between modes
    # that barely matter -- this is what let degree-2/3 fits overfit and
    # get WORSE than degree-1 in the full 26-D space. Note: only xi_train/
    # xi_test (the PCE's input) are reduced -- eta(x) sampling and the FEA
    # solves above already ran on the full physical KL field and are
    # untouched, so the manufacturing-error model itself is unaffected.
    from src.surrogate.kl_sensitivity_diagnostic import (
        diagnose_kl_mode_sensitivity, select_active_modes,
    )
    if comm.rank == 0:
        sobol_report = diagnose_kl_mode_sensitivity(
            xi_train, training_data.compliance_samples,
            xi_test, test_data.compliance_samples,
            hyperbolic_q=opt["pce_hyperbolic_q"],
        )
        active_kl_indices = select_active_modes(sobol_report, margin=3)
    else:
        active_kl_indices = None
    active_kl_indices = comm.bcast(active_kl_indices, root=0)  # must be
    # identical on every rank -- this drives which columns of xi_train/
    # xi_test get sliced below, and every rank calls build_pce_surrogate
    # collectively on the same reduced data.

    xi_train_reduced = xi_train[:, active_kl_indices]
    xi_test_reduced = xi_test[:, active_kl_indices]

    try:
        compliance_pce_result = build_pce_surrogate(
            xi_train_reduced, training_data.compliance_samples,
            xi_test_reduced, test_data.compliance_samples,
            hyperbolic_q=opt["pce_hyperbolic_q"],
            max_degree_attempts=opt["pce_max_degree_attempts"],
        )
    except RuntimeError:
        if comm.rank == 0:
            logger.warning(
                "Compliance PCE still failed its Q^2 gate even after "
                "reducing to %d active modes -- re-raising.",
                active_kl_indices.size,
            )
        raise

    if compliance_pce_result.q2 < opt["pce_q2_threshold"]:
        raise RuntimeError(
            f"Compliance PCE failed Q^2 gate: Q^2={compliance_pce_result.q2:.4f} "
            f"< threshold={opt['pce_q2_threshold']}."
        )

    volume_pce_result = build_pce_surrogate(
        xi_train_reduced, training_data.volume_samples,
        xi_test_reduced, test_data.volume_samples,
        hyperbolic_q=opt["pce_hyperbolic_q"],
        max_degree_attempts=opt["pce_max_degree_attempts"],
    )
    if volume_pce_result.q2 < opt["pce_q2_threshold"]:
        raise RuntimeError(
            f"Volume PCE failed Q^2 gate: Q^2={volume_pce_result.q2:.4f} "
            f"< threshold={opt['pce_q2_threshold']}."
        )

    compliance_pce_model = build_pce_gradient_model(
        compliance_pce_result, xi_train_reduced, training_data.dC_drho_samples,
        active_kl_indices=active_kl_indices,
    )
    volume_pce_model = build_pce_gradient_model(
        volume_pce_result, xi_train_reduced, training_data.dV_drho_samples,
        active_kl_indices=active_kl_indices,
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
    load_cases=None, 
    case_name=None
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
        dict with "rho_robust" ([n_elems_GLOBAL] converged density field,
        gathered via Communicator.gather() and comm.bcast so it is a full
        global array identical on every rank -- NOT a per-rank local slice,
        matching the same convention as this function's own rho_warm_start
        input and topopt.py's saved rho_converged.npy), "mu_C", "sigma_C",
        "mean_volume", "kkt_residual", "tao_converged_reason",
        "iteration_log" (list of per-outer-iteration dicts for CSV export).


    Raises:
        RuntimeError: If TAO reports a non-converged reason at the end of
            the solve, or if any PCE retrain fails its Q^2 gate.

    load_cases: Optional dict[str, list] of named load cases (FEniTop
        multi-case format, e.g. from build_fenitop_dicts/build_box_fenitop_dicts).
        If provided, case_name must also be given -- this function is
        SINGLE-LOAD-CASE ONLY (unlike FEniTop's form_fem_multi_case, it does
        not sum compliance/gradients across multiple cases). The selected
        case's traction_bcs list is merged into a local copy of `fem` before
        calling form_fem(). If load_cases is None, `fem` must already contain
        a "traction_bcs" key (caller has done the merge itself).
    case_name: Name of the single load case to solve, required if load_cases
        is given.
    """

    if load_cases is not None:
        if case_name is None:
            raise ValueError(
                "case_name is required when load_cases is provided -- "
                "run_robust_topopt only solves ONE load case per call, "
                "it does not loop over load_cases internally."
            )
        if case_name not in load_cases:
            raise KeyError(
                f"case_name={case_name!r} not found in load_cases "
                f"(available: {list(load_cases)!r})."
            )
        fem = dict(fem)
        fem["traction_bcs"] = load_cases[case_name]
    elif "traction_bcs" not in fem:
        raise KeyError(
            "fem['traction_bcs'] is missing and no load_cases/case_name "
            "was provided. run_robust_topopt requires EITHER (1) fem "
            "already containing 'traction_bcs' for one case (caller has "
            "done fem = dict(fem); fem['traction_bcs'] = load_cases[name] "
            "itself), OR (2) load_cases + case_name passed to this "
            "function. See this function's docstring -- it is single-"
            "load-case only, unlike FEniTop's form_fem_multi_case."
        )
    
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(
        comm, rho_field, rho_phys_field, opt["filter_radius"], fem["petsc_options"]
    )
    random_heaviside_config = RandomHeavisideConfig(
        kernel_params=opt["kernel_params"],
        transform_params=opt["transform_params"],
        variance_threshold=opt.get("kl_variance_threshold", 0.75),
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
    # Warm-start density (rho_warm_start) already converged under Stage 2's
    # full beta continuation schedule (config: beta_max=128). Restarting the
    # robust loop's projection at beta=1.0 re-introduces a much softer
    # Heaviside projection than the design was actually optimized under,
    # inflating apparent volume (gray elements pulled toward solid) for
    # however many outer iterations it takes beta to re-ramp back to
    # beta_max -- 7 doublings at beta_interval=50 is 350 outer iterations of
    # optimizing against a self-distorted volume/compliance reading before
    # this even matches the design it started from. Seed from beta_max
    # instead, since Stage 2 already did the continuation work.
    state = RobustLoopState(
        beta=float(opt["beta_max"]),
        refresh_policy=PCERefreshPolicy(
            refresh_interval=opt["pce_refresh_interval"],
            max_delta_rho_inf=opt.get("pce_max_delta_rho_inf", 0.05),
        ),
    )
    iteration_log: list[dict] = []


    def objective_gradient_callback(tao: PETSc.TAO, x: PETSc.Vec, g: PETSc.Vec) -> float:
        """TAO objective+gradient callback for the robust scalarized objective J."""
        rho_current = x.getArray(readonly=True).copy()
        rho_field.x.petsc_vec.array[:] = rho_current
        rho_field.x.petsc_vec.ghostUpdate(              
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        density_filter.forward()

        result = evaluate_from_pce(state.compliance_pce, state.volume_pce, n_elems_local)
        delta_rho = rho_current - state.rho_trained_local

        dmu = state.compliance_pce.dmu_drho()
        dsigma = state.compliance_pce.dsigma_drho()
        mu_C_lin = result.mu_C + dmu @ delta_rho
        sigma_C_lin = result.sigma_C + dsigma @ delta_rho

        J_value = mu_C_lin + lambda_tradeoff * sigma_C_lin
        dJ_drho = get_pce_robust_gradient(state.compliance_pce, robust_config)  # unchanged, still constant
        g.setArray(dJ_drho)

        iteration_log.append({
            "outer_iteration": state.outer_iteration,
            "true_outer_iteration": state.true_outer_iteration,
            "beta": state.beta,
            "J": J_value,
            "mu_C": mu_C_lin,
            "sigma_C": sigma_C_lin,
            "mean_volume": result.mean_volume,
        })
        state.outer_iteration += 1
        return J_value


    def inequality_constraint_callback(tao: PETSc.TAO, x: PETSc.Vec, c: PETSc.Vec) -> None:
        """TAO inequality constraint callback: h(rho) = Vfrac - E[V] >= 0."""
        
        rho_current = x.getArray(readonly=True).copy()
        rho_field.x.petsc_vec.array[:] = rho_current
        rho_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        delta_rho = rho_current - state.rho_trained_local
        E_V_lin = state.volume_pce.mu_C + state.volume_pce.dmu_drho() @ delta_rho
        h_value = opt["vol_frac"] - E_V_lin

        # Reduced across ranks so the logged value reflects the true global
        # max design change, not just rank 0's local slice.
        global_delta_rho_inf = comm.allreduce(float(np.abs(delta_rho).max()), op=MPI.MAX)

        if comm.rank == 0:
            logger.info(
                "constraint check: vol_frac=%.6g E_V_lin=%.6g h_value=%.6g delta_rho_inf=%.4g",
                opt["vol_frac"], E_V_lin, h_value, global_delta_rho_inf,
            )

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

    def mma_iteration_monitor(tao: PETSc.TAO) -> None:
        """Fires exactly once per real MMA outer iteration (mma.py calls tao.monitor()
        once per outer loop pass), unlike objective_gradient_callback which TAO/MMA
        invokes multiple times per outer iteration (once to build p/q, once for
        post-step convergence logging). Beta continuation and PCE refresh cadence
        must be driven from here, not from the objective/gradient callback's call
        count, or both escalate roughly 2x faster than opt["beta_interval"] /
        opt["pce_refresh_interval"] intend.
        """
        state.true_outer_iteration += 1

        if (state.true_outer_iteration % opt["beta_interval"] == 0
                and state.beta < opt["beta_max"]):
            state.beta = min(state.beta * 2, opt["beta_max"])
            logger.info("Beta continuation: increased to %.3g at true_outer_iteration=%d",
                        state.beta, state.true_outer_iteration)

        have_delta = state.rho_trained_local is not None
        local_delta_rho_inf = 0.0
        if have_delta:
            local_delta_rho_inf = float(
                np.abs(
                    tao.getSolution().getArray(readonly=True) - state.rho_trained_local
                ).max()
            )
        # MUST be reduced across ranks: tao.getSolution() only exposes this
        # rank's local dof slice, so an un-reduced .max() can cross the
        # refresh threshold on some ranks and not others. Since
        # _retrain_pce_pair()/run_fea_at_samples() is world-collective, a
        # per-rank-divergent refresh decision deadlocks the run (some ranks
        # enter the collective FEA loop, others move on and wait elsewhere).
        delta_rho_inf = comm.allreduce(local_delta_rho_inf, op=MPI.MAX) if have_delta else None

        if state.refresh_policy.needs_refresh(state.true_outer_iteration, delta_rho_inf):
            rho_current = tao.getSolution().getArray(readonly=True).copy()
            state.compliance_pce, state.volume_pce = _retrain_pce_pair(
                fem, opt, rho_current, density_filter, rf_heaviside, sens_problem,
                state.beta, kl_result, linear_problem, rho_field)
            state.rho_trained_local = rho_current.copy()
            state.refresh_policy.last_refresh_iteration = state.true_outer_iteration

        
            
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
    opts[f"{prefix}tao_mma_subsolver_tao_max_it"] = 500            # PETSc default is far too low for this problem's p/q asymmetry
    opts[f"{prefix}tao_mma_subsolver_tao_gatol"] = 1e-8
    opts[f"{prefix}tao_mma_subsolver_tao_grtol"] = 1e-8
    opts[f"{prefix}tao_mma_subsolver_tao_gttol"] = 1e-8


    tao.setFromOptions()

    tao.setMonitor(mma_iteration_monitor)


    state.compliance_pce, state.volume_pce = _retrain_pce_pair(
    fem, opt, rho_warm_start_local, density_filter, rf_heaviside, sens_problem, state.beta, kl_result,
    linear_problem, rho_field)
    state.rho_trained_local = rho_warm_start_local.copy()   # (or rho_warm_start_local at the initial call)
    state.refresh_policy.last_refresh_iteration = 0
    tao.solve()


    converged_reason = tao.getConvergedReason()
    if converged_reason < 0:
        raise RuntimeError(
            f"TAO MMA did not converge: reason code={converged_reason}. "
            "Check iteration_log for divergence pattern before trusting rho_robust."
        )


    rho_robust_local = tao.getSolution().getArray(readonly=True).copy()

    # BUGFIX: tao.getSolution() only exposes THIS RANK's local slice of the
    # design vector (x0 was copied from rho_field's petsc vec, so it shares
    # rho_field's local dof partitioning). Under `mpirun -n >1`, saving/
    # returning that local slice directly (the old behavior) silently wrote
    # a truncated, rank-0-only-partition array to disk and to the caller --
    # correct by accident only when comm.size == 1. Gather it to a true
    # GLOBAL array here, exactly mirroring topopt.py's own
    # rho_S0_comm.gather(rho_field) convention and this function's own
    # warm-start scatter above. warm_start_comm is reused (not rebuilt)
    # since it's already a Communicator on rho_field.function_space, the
    # same layout rho_robust_local was drawn from.
    #
    # Communicator.gather() is a collective MPI call -- must be invoked on
    # EVERY rank unconditionally (not just rank 0), even though only rank 0
    # receives the assembled global array back (every other rank gets None).
    rho_robust_global = warm_start_comm.gather(rho_robust_local)
    rho_robust_global = comm.bcast(rho_robust_global, root=0)

    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
        values = S_comm.gather(rho_phys_field)
        plotter.plot(values)
        save_xdmf(fem["mesh"], rho_phys_field)
        np.save(f"output/rho_robust_lambda{lambda_tradeoff:.4g}.npy", rho_robust_global)


    return {
            "rho_robust": rho_robust_global,
            "mu_C": state.compliance_pce.mu_C,
            "sigma_C": state.compliance_pce.sigma_C,
            "mean_volume": state.volume_pce.mu_C,
            "kkt_residual": float(grad_vec.norm()),
            "tao_converged_reason": converged_reason,
            "iteration_log": iteration_log,
            "compliance_pce": state.compliance_pce,
            "volume_pce": state.volume_pce,
        }