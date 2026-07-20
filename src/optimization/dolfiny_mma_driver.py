"""
src/optimization/dolfiny_mma_driver.py

Stage 5 (Robust Topology Optimization Loop) -- masterContext Section 3.5.

Wires FEniTop's fixed-mesh FEA/sensitivity machinery, the randomized
Heaviside projection (topology/heaviside_projection_glue.py), and the
compliance+volume PCE surrogate pair (surrogate/pce_model.py,
optimization/pce_evaluation.py) into dolfiny's native PETSc TAO MMA
algorithm (mma-13.py's MMA class), per the "never reimplement MMA" rule.

REFACTORED: setup / PCE training / MMA solve are now three separate
entry points (setup_robust_problem, train_pce_pair, run_mma_with_pce)
so a single lambda_sweep in main.py can build the FEA problem and train
the PCE surrogate ONCE and reuse it across every lambda point, instead
of retraining ~250 FEA solves per lambda. The original single-call
run_robust_topopt() is kept at the bottom as a thin wrapper for
backward compatibility (calls all three in sequence, exactly like
before).
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter
from src.fenitop.sensitivity import Sensitivity
from src.fenitop.utility import Communicator, Plotter, save_xdmf
from src.fenitop.mma import MMA  # dolfiny-style native TAO MMA (mma-13.py)

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

comm = MPI.COMM_WORLD

USE_FEA_CACHE = True  # <-- flip this to False to force fresh FEA solves

_FEA_CACHE_DIR = Path("output/cache/fea_at_samples")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unchanged helper: FEA-at-samples caching wrapper.
# ---------------------------------------------------------------------------
def _cached_fea_at_samples(
    fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi, beta,
    linear_problem, rho_field, tag: str,
    cache_file_name: str | None = None,
):
    import hashlib

    if cache_file_name is not None:
        cache_file = _FEA_CACHE_DIR / cache_file_name
    else:
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
            return comm.bcast(data, root=0)
        # cache miss: fall through to compute below instead of returning None

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


# ---------------------------------------------------------------------------
# Unchanged: mutable state threaded through TAO callbacks.
# ---------------------------------------------------------------------------
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
    true_outer_iteration: int = 0  # increments once per real MMA step, via tao.setMonitor
    beta: float = 1.0
    compliance_pce: object = None
    volume_pce: object = None
    refresh_policy: PCERefreshPolicy = field(default_factory=PCERefreshPolicy)
    rho_trained_local: np.ndarray | None = None  # rho at which compliance_pce/volume_pce were fit


# ---------------------------------------------------------------------------
# Unchanged: the expensive PCE-pair training routine (~n_train+n_test FEA solves).
# ---------------------------------------------------------------------------
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
    when RobustLoopState.refresh_policy.needs_refresh() is True (or, in the
    new shared-surrogate flow, exactly once/twice per whole lambda_sweep via
    train_pce_pair() below).

    Returns:
        (compliance_pce_model, volume_pce_model) tuple of PCEGradientModel.

    Raises:
        RuntimeError: If either PCE fails its Q^2 >= threshold gate -- this
            is a hard verification gate per masterContext Section 7 and
            must never be bypassed, even mid-optimization.
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
        linear_problem, rho_field
    )
    test_data = run_fea_at_samples(
        fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi_test, beta,
        linear_problem, rho_field
    )

    # Dimension reduction: a cheap degree-1 fit (n_kl+1 coefficients, never
    # sample-starved) ranks which of the n_kl KL modes actually drive
    # compliance variance. Fitting the PRODUCTION PCE only on the active
    # subset avoids burning the sample budget on cross-terms between modes
    # that barely matter.
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
    # identical on every rank -- drives which columns of xi_train/xi_test
    # get sliced below, and every rank calls build_pce_surrogate
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


# ---------------------------------------------------------------------------
# NEW: holds everything built once per lambda_sweep (mesh, FEA objects,
# warm-start), independent of any single lambda value.
# ---------------------------------------------------------------------------
@dataclass
class RobustProblemContext:
    """Everything about the FEA/PDE problem that is IDENTICAL across every
    lambda in a Pareto sweep. Built once by setup_robust_problem(), then
    passed to train_pce_pair() and run_mma_with_pce() repeatedly."""
    fem: dict
    linear_problem: object
    u_field: object
    lambda_field: object
    rho_field: object
    rho_phys_field: object
    density_filter: DensityFilter
    rf_heaviside: RandomFieldHeaviside
    sens_problem: Sensitivity
    warm_start_comm: Communicator
    rho_warm_start_local: np.ndarray
    n_elems_local: int
    n_elems_global: int
    col_start: int


def setup_robust_problem(
    fem: dict,
    opt: dict,
    rho_warm_start: np.ndarray,
    kl_result: KLExpansionResult,
    load_cases=None,
    case_name=None,
) -> RobustProblemContext:
    """Build the FEA/PDE machinery ONCE, shared across an entire lambda_sweep.

    This is exactly the setup code that used to run at the top of
    run_robust_topopt() on every single call -- form_fem, DensityFilter,
    the random Heaviside projection, Sensitivity, and the warm-start
    scatter. Pulling it out means main.py calls this ONE time instead of
    once per lambda.

    Args mirror run_robust_topopt's docstring exactly (see below).

    Returns:
        A RobustProblemContext to be passed into train_pce_pair() and
        run_mma_with_pce().
    """
    if load_cases is not None:
        if case_name is None:
            raise ValueError(
                "case_name is required when load_cases is provided -- "
                "this function is single-load-case only, it does not loop "
                "over load_cases internally."
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
            "was provided. setup_robust_problem requires EITHER (1) fem "
            "already containing 'traction_bcs' for one case, OR (2) "
            "load_cases + case_name passed to this function."
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
            "a full global array (identical on every rank), not a local slice."
        )

    # Scatter the global warm-start array into this rank's local dof slice,
    # using the same Communicator.bcast() pattern utility.py's Communicator
    # already uses (mirror image of its gather() used at the end of the
    # MMA solve below).
    warm_start_comm = Communicator(rho_field.function_space, fem["mesh_serial"])
    warm_start_comm.bcast(rho_field, rho_warm_start)
    rho_warm_start_local = rho_field.x.petsc_vec.array.copy()

    return RobustProblemContext(
        fem=fem,
        linear_problem=linear_problem,
        u_field=u_field,
        lambda_field=lambda_field,
        rho_field=rho_field,
        rho_phys_field=rho_phys_field,
        density_filter=density_filter,
        rf_heaviside=rf_heaviside,
        sens_problem=sens_problem,
        warm_start_comm=warm_start_comm,
        rho_warm_start_local=rho_warm_start_local,
        n_elems_local=n_elems_local,
        n_elems_global=n_elems_global,
        col_start=col_start,
    )


def train_pce_pair(
    ctx: RobustProblemContext,
    opt: dict,
    kl_result: KLExpansionResult,
    rho_current_local: np.ndarray | None = None,
    beta: float | None = None,
):
    """Train the compliance+volume PCE pair ONCE, to be reused across an
    entire lambda_sweep (or re-called a second time mid-sweep for an
    optional refresh -- see main.py).

    Args:
        ctx: RobustProblemContext from setup_robust_problem().
        opt: FEniTop opt dict (needs pce_n_train/n_test/etc, beta_max).
        kl_result: Stage-3 KL expansion result.
        rho_current_local: design density to train at, THIS RANK's local
            slice. Defaults to ctx.rho_warm_start_local (the nominal SIMP
            warm-start design) if not given -- this is the normal case for
            training once before the lambda_sweep starts.
        beta: Heaviside sharpness for training solves. Defaults to
            opt["beta_max"] / 4, matching the original run_robust_topopt's
            initial training call (state.beta / 4 where state.beta was
            seeded from opt["beta_max"]).

    Returns:
        (compliance_pce, volume_pce, rho_trained_local) -- the third
        element is rho_current_local.copy(), i.e. the design point the
        surrogate was actually fit at, needed by run_mma_with_pce() to
        compute delta_rho during the MMA solve.
    """
    if rho_current_local is None:
        rho_current_local = ctx.rho_warm_start_local
    if beta is None:
        beta = float(opt["beta_max"]) / 2

    compliance_pce, volume_pce = _retrain_pce_pair(
        ctx.fem, opt, rho_current_local, ctx.density_filter, ctx.rf_heaviside,
        ctx.sens_problem, beta, kl_result, ctx.linear_problem, ctx.rho_field,
    )
    return compliance_pce, volume_pce, rho_current_local.copy()


def run_mma_with_pce(
    ctx: RobustProblemContext,
    opt: dict,
    lambda_tradeoff: float,
    compliance_pce,
    volume_pce,
    rho_trained_local: np.ndarray,
    kl_result: KLExpansionResult,
    allow_refresh: bool = False,
) -> dict:
    """Run the TAO MMA outer loop for ONE lambda, against an ALREADY-TRAINED
    PCE surrogate pair.

    This is exactly the TAO setup/callbacks/solve section that used to live
    at the bottom of run_robust_topopt(), unchanged, except:
      (1) it takes compliance_pce/volume_pce/rho_trained_local as arguments
          instead of building them via an unconditional _retrain_pce_pair
          call before tao.solve(), and
      (2) allow_refresh controls whether the refresh_policy inside the MMA
          loop is allowed to fire a mid-solve retrain at all. Pass
          allow_refresh=False (the default) to guarantee ZERO additional
          FEA solves during this call -- the whole 400-iteration MMA solve
          runs purely off the surrogate you already trained.

    Args:
        ctx: RobustProblemContext from setup_robust_problem().
        opt: FEniTop opt dict, extended with PCE/robust-loop keys.
        lambda_tradeoff: Mean-variance scalarization weight for this run.
        compliance_pce, volume_pce: Trained PCEGradientModel pair (from
            train_pce_pair()).
        rho_trained_local: THIS RANK's local slice of the design point the
            surrogate was fit at (third return value of train_pce_pair()).
        kl_result: Stage-3 KL expansion result (still needed in case
            allow_refresh=True triggers a mid-solve retrain).
        allow_refresh: If True, the mma_iteration_monitor's refresh_policy
            can still fire an expensive mid-solve retrain exactly like the
            original code. If False (default), refresh_interval and
            max_delta_rho_inf are both set effectively to infinity so
            needs_refresh() never returns True.

    Returns:
        Same dict shape as the original run_robust_topopt: "rho_robust",
        "mu_C", "sigma_C", "mean_volume", "kkt_residual",
        "tao_converged_reason", "iteration_log", "compliance_pce",
        "volume_pce".

    Raises:
        RuntimeError: If TAO reports a non-converged reason, or if a
            mid-solve refresh (when allow_refresh=True) fails its Q^2 gate.
    """
    fem = ctx.fem
    rho_field = ctx.rho_field
    rho_phys_field = ctx.rho_phys_field
    density_filter = ctx.density_filter
    rf_heaviside = ctx.rf_heaviside
    sens_problem = ctx.sens_problem
    linear_problem = ctx.linear_problem
    warm_start_comm = ctx.warm_start_comm
    n_elems_local = ctx.n_elems_local
    n_elems_global = ctx.n_elems_global
    col_start = ctx.col_start

    robust_config = RobustObjectiveConfig(lambda_tradeoff=lambda_tradeoff)

    if allow_refresh:
        refresh_policy = PCERefreshPolicy(
            refresh_interval=opt["pce_refresh_interval"],
            max_delta_rho_inf=opt.get("pce_max_delta_rho_inf", 0.05),
        )
    else:
        # Effectively disables needs_refresh() for the whole solve -- no
        # interval will ever be reached and no delta_rho_inf will ever
        # exceed this threshold, so the MMA loop below performs ZERO
        # additional FEA solves after this function is entered.
        refresh_policy = PCERefreshPolicy(
            refresh_interval=10**9,
            max_delta_rho_inf=float("inf"),
        )

    state = RobustLoopState(
        beta=float(opt["beta_max"]),
        compliance_pce=compliance_pce,
        volume_pce=volume_pce,
        rho_trained_local=rho_trained_local,
        refresh_policy=refresh_policy,
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
        dJ_drho = get_pce_robust_gradient(state.compliance_pce, robust_config)
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
        """TAO constraint Jacobian callback: dh/drho = -dE[V]/drho."""
        dh_drho = -get_pce_volume_gradient(state.volume_pce)
        global_cols = np.arange(col_start, col_start + n_elems_local, dtype=PETSc.IntType)
        J.setValues([0], global_cols, dh_drho.reshape(1, -1))
        J.assemble()

    tao = PETSc.TAO().create(comm)

    def mma_iteration_monitor(tao: PETSc.TAO) -> None:
        """Fires exactly once per real MMA outer iteration."""
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

        delta_rho_inf = comm.allreduce(local_delta_rho_inf, op=MPI.MAX) if have_delta else None

        if state.refresh_policy.needs_refresh(state.true_outer_iteration, delta_rho_inf):
            rho_current = tao.getSolution().getArray(readonly=True).copy()
            state.compliance_pce, state.volume_pce = _retrain_pce_pair(
                fem, opt, rho_current, density_filter, rf_heaviside, sens_problem,
                state.beta / 2, kl_result, linear_problem, rho_field)
            state.rho_trained_local = rho_current.copy()
            state.refresh_policy.last_refresh_iteration = state.true_outer_iteration

    tao.setType(PETSc.TAO.Type.PYTHON)
    tao.setPythonContext(MMA())

    x0 = rho_field.x.petsc_vec.copy()
    x0.setArray(ctx.rho_warm_start_local)
    tao.setSolution(x0)

    lb = x0.copy(); lb.set(0.0)
    ub = x0.copy(); ub.set(1.0)
    tao.setVariableBounds((lb, ub))

    grad_vec = x0.copy()
    tao.setObjectiveGradient(objective_gradient_callback, grad_vec)

    # NOTE: createSeq requires a size-1 communicator and will raise under
    # `mpirun -n 64`; the single global constraint must instead be a proper
    # distributed (global size 1) PETSc vector.
    constraint_vec = PETSc.Vec().createMPI(1, comm=comm)
    tao.setInequalityConstraints(inequality_constraint_callback, constraint_vec)

    # NOTE (verify against your original file -- exact createDense call
    # signature was not 100% recoverable from the mangled source text):
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

    # NOTE: the original code did an unconditional _retrain_pce_pair() call
    # right here, before tao.solve(). That call is now done ONCE outside
    # this function via train_pce_pair() and passed in as an argument --
    # state.compliance_pce/volume_pce/rho_trained_local are already set
    # above from the constructor, so there is nothing to train here.

    tao.solve()

    converged_reason = tao.getConvergedReason()
    if converged_reason < 0:
        raise RuntimeError(
            f"TAO MMA did not converge (reason code={converged_reason}). "
            "Check iteration_log for divergence pattern before trusting rho_robust."
        )

    rho_robust_local = tao.getSolution().getArray(readonly=True).copy()

    # NOTE (verify gather() signature against your original Communicator
    # class -- reconstructed from context):
    rho_robust_global = warm_start_comm.gather(rho_robust_local)
    rho_robust_global = comm.bcast(rho_robust_global, root=0)

    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    
    values = S_comm.gather(rho_phys_field)
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
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


# ---------------------------------------------------------------------------
# Backward-compatible wrapper: identical behavior to the ORIGINAL
# run_robust_topopt (trains a fresh PCE pair on every call). Kept so any
# other caller that still imports run_robust_topopt directly keeps working
# unchanged. main.py's lambda_sweep loop should call setup_robust_problem /
# train_pce_pair / run_mma_with_pce directly instead (see main.py diff).
# ---------------------------------------------------------------------------
def run_robust_topopt(
    fem: dict,
    opt: dict,
    rho_warm_start: np.ndarray,
    lambda_tradeoff: float,
    kl_result: KLExpansionResult,
    load_cases=None,
    case_name=None,
) -> dict:
    """Original single-call entry point: builds the FEA problem, trains one
    PCE pair, and runs the MMA loop, all for exactly one lambda. Equivalent
    to setup_robust_problem() + train_pce_pair() + run_mma_with_pce()
    called back-to-back with allow_refresh=True (matching the original
    code's refresh_policy behavior inside the MMA loop)."""
    ctx = setup_robust_problem(fem, opt, rho_warm_start, kl_result, load_cases, case_name)
    compliance_pce, volume_pce, rho_trained_local = train_pce_pair(ctx, opt, kl_result)
    return run_mma_with_pce(
        ctx, opt, lambda_tradeoff, compliance_pce, volume_pce, rho_trained_local,
        kl_result, allow_refresh=True,
    )