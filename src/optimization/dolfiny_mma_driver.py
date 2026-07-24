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
from src.surrogate.fea_at_samples import run_fea_at_samples, SurrogateTrainingData
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

USE_FEA_CACHE = False  # <-- flip this to False to force fresh FEA solves

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
    active_kl_indices: np.ndarray | None = None  # cached Sobol-selected active modes, reused by
        # mid-solve refreshes so they skip re-diagnosing from a full 37-dim LHS every time
    refresh_count: int = 0
    last_compliance_q2: float | None = None
    # --- best-feasible-design checkpoint + divergence guard ---
    # Populated from the true (FEA-based) compliance/volume of each successful
    # refresh, so the run can return the best design it actually evaluated
    # rather than a diverged final iterate. See run_mma_with_pce.
    diverged: bool = False
    divergence_regressions: int = 0            # consecutive refreshes with J_true > best
    best_J: float | None = None                # best (lowest) true robust objective seen
    best_rho_local: np.ndarray | None = None   # THIS RANK's local slice of that design
    best_mu_C: float | None = None
    best_sigma_C: float | None = None
    best_mean_volume: float | None = None
    best_compliance_pce: object = None
    best_volume_pce: object = None


# ---------------------------------------------------------------------------
# Unchanged: the expensive PCE-pair training routine (~n_train+n_test FEA solves).
# ---------------------------------------------------------------------------
Q2_REDIAGNOSE_THRESHOLD = 0.95  # if last compliance Q^2 fell below this, force
                                 # a fresh Sobol re-diagnosis on the NEXT retrain

# --- divergence-guard tuning (see run_mma_with_pce) ---
_VOL_FEAS_TOL = 0.05          # E[V] <= vol_frac*(1+tol) counts as feasible for checkpointing
_DIVERGENCE_REL_TOL = 1e-3    # dead-band around best_J: neither a clear improvement nor regression
_DEFAULT_DIVERGENCE_PATIENCE = 3  # consecutive regressing refreshes -> declare divergence


def _freeze_refresh_policy(policy: PCERefreshPolicy) -> None:
    """Neutralize every refresh trigger so no further mid-solve retrain fires
    (used after a divergence is detected or a refresh fails its Q^2 gate)."""
    policy.refresh_interval = 10**9
    policy.max_delta_rho_inf = float("inf")
    policy.mean_delta_rho_threshold = float("inf")
    policy.frac_moved_threshold = float("inf")


def _rediagnose_active_modes(comm, xi_train, compliance_samples, xi_test,
                              test_compliance_samples, opt, reason: str):
    """Run Sobol diagnosis + mode selection, broadcast result to all ranks."""
    from src.surrogate.kl_sensitivity_diagnostic import (
        diagnose_kl_mode_sensitivity, select_active_modes,
    )
    if comm.rank == 0:
        logger.warning("Forcing KL active-mode re-diagnosis: %s", reason)
        sobol_report = diagnose_kl_mode_sensitivity(
            xi_train, compliance_samples,
            xi_test, test_compliance_samples,
            hyperbolic_q=opt["pce_hyperbolic_q"],
        )
        active_kl_indices = select_active_modes(sobol_report, margin=3)
    else:
        active_kl_indices = None
    return comm.bcast(active_kl_indices, root=0)


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
    active_kl_indices: np.ndarray | None = None,
    seed_offset: int = 0,
    force_rediagnose: bool = False,
    group_ctx: "GroupFEAContext | None" = None,
    world_ctx: "RobustProblemContext | None" = None,
) -> tuple:
    """Retrain the compliance+volume PCE pair at the current design iterate.

    This is the expensive path (opt["pce_n_train"] FEA solves). Called only
    when RobustLoopState.refresh_policy.needs_refresh() is True (or, in the
    new shared-surrogate flow, exactly once/twice per whole lambda_sweep via
    train_pce_pair() below).

    Returns:
        (compliance_pce_model, volume_pce_model, active_kl_indices,
         compliance_q2) tuple -- the last element lets the caller decide
        whether to pass force_rediagnose=True on the NEXT call.

    Raises:
        RuntimeError: If either PCE fails its Q^2 >= threshold gate even
            after one same-checkpoint retry with a fresh Sobol diagnosis --
            this is a hard verification gate per masterContext Section 7
            and must never be bypassed, even mid-optimization.
    """
    # Cold start (active_kl_indices is None) is the ONE pass that has to
    # discover which KL modes matter -- it needs the full sample budget.
    # Every later trust-region refresh already knows the active set (handed
    # in by the caller), so a much smaller LHS is enough to refit a stable
    # degree-2 PCE over just those modes -- this is what cuts refresh cost.
    if active_kl_indices is None:
        n_train = opt["pce_n_train"]
        n_test = opt["pce_n_test"]
    else:
        # Scale sample budget to the CURRENT active-mode count rather than
        # a fixed refresh size -- the Sobol diagnostic consistently shows
        # n_kl_effective in the 8-14 range even though n_kl=37, so a fixed
        # 200-sample refresh is oversampling by ~2x. The Q^2 gate below
        # still enforces validity regardless of how n_train was chosen, so
        # this cannot silently pass a bad surrogate.
        n_active = len(active_kl_indices)
        n_train = min(
            opt.get("pce_n_train_max", opt["pce_n_train"]),
            max(opt.get("pce_n_train_min", 40), 8 * n_active),
        )
        n_test = max(20, n_train // 4)

    logger.info(
        "Retraining PCE pair at outer_iteration checkpoint: n_train=%d, beta=%.3g, "
        "active_modes=%s",
        n_train, beta, "reused" if active_kl_indices is not None else "to be diagnosed",
    )

    # -----------------------------------------------------------------------
    # Sample-budget ESCALATION with accumulation.
    #
    # As the robust optimizer sharpens the design, the compliance-vs-eta
    # response gets more nonlinear/heavy-tailed and a fixed small refresh
    # budget (8*n_active samples, degree<=4) can no longer reach the Q^2 gate.
    # Instead of giving up after one same-checkpoint re-diagnosis (which only
    # helps if the WRONG modes were picked, not if the response is genuinely
    # nonlinear), we ADD more independent samples and refit -- accumulating,
    # so prior FEA is never wasted -- until the gate passes or a sample cap is
    # hit. Cheap now that FEA-at-samples is sub-communicator parallel. This
    # only ever ADDS fidelity; the Q^2 gate is still enforced, so no worse
    # surrogate is ever accepted.
    # -----------------------------------------------------------------------
    def _solve_batch(xi):
        if group_ctx is not None:
            return run_fea_at_samples_grouped(world_ctx, group_ctx, opt, rho_current, xi, beta)
        return run_fea_at_samples(
            fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, xi, beta,
            linear_problem, rho_field,
        )

    max_escalations = int(opt.get("pce_max_escalations", 3))
    n_train_cap = int(opt.get("pce_n_train_escalation_cap", 4 * opt["pce_n_train"]))
    q2_thr = opt["pce_q2_threshold"]
    n_elems_local = rho_current.size  # world/current-rank local element count (gradient row width)

    # Accumulators grow across escalation attempts (prior FEA reused, not discarded).
    xi_train_acc = np.empty((0, kl_result.n_kl))
    xi_test_acc = np.empty((0, kl_result.n_kl))
    C_train_acc = np.empty(0); C_test_acc = np.empty(0)
    V_train_acc = np.empty(0); V_test_acc = np.empty(0)
    dC_train_acc = np.empty((0, n_elems_local))
    dV_train_acc = np.empty((0, n_elems_local))

    compliance_pce_result = volume_pce_result = None
    # Force a fresh diagnosis on the first attempt when the caller asked for it
    # (cold start, or last Q^2 fell below the re-diagnose threshold).
    pending_rediagnose = (active_kl_indices is None) or force_rediagnose

    for attempt in range(max_escalations + 1):
        train_set, test_set = generate_train_test_samples(
            kl_result, n_train=n_train, n_test=n_test,
            seed=opt.get("pce_seed", 0) + seed_offset + 1000 * (attempt + 1),
        )
        train_data = _solve_batch(train_set.xi)
        test_data = _solve_batch(test_set.xi)

        xi_train_acc = np.vstack([xi_train_acc, train_set.xi])
        xi_test_acc = np.vstack([xi_test_acc, test_set.xi])
        C_train_acc = np.concatenate([C_train_acc, train_data.compliance_samples])
        C_test_acc = np.concatenate([C_test_acc, test_data.compliance_samples])
        V_train_acc = np.concatenate([V_train_acc, train_data.volume_samples])
        V_test_acc = np.concatenate([V_test_acc, test_data.volume_samples])
        dC_train_acc = np.vstack([dC_train_acc, train_data.dC_drho_samples])
        dV_train_acc = np.vstack([dV_train_acc, train_data.dV_drho_samples])

        # (Re)diagnose active modes on the ACCUMULATED samples when needed (a
        # larger sample set gives a more reliable Sobol ranking).
        if pending_rediagnose:
            reason = (
                f"last compliance Q^2 fell below {Q2_REDIAGNOSE_THRESHOLD:.2f} -- "
                "previous active set may be stale."
                if (force_rediagnose and attempt == 0 and active_kl_indices is not None)
                else "escalation re-diagnosis on enlarged sample set."
                if attempt > 0 else "no active set yet (first checkpoint)."
            )
            active_kl_indices = _rediagnose_active_modes(
                comm, xi_train_acc, C_train_acc, xi_test_acc, C_test_acc, opt, reason,
            )
            pending_rediagnose = False

        xi_tr_red = xi_train_acc[:, active_kl_indices]
        xi_te_red = xi_test_acc[:, active_kl_indices]

        # Fit BOTH surrogates; success requires BOTH to clear the gate.
        try:
            compliance_pce_result = build_pce_surrogate(
                xi_tr_red, C_train_acc, xi_te_red, C_test_acc,
                hyperbolic_q=opt["pce_hyperbolic_q"],
                max_degree_attempts=opt["pce_max_degree_attempts"],
            )
            volume_pce_result = build_pce_surrogate(
                xi_tr_red, V_train_acc, xi_te_red, V_test_acc,
                hyperbolic_q=opt["pce_hyperbolic_q"],
                max_degree_attempts=opt["pce_max_degree_attempts"],
            )
        except RuntimeError as exc:
            compliance_pce_result = volume_pce_result = None
            if comm.rank == 0:
                logger.warning(
                    "PCE build below internal gate at attempt %d (n_train=%d): %s",
                    attempt, xi_train_acc.shape[0], exc,
                )

        passed = (
            compliance_pce_result is not None and volume_pce_result is not None
            and compliance_pce_result.q2 >= q2_thr and volume_pce_result.q2 >= q2_thr
        )
        if passed:
            break

        can_escalate = (attempt < max_escalations) and (xi_train_acc.shape[0] < n_train_cap)
        if can_escalate:
            pending_rediagnose = True  # re-rank modes on the enlarged set next attempt
            if comm.rank == 0:
                cq = f"{compliance_pce_result.q2:.4f}" if compliance_pce_result else "build-failed"
                vq = f"{volume_pce_result.q2:.4f}" if volume_pce_result else "build-failed"
                logger.warning(
                    "PCE pair below Q^2 gate (compliance=%s, volume=%s < %.2f) with "
                    "%d samples; ESCALATING: adding %d more samples + re-diagnosing "
                    "modes (attempt %d/%d).",
                    cq, vq, q2_thr, xi_train_acc.shape[0], n_train,
                    attempt + 1, max_escalations,
                )
            continue

        # Escalation budget exhausted -> raise. The CALLER decides what to do:
        # a mid-solve refresh (run_mma_with_pce) catches this and falls back to
        # the last valid surrogate; a cold-start train (train_pce_pair) lets it
        # propagate, since there is no prior surrogate to fall back to.
        cq = compliance_pce_result.q2 if compliance_pce_result else float("nan")
        vq = volume_pce_result.q2 if volume_pce_result else float("nan")
        raise RuntimeError(
            f"PCE pair failed Q^2 gate after escalating to {xi_train_acc.shape[0]} "
            f"samples (compliance Q^2={cq:.4f}, volume Q^2={vq:.4f}, "
            f"threshold={q2_thr}). The compliance-vs-eta response is likely too "
            "nonlinear to surrogate at this design -- check whether E[V] has "
            "collapsed well below vol_frac (a thin, near-disconnected structure "
            "gives heavy-tailed compliance)."
        )

    # Pass the per-sample QoI values (C_train_acc / V_train_acc) so the model
    # can build the DIRECT sample gradient estimators (dmu = mean, dsigma =
    # centered-sample) rather than an lstsq basis projection -- see
    # build_pce_gradient_model.
    compliance_pce_model = build_pce_gradient_model(
        compliance_pce_result, xi_tr_red, dC_train_acc, C_train_acc,
        active_kl_indices=active_kl_indices,
    )
    volume_pce_model = build_pce_gradient_model(
        volume_pce_result, xi_tr_red, dV_train_acc, V_train_acc,
        active_kl_indices=active_kl_indices,
    )

    logger.info(
        "PCE pair retrained: compliance Q^2=%.4f, volume Q^2=%.4f (n_train=%d, "
        "escalations=%d)",
        compliance_pce_result.q2, volume_pce_result.q2, xi_train_acc.shape[0], attempt,
    )

    return compliance_pce_model, volume_pce_model, active_kl_indices, compliance_pce_result.q2


# ---------------------------------------------------------------------------
# Sample-parallelism across MPI sub-communicators.
#
# The per-sample FEA solve has no cross-sample or cross-group dependency -- its
# only collectives (solve_fem, Sensitivity's allreduces, Heaviside's
# scatter_forward) are scoped to fem["mesh"].comm, never COMM_WORLD. So if we
# build a mesh + FEA problem on a sub-communicator, an entire group solves its
# share of a sample batch independently and concurrently with the other groups.
# Results are recombined through the SAME serial-mesh global ordering the
# existing Communicator already uses, so the caller gets back a
# SurrogateTrainingData identical in shape/ordering/partition to the serial
# path -- nothing downstream (PCE build, MMA) changes.
#
# This is math-exact in the same sense the code already is: each solve now runs
# on a smaller rank partition, so CG+GAMG converges to the same tolerance but
# differs in low-order rounding -- exactly as `-n 32` vs `-n 64` already do.
# ---------------------------------------------------------------------------
@dataclass
class GroupFEAContext:
    """FEA machinery for ONE sub-communicator group, built once per sweep.
    Every field is scoped to `group_comm`; combining across groups goes through
    the serial-mesh global ordering (see run_fea_at_samples_grouped)."""
    group_comm: object
    group_id: int
    n_groups: int
    ranks_per_group: int
    group_fem: dict
    group_linear_problem: object
    group_rho_field: object
    group_rho_phys_field: object
    group_density_filter: DensityFilter
    group_rf_heaviside: RandomFieldHeaviside
    group_sens_problem: Sensitivity
    group_design_comm: Communicator


def build_group_fea_context(fem: dict, opt: dict, kl_result: KLExpansionResult):
    """Split COMM_WORLD into groups and build a per-group FEA problem, or return
    None (single-group fallback) when grouping is disabled/not possible.

    Requires fem["mesh_factory"](comm) -> (mesh, mesh_serial) that deterministically
    rebuilds the mesh on an arbitrary communicator with mesh_serial on COMM_SELF
    (so the serial ordering matches the world mesh exactly -- the correctness
    linchpin). The box path provides this; paths without it fall back silently.
    """
    world = MPI.COMM_WORLD
    world_size = world.size
    rpg = int(opt.get("sample_parallel_ranks_per_group", 8))
    factory = fem.get("mesh_factory")

    enable = (
        factory is not None and 1 <= rpg < world_size and world_size % rpg == 0
    )
    # Ensure every rank makes the identical (collective) decision.
    enable = world.bcast(enable, root=0)
    if not enable:
        if world.rank == 0:
            if factory is None:
                reason = "no mesh_factory in fem dict"
            elif rpg >= world_size:
                reason = f"ranks_per_group={rpg} >= world_size={world_size}"
            else:
                reason = f"world_size={world_size} not divisible by ranks_per_group={rpg}"
            logger.info("Sample-parallelism disabled (%s); using single-group path.", reason)
        return None

    n_groups = world_size // rpg
    color = world.rank // rpg
    group_comm = world.Split(color=color, key=world.rank)

    group_mesh, group_mesh_serial = factory(group_comm)
    group_fem = dict(fem)
    group_fem["mesh"] = group_mesh
    group_fem["mesh_serial"] = group_mesh_serial

    group_linear_problem, group_u, group_lambda, group_rho, group_rho_phys = form_fem(
        group_fem, opt
    )
    group_density_filter = DensityFilter(
        group_comm, group_rho, group_rho_phys, opt["filter_radius"], group_fem["petsc_options"]
    )
    random_heaviside_config = RandomHeavisideConfig(
        kernel_params=opt["kernel_params"],
        transform_params=opt["transform_params"],
        variance_threshold=opt.get("kl_variance_threshold", 0.75),
        seed=opt.get("random_field_seed"),
    )
    group_rf_heaviside = build_random_heaviside_from_function_space(
        group_rho_phys, kl_result, random_heaviside_config,
    )
    group_sens_problem = Sensitivity(
        group_comm, opt, group_linear_problem, group_u, group_lambda, group_rho_phys
    )
    group_design_comm = Communicator(group_rho.function_space, group_mesh_serial)

    if world.rank == 0:
        logger.info(
            "Sample-parallelism enabled: %d group(s) x %d rank(s); each PCE/MC "
            "sample batch is split across groups and solved concurrently.",
            n_groups, rpg,
        )
    return GroupFEAContext(
        group_comm=group_comm, group_id=color, n_groups=n_groups, ranks_per_group=rpg,
        group_fem=group_fem, group_linear_problem=group_linear_problem,
        group_rho_field=group_rho, group_rho_phys_field=group_rho_phys,
        group_density_filter=group_density_filter, group_rf_heaviside=group_rf_heaviside,
        group_sens_problem=group_sens_problem, group_design_comm=group_design_comm,
    )


def run_fea_at_samples_grouped(
    world_ctx, group_ctx: GroupFEAContext, opt: dict,
    rho_current_local: np.ndarray, xi_train: np.ndarray, beta: float,
) -> SurrogateTrainingData:
    """Sample-parallel drop-in for run_fea_at_samples.

    Each group solves a contiguous chunk of xi_train on its own mesh, then
    scalar C/V and per-element gradients are recombined into world-partitioned
    arrays in the ORIGINAL sample order. Return value is field-for-field
    identical (shape/ordering/partition) to the serial run_fea_at_samples.
    """
    world = MPI.COMM_WORLD
    n_train = xi_train.shape[0]
    G = group_ctx.n_groups
    rpg = group_ctx.ranks_per_group
    gid = group_ctx.group_id
    is_group_root = (group_ctx.group_comm.rank == 0)

    world_design_comm = world_ctx.warm_start_comm
    n_elems_world_local = rho_current_local.size
    n_elems_global = world_design_comm.num_global_dofs
    # Element-identity guard: group and world serial meshes must agree, or a
    # gathered gradient row would land on the wrong world elements.
    if group_ctx.group_design_comm.num_global_dofs != n_elems_global:
        raise RuntimeError(
            "Grouped FEA element-count mismatch: group global dofs "
            f"{group_ctx.group_design_comm.num_global_dofs} != world "
            f"{n_elems_global}. mesh_factory must reproduce the world mesh's "
            "serial ordering exactly (create_box on COMM_SELF)."
        )

    # --- rho_nominal: world-local -> serial-global -> group-local ---
    rho_global = world_design_comm.gather(rho_current_local)  # serial-ordered on world rank 0
    rho_global = world.bcast(rho_global, root=0)
    group_ctx.group_design_comm.bcast(group_ctx.group_rho_field, rho_global)
    rho_nominal_group_local = group_ctx.group_rho_field.x.petsc_vec.array.copy()

    # --- contiguous sample assignment; absolute indices preserved everywhere ---
    all_ids = np.array_split(np.arange(n_train), G)
    my_ids = all_ids[gid]
    owner = np.empty(n_train, dtype=np.int64)
    for g, ids in enumerate(all_ids):
        owner[ids] = g

    # --- this group's sub-batch (deferred non-finite handling; see below) ---
    group_data = None
    if my_ids.size > 0:
        group_data = run_fea_at_samples(
            group_ctx.group_fem, opt, rho_nominal_group_local,
            group_ctx.group_density_filter, group_ctx.group_rf_heaviside,
            group_ctx.group_sens_problem, xi_train[my_ids], beta,
            group_ctx.group_linear_problem, group_ctx.group_rho_field,
            raise_on_nonfinite=False,
        )

    # --- reassemble scalar C/V in absolute sample order (SUM: one writer each) ---
    compliance_world = np.zeros(n_train)
    volume_world = np.zeros(n_train)
    if is_group_root and my_ids.size > 0:
        compliance_world[my_ids] = group_data.compliance_samples
        volume_world[my_ids] = group_data.volume_samples
    world.Allreduce(MPI.IN_PLACE, compliance_world, op=MPI.SUM)
    world.Allreduce(MPI.IN_PLACE, volume_world, op=MPI.SUM)

    # --- gather each owned sample's gradient to serial-global on the group root ---
    gradC_by_j: dict[int, np.ndarray] = {}
    gradV_by_j: dict[int, np.ndarray] = {}
    for k, j in enumerate(my_ids):
        gC = group_ctx.group_design_comm.gather(group_data.dC_drho_samples[k])
        gV = group_ctx.group_design_comm.gather(group_data.dV_drho_samples[k])
        if is_group_root:
            gradC_by_j[int(j)] = np.ascontiguousarray(gC, dtype=np.float64)
            gradV_by_j[int(j)] = np.ascontiguousarray(gV, dtype=np.float64)

    # --- stream each sample's serial-global row back to world-local rows ---
    dC_world = np.empty((n_train, n_elems_world_local))
    dV_world = np.empty((n_train, n_elems_world_local))
    widx = world_design_comm.idx
    bufC = np.empty(n_elems_global, dtype=np.float64)
    bufV = np.empty(n_elems_global, dtype=np.float64)
    for j in range(n_train):
        root_world_rank = int(owner[j]) * rpg
        if world.rank == root_world_rank:
            bufC[:] = gradC_by_j[j]
            bufV[:] = gradV_by_j[j]
        world.Bcast(bufC, root=root_world_rank)
        world.Bcast(bufV, root=root_world_rank)
        dC_world[j, :] = bufC[widx]
        dV_world[j, :] = bufV[widx]

    # --- single collective finiteness check on world-identical arrays ---
    if not (np.all(np.isfinite(compliance_world)) and np.all(np.isfinite(volume_world))):
        n_bad = int(np.sum(~np.isfinite(compliance_world)) + np.sum(~np.isfinite(volume_world)))
        raise RuntimeError(
            f"{n_bad} non-finite compliance/volume value(s) in the grouped "
            "FEA-at-samples batch (likely a near-disconnected structure under "
            "some eta(x) draw). Investigate before trusting the PCE surrogate."
        )

    return SurrogateTrainingData(
        compliance_samples=compliance_world,
        volume_samples=volume_world,
        dC_drho_samples=dC_world,
        dV_drho_samples=dV_world,
    )


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
    group: GroupFEAContext | None = None  # sample-parallel sub-comm FEA, or None


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

    # Sample-parallel sub-communicator FEA (built last: form_fem here overwrites
    # opt's UFL expressions against the group fields, and the world Sensitivity
    # above has already compiled its own forms, so ordering is safe). None when
    # grouping is disabled -> callers fall back to the serial run_fea_at_samples.
    group_ctx = build_group_fea_context(fem, opt, kl_result)

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
        group=group_ctx,
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
        beta = 8

    compliance_pce, volume_pce, active_kl_indices, compliance_q2 = _retrain_pce_pair(
        ctx.fem, opt, rho_current_local, ctx.density_filter, ctx.rf_heaviside,
        ctx.sens_problem, beta, kl_result, ctx.linear_problem, ctx.rho_field,
        group_ctx=ctx.group, world_ctx=ctx,
    )
    return compliance_pce, volume_pce, rho_current_local.copy(), active_kl_indices, compliance_q2


def run_mma_with_pce(
    ctx: RobustProblemContext,
    opt: dict,
    lambda_tradeoff: float,
    compliance_pce,
    volume_pce,
    rho_trained_local: np.ndarray,
    kl_result: KLExpansionResult,
    initial_q2,
    allow_refresh: bool = False,
    active_kl_indices: np.ndarray | None = None,
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
            max_delta_rho_inf=opt.get("pce_max_delta_rho_inf", 0.2),
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
        beta=8.0,
        compliance_pce=compliance_pce,
        volume_pce=volume_pce,
        rho_trained_local=rho_trained_local,
        refresh_policy=refresh_policy,
        active_kl_indices=active_kl_indices,
        last_compliance_q2=initial_q2,
    )

    # Seed the best-feasible-design checkpoint from the INITIAL surrogate (fit at
    # rho_trained_local). This guarantees a valid fallback design exists even if
    # the very first refresh diverges. Feasibility: E[V] <= vol_frac*(1+tol).
    divergence_patience = int(opt.get("pce_divergence_patience", _DEFAULT_DIVERGENCE_PATIENCE))
    _init_mu_C = compliance_pce.mu_C
    _init_sigma_C = compliance_pce.sigma_C
    _init_E_V = volume_pce.mu_C
    if _init_E_V <= opt["vol_frac"] * (1.0 + _VOL_FEAS_TOL):
        state.best_J = _init_mu_C + lambda_tradeoff * _init_sigma_C
        state.best_rho_local = np.asarray(rho_trained_local).copy()
        state.best_mu_C = _init_mu_C
        state.best_sigma_C = _init_sigma_C
        state.best_mean_volume = _init_E_V
        state.best_compliance_pce = compliance_pce
        state.best_volume_pce = volume_pce

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

        if False: #(state.true_outer_iteration % opt["beta_interval"] == 0 and state.beta < opt["beta_max"])
            state.beta = min(state.beta * 2, opt["beta_max"])
            logger.info("Beta continuation: increased to %.3g at true_outer_iteration=%d",
                        state.beta, state.true_outer_iteration)

        have_delta = state.rho_trained_local is not None
        delta_rho_inf = None
        delta_rho_mean = None
        delta_rho_frac_moved = None

        if have_delta:
            local_delta_rho = np.abs(
                tao.getSolution().getArray(readonly=True) - state.rho_trained_local
            )

            local_delta_rho_inf = float(local_delta_rho.max())
            delta_rho_inf = comm.allreduce(local_delta_rho_inf, op=MPI.MAX)

            # Bulk-drift metrics computed as proper GLOBAL sum/count
            # allreduces, not an average-of-per-rank-means (which would be
            # skewed if ranks hold different numbers of local elements).
            local_sum = float(local_delta_rho.sum())
            local_count = local_delta_rho.size
            local_moved_count = int((local_delta_rho > 1e-3).sum())

            global_sum = comm.allreduce(local_sum, op=MPI.SUM)
            global_count = comm.allreduce(local_count, op=MPI.SUM)
            global_moved_count = comm.allreduce(local_moved_count, op=MPI.SUM)

            delta_rho_mean = global_sum / global_count
            delta_rho_frac_moved = global_moved_count / global_count

            if comm.rank == 0:
                logger.info(
                    "design drift check: delta_rho_inf=%.4g delta_rho_mean=%.4g "
                    "delta_rho_frac_moved=%.4g",
                    delta_rho_inf, delta_rho_mean, delta_rho_frac_moved,
                )

        if state.refresh_policy.needs_refresh(
            state.true_outer_iteration, delta_rho_inf, delta_rho_mean, delta_rho_frac_moved
        ):
            rho_current = tao.getSolution().getArray(readonly=True).copy()

            force_rediagnose = (
                state.last_compliance_q2 is not None
                and state.last_compliance_q2 < Q2_REDIAGNOSE_THRESHOLD
            )   # <-- NEW

            try:
                new_pce = _retrain_pce_pair(
                    fem, opt, rho_current, density_filter, rf_heaviside, sens_problem, 8,
                    kl_result, linear_problem, rho_field,
                    active_kl_indices=state.active_kl_indices,
                    seed_offset=state.refresh_count,
                    force_rediagnose=force_rediagnose,
                    group_ctx=ctx.group, world_ctx=ctx,  # sample-parallel refresh
                )
            except RuntimeError as exc:
                # GRACEFUL FALLBACK: a mid-solve refresh that cannot pass the Q^2
                # gate even after escalating samples must NOT kill the whole run
                # (previously this RuntimeError propagated through the TAO monitor
                # callback -> PETSc error 101 -> crash). Keep the LAST VALID
                # (gate-passing) surrogate + its training point unchanged, and
                # DISABLE further refreshes for the rest of this lambda so we
                # don't burn FEA re-failing. No bad surrogate is ever used -- only
                # an older good one -- and the mandatory Stage-6 MC validation
                # remains the final accuracy check on the returned design.
                if comm.rank == 0:
                    last_q2 = state.last_compliance_q2 if state.last_compliance_q2 is not None else float("nan")
                    logger.error(
                        "PCE REFRESH FAILED at true_outer_iteration=%d and could not "
                        "be recovered by sample escalation (%s). RETAINING the last "
                        "valid surrogate (compliance Q^2=%.4f) and DISABLING further "
                        "refreshes for lambda=%.3g. This almost always means the "
                        "design has drifted into a thin, near-disconnected regime "
                        "(check E[V] vs vol_frac in the constraint-check logs above) "
                        "where compliance is no longer surrogatable -- treat the "
                        "returned design as provisional and rely on Stage-6 MC.",
                        state.true_outer_iteration, exc, last_q2, lambda_tradeoff,
                    )
                # A refresh that cannot be surrogated at all is itself a
                # divergence signal: freeze refreshes and mark the solve diverged
                # so the best feasible checkpoint is restored at exit.
                _freeze_refresh_policy(state.refresh_policy)
                state.refresh_policy.last_refresh_iteration = state.true_outer_iteration
                state.diverged = True
            else:
                state.compliance_pce, state.volume_pce, state.active_kl_indices, state.last_compliance_q2 = new_pce
                state.rho_trained_local = rho_current.copy()
                state.refresh_policy.last_refresh_iteration = state.true_outer_iteration
                state.refresh_count += 1

                # --- best-feasible-design checkpoint + divergence guard --------
                # state.compliance_pce/volume_pce are now fit to TRUE FEA samples
                # at rho_current, so their mu_C/sigma_C and E[V] are the true
                # values at this design. Track the best feasible one and watch
                # for a sustained regression (the runaway seen in run_log.txt:
                # J climbing geometrically while E[V] collapses below vol_frac).
                mu_C_true = state.compliance_pce.mu_C
                sigma_C_true = state.compliance_pce.sigma_C
                E_V_true = state.volume_pce.mu_C
                J_true = mu_C_true + lambda_tradeoff * sigma_C_true
                feasible = E_V_true <= opt["vol_frac"] * (1.0 + _VOL_FEAS_TOL)

                if feasible and (
                    state.best_J is None or J_true < state.best_J * (1.0 - _DIVERGENCE_REL_TOL)
                ):
                    # New best feasible design -> checkpoint it, reset the counter.
                    state.best_J = J_true
                    state.best_rho_local = rho_current.copy()
                    state.best_mu_C = mu_C_true
                    state.best_sigma_C = sigma_C_true
                    state.best_mean_volume = E_V_true
                    state.best_compliance_pce = state.compliance_pce
                    state.best_volume_pce = state.volume_pce
                    state.divergence_regressions = 0
                elif state.best_J is not None and J_true > state.best_J * (1.0 + _DIVERGENCE_REL_TOL):
                    state.divergence_regressions += 1
                    if comm.rank == 0:
                        logger.warning(
                            "Robust objective regressed at refresh %d: J_true=%.6g > "
                            "best=%.6g, E[V]=%.6g (%d/%d consecutive regressions).",
                            state.refresh_count, J_true, state.best_J, E_V_true,
                            state.divergence_regressions, divergence_patience,
                        )
                    if state.divergence_regressions >= divergence_patience:
                        state.diverged = True
                        if comm.rank == 0:
                            logger.error(
                                "DIVERGENCE DETECTED at refresh %d (robust objective "
                                "rose for %d consecutive refreshes while E[V] drifts "
                                "from vol_frac=%.4g -- the design is shedding material "
                                "into an un-surrogatable thin regime). Freezing "
                                "refreshes; the best feasible design (J=%.6g, "
                                "mu_C=%.6g, E[V]=%.6g) will be RESTORED at exit.",
                                state.refresh_count, divergence_patience,
                                opt["vol_frac"], state.best_J, state.best_mu_C,
                                state.best_mean_volume,
                            )
                        _freeze_refresh_policy(state.refresh_policy)

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
    # A negative reason normally means non-convergence and is fatal. But when we
    # deliberately diverged (froze refreshes and will restore the checkpoint),
    # TAO running on to its iteration limit is EXPECTED, not an error -- so we
    # must not raise in that case, or the divergence guard would just move the
    # crash here. The restored design is the best FEASIBLE one we evaluated.
    if converged_reason < 0 and not state.diverged:
        raise RuntimeError(
            f"TAO MMA did not converge (reason code={converged_reason}). "
            "Check iteration_log for divergence pattern before trusting rho_robust."
        )

    rho_robust_local = tao.getSolution().getArray(readonly=True).copy()

    # --- restore the best feasible checkpoint on divergence ---------------------
    # Pure/robust minimization should improve monotonically; if it diverged, the
    # final iterate is worse than a design we already evaluated. Return that best
    # feasible design instead, and sync rho_field/rho_phys_field so the saved
    # XDMF/plot below reflect the design we actually return.
    if state.diverged and state.best_rho_local is not None:
        if comm.rank == 0:
            logger.error(
                "Restoring best feasible checkpointed design instead of the "
                "diverged final iterate (J=%.6g, mu_C=%.6g, sigma_C=%.6g, "
                "E[V]=%.6g).", state.best_J, state.best_mu_C, state.best_sigma_C,
                state.best_mean_volume,
            )
        rho_robust_local = state.best_rho_local.copy()
        rho_field.x.petsc_vec.array[:] = rho_robust_local
        rho_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        density_filter.forward()  # refresh rho_phys_field for the saved artifacts
        result_compliance_pce = state.best_compliance_pce
        result_volume_pce = state.best_volume_pce
        result_mu_C = state.best_mu_C
        result_sigma_C = state.best_sigma_C
        result_mean_volume = state.best_mean_volume
    else:
        result_compliance_pce = state.compliance_pce
        result_volume_pce = state.volume_pce
        result_mu_C = state.compliance_pce.mu_C
        result_sigma_C = state.compliance_pce.sigma_C
        result_mean_volume = state.volume_pce.mu_C

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
        "mu_C": result_mu_C,
        "sigma_C": result_sigma_C,
        "mean_volume": result_mean_volume,
        # KKT residual belongs to the FINAL iterate; when we restore an earlier
        # checkpoint it does not describe the returned design, so report NaN
        # rather than a misleading value.
        "kkt_residual": float("nan") if state.diverged else float(grad_vec.norm()),
        "tao_converged_reason": converged_reason,
        "diverged": state.diverged,
        "iteration_log": iteration_log,
        "compliance_pce": result_compliance_pce,
        "volume_pce": result_volume_pce,
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
    compliance_pce, volume_pce, rho_trained_local, active_kl_indices, compliance_q2 = train_pce_pair(
        ctx, opt, kl_result
    )
    return run_mma_with_pce(
        ctx, opt, lambda_tradeoff, compliance_pce, volume_pce, rho_trained_local,
        kl_result, initial_q2=compliance_q2, allow_refresh=True, active_kl_indices=active_kl_indices,
    )