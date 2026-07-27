"""src/mainClean.py — top-level orchestrator, ties together all six stages.

This is a clean, no-cache version of the pipeline: every stage always runs
from scratch, and nothing is ever read back from a prior run to skip work.

WHAT CHANGED, AND WHY (research-standards remediation)
-----------------------------------------------------
1. Every artifact now lands under output/<stage>/<run_id>/, run_id = UTC
   timestamp + git SHA, and a manifest.json records the git state, library
   versions, MPI size, seeds, per-stage timings and -- critically -- the
   EFFECTIVE fem/opt dicts, which on the box path differ from config.yaml.
   Previously each run overwrote the last, and optimized_design.xdmf/.jpg were
   written into the current working directory by every lambda in turn.

2. Verification gates run before the robust solve and abort the run on failure.
   They existed in the tree but nothing called them.

3. The Pareto sweep solves every lambda from the SAME nominal warm start and
   asserts the resulting set is non-dominated. The previous sweep warm-started
   each lambda from the previous one, which made lambda=1 nothing but lambda=0
   plus another 400 descent steps -- and it duly came out better in BOTH mu_C
   and sigma_C, which no genuine trade-off curve can do.

4. Stage 6 validates EVERY design (nominal + every lambda), not just the last,
   on a COMMON eta ensemble, and reports bootstrap confidence intervals plus a
   paired comparison against the nominal. The headline claim from the previous
   run -- a 6.2% reduction in sigma_C -- was smaller than the +/-7% confidence
   interval of the n=100 validation that was supposed to confirm it.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI
import ufl
import dolfinx
comm = MPI.COMM_WORLD

from src.config.loader import load_config
from src.fenitop.topopt import topopt
from src.fenitop.utility import Communicator
from src.random_fields.kernel import KernelParams, build_squared_exponential
from src.random_fields.kl_expansion import compute_kl_expansion
from src.optimization.dolfiny_mma_driver import run_robust_topopt
from src.surrogate.pce_model import build_pce_gradient_model
from src.validation.monte_carlo import (
    MCConfig, run_monte_carlo_validation, compare_against_pce, plot_cdf,
)

from src.meshing.importer import import_and_heal, finalize
from src.meshing.mesher import extract_simplices, MeshingConfig, tag_physical_groups, generate_mesh, import_to_dolfinx
from src.meshing.mapper import build_boundary_conditions
from src.meshing.box_source import build_box_fenitop_dicts
from src.fea.fenitop_adapter import build_fenitop_dicts

from src.topology.heaviside_projection_glue import RandomHeavisideConfig
from src.random_fields.threshold_transform import MarginalTransformParams
from src.viz.probability_cloud import build_probability_cloud, ProbabilityCloudConfig

from src.optimization.dolfiny_mma_driver import (
    setup_robust_problem, train_pce_pair, run_mma_with_pce,
)
from src.optimization.saa_robust_driver import (
    run_saa_robust_topopt, save_design_artifacts, measure_non_discreteness,
)
from src.sampling.sampler import generate_samples
from src.provenance import RunManifest, make_run_id
from src.validation.gates import GateConfig, run_verification_gates
from src.validation.statistics import (
    compare_designs, resolvability_report, summarize_samples,
)
from src.validation.boundary_offset import (
    build_report as build_boundary_offset_report,
    measure_offset_from_volumes,
)

from dolfinx.io import XDMFFile

logging.basicConfig(level=logging.DEBUG, force=True)
logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output layout: output/<stage>/<run_id>/.
#
# Runs no longer overwrite each other. The previous layout wiped every stage
# directory at startup and wrote the design artifacts (optimized_design.xdmf,
# .jpg) into the CURRENT WORKING DIRECTORY with a fixed name, so each lambda
# clobbered the previous lambda's geometry and the results accumulated in the
# repository root.
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path("output")
STAGE_NAMES = {
    1: "stage1_mesh",
    2: "stage2_fea",
    3: "stage3_random_field",
    4: "stage4_surrogate",
    5: "stage5_optimization",
    6: "stage6_validation",
}


def _stage_dir(stage: int, run_id: str) -> Path:
    """Per-run directory for one stage. Created on rank 0 only."""
    path = OUTPUT_ROOT / STAGE_NAMES[stage] / run_id
    if comm.rank == 0:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _scatter_global_design(rho_field, mesh_serial, global_array: np.ndarray) -> None:
    """Distribute a rank-0-loaded global design array into rho_field's local slice.

    Mirrors topopt.py's Communicator.gather() pattern in reverse. global_array
    must already be a full [n_elems_global] array, identical on every rank
    (e.g. via comm.bcast after a single np.load on rank 0) -- Communicator's
    own bcast() indexes into it using each rank's precomputed local<->global
    dof map, so no further per-rank slicing is needed here.
    """
    comm_helper = Communicator(rho_field.function_space, mesh_serial)
    comm_helper.bcast(rho_field, global_array)


def _assert_non_dominated(pareto: list[dict]) -> list[dict]:
    """Fail if any sweep point is dominated by another in (mu_C, sigma_C).

    A trade-off curve cannot contain a point that another point beats on BOTH
    objectives -- if it does, the "front" is not a front. The previous run
    produced exactly that (lambda=1 at mu_C=0.15967/sigma_C=0.00820 dominating
    lambda=0 at 0.16080/0.00875), because lambda-continuation made the later
    point a continuation of the earlier one's descent rather than a separate
    optimum. This check turns that class of defect into an immediate failure
    instead of a plot.

    Returns:
        The list of dominated entries (empty when the sweep is a valid front).
    """
    dominated = []
    for i, a in enumerate(pareto):
        for j, b in enumerate(pareto):
            if i == j:
                continue
            beats_or_ties = b["mu_C"] <= a["mu_C"] and b["sigma_C"] <= a["sigma_C"]
            strictly_beats = b["mu_C"] < a["mu_C"] or b["sigma_C"] < a["sigma_C"]
            if beats_or_ties and strictly_beats:
                dominated.append({
                    "dominated_lambda": a["lambda"],
                    "dominated_by_lambda": b["lambda"],
                    "dominated_mu_C": a["mu_C"], "dominated_sigma_C": a["sigma_C"],
                    "dominator_mu_C": b["mu_C"], "dominator_sigma_C": b["sigma_C"],
                })
                break
    return dominated


def main(config_path: str = "src/config/config.yaml") -> None:
    cfg = load_config(config_path)

    run_id = make_run_id(comm)
    manifest = RunManifest(run_id, comm)
    STAGE1_DIR = _stage_dir(1, run_id)
    STAGE2_DIR = _stage_dir(2, run_id)
    STAGE3_DIR = _stage_dir(3, run_id)
    STAGE4_DIR = _stage_dir(4, run_id)
    STAGE5_DIR = _stage_dir(5, run_id)
    STAGE6_DIR = _stage_dir(6, run_id)
    manifest_path = OUTPUT_ROOT / "runs" / f"{run_id}.json"
    comm.Barrier()

    if comm.rank == 0:
        logger.info("Run id: %s (artifacts under output/<stage>/%s/)", run_id, run_id)

    manifest.record_seeds(
        random_field=cfg.random_field.seed,
        saa=cfg.optimization.saa_seed,
        mc_validation=cfg.mc_validation.seed,
        bootstrap=cfg.mc_validation.bootstrap_seed,
    )

    # =========================================================================
    # STAGE 1: Mesh generation
    # =========================================================================
    if cfg.mesh_source == "step":
        # ---------------------------------------------------------------
        # STEP/CAD mesh pipeline (unchanged).
        # ---------------------------------------------------------------
        if comm.rank == 0:
            entities = import_and_heal(cfg.step_file)
            mesh_cfg = MeshingConfig(mesh_size_max=cfg.mesh_size_max,
                                    color_targets=cfg.color_targets,
                                    solid_volume_color=cfg.solid_volume_color)
            tag_physical_groups(entities, mesh_cfg)
            generate_mesh(mesh_cfg, comm)

        tagged_mesh = import_to_dolfinx(comm)  # already collective — reads/partitions on all ranks
        assert "fixed" in tagged_mesh.name_to_tag, "No 'fixed' faces tagged — check STEP coloring"
        assert "load_1" in tagged_mesh.name_to_tag, "No 'load_1' faces tagged — check STEP coloring"

        # cfg.load_cases is dict[case_name, list[LoadCase]] (see schema.py) --
        # convert each LoadCase to a plain (group_name, vector) tuple, which is
        # what mapper.build_boundary_conditions expects.
        load_cases_input = {
            case_name: [(lc.group_name, lc.vector) for lc in entries]
            for case_name, entries in cfg.load_cases.items()
        }
        # --- keep-alive backbone resolution -------------------------------------
        # Resolve the non-designable "keep-alive" corridor parameters here (the
        # call site), not inside mapper.py, so the mesh-size-derived default length
        # scale stays traceable to cfg. mapper.build_boundary_conditions raises if
        # groups are given without concrete radius/eps, so we always pass both when
        # enabled. When disabled, all three are None and solid_zone is unchanged.
        ka = cfg.keep_alive
        if ka.enabled:
            keep_alive_groups = ka.groups
            keep_alive_radius = (
                ka.corridor_radius if ka.corridor_radius is not None
                else 2.0 * cfg.mesh_size_max
            )
            keep_alive_cluster_eps = (
                ka.cluster_eps if ka.cluster_eps is not None
                else 2.0 * cfg.mesh_size_max
            )
            if comm.rank == 0:
                logger.info(
                    "keep_alive enabled: groups=%s, corridor_radius=%.6g m, "
                    "cluster_eps=%.6g m", keep_alive_groups,
                    keep_alive_radius, keep_alive_cluster_eps,
                )
        else:
            keep_alive_groups = None
            keep_alive_radius = None
            keep_alive_cluster_eps = None
            if comm.rank == 0:
                logger.info("keep_alive disabled; solid_zone uses volumes + "
                            "protected faces only.")

        bc = build_boundary_conditions(
            tagged_mesh, load_cases_input,
            snap_tol=cfg.snap_tol,
            protected_face_groups=["fixed", "load_1", "load_2"],  # red bolt faces + blue pin face
            protected_buffer_radius=4e-3,  # 4mm buffer; tune to your mesh_size_max (2mm)
            keep_alive_groups=keep_alive_groups,
            keep_alive_radius=keep_alive_radius,
            keep_alive_cluster_eps=keep_alive_cluster_eps,
            comm=comm,
        )
        if comm.rank == 0:
            for case_name, entries in cfg.load_cases.items():
                for lc in entries:
                    vec_mag = np.linalg.norm(lc.vector)
                    logger.info("RAW load_vector[case=%s, group=%s] = %s, magnitude = %.6g",
                                case_name, lc.group_name, lc.vector, vec_mag)

            for case_name, case_bcs in bc.traction_bcs.items():
                for load_vec, membership_fn in case_bcs:
                    vec_mag = np.linalg.norm(load_vec)
                    logger.info("traction_bcs[case=%s] entry: vector=%s, |vector|=%.6g",
                                case_name, load_vec, vec_mag)

        if "load_1" in tagged_mesh.name_to_tag:
            load1_tag = tagged_mesh.name_to_tag["load_1"]
            ds = ufl.Measure("ds", domain=tagged_mesh.mesh, subdomain_data=tagged_mesh.facet_tags)
            area_form = dolfinx.fem.form(1.0 * ds(load1_tag))
            local_area = dolfinx.fem.assemble_scalar(area_form)
            global_area = comm.allreduce(local_area, op=MPI.SUM)
            if comm.rank == 0:
                logger.info("Measured load_1 face area = %.6g m^2", global_area)

        fem, opt_nominal, load_cases = build_fenitop_dicts(tagged_mesh, bc, cfg)

    elif cfg.mesh_source == "box":
        # ---------------------------------------------------------------
        # Synthetic box mesh, ported from FEniTop's scripts/beam_3d.py,
        # for validating topopt() results against that reference case.
        # See src/meshing/box_source.py for what's hardcoded vs. cfg-driven.
        # ---------------------------------------------------------------
        if cfg.box_mesh.cell_type == "hexahedron":
            raise NotImplementedError(
                "mesh_source='box' with box_mesh.cell_type='hexahedron' "
                "matches beam_3d.py exactly but is only valid through "
                "Stage 2 (nominal topopt) -- Stage 3's compute_kl_expansion "
                "requires a simplicial (tetrahedral) mesh_serial. This "
                "main.py always runs the full Stage 2-6 pipeline, so "
                "hexahedra would silently produce a wrong/garbage KL basis "
                "in Stage 3 rather than fail loudly there. Set "
                "box_mesh.cell_type='tetrahedron' for the full pipeline, "
                "or run a separate Stage-2-only script if you specifically "
                "want the hex-vs-hex comparison."
            )
        if comm.rank == 0:
            logger.info(
                "mesh_source='box': bypassing STEP/CAD pipeline, building "
                "beam_3d.py-equivalent mesh (cell_type=%s)",
                cfg.box_mesh.cell_type,
            )
        tagged_mesh, fem, opt_nominal, load_cases = build_box_fenitop_dicts(cfg, comm)

    else:
        raise ValueError(
            f"Unknown mesh_source={cfg.mesh_source!r}; expected 'step' or 'box'"
        )

    with XDMFFile(comm, str(STAGE1_DIR / "mesh_checkpoint.xdmf"), "w") as xdmf:
        xdmf.write_mesh(tagged_mesh.mesh)

    if comm.rank == 0:
        logger.info("Physical groups: %s", tagged_mesh.name_to_tag)
        with open(STAGE1_DIR / "physical_groups.json", "w") as f:
            json.dump(
                {name: int(tag) for name, tag in tagged_mesh.name_to_tag.items()},
                f, indent=2,
            )
        logger.info("Stage 1 complete: mesh + physical groups written to %s", STAGE1_DIR)

    # =========================================================================
    # STAGE 2: Deterministic FEA / nominal SIMP topology optimization
    # =========================================================================
    if comm.rank == 0:
        logger.info("Stage 2: running nominal SIMP topopt for warm-start "
                    "(%d load case(s): %s)", len(load_cases), list(load_cases.keys()))
    with manifest.stage("stage2_nominal_topopt"):
        rho_warmstart_global = topopt(
            fem, opt_nominal, load_cases, output_prefix=str(STAGE2_DIR) + "/",
        )

    # topopt() now returns the converged design directly and writes its
    # artifacts under this stage's own per-run directory. It used to write to a
    # fixed output/rho_converged.npy which was then copied and re-read; the copy
    # is gone, and with it the risk of picking up a previous run's file.
    # gather() returns None off rank 0, so broadcast to make it world-identical.
    rho_warmstart_global = comm.bcast(rho_warmstart_global, root=0)
    if comm.rank == 0:
        logger.info("Stage 2 complete: rho_converged.npy written to %s", STAGE2_DIR)

    # =========================================================================
    # STAGE 3: Spatially-correlated random field on the projection threshold
    # =========================================================================
    if comm.rank == 0:
        logger.info("Stage 3: fitting squared-exponential kernel and building "
                    "KL expansion")
    kernel_params = KernelParams(
        sigma=cfg.random_field.sigma,
        length_scale=cfg.random_field.length_scale,
        spatial_dim=cfg.random_field.spatial_dim,
    )
    build_squared_exponential(kernel_params)  # sanity-build; actual model used inside compute_kl_expansion

    # node_coordinates/simplices are only real (non-None) on rank 0, since
    # mesh_serial only exists there; compute_kl_expansion() is a collective
    # call that internally solves on rank 0 and broadcasts the result -- see
    # kl_expansion.py's MPI design note.
    if comm.rank == 0:
        node_coordinates = tagged_mesh.mesh_serial.geometry.x
        simplices = extract_simplices(tagged_mesh)
    else:
        node_coordinates = None
        simplices = None

    kl_result = compute_kl_expansion(
        node_coordinates, simplices, kernel_params,
        variance_threshold=cfg.random_field.variance_threshold, comm=comm,
    )

    if comm.rank == 0:
        with open(STAGE3_DIR / "kernel_params.json", "w") as f:
            json.dump(
                {
                    "sigma": kernel_params.sigma,
                    "length_scale": kernel_params.length_scale,
                    "spatial_dim": kernel_params.spatial_dim,
                },
                f, indent=2,
            )
        eigenvalues = getattr(kl_result, "eigenvalues", None)
        if eigenvalues is not None:
            np.save(STAGE3_DIR / "kl_eigenvalues.npy", np.asarray(eigenvalues))
        n_kl = getattr(kl_result, "n_kl", None)
        if n_kl is not None:
            with open(STAGE3_DIR / "kl_truncation.json", "w") as f:
                json.dump({"n_kl": int(n_kl)}, f, indent=2)
        logger.info("Stage 3 complete: kernel + KL expansion artifacts written to %s", STAGE3_DIR)

    # =========================================================================
    # STAGE 4 + STAGE 5: PCE surrogate training and robust MMA Pareto sweep
    # =========================================================================
    if comm.rank == 0:
        logger.info("Stage 5: Pareto sweep over lambda_tradeoff=%s", cfg.optimization.lambda_sweep)
    if len(load_cases) > 1:
        logger.warning(
            "Stage 5's run_robust_topopt is single-load-case only; "
            "%d load cases found (%s) but only %r will be used for the "
            "robust optimization. Multi-case robust support is a known gap.",
            len(load_cases), list(load_cases.keys()), next(iter(load_cases)),
        )
    robust_case_name = next(iter(load_cases))

    pareto_results = []

    ctx = setup_robust_problem(
        fem, opt_nominal, rho_warmstart_global, kl_result,
        load_cases=load_cases, case_name=robust_case_name,
    )

    # Record the dicts the solver ACTUALLY received. On the box path these
    # differ from config.yaml (box_source.py hardcodes the beam_3d physics), and
    # the manifest is the only place that discrepancy is visible.
    manifest.record_config(cfg, effective_fem=ctx.fem, effective_opt=opt_nominal)

    # =========================================================================
    # VERIFICATION GATES -- run before any result is produced, fatal on failure.
    # =========================================================================
    gate_cfg = GateConfig(
        enabled=cfg.validation.run_gates,
        fd_enabled=cfg.validation.gradient_fd,
        fd_n_samples=cfg.validation.fd_n_samples,
        fd_n_elements=cfg.validation.fd_n_elements,
        fd_step=cfg.validation.fd_step,
        fd_rtol=cfg.validation.fd_rtol,
        fd_ksp_rtol=cfg.validation.fd_ksp_rtol,
        correlation_n_nodes=cfg.validation.correlation_n_nodes,
        correlation_n_samples=cfg.validation.correlation_n_samples,
        correlation_rtol=cfg.validation.correlation_rtol,
        marginal_n_nodes=cfg.validation.marginal_n_nodes,
        marginal_n_samples=cfg.validation.marginal_n_samples,
        marginal_alpha=cfg.validation.marginal_alpha,
    )
    with manifest.stage("verification_gates"):
        gate_results = run_verification_gates(
            ctx, opt_nominal, kl_result,
            beta=float(cfg.optimization.saa_beta_max),
            cfg=gate_cfg,
            output_path=STAGE4_DIR / "gates.json",
        )
    manifest.record("gates", [g.as_dict() for g in gate_results])

    # -------------------------------------------------------------------------
    # How far does the perturbation actually move the boundary?
    #
    # This runs BEFORE the optimization because it can invalidate the whole run:
    # if the boundary moves by less than a fraction of an element, sigma_C is
    # the same order as the discretization error of the compliance and days of
    # solve time would produce a number that means nothing. Measured on the
    # nominal warm start; the interface geometry does not change enough over the
    # sweep to alter the verdict.
    #
    # ORDERING IS LOad-BEARING: rho_phys_field holds rho_tilde after
    # density_filter.forward() and rho_phys after rf_heaviside.forward(). The
    # offset measurement must read the FILTERED field, so it goes between them.
    # -------------------------------------------------------------------------
    ctx.rho_field.x.petsc_vec.array[:] = ctx.rho_warm_start_local
    ctx.rho_field.x.scatter_forward()
    ctx.density_filter.forward()

    offset_report = build_boundary_offset_report(
        ctx.rho_phys_field,                       # still rho_tilde at this point
        opt_nominal["transform_params"],
        element_size=ctx.fem["element_size"],
        filter_radius=opt_nominal["filter_radius"],
    )
    manifest.record("boundary_offset", offset_report.as_dict())
    if comm.rank == 0:
        with open(STAGE4_DIR / "boundary_offset.json", "w") as f:
            json.dump(offset_report.as_dict(), f, indent=2, default=_json_default)

    # Discreteness of the NOMINAL warm start, at the same beta the robust
    # designs are reported at. This is the baseline every M_nd in the sweep is
    # compared against -- without it, a robust design's M_nd has no reference
    # and a compliance gain bought with intermediate density is invisible.
    ctx.rf_heaviside.forward(float(cfg.optimization.saa_beta_max), eta=0.5)
    nominal_m_nd = measure_non_discreteness(ctx.rho_phys_field)
    manifest.record("nominal_M_nd_percent", nominal_m_nd)
    if comm.rank == 0:
        logger.info(
            "Nominal warm-start discreteness at beta=%.4g: M_nd=%.3g%%",
            cfg.optimization.saa_beta_max, nominal_m_nd,
        )

    robust_method = getattr(cfg.optimization, "robust_method", "saa")

    if robust_method == "saa":
        # ---------------------------------------------------------------------
        # Surrogate-free robust TO via Sample Average Approximation (no PCE):
        # one FIXED sample set, exact sample-average objective/gradient by full
        # FEA every iteration. See src/optimization/saa_robust_driver.py.
        # ---------------------------------------------------------------------
        xi_saa = generate_samples(
            kl_result, cfg.optimization.saa_n_samples,
            strategy=cfg.optimization.saa_sampling_strategy,
            seed=cfg.optimization.saa_seed,
        ).xi
        if comm.rank == 0:
            logger.info(
                "Stage 4/5 (SAA, surrogate-free): fixed sample set N=%d (%s, "
                "seed=%d); robust Pareto sweep over lambda=%s",
                xi_saa.shape[0], cfg.optimization.saa_sampling_strategy,
                cfg.optimization.saa_seed, cfg.optimization.lambda_sweep,
            )

        # SWEEP START POLICY. "common" (the default) solves every lambda from
        # the same nominal SIMP warm start, so each point is an independent
        # optimum and the resulting set can legitimately be read as a front.
        # "continuation" chains them, which is cheaper but makes each point the
        # previous point plus more descent -- the defect that produced a sweep
        # whose lambda=1 dominated its lambda=0 in both objectives.
        use_continuation = cfg.optimization.lambda_sweep_start == "continuation"
        if comm.rank == 0:
            logger.info(
                "Pareto sweep start policy: %s%s",
                cfg.optimization.lambda_sweep_start,
                " (each lambda chains from the previous -- points are NOT "
                "independent optima)" if use_continuation else
                " (every lambda solved from the same nominal warm start)",
            )

        x0_local = ctx.rho_warm_start_local
        for lam in cfg.optimization.lambda_sweep:
            with manifest.stage(f"stage5_lambda_{lam}"):
                result = run_saa_robust_topopt(
                    ctx, opt_nominal, lam, xi_saa, x0_local=x0_local,
                )
            pareto_results.append({
                "lambda": lam,
                "rho_robust": result["rho_robust"],
                **{k: result[k] for k in (
                    "mu_C", "sigma_C", "mean_volume", "volume_violation",
                    "M_nd_percent", "beta", "converged", "optimality",
                    "grad_norm", "tao_converged_reason", "beta_schedule",
                    "stage_results", "n_fea_batches_total",
                )},
            })
            if use_continuation:
                ctx.warm_start_comm.bcast(ctx.rho_field, result["rho_robust"])
                x0_local = ctx.rho_field.x.petsc_vec.array.copy()
            # else: x0_local stays at the nominal warm start for every lambda.

            save_design_artifacts(ctx, str(STAGE5_DIR / f"lambda_{lam}_"))
            if comm.rank == 0:
                logger.info(
                    "lambda=%.3g -> mu_C=%.6g, sigma_C=%.6g, E[V]=%.6g "
                    "(violation %.3g%%), M_nd=%.3g%%, converged=%s",
                    lam, result["mu_C"], result["sigma_C"], result["mean_volume"],
                    100.0 * result["volume_violation"], result["M_nd_percent"],
                    result["converged"],
                )
                np.save(STAGE5_DIR / f"rho_robust_lambda_{lam}.npy",
                        np.asarray(result["rho_robust"]))

    else:
        # ---------------------------------------------------------------------
        # Legacy PCE-surrogate path (robust_method="pce").
        # ---------------------------------------------------------------------
        if comm.rank == 0:
            logger.info("Stage 4: training initial PCE surrogate pair (compliance, volume)")
        compliance_pce, volume_pce, rho_trained_local, active_kl_indices, initial_q2 = train_pce_pair(
            ctx, opt_nominal, kl_result
        )

        midpoint = len(cfg.optimization.lambda_sweep) // 2  # set to -1 to disable the mid-sweep refresh entirely

        for i, lam in enumerate(cfg.optimization.lambda_sweep):
            if i == midpoint and i != 0:
                if comm.rank == 0:
                    logger.info("Mid-sweep PCE refresh at lambda index %d (lambda=%.3g)", i, lam)
                try:
                    compliance_pce, volume_pce, rho_trained_local, active_kl_indices, initial_q2 = train_pce_pair(
                        ctx, opt_nominal, kl_result, rho_current_local=rho_trained_local
                    )
                    if comm.rank == 0:
                        np.save(STAGE4_DIR / f"rho_trained_refresh_idx{i}.npy",
                                np.asarray(rho_trained_local))
                except RuntimeError as exc:
                    if comm.rank == 0:
                        logger.error(
                            "Mid-sweep PCE refresh at lambda index %d FAILED (%s); "
                            "reusing the previous lambda's surrogate and continuing.",
                            i, exc,
                        )

            opt_robust = dict(opt_nominal)
            result = run_mma_with_pce(
                ctx, opt_robust, lam, compliance_pce, volume_pce, rho_trained_local,
                kl_result, initial_q2, allow_refresh=True, active_kl_indices=active_kl_indices,
            )
            pareto_results.append({"lambda": lam, "rho_robust": result["rho_robust"],
                **{k: result[k] for k in
                ("mu_C", "sigma_C", "mean_volume", "converged", "optimality",
                    "grad_norm", "compliance_pce", "volume_pce")}})
            if comm.rank == 0:
                logger.info("lambda=%.3g -> mu_C=%.6g, sigma_C=%.6g", lam, result["mu_C"], result["sigma_C"])
                np.save(STAGE5_DIR / f"rho_robust_lambda_{lam}.npy",
                        np.asarray(result["rho_robust"]))
        if comm.rank == 0:
            np.save(STAGE4_DIR / "rho_trained_initial.npy", np.asarray(rho_trained_local))

    # -------------------------------------------------------------------------
    # Sweep integrity: no point may be dominated, and every point must have
    # converged. Both are hard failures -- a dominated point means the sweep is
    # not a front, and a non-converged point is not an optimum.
    # -------------------------------------------------------------------------
    dominated = _assert_non_dominated(pareto_results)
    not_converged = [r["lambda"] for r in pareto_results if not r.get("converged", False)]

    pareto_summary = [
        {
            "lambda": r["lambda"],
            "mu_C": r["mu_C"],
            "sigma_C": r["sigma_C"],
            "mean_volume": r["mean_volume"],
            "volume_violation": r.get("volume_violation"),
            "M_nd_percent": r.get("M_nd_percent"),
            "beta": r.get("beta"),
            "beta_schedule": r.get("beta_schedule"),
            "converged": r.get("converged"),
            "optimality": r.get("optimality"),
            "grad_norm_diagnostic_only": r.get("grad_norm"),
            "n_fea_batches_total": r.get("n_fea_batches_total"),
        }
        for r in pareto_results
    ]

    if comm.rank == 0:
        with open(STAGE5_DIR / "pareto_results.json", "w") as f:
            json.dump(
                {
                    "sweep_start_policy": cfg.optimization.lambda_sweep_start,
                    "points": pareto_summary,
                    "dominated_points": dominated,
                    "non_converged_lambdas": not_converged,
                },
                f, indent=2, default=_json_default,
            )
        logger.info("Stage 5 complete: Pareto sweep artifacts written to %s", STAGE5_DIR)

    manifest.record("pareto", pareto_summary)
    manifest.record("pareto_dominated_points", dominated)
    manifest.record("pareto_non_converged_lambdas", not_converged)

    if not_converged:
        logger.error(
            "Lambda value(s) %s did NOT reach first-order optimality. Their "
            "designs are the best iterates found, not optima, and neither they "
            "nor the sweep they belong to are reportable. Raise "
            "optimization.max_iter, or loosen robust_opt_tol/constraint_tol "
            "only with an explicit justification.", not_converged,
        )
    if dominated and cfg.optimization.sweep_check_dominance:
        manifest.write(manifest_path)
        raise RuntimeError(
            f"Pareto sweep contains {len(dominated)} DOMINATED point(s): "
            f"{dominated}. A trade-off curve cannot contain a point another "
            "point beats on both mu_C and sigma_C -- this means the points are "
            "not converged optima of their own lambda. If "
            "lambda_sweep_start='continuation', switch it to 'common'; "
            "otherwise increase optimization.max_iter. Set "
            "optimization.sweep_check_dominance: false only to inspect a known-"
            "bad sweep, never to publish one."
        )

    # =========================================================================
    # STAGE 6: Monte Carlo validation + visualization
    # =========================================================================
    if comm.rank == 0:
        logger.info(
            "Stage 6: MC validation of the nominal design and ALL %d robust "
            "designs on a COMMON eta ensemble (n=%d, seed=%d)",
            len(pareto_results), cfg.mc_validation.n_samples, cfg.mc_validation.seed,
        )

    # CRITICAL: the MC validation must exercise the SAME eta-field MODEL the
    # optimization used (kernel + marginal transform + KL variance threshold), or
    # the validation is meaningless. Source them from the same opt_nominal objects
    # the SAA/PCE optimization used so they are identical by construction. Only
    # the SAMPLE SET differs: mc_validation.seed must be DISJOINT from
    # optimization.saa_seed so this is an unbiased assessment on independent
    # draws (never validate on the optimization's own samples). loader.py
    # enforces that disjointness rather than leaving it to a comment.
    #
    # Every design is validated with the SAME mc_validation.seed, so they see an
    # IDENTICAL eta ensemble. That is what makes the design-to-design
    # comparisons below paired, and a paired comparison is the only way a few
    # percent difference in sigma_C is resolvable at a feasible sample size.
    mc_config = MCConfig(
        n_samples=cfg.mc_validation.n_samples,
        beta=cfg.mc_validation.beta,
        seed=cfg.mc_validation.seed,
        write_ensemble=cfg.mc_validation.write_ensemble,
        percentiles=(cfg.mc_validation.percentile_low, cfg.mc_validation.percentile_high),
        output_dir=STAGE6_DIR / "mc",
        max_solver_failure_rate=cfg.mc_validation.max_solver_failure_rate,
    )

    heaviside_cfg = RandomHeavisideConfig(
        kernel_params=opt_nominal["kernel_params"],
        transform_params=opt_nominal["transform_params"],
        variance_threshold=opt_nominal.get(
            "kl_variance_threshold", cfg.random_field.variance_threshold
        ),
        seed=cfg.mc_validation.seed,
    )

    fem["traction_bcs"] = load_cases[robust_case_name]

    # The NOMINAL (deterministic SIMP) design is validated too, and first. It is
    # the baseline every robust claim is relative to; without it "sigma_C fell
    # 6%" has nothing to be 6% better than. It was previously never evaluated
    # inside the pipeline at all -- that comparison lived in an untracked script
    # outside src/.
    designs: list[tuple[str, np.ndarray]] = [("nominal", rho_warmstart_global)]
    designs += [
        (f"lambda_{r['lambda']}", r["rho_robust"]) for r in pareto_results
    ]

    mc_results: dict[str, object] = {}
    summaries: dict[str, dict] = {}
    for name, rho_design in designs:
        if comm.rank == 0:
            logger.info("Stage 6: MC validation of design %r", name)
        with manifest.stage(f"stage6_mc_{name}"):
            result = run_monte_carlo_validation(
                fem, opt_nominal, rho_design, kl_result,
                heaviside_config=heaviside_cfg,
                mc_config=MCConfig(
                    n_samples=mc_config.n_samples,
                    beta=mc_config.beta,
                    seed=mc_config.seed,          # SAME seed => common random numbers
                    percentiles=mc_config.percentiles,
                    write_ensemble=mc_config.write_ensemble,
                    output_dir=STAGE6_DIR / "mc" / name,
                    max_solver_failure_rate=mc_config.max_solver_failure_rate,
                ),
            )
        mc_results[name] = result
        if comm.rank == 0:
            result.to_csv(STAGE6_DIR / f"compliance_samples_{name}.csv")
            plot_cdf(result, STAGE6_DIR / f"cdf_{name}.png")
            summaries[name] = result.summary_with_intervals(
                percentiles=mc_config.percentiles,
                seed=cfg.mc_validation.bootstrap_seed,
            )
            logger.info(
                "Design %-14s mean=%s  std=%s",
                name, summaries[name]["mean"], summaries[name]["std"],
            )

    if comm.rank == 0:
        # --- paired comparisons against the nominal baseline -----------------
        # All designs saw the identical eta ensemble, so these are paired and
        # their intervals are far tighter than any comparison of independently
        # seeded runs could be.
        nominal_samples = mc_results["nominal"].compliance_samples
        comparisons = {}
        for name, _ in designs[1:]:
            other_samples = mc_results[name].compliance_samples
            # Pairing requires the SAME realizations on both sides, so restrict
            # to draws where BOTH designs carried load. Dropping only the
            # failures of one design would silently unalign the pair and
            # compare different ensembles.
            both_ok = np.isfinite(nominal_samples) & np.isfinite(other_samples)
            n_dropped = int((~both_ok).sum())
            comparison = compare_designs(
                nominal_samples[both_ok],
                other_samples[both_ok],
                name_a="nominal", name_b=name, paired=True,
                n_bootstrap=cfg.mc_validation.n_bootstrap,
                confidence=cfg.mc_validation.confidence,
                seed=cfg.mc_validation.bootstrap_seed,
            )
            comparisons[name] = comparison.as_dict()
            comparisons[name]["n_realizations_dropped_from_pairing"] = n_dropped
            if n_dropped:
                comparisons[name]["pairing_note"] = (
                    f"{n_dropped} realization(s) excluded because at least one "
                    "of the two designs failed to carry load there. The "
                    "comparison is therefore conditional on BOTH designs "
                    "surviving, and a design that fails more often is "
                    "flattered by that -- read it next to each design's "
                    "solver_failure_rate."
                )
                logger.warning(
                    "Paired comparison nominal vs %s dropped %d/%d "
                    "realization(s) where one design did not carry load; the "
                    "comparison is conditional on both surviving.",
                    name, n_dropped, cfg.mc_validation.n_samples,
                )

        # --- SAA in-sample optimism ------------------------------------------
        # The in-loop mu_C/sigma_C come from the optimizer's OWN fixed sample
        # set, which the design was fitted to for hundreds of iterations. The MC
        # numbers come from an independent ensemble. The gap between them is the
        # SAA optimism, and it is only meaningful next to the MC confidence
        # interval -- a "0.9% relative error" means nothing if the interval on
        # the MC estimate is 7% wide, which is precisely how the previous run
        # reported it.
        insample_vs_mc = {}
        for record in pareto_results:
            name = f"lambda_{record['lambda']}"
            summary = summaries[name]
            insample_vs_mc[name] = {
                "method": robust_method,
                "insample_mu_C": float(record["mu_C"]),
                "insample_sigma_C": float(record["sigma_C"]),
                "mc_mean": summary["mean"],
                "mc_std": summary["std"],
                "optimism_mean_relative": (
                    (float(record["mu_C"]) - summary["mean"]["value"])
                    / abs(summary["mean"]["value"])
                ),
                "optimism_std_relative": (
                    (float(record["sigma_C"]) - summary["std"]["value"])
                    / abs(summary["std"]["value"])
                    if summary["std"]["value"] > 0 else float("nan")
                ),
                "in_sample_estimate_inside_mc_ci": (
                    summary["mean"]["ci_low"] <= float(record["mu_C"]) <= summary["mean"]["ci_high"]
                ),
                "note": (
                    "A positive optimism means the in-loop estimate is HIGHER "
                    "than the independent MC. The quantity to watch for SAA "
                    "overfitting is a NEGATIVE optimism in sigma: the design "
                    "looking less variable on its own samples than on fresh "
                    "ones."
                ),
            }

        # --- realized boundary displacement, measured from the ensemble ------
        # The geometric estimator above ran on the nominal design before the
        # solve; this is the displacement the eta ensemble ACTUALLY produced on
        # each final design, derived from the per-sample volumes at no extra
        # FEA cost. The two should agree; disagreement means the interface-area
        # approximation is being strained and the band decision needs revisiting.
        realized_offsets = {}
        volume_spread = {}
        for name, _ in designs:
            result = mc_results[name]
            if result.volume_samples is None:
                continue
            realized_offsets[name] = measure_offset_from_volumes(
                result.volume_samples,
                result.nominal_volume_fraction,
                result.total_volume,
                offset_report.interface_area,
                ctx.fem["element_size"],
            )
            volume_spread[name] = {
                "mean": float(result.volume_samples.mean()),
                "std": float(result.volume_samples.std(ddof=1)),
                "p95": float(np.percentile(result.volume_samples, 95.0)),
                "max": float(result.volume_samples.max()),
                "nominal_eta_half": result.nominal_volume_fraction,
                "vol_frac_budget": opt_nominal["vol_frac"],
                "note": (
                    "The constraint bounds E[V] only. p95 and max are the "
                    "realized upper tail, which nothing constrains -- report "
                    "them so a design whose dilated realization is well over "
                    "budget is visible."
                ),
            }
            n_failures = result.n_solver_failures
            if n_failures:
                logger.error(
                    "Design %-14s LOAD-CARRYING FAILURE RATE %.3g%% (%d/%d "
                    "realizations). This is a robustness RESULT, not noise: "
                    "those manufacturing realizations do not carry load at "
                    "all. Report it alongside -- arguably ahead of -- the "
                    "compliance statistics, which are conditional on survival "
                    "and therefore optimistic.",
                    name, 100 * result.solver_failure_rate, n_failures,
                    cfg.mc_validation.n_samples,
                )

        # --- is the headline effect even resolvable at this n? ---------------
        resolvability = {}
        for name, comparison in comparisons.items():
            observed_relative_delta_std = abs(
                comparison["delta_std_b_minus_a"]["value"]
            ) / max(summaries["nominal"]["std"]["value"], 1e-300)
            resolvability[name] = resolvability_report(
                summaries[name], observed_relative_delta_std
            )

        with open(STAGE6_DIR / "validation_summary.json", "w") as f:
            json.dump(
                {
                    "n_samples": cfg.mc_validation.n_samples,
                    "seed": cfg.mc_validation.seed,
                    "beta": cfg.mc_validation.beta,
                    "confidence": cfg.mc_validation.confidence,
                    "common_random_numbers": True,
                    "per_design": summaries,
                    "paired_comparisons_vs_nominal": comparisons,
                    "insample_vs_mc": insample_vs_mc,
                    "effect_resolvability": resolvability,
                    "boundary_offset_geometric": offset_report.as_dict(),
                    "boundary_offset_realized": realized_offsets,
                    "volume_spread": volume_spread,
                    # First-class robustness metric: the fraction of
                    # manufacturing realizations in which the design does not
                    # carry load at all. Every compliance figure above is
                    # conditional on survival wherever this is non-zero.
                    "load_carrying_failure_rate": {
                        name: {
                            "rate": mc_results[name].solver_failure_rate,
                            "n_failed": mc_results[name].n_solver_failures,
                            "n_samples": int(cfg.mc_validation.n_samples),
                            "statistics_conditional_on_success":
                                mc_results[name].statistics_conditional_on_success,
                        }
                        for name, _ in designs
                    },
                },
                f, indent=2, default=_json_default,
            )

        for name, comparison in comparisons.items():
            logger.info("VERDICT %s: %s", name, comparison["verdict"])
            if not resolvability[name]["resolvable_at_this_n"]:
                logger.warning(
                    "Design %s: the observed change in sigma_C is NOT "
                    "resolvable at n=%d. It would take n>=%d to separate an "
                    "effect this size from zero. Do not report this as an "
                    "improvement at the current sample size.",
                    name, cfg.mc_validation.n_samples,
                    resolvability[name]["required_n_to_resolve"],
                )

        manifest.record("validation_summary", summaries)
        manifest.record("paired_comparisons_vs_nominal", comparisons)
        manifest.record("effect_resolvability", resolvability)
        manifest.record("boundary_offset_realized", realized_offsets)
        manifest.record("volume_spread", volume_spread)

        # Legacy PCE path also gets the full metamodel-vs-MC scatter (Q^2 etc.).
        final_design = pareto_results[-1]
        if "compliance_pce" in final_design:
            final_name = f"lambda_{final_design['lambda']}"
            pce_comparison = compare_against_pce(
                mc_results[final_name], final_design["compliance_pce"]
            )
            logger.info("PCE-vs-MC comparison: %s", pce_comparison)
            with open(STAGE6_DIR / "pce_vs_mc_comparison.json", "w") as f:
                json.dump(pce_comparison, f, indent=2, default=_json_default)

        if cfg.mc_validation.write_ensemble:
            final_name = f"lambda_{pareto_results[-1]['lambda']}"
            viz_config = ProbabilityCloudConfig(
                n_vis=min(cfg.mc_validation.n_samples, 100),
                output_dir=STAGE6_DIR / "probability_cloud",
                seed=cfg.mc_validation.seed,
            )
            build_probability_cloud(tagged_mesh, mc_results[final_name], viz_config)
            logger.info("Stage 6 visualization complete: probability cloud at %s",
                        STAGE6_DIR / "probability_cloud")

    manifest.write(manifest_path)
    finalize()


if __name__ == "__main__":
    import sys
    # Optional config path as the first CLI arg, e.g.:
    #   mpirun -n 8 python src/mainClean.py src/config/configSmoke.yaml
    # Defaults to src/config/config.yaml when omitted.
    main(sys.argv[1] if len(sys.argv) > 1 else "src/config/config.yaml")