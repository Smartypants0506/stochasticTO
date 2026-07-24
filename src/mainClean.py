"""src/main.py — top-level orchestrator, ties together all six stages.

This is a clean, no-cache version of the pipeline: every stage always runs
from scratch, and nothing is ever read back from a prior run to skip work.
Each stage still persists its own results to disk (under output/<stage>/)
purely as a durable record/deliverable for the user -- those files are never
read back in on a subsequent run to shortcut computation.
"""
from __future__ import annotations
import json
import logging
import shutil
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
from src.optimization.saa_robust_driver import run_saa_robust_topopt
from src.sampling.sampler import generate_samples

from dolfinx.io import XDMFFile

logging.basicConfig(level=logging.DEBUG, force=True)
logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output layout: every stage writes exclusively to its own directory. These
# directories are recreated fresh on every run -- nothing here is ever read
# back in to skip a computation.
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path("output")
STAGE1_DIR = OUTPUT_ROOT / "stage1_mesh"
STAGE2_DIR = OUTPUT_ROOT / "stage2_fea"
STAGE3_DIR = OUTPUT_ROOT / "stage3_random_field"
STAGE4_DIR = OUTPUT_ROOT / "stage4_surrogate"
STAGE5_DIR = OUTPUT_ROOT / "stage5_optimization"
STAGE6_DIR = OUTPUT_ROOT / "stage6_validation"


def _fresh_dir(path: Path) -> Path:
    """Ensure `path` exists as an empty directory for this run's outputs."""
    if path.exists():
        shutil.rmtree(path)
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


def main(config_path: str = "src/config/config.yaml") -> None:
    cfg = load_config(config_path)

    if comm.rank == 0:
        _fresh_dir(STAGE1_DIR)
        _fresh_dir(STAGE2_DIR)
        _fresh_dir(STAGE3_DIR)
        _fresh_dir(STAGE4_DIR)
        _fresh_dir(STAGE5_DIR)
        _fresh_dir(STAGE6_DIR)
    comm.Barrier()

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
    topopt(fem, opt_nominal, load_cases)

    # topopt() writes its converged design to output/rho_converged.npy; copy
    # that artifact into this stage's own output directory as the durable
    # record, then load it fresh (no prior-run cache involved -- this is the
    # design we *just* computed above, in this run, on this call).
    if comm.rank == 0:
        shutil.copy("output/rho_converged.npy", STAGE2_DIR / "rho_converged.npy")
        rho_warmstart_global = np.load(STAGE2_DIR / "rho_converged.npy")
        logger.info("Stage 2 complete: rho_converged.npy written to %s", STAGE2_DIR)
    else:
        rho_warmstart_global = None
    rho_warmstart_global = comm.bcast(rho_warmstart_global, root=0)

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

        x0_local = ctx.rho_warm_start_local  # first lambda warm-starts from nominal SIMP
        for lam in cfg.optimization.lambda_sweep:
            result = run_saa_robust_topopt(
                ctx, opt_nominal, lam, xi_saa, x0_local=x0_local,
            )
            pareto_results.append({"lambda": lam, "rho_robust": result["rho_robust"],
                **{k: result[k] for k in ("mu_C", "sigma_C", "mean_volume", "kkt_residual")}})
            # lambda-continuation: warm-start the next lambda from this converged
            # design (scatter the global result back into this rank's local slice).
            ctx.warm_start_comm.bcast(ctx.rho_field, result["rho_robust"])
            x0_local = ctx.rho_field.x.petsc_vec.array.copy()
            if comm.rank == 0:
                logger.info(
                    "lambda=%.3g -> mu_C=%.6g, sigma_C=%.6g, E[V]=%.6g, kkt=%.4g",
                    lam, result["mu_C"], result["sigma_C"], result["mean_volume"],
                    result["kkt_residual"],
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
                ("mu_C", "sigma_C", "mean_volume", "kkt_residual",
                    "compliance_pce", "volume_pce")}})
            if comm.rank == 0:
                logger.info("lambda=%.3g -> mu_C=%.6g, sigma_C=%.6g", lam, result["mu_C"], result["sigma_C"])
                np.save(STAGE5_DIR / f"rho_robust_lambda_{lam}.npy",
                        np.asarray(result["rho_robust"]))
        if comm.rank == 0:
            np.save(STAGE4_DIR / "rho_trained_initial.npy", np.asarray(rho_trained_local))

    if comm.rank == 0:
        logger.info("Stage 4/5 complete: robust Pareto artifacts written to %s", STAGE5_DIR)

        pareto_summary = [
            {
                "lambda": r["lambda"],
                "mu_C": r["mu_C"],
                "sigma_C": r["sigma_C"],
                "mean_volume": r["mean_volume"],
                "kkt_residual": r["kkt_residual"],
            }
            for r in pareto_results
        ]
        with open(STAGE5_DIR / "pareto_results.json", "w") as f:
            json.dump(pareto_summary, f, indent=2, default=_json_default)
        logger.info("Stage 5 complete: Pareto sweep artifacts written to %s", STAGE5_DIR)

    # =========================================================================
    # STAGE 6: Monte Carlo validation + visualization
    # =========================================================================
    if comm.rank == 0:
        logger.info("Stage 6: full-scale MC validation on final robust design")
    final_design = pareto_results[-1]

    # CRITICAL: the MC validation must exercise the SAME eta-field MODEL the
    # optimization used (kernel + marginal transform + KL variance threshold), or
    # the validation is meaningless. Source them from the same opt_nominal objects
    # the SAA/PCE optimization used so they are identical by construction. Only
    # the SAMPLE SET differs: mc_validation.seed must be DISJOINT from
    # optimization.saa_seed so this is an unbiased assessment on independent
    # draws (never validate on the optimization's own samples).
    mc_config = MCConfig(
        n_samples=cfg.mc_validation.n_samples,
        beta=cfg.mc_validation.beta,
        seed=cfg.mc_validation.seed,
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
    mc_result = run_monte_carlo_validation(
        fem, opt_nominal, final_design["rho_robust"], kl_result,
        heaviside_config=heaviside_cfg, mc_config=mc_config,
    )

    if comm.rank == 0:
        mc_result.to_csv(STAGE6_DIR / "compliance_samples.csv")
        plot_cdf(mc_result, STAGE6_DIR / "cdf.png")
        logger.info("Stage 6 complete: mean=%.6g, std=%.6g", mc_result.mean, mc_result.std)
        with open(STAGE6_DIR / "mc_summary.json", "w") as f:
            json.dump({"mean": float(mc_result.mean), "std": float(mc_result.std)},
                    f, indent=2)

        # Unbiased fidelity check: the in-loop robust estimate (mu_C, sigma_C at
        # the final design, over the optimization's OWN sample set) vs the
        # INDEPENDENT high-fidelity MC (different seed, same eta-field model).
        # For SAA there is no PCE metamodel, so this is the primary validation;
        # close agreement means the design generalizes (not overfit to its
        # sample set). Large N shrinks any residual SAA optimism.
        insample_mu = float(final_design["mu_C"])
        insample_sigma = float(final_design["sigma_C"])
        insample_vs_mc = {
            "method": robust_method,
            "insample_mu_C": insample_mu,
            "insample_sigma_C": insample_sigma,
            "mc_mean": float(mc_result.mean),
            "mc_std": float(mc_result.std),
            "rel_err_mean": abs(insample_mu - mc_result.mean) / abs(mc_result.mean),
            "rel_err_std": (abs(insample_sigma - mc_result.std) / abs(mc_result.std)
                            if mc_result.std > 0 else float("nan")),
            "mc_n_samples": int(mc_result.compliance_samples.size),
        }
        logger.info("In-loop-vs-independent-MC fidelity check: %s", insample_vs_mc)
        with open(STAGE6_DIR / "insample_vs_mc.json", "w") as f:
            json.dump(insample_vs_mc, f, indent=2, default=_json_default)

        # Legacy PCE path also gets the full metamodel-vs-MC scatter (Q^2 etc.).
        if "compliance_pce" in final_design:
            pce_comparison = compare_against_pce(mc_result, final_design["compliance_pce"])
            logger.info("PCE-vs-MC comparison: %s", pce_comparison)
            with open(STAGE6_DIR / "pce_vs_mc_comparison.json", "w") as f:
                json.dump(pce_comparison, f, indent=2, default=_json_default)

        viz_config = ProbabilityCloudConfig(
            n_vis=min(cfg.mc_validation.n_samples, 100),
            output_dir=STAGE6_DIR / "probability_cloud",
            seed=cfg.mc_validation.seed,
        )
        build_probability_cloud(tagged_mesh, mc_result, viz_config)
        logger.info("Stage 6 visualization complete: probability cloud written to %s",
                    STAGE6_DIR / "probability_cloud")

    finalize()


if __name__ == "__main__":
    import sys
    # Optional config path as the first CLI arg, e.g.:
    #   mpirun -n 8 python src/mainClean.py src/config/configSmoke.yaml
    # Defaults to src/config/config.yaml when omitted.
    main(sys.argv[1] if len(sys.argv) > 1 else "src/config/config.yaml")