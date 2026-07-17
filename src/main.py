"""src/main.py — top-level orchestrator, ties together all six stages."""
from __future__ import annotations
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

from dolfinx.io import XDMFFile
import hashlib
import json
import pickle
import shutil

USE_MC_CACHE = True   # <-- flip this to False to force a fresh Stage 6 MC run

MC_CACHE_FILE = Path("output/cache/mc_validation/mc_result.pkl")


logging.basicConfig(level=logging.DEBUG, force=True)
logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)
logger = logging.getLogger(__name__)


def _stable_hash(*objs) -> str:
    """Order-independent, content-based hash for cache keys."""
    h = hashlib.sha256()
    for obj in objs:
        try:
            h.update(json.dumps(obj, sort_keys=True, default=str).encode())
        except TypeError:
            h.update(pickle.dumps(obj))
    return h.hexdigest()[:16]

def _stage2_cache_dir(cache_key: str) -> Path:
    return Path("output/cache/stage2") / cache_key

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
        logger.info("material.youngs_modulus (raw config value) = %.6g", cfg.material.youngs_modulus)
        logger.info("material.poissons_ratio (raw config value) = %.6g", cfg.material.poissons_ratio)
    

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

    with XDMFFile(comm, "meshes/mesh_checkpoint.xdmf", "w") as xdmf:
        xdmf.write_mesh(tagged_mesh.mesh)

    if comm.rank == 0:
        logger.info("Physical groups: %s", tagged_mesh.name_to_tag)

    if comm.rank == 0:
        cache_key = _stable_hash(
            opt_nominal,
            load_cases,
            cfg.mesh_source,
            cfg.model_dump() if hasattr(cfg, "model_dump") else str(cfg),
        )
    else:
        cache_key = None
    cache_key = comm.bcast(cache_key, root=0)

    #cache_dir = _stage2_cache_dir(cache_key)
    cache_file = "output/cache/stage2/706bf66af2b64be2/rho_converged.npy"
    #cache_hit = cache_file.exists()

    if True:
        if comm.rank == 0:
            logger.info("Stage 2: cache hit (key=%s) -- skipping topopt, "
                        "reusing cached rho_converged.npy", cache_key)
    else:
        if comm.rank == 0:
            logger.info("Stage 2: cache miss (key=%s) -- running nominal SIMP "
                        "topopt for warm-start (%d load case(s): %s)",
                        cache_key, len(load_cases), list(load_cases.keys()))
        topopt(fem, opt_nominal, load_cases)
        if comm.rank == 0:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy("output/rho_converged.npy", cache_file)

    # rho_converged.npy is now a GLOBAL array (see topopt.py fix); load it
    # identically on every rank via comm.bcast so all ranks agree on its
    # contents before slicing down to local dofs later.
    if comm.rank == 0:
        rho_warmstart_global = np.load(cache_file)
    else:
        rho_warmstart_global = None
    rho_warmstart_global = comm.bcast(rho_warmstart_global, root=0)

    if comm.rank == 0:
        logger.info("Stage 3 (KL expansion only, metrology fit deferred): building kernel")
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

    kl_result = compute_kl_expansion(node_coordinates, simplices, kernel_params, comm=comm)

    if comm.rank == 0:
        logger.info("Stage 5: Pareto sweep over lambda_tradeoff=%s", cfg.optimization.lambda_sweep)
        if len(load_cases) > 1:
            logger.warning(
                "Stage 5's run_robust_topopt is single-load-case only; "
                "%d load cases found (%s) but only %r will be used for the "
                "robust optimization. Multi-case robust support is a known gap.",
                len(load_cases), list(load_cases.keys()), next(iter(load_cases)),
            )
    pareto_results = []
    robust_case_name = next(iter(load_cases))
    for lam in cfg.optimization.lambda_sweep:
        opt_robust = dict(opt_nominal)
        result = run_robust_topopt(
            fem, opt_robust, rho_warmstart_global, lambda_tradeoff=lam, kl_result=kl_result,
            load_cases=load_cases, case_name=robust_case_name,
        )
        pareto_results.append({"lambda": lam, "rho_robust": result["rho_robust"],
                    **{k: result[k] for k in
                       ("mu_C", "sigma_C", "mean_volume", "kkt_residual",
                        "compliance_pce", "volume_pce")}})
        if comm.rank == 0:
            logger.info("lambda=%.3g -> mu_C=%.6g, sigma_C=%.6g", lam, result["mu_C"], result["sigma_C"])

    if comm.rank == 0:
        logger.info("Stage 6: full-scale MC validation on final robust design")
    final_design = pareto_results[-1]

    mc_config = MCConfig(
        n_samples=cfg.mc_validation.n_samples,
        beta=cfg.optimization.beta_max,
        seed=cfg.mc_validation.seed,
    )

    heaviside_cfg = RandomHeavisideConfig(
        kernel_params=kernel_params,
        transform_params=MarginalTransformParams(
            eta_min=0.3, eta_max=0.7, alpha=2.0, beta=2.0
        ),
        variance_threshold=0.75,
        seed=cfg.mc_validation.seed,
    )

    if USE_MC_CACHE:
        mc_cache_hit = MC_CACHE_FILE.exists() if comm.rank == 0 else False
        mc_cache_hit = comm.bcast(mc_cache_hit, root=0)
    else:
        mc_cache_hit = False

    if mc_cache_hit:
        if comm.rank == 0:
            logger.info("Stage 6 MC validation cache HIT: %s", MC_CACHE_FILE)
            with open(MC_CACHE_FILE, "rb") as f:
                mc_result = pickle.load(f)
        else:
            mc_result = None
        mc_result = comm.bcast(mc_result, root=0)
    else:
        mc_result = run_monte_carlo_validation(
            fem, opt_nominal, final_design["rho_robust"], kl_result,
            heaviside_config=heaviside_cfg, mc_config=mc_config,
        )
        if USE_MC_CACHE and comm.rank == 0:
            MC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(MC_CACHE_FILE, "wb") as f:
                pickle.dump(mc_result, f)
            logger.info("Stage 6 MC validation cache WRITTEN: %s", MC_CACHE_FILE)

    if comm.rank == 0:
        mc_result.to_csv(Path("output/mc_validation/compliance_samples.csv"))
        plot_cdf(mc_result, Path("output/mc_validation/cdf.png"))
        logger.info("Stage 6 complete: mean=%.6g, std=%.6g", mc_result.mean, mc_result.std)
        pce_comparison = compare_against_pce(mc_result, final_design["compliance_pce"])
        logger.info("PCE-vs-MC comparison: %s", pce_comparison)

    finalize()


if __name__ == "__main__":
    main()