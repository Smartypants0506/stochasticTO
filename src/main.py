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
from src.fea.fenitop_adapter import build_fenitop_dicts

from src.topology.heaviside_projection_glue import RandomHeavisideConfig
from src.random_fields.threshold_transform import MarginalTransformParams

from dolfinx.io import XDMFFile

logging.basicConfig(level=logging.DEBUG, force=True)
logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)
logger = logging.getLogger(__name__)



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

    with XDMFFile(comm, "meshes/mesh_checkpoint.xdmf", "w") as xdmf:
        xdmf.write_mesh(tagged_mesh.mesh)

    if comm.rank == 0:
        logger.info("Physical groups: %s", tagged_mesh.name_to_tag)
        logger.info("Stage 2: running nominal SIMP topopt for warm-start "
                     "(%d load case(s): %s)", len(load_cases), list(load_cases.keys()))
    topopt(fem, opt_nominal, load_cases)

    # rho_converged.npy is now a GLOBAL array (see topopt.py fix); load it
    # identically on every rank via comm.bcast so all ranks agree on its
    # contents before slicing down to local dofs later.
    if comm.rank == 0:
        rho_warmstart_global = np.load("output/rho_converged.npy")
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
    pareto_results = []
    for lam in cfg.optimization.lambda_sweep:
        opt_robust = dict(opt_nominal)
        result = run_robust_topopt(
            fem, opt_robust, rho_warmstart_global, lambda_tradeoff=lam, kl_result=kl_result,
        )
        pareto_results.append({"lambda": lam, "rho_robust": result["rho_robust"],
                            **{k: result[k] for k in ("mu_C", "sigma_C", "mean_volume", "kkt_residual")}})
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
        variance_threshold=0.95,
        seed=cfg.mc_validation.seed,
    )

    mc_result = run_monte_carlo_validation(
    fem, opt_nominal, final_design["rho_robust"], kl_result,
    heaviside_config=heaviside_cfg, mc_config=mc_config,
)

    if comm.rank == 0:
        mc_result.to_csv(Path("output/mc_validation/compliance_samples.csv"))
        plot_cdf(mc_result, Path("output/mc_validation/cdf.png"))
        logger.info("Stage 6 complete: mean=%.6g, std=%.6g", mc_result.mean, mc_result.std)
    # PCE-vs-MC comparison requires the last trained PCE pair from this lambda's run
    # (must be returned/exposed by run_robust_topopt -- see gap below)

    finalize()


if __name__ == "__main__":
    main()