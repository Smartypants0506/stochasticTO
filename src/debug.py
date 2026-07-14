"""src/debug_main.py — fast-path debug driver.

Skips Stage 2 (nominal SIMP topopt) by loading a cached rho_converged.npy
checkpoint. Meshing (Stage 1) and the KL expansion (Stage 3) still run,
since run_robust_topopt requires fem/opt dicts and kl_result -- but the
expensive iterative SIMP loop (which is what actually eats most of the
hour) is removed entirely.

IMPORTANT: rho_converged.npy must have been produced by the SAME mesh
(same STEP file, same mesh_size_max, same comm.size) as this run -- its
shape must match the current rho_field's local dof count exactly, or you
will hit the same local/global shape mismatch bugs we already fixed
elsewhere in this pipeline. If comm.size differs from the run that
produced the checkpoint, this WILL silently corrupt the design field.
"""
from __future__ import annotations
import logging
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI

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

comm = MPI.COMM_WORLD
logging.basicConfig(
    level=logging.INFO if comm.rank == 0 else logging.WARNING,
    force=True,
)
logger = logging.getLogger(__name__)


def main(
    config_path: str = "src/config/config.yaml",
    rho_checkpoint_path: str = "output/rho_converged.npy",
    lambda_override: float | None = None,
) -> None:
    comm = MPI.COMM_WORLD

    cfg = load_config(config_path)

    entities = import_and_heal(cfg.step_file)
    mesh_cfg = MeshingConfig(mesh_size_max=cfg.mesh_size_max,
                            color_targets=cfg.color_targets,
                            solid_volume_color=cfg.solid_volume_color)
    tag_physical_groups(entities, mesh_cfg)
    generate_mesh(mesh_cfg, comm)
    tagged_mesh = import_to_dolfinx(comm)
    assert "fixed" in tagged_mesh.name_to_tag, "No 'fixed' faces tagged — check STEP coloring"
    assert "load_1" in tagged_mesh.name_to_tag, "No 'load_1' faces tagged — check STEP coloring"

    load_vectors = {lc.group_name: lc.vector for lc in cfg.load_cases}
    bc = build_boundary_conditions(tagged_mesh, load_vectors, snap_tol=cfg.snap_tol)

    fem, opt_nominal = build_fenitop_dicts(tagged_mesh, bc, cfg)

    logger.info("Loading cached warm-start from %s (skipping SIMP topopt)", rho_checkpoint_path)

    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt_nominal)
    S0_comm = Communicator(rho_field.function_space, fem["mesh_serial"])

    if comm.rank == 0:
        rho_warmstart_global = np.load(rho_checkpoint_path)
        if rho_warmstart_global.shape[0] != S0_comm.num_global_dofs:
            raise ValueError(
                f"rho_converged.npy has {rho_warmstart_global.shape[0]} global dofs, "
                f"but this mesh has {S0_comm.num_global_dofs} global DG0 dofs. "
                f"The checkpoint was produced with a different mesh_size_max or STEP file."
            )
    else:
        rho_warmstart_global = None

    rho_warmstart_global = comm.bcast(rho_warmstart_global, root=0)
    S0_comm.bcast(rho_field, rho_warmstart_global)
    rho_warmstart = rho_field.x.petsc_vec.array.copy()  # this rank's local slice, correctly ordered

    # --- Stage 3: KL expansion (still needed by run_robust_topopt) ---
    logger.info("Building KL expansion")
    kernel_params = KernelParams(
        sigma=cfg.random_field.sigma,
        length_scale=cfg.random_field.length_scale,
        spatial_dim=cfg.random_field.spatial_dim,
    )
    node_coordinates = tagged_mesh.mesh_serial.geometry.x if comm.rank == 0 else None
    simplices = extract_simplices(tagged_mesh) if comm.rank == 0 else None
    kl_result = compute_kl_expansion(node_coordinates, simplices, kernel_params, comm=comm)

    # --- Stage 5: run just ONE lambda to reproduce the crash fast ---
    lambdas = [lambda_override] if lambda_override is not None else cfg.optimization.lambda_sweep[:1]
    logger.info("Running robust loop for lambda_tradeoff=%s only (debug mode)", lambdas)

    for lam in lambdas:
        opt_robust = dict(opt_nominal)
        result = run_robust_topopt(
            fem, opt_robust, rho_warmstart, lambda_tradeoff=lam, kl_result=kl_result,
        )
        logger.info(
            "lambda=%.3g -> mu_C=%.6g, sigma_C=%.6g",
            lam, result["mu_C"], result["sigma_C"],
        )


if __name__ == "__main__":
    lam_arg = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lambda_override=lam_arg)