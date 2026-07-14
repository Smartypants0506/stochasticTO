"""standalone_stage6.py — rerun only Stage 6 MC validation using saved rho_robust checkpoints."""
import logging
import numpy as np
from mpi4py import MPI

from src.config.loader import load_config
from src.random_fields.kernel import KernelParams, build_squared_exponential
from src.meshing.importer import import_and_heal, finalize
from src.meshing.mesher import extract_simplices, MeshingConfig, tag_physical_groups, generate_mesh, import_to_dolfinx
from src.meshing.mapper import build_boundary_conditions
from src.fea.fenitop_adapter import build_fenitop_dicts
from src.topology.heaviside_projection_glue import RandomHeavisideConfig
from src.random_fields.threshold_transform import MarginalTransformParams
from src.validation.monte_carlo import MCConfig, run_monte_carlo_validation, plot_cdf
from pathlib import Path

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

def main(config_path: str = "src/config/configSmoke.yaml", lambda_pick: str = "1"):
    cfg = load_config(config_path)
    entities = import_and_heal(cfg.step_file)
    mesh_cfg = MeshingConfig(mesh_size_max=cfg.mesh_size_max,
                              color_targets=cfg.color_targets,
                              solid_volume_color=cfg.solid_volume_color)
    comm = MPI.COMM_WORLD
    tag_physical_groups(entities, mesh_cfg)
    generate_mesh(mesh_cfg, comm)
    tagged_mesh = import_to_dolfinx(comm)

    load_vectors = {lc.group_name: lc.vector for lc in cfg.load_cases}
    bc = build_boundary_conditions(tagged_mesh, load_vectors, snap_tol=cfg.snap_tol)
    fem, opt_nominal = build_fenitop_dicts(tagged_mesh, bc, cfg)

    kernel_params = KernelParams(
        sigma=cfg.random_field.sigma,
        length_scale=cfg.random_field.length_scale,
        spatial_dim=cfg.random_field.spatial_dim,
    )
    node_coordinates = tagged_mesh.mesh_serial.geometry.x
    simplices = extract_simplices(tagged_mesh)

    rho_robust = np.load("output/rho_robust_lambda1.npy")
    logger.info("Loaded rho_robust from lambda=%s checkpoint, shape=%s", lambda_pick, rho_robust.shape)

    heaviside_cfg = RandomHeavisideConfig(
        kernel_params=kernel_params,
        transform_params=MarginalTransformParams(
            eta_min=0.3, eta_max=0.7, alpha=2.0, beta=2.0
        ),
        variance_threshold=0.95,
        seed=cfg.mc_validation.seed,
    )
    mc_config = MCConfig(
        n_samples=cfg.mc_validation.n_samples,
        beta=cfg.optimization.beta_max,
        seed=cfg.mc_validation.seed,
    )

    mc_result = run_monte_carlo_validation(
        fem, opt_nominal, rho_robust, node_coordinates, simplices,
        heaviside_config=heaviside_cfg, mc_config=mc_config,
    )

    mc_result.to_csv(Path("output/mc_validation/compliance_samples.csv"))
    plot_cdf(mc_result, Path("output/mc_validation/cdf.png"))
    logger.info("Stage 6 complete: mean=%.6g, std=%.6g", mc_result.mean, mc_result.std)

if __name__ == "__main__":
    main()