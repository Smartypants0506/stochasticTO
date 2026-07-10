"""src/main.py — top-level orchestrator, ties together all six stages."""
from __future__ import annotations
import logging
import numpy as np

from src.config.loader import load_config
from src.fenitop.topopt import topopt
from src.random_fields.kernel import KernelParams, build_squared_exponential
from src.random_fields.kl_expansion import compute_kl_expansion
from src.optimization.dolfiny_mma_driver import run_robust_topopt
from src.surrogate.pce_model import build_pce_gradient_model
from src.validation.monte_carlo import MCConfig, run_monte_carlo_validation, compare_against_pce

from src.meshing.importer import import_and_heal, finalize
from src.meshing.mesher import extract_simplices, MeshingConfig, tag_physical_groups, generate_mesh, import_to_dolfinx
from mpi4py import MPI
from src.meshing.mapper import build_boundary_conditions
from src.fea.fenitop_adapter import build_fenitop_dicts

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

def main(config_path: str = "src/config/configSmoke.yaml") -> None:
    cfg = load_config(config_path)
    entities = import_and_heal(cfg.step_file)
    mesh_cfg = MeshingConfig(mesh_size_max=cfg.mesh_size_max,
                            color_targets=cfg.color_targets,
                            solid_volume_color=cfg.solid_volume_color)
    comm = MPI.COMM_WORLD
    tag_physical_groups(entities, mesh_cfg)
    generate_mesh(mesh_cfg, comm)
    tagged_mesh = import_to_dolfinx(comm)
    assert "fixed" in tagged_mesh.name_to_tag, "No 'fixed' faces tagged — check STEP coloring"
    assert "load_1" in tagged_mesh.name_to_tag, "No 'load_1' faces tagged — check STEP coloring"

    load_vectors = {lc.group_name: lc.vector for lc in cfg.load_cases}
    bc = build_boundary_conditions(tagged_mesh, load_vectors, snap_tol=cfg.snap_tol)

    fem, opt_nominal = build_fenitop_dicts(tagged_mesh, bc, cfg)

    logger.info("Stage 2: running nominal SIMP topopt for warm-start")
    topopt(fem, opt_nominal)
    rho_warmstart = np.load("output/rho_converged.npy")

    logger.info("Stage 3 (KL expansion only, metrology fit deferred): building kernel")
    kernel_params = KernelParams(
        sigma=cfg.random_field.sigma,
        length_scale=cfg.random_field.length_scale,
        spatial_dim=cfg.random_field.spatial_dim,
    )
    covariance_model = build_squared_exponential(kernel_params)
    node_coordinates = tagged_mesh.mesh_serial.geometry.x
    simplices = extract_simplices(tagged_mesh)
    kl_result = compute_kl_expansion(node_coordinates, simplices, kernel_params)
    logger.info("Stage 5: Pareto sweep over lambda_tradeoff=%s", cfg.optimization.lambda_sweep)
    pareto_results = []
    for lam in cfg.optimization.lambda_sweep:
        opt_robust = dict(opt_nominal)
        result = run_robust_topopt(fem, opt_robust, rho_warmstart, lambda_tradeoff=lam, kl_result=kl_result)
        pareto_results.append({"lambda": lam, "rho_robust": result["rho_robust"],
                            **{k: result[k] for k in ("mu_C", "sigma_C", "mean_volume", "kkt_residual")}})
        logger.info("lambda=%.3g -> mu_C=%.6g, sigma_C=%.6g", lam, result["mu_C"], result["sigma_C"])

    logger.info("Stage 6: full-scale MC validation on final robust design")
    final_design = pareto_results[-1]
    mc_config = MCConfig(
        n_samples=cfg.mc_validation.n_samples,
        beta=cfg.optimization.beta_max,
        seed=cfg.mc_validation.seed,
    )
    mc_result = run_monte_carlo_validation(
        fem, opt_nominal, final_design["rho_robust"], node_coordinates, simplices,
        heaviside_config=kl_result, mc_config=mc_config,
    )
    # PCE-vs-MC comparison requires the last trained PCE pair from this lambda's run
    # (must be returned/exposed by run_robust_topopt -- see gap below)

if __name__ == "__main__":
    main()