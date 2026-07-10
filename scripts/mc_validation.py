"""End-to-end driver: nominal TO (Steps 1-2) -> MC validation (Step 5).

Requires a real dolfinx/FEniTop environment (not standalone-mockable, unlike
Steps 3-4). Run inside the container where fenitop is installed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.fenitop.topopt import topopt
from src.fenitop.fem import form_fem

from src.random_fields.kernel import KernelParams
from src.random_fields.threshold_transform import MarginalTransformParams
from src.topology.heaviside_projection_glue import RandomHeavisideConfig
from src.validation.monte_carlo import MCConfig, run_monte_carlo_validation, plot_cdf

from dolfinx.mesh import create_rectangle, CellType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [60, 20]],
                        [200, 60], CellType.quadrilateral)

    if MPI.COMM_WORLD.rank == 0:
        mesh_serial = create_rectangle(MPI.COMM_SELF, [[0, 0], [60, 20]],
                                   [200, 60], CellType.quadrilateral)
    else:
        mesh_serial = None
        
    fem_config = {  # FEM parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[0], 0),
    "traction_bcs": [[(0, -0.2),
                      lambda x: (np.isclose(x[0], 60) & np.greater(x[1], 8) & np.less(x[1], 12))]],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}
    opt_config = {  # Topology optimization parameters
    "max_iter": 400,
    "opt_tol": 1e-5,
    "vol_frac": 0.5,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 1.2,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": True,
    "move": 0.02,
    "opt_compliance": True,
}
    if not fem_config or not opt_config:
        raise RuntimeError(
            "Fill in fem_config/opt_config with your verified Step 1/2 "
            "cantilever setup before running this script."
        )

    logger.info("=== Running nominal deterministic TO (Steps 1-2 baseline) ===")
    topopt(fem_config, opt_config)  # writes optimized_design.xdmf per topopt.py's own logic

    logger.info("=== Re-extracting converged density for MC validation ===")
    # topopt() does not return rho_field directly; rebuild the FEM problem and
    # re-run form_fem to get a fresh Function, then load the converged values
    # from the XDMF/checkpoint written by topopt(). For the MVP, re-run
    # form_fem and assume the caller has separately saved rho_field.vector.array
    # to disk (e.g., via np.save inside a modified topopt loop, or by
    # capturing it directly if you control the topopt() call site).
    _, _, _, rho_field, rho_phys_field = form_fem(fem_config, opt_config)
    rho_converged_path = Path("output/rho_converged.npy")
    if not rho_converged_path.exists():
        raise RuntimeError(
            f"{rho_converged_path} not found. Modify your Step 1/2 topopt run "
            "to save np.save('output/rho_converged.npy', rho_field.vector.array) "
            "after convergence, then re-run this script."
        )
    rho_converged = np.load(rho_converged_path)

    node_coordinates = rho_phys_field.function_space.tabulate_dof_coordinates()[:, :2]
    # simplices: extract from the mesh_serial connectivity (Section 8's mesh mapper.py
    # is where this normally comes from; for MVP, derive via a Delaunay pass consistent
    # with the mesh nodes, matching the standalone test pattern from Steps 3-4).
    from scipy.spatial import Delaunay
    simplices = Delaunay(node_coordinates).simplices

    domain_size = node_coordinates.max(axis=0) - node_coordinates.min(axis=0)
    length_scale = 0.2 * domain_size.min() 

    heaviside_config = RandomHeavisideConfig(
        kernel_params=KernelParams(sigma=1.0, length_scale=length_scale, spatial_dim=2),
        transform_params=MarginalTransformParams(eta_min=0.45, eta_max=0.55, alpha=2.0, beta=2.0),
        seed=42,
    )
    mc_config = MCConfig(n_samples=100, beta=12, seed=0)

    print("Domain bounding box:")
    print(f"  x: [{node_coordinates[:, 0].min():.4g}, {node_coordinates[:, 0].max():.4g}]")
    print(f"  y: [{node_coordinates[:, 1].min():.4g}, {node_coordinates[:, 1].max():.4g}]")
    domain_size = node_coordinates.max(axis=0) - node_coordinates.min(axis=0)
    print(f"  domain_size: {domain_size}")
    print(f"  current length_scale={heaviside_config.kernel_params.length_scale}, "
        f"ratio to smallest dimension: {heaviside_config.kernel_params.length_scale / domain_size.min():.4f}")

    logger.info("=== Running Monte Carlo validation loop (Step 5) ===")
    result = run_monte_carlo_validation(
        fem_config, opt_config, rho_converged, node_coordinates, simplices,
        heaviside_config, mc_config,
    )

    output_dir = Path("output/mc_validation")
    result.to_csv(output_dir / "compliance_samples.csv")
    plot_cdf(result, output_dir / "cdf.png")
    

    print(f"\nMC validation summary (n={mc_config.n_samples}):")
    print(f"  mean C     = {result.mean:.6g}")
    print(f"  std C      = {result.std:.6g}")
    print(f"  5th pct    = {result.percentile_low:.6g}")
    print(f"  95th pct   = {result.percentile_high:.6g}")
    print(f"  N_kl used  = {result.n_kl} ({result.variance_explained:.2%} variance)")


if __name__ == "__main__":
    main()