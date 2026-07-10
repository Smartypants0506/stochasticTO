"""End-to-end driver: load mesh ONCE, build FEniTop objects ONCE, run the
robust objective/gradient (Step 6) and its finite-difference verification
gate, using the ACTUAL verified beam_2d.py cantilever configuration.

Fixes applied:
1. HDF5/PETSc benign diagnostic noise suppressed (does not affect results).
2. KL expansion computed on a COARSE auxiliary grid sized to length_scale,
   then interpolated up to the full FEA mesh -- avoids the O(N^3) eigen-
   decomposition cost of running KL directly on ~12,261 FEA mesh nodes,
   which was the actual source of the long runtime.
"""
from __future__ import annotations

import os
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

import logging
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from dolfinx.mesh import create_rectangle, CellType

PETSc.Sys.pushErrorHandler("ignore")  # suppress benign PETSc/HDF5 diagnostic stack traces

from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter
from src.fenitop.sensitivity import Sensitivity

from src.random_fields.kernel import KernelParams
from src.random_fields.kl_expansion import compute_kl_expansion, sample_gaussian_field
from src.random_fields.threshold_transform import MarginalTransformParams, ThresholdMarginalTransform
from src.topology.heaviside_projection_glue import RandomFieldHeaviside, RandomHeavisideConfig
from src.optimization.robust_objective import (
    evaluate_robust_samples,
    compute_robust_objective_value,
    compute_mean_volume_constraint,
    RobustObjectiveConfig,
)
from src.optimization.robust_gradient import (
    compute_robust_gradient,
    compute_mean_volume_gradient,
    verify_robust_gradient_fd,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
if comm.size > 1:
    raise RuntimeError(
        "This driver requires serial execution (comm.size == 1) because "
        "RandomFieldHeaviside is not yet MPI-consistent. Run with plain "
        "'python3 scripts/run_robust_gradient_check.py', not mpirun."
    )

# --- STEP 1: Build mesh ONCE, in memory, matching beam_2d.py exactly ---
DOMAIN_LENGTH, DOMAIN_HEIGHT = 60, 20
NX, NY = 200, 60

mesh = create_rectangle(
    comm, [[0, 0], [DOMAIN_LENGTH, DOMAIN_HEIGHT]], [NX, NY], CellType.quadrilateral
)
logger.info(
    "Mesh built in-memory: %d cells, %d vertices (no file I/O)",
    mesh.topology.index_map(mesh.topology.dim).size_local,
    mesh.topology.index_map(0).size_local,
)

# --- STEP 2: fem_config / opt_config, matching beam_2d.py's verified values ---
fem_config = {
    "mesh": mesh,
    "mesh_serial": mesh,  # serial run: mesh_serial == mesh, no MPI gather needed
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

opt_config = {
    "vol_frac": 0.5,
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 1.2,
    "opt_compliance": True,
    "use_oc": False,  # robust objective uses gradient-based MMA path, not OC
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

# --- STEP 3: Build all FEniTop objects ONCE ---
linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem_config, opt_config)

density_filter = DensityFilter(
    comm, rho_field, rho_phys_field,
    opt_config["filter_radius"], fem_config["petsc_options"],
)

sens_problem = Sensitivity(comm, opt_config, linear_problem, u_field, lambda_field, rho_phys_field)

logger.info("FEniTop objects constructed once: form_fem, DensityFilter, Sensitivity")

# --- STEP 4: Build the random-field Heaviside glue using a COARSE KL grid ---
node_coordinates = rho_phys_field.function_space.tabulate_dof_coordinates()[:, :2]

domain_size = node_coordinates.max(axis=0) - node_coordinates.min(axis=0)
length_scale = 0.2 * domain_size.min()
logger.info("Domain size: %s, length_scale set to: %.4g", domain_size, length_scale)

# Coarse auxiliary grid sized relative to length_scale (NOT the full FEA mesh)
# -- avoids the O(N^3) KL eigendecomposition cost at ~12,261 FEA nodes, which
# was the actual source of the long runtime.
coarse_nx = max(4, int(domain_size[0] / (length_scale / 2)))
coarse_ny = max(4, int(domain_size[1] / (length_scale / 2)))
coarse_x = np.linspace(node_coordinates[:, 0].min(), node_coordinates[:, 0].max(), coarse_nx)
coarse_y = np.linspace(node_coordinates[:, 1].min(), node_coordinates[:, 1].max(), coarse_ny)
coarse_xx, coarse_yy = np.meshgrid(coarse_x, coarse_y)
coarse_coordinates = np.column_stack([coarse_xx.ravel(), coarse_yy.ravel()])
logger.info(
    "Coarse KL grid: %d nodes (vs %d full FEA nodes)",
    coarse_coordinates.shape[0], node_coordinates.shape[0],
)
coarse_simplices = Delaunay(coarse_coordinates).simplices

kernel_params = KernelParams(sigma=1.0, length_scale=length_scale, spatial_dim=2)
transform_params = MarginalTransformParams(eta_min=0.45, eta_max=0.55, alpha=2.0, beta=2.0)

t0 = time.perf_counter()
kl_result = compute_kl_expansion(
    coarse_coordinates, coarse_simplices, kernel_params, variance_threshold=0.95
)
logger.info(
    "Coarse KL ready in %.2fs: N_kl=%d, variance_explained=%.4f",
    time.perf_counter() - t0, kl_result.n_kl, kl_result.variance_explained,
)

transform = ThresholdMarginalTransform(transform_params)

def sample_eta_on_fine_mesh(seed: int) -> np.ndarray:
    """Sample eta(x) on the coarse KL grid, then interpolate to the fine FEA mesh."""
    g_coarse = sample_gaussian_field(kl_result, n_samples=1, seed=seed)[0]
    eta_coarse = transform.transform(g_coarse)
    eta_fine = griddata(coarse_coordinates, eta_coarse, node_coordinates, method="linear")
    nan_mask = np.isnan(eta_fine)
    if nan_mask.any():
        eta_fine[nan_mask] = griddata(
            coarse_coordinates, eta_coarse, node_coordinates[nan_mask], method="nearest"
        )
    return np.clip(eta_fine, 1e-6, 1.0 - 1e-6)

heaviside_config = RandomHeavisideConfig(
    kernel_params=kernel_params,
    transform_params=transform_params,
    seed=42,
)

# Bypass RandomFieldHeaviside.__init__'s expensive full-mesh KL computation:
# construct the object without calling __init__, then wire in forward/backward
# (pure NumPy math, unchanged) and the fast coarse-to-fine resample() above.
rf_heaviside = RandomFieldHeaviside.__new__(RandomFieldHeaviside)
rf_heaviside.rho_phys = rho_phys_field
rf_heaviside.config = heaviside_config
rf_heaviside.drho = None
rf_heaviside._current_eta = None
rf_heaviside.kl_result = kl_result  # coarse KL, retained for logging/introspection
rf_heaviside.transform = transform
rf_heaviside.resample = lambda seed=None: (
    sample_eta_on_fine_mesh(seed if seed is not None else heaviside_config.seed)
)

logger.info(
    "RandomFieldHeaviside ready (coarse-grid KL): N_kl=%d, variance_explained=%.4f",
    rf_heaviside.kl_result.n_kl, rf_heaviside.kl_result.variance_explained,
)

# --- STEP 5: Load the converged nominal design, or fall back to uniform vol_frac ---
rho_converged_path = "output/rho_converged.npy"
try:
    rho_current = np.load(rho_converged_path)
    logger.info("Loaded converged design from %s", rho_converged_path)
except FileNotFoundError:
    logger.warning(
        "%s not found -- using a uniform vol_frac=%.2f initial guess instead. "
        "Run topopt.py (with the np.save addition) first for a real converged design.",
        rho_converged_path, opt_config["vol_frac"],
    )
    rho_current = np.full(rho_field.vector.array.shape, opt_config["vol_frac"])

if rho_current.shape != rho_field.vector.array.shape:
    raise ValueError(
        f"rho_converged.npy shape {rho_current.shape} does not match this "
        f"mesh's dof shape {rho_field.vector.array.shape}. Ensure NX, NY here "
        "match the resolution used to generate that file."
    )

# --- STEP 6: Define evaluate_fn as a closure reusing the objects built above ---
robust_config = RobustObjectiveConfig(lambda_tradeoff=0.5, n_mc_samples=20, beta=8.0, seed=0)

def evaluate_fn(rho_values: np.ndarray):
    return evaluate_robust_samples(
        rho_values, linear_problem, density_filter, rf_heaviside,
        sens_problem, rho_field, robust_config,
    )

# --- STEP 7: Run one full robust evaluation at the current design ---
logger.info("Running one robust evaluation at the current design...")
t0 = time.perf_counter()
result = evaluate_fn(rho_current)
logger.info("Robust evaluation (%d MC samples) took %.2fs", robust_config.n_mc_samples, time.perf_counter() - t0)

J_value = compute_robust_objective_value(result, robust_config)
dJ_drho = compute_robust_gradient(result, robust_config)
g_value = compute_mean_volume_constraint(result, opt_config["vol_frac"])
dg_drho = compute_mean_volume_gradient(result)

print(f"J (robust objective)     = {J_value:.6g}")
print(f"mu_C                      = {result.mu_C:.6g}")
print(f"sigma_C                   = {result.sigma_C:.6g}")
print(f"mean volume constraint g  = {g_value:.6g} (feasible if <= 0)")
print(f"dJ/drho norm              = {np.linalg.norm(dJ_drho):.6g}")
print(f"dg/drho norm              = {np.linalg.norm(dg_drho):.6g}")

# --- STEP 8: Finite-difference verification gate (Section 7 mandatory gate) ---
logger.info("Running finite-difference verification gate...")
fd_result = verify_robust_gradient_fd(
    rho_current, evaluate_fn, robust_config,
    n_check_elements=5, fd_step=1e-6, rtol=1e-2, rng_seed=0,
)

print(f"\nFD verification passed   = {fd_result['passed']}")
print(f"Max relative error        = {fd_result['max_relative_error']:.6g}")
print(f"Checked element indices   = {fd_result['checked_indices']}")
print(f"Analytic gradient (subset)= {fd_result['analytic_grad']}")
print(f"FD gradient (subset)      = {fd_result['fd_grad']}")

if not fd_result["passed"]:
    logger.warning(
        "FD check FAILED (max_rel_error=%.4g >= rtol=%.4g). Do NOT proceed "
        "to the MMA optimizer with this gradient until resolved -- try "
        "increasing n_mc_samples (currently %d) first, since MC sampling "
        "noise is the most likely cause of failure at low sample counts.",
        fd_result["max_relative_error"], 1e-2, robust_config.n_mc_samples,
    )
else:
    logger.info("FD check PASSED -- robust gradient is verified and safe to use in MMA.")