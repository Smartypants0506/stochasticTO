"""
diagnose_pce_beta_smoothness.py

STANDALONE DIAGNOSTIC -- not part of the verified optimization path, not
wired into main.py's pipeline, and never feeds a PCEGradientModel into the
robust loop. Deliberately bypasses pce_builder.build_pce_surrogate's Q^2
gate (same spirit as src/surrogate/kl_sensitivity_diagnostic.py) so we can
see the FULL Q^2-vs-degree curve at several Heaviside sharpness (beta)
values, instead of only the pass/fail result at whatever beta the robust
loop happens to be using.

WHY THIS SCRIPT EXISTS
-----------------------
Stage 5 retraining is failing its Q^2 >= 0.99 gate at beta=64, with Q^2
rising through degree 3 (~0.85) then DEGRADING at higher degree -- the
classic signature of a smooth polynomial basis trying to track a
kink/near-discontinuity, not a "need more training samples" signature.
RandomFieldHeaviside.forward() uses tanh(beta * (rho_tilde - eta)), whose
transition band narrows as ~1/beta, so as beta increases the projection
gets closer to a true step function and any fixed-degree PCE should get
harder to fit -- not easier.

This script tests that hypothesis directly and cheaply: it reuses the
EXACT same rho_current, KL expansion, and (critically) the EXACT same
xi_train/xi_test samples across every beta in BETA_SWEEP (same seed ->
identical sample locations in KL-coefficient space), varying ONLY beta.
That isolates beta as the single variable under test. If Q^2 climbs
cleanly toward 0.99 as beta drops, the ceiling is confirmed as
projection-sharpness, not sample budget, solver noise, or KL truncation.

WHAT IT DOES NOT DO
-------------------
- Does not modify rho_current (no MMA/TAO stepping).
- Does not touch main.py, dolfiny_mma_driver.py's actual retrain path, or
  any config.yaml default.
- Does not accept/return a PCEGradientModel or otherwise touch
  robust_objective.py / robust_gradient.py.
- Does not bypass caching silently: each (beta, tag) FEA batch is cached
  to output/cache/fea_at_samples/ via the driver's own
  _cached_fea_at_samples(), so re-running this script to add more betas
  or degrees does not re-solve FEA for betas already computed.

USAGE
-----
    mpirun -n <ranks> python diagnose_pce_beta_smoothness.py \\
        --config src/config/config.yaml \\
        --betas 8 16 32 64 \\
        --max-degree 6

Results are logged to stdout (rank 0 only) and written to
output/diagnostics/pce_beta_smoothness.csv as they complete, so a partial
run (e.g. killed after 2 of 4 betas) still leaves usable data.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

from src.config.loader import load_config
from src.random_fields.kernel import KernelParams, build_squared_exponential
from src.random_fields.kl_expansion import compute_kl_expansion
from src.random_fields.threshold_transform import MarginalTransformParams
from src.meshing.importer import import_and_heal, finalize
from src.meshing.mesher import (
    extract_simplices, MeshingConfig, tag_physical_groups, generate_mesh, import_to_dolfinx,
)
from src.meshing.mapper import build_boundary_conditions
from src.meshing.box_source import build_box_fenitop_dicts
from src.fea.fenitop_adapter import build_fenitop_dicts

from src.sampling.sampler import generate_train_test_samples
from src.optimization.dolfiny_mma_driver import (
    setup_robust_problem,
    _cached_fea_at_samples,
)
from src.surrogate.pce_builder import _fit_chaos_at_degree, _compute_q2

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)
logger = logging.getLogger(__name__)

# Stage 2 cache file main.py currently reads its warm-start design from --
# kept as a literal here (mirroring main.py) rather than re-running SIMP.
STAGE2_CACHE_FILE = Path("output/rho_converged.npy")

OUTPUT_CSV = Path("output/diagnostics/pce_beta_smoothness.csv")


def _build_problem_context(cfg):
    """Reconstruct fem/opt/tagged_mesh/kl_result/rho_warmstart exactly as
    main.py does through the end of Stage 3, so this script fits at
    IDENTICAL rho/KL conditions to the real failing run. No new SIMP or
    Stage-2 solve happens here -- it requires the Stage 2 cache file to
    already exist (i.e. run main.py at least once first).
    """
    if cfg.mesh_source == "step":
        if comm.rank == 0:
            entities = import_and_heal(cfg.step_file)
            mesh_cfg = MeshingConfig(
                mesh_size_max=cfg.mesh_size_max,
                color_targets=cfg.color_targets,
                solid_volume_color=cfg.solid_volume_color,
            )
            tag_physical_groups(entities, mesh_cfg)
            generate_mesh(mesh_cfg, comm)
        tagged_mesh = import_to_dolfinx(comm)

        load_cases_input = {
            case_name: [(lc.group_name, lc.vector) for lc in entries]
            for case_name, entries in cfg.load_cases.items()
        }
        bc = build_boundary_conditions(
            tagged_mesh, load_cases_input,
            snap_tol=cfg.snap_tol,
            protected_face_groups=["fixed", "load_1", "load_2"],
            protected_buffer_radius=4e-3,
            keep_alive_groups=None,
            keep_alive_radius=None,
            keep_alive_cluster_eps=None,
            comm=comm,
        )
        fem, opt, load_cases = build_fenitop_dicts(tagged_mesh, bc, cfg)
    elif cfg.mesh_source == "box":
        tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    else:
        raise ValueError(f"Unknown mesh_source={cfg.mesh_source!r}")

    if comm.rank == 0:
        if not STAGE2_CACHE_FILE.exists():
            raise FileNotFoundError(
                f"{STAGE2_CACHE_FILE} not found -- run main.py at least once "
                "first so a converged nominal design exists to diagnose against."
            )
        rho_warmstart_global = np.load(STAGE2_CACHE_FILE)
    else:
        rho_warmstart_global = None
    rho_warmstart_global = comm.bcast(rho_warmstart_global, root=0)

    kernel_params = KernelParams(
        sigma=cfg.random_field.sigma,
        length_scale=cfg.random_field.length_scale,
        spatial_dim=cfg.random_field.spatial_dim,
    )
    build_squared_exponential(kernel_params)

    if comm.rank == 0:
        node_coordinates = tagged_mesh.mesh_serial.geometry.x
        simplices = extract_simplices(tagged_mesh)
    else:
        node_coordinates = None
        simplices = None
    kl_result = compute_kl_expansion(node_coordinates, simplices, kernel_params, comm=comm)

    opt.setdefault("kernel_params", kernel_params)
    opt.setdefault(
        "transform_params",
        MarginalTransformParams(
            eta_min=cfg.random_field.eta_min, eta_max=cfg.random_field.eta_max,
            alpha=cfg.random_field.alpha, beta=cfg.random_field.beta,
        ),
    )

    robust_case_name = next(iter(load_cases))
    ctx = setup_robust_problem(
        fem, opt, rho_warmstart_global, kl_result,
        load_cases=load_cases, case_name=robust_case_name,
    )
    return ctx, opt, kl_result, tagged_mesh


def _q2_curve_at_beta(ctx, opt, kl_result, beta: float, max_degree: int,
                       xi_train: np.ndarray, xi_test: np.ndarray) -> list[dict]:
    """Run FEA at `beta` for the FIXED xi_train/xi_test, then fit PCE at
    every degree 1..max_degree (bypassing the Q^2 gate -- see module
    docstring), returning the full curve rather than stopping early.
    """
    training_data = _cached_fea_at_samples(
        ctx.fem, opt, ctx.rho_warm_start_local, ctx.density_filter, ctx.rf_heaviside,
        ctx.sens_problem, xi_train, beta, ctx.linear_problem, ctx.rho_field,
        tag=f"beta_smoothness_train_b{beta:g}",
    )
    test_data = _cached_fea_at_samples(
        ctx.fem, opt, ctx.rho_warm_start_local, ctx.density_filter, ctx.rf_heaviside,
        ctx.sens_problem, xi_test, beta, ctx.linear_problem, ctx.rho_field,
        tag=f"beta_smoothness_test_b{beta:g}",
    )

    n_kl = xi_train.shape[1]
    c_train = training_data.compliance_samples
    c_test = test_data.compliance_samples

    rows = []
    for degree in range(1, max_degree + 1):
        chaos_result = _fit_chaos_at_degree(
            xi_train, c_train, n_kl, degree, hyperbolic_q=.5,
        )
        q2, rmse = _compute_q2(chaos_result, xi_test, c_test)
        if comm.rank == 0:
            logger.info(
                "beta=%.4g degree=%d: Q^2=%.5f, RMSE=%.5g", beta, degree, q2, rmse,
            )
        rows.append({"beta": beta, "degree": degree, "q2": q2, "rmse": rmse})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--betas", type=float, nargs="+", default=[8.0, 16.0, 32.0, 64.0])
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=None,
                         help="Override surrogate.n_train from config (defaults to config value).")
    parser.add_argument("--n-test", type=int, default=None,
                         help="Override surrogate.n_test from config (defaults to config value).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ctx, opt, kl_result, tagged_mesh = _build_problem_context(cfg)

    n_train = args.n_train or opt["pce_n_train"]
    n_test = args.n_test or opt["pce_n_test"]

    # Same seed on every beta -> IDENTICAL xi_train/xi_test across the sweep,
    # so beta is the only thing that changes between rows. This is the
    # whole point of the diagnostic; do not vary the seed per beta.
    train_set, test_set = generate_train_test_samples(
        kl_result, n_train=n_train, n_test=n_test, seed=opt.get("pce_seed", 0),
    )
    xi_train, xi_test = train_set.xi, test_set.xi

    if comm.rank == 0:
        logger.info(
            "Beta-smoothness diagnostic: n_kl=%d, n_train=%d, n_test=%d, "
            "betas=%s, max_degree=%d",
            kl_result.n_kl, n_train, n_test, args.betas, args.max_degree,
        )
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not OUTPUT_CSV.exists()
        csv_file = open(OUTPUT_CSV, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=["beta", "degree", "q2", "rmse"])
        if write_header:
            writer.writeheader()

    for beta in args.betas:
        rows = _q2_curve_at_beta(ctx, opt, kl_result, beta, args.max_degree, xi_train, xi_test)
        if comm.rank == 0:
            best = max(rows, key=lambda r: r["q2"])
            logger.info(
                "beta=%.4g SUMMARY: best Q^2=%.5f at degree=%d",
                beta, best["q2"], best["degree"],
            )
            for row in rows:
                writer.writerow(row)
            csv_file.flush()

    if comm.rank == 0:
        csv_file.close()
        logger.info("Diagnostic complete. Results written to %s", OUTPUT_CSV)
        logger.info(
            "Read the CSV as a beta x degree grid: if best-Q^2-per-beta climbs "
            "toward 0.99 as beta decreases, the Q^2 shortfall is confirmed as "
            "Heaviside projection sharpness, not n_train/hyperbolic_q/degree."
        )

    finalize()


if __name__ == "__main__":
    main()