"""viz/compare_designs_mc.py -- the head-to-head that justifies the project.

WHAT IS MISSING FROM THE PIPELINE TODAY
---------------------------------------
src/mainClean.py Stage 6 runs Monte Carlo validation on ONE design:

    final_design = pareto_results[-1]          # the robust design, lambda = 1
    mc_result = run_monte_carlo_validation(..., final_design["rho_robust"], ...)

So `output/stage6_validation/` tells you how the robust design behaves under
manufacturing variation -- but says nothing about how the ordinary
deterministic SIMP design would have behaved under the SAME variation. Without
that second number there is no comparison, and therefore no argument. Every
plot in the repo right now is a description of one design, not evidence that
robust TO was worth doing.

This script closes that gap. It runs the IDENTICAL Monte Carlo ensemble --
same KL basis, same seed, so realization i is literally the same manufacturing
defect field for every design -- against each design you point it at:

    output/stage2_fea/rho_converged.npy               deterministic SIMP
    output/stage5_optimization/rho_robust_lambda_0.0.npy   robust, mean only
    output/stage5_optimization/rho_robust_lambda_1.0.npy   robust, mean + std

Because the perturbation fields are shared, differences between the resulting
compliance distributions are attributable to the design alone. That is a
paired comparison, which is far stronger evidence than two independent runs
and lets you report a per-realization win rate, not just a shift in means.

OUTPUTS (under --out-dir, default output/comparison/)
-----------------------------------------------------
  <name>/compliance_samples.csv    per-realization compliance for that design
  <name>/reliability_map.vtu       per-node mean_density / std_density /
                                   prob_void across the ensemble
  <name>/ensemble/*.vtu            per-realization density (only with
                                   --write-ensemble; off by default, it is the
                                   dominant cost and you rarely need two copies)
  paired_compliance.csv            one row per realization, one column per
                                   design -- the paired dataset
  risk_delta.vtu                   nominal-minus-robust fields on one mesh:
                                     d_prob_void   where robust TO removed
                                                   the risk of a feature
                                                   disappearing (positive =
                                                   robust is safer here)
                                     d_std_density where robust TO stabilized
                                                   the boundary
                                   This single file is the most persuasive
                                   spatial artifact in the whole project: it
                                   shows *where* the robustness went.
  summary.json                     mean / std / p95 / worst-case / CV, the
                                   paired win rate, and the tail-risk
                                   reduction, per design

USAGE (dolfinx container, repo root)
------------------------------------
    mpirun -n 8 python viz/compare_designs_mc.py

    mpirun -n 8 python viz/compare_designs_mc.py \
        --design nominal=output/stage2_fea/rho_converged.npy \
        --design robust=output/stage5_optimization/rho_robust_lambda_1.0.npy \
        --n-samples 500

Note on sample count: cfg.mc_validation.n_samples is 100 right now. 100 is
enough to separate the means but thin for the 95th-percentile and worst-case
claims that make the strongest slide. If you want tail numbers a reviewer will
not push back on, run this with --n-samples 500 or more; it is a post-hoc
check, so it costs solves but no optimization.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from mpi4py import MPI

from src.config.loader import load_config
from src.random_fields.kernel import KernelParams
from src.random_fields.kl_expansion import compute_kl_expansion
from src.topology.heaviside_projection_glue import RandomHeavisideConfig
from src.validation.monte_carlo import MCConfig, run_monte_carlo_validation

from viz.enrich_ensemble_fea import build_stage1

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD

_DEFAULT_DESIGNS = [
    ("nominal", "output/stage2_fea/rho_converged.npy"),
    ("robust_lambda0", "output/stage5_optimization/rho_robust_lambda_0.0.npy"),
    ("robust_lambda1", "output/stage5_optimization/rho_robust_lambda_1.0.npy"),
]


def _stats(C: np.ndarray) -> dict:
    return {
        "mean": float(C.mean()),
        "std": float(C.std(ddof=1)),
        "cv": float(C.std(ddof=1) / C.mean()),
        "p05": float(np.percentile(C, 5)),
        "p50": float(np.percentile(C, 50)),
        "p95": float(np.percentile(C, 95)),
        "worst": float(C.max()),
        "range": float(C.max() - C.min()),
        "n": int(C.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="src/config/config.yaml")
    ap.add_argument("--design", action="append", default=None,
                    metavar="NAME=PATH",
                    help="repeatable; defaults to nominal + both robust designs")
    ap.add_argument("--out-dir", type=Path, default=Path("output/comparison"))
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--write-ensemble", action="store_true",
                    help="also dump every per-realization density VTU per design")
    ap.add_argument("--baseline", default="nominal",
                    help="design name used as the reference in risk_delta.vtu "
                         "and in the paired win-rate")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if comm.rank == 0 else logging.ERROR,
                        force=True)
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.mc_validation.seed
    n_samples = args.n_samples if args.n_samples is not None else cfg.mc_validation.n_samples

    if args.design:
        designs = [tuple(d.split("=", 1)) for d in args.design]
    else:
        designs = [(n, p) for n, p in _DEFAULT_DESIGNS if Path(p).exists()]
    if comm.rank == 0:
        missing = [p for _, p in designs if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"design file(s) not found: {missing}")
        logger.info("Comparing %d design(s) over %d shared realizations "
                    "(seed=%d): %s", len(designs), n_samples, seed,
                    [n for n, _ in designs])

    tagged_mesh, fem, opt, load_cases = build_stage1(cfg, comm)
    case_name = next(iter(load_cases))
    fem["traction_bcs"] = load_cases[case_name]

    kernel_params = KernelParams(sigma=cfg.random_field.sigma,
                                 length_scale=cfg.random_field.length_scale,
                                 spatial_dim=cfg.random_field.spatial_dim)
    if comm.rank == 0:
        from src.meshing.mesher import extract_simplices
        node_coordinates = tagged_mesh.mesh_serial.geometry.x
        simplices = extract_simplices(tagged_mesh)
    else:
        node_coordinates = simplices = None
    kl_result = compute_kl_expansion(
        node_coordinates, simplices, kernel_params,
        variance_threshold=cfg.random_field.variance_threshold, comm=comm)

    hv_cfg = RandomHeavisideConfig(
        kernel_params=opt["kernel_params"],
        transform_params=opt["transform_params"],
        variance_threshold=opt.get("kl_variance_threshold",
                                   cfg.random_field.variance_threshold),
        seed=seed,
    )

    results: dict[str, dict] = {}
    curves: dict[str, np.ndarray] = {}
    reliability: dict[str, tuple] = {}

    for name, path in designs:
        if comm.rank == 0:
            logger.info("=== MC on design %r (%s) ===", name, path)
        rho_global = np.load(path) if comm.rank == 0 else None
        rho_global = comm.bcast(rho_global, root=0)

        mc_cfg = MCConfig(
            n_samples=n_samples,
            beta=cfg.mc_validation.beta,
            seed=seed,                       # SHARED -> paired comparison
            output_dir=args.out_dir / name,
            write_ensemble=True,             # needed for the reliability map
            ensemble_dir=(args.out_dir / name / "ensemble" if args.write_ensemble
                          else args.out_dir / name / "_scratch_ensemble"),
        )
        mc = run_monte_carlo_validation(fem, opt, rho_global, kl_result,
                                        heaviside_config=hv_cfg, mc_config=mc_cfg)
        curves[name] = mc.compliance_samples.copy()
        if comm.rank == 0:
            mc.to_csv(args.out_dir / name / "compliance_samples.csv")
            results[name] = _stats(mc.compliance_samples)
            reliability[name] = (mc.reliability_mean, mc.reliability_std,
                                 mc.reliability_prob_void)
            if not args.write_ensemble:
                import shutil
                shutil.rmtree(mc_cfg.ensemble_dir, ignore_errors=True)

    if comm.rank != 0:
        return

    names = [n for n, _ in designs]
    paired = np.column_stack([curves[n] for n in names])
    np.savetxt(args.out_dir / "paired_compliance.csv", paired, delimiter=",",
               header=",".join(names), comments="", fmt="%.10e")

    base = args.baseline if args.baseline in names else names[0]
    for name in names:
        if name == base:
            continue
        d = curves[base] - curves[name]
        results[name]["vs_" + base] = {
            "win_rate": float((d > 0).mean()),
            "mean_reduction_pct": float(100 * (results[base]["mean"] - results[name]["mean"])
                                        / results[base]["mean"]),
            "std_reduction_pct": float(100 * (results[base]["std"] - results[name]["std"])
                                       / results[base]["std"]),
            "p95_reduction_pct": float(100 * (results[base]["p95"] - results[name]["p95"])
                                       / results[base]["p95"]),
            "worst_case_reduction_pct": float(100 * (results[base]["worst"] - results[name]["worst"])
                                              / results[base]["worst"]),
        }

    (args.out_dir / "summary.json").write_text(
        json.dumps({"seed": seed, "n_samples": n_samples, "baseline": base,
                    "designs": dict(designs), "stats": results}, indent=2))

    # --- risk_delta.vtu ----------------------------------------------------
    # Both reliability maps live on the same serial mesh (create_box on
    # COMM_SELF is deterministic, and the STEP path reuses one mesh_serial),
    # so a node-by-node difference is well-defined.
    import pyvista as pv
    import dolfinx.plot
    ms = fem["mesh_serial"]
    elements, cell_types, nodes = dolfinx.plot.vtk_mesh(ms, ms.topology.dim)
    grid = pv.UnstructuredGrid(elements, cell_types, nodes)
    for name in names:
        mean_d, std_d, pv_void = reliability[name]
        grid.point_data[f"mean_density_{name}"] = mean_d
        grid.point_data[f"std_density_{name}"] = std_d
        grid.point_data[f"prob_void_{name}"] = pv_void
    for name in names:
        if name == base:
            continue
        grid.point_data[f"d_prob_void_{name}"] = (
            reliability[base][2] - reliability[name][2])
        grid.point_data[f"d_std_density_{name}"] = (
            reliability[base][1] - reliability[name][1])
    grid.save(str(args.out_dir / "risk_delta.vtu"))

    logger.info("Comparison complete -> %s", args.out_dir)
    for name in names:
        s = results[name]
        logger.info("  %-16s mean=%.6g std=%.6g p95=%.6g worst=%.6g CV=%.4f",
                    name, s["mean"], s["std"], s["p95"], s["worst"], s["cv"])


if __name__ == "__main__":
    main()
