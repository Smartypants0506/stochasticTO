"""Is the erode/dilate driver actually solving? -- isolate it from the SAA arm.

    mpirun -n 32 python scripts/erode_dilate_probe.py [config.yaml]

WHY THIS EXISTS
---------------
scripts/baseline_comparison.py runs a full SAA solve (hours at N=512) BEFORE it
touches erode/dilate, so it is a terrible debugging loop for the baseline arm.
This runs the baseline alone: 3 FEA solves per iteration, minutes not hours.

WHAT TO LOOK FOR
----------------
The epigraph formulation is healthy when:

  * worst-case compliance DECREASES across iterations. If it is pinned at a
    constant, the worst branch is saturated -- typically the eroded realization
    has stopped carrying load, which makes its gradient ~0 and leaves the
    optimizer no signal. That is a property of the mesh/eta band, not of the
    solver.
  * V_dilated settles near vol_frac rather than collapsing toward zero.
  * dx <= move_limit * t_range. A dx far above the design variables' move limit
    means the epigraph variable is running away (fixed 2026-07-30 by
    normalizing t; see _run_stage's comment block).

configSmoke.yaml CANNOT validate this driver: at R/h = 0.6 the filter is
narrower than one element, so eta = 0.75 erodes the structure to nothing and the
worst branch is degenerate by construction. Use configStudy.yaml.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.config.loader import load_config
from src.fenitop.topopt import topopt
from src.meshing.box_source import build_box_fenitop_dicts
from src.optimization.erode_dilate_driver import run_erode_dilate_topopt
from src.provenance import RunManifest, make_run_id
from src.study_support import build_stage3_kl, setup_context

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "erode_dilate_probe"


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/config/configStudy.yaml"
    cfg = load_config(config_path)

    run_id = make_run_id(comm)
    manifest = RunManifest(run_id, comm)
    run_dir = OUTPUT_ROOT / run_id
    if comm.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    manifest.record_config(cfg, effective_fem=fem, effective_opt=opt)
    rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "probe_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    with manifest.stage("erode_dilate"):
        result = run_erode_dilate_topopt(
            ctx, opt,
            eta_lo=cfg.random_field.eta_min,
            eta_hi=cfg.random_field.eta_max,
        )

    payload = {
        "config": config_path,
        "eta_band": [cfg.random_field.eta_min, cfg.random_field.eta_max],
        "beta_schedule": result["beta_schedule"],
        "worst_compliance": result["worst_compliance"],
        "compliances": result["compliances"],
        "epigraph_t": result["epigraph_t"],
        "epigraph_t_normalized": result.get("epigraph_t_normalized"),
        "epigraph_c_ref": result.get("epigraph_c_ref"),
        "volume_dilated": result["volume_dilated"],
        "volume_violation": result["volume_violation"],
        "vol_frac_target": opt["vol_frac"],
        "M_nd_percent": result["M_nd_percent"],
        "converged": result["converged"],
        "optimality": result.get("optimality"),
        "n_fea_solves_total": result["n_fea_solves_total"],
        "stage_results": result.get("stage_results"),
    }
    if comm.rank == 0:
        with open(run_dir / "erode_dilate_probe.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        np.save(run_dir / "rho_erode_dilate.npy", np.asarray(result["rho_robust"]))

        logger.info("=== erode/dilate probe verdict ===")
        logger.info("  worst compliance  : %.6g", result["worst_compliance"])
        logger.info("  branch compliances: %s",
                    ", ".join(f"{c:.6g}" for c in result["compliances"]))
        logger.info("  V_dilated / target: %.6g / %.6g",
                    result["volume_dilated"], opt["vol_frac"])
        logger.info("  M_nd              : %.3g%%", result["M_nd_percent"])
        logger.info("  converged         : %s", result["converged"])

        branches = np.asarray(result["compliances"], dtype=float)
        # Eroded is thresholds[0] = eta_hi (the THINNEST realization) in the
        # driver's ordering; a saturated worst branch shows up as the eroded and
        # mid values coinciding.
        if branches.size >= 2 and np.isclose(branches[0], branches[1], rtol=1e-3):
            logger.warning(
                "Eroded and intermediate compliance coincide (%.6g vs %.6g). "
                "The worst branch is SATURATED -- that realization carries no "
                "load, its gradient is ~0, and the optimizer has no signal. "
                "This is a mesh/eta-band property, not a solver bug.",
                branches[0], branches[1],
            )
        if result["volume_dilated"] < 0.5 * opt["vol_frac"]:
            logger.warning(
                "V_dilated %.4g collapsed to well under the %.4g budget -- the "
                "volume constraint is not binding, consistent with a saturated "
                "worst branch driving the design to nothing.",
                result["volume_dilated"], opt["vol_frac"],
            )

    manifest.record("erode_dilate_probe", payload)
    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
