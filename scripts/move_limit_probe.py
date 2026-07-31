"""Does move-limit continuation let the SAA solve actually reach optimality?

    mpirun -n 32 python scripts/move_limit_probe.py 0.7 [config.yaml]

THE MEASURED PROBLEM
--------------------
With a FIXED move limit (box_source._MOVE = 0.02) the design change dx equals
the move limit on EVERY iteration of EVERY beta stage -- the MMA subproblem
solution is permanently on the trust-region boundary. The relative stationarity
residual then drops for ~6 iterations and plateaus, oscillating rather than
decaying. Measured on the study mesh at beta=128:

    it= 2  stat_rel=0.2303      it=18  stat_rel=0.1335
    it= 6  stat_rel=0.0832      it=22  stat_rel=0.1006
    it=10  stat_rel=0.0882      it=26  stat_rel=0.1193
    it=14  stat_rel=0.0800      it=30  stat_rel=0.1244   <- budget exhausted

against robust_opt_tol = 1e-3. The gap study's seven independent field designs
land at stat_rel 0.038-0.106, so this is systemic, not a bad seed: NO run in
this project has ever met its configured tolerance. More iterations cannot fix
it -- the iterate is not converging slowly, it is stepping the maximum distance
forever.

WHAT THIS PROBE MEASURES
------------------------
One uniform-eta arm solve (N=64, cheap, ~1 h at 32 ranks) with a move limit that
shrinks by `factor` per beta stage. The baseline to beat is the P0-A run already
on disk, which used the same config and reached:

    final stat_rel = 0.1244,  mu_C = 0.2718,  sigma_C = 0.1387,  M_nd = 0.234%

A pass is a materially lower final stat_rel WITHOUT a worse objective -- a
smaller move limit trivially reduces dx, so dx alone proves nothing. The
objective and the discreteness have to hold up too, otherwise the "convergence"
is just a design that stopped moving before it was any good.
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
from src.optimization.saa_robust_driver import run_saa_robust_topopt
from src.random_fields.kl_expansion import build_uniform_eta_kl
from src.sampling.sampler import generate_samples
from src.study_support import build_stage3_kl, setup_context

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "move_limit_probe"
N_SAA = 64

# The P0-A fixed-move-limit run, for a like-for-like comparison.
BASELINE = {
    "move_reduction": 1.0,
    "final_stat_rel": 0.1243845783248236,
    "mu_C": 0.2717997059134909,
    "sigma_C": 0.13865353340209827,
    "M_nd_percent": 0.2335855958557593,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)

    argv = sys.argv[1:]
    configs = [a for a in argv if a.endswith((".yaml", ".yml"))]
    factors = [a for a in argv if not a.endswith((".yaml", ".yml"))]
    factor = float(factors[0]) if factors else 0.7
    cfg = load_config(configs[0] if configs else "src/config/configStudy.yaml")

    if comm.rank == 0:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(OUTPUT_ROOT / "nominal_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_uniform = build_uniform_eta_kl(build_stage3_kl(cfg, tagged_mesh, comm))
    ctx = setup_context(fem, opt, rho_nominal, kl_uniform, load_cases, case_name)

    # The ONLY difference from the P0-A baseline run.
    opt = dict(opt)
    opt["move_reduction"] = factor

    lam = float(cfg.optimization.lambda_sweep[-1])
    xi = generate_samples(
        kl_uniform, N_SAA,
        strategy=cfg.optimization.saa_sampling_strategy,
        seed=cfg.optimization.saa_seed,
    ).xi

    if comm.rank == 0:
        logger.info(
            "=== move-limit probe: factor=%.3g, base move=%.4g, lambda=%.3g ===",
            factor, opt["move"], lam,
        )
    solved = run_saa_robust_topopt(ctx, opt, lam, xi)

    if comm.rank != 0:
        return

    final_stat = solved["optimality"]["stationarity_rel"]
    improvement = (BASELINE["final_stat_rel"] - final_stat) / BASELINE["final_stat_rel"]
    objective = solved["mu_C"] + lam * solved["sigma_C"]
    baseline_objective = BASELINE["mu_C"] + lam * BASELINE["sigma_C"]
    objective_change = (objective - baseline_objective) / baseline_objective

    payload = {
        "move_reduction": factor,
        "move_schedule": solved["move_schedule"],
        "beta_schedule": solved["beta_schedule"],
        "final_stationarity_rel": final_stat,
        "converged": solved["converged"],
        "mu_C": solved["mu_C"],
        "sigma_C": solved["sigma_C"],
        "objective": objective,
        "M_nd_percent": solved["M_nd_percent"],
        "baseline": BASELINE,
        "stationarity_improvement_relative": improvement,
        "objective_change_relative": objective_change,
        "per_stage": [
            {
                "beta": s["beta"], "move_limit": s["move_limit"],
                "stationarity_rel": s["optimality"]["stationarity_rel"],
                "design_change": s["optimality"]["design_change"],
                "mu_C": s["mu_C"], "sigma_C": s["sigma_C"],
                "M_nd_percent": s["M_nd_percent"],
            }
            for s in solved["stage_results"]
        ],
        "verdict_rule": (
            "PASS requires a materially lower final stationarity_rel AND an "
            "objective no worse than baseline. A smaller move limit trivially "
            "shrinks dx, so dx alone proves nothing -- a design that merely "
            "stopped moving early is not a converged design."
        ),
    }
    with open(OUTPUT_ROOT / f"probe_factor_{factor:g}.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    np.save(OUTPUT_ROOT / f"rho_factor_{factor:g}.npy", np.asarray(solved["rho_robust"]))

    logger.info("per-stage: beta / move / stat_rel / dx / M_nd")
    for s in payload["per_stage"]:
        logger.info(
            "  beta=%-6g move=%-8.4g stat_rel=%-9.4g dx=%-8.4g M_nd=%.3g%%",
            s["beta"], s["move_limit"], s["stationarity_rel"],
            s["design_change"], s["M_nd_percent"],
        )
    logger.info(
        "FINAL stat_rel %.4g vs baseline %.4g  (%+.1f%%);  objective %.5g vs "
        "%.5g (%+.1f%%);  M_nd %.3g%% vs %.3g%%",
        final_stat, BASELINE["final_stat_rel"], -100 * improvement,
        objective, baseline_objective, 100 * objective_change,
        solved["M_nd_percent"], BASELINE["M_nd_percent"],
    )
    if improvement > 0.5 and objective_change < 0.02:
        logger.info(
            "PASS: continuation cuts the stationarity residual by %.0f%% without "
            "costing objective. Worth enabling for production.", 100 * improvement,
        )
    elif improvement > 0.5:
        logger.warning(
            "MIXED: stationarity improved %.0f%% but the objective got %.1f%% "
            "worse -- the design may simply have stopped moving. Inspect the "
            "per-stage table before enabling.", 100 * improvement, 100 * objective_change,
        )
    else:
        logger.warning(
            "FAIL: stationarity %.4g is not materially better than the fixed-move "
            "baseline %.4g. The plateau is not caused by the move limit -- most "
            "likely it is the beta=128 projection stiffness (the responsive band "
            "is only ~19/beta wide, so small design changes flip which nodes "
            "carry gradient). Report the achieved residual honestly instead.",
            final_stat, BASELINE["final_stat_rel"],
        )


if __name__ == "__main__":
    main()
