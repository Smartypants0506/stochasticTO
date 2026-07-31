"""How much of the reported robustness is the design overfitting its own samples?

    mpirun -n 64 python scripts/saa_gap_study.py [config.yaml]

THE PROBLEM THIS MEASURES
-------------------------
The SAA driver optimizes over ONE fixed set of N samples for hundreds of
iterations. That is exactly the regime where a design can learn its own sample
set: sigma_C looks small on those N draws and larger on fresh ones. The pipeline
previously compared the in-loop estimate against a single n=100 Monte Carlo run
and reported a bare relative error, which cannot separate genuine robustness
from that overfitting.

THE ESTIMATOR (Mak, Morton & Wood 1999; Kleywegt, Shapiro & Homem-de-Mello 2002)
--------------------------------------------------------------------------------
For a minimization, the SAA optimal value is optimistically biased:

    E[ v_hat_N ] <= v*

so averaging the optimal values of M INDEPENDENTLY sampled SAA problems gives a
statistical LOWER bound on the true optimum:

    L = (1/M) sum_m v_hat_N^(m)

Any feasible design gives an UPPER bound, so evaluating one candidate design on
a large independent sample set N' >> N gives

    U = J_N'( x_bar )

and the optimality gap is estimated by U - L, with a confidence interval from
the variance of the M replications and of the N' evaluation. A gap that is large
relative to the differences being claimed between designs means N is too small
and the reported improvements are partly artifacts of the sample set.

The by-product is just as useful: the spread of sigma_C across the M
replications is the run-to-run variability of the whole method, which is the
right yardstick for asking whether a difference between two lambda values is
real.
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
from src.optimization.saa_robust_driver import _evaluate_saa, run_saa_robust_topopt
from src.provenance import RunManifest, make_run_id
from src.study_support import build_stage3_kl, setup_context
from src.validation.statistics import summarize_samples

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "saa_gap"

N_REPLICATIONS = 10
# Independent evaluation set for the upper bound. Must be >> the SAA N, and must
# use a seed disjoint from every replication's.
N_EVALUATION = 5000
EVALUATION_SEED_OFFSET = 900_000


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
    with manifest.stage("nominal"):
        rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "nominal_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    lam = float(cfg.optimization.lambda_sweep[-1])
    n_saa = int(cfg.optimization.saa_n_samples)
    beta_max = float(cfg.optimization.saa_beta_max)

    # Evaluation set, drawn once, disjoint from every replication.
    xi_evaluation = np.random.default_rng(
        cfg.optimization.saa_seed + EVALUATION_SEED_OFFSET
    ).standard_normal(size=(N_EVALUATION, kl_result.n_kl))

    replications = []
    for m in range(N_REPLICATIONS):
        seed = cfg.optimization.saa_seed + 1000 * (m + 1)
        xi = np.random.default_rng(seed).standard_normal(size=(n_saa, kl_result.n_kl))
        if comm.rank == 0:
            logger.info(
                "=== replication %d/%d (seed=%d, N=%d, lambda=%.3g) ===",
                m + 1, N_REPLICATIONS, seed, n_saa, lam,
            )
        with manifest.stage(f"replication_{m}"):
            solved = run_saa_robust_topopt(ctx, opt, lam, xi)

        v_hat = solved["mu_C"] + lam * solved["sigma_C"]

        # Out-of-sample: the SAME design, on the independent evaluation set.
        ctx.warm_start_comm.bcast(ctx.rho_field, solved["rho_robust"])
        design_local = ctx.rho_field.x.petsc_vec.array.copy()
        evaluation = _evaluate_saa(
            ctx, opt, design_local, xi_evaluation, beta_max, accumulate_gradients=True,
        )
        j_out = evaluation.mu_C + lam * evaluation.sigma_C

        replications.append({
            "replication": m,
            "seed": seed,
            "in_sample_objective": v_hat,
            "in_sample_mu_C": solved["mu_C"],
            "in_sample_sigma_C": solved["sigma_C"],
            "out_of_sample_objective": j_out,
            "out_of_sample_mu_C": evaluation.mu_C,
            "out_of_sample_sigma_C": evaluation.sigma_C,
            "converged": solved["converged"],
            "rho": np.asarray(solved["rho_robust"]),
            "compliance_samples_out": evaluation.compliance_samples,
        })
        if comm.rank == 0:
            logger.info(
                "replication %d: in-sample J=%.6g, out-of-sample J=%.6g "
                "(optimism %+.3g%%); sigma %.6g -> %.6g",
                m, v_hat, j_out, 100 * (j_out - v_hat) / abs(j_out),
                solved["sigma_C"], evaluation.sigma_C,
            )

    payload: dict = {}
    if comm.rank == 0:
        in_sample = np.array([r["in_sample_objective"] for r in replications])
        out_sample = np.array([r["out_of_sample_objective"] for r in replications])

        # Lower bound: E[v_hat_N] <= v*, so the replication mean bounds the true
        # optimum from below.
        lower_bound = float(in_sample.mean())
        lower_bound_se = float(in_sample.std(ddof=1) / np.sqrt(in_sample.size))

        # Upper bound: the best candidate design evaluated out of sample. Any
        # feasible design's true objective is an upper bound on the optimum.
        best_index = int(np.argmin(out_sample))
        upper_bound = float(out_sample[best_index])
        best_summary = summarize_samples(
            replications[best_index]["compliance_samples_out"],
            seed=cfg.mc_validation.bootstrap_seed,
        )
        upper_bound_se = float(best_summary["mean"]["standard_error"])

        gap = upper_bound - lower_bound
        gap_se = float(np.sqrt(lower_bound_se ** 2 + upper_bound_se ** 2))

        sigma_in = np.array([r["in_sample_sigma_C"] for r in replications])
        sigma_out = np.array([r["out_of_sample_sigma_C"] for r in replications])

        payload = {
            "n_replications": N_REPLICATIONS,
            "saa_n_samples": n_saa,
            "n_evaluation": N_EVALUATION,
            "lambda": lam,
            "lower_bound": lower_bound,
            "lower_bound_standard_error": lower_bound_se,
            "upper_bound": upper_bound,
            "upper_bound_standard_error": upper_bound_se,
            "optimality_gap": gap,
            "optimality_gap_standard_error": gap_se,
            "optimality_gap_relative": gap / abs(upper_bound),
            "gap_ci_95": [gap - 1.96 * gap_se, gap + 1.96 * gap_se],
            "sigma_C_in_sample": {
                "mean": float(sigma_in.mean()), "std": float(sigma_in.std(ddof=1)),
            },
            "sigma_C_out_of_sample": {
                "mean": float(sigma_out.mean()), "std": float(sigma_out.std(ddof=1)),
            },
            "sigma_optimism_relative": float(
                (sigma_in.mean() - sigma_out.mean()) / sigma_out.mean()
            ),
            "run_to_run_sigma_variability_relative": float(
                sigma_out.std(ddof=1) / sigma_out.mean()
            ),
            "replications": [
                {k: v for k, v in r.items()
                 if k not in ("rho", "compliance_samples_out")}
                for r in replications
            ],
            "interpretation": (
                "optimality_gap_relative is how far N="
                f"{n_saa} SAA is from the true robust optimum, as a fraction of "
                "the objective. sigma_optimism_relative is (in-sample sigma_C "
                "minus out-of-sample sigma_C) over out-of-sample sigma_C -- "
                "NEGATIVE means the design is overfitting its sample set, i.e. "
                "it looks less variable on the samples it was fitted to than on "
                "fresh ones. run_to_run_sigma_variability_relative is the spread of "
                "sigma_C across independent replications: any claimed "
                "difference between designs that is smaller than this is not a "
                "property of the method, it is the seed."
            ),
        }
        with open(run_dir / "saa_gap.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

        logger.info(
            "SAA gap at N=%d: %.6g +/- %.3g (%.3g%% of the objective). "
            "sigma optimism %+.3g%%. Run-to-run sigma variability %.3g%%.",
            n_saa, gap, 1.96 * gap_se, 100 * payload["optimality_gap_relative"],
            100 * payload["sigma_optimism_relative"],
            100 * payload["run_to_run_sigma_variability_relative"],
        )
        logger.warning(
            "Any sigma_C difference between designs smaller than %.3g%% is "
            "within this method's own run-to-run variability and must not be "
            "reported as an improvement.",
            100 * payload["run_to_run_sigma_variability_relative"],
        )
        np.save(run_dir / "best_design.npy", replications[best_index]["rho"])

    manifest.record("saa_gap", payload)
    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
