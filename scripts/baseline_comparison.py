"""SAA random-field robust TO vs. the erode/dilate baseline, head to head.

    mpirun -n 64 python scripts/baseline_comparison.py [config.yaml]

Runs both methods on the SAME problem, the SAME eta band and the SAME beta
schedule, then evaluates both designs on ONE COMMON eta ensemble so the
comparison is paired. Reports robustness AND cost: the SAA path spends N FEA
solves per iteration, the baseline spends 3, so the burden is on the SAA result
to show that ~170x buys something. That is the comparison a reviewer makes
first, and nothing in this project made it.
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
from src.optimization.saa_robust_driver import _evaluate_saa, run_saa_robust_topopt
from src.provenance import RunManifest, make_run_id
from src.sampling.sampler import generate_samples
from src.study_support import build_stage3_kl, setup_context
from src.validation.statistics import compare_designs, summarize_samples

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "baseline_comparison"


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
    manifest.record_config(cfg, effective_fem=ctx.fem, effective_opt=opt)

    lam = float(cfg.optimization.lambda_sweep[-1])
    beta_max = float(cfg.optimization.saa_beta_max)

    # --- SAA ---------------------------------------------------------------
    xi_saa = generate_samples(
        kl_result, cfg.optimization.saa_n_samples,
        strategy=cfg.optimization.saa_sampling_strategy,
        seed=cfg.optimization.saa_seed,
    ).xi
    with manifest.stage("saa"):
        saa = run_saa_robust_topopt(ctx, opt, lam, xi_saa)
    saa_solves = int(saa["n_fea_batches_total"]) * int(cfg.optimization.saa_n_samples)

    # --- erode/dilate baseline, SAME band and schedule ---------------------
    with manifest.stage("erode_dilate"):
        baseline = run_erode_dilate_topopt(
            ctx, opt,
            eta_lo=cfg.random_field.eta_min,
            eta_hi=cfg.random_field.eta_max,
        )
    baseline_solves = int(baseline["n_fea_solves_total"])

    # --- paired evaluation on ONE common ensemble --------------------------
    xi_common = np.random.default_rng(cfg.mc_validation.seed).standard_normal(
        size=(cfg.mc_validation.n_samples, kl_result.n_kl)
    )
    evaluations = {}
    for name, rho_global in (
        ("nominal", rho_nominal),
        ("saa", saa["rho_robust"]),
        ("erode_dilate", baseline["rho_robust"]),
    ):
        ctx.warm_start_comm.bcast(ctx.rho_field, np.asarray(rho_global))
        evaluations[name] = _evaluate_saa(
            ctx, opt, ctx.rho_field.x.petsc_vec.array.copy(), xi_common,
            beta_max, accumulate_gradients=True,
        )

    payload: dict = {}
    if comm.rank == 0:
        summaries = {
            name: summarize_samples(
                result.compliance_samples, seed=cfg.mc_validation.bootstrap_seed
            )
            for name, result in evaluations.items()
        }
        comparisons = {
            name: compare_designs(
                evaluations["nominal"].compliance_samples,
                evaluations[name].compliance_samples,
                name_a="nominal", name_b=name, paired=True,
                n_bootstrap=cfg.mc_validation.n_bootstrap,
                confidence=cfg.mc_validation.confidence,
                seed=cfg.mc_validation.bootstrap_seed,
            ).as_dict()
            for name in ("saa", "erode_dilate")
        }
        # SAA vs baseline directly -- the comparison the paper turns on.
        head_to_head = compare_designs(
            evaluations["erode_dilate"].compliance_samples,
            evaluations["saa"].compliance_samples,
            name_a="erode_dilate", name_b="saa", paired=True,
            n_bootstrap=cfg.mc_validation.n_bootstrap,
            confidence=cfg.mc_validation.confidence,
            seed=cfg.mc_validation.bootstrap_seed,
        ).as_dict()

        payload = {
            "lambda": lam,
            "eta_band": [cfg.random_field.eta_min, cfg.random_field.eta_max],
            "beta_schedule": saa["beta_schedule"],
            "n_evaluation_samples": int(cfg.mc_validation.n_samples),
            "common_random_numbers": True,
            "cost": {
                "saa_fea_solves": saa_solves,
                "erode_dilate_fea_solves": baseline_solves,
                "cost_ratio": saa_solves / max(baseline_solves, 1),
                "saa_solves_per_iteration": int(cfg.optimization.saa_n_samples),
                "erode_dilate_solves_per_iteration": 3,
            },
            "discreteness": {
                "saa_M_nd_percent": saa["M_nd_percent"],
                "erode_dilate_M_nd_percent": baseline["M_nd_percent"],
            },
            "convergence": {
                "saa_converged": saa["converged"],
                "erode_dilate_converged": baseline["converged"],
            },
            "volume": {
                "saa_mean_volume": saa["mean_volume"],
                "saa_volume_violation": saa["volume_violation"],
                "erode_dilate_volume_dilated": baseline["volume_dilated"],
                "erode_dilate_volume_violation": baseline["volume_violation"],
                "note": (
                    "The two methods constrain DIFFERENT things: SAA bounds "
                    "E[V], erode/dilate bounds the volume of the DILATED "
                    "realization. That is a genuine difference between the "
                    "formulations, not an inconsistency to normalize away, and "
                    "it must be stated when the robustness numbers are compared."
                ),
            },
            "per_design": summaries,
            "paired_vs_nominal": comparisons,
            "saa_vs_erode_dilate": head_to_head,
        }
        with open(run_dir / "baseline_comparison.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

        logger.info("COST: SAA %d FEA solves vs erode/dilate %d (%.0fx)",
                    saa_solves, baseline_solves, payload["cost"]["cost_ratio"])
        logger.info("HEAD TO HEAD: %s", head_to_head["verdict"])
        if not head_to_head["std_difference_resolvable"]:
            logger.warning(
                "The SAA design is NOT distinguishable from the erode/dilate "
                "baseline in sigma_C at n=%d, despite costing %.0fx more FEA. "
                "That is the central result of this comparison and must be "
                "reported as such -- either the extra cost buys something "
                "measurable, or the honest conclusion is that it does not.",
                cfg.mc_validation.n_samples, payload["cost"]["cost_ratio"],
            )
        np.save(run_dir / "rho_saa.npy", np.asarray(saa["rho_robust"]))
        np.save(run_dir / "rho_erode_dilate.npy", np.asarray(baseline["rho_robust"]))

    manifest.record("baseline_comparison", payload)
    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
