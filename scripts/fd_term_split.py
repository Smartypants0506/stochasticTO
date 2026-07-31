"""Which TERM disagrees with finite differences -- mu_C, sigma_C, or E[V]?

    mpirun -n 8 python scripts/fd_term_split.py [config.yaml]

WHY THIS AND NOT THE EXISTING GATE
-----------------------------------
gate_gradient_fd checks the COMBINED objective J = mu_C + lambda*sigma_C. When
that fails you learn only that something in the chain is off. Three hypotheses
were tested against the combined check and all three were disproved (near-zero
denominators, FD truncation at high beta, warm-start jitter), which is what a
non-localising test buys you.

The gradient is separable and the pieces are already exposed:
    compute_dmu_drho     -> mean of the per-sample dC/drho
    compute_dsigma_drho  -> centred sum / ((N-1) * sigma_C)
    compute_mean_volume_gradient
So each can be finite-differenced against its OWN scalar, which localises the
disagreement to one term instead of pointing at the sum.

The prior is sharp. dE[V]/drho already matches central differences to 5-6
significant figures in every gate run, and it traverses the identical
filter -> projection -> backward chain. So that chain is sound, and the suspect
is dsigma_C/drho -- which is the only piece involving the centred sum and the
1/sigma_C factor.

WHAT EACH OUTCOME MEANS
-----------------------
  mu OK, sigma OK      -> the terms are fine; the combined-J failure is an
                          artefact of how J was assembled or scaled.
  mu OK, sigma BAD     -> the defect is in the sigma_C gradient specifically.
                          That is a real bug and production results built on it
                          are suspect.
  mu BAD, sigma BAD    -> the per-sample dC/drho is wrong, which would also
                          contradict the clean dE[V] result -- look for a
                          sample-indexing or accumulation error.
  everything OK here   -> finite differences are simply an unreliable reference
                          for this objective, and the gate needs rethinking
                          rather than the gradient needing fixing.
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
from src.optimization.robust_gradient import (
    compute_dmu_drho, compute_dsigma_drho, compute_mean_volume_gradient,
)
from src.optimization.saa_robust_driver import _evaluate_saa
from src.provenance import RunManifest, make_run_id
from src.sampling.sampler import generate_samples
from src.study_support import build_stage3_kl, setup_context
from src.validation.gates import (
    _gather_local_to_global, _iter_linear_problems, _select_global_entries,
)

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "fd_term_split"

BETAS = (32.0, 128.0)
STEP = 1.0e-3
N_ELEMENTS = 8
N_SAMPLES = 8
KSP_RTOL = 1.0e-12


def main() -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING if comm.rank == 0 else logging.ERROR)

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
    rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "split_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)
    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    transform_params = opt["transform_params"]
    eta_mid = 0.5 * (transform_params.eta_min + transform_params.eta_max)
    rho0 = eta_mid + 0.4 * (np.asarray(ctx.rho_warm_start_local, dtype=float) - eta_mid)

    xi = generate_samples(kl_result, N_SAMPLES, strategy="monte_carlo", seed=0).xi

    for problem in _iter_linear_problems(ctx):
        problem.solver.setTolerances(rtol=KSP_RTOL, atol=1e-50)

    index_map = ctx.rho_field.function_space.dofmap.index_map
    col_start = index_map.local_range[0]
    n_global = index_map.size_global

    out = {}
    for beta in BETAS:
        base = _evaluate_saa(ctx, opt, rho0, xi, beta)
        analytic = {
            "mu_C": compute_dmu_drho(base),
            "sigma_C": compute_dsigma_drho(base),
            "mean_volume": compute_mean_volume_gradient(base),
        }

        grad_abs = _gather_local_to_global(np.abs(analytic["mu_C"]), col_start, n_global)
        rho_g = _gather_local_to_global(rho0, col_start, n_global)
        if comm.rank == 0:
            ok = np.where((rho_g > 0.05) & (rho_g < 0.95)
                          & (grad_abs > np.median(grad_abs)))[0]
            chosen = np.random.default_rng(0).choice(
                ok, size=min(N_ELEMENTS, ok.size), replace=False).astype(np.int64)
        else:
            chosen = None
        chosen = comm.bcast(chosen, root=0)

        analytic_sel = {k: _select_global_entries(v, chosen, col_start)
                        for k, v in analytic.items()}
        fd = {k: np.empty(chosen.size) for k in analytic}

        for k, gidx in enumerate(chosen):
            local = int(gidx) - col_start
            owned = 0 <= local < rho0.size

            pert = rho0.copy()
            if owned:
                pert[local] += STEP
            plus = _evaluate_saa(ctx, opt, pert, xi, beta)

            pert = rho0.copy()
            if owned:
                pert[local] -= STEP
            minus = _evaluate_saa(ctx, opt, pert, xi, beta)

            # Each term finite-differenced against ITS OWN scalar.
            fd["mu_C"][k] = (plus.mu_C - minus.mu_C) / (2 * STEP)
            fd["sigma_C"][k] = (plus.sigma_C - minus.sigma_C) / (2 * STEP)
            fd["mean_volume"][k] = (
                plus.mean_volume - minus.mean_volume) / (2 * STEP)

        rows = {}
        for term in analytic:
            a, f = analytic_sel[term], fd[term]
            scale = np.maximum(np.abs(a), np.abs(f))
            rel = np.abs(a - f) / np.where(scale > 0, scale, 1.0)
            rows[term] = {
                "max_rel_err": float(rel.max()),
                "median_rel_err": float(np.median(rel)),
                "analytic": [float(v) for v in a],
                "fd": [float(v) for v in f],
            }
            if comm.rank == 0:
                logger.warning(
                    "beta=%-6g %-12s max=%9.4f%%  median=%9.4f%%  %s",
                    beta, term, 100 * rel.max(), 100 * np.median(rel),
                    "OK" if rel.max() < 0.01 else "DISAGREES",
                )
        out[str(beta)] = rows

    if comm.rank == 0:
        with open(run_dir / "fd_term_split.json", "w") as h:
            json.dump({"step": STEP, "n_elements": N_ELEMENTS,
                       "n_samples": N_SAMPLES, "betas": list(BETAS),
                       "results": out}, h, indent=2)
        logger.warning("=== verdict ===")
        for beta, rows in out.items():
            bad = [t for t, r in rows.items() if r["max_rel_err"] >= 0.01]
            logger.warning("beta=%-6s disagreeing terms: %s", beta,
                           ", ".join(bad) if bad else "NONE (all within 1%)")
        logger.warning(
            "dE[V] is the control: it has matched to 5-6 significant figures in "
            "every gate run, so if it disagrees HERE the test setup is at fault, "
            "not the gradient.")

    manifest.record("fd_term_split", out)
    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
