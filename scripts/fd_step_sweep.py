"""Is the FD gate's step size too large at high beta? -- the decisive test.

    mpirun -n 32 python scripts/fd_step_sweep.py [config.yaml]

THE QUESTION
------------
gate_gradient_fd compares the SAA adjoint against a central difference with a
FIXED step (fd_step = 1e-3, in every config, at every beta). The beta sweep in
scripts/fd_gate_probe.py showed the disagreement growing monotonically with
beta -- and growing on LARGE-magnitude gradient entries, not just near-zero ones
(0.60% -> 2.11% -> 3.04% at beta = 32 -> 64 -> 128). A magnitude floor does not
explain it: the worst offender sits at 0.24x the median |dJ|, well clear of any
sensible floor.

The remaining explanation is the finite difference itself. Central-difference
truncation error is (h^2/6) * f''', and the tanh projection's third derivative
grows like beta^3, so a step tuned at beta = 8 is far too large at beta = 128.
If that is right, the ADJOINT is fine and the REFERENCE is wrong -- which is the
opposite of what the gate currently reports.

WHAT THIS MEASURES
------------------
For a handful of elements, evaluate the central difference at a range of step
sizes and compare each against the analytic gradient. Truncation error falls as
h^2 as the step shrinks, until KSP noise (amplified by 1/h) takes over and the
error rises again. The minimum of that V is the usable step.

  * If the error falls sharply as h shrinks -> truncation confirmed, and the fix
    is a beta-scaled step, not a looser tolerance.
  * If the error is flat in h -> the adjoint really does disagree, and this is a
    genuine gradient bug that must be found before production runs.
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
from src.optimization.robust_gradient import compute_robust_gradient
from src.optimization.robust_objective import (
    RobustObjectiveConfig, compute_robust_objective_value,
)
from src.optimization.saa_robust_driver import _evaluate_saa
from src.provenance import RunManifest, make_run_id
from src.sampling.sampler import generate_samples
from src.study_support import build_stage3_kl, setup_context
from src.validation.gates import _gather_local_to_global, _select_global_entries

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "fd_step_sweep"

STEPS = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5)
BETAS = (32.0, 128.0)
N_ELEMENTS = 6
N_SAMPLES = 4
KSP_RTOL = 1.0e-12
WARM_START = False


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
    rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "sweep_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)
    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    # Same design construction the gate uses: blended toward the eta midpoint so
    # the projection is actually active.
    transform_params = opt["transform_params"]
    eta_mid = 0.5 * (transform_params.eta_min + transform_params.eta_max)
    rho0 = eta_mid + 0.4 * (np.asarray(ctx.rho_warm_start_local, dtype=float) - eta_mid)

    xi = generate_samples(kl_result, N_SAMPLES, strategy="monte_carlo", seed=0).xi
    robust_config = RobustObjectiveConfig(lambda_tradeoff=1.0)

    # DISABLE WARM START for the whole sweep. fea_at_samples enables it
    # unconditionally, and while it is "math-exact" to solver tolerance, a
    # finite difference subtracts two nearly-equal J values -- so a solution
    # that depends on its starting guess at the rtol level leaves exactly that
    # jitter behind after the cancellation. That is a noise floor no step size
    # can escape, and it is the prime suspect for the sweep's inverted trend.
    for problem in _iter_problems(ctx):
        problem.solver.setTolerances(rtol=KSP_RTOL, atol=1e-50)
        problem.enable_warm_start(WARM_START)

    index_map = ctx.rho_field.function_space.dofmap.index_map
    col_start = index_map.local_range[0]
    n_global = index_map.size_global

    results = {}
    for beta in BETAS:
        base = _evaluate_saa(ctx, opt, rho0, xi, beta)
        analytic = compute_robust_gradient(base, robust_config)

        grad_abs = _gather_local_to_global(np.abs(analytic), col_start, n_global)
        rho_g = _gather_local_to_global(rho0, col_start, n_global)
        if comm.rank == 0:
            ok = np.where((rho_g > 0.05) & (rho_g < 0.95)
                          & (grad_abs > np.median(grad_abs)))[0]
            chosen = np.random.default_rng(0).choice(
                ok, size=min(N_ELEMENTS, ok.size), replace=False).astype(np.int64)
        else:
            chosen = None
        chosen = comm.bcast(chosen, root=0)
        analytic_sel = _select_global_entries(analytic, chosen, col_start)

        rows = []
        for step in STEPS:
            fd = np.empty(chosen.size)
            for _p in _iter_problems(ctx):
                _p.enable_warm_start(WARM_START)
            for k, gidx in enumerate(chosen):
                local = int(gidx) - col_start
                owned = 0 <= local < rho0.size

                pert = rho0.copy()
                if owned:
                    pert[local] += step
                Jp = compute_robust_objective_value(
                    _evaluate_saa(ctx, opt, pert, xi, beta), robust_config)

                pert = rho0.copy()
                if owned:
                    pert[local] -= step
                Jm = compute_robust_objective_value(
                    _evaluate_saa(ctx, opt, pert, xi, beta), robust_config)

                fd[k] = (Jp - Jm) / (2.0 * step)

            scale = np.maximum(np.abs(analytic_sel), np.abs(fd))
            rel = np.abs(analytic_sel - fd) / np.where(scale > 0, scale, 1.0)
            rows.append({"step": step, "max_rel_err": float(rel.max()),
                         "median_rel_err": float(np.median(rel))})
            if comm.rank == 0:
                logger.warning("beta=%-6g step=%.1e  max=%.4f%%  median=%.4f%%",
                               beta, step, 100 * rel.max(), 100 * np.median(rel))
        results[str(beta)] = rows

    if comm.rank == 0:
        payload = {"steps": list(STEPS), "betas": list(BETAS),
                   "n_elements": N_ELEMENTS, "ksp_rtol": KSP_RTOL,
                   "results": results}
        with open(run_dir / "fd_step_sweep.json", "w") as h:
            json.dump(payload, h, indent=2)
        logger.warning("=== verdict ===")
        for beta, rows in results.items():
            best = min(rows, key=lambda r: r["max_rel_err"])
            worst_at_default = rows[0]["max_rel_err"]
            logger.warning(
                "beta=%-6s default step 1e-3 -> %.4f%% ; best step %.1e -> %.4f%% "
                "(%.1fx better)", beta, 100 * worst_at_default, best["step"],
                100 * best["max_rel_err"],
                worst_at_default / max(best["max_rel_err"], 1e-30))
        logger.warning(
            "If the error drops sharply with smaller steps, the ADJOINT is fine "
            "and fd_step must scale with beta. If it is flat, the gradient is "
            "genuinely wrong and production must not run.")
    manifest.record("fd_step_sweep", results)
    manifest.write(run_dir / "manifest.json")


def _iter_problems(ctx):
    from src.validation.gates import _iter_linear_problems
    return _iter_linear_problems(ctx)


if __name__ == "__main__":
    main()
