"""Is the gradient_fd gate failure a beta=128 effect or a bad seed?

    mpirun -n 32 python scripts/fd_gate_probe.py [config.yaml]

THE QUESTION
------------
The study-mesh rehearsal failed `gradient_fd` at beta=128 with max relative
error 0.47% against a 0.1% tolerance. Inspection of the 32-element detail showed
28 of 32 elements passing comfortably (median error 0.02%) and the single worst
offender sitting on the smallest-magnitude gradient entry in the batch -- the
classic signature of a relative-error ratio blowing up near zero rather than of
a wrong gradient. The volume gradient matched to 5-6 significant figures at
every element, which exonerates the FD machinery itself.

Two competing explanations:
  (a) one unlucky element-selection seed, or
  (b) a real, systematic degradation that grows with beta -- expected, because
      the Heaviside derivative beta*sech^2(beta*(rho_tilde - eta)) is a spike
      only ~19/beta wide, so at beta=128 the gradient is carried by a thin
      shifting interface layer and a finite step of 1e-3 straddles it.

This runs the gate ONLY -- no optimization -- across several seeds at each of
beta = 32, 64, 128. If the error grows with beta and is stable across seeds,
(b) is confirmed, and the right response is a magnitude floor in the gate's
error metric plus an honest statement in the paper's limitations. If it is
seed noise, (a), the gate is simply flaky and needs more elements.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.config.loader import load_config
from src.fenitop.topopt import topopt
from src.meshing.box_source import build_box_fenitop_dicts
from src.provenance import RunManifest, make_run_id
from src.study_support import build_stage3_kl, setup_context
from src.validation.gates import GateConfig, gate_gradient_fd

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "fd_gate_probe"

BETAS = (32.0, 64.0, 128.0)
SEEDS = (0, 1, 2)


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
    rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "fd_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    base = GateConfig(
        fd_enabled=True,
        fd_n_samples=cfg.validation.fd_n_samples,
        fd_n_elements=cfg.validation.fd_n_elements,
        fd_step=cfg.validation.fd_step,
        fd_rtol=cfg.validation.fd_rtol,
        fd_ksp_rtol=cfg.validation.fd_ksp_rtol,
    )

    rows = []
    for beta in BETAS:
        for seed in SEEDS:
            gcfg = replace(base, fd_seed=seed)
            result = gate_gradient_fd(ctx, opt, beta, gcfg)
            det = result.detail
            a = np.asarray(det["analytic_dJ"], dtype=float)
            f = np.asarray(det["fd_dJ"], dtype=float)
            scale = np.maximum(np.abs(a), np.abs(f))
            rel = np.abs(a - f) / np.where(scale > 0, scale, 1.0)

            # The hypothesis is that error concentrates on SMALL-magnitude
            # entries, so report the error restricted to the top half by
            # magnitude alongside the raw max.
            big = np.abs(a) >= np.median(np.abs(a))
            row = {
                "beta": beta,
                "seed": seed,
                "passed": bool(result.passed),
                "max_rel_err": float(rel.max()),
                "median_rel_err": float(np.median(rel)),
                "max_rel_err_large_magnitude_half": float(rel[big].max()),
                "max_rel_err_dEV": float(det["max_relative_error_dEV_drho"]),
                "worst_entry_magnitude": float(np.abs(a)[np.argmax(rel)]),
                "median_entry_magnitude": float(np.median(np.abs(a))),
                "tolerance": float(det["tolerance"]),
            }
            rows.append(row)
            if comm.rank == 0:
                logger.info(
                    "beta=%-6g seed=%d  max=%.4f%%  median=%.4f%%  "
                    "max|large-half=%.4f%%  dEV=%.2e  worst-entry=%.2e  %s",
                    beta, seed, 100 * row["max_rel_err"],
                    100 * row["median_rel_err"],
                    100 * row["max_rel_err_large_magnitude_half"],
                    row["max_rel_err_dEV"], row["worst_entry_magnitude"],
                    "PASS" if row["passed"] else "FAIL",
                )

    payload = {"betas": list(BETAS), "seeds": list(SEEDS), "rows": rows}
    if comm.rank == 0:
        by_beta = {}
        for b in BETAS:
            sel = [r for r in rows if r["beta"] == b]
            by_beta[str(b)] = {
                "max_rel_err_across_seeds": max(r["max_rel_err"] for r in sel),
                "median_of_median_rel_err": float(np.median([r["median_rel_err"] for r in sel])),
                "max_rel_err_large_magnitude_half": max(
                    r["max_rel_err_large_magnitude_half"] for r in sel
                ),
                "n_passed": sum(1 for r in sel if r["passed"]),
                "n_seeds": len(sel),
            }
        payload["by_beta"] = by_beta
        with open(run_dir / "fd_gate_probe.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

        logger.info("=== summary: does FD error grow with beta? ===")
        for b in BETAS:
            s = by_beta[str(b)]
            logger.info(
                "  beta=%-6g max=%.4f%%  median=%.4f%%  large-magnitude-half max=%.4f%%  passed %d/%d",
                float(b), 100 * s["max_rel_err_across_seeds"],
                100 * s["median_of_median_rel_err"],
                100 * s["max_rel_err_large_magnitude_half"],
                s["n_passed"], s["n_seeds"],
            )
        logger.info(
            "If the LARGE-MAGNITUDE-half column stays small while the raw max "
            "grows, the failure is a near-zero-denominator artifact and the "
            "gate needs a magnitude floor, not a looser tolerance."
        )

    manifest.record("fd_gate_probe", payload)
    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
