"""How much over-etching does a finished design survive? -- the beta=128 defence.

    mpirun -n 32 python scripts/measure_feature_size.py <rho.npy> [config.yaml]
    mpirun -n 32 python scripts/measure_feature_size.py --all [config.yaml]

Schevenels, Lazarov & Sigmund (CMAME 2011) cap beta at 32 and explicitly refuse
128, on the grounds that high beta yields single-element-wide features that
cannot be manufactured. Their filter was resolved at R/h = 8.4; this project
runs beta = 128 at R/h = 1.5. The objection therefore applies with MORE force
here, and M_nd does not answer it -- a single-element strut is also near-binary.

This sweeps the projection threshold upward (a morphological erosion of depth
(eta - 0.5) * 2R) and finds where the load path fails. See
src/validation/feature_size.py for what that does and does not measure.

Deliberately standalone: it reads a saved design and touches nothing in the
running pipeline, so it can be applied after the fact to any .npy the drivers
left behind.
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
from src.provenance import RunManifest, make_run_id
from src.study_support import build_stage3_kl, setup_context
from src.validation.feature_size import measure_erosion_robustness

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "feature_size"

# Designs worth measuring, newest first, when --all is given.
_DESIGN_GLOBS = (
    "output/studies/uniform_eta/rho_*.npy",
    "output/studies/correlation_length/*/rho_lc_*.npy",
    "output/studies/baseline_comparison/*/rho_*.npy",
    "output/stage5_optimization/*/rho_robust_lambda_*.npy",
)


def _collect(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    import glob
    hits: list[Path] = []
    for pattern in _DESIGN_GLOBS:
        hits.extend(Path(p) for p in sorted(glob.glob(pattern)))
    return hits


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)

    argv = sys.argv[1:]
    configs = [a for a in argv if a.endswith((".yaml", ".yml"))]
    rest = [a for a in argv if not a.endswith((".yaml", ".yml"))]
    cfg = load_config(configs[0] if configs else "src/config/configStudy.yaml")

    explicit = [a for a in rest if a != "--all"]
    designs = _collect(explicit)
    if not designs:
        raise SystemExit("no designs found; pass a .npy path or --all")

    run_id = make_run_id(comm)
    manifest = RunManifest(run_id, comm)
    run_dir = OUTPUT_ROOT / run_id
    if comm.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    manifest.record_config(cfg, effective_fem=fem, effective_opt=opt)
    rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "fs_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    beta = float(cfg.optimization.saa_beta_max)
    radius = float(opt["filter_radius"])

    results = {}
    for path in designs:
        rho = np.load(path) if comm.rank == 0 else None
        rho = comm.bcast(rho, root=0)
        if comm.rank == 0:
            logger.info("=== %s ===", path)
        ctx.warm_start_comm.bcast(ctx.rho_field, np.asarray(rho))
        ctx.density_filter.forward()
        results[str(path)] = measure_erosion_robustness(
            ctx, opt, beta=beta, filter_radius=radius,
        )

    payload = {
        "beta": beta,
        "filter_radius": radius,
        "min_feature_size_2R": 2.0 * radius,
        "element_size": fem.get("element_size"),
        "designs": results,
        "why": (
            "Answers the Schevenels et al. (2011) objection to beta >= 128. "
            "M_nd shows a design is near-binary; it does not show the members "
            "are thick enough to make. This measures the erosion depth the "
            "load path survives, which does."
        ),
    }
    if comm.rank == 0:
        with open(run_dir / "feature_size.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        for name, r in results.items():
            logger.info(
                "%-58s survives eta<=%.3f, depth %.4g (%.2f el), "
                "implied thickness >= %.4g = %.0f%% of 2R",
                Path(name).name, r["max_survived_eta"], r["max_survived_erosion"],
                r["max_survived_erosion_elements"] or float("nan"),
                r["implied_min_thickness"],
                100.0 * r["implied_min_thickness"] / (2.0 * radius),
            )
    manifest.record("feature_size", payload)
    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
