"""Locate artifacts under the run-id output layout.

The other scripts in viz/ were written against the pre-run-id layout
(`output/mc_validation/`, `output/stage2_fea/rho_converged.npy`, a single
lambda = 1.0 design). src/mainClean.py now writes every stage into
`output/stage<N>_<name>/<run_id>/`, and the study scripts into
`output/studies/<study>/<run_id>/`, so hard-coded paths silently resolve to
stale artifacts from July runs instead of failing loudly.

This module resolves "the newest run" once, so a figure is never accidentally
built from a mixture of runs.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def newest(pattern: str) -> Path | None:
    """Newest path matching a repo-relative glob, or None."""
    hits = sorted(glob.glob(str(ROOT / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return Path(hits[-1]) if hits else None


def load_json(pattern: str):
    """(payload, path) for the newest match, or (None, None)."""
    path = newest(pattern)
    if path is None:
        return None, None
    try:
        with open(path) as handle:
            return json.load(handle), path
    except (OSError, ValueError):
        return None, path


# Canonical locations of the artifacts the research figures need.
ARTIFACTS = {
    "lc_sweep": "output/studies/correlation_length/*/correlation_length_fixed_design.json",
    "lc_sweep_vthr": "output/studies/correlation_length/*/correlation_length_fixed_design_vthr*.json",
    "lc_reopt": "output/studies/correlation_length/*/correlation_length_reoptimize.json",
    "mesh": "output/studies/mesh/*/mesh_convergence.json",
    "n_fixed": "output/studies/n-fixed/*/n_convergence_fixed_design.json",
    "gap": "output/studies/saa_gap/*/saa_gap.json",
    "baseline": "output/studies/baseline_comparison/*/baseline_comparison.json",
    "uniform_eta": "output/studies/uniform_eta/uniform_eta_comparison.json",
    "pareto": "output/stage5_optimization/*/pareto_results.json",
    "validation": "output/stage6_validation/*/validation_summary.json",
}


def artifact(name: str):
    """(payload, path) for a named artifact. Raises on an unknown name so a
    typo fails immediately rather than yielding an empty figure."""
    if name not in ARTIFACTS:
        raise KeyError(f"unknown artifact {name!r}; known: {sorted(ARTIFACTS)}")
    return load_json(ARTIFACTS[name])


def report_available() -> str:
    lines = []
    for name, pattern in ARTIFACTS.items():
        path = newest(pattern)
        mark = "x" if path else " "
        where = str(path.relative_to(ROOT)) if path else "(absent)"
        lines.append(f"  [{mark}] {name:<16} {where}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("artifacts discoverable under the current output layout:")
    print(report_available())
