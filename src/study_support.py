"""Shared setup for the offline studies (convergence, SAA gap, baselines).

The study scripts under scripts/ need the same Stage-1..3 setup that
src/mainClean.py performs, but not the rest of the pipeline. Factoring it here
keeps the scripts short AND, more importantly, guarantees they build the eta
field with EXACTLY the same code path as the production run -- a study that
quietly differs from the pipeline it is meant to validate is worse than no
study.
"""
from __future__ import annotations

import logging

from mpi4py import MPI

from src.meshing.mesher import extract_simplices
from src.optimization.dolfiny_mma_driver import (
    RobustProblemContext, setup_robust_problem,
)
from src.random_fields.kernel import KernelParams
from src.random_fields.kl_expansion import KLExpansionResult, compute_kl_expansion

logger = logging.getLogger(__name__)


def build_stage3_kl(
    cfg, tagged_mesh, comm: MPI.Comm, length_scale: float | None = None,
    variance_threshold: float | None = None,
) -> KLExpansionResult:
    """Stage 3 exactly as mainClean.py runs it: KL expansion of the underlying
    Gaussian field on this mesh.

    Collective. node_coordinates/simplices exist only on rank 0 (mesh_serial
    lives there); compute_kl_expansion solves on rank 0 and broadcasts.

    The KL cache in kl_expansion.py is keyed on the node coordinates, so each
    mesh refinement level correctly gets its own expansion rather than a stale
    hit from another level.

    Args:
        length_scale: Overrides cfg.random_field.length_scale. Used by
            scripts/correlation_length_study.py to sweep l_c without mutating
            the config -- the sweep must run the SAME code path as production,
            differing only in the one parameter under study.
        variance_threshold: Overrides cfg.random_field.variance_threshold. Used
            for the truncation-robustness spot check: n_kl swings from 183 at
            l_c=1 to 4 at l_c=16 under the 95% rule, so "is 4 modes enough?" is
            a fair reviewer question. Re-running the extremes at 99% answers it
            with a number instead of an argument.
    """
    kernel_params = KernelParams(
        sigma=cfg.random_field.sigma,
        length_scale=(
            cfg.random_field.length_scale if length_scale is None else float(length_scale)
        ),
        spatial_dim=cfg.random_field.spatial_dim,
    )
    if comm.rank == 0:
        node_coordinates = tagged_mesh.mesh_serial.geometry.x
        simplices = extract_simplices(tagged_mesh)
    else:
        node_coordinates = simplices = None

    return compute_kl_expansion(
        node_coordinates, simplices, kernel_params,
        variance_threshold=(
            cfg.random_field.variance_threshold if variance_threshold is None
            else float(variance_threshold)
        ),
        comm=comm,
    )


def setup_context(
    fem: dict, opt: dict, rho_warm_start, kl_result: KLExpansionResult,
    load_cases: dict, case_name: str,
) -> RobustProblemContext:
    """The same RobustProblemContext the pipeline builds (FEA machinery, random
    Heaviside, sample-parallel groups, warm start)."""
    return setup_robust_problem(
        fem, opt, rho_warm_start, kl_result,
        load_cases=load_cases, case_name=case_name,
    )
