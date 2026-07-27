"""Convergence studies: does sigma_C mean anything, and is N big enough?

    mpirun -n 64 python scripts/convergence_studies.py mesh   [config]
    mpirun -n 64 python scripts/convergence_studies.py n-fixed [config]
    mpirun -n 64 python scripts/convergence_studies.py n-opt   [config]

WHY THESE TWO STUDIES
---------------------
`mesh` is the gating one. sigma_C is only a meaningful quantity if it is a
property of the CONTINUUM problem rather than of the discretization. Because the
manufacturing perturbation displaces the boundary by a fraction of an element,
that is a real question here, not a formality: if sigma_C/mu_C keeps moving as h
shrinks, then it is measuring discretization error and no downstream result --
no Pareto front, no baseline comparison -- survives. RUN THIS FIRST, and stop if
it does not converge.

The filter radius R is FIXED in absolute units across levels (box_source.py).
Scaling R with h would shrink the minimum feature size at every level, so the
design itself would keep changing and the study would converge to nothing while
looking perfectly well behaved.

`n-fixed` and `n-opt` answer different questions and both are needed:
  * n-fixed: how many samples does the ESTIMATOR need? Evaluate sigma_C at one
    frozen design over growing N. Cheap (no optimization).
  * n-opt: how many samples does the DESIGN need? Re-optimize at each N and
    compare the resulting designs. Expensive, and the one that justifies
    saa_n_samples.

Use monte_carlo sampling for these, not lhs. An optimized LHS design is NOT
nested -- the first 512 points of a 1024-point design are not a 512-point design
-- so an LHS curve carries design-to-design jitter that looks like convergence
and is not. The MC draw here is prefix-stable by construction.
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
from src.meshing.box_source import (
    _load_patch_halfwidth, build_box_fenitop_dicts, elements_for_refinement,
    loaded_area, realized_element_size,
)
from src.study_support import build_stage3_kl, setup_context
from src.optimization.saa_robust_driver import (
    _evaluate_saa, run_saa_robust_topopt,
)
from src.provenance import RunManifest, make_run_id
from src.validation.boundary_offset import build_report as build_boundary_offset_report
from src.validation.statistics import summarize_samples

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies"

# Refinement levels for the mesh study: h = 0.625, 0.476, 0.400.
#
# THESE ARE NOT FREELY CHOSEN. Three constraints intersect and leave very little
# room, and the reader of any convergence plot from this study needs to know it:
#
#   1. h <= 0.667 (R/h >= 0.9). Below that the Helmholtz filter is sub-element,
#      the filtered field jumps 0 to 1 inside one element, and there is no
#      interface band for eta to act on -- the eta model is degenerate, not just
#      inaccurate. (This is what made a smoke run return sigma_C exactly zero.)
#   2. h >= 0.4. The KL expansion is a DENSE O(N_nodes^2) eigensolve:
#      production's 51k nodes already need a 21 GB covariance matrix, and
#      halving h would need ~175 GB. The study cannot refine past production.
#   3. The resolved traction area should agree across levels (see
#      study_mesh's docstring). Vertex-based facet selection quantizes the patch
#      to O(h), so only particular level combinations agree closely.
#
# Together these cap the h-range at about 1.6x, which is a NARROW basis for a
# convergence claim -- roughly one refinement step, not the 4x a textbook study
# would use. The levels below are the best available compromise: the widest
# feasible h-range with a resolved interface everywhere, a load-area spread of
# ~1.4x, and the finest level equal to the production mesh so the study speaks
# directly to the production result.
#
# TO DO BETTER, the KL expansion has to stop being O(N_nodes^2) in the FEA mesh.
# The eta field is smooth (length_scale 4.0 against h ~ 0.5), so it could be
# expanded on a coarse auxiliary mesh and interpolated onto the FEA nodes,
# decoupling the two and allowing h down to ~0.2. That is a real change to
# RandomFieldHeaviside's coordinate matching, not a parameter tweak.
MESH_LEVELS = (0.62, 0.825, 0.985)

# Sample counts. Nested by construction under monte_carlo sampling.
N_LEVELS_FIXED_DESIGN = (32, 64, 128, 256, 512, 1024, 2048)
N_LEVELS_REOPTIMIZE = (64, 128, 256, 512)


def _write(payload: dict, path: Path) -> None:
    if comm.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        logger.info("Wrote %s", path)


def study_mesh(cfg, run_dir: Path, manifest: RunManifest) -> dict:
    """sigma_C/mu_C versus mesh size, at fixed R, fixed load and fixed eta model.

    THREE things are pinned across levels, and each one is pinned for the same
    reason: a mesh study is only meaningful if every level solves the SAME
    continuum problem.

      * the filter radius R (box_source._FILTER_RADIUS) -- scaling it with h
        would shrink the minimum feature size at every level;
      * the eta model (band, kernel, variance threshold) -- unchanged by
        construction, it comes from cfg;
      * the TRACTION PATCH, pinned here. The automatic per-mesh rule
        max(0.5, 1.5h) has to widen the patch on coarse meshes so that any facet
        is selected at all, which means the patch SHRINKS as the mesh refines
        and the load concentrates progressively. Left alone, each level would
        solve a slightly different problem. Pinning it to the value the COARSEST
        level needs makes the load identical everywhere.

    The cost of pinning is that no level reproduces beam_3d's own load exactly
    (its patch is narrower than the coarsest mesh can resolve). That is the
    right trade: this study is about convergence, not about reproducing the
    reference case, and the production run in config.yaml is unaffected.
    """
    coarsest = min(MESH_LEVELS)
    coarsest_h = realized_element_size(elements_for_refinement(coarsest))
    pinned_halfwidth = _load_patch_halfwidth(coarsest_h)

    # Report the h-range up front: it is only ~1.6x (see MESH_LEVELS), which
    # bounds how strong a convergence claim this study can support.
    h_values = [realized_element_size(elements_for_refinement(r)) for r in MESH_LEVELS]
    if comm.rank == 0:
        logger.warning(
            "Mesh study spans h = %s, a range of only %.2fx. That is about one "
            "refinement step and is a NARROW basis for a convergence claim -- "
            "state the range alongside any conclusion drawn from it. The cap "
            "comes from the dense KL eigensolve at the fine end and the "
            "R/h >= 0.9 interface-resolution requirement at the coarse end.",
            [round(h, 4) for h in h_values], max(h_values) / min(h_values),
        )
    cfg.box_mesh.load_patch_halfwidth = pinned_halfwidth
    if comm.rank == 0:
        logger.info(
            "Mesh study: pinning the traction patch half-width to %.4g (the "
            "value the coarsest level, refinement=%.4g, requires) so every "
            "level solves the same load case. This deliberately differs from "
            "beam_3d's own 0.5.", pinned_halfwidth, coarsest,
        )
    manifest.record("pinned_load_patch_halfwidth", pinned_halfwidth)

    results = []
    for refinement in MESH_LEVELS:
        cfg.box_mesh.refinement = refinement
        elements = elements_for_refinement(refinement)
        if comm.rank == 0:
            logger.info("=== mesh level refinement=%.4g -> %s ===", refinement, elements)

        tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
        case_name = next(iter(load_cases))

        with manifest.stage(f"mesh_{refinement}_nominal"):
            rho_nominal = topopt(
                fem, opt, load_cases,
                output_prefix=str(run_dir / f"mesh_{refinement}_"),
            )
        rho_nominal = comm.bcast(rho_nominal, root=0)

        kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
        ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

        # Boundary offset at this level: the perturbation is fixed in absolute
        # units, so offset/h grows as the mesh refines. Reporting it alongside
        # sigma_C is what lets a reader see whether a change in sigma_C tracks
        # resolution rather than physics.
        ctx.rho_field.x.petsc_vec.array[:] = ctx.rho_warm_start_local
        ctx.rho_field.x.scatter_forward()
        ctx.density_filter.forward()
        offset = build_boundary_offset_report(
            ctx.rho_phys_field, opt["transform_params"],
            element_size=fem["element_size"], filter_radius=opt["filter_radius"],
        )

        xi = np.random.default_rng(cfg.optimization.saa_seed).standard_normal(
            size=(cfg.mc_validation.n_samples, kl_result.n_kl)
        )
        with manifest.stage(f"mesh_{refinement}_evaluate"):
            evaluation = _evaluate_saa(
                ctx, opt, ctx.rho_warm_start_local, xi,
                float(cfg.optimization.saa_beta_max), accumulate_gradients=True,
            )

        summary = (
            summarize_samples(evaluation.compliance_samples,
                              seed=cfg.mc_validation.bootstrap_seed)
            if comm.rank == 0 else None
        )
        results.append({
            "refinement": refinement,
            "elements": elements,
            "element_size_h": fem["element_size"],
            "filter_radius_R": opt["filter_radius"],
            "R_over_h": opt["filter_radius"] / fem["element_size"],
            # Recorded per level so a reader can confirm the load really was
            # identical everywhere -- these two must not vary down the table.
            "load_patch_halfwidth": pinned_halfwidth,
            "loaded_area": loaded_area(fem["element_size"], pinned_halfwidth),
            "n_kl": int(kl_result.n_kl),
            "variance_explained": float(kl_result.variance_explained),
            "mu_C": evaluation.mu_C,
            "sigma_C": evaluation.sigma_C,
            "cov_sigma_over_mu": evaluation.sigma_C / evaluation.mu_C,
            "boundary_offset": offset.as_dict(),
            "statistics": summary,
        })
        if comm.rank == 0:
            logger.info(
                "refinement=%.4g h=%.4g: mu_C=%.6g sigma_C=%.6g "
                "sigma/mu=%.4g offset=%.3g h n_kl=%d",
                refinement, fem["element_size"], evaluation.mu_C,
                evaluation.sigma_C, evaluation.sigma_C / evaluation.mu_C,
                offset.offset_std_elements, kl_result.n_kl,
            )

    verdict = _mesh_verdict(results)
    payload = {
        "levels": results,
        "verdict": verdict,
        "h_range": max(h_values) / min(h_values),
        "pinned_load_patch_halfwidth": pinned_halfwidth,
        "limitations": (
            "h-range is capped at ~1.6x by the dense O(N_nodes^2) KL eigensolve "
            "at the fine end and by the R/h >= 0.9 interface-resolution "
            "requirement at the coarse end. The resolved traction area also "
            "varies by ~1.4x because facet selection quantizes the fixed "
            "geometric patch to O(h). Both bound how strongly this study can "
            "support a convergence claim; see MESH_LEVELS for what it would "
            "take to widen it."
        ),
    }
    _write(payload, run_dir / "mesh_convergence.json")
    if comm.rank == 0 and not verdict["converged"]:
        logger.error(
            "sigma_C/mu_C has NOT converged between the two finest levels "
            "(%.4g vs %.4g, relative change %.3g). STOP -- no Pareto front or "
            "baseline comparison built on it is meaningful until this is "
            "resolved.",
            verdict["coarser_cov"], verdict["finest_cov"], verdict["relative_change"],
        )
        if verdict["load_discretization_is_a_candidate_confounder"]:
            logger.error(
                "BEFORE concluding sigma_C is a discretization artifact: the "
                "resolved traction area varied by %.2fx across levels (total "
                "force was conserved exactly, but facet selection resolves the "
                "fixed geometric patch to its inner envelope, which depends on "
                "node placement). The levels therefore did not see quite the "
                "same load distribution, which alone could move sigma_C. Rule "
                "that out -- e.g. by re-running on refinement levels whose "
                "resolved areas agree closely -- before blaming the physics.",
                verdict["loaded_area_spread_across_levels"],
            )
    return payload


def _mesh_verdict(results: list[dict]) -> dict:
    """Compare the two finest levels against the bootstrap CI of the finest."""
    ordered = sorted(results, key=lambda r: r["element_size_h"])
    finest, coarser = ordered[0], ordered[1]
    relative_change = abs(
        finest["cov_sigma_over_mu"] - coarser["cov_sigma_over_mu"]
    ) / abs(finest["cov_sigma_over_mu"])

    tolerance = 0.10
    stats = finest.get("statistics")
    if stats:
        # If the CI on sigma at the finest level is wider than 10%, demanding
        # 10% agreement is demanding more than the measurement can deliver.
        tolerance = max(tolerance, stats["std"]["relative_half_width"])

    # Confounder check. The traction patch is a fixed GEOMETRIC region, but
    # vertex-based facet selection resolves it to the inner envelope, so the
    # loaded AREA varies with h even though the total force is conserved
    # exactly. If sigma_C has not converged, this has to be ruled out before
    # blaming the physics -- a load whose distribution changed between levels is
    # a perfectly good reason for the response to change too.
    areas = [r["loaded_area"] for r in results]
    load_area_spread = max(areas) / min(areas) if min(areas) > 0 else float("inf")

    converged = bool(relative_change <= tolerance)
    return {
        "finest_h": finest["element_size_h"],
        "coarser_h": coarser["element_size_h"],
        "finest_cov": finest["cov_sigma_over_mu"],
        "coarser_cov": coarser["cov_sigma_over_mu"],
        "relative_change": relative_change,
        "tolerance": tolerance,
        "converged": converged,
        "loaded_area_spread_across_levels": load_area_spread,
        "load_discretization_is_a_candidate_confounder": bool(
            not converged and load_area_spread > 1.2
        ),
        "note": (
            "Converged means sigma_C/mu_C changes between the two finest meshes "
            "by less than the bootstrap CI of the finest (floored at 10%). If "
            "this is False, sigma_C may be a discretization artifact -- but "
            "check loaded_area_spread_across_levels first: the total force is "
            "conserved exactly, yet the resolved patch AREA varies with h "
            "because facet selection takes the inner envelope of the geometric "
            "patch. A spread well above 1 means the levels did not see quite "
            "the same load distribution, which is an alternative explanation "
            "for a moving sigma_C and must be excluded before drawing a "
            "conclusion about the continuum problem."
        ),
    }


def study_n_fixed_design(cfg, run_dir: Path, manifest: RunManifest) -> dict:
    """sigma_C at ONE frozen design over growing N. Answers: how many samples
    does the estimator need? Cheap -- evaluation only, no optimization."""
    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    with manifest.stage("n_fixed_nominal"):
        rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "n_fixed_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    # Prefix-stable: one draw of the largest block, then take leading slices, so
    # the N-curve nests instead of resampling at every point.
    n_max = max(N_LEVELS_FIXED_DESIGN)
    xi_all = np.random.default_rng(cfg.optimization.saa_seed).standard_normal(
        size=(n_max, kl_result.n_kl)
    )

    results = []
    for n in N_LEVELS_FIXED_DESIGN:
        evaluation = _evaluate_saa(
            ctx, opt, ctx.rho_warm_start_local, xi_all[:n],
            float(cfg.optimization.saa_beta_max), accumulate_gradients=True,
        )
        summary = (
            summarize_samples(evaluation.compliance_samples,
                              seed=cfg.mc_validation.bootstrap_seed)
            if comm.rank == 0 else None
        )
        results.append({
            "N": n, "mu_C": evaluation.mu_C, "sigma_C": evaluation.sigma_C,
            "statistics": summary,
        })
        if comm.rank == 0:
            logger.info("N=%5d: mu_C=%.6g sigma_C=%.6g", n, evaluation.mu_C,
                        evaluation.sigma_C)

    payload = {"levels": results, "nested": True, "sampling": "monte_carlo"}
    _write(payload, run_dir / "n_convergence_fixed_design.json")
    return payload


def study_n_reoptimize(cfg, run_dir: Path, manifest: RunManifest) -> dict:
    """Re-optimize at each N. Answers: how many samples does the DESIGN need?

    Designs are compared by the L1 distance between them and by their
    out-of-sample performance on one common large evaluation set -- comparing
    each design on its OWN sample set would compare designs on different
    yardsticks and reward overfitting.
    """
    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    with manifest.stage("n_opt_nominal"):
        rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "n_opt_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    n_max = max(N_LEVELS_REOPTIMIZE)
    xi_all = np.random.default_rng(cfg.optimization.saa_seed).standard_normal(
        size=(n_max, kl_result.n_kl)
    )
    # Independent evaluation set -- disjoint seed, so no design is scored on the
    # samples it was fitted to.
    xi_eval = np.random.default_rng(cfg.mc_validation.seed).standard_normal(
        size=(cfg.mc_validation.n_samples, kl_result.n_kl)
    )

    lam = float(cfg.optimization.lambda_sweep[-1])
    results, designs = [], {}
    for n in N_LEVELS_REOPTIMIZE:
        with manifest.stage(f"n_opt_N{n}"):
            solved = run_saa_robust_topopt(ctx, opt, lam, xi_all[:n])
        designs[n] = np.asarray(solved["rho_robust"])

        ctx.warm_start_comm.bcast(ctx.rho_field, solved["rho_robust"])
        out_of_sample = _evaluate_saa(
            ctx, opt, ctx.rho_field.x.petsc_vec.array.copy(), xi_eval,
            float(cfg.optimization.saa_beta_max), accumulate_gradients=True,
        )
        results.append({
            "N": n,
            "in_sample_mu_C": solved["mu_C"],
            "in_sample_sigma_C": solved["sigma_C"],
            "out_of_sample_mu_C": out_of_sample.mu_C,
            "out_of_sample_sigma_C": out_of_sample.sigma_C,
            "optimism_sigma_relative": (
                (solved["sigma_C"] - out_of_sample.sigma_C) / out_of_sample.sigma_C
                if out_of_sample.sigma_C > 0 else float("nan")
            ),
            "converged": solved["converged"],
            "M_nd_percent": solved["M_nd_percent"],
        })
        if comm.rank == 0:
            logger.info(
                "N=%4d: in-sample sigma=%.6g, out-of-sample sigma=%.6g "
                "(optimism %+.2f%%)", n, solved["sigma_C"],
                out_of_sample.sigma_C, 100 * results[-1]["optimism_sigma_relative"],
            )

    if comm.rank == 0:
        reference = designs[max(N_LEVELS_REOPTIMIZE)]
        for record in results:
            design = designs[record["N"]]
            record["mean_abs_design_difference_vs_largest_N"] = float(
                np.mean(np.abs(design - reference))
            )

    payload = {"levels": results, "lambda": lam}
    _write(payload, run_dir / "n_convergence_reoptimize.json")
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)

    if len(sys.argv) < 2 or sys.argv[1] not in ("mesh", "n-fixed", "n-opt"):
        raise SystemExit(
            "usage: convergence_studies.py {mesh|n-fixed|n-opt} [config.yaml]\n"
            "Run 'mesh' FIRST -- it is a stop condition, not a checkbox."
        )
    mode = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "src/config/configStudy.yaml"
    cfg = load_config(config_path)

    run_id = make_run_id(comm)
    manifest = RunManifest(run_id, comm)
    run_dir = OUTPUT_ROOT / mode / run_id
    if comm.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    manifest.record("study", mode)
    manifest.record("config_path", config_path)

    if mode == "mesh":
        study_mesh(cfg, run_dir, manifest)
    elif mode == "n-fixed":
        study_n_fixed_design(cfg, run_dir, manifest)
    else:
        study_n_reoptimize(cfg, run_dir, manifest)

    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
