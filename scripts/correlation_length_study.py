"""When does the spatial correlation of the manufacturing error actually matter?

    mpirun -n 64 python scripts/correlation_length_study.py fixed [cfg]
    mpirun -n 64 python scripts/correlation_length_study.py reoptimize 4 32 [cfg]

THE GAP THIS FILLS
------------------
Schevenels, Lazarov & Sigmund (CMAME 200:3613-3627, 2011) model the projection
threshold as a random field with a squared-exponential kernel, pick ONE
correlation length (l_c = 0.3L), and conclude from it that designs optimized
against uniform errors are just as robust to non-uniform ones. That conclusion
is drawn from a single point in a parameter the theory says is decisive:

  * as l_c -> infinity the field becomes spatially constant, and the model
    collapses onto the uniform-error case by construction;
  * as l_c -> 0 the erosions and dilations average out within any finite
    region -- which is precisely the cancellation they invoke to explain why
    their heat sink's sigma_C HALVED under spatial variation;
  * so the effect must be non-monotonic, with a worst l_c somewhere in between,
    and nobody has measured where.

This script measures it. The deliverable is sigma_C/mu_C versus l_c with the
uniform limit marked -- the curve that says WHEN spatial correlation matters
rather than assuming it always does (or, per their single data point, never
does).

The expected scale of the peak is set by the geometry, not by the filter: the
error correlates over l_c, the beam is 30 long and 10 across, so the transition
from "many independent weak cross-sections" to "one global erosion" should
happen as l_c passes through the transverse dimension. Both ratios are reported.

RELATION TO THE OTHER SCRIPTS
-----------------------------
The l_c -> infinity end of this curve is exactly the uniform arm of
scripts/uniform_eta_baseline.py, and this script evaluates that limit through
the same build_uniform_eta_kl() degenerate expansion, so the two studies are
directly comparable rather than merely similar. Only the length scale varies;
the mesh, the filter radius R = 0.6, the eta band, the beta schedule and the
evaluation ensemble are all held fixed.
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
from src.meshing.box_source import _DOMAIN, build_box_fenitop_dicts
from src.optimization.saa_robust_driver import _evaluate_saa, run_saa_robust_topopt
from src.provenance import RunManifest, make_run_id
from src.random_fields.kl_expansion import build_uniform_eta_kl
from src.sampling.sampler import generate_samples
from src.study_support import build_stage3_kl, setup_context
from src.validation.statistics import summarize_samples

comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "correlation_length"

# Domain is 10 x 30 x 10, so this brackets both the transverse dimension (10)
# and the axial one (30). l_c = 32 exceeds the whole domain and must reproduce
# the uniform limit -- it is included as an internal consistency check, not for
# its own sake. The config default (4.0) sits deliberately inside the range.
L_C_LEVELS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

# The uniform arm's stochastic dimension is 1, so it needs far fewer samples for
# the SAA loop -- but the EVALUATION ensemble size is always mc_validation.n_samples
# so every point on the curve carries the same estimator noise.
N_UNIFORM_SAA = 64

_DOMAIN_SPAN = np.asarray(_DOMAIN[1], dtype=float) - np.asarray(_DOMAIN[0], dtype=float)
_L_AXIAL = float(_DOMAIN_SPAN.max())
_L_TRANSVERSE = float(_DOMAIN_SPAN.min())


def _write(payload: dict, path: Path) -> None:
    if comm.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)


def _level_label(length_scale: float | None) -> str:
    return "uniform" if length_scale is None else f"{length_scale:g}"


def _build_kl(cfg, tagged_mesh, length_scale: float | None, vthr: float | None = None):
    """The expansion for one level. `None` means the uniform (l_c -> infinity)
    limit, built as the same degenerate one-mode expansion the baseline study
    uses so the two are the same object, not two implementations of one idea."""
    if length_scale is None:
        return build_uniform_eta_kl(
            build_stage3_kl(cfg, tagged_mesh, comm, variance_threshold=vthr)
        )
    return build_stage3_kl(
        cfg, tagged_mesh, comm, length_scale=length_scale, variance_threshold=vthr,
    )


def _ratios(length_scale: float | None) -> dict:
    if length_scale is None:
        return {"l_c_over_axial": None, "l_c_over_transverse": None}
    return {
        "l_c_over_axial": length_scale / _L_AXIAL,
        "l_c_over_transverse": length_scale / _L_TRANSVERSE,
    }


def study_fixed(cfg, run_dir: Path, manifest: RunManifest,
                only_levels: tuple | None = None, vthr: float | None = None) -> dict:
    """sigma_C of ONE frozen design across l_c. Evaluation only -- this locates
    the peak cheaply, before any re-optimization is spent on it.

    Args:
        only_levels: Subset of L_C_LEVELS to run (plus the uniform limit). Used
            by the truncation spot check, which only needs the extremes.
        vthr: KL variance_threshold override. Re-running the l_c extremes at a
            tighter threshold tests whether the curve is an artifact of
            truncation -- n_kl is 183 at l_c=1 but only 4 at l_c=16 under the
            95% rule, and a reviewer will ask whether 4 modes is enough.
    """
    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    manifest.record_config(cfg, effective_fem=fem, effective_opt=opt)
    with manifest.stage("lc_fixed_nominal"):
        rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "lc_fixed_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    beta_max = float(cfg.optimization.saa_beta_max)
    n_eval = int(cfg.mc_validation.n_samples)
    # NOT named `levels`: that would shadow the only_levels parameter, and since
    # an empty list is falsy-but-not-None the sweep would silently collapse to
    # the uniform level alone.
    results: list[dict] = []

    sweep = (*(only_levels if only_levels is not None else L_C_LEVELS), None)
    for length_scale in sweep:
        label = _level_label(length_scale)
        kl_result = _build_kl(cfg, tagged_mesh, length_scale, vthr)
        ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

        # Drawn per level because n_kl differs between levels; the SEED is fixed
        # so the underlying standard normals are as comparable as differing
        # dimensions allow.
        xi = np.random.default_rng(cfg.mc_validation.seed).standard_normal(
            size=(n_eval, kl_result.n_kl)
        )
        evaluation = _evaluate_saa(
            ctx, opt, ctx.rho_warm_start_local, xi, beta_max, accumulate_gradients=True,
        )
        summary = (
            summarize_samples(evaluation.compliance_samples,
                              seed=cfg.mc_validation.bootstrap_seed)
            if comm.rank == 0 else None
        )
        # Persist the RAW ensemble, not just its summary. Without this, adding
        # any new statistic later -- a ratio CI, a tail index, a different
        # confidence level -- costs a full re-run of 1000 FEA solves per level.
        # With it, every future statistic is post-processing on a 8 kB array.
        if comm.rank == 0:
            np.save(run_dir / f"compliance_samples_lc_{label}.npy",
                    np.asarray(evaluation.compliance_samples))
        results.append({
            "length_scale": length_scale,
            "label": label,
            **_ratios(length_scale),
            "n_kl": int(kl_result.n_kl),
            "variance_explained": float(kl_result.variance_explained),
            "mu_C": evaluation.mu_C,
            "sigma_C": evaluation.sigma_C,
            "cv": evaluation.sigma_C / evaluation.mu_C,
            "statistics": summary,
        })
        if comm.rank == 0:
            cv_est = (summary or {}).get("cv", {})
            logger.info(
                "l_c=%-8s n_kl=%-4d var_expl=%.4f  mu_C=%.6g sigma_C=%.6g "
                "cv=%.5g [%.5g, %.5g]",
                label, kl_result.n_kl, kl_result.variance_explained,
                evaluation.mu_C, evaluation.sigma_C,
                evaluation.sigma_C / evaluation.mu_C,
                cv_est.get("ci_low", float("nan")),
                cv_est.get("ci_high", float("nan")),
            )
            target_vthr = vthr if vthr is not None else cfg.random_field.variance_threshold
            if kl_result.variance_explained < target_vthr - 1e-9:
                logger.warning(
                    "l_c=%s retained only %.4f of the variance, below the "
                    "configured threshold %.4f. The mesh cannot resolve this "
                    "correlation length; this point UNDERSTATES sigma_C and "
                    "must not be read as a physical trend.",
                    label, kl_result.variance_explained, target_vthr,
                )

    payload = _summarize_curve(results, cfg, mode="fixed_design")
    payload["variance_threshold_used"] = (
        vthr if vthr is not None else cfg.random_field.variance_threshold
    )
    suffix = "" if vthr is None else f"_vthr{vthr:g}"
    _write(payload, run_dir / f"correlation_length_fixed_design{suffix}.json")
    return payload


def study_reoptimize(cfg, run_dir: Path, manifest: RunManifest,
                     length_scales: list[float | None]) -> dict:
    """Re-optimize at selected l_c, then score EVERY design under EVERY error
    model -- the cross-evaluation that tests conservatism at the design level.

    WHY A GRID AND NOT A SINGLE YARDSTICK
    -------------------------------------
    The fixed-design sweep shows that, for one frozen design, response variance
    rises monotonically with correlation length and is maximal in the uniform
    limit. That bounds the RESPONSE. It does not by itself establish the claim
    that matters for practice: that a design optimized against the cheap scalar
    model is at least as robust, under any correlation length, as one optimized
    against the full field.

    Testing that needs design-at-X scored-at-Y for all pairs. Scoring every
    design at one reference l_c (the previous behaviour) cannot see it: it
    reports how the designs differ under ONE error model, not whether any
    design is exposed under some OTHER one. The conservatism claim is a
    statement about each design's WORST case across the row.

    Every column uses one common ensemble with common random numbers, so
    designs are paired within an error model. Rows are not paired across
    columns -- different l_c means a different population, and n_kl differs.
    """
    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    manifest.record_config(cfg, effective_fem=fem, effective_opt=opt)
    with manifest.stage("lc_reopt_nominal"):
        rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(run_dir / "lc_reopt_"))
    rho_nominal = comm.bcast(rho_nominal, root=0)

    lam = float(cfg.optimization.lambda_sweep[-1])
    beta_max = float(cfg.optimization.saa_beta_max)
    n_eval = int(cfg.mc_validation.n_samples)

    # ---- Phase A: one optimization per correlation length ------------------
    # The nominal (deterministic) design is carried along as the control row:
    # without it there is nothing to show the robust formulation did anything.
    designs: dict[str, np.ndarray] = {"nominal": np.asarray(rho_nominal)}
    optimized: list[dict] = []

    for length_scale in length_scales:
        label = _level_label(length_scale)
        kl_result = _build_kl(cfg, tagged_mesh, length_scale)
        ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

        n_saa = (
            N_UNIFORM_SAA if length_scale is None
            else int(cfg.optimization.saa_n_samples)
        )
        xi = generate_samples(
            kl_result, n_saa,
            strategy=cfg.optimization.saa_sampling_strategy,
            seed=cfg.optimization.saa_seed,
        ).xi
        if comm.rank == 0:
            logger.info("=== optimizing at l_c=%s (n_kl=%d, N=%d) ===",
                        label, kl_result.n_kl, n_saa)
        with manifest.stage(f"lc_reopt_{label}"):
            solved = run_saa_robust_topopt(ctx, opt, lam, xi)

        designs[label] = np.asarray(solved["rho_robust"])
        optimized.append({
            "design": label,
            "length_scale": length_scale,
            **_ratios(length_scale),
            "n_kl": int(kl_result.n_kl),
            "saa_n_samples": n_saa,
            "in_sample_mu_C": solved["mu_C"],
            "in_sample_sigma_C": solved["sigma_C"],
            "M_nd_percent": solved["M_nd_percent"],
            "converged": solved["converged"],
            "volume_violation": solved["volume_violation"],
        })
        if comm.rank == 0:
            np.save(run_dir / f"rho_lc_{label}.npy", designs[label])

    # ---- Phase B: score every design under every error model ---------------
    grid: dict[str, dict[str, dict]] = {name: {} for name in designs}
    for length_scale in length_scales:
        eval_label = _level_label(length_scale)
        kl_eval = _build_kl(cfg, tagged_mesh, length_scale)
        ctx_eval = setup_context(fem, opt, rho_nominal, kl_eval, load_cases, case_name)
        xi_common = np.random.default_rng(cfg.mc_validation.seed).standard_normal(
            size=(n_eval, kl_eval.n_kl)
        )
        for design_label, rho in designs.items():
            ctx_eval.warm_start_comm.bcast(ctx_eval.rho_field, rho)
            evaluation = _evaluate_saa(
                ctx_eval, opt, ctx_eval.rho_field.x.petsc_vec.array.copy(),
                xi_common, beta_max, accumulate_gradients=True,
            )
            cell = {
                "mu_C": evaluation.mu_C,
                "sigma_C": evaluation.sigma_C,
                "cv": evaluation.sigma_C / evaluation.mu_C,
            }
            if comm.rank == 0:
                cell["statistics"] = summarize_samples(
                    evaluation.compliance_samples,
                    seed=cfg.mc_validation.bootstrap_seed,
                )
                np.save(
                    run_dir / f"compliance_design_{design_label}_at_{eval_label}.npy",
                    np.asarray(evaluation.compliance_samples),
                )
                logger.info(
                    "  design=%-8s scored at l_c=%-8s cv=%.5g",
                    design_label, eval_label, cell["cv"],
                )
            grid[design_label][eval_label] = cell

    # ---- The conservatism test --------------------------------------------
    # A design's exposure is its WORST cv across the error models, because a
    # manufacturer does not get to choose the correlation length.
    worst: dict[str, dict] = {}
    for design_label, row in grid.items():
        if not row:
            continue
        at, cell = max(row.items(), key=lambda kv: kv[1]["cv"])
        worst[design_label] = {"worst_cv": cell["cv"], "attained_at_l_c": at}

    robust_rows = {k: v for k, v in worst.items() if k != "nominal"}
    best_design = min(robust_rows, key=lambda k: robust_rows[k]["worst_cv"]) if robust_rows else None
    uniform_worst = robust_rows.get("uniform", {}).get("worst_cv")
    conservative = None
    if uniform_worst is not None and robust_rows:
        # "Conservative" here means: optimizing against the cheap scalar model
        # leaves you no worse off, in the worst case, than optimizing against
        # any field model tested.
        conservative = bool(
            uniform_worst <= min(v["worst_cv"] for v in robust_rows.values()) + 1e-12
        )

    payload = {
        "mode": "reoptimize",
        "lambda": lam,
        "n_evaluation_samples": n_eval,
        "designs_optimized": optimized,
        "cross_evaluation_cv": {
            d: {e: c["cv"] for e, c in row.items()} for d, row in grid.items()
        },
        "cross_evaluation_full": grid,
        "worst_case_per_design": worst,
        "best_worst_case_design": best_design,
        "uniform_model_is_conservative": conservative,
        "interpretation": (
            "Rows are designs, columns are the error model they were scored "
            "under. worst_case_per_design is each design's largest cv across "
            "the row -- its exposure, since the correlation length is not the "
            "designer's to choose. uniform_model_is_conservative is True when "
            "the design optimized against a scalar random eta has the smallest "
            "(or equal-smallest) worst case, i.e. the cheap model costs nothing "
            "in robustness. That is the practical form of the bound the "
            "fixed-design sweep establishes for the response."
        ),
    }
    _write(payload, run_dir / "correlation_length_reoptimize.json")

    if comm.rank == 0:
        logger.info("cross-evaluation cv (rows=design, cols=error model):")
        cols = [_level_label(l) for l in length_scales]
        logger.info("  %-10s %s", "design", "  ".join(f"{c:>9}" for c in cols))
        for d, row in grid.items():
            logger.info("  %-10s %s", d,
                        "  ".join(f"{row.get(c, {}).get('cv', float('nan')):>9.4f}" for c in cols))
        for d, w in worst.items():
            logger.info("  worst case %-10s cv=%.4f at l_c=%s",
                        d, w["worst_cv"], w["attained_at_l_c"])
        if conservative is not None:
            logger.info("uniform model conservative for DESIGN: %s", conservative)
    return payload


def _summarize_curve(levels: list[dict], cfg, mode: str) -> dict:
    """Locate the peak of cv = sigma_C/mu_C and state how far the uniform limit
    sits below it -- the two numbers the figure exists to deliver."""
    correlated = [entry for entry in levels if entry["length_scale"] is not None]
    uniform = next((e for e in levels if e["length_scale"] is None), None)

    peak = max(correlated, key=lambda e: e["cv"]) if correlated else None
    interpretation = (
        "cv = sigma_C/mu_C versus l_c. The uniform entry is the l_c -> infinity "
        "limit (a scalar random eta) and is the Schevenels et al. (2011) "
        "control. If the peak of the correlated curve is not meaningfully above "
        "the uniform value, spatial correlation does not matter for this "
        "problem at any correlation length, and the single-l_c conclusion of "
        "the 2011 paper generalizes. If it is, the gap between the peak and the "
        "uniform limit is the quantity this project contributes -- and the "
        "production l_c should be the peak, not an arbitrary choice."
    )
    payload = {
        "mode": mode,
        "domain_axial": _L_AXIAL,
        "domain_transverse": _L_TRANSVERSE,
        "filter_radius": 0.6,
        "config_length_scale": cfg.random_field.length_scale,
        "n_evaluation_samples": int(cfg.mc_validation.n_samples),
        "levels": levels,
        "peak": peak,
        "uniform_limit": uniform,
        "interpretation": interpretation,
    }
    # Is the ENDPOINT contrast resolvable? That, not the level-by-level
    # ordering, is what the "uniform bounds the variance from above" claim
    # rests on -- and it is a much larger effect, so it can survive even if
    # adjacent levels overlap.
    if correlated and uniform is not None:
        shortest = min(correlated, key=lambda e: e["length_scale"])
        lo = (shortest.get("statistics") or {}).get("cv", {})
        hi = (uniform.get("statistics") or {}).get("cv", {})
        if lo.get("ci_high") is not None and hi.get("ci_low") is not None:
            payload["endpoint_contrast"] = {
                "shortest_l_c": shortest["length_scale"],
                "cv_shortest": lo.get("value"),
                "cv_shortest_ci": [lo.get("ci_low"), lo.get("ci_high")],
                "cv_uniform": hi.get("value"),
                "cv_uniform_ci": [hi.get("ci_low"), hi.get("ci_high")],
                "ratio": (hi.get("value") or float("nan")) / (lo.get("value") or float("nan")),
                "intervals_disjoint": bool(lo["ci_high"] < hi["ci_low"]),
                "note": (
                    "intervals_disjoint True means the uniform limit is "
                    "resolvably more variable than the shortest correlation "
                    "length. That is the bound claim. Level-by-level "
                    "monotonicity is a separate, weaker-powered question -- "
                    "check each level's cv interval before using the word."
                ),
            }

    if peak is not None and uniform is not None and uniform["cv"] > 0:
        excess = (peak["cv"] - uniform["cv"]) / uniform["cv"]
        # A peak at an ENDPOINT is not a peak: it means the curve is monotonic
        # over the range swept and there is no interior worst correlation
        # length. That is the opposite conclusion from an interior maximum and
        # must not be reported as one.
        l_c_values = [e["length_scale"] for e in correlated]
        peak_is_interior = (
            len(l_c_values) > 2
            and min(l_c_values) < peak["length_scale"] < max(l_c_values)
        )
        payload["peak_excess_over_uniform_relative"] = excess
        payload["peak_is_interior"] = peak_is_interior
        if comm.rank == 0:
            logger.info(
                "PEAK at l_c=%g (cv=%.5g); uniform limit cv=%.5g -> the worst "
                "correlation length is %+.3g%% above the uniform case.",
                peak["length_scale"], peak["cv"], uniform["cv"], 100 * excess,
            )
            if not peak_is_interior:
                logger.warning(
                    "That peak sits at an ENDPOINT of the swept range, so cv is "
                    "MONOTONIC here and there is no interior worst-case "
                    "correlation length. If the maximum is at the large-l_c end "
                    "the uniform case is the worst case, which SUPPORTS the "
                    "single-l_c conclusion of Schevenels et al. (2011) and "
                    "undercuts the premise that a spatially-correlated model is "
                    "needed. Widen L_C_LEVELS before concluding either way."
                )
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO if comm.rank == 0 else logging.ERROR)

    argv = sys.argv[1:]

    # Optional flags, pulled out before positional parsing.
    vthr: float | None = None
    only_levels: tuple | None = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--variance-threshold":
            vthr = float(argv[i + 1]); i += 2
        elif argv[i] == "--levels":
            only_levels = tuple(float(x) for x in argv[i + 1].split(",")); i += 2
        else:
            rest.append(argv[i]); i += 1

    configs = [a for a in rest if a.endswith((".yaml", ".yml"))]
    args = [a for a in rest if not a.endswith((".yaml", ".yml"))]
    cfg = load_config(configs[0] if configs else "src/config/configStudy.yaml")

    if not args:
        raise SystemExit(__doc__)
    mode = args[0]

    run_id = make_run_id(comm)
    manifest = RunManifest(run_id, comm)
    run_dir = OUTPUT_ROOT / run_id
    if comm.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    if mode == "fixed":
        payload = study_fixed(cfg, run_dir, manifest, only_levels=only_levels, vthr=vthr)
    elif mode == "reoptimize":
        requested = args[1:] or [str(cfg.random_field.length_scale), "uniform"]
        length_scales: list[float | None] = [
            None if a == "uniform" else float(a) for a in requested
        ]
        payload = study_reoptimize(cfg, run_dir, manifest, length_scales)
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected fixed|reoptimize")

    manifest.record(f"correlation_length_{mode}", payload)
    manifest.write(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
