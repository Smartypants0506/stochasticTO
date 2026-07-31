"""Does the spatial correlation buy anything? -- the Schevenels et al. control.

    mpirun -n 64 python scripts/uniform_eta_baseline.py optimize uniform [cfg]
    mpirun -n 64 python scripts/uniform_eta_baseline.py optimize field   [cfg]
    mpirun -n 64 python scripts/uniform_eta_baseline.py evaluate uniform [cfg]
    mpirun -n 64 python scripts/uniform_eta_baseline.py evaluate field   [cfg]
    python scripts/uniform_eta_baseline.py report [cfg]          # rank-0, no MPI

THE QUESTION
------------
Schevenels, Lazarov & Sigmund (CMAME 200:3613-3627, 2011) model exactly this
project's uncertainty -- a spatially-correlated random Heaviside projection
threshold, memoryless-transformed from an underlying Gaussian field -- and
report, for BOTH of their test problems:

    "the design obtained assuming uniform manufacturing errors is equally
     robust with respect to non-uniform errors, and vice versa"

Their gripper: uniform-optimized design (m=-1.817, sigma=0.035) vs
non-uniform-optimized (m=-1.821, sigma=0.036) under non-uniform error --
indistinguishable. Their heat sink: the UNIFORM-optimized design is slightly
BETTER under non-uniform error than the non-uniform-optimized one.

If that carries over to 3D compliance, then the KL expansion, the 512 FEA
solves per iteration and the ~142 h production run buy nothing over a scalar
random eta at ~64 samples, and this project has no result. Nothing currently in
the repository can answer that: the erode/dilate driver is the Wang/Lazarov/
Sigmund WORST-CASE three-point scheme, which is a different control.

WHY IT MIGHT NOT CARRY OVER (the hypothesis under test)
-------------------------------------------------------
Their explanation for the heat sink's REDUCED sigma under spatial variation is
cancellation: "dilations occur in some regions ... erosions in other regions.
The total amount of material does not change as strongly." That argument needs
the QoI to be roughly additive over the domain. Heat transfer under a
distributed source is. Compliance of a slender loaded beam is NOT -- it is a
series / weakest-link quantity, so a correlated erosion at one cross-section is
not compensated by a dilation elsewhere. If that is right, 3D beam compliance is
precisely the case where spatial correlation SHOULD matter and their negative
result SHOULD fail. That is this project's actual thesis, and this script is the
experiment that decides it.

WHAT MAKES THE COMPARISON FAIR
------------------------------
The uniform arm is the SAME code path with a degenerate one-mode KL expansion
whose eigenfunction is constant (src/study_support.py::build_uniform_eta_kl).
Because the projection standardizes G(x) by its pointwise std before the
marginal transform, both arms see an EXACT Beta(alpha,beta) marginal on
[eta_min, eta_max]; the only difference between them is whether eta varies in
space. Same driver, same beta continuation, same optimality test, same volume
constraint, same evaluation ensemble.

WHY IT RUNS AS SEPARATE INVOCATIONS
-----------------------------------
Each arm needs its own RobustProblemContext, and setup_robust_problem() splits
COMM_WORLD and builds a per-group mesh. Holding two of those live doubles the
FEA memory footprint for no reason, so each step runs in its own process and
hands off through .npy files in a shared, fixed directory. Each step writes its
own provenance manifest.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.config.loader import load_config

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output") / "studies" / "uniform_eta"

# Samples for the uniform arm's SAA loop. eta is a single scalar there, so the
# stochastic dimension is 1 and far fewer samples resolve the same marginal than
# the field arm's n_kl ~ 20 needs. Deliberately generous for a 1-D quadrature --
# the arm still costs ~1/8 of a field run.
N_UNIFORM_SAA = 64

# Fallback for the decision rule if no completed SAA gap study is on disk. This
# is the run-to-run sigma variability of the method itself: any difference
# between two designs smaller than this is the seed, not the formulation.
DEFAULT_RUN_TO_RUN_SIGMA_VARIABILITY = 0.059

DESIGNS = ("nominal", "uniform", "field")


def _samples_path(error_model: str, design: str) -> Path:
    return OUTPUT_ROOT / f"samples_under_{error_model}__{design}.npy"


def _load_run_to_run_variability() -> tuple[float, str]:
    """Measured run-to-run sigma spread from the newest completed SAA gap study.

    Returns (threshold, description, diagnostics).

    WHY THIS IS NOT JUST `run_to_run_sigma_variability_relative`
    ------------------------------------------------------------
    That field is std/mean over the replications, which summarizes a Gaussian.
    The measured distribution is not Gaussian: across the first seven
    replications six out-of-sample sigmas cluster within ~7% and one sits at
    2.47x the median. std/mean of that sample is 48.9%, driven entirely by the
    outlier, while the robust spread is 7.2%. Using 48.9% as the decision
    threshold would declare essentially any result unresolvable; using it as
    "the typical run-to-run spread" would misdescribe six of seven runs.

    So the threshold returned is the ROBUST spread (1.4826*MAD/median, the
    MAD-based estimator scaled to match sigma for a normal), and the raw value
    and the outlier count are returned alongside it. Both belong in the paper:
    the robust number characterizes a typical run, and the outlier rate is
    itself a headline result -- roughly one SAA solve in seven produces a design
    2.5x less robust than its siblings, and its IN-SAMPLE statistics give no
    warning (rep 6 ranked mid-pack in-sample). A single-solve comparison
    therefore carries a real chance of drawing the outlier, which is why the
    verdict this feeds is provisional unless replicated.
    """
    candidates = sorted(Path("output/studies/saa_gap").glob("*/saa_gap.json"))
    for path in reversed(candidates):
        try:
            with open(path) as handle:
                payload = json.load(handle)
            sigmas = np.array(
                [float(r["out_of_sample_sigma_C"]) for r in payload["replications"]],
                dtype=float,
            )
        except (OSError, KeyError, ValueError, TypeError):
            continue
        if sigmas.size < 3:
            continue
        median = float(np.median(sigmas))
        mad = float(np.median(np.abs(sigmas - median)))
        robust = 1.4826 * mad / median if median > 0 else float("nan")
        raw = float(sigmas.std(ddof=1) / sigmas.mean())
        # "Outlier" here means beyond 3 robust standard deviations of the
        # median -- the same yardstick the threshold itself uses.
        outliers = int(np.sum(np.abs(sigmas - median) > 3.0 * 1.4826 * mad)) if mad > 0 else 0
        return robust, str(path), {
            "n_replications": int(sigmas.size),
            "robust_spread_mad_based": robust,
            "raw_spread_std_over_mean": raw,
            "median_sigma_C": median,
            "max_over_median": float(sigmas.max() / median) if median > 0 else None,
            "n_outliers_beyond_3_robust_sd": outliers,
            "outlier_rate": outliers / sigmas.size,
            "note": (
                "threshold = robust (MAD-based) spread. raw std/mean is reported "
                "because it is what saa_gap.json's headline field contains and "
                "it differs a lot when the distribution is heavy-tailed."
            ),
        }
    return (
        DEFAULT_RUN_TO_RUN_SIGMA_VARIABILITY,
        f"default ({DEFAULT_RUN_TO_RUN_SIGMA_VARIABILITY:.3g}); no completed gap study found",
        {},
    )


# --------------------------------------------------------------------------
# MPI stages
# --------------------------------------------------------------------------

def _build(cfg, arm: str):
    """Mesh + nominal warm start + the arm's expansion + robust context.

    The nominal SIMP solve is repeated in every invocation rather than cached.
    It is deterministic given the config, and a cache keyed on anything less
    than the full effective configuration is the exact failure mode
    (`recompute:` in the deleted main.py) that let a run reuse artifacts from a
    different problem.
    """
    from mpi4py import MPI

    from src.fenitop.topopt import topopt
    from src.meshing.box_source import build_box_fenitop_dicts
    from src.random_fields.kl_expansion import build_uniform_eta_kl
    from src.study_support import build_stage3_kl, setup_context

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    rho_nominal = topopt(
        fem, opt, load_cases, output_prefix=str(OUTPUT_ROOT / "nominal_")
    )
    rho_nominal = comm.bcast(rho_nominal, root=0)

    kl_field = build_stage3_kl(cfg, tagged_mesh, comm)
    kl_result = build_uniform_eta_kl(kl_field) if arm == "uniform" else kl_field
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)
    return comm, ctx, opt, kl_result, kl_field, rho_nominal


def stage_optimize(cfg, arm: str) -> None:
    """Optimize one arm and save its design."""
    from src.optimization.saa_robust_driver import run_saa_robust_topopt
    from src.provenance import RunManifest, make_run_id
    from src.sampling.sampler import generate_samples

    comm, ctx, opt, kl_result, _, rho_nominal = _build(cfg, arm)
    manifest = RunManifest(make_run_id(comm), comm)
    manifest.record_config(cfg, effective_fem=ctx.fem, effective_opt=opt)

    n_field = int(cfg.optimization.saa_n_samples)
    # Never let the control cost more than the treatment: on a small config
    # (configSmoke has saa_n_samples=16) a fixed 64 would make the cheap arm the
    # expensive one and invert the cost ratio the comparison is meant to report.
    n_saa = min(N_UNIFORM_SAA, n_field) if arm == "uniform" else n_field
    lam = float(cfg.optimization.lambda_sweep[-1])

    xi = generate_samples(
        kl_result, n_saa,
        strategy=cfg.optimization.saa_sampling_strategy,
        seed=cfg.optimization.saa_seed,
    ).xi

    if comm.rank == 0:
        logger.info(
            "=== arm=%s: n_kl=%d, N=%d, lambda=%.3g ===",
            arm, kl_result.n_kl, n_saa, lam,
        )
    with manifest.stage(f"optimize_{arm}"):
        solved = run_saa_robust_topopt(ctx, opt, lam, xi)

    payload = {
        "arm": arm,
        "n_kl": int(kl_result.n_kl),
        "saa_n_samples": n_saa,
        "lambda": lam,
        "in_sample_mu_C": solved["mu_C"],
        "in_sample_sigma_C": solved["sigma_C"],
        "mean_volume": solved["mean_volume"],
        "volume_violation": solved["volume_violation"],
        "M_nd_percent": solved["M_nd_percent"],
        "converged": solved["converged"],
        "optimality": solved["optimality"],
        "beta_schedule": solved["beta_schedule"],
        "fea_solves_total": int(solved["n_fea_batches_total"]) * n_saa,
        "note": (
            "in_sample statistics are computed on the arm's OWN sample set and "
            "under its OWN error model; they are NOT comparable across arms. "
            "Only the evaluate/report stages produce comparable numbers."
        ),
    }
    if comm.rank == 0:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_ROOT / f"rho_{arm}.npy", np.asarray(solved["rho_robust"]))
        np.save(OUTPUT_ROOT / "rho_nominal.npy", np.asarray(rho_nominal))
        with open(OUTPUT_ROOT / f"optimize_{arm}.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
    manifest.record(f"optimize_{arm}", payload)
    manifest.write(OUTPUT_ROOT / f"manifest_optimize_{arm}.json")


def stage_evaluate(cfg, error_model: str) -> None:
    """Evaluate every available design under ONE error model, on a common
    ensemble with common random numbers so the designs are paired."""
    from src.optimization.saa_robust_driver import _evaluate_saa
    from src.provenance import RunManifest, make_run_id

    comm, ctx, opt, kl_result, _, rho_nominal = _build(cfg, error_model)
    manifest = RunManifest(make_run_id(comm), comm)
    manifest.record_config(cfg, effective_fem=ctx.fem, effective_opt=opt)

    beta_max = float(cfg.optimization.saa_beta_max)
    n_mc = int(cfg.mc_validation.n_samples)
    # Same seed for both error models: the ensembles are different populations
    # (n_kl differs), so they are not paired ACROSS models -- only across
    # designs WITHIN a model, which is what the comparison needs.
    xi_common = np.random.default_rng(cfg.mc_validation.seed).standard_normal(
        size=(n_mc, kl_result.n_kl)
    )

    designs: dict[str, np.ndarray] = {"nominal": np.asarray(rho_nominal)}
    for arm in ("uniform", "field"):
        path = OUTPUT_ROOT / f"rho_{arm}.npy"
        if path.exists():
            designs[arm] = np.load(path)
        elif comm.rank == 0:
            logger.warning("%s not found -- run `optimize %s` first", path, arm)

    summary = {}
    for name, rho_global in designs.items():
        ctx.warm_start_comm.bcast(ctx.rho_field, np.asarray(rho_global))
        result = _evaluate_saa(
            ctx, opt, ctx.rho_field.x.petsc_vec.array.copy(), xi_common,
            beta_max, accumulate_gradients=True,
        )
        summary[name] = {"mu_C": result.mu_C, "sigma_C": result.sigma_C}
        if comm.rank == 0:
            np.save(_samples_path(error_model, name), result.compliance_samples)
            logger.info(
                "under %s error, design=%s: mu_C=%.6g sigma_C=%.6g (cv=%.4g)",
                error_model, name, result.mu_C, result.sigma_C,
                result.sigma_C / result.mu_C,
            )

    payload = {
        "error_model": error_model,
        "n_kl": int(kl_result.n_kl),
        "n_evaluation_samples": n_mc,
        "beta": beta_max,
        "common_random_numbers": True,
        "designs": summary,
    }
    if comm.rank == 0:
        with open(OUTPUT_ROOT / f"evaluate_under_{error_model}.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
    manifest.record(f"evaluate_under_{error_model}", payload)
    manifest.write(OUTPUT_ROOT / f"manifest_evaluate_{error_model}.json")


# --------------------------------------------------------------------------
# Report (serial)
# --------------------------------------------------------------------------

def stage_report(cfg) -> None:
    """The 2x2 table, its confidence intervals, and the decision rule."""
    from src.validation.statistics import compare_designs, summarize_samples

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        n_bootstrap=cfg.mc_validation.n_bootstrap,
        confidence=cfg.mc_validation.confidence,
        seed=cfg.mc_validation.bootstrap_seed,
    )

    samples: dict[tuple[str, str], np.ndarray] = {}
    for error_model in ("uniform", "field"):
        for design in DESIGNS:
            path = _samples_path(error_model, design)
            if path.exists():
                samples[(error_model, design)] = np.load(path)

    missing = [
        f"{e}/{d}" for e in ("uniform", "field") for d in DESIGNS
        if (e, d) not in samples
    ]
    if ("field", "uniform") not in samples or ("field", "field") not in samples:
        raise SystemExit(
            "Cannot report: the decisive cells (both designs under the "
            "spatially-correlated error model) are missing. Run the optimize "
            f"and evaluate stages first. Missing: {missing}"
        )

    table = {
        f"{error_model}_error": {
            design: summarize_samples(array, seed=cfg.mc_validation.bootstrap_seed)
            for (error_model_key, design), array in samples.items()
            if error_model_key == error_model
        }
        for error_model in ("uniform", "field")
    }

    # The decisive comparison: both designs, under the spatially-correlated
    # error model that the project's whole apparatus exists to represent.
    decisive = compare_designs(
        samples[("field", "uniform")], samples[("field", "field")],
        name_a="uniform_optimized", name_b="field_optimized", paired=True, **kwargs,
    )

    # Schevenels' reciprocal check: does the field-optimized design also hold up
    # under uniform error? Their finding was that both directions are a wash.
    reciprocal = None
    if ("uniform", "uniform") in samples and ("uniform", "field") in samples:
        reciprocal = compare_designs(
            samples[("uniform", "uniform")], samples[("uniform", "field")],
            name_a="uniform_optimized", name_b="field_optimized", paired=True, **kwargs,
        ).as_dict()

    sigma_uniform = float(np.std(samples[("field", "uniform")], ddof=1))
    sigma_field = float(np.std(samples[("field", "field")], ddof=1))
    relative_gain = (sigma_uniform - sigma_field) / sigma_uniform

    noise_floor, noise_source, noise_diagnostics = _load_run_to_run_variability()
    exceeds_noise = abs(relative_gain) > noise_floor
    resolvable = bool(decisive.std_difference_resolvable)
    significant = bool(exceeds_noise and resolvable)
    # Three outcomes, not two. "The field arm is significantly WORSE" is a
    # distinct finding from "the two are indistinguishable" and must not be
    # collapsed into it: it means either the field arm is under-converged or
    # optimizing against the correlated model actively costs robustness, and
    # both demand investigation rather than a write-up.
    if significant and relative_gain > 0:
        verdict = "spatial_correlation_matters"
    elif significant:
        verdict = "field_arm_significantly_worse"
    else:
        verdict = "indistinguishable"
    spatial_correlation_matters = verdict == "spatial_correlation_matters"

    cost = {}
    for arm in ("uniform", "field"):
        path = OUTPUT_ROOT / f"optimize_{arm}.json"
        if path.exists():
            with open(path) as handle:
                cost[arm] = json.load(handle).get("fea_solves_total")
    cost_ratio = (
        cost["field"] / cost["uniform"]
        if cost.get("field") and cost.get("uniform") else None
    )

    payload = {
        "eta_band": [cfg.random_field.eta_min, cfg.random_field.eta_max],
        "correlation_length": cfg.random_field.length_scale,
        "n_evaluation_samples": int(cfg.mc_validation.n_samples),
        "table_2x2": table,
        "decisive_comparison_under_correlated_error": decisive.as_dict(),
        "reciprocal_comparison_under_uniform_error": reciprocal,
        "sigma_C_under_correlated_error": {
            "uniform_optimized": sigma_uniform,
            "field_optimized": sigma_field,
            "relative_reduction_from_field_arm": relative_gain,
        },
        "fea_solves": {**cost, "field_over_uniform_ratio": cost_ratio},
        "decision_rule": {
            "run_to_run_sigma_variability": noise_floor,
            "run_to_run_source": noise_source,
            "run_to_run_diagnostics": noise_diagnostics,
            "exceeds_run_to_run_noise": exceeds_noise,
            "resolvable_at_this_n": resolvable,
            "verdict": verdict,
            "spatial_correlation_matters": spatial_correlation_matters,
            "rule": (
                "The field arm is judged to buy something only if its sigma_C "
                "under correlated error is lower than the uniform arm's by MORE "
                "than the method's own run-to-run variability AND the paired "
                "bootstrap CI on the difference excludes zero. Anything less "
                "reproduces Schevenels et al. (2011) in 3D, which is a "
                "publishable result but a different one -- and it means the "
                "spatially-correlated production run should be re-scoped."
            ),
        },
        "missing_cells": missing,
    }
    with open(OUTPUT_ROOT / "uniform_eta_comparison.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=str)

    logger.info("2x2 sigma_C table (rows = error model, cols = design):")
    for error_model in ("uniform", "field"):
        row = table.get(f"{error_model}_error", {})
        cells = "  ".join(
            f"{d}={row[d]['std']['value']:.6g}" for d in DESIGNS if d in row
        )
        logger.info("  %-8s error : %s", error_model, cells)
    logger.info("DECISIVE: %s", decisive.verdict())
    logger.info(
        "sigma_C under correlated error: uniform-optimized %.6g vs "
        "field-optimized %.6g (%.3g%% reduction); run-to-run noise floor "
        "%.3g%% [%s]",
        sigma_uniform, sigma_field, 100 * relative_gain, 100 * noise_floor,
        noise_source,
    )
    if noise_diagnostics:
        logger.info(
            "noise floor detail: robust(MAD) %.3g%% vs raw std/mean %.3g%% over "
            "%d replications; worst/median %.2fx; %d outlier(s) beyond 3 robust "
            "SD (rate %.0f%%). The robust value is the threshold.",
            100 * noise_diagnostics["robust_spread_mad_based"],
            100 * noise_diagnostics["raw_spread_std_over_mean"],
            noise_diagnostics["n_replications"],
            noise_diagnostics["max_over_median"] or float("nan"),
            noise_diagnostics["n_outliers_beyond_3_robust_sd"],
            100 * noise_diagnostics["outlier_rate"],
        )
        if noise_diagnostics["n_outliers_beyond_3_robust_sd"]:
            logger.warning(
                "The replication set is HEAVY-TAILED. A single-solve comparison "
                "like this one can draw the outlier, so treat the verdict below "
                "as provisional unless it is replicated across seeds."
            )

    cost_phrase = (
        f"{cost_ratio:.1f}x the FEA solves" if cost_ratio
        else "its extra FEA cost"
    )
    if verdict == "spatial_correlation_matters":
        logger.info(
            "VERDICT: the spatially-correlated model earns %s. The field arm "
            "reduces sigma_C by %.3g%%, beyond both the run-to-run noise floor "
            "and the paired CI.", cost_phrase, 100 * relative_gain,
        )
    elif verdict == "field_arm_significantly_worse":
        logger.error(
            "VERDICT: the field arm is significantly WORSE than a scalar random "
            "eta (sigma_C %.3g%% HIGHER, CI excludes zero) while costing %s. "
            "This is not the Schevenels null result -- it is an anomaly. Check "
            "first that the field arm actually converged (a max-iter stop makes "
            "this exact signature) and that both arms ran the same beta "
            "schedule and iteration budget, before drawing any conclusion.",
            -100 * relative_gain, cost_phrase,
        )
    else:
        logger.warning(
            "VERDICT: the field arm is NOT distinguishable from a scalar random "
            "eta (difference %.3g%%, noise floor %.3g%%). This REPRODUCES "
            "Schevenels, Lazarov & Sigmund (2011) in 3D compliance and must be "
            "reported as such. Do not launch the spatially-correlated "
            "production run on the current premise: the KL expansion and %s "
            "are unjustified by this evidence.",
            100 * relative_gain, 100 * noise_floor, cost_phrase,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = [a for a in sys.argv[1:] if not a.endswith((".yaml", ".yml"))]
    configs = [a for a in sys.argv[1:] if a.endswith((".yaml", ".yml"))]
    config_path = configs[0] if configs else "src/config/configStudy.yaml"

    if not args:
        raise SystemExit(__doc__)
    mode = args[0]

    if mode == "report":
        stage_report(load_config(config_path))
        return

    from mpi4py import MPI
    logging.getLogger().setLevel(
        logging.INFO if MPI.COMM_WORLD.rank == 0 else logging.ERROR
    )
    if len(args) < 2 or args[1] not in ("uniform", "field"):
        raise SystemExit(f"`{mode}` needs an arm: uniform | field")
    arm = args[1]
    cfg = load_config(config_path)

    if mode == "optimize":
        stage_optimize(cfg, arm)
    elif mode == "evaluate":
        stage_evaluate(cfg, arm)
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected optimize|evaluate|report")


if __name__ == "__main__":
    main()
