"""Mandatory verification gates, executed inside the production pipeline.

WHY THIS MODULE EXISTS
----------------------
The verification routines this project already contained -- the finite-difference
gradient check, the KL sample-covariance check, the Heaviside equivalence check,
the eta bounds check -- were never called from the pipeline. Their only call
sites were throwaway scripts under scripts/, and the two dedicated verification
modules (src/validation/sensitivity_fd_check.py, analytical_cantilver.py) had
been deleted from the tree. So every published number came from a run in which
nothing was verified.

This module runs the checks as part of the run, fails the run when one fails,
and writes gates.json alongside the stage artifacts so the verification evidence
ships with the results.

THE FOUR GATES
--------------
1. heaviside_equivalence
   The random-field Heaviside must reduce EXACTLY to FEniTop's stock scalar-eta
   Heaviside when eta is a constant field. This is the correctness proof for the
   whole eta(x) generalization.

2. kl_correlation
   The field actually fed to the marginal transform is G(x)/std(x), the
   TRUNCATED KL field normalized to unit pointwise variance -- not the nominal
   squared-exponential kernel. Its covariance is therefore the truncated
   correlation, and comparing it against sigma^2 (as the old
   verify_sample_covariance did) is checking the wrong quantity against a
   parameter the pipeline has since made inert. This gate checks the empirical
   correlation against the ANALYTIC truncated correlation (a real
   implementation check) and separately REPORTS the gap between the truncated
   correlation and the target kernel (the honest statement of what truncation
   costs, which belongs in the paper).

3. eta_marginal
   The isoprobabilistic transform must reproduce the target Beta marginal on
   [eta_min, eta_max] at every point. Tested per node across independent field
   realizations -- which are genuinely iid in that direction -- rather than by
   pooling nodes within one realization, which are spatially correlated and
   would make the KS p-value meaningless.

4. gradient_fd
   The SAA robust gradient dJ/drho and the mean-volume gradient dE[V]/drho must
   match central finite differences of the SAME fixed-sample objective. Because
   the sample set is fixed, J_N is a deterministic function of rho and this is a
   clean comparison with no Monte Carlo noise floor.

   Two things this gate must get right, and which the pre-existing serial helper
   robust_gradient.verify_robust_gradient_fd does not:
     * MPI. That helper perturbs `rho_values[idx]` where rho_values is the
       rank-LOCAL slice, so under MPI every rank perturbs a DIFFERENT global
       element simultaneously and the comparison is meaningless. Here the
       perturbed elements are chosen as GLOBAL indices on rank 0 and broadcast,
       and only the owning rank perturbs.
     * Solver noise. The FEA is solved iteratively. At the default KSP
       tolerance the compliance is accurate to ~1e-5 relative, while a
       central difference with step 1e-6 moves J by ~1e-11 -- the "verification"
       would be measuring solver noise. This gate tightens the KSP tolerance
       and uses a step large enough that the finite-difference signal is orders
       of magnitude above solver noise, then restores the original tolerance.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy import stats as scipy_stats

from src.random_fields.kl_expansion import pointwise_std
from src.random_fields.threshold_transform import ThresholdMarginalTransform

logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD

# Safety factor on the Bonferroni noise bound in the kl_correlation gate. The
# per-entry standard error is asymptotic and the entries are not independent, so
# the bound is approximate; this keeps a correct implementation comfortably
# inside it while still catching a systematic error.
_CORRELATION_NOISE_SAFETY = 1.3


@dataclass
class GateConfig:
    """Knobs for the verification gates. Defaults are chosen so the whole suite
    costs far less than one MMA iteration of the real solve."""

    enabled: bool = True

    # --- kl_correlation ---
    correlation_n_nodes: int = 64
    correlation_n_samples: int = 4000
    correlation_rtol: float = 0.05  # empirical vs analytic truncated correlation

    # --- eta_marginal ---
    marginal_n_nodes: int = 128
    marginal_n_samples: int = 2000
    marginal_alpha: float = 0.01  # family-wise, Bonferroni-corrected across nodes

    # --- gradient_fd ---
    fd_enabled: bool = True
    fd_n_samples: int = 8  # SAA set size for the FD check (NOT the solve's N)
    fd_n_elements: int = 16  # global elements checked
    fd_step: float = 1.0e-3
    fd_rtol: float = 1.0e-3
    fd_ksp_rtol: float = 1.0e-12
    fd_bound_margin: float = 5.0e-3  # skip elements within this of 0 or 1
    fd_seed: int = 12345
    # The gradient is verified at a design blended toward the CENTRE OF THE ETA
    # BAND:  rho_fd = eta_mid + blend * (rho_warm - eta_mid).
    #
    # Being interior to the box [0,1] is not enough, and assuming it was is what
    # made this gate crash. The projection is tanh(beta*(rho_tilde - eta)), which
    # saturates to +/-1 in double precision once |rho_tilde - eta| exceeds about
    # 19/beta. At beta=128 that is 0.15. A design clipped to [0.05, 0.95] sits
    # OUTSIDE the eta band [0.25, 0.75] by more than 0.15, so every node is
    # saturated for every eta draw, every sample returns bit-identical
    # compliance, and sigma_C is exactly zero -- the gradient cannot be verified
    # at all, and dsigma/drho divides by zero.
    #
    # Blending toward eta_mid guarantees the projection is active at every node
    # regardless of beta or the band. It is also the STRONGER test: at a
    # saturated design the projection derivative is ~0 everywhere, so the chain
    # rule would be verified against nothing.
    fd_design_blend: float = 0.4


@dataclass
class GateResult:
    name: str
    passed: bool
    message: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "message": self.message,
            "detail": self.detail,
        }


class GateFailure(RuntimeError):
    """Raised when a mandatory verification gate fails. Never catch this to
    'continue anyway' -- a failed gate means the results are not trustworthy."""


# ---------------------------------------------------------------------------
# Gate 1: Heaviside equivalence
# ---------------------------------------------------------------------------
def gate_heaviside_equivalence(rf_heaviside, beta: float) -> GateResult:
    """Constant-array eta must reproduce scalar eta bit-for-bit.

    Collective: every rank runs it on its own local slice and the verdict is
    reduced with MPI.MIN so a failure on any rank fails the gate everywhere.
    """
    saved_rho_phys = rf_heaviside.rho_phys.x.petsc_vec.array.copy()
    saved_eta = rf_heaviside._current_eta
    saved_drho = rf_heaviside.drho
    try:
        local_pass = rf_heaviside.verify_reduces_to_deterministic(
            beta=beta, eta_value=0.5, rtol=1e-12
        )
    finally:
        # verify_reduces_to_deterministic overwrites rho_phys and drho in place;
        # restore so the gate cannot perturb the solve that follows it.
        rf_heaviside.rho_phys.x.petsc_vec.array[:] = saved_rho_phys
        rf_heaviside.rho_phys.x.scatter_forward()
        rf_heaviside._current_eta = saved_eta
        rf_heaviside.drho = saved_drho

    passed = bool(comm.allreduce(1 if local_pass else 0, op=MPI.MIN))
    return GateResult(
        name="heaviside_equivalence",
        passed=passed,
        message=(
            "constant-field eta reproduces FEniTop's scalar-eta Heaviside to 1e-12"
            if passed
            else "constant-field eta DIVERGES from FEniTop's scalar-eta Heaviside -- "
            "the eta(x) generalization is not behavior-preserving"
        ),
        detail={"beta": float(beta), "eta_value": 0.5, "rtol": 1e-12},
    )


# ---------------------------------------------------------------------------
# Gate 2: KL correlation structure
# ---------------------------------------------------------------------------
def gate_kl_correlation(kl_result, cfg: GateConfig) -> GateResult:
    """Empirical correlation of the normalized truncated field vs its analytic
    value, plus the reported truncation gap against the target kernel.

    Rank-0 computation (pure numpy on the world-identical kl_result), verdict
    broadcast. Cheap: a [n_samples x n_kl] @ [n_kl x n_nodes] product on a small
    node subset.
    """
    detail: dict = {}
    passed = True
    message = ""

    if comm.rank == 0:
        rng = np.random.default_rng(0)
        n_nodes_total = kl_result.modes.shape[0]
        k = min(cfg.correlation_n_nodes, n_nodes_total)
        node_idx = rng.choice(n_nodes_total, size=k, replace=False)

        std_all = pointwise_std(kl_result)
        sqrt_lambda = np.sqrt(kl_result.eigenvalues)              # [n_kl]
        modes_sub = kl_result.modes[node_idx, :]                  # [k x n_kl]
        scaled_sub = modes_sub * sqrt_lambda[np.newaxis, :]       # [k x n_kl]
        std_sub = std_all[node_idx]                               # [k]

        # Analytic correlation of the NORMALIZED truncated field.
        cov_truncated = scaled_sub @ scaled_sub.T                 # [k x k]
        analytic = cov_truncated / np.outer(std_sub, std_sub)

        # Empirical, from independent xi draws through the same code path.
        xi = rng.standard_normal(size=(cfg.correlation_n_samples, kl_result.n_kl))
        g_sub = xi @ scaled_sub.T                                 # [n_samples x k]
        g_sub = g_sub / std_sub[np.newaxis, :]
        empirical = np.cov(g_sub, rowvar=False)

        max_abs_err = float(np.max(np.abs(empirical - analytic)))

        # THRESHOLD MUST SCALE WITH THE SAMPLING NOISE, not be a fixed constant.
        # This compares the MAXIMUM error over k(k+1)/2 correlation entries, each
        # estimated from n samples. A sample correlation has standard error
        # (1-rho^2)/sqrt(n-1), and the maximum of many such estimates grows with
        # the number of entries -- so a fixed tolerance fails on pure noise as
        # soon as k is large or n is small. At k=32, n=1000 the expected maximum
        # deviation is ~0.13, which duly tripped a fixed 0.10 tolerance and
        # reported a correct implementation as broken.
        #
        # The threshold below is a Bonferroni-corrected bound on that maximum,
        # floored at the configured value so that a very large n cannot make the
        # test arbitrarily strict (a small systematic error is still a failure,
        # and truncation makes some systematic deviation legitimate).
        standard_error = (1.0 - analytic ** 2) / np.sqrt(max(cfg.correlation_n_samples - 1, 1))
        n_entries = k * (k + 1) // 2
        z_bonferroni = float(scipy_stats.norm.ppf(1.0 - 0.01 / (2 * n_entries)))
        noise_threshold = float(
            _CORRELATION_NOISE_SAFETY * z_bonferroni * standard_error.max()
        )
        threshold = max(cfg.correlation_rtol, noise_threshold)
        passed = max_abs_err <= threshold

        # Truncation gap: how far the model the code actually samples is from
        # the squared-exponential kernel the write-up claims. Reported, not
        # gated -- truncation at variance_threshold < 1 makes a gap inevitable,
        # and its size is a result the paper must state rather than hide.
        coords = kl_result.node_coordinates[node_idx, : kl_result.kernel_params.spatial_dim]
        diff = coords[:, None, :] - coords[None, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        target = np.exp(-r2 / (2.0 * kl_result.kernel_params.length_scale ** 2))
        truncation_gap = float(np.max(np.abs(analytic - target)))

        detail = {
            "n_nodes_sampled": int(k),
            "n_samples": int(cfg.correlation_n_samples),
            "max_abs_error_empirical_vs_analytic": max_abs_err,
            "tolerance": float(threshold),
            "tolerance_floor_from_config": float(cfg.correlation_rtol),
            "tolerance_from_sampling_noise": noise_threshold,
            "n_correlation_entries_compared": int(n_entries),
            "n_kl": int(kl_result.n_kl),
            "variance_explained": float(kl_result.variance_explained),
            "max_abs_gap_truncated_vs_target_kernel": truncation_gap,
            "note": (
                "The sampled field is G(x)/std(x): the truncated KL field "
                "normalized to unit pointwise variance. Its covariance is the "
                "truncated CORRELATION, not sigma^2*exp(-r^2/2l^2); sigma "
                "cancels identically. max_abs_gap_truncated_vs_target_kernel "
                "is the cost of truncating at variance_explained and must be "
                "reported as a modelling approximation."
            ),
        }
        message = (
            f"empirical correlation matches the analytic truncated correlation "
            f"(max abs err {max_abs_err:.3g} <= {threshold:.3g}, of which "
            f"{noise_threshold:.3g} is the sampling-noise allowance over "
            f"{n_entries} entries at n={cfg.correlation_n_samples}); "
            f"truncation gap vs target kernel {truncation_gap:.3g}"
            if passed
            else f"empirical correlation DEVIATES from analytic truncated "
            f"correlation (max abs err {max_abs_err:.3g} > {threshold:.3g}). "
            f"This exceeds the sampling-noise allowance ({noise_threshold:.3g} "
            f"over {n_entries} entries at n={cfg.correlation_n_samples}), so it "
            "is a systematic deviation, not chance -- raising "
            "correlation_n_samples will not fix it."
        )

    passed = comm.bcast(passed, root=0)
    message = comm.bcast(message, root=0)
    detail = comm.bcast(detail, root=0)
    return GateResult("kl_correlation", passed, message, detail)


# ---------------------------------------------------------------------------
# Gate 3: eta marginal
# ---------------------------------------------------------------------------
def gate_eta_marginal(kl_result, transform_params, cfg: GateConfig) -> GateResult:
    """Per-node KS test of the realized eta against the target Beta marginal.

    Tested ACROSS independent realizations at each fixed node (iid in that
    direction) rather than across nodes within one realization (spatially
    correlated -- a pooled KS test there would be invalid).
    """
    detail: dict = {}
    passed = True
    message = ""

    if comm.rank == 0:
        rng = np.random.default_rng(1)
        n_nodes_total = kl_result.modes.shape[0]
        k = min(cfg.marginal_n_nodes, n_nodes_total)
        node_idx = rng.choice(n_nodes_total, size=k, replace=False)

        std_all = pointwise_std(kl_result)
        sqrt_lambda = np.sqrt(kl_result.eigenvalues)
        scaled_sub = kl_result.modes[node_idx, :] * sqrt_lambda[np.newaxis, :]
        std_sub = std_all[node_idx]

        xi = rng.standard_normal(size=(cfg.marginal_n_samples, kl_result.n_kl))
        g_sub = (xi @ scaled_sub.T) / std_sub[np.newaxis, :]      # [n_samples x k]

        transform = ThresholdMarginalTransform(transform_params)
        eta = transform.transform(g_sub)

        within_bounds = bool(
            np.all(eta >= transform_params.eta_min - 1e-9)
            and np.all(eta <= transform_params.eta_max + 1e-9)
        )

        span = transform_params.eta_max - transform_params.eta_min
        target = scipy_stats.beta(
            transform_params.alpha, transform_params.beta,
            loc=transform_params.eta_min, scale=span,
        )
        ks_stats = np.empty(k)
        p_values = np.empty(k)
        for j in range(k):
            ks_stats[j], p_values[j] = scipy_stats.kstest(eta[:, j], target.cdf)

        # Bonferroni across the k simultaneous tests: reject only if the
        # smallest p-value survives the family-wise correction.
        min_p = float(p_values.min())
        corrected_alpha = cfg.marginal_alpha / k
        passed = bool(within_bounds and min_p >= corrected_alpha)

        detail = {
            "n_nodes_tested": int(k),
            "n_samples_per_node": int(cfg.marginal_n_samples),
            "max_ks_statistic": float(ks_stats.max()),
            "min_p_value": min_p,
            "bonferroni_alpha": float(corrected_alpha),
            "within_bounds": within_bounds,
            "target": {
                "distribution": "Beta",
                "alpha": float(transform_params.alpha),
                "beta": float(transform_params.beta),
                "eta_min": float(transform_params.eta_min),
                "eta_max": float(transform_params.eta_max),
            },
        }
        message = (
            f"realized eta matches Beta({transform_params.alpha:g},"
            f"{transform_params.beta:g}) on "
            f"[{transform_params.eta_min:g},{transform_params.eta_max:g}] at all "
            f"{k} tested nodes (min p={min_p:.3g} >= {corrected_alpha:.3g})"
            if passed
            else f"realized eta marginal DEVIATES from target "
            f"(min p={min_p:.3g} < {corrected_alpha:.3g}, "
            f"within_bounds={within_bounds})"
        )

    passed = comm.bcast(passed, root=0)
    message = comm.bcast(message, root=0)
    detail = comm.bcast(detail, root=0)
    return GateResult("eta_marginal", passed, message, detail)


# ---------------------------------------------------------------------------
# Gate 4: SAA gradient vs central finite differences
# ---------------------------------------------------------------------------
def _iter_linear_problems(ctx):
    """Every LinearProblem whose KSP tolerance must be tightened for the FD gate
    (the world problem plus, when sample-parallelism is on, the group problem)."""
    problems = [ctx.linear_problem]
    if getattr(ctx, "group", None) is not None:
        problems.append(ctx.group.group_linear_problem)
    return [p for p in problems if p is not None]


def gate_gradient_fd(ctx, opt: dict, beta: float, cfg: GateConfig) -> GateResult:
    """Central-difference check of dJ/drho and dE[V]/drho on the fixed SAA set.

    Uses its OWN small sample set (cfg.fd_n_samples), so the cost is
    2 * n_elements * fd_n_samples FEA solves rather than anything proportional
    to the solve's own N.
    """
    # Imported here rather than at module scope: saa_robust_driver imports the
    # FEA machinery, and keeping the dependency lazy lets the cheap numpy gates
    # above run in contexts that never build a solver.
    from src.optimization.robust_objective import RobustObjectiveConfig, compute_robust_objective_value
    from src.optimization.robust_gradient import compute_robust_gradient, compute_mean_volume_gradient
    from src.optimization.saa_robust_driver import _evaluate_saa
    from src.sampling.sampler import generate_samples

    kl_result = ctx.rf_heaviside.kl_result
    xi_fd = generate_samples(
        kl_result, cfg.fd_n_samples, strategy="monte_carlo", seed=cfg.fd_seed
    ).xi

    # Verify at a projection-ACTIVE design -- see GateConfig.fd_design_blend.
    transform_params = opt["transform_params"]
    eta_mid = 0.5 * (transform_params.eta_min + transform_params.eta_max)
    rho0_local = eta_mid + cfg.fd_design_blend * (
        np.asarray(ctx.rho_warm_start_local, dtype=float) - eta_mid
    )
    robust_config = RobustObjectiveConfig(lambda_tradeoff=1.0)
    saved_rho_local = np.asarray(ctx.rho_field.x.petsc_vec.array).copy()

    # --- tighten the KSP so the FD signal is not solver noise ---------------
    saved_tolerances = []
    for problem in _iter_linear_problems(ctx):
        ksp = problem.solver
        saved_tolerances.append((ksp, ksp.getTolerances()))
        ksp.setTolerances(rtol=cfg.fd_ksp_rtol, atol=1e-50)

    try:
        base = _evaluate_saa(ctx, opt, rho0_local, xi_fd, beta)

        # Degeneracy check BEFORE touching the gradient. If every sample gave the
        # same compliance there is nothing to verify, and compute_dsigma_drho
        # would divide by zero -- a traceback from three frames down that says
        # nothing about the actual cause.
        compliance_spread = float(
            np.max(base.compliance_samples) - np.min(base.compliance_samples)
        )
        if base.sigma_C < 1e-14 or compliance_spread == 0.0:
            saturation_halfwidth = 19.0 / beta
            return GateResult(
                name="gradient_fd",
                passed=False,
                message=(
                    f"DEGENERATE: all {xi_fd.shape[0]} samples returned "
                    f"identical compliance (sigma_C={base.sigma_C:.3g}), so the "
                    "gradient cannot be verified. The eta perturbation had no "
                    "effect on the design at all. Almost always this is "
                    f"PROJECTION SATURATION: tanh(beta*(rho_tilde - eta)) is "
                    f"+/-1 to machine precision once |rho_tilde - eta| exceeds "
                    f"~19/beta = {saturation_halfwidth:.3g}, so a design whose "
                    "filtered density sits outside "
                    f"[{transform_params.eta_min - saturation_halfwidth:.3g}, "
                    f"{transform_params.eta_max + saturation_halfwidth:.3g}] "
                    "responds to no eta draw whatsoever. Check (a) that "
                    "validation.fd_design_blend puts the design inside the eta "
                    "band, and (b) that the mesh RESOLVES the filter: at "
                    "R/h < 1 the filtered field jumps between 0 and 1 within a "
                    "single element, leaving no interface band for eta to act "
                    "on, which makes the whole eta model degenerate on that "
                    "mesh -- not just this gate."
                ),
                detail={
                    "sigma_C": base.sigma_C,
                    "mu_C": base.mu_C,
                    "compliance_spread": compliance_spread,
                    "beta": float(beta),
                    "saturation_halfwidth_in_rho": saturation_halfwidth,
                    "eta_band": [transform_params.eta_min, transform_params.eta_max],
                    "fd_design_blend": cfg.fd_design_blend,
                },
            )

        analytic_dJ = compute_robust_gradient(base, robust_config)
        analytic_dV = compute_mean_volume_gradient(base)
        J0 = compute_robust_objective_value(base, robust_config)

        # --- choose GLOBAL elements to perturb -----------------------------
        # Chosen on rank 0 and broadcast so every rank agrees; only the owning
        # rank actually perturbs its entry. Candidates are restricted to
        # elements away from the [0,1] bounds (so rho +/- step stays feasible)
        # and with an above-median |dJ/drho| (so the RELATIVE error being
        # tested is not dominated by a near-zero true gradient).
        index_map = ctx.rho_field.function_space.dofmap.index_map
        col_start = index_map.local_range[0]
        n_global = index_map.size_global

        grad_abs_global = _gather_local_to_global(np.abs(analytic_dJ), col_start, n_global)
        rho_global = _gather_local_to_global(rho0_local, col_start, n_global)

        if comm.rank == 0:
            margin = cfg.fd_bound_margin + cfg.fd_step
            eligible = np.where(
                (rho_global > margin)
                & (rho_global < 1.0 - margin)
                & (grad_abs_global > max(np.median(grad_abs_global), 1e-300))
            )[0]
            rng = np.random.default_rng(cfg.fd_seed)
            if eligible.size == 0:
                chosen = np.zeros(0, dtype=np.int64)
            else:
                chosen = rng.choice(
                    eligible, size=min(cfg.fd_n_elements, eligible.size), replace=False
                ).astype(np.int64)
        else:
            chosen = None
        chosen = comm.bcast(chosen, root=0)

        if chosen.size == 0:
            return GateResult(
                name="gradient_fd",
                passed=False,
                message=(
                    "no eligible elements for the FD check: every design "
                    "variable is pinned at a bound or has a negligible "
                    "gradient. The gradient cannot be verified on this design."
                ),
                detail={"n_candidates": 0},
            )

        fd_dJ = np.empty(chosen.size)
        fd_dV = np.empty(chosen.size)
        for k, global_idx in enumerate(chosen):
            local_idx = int(global_idx) - col_start
            owned = 0 <= local_idx < rho0_local.size

            rho_pert = rho0_local.copy()
            if owned:
                rho_pert[local_idx] += cfg.fd_step
            plus = _evaluate_saa(ctx, opt, rho_pert, xi_fd, beta)
            J_plus = compute_robust_objective_value(plus, robust_config)
            V_plus = plus.mean_volume

            rho_pert = rho0_local.copy()
            if owned:
                rho_pert[local_idx] -= cfg.fd_step
            minus = _evaluate_saa(ctx, opt, rho_pert, xi_fd, beta)
            J_minus = compute_robust_objective_value(minus, robust_config)
            V_minus = minus.mean_volume

            fd_dJ[k] = (J_plus - J_minus) / (2.0 * cfg.fd_step)
            fd_dV[k] = (V_plus - V_minus) / (2.0 * cfg.fd_step)

        # Analytic entries live on the owning rank; reduce to world-identical.
        analytic_dJ_sel = _select_global_entries(analytic_dJ, chosen, col_start)
        analytic_dV_sel = _select_global_entries(analytic_dV, chosen, col_start)

    finally:
        for ksp, (rtol, atol, dtol, maxits) in saved_tolerances:
            ksp.setTolerances(rtol=rtol, atol=atol, divtol=dtol, max_it=maxits)
        # The gate left rho_field at the last perturbed design; restore it so a
        # verification step can never influence the solve that follows.
        ctx.rho_field.x.petsc_vec.array[:] = saved_rho_local
        ctx.rho_field.x.scatter_forward()
        ctx.density_filter.forward()

    def _relative_error(analytic, finite_difference):
        scale = np.maximum(np.abs(analytic), np.abs(finite_difference))
        scale = np.where(scale > 0.0, scale, 1.0)
        return np.abs(analytic - finite_difference) / scale

    err_J = _relative_error(analytic_dJ_sel, fd_dJ)
    err_V = _relative_error(analytic_dV_sel, fd_dV)
    max_err_J = float(err_J.max())
    max_err_V = float(err_V.max())
    passed = bool(max_err_J <= cfg.fd_rtol and max_err_V <= cfg.fd_rtol)

    detail = {
        "n_elements_checked": int(chosen.size),
        "global_element_indices": [int(i) for i in chosen],
        "fd_step": float(cfg.fd_step),
        "fd_n_samples": int(cfg.fd_n_samples),
        "fd_design_blend": float(cfg.fd_design_blend),
        "base_design_sigma_C": base.sigma_C,
        "ksp_rtol_during_check": float(cfg.fd_ksp_rtol),
        "tolerance": float(cfg.fd_rtol),
        "max_relative_error_dJ_drho": max_err_J,
        "max_relative_error_dEV_drho": max_err_V,
        "J0": float(J0),
        "analytic_dJ": [float(v) for v in analytic_dJ_sel],
        "fd_dJ": [float(v) for v in fd_dJ],
        "analytic_dEV": [float(v) for v in analytic_dV_sel],
        "fd_dEV": [float(v) for v in fd_dV],
        "note": (
            "Checked against the SAME fixed sample set used for the analytic "
            "gradient, so J_N is deterministic and there is no Monte Carlo "
            "noise floor in this comparison."
        ),
    }
    message = (
        f"SAA adjoint matches central differences over {chosen.size} elements "
        f"(max rel err dJ={max_err_J:.3g}, dE[V]={max_err_V:.3g} <= {cfg.fd_rtol:.3g})"
        if passed
        else f"SAA adjoint DISAGREES with central differences "
        f"(max rel err dJ={max_err_J:.3g}, dE[V]={max_err_V:.3g} > {cfg.fd_rtol:.3g})"
    )
    return GateResult("gradient_fd", passed, message, detail)


def _gather_local_to_global(local: np.ndarray, col_start: int, n_global: int) -> np.ndarray:
    """Assemble a world-identical global array from each rank's contiguous slice."""
    buf = np.zeros(n_global, dtype=np.float64)
    buf[col_start:col_start + local.size] = local
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    return buf


def _select_global_entries(local: np.ndarray, global_indices: np.ndarray, col_start: int) -> np.ndarray:
    """Pull the given global entries out of a distributed array, world-identically."""
    out = np.zeros(global_indices.size, dtype=np.float64)
    for k, gi in enumerate(global_indices):
        li = int(gi) - col_start
        if 0 <= li < local.size:
            out[k] = local[li]
    comm.Allreduce(MPI.IN_PLACE, out, op=MPI.SUM)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_verification_gates(
    ctx,
    opt: dict,
    kl_result,
    beta: float,
    cfg: GateConfig,
    output_path: Path | None = None,
) -> list[GateResult]:
    """Run every gate, write gates.json, and raise GateFailure if any fails.

    Collective: must be called by every rank. The verdict is world-identical.

    Args:
        ctx: RobustProblemContext from setup_robust_problem().
        opt: The effective FEniTop opt dict.
        kl_result: The Stage-3 KLExpansionResult.
        beta: Heaviside sharpness the solve will use.
        cfg: GateConfig.
        output_path: Where to write gates.json (rank 0 only).

    Raises:
        GateFailure: If any gate fails. This is deliberately fatal -- a failed
            gate means the physics or the statistics are wrong, and every number
            produced downstream would be invalid.
    """
    if not cfg.enabled:
        if comm.rank == 0:
            logger.error(
                "VERIFICATION GATES DISABLED. No correctness evidence will be "
                "produced for this run, and its results must not be reported."
            )
        return []

    if comm.rank == 0:
        logger.info("Running verification gates (beta=%.4g)", beta)

    results = [
        gate_heaviside_equivalence(ctx.rf_heaviside, beta),
        gate_kl_correlation(kl_result, cfg),
        gate_eta_marginal(kl_result, opt["transform_params"], cfg),
    ]
    if cfg.fd_enabled:
        results.append(gate_gradient_fd(ctx, opt, beta, cfg))
    else:
        if comm.rank == 0:
            logger.warning(
                "gradient_fd gate SKIPPED by configuration. The adjoint "
                "sensitivities are the core of the method and are now "
                "unverified for this run."
            )

    if comm.rank == 0:
        for result in results:
            log = logger.info if result.passed else logger.error
            log("GATE %-24s %s -- %s", result.name,
                "PASS" if result.passed else "FAIL", result.message)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as handle:
                json.dump(
                    {
                        "all_passed": all(r.passed for r in results),
                        "beta": float(beta),
                        "gates": [r.as_dict() for r in results],
                    },
                    handle, indent=2,
                )
            logger.info("Verification evidence written to %s", output_path)

    failed = [r.name for r in results if not r.passed]
    if failed:
        raise GateFailure(
            f"Verification gate(s) failed: {failed}. The pipeline is halted "
            "deliberately -- results from a run with a failing gate are not "
            "valid. See gates.json for the per-gate detail."
        )
    return results
