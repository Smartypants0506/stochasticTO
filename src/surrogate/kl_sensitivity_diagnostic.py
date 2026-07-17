"""
src/surrogate/kl_sensitivity_diagnostic.py

DIAGNOSTIC ONLY -- not part of the verified optimization path. Never wire
this into robust_objective.py/robust_gradient.py or feed its output into
a PCEGradientModel used by the optimizer; it deliberately bypasses
pce_builder.py's Q^2 >= threshold accuracy gate (Section 7's mandatory
verification gate), which exists precisely to prevent an under-fit
surrogate from driving the optimizer.

Purpose: when build_pce_surrogate() fails its Q^2 gate, we still want to
know WHY -- specifically, whether compliance variance is concentrated in
a handful of KL modes (a truncation problem, fixable by lowering n_kl,
which cuts required training samples roughly proportionally) or spread
evenly across all of them (a genuine smoothness/sample-count problem --
likely the sharp beta=128 Heaviside projection making C(xi) non-smooth --
that re-truncating n_kl will NOT fix).

A degree-1 (purely linear) PCE fit is the natural tool for this: it uses
exactly n_kl+1 coefficients (one constant + one linear term per mode), the
minimum possible for any PCE, so it is the fit LEAST likely to be
sample-starved -- if variance concentrates cleanly in a few modes even at
degree 1, that ranking is trustworthy regardless of whether the
production (higher-degree, gated) fit is failing for sample-count or
smoothness reasons.
"""
from __future__ import annotations

import logging

import numpy as np

from src.surrogate.pce_builder import PCEBuildResult, _fit_chaos_at_degree, _compute_q2
from src.surrogate.sobol import compute_sobol_indices, SobolReport

logger = logging.getLogger(__name__)

_DIAGNOSTIC_DEGREE = 1


def diagnose_kl_mode_sensitivity(
    xi_train: np.ndarray,
    c_train: np.ndarray,
    xi_test: np.ndarray,
    c_test: np.ndarray,
    hyperbolic_q: float = 0.75,
) -> SobolReport:
    """Fit a degree-1 PCE (bypassing the Q^2 gate) and report KL-mode Sobol ranking.

    Args:
        xi_train: [n_train x n_kl] KL coefficient training samples (reuse
            the exact same array the failed production fit used -- no new
            FEA solves needed).
        c_train: [n_train] compliance values matching xi_train (reuse
            training_data.compliance_samples from the failed run).
        xi_test: [n_test x n_kl] held-out KL coefficients.
        c_test: [n_test] held-out compliance values.
        hyperbolic_q: Same truncation exponent used by the production fit,
            for an apples-to-apples basis construction (irrelevant at
            degree 1 in practice, since hyperbolic truncation only prunes
            interaction/higher-order terms, but passed through for
            consistency).

    Returns:
        A SobolReport (see sobol.py) with first-order/total Sobol indices
        per KL mode, ranked and logged.

    Note:
        This function does NOT check q2 against any gate and does not
        raise on a low value -- that's the point. The degree-1 q2 is
        logged for context only.
    """
    n_kl = xi_train.shape[1]

    chaos_result = _fit_chaos_at_degree(xi_train, c_train, n_kl, _DIAGNOSTIC_DEGREE, hyperbolic_q)
    q2, rmse = _compute_q2(chaos_result, xi_test, c_test)

    logger.warning(
        "KL sensitivity diagnostic: degree-1 fit Q^2=%.4f, RMSE=%.5g "
        "(this fit is UNGATED and must never be used for optimization -- "
        "diagnostic ranking only)",
        q2, rmse,
    )

    pce_result = PCEBuildResult(
        chaos_result=chaos_result, q2=q2, degree=_DIAGNOSTIC_DEGREE, n_kl=n_kl, rmse_test=rmse,
    )
    report = compute_sobol_indices(pce_result)

    order = np.argsort(report.first_order)[::-1]
    ranked = [(int(i), float(report.first_order[i])) for i in order]

    logger.warning(
        "KL sensitivity diagnostic: ranked first-order indices (top 8 of %d): %s",
        n_kl, ranked[:8],
    )
    logger.warning(
        "KL sensitivity diagnostic: n_kl_effective=%d/%d modes needed to reach "
        "99%% cumulative first-order variance",
        report.n_kl_effective, n_kl,
    )

    if report.n_kl_effective <= max(3, n_kl // 3):
        logger.warning(
            "KL sensitivity diagnostic: variance is CONCENTRATED (%d/%d modes "
            "explain 99%% of first-order variance) -- re-truncating n_kl "
            "down to ~%d is a reasonable next step, would cut required "
            "training samples roughly proportionally.",
            report.n_kl_effective, n_kl, report.n_kl_effective,
        )
    else:
        logger.warning(
            "KL sensitivity diagnostic: variance is SPREAD across most modes "
            "(%d/%d needed for 99%%) -- re-truncating n_kl is unlikely to "
            "help; the Q^2 shortfall is more likely a smoothness/sample-count "
            "issue (e.g. the sharp beta=%s Heaviside projection making C(xi) "
            "non-smooth) than a truncation problem.",
            report.n_kl_effective, n_kl, "128",
        )

    return report


def select_active_modes(report: SobolReport, margin: int = 3) -> np.ndarray:
    """Turn a Sobol ranking into a concrete sorted array of active KL indices.

    Takes the top (n_kl_effective + margin) modes by first-order index. The
    margin exists because n_kl_effective is computed from a single degree-1
    fit on one training draw -- a small buffer guards against the ranking
    shifting slightly on a different sample set or at a higher fit degree,
    without giving back the dimensionality-reduction benefit (margin should
    stay small, e.g. 2-5, not doubled).

    Args:
        report: Output of diagnose_kl_mode_sensitivity (or
            sobol.compute_sobol_indices on any PCEBuildResult).
        margin: Extra modes to include beyond n_kl_effective, for safety.

    Returns:
        Sorted (ascending) array of KL mode indices into the FULL xi space
        -- ascending order so slicing xi[:, active_indices] is deterministic
        and reproducible across calls given the same report.
    """
    order = np.argsort(report.first_order)[::-1]
    n_keep = min(report.n_kl_effective + margin, report.n_kl)
    active = np.sort(order[:n_keep])
    logger.warning(
        "KL sensitivity diagnostic: selected %d/%d active modes for the "
        "production PCE fit (n_kl_effective=%d + margin=%d): %s",
        n_keep, report.n_kl, report.n_kl_effective, margin, active.tolist(),
    )
    return active