"""How much over-etching does this design survive? -- the beta=128 defence.

THE OBJECTION THIS ANSWERS
--------------------------
Schevenels, Lazarov & Sigmund (CMAME 2011) cap their Heaviside sharpness at
beta = 32 and explicitly refuse 128 or 256, because at high beta the projection
produces "a structure with very fine features (a single element wide ...) which
is impossible to produce". Their filter was resolved at R/h = 8.4. This project
runs beta = 128 at R/h = 1.5 -- 5.6x less resolved -- so the objection lands
harder here, not softer, and it has to be answered with a measurement.

M_nd (Sigmund's measure of non-discreteness) does NOT answer it. M_nd = 0.23%
says the design is near-binary; a single-element-wide strut is also near-binary.
Discreteness and manufacturability are different properties.

WHAT IS MEASURED
----------------
The erosion depth the structure survives. Increasing the projection threshold
above eta_0 = 0.5 is exactly a morphological erosion of the solid phase: for the
Helmholtz filter the 1-D step response gives |grad rho_tilde| = 1/(2R) at the
interface, so a threshold shift of delta_eta retracts every boundary by

    epsilon(eta) = (eta - 0.5) * 2R

This sweeps eta upward and records the compliance of each eroded realization.
Thin members disappear first; when the load path is severed compliance diverges.
The largest erosion depth still carrying load is a direct, physical lower bound
on the load-bearing minimum half-thickness -- which is the quantity that decides
whether the part can actually be made.

WHAT IT DOES NOT MEASURE
------------------------
This is the LOAD-PATH-critical feature, not the global geometric minimum. A
non-structural whisker can vanish early without moving compliance at all. That
is arguably the more relevant number for a compliance-designed part, but the
distinction must be stated rather than glossed: report it as "survives erosion
to epsilon = X", not as "the minimum feature size is X".
"""
from __future__ import annotations

import logging

import numpy as np
from mpi4py import MPI

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD

# Compliance ratio, relative to the nominal eta=0.5 design, above which the
# structure is judged to have lost its load path. A severed beam's compliance
# rises by orders of magnitude, so this is not a sensitive threshold -- it just
# has to sit well above ordinary stiffness loss from uniform thinning.
_SEVERED_COMPLIANCE_RATIO = 50.0


def measure_erosion_robustness(
    ctx,
    opt: dict,
    beta: float,
    filter_radius: float,
    eta_max: float = 0.95,
    n_steps: int = 10,
) -> dict:
    """Sweep the projection threshold upward and find where the load path fails.

    The design must already be loaded into ctx.rho_field (the drivers leave it
    there; save_design_artifacts relies on the same thing).

    World-collective: every rank must call this together, because each threshold
    costs one FEA solve on the shared communicator.

    Args:
        ctx: RobustProblemContext holding the design to test.
        opt: FEniTop opt dict (unused directly; kept for call-site symmetry).
        beta: Heaviside sharpness the design was produced at -- erosion measured
            at a softer beta would describe a different structure.
        filter_radius: R, needed to convert a threshold shift into a length.
        eta_max: Highest threshold to try. Above ~0.95 the tanh saturates and
            the projection stops responding.
        n_steps: Number of thresholds between 0.5 and eta_max.

    Returns:
        dict with the eta sweep, the erosion depths in absolute and element
        units, the survival limit, and the implied minimum half-thickness.
    """
    thresholds = np.linspace(0.5, eta_max, n_steps + 1)
    element_size = float(ctx.fem.get("element_size", float("nan")))

    records: list[dict] = []
    nominal_compliance: float | None = None

    for eta in thresholds:
        ctx.rf_heaviside.forward(beta, eta=float(eta))
        try:
            ctx.linear_problem.solve_fem()
            # evaluate() -> (func_values, sensitivities) with
            # func_values = (C_value, V_value, _), matching fea_at_samples.py.
            func_values, _ = ctx.sens_problem.evaluate()
            compliance = float(func_values[0])
        except Exception as exc:  # a severed structure can fail to solve at all
            logger.info("erosion eta=%.4f: solve failed (%s) -- treating as severed",
                        eta, type(exc).__name__)
            compliance = float("inf")

        if not np.isfinite(compliance) or compliance <= 0.0:
            compliance = float("inf")
        if nominal_compliance is None:
            nominal_compliance = compliance

        offset = (float(eta) - 0.5) * 2.0 * filter_radius
        ratio = (
            compliance / nominal_compliance
            if nominal_compliance and np.isfinite(nominal_compliance) else float("inf")
        )
        records.append({
            "eta": float(eta),
            "erosion_depth": offset,
            "erosion_depth_elements": offset / element_size if element_size else None,
            "compliance": compliance if np.isfinite(compliance) else None,
            "compliance_ratio": ratio if np.isfinite(ratio) else None,
            "severed": bool(not np.isfinite(ratio) or ratio > _SEVERED_COMPLIANCE_RATIO),
        })

    survived = [r for r in records if not r["severed"]]
    limit = survived[-1] if survived else records[0]
    first_severed = next((r for r in records if r["severed"]), None)

    # Restore the nominal projection so the caller's field is not left eroded.
    ctx.rf_heaviside.forward(beta, eta=0.5)

    result = {
        "beta": float(beta),
        "filter_radius": float(filter_radius),
        "element_size": element_size,
        "min_feature_size_2R": 2.0 * float(filter_radius),
        "severed_compliance_ratio_threshold": _SEVERED_COMPLIANCE_RATIO,
        "sweep": records,
        "max_survived_erosion": limit["erosion_depth"],
        "max_survived_erosion_elements": limit["erosion_depth_elements"],
        "max_survived_eta": limit["eta"],
        "severed_at_eta": first_severed["eta"] if first_severed else None,
        # A member that survives erosion by epsilon has half-thickness >= epsilon,
        # so its full thickness is at least twice that.
        "implied_min_thickness": 2.0 * limit["erosion_depth"],
        "implied_min_thickness_elements": (
            2.0 * limit["erosion_depth_elements"]
            if limit["erosion_depth_elements"] is not None else None
        ),
        "survived_full_sweep": first_severed is None,
        "note": (
            "Load-path-critical, not global-geometric: a non-structural whisker "
            "can vanish without moving compliance. Report as 'survives erosion "
            "to epsilon = X', not as 'the minimum feature size is X'. If "
            "survived_full_sweep is True the design carried load at every "
            "threshold tried, and the limit is a lower bound only."
        ),
    }
    if comm.rank == 0:
        logger.info(
            "erosion robustness at beta=%g: survives to eta=%.3f "
            "(depth %.4g = %.2f elements); implied min thickness >= %.4g "
            "(%.0f%% of 2R=%.3g)",
            beta, limit["eta"], limit["erosion_depth"],
            limit["erosion_depth_elements"] or float("nan"),
            result["implied_min_thickness"],
            100.0 * result["implied_min_thickness"] / (2.0 * filter_radius),
            2.0 * filter_radius,
        )
    return result
