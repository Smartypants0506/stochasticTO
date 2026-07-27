"""First-order (KKT) optimality measures for bound- and inequality-constrained
topology optimization.

WHY THIS MODULE EXISTS
----------------------
The previous convergence test in src/fenitop/mma.py was

    ||grad f(x)|| <= gatol

and the value reported downstream as "kkt_residual" was that same raw
objective-gradient norm. Neither is an optimality measure for this problem:

  * The design is bound constrained (rho in [0, 1]) and volume constrained
    (E[V] <= vol_frac). At a KKT point the RAW objective gradient is NOT zero --
    it is balanced by the volume multiplier and the active bound multipliers.
    grad f only vanishes at an unconstrained stationary point, which a
    compliance-minimization-under-volume problem never has (compliance strictly
    decreases with added material). So the test could never fire, TAO always ran
    to its iteration cap, and the run was nevertheless reported as finished.
  * Reporting ||grad f|| as "the KKT residual" therefore reported a quantity
    that is expected to be non-zero at the solution, and whose magnitude tracks
    the objective's units rather than the distance to optimality.

What is computed here instead, at a point x with multipliers mu >= 0 for the
inequality constraints g(x) <= 0:

    Lagrangian gradient      grad_L = grad f + sum_j mu_j * grad g_j
    stationarity             ||P(grad_L)||_inf, where P is the projection onto
                             the directions in which x can still move inside
                             [lb, ub] (the standard active-set projected
                             gradient: components pinned at a bound only count
                             when the gradient pushes further INTO that bound)
    stationarity_rel         stationarity / ||grad f||_inf -- scale free, O(1)
                             at a poor iterate, -> 0 at a KKT point. This is the
                             quantity to converge on and to report.
    feasibility              max_j max(0, g_j) / scale_j
    complementarity          max_j |mu_j * g_j| / scale_j
    design_change            ||x_k - x_{k-1}||_inf

All three of stationarity, feasibility and complementarity must be small for a
point to be first-order optimal; reporting only one of them is what allowed a
non-converged, slightly-infeasible design to be presented as a solution.

Everything is MPI-collective and returns world-identical scalars: the design
vector is distributed, so the inf-norms are global MAX allreduces, not per-rank
maxima.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

logger = logging.getLogger(__name__)

# A design variable within this distance of a bound counts as sitting ON it.
# MMA projects onto [alpha, beta] which it clips to [lb, ub], so converged
# solid/void elements land exactly on the bound; the tolerance only guards
# against round-off in that projection.
_BOUND_ACTIVE_TOL = 1e-8

# Floor for the relative-stationarity denominator, so a vanishing objective
# gradient cannot manufacture a spurious "converged" verdict by division.
_GRAD_SCALE_FLOOR = 1e-300


@dataclass(frozen=True)
class FirstOrderOptimality:
    """First-order optimality diagnostics at one design iterate.

    Attributes:
        stationarity: ||P(grad_L)||_inf, absolute. Carries the objective's
            units, so it is only comparable across runs of the SAME problem.
        stationarity_rel: stationarity / ||grad f||_inf. Dimensionless and
            scale free -- this is the number to converge on and to publish.
        feasibility: max_j max(0, g_j) / scale_j. Zero when every inequality
            constraint holds. Positive values are constraint VIOLATION, in
            units of the constraint's own scale (e.g. a fraction of vol_frac).
        complementarity: max_j |mu_j * g_j| / scale_j. Zero when every
            constraint is either exactly active or has a zero multiplier.
        design_change: ||x_k - x_{k-1}||_inf, the move actually taken by the
            previous step. Reported because MMA can stall against its move
            limit with a still-large stationarity residual, and the two
            together distinguish "converged" from "stuck".
        grad_norm: ||grad f||_2. DIAGNOSTIC ONLY -- retained because it is what
            the old code reported as "kkt_residual", so runs stay comparable,
            but it is not an optimality measure. See module docstring.
        grad_inf: ||grad f||_inf, the denominator used for stationarity_rel.
        multipliers: The inequality multipliers mu_j used, in the g <= 0 sign
            convention (mu_j >= 0).
        constraints: The constraint values g_j, same convention (feasible <= 0).
    """

    stationarity: float
    stationarity_rel: float
    feasibility: float
    complementarity: float
    design_change: float
    grad_norm: float
    grad_inf: float
    multipliers: tuple[float, ...] = ()
    constraints: tuple[float, ...] = ()

    def satisfied(self, stationarity_tol: float, constraint_tol: float) -> bool:
        """True only when ALL THREE first-order conditions hold.

        Args:
            stationarity_tol: Threshold on stationarity_rel (dimensionless).
            constraint_tol: Threshold on both feasibility and complementarity.
        """
        return (
            self.stationarity_rel <= stationarity_tol
            and self.feasibility <= constraint_tol
            and self.complementarity <= constraint_tol
        )

    def as_dict(self) -> dict:
        """JSON-serializable form for pareto_results.json / the run manifest."""
        return {
            "stationarity": self.stationarity,
            "stationarity_rel": self.stationarity_rel,
            "feasibility": self.feasibility,
            "complementarity": self.complementarity,
            "design_change": self.design_change,
            "grad_norm_diagnostic_only": self.grad_norm,
            "grad_inf_diagnostic_only": self.grad_inf,
            "multipliers": list(self.multipliers),
            "constraints": list(self.constraints),
        }

    def summary(self) -> str:
        return (
            f"stat_rel={self.stationarity_rel:.4g} "
            f"(abs={self.stationarity:.4g}) feas={self.feasibility:.4g} "
            f"comp={self.complementarity:.4g} dx={self.design_change:.4g} "
            f"[|grad f|={self.grad_norm:.4g}, diagnostic only]"
        )


def _gather_global_vector(vec: PETSc.Vec | None) -> np.ndarray:
    """Materialize a small distributed PETSc Vec identically on every rank.

    Used for the constraint and multiplier vectors, which have global size equal
    to the number of constraints (1 here) but are laid out with all entries on
    rank 0. Every rank needs the values to compute world-identical scalars.
    """
    if vec is None:
        return np.zeros(0)
    n_global = vec.getSize()
    buf = np.zeros(n_global, dtype=np.float64)
    lo, hi = vec.getOwnershipRange()
    if hi > lo:
        buf[lo:hi] = np.asarray(vec.getArray(readonly=True), dtype=np.float64)
    comm = vec.getComm().tompi4py()
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    return buf


def compute_first_order_optimality(
    x: PETSc.Vec,
    gradient: PETSc.Vec,
    lb: PETSc.Vec,
    ub: PETSc.Vec,
    constraint_vec: PETSc.Vec | None = None,
    jacobian: PETSc.Mat | None = None,
    multipliers: PETSc.Vec | None = None,
    x_previous: PETSc.Vec | None = None,
    constraint_scales: tuple[float, ...] | None = None,
    work_vec: PETSc.Vec | None = None,
) -> FirstOrderOptimality:
    """Compute the first-order optimality diagnostics at x. MPI-collective.

    Args:
        x: Current design iterate (distributed).
        gradient: grad f(x), same layout as x.
        lb, ub: Variable bounds, same layout as x.
        constraint_vec: g(x) in the g <= 0 convention, or None for an
            unconstrained-except-bounds problem.
        jacobian: dg/dx as an [n_constraints x n_design] matrix in the SAME
            sign convention as constraint_vec.
        multipliers: mu >= 0, same layout as constraint_vec.
        x_previous: Previous iterate, for the design-change measure. Skipped
            (reported as NaN) when None.
        constraint_scales: Per-constraint normalizers for feasibility and
            complementarity, e.g. (vol_frac,) so a violation reads as a
            fraction of the budget rather than an absolute volume. Defaults to
            all ones.
        work_vec: Optional scratch vector matching x's layout, to avoid an
            allocation per iteration.

    Returns:
        A FirstOrderOptimality with world-identical scalars on every rank.

    Notes:
        The sign convention MUST be g <= 0 with mu >= 0. src/fenitop/mma.py
        flips TAO's h(x) >= 0 form into this convention before calling here;
        passing the unflipped values would silently invert the feasibility and
        complementarity measures.
    """
    comm = x.getComm().tompi4py()

    # --- Lagrangian gradient: grad f + J^T mu -------------------------------
    grad_L = work_vec if work_vec is not None else x.duplicate()
    gradient.copy(grad_L)
    if jacobian is not None and multipliers is not None:
        contribution = x.duplicate()
        # multTranspose handles a MATTRANSPOSE wrapper natively, so no special
        # casing is needed for the transposed Jacobian TAO may hand back.
        jacobian.multTranspose(multipliers, contribution)
        grad_L += contribution
        contribution.destroy()

    # --- active-set projected gradient --------------------------------------
    # A component pinned at a bound is stationary if the gradient pushes it
    # further into that bound (the bound multiplier absorbs it); it is NOT
    # stationary if the gradient would move it back into the interior.
    x_arr = np.asarray(x.getArray(readonly=True))
    gL_arr = np.asarray(grad_L.getArray(readonly=True))
    lb_arr = np.asarray(lb.getArray(readonly=True))
    ub_arr = np.asarray(ub.getArray(readonly=True))

    projected = gL_arr.copy()
    at_lower = x_arr <= lb_arr + _BOUND_ACTIVE_TOL
    at_upper = x_arr >= ub_arr - _BOUND_ACTIVE_TOL
    projected[at_lower] = np.minimum(projected[at_lower], 0.0)
    projected[at_upper] = np.maximum(projected[at_upper], 0.0)

    local_stationarity = float(np.abs(projected).max()) if projected.size else 0.0
    stationarity = float(comm.allreduce(local_stationarity, op=MPI.MAX))

    grad_inf = float(gradient.norm(PETSc.NormType.INFINITY))
    grad_norm = float(gradient.norm())
    stationarity_rel = stationarity / max(grad_inf, _GRAD_SCALE_FLOOR)

    # --- feasibility and complementarity ------------------------------------
    g_values = _gather_global_vector(constraint_vec)
    mu_values = _gather_global_vector(multipliers)
    n_g = g_values.size

    if constraint_scales is None:
        scales = np.ones(n_g)
    else:
        scales = np.abs(np.asarray(constraint_scales, dtype=float))
        if scales.size != n_g:
            raise ValueError(
                f"constraint_scales has {scales.size} entries but there are "
                f"{n_g} constraints."
            )
        scales = np.where(scales > 0.0, scales, 1.0)

    if n_g:
        feasibility = float(np.max(np.maximum(g_values, 0.0) / scales))
        # mu is only meaningful when it has been solved for; a shorter mu array
        # (no dual yet) is treated as mu = 0, which makes complementarity 0 and
        # correctly prevents a "converged" verdict from the stationarity term
        # alone on the very first iteration, since feasibility still applies.
        mu_aligned = mu_values if mu_values.size == n_g else np.zeros(n_g)
        complementarity = float(np.max(np.abs(mu_aligned * g_values) / scales))
    else:
        feasibility = 0.0
        complementarity = 0.0

    # --- design change -------------------------------------------------------
    if x_previous is not None:
        prev_arr = np.asarray(x_previous.getArray(readonly=True))
        local_change = float(np.abs(x_arr - prev_arr).max()) if x_arr.size else 0.0
        design_change = float(comm.allreduce(local_change, op=MPI.MAX))
    else:
        design_change = float("nan")

    if work_vec is None:
        grad_L.destroy()

    return FirstOrderOptimality(
        stationarity=stationarity,
        stationarity_rel=stationarity_rel,
        feasibility=feasibility,
        complementarity=complementarity,
        design_change=design_change,
        grad_norm=grad_norm,
        grad_inf=grad_inf,
        multipliers=tuple(float(v) for v in mu_values),
        constraints=tuple(float(v) for v in g_values),
    )
