"""The first-order optimality measure, against problems with known KKT points.

The defect this replaces was a convergence test of ||grad f|| <= gatol on a
volume-constrained problem, where grad f is NOT zero at the optimum. So the
tests that matter are: the measure must be ~0 at a genuine KKT point of a
CONSTRAINED problem (where the old test would report a large residual), and it
must be large at a non-optimal point (where the old test could be small).
"""
from __future__ import annotations

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py", reason="needs PETSc")
from petsc4py import PETSc  # noqa: E402

from src.optimization.optimality import compute_first_order_optimality  # noqa: E402

pytestmark = pytest.mark.dolfinx


def _vec(values):
    v = PETSc.Vec().createSeq(len(values))
    v.setArray(np.asarray(values, dtype=float))
    return v


def _row_matrix(row):
    row = np.asarray(row, dtype=float)
    m = PETSc.Mat().createDense((1, row.size))
    m.setUp()
    m.setValues([0], list(range(row.size)), row.reshape(1, -1))
    m.assemble()
    return m


def test_stationarity_is_zero_at_a_constrained_kkt_point():
    """min sum x_i^2 s.t. sum x_i >= 1 has optimum x_i = 1/n with multiplier
    2/n. grad f = 2x = 2/n per component -- NONZERO, which is exactly why the
    old ||grad f|| test could never fire. The Lagrangian gradient IS zero."""
    n = 4
    x = _vec([1.0 / n] * n)
    gradient = _vec([2.0 / n] * n)          # grad of sum x^2
    lb, ub = _vec([0.0] * n), _vec([1.0] * n)
    # g(x) = 1 - sum x <= 0, so dg/dx = -1 per component.
    constraint = _vec([0.0])                 # active
    jacobian = _row_matrix([-1.0] * n)
    multipliers = _vec([2.0 / n])

    result = compute_first_order_optimality(
        x, gradient, lb, ub, constraint, jacobian, multipliers, x_previous=x,
    )

    assert result.stationarity == pytest.approx(0.0, abs=1e-12)
    assert result.stationarity_rel == pytest.approx(0.0, abs=1e-12)
    assert result.feasibility == pytest.approx(0.0, abs=1e-12)
    assert result.complementarity == pytest.approx(0.0, abs=1e-12)
    assert result.satisfied(1e-8, 1e-8)
    # The quantity the old code reported as "the KKT residual" is large here.
    assert result.grad_norm > 0.9


def test_non_optimal_point_is_reported_as_non_optimal():
    n = 4
    x = _vec([0.5] * n)
    gradient = _vec([1.0] * n)
    lb, ub = _vec([0.0] * n), _vec([1.0] * n)
    constraint = _vec([0.5])                 # violated: g > 0
    jacobian = _row_matrix([-1.0] * n)
    multipliers = _vec([0.0])

    result = compute_first_order_optimality(
        x, gradient, lb, ub, constraint, jacobian, multipliers,
    )
    assert result.stationarity_rel == pytest.approx(1.0, rel=1e-9)
    assert result.feasibility == pytest.approx(0.5)
    assert not result.satisfied(1e-3, 1e-3)


def test_infeasible_but_stationary_point_does_not_pass():
    """The reason all three conditions are required: a point can have zero
    stationarity residual and still violate its constraint."""
    n = 3
    x = _vec([1.0] * n)
    gradient = _vec([0.0] * n)               # stationarity trivially satisfied
    lb, ub = _vec([0.0] * n), _vec([1.0] * n)
    constraint = _vec([0.3])                 # infeasible
    jacobian = _row_matrix([1.0] * n)
    multipliers = _vec([0.0])

    result = compute_first_order_optimality(
        x, gradient, lb, ub, constraint, jacobian, multipliers,
    )
    assert result.stationarity == pytest.approx(0.0, abs=1e-14)
    assert not result.satisfied(1e-8, 1e-8)


def test_gradient_pushing_into_an_active_bound_is_stationary():
    """A variable pinned at a bound with the gradient pushing further into it is
    stationary (the bound multiplier absorbs it); pushing back into the interior
    is not."""
    lb, ub = _vec([0.0, 0.0]), _vec([1.0, 1.0])

    into = compute_first_order_optimality(
        _vec([0.0, 1.0]), _vec([+5.0, -5.0]), lb, ub,
    )
    assert into.stationarity == pytest.approx(0.0, abs=1e-14)

    outward = compute_first_order_optimality(
        _vec([0.0, 1.0]), _vec([-5.0, +5.0]), lb, ub,
    )
    assert outward.stationarity == pytest.approx(5.0)


def test_constraint_scales_normalize_the_feasibility_residual():
    """A volume overshoot should read as a FRACTION of the budget."""
    lb, ub = _vec([0.0]), _vec([1.0])
    result = compute_first_order_optimality(
        _vec([0.5]), _vec([1.0]), lb, ub,
        constraint_vec=_vec([0.008]),         # 10% over a 0.08 budget
        jacobian=_row_matrix([1.0]),
        multipliers=_vec([0.0]),
        constraint_scales=(0.08,),
    )
    assert result.feasibility == pytest.approx(0.1)


def test_design_change_is_reported():
    lb, ub = _vec([0.0, 0.0]), _vec([1.0, 1.0])
    result = compute_first_order_optimality(
        _vec([0.5, 0.5]), _vec([0.0, 0.0]), lb, ub,
        x_previous=_vec([0.5, 0.2]),
    )
    assert result.design_change == pytest.approx(0.3)
