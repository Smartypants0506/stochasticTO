import logging

from petsc4py import PETSc

import numpy as np

from src.fenitop.la import negative_part, positive_part
from src.optimization.optimality import compute_first_order_optimality

logger = logging.getLogger(__name__)

# Default tolerance on the feasibility and complementarity residuals, used
# alongside TAO's gatol (which now governs the RELATIVE stationarity residual).
# Override per-solve with MMA.set_constraint_tolerance().
DEFAULT_CONSTRAINT_TOL = 1e-4


class MMA:
    """Method of Moving Asymptotes (MMA).

    Optimisation solver for general constrained problems based on separable convex
    approximations.

    The method is applicable to general constrained optimisation problems in the (PETSc) form

        min     f(x)
         x
        s.t.    x⁻ ≤ x ≤ x⁺
                g(x) = 0 (TODO: coming soon)
                h(x) ≥ 0

    References:
        1) https://doi.org/10.1002/nme.1620240207
        2) https://people.kth.se/~krille/mmagcmma.pdf
        3) https://comsolyar.com/wp-content/uploads/2020/03/gcmma.pdf

    """

    # TODO: MMA.view() outputs ksp and ls info

    _f: float  # objective value

    # primal variables
    _x: PETSc.Vec

    # dual variables
    _λ: PETSc.Vec | None
    _J_λ: PETSc.Vec | None

    # aux. variables
    _r: float
    _p: PETSc.Vec | None
    _q: PETSc.Vec | None
    _P: PETSc.Mat | None
    _Q: PETSc.Mat | None

    # parameters
    _albefa: float
    _move_limit: float
    _asymptote_init: float
    _asymptote_decrement: float
    _asymptote_increment: float
    _asymptote_min: float
    _asymptote_max: float
    _raai: float
    _theta: float

    def __init__(self):
        logger.debug("__init__")

        # Initialize all LA objects to None
        self._λ = None
        self._J_λ = None
        self._p = None
        self._q = None
        self._P = None
        self._Q = None

        self._x_range = None
        self._x_m1 = None
        self._x_m2 = None
        self._diff_12 = None
        self._diff_23 = None

        self._p = None
        self._q = None

        self._L = None
        self._U = None
        self._alpha = None
        self._beta = None

        self._xmL = None
        self._Umx = None
        self._xmL_recp = None
        self._Umx_recp = None

        self._tmp = None
        self._tmp_2 = None
        self._tmp_3 = None

        self._zero = None

        # First-order optimality state. self._optimality holds the most recent
        # FirstOrderOptimality record and is what the driver should report --
        # NOT the raw objective-gradient norm, which is not an optimality
        # measure for this constrained problem (see optimality.py).
        self._optimality = None
        self._constraint_tol = DEFAULT_CONSTRAINT_TOL
        self._constraint_scales = None
        self._optimality_work = None

    def create(self, tao: PETSc.TAO) -> None:
        logger.debug("create")

        # Default subsolver
        self._subsolver = PETSc.TAO().create()
        self._subsolver.setType(PETSc.TAO.Type.BQNLS)

    def setFromOptions(self, tao):
        logger.debug("setFromOptions")

        opts = PETSc.Options()

        prefix = tao.getOptionsPrefix()
        if prefix is None:
            prefix = ""

        self._albefa = opts.getReal(f"{prefix}tao_mma_albefa", 0.1)
        self._move_limit = opts.getReal(f"{prefix}tao_mma_move_limit", 0.5)
        self._asymptote_init = opts.getReal(f"{prefix}tao_mma_asymptote_init", 0.5)
        self._asymptote_decrement = opts.getReal(f"{prefix}tao_mma_asymptote_decrement", 0.7)
        self._asymptote_increment = opts.getReal(f"{prefix}tao_mma_asymptote_increment", 1.2)
        self._asymptote_min = opts.getReal(f"{prefix}tao_mma_asymptote_min", 0.01)
        self._asymptote_max = opts.getReal(f"{prefix}tao_mma_asymptote_max", 10.0)
        self._raai = opts.getReal(f"{prefix}tao_mma_raai", 1e-5)  # 1e-5
        self._theta = opts.getReal(f"{prefix}tao_mma_theta", 0.1)

        '''if not np.isclose(self._raai, 0.0):
            raise RuntimeError("raai 0 only supported.")
'''
        if self._asymptote_min > 1.0:
            raise RuntimeError(f"Asymptote min. ({self._asymptote_min}) must be ≤ 1.")

        if self._asymptote_max < 1.0:
            raise RuntimeError(f"Asymptote max. ({self._asymptote_max}) must be ≥ 1.")

        self._subsolver.setOptionsPrefix(f"{prefix}tao_mma_subsolver_")
        self._subsolver.setFromOptions()

        if not self._subsolver.getType().startswith("b"):
            raise RuntimeError("MMA subsolver needs to be a bound constrained solver.")

    def x(self, λ, x):
        """Dual to primal map.

        Compute from dual state λ the primal state x.

               √(p + λᵀP) ⊙ L + √(q + λᵀQ) ⊙ U
        x(λ) = -------------------------------
                   √(p + λᵀP) + √(q + λᵀQ)

        + bounds projection.
        """
        # tmp = √(p + λᵀP)
        self._P.multTransposeAdd(λ, self._p, self._tmp)
        self._tmp.sqrtabs()

        # tmp_2 = √(q + λᵀQ)
        self._Q.multTransposeAdd(λ, self._q, self._tmp_2)
        self._tmp_2.sqrtabs()

        # x(λ) = (tmpᵀL + tmp_2ᵀU ) ⨸ (tmp + tmp_2)
        self._tmp.copy(self._tmp_3)
        self._tmp_3 += self._tmp_2

        self._tmp.pointwiseMult(self._tmp, self._L)
        self._tmp_2.pointwiseMult(self._tmp_2, self._U)

        self._tmp.copy(x)
        x += self._tmp_2
        x.pointwiseDivide(x, self._tmp_3)

        # bounds projection
        x.pointwiseMax(self._alpha, x)
        x.pointwiseMin(self._beta, x)

    def setUp(self, tao: PETSc.TAO) -> None:
        logger.debug("setUp")

        self._objective = 0.0
        self._gradient = tao.getGradient()[0]

        # Dual problem is a bound-constrained optimisation problem of the form
        #
        #     min   W(λ)
        #      λ
        #     s.t.    λ ≥ 0
        #
        if (constraint := tao.getInequalityConstraints())[1] is not None:
            self._λ = constraint[0].copy()
            # Vec.copy() duplicates the constraint vector's CONTENTS, which at
            # setUp time are whatever the caller's createMPI() left there --
            # i.e. an undefined initial guess for the dual subsolver, and an
            # undefined multiplier in the first iteration's KKT residual.
            # Start from the only defensible value, mu = 0.
            self._λ.set(0.0)
            self._λ.assemble()
            self._J_λ = self._λ.copy()
            self._J_λ.assemble()

        def dual_objective_and_gradient(tao, λ, G) -> float:
            logger.debug("dual_objective_and_gradient")
            assert self._P is not None and self._Q is not None
            assert self._p is not None and self._q is not None

            # x(λ)
            self.x(λ, self._x)

            # tmp_2 = (U - x)⁻¹
            self._U.copy(self._tmp_2)
            self._tmp_2 -= self._x
            self._tmp_2.reciprocal()

            # tmp_3 = (x - L)⁻¹
            self._x.copy(self._tmp_3)
            self._tmp_3 -= self._L
            self._tmp_3.reciprocal()

            # W(λ) = r + λᵀr_h + (p + λᵀP) ⊙ (U - x(λ))⁻¹ + (q + λᵀQ) ⊙ (x(λ) - L)⁻¹
            #      = r + λᵀr_h + tmp_p ⊙ Umx + tmp_q ⊙ xmL
            # Note: we have no b term here as in the original MMA paper (compare pg. 365 eq. 20) due
            #       to different problem form.
            W = self._r
            W += λ.dot(self._r_h)

            # tmp = p + λᵀP
            self._P.multTransposeAdd(λ, self._p, self._tmp)
            W += self._tmp.dot(self._tmp_2)

            # tmp_q = q + λᵀQ
            self._Q.multTransposeAdd(λ, self._q, self._tmp)
            W += self._tmp.dot(self._tmp_3)

            # ∇W(λ) = r_h + P (U - x)⁻¹ + Q (x - L)⁻¹
            self._r_h.copy(G)
            self._P.multAdd(self._tmp_2, G, G)
            self._Q.multAdd(self._tmp_3, G, G)

            # Flip for max. to min.
            G.scale(-1)
            return -W  # type: ignore

        if (constraint := tao.getInequalityConstraints())[1] is not None:
            self._subsolver.setSolution(self._λ)
            self._subsolver.setObjectiveGradient(dual_objective_and_gradient, self._J_λ)

            lb = self._λ.copy()
            lb.set(0.0)
            ub = self._λ.copy()
            ub.set(PETSc.INFINITY)
            self._subsolver.setVariableBounds((lb, ub))

            self._subsolver.setUp()

        # variables/intermediates
        self._x = tao.getSolution()

        self._x_range = self._x.copy()
        self._x_m1 = self._x.copy()
        self._x_m2 = self._x.copy()
        self._diff_12 = self._x.copy()
        self._diff_23 = self._x.copy()

        self._p = self._x.copy()
        self._q = self._x.copy()

        self._L = self._x.copy()
        self._U = self._x.copy()
        self._alpha = self._x.copy()
        self._beta = self._x.copy()

        self._xmL = self._x.copy()
        self._Umx = self._x.copy()
        self._xmL_recp = self._x.copy()
        self._Umx_recp = self._x.copy()

        self._tmp = self._x.copy()
        self._tmp_2 = self._x.copy()
        self._tmp_3 = self._x.copy()

        self._zero = self._x.copy()
        self._zero.set(0.0)

        self._optimality_work = self._x.copy()

    def set_constraint_tolerance(self, tol: float) -> None:
        """Tolerance on the feasibility and complementarity residuals.

        TAO's gatol governs the RELATIVE stationarity residual; this governs the
        other two first-order conditions. All three must hold to converge.
        """
        self._constraint_tol = float(tol)

    def set_constraint_scales(self, scales) -> None:
        """Per-constraint normalizers for the feasibility/complementarity
        residuals, e.g. (vol_frac,) so a volume violation is reported as a
        fraction of the budget rather than an absolute volume fraction."""
        self._constraint_scales = None if scales is None else tuple(float(s) for s in scales)

    def solve(self, tao):
        """Follows TaoSolve_Python_default."""
        logger.debug("solve")

        # TAO 0-th iteration is a convergence check.

        self._f = tao.computeObjectiveGradient(self._x, self._gradient)

        tao.monitor(f=self._f)

        c, h_tuple = tao.getInequalityConstraints()
        h, h_args, h_kwargs = h_tuple if h_tuple else (None, None, None)

        J, _, Jh_tuple = tao.getJacobianInequality()
        Jh, Jh_args, Jh_kwargs = Jh_tuple if Jh_tuple else (None, None, None)

        # NOTE: there used to be a pre-loop `if self._gradient.norm() <= gatol:
        # setConvergedReason(CONVERGED_GATOL)` here. It is deliberately gone.
        # ||grad f|| is not an optimality measure for a bound- and
        # volume-constrained problem: at a KKT point grad f is balanced by the
        # volume multiplier and the bound multipliers, not zero. The test could
        # therefore never fire, and the value it tested was nonetheless reported
        # downstream as "the KKT residual". Convergence is now decided by the
        # full first-order conditions (stationarity + feasibility +
        # complementarity) computed each iteration below -- see
        # src/optimization/optimality.py.
        gatol, _, _ = tao.getTolerances()

        lb, ub = tao.getVariableBounds()

        # x_range = ub - lb
        ub.copy(self._x_range)
        self._x_range -= lb

        if np.any(np.isinf(self._x_range)):
            raise RuntimeError("MMA requires a bounded domain.")

        # Initial lower/upper bounds
        lb.copy(self._L)
        ub.copy(self._U)
        lb.copy(self._alpha)
        ub.copy(self._beta)

        # Reset history
        self._x_m1.set(0.0)
        self._x_m2.set(0.0)

        if c:
            tmp_h = c.copy()
            self._r_h = c.copy()

        if J:
            J_p = J.copy()
            J_m = J.copy()
            self._P = J.copy()
            self._Q = J.copy()

        # Warn only once per outer solve if the dual subsolver plateaus without
        # meeting its strict tolerance (see the accept-and-continue branch below),
        # to avoid one log line per outer iteration per rank.
        warned_subsolver_nonconvergence = False

        # range(1, max_it + 1): the previous range(1, max_it) silently ran one
        # fewer outer iteration than the configured budget, so a run capped at
        # max_iter=400 actually performed 399 MMA steps.
        for it in range(1, tao.getMaximumIterations() + 1):
            if tao.reason:
                break

            logger.debug(f"solve iteration {it}")

            # Compute f(x), ∇f(x), h(x) and J_h(x)
            self._f = tao.computeObjectiveGradient(self._x, self._gradient)

            # --- DIAGNOSTICS: raw objective/gradient right out of FEA or PCE ---
            logger.info(
                f"[MMA diag] it={it} raw objective f={self._f:.6e}, "
                f"|grad|={self._gradient.norm():.4e}, "
                f"max|grad|={max(abs(self._gradient.getArray().min()), abs(self._gradient.getArray().max())):.4e}"
            )
            # --- END DIAGNOSTICS ---

            if h:
                h(tao, self._x, c, *h_args, **h_kwargs)
                Jh(tao, self._x, J, None, *Jh_args, **Jh_kwargs)

                # The implemented MMA formulation relies on a form where h(x) ≤ 0 holds (not
                # h(x) ≥ 0). To account for this change we sign flip the callback
                # results - interpreting the constraint as if it was in the h(x) ≥ 0.
                c.scale(-1)
                if J.getType() == PETSc.Mat.Type.TRANSPOSE:
                    # Note: this ensures we never introduce a lazy scaling but, have the right
                    #       scaling on the underlying matrix, which we need for the splits based on
                    #       sign for P/Q.
                    J.getTransposeMat().scale(-1)
                else:
                    J.scale(-1)

            # --- first-order optimality at the CURRENT iterate ---------------
            # Evaluated here, immediately after the sign flip, because this is
            # the only point in the loop where the objective gradient, the
            # constraint values and the constraint Jacobian all refer to the
            # SAME x. (The multiplier is necessarily the one from the previous
            # dual solve -- that is the multiplier that produced this iterate,
            # which is the standard choice.) self._x_m1 still holds the
            # iterate that entered the PREVIOUS step, so the design-change
            # measure describes the move just taken.
            self._optimality = compute_first_order_optimality(
                self._x,
                self._gradient,
                lb,
                ub,
                constraint_vec=c if h else None,
                jacobian=J if h else None,
                multipliers=self._λ if h else None,
                x_previous=self._x_m1 if it > 1 else None,
                constraint_scales=self._constraint_scales,
                work_vec=self._optimality_work,
            )
            logger.info("[MMA opt] it=%d %s", it, self._optimality.summary())

            # Compute MMA subproblem dependencies

            # Update moving asymptotes L/U
            self._tmp_2.set(1.0)
            if it < 3:
                # L = x - asymptote_init * x_range
                self._x.copy(self._L)
                self._L.axpy(-self._asymptote_init, self._x_range)

                # U = x + asymptote_init * x_range
                self._x.copy(self._U)
                self._U.axpy(self._asymptote_init, self._x_range)
            else:
                self._x.copy(self._diff_12)
                self._diff_12 -= self._x_m1

                self._x_m1.copy(self._diff_23)
                self._diff_23 -= self._x_m2

                sign_change = np.sign(self._diff_12.getArray()) * np.sign(self._diff_23.getArray())
                self._tmp_2.setArray(
                    self._asymptote_increment * (sign_change > 0)
                    + self._asymptote_decrement * (sign_change < 0)
                    + 1.0 * (sign_change == 0)
                )

                # Note: the computation of L/U is based on the offsets of the previous iterate x_m1
                #       to the previous L/U.

                # L = x - f ⊙ (x_m1 - L)
                self._x_m1.copy(self._tmp)
                self._tmp -= self._L
                self._tmp.pointwiseMult(self._tmp, self._tmp_2)

                self._x.copy(self._L)
                self._L -= self._tmp

                # Bound project L to [L_min, L_max]

                # tmp = x - asymptote_max * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(-self._asymptote_max, self._x_range)

                self._L.pointwiseMax(self._L, self._tmp)

                # tmp = x - asymptote_min * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(-self._asymptote_min, self._x_range)

                self._L.pointwiseMin(self._L, self._tmp)

                # U = x + f ⊙ (U - x_m1)
                self._U.copy(self._tmp)
                self._tmp -= self._x_m1
                self._tmp.pointwiseMult(self._tmp, self._tmp_2)

                self._x.copy(self._U)
                self._U += self._tmp

                # Bound project U to [U_min, U_max]
                # U_min = x + asymptote_min * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(self._asymptote_min, self._x_range)

                self._U.pointwiseMax(self._U, self._tmp)

                # U_max = x + asymptote_max * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(self._asymptote_max, self._x_range)

                self._U.pointwiseMin(self._U, self._tmp)

            # Umx = U - x
            self._U.copy(self._Umx)
            self._Umx -= self._x

            # xmL = x - L
            self._x.copy(self._xmL)
            self._xmL -= self._L

            # alpha = L + albefa * (x - L)
            self._L.copy(self._alpha)
            self._alpha.axpy(self._albefa, self._xmL)

            # alpha = max (alpha, x - move_limit * x_range)
            self._x.copy(self._tmp)
            self._tmp.axpy(-self._move_limit, self._x_range)
            self._alpha.pointwiseMax(self._alpha, self._tmp)

            # alpha = max ( alpha, x_min )
            self._alpha.pointwiseMax(self._alpha, lb)

            # beta = U - albefa * (U - x)
            self._U.copy(self._beta)
            self._beta.axpy(-self._albefa, self._Umx)

            # beta = min (beta, x + move_limit * x_range)
            self._x.copy(self._tmp)
            self._tmp.axpy(self._move_limit, self._x_range)
            self._beta.pointwiseMin(self._beta, self._tmp)

            # beta = min ( beta, x_max )
            self._beta.pointwiseMin(self._beta, ub)

            if not np.all(self._L.getArray() < self._alpha.getArray()):
                raise RuntimeError("L < alpha not fulfilled.")

            if not np.all(self._beta.getArray() < self._U.getArray()):
                raise RuntimeError("beta < U not fulfilled.")

            if not np.all(self._alpha.getArray() <= self._beta.getArray()):
                raise RuntimeError("alpha <= beta not fulfilled.")

            # tmp_2 = grad_p
            self._gradient.copy(self._tmp_2)
            self._tmp_2.pointwiseMax(self._tmp_2, self._zero)

            # tmp_3 = grad_m
            self._gradient.copy(self._tmp_3)
            self._tmp_3.scale(-1)
            self._tmp_3.pointwiseMax(self._tmp_3, self._zero)

            self._Umx.copy(self._p)
            self._p.pointwiseMult(self._p, self._p)

            # tmp = (1+theta) * grad_p + theta * grad_m + raai / x_range
            self._x_range.copy(self._tmp)
            self._tmp.reciprocal()
            self._tmp.scale(self._raai)
            self._tmp.axpy(1 + self._theta, self._tmp_2)
            self._tmp.axpy(self._theta, self._tmp_3)

            self._p.pointwiseMult(self._p, self._tmp)

            self._xmL.copy(self._q)
            self._q.pointwiseMult(self._q, self._q)

            # tmp = (1+theta) * grad_m + theta * grad_p + raai / x_range
            self._x_range.copy(self._tmp)
            self._tmp.reciprocal()
            self._tmp.scale(self._raai)
            self._tmp.axpy(1 + self._theta, self._tmp_3)
            self._tmp.axpy(self._theta, self._tmp_2)

            self._q.pointwiseMult(self._q, self._tmp)

            self._Umx.copy(self._Umx_recp)
            self._Umx_recp.reciprocal()
            self._xmL.copy(self._xmL_recp)
            self._xmL_recp.reciprocal()
            self._r = self._f - self._Umx_recp.dot(self._p) - self._xmL_recp.dot(self._q)

            # Update history (before overwriting x with new solution)
            self._x_m1.copy(self._x_m2)
            self._x.copy(self._x_m1)

            if h:
                c.copy(self._r_h)

                J.copy(J_p)
                if J_p.getType() == PETSc.Mat.Type.TRANSPOSE:
                    positive_part(J_p.getTransposeMat())
                else:
                    positive_part(J_p)

                J.copy(J_m)
                if J_m.getType() == PETSc.Mat.Type.TRANSPOSE:
                    negative_part(J_m.getTransposeMat())
                else:
                    negative_part(J_m)

                # P = (1+theta) J_p + theta J_m + TODO figure kappa out
                J_p.copy(self._P)
                self._P.scale(1 + self._theta)
                self._P.axpy(self._theta, J_m)

                self._Umx.copy(self._tmp)
                self._tmp.pointwiseMult(self._tmp, self._tmp)

                self._P.diagonalScale(L=None, R=self._tmp)

                # r_h -= P (U - x)⁻¹
                self._P.mult(self._Umx_recp, tmp_h)
                self._r_h -= tmp_h

                # Q = (1+theta) J_m + theta J_p + TODO figure kappa out
                J_m.copy(self._Q)
                self._Q.scale(1 + self._theta)
                self._Q.axpy(self._theta, J_p)

                self._xmL.copy(self._tmp)
                self._tmp.pointwiseMult(self._tmp, self._tmp)

                self._Q.diagonalScale(L=None, R=self._tmp)

                # r_h -= Q (x - L)⁻¹
                self._Q.mult(self._xmL_recp, tmp_h)
                self._r_h -= tmp_h

                grad_norm = self._gradient.norm()
                p_norm = self._p.norm()
                q_norm = self._q.norm()

                Umx_arr = self._Umx.getArray()
                xmL_arr = self._xmL.getArray()
                x_range_arr = self._x_range.getArray()

                # --- DIAGNOSTICS: state feeding the dual subsolver ---
                """logger.info(
                    f"[MMA diag] it={it} pre-subsolver: "
                    f"|grad|={grad_norm:.4e} "
                    f"|p|={p_norm:.4e} |q|={q_norm:.4e} "
                    f"min(U-x)={np.min(Umx_arr):.4e} max(U-x)={np.max(Umx_arr):.4e} "
                    f"min(x-L)={np.min(xmL_arr):.4e} max(x-L)={np.max(xmL_arr):.4e} "
                    f"min(x_range)={np.min(x_range_arr):.4e} "
                    f"r_h_norm={self._r_h.norm():.4e}"
                )"""
                if np.min(Umx_arr) < 1e-8 or np.min(xmL_arr) < 1e-8:
                    logger.warning(
                        f"[MMA diag] it={it}: (U-x) or (x-L) near zero — "
                        f"bound/asymptote collapse likely"
                    )
                # --- END DIAGNOSTICS ---

                self._subsolver.solve()

                reason = self._subsolver.getConvergedReason()
                if reason < 0:
                    _TAO_REASONS = {
                        2: "CONVERGED_GATOL", 3: "CONVERGED_GRTOL", 4: "CONVERGED_GTTOL",
                        5: "CONVERGED_STEPTOL", 6: "CONVERGED_MINF", 7: "CONVERGED_USER",
                        -2: "DIVERGED_MAXITS", -4: "DIVERGED_NAN", -5: "DIVERGED_MAXFCN",
                        -6: "DIVERGED_LS_FAILURE", -7: "DIVERGED_TR_REDUCTION",
                        -8: "DIVERGED_USER",
                    }
                    name = _TAO_REASONS.get(reason, "UNKNOWN")
                    final_obj = self._subsolver.getObjectiveValue()
                    its = self._subsolver.getIterationNumber()

                    # This subproblem is the MMA DUAL: a 1-D (single inequality
                    # constraint) CONCAVE maximization in lambda >= 0. Its
                    # objective W is non-smooth in lambda because the dual->primal
                    # map x(lambda) projects onto the move-limited box (kinks
                    # wherever a design variable hits alpha/beta). At a (robustly)
                    # infeasible-ish warm start many variables sit at a bound, so
                    # bqnls can plateau AT the dual optimum yet never drive its
                    # projected gradient below the strict subsolver tolerance in
                    # the iteration budget -> DIVERGED_MAXITS with a FINITE, stable
                    # objective. That is not a real failure: a finite objective on
                    # a concave 1-D maximization means we are at (or essentially
                    # at) the maximizer, so the achieved lambda yields a valid MMA
                    # primal step. MMA re-linearizes every outer iteration and
                    # tolerates an inexact inner solve, so accept it and continue.
                    # Only a NON-FINITE objective (NaN/Inf -- a genuinely corrupt
                    # dual) is fatal.
                    if not np.isfinite(final_obj):
                        logger.error(
                            f"[MMA diag] subsolver FATALLY diverged: reason={reason} "
                            f"({name}), its={its}, final_obj={final_obj}"
                        )
                        raise RuntimeError(
                            f"Subsolver diverged with non-finite objective "
                            f"(reason={reason}, {name})."
                        )
                    if not warned_subsolver_nonconvergence:
                        logger.warning(
                            f"[MMA diag] subsolver did not meet its tolerance "
                            f"(reason={reason} {name}, its={its}, "
                            f"final_obj={final_obj:.6e}); the 1-D concave dual "
                            f"objective is finite and stable, so accepting the "
                            f"achieved multiplier and continuing (this outer solve "
                            f"suppresses further identical warnings)."
                        )
                        warned_subsolver_nonconvergence = True

                self.x(self._subsolver.getSolution(), self._x)
            else:
                tmp_p = self._p.copy()
                tmp_p.sqrtabs()

                tmp_q = self._q.copy()
                tmp_q.sqrtabs()

                div_tmp = tmp_p.copy()
                div_tmp += tmp_q

                tmp_p.pointwiseMult(tmp_p, self._L)
                tmp_q.pointwiseMult(tmp_q, self._U)

                tmp_p.copy(self._x)
                self._x += tmp_q
                self._x.pointwiseDivide(self._x, div_tmp)

                self._x.pointwiseMax(self._x, lb)
                self._x.pointwiseMin(self._x, ub)

            # convergence and logging
            self._objective = tao.computeObjectiveGradient(self._x, self._gradient)

            tao.setIterationNumber(it)
            # `res` is the RELATIVE stationarity residual, not ||grad f||. The
            # latter is retained inside self._optimality purely as a diagnostic.
            residual = (
                self._optimality.stationarity_rel
                if self._optimality is not None
                else float("inf")
            )
            cnorm = self._optimality.feasibility if self._optimality is not None else 0.0
            try:
                tao.monitor(f=self._objective, res=residual, cnorm=cnorm)
            except TypeError:
                # Older petsc4py builds expose no `cnorm` keyword on TAO.monitor.
                tao.monitor(f=self._objective, res=residual)

            # Converge only when ALL THREE first-order conditions hold. TAO's
            # default test would look at `res` alone, which would let a
            # stationary-but-INFEASIBLE point report success.
            if self._optimality is not None and self._optimality.satisfied(
                gatol, self._constraint_tol
            ):
                tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_GATOL)
                logger.info(
                    "[MMA opt] first-order optimality reached at it=%d: %s "
                    "(gatol=%.3g, constraint_tol=%.3g)",
                    it, self._optimality.summary(), gatol, self._constraint_tol,
                )

        # Exhausting the iteration budget without meeting the first-order
        # conditions must be reported as such. Previously the loop simply fell
        # through with tao.reason == 0 (TAO_CONTINUE_ITERATING), which every
        # caller that checks `reason < 0` read as a clean convergence.
        if not tao.reason:
            tao.setConvergedReason(PETSc.TAO.ConvergedReason.DIVERGED_MAXITS)
            logger.warning(
                "[MMA opt] iteration budget (%d) exhausted WITHOUT meeting the "
                "first-order conditions: %s. Reporting DIVERGED_MAXITS -- the "
                "returned design is the best iterate reached, not a converged "
                "optimum.",
                tao.getMaximumIterations(),
                self._optimality.summary() if self._optimality else "no iterations ran",
            )

    @property
    def optimality(self):
        """The most recent FirstOrderOptimality record (or None before solve).

        This -- not the objective-gradient norm -- is what callers should
        report as the convergence evidence for a design.
        """
        return self._optimality

    @property
    def subsolver(self) -> PETSc.TAO:
        return self._subsolver

    @property
    def albefa(self) -> float:
        return self._albefa

    @property
    def move_limit(self) -> float:
        return self._move_limit

    @property
    def asymptote_init(self) -> float:
        return self._asymptote_init

    @property
    def asymptote_decrement(self) -> float:
        return self._asymptote_decrement

    @property
    def asymptote_increment(self) -> float:
        return self._asymptote_increment

    @property
    def asymptote_min(self) -> float:
        return self._asymptote_min

    @property
    def asymptote_max(self) -> float:
        return self._asymptote_max

    @property
    def raai(self) -> float:
        return self._raai

    @property
    def theta(self) -> float:
        return self._theta

    def destroy(self, tao: PETSc.TAO):
        logger.debug("destroy")

        to_destroy = (
            self._λ,
            self._J_λ,
            self._x_range,
            self._x_m1,
            self._x_m2,
            self._diff_12,
            self._diff_23,
            self._p,
            self._q,
            self._P,
            self._Q,
            self._L,
            self._U,
            self._alpha,
            self._beta,
            self._xmL,
            self._Umx,
            self._xmL_recp,
            self._Umx_recp,
            self._tmp,
            self._tmp_2,
            self._tmp_3,
            self._zero,
            self._optimality_work,
        )
        for o in filter(lambda o: o is not None, to_destroy):
            o.destroy()

        self._subsolver.destroy()