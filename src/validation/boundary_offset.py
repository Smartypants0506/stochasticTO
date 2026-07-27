"""How far does the manufacturing perturbation actually move the boundary?

WHY THIS MODULE EXISTS
----------------------
The whole physical premise of this project is that a shift in the projection
threshold eta models a shift of the manufactured boundary. Nothing measured how
far the boundary actually moves, so nobody noticed that at eta in [0.45, 0.55]
it moved by roughly one TWELFTH of an element -- far below the mesh's ability to
resolve it, which makes sigma_C the same order as the discretization error of
the compliance and leaves the robust design no room to differ from the nominal.

THE ESTIMATORS
--------------
The projected boundary is the level set rho_tilde = eta, so a threshold shift
d_eta displaces it by

    d_s(x) = d_eta / |grad rho_tilde(x)|

which varies over the interface. Two independent estimators are provided,
because they fail differently and agreeing is evidence.

1. GEOMETRIC (measure_interface_geometry). Uses the coarea formula, which turns
   volume integrals of |grad u| into integrals over level sets:

       integral f |grad u| dx = integral over t of ( integral f dS on {u=t} ) dt

   Take chi = 1 on the interface band lo < rho_tilde < hi. Then

       integral chi dx        = (hi-lo) * <1/|grad|>_A * A
       integral chi |grad| dx = (hi-lo) * A

   so the AREA-WEIGHTED mean displacement per unit d_eta is exactly

       <1/|grad rho_tilde|>_A = (integral chi dx) / (integral chi |grad| dx)

   and the interface area is A = (integral chi |grad| dx) / (hi - lo).

   Two scalar assemblies, no interpolation, no version-fragile element API, and
   MPI-collective by construction.

2. EMPIRICAL (measure_offset_from_volumes). Each Monte Carlo sample already
   records its volume fraction V_i. A boundary displaced outward by d_s over an
   interface of area A changes the volume by A * d_s, so

       d_s_i = (V_i - V_nominal) * total_volume / A

   This needs NO new FEA at all -- the volumes are already collected by
   Sensitivity.evaluate() and stored in every MCResult. It is the measured
   realized displacement of the actual ensemble, which is why it is the one to
   quote; the geometric estimator is the independent cross-check.

WHAT TO DO WITH THE ANSWER
--------------------------
Report the displacement in three ways, because they answer different questions:
  * in units of h        -- can the mesh resolve it? (the numerical question)
  * in absolute units    -- reproducibility
  * as a fraction of the minimum feature size ~2R -- is it a plausible
    manufacturing tolerance, or a coarse-casting envelope? (the physical
    question)

A band can pass the first and fail the third; [0.25, 0.75] does exactly that,
which is why the write-up must present it as a robustness envelope rather than a
calibrated tolerance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import ufl
from dolfinx.fem import assemble_scalar, form
from mpi4py import MPI

logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD

# Interface band for the coarea estimator. Wide enough that the band contains
# enough quadrature points to integrate accurately, narrow enough that the
# gradient it averages is representative of the rho_tilde = 0.5 level set.
_BAND_LO = 0.4
_BAND_HI = 0.6

# Below this, the perturbation is not resolved by the mesh and sigma_C cannot be
# distinguished from discretization error. See the module docstring.
RESOLVABLE_OFFSET_STD_IN_ELEMENTS = 0.25

_GRAD_FLOOR = 1e-12


@dataclass
class BoundaryOffsetReport:
    """Realized boundary displacement, in the three units that matter."""

    element_size: float
    filter_radius: float
    min_feature_size: float           # ~2R, the smallest member the filter admits
    eta_min: float
    eta_max: float
    eta_std: float

    interface_area: float
    mean_displacement_per_unit_eta: float   # <1/|grad rho_tilde|>_A

    offset_std_absolute: float
    offset_std_elements: float
    offset_std_feature_fraction: float
    offset_range_absolute: float
    offset_range_elements: float

    # Empirical estimator (populated when MC volumes are supplied).
    empirical: dict = field(default_factory=dict)

    @property
    def resolvable(self) -> bool:
        return self.offset_std_elements >= RESOLVABLE_OFFSET_STD_IN_ELEMENTS

    def as_dict(self) -> dict:
        return {
            "element_size_h": self.element_size,
            "filter_radius_R": self.filter_radius,
            "min_feature_size_2R": self.min_feature_size,
            "eta_band": [self.eta_min, self.eta_max],
            "eta_std": self.eta_std,
            "interface_area": self.interface_area,
            "mean_displacement_per_unit_eta": self.mean_displacement_per_unit_eta,
            "offset_std_absolute": self.offset_std_absolute,
            "offset_std_elements": self.offset_std_elements,
            "offset_std_as_fraction_of_min_feature": self.offset_std_feature_fraction,
            "offset_range_absolute": self.offset_range_absolute,
            "offset_range_elements": self.offset_range_elements,
            "resolvable": self.resolvable,
            "resolvable_threshold_elements": RESOLVABLE_OFFSET_STD_IN_ELEMENTS,
            "empirical": self.empirical,
            "note": (
                "offset_std_elements answers the NUMERICAL question (can the "
                "mesh resolve the perturbation). "
                "offset_std_as_fraction_of_min_feature answers the PHYSICAL "
                "question (is this a plausible process tolerance). A band can "
                "pass the first and fail the second; report both."
            ),
        }

    def summary(self) -> str:
        verdict = "RESOLVABLE" if self.resolvable else "NOT RESOLVABLE BY THIS MESH"
        return (
            f"boundary offset std = {self.offset_std_absolute:.4g} "
            f"({self.offset_std_elements:.3g} h, "
            f"{100 * self.offset_std_feature_fraction:.1f}% of min feature); "
            f"range = {self.offset_range_elements:.3g} h -- {verdict}"
        )


def measure_interface_geometry(rho_tilde_field, quadrature_degree: int = 2) -> tuple[float, float]:
    """Area-weighted mean of 1/|grad rho_tilde| over the interface, and the area.

    .. warning::
       ``rho_tilde_field`` must be the FILTERED field, i.e. read it immediately
       after ``density_filter.forward()`` and BEFORE ``rf_heaviside.forward()``.
       In this codebase both live in the same dolfinx Function
       (``rho_phys_field``), which holds rho_tilde before the projection and
       rho_phys after it. Measuring after the projection would report the
       gradient of a near-binary field -- enormous, and meaningless here.
       A heuristic check below warns when the field looks already projected.

    Collective: assembles two scalars and allreduces them.

    Returns:
        (mean_displacement_per_unit_eta, interface_area)
    """
    local = np.asarray(rho_tilde_field.x.petsc_vec.array)
    intermediate_fraction = (
        float(np.mean((local > 0.05) & (local < 0.95))) if local.size else 0.0
    )
    global_intermediate = comm.allreduce(intermediate_fraction * local.size, op=MPI.SUM)
    global_count = comm.allreduce(local.size, op=MPI.SUM)
    if global_count and global_intermediate / global_count < 0.01:
        logger.warning(
            "measure_interface_geometry: only %.3g%% of the field lies strictly "
            "between 0.05 and 0.95. That is what a PROJECTED field looks like, "
            "not a filtered one. If this was called after "
            "rf_heaviside.forward(), the result is meaningless -- call it "
            "immediately after density_filter.forward() instead.",
            100 * global_intermediate / max(global_count, 1),
        )

    mesh = rho_tilde_field.function_space.mesh
    metadata = {"quadrature_degree": quadrature_degree}
    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)

    grad_magnitude = ufl.sqrt(
        ufl.dot(ufl.grad(rho_tilde_field), ufl.grad(rho_tilde_field)) + _GRAD_FLOOR
    )
    in_band = ufl.conditional(
        ufl.And(ufl.gt(rho_tilde_field, _BAND_LO), ufl.lt(rho_tilde_field, _BAND_HI)),
        1.0, 0.0,
    )

    band_volume = comm.allreduce(assemble_scalar(form(in_band * dx)), op=MPI.SUM)
    band_area_integral = comm.allreduce(
        assemble_scalar(form(in_band * grad_magnitude * dx)), op=MPI.SUM
    )

    if band_area_integral <= 0.0:
        raise RuntimeError(
            "The interface band integral is zero: no part of the design has "
            f"rho_tilde in ({_BAND_LO}, {_BAND_HI}). Either the design is "
            "entirely solid/void, or a projected field was passed instead of a "
            "filtered one."
        )

    # Coarea identities -- see the module docstring.
    mean_displacement_per_unit_eta = band_volume / band_area_integral
    interface_area = band_area_integral / (_BAND_HI - _BAND_LO)
    return float(mean_displacement_per_unit_eta), float(interface_area)


def _beta_std(alpha: float, beta: float, band: float) -> float:
    """Standard deviation of Beta(alpha, beta) rescaled onto a band of that width."""
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return band * float(np.sqrt(variance))


def measure_offset_from_volumes(
    volume_samples: np.ndarray,
    nominal_volume_fraction: float,
    total_volume: float,
    interface_area: float,
    element_size: float,
) -> dict:
    """Realized displacement of each MC sample, from volumes already collected.

    A boundary displaced by d_s over an interface of area A changes the enclosed
    volume by A*d_s, so d_s_i = (V_i - V_nominal) * total_volume / A. Costs no
    FEA: run_monte_carlo_validation already records V_i for every sample.

    Args:
        volume_samples: [n] per-sample volume FRACTIONS (FEniTop's convention:
            Sensitivity.evaluate returns actual_volume / total_volume).
        nominal_volume_fraction: Volume fraction of the same design at eta = 0.5.
        total_volume: Domain volume, from Sensitivity.total_volume.
        interface_area: From measure_interface_geometry.
        element_size: Mesh h, for reporting the result in elements.

    Returns:
        Dict of the realized displacement distribution, in absolute units and in
        elements.
    """
    volume_samples = np.asarray(volume_samples, dtype=float).ravel()
    displacement = (volume_samples - nominal_volume_fraction) * total_volume / interface_area

    return {
        "n_samples": int(displacement.size),
        "mean_absolute": float(displacement.mean()),
        "std_absolute": float(displacement.std(ddof=1)) if displacement.size > 1 else float("nan"),
        "min_absolute": float(displacement.min()),
        "max_absolute": float(displacement.max()),
        "mean_elements": float(displacement.mean() / element_size),
        "std_elements": (
            float(displacement.std(ddof=1) / element_size)
            if displacement.size > 1 else float("nan")
        ),
        "range_elements": float((displacement.max() - displacement.min()) / element_size),
        "note": (
            "Displacement realized by the actual eta ensemble, derived from the "
            "per-sample volumes with no additional FEA. This is the measured "
            "number; the geometric estimator is an independent cross-check and "
            "the two should agree to within the interface-area approximation."
        ),
    }


def build_report(
    rho_tilde_field,
    transform_params,
    element_size: float,
    filter_radius: float,
    volume_samples: np.ndarray | None = None,
    nominal_volume_fraction: float | None = None,
    total_volume: float | None = None,
) -> BoundaryOffsetReport:
    """Full boundary-offset report, geometric estimator plus (optionally) the
    empirical one.

    .. warning::
       See measure_interface_geometry: rho_tilde_field must be the FILTERED
       field, read before the Heaviside projection is applied.

    Args:
        rho_tilde_field: The filtered density field.
        transform_params: MarginalTransformParams (eta band and Beta shapes).
        element_size: Mesh h, from fem["element_size"].
        filter_radius: Helmholtz filter length R, from opt["filter_radius"].
        volume_samples: Optional per-sample volume fractions for the empirical
            estimator.
        nominal_volume_fraction: Volume fraction at eta = 0.5.
        total_volume: Domain volume (Sensitivity.total_volume).

    Returns:
        A BoundaryOffsetReport, world-identical.
    """
    displacement_per_eta, interface_area = measure_interface_geometry(rho_tilde_field)

    band = transform_params.eta_max - transform_params.eta_min
    eta_std = _beta_std(transform_params.alpha, transform_params.beta, band)

    offset_std = eta_std * displacement_per_eta
    offset_range = band * displacement_per_eta
    min_feature = 2.0 * filter_radius

    report = BoundaryOffsetReport(
        element_size=element_size,
        filter_radius=filter_radius,
        min_feature_size=min_feature,
        eta_min=transform_params.eta_min,
        eta_max=transform_params.eta_max,
        eta_std=eta_std,
        interface_area=interface_area,
        mean_displacement_per_unit_eta=displacement_per_eta,
        offset_std_absolute=offset_std,
        offset_std_elements=offset_std / element_size,
        offset_std_feature_fraction=offset_std / min_feature,
        offset_range_absolute=offset_range,
        offset_range_elements=offset_range / element_size,
    )

    if volume_samples is not None and nominal_volume_fraction is not None and total_volume:
        report.empirical = measure_offset_from_volumes(
            volume_samples, nominal_volume_fraction, total_volume,
            interface_area, element_size,
        )

    if comm.rank == 0:
        logger.info("Boundary offset: %s", report.summary())
        if not report.resolvable:
            logger.error(
                "The manufacturing perturbation is NOT resolved by this mesh "
                "(offset std %.3g h < %.3g h). sigma_C is then the same order "
                "as the discretization error of the compliance and cannot be "
                "shown to be a property of the continuum problem. Widen "
                "random_field.eta_min/eta_max -- refining the mesh does NOT "
                "help, because the displacement is fixed in absolute units and "
                "depends on the eta band and the filter radius, not on h.",
                report.offset_std_elements, RESOLVABLE_OFFSET_STD_IN_ELEMENTS,
            )
    return report
