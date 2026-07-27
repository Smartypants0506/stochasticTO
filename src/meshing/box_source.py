"""
src/meshing/box_source.py

Synthetic box-mesh source, ported directly from FEniTop's own
scripts/beam_3d.py reference example. Exists ONLY to validate this
project's multi-load-case topopt() pipeline (Stage 2 nominal warm-start,
and optionally Stages 3-6) against FEniTop's known-good single-case
result -- it is not a general-purpose box-mesh feature.

Deliberately hardcodes beam_3d.py's domain, element counts, material
properties, SIMP/optimization parameters, disp_bc region, and traction
location -- rather than sourcing them from cfg -- because this project's
config.yaml carries real GE-bracket SI values for the STEP path (a
different optimization problem entirely). Pulling from cfg here would
silently validate against the wrong case. Only Stage 3-6 parameters
(random field, surrogate, MC validation), which have no beam_3d.py
equivalent at all, are sourced from cfg, exactly mirroring how
fenitop_adapter.build_fenitop_dicts does it for the STEP path -- so
build_fenitop_dicts itself did not need to change.

CELL TYPE / SIMPLEX NOTE:
beam_3d.py uses CellType.hexahedron. This project's Stage 3+
(src/random_fields/kl_expansion.py's compute_kl_expansion, via
mesher.extract_simplices) builds an OpenTURNS FEM mesh that requires a
SIMPLICIAL mesh_serial -- hexahedra will not work there. Since main.py's
pipeline is expected to run end-to-end (Stages 2-6), this module defaults
to CellType.tetrahedron (same domain, same [75, 225, 75] element counts
as beam_3d.py) so the KL/PCE/robust-loop/MC stages behave correctly. This
is the one deliberate deviation from beam_3d.py -- if you only care about
a Stage-2-only apples-to-apples hex comparison, set
cfg.box_mesh.cell_type = "hexahedron" and stop the pipeline after
Stage 2 (main.py raises if you try to proceed past it with hexahedra).
"""
from __future__ import annotations

import logging

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType

from src.config.schema import ProjectConfig
from src.meshing.mesher import TaggedMesh
from src.random_fields.kernel import KernelParams
from src.random_fields.threshold_transform import MarginalTransformParams

logger = logging.getLogger(__name__)

# --- beam_3d.py's fixed reference case (do not make config-driven) ---------
# Every value below is copied verbatim from scripts/beam_3d.py's `fem`/`opt`
# dicts. This project's own config.yaml carries real GE-bracket SI values
# (E=68.9 GPa, vol_frac=0.1, filter_radius=0.006, ...) for the STEP path,
# which are a DIFFERENT optimization problem -- pulling material/SIMP
# params from cfg here would silently validate against the wrong case.
# Only Stage 3-6 params (random field / surrogate / MC), which have no
# beam_3d.py equivalent at all, are sourced from cfg below.
_DOMAIN = [[0, 0, 0], [10, 30, 10]]
_ELEMENTS = [25, 75, 25]          # refinement level 1.0 -> h = 0.4
_YOUNGS_MODULUS = 100
_POISSONS_RATIO = 0.25
_VOL_FRAC = 0.08
_PENALTY = 3.0
_EPSILON = 1e-6

# --- FILTER RADIUS: FIXED IN ABSOLUTE UNITS ACROSS EVERY REFINEMENT LEVEL ---
# Helmholtz length 0.6 = 1.5*h at the reference mesh (h=0.4), i.e. roughly a
# 5-element classical filter radius.
#
# DO NOT scale this with the mesh. The filter is part of the CONTINUUM problem
# the mesh study converges to: holding R fixed while h shrinks converges to a
# single well-posed problem, whereas tying R to h (R = 1.5h at every level)
# defines a DIFFERENT problem at every level and produces a convergence plot
# that looks clean and proves nothing -- the minimum feature size would shrink
# with the mesh, so the design, its compliance and its sigma_C would all keep
# moving and never converge to anything.
#
# config.optimization.filter_radius is intentionally NOT honored on the box
# path (beam_3d physics is frozen); loader.py rejects it. Tune it here.
_FILTER_RADIUS = 0.6
_BETA_INTERVAL = 50
_BETA_MAX = 128
_USE_OC = True
_MOVE = 0.02
_OPT_COMPLIANCE = True
_MAX_ITER = 400
_OPT_TOL = 1e-5
_QUADRATURE_DEGREE = 2
_BODY_FORCE = (0, 0, 0)
# fgmres, NOT cg: under high SIMP contrast (epsilon small) a random eta(x) draw
# can make the structure near-disconnected, and GAMG then produces a slightly
# non-SPD preconditioner application. CG detects that and bails immediately with
# DIVERGED_INDEFINITE_PC (reason=-8), crashing the batch. FGMRES minimizes the
# residual regardless of PC definiteness and converges on the same SPD system to
# the same tolerance (math-exact -- same solution), just a bit more work/memory.
_PETSC_OPTIONS = {
    "ksp_type": "fgmres",
    "ksp_gmres_restart": 100,
    "pc_type": "gamg",
    "ksp_max_it": 2000,
    "ksp_error_if_not_converged": None,
}

_CELL_TYPES = {
    "tetrahedron": CellType.tetrahedron,
    "hexahedron": CellType.hexahedron,
}


def elements_for_refinement(refinement: float) -> list[int]:
    """Element counts at a given refinement level, domain and filter radius fixed.

    refinement=1.0 reproduces beam_3d.py's [25, 75, 25] (h = 0.4) exactly.
    Values below 1 coarsen (study tier), above 1 refine.

    Element counts are rounded to integers, so the requested and realized h can
    differ slightly; realized_element_size() reports what was actually built and
    that is the value written into the run manifest. The domain aspect ratio
    (1:3:1) is preserved so h stays isotropic.

    Args:
        refinement: Multiplier on the reference element counts. Must be > 0.

    Returns:
        [nx, ny, nz], each at least 1.
    """
    if refinement <= 0:
        raise ValueError(f"box_mesh.refinement must be > 0, got {refinement}")
    return [max(1, int(round(n * refinement))) for n in _ELEMENTS]


def realized_element_size(elements: list[int]) -> float:
    """Actual element size h of a built mesh, for reporting and for expressing
    the boundary offset in units of h."""
    return (_DOMAIN[1][0] - _DOMAIN[0][0]) / elements[0]


def _disp_bc(x):
    """Identical to beam_3d.py's fem['disp_bc']: clamp the two short edges
    at y=0."""
    return np.isclose(x[1], 0) & (np.less(x[0], 1.5) | np.greater(x[0], 8.5))


# beam_3d.py's load: traction (0,0,-2.0) over the patch |x-5|<0.5, |z-5|<0.5 at
# y=30. At h=0.4 that patch admits exactly the quad [4.8,5.2]x[4.8,5.2], so the
# reference TOTAL FORCE is 2.0 * 0.4 * 0.4:
_REFERENCE_LOAD_TOTAL_FORCE = 0.32
_LOAD_CENTRE = (5.0, 5.0)
_LOAD_NOMINAL_HALFWIDTH = 0.5


def _load_patch_halfwidth(element_size: float, override: float | None = None) -> float:
    """Half-width of the traction patch at a given mesh size.

    WHY THIS IS NOT SIMPLY beam_3d's 0.5. dolfinx's locate_entities_boundary
    selects a facet only when ALL of its vertices satisfy the predicate, so the
    patch must contain two CONSECUTIVE grid nodes to admit any facet at all.
    A fixed half-width of 0.5 does that only for a handful of meshes: at h=1.0,
    0.833, 0.625 and 0.5 it contains the single node x=5.0, no facet is
    selected, the traction is applied to nothing, and the solve returns u=0 and
    compliance=0 -- silently, with no error anywhere. That is exactly what
    happened when the refinement knob was added: of the mesh-study levels
    (0.48, 0.64, 1.0) only 1.0 had any load.

    max(0.5, 1.5h) guarantees at least two nodes inside at any h. At the
    reference mesh h=0.4 it evaluates to 0.6, which is WIDER than beam_3d's 0.5
    yet selects exactly the same facets: the patch bounds 4.4 and 5.6 are not
    grid nodes, so the selected set is still {4.8, 5.2} and the normalized
    traction still comes out at exactly -2.0. beam_3d is reproduced bit for bit.

    `override` pins the half-width to one value across a family of meshes. A
    mesh-convergence study MUST use it: with the per-mesh rule the patch shrinks
    as h refines (half-width 1.25 at h=0.833 down to 0.6 at h=0.4), so the load
    becomes progressively more concentrated and each level solves a slightly
    DIFFERENT problem -- the same defect that scaling the filter radius with the
    mesh would introduce. Pinning it to the coarsest level's value makes every
    level solve one problem, at the cost of no longer matching beam_3d.
    """
    if override is not None:
        return float(override)
    return max(_LOAD_NOMINAL_HALFWIDTH, 1.5 * element_size)


def loaded_area(element_size: float, halfwidth_override: float | None = None) -> float:
    """Area of the facets the traction predicate will actually select.

    Computed analytically from the structured grid: the selected facets are the
    cells whose two bounding nodes both lie strictly inside the patch, in x and
    in z. Used to normalize the traction so the TOTAL applied force is identical
    at every refinement level -- without that, a coarser mesh gets a wider patch
    and therefore a larger total load, i.e. a different physical problem at each
    level, which would make a mesh-convergence study meaningless in precisely
    the way the fixed filter radius is meant to prevent.
    """
    halfwidth = _load_patch_halfwidth(element_size, halfwidth_override)
    n_nodes = int(round((_DOMAIN[1][0] - _DOMAIN[0][0]) / element_size))
    nodes = np.arange(n_nodes + 1) * element_size
    inside = nodes[
        (nodes > _LOAD_CENTRE[0] - halfwidth) & (nodes < _LOAD_CENTRE[0] + halfwidth)
    ]
    n_intervals = inside.size - 1
    if n_intervals < 1:
        raise RuntimeError(
            f"The traction patch (half-width {halfwidth:.4g}) contains "
            f"{inside.size} grid node(s) at h={element_size:.4g}, so NO facet "
            "can be selected and the load would be applied to nothing "
            "(compliance would come out exactly zero). Widen the patch: either "
            "raise box_mesh.load_patch_halfwidth, or leave it null to use the "
            "automatic max(0.5, 1.5h) rule."
        )
    return (n_intervals * element_size) ** 2


def _make_load_membership(element_size: float, halfwidth_override: float | None = None):
    """beam_3d's traction patch, widened just enough to remain resolvable."""
    halfwidth = _load_patch_halfwidth(element_size, halfwidth_override)
    x0, z0 = _LOAD_CENTRE

    def _membership(x):
        return (np.isclose(x[1], 30)
                & np.greater(x[0], x0 - halfwidth) & np.less(x[0], x0 + halfwidth)
                & np.greater(x[2], z0 - halfwidth) & np.less(x[2], z0 + halfwidth))

    return _membership


def _solid_zone(x):
    """beam_3d.py has no protected solid regions."""
    return np.full(x.shape[1], False)


def _void_zone(x):
    """beam_3d.py has no forced-void regions."""
    return np.full(x.shape[1], False)


def build_box_mesh(comm: MPI.Comm, cell_type: str, refinement: float = 1.0):
    """Build the beam_3d domain at a given refinement level.

    At refinement=1.0 this reproduces beam_3d.py's mesh exactly. The DOMAIN and
    the filter radius are identical at every level -- only h changes -- which is
    what makes a mesh-convergence study measure convergence to the continuum
    problem rather than tracking a moving target.

    mesh_serial is built on COMM_SELF and is therefore byte-identical across the
    world communicator and every sample-parallel sub-communicator, which is the
    correctness linchpin for recombining grouped results (see
    build_group_fea_context).
    """
    if cell_type not in _CELL_TYPES:
        raise ValueError(
            f"box_mesh.cell_type={cell_type!r} not supported; expected one "
            f"of {list(_CELL_TYPES)}"
        )
    dolfinx_cell_type = _CELL_TYPES[cell_type]
    elements = elements_for_refinement(refinement)

    mesh = create_box(comm, _DOMAIN, elements, dolfinx_cell_type)
    if comm.rank == 0:
        mesh_serial = create_box(MPI.COMM_SELF, _DOMAIN, elements, dolfinx_cell_type)
    else:
        mesh_serial = None
    return mesh, mesh_serial


def build_box_fenitop_dicts(config: ProjectConfig, comm: MPI.Comm):
    """Box-mesh analogue of fenitop_adapter.build_fenitop_dicts().

    Returns (tagged_mesh, fem, opt, load_cases) in the exact same shape
    main.py's STEP path already produces, so every downstream stage
    (XDMF checkpoint write, topopt, KL expansion, Pareto sweep, MC
    validation) runs completely unmodified.

    tagged_mesh is a real mesher.TaggedMesh with cell_tags/facet_tags/
    name_to_tag left empty/None (beam_3d.py has no physical-group
    tagging), so main.py's `tagged_mesh.mesh` / `tagged_mesh.mesh_serial`
    / `tagged_mesh.name_to_tag` accesses keep working unmodified.

    Single load case ("beam_3d_reference"), matching beam_3d.py's single
    traction_bcs entry -- topopt.py's form_fem_multi_case treats a
    one-entry load_cases dict identically to a multi-entry one, so no
    separate downstream code path is required.
    """
    cell_type = config.box_mesh.cell_type
    refinement = config.box_mesh.refinement
    elements = elements_for_refinement(refinement)
    element_size = realized_element_size(elements)
    mesh, mesh_serial = build_box_mesh(comm, cell_type, refinement)

    tagged_mesh = TaggedMesh(
        mesh=mesh,
        mesh_serial=mesh_serial,
        cell_tags=None,
        facet_tags=None,
        cell_tags_serial=None,
        facet_tags_serial=None,
        name_to_tag={},
    )

    fem = {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "young's modulus": _YOUNGS_MODULUS,
        "poisson's ratio": _POISSONS_RATIO,
        "disp_bc": _disp_bc,
        "body_force": _BODY_FORCE,
        "quadrature_degree": _QUADRATURE_DEGREE,
        "petsc_options": dict(_PETSC_OPTIONS),
        # No physical-group tagging in the box source -- matches
        # beam_3d.py, which has no bolt/mount pinning. topopt.py falls
        # back to its "no solid/cell_tags found" warning path, same as
        # beam_3d.py's own solid_zone/void_zone lambdas (both all-False).
        "cell_tags": None,
        "solid_tag": None,
        # Deterministic mesh rebuild on an arbitrary communicator, used by the
        # sub-communicator sample-parallelism path to construct a per-group mesh.
        # create_box(COMM_SELF, ...) makes mesh_serial byte-identical across the
        # world and every group, so their serial cell/node ordering matches and
        # gathered global arrays remain interchangeable (see build_group_fea_context).
        # refinement is bound here so the group meshes match the world mesh --
        # a group built at a different level would silently produce gradient
        # rows for the wrong elements.
        "mesh_factory": (
            lambda c, ct=cell_type, r=refinement: build_box_mesh(c, ct, r)
        ),
        # Geometry facts the downstream measurements need. element_size is what
        # the boundary offset is reported in units of; filter_radius is repeated
        # here (it is also in opt) because the minimum feature size ~ 2R is the
        # other scale the offset must be compared against.
        "element_size": element_size,
        "domain": _DOMAIN,
        "elements": elements,
        "refinement": refinement,
    }

    rf_cfg = config.random_field
    surrogate_cfg = config.surrogate
    # pce_refresh_interval is a Stage 5 scheduling knob specific to this
    # project's robust loop -- no beam_3d.py equivalent -- so it's sourced
    # from cfg like the other Stage 3-6 params, not hardcoded.
    pce_refresh_interval = config.optimization.pce_refresh_interval

    opt = {
        # max_iter is run-control (iteration budget), NOT beam_3d.py physics, so
        # it is sourced from cfg -- this lets a smoke config shorten the run
        # (Stage 2 warm-start + Stage 5 robust loop) without changing the
        # problem. Defaults to 400 via config.yaml, preserving the reference run.
        "max_iter": config.optimization.max_iter,
        "opt_tol": _OPT_TOL,
        "vol_frac": _VOL_FRAC,
        "solid_zone": _solid_zone,
        "void_zone": _void_zone,
        "penalty": _PENALTY,
        # epsilon (void/solid stiffness ratio E_min/E_0) is sourced from cfg: it
        # is the dominant FEA *conditioning* knob. The beam_3d default 1e-6 is a
        # 1e6 contrast that makes GAMG fragile (indefinite-PC failures) on
        # near-disconnected random-eta draws; config.yaml's 1e-4 is far more
        # solvable with a negligible effect on the (essentially 0/1) design.
        "epsilon": config.optimization.epsilon,
        "filter_radius": _FILTER_RADIUS,
        "beta_interval": _BETA_INTERVAL,
        "beta_max": _BETA_MAX,
        "use_oc": _USE_OC,
        "move": _MOVE,
        "opt_compliance": _OPT_COMPLIANCE,

        "kl_model": None,
        "kernel_params": KernelParams(
            sigma=rf_cfg.sigma,
            length_scale=rf_cfg.length_scale,
            spatial_dim=rf_cfg.spatial_dim,
        ),
        "transform_params": MarginalTransformParams(
            eta_min=rf_cfg.eta_min,
            eta_max=rf_cfg.eta_max,
            alpha=rf_cfg.alpha,
            beta=rf_cfg.beta,
        ),
        "kl_variance_threshold": rf_cfg.variance_threshold,
        "random_field_seed": rf_cfg.seed,
        "n_kl": rf_cfg.n_kl_hint,

        "pce_n_train": surrogate_cfg.n_train,
        "pce_n_test": surrogate_cfg.n_test,
        "pce_hyperbolic_q": surrogate_cfg.hyperbolic_q,
        "pce_max_degree_attempts": surrogate_cfg.max_degree_attempts,
        "pce_q2_threshold": surrogate_cfg.q2_threshold,

        "pce_refresh_interval": pce_refresh_interval,
        "sample_parallel_ranks_per_group": config.optimization.sample_parallel_ranks_per_group,
        "pce_max_escalations": config.optimization.pce_max_escalations,
        "pce_n_train_escalation_cap": config.optimization.pce_n_train_escalation_cap,
        "pce_divergence_patience": config.optimization.pce_divergence_patience,

        # --- robust-solve control (config-driven: these are run control, not
        # beam_3d physics, so they belong in config.yaml and are read here) ---
        "saa_beta": config.optimization.saa_beta,
        "saa_beta_max": config.optimization.saa_beta_max,
        "saa_beta_continuation": config.optimization.saa_beta_continuation,
        # robust_opt_tol is the MMA stationarity tolerance and is deliberately a
        # SEPARATE key from opt_tol above, which is the nominal Stage-2 OC
        # design-change threshold (_OPT_TOL, frozen beam_3d run control). The
        # two used to share one key while meaning different things.
        "robust_opt_tol": config.optimization.robust_opt_tol,
        "constraint_tol": config.optimization.constraint_tol,
    }

    # Traction is normalized to hold the TOTAL applied force fixed across
    # refinement levels: the patch has to widen on coarse meshes to remain
    # resolvable (see _load_patch_halfwidth), and a fixed traction over a wider
    # patch would be a larger -- i.e. different -- load at every level. At
    # h=0.4 the patch area is 0.16 and this returns exactly beam_3d's -2.0.
    halfwidth_override = config.box_mesh.load_patch_halfwidth
    halfwidth = _load_patch_halfwidth(element_size, halfwidth_override)
    area = loaded_area(element_size, halfwidth_override)
    traction_z = -_REFERENCE_LOAD_TOTAL_FORCE / area
    load_cases = {
        "beam_3d_reference": [
            (
                (0.0, 0.0, traction_z),
                _make_load_membership(element_size, halfwidth_override),
            )
        ],
    }
    if comm.rank == 0:
        logger.info(
            "Traction patch: half-width %.4g (%s), area %.6g, traction_z %.6g "
            "(total force %.6g held fixed across refinement levels)",
            halfwidth,
            "PINNED via box_mesh.load_patch_halfwidth -- required for a "
            "mesh-convergence study, and NOT beam_3d-equivalent unless it "
            "happens to select the same facets"
            if halfwidth_override is not None else "auto max(0.5, 1.5h)",
            area, traction_z, _REFERENCE_LOAD_TOTAL_FORCE,
        )

    if comm.rank == 0:
        logger.info(
            "Built box-mesh fenitop dicts (beam_3d.py reference): "
            "cell_type=%s, domain=%s, refinement=%.4g -> elements=%s, h=%.4g, "
            "R=%.4g (R/h=%.3g, FIXED across levels), E=%.6g, nu=%.6g, "
            "vol_frac=%.3g",
            cell_type, _DOMAIN, refinement, elements, element_size,
            _FILTER_RADIUS, _FILTER_RADIUS / element_size,
            fem["young's modulus"], fem["poisson's ratio"], opt["vol_frac"],
        )

    return tagged_mesh, fem, opt, load_cases