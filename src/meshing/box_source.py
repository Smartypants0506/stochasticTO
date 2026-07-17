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
_ELEMENTS = [25, 75, 25]
_YOUNGS_MODULUS = 100
_POISSONS_RATIO = 0.25
_VOL_FRAC = 0.08
_PENALTY = 3.0
_EPSILON = 1e-6
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
_PETSC_OPTIONS = {
    "ksp_type": "cg",
    "pc_type": "gamg",
    "ksp_max_it": 2000,
    "ksp_converged_reason": None,     # PETSc bool-flag options: value is ignored, just needs the key present
    "ksp_error_if_not_converged": None,
}

_CELL_TYPES = {
    "tetrahedron": CellType.tetrahedron,
    "hexahedron": CellType.hexahedron,
}


def _disp_bc(x):
    """Identical to beam_3d.py's fem['disp_bc']: clamp the two short edges
    at y=0."""
    return np.isclose(x[1], 0) & (np.less(x[0], 1.5) | np.greater(x[0], 8.5))


def _load_membership(x):
    """Identical to beam_3d.py's traction_bcs membership fn: a small patch
    at y=30, centered at x=5, z=5."""
    return (np.isclose(x[1], 30)
            & np.greater(x[0], 4.5) & np.less(x[0], 5.5)
            & np.greater(x[2], 4.5) & np.less(x[2], 5.5))


def _solid_zone(x):
    """beam_3d.py has no protected solid regions."""
    return np.full(x.shape[1], False)


def _void_zone(x):
    """beam_3d.py has no forced-void regions."""
    return np.full(x.shape[1], False)


def build_box_mesh(comm: MPI.Comm, cell_type: str):
    """Recreate beam_3d.py's `mesh` + `mesh_serial` exactly (module-level
    call in beam_3d_DOMAIN = [[0, 0, 0], [10, 30, 10]]
_ELEMENTS = [25, 75, 25].py; wrapped in a function here so main.py controls
    when/whether it runs)."""
    if cell_type not in _CELL_TYPES:
        raise ValueError(
            f"box_mesh.cell_type={cell_type!r} not supported; expected one "
            f"of {list(_CELL_TYPES)}"
        )
    dolfinx_cell_type = _CELL_TYPES[cell_type]

    mesh = create_box(comm, _DOMAIN, _ELEMENTS, dolfinx_cell_type)
    if comm.rank == 0:
        mesh_serial = create_box(MPI.COMM_SELF, _DOMAIN, _ELEMENTS, dolfinx_cell_type)
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
    mesh, mesh_serial = build_box_mesh(comm, cell_type)

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
    }

    rf_cfg = config.random_field
    surrogate_cfg = config.surrogate
    # pce_refresh_interval is a Stage 5 scheduling knob specific to this
    # project's robust loop -- no beam_3d.py equivalent -- so it's sourced
    # from cfg like the other Stage 3-6 params, not hardcoded.
    pce_refresh_interval = config.optimization.pce_refresh_interval

    opt = {
        "max_iter": _MAX_ITER,
        "opt_tol": _OPT_TOL,
        "vol_frac": _VOL_FRAC,
        "solid_zone": _solid_zone,
        "void_zone": _void_zone,
        "penalty": _PENALTY,
        "epsilon": _EPSILON,
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
    }

    # Vector kept as a plain tuple (not np.array) to match beam_3d.py's
    # own traction_bcs entry verbatim: [(0, 0, -2.0), lambda ...].
    load_cases = {
        "beam_3d_reference": [((0.0, 0.0, -2.0), _load_membership)],
    }

    if comm.rank == 0:
        logger.info(
            "Built box-mesh fenitop dicts (beam_3d.py reference): "
            "cell_type=%s, domain=%s, elements=%s, E=%.6g, nu=%.6g, "
            "vol_frac=%.3g",
            cell_type, _DOMAIN, _ELEMENTS,
            fem["young's modulus"], fem["poisson's ratio"], opt["vol_frac"],
        )

    return tagged_mesh, fem, opt, load_cases