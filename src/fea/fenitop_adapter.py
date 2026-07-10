"""
src/fea/fenitop_adapter.py

Thin wrapper around FEniTop's src.fenitop.fem.form_fem and
src.fenitop.topopt.topopt. Converts this project's ProjectConfig +
meshing outputs (TaggedMesh, BoundaryConditions) into the exact `fem`
and `opt` dict schema FEniTop expects -- confirmed against fem-11.py's
form_fem() signature and topopt-17.py's topopt() signature.

Per masterContext.md Section 3.1: "Custom glue code retained: hooking
FEniTop's eta parameter to the randomized eta(x) field" -- but that
happens in src/topology/heaviside_projection_glue.py, NOT here. This
adapter only prepares the deterministic fem/opt dicts; it never touches
FEniTop's internal solver/filter/projection code.
"""
from __future__ import annotations

import logging

from src.config.schema import ProjectConfig
from src.meshing.mapper import BoundaryConditions
from src.meshing.mesher import TaggedMesh
from src.meshing.mesher import extract_simplices

from src.random_fields.kernel import KernelParams
from src.random_fields.threshold_transform import MarginalTransformParams

logger = logging.getLogger(__name__)


def build_fem_dict(tagged_mesh: TaggedMesh, bc: BoundaryConditions,
                    config: ProjectConfig) -> dict:
    """Build the `fem` dict exactly as form_fem() in fem-11.py expects it.

    Required keys per form_fem(): "mesh", "young's modulus", "poisson's
    ratio", "disp_bc", "traction_bcs", "body_force", "quadrature_degree",
    "petsc_options". "mesh_serial" is additionally required by topopt()
    (topopt-17.py) for plotting, so it is included here too.
    """
    traction_bcs = [[list(lc.vector), fn]
                     for lc, fn in zip(config.load_cases,
                                        [t[1] for t in bc.traction_bcs])]

    
    return {
        "mesh": tagged_mesh.mesh,
        "mesh_serial": tagged_mesh.mesh_serial,
        "mesh_simplices": extract_simplices(tagged_mesh),
        "young's modulus": config.material.youngs_modulus,
        "poisson's ratio": config.material.poissons_ratio,
        "disp_bc": bc.disp_bc,
        "traction_bcs": traction_bcs,
        "body_force": config.optimization.body_force,
        "quadrature_degree": config.optimization.quadrature_degree,
        "petsc_options": config.petsc.to_options_dict(),
    }


def build_opt_dict(bc: BoundaryConditions, config: ProjectConfig, kl_result: "KLExpansionResult") -> dict:
    """Build the `opt` dict exactly as form_fem()/topopt() expect it, extended
    with the Stage 3/4/5 robust-loop keys consumed by
    src/optimization/dolfiny_mma_driver.py.

    Required keys per form_fem(): "penalty", "epsilon", "opt_compliance"
    (plus "in_spring"/"out_spring" only if opt_compliance is False --
    not wired here since this project's verification/robust-TO paths are
    compliance-based). Required keys per topopt() additionally: "filter_radius",
    "beta_interval", "beta_max", "use_oc", "move", "vol_frac", "opt_tol",
    "max_iter", "solid_zone", "void_zone".

    Additional keys required by src/optimization/dolfiny_mma_driver.py's
    robust loop: "kernel_params" (KernelParams, matching kernel.py's real
    sigma/length_scale/spatial_dim fields), "transform_params"
    (MarginalTransformParams, matching threshold_transform.py's real
    eta_min/eta_max/alpha/beta fields), "kl_variance_threshold",
    "random_field_seed", "pce_n_train", "n_kl", "pce_hyperbolic_q",
    "pce_max_degree", "pce_q2_threshold", "pce_refresh_interval" -- all
    sourced from config.random_field / config.surrogate / config.optimization.

    Raises:
        NotImplementedError: If opt_compliance is False (compliant-mechanism
            mode requires in_spring/out_spring config fields not yet wired
            in this adapter -- do not silently fall back to compliance mode).
    """
    opt_cfg = config.optimization
    rf_cfg = config.random_field
    surrogate_cfg = config.surrogate

    if not opt_cfg.opt_compliance:
        raise NotImplementedError(
            "Compliant-mechanism mode (opt_compliance=False) requires "
            "in_spring/out_spring config fields not yet wired in this "
            "adapter -- do not silently fall back to compliance mode."
        )

    return {
        # FEniTop nominal-SIMP keys (form_fem() / topopt() contract)
        "max_iter": opt_cfg.max_iter,
        "opt_tol": opt_cfg.opt_tol,
        "vol_frac": opt_cfg.vol_frac,
        "solid_zone": bc.solid_zone,
        "void_zone": bc.void_zone,
        "penalty": opt_cfg.penalty,
        "epsilon": opt_cfg.epsilon,
        "filter_radius": opt_cfg.filter_radius,
        "beta_interval": opt_cfg.beta_interval,
        "kl_model": kl_result,
        "beta_max": opt_cfg.beta_max,
        "use_oc": opt_cfg.use_oc,
        "move": opt_cfg.move,
        "opt_compliance": opt_cfg.opt_compliance,

        # Stage 3 random-field keys (RandomFieldHeaviside / KL expansion)
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

        # Stage 4 PCE surrogate keys
        "pce_n_train": surrogate_cfg.n_train,
        "pce_n_test": surrogate_cfg.n_test,
        "pce_hyperbolic_q": surrogate_cfg.hyperbolic_q,
        "pce_max_degree_attempts": surrogate_cfg.max_degree,
        "pce_q2_threshold": surrogate_cfg.q2_threshold,

        # Stage 5 robust-loop scheduling key
        "pce_refresh_interval": opt_cfg.pce_refresh_interval,
    }


def build_fenitop_dicts(tagged_mesh: TaggedMesh, bc: BoundaryConditions,
                         config: ProjectConfig) -> tuple[dict, dict]:
    """Single entry point: returns (fem, opt) dicts ready for
    src.fenitop.fem.form_fem() or src.fenitop.topopt.topopt()."""
    fem_dict = build_fem_dict(tagged_mesh, bc, config)
    opt_dict = build_opt_dict(bc, config, kl_result)
    logger.info(
        "Built FEniTop dicts: E=%.3g, nu=%.3g, vol_frac=%.3g, %d traction BCs",
        fem_dict["young's modulus"], fem_dict["poisson's ratio"],
        opt_dict["vol_frac"], len(fem_dict["traction_bcs"]),
    )
    return fem_dict, opt_dict