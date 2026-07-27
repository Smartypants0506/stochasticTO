"""The config must describe the run it configures.

The motivating defect: with mesh_source "box", box_source.py hardcodes the
beam_3d physics and IGNORES config.yaml's vol_frac, filter_radius, material and
load_cases. Those values sat in the file looking authoritative, so anyone
reconstructing the study from it would have reported the wrong number for
essentially every parameter. The loader now rejects them, and these tests pin
that behaviour down -- including that the shipped configs actually load.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigError, load_config

MINIMAL_BOX = {
    "mesh_source": "box",
    "box_mesh": {"cell_type": "tetrahedron", "refinement": 0.64},
    "optimization": {
        "max_iter": 20,
        "epsilon": 1.0e-4,
        "robust_method": "saa",
        "saa_n_samples": 16,
        "saa_seed": 7,
        "saa_beta": 8.0,
        "saa_beta_max": 128.0,
        "lambda_sweep": [0.0, 1.0],
    },
    "random_field": {
        "sigma": 1.0, "length_scale": 4.0, "spatial_dim": 3,
        "eta_min": 0.25, "eta_max": 0.75, "alpha": 2.0, "beta": 2.0,
        "variance_threshold": 0.95, "seed": 42,
    },
    "mc_validation": {"n_samples": 64, "beta": 128.0, "seed": 42},
}


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_minimal_box_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL_BOX))
    assert cfg.mesh_source == "box"
    assert cfg.box_mesh.refinement == 0.64
    # STEP-only fields are absent, not defaulted to something misleading.
    assert cfg.material is None
    assert cfg.load_cases == {}


@pytest.mark.parametrize(
    "key,value",
    [("vol_frac", 0.15), ("filter_radius", 0.006), ("penalty", 4.0), ("move", 0.05)],
)
def test_inert_optimization_keys_are_rejected_on_the_box_path(tmp_path, key, value):
    payload = {**MINIMAL_BOX, "optimization": {**MINIMAL_BOX["optimization"], key: value}}
    with pytest.raises(ConfigError, match="hardcoded"):
        load_config(_write(tmp_path, payload))


@pytest.mark.parametrize("key", ["step_file", "mesh_size_max", "material", "load_cases"])
def test_step_only_keys_are_rejected_on_the_box_path(tmp_path, key):
    payload = {**MINIMAL_BOX, key: {"dummy": 1} if key in ("material", "load_cases") else 1}
    with pytest.raises(ConfigError, match="mesh_source='step'"):
        load_config(_write(tmp_path, payload))


def test_unknown_keys_are_rejected(tmp_path):
    payload = {**MINIMAL_BOX, "recompute": {"stage1_mesh": False}}
    with pytest.raises(ConfigError, match="Unknown top-level config key"):
        load_config(_write(tmp_path, payload))


def test_seed_collision_between_optimization_and_validation_is_rejected(tmp_path):
    """Sharing a seed would validate the design on the very samples it was
    optimized against, hiding the SAA overfitting the check exists to find."""
    payload = {**MINIMAL_BOX,
               "optimization": {**MINIMAL_BOX["optimization"], "saa_seed": 42}}
    with pytest.raises(ConfigError, match="disjoint seeds"):
        load_config(_write(tmp_path, payload))


def test_validation_beta_must_match_the_final_optimization_beta(tmp_path):
    """Validating a beta=128 design at beta=8 measures a different structure
    than the one that was designed."""
    payload = {**MINIMAL_BOX,
               "mc_validation": {**MINIMAL_BOX["mc_validation"], "beta": 8.0}}
    with pytest.raises(ConfigError, match="projection\\s+sharpness"):
        load_config(_write(tmp_path, payload))


def test_single_point_lambda_sweep_is_rejected(tmp_path):
    payload = {**MINIMAL_BOX,
               "optimization": {**MINIMAL_BOX["optimization"], "lambda_sweep": [1.0]}}
    with pytest.raises(ConfigError, match="at least 2"):
        load_config(_write(tmp_path, payload))


def test_inverted_eta_band_is_rejected(tmp_path):
    payload = {**MINIMAL_BOX,
               "random_field": {**MINIMAL_BOX["random_field"],
                                "eta_min": 0.75, "eta_max": 0.25}}
    with pytest.raises((ConfigError, ValueError)):
        load_config(_write(tmp_path, payload))


SHIPPED_CONFIGS = [
    "src/config/config.yaml",
    "src/config/configStudy.yaml",
    "src/config/configSmoke.yaml",
]


@pytest.mark.parametrize("path", SHIPPED_CONFIGS)
def test_shipped_configs_load(path):
    """Every config in the repository must satisfy the loader it ships with."""
    cfg = load_config(path)
    assert cfg.mesh_source in ("box", "step")
    assert cfg.mc_validation.beta == cfg.optimization.saa_beta_max
    assert cfg.mc_validation.seed != cfg.optimization.saa_seed


# At least this many CG1 nodes must span the filter's interface transition for
# an intermediate density -- and hence a responsive projection -- to exist at
# all. The transition is O(R) wide, so it contains roughly R/h nodes.
_MIN_R_OVER_H = 0.9
# tanh saturates to +/-1 in double precision at an argument of about 19.
_TANH_SATURATION_ARGUMENT = 19.0


@pytest.mark.parametrize("path", SHIPPED_CONFIGS)
def test_shipped_configs_are_not_projection_degenerate(path):
    """Guards the failure that killed a smoke run: a mesh too coarse to resolve
    the filter, combined with a beta sharp enough to saturate the projection.

    tanh(beta*(rho_tilde - eta)) is +/-1 to machine precision once
    |rho_tilde - eta| > ~19/beta. If NO node's filtered density lands inside
    that window around the eta band, every eta draw returns identical
    compliance, sigma_C is exactly zero, and dsigma_C/drho divides by zero.

    A config is safe if EITHER
      * the mesh resolves the filter (R/h >= 0.9), so a real interface band of
        intermediate densities exists for eta to act on, OR
      * beta is soft enough that the responsive window covers the whole [0, 1]
        range regardless of the mesh.

    configSmoke.yaml deliberately takes the second route: it keeps a very coarse
    mesh (cheap KL eigensolve) and caps beta_max instead.
    """
    from src.meshing.box_source import (
        _FILTER_RADIUS, elements_for_refinement, realized_element_size,
    )

    cfg = load_config(path)
    if cfg.mesh_source != "box":
        pytest.skip("filter/mesh ratio is a box-path property")

    element_size = realized_element_size(
        elements_for_refinement(cfg.box_mesh.refinement)
    )
    r_over_h = _FILTER_RADIUS / element_size

    saturation_halfwidth = _TANH_SATURATION_ARGUMENT / cfg.optimization.saa_beta_max
    responsive_low = cfg.random_field.eta_min - saturation_halfwidth
    responsive_high = cfg.random_field.eta_max + saturation_halfwidth
    beta_covers_everything = responsive_low <= 0.0 and responsive_high >= 1.0

    assert r_over_h >= _MIN_R_OVER_H or beta_covers_everything, (
        f"{path} is projection-degenerate: R/h = {r_over_h:.3g} (< "
        f"{_MIN_R_OVER_H}) so the mesh does not resolve the filter, AND "
        f"beta_max = {cfg.optimization.saa_beta_max:g} only makes rho_tilde in "
        f"[{responsive_low:.3g}, {responsive_high:.3g}] responsive to eta. "
        "Every eta draw would give the same compliance. Refine the mesh or "
        "lower saa_beta_max."
    )


def test_solver_failure_tolerance_tightens_toward_production():
    """A failed solve means that manufacturing realization does not carry load.
    It is a robustness RESULT and is always reported, but it also makes the
    compliance statistics conditional on survival -- so the tolerance must be
    loosest on the throwaway smoke mesh and tightest in production, never the
    other way round.
    """
    rates = {
        path: load_config(path).mc_validation.max_solver_failure_rate
        for path in SHIPPED_CONFIGS
    }
    smoke = rates["src/config/configSmoke.yaml"]
    study = rates["src/config/configStudy.yaml"]
    production = rates["src/config/config.yaml"]

    assert production <= study <= smoke, (
        f"failure tolerance must tighten toward production, got "
        f"production={production}, study={study}, smoke={smoke}"
    )
    # Production must not quietly tolerate a large fraction of the ensemble
    # failing: at that point the compliance numbers describe a different,
    # self-selected population of designs.
    assert production <= 0.02
    # The smoke tier must tolerate the rate actually observed on its mesh
    # (2/32 = 6.25%), or the smoke test can never complete.
    assert smoke >= 2 / 32


def test_the_configuration_that_actually_crashed_is_detected():
    """The exact combination from the failed run -- refinement 0.24 (R/h = 0.36)
    with beta_max = 128 -- must be recognized as degenerate. This is the
    regression, stated as arithmetic rather than as a comment."""
    from src.meshing.box_source import (
        _FILTER_RADIUS, elements_for_refinement, realized_element_size,
    )

    element_size = realized_element_size(elements_for_refinement(0.24))
    r_over_h = _FILTER_RADIUS / element_size
    assert r_over_h == pytest.approx(0.36, abs=0.01)

    saturation_halfwidth = _TANH_SATURATION_ARGUMENT / 128.0
    assert saturation_halfwidth < 0.16
    # Responsive window [0.25-0.148, 0.75+0.148] = [0.10, 0.90] -- excludes the
    # near-binary densities such a mesh actually produces.
    assert 0.25 - saturation_halfwidth > 0.0
    assert r_over_h < _MIN_R_OVER_H
