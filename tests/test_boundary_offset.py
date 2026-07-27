"""The boundary-offset arithmetic that decided the eta band.

config.yaml's eta-band table is the single most consequential piece of reasoning
in the project -- it is why the band moved from [0.45, 0.55] to [0.25, 0.75].
These tests pin the arithmetic behind it so a later edit cannot quietly change
the conclusion, and check that the empirical estimator inverts the geometry it
claims to.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.random_fields.threshold_transform import MarginalTransformParams
from src.validation.boundary_offset import (
    RESOLVABLE_OFFSET_STD_IN_ELEMENTS, _beta_std, measure_offset_from_volumes,
)


def test_beta_std_matches_the_closed_form():
    """Beta(2,2) has variance 1/20, so std = 0.2236 on a unit band."""
    assert _beta_std(2.0, 2.0, 1.0) == pytest.approx(np.sqrt(0.05))
    assert _beta_std(2.0, 2.0, 0.1) == pytest.approx(0.02236068, rel=1e-6)
    assert _beta_std(2.0, 2.0, 0.5) == pytest.approx(0.1118034, rel=1e-6)


@pytest.mark.parametrize(
    "band,expected_offset_elements",
    [
        (0.10, 0.080),   # eta in [0.45, 0.55] -- the old, unresolvable setting
        (0.30, 0.240),   # eta in [0.35, 0.65]
        (0.50, 0.399),   # eta in [0.25, 0.75] -- current
    ],
)
def test_offset_table_in_config_is_reproduced(band, expected_offset_elements):
    """Reproduces config.yaml's table: displacement std in elements, using the
    1-D step estimate |grad rho_tilde| = 1/(2R) at R = 0.6, h = 0.4.

    If this changes, the justification written into config.yaml is stale and the
    band decision has to be revisited."""
    filter_radius, element_size = 0.6, 0.4
    # The band-averaged gradient is a little below the level-set value; the
    # config table uses 0.7 as the representative figure.
    representative_gradient = 0.7
    offset_std = _beta_std(2.0, 2.0, band) / representative_gradient
    assert offset_std / element_size == pytest.approx(expected_offset_elements, abs=0.005)


def test_old_band_is_below_the_resolvability_threshold_and_new_one_is_above():
    """The finding, as an assertion."""
    element_size, gradient = 0.4, 0.7
    old = _beta_std(2.0, 2.0, 0.10) / gradient / element_size
    new = _beta_std(2.0, 2.0, 0.50) / gradient / element_size
    assert old < RESOLVABLE_OFFSET_STD_IN_ELEMENTS
    assert new > RESOLVABLE_OFFSET_STD_IN_ELEMENTS


def test_mesh_refinement_cannot_rescue_the_old_band():
    """The displacement is fixed in ABSOLUTE units, so reaching 0.5h at the old
    band needs an unaffordable mesh. This is why 'just refine' was the wrong
    answer."""
    gradient = 0.7
    offset_std = _beta_std(2.0, 2.0, 0.10) / gradient
    required_h = offset_std / 0.5
    refinement_factor = 0.4 / required_h
    assert refinement_factor > 6.0
    assert 46875 * refinement_factor ** 3 > 1.0e7      # elements, i.e. unaffordable


def test_empirical_estimator_inverts_the_geometry():
    """A boundary displaced by d_s over area A changes the volume by A*d_s, so
    feeding synthetic volumes back through must recover d_s."""
    total_volume, interface_area, element_size = 3000.0, 250.0, 0.4
    nominal_fraction = 0.08
    displacements = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    volumes = nominal_fraction + displacements * interface_area / total_volume

    report = measure_offset_from_volumes(
        volumes, nominal_fraction, total_volume, interface_area, element_size
    )
    assert report["mean_absolute"] == pytest.approx(0.0, abs=1e-12)
    assert report["std_absolute"] == pytest.approx(displacements.std(ddof=1))
    assert report["range_elements"] == pytest.approx(0.4 / element_size)


def test_offset_is_reported_against_both_scales():
    """A band can be resolvable by the mesh yet implausible as a tolerance --
    which is exactly the [0.25, 0.75] situation -- so both denominators have to
    be reported."""
    filter_radius, element_size = 0.6, 0.4
    offset_std = _beta_std(2.0, 2.0, 0.5) / 0.7
    in_elements = offset_std / element_size
    as_feature_fraction = offset_std / (2 * filter_radius)
    assert in_elements > RESOLVABLE_OFFSET_STD_IN_ELEMENTS    # numerically fine
    assert as_feature_fraction > 0.10                          # physically coarse
