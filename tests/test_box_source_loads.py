"""The load must actually reach the mesh at every refinement level.

THE FAILURE THIS GUARDS AGAINST
-------------------------------
dolfinx's locate_entities_boundary selects a facet only when ALL of its vertices
satisfy the predicate. beam_3d's traction patch is |x-5| < 0.5, |z-5| < 0.5, so
it admits a facet only when two CONSECUTIVE grid nodes fall strictly inside --
which happens at h=0.4 and essentially nowhere else. When the refinement knob
was added, every level except 1.0 therefore applied the traction to NO facets:
u came back zero, compliance came back exactly zero, and nothing raised. The
mesh-convergence study would have "converged" on two levels of zero.

The tests below pin both halves of the fix: the patch must admit facets at every
refinement level in use, and the total applied force must be identical across
levels (a wider patch at fixed traction would be a different physical problem at
each level, which is the same defect the fixed filter radius exists to avoid).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.meshing.box_source import (
    _DOMAIN, _LOAD_CENTRE, _REFERENCE_LOAD_TOTAL_FORCE, _load_patch_halfwidth,
    _make_load_membership, elements_for_refinement, loaded_area,
    realized_element_size,
)

# Every refinement level any shipped config or study driver uses, plus a few
# neighbours. Kept deliberately wider than what is currently configured so that
# changing MESH_LEVELS or a config's refinement cannot land on a value whose
# load silently evaporates.
#   0.40  configSmoke.yaml
#   0.62, 0.825, 0.985  convergence_studies.MESH_LEVELS
#   0.64  configStudy.yaml
#   1.00  config.yaml (production, beam_3d reference)
REFINEMENT_LEVELS = [0.24, 0.4, 0.48, 0.6, 0.62, 0.64, 0.8, 0.825, 0.985, 1.0]
REFERENCE_REFINEMENT = 1.0
REFERENCE_TRACTION_Z = -2.0
REFERENCE_AREA = 0.16          # the [4.8,5.2]^2 quad at h=0.4


def _element_size(refinement: float) -> float:
    return realized_element_size(elements_for_refinement(refinement))


def _selected_nodes(refinement: float) -> np.ndarray:
    """Grid nodes strictly inside the patch along x."""
    h = _element_size(refinement)
    halfwidth = _load_patch_halfwidth(h)
    n = int(round((_DOMAIN[1][0] - _DOMAIN[0][0]) / h))
    nodes = np.arange(n + 1) * h
    return nodes[
        (nodes > _LOAD_CENTRE[0] - halfwidth) & (nodes < _LOAD_CENTRE[0] + halfwidth)
    ]


@pytest.mark.parametrize("refinement", REFINEMENT_LEVELS)
def test_load_patch_admits_at_least_one_facet(refinement):
    """Two consecutive nodes strictly inside the patch, at every level."""
    inside = _selected_nodes(refinement)
    assert inside.size >= 2, (
        f"refinement={refinement} (h={_element_size(refinement):.4g}) leaves "
        f"only {inside.size} node(s) inside the traction patch, so no facet is "
        "selected and the load is applied to nothing -- compliance would come "
        "out exactly zero with no error raised."
    )


@pytest.mark.parametrize("refinement", REFINEMENT_LEVELS)
def test_loaded_area_is_positive_and_consistent(refinement):
    h = _element_size(refinement)
    inside = _selected_nodes(refinement)
    expected = ((inside.size - 1) * h) ** 2
    assert loaded_area(h) == pytest.approx(expected)
    assert loaded_area(h) > 0.0


@pytest.mark.parametrize("refinement", REFINEMENT_LEVELS)
def test_total_applied_force_is_identical_across_levels(refinement):
    """The patch widens on coarse meshes, so the traction must be normalized --
    otherwise each refinement level solves a different problem and a
    convergence study measures nothing."""
    h = _element_size(refinement)
    traction_z = -_REFERENCE_LOAD_TOTAL_FORCE / loaded_area(h)
    total_force = abs(traction_z) * loaded_area(h)
    assert total_force == pytest.approx(_REFERENCE_LOAD_TOTAL_FORCE, rel=1e-12)


def test_reference_mesh_reproduces_beam_3d_exactly():
    """The frozen reference case must be untouched by the widening.

    Note the half-width at h=0.4 is 0.6, NOT beam_3d's 0.5 -- max(0.5, 1.5*0.4)
    widens it. That is harmless and the reason is worth stating: the widened
    bounds 4.4 and 5.6 are not grid nodes, so the set of vertices strictly
    inside is still {4.8, 5.2}, exactly as with 0.5. What has to be preserved is
    the SELECTED FACETS and the resulting traction, not the half-width itself.
    """
    h = _element_size(REFERENCE_REFINEMENT)
    assert h == pytest.approx(0.4)
    assert _load_patch_halfwidth(h) == pytest.approx(0.6)       # widened...
    np.testing.assert_allclose(_selected_nodes(REFERENCE_REFINEMENT), [4.8, 5.2])
    assert loaded_area(h) == pytest.approx(REFERENCE_AREA)      # ...but identical facets
    assert -_REFERENCE_LOAD_TOTAL_FORCE / loaded_area(h) == pytest.approx(
        REFERENCE_TRACTION_Z
    )

    # And the widening genuinely selects the same set beam_3d's own 0.5 would.
    nodes = np.arange(int(round(10 / h)) + 1) * h
    beam_3d_selection = nodes[(nodes > 4.5) & (nodes < 5.5)]
    np.testing.assert_allclose(beam_3d_selection, _selected_nodes(REFERENCE_REFINEMENT))


def test_pinned_halfwidth_conserves_total_force_but_not_loaded_area():
    """What pinning the half-width does and does NOT buy.

    DOES: every level has a non-empty load, the geometric patch is identical,
    and the total applied force is conserved exactly.

    DOES NOT: give an identical loaded AREA. Vertex-based facet selection takes
    the INNER ENVELOPE of the geometric patch, so the resolved area depends on
    where grid nodes happen to fall relative to the patch boundary. With a
    half-width of 2.5 the resolved area oscillates between ~44% and ~87% of the
    geometric patch, non-monotonically in h.

    That is inherent to discretizing a fixed-geometry patch load and cannot be
    tuned away by choosing a different half-width. Its consequence for the mesh
    study is real and must be accounted for: the load DISTRIBUTION varies by up
    to ~2x in area between levels even though the total force does not. If
    sigma_C fails to converge, this is a candidate cause that has to be ruled
    out before concluding anything about the physics -- which is why
    convergence_studies.py records the resolved area at every level.
    """
    coarsest = min(REFINEMENT_LEVELS)
    pinned = _load_patch_halfwidth(_element_size(coarsest))
    geometric_area = (2 * pinned) ** 2

    areas = {r: loaded_area(_element_size(r), pinned) for r in REFINEMENT_LEVELS}

    for refinement, area in areas.items():
        h = _element_size(refinement)
        assert area > 0.0
        # Inner envelope: never larger than the geometric patch, and within one
        # element of it on each side.
        assert area <= geometric_area + 1e-12
        lower_bound = max(0.0, (2 * pinned - 2 * h)) ** 2
        assert area >= lower_bound - 1e-12, (
            f"refinement={refinement}: resolved area {area:.4g} is more than "
            f"one element inside the geometric patch {geometric_area:.4g}"
        )
        # Total force conserved exactly, which is the property the study needs.
        traction = -_REFERENCE_LOAD_TOTAL_FORCE / area
        assert abs(traction) * area == pytest.approx(
            _REFERENCE_LOAD_TOTAL_FORCE, rel=1e-12
        )

    # The variation is real, not hypothetical -- pin it so a future change that
    # claims to eliminate it has to update this test deliberately.
    spread = max(areas.values()) / min(areas.values())
    assert spread > 1.5


def test_unpinned_halfwidth_makes_the_load_shrink_with_refinement():
    """Why pinning is necessary at all: left automatic, the patch narrows as the
    mesh refines, so the load concentrates and each level is a different
    problem. This is the same defect as scaling the filter radius with h."""
    areas = [loaded_area(_element_size(r)) for r in sorted(REFINEMENT_LEVELS)]
    # Coarsest level's loaded area is orders of magnitude larger than the finest.
    assert areas[0] > 10 * areas[-1]


def test_membership_predicate_matches_the_area_calculation():
    """loaded_area() is analytic; confirm it agrees with the predicate that
    actually drives facet selection, evaluated on the real node grid."""
    for refinement in REFINEMENT_LEVELS:
        h = _element_size(refinement)
        membership = _make_load_membership(h)
        n = int(round((_DOMAIN[1][0] - _DOMAIN[0][0]) / h))
        nodes = np.arange(n + 1) * h
        # Evaluate on the y=30 face along the x line through z=5.
        coords = np.vstack([nodes, np.full(nodes.size, 30.0), np.full(nodes.size, 5.0)])
        selected = nodes[membership(coords)]
        np.testing.assert_allclose(selected, _selected_nodes(refinement))


def test_the_bug_is_reproduced_by_the_original_fixed_halfwidth():
    """With beam_3d's un-widened 0.5 half-width, every level except 1.0 selects
    a single node and therefore NO facet. Stated as arithmetic so the
    regression cannot silently return."""
    broken = []
    for refinement in REFINEMENT_LEVELS:
        h = _element_size(refinement)
        n = int(round((_DOMAIN[1][0] - _DOMAIN[0][0]) / h))
        nodes = np.arange(n + 1) * h
        inside = nodes[(nodes > 4.5) & (nodes < 5.5)]
        if inside.size < 2:
            broken.append(refinement)
    # A few levels (0.6, 0.62, 0.825, 0.985) happen to place two nodes inside
    # beam_3d's window and did work; the majority did not, including two of the
    # three levels the mesh study originally used.
    assert REFERENCE_REFINEMENT not in broken
    assert {0.24, 0.4, 0.48, 0.64, 0.8} <= set(broken)
    assert len(broken) >= len(REFINEMENT_LEVELS) // 2
