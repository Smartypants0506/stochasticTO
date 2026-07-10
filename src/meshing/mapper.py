"""
src/meshing/mapper.py

Stage 1 (BC / Zone Mapping) — masterContext.md Section 3.2.
Converts tagged mesh entities (from mesher.py) into the membership
functions FEniTop's fem.py expects for `disp_bc`, `traction_bcs`,
`solid_zone`, and `void_zone`.

This module has no Gmsh dependency -- it operates purely on the dolfinx
Mesh/MeshTags objects produced by mesher.import_to_dolfinx(), keeping the
Gmsh-specific logic isolated to mesher.py per the module-separation rule.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from src.meshing.mesher import TaggedMesh

logger = logging.getLogger(__name__)

MembershipFn = Callable[[np.ndarray], np.ndarray]


def facet_group_points(tagged_mesh: TaggedMesh, group_name: str) -> np.ndarray:
    """Return the [N x 3] coordinates of mesh vertices on facets tagged
    with the given physical group name (e.g. "fixed", "load_1")."""
    mesh = tagged_mesh.mesh
    name_to_tag = tagged_mesh.name_to_tag
    if group_name not in name_to_tag:
        return np.empty((0, 3))

    tag = name_to_tag[group_name]
    facet_indices = tagged_mesh.facet_tags.find(tag)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f_to_v = mesh.topology.connectivity(fdim, 0)

    if len(facet_indices) == 0:
        return np.empty((0, 3))

    vertex_ids = np.unique(np.hstack([f_to_v.links(f) for f in facet_indices]))
    return mesh.geometry.x[vertex_ids]


def make_membership_fn(points: np.ndarray, tol: float) -> MembershipFn:
    """Build a `lambda x: bool_mask` membership function from a KDTree
    over reference points, matching FEniTop's expected `disp_bc`/
    `traction_bcs` callable signature (x is [3 x N])."""
    if len(points) == 0:
        return lambda x: np.full(x.shape[1], False)
    tree = cKDTree(points)

    def fn(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x.T, k=1)
        return dist < tol
    return fn


def make_solid_zone_from_cells(tagged_mesh: TaggedMesh, group_name: str) -> MembershipFn:
    """Build a solid_zone(x) membership function from actual tagged cell
    centroids (volume-based), not coordinate proximity -- mirrors the
    fenitop GE-bracket demo's "bolts" volume treatment exactly."""
    mesh = tagged_mesh.mesh
    tag = tagged_mesh.name_to_tag.get(group_name)
    if tag is None:
        return lambda x: np.full(x.shape[1], False)

    cell_indices = tagged_mesh.cell_tags.find(tag)
    if len(cell_indices) == 0:
        return lambda x: np.full(x.shape[1], False)

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c_to_v = mesh.topology.connectivity(tdim, 0)
    centroids = np.array([
        mesh.geometry.x[c_to_v.links(c)].mean(axis=0) for c in cell_indices
    ])
    tree = cKDTree(centroids)

    def fn(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x.T, k=1)
        return dist < 1e-9  # centroids match exactly (same DG0 dof coords)
    return fn


@dataclass
class BoundaryConditions:
    """Bundled BC/zone functions ready to drop into FEniTop's fem_config
    and opt_config dicts (see fem-11.py's `form_fem` signature)."""
    disp_bc: MembershipFn
    traction_bcs: list[list]  # [[load_vector, membership_fn], ...]
    solid_zone: MembershipFn
    void_zone: MembershipFn


def build_boundary_conditions(
    tagged_mesh: TaggedMesh,
    load_vectors: dict[str, tuple[float, float, float]],
    snap_tol: float = 1e-4,
) -> BoundaryConditions:
    """Build the full BoundaryConditions bundle from a tagged mesh.

    Parameters
    ----------
    load_vectors:
        Maps physical group name (e.g. "load_1", "load_2") to its applied
        traction vector. Must match names tagged in mesher.tag_physical_groups().
    snap_tol:
        Distance tolerance (metres) for matching mesh points to tagged facets.
    """
    fixed_pts = facet_group_points(tagged_mesh, "fixed")
    disp_bc = make_membership_fn(fixed_pts, snap_tol)

    traction_bcs = []
    for name, vector in load_vectors.items():
        pts = facet_group_points(tagged_mesh, name)
        if len(pts) == 0:
            logger.warning("No points found for load group '%s'; skipping.", name)
            continue
        traction_bcs.append([vector, make_membership_fn(pts, snap_tol)])

    solid_zone = make_solid_zone_from_cells(tagged_mesh, "solid")
    void_zone = lambda x: np.full(x.shape[1], False)  # no default void region

    return BoundaryConditions(
        disp_bc=disp_bc, traction_bcs=traction_bcs,
        solid_zone=solid_zone, void_zone=void_zone,
    )