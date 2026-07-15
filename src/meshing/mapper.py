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
from mpi4py import MPI
from scipy.spatial import cKDTree

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from src.meshing.mesher import TaggedMesh, extract_simplices

logger = logging.getLogger(__name__)

MembershipFn = Callable[[np.ndarray], np.ndarray]


def facet_group_points(tagged_mesh: TaggedMesh, group_name: str,
                        comm: MPI.Comm | None = None) -> np.ndarray:
    """Return the [N x 3] coordinates of vertices on facets tagged with the
    given physical group name (e.g. "fixed", "load_1").

    Reads from `tagged_mesh.mesh_serial`/`facet_tags_serial` (the full,
    unpartitioned COMM_SELF mesh, populated only on rank 0) rather than the
    parallel-distributed `tagged_mesh.mesh`. Small, spatially concentrated
    surfaces (e.g. a single load face) can otherwise end up with zero
    facets in whichever rank's local partition is doing the lookup, even
    though the group is correctly tagged in the global mesh. The result is
    broadcast to every rank so all ranks build identical membership
    functions from the same complete point set.
    """
    name_to_tag = tagged_mesh.name_to_tag
    if group_name not in name_to_tag:
        points = np.empty((0, 3))
        return comm.bcast(points, root=0) if comm is not None else points

    tag = name_to_tag[group_name]

    if comm is None or comm.rank == 0:
        mesh_serial = tagged_mesh.mesh_serial
        facet_tags_serial = tagged_mesh.facet_tags_serial
        fdim = mesh_serial.topology.dim - 1
        mesh_serial.topology.create_connectivity(fdim, 0)
        f_to_v = mesh_serial.topology.connectivity(fdim, 0)

        facet_indices = facet_tags_serial.find(tag)
        if len(facet_indices) == 0:
            points = np.empty((0, 3))
        else:
            vertex_ids = np.unique(np.hstack([f_to_v.links(f) for f in facet_indices]))
            points = mesh_serial.geometry.x[vertex_ids]
    else:
        points = None

    if comm is not None:
        points = comm.bcast(points, root=0)
    return points


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

def make_protected_zone_from_faces(
    tagged_mesh: TaggedMesh,
    group_names: list[str],
    buffer_radius: float,
    comm: MPI.Comm | None = None,
) -> MembershipFn:
    """Build a protected_zone(x) membership function that hard-fixes density
    to 1 for any point within `buffer_radius` of any facet tagged with one of
    `group_names` (e.g. bolt faces, pin faces).

    Unlike make_solid_zone_from_cells(), this does NOT require the protected
    region to be its own separate colored volume in the CAD file -- it works
    directly off facet points, so it protects material *around* a face that
    sits on the boundary of the single optimizable volume (e.g. bolt holes,
    pin holes), not just pre-separated bolt/mount volumes.

    Reads from the full serial mesh on rank 0 and broadcasts, for the same
    partition-safety reason as facet_group_points().
    """
    all_points = []
    for name in group_names:
        pts = facet_group_points(tagged_mesh, name, comm=comm)
        if len(pts) > 0:
            all_points.append(pts)

    if not all_points:
        logger.warning(
            "No points found for any protected-zone group in %s; "
            "protected zone will be empty.", group_names,
        )
        return lambda x: np.full(x.shape[1], False)

    combined_points = np.vstack(all_points)
    tree = cKDTree(combined_points)

    def fn(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x.T, k=1)
        return dist < buffer_radius
    return fn

def make_solid_zone_from_cells(tagged_mesh: TaggedMesh, group_name: str,
                                comm: MPI.Comm | None = None) -> MembershipFn:
    """Build a solid_zone(x) membership function from actual tagged cell
    centroids (volume-based), not coordinate proximity -- mirrors the
    fenitop GE-bracket demo's "bolts" volume treatment exactly.

    Like facet_group_points(), reads from the serial mesh/cell_tags on
    rank 0 and broadcasts, since small solid regions (e.g. bolt volumes)
    are equally susceptible to being absent from a given rank's partition.
    """
    tag = tagged_mesh.name_to_tag.get(group_name)
    if tag is None:
        return lambda x: np.full(x.shape[1], False)

    if comm is None or comm.rank == 0:
        mesh_serial = tagged_mesh.mesh_serial
        cell_tags_serial = tagged_mesh.cell_tags_serial
        cell_indices = cell_tags_serial.find(tag)

        if len(cell_indices) == 0:
            centroids = np.empty((0, 3))
        else:
            tdim = mesh_serial.topology.dim
            mesh_serial.topology.create_connectivity(tdim, 0)
            c_to_v = mesh_serial.topology.connectivity(tdim, 0)
            centroids = np.array([
                mesh_serial.geometry.x[c_to_v.links(c)].mean(axis=0) for c in cell_indices
            ])
    else:
        centroids = None

    if comm is not None:
        centroids = comm.bcast(centroids, root=0)

    if len(centroids) == 0:
        return lambda x: np.full(x.shape[1], False)

    tree = cKDTree(centroids)

    def fn(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x.T, k=1)
        return dist < 1e-9  # centroids match exactly (same DG0 dof coords)
    return fn

def cluster_group_instances(
    points: np.ndarray, eps: float
) -> list[np.ndarray]:
    """Split one facet group's merged point cloud into distinct physical
    instances (e.g. 4 separate bolt holes sharing one "fixed" tag).

    Single-linkage clustering: two points are in the same instance if they
    are within `eps` of each other (transitively). Implemented as connected
    components of the eps-radius neighborhood graph via scipy.sparse.csgraph,
    so no sklearn dependency and no need to know the instance count up front.

    Args:
        points: [N x 3] facet vertex coordinates for a single group.
        eps: Neighborhood radius. Points closer than eps are linked. Choose
            larger than the mesh edge length but smaller than the gap between
            distinct holes (a few x mesh_size_max is typical).

    Returns:
        List of [Ni x 3] arrays, one per detected instance. Empty list if
        `points` is empty.
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [points]
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    n = len(points)
    if len(pairs) == 0:
        # No links: every point is its own instance. Degenerate but valid.
        return [points[i:i + 1] for i in range(n)]
    data = np.ones(len(pairs), dtype=np.int8)
    graph = coo_matrix(
        (data, (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    n_comp, labels = connected_components(graph, directed=False)
    return [points[labels == k] for k in range(n_comp)]


def _build_mesh_graph(
    coords: np.ndarray, simplices: np.ndarray
) -> "csr_matrix":
    """Build a sparse, symmetric Euclidean-weighted edge graph over mesh
    vertices from element connectivity, for geodesic (edge-path) distance.

    Every pair of nodes sharing a simplex becomes an undirected edge whose
    weight is their Euclidean distance. This is the graph Dijkstra runs on:
    shortest paths follow real mesh edges and therefore stay inside the
    meshed (solid) region, never cutting across void space -- the whole
    reason we don't use straight-line distance on a bent bracket.

    Args:
        coords: [N x 3] vertex coordinates (mesh_serial.geometry.x).
        simplices: [M x (dim+1)] node indices per element.

    Returns:
        [N x N] CSR sparse distance matrix (symmetric).
    """
    from scipy.sparse import csr_matrix

    # All undirected edges of each simplex (every vertex pair within it).
    dim1 = simplices.shape[1]
    edge_pairs = []
    for a in range(dim1):
        for b in range(a + 1, dim1):
            edge_pairs.append(simplices[:, [a, b]])
    edges = np.vstack(edge_pairs)                     # [E x 2]
    i, j = edges[:, 0], edges[:, 1]
    lengths = np.linalg.norm(coords[i] - coords[j], axis=1)
    # Symmetrize.
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.concatenate([lengths, lengths])
    n = len(coords)
    return csr_matrix((data, (rows, cols)), shape=(n, n))

def make_line_corridor_zone(
    waypoints: list[np.ndarray], radius: float
) -> MembershipFn:
    """Build a membership function that is True within `radius` of any
    segment in a polyline (chain of cylinders / capsules).

    Point-to-segment distance for each consecutive waypoint pair; a query
    point is inside the corridor if it is within `radius` of ANY segment.
    Used to realize one geodesic keep-alive path as a solid tube following
    the mesh, so it hugs a bent bracket instead of chording through void.

    Args:
        waypoints: Ordered list of [3] points along one path (>= 1 point).
        radius: Cylinder/capsule radius (m).

    Returns:
        MembershipFn matching FEniTop's `x is [3 x N]` convention.
    """
    r2 = radius * radius

    if len(waypoints) < 2:
        # Degenerate path (single anchor): a ball around the point.
        if len(waypoints) == 1:
            p = np.asarray(waypoints[0], dtype=float).reshape(3)

            def fn_ball(x: np.ndarray) -> np.ndarray:
                d = x - p[:, None]                       # [3 x N]
                return np.einsum("ij,ij->j", d, d) < r2

            return fn_ball
        return lambda x: np.full(x.shape[1], False)

    # Precompute segment endpoints as [S x 3] arrays.
    pts = np.asarray(waypoints, dtype=float).reshape(-1, 3)
    seg_a = pts[:-1]                                     # [S x 3]
    seg_b = pts[1:]                                      # [S x 3]
    seg_d = seg_b - seg_a                                # [S x 3]
    seg_len2 = np.einsum("ij,ij->i", seg_d, seg_d)       # [S]
    seg_len2 = np.where(seg_len2 > 0.0, seg_len2, 1.0)   # guard zero-length

    def fn(x: np.ndarray) -> np.ndarray:
        # x: [3 x N] -> work with q: [N x 3].
        q = x.T                                          # [N x 3]
        # Vector from each segment start to each query point: [N x S x 3].
        w = q[:, None, :] - seg_a[None, :, :]            # [N x S x 3]
        # Projection parameter t of q onto each segment, clamped to [0, 1].
        t = np.einsum("nsj,sj->ns", w, seg_d) / seg_len2  # [N x S]
        t = np.clip(t, 0.0, 1.0)
        # Closest point on each segment: seg_a + t * seg_d.
        proj = seg_a[None, :, :] + t[:, :, None] * seg_d[None, :, :]  # [N x S x 3]
        diff = q[:, None, :] - proj                      # [N x S x 3]
        d2 = np.einsum("nsj,nsj->ns", diff, diff)        # [N x S]
        # Inside if within radius of ANY segment.
        return d2.min(axis=1) < r2

    return fn


def build_keep_alive_zone(
    tagged_mesh: TaggedMesh,
    groups: list[str],
    corridor_radius: float,
    cluster_eps: float,
    comm: MPI.Comm | None = None,
) -> MembershipFn:
    """Build a non-designable keep-alive corridor membership function that
    guarantees every attachment instance is connected into one component.

    Pipeline (all heavy work is rank-0 serial, then broadcast -- same
    partition-safety pattern as facet_group_points()):

      1. For each group in `groups`, pull its merged facet points and split
         them into distinct physical instances via cluster_group_instances()
         (so 4 bolts sharing the "fixed" tag become 4 anchors, not one blob).
      2. Take each instance's centroid as its anchor point.
      3. Build the mesh edge graph and snap every anchor to its nearest
         mesh vertex (graph node).
      4. Compute all-pairs *geodesic* (in-mesh Dijkstra) distances between
         anchor nodes, build a complete graph weighted by those distances,
         and take its minimum spanning tree -- the least guaranteed material
         that still makes everything one connected component.
      5. For each MST edge, reconstruct the geodesic waypoint chain and
         realize it as a piecewise cylinder corridor. OR all corridors.

    Returns a MembershipFn identical on every rank. If fewer than 2 anchors
    are found, returns an all-False function (nothing to connect).

    Args:
        tagged_mesh: Output of import_to_dolfinx (needs mesh_serial on rank 0).
        groups: Facet-group names whose instances must all be connected.
        corridor_radius: Radius (m) of each keep-alive cylinder.
        cluster_eps: Single-linkage radius (m) for splitting a group's
            facet points into distinct instances.
        comm: MPI communicator (rank-0 does the work; result is broadcast).
    """
    if comm is None or comm.rank == 0:
        # --- 1 & 2: cluster each group into instances, take centroids ---
        anchors: list[np.ndarray] = []
        for name in groups:
            pts = facet_group_points(tagged_mesh, name, comm=None)  # rank-0 read
            instances = cluster_group_instances(pts, cluster_eps)
            for inst in instances:
                anchors.append(inst.mean(axis=0))
            logger.info(
                "keep_alive: group '%s' -> %d instance(s).", name, len(instances)
            )

        if len(anchors) < 2:
            logger.info(
                "keep_alive: %d anchor(s) found across groups %s; nothing to "
                "connect. Keep-alive zone is empty.", len(anchors), groups,
            )
            payload = {"waypoint_paths": []}
        else:
            anchor_pts = np.vstack(anchors)                    # [A x 3]

            # --- 3: mesh graph + snap anchors to nearest mesh vertex ---
            mesh_serial = tagged_mesh.mesh_serial
            coords = mesh_serial.geometry.x                    # [N x 3]
            simplices = extract_simplices(tagged_mesh)         # [M x (dim+1)]
            graph = _build_mesh_graph(coords, simplices)

            node_tree = cKDTree(coords)
            _, anchor_nodes = node_tree.query(anchor_pts, k=1)  # [A]
            anchor_nodes = np.atleast_1d(anchor_nodes)

            # --- 4: geodesic all-pairs among anchors, then MST ---
            # Dijkstra from each anchor node; also return predecessors so we
            # can reconstruct the actual geodesic waypoint chains.
            dist_matrix, predecessors = dijkstra(
                graph, directed=False, indices=anchor_nodes,
                return_predecessors=True,
            )
            # dist_matrix: [A x N]; anchor-to-anchor block is [A x A].
            geodesic = dist_matrix[:, anchor_nodes]            # [A x A]

            # Unreachable anchors -> inf; MST will simply omit them (they'd
            # be a separate disconnected component in the mesh graph, which
            # itself signals a meshing problem). Warn if so.
            if not np.isfinite(geodesic[np.triu_indices(len(anchor_nodes), 1)]).all():
                logger.warning(
                    "keep_alive: some anchors are not mutually reachable "
                    "through the mesh edge graph (disconnected mesh?). The "
                    "MST will connect only what is reachable."
                )
                geodesic = np.where(np.isfinite(geodesic), geodesic, 0.0)
                # Zeroing infs would create spurious edges; instead mask them
                # out by leaving them as no-edge in the sparse MST input below.

            # Build MST over the anchor-anchor geodesic graph. Use a sparse
            # matrix so np.inf / missing entries become "no edge".
            A = len(anchor_nodes)
            iu, ju = np.triu_indices(A, k=1)
            w = geodesic[iu, ju]
            finite = np.isfinite(w) & (w > 0.0)
            from scipy.sparse import csr_matrix
            mst_input = csr_matrix(
                (w[finite], (iu[finite], ju[finite])), shape=(A, A)
            )
            mst = minimum_spanning_tree(mst_input)             # [A x A] sparse
            mst_coo = mst.tocoo()

            # --- 5: reconstruct geodesic waypoint chain for each MST edge ---
            def _path_nodes(src_row: int, dst_node: int) -> list[int]:
                """Walk predecessors[src_row] back from dst_node to the
                source anchor node, returning node indices source->dest."""
                path = []
                node = dst_node
                # predecessors uses -9999 for "no predecessor" (the source).
                guard = 0
                while node != -9999 and guard <= len(coords):
                    path.append(node)
                    node = predecessors[src_row, node]
                    guard += 1
                path.reverse()
                return path

            waypoint_paths: list[list[list[float]]] = []
            for a_idx, b_idx in zip(mst_coo.row, mst_coo.col):
                src_node = anchor_nodes[a_idx]
                dst_node = anchor_nodes[b_idx]
                node_path = _path_nodes(a_idx, dst_node)
                if not node_path or node_path[0] != src_node:
                    # Fallback: straight anchor-to-anchor segment if the
                    # predecessor walk failed for any reason.
                    logger.warning(
                        "keep_alive: geodesic reconstruction failed for MST "
                        "edge (%d, %d); falling back to straight segment.",
                        a_idx, b_idx,
                    )
                    node_path = [src_node, dst_node]
                waypoint_paths.append([coords[n].tolist() for n in node_path])

            logger.info(
                "keep_alive: connected %d anchors with %d MST corridor(s).",
                A, len(waypoint_paths),
            )
            payload = {"waypoint_paths": waypoint_paths}
    else:
        payload = None

    if comm is not None:
        payload = comm.bcast(payload, root=0)

    waypoint_paths = payload["waypoint_paths"]
    if not waypoint_paths:
        return lambda x: np.full(x.shape[1], False)

    # Build one corridor membership fn per path, then OR them all.
    corridor_fns = [
        make_line_corridor_zone([np.asarray(p) for p in path], corridor_radius)
        for path in waypoint_paths
    ]

    def fn(x: np.ndarray) -> np.ndarray:
        mask = np.full(x.shape[1], False)
        for cfn in corridor_fns:
            mask |= cfn(x)
        return mask

    return fn

@dataclass
class BoundaryConditions:
    """Bundled BC/zone functions ready to drop into FEniTop's fem_config
    and opt_config dicts (see fem-11.py's `form_fem` signature).

    traction_bcs is now dict[case_name, list[[load_vector, membership_fn]]]
    -- one independent traction_bcs list per named load case -- rather than
    a single flat list. This is what gets handed directly to
    src.fenitop.topopt.form_fem_multi_case as its `load_cases` argument;
    each entry becomes its own independently-solved equilibrium problem
    sharing one density field, instead of every load being summed into one
    combined RHS.
    """
    disp_bc: MembershipFn
    traction_bcs: dict[str, list[list]]  # {case_name: [[load_vector, membership_fn], ...]}
    solid_zone: MembershipFn
    void_zone: MembershipFn


def build_boundary_conditions(
    tagged_mesh: TaggedMesh,
    load_cases: dict[str, list[tuple[str, tuple[float, float, float]]]],
    snap_tol: float = 1e-4,
    protected_face_groups: list[str] | None = None,
    protected_buffer_radius: float = 5e-3,
    keep_alive_groups: list[str] | None = None,
    keep_alive_radius: float | None = None,
    keep_alive_cluster_eps: float | None = None,
    comm: MPI.Comm | None = None,
) -> BoundaryConditions:
    """Build shared disp_bc/solid_zone/void_zone plus a per-case traction_bcs dict.

    load_cases: {case_name: [(group_name, vector), ...], ...} -- e.g.
    {"vertical_up": [("load_1", (0,0,9.34e7))],
     "torsion": [("load_1", (0,-2.9e7,0))]}.

    Multiple cases commonly reuse the same group_name with different
    vectors (facet lookup is cached per group_name below so that pattern
    doesn't repeat the KDTree build), but each case's list can also
    reference entirely different groups.

    solid_zone is the OR of up to three contributions:
      * volume-tagged solid regions (make_solid_zone_from_cells),
      * a buffer around protected facet groups (make_protected_zone_from_faces),
      * a non-designable keep-alive backbone (build_keep_alive_zone) that
        geodesically connects all attachment instances into one connected
        component so the assembly is always physically joinable, regardless
        of what the stress-based optimizer would otherwise carve away.
    keep_alive_* args are only honored when keep_alive_groups is non-empty;
    keep_alive_radius / keep_alive_cluster_eps default to a small multiple of
    the local mesh size at the call site (mapper does not read config).
    """
    fixed_pts = facet_group_points(tagged_mesh, "fixed", comm=comm)
    disp_bc = make_membership_fn(fixed_pts, snap_tol)

    # Cache membership fns per facet group so cases that reuse the same
    # tagged group (the common "same group, different vector" pattern)
    # don't redo the facet lookup + KDTree build for every case.
    membership_cache: dict[str, MembershipFn | None] = {}

    def _get_membership_fn(group_name: str) -> MembershipFn | None:
        if group_name not in membership_cache:
            pts = facet_group_points(tagged_mesh, group_name, comm=comm)
            if len(pts) == 0:
                logger.warning("No points found for load group '%s'; skipping.", group_name)
                membership_cache[group_name] = None
            else:
                membership_cache[group_name] = make_membership_fn(pts, snap_tol)
        return membership_cache[group_name]

    traction_bcs: dict[str, list[list]] = {}
    for case_name, entries in load_cases.items():
        case_bcs = []
        for group_name, vector in entries:
            fn = _get_membership_fn(group_name)
            if fn is None:
                continue
            case_bcs.append([vector, fn])
        if not case_bcs:
            logger.warning(
                "Load case '%s' resolved to zero valid facet groups; it "
                "will contribute no traction (check group names/tags).",
                case_name,
            )
        traction_bcs[case_name] = case_bcs

    # --- solid_zone assembly: volumes | protected faces | keep-alive backbone ---
    zone_fns: list[MembershipFn] = [
        make_solid_zone_from_cells(tagged_mesh, "solid", comm=comm)
    ]

    if protected_face_groups:
        zone_fns.append(
            make_protected_zone_from_faces(
                tagged_mesh, protected_face_groups,
                protected_buffer_radius, comm=comm,
            )
        )

    if keep_alive_groups:
        if keep_alive_radius is None or keep_alive_cluster_eps is None:
            raise ValueError(
                "keep_alive_groups was provided but keep_alive_radius and/or "
                "keep_alive_cluster_eps is None. Resolve these at the call "
                "site (e.g. from config.keep_alive, defaulting to a multiple "
                "of mesh_size_max) rather than letting mapper.py invent a "
                "physical length scale."
            )
        zone_fns.append(
            build_keep_alive_zone(
                tagged_mesh,
                groups=keep_alive_groups,
                corridor_radius=keep_alive_radius,
                cluster_eps=keep_alive_cluster_eps,
                comm=comm,
            )
        )

    if len(zone_fns) == 1:
        solid_zone = zone_fns[0]
    else:
        def solid_zone(x: np.ndarray) -> np.ndarray:
            mask = np.full(x.shape[1], False)
            for zfn in zone_fns:
                mask |= zfn(x)
            return mask

    void_zone = lambda x: np.full(x.shape[1], False)

    return BoundaryConditions(
        disp_bc=disp_bc, traction_bcs=traction_bcs,
        solid_zone=solid_zone, void_zone=void_zone,
    )