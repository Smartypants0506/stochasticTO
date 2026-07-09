"""
topopt_3d_custom_step.py

Run 3D topology optimization (via FEniTop) on a CUSTOM STEP geometry whose
faces have been colour-coded in a CAD tool (e.g. Onshape) to denote:
  - fixed / boundary-condition faces (default: GREEN)
  - load-application faces (default: RED, optionally BLUE for a 2nd load case)

Additionally, following the same strategy as topopt_3d_ge_bracket.py, any
mounting/bolt regions that must stay fully solid (not optimized away) should
be modelled as SEPARATE SOLID BODIES in the CAD tool, coloured YELLOW. These
become their own gmsh volume (not just a coordinate-based guess), and their
density is hard-fixed to 1 for the entire optimization -- exactly like the
"bolts" volume in the GE bracket demo.

Usage:
    python3 topopt_3d_custom_step.py --step my_part.step
"""
import argparse
import numpy as np
from mpi4py import MPI
import gmsh
import dolfinx
from scipy.spatial import cKDTree

from fenitop.topopt import topopt

# ----------------------------------------------------------------------------
# 1. CLI / configuration
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--step", required=True, help="Path to coloured STEP file")
parser.add_argument("--mesh-size-max", type=float, default=3e-3, help="Max element size in metres (STEP assumed mm)")
parser.add_argument("--tol", type=float, default=1e-4, help="Snap tolerance (m) matching mesh points to coloured facets")
args = parser.parse_args()

comm = MPI.COMM_WORLD

# Surface colours to look for (RGBA 0-255). Edit as needed.
COLOR_TARGETS = {
    "fixed": (0, 255, 0, 255),  # green -> Dirichlet (u = 0)
    "load_1": (255, 0, 0, 255),  # red -> traction load case 1
    "load_2": (0, 0, 255, 255),  # blue -> traction load case 2 (optional)
}

# Volume colour for solid bodies that must stay fully solid (bolt bosses,
# mounting bosses, etc.), matching the "bolts" volume in the GE bracket demo.
SOLID_VOLUME_COLOR = (255, 255, 0, 255)  # yellow

COLOR_MATCH_TOL = 40  # per-channel tolerance (0-255)

# Load vectors applied on matching coloured faces. Set "load_2" to None if unused.
LOAD_VECTORS = {
    "load_1": (0.0, 0.0, 100.0 / .00007125598652),
    "load_2": None,
}

# ----------------------------------------------------------------------------
# 2. Mesh the coloured STEP file with gmsh, tagging surfaces and volumes
# ----------------------------------------------------------------------------
if comm.rank == 0:
    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.option.set_string("Geometry.OCCTargetUnit", "M")  # STEP assumed in mm
    gmsh.open(args.step)
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    _, _, surfaces, volumes = (gmsh.model.occ.get_entities(d) for d in range(4))

    def color_close(c1, c2, tol=COLOR_MATCH_TOL):
        return all(abs(a - b) <= tol for a, b in zip(c1[:3], c2[:3]))

    def extract_tags_by_color(entities, target_color):
        return [e[1] for e in entities if color_close(gmsh.model.get_color(*e), target_color)]

    # -- Surface physical groups (fixed / load faces) --
    for name, target_color in COLOR_TARGETS.items():
        if LOAD_VECTORS.get(name, "fixed") is None:
            continue
        tags = extract_tags_by_color(surfaces, target_color)
        if not tags:
            print(f"[warn] No faces found for colour group '{name}' "
                  f"(target RGBA={target_color}). Check your STEP colouring.")
            continue
        gmsh.model.addPhysicalGroup(2, tags, name=name)

    # -- Volume physical groups: split into "solid" (fixed density = 1) and
    #    "volume" (the optimizable region), same pattern as the GE bracket demo --
    solid_volume_tags = extract_tags_by_color(volumes, SOLID_VOLUME_COLOR)
    all_volume_tags = [v[1] for v in volumes]
    optimizable_volume_tags = [v for v in all_volume_tags if v not in solid_volume_tags]

    if not optimizable_volume_tags:
        raise RuntimeError("No optimizable volume found. Check that at least one solid body is NOT coloured yellow (SOLID_VOLUME_COLOR).")

    gmsh.model.addPhysicalGroup(3, optimizable_volume_tags, name="volume")
    if solid_volume_tags:
        gmsh.model.addPhysicalGroup(3, solid_volume_tags, name="solid")
    else:
        print("[warn] No yellow-coloured solid volumes found -- no bolt/mounting "
              "region will be protected from optimization.")

    gmsh.option.setNumber("General.NumThreads", comm.size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size_max)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 8)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model if comm.rank == 0 else None, comm, rank=0
)

if comm.rank == 0:
    mesh_serial, cell_tags_serial, facet_tags_serial = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_SELF, rank=0
    )
    name_to_tag = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        name_to_tag[name] = tag
    gmsh.finalize()
else:
    mesh_serial = None
    name_to_tag = {}

name_to_tag = comm.bcast(name_to_tag, root=0)

for name in list(COLOR_TARGETS) + ["volume", "solid"]:
    tag = name_to_tag.get(name)
    print(f"{name}: physical group tag = {tag}")

# ----------------------------------------------------------------------------
# 3. Turn coloured facet tags into geometric "is-this-point-in-group" lambdas
# ----------------------------------------------------------------------------
def facet_group_points(name_to_tag, facet_tags, group_name):
    if group_name not in name_to_tag:
        return np.empty((0, 3))
    tag = name_to_tag[group_name]
    facet_indices = facet_tags.find(tag)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f_to_v = mesh.topology.connectivity(fdim, 0)
    vertex_ids = (np.unique(np.hstack([f_to_v.links(f) for f in facet_indices]))
                  if len(facet_indices) else np.array([], dtype=np.int32))
    return mesh.geometry.x[vertex_ids] if len(vertex_ids) else np.empty((0, 3))

def make_membership_fn(points, tol):
    if len(points) == 0:
        return lambda x: np.full(x.shape[1], False)
    tree = cKDTree(points)
    def fn(x):
        dist, _ = tree.query(x.T, k=1)
        return dist < tol
    return fn

fixed_pts = facet_group_points(name_to_tag, facet_tags, "fixed")
load1_pts = facet_group_points(name_to_tag, facet_tags, "load_1")
load2_pts = facet_group_points(name_to_tag, facet_tags, "load_2")

disp_bc = make_membership_fn(fixed_pts, args.tol)
traction_bcs = [[LOAD_VECTORS["load_1"], make_membership_fn(load1_pts, args.tol)]]
if LOAD_VECTORS.get("load_2") is not None and len(load2_pts):
    traction_bcs.append([LOAD_VECTORS["load_2"], make_membership_fn(load2_pts, args.tol)])

# ----------------------------------------------------------------------------
# 3b. Solid zone from actual cell tags (volume-based), not coordinate proximity
#     -- mirrors the GE bracket demo's "bolts" volume treatment.
# ----------------------------------------------------------------------------
def make_solid_zone_from_cells(mesh, cell_tags, name_to_tag, group_name):
    """Build a solid_zone(x) function that is True for centroids of cells
    tagged as `group_name` (e.g. the yellow-coloured solid bolt volume)."""
    tag = name_to_tag.get(group_name)
    if tag is None:
        return lambda x: np.full(x.shape[1], False)
    cell_indices = cell_tags.find(tag)
    if len(cell_indices) == 0:
        return lambda x: np.full(x.shape[1], False)

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c_to_v = mesh.topology.connectivity(tdim, 0)
    centroids = np.array([
        mesh.geometry.x[c_to_v.links(c)].mean(axis=0) for c in cell_indices
    ])
    tree = cKDTree(centroids)
    def fn(x):
        dist, _ = tree.query(x.T, k=1)
        return dist < 1e-9  # centroids should match exactly (same DG0 dof coords)
    return fn

solid_zone_fn = make_solid_zone_from_cells(mesh, cell_tags, name_to_tag, "solid")

# ----------------------------------------------------------------------------
# 4. FenitTop FEA / topology-optimisation setup (same interface as beam_3d.py)
# ----------------------------------------------------------------------------
fem = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 68.9e9,
    "poisson's ratio": 0.33,
    "disp_bc": disp_bc,
    "traction_bcs": traction_bcs,
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {"ksp_type": "cg", "pc_type": "gamg"},
}

opt = {
    "max_iter": 400,
    "opt_tol": 1e-5,
    "vol_frac": 0.3,
    "solid_zone": solid_zone_fn,
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 0.006,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": True,
    "move": 0.02,
    "opt_compliance": True,
}

if __name__ == "__main__":
    topopt(fem, opt)