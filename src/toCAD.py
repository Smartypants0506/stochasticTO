"""
Export a topology-optimized density field to a printable/viewable STL surface.

Run:
    python export_optimized_stl.py

Optional:
    python export_optimized_stl.py --density output/rho_robust_lambda1.npy \
        --output output/optimized_part_lambda1.stl --iso 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from mpi4py import MPI
from dolfinx import plot

from src.config.loader import load_config
from src.meshing.importer import import_and_heal
from src.meshing.mesher import (
    MeshingConfig,
    generate_mesh,
    import_to_dolfinx,
    tag_physical_groups,
)


def rebuild_optimization_mesh(config_path: str):
    """Recreate the same FEM mesh generation sequence used by main.py."""
    cfg = load_config(config_path)

    entities = import_and_heal(cfg.step_file)
    mesh_cfg = MeshingConfig(
        mesh_size_max=cfg.mesh_size_max,
        color_targets=cfg.color_targets,
        solid_volume_color=cfg.solid_volume_color,
    )

    comm = MPI.COMM_WORLD
    tag_physical_groups(entities, mesh_cfg)
    generate_mesh(mesh_cfg, comm)

    return import_to_dolfinx(comm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an optimized density .npy field to an STL surface."
    )
    parser.add_argument(
        "--config",
        default="src/config/configSmoke.yaml",
        help="The exact config used during optimization.",
    )
    parser.add_argument(
        "--density",
        default="output/rho_robust_lambda1.npy",
        help="Input robust-density NumPy file.",
    )
    parser.add_argument(
        "--output",
        default="output/optimized_part_lambda1.stl",
        help="Output STL path.",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=0.5,
        help="Material-density threshold, normally 0.5.",
    )
    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help="Keep disconnected material islands instead of only the largest piece.",
    )
    args = parser.parse_args()

    if not 0.0 < args.iso < 1.0:
        raise ValueError("--iso must be strictly between 0 and 1.")

    density_path = Path(args.density)
    output_path = Path(args.output)

    if not density_path.is_file():
        raise FileNotFoundError(f"Density file does not exist: {density_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tagged_mesh = rebuild_optimization_mesh(args.config)
    mesh = tagged_mesh.mesh_serial

    rho = np.asarray(np.load(density_path), dtype=np.float64).ravel()

    # Build a VTK/PyVista representation of the 3-D tetrahedral FEM mesh.
    topology, cell_types, geometry = plot.vtk_mesh(mesh, dim=3)
    volume_grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    n_cells = volume_grid.n_cells
    if rho.size != n_cells:
        raise ValueError(
            f"Density array has {rho.size} values, but the rebuilt mesh has "
            f"{n_cells} 3-D cells. Use precisely the same config, STEP input, "
            "meshing settings, and MPI rank count as the optimization run."
        )

    volume_grid.cell_data["rho"] = rho

    # Contour needs point data; PyVista averages adjacent cell values at vertices.
    nodal_grid = volume_grid.cell_data_to_point_data(pass_cell_data=True)
    surface = nodal_grid.contour(isosurfaces=[args.iso], scalars="rho")

    if surface.n_cells == 0:
        raise RuntimeError(
            f"No surface was found at rho={args.iso}. "
            f"Density range: {rho.min():.4f} to {rho.max():.4f}. "
            "Try --iso 0.4 or --iso 0.3."
        )

    surface = surface.clean(tolerance=1e-9)

    if not args.keep_all_components:
        surface = surface.connectivity(extraction_mode="largest").extract_surface()

    surface = surface.triangulate().clean(tolerance=1e-9)

    # STL is the 3-D surface model; VT P/VTU preserves the density field for review.
    surface.save(output_path)
    nodal_grid.save(output_path.with_suffix(".vtu"))

    print(f"Loaded:       {density_path}")
    print(f"Density range: {rho.min():.6f} to {rho.max():.6f}")
    print(f"Threshold:    rho = {args.iso}")
    print(f"STL written:  {output_path}")
    print(f"VTU written:  {output_path.with_suffix('.vtu')}")
    print(f"Triangles:    {surface.n_cells}")


if __name__ == "__main__":
    main()