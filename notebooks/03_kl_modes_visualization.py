"""
Stage 3: Random Field Modeling on a Gmsh-Generated Tetrahedral Volume Mesh
using Real Aero-Engine Compressor Blade Metrology Data (OpenTURNS + PyVista).

Conforms to Master Project Context (masterContext.md), Section 3.3:
Manufacturing Uncertainty - Random Field Modeling.

Pipeline
--------
1. Ingest real CMM-style deviation data (leading-edge radius deviation,
   or any other measured sheet) from the metrology spreadsheet.
2. Fit a stationary covariance kernel (squared-exponential or Matern) to
   the empirical spatial covariance across measurement stations, and fit
   a spanwise mean deviation profile mu(x).
3. Read a tagged, tetrahedral (3D volume) Gmsh mesh of the physical part.
4. Run KarhunenLoeveP1Algorithm to decompose the fitted covariance kernel
   into eigenmodes phi_i(x) and eigenvalues lambda_i directly on the FE
   mesh nodes, truncating at >= 95% retained variance.
5. Sample KL coefficients xi_i ~ N(0,1) and reconstruct one realization
   of the manufacturing deviation field:
       Z(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i
6. Visualize Z(x) on the 3D volume mesh with PyVista, annotated with the
   Gmsh physical-group feature tags.

Usage
-----
    python random_field_stage3.py mesh.msh metrology.xlsx \\
        --sheet "Leading-edge radius deviation" \\
        --variance-target 0.95
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import openturns as ot
import gmsh


# ---------------------------------------------------------------------------
# 1. Metrology ingestion - src/metrology/
# ---------------------------------------------------------------------------
@dataclass
class MetrologyProfile:
    """Empirical spanwise mean and covariance fitted from CMM/scan data."""

    station_positions: np.ndarray   # normalized [0, 1] spanwise positions
    mean_profile: np.ndarray        # mu at each station, shape (n_stations,)
    covariance_matrix: np.ndarray   # empirical covariance, shape (n_stations, n_stations)
    sigma: float                    # marginal std dev, sqrt(mean of diag)
    correlation_length: float       # fitted length scale l


def load_metrology_data(xlsx_path: str, sheet_name: str) -> MetrologyProfile:
    """
    Reads a metrology sheet formatted as Sample x Station (H1..Hn) and
    returns the empirical mean profile, covariance matrix, and a fitted
    correlation length for a squared-exponential kernel.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0, skiprows=1)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    values = df.to_numpy(dtype=float)          # shape (n_samples, n_stations)
    n_stations = values.shape[1]

    station_positions = np.linspace(0.0, 1.0, n_stations)
    mean_profile = values.mean(axis=0)
    covariance_matrix = np.cov(values, rowvar=False)
    sigma = float(np.sqrt(np.mean(np.diag(covariance_matrix))))

    correlation_length = _fit_correlation_length(station_positions, covariance_matrix, sigma)
    return MetrologyProfile(station_positions, mean_profile, covariance_matrix,
                             sigma, correlation_length)


def _fit_correlation_length(positions: np.ndarray, cov_matrix: np.ndarray,
                             sigma: float) -> float:
    """
    Fits an isotropic squared-exponential length scale l to the empirical
    covariance via nonlinear least squares (variogram-style fit), per
    Stage 3.3: 'kernel parameters validated against empirical variogram'.
    """
    from scipy.optimize import curve_fit

    n = len(positions)
    dist, cov_vals = [], []
    for i in range(n):
        for j in range(n):
            dist.append(abs(positions[i] - positions[j]))
            cov_vals.append(cov_matrix[i, j])
    dist = np.array(dist)
    cov_vals = np.array(cov_vals)

    def model(d, l):
        return sigma ** 2 * np.exp(-(d ** 2) / (2.0 * l ** 2))

    popt, _ = curve_fit(model, dist, cov_vals, p0=[0.2], bounds=(1e-3, 5.0))
    return float(popt[0])


# ---------------------------------------------------------------------------
# 2. Gmsh mesh ingestion - src/meshing/ (3D tetrahedral volume mesh only)
# ---------------------------------------------------------------------------
@dataclass
class VolumeMesh:
    vertices: np.ndarray
    tetrahedra: np.ndarray
    feature_tags: dict[str, np.ndarray]
    span_axis: int


def read_gmsh_volume_mesh(filename: str, span_axis: int = 1) -> VolumeMesh:
    """
    Reads a tetrahedral (3D volume) .msh file and its physical groups.
    Raises if no tetrahedra are present, per project standard that Stage 3
    perturbation acts on the full FE volume mesh, not a surface mesh.
    """
    gmsh.initialize()
    gmsh.open(filename)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)
    tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

    tets = []
    for dim, vol_tag in gmsh.model.getEntities(dim=3):
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim, vol_tag)
        for etype, enodes_list in zip(elem_types, elem_node_tags):
            if etype == 4:  # 4-node tetrahedron
                enodes = np.array(enodes_list).reshape(-1, 4)
                for tet_tags in enodes:
                    tets.append([tag_to_index[t] for t in tet_tags])

    if not tets:
        gmsh.finalize()
        raise RuntimeError(
            "No tetrahedra found. Run Mesh > 3D in Gmsh before exporting "
            "-- this pipeline requires a volume mesh, not a surface mesh."
        )
    tetrahedra = np.array(tets, dtype=int)

    feature_tags: dict[str, np.ndarray] = {}
    for dim, phys_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, phys_tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
        idx = set()
        for ent_tag in entities:
            etypes, _, enode_tags = gmsh.model.mesh.getElements(dim, ent_tag)
            for _, enodes_list in zip(etypes, enode_tags):
                for node_tag in np.array(enodes_list):
                    idx.add(tag_to_index[node_tag])
        feature_tags[name] = np.array(sorted(idx), dtype=int)

    gmsh.finalize()
    return VolumeMesh(coords, tetrahedra, feature_tags, span_axis)


# ---------------------------------------------------------------------------
# 3. Random field construction - src/random_fields/ (KL expansion, real data)
# ---------------------------------------------------------------------------
def build_kl_random_field(mesh: VolumeMesh, profile: MetrologyProfile,
                           variance_target: float = 0.95):
    """
    Builds a truncated Karhunen-Loeve expansion of the fitted covariance
    kernel directly on the FE mesh nodes, per Stage 3.3:
        Z(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i
    Truncation order chosen so retained modes explain >= variance_target
    of total variance.
    """
    coords = mesh.vertices
    span = coords[:, mesh.span_axis]
    span_norm = (span - span.min()) / (span.max() - span.min() + 1e-12)

    mu_x = np.interp(span_norm, profile.station_positions, profile.mean_profile)

    ot_mesh = ot.Mesh(coords.tolist(), mesh.tetrahedra.tolist(), True)
    cov_model = ot.SquaredExponential([profile.correlation_length], [profile.sigma])

    algo = ot.KarhunenLoeveP1Algorithm(ot_mesh, cov_model, 1e-3)
    algo.run()
    result = algo.getResult()

    eigenvalues = np.array(result.getEigenvalues())
    total_variance = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / total_variance
    n_kl = int(np.searchsorted(cumulative, variance_target) + 1)
    n_kl = max(1, min(n_kl, len(eigenvalues)))

    modes = result.getModesAsProcessSample()
    phi = np.array([np.array(modes.getField(i).getValues())[:, 0]
                    for i in range(n_kl)])          # shape (n_kl, n_nodes)
    lam = eigenvalues[:n_kl]

    xi = np.random.standard_normal(n_kl)
    deviation_field = mu_x + (np.sqrt(lam)[:, None] * phi).T @ xi

    return deviation_field, n_kl, cumulative[n_kl - 1]


# ---------------------------------------------------------------------------
# 4. Visualization - src/viz/ (PyVista, per approved stack)
# ---------------------------------------------------------------------------
def plot_field_3d(mesh: VolumeMesh, field: np.ndarray) -> None:
    import pyvista as pv

    n_cells = mesh.tetrahedra.shape[0]
    cells = np.hstack([np.full((n_cells, 1), 4), mesh.tetrahedra]).astype(np.int64).flatten()
    cell_types = np.full(n_cells, 10, dtype=np.uint8)  # VTK_TETRA

    grid = pv.UnstructuredGrid(cells, cell_types, mesh.vertices)
    grid["deviation_mm"] = field

    plotter = pv.Plotter()
    vmax = np.max(np.abs(field))
    plotter.add_mesh(grid, scalars="deviation_mm", cmap="RdBu_r",
                      clim=[-vmax, vmax], show_edges=False,
                      scalar_bar_args={"title": "Manufacturing deviation (mm)"})

    colors = ["lime", "orange", "yellow", "magenta", "cyan"]
    for i, (name, idx) in enumerate(mesh.feature_tags.items()):
        if len(idx) == 0:
            continue
        plotter.add_points(mesh.vertices[idx], color=colors[i % len(colors)],
                            point_size=8, render_points_as_spheres=True, label=name)

    plotter.add_legend()
    plotter.add_title("KL-expansion realization of real compressor-blade deviation field")
    plotter.show()


# ---------------------------------------------------------------------------
# 5. CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_file", help="Path to a tetrahedral .msh file (3D volume mesh)")
    parser.add_argument("metrology_file", help="Path to the metrology .xlsx file")
    parser.add_argument("--sheet", default="Leading-edge radius deviation",
                         help="Sheet name in the metrology workbook")
    parser.add_argument("--variance-target", type=float, default=0.95,
                         help="Minimum fraction of variance retained by KL truncation")
    parser.add_argument("--span-axis", type=int, default=1, choices=[0, 1, 2],
                         help="Mesh coordinate axis (0=x,1=y,2=z) treated as blade span")
    args = parser.parse_args()

    profile = load_metrology_data(args.metrology_file, args.sheet)
    print(f"Fitted correlation length l = {profile.correlation_length:.4f}, "
          f"sigma = {profile.sigma:.4f}")

    mesh = read_gmsh_volume_mesh(args.mesh_file, span_axis=args.span_axis)
    print(f"Loaded {len(mesh.vertices)} nodes, {len(mesh.tetrahedra)} tetrahedra.")
    print(f"Physical groups: {list(mesh.feature_tags.keys())}")

    field, n_kl, retained = build_kl_random_field(mesh, profile, args.variance_target)
    print(f"KL truncation order N_KL = {n_kl}, retained variance = {retained:.3%}")

    plot_field_3d(mesh, field)


if __name__ == "__main__":
    main()