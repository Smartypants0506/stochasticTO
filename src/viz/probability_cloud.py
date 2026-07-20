"""src/viz/probability_cloud.py — Stage 6 probability-cloud export.

Builds the "probability cloud" ensemble described in masterContext.md
section 4.1: N_vis (500-5,000) perturbed meshes with scalar fields
(compliance, per-element coefficient of variation, von Mises stress),
with opacity mapped to P(sample) (low-probability/extreme geometries render
more transparent). Exports a ParaView time-collection (ensemble.pvd, one
.vtu per sample) plus a merged surface cloud (probability_cloud.vtp), both
of which can be opened directly in ParaView Desktop.

Per project rule "Always Use Premade Tools -- Never Reimplement": all mesh
I/O and file writing goes through PyVista/VTK. The only custom code here is
(a) the thin PVD collection-XML wrapper, since PyVista has no high-level
writer for a non-time-varying multi-file VTK collection, and (b) the
probability-weight/opacity mapping, which is this project's own logic.

Expected MCResult sample schema (produced by
src/validation/monte_carlo.py's run_monte_carlo_validation). Each element
of `mc_result.samples` must expose:
    .xi          : np.ndarray [N_KL]     -- iid N(0,1) KL coordinates drawn for this sample
    .compliance  : float                 -- FEniTop compliance C(xi)
    .rho_hat     : np.ndarray [n_elem]   -- projected density field rho_hat(x; eta(xi))
    .von_mises   : np.ndarray [n_elem]   -- elementwise von Mises stress
    .points      : np.ndarray [n_nodes,3] | None
                    -- explicit Stage-6 perturbed node coordinates (see
                    masterContext.md 3.2/3.6); if None the nominal mesh
                    geometry is reused and only scalar fields vary per sample.
If monte_carlo.py's MCResult does not yet carry these fields, extend it
there -- this module intentionally does not reimplement Stage 6 sampling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityCloudConfig:
    n_vis: int = 500                          # masterContext 4.1: 500-5000 perturbed meshes
    output_dir: Path = field(default_factory=lambda: Path("output/viz/probability_cloud"))
    seed: int = 0
    min_opacity: float = 0.05                 # fully-transparent floor for extreme samples
    max_opacity: float = 1.0


def build_probability_cloud(tagged_mesh, mc_result, config: ProbabilityCloudConfig | None = None) -> dict[str, Path]:
    """Assemble and export the Stage 6 probability cloud for ParaView Desktop.

    Parameters
    ----------
    tagged_mesh : the dolfinx-wrapping mesh object built in Stage 1
        (meshing.importer/meshing.mesher), used for nominal topology/geometry.
    mc_result : MCResult returned by validation.monte_carlo.run_monte_carlo_validation
        (see module docstring for the required `.samples` schema).
    config : ProbabilityCloudConfig, optional.

    Returns
    -------
    dict with 'pvd' and 'vtp' Path entries for the written files. Both are
    native ParaView formats: open `ensemble.pvd` in ParaView Desktop to
    browse per-sample .vtu files as timesteps, or open
    `probability_cloud.vtp` for the single merged surface cloud.
    """
    config = config or ProbabilityCloudConfig()
    samples = getattr(mc_result, "samples", None)
    if not samples:
        raise ValueError(
            "mc_result.samples is empty/missing -- build_probability_cloud "
            "requires per-sample xi/compliance/rho_hat/von_mises records "
            "from run_monte_carlo_validation; see module docstring."
        )

    samples = _subsample(samples, config.n_vis, config.seed)
    logger.info("Stage 6 viz: building probability cloud from %d/%d MC samples",
                len(samples), len(mc_result.samples))

    xi = np.stack([np.asarray(s.xi, dtype=float) for s in samples])
    weights = _sample_probability_weights(xi)
    opacity = _map_to_opacity(weights, config.min_opacity, config.max_opacity)

    base_grid = _mesh_to_pyvista(tagged_mesh)
    density_stack = np.stack(
        [_match_length(np.asarray(s.rho_hat, dtype=float), base_grid.n_cells) for s in samples]
    )
    elem_cov = _coefficient_of_variation(density_stack)

    output_dir = Path(config.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    vtu_paths: list[Path] = []
    merged: pv.UnstructuredGrid | None = None
    for i, (sample, p, w) in enumerate(zip(samples, opacity, weights)):
        grid = _build_sample_grid(base_grid, sample, elem_cov, opacity=p, probability=w)
        vtu_path = samples_dir / f"sample_{i:05d}.vtu"
        grid.save(str(vtu_path))            # PyVista/VTK native writer
        vtu_paths.append(vtu_path)
        merged = grid if merged is None else merged.merge(grid)  # PyVista native merge

    pvd_path = output_dir / "ensemble.pvd"
    _write_pvd_collection(pvd_path, vtu_paths)

    vtp_path = output_dir / "probability_cloud.vtp"
    merged.extract_surface().save(str(vtp_path))  # PyVista native surface extraction/writer

    logger.info("Stage 6 viz: wrote %s, %s (%d samples) -- open either directly in ParaView Desktop",
                pvd_path, vtp_path, len(samples))
    return {"pvd": pvd_path, "vtp": vtp_path}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _subsample(samples: list, n_vis: int, seed: int) -> list:
    if n_vis >= len(samples):
        return list(samples)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(samples), size=n_vis, replace=False))
    return [samples[i] for i in idx]


def _sample_probability_weights(xi: np.ndarray) -> np.ndarray:
    """P(sample) under the underlying iid-N(0,1) KL-coordinate density,
    normalized so the modal (xi=0, highest-probability) sample -> weight 1.0.
    """
    n_kl = xi.shape[1]
    mvn = multivariate_normal(mean=np.zeros(n_kl), cov=np.eye(n_kl))
    log_pdf = mvn.logpdf(xi) - mvn.logpdf(np.zeros(n_kl))
    return np.exp(log_pdf)


def _map_to_opacity(weights: np.ndarray, lo: float, hi: float) -> np.ndarray:
    w_min, w_max = weights.min(), weights.max()
    span = max(w_max - w_min, 1e-12)
    return lo + (weights - w_min) / span * (hi - lo)


def _coefficient_of_variation(stack: np.ndarray) -> np.ndarray:
    """Per-element CoV of the density field across the MC ensemble
    (masterContext 4.1: 'coefficient of variation per element')."""
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    return np.divide(std, mean, out=np.zeros_like(mean), where=mean > 1e-8)


def _mesh_to_pyvista(tagged_mesh) -> pv.UnstructuredGrid:
    """Convert the dolfinx mesh to PyVista via dolfinx's own VTK-topology
    export helper -- no custom cell-connectivity code.

    Uses `tagged_mesh.mesh_serial` (the rank-0-only, un-partitioned full mesh
    already relied on by Stage 3's KL expansion, see main.py) rather than the
    MPI-partitioned `tagged_mesh.mesh`: the latter only holds each rank's
    local sub-mesh, which would silently export a fragment of the geometry
    if this function is ever called under more than one rank.
    """
    from dolfinx.plot import vtk_mesh
    mesh = getattr(tagged_mesh, "mesh_serial", None)
    if mesh is None:
        mesh = tagged_mesh.mesh
    topology, cell_types, geometry = vtk_mesh(mesh)
    return pv.UnstructuredGrid(topology, cell_types, geometry)


def _match_length(arr: np.ndarray, n: int) -> np.ndarray:
    """Coerce a per-element array to length n (defensive against a DG0
    density field / plotting-mesh cell-count mismatch)."""
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] > n:
        return arr[:n]
    return np.pad(arr, (0, n - arr.shape[0]), mode="edge")


def _build_sample_grid(base_grid: pv.UnstructuredGrid, sample, elem_cov: np.ndarray,
                        opacity: float, probability: float) -> pv.UnstructuredGrid:
    grid = base_grid.copy()

    points = getattr(sample, "points", None)
    if points is not None:
        points = np.asarray(points, dtype=float)
        if points.shape == grid.points.shape:
            grid.points = points  # Stage-6 explicit geometry perturbation (masterContext 3.6)
        else:
            logger.warning("perturbed point count %s != mesh point count %s; "
                            "keeping nominal geometry for this sample",
                            points.shape, grid.points.shape)

    n_cells = grid.n_cells
    grid.cell_data["density"] = _match_length(np.asarray(sample.rho_hat, dtype=float), n_cells)
    grid.cell_data["von_mises_stress"] = _match_length(np.asarray(sample.von_mises, dtype=float), n_cells)
    grid.cell_data["density_cov"] = elem_cov
    grid.cell_data["compliance"] = np.full(n_cells, float(sample.compliance))
    grid.cell_data["sample_probability"] = np.full(n_cells, float(probability))
    grid.cell_data["opacity"] = np.full(n_cells, float(opacity))
    return grid


def _write_pvd_collection(pvd_path: Path, vtu_paths: list[Path]) -> None:
    """Write a ParaView .pvd collection referencing each per-sample .vtu.

    This is the standard minimal XML wrapper ParaView expects for grouping
    many datasets into one time-browsable collection (VTKFile type=
    "Collection"); PyVista has no higher-level writer for this non-time-series
    use case, so the wrapper itself is the only hand-written part -- each
    referenced .vtu was written by PyVista/VTK directly. Opening this .pvd
    file in ParaView Desktop lets you scrub through samples as timesteps.
    """
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             "  <Collection>"]
    for i, vtu_path in enumerate(vtu_paths):
        rel = vtu_path.relative_to(pvd_path.parent)
        lines.append(f'    <DataSet timestep="{i}" group="" part="0" file="{rel.as_posix()}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    pvd_path.write_text("\n".join(lines) + "\n")