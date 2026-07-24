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

    # run_monte_carlo_validation(write_ensemble=True) has ALREADY written one
    # .vtu per sample (each carrying point_data["density"] = that sample's
    # rho_phys), the ensemble.pvd collection, and the normalized opacity weights.
    # We consume those on-disk artifacts to assemble the single merged,
    # opacity-weighted probability_cloud.vtp (masterContext 4.1), rather than
    # requiring MCResult to also hold the (large) per-sample nodal density
    # fields in memory. (The legacy `.samples`-record schema this module's
    # docstring describes was never implemented in MCResult.)
    pvd_path = getattr(mc_result, "ensemble_pvd_path", None)
    if pvd_path is None:
        logger.warning(
            "MCResult has no ensemble_pvd_path (run_monte_carlo_validation ran "
            "with write_ensemble=False) -- no per-sample ensemble on disk to "
            "assemble; skipping probability_cloud.vtp."
        )
        return {"pvd": None, "vtp": None}
    pvd_path = Path(pvd_path)
    ensemble_dir = pvd_path.parent / "ensemble"
    vtu_files = sorted(ensemble_dir.glob("sample_*.vtu"))
    if not vtu_files:
        logger.warning("No per-sample VTUs under %s; skipping probability_cloud.vtp.",
                       ensemble_dir)
        return {"pvd": pvd_path, "vtp": None}
    n_total = len(vtu_files)

    # Opacity weights: prefer the ones MC already computed & normalized; else
    # derive them from the retained iid-N(0,1) KL coordinates.
    weights = getattr(mc_result, "probability_weights", None)
    if weights is None:
        weights = _sample_probability_weights(np.asarray(mc_result.xi_samples, dtype=float))
    weights = np.asarray(weights, dtype=float)
    compliance = np.asarray(mc_result.compliance_samples, dtype=float)

    # Defensive: n_total came from a glob of ensemble_dir on disk, which is
    # only guaranteed to match THIS run's sample count if the directory was
    # freshened before writing (see monte_carlo.py). Don't trust it blindly --
    # clamp to the smallest of the three so a stale/pre-populated directory
    # can't index past this run's actual weights/compliance arrays.
    n_matched = min(n_total, weights.size, compliance.size)
    if n_matched < n_total:
        logger.warning(
            "Ensemble dir %s has %d VTU file(s) but this run's MCResult only "
            "has %d sample(s) (probability_weights/compliance_samples) -- "
            "likely stale files left by a previous run. Using only the "
            "first %d.", ensemble_dir, n_total, n_matched, n_matched,
        )
    n_total = n_matched
    vtu_files = vtu_files[:n_total]

    # Deterministic subsample to n_vis.
    idx = np.arange(n_total)
    if config.n_vis < n_total:
        rng = np.random.default_rng(config.seed)
        idx = np.sort(rng.choice(n_total, size=config.n_vis, replace=False))

    opacity = _map_to_opacity(weights[idx], config.min_opacity, config.max_opacity)

    grids: list[pv.UnstructuredGrid] = []
    for k, i in enumerate(idx):
        grid = pv.read(str(vtu_files[i]))  # PyVista/VTK native reader (has "density")
        n_pts = grid.n_points
        grid.point_data["opacity"] = np.full(n_pts, float(opacity[k]))
        grid.point_data["sample_probability"] = np.full(
            n_pts, float(weights[i]) if i < weights.size else np.nan)
        grid.point_data["compliance"] = np.full(
            n_pts, float(compliance[i]) if i < compliance.size else np.nan)
        grids.append(grid)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtp_path = output_dir / "probability_cloud.vtp"

    # Single batched merge (DataSet.merge accepts a list -> one pass), then a
    # native surface extraction/writer.
    merged = grids[0].merge(grids[1:]) if len(grids) > 1 else grids[0]
    merged.extract_surface().save(str(vtp_path))

    logger.info(
        "Stage 6 viz: wrote %s from %d/%d MC ensemble samples (per-sample "
        "collection already at %s) -- open either directly in ParaView Desktop.",
        vtp_path, len(idx), n_total, pvd_path,
    )
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