"""Poster-quality 3D renders of the designs and the eta random field.

    mpirun -n 1 python viz/poster_figures.py p1 [config.yaml]     # erode/nominal/dilate
    mpirun -n 1 python viz/poster_figures.py p3 [config.yaml]     # eta(x) field
    mpirun -n 1 python viz/poster_figures.py p2 [config.yaml]     # nominal vs robust
    mpirun -n 1 python viz/poster_figures.py all [config.yaml]

WHY THIS EXISTS
---------------
viz/plot_research_figures.py produces the three line plots that carry the
argument, but a poster needs geometry. The only 3D render the pipeline emits is
FEniTop's Plotter (src/fenitop/utility.py), which is built for a quick sanity
check, not for print: flat default lighting, a grey constant colour, the model
small in a square frame, and no indication of where the load or the supports
are.

Everything needed for better renders is already on disk. The design variables
are saved per study as .npy (DG0, one value per tet), the mesh rebuilds
deterministically from build_box_fenitop_dicts, and
RandomFieldHeaviside.forward(beta, eta=<float>) re-projects any design at any
threshold -- which is exactly what makes the erode/dilate triptych possible from
a single stored design.

RUN IT ON ONE RANK. Rendering is rank-0 work and takes seconds; the production
sweep holds the other cores. The gather path used here is the same one
save_design_artifacts uses, so nodal ordering matches the mesh.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "figures" / "poster"

# --- look ------------------------------------------------------------------
# One camera and one light rig for every panel, so side-by-side comparisons are
# honest: a design cannot look thinner merely because it was rendered from a
# different angle.
CAMERA = [(38.0, -34.0, 26.0), (5.0, 15.0, 5.0), (0.0, 0.0, 1.0)]
WINDOW = (1700, 1300)
# reset_camera() fits the model to the viewport along the CAMERA direction, so
# framing is driven by the mesh bounds rather than a hand-tuned distance. Zoom
# is then a small trim, never the thing doing the fitting -- a large zoom is how
# the first version cropped the beam's fixed end out of every panel.
ZOOM = 1.35
# The stacked strip panels are wide and short (1500x500), so an isometric view
# wastes the width: the beam's long axis (y, 0..30) recedes into the screen.
# CAMERA_WIDE looks roughly along -x instead, mapping the long axis to screen
# horizontal, with enough offset to keep the render three-dimensional.
CAMERA_WIDE = [(58.0, -10.0, 20.0), (5.0, 15.0, 5.0), (0.0, 0.0, 1.0)]
ZOOM_STRIP = 1.45

COLOURS = {
    "nominal": "#8899a6",     # neutral slate -- the deterministic reference
    "robust": "#2f7d5d",      # green -- the design that survives
    "eroded": "#c0504d",      # red -- material removed, the dangerous case
    "dilated": "#4f81bd",     # blue -- material added
}


def _pv():
    import pyvista as pv
    pv.OFF_SCREEN = True
    return pv


def _style(plotter, grid, colour, opacity=1.0):
    """Consistent material + lighting. smooth_shading hides the tet facets that
    make the default render look like a low-poly blob."""
    plotter.add_mesh(
        grid, color=colour, opacity=opacity,
        smooth_shading=True, specular=0.35, specular_power=18,
        ambient=0.22, diffuse=0.82, show_scalar_bar=False,
    )


def _aim(plotter, zoom: float = ZOOM, camera=None):
    """Point the camera, then let it fit the model. Order matters.

    reset_camera fits the model's bounding SPHERE, which is generous for a beam
    that is 30 long and 10 across seen at an angle -- roughly a third of the
    frame ends up empty. The zoom afterwards reclaims that. It is safe to push
    because the fit has already happened; it is not doing the framing itself.
    """
    plotter.camera_position = camera if camera is not None else CAMERA
    plotter.reset_camera()
    plotter.camera.zoom(zoom)


def _finish(plotter, path: Path, transparent=False):
    plotter.set_background("white")
    _aim(plotter)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(path), transparent_background=transparent)
    plotter.close()
    print(f"  wrote {path.relative_to(ROOT)}")


def _surface(pv, grid_template, nodal_density, threshold=0.5, smooth_iter=60):
    """Isosurface of the projected density -- the actual structure."""
    grid = grid_template.copy()
    grid.point_data["density"] = np.asarray(nodal_density, dtype=float)
    solid = grid.threshold(threshold, scalars="density")
    if solid.n_cells == 0:
        return None
    surf = solid.extract_surface()
    return surf.smooth(n_iter=smooth_iter) if smooth_iter else surf


# --------------------------------------------------------------------------
# FEA-side setup
# --------------------------------------------------------------------------

def _build(config_path: str):
    """Mesh + context on ONE rank, plus a pyvista grid template."""
    from mpi4py import MPI
    import dolfinx

    from src.config.loader import load_config
    from src.fenitop.topopt import topopt
    from src.fenitop.utility import Communicator
    from src.meshing.box_source import build_box_fenitop_dicts
    from src.study_support import build_stage3_kl, setup_context

    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise SystemExit(
            f"Run this on ONE rank (got {comm.size}). Rendering is rank-0 work "
            "and the sample-parallel group split is pointless here."
        )

    # topopt writes its own preview image to output_prefix, so the directory has
    # to exist before the warm start runs, not just before the first render.
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    rho_nominal = topopt(fem, opt, load_cases, output_prefix=str(OUT / "_warm_"))
    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_nominal, kl_result, load_cases, case_name)

    pv = _pv()
    cells, cell_types, nodes = dolfinx.plot.vtk_mesh(fem["mesh"], fem["mesh"].topology.dim)
    grid_template = pv.UnstructuredGrid(cells, cell_types, nodes)
    phys_comm = Communicator(ctx.rho_phys_field.function_space, fem["mesh_serial"])

    return cfg, ctx, opt, kl_result, pv, grid_template, phys_comm


def _project(ctx, phys_comm, rho_design: np.ndarray, beta: float, eta) -> np.ndarray:
    """rho (DG0) -> filter -> Heaviside(beta, eta) -> nodal physical density."""
    ctx.rho_field.x.petsc_vec.array[:] = np.asarray(rho_design, dtype=float)
    ctx.rho_field.x.scatter_forward()
    ctx.density_filter.forward()
    ctx.rf_heaviside.forward(beta, eta=eta)
    return np.asarray(phys_comm.gather(ctx.rho_phys_field))


def _load_design(name: str) -> np.ndarray:
    path = ROOT / "output" / "studies" / "uniform_eta" / name
    if not path.exists():
        raise SystemExit(f"missing design {path} -- run E1 first")
    return np.load(path)


# --------------------------------------------------------------------------
# P1 -- erode / nominal / dilate
# --------------------------------------------------------------------------

def figure_p1(cfg, ctx, opt, pv, grid_template, phys_comm) -> None:
    """The same design projected at three thresholds.

    This is the figure that makes the premise physical: identical design
    variables, three manufacturing outcomes. eta HIGH removes material (eroded),
    eta LOW adds it (dilated) -- the inverse relationship is the thing readers
    get wrong, so the panels are labelled with both.
    """
    beta = float(cfg.optimization.saa_beta_max)
    eta_lo, eta_hi = cfg.random_field.eta_min, cfg.random_field.eta_max
    design = _load_design("rho_uniform.npy")

    panels = [
        (eta_hi, "eroded", f"eta = {eta_hi:g}  (over-etched, thinner)"),
        (0.5, "nominal", "eta = 0.5  (as designed)"),
        (eta_lo, "dilated", f"eta = {eta_lo:g}  (under-etched, thicker)"),
    ]

    # Individual panels, so they can be laid out freely on the poster...
    for eta, key, label in panels:
        nodal = _project(ctx, phys_comm, design, beta, float(eta))
        surf = _surface(pv, grid_template, nodal)
        p = pv.Plotter(off_screen=True, window_size=WINDOW)
        if surf is not None:
            _style(p, surf, COLOURS[key])
        _finish(p, OUT / f"P1_{key}.png", transparent=True)
        print(f"    {label}: volume fraction {nodal.mean():.4f}")

    # ...and one combined strip, which is usually what actually gets used.
    # STACKED VERTICALLY, not side by side: the beam is 30 long and 10 across,
    # so three narrow columns crop its fixed end. Three wide rows fit it.
    p = pv.Plotter(off_screen=True, window_size=(1500, 1500), shape=(3, 1))
    for i, (eta, key, label) in enumerate(panels):
        nodal = _project(ctx, phys_comm, design, beta, float(eta))
        surf = _surface(pv, grid_template, nodal)
        p.subplot(i, 0)
        if surf is not None:
            _style(p, surf, COLOURS[key])
        # Normalized viewport coords, not "upper_left": the named position
        # clips the first glyph against the subplot border.
        p.add_text(label, font_size=12, color="black", position=(0.03, 0.86),
                   viewport=True)
        _aim(p, ZOOM_STRIP, CAMERA_WIDE)
    p.set_background("white")
    OUT.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(OUT / "P1_triptych.png"))
    p.close()
    print(f"  wrote {(OUT / 'P1_triptych.png').relative_to(ROOT)}")


# --------------------------------------------------------------------------
# P2 -- deterministic vs robust
# --------------------------------------------------------------------------

def figure_p2(cfg, ctx, opt, pv, grid_template, phys_comm) -> None:
    """The payoff: same volume budget, very different sensitivity."""
    beta = float(cfg.optimization.saa_beta_max)
    for name, key, label in (
        ("rho_nominal.npy", "nominal", "deterministic"),
        ("rho_uniform.npy", "robust", "robust"),
    ):
        design = _load_design(name)
        nodal = _project(ctx, phys_comm, design, beta, 0.5)
        surf = _surface(pv, grid_template, nodal)
        p = pv.Plotter(off_screen=True, window_size=WINDOW)
        if surf is not None:
            _style(p, surf, COLOURS[key])
        _finish(p, OUT / f"P2_{label}.png", transparent=True)
        print(f"    {label}: volume fraction {nodal.mean():.4f}")


# --------------------------------------------------------------------------
# P3 -- the eta random field itself
# --------------------------------------------------------------------------

def figure_p3(cfg, ctx, kl_result, pv, grid_template) -> None:
    """A realization of eta(x), colour-mapped on the domain.

    No FEA involved -- this is the input to the projection, and it is what makes
    the method visibly different from a scalar threshold. Two realizations are
    rendered so a reader can see that the PATTERN changes while the marginal
    does not.
    """
    from src.random_fields.kl_expansion import evaluate_field_from_xi, pointwise_std
    from src.random_fields.threshold_transform import (
        MarginalTransformParams, ThresholdMarginalTransform,
    )

    transform = ThresholdMarginalTransform(MarginalTransformParams(
        eta_min=cfg.random_field.eta_min, eta_max=cfg.random_field.eta_max,
        alpha=cfg.random_field.alpha, beta=cfg.random_field.beta,
    ))
    std = pointwise_std(kl_result)

    for k, seed in enumerate((3, 11)):
        xi = np.random.default_rng(seed).standard_normal(kl_result.n_kl)
        eta = transform.transform(evaluate_field_from_xi(kl_result, xi) / std)

        grid = grid_template.copy()
        grid.point_data["eta"] = eta
        p = pv.Plotter(off_screen=True, window_size=WINDOW)
        p.add_mesh(
            grid, scalars="eta", cmap="RdYlBu_r",
            clim=[cfg.random_field.eta_min, cfg.random_field.eta_max],
            smooth_shading=True, show_scalar_bar=True,
            scalar_bar_args=dict(title="eta(x)", color="black", n_labels=3,
                                 title_font_size=18, label_font_size=15),
        )
        _finish(p, OUT / f"P3_eta_field_{k}.png")
        print(f"    realization {k}: eta in [{eta.min():.3f}, {eta.max():.3f}], "
              f"mean {eta.mean():.3f}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    args = [a for a in sys.argv[1:] if not a.endswith((".yaml", ".yml"))]
    cfgs = [a for a in sys.argv[1:] if a.endswith((".yaml", ".yml"))]
    which = (args[0] if args else "all").lower()
    config_path = cfgs[0] if cfgs else "src/config/configStudy.yaml"

    cfg, ctx, opt, kl_result, pv, grid_template, phys_comm = _build(config_path)
    print(f"building poster figures ({which}) from {config_path}")

    if which in ("p1", "all"):
        figure_p1(cfg, ctx, opt, pv, grid_template, phys_comm)
    if which in ("p3", "all"):
        figure_p3(cfg, ctx, kl_result, pv, grid_template)
    if which in ("p2", "all"):
        figure_p2(cfg, ctx, opt, pv, grid_template, phys_comm)


if __name__ == "__main__":
    main()
