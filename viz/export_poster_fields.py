"""Export every poster-relevant field into ONE ParaView file.

    mpirun -n 1 python viz/export_poster_fields.py [config.yaml]

    -> output/paraview/poster/poster_fields.vtu

WHY ONE FILE
------------
The designs this week's studies produced are stored as .npy of the DG0 design
variable rho -- not as anything ParaView can open, and not as the projected
physical density that actually defines the structure. The eroded and dilated
realizations do not exist on disk at all; they are produced by re-projecting a
stored design at a different threshold.

Rather than emit a dozen files, this writes a single unstructured grid carrying
every field as a separate POINT DATA array. In ParaView you then load one file
and switch arrays from the dropdown, which keeps the camera, the Threshold
filter and the colour map fixed while you flip between cases -- so panels are
directly comparable instead of accidentally rendered at different angles.

ARRAYS WRITTEN
--------------
  density_eta075   robust design, over-etched   (thinner)
  density_eta050   robust design, as designed
  density_eta025   robust design, under-etched  (thicker)
  density_nominal  deterministic design at eta = 0.5
  density_robust   robust design at eta = 0.5   (same as density_eta050)
  density_field    field-optimized design at eta = 0.5
  eta_sample_0/1   two realizations of the eta(x) random field itself

Every density array is the PROJECTED physical density in [0,1]; threshold at
0.5 to get the structure. eta_* arrays are the threshold field, in
[eta_min, eta_max] -- colour-map those, do not threshold them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "paraview" / "poster"


def main() -> None:
    from mpi4py import MPI
    import dolfinx
    import pyvista as pv

    from src.config.loader import load_config
    from src.fenitop.topopt import topopt
    from src.fenitop.utility import Communicator
    from src.meshing.box_source import build_box_fenitop_dicts
    from src.random_fields.kl_expansion import evaluate_field_from_xi, pointwise_std
    from src.random_fields.threshold_transform import (
        MarginalTransformParams, ThresholdMarginalTransform,
    )
    from src.study_support import build_stage3_kl, setup_context

    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise SystemExit(f"run on ONE rank (got {comm.size})")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/config/configStudy.yaml"
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    tagged_mesh, fem, opt, load_cases = build_box_fenitop_dicts(cfg, comm)
    case_name = next(iter(load_cases))
    rho_warm = topopt(fem, opt, load_cases, output_prefix=str(OUT / "_warm_"))
    kl_result = build_stage3_kl(cfg, tagged_mesh, comm)
    ctx = setup_context(fem, opt, rho_warm, kl_result, load_cases, case_name)

    cells, cell_types, nodes = dolfinx.plot.vtk_mesh(
        fem["mesh"], fem["mesh"].topology.dim
    )
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    phys = Communicator(ctx.rho_phys_field.function_space, fem["mesh_serial"])
    beta = float(cfg.optimization.saa_beta_max)

    def project(design: np.ndarray, eta: float) -> np.ndarray:
        ctx.rho_field.x.petsc_vec.array[:] = np.asarray(design, dtype=float)
        ctx.rho_field.x.scatter_forward()
        ctx.density_filter.forward()
        ctx.rf_heaviside.forward(beta, eta=float(eta))
        return np.asarray(phys.gather(ctx.rho_phys_field))

    def load(name: str) -> np.ndarray | None:
        p = ROOT / "output" / "studies" / "uniform_eta" / name
        if not p.exists():
            print(f"  [skip] {name} not found")
            return None
        return np.load(p)

    eta_lo = float(cfg.random_field.eta_min)
    eta_hi = float(cfg.random_field.eta_max)

    robust = load("rho_uniform.npy")
    if robust is not None:
        for eta, tag in ((eta_hi, "075"), (0.5, "050"), (eta_lo, "025")):
            arr = project(robust, eta)
            grid.point_data[f"density_eta{tag}"] = arr
            print(f"  density_eta{tag}: volume fraction {arr.mean():.4f}")
        grid.point_data["density_robust"] = grid.point_data["density_eta050"]

    nominal = load("rho_nominal.npy")
    if nominal is not None:
        grid.point_data["density_nominal"] = project(nominal, 0.5)

    field = load("rho_field.npy")
    if field is not None:
        grid.point_data["density_field"] = project(field, 0.5)

    # eta(x) itself -- no FEA, this is the INPUT to the projection.
    transform = ThresholdMarginalTransform(MarginalTransformParams(
        eta_min=eta_lo, eta_max=eta_hi,
        alpha=cfg.random_field.alpha, beta=cfg.random_field.beta,
    ))
    std = pointwise_std(kl_result)
    for k, seed in enumerate((3, 11)):
        xi = np.random.default_rng(seed).standard_normal(kl_result.n_kl)
        eta_field = transform.transform(evaluate_field_from_xi(kl_result, xi) / std)
        grid.point_data[f"eta_sample_{k}"] = eta_field
        print(f"  eta_sample_{k}: range [{eta_field.min():.3f}, {eta_field.max():.3f}]")

    out = OUT / "poster_fields.vtu"
    grid.save(out)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"arrays: {', '.join(grid.point_data.keys())}")


if __name__ == "__main__":
    main()
