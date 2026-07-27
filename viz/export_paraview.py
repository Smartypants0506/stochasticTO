"""viz/export_paraview.py -- assemble a tidy, ready-to-open output/paraview/.

No `paraview` import: everything here is pyvista/VTK, which is already in the
container image (requirements.txt pins pyvista==0.44.2, vtk==9.3.1). It reads
whatever the earlier stages left under output/ and writes a single flat
directory of self-contained files you open by hand in ParaView Desktop.

It is safe to run at any point -- each section is skipped with a printed note
if its inputs are missing, so you can run it after step 1 and again after
step 3 without cleaning anything up.

WHAT IT WRITES (output/paraview/)
---------------------------------
  README.txt                     per-file cheat sheet: which array to colour
                                 by, which range, which representation
  field_ranges.csv               global min/max/p01/p99 of every field across
                                 the whole ensemble -- paste these into
                                 ParaView's "Rescale to custom range" so an
                                 ensemble animation stays comparable frame to
                                 frame (see NOTE below)

  01_cloud_all.vtp               all N boundary surfaces, layered
  01_cloud_likely_top50.vtp      the more-likely-than-median half
  01_cloud_likely_top10.vtp      the 10% most typical parts
  01_cloud_worst5pct.vtp         the 5% worst-compliance parts
  01_cloud_extreme5pct.vtp       the 5% rarest geometries
  01_mean_shape.vtp              the ensemble-mean boundary, as a reference

  02_ensemble_fea.pvd            enriched per-realization fields, as a time
                                 series (one timestep = one realization)
  02_ensemble_density.pvd        the raw density-only ensemble, same idea

  03_risk_delta.vtu              every design's mean/std/prob_void plus the
                                 baseline-minus-design differences
  03_uncertain_<design>.vtu      only the nodes with 0.02 < prob_void < 0.98,
                                 i.e. the material whose existence is a coin
                                 flip at manufacturing time
  03_shape_<design>.vtp          each design's mean rho = 0.5 boundary

NOTE on field_ranges.csv -- the one thing that is easy to get wrong by hand.
ParaView defaults to rescaling the colour map to the current timestep. On an
ensemble that is actively misleading: every frame renders equally hot and the
variation you are trying to show disappears. Turn off "Rescale on Play"
(Edit -> Settings -> General -> "Automatically rescale...") and set a custom
range from this file instead. For stress use the p01/p99 columns rather than
min/max -- a handful of nodes at a re-entrant corner otherwise own the whole
scale.

USAGE
    python viz/export_paraview.py
    python viz/export_paraview.py --out-dir output/paraview
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv


def _merge_polydata(pieces: list[pv.PolyData]) -> pv.PolyData:
    """Merge and guarantee a PolyData back, so `.save('*.vtp')` is legal.

    pyvista's DataSet.merge returns an UnstructuredGrid for some input
    combinations; extract_surface() is a no-op on data that is already a
    surface, so this is cheap insurance rather than a conversion.
    """
    merged = pieces[0].merge(pieces[1:]) if len(pieces) > 1 else pieces[0].copy()
    if not isinstance(merged, pv.PolyData):
        merged = merged.extract_surface()
    return merged


def _read_index(path: Path) -> tuple[list[str], np.ndarray]:
    header = path.read_text().splitlines()[0].split(",")
    return header, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def _write_pvd(pvd_path: Path, entries: list[tuple[float, str]]) -> None:
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             "  <Collection>"]
    for t, rel in entries:
        lines.append(f'    <DataSet timestep="{t}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    pvd_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 01 -- the probability cloud and its pre-filtered subsets
# ---------------------------------------------------------------------------
def export_cloud(viz_dir: Path, mc_dir: Path, out: Path, notes: list[str]) -> None:
    index_path = viz_dir / "sample_index.csv"
    surf_dir = viz_dir / "surfaces"
    if not index_path.exists() or not surf_dir.exists():
        notes.append("01_* skipped: run `python viz/build_cloud_index.py` first.")
        return

    header, table = _read_index(index_path)
    col = {name: i for i, name in enumerate(header)}
    like = table[:, col["occurrence_likelihood"]]
    cpct = table[:, col["compliance_percentile"]]
    ids = table[:, col["sample_index"]].astype(int)

    cache: dict[int, pv.PolyData] = {}

    def load(i: int) -> pv.PolyData | None:
        if i not in cache:
            p = surf_dir / f"sample_{i:05d}.vtp"
            cache[i] = pv.read(str(p)) if p.exists() else None
        return cache[i]

    subsets = {
        "01_cloud_all.vtp": np.ones(ids.size, dtype=bool),
        "01_cloud_likely_top50.vtp": like >= np.median(like),
        "01_cloud_likely_top10.vtp": like >= np.percentile(like, 90),
        "01_cloud_worst5pct.vtp": cpct > 0.95,
        "01_cloud_extreme5pct.vtp": like <= np.percentile(like, 5),
    }
    for name, mask in subsets.items():
        pieces = [s for s in (load(int(i)) for i in ids[mask]) if s is not None]
        if not pieces:
            notes.append(f"{name} skipped: no surfaces matched.")
            continue
        merged = _merge_polydata(pieces)
        merged.save(str(out / name))
        notes.append(f"{name}: {len(pieces)} realizations, "
                     f"{merged.n_cells} triangles.")

    rel = mc_dir / "reliability_map.vtu"
    if rel.exists():
        shape = pv.read(str(rel)).contour(isosurfaces=[0.5], scalars="mean_density")
        shape.save(str(out / "01_mean_shape.vtp"))
        notes.append("01_mean_shape.vtp: ensemble-mean boundary.")

    # Written only when build_cloud_index.py ran with --deviation-scale != 1:
    # the TRUE-scale boundary the exaggerated cloud is measured from.
    ref = viz_dir / "reference_surface.vtp"
    if ref.exists():
        shutil.copy2(ref, out / "01_reference_surface.vtp")
        notes.append("01_reference_surface.vtp: unamplified reference boundary "
                     "(the cloud around it is exaggerated -- see its "
                     "deviation_scale array).")


# ---------------------------------------------------------------------------
# 02 -- ensemble time series + the global field ranges
# ---------------------------------------------------------------------------
def export_ensemble(fea_dir: Path, mc_dir: Path, out: Path,
                    notes: list[str]) -> None:
    ranges: dict[str, tuple[float, float, float, float]] = {}

    def collect(ens_dir: Path, pvd_name: str, label: str, do_ranges: bool) -> None:
        vtus = sorted(ens_dir.glob("sample_*.vtu"))
        if not vtus:
            notes.append(f"{pvd_name} skipped: no VTUs under {ens_dir}.")
            return
        entries = [(float(t), os.path.relpath(p, start=out))
                   for t, p in enumerate(vtus)]
        _write_pvd(out / pvd_name, entries)
        notes.append(f"{pvd_name}: {len(vtus)} realizations ({label}).")

        if not do_ranges:
            return
        # Streaming min/max plus a pooled sample for robust percentiles, so we
        # never hold the whole ensemble in memory.
        pooled: dict[str, list[np.ndarray]] = {}
        for p in vtus:
            g = pv.read(str(p))
            for name in g.point_data.keys():
                a = np.asarray(g.point_data[name])
                a = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
                lo, hi = float(a.min()), float(a.max())
                if name in ranges:
                    plo, phi, _, _ = ranges[name]
                    ranges[name] = (min(plo, lo), max(phi, hi), 0.0, 0.0)
                else:
                    ranges[name] = (lo, hi, 0.0, 0.0)
                pooled.setdefault(name, []).append(a[::37])  # thin, unbiased
        for name, chunks in pooled.items():
            a = np.concatenate(chunks)
            lo, hi, _, _ = ranges[name]
            ranges[name] = (lo, hi, float(np.percentile(a, 1)),
                            float(np.percentile(a, 99)))

    if fea_dir.exists():
        collect(fea_dir / "ensemble", "02_ensemble_fea.pvd",
                "enriched: eta / displacement / von Mises / SED", True)
    else:
        notes.append("02_ensemble_fea.pvd skipped: run "
                     "`mpirun -n 8 python viz/enrich_ensemble_fea.py`.")
    collect(mc_dir / "ensemble", "02_ensemble_density.pvd", "density only",
            not fea_dir.exists())

    if ranges:
        lines = ["field,min,max,p01,p99"]
        for name in sorted(ranges):
            lo, hi, p01, p99 = ranges[name]
            lines.append(f"{name},{lo:.10e},{hi:.10e},{p01:.10e},{p99:.10e}")
        (out / "field_ranges.csv").write_text("\n".join(lines) + "\n")
        notes.append(f"field_ranges.csv: {len(ranges)} fields.")


# ---------------------------------------------------------------------------
# 03 -- the robust-vs-deterministic comparison
# ---------------------------------------------------------------------------
def export_comparison(comparison_dir: Path, out: Path, notes: list[str]) -> None:
    delta = comparison_dir / "risk_delta.vtu"
    if not delta.exists():
        notes.append("03_* skipped: run "
                     "`mpirun -n 8 python viz/compare_designs_mc.py`.")
        return

    grid = pv.read(str(delta))
    grid.save(str(out / "03_risk_delta.vtu"))
    arrays = list(grid.point_data.keys())
    designs = sorted({a[len("prob_void_"):] for a in arrays
                      if a.startswith("prob_void_")})
    notes.append(f"03_risk_delta.vtu: designs = {designs}")

    for name in designs:
        band = grid.threshold((0.02, 0.98), scalars=f"prob_void_{name}")
        if band.n_cells:
            band.save(str(out / f"03_uncertain_{name}.vtu"))
            frac = 100.0 * band.n_cells / grid.n_cells
            notes.append(f"03_uncertain_{name}.vtu: {frac:.1f}% of the mesh is "
                         "neither reliably solid nor reliably void.")
        else:
            notes.append(f"03_uncertain_{name}.vtu skipped: prob_void is 0/1 "
                         "everywhere -- the eta band is too narrow to move the "
                         "boundary. Widen random_field.eta_min/eta_max.")
        shape = grid.contour(isosurfaces=[0.5], scalars=f"mean_density_{name}")
        if shape.n_points:
            shape.save(str(out / f"03_shape_{name}.vtp"))


_README = """\
output/paraview/ -- open these directly in ParaView Desktop.
Written by viz/export_paraview.py. Nothing here needs a Python shell.

GLOBAL SETUP, DO THIS ONCE
  Edit -> Settings -> General -> untick "Automatically rescale to data range".
  Otherwise ParaView re-scales the colour map on every timestep and the whole
  ensemble renders equally hot. Set ranges by hand from field_ranges.csv.
  For stress use the p01/p99 columns, not min/max -- a few nodes at a
  re-entrant corner otherwise own the entire scale.

--------------------------------------------------------------------------
01 -- THE PROBABILITY CLOUD (layered possible geometries)
--------------------------------------------------------------------------
  01_cloud_all.vtp            every realization's rho = 0.5 boundary, stacked
  01_cloud_likely_top50.vtp   the more-likely-than-median half
  01_cloud_likely_top10.vtp   the 10% most typical parts
  01_cloud_worst5pct.vtp      the 5% worst-compliance parts
  01_cloud_extreme5pct.vtp    the 5% rarest geometries
  01_mean_shape.vtp           the ensemble-mean boundary
  01_reference_surface.vtp    the true-scale reference boundary (only when the
                              cloud was built with --deviation-scale != 1)

  Suggested: Representation = Surface, colour by compliance_z with
  "Cool to Warm (Extended)" rescaled to [-3, 3], and let the baked-in alpha
  do the fading: Properties -> Opacity By Array -> `opacity`, then tick
  "Enable opacity mapping for surfaces" on the colour map and set its opacity
  ramp to the identity on [0, 1]. Typical (near-nominal) geometries then
  render solid and rare, far-out ones fade toward invisible, so the cloud
  reads as a dense core inside a faint envelope. Without that, a flat
  Opacity ~ 0.10 for every layer is the fallback.
  Load 01_mean_shape.vtp alongside, opaque dark grey, as the reference
  silhouette. Where the cloud hugs the mean shape the geometry is certain;
  where it fans out, manufacturing variation is really moving the boundary.

  All five files carry the same point arrays, so you can also open
  01_cloud_all.vtp alone and drive it with a Threshold filter:
      opacity                 per-layer alpha, high for typical geometries and
                              low for rare/far-out ones. Use it as the
                              separate opacity array (above), not as a colour.
      deviation_scale         the exaggeration factor these surfaces were
                              built with. 1 = true geometry; anything else
                              MUST be stated in the figure caption.
      occurrence_likelihood   1 = a perfectly typical part, 0 = a rare one.
                              Threshold [0.5, 1] = "what the process
                              routinely produces". Whatever spread survives
                              at [0.75, 1] is ROUTINE variation, not tail risk.
      radius_percentile       1 - occurrence_likelihood.
      compliance_percentile   1 = the softest/worst part in the ensemble.
      compliance_z            (C - mean)/std for that realization.
      is_tail_95              1 for the worst 5% of outcomes.
      sample_index            maps back to ensemble/sample_XXXXX.vtu.

  Worth checking: open 01_cloud_worst5pct.vtp and colour by
  occurrence_likelihood. If the worst parts are ORDINARY realizations rather
  than freak ones, the risk is not a tail curiosity -- it is the median
  outcome of the process, and the deterministic optimum was never describing
  the part you would actually get.

  Why not output/stage6_validation/probability_cloud/probability_cloud.vtp:
  it is 256 MB of merged full-volume tetrahedra, and its opacity array is
  exp(-0.5*||xi||^2) in 37 dimensions, which spans 10 orders of magnitude
  across this ensemble -- any linear transfer function on it shows one
  sample and hides the other 99. See viz/build_cloud_index.py.

--------------------------------------------------------------------------
02 -- FEA FIELDS PER REALIZATION
--------------------------------------------------------------------------
  02_ensemble_fea.pvd      one timestep per realization; use the VCR controls
  02_ensemble_density.pvd  the original density-only ensemble

  Apply Threshold on density in [0.5, 1] FIRST, then colour. Fields:
      eta                    the sampled manufacturing threshold -- the CAUSE.
                             eta > 0.5 = under-deposition at that point.
      von_mises              macroscopic stress. MEANINGLESS in void: it
                             carries the SIMP scaling. Always threshold first.
      von_mises_solid        stress in the actual solid phase, SIMP factor
                             divided out. This is the yielding-relevant number
                             and the one that spikes in thin ligaments.
      strain_energy_density  where compliance is spent; integrates to that
                             sample's C. Try log scale.
      displacement           3-vector -> Warp By Vector. Fix the scale factor
                             manually, the same value for every frame, or the
                             animation is meaningless.
      displacement_magnitude scalar deformation.
      density                rho_phys, for masking.

  The thing to look for: scrub von_mises_solid through the ensemble. If the
  hot spot MOVES and SPIKES between realizations, the design depends on
  ligaments that manufacturing variation is thinning. If it stays put, the
  design is insensitive. That is the failure mode robust TO insures against,
  and it is visible frame by frame rather than argued.

--------------------------------------------------------------------------
03 -- ROBUST vs DETERMINISTIC (the argument)
--------------------------------------------------------------------------
  03_risk_delta.vtu        all designs on one mesh:
        mean_density_<d>     ensemble-mean density
        std_density_<d>      boundary jitter across the ensemble
        prob_void_<d>        P(this point is void)
        d_prob_void_<d>      baseline minus design. POSITIVE = robust TO
                             removed manufacturing risk at that point.
        d_std_density_<d>    baseline minus design boundary jitter.
  03_uncertain_<design>.vtu  only the nodes with 0.02 < prob_void < 0.98
  03_shape_<design>.vtp      each design's mean boundary

  Suggested: colour d_prob_void_robust_lambda1 with "Cool to Warm (Extended)"
  on a SYMMETRIC range so zero is white and the sign is readable. Warm = the
  robust design removed the risk of that feature not existing. This localizes
  the benefit instead of asserting it, and it is the image people remember.

  03_uncertain_<design>.vtu answers a question deterministic TO cannot even
  ask: how much of your structure is not guaranteed to exist? Compare the
  nominal and robust files side by side in a split view.

  Load every 03_shape_*.vtp at once, different solid colours, robust ones at
  ~45% opacity, to see HOW the designs differ -- robust designs usually show
  thicker members, fewer knife-edge junctions, more redundant load paths.
  Being able to point at that turns a statistics claim into a design claim.

  Numbers to quote alongside: output/comparison/summary.json (per-design mean
  / std / p95 / worst case, and the paired win rate) and output/figures/
  fig1..fig5 from viz/plot_comparison.py.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viz-dir", type=Path, default=Path("output/viz"))
    ap.add_argument("--mc-dir", type=Path, default=Path("output/mc_validation"))
    ap.add_argument("--fea-dir", type=Path, default=Path("output/viz/ensemble_fea"))
    ap.add_argument("--comparison-dir", type=Path, default=Path("output/comparison"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/paraview"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    export_cloud(args.viz_dir, args.mc_dir, args.out_dir, notes)
    export_ensemble(args.fea_dir, args.mc_dir, args.out_dir, notes)
    export_comparison(args.comparison_dir, args.out_dir, notes)

    (args.out_dir / "README.txt").write_text(
        _README + "\n\nTHIS RUN\n--------\n" + "\n".join(f"  {n}" for n in notes) + "\n")
    print(f"\n-> {args.out_dir}")
    for n in notes:
        print(f"   {n}")
    print(f"\n   open {args.out_dir}/README.txt for what to colour by.")


if __name__ == "__main__":
    main()
