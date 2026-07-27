"""viz/build_cloud_index.py -- turn the raw Stage-6 MC ensemble into something
ParaView can actually filter by likelihood.

WHY THIS EXISTS
---------------
`output/mc_validation/probability_weights.csv` (written by
src/validation/monte_carlo.py) stores

    w_i = exp(-0.5 * ||xi_i||^2)   normalized so max_i w_i = 1

That is proportional to the true N(0, I) *density* at xi_i, and it is the right
thing mathematically -- but it is the wrong thing to hand to a ParaView
transfer function in N_KL = 37 dimensions. In 37D the density at the mode
(xi = 0) is astronomically larger than the density anywhere the samples
actually live: this run's weights span ~10 orders of magnitude
(5.7e-11 .. 1.0) with a median of 3.1e-4. A linear opacity/threshold on that
array shows exactly one sample and hides the other 99.

The statistically correct notion of "how likely is a geometry like this one"
in high dimensions is based on the *radius*, not the density. For iid
xi ~ N(0, I_d), ||xi||^2 ~ chi^2_d, so

    radius_percentile_i = F_{chi^2_d}( ||xi_i||^2 )

is uniform on [0, 1] by construction. radius_percentile = 0.5 is a perfectly
typical realization; 0.99 means "only 1% of manufacturable parts deviate more
than this one". We expose

    occurrence_likelihood = 1 - radius_percentile

so that a ParaView Threshold of `occurrence_likelihood > 0.5` reads literally
as "show me only the more-likely-than-median half of the deformation modes",
and `occurrence_likelihood < 0.05` isolates the 5% rare/extreme tail.

FADING THE UNLIKELY LAYERS OUT (`opacity`)
------------------------------------------
Every surface also carries a ready-to-render alpha,

    opacity = lo + (hi - lo) * occurrence_likelihood ** gamma

(defaults lo=0.02, hi=0.30, gamma=1). Because occurrence_likelihood is uniform
on [0, 1] by construction, this spreads the ensemble evenly across the alpha
band instead of piling it up at one end: the typical, near-nominal geometries
render solid and the rare, far-out ones fade to nearly nothing, so the cloud
reads as a dense core with a wispy envelope rather than 100 equal shells. Drive
it in ParaView with the representation's **Use Separate Opacity Array** ->
`opacity` (viz/make_paraview_state.py does this for you), with the colour map's
"Enable opacity mapping for surfaces" ticked and an identity opacity ramp.

This is NOT `opacity_weight`/`log10_opacity_weight`, which are the raw
exp(-0.5||xi||^2) and are unusable as alpha for the reason given above; those
stay in the output only for provenance.

EXAGGERATING THE DEVIATIONS (--deviation-scale)
-----------------------------------------------
The manufacturing deviations are physically small: at rho = 0.5 the 100
realizations sit within a fraction of a filter radius of each other, so the
raw cloud renders as one solid-looking shell. `--deviation-scale K` scales
each realization's departure from a reference field before contouring,

    rho_i^amplified(x) = rho_ref(x) + K * ( rho_i(x) - rho_ref(x) )

which moves that sample's rho = 0.5 boundary by approximately K times its true
normal offset (exactly K x, to first order: the level set shifts by
-delta_rho / |grad rho|, and the amplification is linear in delta_rho). K = 1
is the untouched geometry and is still the default; K = 10 is a good starting
point for a figure. This is the same convention as a mode-shape plot in FEA --
the *shape* of the variability is faithful, the magnitude is not, so any
figure made with K != 1 must say so. The factor is baked into every surface as
the point array `deviation_scale` so it cannot be lost.

The linearization degrades as K grows: away from the boundary rho saturates at
0 / 1 and the sample-to-sample deviation decays to ~1e-6 there, so amplified
interior artifacts are not a practical concern, but at very large K (>~50 for
this run) the amplified boundary offset stops being proportional to the true
one. Prefer the smallest K that separates the layers.

WHAT IT WRITES (all under output/viz/)
--------------------------------------
  sample_index.csv                 per-sample scalars (see COLUMNS below)
  probability_cloud_surfaces.vtp   merged rho = 0.5 iso-surface of every
                                   sample, each carrying the per-sample
                                   scalars (plus `opacity` and
                                   `deviation_scale`) as point data -> the
                                   layered "probability cloud" (deliverable #1)
  reference_surface.vtp            the rho = 0.5 iso-surface of the reference
                                   field the deviations are measured from --
                                   the "unamplified nominal" to show the cloud
                                   against
  surfaces/sample_XXXXX.vtp        the individual iso-surfaces
  surfaces_by_likelihood.pvd       the same surfaces as a ParaView time
                                   series ordered most-likely -> least-likely,
                                   so scrubbing the animation walks outward
                                   from the nominal geometry
  ensemble_by_likelihood.pvd       the ORIGINAL full-volume ensemble VTUs,
                                   re-indexed in the same likelihood order.
                                   NOTE: these point at the raw .vtu files on
                                   disk, so they are never amplified --
                                   --deviation-scale only affects the
                                   iso-surfaces this script writes itself.

COLUMNS in sample_index.csv
---------------------------
  sample_index            0..N-1, matches ensemble/sample_XXXXX.vtu
  xi_norm2                ||xi_i||^2
  radius_percentile       F_chi2(||xi||^2; N_KL) in [0,1]; 1 = extreme
  occurrence_likelihood   1 - radius_percentile; 1 = typical, 0 = rare
  likelihood_rank         0 = most typical geometry
  opacity_weight          the raw exp(-0.5||xi||^2) from monte_carlo.py
  log10_opacity_weight    log of the above (usable as a color-by array)
  compliance              C(xi_i) from stage6_validation/compliance_samples.csv
  compliance_percentile   empirical rank / N; 1 = worst (softest) sample
  compliance_z            (C - mean) / std
  is_tail_95              1 if this sample is in the worst 5% of compliance

Run (plain python, needs numpy/scipy/pyvista -- no dolfinx, no MPI):
    python viz/build_cloud_index.py
    python viz/build_cloud_index.py --no-merge          # skip the big .vtp
    python viz/build_cloud_index.py --decimate 0.7      # lighter merged cloud
    python viz/build_cloud_index.py --deviation-scale 10   # 10x exaggeration
    python viz/build_cloud_index.py --deviation-scale 25 --reference typical
    python viz/build_cloud_index.py --deviation-scale 0.5  # damp them instead
    python viz/build_cloud_index.py -k 10 --opacity-gamma 2  # fade the tails harder
    python viz/build_cloud_index.py --opacity-range 0.01 0.5 # stronger contrast
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
from scipy.stats import chi2

# The per-sample scalars that get baked into every iso-surface as constant
# point data, so ParaView can Threshold / color by any of them.
_CLOUD_ARRAYS = (
    "sample_index",
    "occurrence_likelihood",
    "radius_percentile",
    "compliance",
    "compliance_percentile",
    "compliance_z",
    "log10_opacity_weight",
    "is_tail_95",
)

# Default alpha for the rarest / most-deviated realization and for the most
# typical one. 100 stacked layers at 0.3 read as a near-solid core that frays
# outward, which is exactly the picture: the typical part is where the mass is.
_OPACITY_LO, _OPACITY_HI = 0.02, 0.30

# The nodal array every ensemble .vtu carries (written by
# monte_carlo.run_monte_carlo_validation as point_data["density"]).
_DENSITY = "density"


def _write_pvd(pvd_path: Path, entries: list[tuple[float, str]]) -> None:
    """Minimal ParaView collection file. `entries` is [(timestep, relpath)].

    Same format monte_carlo._write_pvd_collection emits; duplicated here only
    so this script stays importable without dolfinx on the path.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, rel in entries:
        lines.append(f'    <DataSet timestep="{t}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    pvd_path.write_text("\n".join(lines) + "\n")


def build_index(mc_dir: Path, stage6_dir: Path, kl_dir: Path,
                config_path: Path) -> np.ndarray:
    """Recompute each sample's xi and assemble the per-sample scalar table.

    xi is NOT stored on disk by the pipeline, but it is fully deterministic:
    monte_carlo.run_monte_carlo_validation draws sample i as
    `np.random.default_rng(seed + i).standard_normal(n_kl)`. Reproducing it
    here is exact, not an approximation.
    """
    n_kl = json.loads((kl_dir / "kl_truncation.json").read_text())["n_kl"]
    cfg = yaml.safe_load(config_path.read_text())
    seed = int(cfg["mc_validation"]["seed"])

    weights = np.loadtxt(mc_dir / "probability_weights.csv",
                         delimiter=",", skiprows=1)
    compliance = np.loadtxt(stage6_dir / "compliance_samples.csv",
                            delimiter=",", skiprows=1)
    if weights.shape[0] != compliance.shape[0]:
        raise ValueError(
            f"probability_weights.csv has {weights.shape[0]} rows but "
            f"compliance_samples.csv has {compliance.shape[0]}. These must come "
            "from the same run -- rerun Stage 6 or delete the stale one."
        )
    n = weights.shape[0]

    xi_norm2 = np.empty(n)
    for i in range(n):
        xi = np.random.default_rng(seed + i).standard_normal(n_kl)
        xi_norm2[i] = float(xi @ xi)

    # Consistency check: the reproduced xi must regenerate monte_carlo.py's own
    # weights (up to its max-normalization). If this fails, `seed` or `n_kl`
    # disagree with the run that produced the ensemble on disk.
    w_repro = np.exp(-0.5 * xi_norm2)
    w_repro = w_repro / w_repro.max() if w_repro.max() > 0 else w_repro
    if not np.allclose(w_repro, weights[:, 1], rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"Reproduced xi does not match probability_weights.csv "
            f"(seed={seed}, n_kl={n_kl}). The ensemble on disk was written by a "
            "run with different mc_validation.seed or a different KL truncation "
            "than the current config/stage3 artifacts."
        )

    radius_percentile = chi2.cdf(xi_norm2, df=n_kl)
    occurrence_likelihood = 1.0 - radius_percentile
    likelihood_rank = np.argsort(np.argsort(-occurrence_likelihood))

    C = compliance[:, 1]
    compliance_percentile = (np.argsort(np.argsort(C)) + 1) / n
    compliance_z = (C - C.mean()) / C.std(ddof=1)

    table = np.column_stack([
        np.arange(n, dtype=float),
        xi_norm2,
        radius_percentile,
        occurrence_likelihood,
        likelihood_rank.astype(float),
        weights[:, 1],
        np.log10(np.maximum(weights[:, 1], 1e-300)),
        C,
        compliance_percentile,
        compliance_z,
        (compliance_percentile > 0.95).astype(float),
    ])
    return table


_HEADER = ("sample_index,xi_norm2,radius_percentile,occurrence_likelihood,"
           "likelihood_rank,opacity_weight,log10_opacity_weight,compliance,"
           "compliance_percentile,compliance_z,is_tail_95")


def _sample_vtu(ensemble_dir: Path, i: int) -> Path:
    vtu = ensemble_dir / f"sample_{i:05d}.vtu"
    if not vtu.exists():
        raise FileNotFoundError(
            f"{vtu} is missing but sample {i} is listed in the index. The "
            "ensemble directory and the CSVs are out of sync."
        )
    return vtu


def resolve_reference(table: np.ndarray, ensemble_dir: Path, mc_dir: Path,
                      spec: str, n_points: int) -> tuple[np.ndarray, str]:
    """Nodal density field that --deviation-scale measures deviations from.

    `spec` is one of:
      "mean"     the pointwise ensemble mean E[rho_phys(x)] -- the closest
                 thing on disk to the nominal (xi = 0) geometry, and the
                 reference that makes the amplified cloud symmetric about the
                 middle of the manufacturing envelope. Taken from
                 reliability_map.vtu when it matches, otherwise streamed off
                 the ensemble (one extra read pass, constant memory).
      "typical"  the single most-likely realization (likelihood_rank 0), i.e.
                 an actual manufacturable part rather than an average of many.
      "<int>"    that sample index, for comparing against one chosen part.

    Returns (reference_density, human-readable description).
    """
    if spec == "typical" or spec.isdigit() or (spec[:1] == "-" and spec[1:].isdigit()):
        if spec == "typical":
            idx = int(table[int(np.argmin(table[:, 4])), 0])
            what = f"most-typical sample {idx} (likelihood_rank 0)"
        else:
            idx = int(spec)
            if idx not in table[:, 0].astype(int):
                raise ValueError(
                    f"--reference {idx} is not a sample index in the ensemble "
                    f"(have 0..{int(table[:, 0].max())})."
                )
            what = f"sample {idx}"
        ref = np.asarray(pv.read(str(_sample_vtu(ensemble_dir, idx)))
                         .point_data[_DENSITY], dtype=float)
    elif spec == "mean":
        rel_path = mc_dir / "reliability_map.vtu"
        ref = None
        if rel_path.exists():
            rel = pv.read(str(rel_path))
            if "mean_density" in rel.point_data and rel.n_points == n_points:
                ref = np.asarray(rel.point_data["mean_density"], dtype=float)
                what = f"ensemble mean density from {rel_path}"
        if ref is None:
            # reliability_map.vtu absent or from a different mesh -- recompute.
            # Streamed rather than stacked: an n_samples x n_nodes array is the
            # one thing here that does not fit at full MC scale.
            acc = np.zeros(n_points)
            for row in table:
                acc += np.asarray(
                    pv.read(str(_sample_vtu(ensemble_dir, int(row[0]))))
                    .point_data[_DENSITY], dtype=float)
            ref = acc / table.shape[0]
            what = f"ensemble mean density recomputed over {table.shape[0]} samples"
    else:
        raise ValueError(
            f"--reference must be 'mean', 'typical', or a sample index; got {spec!r}"
        )

    if ref.shape[0] != n_points:
        raise ValueError(
            f"reference field has {ref.shape[0]} nodes but the ensemble VTUs "
            f"have {n_points}. The reference and the ensemble are not on the "
            "same mesh."
        )
    return ref, what


def build_surfaces(table: np.ndarray, ensemble_dir: Path, out_dir: Path,
                   iso: float, merge: bool, decimate: float,
                   deviation_scale: float = 1.0,
                   reference: str = "mean",
                   opacity_range: tuple[float, float] = (_OPACITY_LO, _OPACITY_HI),
                   opacity_gamma: float = 1.0) -> None:
    """Contour each sample at rho = `iso` and tag it with its scalars.

    The rho = 0.5 level set IS the manufactured boundary of that realization,
    so a stack of 100 of them is a direct picture of "every part this process
    could produce". It is also ~50x lighter than the merged full-volume
    probability_cloud.vtp that src/viz/probability_cloud.py writes (256 MB
    here), which is what makes interactive filtering feasible.

    `deviation_scale` != 1 exaggerates (or damps) each realization's departure
    from `reference` before contouring -- see the module docstring. The scalar
    tables and the likelihood ordering are untouched by it; only the geometry
    of the emitted surfaces changes.

    `opacity_range` / `opacity_gamma` set the per-surface `opacity` array that
    fades the rare, far-from-nominal layers out; also module docstring.
    """
    op_lo, op_hi = float(opacity_range[0]), float(opacity_range[1])
    surf_dir = out_dir / "surfaces"
    surf_dir.mkdir(parents=True, exist_ok=True)

    amplify = not np.isclose(deviation_scale, 1.0)
    ref_density = None
    if amplify:
        probe = pv.read(str(_sample_vtu(ensemble_dir, int(table[0, 0]))))
        ref_density, ref_what = resolve_reference(
            table, ensemble_dir, ensemble_dir.parent, reference, probe.n_points)
        print(f"  deviation_scale = {deviation_scale:g}x about {ref_what}")

        # The unamplified reference boundary, so a figure can show the true
        # nominal shape next to the exaggerated cloud.
        probe.point_data[_DENSITY] = ref_density
        ref_surf = probe.contour(isosurfaces=[iso], scalars=_DENSITY)
        if ref_surf.n_points > 0:
            if decimate > 0.0:
                ref_surf = ref_surf.decimate_pro(decimate, preserve_topology=True)
            ref_surf.point_data["deviation_scale"] = np.full(ref_surf.n_points, 1.0)
            # Opaque: it is the reference silhouette, not one of the layers.
            ref_surf.point_data["opacity"] = np.full(ref_surf.n_points, 1.0)
            ref_surf.save(str(out_dir / "reference_surface.vtp"))
            print(f"  reference surface: {out_dir / 'reference_surface.vtp'}")
        del probe

    print(f"  opacity ramp: rare={op_lo:g} -> typical={op_hi:g} "
          f"(gamma={opacity_gamma:g}), baked in as point array 'opacity'")

    max_dev = 0.0
    pieces: list[pv.PolyData] = []
    written: list[tuple[int, Path]] = []
    for row in table:
        i = int(row[0])
        vtu = _sample_vtu(ensemble_dir, i)
        grid = pv.read(str(vtu))
        if amplify:
            dev = np.asarray(grid.point_data[_DENSITY], dtype=float) - ref_density
            max_dev = max(max_dev, float(np.abs(dev).max()))
            # Linear in the deviation, so the rho = iso level set moves by
            # ~deviation_scale x its true normal offset. Deliberately NOT
            # clipped to [0, 1]: clipping would flatten the amplified field
            # right where the boundary needs to travel.
            grid.point_data[_DENSITY] = ref_density + deviation_scale * dev
        surf = grid.contour(isosurfaces=[iso], scalars=_DENSITY)
        if surf.n_points == 0:
            # A realization can, in principle, project to all-solid or all-void.
            # Skip it rather than emit a degenerate file, but say so loudly.
            print(f"  sample {i:5d}: empty iso-surface at rho={iso}, skipped")
            continue
        if decimate > 0.0:
            surf = surf.decimate_pro(decimate, preserve_topology=True)

        n_pts = surf.n_points
        for name, col in zip(_CLOUD_ARRAYS,
                             (row[0], row[3], row[2], row[7], row[8],
                              row[9], row[6], row[10])):
            surf.point_data[name] = np.full(n_pts, float(col))
        # Travels with the geometry so an exaggerated figure can never be
        # mistaken for a true-scale one.
        surf.point_data["deviation_scale"] = np.full(n_pts, float(deviation_scale))
        # Ready-to-render alpha: typical geometries solid, rare ones nearly
        # invisible. row[3] is occurrence_likelihood, uniform on [0, 1].
        alpha = op_lo + (op_hi - op_lo) * float(row[3]) ** opacity_gamma
        surf.point_data["opacity"] = np.full(n_pts, alpha)

        path = surf_dir / f"sample_{i:05d}.vtp"
        surf.save(str(path))
        written.append((i, path))
        pieces.append(surf)

    # Time series ordered most-likely -> least-likely: scrubbing the animation
    # walks outward from the nominal geometry into the rare tails.
    order = np.argsort(table[:, 4])  # likelihood_rank
    by_likelihood = []
    lookup = dict(written)
    for t, idx in enumerate(order):
        i = int(table[idx, 0])
        if i in lookup:
            by_likelihood.append((float(t), f"surfaces/{lookup[i].name}"))
    _write_pvd(out_dir / "surfaces_by_likelihood.pvd", by_likelihood)

    ens_rel = os.path.relpath(ensemble_dir, start=out_dir)
    ens_by_likelihood = [
        (float(t), f"{ens_rel}/sample_{int(table[idx, 0]):05d}.vtu")
        for t, idx in enumerate(order)
    ]
    _write_pvd(out_dir / "ensemble_by_likelihood.pvd", ens_by_likelihood)

    if merge and pieces:
        merged = pieces[0].merge(pieces[1:]) if len(pieces) > 1 else pieces[0].copy()
        if not isinstance(merged, pv.PolyData):
            # pyvista's merge can hand back an UnstructuredGrid; .vtp needs
            # PolyData. extract_surface() is a no-op on an actual surface.
            merged = merged.extract_surface()
        merged_path = out_dir / "probability_cloud_surfaces.vtp"
        merged.save(str(merged_path))
        print(f"  merged cloud: {merged_path} "
              f"({merged.n_points} pts, {merged.n_cells} cells)")

    if amplify:
        print(f"  max |rho_i - rho_ref| over the ensemble: {max_dev:.4g} "
              f"-> amplified to {deviation_scale * max_dev:.4g}")
        print(f"  NOTE: these surfaces are exaggerated {deviation_scale:g}x. "
              "Say so in any caption; rerun without --deviation-scale for "
              "true-scale geometry.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-dir", type=Path, default=Path("output/mc_validation"))
    ap.add_argument("--stage6-dir", type=Path, default=Path("output/stage6_validation"))
    ap.add_argument("--kl-dir", type=Path, default=Path("output/stage3_random_field"))
    ap.add_argument("--config", type=Path, default=Path("src/config/config.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/viz"))
    ap.add_argument("--iso", type=float, default=0.5,
                    help="density level set treated as the part boundary")
    ap.add_argument("--decimate", type=float, default=0.0,
                    help="fraction of triangles to remove per surface, 0..0.95")
    ap.add_argument("--deviation-scale", "-k", type=float, default=1.0,
                    help="exaggerate (>1) or damp (<1) each realization's "
                         "deviation from the reference field before contouring. "
                         "1.0 = true geometry (default); 10 is a good starting "
                         "point for a figure. The boundary moves ~this many "
                         "times its true offset -- label any figure that uses it.")
    ap.add_argument("--reference", default="mean",
                    help="field the deviations are measured from: 'mean' "
                         "(ensemble mean, default), 'typical' (the most-likely "
                         "realization), or a sample index. Only used when "
                         "--deviation-scale != 1.")
    ap.add_argument("--opacity-range", type=float, nargs=2,
                    metavar=("RARE", "TYPICAL"),
                    default=[_OPACITY_LO, _OPACITY_HI],
                    help="alpha baked into the per-surface 'opacity' array for "
                         "the rarest and the most typical realization "
                         f"(default {_OPACITY_LO} {_OPACITY_HI}). RARE < "
                         "TYPICAL makes far-out deviations fade; swap them to "
                         "highlight the tails instead.")
    ap.add_argument("--opacity-gamma", type=float, default=1.0,
                    help="exponent on occurrence_likelihood before the alpha "
                         "ramp. >1 fades the unlikely layers out harder, <1 "
                         "keeps them visible (default 1.0 = linear).")
    ap.add_argument("--no-merge", dest="merge", action="store_false",
                    help="skip the merged probability_cloud_surfaces.vtp")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = build_index(args.mc_dir, args.stage6_dir, args.kl_dir, args.config)
    csv_path = args.out_dir / "sample_index.csv"
    np.savetxt(csv_path, table, delimiter=",", header=_HEADER, comments="",
               fmt=["%d", "%.10e", "%.10e", "%.10e", "%d", "%.10e", "%.10e",
                    "%.10e", "%.10e", "%.10e", "%d"])
    print(f"wrote {csv_path} ({table.shape[0]} samples)")

    if args.deviation_scale <= 0.0:
        raise SystemExit("--deviation-scale must be > 0 "
                         "(1.0 = true geometry, >1 exaggerates, <1 damps)")
    if not all(0.0 <= v <= 1.0 for v in args.opacity_range):
        raise SystemExit("--opacity-range values are alphas and must be in [0, 1]")
    if args.opacity_gamma <= 0.0:
        raise SystemExit("--opacity-gamma must be > 0")

    build_surfaces(table, args.mc_dir / "ensemble", args.out_dir,
                   args.iso, args.merge, args.decimate,
                   deviation_scale=args.deviation_scale,
                   reference=str(args.reference),
                   opacity_range=tuple(args.opacity_range),
                   opacity_gamma=args.opacity_gamma)
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
