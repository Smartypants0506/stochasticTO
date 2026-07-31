"""viz/make_paraview_state.py -- build the whole ParaView session, then save it
as a .pvsm you can reopen with File -> Load State.

HOW TO RUN IT (once)
--------------------
A .pvsm is a proxy-ID-keyed XML that only ParaView itself can emit correctly,
and the dolfinx image does not ship ParaView. So this runs inside ParaView,
not in the container:

  ParaView Desktop:  View -> Python Shell, then
      exec(open('/raid/ovb/stochasticTO/viz/make_paraview_state.py').read())

  or headless, if you have a ParaView install:
      pvpython viz/make_paraview_state.py

Either way it writes  output/paraview/stochasticTO.pvsm  and from then on you
just do File -> Load State on that file. The scene is rebuilt exactly, so you
never have to run this again unless the data changes.

WHAT YOU GET -- three layout tabs
---------------------------------
  [1 Probability Cloud]  one view. The layered boundary cloud with a live
      Threshold on occurrence_likelihood already wired up: select
      "likelihood_filter" in the pipeline browser and drag the lower bound to
      watch the cloud collapse toward the nominal shape.

  [2 Ensemble FEA]  four camera-linked views over the same realization --
      von Mises (solid phase), the eta field that caused it, strain energy
      density, and the deformed shape. Hit Play and all four advance through
      the 100 realizations together. Colour ranges are LOCKED across
      timesteps, which is the thing hand-setup usually gets wrong.

  [3 Robust vs Nominal]  four views: uncertain material for the nominal
      design, the same for the robust design (camera-linked, so they stay
      comparable), the P(void) reduction map, and the three mean shapes
      overlaid. Headline statistics are drawn on as text annotations.

Everything reads from output/paraview/, so run viz/export_paraview.py first.
Ranges come from field_ranges.csv and stats from output/comparison/summary.json,
so re-running after new data picks up the new numbers automatically.

NOTE ON PATHS: a .pvsm stores absolute file paths. If you move the repo or
open the state on a machine with a different mount point, ParaView will prompt
you to relocate the files -- point it at output/paraview/ and it resolves the
rest itself.
"""
from paraview.simple import *  # noqa: F401,F403

import csv
import json
import os

# --------------------------------------------------------------------------
# Locate the data. __file__ is undefined when pasted into the Python Shell via
# exec(), so fall back to the known repo path.
# --------------------------------------------------------------------------
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = "/raid/ovb/stochasticTO"
PV = os.path.join(ROOT, "output", "paraview")
STATE = os.path.join(PV, "stochasticTO.pvsm")

if not os.path.isdir(PV):
    raise SystemExit(f"{PV} not found -- run `python viz/export_paraview.py` first.")


def f(name):
    return os.path.join(PV, name)


def have(name):
    return os.path.exists(f(name))


def point_arrays(source):
    """Point-array names on a source; [] if the pipeline cannot report them.

    Lets the state adapt to data written by an older build_cloud_index.py
    (no `opacity` array) instead of erroring out mid-build.
    """
    try:
        source.UpdatePipeline()
        return list(source.PointData.keys())
    except Exception:
        return []


# --- ranges + stats, read rather than hardcoded ---------------------------
RANGES = {}
if have("field_ranges.csv"):
    with open(f("field_ranges.csv")) as fh:
        for row in csv.DictReader(fh):
            RANGES[row["field"]] = {k: float(row[k]) for k in ("min", "max", "p01", "p99")}

SUMMARY = {}
_summary_path = os.path.join(ROOT, "output", "comparison", "summary.json")
if os.path.exists(_summary_path):
    SUMMARY = json.load(open(_summary_path))

BASELINE = SUMMARY.get("baseline", "nominal")
DESIGNS = sorted(SUMMARY.get("stats", {}).keys()) or ["nominal", "robust_lambda1"]
ROBUST = next((d for d in DESIGNS if d != BASELINE), None)


def rng(field, robust=True):
    """Colour range for a field: p01..p99 by default.

    min..max is the wrong default for stress -- a handful of nodes at a
    re-entrant corner own the entire scale and everything else renders flat.
    """
    r = RANGES.get(field)
    if not r:
        return None
    return (r["p01"], r["p99"]) if robust else (r["min"], r["max"])


# --------------------------------------------------------------------------
# Small wrappers over the parts of the API that moved between versions.
# --------------------------------------------------------------------------
def set_threshold(t, lo, hi):
    """ParaView >= 5.10 split ThresholdRange into Lower/UpperThreshold."""
    try:
        t.LowerThreshold = lo
        t.UpperThreshold = hi
        t.ThresholdMethod = "Between"
    except AttributeError:
        t.ThresholdRange = [lo, hi]


def paint(display, view, array, preset, value_range=None, log=False,
          assoc="POINTS", title=None):
    ColorBy(display, (assoc, array))
    lut = GetColorTransferFunction(array)
    lut.ApplyPreset(preset, True)
    if value_range:
        lo, hi = value_range
        if log:
            lo = max(lo, hi * 1e-6)          # log scale needs a positive floor
        # Rescale BEFORE enabling log: ParaView rejects a log scale while the
        # current range still touches zero.
        lut.RescaleTransferFunction(lo, hi)
        pwf = GetOpacityTransferFunction(array)
        pwf.RescaleTransferFunction(lo, hi)
        if log:
            lut.UseLogScale = 1
        # Stop ParaView rescaling per timestep -- otherwise every frame of the
        # ensemble renders equally hot and the variation vanishes.
        lut.AutomaticRescaleRangeMode = "Never"
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = title or array
    bar.ComponentTitle = ""
    bar.TitleFontSize = 14
    bar.LabelFontSize = 12
    return lut


def paint_diverging_grey(display, view, array, cap, title=None,
                         cool=(0.231, 0.298, 0.753), warm=(0.706, 0.016, 0.150),
                         grey=(0.25, 0.25, 0.25)):
    """Diverging colour map for a signed, symmetric-about-zero array (e.g.
    compliance_z): dark grey at 0 (typical realizations), fading out to
    `cool`/`warm` at -cap/+cap (default a blue/red pair) -- matches the
    build_cloud_index.py --opacity-z-cap tails visually, since the same
    samples that are nearly-transparent are also the ones furthest from grey.
    """
    ColorBy(display, ("POINTS", array))
    lut = GetColorTransferFunction(array)
    lut.RGBPoints = [-cap, *cool, 0.0, *grey, cap, *warm]
    lut.RescaleTransferFunction(-cap, cap)
    pwf = GetOpacityTransferFunction(array)
    pwf.RescaleTransferFunction(-cap, cap)
    lut.AutomaticRescaleRangeMode = "Never"
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = title or array
    bar.ComponentTitle = ""
    bar.TitleFontSize = 14
    bar.LabelFontSize = 12
    return lut


def fade_by_array(display, opacity_array, color_array):
    """Alpha per point from `opacity_array` instead of one global opacity.

    build_cloud_index.py bakes a ready-to-use alpha into `opacity` (high for
    typical geometries, low for rare/far-out ones), so the transfer function
    here is deliberately the identity on [0, 1] -- the ramp shape is decided
    when the surfaces are written, not at render time.

    Needs BOTH switches: the representation's "Use Separate Opacity Array"
    (which array supplies alpha) and the colour map's "Enable opacity mapping
    for surfaces" (whether surfaces honour alpha at all). Both moved into the
    Python API around 5.9, so this degrades to a flat opacity if either is
    missing rather than failing the whole state build.
    """
    try:
        display.UseSeparateOpacityArray = 1
        display.OpacityArrayName = ["POINTS", opacity_array]
        lut = GetColorTransferFunction(color_array)
        lut.EnableOpacityMapping = 1
        # Overwrite whatever paint() rescaled this to: when a separate opacity
        # array drives it, the pwf domain is that array's range, not the
        # colour array's.
        pwf = GetOpacityTransferFunction(color_array)
        pwf.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
        display.Opacity = 1.0     # the array is the alpha; don't double-dim it
        return True
    except (AttributeError, NameError, RuntimeError) as exc:
        print(f"  note: separate opacity array unavailable ({exc}); "
              "falling back to a flat opacity. In the GUI: Properties -> "
              f"Opacity By Array -> {opacity_array}, and tick 'Enable opacity "
              "mapping for surfaces' on the colour map.")
        return False


def solid(display, rgb, opacity=1.0):
    ColorBy(display, None)
    display.AmbientColor = list(rgb)
    display.DiffuseColor = list(rgb)
    display.Opacity = opacity


def annotate(view, text, location="Upper Left Corner", size=13,
             color=(0.1, 0.1, 0.12)):
    t = Text(registrationName=f"note_{abs(hash(text)) % 100000}")
    t.Text = text
    d = Show(t, view)
    d.WindowLocation = location
    d.FontSize = size
    d.Color = list(color)
    d.Bold = 0
    return t


def new_view():
    v = CreateRenderView()
    v.Background = [1.0, 1.0, 1.0]
    v.UseColorPaletteForBackground = 0
    v.OrientationAxesVisibility = 1
    return v


def quad(layout, views):
    """Drop up to four views into a 2x2 grid in `layout`."""
    layout.SplitHorizontal(0, 0.5)
    layout.SplitVertical(1, 0.5)
    layout.SplitVertical(2, 0.5)
    for view, hint in zip(views, (3, 4, 5, 6)):
        AssignViewToLayout(view=view, layout=layout, hint=hint)


# ==========================================================================
# TAB 1 -- the probability cloud
# ==========================================================================
def tab_probability_cloud():
    if not have("01_cloud_all.vtp"):
        print("skip tab 1: 01_cloud_all.vtp missing")
        return None

    layout = CreateLayout(name="1 Probability Cloud")
    view = new_view()
    AssignViewToLayout(view=view, layout=layout, hint=0)

    cloud = XMLPolyDataReader(registrationName="cloud_all",
                              FileName=[f("01_cloud_all.vtp")])
    cloud.UpdatePipeline()

    # The filter the whole deliverable hangs on. Select it in the pipeline
    # browser and drag LowerThreshold: 0.0 shows every part the process can
    # make, 0.75 shows only what it routinely makes. Spread that survives at
    # 0.75 is ROUTINE variation, not tail risk.
    likelihood = Threshold(registrationName="likelihood_filter", Input=cloud)
    likelihood.Scalars = ["POINTS", "occurrence_likelihood"]
    set_threshold(likelihood, 0.0, 1.0)

    d = Show(likelihood, view)
    d.SetRepresentationType("Surface")
    d.Opacity = 0.08
    paint_diverging_grey(d, view, "compliance_z", cap=3.0,
                         title="compliance z-score")
    # Per-layer alpha: the further a realization is from nominal, the rarer it
    # is and the more transparent it renders, so the cloud reads as a dense
    # typical core inside a faint envelope of what the process *could* do.
    if "opacity" in point_arrays(cloud):
        fade_by_array(d, "opacity", "compliance_z")

    if have("01_mean_shape.vtp"):
        mean_shape = XMLPolyDataReader(registrationName="mean_shape",
                                       FileName=[f("01_mean_shape.vtp")])
        solid(Show(mean_shape, view), (0.24, 0.24, 0.28), 1.0)
    elif have("01_reference_surface.vtp"):
        ref = XMLPolyDataReader(registrationName="reference_surface",
                               FileName=[f("01_reference_surface.vtp")])
        solid(Show(ref, view), (0.24, 0.24, 0.28), 1.0)

    # One click away, loaded but hidden.
    for name, reg in (("01_cloud_worst5pct.vtp", "cloud_worst5pct"),
                      ("01_cloud_likely_top10.vtp", "cloud_likely_top10"),
                      ("01_cloud_extreme5pct.vtp", "cloud_extreme5pct")):
        if have(name):
            r = XMLPolyDataReader(registrationName=reg, FileName=[f(name)])
            Hide(r, view)

    annotate(view,
             "PROBABILITY CLOUD -- every geometry this process could produce\n"
             "Select 'likelihood_filter' and drag LowerThreshold 0 -> 0.75.\n"
             "Colour = compliance z-score of that realization\n"
             "(dark grey at z=0, blue/red toward the +/-3 sigma tails).\n"
             "Transparency = rarity: the further from nominal, the fainter.\n"
             "Solid dark grey = ensemble-mean boundary (reference).")
    ResetCamera(view)
    return view


# ==========================================================================
# TAB 2 -- FEA fields across the ensemble
# ==========================================================================
def tab_ensemble():
    if not have("02_ensemble_fea.pvd"):
        print("skip tab 2: 02_ensemble_fea.pvd missing "
              "(run viz/enrich_ensemble_fea.py)")
        return None

    layout = CreateLayout(name="2 Ensemble FEA")
    views = [new_view() for _ in range(4)]
    quad(layout, views)

    reader = PVDReader(registrationName="ensemble_fea",
                       FileName=f("02_ensemble_fea.pvd"))
    reader.UpdatePipeline()
    GetAnimationScene().UpdateAnimationUsingDataTimeSteps()

    # Stress is meaningless in void -- it carries the SIMP scaling. Mask first.
    solid_only = Threshold(registrationName="solid_material", Input=reader)
    solid_only.Scalars = ["POINTS", "density"]
    set_threshold(solid_only, 0.5, 1.0)
    solid_only.UpdatePipeline()

    panels = [
        ("von_mises_solid", "Inferno (matplotlib)", False,
         "VON MISES (solid phase)\nSIMP scaling divided out -- this is the\n"
         "yielding-relevant stress. Watch the hot spot as you\n"
         "play: if it MOVES between realizations, the design\n"
         "leans on ligaments that variation is thinning."),
        ("eta", "Cool to Warm", False,
         "ETA -- the sampled manufacturing threshold.\n"
         "This is the CAUSE; the other panels are the effect.\n"
         "eta > 0.5 = under-deposition at that point."),
        ("strain_energy_density", "Viridis (matplotlib)", True,
         "STRAIN ENERGY DENSITY (log)\nWhere compliance is being spent.\n"
         "Integrates to this sample's C."),
    ]
    for view, (field, preset, log, note) in zip(views, panels):
        d = Show(solid_only, view)
        paint(d, view, field, preset, rng(field), log=log)
        annotate(view, note, size=11)
        ResetCamera(view)

    # --- fourth panel: deformed shape, one fixed exaggeration for all frames
    view = views[3]
    warp = WarpByVector(registrationName="deformed", Input=solid_only)
    warp.Vectors = ["POINTS", "displacement"]
    umax = RANGES.get("displacement_magnitude", {}).get("max", 0.0)
    bounds = solid_only.GetDataInformation().GetBounds()
    diag = ((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2) ** 0.5
    # Largest deformation in the ensemble renders as ~2% of the bounding-box
    # diagonal. Fixed, so frames stay comparable.
    warp.ScaleFactor = (0.02 * diag / umax) if umax > 0 else 1.0

    d = Show(warp, view)
    paint(d, view, "displacement_magnitude", "Viridis (matplotlib)",
          rng("displacement_magnitude"))
    annotate(view,
             f"DEFORMED SHAPE, exaggerated x{warp.ScaleFactor:.3g}\n"
             "Fixed scale for every realization -- an auto-scaled\n"
             "warp would make all frames look identical.", size=11)
    ResetCamera(view)

    for other in views[1:]:
        AddCameraLink(views[0], other, f"ens_cam_{id(other)}")
    return views[0]


# ==========================================================================
# TAB 3 -- robust vs deterministic
# ==========================================================================
def _headline():
    s = SUMMARY.get("stats", {}).get(ROBUST, {}).get(f"vs_{BASELINE}")
    if not s:
        return ""
    return (f"{ROBUST} vs {BASELINE}, {SUMMARY.get('n_samples', '?')} shared "
            f"realizations:\n"
            f"  wins {100 * s['win_rate']:.0f}% of realizations (paired)\n"
            f"  std      -{s['std_reduction_pct']:.1f}%\n"
            f"  95th pct -{s['p95_reduction_pct']:.1f}%\n"
            f"  worst    -{s['worst_case_reduction_pct']:.1f}%\n"
            f"  mean     -{s['mean_reduction_pct']:.1f}%")


def tab_comparison():
    if not have("03_risk_delta.vtu"):
        print("skip tab 3: 03_risk_delta.vtu missing "
              "(run viz/compare_designs_mc.py)")
        return None

    layout = CreateLayout(name="3 Robust vs Nominal")
    views = [new_view() for _ in range(4)]
    quad(layout, views)

    # --- panels 1-2: uncertain material, same scale, camera-linked ---------
    pair = []
    for view, design in zip(views[:2], (BASELINE, ROBUST)):
        name = f"03_uncertain_{design}.vtu"
        if not design or not have(name):
            continue
        r = XMLUnstructuredGridReader(registrationName=f"uncertain_{design}",
                                      FileName=[f(name)])
        d = Show(r, view)
        paint(d, view, f"prob_void_{design}", "Inferno (matplotlib)", (0.0, 1.0),
              title="P(void)")
        annotate(view,
                 f"UNCERTAIN MATERIAL -- {design}\n"
                 "Only nodes with 0.02 < P(void) < 0.98.\n"
                 "Every voxel shown is material whose existence\n"
                 "is a coin flip at manufacturing time.", size=11)
        ResetCamera(view)
        pair.append(view)
    if len(pair) == 2:
        AddCameraLink(pair[0], pair[1], "uncertain_cam")

    # --- panel 3: where the robustness went -------------------------------
    view = views[2]
    if ROBUST:
        delta = XMLUnstructuredGridReader(registrationName="risk_delta",
                                          FileName=[f("03_risk_delta.vtu")])
        array = f"d_prob_void_{ROBUST}"
        # Two thresholds rather than one: the interesting signal is the two
        # tails, and a single Between-threshold cannot exclude the near-zero
        # bulk that would otherwise hide everything behind it.
        for reg, lo, hi in ((f"risk_removed_{ROBUST}", 0.02, 1.0),
                            (f"risk_added_{ROBUST}", -1.0, -0.02)):
            t = Threshold(registrationName=reg, Input=delta)
            t.Scalars = ["POINTS", array]
            set_threshold(t, lo, hi)
            d = Show(t, view)
            paint(d, view, array, "Cool to Warm (Extended)", (-1.0, 1.0),
                  title="P(void) reduction")
        annotate(view,
                 f"WHERE THE ROBUSTNESS WENT\n{BASELINE} minus {ROBUST}.\n"
                 "WARM = robust TO removed the risk of that\n"
                 "feature not existing. COOL = it accepted risk there.\n"
                 "Near-zero bulk hidden so the signal is visible.", size=11)
        ResetCamera(view)

    # --- panel 4: how the shapes actually differ --------------------------
    view = views[3]
    palette = {BASELINE: (0.75, 0.20, 0.16)}
    others = [(0.12, 0.44, 0.71), (0.88, 0.56, 0.05)]
    for design in DESIGNS:
        name = f"03_shape_{design}.vtp"
        if not have(name):
            continue
        r = XMLPolyDataReader(registrationName=f"shape_{design}",
                              FileName=[f(name)])
        rgb = palette.get(design) or others[(DESIGNS.index(design) - 1) % 2]
        solid(Show(r, view), rgb, 1.0 if design == BASELINE else 0.45)
    annotate(view,
             "MEAN SHAPES OVERLAID\n"
             f"red = {BASELINE} (opaque), blue/orange = robust (45%).\n"
             "Look for thicker members, fewer knife-edge\n"
             "junctions, more redundant load paths.", size=11)
    head = _headline()
    if head:
        annotate(view, head, location="Lower Right Corner", size=12,
                 color=(0.10, 0.30, 0.55))
    ResetCamera(view)
    return views[0]


# ==========================================================================
def main():
    try:
        LoadPalette("WhiteBackground")
    except Exception:
        pass

    first = None
    for build in (tab_probability_cloud, tab_ensemble): # tab_probability_cloud, tab_ensemble, tab_comparison
        try:
            v = build()
            first = first or v
        except Exception as exc:
            print(f"{build.__name__} failed: {type(exc).__name__}: {exc}")

    if first:
        SetActiveView(first)
    Render()
    SaveState(STATE)
    print(f"\nstate saved -> {STATE}")
    print("reopen it any time with File -> Load State")
    if SUMMARY:
        print("\n" + _headline())


main()
