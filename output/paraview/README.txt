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


THIS RUN
--------
  01_cloud_all.vtp: 100 realizations, 2055088 triangles.
  01_cloud_likely_top50.vtp: 50 realizations, 1028920 triangles.
  01_cloud_likely_top10.vtp: 10 realizations, 198608 triangles.
  01_cloud_worst5pct.vtp: 5 realizations, 75728 triangles.
  01_cloud_extreme5pct.vtp: 5 realizations, 89974 triangles.
  01_mean_shape.vtp: ensemble-mean boundary.
  01_reference_surface.vtp: unamplified reference boundary (the cloud around it is exaggerated -- see its deviation_scale array).
  02_ensemble_fea.pvd: 100 realizations (enriched: eta / displacement / von Mises / SED).
  02_ensemble_density.pvd: 100 realizations (density only).
  field_ranges.csv: 7 fields.
  03_risk_delta.vtu: designs = ['nominal', 'robust_lambda0', 'robust_lambda1']
  03_uncertain_nominal.vtu: 5.6% of the mesh is neither reliably solid nor reliably void.
  03_uncertain_robust_lambda0.vtu: 5.4% of the mesh is neither reliably solid nor reliably void.
  03_uncertain_robust_lambda1.vtu: 5.3% of the mesh is neither reliably solid nor reliably void.
