# Visualizing the Stage-6 outputs in ParaView

Everything here reads artifacts a pipeline run leaves under `output/`. Nothing
in `src/` is modified.

---

## 1. What is actually on disk, and which code wrote it

`src/mainClean.py` Stage 6 (lines 439–519) does three things: it runs the MC
validation, writes the summary artifacts, then builds the probability cloud.

| Artifact | Written by | Contents |
|---|---|---|
| `output/mc_validation/ensemble/sample_XXXXX.vtu` | [`monte_carlo.py:504-513`](../src/validation/monte_carlo.py#L504-L513) | **One array only:** PointData `density` = `rho_phys` for that realization. 51 376 points, 281 250 tets. |
| `output/mc_validation/ensemble.pvd` | [`monte_carlo.py:243-265`](../src/validation/monte_carlo.py#L243-L265) | Collection indexing the above; sample index is the "timestep". |
| `output/mc_validation/reliability_map.vtu` | [`monte_carlo.py:576-582`](../src/validation/monte_carlo.py#L576-L582) | PointData `mean_density`, `std_density`, `prob_void` across the ensemble. |
| `output/mc_validation/probability_weights.csv` | [`monte_carlo.py:570-574`](../src/validation/monte_carlo.py#L570-L574) | `exp(-0.5·‖ξᵢ‖²)` normalized to max = 1. |
| `output/stage6_validation/compliance_samples.csv` | [`monte_carlo.py:227-239`](../src/validation/monte_carlo.py#L227-L239) | Per-realization compliance C(ξᵢ). |
| `output/stage6_validation/cdf.png` | [`monte_carlo.py:677-711`](../src/validation/monte_carlo.py#L677-L711) | Empirical CDF of the above. |
| `output/stage6_validation/mc_summary.json` | `mainClean.py:478-480` | mean = 0.158 19, std = 0.007 824. |
| `output/stage6_validation/insample_vs_mc.json` | `mainClean.py:490-503` | SAA in-loop µ/σ vs independent MC. Currently 0.94 % and 4.85 % relative error — the design generalizes off its own sample set. |
| `output/stage6_validation/probability_cloud/probability_cloud.vtp` | [`probability_cloud.py:130-148`](../src/viz/probability_cloud.py#L130-L148) | 100 merged full-volume surfaces, 256 MB, arrays `density`, `opacity`, `sample_probability`, `compliance`. |

The design under test is `pareto_results[-1]`, i.e.
`output/stage5_optimization/rho_robust_lambda_1.0.npy` (λ = 1, mean + std).

### Two things you need to know before plotting anything

**(a) There are no FEA fields in the ensemble.** Only `density`. The docstring
of `src/viz/probability_cloud.py` advertises a `.von_mises` field, but nothing
in the pipeline ever computes one — `_build_sample_grid()`, the function that
would consume it, is dead code (`build_probability_cloud` takes the on-disk
path instead). So request #2 cannot be satisfied from the current files;
`viz/enrich_ensemble_fea.py` below re-solves the ensemble to produce them.

**(b) `probability_weights.csv` is not usable as a ParaView opacity/threshold
array.** It is the N(0, I) density at ξᵢ, which is correct mathematically but
degenerate in N_KL = 37 dimensions: this run's weights span
5.7 × 10⁻¹¹ … 1.0 with a median of 3.1 × 10⁻⁴. Any linear transfer function on
it renders one sample and hides 99. In high dimensions "likely" is a statement
about *radius*, not density — almost all probability mass sits in a thin shell
at ‖ξ‖² ≈ 37, and the mode ξ = 0 is itself an atypical point. So
`viz/build_cloud_index.py` computes

```
radius_percentile    = F_χ²(‖ξᵢ‖² ; 37)        uniform on [0,1] by construction
occurrence_likelihood = 1 − radius_percentile   1 = typical part, 0 = rare part
```

which is what the ParaView filters below actually threshold on. `Threshold:
occurrence_likelihood > 0.5` reads exactly as "show me only the
more-likely-than-median half of the deformation modes".

---

## 2. Order of operations

```bash
# from the repo root, inside the dolfinx container

# (1) index + iso-surface cloud. numpy/scipy/pyvista only, seconds.
python viz/build_cloud_index.py

# (2) real FEA fields on every realization. ~100 linear solves.
source mpi-env.sh
mpirun -n 8 python viz/enrich_ensemble_fea.py

# (3) the head-to-head vs the deterministic design. ~300 linear solves.
mpirun -n 8 python viz/compare_designs_mc.py

# (4) 2D figures
python viz/plot_comparison.py

# (5) collect everything into one flat, ready-to-open directory
python viz/export_paraview.py
```

Then, **once**, inside ParaView Desktop (View → Python Shell):

```python
exec(open('/raid/ovb/stochasticTO/viz/make_paraview_state.py').read())
```

That builds the full three-tab session and saves
`output/paraview/stochasticTO.pvsm`. From then on it is just File → Load
State. See §7 for what the tabs contain.

Steps 1–5 import nothing from `paraview` and need no `pvbatch` — step 5 is
pyvista only, and `output/paraview/README.txt` is generated alongside the data
with the suggested array, range and representation for each file, if you'd
rather set things up by hand.

`export_paraview.py` skips any section whose inputs are missing and says so,
so it is safe to run after step 1 and again after step 3.

Steps 3–5 are the ones that make the case for the project; steps 1–2 make the
pictures.

---

## 3. Request #1 — the layered density cloud, filtered by likelihood

`viz/build_cloud_index.py` writes `output/viz/probability_cloud_surfaces.vtp`:
the ρ = 0.5 level set of each realization (that *is* the manufactured boundary
of that part), merged, with per-sample scalars baked in as constant point
arrays. It is roughly 50× lighter than the 256 MB `probability_cloud.vtp`,
which is what makes the filtering interactive.

`viz/export_paraview.py` also writes pre-filtered subsets so you can just open
one file: `01_cloud_likely_top50.vtp`, `01_cloud_likely_top10.vtp`,
`01_cloud_worst5pct.vtp`, `01_cloud_extreme5pct.vtp`.

To drive it yourself from `01_cloud_all.vtp`:

1. **Threshold** → Scalars `occurrence_likelihood`, range `[floor, 1.0]`.
   Drag `floor` from 0 → 0.75 and watch the cloud collapse toward the nominal
   shape. Whatever spread survives at 0.75 is *routine* variation, not tail risk.
2. Colour by `compliance_z` with **Cool to Warm (Extended)**, range −3…3.
   Colour now encodes the outcome while the layering encodes the geometry.
3. **Use Separate Opacity Array** → `opacity` (plus *Enable opacity mapping for
   surfaces* on the colour map, with an identity ramp on [0, 1]), so rare
   geometries fade out on their own — see below. Flat Opacity ≈ 0.1 is the
   fallback if your ParaView lacks the option.
4. Add `01_mean_shape.vtp`, opaque dark grey, as the reference silhouette.

Other useful thresholds on the same dataset, no reload needed:

| Filter | Question it answers |
|---|---|
| `compliance_percentile > 0.95` | What do the worst 5 % of parts look like? |
| `occurrence_likelihood > 0.5` | What does the process routinely produce? |
| `is_tail_95 == 1` + colour by `occurrence_likelihood` | Are the bad parts rare, or ordinary? |

That last one is the interesting one. If the worst-compliance realizations have
*high* `occurrence_likelihood`, the risk is not a tail curiosity — it is the
median outcome of the process, and the deterministic design was never
describing the part you would actually get. `fig5` in
`viz/plot_comparison.py` is the 2D version of this.

`surfaces_by_likelihood.pvd` is the same surfaces as a time series ordered
most-likely → least-likely, so scrubbing the animation walks outward from the
nominal geometry into the tails.

### Exaggerating the deviations

At true scale the 100 boundaries sit within a fraction of a filter radius of
each other and the cloud renders as one solid shell. `--deviation-scale K`
rebuilds the surfaces with each realization's departure from a reference field
scaled by `K`:

```bash
python viz/build_cloud_index.py --deviation-scale 10             # 10× spread
python viz/build_cloud_index.py -k 25 --reference typical        # about the most-likely part
python viz/build_cloud_index.py -k 1                             # back to true scale
```

It amplifies the field, not the picture — `ρᵢ ← ρ_ref + K(ρᵢ − ρ_ref)` before
contouring — so the ρ = 0.5 boundary moves by ≈ K × its true normal offset and
the *shape* of the variability stays faithful, exactly like the amplitude
scaling on a mode-shape plot. Start at K = 10 and raise it until the layers
separate; past K ≈ 50 on this run the offset stops being proportional to the
true one.

`--reference` picks what the deviation is measured from: `mean` (default, the
pointwise ensemble mean, read from `reliability_map.vtu`), `typical` (the
most-likely realization), or a sample index. The unamplified reference boundary
is written alongside as `output/viz/reference_surface.vtp` — render it opaque
under the exaggerated cloud.

Only the iso-surfaces this script writes are affected; `sample_index.csv`, the
likelihood ordering and `ensemble_by_likelihood.pvd` (which points at the raw
VTUs) are untouched. Every surface carries `deviation_scale` as a point array,
and **any figure made with K ≠ 1 has to say so in its caption**.

### Fading the unlikely layers out

Every surface also carries a ready-to-render alpha in the point array
`opacity`:

```
opacity = lo + (hi − lo) · occurrence_likelihood^γ      (default 0.02, 0.30, γ = 1)
```

so a near-nominal realization renders solid and a far-out, rare one fades to
almost nothing. Since `occurrence_likelihood` is uniform on [0, 1] by
construction, the ensemble spreads evenly across the alpha band instead of
piling up at one end — the cloud reads as a dense typical core inside a wispy
envelope of what the process *could* produce, which is the picture the
deliverable is after.

```bash
python viz/build_cloud_index.py -k 10 --opacity-gamma 2     # fade the tails harder
python viz/build_cloud_index.py --opacity-range 0.01 0.5    # stronger contrast
python viz/build_cloud_index.py --opacity-range 0.3 0.02    # invert: highlight the tails
```

`make_paraview_state.py` wires this up automatically (`UseSeparateOpacityArray`
→ `opacity`, `EnableOpacityMapping` on, identity ramp, representation opacity
back to 1.0); if your ParaView build doesn't expose those it says so and leaves
a flat opacity. To do it by hand: Properties → **Opacity By Array** → `opacity`,
then tick **Enable opacity mapping for surfaces** on the colour map.

Note this is *not* `opacity_weight` / `log10_opacity_weight` — those are the raw
exp(−½‖ξ‖²) values, unusable as alpha for the reason in §1(b), kept only for
provenance.

---

## 4. Request #2 — FEA colormaps on each ensemble file

Run `viz/enrich_ensemble_fea.py` first. It reloads the fixed design, rebuilds
the identical KL basis, and replays realization *i* using
`default_rng(seed + i).standard_normal(n_kl)` — the same draw
`monte_carlo.py` used, so the realizations are bit-identical, not a
re-sampling. It checks itself against `compliance_samples.csv` and records the
result in `provenance.json`.

Fields written to `output/viz/ensemble_fea/ensemble/sample_XXXXX.vtu`, all CG1
nodal:

| Field | Use |
|---|---|
| `eta` | The sampled manufacturing threshold — the *cause*. η > 0.5 = under-deposition. |
| `von_mises` | Macroscopic stress. **Meaningless in void**: threshold `density > 0.5` first. |
| `von_mises_solid` | `von_mises / (ε + (1−ε)ρ^p)` — stress in the actual solid phase. This is the yielding-relevant number and the one that spikes in thin ligaments. |
| `strain_energy_density` | Where compliance is spent; integrates to that sample's C. |
| `displacement` (3-vector) | Feed **Warp By Vector**. |
| `displacement_magnitude` | Scalar deformation. |
| `density` | Kept for masking/thresholding. |

Open `output/paraview/02_ensemble_fea.pvd` — one timestep per realization, use
the VCR controls. Apply **Threshold** on `density` in [0.5, 1] *before*
colouring by any stress field.

One detail that is easy to get wrong by hand: **lock one colour range across
the whole ensemble.** ParaView's default is to rescale to the current
timestep, which makes an ensemble animation actively misleading — every frame
looks equally hot and the variation disappears. Untick Edit → Settings →
General → "Automatically rescale to data range", and set custom ranges from
the generated `output/paraview/field_ranges.csv` (use its `p01`/`p99` columns
for stress, not `min`/`max` — a few nodes at a re-entrant corner otherwise own
the whole scale).

The story to look for: scrub `von_mises_solid` through the deterministic
design's ensemble and the hot spot *moves and spikes* as different ligaments
thin. On the robust design it stays put. That instability is the failure mode
being insured against, and it is visible frame by frame rather than argued.

---

## 5. Request #3 — making the case for robust TO

### The gap this fills

`mainClean.py` Stage 6 validates **one** design:

```python
final_design = pareto_results[-1]
mc_result = run_monte_carlo_validation(fem, opt_nominal, final_design["rho_robust"], ...)
```

Nothing in the repo ever runs the deterministic SIMP design through the same
manufacturing variation. Without that number there is no comparison and
therefore no argument — every current plot describes one design rather than
demonstrating that robust TO was worth doing.

`viz/compare_designs_mc.py` runs the identical ensemble (**same seed**, so
realization *i* is literally the same defect field for every design) against
`rho_converged.npy`, `rho_robust_lambda_0.0.npy` and
`rho_robust_lambda_1.0.npy`. Sharing the perturbation fields makes it a
**paired** comparison, which supports a per-realization win rate — much harder
to wave away than a shift in means.

### What to show, in order

1. **`fig1_cdf_overlay.png`** — the CDFs. Robust TO does not promise a better
   nominal part; it promises a tighter, left-shifted *distribution*. The 95th-
   percentile verticals make the tail claim concrete.
2. **`fig2_distributions.png`**, right panel — the paired scatter. Every dot is
   a controlled A/B on one manufacturing defect field. A win rate near 100 % is
   the strongest single number available.
3. **`03_risk_delta.vtu`**, coloured by `d_prob_void_robust_lambda1` on a
   symmetric Cool-to-Warm range. Warm = robust TO removed the risk of that
   feature not existing. This localizes the benefit instead of asserting it,
   and it is the image people remember.
4. **`03_uncertain_<design>.vtu`** — only the nodes with
   0.02 < `prob_void` < 0.98, i.e. neither reliably solid nor reliably void.
   Everything visible is a location whose existence is a coin flip at
   manufacturing time. *How much of your structure is not guaranteed to
   exist* is a question deterministic TO cannot ask. Compare nominal against
   robust in a split view.
5. **`fig4_pareto.png`** — µ_C vs σ_C along the λ sweep with the independent MC
   point overlaid. Two claims at once: robustness is a tunable knob, not a
   fixed tax; and the in-loop SAA estimate is confirmed out-of-sample
   (0.94 % / 4.85 % relative error on this run). Showing the honesty check
   unprompted does more for credibility than another performance number.
6. **`fig5_likelihood_vs_compliance.png`** — are the bad outcomes rare or
   routine? If routine, the deterministic optimum was never the part you get.
7. **`03_shape_<design>.vtp`**, all loaded at once in different solid colours
   with the robust ones at ~45 % opacity — mean shapes superimposed. Robust designs usually
   differ in an explainable way (thicker members, fewer knife-edge junctions,
   redundant load paths). Being able to point at the design difference turns a
   statistics claim into a design-insight claim.

### Results from the run currently on disk

From `output/comparison/summary.json`, 100 shared realizations, seed 42:

| | mean C | std C | 95th pct | worst case | CV |
|---|---|---|---|---|---|
| `nominal` (deterministic SIMP) | 0.17301 | 0.012157 | 0.19583 | 0.21218 | 7.03 % |
| `robust_lambda0` (mean only) | 0.15913 | 0.008259 | 0.17104 | 0.18801 | 5.19 % |
| `robust_lambda1` (mean + std) | 0.15819 | 0.007824 | 0.16954 | 0.18564 | 4.95 % |

`robust_lambda1` vs `nominal`: **win rate 100 %** (it is better on every one
of the 100 realizations, paired), std **−35.6 %**, 95th percentile **−13.4 %**,
worst case **−12.5 %**, mean **−8.6 %**.

The mean improving by 8.6 % is worth flagging honestly: the robust design is
not trading nominal performance for variance here, it is better on both axes.
That happens when the deterministic optimum sits on a knife edge that the
projection threshold moves off — which is itself the argument, but do not
present it as the general case, because usually there *is* a mean/variance
trade-off. `fig4_pareto.png` shows the actual trade-off curve.

Also note ~5.3–5.6 % of the mesh is neither reliably solid nor reliably void
across all three designs (from `export_paraview.py`'s run notes), so the
perturbation is doing real work — this is not a degenerate ensemble.

### Honest caveats to state rather than hide

- **n = 100.** Enough to separate means, thin for 95th-percentile and
  worst-case claims. Run `compare_designs_mc.py --n-samples 500` (or more)
  before putting tail numbers in front of someone who will push on them.
  `MCConfig` itself warns below 5 000 (`FULL_SPEC_N_MC`).
- **The perturbation band is narrow.** `eta_min/eta_max = 0.45/0.55` with
  `length_scale = 4` is ~0.1–0.25 elements of boundary motion per sample. The
  observed compliance CV is ≈ 4.9 %. If `prob_void` turns out to be ~0/1
  everywhere with no intermediate band, the variation is too weak to
  discriminate designs — widen the band and re-run. That is a finding worth
  reporting, not a plotting failure.
- **Single load case.** `mainClean.py:318-324` warns that robust optimization
  uses only the first of the six configured load cases. The comparison
  inherits that; say so.
- **`compare_designs_mc.py` writes and deletes a scratch ensemble** per design
  (~400 MB of transient I/O). `run_monte_carlo_validation` only populates the
  reliability arrays when `write_ensemble=True`, so this is the cost of getting
  reliability maps without editing pipeline code. Pass `--write-ensemble` to
  keep them.

---

## 6. File map

```
viz/
  build_cloud_index.py    sample_index.csv + iso-surface cloud + likelihood-ordered PVDs
  enrich_ensemble_fea.py  replays the ensemble, adds u / von Mises / SED / eta
  compare_designs_mc.py   paired MC over nominal + robust designs, risk_delta.vtu
  plot_comparison.py      fig1..fig5 (matplotlib)
  export_paraview.py      collects it all into output/paraview/ + a README.txt
  make_paraview_state.py  run once INSIDE ParaView -> stochasticTO.pvsm
```

Only `make_paraview_state.py` imports `paraview`, and it is the one script that
runs *inside* ParaView rather than in the container — which matters, because
the [Dockerfile](../Dockerfile) installs pyvista but not ParaView. Everything
else uses the pyvista/VTK already pinned in
[requirements.txt](../requirements.txt), and writes plain `.vtp` / `.vtu` /
`.pvd` that ParaView Desktop opens directly.

---

## 7. The saved state file

`viz/make_paraview_state.py`, run once in ParaView's Python Shell, writes
`output/paraview/stochasticTO.pvsm`. It reads its colour ranges from
`field_ranges.csv` and its annotations from `output/comparison/summary.json`,
so re-running it after new data picks up the new numbers rather than baking in
stale ones.

**Tab 1 — Probability Cloud.** `01_cloud_all.vtp` coloured by
`compliance_z` on [−2, 3.5], with alpha driven per layer by the `opacity`
array — rare, far-from-nominal geometries render faint, typical ones solid
(8 % flat opacity is the fallback on ParaView builds without a separate
opacity array) — and `01_mean_shape.vtp` opaque underneath as the
reference silhouette. The `likelihood_filter` Threshold is already wired to
`occurrence_likelihood` — select it and drag `LowerThreshold` from 0 to 0.75.
The worst-5 %, top-10 % and extreme-5 % subsets are loaded but hidden, one
eyeball-click away.

**Tab 2 — Ensemble FEA.** Four camera-linked views on a shared
`density > 0.5` Threshold: `von_mises_solid`, `eta`, `strain_energy_density`
(log), and a `WarpByVector` deformed shape. Press Play and all four advance
through the 100 realizations together. Every colour map has
`AutomaticRescaleRangeMode = 'Never'` and a p01–p99 range, so frames stay
comparable; the warp uses one fixed scale factor for the same reason.

**Tab 3 — Robust vs Nominal.** Uncertain material for `nominal` and for
`robust_lambda1` in two camera-linked views on the same P(void) scale; the
`d_prob_void` reduction map as two thresholds (±0.02 outward, so the near-zero
bulk does not hide the signal); and all three mean shapes overlaid with the
headline statistics drawn on.

If you move the repo, ParaView will prompt to relocate the files when loading
the state — point it at `output/paraview/` and it resolves the rest.
