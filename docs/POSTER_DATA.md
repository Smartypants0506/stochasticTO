# Poster data pack — everything you need to write, in one place

Generated 2026-07-30. **Every number here is FINAL** and traceable to a JSON under
`output/studies/`. Nothing in this file depends on the production run.

---

## The one-sentence takeaway

> The spatially uniform threshold bounds the response variance from above —
> spatial correlation can only reduce it — so the inexpensive scalar model is
> the conservative choice.

This inverts the project's original premise (that a spatially-correlated random
field was *needed*). It is a stronger, more useful claim, and it explains
mechanistically why Schevenels et al. (2011) found uniform and non-uniform
designs equally robust at their single correlation length.

---

## HEADLINE RESULT — σ_C/μ_C vs correlation length

Nominal design, study mesh, 1000-sample evaluation ensemble. `cv = σ_C/μ_C`.

| ℓ_c | ℓ_c/L (L=30) | n_kl | cv | 95% CI |
|---|---|---|---|---|
| 1 | 0.033 | 183 | **1.2337** | [1.1676, 1.3173] |
| 2 | 0.067 | 141 | **1.4611** | [1.3889, 1.5470] |
| 4 | 0.133 | 37 | **1.9305** | [1.8431, 2.0338] |
| 8 | 0.267 | 9 | **2.3502** | [2.2170, 2.5021] |
| 16 | 0.533 | 4 | **3.0260** | [2.8401, 3.2304] |
| 32 | 1.067 | 2 | **3.3059** | [3.0967, 3.5334] |
| **uniform** | ∞ | 1 | **3.1788** | [2.9717, 3.4062] |

**Claims you can make, and the exact wording:**

- "cv increases **monotonically** with correlation length" — **all four adjacent
  95% CIs from ℓ_c = 1 to 16 are disjoint.** This is provable, not eyeballed.
- Endpoint contrast: **2.58×**, intervals disjoint.
- ℓ_c = 16, 32 and the uniform limit are **mutually indistinguishable** — which
  is the point: all three are effectively the uniform case.
- ℓ_c = 32 sits 4.0% above the uniform line because at ℓ_c > domain the
  expansion still retains 2 modes and is not yet exactly constant. Mention it;
  do not call it a peak.

**Do NOT write** "the scalar model is conservative *for design*." That is a
design-level claim requiring the cross-evaluation grid, which was never run.
The defensible claim is about the **response**.

Figure: `output/figures/figA_correlation_length.png`

---

## VERIFICATION — mesh convergence

| h | R/h | μ_C | σ_C | cv |
|---|---|---|---|---|
| 0.625 (study) | 0.96 | 0.7327 | 1.4674 | 2.0028 |
| 0.476 | 1.26 | 0.8899 | 1.7634 | 1.9817 |
| 0.400 (production) | 1.50 | 1.0017 | 2.0109 | 2.0076 |

- cv spread across a 1.6× range of element size: **1.3%**
- μ_C and σ_C **individually** move ~37%, because the resolved traction area
  varies 1.4× as facet selection quantizes the fixed load patch to O(h)
- **Therefore: report σ_C/μ_C, never absolute compliance.** This is a real
  methodological point worth a sentence.
- Study and production meshes agree on cv to **0.24%**

Figure: `output/figures/figB_mesh_convergence.png`

---

## SAMPLING ERROR — the most novel content

SAA gap study, 10 independent seeds, N=512, λ=4, each design re-scored on the
same independent 5000-sample set (Mak/Morton/Wood).

| quantity | value |
|---|---|
| σ optimism (in- vs out-of-sample) | **−33.9%** (0.0531 vs 0.0804) |
| Optimality gap | **+3.1%**, 95% CI [0.58%, 2.36%] — **excludes zero** |
| Noise floor, robust (1.4826·MAD/median) | **15.5%** |
| Noise floor, raw std/mean | 38.8% |
| Converged | 0/10 (expected on the study tier) |

Per-replication out-of-sample σ_C — **bimodal plus an outlier**:

| rep | in-sample σ | out-of-sample σ | × median |
|---|---|---|---|
| 0 | 0.0546 | 0.0610 | 0.93 |
| 1 | 0.0530 | 0.0652 | 1.00 |
| 2 | 0.0533 | 0.0560 | 0.86 |
| 3 | 0.0518 | 0.0654 | 1.00 |
| 4 | 0.0527 | 0.0652 | 1.00 |
| 5 | 0.0576 | 0.0620 | 0.95 |
| **6** | 0.0548 | **0.1610** | **2.47** |
| 7 | 0.0484 | 0.0890 | 1.36 |
| 8 | 0.0526 | 0.0886 | 1.36 |
| 9 | 0.0521 | 0.0903 | 1.38 |

**The claim:** 4 of 10 designs are materially worse out of sample, one by 2.5×,
and **in-sample σ is 0.048–0.058 for all ten** — it gives no warning whatsoever.
Rep 6 is not a solver artifact: every replication shows exactly 5
`DIVERGED_MAXITS` (one per β stage) and zero KSP failures, and rep 6 converged
about as well as the rest (stat_rel 0.044).

**Why this matters:** Schevenels et al. justified N=100 with a *single*
10,000-sample check on *one* design and concluded "sufficient." That procedure
would have missed this in 6 cases out of 10. Their own published data already
contains the effect unremarked — heat sink, non-uniform error, σ̂ = 0.042
in-sample vs σ = 0.050 reference, a −16% optimism.

Figure: `output/figures/figC_gap_replications.png`

---

## THE PAYOFF — robust vs deterministic

Uniform-η arm, N=64 (stochastic dimension is 1), λ=4, β continuation to 128,
9,920 FEA solves.

| β | μ_C | σ_C | M_nd | stat_rel |
|---|---|---|---|---|
| 8 | 0.3645 | 0.3651 | 6.05% | 0.233 |
| 16 | 0.3168 | 0.2445 | 2.57% | 0.149 |
| 32 | 0.2928 | 0.1882 | 1.15% | 0.128 |
| 64 | 0.2800 | 0.1573 | 0.511% | 0.140 |
| **128** | **0.2718** | **0.1387** | **0.234%** | 0.124 |

**cv = 0.51 vs 2.00 for the nominal design → ~4× more robust**, with a
near-binary layout (M_nd 0.234%).

---

## KL TRUNCATION ROBUSTNESS (supporting)

95% vs 99% retained variance at the sweep's extremes:

| level | n_kl 95%→99% | cv 95% | cv 99% | change |
|---|---|---|---|---|
| ℓ_c = 1 | 183 → 197 | 1.2337 | 1.2328 | −0.07% |
| ℓ_c = 16 | 4 → 6 | 3.0260 | 2.9776 | −1.60% |

Both 95% point estimates fall inside the 99% run's own CI. Answers "is 4 modes
enough at ℓ_c=16?" — yes, within noise. Also closes the EOLE-vs-KL question: a
different expansion method would not have changed the finding.

---

## METHOD (for the formulation block)

SIMP `E(ρ) = E₀ρᵖ`, p=3 → Helmholtz PDE filter `−R²∇²ρ̃ + ρ̃ = ρ`, R=0.6 →
smooth Heaviside projection with threshold η, β continuation 8→128.

η is a **non-Gaussian random field**: an underlying Gaussian field is
discretized by a Karhunen–Loève expansion on the FEM mesh (≥95% retained
variance), standardized by its pointwise std, then pushed through a memoryless
isoprobabilistic transform to a **Beta(2,2) marginal on [0.25, 0.75]**.

    min_ρ  J = μ_C + λ σ_C     s.t.  E[V(ρ)] ≤ V*,  0 ≤ ρ ≤ 1

Solved by **sample average approximation**: one fixed set of N=512 η
realizations, drawn once and reused every design iteration (common random
numbers), each evaluated by full FEA. Exact sample-average gradients. MMA via
PETSc TAO, samples parallelized across MPI sub-communicators.

Test case: 3D cantilever beam, domain 10×30×10 (dimensionless), E=100, ν=0.25,
vol_frac 0.08, 25×75×25 → tets, h=0.4, ~154k dofs.

---

## LIMITATIONS — write these, they strengthen the poster

1. **Not metrology-calibrated.** The η band was chosen for mesh resolvability,
   not fitted to a process. This is a robustness *envelope*, not a process
   tolerance.
2. **Aggressive perturbation relative to feature size.** ε_max/R = **0.50** here
   vs **0.108** in Schevenels et al., forced by 3D affordability: their R/h was
   8.4, this is 1.5. R/h = 8.4 in 3D would need ~8.2M hexes.
   Boundary offset std = **0.41 elements** (measured, resolvable).
3. **First-order optimality not achieved.** stat_rel plateaus at 0.04–0.12
   against a 1e-3 tolerance. Move-limit continuation was tested and **failed**
   (0.1413 vs 0.1244 baseline, objective 8% worse); `dx/move_limit = 1.000` at
   every stage proves the move limit is not the cause. Mechanism is **β=128
   projection stiffness** — the responsive band is only ~19/β = 0.15 wide.
4. **The FD gradient gate is disabled for the production run — and the reason is
   now understood.** A term-split finite-difference test localised the
   disagreement precisely:

   | term | max rel err | median | verdict |
   |---|---|---|---|
   | `dE[V]/dρ` (control) | 0.0004% | 0.0002% | exact |
   | `dμ_C/dρ` | 0.569% | 0.041% | fine |
   | `dσ_C/dρ` | 4.410% | 0.164% | the outlier |

   The error is confined to `dσ_C/dρ`. **That formula and its implementation are
   nonetheless provably correct**: `compute_dsigma_drho` reproduces an exact
   finite difference to **3.5 × 10⁻¹⁰** on a synthetic noiseless problem, and
   the derivation checks out —
   `σ² = Σ(Cᵢ−μ)²/(N−1)` ⟹ `dσ/dρ = Σ(Cᵢ−μ)(dCᵢ/dρ)/((N−1)σ)`,
   with the `dμ/dρ·Σ(Cᵢ−μ)` term vanishing identically.

   The conclusion is therefore about the *reference*, not the gradient: a
   central difference of a **second-moment** quantity subtracts two nearly-equal
   standard deviations, so it inherits amplified noise from the individual
   compliance solves in a way that `μ_C` and `E[V]` — both first moments — do
   not. Three alternative explanations were tested and disproved along the way
   (near-zero denominators, FD truncation at high β, warm-start jitter).

   **Honest statement for the poster:** first-moment sensitivities are verified
   to 4–6 significant figures; the second-moment sensitivity is verified
   analytically and to ~0.16% median against a reference that is itself noisy at
   that scale. The gate's 0.1% tolerance is simply not attainable for σ_C by
   finite differences on this problem.
5. **Forked upstream.** FEniTop and dolfiny's MMA are vendored and modified —
   four MMA defects (including a convergence check that could terminate at
   iteration 0 and uninitialized multipliers) and a `solve_fem` retry that was
   unreachable in the MC-loop configuration, silently admitting 2/32 failed
   solves as converged. Two of those changed results.

---

## PRIOR ART — how to position

Closest work by a wide margin: **Schevenels, Lazarov & Sigmund, CMAME
200:3613–3627 (2011)**. Same uncertainty model (KL random field on the Heaviside
threshold, memoryless transform, μ + λσ objective, mean-volume constraint).

| | Schevenels et al. | this work |
|---|---|---|
| Physics | 2D mechanism + 2D heat sink | **3D linear elastic compliance** |
| Filter | linear hat, R/h = 8.4 | Helmholtz, R/h = 1.5 |
| Field discretization | EOLE, 100 nodes, no truncation | KL on FEM mesh, ≥95% variance |
| Correlation length | **one value** (0.3L) | **swept 32-fold** |
| Sample-size justification | one 10,000-sample check | 10 replications + gap estimator + CIs |
| Objective | κ = 1 only | λ sweep |
| Convergence test | fixed 300 iterations, none | projected-Lagrangian residual |

**Two things that are NOT novel and must not be claimed:**
1. Surrogate-free SAA — their 100 fixed realizations reused every iteration *is*
   SAA with common random numbers.
2. The boundary-offset / resolvability analysis — their §3.1 and Figs. 8–9
   already give the η → ε/R map.

**What IS novel:** the correlation-length sweep and its finding; the sampling-
error quantification.

---

## FIGURES

**Line plots (ready):** `output/figures/`
- `figA_correlation_length.png` — the headline
- `figB_mesh_convergence.png` — cv converged, μ_C and σ_C not
- `figC_gap_replications.png` — in vs out-of-sample σ, the outlier visible

**3D geometry (ParaView):** `output/paraview/poster/poster_fields.vtu`
One file, eight point-data arrays. Recipe: **Threshold** (scalars = the array,
range 0.5→1.0) → **Extract Surface** → **Smooth** (50–100 iters). Set the camera
ONCE, then only change the Threshold's Scalars dropdown — that keeps panels
honestly comparable.

| array | what | volume fraction |
|---|---|---|
| `density_eta075` | over-etched (thinner) | 0.0262 |
| `density_eta050` | as designed | 0.0809 |
| `density_eta025` | under-etched (thicker) | 0.1497 |
| `density_nominal` | deterministic design | — |
| `density_field` | field-optimized design | — |
| `eta_sample_0/1` | the η(x) field itself | colour-map, **don't** threshold; fix range 0.25–0.75 |

Those three volume fractions (0.026 / 0.081 / 0.150) are a caption in
themselves — the eroded case retains under a third of the intended material.

---

## STILL BLOCKED (leave placeholders)

- **Production Pareto front** — running now (80 ranks, started 12:19). 5 λ points
  ordered [0, 4, 1, 2, 0.5] so any prefix of 3 spans the trade-off. Designs save
  incrementally, so a partial run is still usable.
- **Erode/dilate cost comparison** — the baseline driver had three bugs (MPI
  constraint layout, epigraph variable runaway, missing adaptive volume target);
  first two fixed, the third is blocked on an MMA/epigraph structural issue.
  No cost-ratio table yet.
