# stochasticTO — complete project context brief

**Self-contained.** Written 2026-07-29 23:00 UTC-4. Assumes no filesystem access
and no prior conversation. Every number here is measured, not estimated. Deadline
for the work: **~2026-08-05**.

---

## 1. What this project is

A robust topology optimization framework that treats **manufacturing error as a
spatially correlated random field on the Heaviside projection threshold**, applied
to 3D linear-elastic compliance minimization. Built on FEniCSx/dolfinx via a
vendored FEniTop, with MMA from dolfiny running as a native PETSc TAO algorithm.

Originally scoped as a NIST project to produce "the first robust TO framework
grounded in real process metrology data." **That framing is no longer supportable**
— see §3.

Test case: a 3D cantilever beam (FEniTop's `beam_3d`), dimensionless, domain
10 × 30 × 10, E = 100, ν = 0.3, vol_frac 0.08, SIMP p = 3, Helmholtz filter
radius R = 0.6. Production mesh 25×75×25 hex→tet, h = 0.400, ~154 k dofs, ~51 k
nodes. Study mesh 16×48×16, h = 0.625, ~42 k dofs, ~14 k nodes.

---

## 2. The formulation

**Design chain.** Design variables ρ → Helmholtz PDE filter
(−R²∇²ρ̃ + ρ̃ = ρ) → smooth Heaviside projection

    ρ̂ = [tanh(βη) + tanh(β(ρ̃ − η))] / [tanh(βη) + tanh(β(1 − η))]

**The uncertainty.** η is not a scalar but a random field η(x). An underlying
Gaussian field G(x) is discretized by a Karhunen–Loève expansion on the FEM nodal
mesh (OpenTURNS `KarhunenLoeveP1Algorithm`), truncated at ≥95 % retained
variance. G is then **standardized by its own pointwise std** and pushed through a
memoryless isoprobabilistic transform to a Beta(2,2) marginal on **[0.25, 0.75]**.

The standardization is load-bearing: it makes the η marginal *exactly* Beta
regardless of σ, of the modes, or of the truncation level. Consequence — the
perturbation magnitude is set solely by the band and the Beta shape, and `sigma`
in the kernel is inert.

**Robust problem.**

    min_ρ  J = μ_C + λ σ_C     s.t.  E[V(ρ)] ≤ V*,  0 ≤ ρ ≤ 1

**Solution method — surrogate-free SAA.** One *fixed* set of N = 512 η
realizations, drawn once and reused at every design iteration (common random
numbers), each evaluated by full FEA. Exact sample-average gradients:

    ∂μ_C/∂ρ = mean_i ∂C_i/∂ρ
    ∂σ_C/∂ρ = (1/σ_C) mean_i [(C_i − μ_C) ∂C_i/∂ρ]

β continuation 8 → 16 → 32 → 64 → 128, each stage a separately converged solve.
MMA via PETSc TAO. Samples parallelize across MPI sub-communicators
(`ranks_per_group = 8`), which is what makes 512 FEA/iteration affordable.

A PCE surrogate path exists but is **not used** — see §7.

---

## 3. THE CENTRAL FINDING — read this before anything else

**The project's original premise was disproved, and the replacement is stronger.**

The premise: spatially correlated error requires a KL random field, and this buys
robustness that a cheap scalar model cannot.

The measurement: sweep the correlation length l_c and evaluate σ_C/μ_C of a fixed
nominal design on a 1000-sample ensemble. Result — **cv rises monotonically with
l_c, and the spatially uniform limit is the worst case.**

| l_c | l_c/L (axial) | n_kl | cv = σ_C/μ_C | 95 % CI |
|---|---|---|---|---|
| 1 | 0.033 | 183 | 1.2337 | [1.1676, 1.3173] |
| 2 | 0.067 | 141 | 1.4611 | [1.3889, 1.5470] |
| 4 | 0.133 | 37 | 1.9305 | [1.8431, 2.0338] |
| 8 | 0.267 | 9 | 2.3502 | [2.2170, 2.5021] |
| 16 | 0.533 | 4 | 3.0260 | [2.8401, 3.2304] |
| 32 | 1.067 | 2 | 3.3059 | [3.0967, 3.5334] |
| **uniform** | ∞ | 1 | **3.1788** | [2.9717, 3.4062] |

**All four adjacent intervals from l_c = 1 to 16 are disjoint** → monotonicity is
statistically established, not eyeballed. l_c = 16, 32 and uniform mutually
overlap, which is the point: all three are effectively the uniform case.
Endpoint contrast **2.58×, disjoint**.

**Mechanism.** At short l_c, erosions in some regions are offset by dilations in
others and the net effect on a global quantity like compliance partially cancels.
As l_c grows the whole structure erodes or dilates together, so the variance is
maximal. This is exactly the cancellation argument Schevenels et al. (2011)
invoked to explain why *their* heat sink's σ halved under spatial variation — now
measured across the whole range instead of asserted at one point.

**The framing the paper must use:**

> The spatially uniform threshold bounds the response variance from above;
> spatial correlation can only reduce it. The inexpensive scalar model is
> therefore **conservative** — which explains Schevenels et al.'s equivalence
> finding mechanistically, extends it from a single correlation length to the
> whole curve, and quantifies the sampling error, which prior work did not do.

Practically actionable: use scalar η at ~64 samples rather than a 512-sample
field. ~8× cheaper, with a conservatism argument.

**Do not attempt to rescue "spatial correlation matters."** It is not what the
data says.

---

## 4. All measured results

### 4.1 Mesh convergence — σ_C/μ_C is a continuum property

| h | R/h | n_kl | μ_C | σ_C | cv |
|---|---|---|---|---|---|
| 0.625 (study) | 0.96 | 37 | 0.7327 | 1.4674 | 2.0028 |
| 0.476 | 1.26 | 37 | 0.8899 | 1.7634 | 1.9817 |
| 0.400 (production) | 1.50 | 37 | 1.0017 | 2.0109 | 2.0076 |

cv spread **1.3 %** over a 1.6× range of h. But μ_C and σ_C *individually* move
~37 %, because the resolved traction area varies 1.4× as facet selection
quantizes the fixed geometric load patch to O(h).

**→ Report σ_C/μ_C. Never report absolute compliance across meshes.**

Corollary: study-mesh and production-mesh cv agree to **0.24 %**, so the study
tier alone supports the paper's claims. Production is the headline deliverable,
not the evidence.

### 4.2 SAA gap study — COMPLETE, 10 replications, N = 512, λ = 4

Ten independently seeded SAA solves; each design re-scored on the *same*
independent 5000-sample set (Mak/Morton/Wood estimator).

| | value |
|---|---|
| σ optimism (in-sample vs out-of-sample) | **−33.9 %** (0.0531 vs 0.0804) |
| Optimality gap | **+3.1 %**, 95 % CI [0.58 %, 2.36 %] — **excludes zero** |
| Run-to-run noise floor, robust (1.4826·MAD/median) | **15.5 %** |
| Run-to-run noise floor, raw std/mean | 38.8 % |
| converged | 0/10 (expected on the study tier) |

Per-replication out-of-sample σ_C — **the distribution is bimodal plus an
outlier**, not a tight cluster:

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

**4 of 10 designs are materially worse than the median cluster**, and in-sample σ
is 0.048–0.058 for all ten — it gives no warning at all. Rep 6 is **not** a
solver artifact: every replication shows exactly 5 `DIVERGED_MAXITS` (one per β
stage) and **zero** KSP failures or non-finite compliance, and rep 6 converged
about as well as the rest (`stat_rel` 0.044). It is a genuine heavy tail.

This directly indicts standard practice. Schevenels et al. justified N = 100 with
a *single* 10 000-sample check on *one* design and concluded "100 samples is
sufficient." That procedure would have missed this in 6 cases out of 10.

Their own published data already contains the effect, unremarked: heat sink,
non-uniform error, σ̂ = 0.042 in-sample vs σ = 0.050 reference — a −16 %
in-sample optimism.

### 4.3 KL truncation robustness — the l_c result is not a discretization artifact

Re-ran the sweep's extremes at a 99 % variance threshold instead of 95 %:

| level | n_kl 95 %→99 % | cv 95 % | cv 99 % | change |
|---|---|---|---|---|
| l_c = 1 | 183 → 197 | 1.2337 | 1.2328 | −0.07 % |
| l_c = 16 | 4 → 6 | 3.0260 | 2.9776 | −1.60 % |
| uniform | 1 → 1 | 3.1788 | 3.1788 | 0.00 % |

Both 95 % point estimates fall inside the 99 % run's own bootstrap CI, and the
shift is *away* from the trend, not toward it. Answers "is 4 modes enough at
l_c = 16?" — yes, within noise. **This also closes the EOLE-vs-KL question:** a
different expansion method would not have changed the finding.

### 4.4 Robust vs deterministic design (E1 uniform arm, complete)

Uniform-η arm, N = 64 (stochastic dimension is 1), λ = 4, β continuation to 128,
9 920 FEA solves. Per-stage:

| β | μ_C | σ_C | M_nd | stat_rel |
|---|---|---|---|---|
| 8 | 0.3645 | 0.3651 | 6.05 % | 0.233 |
| 16 | 0.3168 | 0.2445 | 2.57 % | 0.149 |
| 32 | 0.2928 | 0.1882 | 1.15 % | 0.128 |
| 64 | 0.2800 | 0.1573 | 0.511 % | 0.140 |
| **128** | **0.2718** | **0.1387** | **0.234 %** | 0.124 |

cv = 0.51 versus 2.00 for the nominal deterministic design → **~4× more robust**,
with a near-binary layout (M_nd 0.234 %).

### 4.5 Boundary-offset resolvability (measured, not assumed)

The projected boundary is the level set ρ̃ = η, so a threshold shift displaces it
by δs = δη/|∇ρ̃|; the Helmholtz filter's 1-D step response gives
|∇ρ̃| = 1/(2R). Measured at the study mesh: offset std **0.41 elements**,
i.e. resolvable. ε_max/R = **0.50**.

For comparison, Schevenels et al.: ε = ±0.91 at R = 8.4, h = 1.0 → ε_max/R =
**0.108**. In *element* units the two are comparable (0.75 h vs 0.91 h) — but
relative to the filter this project's perturbation is ~5× more aggressive.

Cause: they had R/h = 8.4; this project has R/h = 1.5. R/h = 8.4 in 3D would
need h = 0.071, i.e. ~8.2 M hexes. **The 2D→3D move forces the coarse filter,
which forces the wide η band.** State that causal chain; it is defensible.

### 4.6 Convergence — never achieved, and now diagnosed

**No run in this project has ever met `robust_opt_tol = 1e-3.`** `stat_rel`
plateaus at 0.04–0.12 and oscillates.

Move-limit continuation was implemented and probed (shrink the MMA move limit
0.7× per β stage). It **FAILED**: final `stat_rel` 0.1413 vs 0.1244 baseline
(13.6 % worse), objective 8 % worse, M_nd 0.312 % vs 0.234 %.

Decisive evidence: **`dx / move_limit = 1.000` at every β stage even after
shrinking the limit 4×** (0.02 → 0.0048). The iterate never leaves the
trust-region boundary regardless of box size, so the move limit is not the cause.

| β | move limit | baseline stat_rel | probe stat_rel |
|---|---|---|---|
| 8 | 0.0200 | 0.233 | 0.233 |
| 16 | 0.0140 | 0.149 | 0.171 |
| 32 | 0.0098 | 0.128 | 0.123 |
| 64 | 0.0069 | 0.140 | **0.104** |
| 128 | 0.0048 | 0.124 | **0.141** |

Smaller steps help at moderate β and hurt at the sharpest projection — the
signature of **β = 128 projection stiffness**. The Heaviside derivative
β·sech²(β(ρ̃−η)) is a spike only ~19/β = 0.15 wide in ρ̃, so as the design moves,
*which* nodes carry gradient flips, and the objective is effectively non-smooth
at the optimizer's working scale.

**Resolution: leave move-limit continuation off** (`move_reduction` defaults to
1.0). Report the achieved residual honestly and put the mechanism in limitations.
**Never loosen `robust_opt_tol` to make `converged: true` appear.**

### 4.7 N convergence

σ_C still rising at N = 512 (+9.8 % over N = 256), flattening by N = 2048
(+0.6 %). Heavy right tail. Consistent with the +3.1 % optimality gap.

---

## 5. Prior art — Schevenels, Lazarov & Sigmund, CMAME 200:3613–3627 (2011)

The closest prior work by a wide margin: same uncertainty model (spatially
correlated random Heaviside threshold, memoryless-transformed from a Gaussian
field, `F = m_f + κσ_f`, mean-volume constraint).

| axis | Schevenels et al. | this project |
|---|---|---|
| Physics | 2D compliant mechanism + 2D heat conduction, 200×200 | 3D linear-elastic compliance, ~154 k dofs |
| Filter | linear hat `max(0, R−r)`, R = 8.4, **R/h = 8.4** | Helmholtz PDE, R = 0.6, **R/h = 1.5** |
| Field discretization | **EOLE**, 100 nodes, no truncation | KL (Galerkin P1) on the FEM mesh, ≥95 % variance |
| η marginal | Uniform [0.4, 0.6] | Beta(2,2) on [0.25, 0.75] |
| Sampling | MC, N = 100, fixed across iterations | MC/LHS, N = 512, fixed across iterations |
| Sample-size justification | one 10 000-sample check → "sufficient" | 10 replications, Mak/Morton/Wood gap, N curves, bootstrap CIs |
| Objective | κ = 1 only | λ sweep, common start, dominance assertion |
| Convergence test | fixed 300 iterations, none | projected-Lagrangian stationarity + feasibility + complementarity |
| β ceiling | 1 → 32, **explicitly refuses 128** | 8 → 128 |

**Two things are NOT differentiators and must not be claimed as such:**

1. **Surrogate-free SAA.** Their Eqs. (26)–(27) are the same estimators, and
   their 100 realizations are "used throughout the entire iteration history" —
   that *is* SAA with common random numbers.
2. **The boundary-offset / resolvability analysis.** Their §3.1 and Figs. 8–9
   already give the η → ε/R map, including the caveat that it holds only for
   features wider than 2R.

**What genuinely differs:** 3D compliance; the l_c sweep instead of a single
point; the error control (replication, gap estimation, CIs on everything); a
Pareto sweep instead of one κ; a real optimality residual.

**Bibliographic note:** the project's original spec cites this as "Chevens et
al. (2011)". That is a garbled rendering of Schevenels — fix it.

**Their central finding, which this project's l_c curve explains:** for both
their test problems, "the design obtained assuming uniform manufacturing errors
is equally robust with respect to non-uniform errors, and vice versa."

---

## 6. Code architecture

```
src/
  meshing/box_source.py          beam_3d geometry + physics. SINGLE SOURCE OF
                                 TRUTH for vol_frac, R, E, move limit, loads.
                                 _MOVE = 0.02, _FILTER_RADIUS = 0.6 (fixed at
                                 every refinement level - it is part of the
                                 continuum problem).
  fenitop/                       VENDORED FeniTop (Jia/Wang/Zhang) + dolfiny MMA
                                 as mma.py. FORKED, not pinned - see §9.
  random_fields/
    kernel.py                    squared-exponential covariance
    kl_expansion.py              OpenTURNS KL + build_uniform_eta_kl()
    threshold_transform.py       memoryless transform to bounded Beta
  topology/heaviside_projection_glue.py
                                 RandomFieldHeaviside: accepts scalar OR
                                 per-node eta. Standardizes G by pointwise_std
                                 BEFORE the transform - this is why the marginal
                                 is exact.
  optimization/
    saa_robust_driver.py         THE production path. SAA + beta continuation.
    optimality.py                first-order optimality (see §9)
    erode_dilate_driver.py       Wang/Lazarov/Sigmund 3-point worst-case baseline
    dolfiny_mma_driver.py        PCE path (unused) + setup_robust_problem +
                                 MPI sample-parallel group machinery
  validation/
    gates.py                     4 FATAL verification gates
    statistics.py                BCa bootstrap, paired CRN comparison, cv ratio
    boundary_offset.py           coarea-based offset measurement
    monte_carlo.py               Stage 6 ensemble
    feature_size.py              erosion-survival measurement (see §8)
  provenance.py                  run manifests: git SHA, versions, seeds, timings
  mainClean.py                   THE ENTRY POINT. 6 stages, always from scratch.

scripts/
  uniform_eta_baseline.py        E1: uniform-eta control + 2x2 cross-evaluation
  correlation_length_study.py    E2: l_c sweep (fixed) + cross-eval grid (reopt)
  saa_gap_study.py               Mak/Morton/Wood optimality gap, M replications
  convergence_studies.py         mesh / n-fixed / n-opt
  baseline_comparison.py         SAA vs erode/dilate head-to-head
  move_limit_probe.py            the convergence probe (§4.6)
  measure_feature_size.py        beta=128 defence runner
  watch_progress.sh              status dashboard
  job_watch.sh                   ntfy push notifier daemon
  summarize_result.py            phone-readable summary of any finished study
  phase1_chain.sh                overnight auto-chain

viz/
  paths.py                       run-id-aware artifact resolution
  plot_research_figures.py       figA (l_c), figB (mesh), figC (replications)
  plot_comparison.py             CDF / distributions / risk / Pareto
  + ParaView state builders (written against the OLD pre-run-id layout)

tests/  108 tests, no cluster needed. pytest tests/ -q
```

**Configs.** `config.yaml` (production, h = 0.400, λ ∈ {0, 0.5, 1, 2, 4},
max_iter 400, N = 512), `configStudy.yaml` (h = 0.625, λ ∈ {0, 1, 4},
max_iter 150), `configSmoke.yaml` (throwaway, β capped at 32).

The loader **rejects any key inert on the active path**. The previous config
declared vol_frac 0.15, filter_radius 0.006 m, E = 68.9 GPa and six SI load cases
— none of which reached the solver, because `box_source.py` overrode all of them.
Anyone reconstructing the study from that file would have reported the wrong
number for essentially every physical parameter, in the wrong unit system.

**Verification gates** (all fatal, written to `gates.json`): Heaviside
forward/backward vs closed form; KL sample correlation vs the theoretical kernel
(noise-aware, Bonferroni-corrected); η marginal KS test; finite-difference
gradient check (tightens KSP to 1e-12 first, because at default tolerance a 1e-6
step measures solver noise).

---

## 7. Deliberate departures from the original spec

- **PCE surrogate bypassed.** The map ξ→C depends on ρ, so the surrogate is stale
  the moment the design moves and must be refit every iteration; and Q² ≥ 0.99 is
  a *response*-variance criterion dominated by the mean — a surrogate can pass it
  while being materially wrong about σ, which is the entire quantity of interest.
  SAA has zero surrogate error and its sampling error is estimable.
  **Consequence: no Sobol indices** (they lived in the PCE path).
- **No metrology.** `src/metrology/` deleted, Open3D not a dependency. Kernel
  parameters come from config, not data. The η band was chosen for *mesh
  resolvability*, not fitted to a process. **The framework is therefore a
  robustness envelope in the erode/dilate tradition, NOT a calibrated process
  tolerance.** Any claim otherwise is unsupportable and is the single thing most
  likely to sink a review.
- **Hardcoded beam_3d instead of STEP/CAD.** Needed a case with predictable cost,
  a published reference, and a controlled refinement family.
- **Analytical cantilever check (PL³/3EI) dropped.** Euler–Bernoulli does not
  verify a 3D SIMP compliance with a patch load and a filter; agreement would be
  coincidence. Replaced by mesh convergence of the actual QoI, which passed.
- **λ sweep is common-start.** Chaining warm starts made the sweep one continued
  descent and produced a *dominated* point (λ=1 beat λ=0 in both μ_C and σ_C,
  which is impossible for genuine optima). Now every λ starts from the nominal
  design and the run asserts non-dominance.
- **Two-tier compute** (study mesh for everything that must be converged or
  replicated; one production run at the end) rather than the spec's two-tier UQ.

---

## 8. State right now, and what happens next

### Running
**Phase 1 auto-chain**, started 2026-07-29 22:05 on 128 ranks, ~7–8 h:
E1 field arm (currently β stage 2/5, 61 iterations) → E1 evaluate ×2 → E1 report
→ erode/dilate baseline → **study-mesh full pipeline (the production rehearsal)**.
Expected complete 05:00–06:00.

### Complete
mesh convergence · N convergence (fixed + reopt) · SAA gap (10/10) · l_c sweep
with cv CIs · KL truncation check · E1 uniform arm · move-limit probe ·
figA/figB/figC.

### Not yet run
- **Cross-evaluation grid** (`correlation_length_study.py reoptimize 1 4 uniform`,
  ~8 h). Optimize at l_c = X, score at Y, all pairs. **This is what upgrades the
  "conservative" claim from a statement about the response to one about the
  design.** Without it, soften the abstract accordingly.
- **Feature-size measurement** (`measure_feature_size.py --all`, ~15 min). Sweeps
  η upward — a morphological erosion of depth (η−0.5)·2R — and finds where the
  load path severs. **This is the answer to Schevenels' β ≥ 128 objection.**
  M_nd does not answer it: a single-element strut is also near-binary.
- **Production run.** 5-point λ sweep, `mainClean.py src/config/config.yaml`.
  **Not run on the current box** (shared, 128 cores, and another user is using
  ~40 of them). 67 h at 128 ranks / 33 h at 256 / **17 h at 512**. Only
  constraint is `world_size % 8 == 0`. Each MPI group holds its own copy of the
  154 k-dof mesh, so check RAM before maxing ranks.

### Immediate next actions
1. Read E1's verdict, the erode/dilate cost ratio, and the rehearsal's
   `gates.json` + Pareto front.
2. If the rehearsal passes, the production config is validated end-to-end (same
   code path, same five β stages, same gates) — port and launch on bigger
   hardware.
3. Run the cross-eval grid and feature-size measurement in parallel on the
   current box.
4. Build remaining figures, then write.

**⚠ E1 is underpowered and will probably return "indistinguishable."** Its
decision rule compares the two arms against the gap study's run-to-run noise
floor, which is **15.5 %** — a single-solve-per-arm comparison cannot resolve
less than that. Report it as underpowered rather than as a clean null.

---

## 9. Honest limitations (these belong in the paper)

1. **Not metrology-calibrated.** The η band is resolvability-driven. It is a
   robustness envelope, not a process tolerance.
2. **The perturbation is aggressive relative to feature size** — ε_max/R = 0.50
   vs 0.108 in prior work — forced by 3D affordability (R/h = 1.5 vs 8.4).
3. **First-order optimality is not achieved.** `stat_rel` ≈ 0.04–0.06 at N = 512
   against a 1e-3 tolerance, diagnosed as β = 128 projection stiffness (§4.6).
4. **KL is expanded on the FEM mesh**, so the dense eigensolve is O(N_nodes²):
   1.6 GB at the study mesh, **21 GB at production**. This caps refinement. An
   EOLE-style coarse auxiliary grid would decouple the two — at l_c = 4 that is
   ~1500 points (17 MB) instead of 51 k — but the saving inverts at small l_c
   (l_c = 1 at spacing l_c/3 needs ~81 k points, worse than the FEM mesh).
5. **KL, not EOLE**, where Li & Der Kiureghian argue EOLE is more efficient for
   squared-exponential kernels; and truncated to ~20–40 modes where the prior
   work retained all 100. §4.3 shows this does not affect the conclusion.
6. **FEniTop and dolfiny are FORKED, not pinned.** The spec said "fork only if
   strictly necessary and document why." The why:
   - `mma.py` — a pre-loop `if ‖∇f‖ ≤ gatol: return` that could terminate at
     iteration 0; the multiplier vector never initialized; an off-by-one in the
     iteration range; `DIVERGED_MAXITS` never set on budget exhaustion, so
     running out of iterations was indistinguishable from converging.
   - `utility.py::solve_fem` — the zero-initial-guess retry after a KSP failure
     was gated on a branch unreachable in exactly the MC-loop configuration.
     2 of 32 solves were failing with `DIVERGED_DTOL` and entering the statistics
     as if converged.
   - `sensitivity.py`/`topopt.py` — accumulate-gradients rewrite (S1/S2 running
     sums), turning 1024 world broadcasts per objective evaluation into 3
     reductions.
   The first two **changed results**, not just speed. Say so.
7. **`kkt_residual` used to be `‖∇f‖`**, which cannot vanish on a
   volume-constrained problem (at the optimum the objective gradient balances the
   constraint gradient). `reason = 0` — TAO for "still iterating" — was being read
   as success. Replaced by an active-set projected-Lagrangian stationarity
   residual, normalized, reported with feasibility and complementarity.

---

## 10. Operational facts

- Host `pyrite.cam.nist.gov`, user `ovb`, **128 cores**, 503 GB RAM. **Shared** —
  another user (`dlb8`) is running a ~40-core training job, which slowed the gap
  study from 3h04 to 3h25 per replication.
- Container `fenitop_container` (image based on `dolfinx/dolfinx:stable`).
  **Every docker command needs `sudo docker`.** Host `/raid/ovb/stochasticTO` →
  container `/shared`. **`cd /shared` before every launch** — scripts hardcode a
  relative `Path("output")`.
- **MPICH, not OpenMPI** — no `--allow-run-as-root`.
- **`source mpi-env.sh`** before every `mpirun` (pins `OMP_NUM_THREADS=1`).
- `output/` is written by the container as **root** into an `ovb`-owned tree:
  readable, not writable from the host. Run anything that writes there inside the
  container.
- Notifications: `scripts/notify_push.sh` → `ntfy.sh/stochasticTO-488039e71edb`.
  `scripts/job_watch.sh` runs detached on the host and pushes full results on
  completion, HIGH priority on crash. Email does not work (the NIST relay accepts
  and silently drops).
- `run_id` is always `..._nogit` because git fails inside the container.

### Traps that already cost time

- **`pgrep -f <script.py>` cannot detect these jobs.** The wrapper chain
  (`sudo` → `docker` → `bash -c '...'`) carries every script name in its own
  command line, so a finished job looks alive and a wrapper exit looks like a
  crash. Match on process name:
  `ps -eo comm=,args= | awk '$1 ~ /^python/ && index($0,p)>0'`.
- **`pkill -f <script.py>` kills the parent block and every sibling job.** It
  happened. Kill by PID.
- **Long log silence is normal.** Only rank 0's sample-parallel *group* logs, so
  with 4 groups three are invisible; and each new l_c is a KL cache miss → a dense
  eigensolve on ~14 k nodes. 6–10 min of silence is healthy.
- **Don't trust the cost table in `configStudy.yaml`.** Its "erode/dilate
  baseline … 3 min" is the baseline *arm* only; `baseline_comparison.py` also runs
  a full SAA solve (~5 h at 32 ranks). Measure, don't estimate.
- **cv must be bootstrapped as one statistic**, not propagated from separate μ and
  σ intervals. They come from the same draws and are positively correlated;
  propagation is 2.1× too wide on this data, enough to make l_c = 1 and l_c = 2
  overlap when they are in fact disjoint. This is why monotonicity is provable.

---

## 11. Draft abstract (numbers current as of the completed gap study)

> This paper examines **when** spatially correlated manufacturing error must be
> modelled as a random field in robust topology optimization, and when a single
> random variable suffices. The focus is on structures produced by milling or
> etching, where over- or under-etching causes parts of the structure to become
> thinner or thicker than intended. Following the established projection-based
> tradition, this error is modelled by applying a density filter followed by a
> smooth Heaviside projection whose threshold η is randomized: a low threshold
> simulates under-etching and a high threshold over-etching. Spatial variation is
> introduced by representing η as a non-Gaussian random field, obtained as a
> memoryless transformation of an underlying Gaussian field discretized by a
> Karhunen–Loève expansion on the finite element mesh, with the threshold band
> selected so that the induced boundary displacement is resolvable on that mesh
> (standard deviation 0.41 elements). Prior work established this formulation in
> two dimensions at a single correlation length and reported that designs
> optimized against uniform error are equally robust to non-uniform error;
> whether that finding generalizes was left open. Here the correlation length is
> instead treated as the independent variable and swept over a 32-fold range in
> three-dimensional linear elasticity. The robust problem minimizes a weighted
> sum of the mean and standard deviation of compliance subject to a mean-volume
> constraint, and is solved by sample average approximation: a fixed set of 512
> realizations is evaluated by full finite element analysis at every design
> iteration, using exact sample-average gradients and Heaviside continuation to
> β = 128. The ratio σ_C/μ_C varies by less than 1.4 % across a 1.6-fold range of
> element size, and increases monotonically with correlation length — every
> adjacent 95 % confidence interval from ℓ_c = 0.03L to 0.53L is disjoint — from
> 1.23 to 3.03, with the spatially uniform limit at 3.18 and statistically
> indistinguishable from the two longest correlation lengths. The spatially
> uniform threshold therefore bounds the response variance from above: spatial
> correlation can only reduce it, and the inexpensive scalar model is
> conservative. Robust design reduces σ_C/μ_C from 2.00 to 0.51 while retaining a
> near-discrete layout (measure of non-discreteness 0.23 %). Ten independently
> seeded solves further show that the in-sample standard deviation is optimistic
> by 34 %, that the sample-average approximation lands 3.1 % above the true
> robust optimum at N = 512 (95 % CI excluding zero), and that four designs in
> ten are materially less robust out of sample — one by a factor of 2.5 — with no
> in-sample indication whatsoever, a failure mode invisible to the
> single-verification-run practice of prior work.

**Do not add to it:** metrology calibration; a first-order optimality claim; or a
cost ratio (the erode/dilate baseline runs in Phase 1 and is not yet in).
