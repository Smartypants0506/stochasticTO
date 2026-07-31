# Architecture divergence: `masterContext.md` vs. the code as built

*Status: current as of commit `fb28b16`, 2026-07-29. Update this document whenever a
divergence is opened or closed.*

## Context

[`masterContext.md`](../masterContext.md) is the project's design specification (Revision 2,
prebuilt-software integration). The code that exists now departs from it in eleven material
ways. Some of those departures are deliberate scope cuts, some were forced by numerics the
spec did not anticipate, and three are corrections of things the spec got wrong. None of them
was hidden — but until now none was written down either, and a reviewer reading the spec would
form a materially wrong picture of what was actually run. This document is the reconciliation.

The organizing question for each entry is not "did we follow the spec" but **"what would a
reviewer conclude if they read the spec and then read the results?"**

### Summary

| # | Specified | Built | Class |
|---|---|---|---|
| D1 | PCE surrogate drives the robust loop | Surrogate-free SAA | Forced (accuracy) |
| D2 | Metrology-calibrated η via Open3D ICP | Not implemented; `src/metrology/` deleted | **Scope cut — changes the claim** |
| D3 | η calibrated from process data | η ∈ [0.25, 0.75], chosen for resolvability | Forced (discretization) |
| D4 | STEP/CAD → Gmsh pipeline | Hardcoded `beam_3d` box | Scope cut |
| D5 | 8 verification gates incl. PL³/3EI, Q²≥0.99 | 4 different gates, all fatal | Mixed |
| D6 | MC 5000+, validates the PCE | MC 2000, paired CRN, validates *designs* | Forced (purpose changed) |
| D7 | Two-tier UQ (PCE / MC) | Two-tier **compute** (study mesh / production) | Consequence of D1 |
| D8 | Probability cloud, CAVE XR, 500–5000 meshes | Code retained, `write_ensemble: false` | Deferred |
| D9 | `external/` pinned FEniTop + dolfiny | Vendored in-tree and **modified** | **Rule violation — justified** |
| D10 | λ sweep warm-started from nominal | Common-start + dominance assertion | **Correction of a bug** |
| D11 | `src/main.py`, `tests/{unit,regression,integration}` | `src/mainClean.py`, flat `tests/` | Correction |

---

## Part 1 — Divergences forced by numerics

### D1. The PCE surrogate is bypassed; SAA is the production path

**Spec** (§3.4, §3.5): a sparse PCE is fit to C(ξ); moments come out analytically
(`μ = c₀`, `σ² = Σ_{α≠0} c_α²`); "no additional FEA solve needed per objective evaluation";
a Q² ≥ 0.99 gate on a held-out set governs deployment.

**Built:** `robust_method: "saa"` is the default ([schema.py:145](../src/config/schema.py#L145)).
Stage 4/5 skip the surrogate entirely and evaluate the exact sample average over a fixed set of
N = 512 η-realizations by FEA at every design iteration
([saa_robust_driver.py](../src/optimization/saa_robust_driver.py)). The PCE path still exists
and is selectable — `pce_builder.py`, `pce_model.py`, `pce_evaluation.py`,
`dolfiny_mma_driver.py` are intact — but it is not what produces results.

**Why.** Three reasons, in order of weight:

1. **The surrogate has to be refit every iteration anyway.** The map ξ ↦ C depends on the
   design ρ. As ρ moves, the PCE is stale. The spec's headline saving — "no FEA per objective
   evaluation" — only holds *within* one design point; across a 400-iteration solve the
   retraining cost is the same order as just evaluating the samples.
2. **Q² ≥ 0.99 does not certify σ.** Q² is a fraction of *response* variance explained, and the
   response is dominated by its mean. A surrogate can clear Q² = 0.99 while being materially
   wrong about the standard deviation, which is the entire quantity of interest here. The
   spec's gate protects the wrong number.
3. **Samples parallelize perfectly.** 512 FEA solves per iteration is affordable when they run
   as 8 MPI sub-communicator groups × 64 sequential solves
   (`sample_parallel_ranks_per_group: 8`). This is the structural fact that makes
   surrogate-free viable, and the spec — written before the parallel layout existed — could not
   assume it.

SAA has **zero surrogate error** by construction. Its error is sampling error, which is
estimable — see the Mak/Morton/Wood gap study,
[`scripts/saa_gap_study.py`](../scripts/saa_gap_study.py) — where PCE error is not.

**Consequences that must be stated in the paper:**
- No Q² gate is reported, because no surrogate is deployed.
- Moments are not analytic; they are sample averages with bootstrap CIs.
- **Sobol indices are not produced** (spec §4.3). `sobol.py` and `kl_sensitivity_diagnostic.py`
  survive but only the PCE branch calls them. The spec used Sobol indices to justify the KL
  truncation order; that justification is now made directly by the ≥95% retained-variance
  criterion, which is weaker. This is a genuine loss of a deliverable.

### D3. The η band is [0.25, 0.75] and the reason is the mesh, not the process

**Spec** (§3.3): η's marginal is "calibrated from metrology data rather than assumed uniform."
No band is given.

**Built:** η ~ Beta(2, 2) on **[0.25, 0.75]**, identical at every compute tier.

**Why — this is the most important number in the project, and it is set by discretization.**
The projected boundary is the level set ρ̃ = η, so a threshold shift δη displaces it by
δs = δη / |∇ρ̃|. For the Helmholtz filter (−R²∇²ρ̃ + ρ̃ = ρ) the 1-D step response gives
|∇ρ̃| = 1/(2R) = 0.83 at R = 0.6. At the production mesh h = 0.4:

| η band | boundary-offset std | in elements |
|---|---|---|
| ±0.05 (a plausible calibrated tolerance) | 0.032 | **0.080 h** |
| [0.35, 0.65] | 0.096 | 0.24 h |
| **[0.25, 0.75] (built)** | **0.16** | **0.40 h** |

A ±0.05 band moves the boundary **one twelfth of an element**. The compliance variation being
optimized would then be the same order as the discretization error of compliance itself, and
σ_C could not be shown to be a property of the continuum problem at all.

**Refinement cannot rescue it.** δs is fixed in absolute units — it depends on the band and R,
not on h — so refining only grows δs/h linearly. Reaching 0.5 h needs h = 0.064, i.e. 11.5 M
hexes (~70 M tets). Not affordable at 4 s per 154 k-dof solve.

So the band was widened until it was resolvable, and [0.25, 0.75] is the landing point because
it is simultaneously the smallest resolvable band *and* the standard Wang/Lazarov/Sigmund
erode/dilate band — which makes the baseline comparison direct rather than contrived.

This is **measured, not asserted**:
[`src/validation/boundary_offset.py`](../src/validation/boundary_offset.py) computes the offset
two ways (pointwise |∇ρ̃| over the interface band, and the coarea identity
offset ≈ (V(η) − V(0.5))/A with A = ∫|∇ρ̃| dx) and reports it in absolute units, in units of h,
and as a fraction of the minimum feature size.

**The cost, which the paper must state plainly:** the offset range is ~60% of the minimum
feature size (~2R). This is a **robustness envelope in the erode/dilate tradition, not a
calibrated process tolerance.** A calibrated ±0.05 tolerance is not resolvable on any mesh this
project can afford. That belongs in the limitations section, not a footnote.

### D6. Monte Carlo validation: fewer samples, different job

**Spec** (§3.6): N_mc = 5000+; the job is validating the PCE against brute force; pass/fail on
Q² ≥ 0.99; flags PCE tail underprediction.

**Built:** `n_samples: 2000` in production. Every design — nominal *and* every λ — is evaluated
on **one common ensemble** with common random numbers; statistics are BCa bootstrap CIs and
paired comparisons ([`src/validation/statistics.py`](../src/validation/statistics.py)).

**Why the purpose changed.** With no PCE there is nothing to validate against. Stage 6 stopped
being a cross-check and became **the primary estimator** of μ_C and σ_C, and the question it
answers is no longer "is the surrogate faithful" but "**is the difference between these two
designs larger than the noise in my estimate of it**". For that question, paired CRN comparison
extracts far more power per sample than two independent 5000-sample runs, because the shared
η-realizations cancel. `resolvability_report()` states the n that *would* be required for
whatever effect actually turns up, so an unresolvable difference gets reported as unresolvable
rather than as a result.

This directly addresses a defect in the earlier work: a 6.2% σ_C improvement was being claimed
from an n = 100 run whose CI on σ was ±7%. The claim was inside its own error bar.

**Added beyond the spec:** `max_solver_failure_rate`. At η = 0.75 the structure is heavily
eroded and may not carry load. The spec had no policy for this. The code logs the failure rate,
aborts if it exceeds the tier's threshold (1% production, 5% study), and — critically — flags
the surviving statistics as **conditional on survival**, because they describe a self-selected
population. A realization that does not carry load is a robustness *result*, not an exception
to swallow.

---

## Part 2 — Deliberate scope cuts

### D2. Metrology calibration is not implemented — and this changes the novelty claim

**Spec:** this is the centerpiece. §1 calls the project "the **first robust topology
optimization framework** grounded in real process metrology data." §3.3 and §5 specify Open3D
colored ICP registration of CMM/laser point clouds, Cp/Cpk process capability statistics,
maximum-likelihood or variogram fitting of the squared-exponential kernel to *measured*
deviation fields, and a marginal transform T(·) calibrated from those deviations. §8 lists
`src/metrology/{ingestion,registration,deviation,process_stats}.py`. §7 makes "use Open3D for
ALL point cloud registration" a hard rule.

**Built:** none of it. `src/metrology/` is deleted. Open3D is not in `requirements.txt`. The
kernel parameters σ = 1.0 and ℓ = 4.0 come from `config.yaml`, not from data.

**Why.** No metrology dataset was ever ingested. The four modules were scaffolding wrapped
around data that does not exist; they were never exercised, and shipping unexercised code whose
docstrings say "calibrated from metrology" is worse than shipping nothing, because a reader
grepping the tree would conclude calibration happened.

**What this costs, stated bluntly.** Combined with D3 — where the η band is set by what the
mesh can resolve rather than by any process — the framework as built is **not
metrology-calibrated**. It is a robustness envelope over a spatially-correlated threshold
perturbation. That is a real contribution, but it is not the one §1 claims, and the prior art
is closer than the spec suggests: **Schevenels, Lazarov & Sigmund, CMAME 2011** already put a
KL-expanded random field on the projection threshold. (masterContext §9 cites this as "Chevens
et al. (2011)", which appears to be a garbled rendering of the same paper — worth correcting in
the bibliography regardless.)

The differentiators that remain and can be defended: the extension to 3D compliance; the
quantified SAA optimality gap and the replication machinery around it; the direct erode/dilate
cost/quality comparison of D10; and — if the experiments support it — the weakest-link mechanism
and the l_c curve. **Two things that earlier drafts listed here are NOT differentiators and have
been struck**: surrogate-free SAA (Schevenels et al. already do exactly this) and the
resolvability analysis (their Figs. 8–9 already give the η → ε/R map). See the prior-art
appendix. **The novelty paragraph has to be rewritten around what survives.**

### D4. Geometry is a hardcoded `beam_3d` box, not CAD

**Spec** (§2, §3.2): the pipeline begins at a STEP file; Gmsh imports, heals, tags physical
surfaces, meshes with local refinement.

**Built:** `mesh_source: "box"`. [`src/meshing/box_source.py`](../src/meshing/box_source.py) is
the single source of truth for the physics: domain 10 × 30 × 10, E = 100, vol_frac 0.08, p = 3,
R = 0.6, one tip load, dimensionless throughout. The STEP path (`importer.py`, `mesher.py`,
`mapper.py`, `fea/fenitop_adapter.py`) is intact and selectable but is not held to the same
verification standard.

**Why.** Every study in the plan — mesh convergence, N convergence, 10-replication gap
estimation, baseline comparison — needs a case with predictable cost and a published reference.
`beam_3d` is FEniTop's own example. A STEP-derived mesh has no refinement family (you cannot
"halve h" on a healed CAD mesh in a controlled way), which alone makes the mesh-convergence
study impossible on that path.

**The correction this forced.** `config.yaml` had been declaring `vol_frac: 0.15`,
`filter_radius: 0.006` (metres), `opt_tol: 1e-3`, `E: 68.9e9` and six SI load cases —
**none of which reached the solver**, because `box_source.py` overrode all of them. Anyone
reconstructing the study from the config file would have reported the wrong number for
essentially every physical parameter, in the wrong unit system. The loader now hard-rejects any
key that is inert on the active path
([`loader.py::_reject_unknown_and_inert`](../src/config/loader.py)), and
[`tests/test_config_loader.py`](../tests/test_config_loader.py) pins that behaviour.

### D8. Probability cloud and CAVE XR are deferred, not deleted

**Spec** (§4.1): 500–5000 perturbed VTK meshes, opacity mapped to sample probability, ParaView
CAVE export with stereo and head tracking.

**Built:** [`src/viz/probability_cloud.py`](../src/viz/probability_cloud.py) survives and is
wired into Stage 6, but `write_ensemble: false` in both shipped configs.
`viz/ensemble_generator.py` and `viz/pareto_plot.py` are deleted.

**Why.** At 2000 samples × 154 k dofs, gathering and writing full-resolution per-sample fields
dominates Stage 6 runtime. It produces a figure, not a result, and it is one config flag away —
so it runs once, at the end, on the final design only. Nothing about it needs to be re-derived.

---

## Part 3 — Corrections of the specification

These are places where the spec asked for something that was wrong, not merely impractical.

### D5. The verification gates are different gates

| Spec gate (§7) | Status | Why |
|---|---|---|
| Cantilever δ = PL³/3EI | **Dropped** | See below |
| FD check, 1e-6 step, **all** elements, rel. err < 1e-5 | **Changed** | See below |
| Open3D `evaluate_registration` fitness | Not run | No metrology (D2) |
| KL sample covariance vs. theoretical kernel | **Kept, hardened** | `gate_kl_correlation` |
| PCE Q² ≥ 0.99 | Not run | No surrogate (D1) |
| Kernel/marginal vs. empirical variogram | Not run | No metrology (D2) |
| MMA KKT residual from TAO diagnostics | **Rewritten** | See below |
| MC 500+ samples | Kept (2000) | D6 |
| — | **Added** | `gate_heaviside_equivalence`, `gate_eta_marginal` |

All four implemented gates live in [`src/validation/gates.py`](../src/validation/gates.py), are
**fatal on failure**, and write `gates.json`. Previously all four existed in some form and
*nothing called them*.

**PL³/3EI dropped.** `src/validation/analytical_cantilver.py` is deleted. Euler–Bernoulli beam
theory is not a verification of a 3D SIMP compliance with a distributed patch load, a Helmholtz
filter and a projection: agreement would be coincidence and disagreement uninformative. It was
replaced with something that actually tests the claim — mesh convergence of the QoI itself
([`scripts/convergence_studies.py`](../scripts/convergence_studies.py) `mesh`), which
**passed**: σ_C/μ_C changed 1.3% between the two finest levels, establishing σ_C as a continuum
property rather than a discretization artifact. That is the check the spec was reaching for.

**FD tolerance 1e-5 → 1e-3, on sampled elements.** Two independent reasons. (a) At the solver's
default KSP tolerance, a 1e-6 finite-difference step measures *solver noise*, not the gradient
— so the gate now tightens KSP to 1e-12 first (`fd_ksp_rtol`), and even then 1e-5 sits below
the floor set by the remaining solve tolerance. (b) With ~154 k design variables an all-element
check is not affordable; the gate samples 32 elements × 8 η-draws. The spec's "all elements"
was written for a much smaller problem.

**KKT residual rewritten — this was a bug.** The spec says to read "KKT residual ‖∇L‖_∞ < tol"
from TAO's convergence diagnostics. TAO does not provide that for this problem, and what the
code was actually reporting was **‖∇f‖_∞** — the norm of the *objective* gradient, which
**cannot vanish** at the optimum of a volume-constrained problem, because there the objective
gradient balances the constraint gradient rather than going to zero. The convergence test could
therefore never be satisfied, and `reason = 0` — TAO for "still iterating" — was being read as
success. [`src/optimization/optimality.py`](../src/optimization/optimality.py) replaces it with
an active-set projected-Lagrangian stationarity residual (components at a bound count only when
the gradient pushes *into* the bound), normalized by ‖∇f‖_∞, reported alongside feasibility and
complementarity. `‖∇f‖` is retained but relabelled `grad_norm` and marked diagnostic-only.

### D10. The λ sweep is common-start, because chained warm starts do not produce a Pareto front

**Spec** (§3.5): "starts from nominal FEniTop SIMP solution as warm start."

**Built:** `lambda_sweep_start: "common"`, `sweep_check_dominance: true`, and
`_assert_non_dominated()` in [`src/mainClean.py`](../src/mainClean.py) aborts the run if the
front is not a front.

**Why.** The implementation had been chaining — each λ warm-started from the *previous λ's*
answer — which makes the sweep one continued descent rather than a set of independent optima.
The observable symptom: λ = 1 beat λ = 0 in **both** μ_C and σ_C. That is impossible for genuine
optima of J = μ + λσ (the λ = 0 point minimizes μ by definition, so nothing can beat it on μ),
and it means the reported "Pareto front" contained a dominated point. Every λ now starts from
the same nominal design, and the run asserts non-dominance rather than trusting it.

**New, and not in the spec at all:**
[`src/optimization/erode_dilate_driver.py`](../src/optimization/erode_dilate_driver.py). An
epigraph reformulation `min t s.t. C(η_k) − t ≤ 0` over {η_lo, 0.5, η_hi} with the volume
constraint on the dilated realization — the Wang/Lazarov/Sigmund three-field baseline.
**3 FEA per iteration versus 512, a 171× cost ratio.** The spec specified no baseline
whatsoever, which would have left the paper with a method and nothing to compare it to. The
cost ratio is itself a headline number.

### D11. Entry point and test layout

**Spec** (§8): `python src/main.py --config configs/config.yaml`;
`tests/{unit,regression,integration}`.

**Built:** `src/main.py` is deleted; the entry point is
[`src/mainClean.py`](../src/mainClean.py). `tests/` is flat — 8 modules, ~1,120 lines — with
`.github/workflows/tests.yml` running the non-MPI subset in the dolfinx image. Configs live in
`src/config/`, not `configs/`.

**Why.** `main.py` had accreted a `recompute:` caching layer that let a run reuse Stage-1/2/3
artifacts from a *different configuration* — the fastest available route to publishing a number
that no single configuration ever produced. `mainClean.py` always runs the full pipeline from
scratch into a run-id directory with a provenance manifest (git SHA, library versions, seeds,
timings, and the **effective** options dict, not the requested one). Nothing is ever read back
from a prior run to skip work. The flat test layout is what 8 modules warrant; the three-way
split was premature structure.

---

## Part 4 — D7 and D9: consequences worth naming separately

### D7. "Two-tier UQ" became "two-tier compute"

The spec's tiering was fidelity: PCE inside the loop, Monte Carlo at the end. D1 removed the
low-fidelity tier. What actually needed tiering turned out to be the **study budget** — the gap
study alone is 26 h, and running convergence, replication and baseline work at production mesh
cost was never viable.

So: [`configStudy.yaml`](../src/config/configStudy.yaml) (h = 0.625, ~1 s/solve) carries
everything that must be converged or replicated; [`config.yaml`](../src/config/config.yaml)
(h = 0.400, ~4 s/solve) is spent **once**, at the end, on parameters the study tier has already
settled. [`src/study_support.py`](../src/study_support.py) ensures both tiers execute the
identical code path, and **R = 0.6 is fixed in absolute units at both** — the filter is part of
the continuum problem being converged to, so scaling it with h would converge to a different
problem at every level and make the study meaningless.

### D9. FEniTop and dolfiny are vendored in-tree and modified — a rule violation, documented

**Spec** (§8): `external/{fenitop,dolfiny,touu_reference}`, pinned. **Rule** (§7): "Modify
FEniTop's internals — extend via its documented η/filter interfaces, or **fork only if strictly
necessary and document why**."

**Built:** `external/` does not exist. FEniTop lives in `src/fenitop/` with the original
Jia/Wang/Zhang authorship headers intact; `src/fenitop/mma.py` is dolfiny's MMA, likewise
retaining its Svanberg references. TOuU appears only as citation comments in
`pce_evaluation.py` and `pce_model.py` — it informed the stochastic-gradient orchestration and
was never executed.

**This is the "fork only if strictly necessary" clause being invoked. The why:**

- **`mma.py`** — four defects, each of which independently invalidates a solve: a pre-loop
  `if ‖∇f‖ ≤ gatol: return` that could terminate at iteration 0; the multiplier vector `self._λ`
  never initialized before use; an off-by-one in the iteration range; and `DIVERGED_MAXITS`
  never set when the iteration budget was exhausted, so a run that simply ran out of iterations
  was indistinguishable from a converged one.
- **`utility.py::solve_fem`** — the zero-initial-guess retry after a KSP failure was gated on
  `_reuse_pc_active`, a branch unreachable in the exact configuration the MC loop uses (warm
  start on, PC reuse off). 2 of 32 solves were failing with `DIVERGED_DTOL` and entering the
  statistics as if converged. Making the retry unconditional took the failure count to 0.
- **`sensitivity.py` / `topopt.py`** — the accumulate-gradients rewrite: per-sample gradient
  rows and 2N world-broadcasts per objective evaluation became S1/S2 running sums with a
  shifted centering. Algebraically identical, 1024 broadcasts → 3 reductions.

None of these is reachable through a documented extension interface. They are bug fixes and a
performance rewrite in upstream code. The honest position for the paper is that the vendored
copies are **forks**, that the diffs are enumerated above, and that the first two changed
results rather than merely speed.

---

## Part 5 — What still matches the specification

Worth stating explicitly, because the divergence list above is long while the formulation
itself is unchanged:

- SIMP `E(ρ) = E₀ρᵖ`, p = 3 — unchanged, FEniTop's implementation.
- Helmholtz PDE filter `−R²∇²ρ̃ + ρ̃ = ρ` followed by smooth Heaviside projection — unchanged.
- Compliance `C = UᵀKU`; robust objective `J = μ[C] + λσ[C]`; mean volume constraint
  `E[V(ρ)] ≤ V_frac` — all exactly as specified in §7.
- Squared-exponential kernel `k(x,x') = σ²exp(−‖x−x'‖²/2ℓ²)` — unchanged.
- KL expansion via OpenTURNS `KarhunenLoeveP1Algorithm` on the FEM nodal basis, truncated at
  ≥ 95% retained variance — unchanged.
- Memoryless isoprobabilistic marginal transform η(x) = T(G(x)) to a bounded Beta — unchanged,
  and still the piece of genuinely custom mathematics the spec identified as such.
- dolfiny MMA as a native PETSc TAO algorithm; no custom MMA math — unchanged (modulo D9's bug
  fixes).
- OpenTURNS for all UQ, Gmsh for all meshing, scikit-learn for regression — unchanged.
- Code standards: Python 3.11+, `from __future__ import annotations`, type hints, vectorized
  NumPy, `logging` not `print`, config-driven — unchanged, and the config-driven clause is now
  enforced by the loader rather than merely stated.

**Added beyond the spec entirely:** [`src/provenance.py`](../src/provenance.py) (run manifests),
[`src/validation/statistics.py`](../src/validation/statistics.py) (BCa bootstrap, paired CRN
comparison, resolvability), [`src/validation/boundary_offset.py`](../src/validation/boundary_offset.py),
[`src/optimization/optimality.py`](../src/optimization/optimality.py),
[`src/optimization/erode_dilate_driver.py`](../src/optimization/erode_dilate_driver.py),
[`scripts/convergence_studies.py`](../scripts/convergence_studies.py),
[`scripts/saa_gap_study.py`](../scripts/saa_gap_study.py),
[`scripts/baseline_comparison.py`](../scripts/baseline_comparison.py), `tests/` + CI.

---

## What this means for the write-up

Three things follow that are the author's call:

1. **The novelty paragraph must be rewritten** (D2). "First metrology-grounded robust TO" is not
   supportable by the code that exists. Neither is "surrogate-free SAA" nor the resolvability
   analysis — Schevenels/Lazarov/Sigmund 2011 has both. What survives is the 3D compliance
   extension, the error control, and the two experiments in the prior-art appendix. Position
   against that paper explicitly rather than as unprecedented.
2. **D3 belongs in the limitations section in full**, including the arithmetic — that a
   calibrated ±0.05 tolerance would require an 11.5 M-element mesh. It is a stronger paper for
   stating the bound than for omitting it.
3. **D9 requires an acknowledgement that FEniTop and dolfiny are forked**, with the four MMA
   defects and the `solve_fem` retry named. Two of those changed results.

---

## Appendix — prior art: Schevenels, Lazarov & Sigmund (CMAME 200:3613–3627, 2011)

masterContext §9 cites this paper as "Chevens et al. (2011)". It is the closest prior work by a
wide margin — it models the *same* uncertainty (a spatially-correlated random Heaviside
projection threshold, memoryless-transformed from an underlying Gaussian field, optimized
against `F = m_f + κσ_f` with a mean-volume constraint) — so the differential matters more than
any other citation in the bibliography.

### Two corrections to earlier drafts of this document

1. **The boundary-offset / resolvability analysis is not novel here.** Their Section 3.1 and
   Figs. 8–9 give the η → ε/R map, including the caveat that it holds only for features wider
   than 2R and that erosion is stronger on thin members where the filtered slope is shallower.
   This project reproduces that analysis on a different filter and uses it as a *design
   constraint on the η band*; it does not originate it.
2. **"Surrogate-free SAA" is not a differentiator against this paper.** Their Eqs. (26)–(27) are
   the same estimators, and their 100 realizations are "used throughout the entire iteration
   history" — a fixed sample set with common random numbers, i.e. SAA. They also note that PCE
   and sparse grids would be more efficient in low stochastic dimension, so avoiding a surrogate
   was their choice too. What is new here is the **error control**, not the method.

### The differential

| Axis | Schevenels et al. 2011 | This project |
|---|---|---|
| Physics | 2D compliant mechanism (l ≠ f) + 2D heat conduction, 200×200 | 3D linear-elastic compliance, ~154 k dofs |
| Filter | Linear hat `w = max(0, R−r)`, R = 8.4, **R/h = 8.4** | Helmholtz PDE, R = 0.6, **R/h = 1.5** |
| Field discretization | **EOLE**, N = 100 nodes, M = 100, no truncation | KL (OpenTURNS P1), truncated at ≥95% variance, n_kl ≈ 20 |
| η marginal | Uniform [0.4, 0.6] | Beta(2,2) on [0.25, 0.75] |
| Sampling | MC, N = 100, fixed across iterations | MC/LHS, N = 512, fixed across iterations |
| Sample-size justification | One 10,000-sample check → "100 is sufficient" | M = 10 replications, Mak/Morton/Wood gap, N-curves, bootstrap CIs |
| Objective | κ = 1 only | λ sweep, common start, dominance assertion |
| Convergence | Fixed 300 iterations, no optimality test | Projected-Lagrangian stationarity + feasibility + complementarity |
| β ceiling | 1 → 32; explicitly refuses 128/256 | 8 → 128 |

**The resolution gap.** Their ε = ±0.91 at R = 8.4 and h = 1.0; this project's ε_max = δη·2R =
0.30 at R = 0.6 and h = 0.4. In *element* units the two are comparable (0.91 h vs 0.75 h) — so
the D3 band widening put this project in the same resolvable regime they were already in.
Relative to the *filter* they are not: ε_max/R = 0.108 vs **0.50**. They bought resolvability by
resolving the filter; this project bought it by enlarging the perturbation. Only one of those is
a tolerance model, and it is not this one. R/h = 8.4 is unavailable in 3D — at R = 0.6 it needs
h = 0.071, i.e. ~8.2 M hexes.

One point in this project's favour, from their own numbers: the hat filter's interface slope is
|∇ρ̃| ≈ 0.92/R against 0.5/R for the Helmholtz filter, so **the Helmholtz filter is ~1.85× more
η-sensitive per unit R**.

### The finding that had to be tested

Their conclusion, for **both** test problems: *"the design obtained assuming uniform
manufacturing errors is equally robust with respect to non-uniform errors, and vice versa."*
Their heat sink goes further — σ_C under spatial variation is *half* the uniform value (0.050 vs
0.098), explained by cancellation: "dilations occur in some regions … erosions in other regions.
The total amount of material does not change as strongly."

If that carries to 3D compliance, the KL expansion and the sample-average loop buy nothing over
a scalar random η. That cancellation argument, though, requires the QoI to be roughly additive
over the domain — true for heat transfer under a distributed source, **false for the compliance
of a slender loaded beam**, which is a series / weakest-link quantity. That is the hypothesis
this project tests, and it is the paper's actual thesis.

Two experiments now exist for it, and neither existed before:

- [`scripts/uniform_eta_baseline.py`](../scripts/uniform_eta_baseline.py) — the uniform-η control
  and Schevenels' 2×2 cross-evaluation, with a decision rule fixed in advance against the gap
  study's measured run-to-run noise floor.
- [`scripts/correlation_length_study.py`](../scripts/correlation_length_study.py) — σ_C/μ_C
  versus l_c with the uniform limit marked. They drew a general conclusion from a single l_c;
  this measures the curve.

Both reuse the production code path exactly. The uniform arm is the *same* SAA driver, FEA,
projection and marginal transform, handed a degenerate one-mode expansion with a constant
eigenfunction ([`kl_expansion.build_uniform_eta_kl`](../src/random_fields/kl_expansion.py)).
Because the projection standardizes G(x) by its pointwise std before the marginal transform, the
uniform arm draws η from the *identical* Beta marginal — so the only difference between the arms
is spatial structure, not error magnitude.
[`tests/test_uniform_eta_baseline.py`](../tests/test_uniform_eta_baseline.py) pins that, and the
l_c sweep provides an independent check: at l_c = 32 (beyond the 30-long domain) the real KL
collapses to n_kl = 1 and reproduces the degenerate expansion's cv to 11 significant figures.

### Consequences for the write-up

1. The novelty paragraph cannot claim the KL-threshold-field model, the robust objective, or the
   SAA method. It can claim the 3D compliance extension, the error control, and — if the
   experiments support it — the weakest-link mechanism and the l_c curve.
2. **β = 128 must be defended, not merely reported.** They refuse β ≥ 128 as producing
   single-element features "impossible to produce", at an R/h 5.6× better resolved than here.
   This needs `measure_non_discreteness` per design plus a minimum-feature-size check against 2R;
   no M_nd value currently appears in any output file.
3. **KL vs EOLE.** They cite Li & Der Kiureghian that EOLE is more efficient than KL for
   squared-exponential kernels, and retained all 100 modes where this project truncates to ~20.
   Report retained variance and n_kl per run so the truncation is visibly not doing hidden work.

---

*Traceability: every claim above resolves to a file in this tree or to a committed study
output. Mesh-convergence, N-convergence and SAA gap results are under `output/studies/`; gate
results are in each run's `gates.json`; the forked-upstream diffs are `git diff` against the
vendored baselines at commit `d345127`.*
