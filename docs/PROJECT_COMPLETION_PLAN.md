# From here to a finished paper — concrete steps

Written 2026-07-29 14:30. Deadline ~2026-08-05 (7 days). Every command runs from
the repo root on the host; `sudo docker` is required, and `cd /shared` inside the
container. Time estimates are measured, not guessed.

**Status key:** ⏳ running · ⬜ you do it · 🤖 automatic · 🔀 decision point

---

## STEP 1 — Tonight (🤖 automatic, no action)

Four jobs are chained and self-triggering. You get a push per job.

| job | ranks | trigger | ~cost |
|---|---|---|---|
| ⏳ SAA gap study | 64 | already running | ends 19:30–22:30 |
| ⏳ move-limit probe | 32 | after Phase 0 | ~1 h |
| 🤖 KL truncation check (95 % vs 99 %) | 32 | after probe | ~20 min |
| 🤖 l_c re-run with cv confidence intervals | 32 | after truncation | ~1 h |
| 🤖 **Phase 1 chain** | 128 | after gap + all 32-rank jobs idle | ~7–8 h |

Phase 1 = E1 field arm → E1 evaluate ×2 → E1 report → erode/dilate baseline →
**study-mesh pipeline (the production rehearsal)**.

Expect everything finished **04:00–06:00** tomorrow.

Nothing to do. If a push says CRASH, see §4 of `AUTONOMOUS_PIPELINE_HANDOFF.md`.

---

## STEP 2 — Tomorrow morning, ~30 min (⬜ read results, 🔀 two decisions)

```bash
./scripts/watch_progress.sh
for k in gap probe lc uniform baseline pipeline; do
  echo "--- $k ---"; python3 scripts/summarize_result.py $k
done
```

### ~~🔀 Decision A — move-limit continuation~~ ✅ RESOLVED 2026-07-29: **FAIL**

The probe ran and failed cleanly. `stat_rel` 0.1413 vs baseline 0.1244 (13.6 %
worse), objective 8 % worse. Decisive evidence: `dx / move_limit = 1.000` at
every β stage even after shrinking the limit 4×, so the iterate never leaves the
trust-region boundary and the move limit is not the cause. The plateau is
**β = 128 projection stiffness** (responsive band ~19/β = 0.15 wide).

**Action: none.** `move_reduction` already defaults to 1.0. Do not touch
`box_source.py`. Report the achieved `stat_rel` (~0.04–0.06 at N = 512) as a
measured quantity in the results, and put the mechanism in limitations.

**Never** loosen `robust_opt_tol` to make `converged: true` appear.

### 🔀 Decision B — does the rehearsal clear production?

```bash
python3 scripts/summarize_result.py pipeline
ls output/stage4_surrogate/*/gates.json | tail -1 | xargs cat | head -30
```

Required: all four gates pass, sweep non-dominated, Stage 6 wrote
`validation_summary.json`. If yes, the production config is validated end to end
— same code path, same five β stages, same gates — and is safe to launch
elsewhere. If not, **fix before porting**; a failure 3 h into a 67 h run is
unrecoverable inside the deadline.

---

## STEP 3 — Tomorrow, port and launch production (⬜)

Follow `docs/PRODUCTION_LAUNCH_CHECKLIST.md`. Summary:

```bash
# on the new machine, after copying repo + container image
source mpi-env.sh
python -m pytest tests/ -q                 # expect 108 passed
mpirun -n <ALL CORES> python src/mainClean.py src/config/config.yaml
nohup scripts/job_watch.sh > watch_log.txt 2>&1 &   # notifications there too
```

| ranks | 5-λ sweep |
|---|---|
| 128 | 67 h |
| 256 | 33 h |
| **512** | **17 h** |

Constraint: `world_size % 8 == 0`. Check RAM — each group holds its own copy of
the 154 k-dof mesh.

Watch the first λ's first β stage (gates + first `[SAA] outer_iter=` lines), then
leave it.

---

## STEP 4 — While production runs, use pyrite (⬜, ~8 h, runs in parallel)

pyrite's 128 cores are free once production moves to the other machine. Two
things still missing, both study-tier:

```bash
# (a) design-level conservatism: optimize at X, score at Y, all pairs (~8 h)
mpirun -n 128 python scripts/correlation_length_study.py reoptimize 1 4 uniform \
    src/config/configStudy.yaml

# (b) the beta=128 defence (~15 min)
mpirun -n 32 python scripts/measure_feature_size.py --all src/config/configStudy.yaml
```

**(a) is what upgrades the abstract's second clause.** The fixed sweep bounds the
*response*; this bounds the *design*. Its `uniform_model_is_conservative` field is
the direct test. Without it, soften "the inexpensive scalar model is
conservative" to a statement about response variance only.

**(b)** answers Schevenels' objection to β = 128 with a number. M_nd = 0.23 % is
not an answer — a single-element strut is also near-binary.

---

## STEP 5 — Production completes (⬜, ~1 h)

```bash
python3 scripts/summarize_result.py pipeline     # 5-point Pareto + convergence
```

Check: `converged` per λ (whatever Decision A settled), sweep non-dominated
(mainClean asserts it), all four gates in `gates.json`, Stage 6 CIs present.

Copy the production `output/` tree back to pyrite (or run the remaining figure
steps on the new machine — they need only matplotlib).

---

## STEP 6 — Figures and tables (⬜, ~2 h)

```bash
sudo docker exec fenitop_container bash -c "cd /shared && python viz/plot_research_figures.py"
sudo docker exec fenitop_container bash -c "cd /shared && python viz/plot_comparison.py"
python3 viz/paths.py        # confirm every artifact resolves to the NEWEST run
```

| figure | source | status |
|---|---|---|
| **A** cv vs l_c, uniform limit marked | `lc_sweep` | ✅ built; gains CIs after tonight |
| **B** mesh convergence, cv vs μ_C vs σ_C | `mesh` | ✅ built |
| **C** in-sample vs out-of-sample σ scatter | `gap` | renders once gap lands |
| **D** Pareto front with CIs | `pareto` | `viz/plot_comparison.py` fig4 |
| **E** CDF overlay nominal vs robust | `validation` | fig1 |
| **F** design renders (nominal / robust / eroded) | ParaView | `viz/make_paraview_state.py` |

| table | source |
|---|---|
| verification gates (4, all fatal) | `gates.json` |
| cost: SAA vs erode/dilate FEA solves | `baseline` |
| cross-evaluation grid + worst case per design | `lc_reopt` |
| erosion survival / implied thickness | `feature_size.json` |
| SAA gap: optimism, noise floor, outlier rate | `gap` |

---

## STEP 7 — Write the paper (⬜, 3 days)

Draft abstract exists in the session. Positioning material is already written in
`docs/architecture_divergence.md` (prior-art appendix + the two struck claims).

### Outline, each section mapped to evidence

1. **Introduction** — robust TO for manufacturing error; Sigmund 2009 and Wang
   et al. 2011 (worst-case, uniform); Schevenels et al. 2011 (probabilistic,
   spatially correlated, 2D, one l_c). The open question: *when* does spatial
   correlation matter, and is it worth its cost?

2. **Formulation** — SIMP + Helmholtz filter + tanh projection; η as a
   KL-expanded, memoryless-transformed Beta field; J = μ_C + λσ_C; E[V] ≤ V*.
   State the η-band resolvability argument (offset std 0.41 elements) and that
   this makes it a **robustness envelope, not a calibrated process tolerance**.

3. **Solution method** — surrogate-free SAA, N = 512 fixed sample set with
   common random numbers, exact sample-average gradients, β continuation 8→128,
   MMA via PETSc TAO, sample-parallel over MPI sub-communicators.

4. **Verification** — four fatal gates; mesh convergence (**figB**); N
   convergence; FD gradient check. Make the point that σ_C/μ_C is the converged
   statistic and absolute compliance is not.

5. **Results** — **figA** and the l_c table; the monotonic rise and the uniform
   bound; the cross-evaluation grid; robust vs nominal (cv 2.00 → 0.51, **figE**,
   **figD**); erode/dilate cost table; **figF** renders.

6. **Sampling error** — **figC**, the gap study: σ optimism, robust noise floor,
   the ~1-in-7 heavy tail invisible in-sample. This is the section that indicts
   the single-verification-run practice.

7. **Limitations** — no metrology calibration (band is resolvability-driven);
   ε/R = 0.50 vs 0.108 in prior work, i.e. a more aggressive perturbation
   relative to feature size, forced by 3D affordability (R/h = 1.5 vs 8.4);
   first-order optimality achieved only to `stat_rel` ≈ 0.05; KL truncation on
   the FEM mesh caps refinement (21 GB dense eigensolve at production) where an
   EOLE-style auxiliary grid would not; FEniTop and dolfiny are **forked**, with
   the four MMA defects and the `solve_fem` retry named.

8. **Conclusion** — the practical rule: use the scalar-η model; it is the
   conservative bound and ~8× cheaper.

### Framing to hold to

The contribution is **not** "spatial correlation matters." It is:

> The spatially uniform threshold bounds the response variance from above;
> spatial correlation can only reduce it. The cheap scalar model is therefore
> conservative, which explains Schevenels et al.'s equivalence finding
> mechanistically and extends it from a single correlation length to the whole
> curve — with the sampling error quantified, which prior work did not do.

---

## Timeline and slack

| day | work |
|---|---|
| Jul 29 (tonight) | 🤖 everything queued |
| Jul 30 | STEP 2 decisions, STEP 3 port + launch, STEP 4 starts on pyrite |
| Jul 31 – Aug 1 | production runs (17 h at 512 ranks, 67 h at 128) |
| Aug 1 | STEP 5 + STEP 6 |
| Aug 2 – 5 | STEP 7 writing |

**At ≥256 cores this leaves 4 days to write.** At 128 it leaves ~3. Securing
cores is still the highest-leverage action.

**If the new machine falls through:** the study tier alone supports the paper.
Mesh convergence already shows σ_C/μ_C differs 0.24 % between study and
production meshes. Production is the headline deliverable, not the evidence.
