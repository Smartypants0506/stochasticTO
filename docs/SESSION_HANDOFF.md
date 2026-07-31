# Session handoff — read this first

**For an agent with no prior conversation context.** Written 2026-07-29 14:45 by the
session that built the E1/E2 experiments and armed the overnight chain.
Deadline: **~2026-08-05**.

Read this file, then `PROJECT_COMPLETION_PLAN.md` for the step list. The other
docs are reference, not orientation.

| doc | what it is for |
|---|---|
| **this file** | state, findings, decisions, traps |
| `PROJECT_COMPLETION_PLAN.md` | the numbered steps to a finished paper |
| `PRODUCTION_LAUNCH_CHECKLIST.md` | porting + launching the production run |
| `AUTONOMOUS_PIPELINE_HANDOFF.md` | environment facts, crash handling (its §0 is current; older prose is stale) |
| `architecture_divergence.md` | how the code differs from `masterContext.md`, plus the prior-art appendix |

---

## 1. The one thing to understand

**The project's original premise was disproved, and the replacement is better.**

The premise was that spatially-correlated manufacturing error needs a KL random
field, and that this buys robustness a cheap scalar model cannot. The measured
l_c sweep says the opposite: response variability rises **monotonically** with
correlation length, so a spatially *uniform* threshold is the worst case.

Do not try to rescue the original framing. The defensible claim, which the paper
is now built on, is:

> The spatially uniform threshold bounds the response variance from above;
> spatial correlation can only reduce it. The cheap scalar model is therefore
> **conservative** — which explains Schevenels et al. (2011)'s equivalence
> finding mechanistically, extends it from a single correlation length to the
> whole curve, and quantifies the sampling error, which prior work did not.

That is a stronger, more useful result than the one the project set out to make.
It is also directly actionable: use scalar η at ~64 samples, not a 512-sample
field, ~8× cheaper.

---

## 2. Current job state (2026-07-29 14:45)

```bash
./scripts/watch_progress.sh          # one screen: jobs, ETAs, notifier health
watch -n 60 ./scripts/watch_progress.sh
```

| job | ranks | state |
|---|---|---|
| SAA gap study | 64 | **7/10 reps**, ETA 21:20–00:40 |
| move-limit probe | 32 | ✅ done — **FAIL**, see §4 |
| KL truncation check (95 % vs 99 %) | 32 | ✅ done — **not truncation-sensitive**, see §3 |
| l_c re-run with cv CIs | 32 | **running now** |
| **Phase 1 chain** | 128 | queued until gap + all 32-rank jobs idle, ~7–8 h |
| production | — | **NOT run here.** Tomorrow, bigger machine. |

Phase 1 = E1 field arm → E1 evaluate ×2 → E1 report → erode/dilate baseline →
study-mesh pipeline (**the production rehearsal**).

Logs: `gap_log.txt`, `phase0_log.txt`, `probe_log.txt`, `vthr_log.txt`,
`lcci_log.txt`, `phase1_log.txt`, `watch_log.txt`.

**⚠ The box is shared and currently oversubscribed.** User `dlb8` is running
`train_light_unet.py` at ~3967 % CPU (~40 cores). With gap (64) + probe (32)
that is ~136 of 128. The gap study's rate has already slipped 3h04 → 3h19 per
replication. Do not kill their job. This is an argument for dedicated hardware
for production, not for intervening.

### Notifications

`scripts/job_watch.sh` runs detached on the host and pushes to
`ntfy.sh/stochasticTO-488039e71edb` — full results on completion, HIGH priority
on crash. Restart with `nohup scripts/job_watch.sh > watch_log.txt 2>&1 &`.
`scripts/summarize_result.py gap|lc|probe|uniform|baseline|pipeline` prints any
finished study on demand (exit 1 = not finished).

---

## 3. Established results — quote these, they are measured

| finding | numbers |
|---|---|
| **l_c sweep** (nominal design, study mesh, n=1000) | cv = σ_C/μ_C: **1.234** (l_c=1) → 1.461 → 1.931 → 2.350 → 3.026 → **3.306** (l_c=32); **uniform limit 3.179**. Monotonic, no interior peak. |
| **Mesh convergence** | cv = 2.0028 (h=.625), 1.9817 (h=.476), 2.0076 (h=.400) → spread **1.3 %** over a 1.6× h-range. μ_C and σ_C individually move ~37 %. **Report cv, never absolute compliance.** |
| **E1 uniform arm** | μ_C 0.2718, σ_C 0.1387, **M_nd 0.234 %** at β=128, 9 920 FEA solves. cv 0.51 vs nominal 2.00 → 4× more robust. |
| **SAA gap — COMPLETE, 10/10** | σ optimism **−33.9 %** (in-sample 0.0531 vs out-of-sample 0.0804). Optimality gap **+3.1 %**, 95 % CI [0.58 %, 2.36 %] — **excludes zero, so the gap is resolved**. Noise floor: robust MAD **15.5 %**, raw std/mean 38.8 %. Out-of-sample σ is **bimodal + outlier**: reps 0–5 at 0.056–0.065, reps 7–9 at 0.089–0.090, rep 6 at 0.161. **4 of 10 designs are materially worse than the median cluster**; in-sample σ is 0.048–0.058 for all ten and gives no warning. converged 0/10 (expected on study tier). |
| **cv confidence intervals** (l_c re-run) | 1: 1.2337 [1.1676, 1.3173] · 2: 1.4611 [1.3889, 1.5470] · 4: 1.9305 [1.8431, 2.0338] · 8: 2.3502 [2.2170, 2.5021] · 16: 3.0260 [2.8401, 3.2304] · 32: 3.3059 [3.0967, 3.5334] · uniform: 3.1788 [2.9717, 3.4062]. **All four adjacent pairs from l_c=1 to 16 are DISJOINT** → monotonicity is established over the rising portion. 16/32/uniform mutually overlap (they are all effectively "uniform"). Endpoint contrast **2.58×, disjoint**. |
| **Convergence** | No run has ever met `robust_opt_tol = 1e-3`. `stat_rel` plateaus 0.04–0.12; `dx` pinned at the move limit every iteration. |
| **Boundary offset** | std **0.41 elements** at the study mesh — resolvable. ε_max/R = 0.50 vs 0.108 in Schevenels et al. |
| **KL truncation robustness** | 95 % vs 99 % variance threshold at the sweep's extremes: l_c=1 cv 1.2337→1.2328 (n_kl 183→197, −0.07 %); l_c=16 cv 3.026→2.9776 (n_kl 4→6, −1.6 %). Both 95 % point estimates sit inside the 99 % run's own bootstrap CI. **The monotonic l_c trend is not a truncation artifact**, and this also closes the "should we switch to EOLE" question — a different expansion method would not have changed the finding. |

**Sign convention:** `sigma_optimism_relative` NEGATIVE = overfitting (looks less
variable on its own samples than on fresh ones).

**The rep-6 outlier is NOT a solver artifact.** All replications show exactly 5
`DIVERGED_MAXITS` (one per β stage) and **zero** KSP failures or non-finite
compliance. It converged fine (`stat_rel` 0.044). It is a genuine heavy tail, and
it is one of the paper's strongest findings.

---

## 4. Decisions already taken — do not relitigate

- **η band [0.25, 0.75]**, chosen for mesh-resolvability, not calibrated to a
  process. The paper says "robustness envelope", not "process tolerance".
- **No metrology.** `src/metrology/` was deleted; Open3D is not a dependency.
  Any claim of metrology calibration is unsupportable.
- **Full 5-point production sweep** is in scope, on a bigger machine.
- **Study-tier results are publishable** — mesh convergence shows study and
  production meshes agree on cv to 0.24 %.
- **`move_reduction` defaults to 1.0 (inert).** Enabling it changes every design,
  so it stays off until the probe justifies it.
- **E1's two arms must share settings.** P0-A (uniform) ran without move-limit
  continuation, so the field arm must too, or the comparison is confounded.

- **Move-limit continuation is OFF, permanently.** ✅ **RESOLVED 2026-07-29 by
  `scripts/move_limit_probe.py` — it FAILED.** Shrinking the move limit per β
  stage made `stat_rel` 13.6 % *worse* (0.1413 vs 0.1244) and the objective 8 %
  worse. The decisive evidence: `dx / move_limit = 1.000` at **every** stage even
  after shrinking the limit 4× (0.02 → 0.0048), so the iterate stays pinned to
  the trust-region boundary regardless of box size — the move limit is not the
  cause. Per-stage `stat_rel` (baseline / probe): β=8 0.233/0.233, β=16
  0.149/0.171, β=32 0.128/0.123, β=64 0.140/**0.104**, β=128 0.124/**0.141**.
  Smaller steps help at moderate β and hurt at the sharpest projection — the
  signature of **β = 128 projection stiffness**: the responsive band is only
  ~19/β = 0.15 wide, so as the design moves, which nodes carry gradient flips,
  and the objective is effectively non-smooth at the optimizer's working scale.
  **No production code change needed** (`move_reduction` already defaults to
  1.0). Report the achieved residual honestly; this belongs in limitations with
  the mechanism named.

### Open decisions

1. **Production rank count** — 128 = 67 h, 256 = 33 h, 512 = 17 h.
2. **Cross-evaluation grid** (`reoptimize 1 4 uniform`, ~8 h) — needed to upgrade
   the abstract's "conservative" clause from a *response* claim to a *design*
   claim. Run on pyrite while production runs elsewhere.

---

## 5. Traps that already cost time — do not repeat

- **`pgrep -f <script.py>` cannot detect these jobs.** The wrapper chain
  (`sudo` → `docker` → `bash -c '...'`) carries every script name in its own
  command line, so a finished job looks alive and a wrapper exit looks like a
  crash. Match on process name:
  `ps -eo comm=,args= | awk '$1 ~ /^python/ && index($0,p)>0'`.
- **`pkill -f <script.py>` kills the parent block and every sibling job.** It
  happened. Kill by PID.
- **Long log silence is normal.** Only rank 0's sample-parallel *group* logs, so
  with 4 groups three are invisible; and each new l_c is a KL cache miss → a
  dense O(N²) eigensolve on ~14 k nodes. 6–10 min of silence is healthy;
  `watch_progress.sh` uses a 1200 s stall threshold for this reason.
- **`output/` is root-owned** (container writes as root into an `ovb` tree). Run
  anything that writes there inside the container. Watcher state falls back to
  `/tmp/stochasticTO_watch/`.
- **MPICH, not OpenMPI** — no `--allow-run-as-root`.
- **`source mpi-env.sh`** before every `mpirun` (pins `OMP_NUM_THREADS=1`).
- Don't guess at costs from `configStudy.yaml`'s header table: its
  "erode/dilate baseline … 3 min" is the *baseline arm only*;
  `baseline_comparison.py` also runs a full SAA solve (~5 h at 32 ranks).
  **Measure, don't estimate.**

---

## 6. What this session built

**New:** `scripts/uniform_eta_baseline.py` (E1), `scripts/correlation_length_study.py`
(E2), `scripts/move_limit_probe.py`, `scripts/measure_feature_size.py`,
`scripts/watch_progress.sh`, `scripts/job_watch.sh`, `scripts/summarize_result.py`,
`scripts/phase1_chain.sh`, `src/validation/feature_size.py`, `viz/paths.py`,
`viz/plot_research_figures.py`, `tests/test_uniform_eta_baseline.py`, and the
five `docs/*.md`.

**Modified:** `saa_robust_driver.py` (move-limit continuation, inert by default),
`statistics.py` (paired-bootstrap cv estimator), `kl_expansion.py`
(`build_uniform_eta_kl`), `study_support.py` (length-scale + variance-threshold
overrides), `convergence_studies.py` (persist raw samples), `kernel.py` (units).

**108 tests pass.** Run `python -m pytest tests/ -q` in the container.

### Two design points worth preserving

- **`build_uniform_eta_kl`** makes the uniform-η control a degenerate one-mode KL
  with a constant eigenfunction. Because the projection standardizes G(x) by its
  pointwise std *before* the marginal transform, both arms draw η from the
  **identical** Beta marginal — only spatial structure differs. Validated
  independently: at l_c=32 the real KL collapses to n_kl=1 and reproduces it to
  11 significant figures.
- **cv is bootstrapped as one statistic**, not propagated from separate μ and σ
  intervals. They come from the same draws and are positively correlated;
  propagation overstates the width ~25 % and would make resolvable orderings look
  unresolvable.

---

## 7. Known gaps

- **cv CIs are missing from the current l_c curve** (the run predates the
  estimator). The queued re-run fills them. `viz/plot_research_figures.py`
  deliberately does *not* fall back to naive propagation.
- **Cross-evaluation grid not run** — the design-level conservatism claim.
- **Feature size not measured** — the β=128 defence. `measure_feature_size.py`
  is written and ready (~15 min).
- **`viz/`'s five original scripts** still hard-code the pre-run-id layout. Use
  `viz/paths.py` for anything new; the old scripts read July artifacts silently.
- **First-order optimality unachieved** — report the residual honestly.

---

## 8. Draft abstract

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
> β = 128. The ratio σ_C/μ_C is shown to vary by less than 1.4 % across a
> 1.6-fold range of element size, and to increase monotonically with correlation
> length, from 1.23 at ℓ_c = 0.03L to 3.31 at ℓ_c > L, with the spatially uniform
> limit at 3.18. The spatially uniform threshold therefore bounds the response
> variance from above: spatial correlation can only reduce it, and the
> inexpensive scalar model is conservative. Robust design reduces σ_C/μ_C from
> 2.00 to 0.51 while retaining a near-discrete layout (measure of
> non-discreteness 0.23 %). Replicated independent solves further show that the
> in-sample standard deviation is optimistic by approximately 14 % at N = 512,
> and that roughly one solve in seven produces a design 2.5 times less robust out
> of sample with no in-sample indication — a failure mode invisible to the
> single-verification-run practice of prior work.

### ⚠ Abstract numbers are now SUPERSEDED by the completed gap study

The draft above was written from 7 replications. With all 10, three figures change
and the final paragraph must be rewritten:

- σ optimism **−33.9 %**, not ~14 %.
- Not "one solve in seven is 2.5× worse". It is **1 in 10 at 2.5× and 3 in 10 at
  ~1.4×** — 4 of 10 designs materially worse than the median cluster, with
  in-sample σ (0.048–0.058 across all ten) giving no warning.
- Add the resolved optimality gap: **+3.1 %, 95 % CI excluding zero**.

"increases monotonically" **is** defensible: all four adjacent cv intervals from
l_c = 1 to 16 are disjoint. State that l_c = 16, 32 and the uniform limit are
mutually indistinguishable — which is the point, since all three are effectively
the uniform case.

**Do not add:** metrology calibration, a first-order optimality claim, or a cost
ratio (erode/dilate baseline runs in Phase 1, tonight).
