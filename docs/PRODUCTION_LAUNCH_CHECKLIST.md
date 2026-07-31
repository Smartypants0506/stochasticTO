# Production run — port and launch checklist

**For the bigger machine, 2026-07-30.** Production was deliberately NOT run on `pyrite`
(128 cores): the 5-point sweep is 67 h there against a 7-day deadline, with no slack for a
failed run.

---

## 1. What you are launching

```bash
cd <repo root>
source mpi-env.sh
mpirun -n <ALL CORES> python src/mainClean.py src/config/config.yaml
```

`config.yaml` is the production tier: mesh h=0.400 (R/h=1.5), `lambda_sweep: [0, 0.5, 1, 2, 4]`,
`max_iter: 400`, `saa_n_samples: 512`, β continuation 8→128, `mc_validation.n_samples: 2000`.

**Cost scales almost linearly with cores**, because sample parallelism is embarrassingly parallel
across groups. The only constraint is `world_size % 8 == 0`
(`sample_parallel_ranks_per_group: 8`):

| ranks | groups | solves/group/iter | 5-λ sweep |
|---|---|---|---|
| 128 | 16 | 32 | 67 h |
| 256 | 32 | 16 | 33 h |
| 512 | 64 | 8 | **17 h** |
| 1024 | 128 | 4 | ~9 h |

Memory: each group holds its own copy of the production mesh (~154 k dofs). At 64 groups that is
64 mesh copies — check RAM on the new box before choosing the largest rank count.

---

## 2. Pre-flight — do not skip

```bash
python -m pytest tests/ -q                     # expect 105 passed
./scripts/watch_progress.sh                    # nothing else should be running
python3 scripts/summarize_result.py pipeline   # the study-mesh REHEARSAL
```

The rehearsal is the thing that de-risks this. It runs the identical code path — same five β
stages, same four gates, same Stage 6 — at study resolution. **If it completed with gates passing
and a non-dominated sweep, the production config is validated end to end.** Confirm:

- `gates.json` — all four gates pass (they are fatal, so a run that finished has passed them)
- `pareto_results.json` — sweep is non-dominated (`mainClean.py` asserts this and aborts otherwise)
- Stage 6 wrote `validation_summary.json`

If the rehearsal did not run or failed, **run it on the new machine first** (~1–2 h there):

```bash
mpirun -n <cores> python src/mainClean.py src/config/configStudy.yaml
```

---

## 3. The one decision to make before launching

**Move-limit continuation.** No run in this project has ever met `robust_opt_tol = 1e-3`: `dx`
sits at the move limit on every iteration and `stat_rel` plateaus at 0.04–0.12, oscillating.
More iterations do not fix it — production's 80 iterations/stage would just oscillate longer and
report `converged: false` at all five λ, after 67 h.

`scripts/move_limit_probe.py` tested a fix. Read its verdict:

```bash
python3 scripts/summarize_result.py probe
```

| probe verdict | what to do |
|---|---|
| **PASS** | Enable it: set `_MOVE_REDUCTION = 0.7` in `src/meshing/box_source.py` and add `"move_reduction": _MOVE_REDUCTION` to the opt dict. Production then reports genuine first-order convergence. |
| **MIXED** | Inspect the per-stage table. Stationarity improved but objective got worse usually means the design stopped moving early — not a real convergence. |
| **FAIL** | Leave it off. The plateau is β=128 projection stiffness, not the move limit. Report the achieved `stat_rel` honestly as a measured quantity — still ahead of Schevenels et al., who report no optimality measure at all. |

Whatever you choose, it must be stated in the paper. Do not quietly loosen `robust_opt_tol` to
make `converged: true` appear — that is tolerance-fudging and this project has spent its whole
remediation removing exactly that.

---

## 4. Porting to the new machine

1. **Repo**: copy `/raid/ovb/stochasticTO` (or `git clone` the `test` branch — note `output/` has
   many modified tracked VTU artifacts; they are not needed).
2. **Container**: image `fenitop-image`, based on `dolfinx/dolfinx:stable`. Rebuild from
   `Dockerfile`, or `docker save`/`docker load` the existing image.
3. **Mount**: repo root → `/shared` inside the container. Every script hardcodes a relative
   `Path("output")`, so **always `cd /shared` before launching** or output scatters.
4. **`source mpi-env.sh`** — pins `OMP_NUM_THREADS=1` so ranks do not oversubscribe.
5. Re-run the pre-flight in §2 on the new box before the real launch.

The KL cache at `output/cache/kl_expansion/` is keyed on node coordinates. Copying it warms the
production mesh's eigensolve; omitting it just costs one eigensolve.

---

## 5. Notifications on the new machine

```bash
nohup scripts/job_watch.sh > watch_log.txt 2>&1 &
```

Pushes to the same ntfy topic. It watches `pipeline` (i.e. `src/mainClean.py`) among others, so
the production run will report itself on completion — and fire a HIGH-priority alert if it dies.

`scripts/job_watch.sh` runs on the **host**, not in the container (it needs outbound curl). Its
liveness check matches on process name, not `pgrep -f <script>` — see the handoff doc §0 for why
that distinction matters.

---

## 6. While production runs (~17–67 h)

Watch the first λ only, then leave it:

- first β stage writes `gates.json` — if any gate fails the run aborts immediately, by design
- first `[SAA] outer_iter=` lines confirm the sample-parallel groups are live
- `constraint check: vol_frac=0.08 E[V]=...` confirms the volume constraint is being enforced

After that it is hours of identical output. The notifier will tell you when it ends.

---

## 7. What production does NOT need to establish

The mesh-convergence study already passed: σ_C/μ_C is **2.0028** at the study mesh vs **2.0076**
at production — a 0.24% difference against a ±14% bootstrap CI. The dimensionless robustness
measure is mesh-converged.

So production is the headline deliverable — a 5-point Pareto front at full resolution — but it is
**not** load-bearing for the scientific claim. If the new machine falls through, the study-tier
results still support the paper.

One caveat that must appear in the write-up either way: μ_C and σ_C *individually* move ~37% with
h, because the resolved traction area varies 1.4× as facet selection quantizes the load patch to
O(h) (`loaded_area_spread_across_levels` in `mesh_convergence.json`). **Report σ_C/μ_C, not
absolute compliance.**
