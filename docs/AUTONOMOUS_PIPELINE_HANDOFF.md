# Autonomous pipeline babysitter — handoff playbook

**Read this whole file before doing anything.** It is written for an agent with no prior
conversation context. Written 2026-07-28; **substantially updated 2026-07-29** — §§2, 6, 10, 11
changed. Where this header and the older prose disagree, this header wins.

---

## 0. UPDATE 2026-07-29 13:15 — read this first

### Tooling now exists; use it instead of ad-hoc commands

| Command | What it does |
|---|---|
| `./scripts/watch_progress.sh` | one-screen status: every job, ETA, notifier health |
| `watch -n 60 ./scripts/watch_progress.sh` | same, live |
| `scripts/job_watch.sh` | **daemon**: pushes a full result to ntfy the moment a job ends, HIGH-priority alert if one dies. `nohup scripts/job_watch.sh > watch_log.txt 2>&1 &` |
| `python3 scripts/summarize_result.py gap\|lc\|probe\|uniform\|baseline\|pipeline` | phone-readable summary of a finished study; exit 1 = not finished |

State markers live in `/tmp/stochasticTO_watch/` (NOT `output/studies/_watch` — that tree is
root-owned and unwritable, so the script falls back automatically).

> **Do not detect job liveness with `pgrep -f <script.py>`.** The wrapper chain
> (`sudo` → `docker` → `bash -c '...'`) carries every script name in its own command line, so a
> finished job still looks alive and a wrapper exit looks like a crash. Match on the process
> name instead — `ps -eo comm=,args= | awk '$1 ~ /^python/ && index($0,p)>0'` — which is what
> `job_watch.sh::alive()` does. The same trap applies to `pkill -f`: it will kill the parent
> block and every sibling job with it.

### Production decision has CHANGED

§6 Step E below says "do not launch production". **That is superseded.** The user has approved
the **full 5-point production sweep** (`lambda_sweep: [0, 0.5, 1, 2, 4]`, `max_iter: 400`,
N=512). Still launch it only after the Phase-1 study-mesh rehearsal passes.

Cost: **67 h at 128 ranks / 33 h at 256 / 17 h at 512.** The box has **128 cores and the gap
study only uses 64** — always check `nproc` before choosing `-n`.

### Two open findings a remote session must know

1. **No run has ever met `robust_opt_tol = 1e-3`.** `dx` equals the move limit on every
   iteration of every stage; `stat_rel` plateaus at 0.04–0.12 and oscillates. Move-limit
   continuation is implemented in `saa_robust_driver.py` (`opt["move_reduction"]`,
   **default 1.0 = inert**) and is being probed by `scripts/move_limit_probe.py`.
2. **The gap study's replication set is heavy-tailed.** Six of seven out-of-sample σ cluster
   near 0.062; one sits at 0.161. Raw std/mean = 48.9%, robust MAD-based = 7.2%. Use the robust
   figure as the noise floor and report the outlier rate separately — the outlier converged
   *fine* (stat_rel 0.044), so it is a genuine tail, not a solver artifact.

### Current job state (2026-07-29 13:15)

| Job | Ranks | State |
|---|---|---|
| SAA gap study | 64 | 7/10 reps, ETA 19:00–22:00 |
| l_c sweep (E2 fixed) | 32 | running, 1/7 levels |
| move-limit probe | 32 | queued, auto-starts on `PHASE0 DONE` |
| E1 uniform arm | — | ✅ done: μ_C 0.2718, σ_C 0.1387, M_nd **0.234%** |

---

Your job: wait for the running study to finish, read its results, fix anything that blocks
the next step, launch the next step, and report at each hand-off. **Stop before the
production run.**

---

## 1. Environment — verified facts, do not re-derive

| Fact | Value |
|---|---|
| Host | `pyrite.cam.nist.gov`, running as user `ovb` |
| Are you in the container? | **No.** You are on the host. `/.dockerenv` does not exist. |
| Container | `fenitop_container` (image `fenitop-image`, based on `dolfinx/dolfinx:stable`) |
| Mount | host `/raid/ovb/stochasticTO` → container `/shared` |
| Repo root (host) | `/raid/ovb/stochasticTO` |
| Repo root (container) | `/shared` |
| Docker | passwordless sudo works. **Every docker command must be `sudo docker`**, never bare `docker`. |
| Notifications | push via `scripts/notify_push.sh` → `ntfy.sh/stochasticTO-488039e71edb`. **Email does not work — see §8.** |
| Git branch | `test` (main branch is `master`) |

**Every study script hardcodes a relative `Path("output")` root.** They must be launched with
CWD = the repo root or they will scatter output into the wrong place. Always `cd /shared` first.

Artifacts are written by the container as **root** into an `ovb`-owned tree. You can read them
(`-rw-r--r--`) but cannot modify or delete them without `sudo`. Don't try to clean them up.

---

## 2. Where the pipeline stands

Four-step run sequence. Steps 1, 2 and 3a are done or running:

| Step | Command | Status |
|---|---|---|
| 1. Mesh convergence (**stop condition**) | `convergence_studies.py mesh` | ✅ **PASSED** |
| 2a. N-convergence, fixed design | `convergence_studies.py n-fixed` | ✅ done |
| 2b. N-convergence, re-optimizing | `convergence_studies.py n-opt` | ✅ done |
| 3a. SAA gap study | `saa_gap_study.py` | ⏳ **RUNNING** |
| 3b. Baseline comparison | `baseline_comparison.py` | ⬜ you launch this |
| 4. Production | `src/mainClean.py src/config/config.yaml` | 🛑 **DO NOT LAUNCH** |

**The mesh gate passed**, so nothing downstream is blocked:
`output/studies/mesh/20260727T213136Z_nogit/mesh_convergence.json` → `verdict.converged: true`,
`load_discretization_is_a_candidate_confounder: false`. (Note that the confounder flag can only ever
be `true` when `converged` is `false` — it is a "before you blame the physics" rider, not an
independent check.)

### The currently running job

```
mpirun -n 64 python scripts/saa_gap_study.py src/config/configStudy.yaml
```

- Started **2026-07-28 19:25 UTC**, PID 1921917 plus 64 children, running as root in `fenitop_container`.
- Run dir: `output/studies/saa_gap/20260728T192558Z_nogit/`
- Log: `/raid/ovb/stochasticTO/gap_log.txt` (shell redirect, root-owned, no timestamps in lines)
- **ETA ≈ 26–27 h → finishes roughly 2026-07-29 21:00–23:00 UTC.**

That ETA is derived, not guessed: `output/studies/n-opt/20260728T142558Z_nogit/manifest.json`
records `n_opt_N512 = 8894 s` for one SAA solve at N=512. Ten replications ≈ 24.7 h, plus ten
5000-sample out-of-sample evaluations ≈ 1.6 h.

> The `configStudy.yaml` header used to say "7 h". That was written when `saa_n_samples` was 128;
> it is now 512. The comment has been corrected — if you see 7 h anywhere else, it is stale.

### It is all-or-nothing

`saa_gap_study.py` holds all 10 replications in memory and writes **nothing** between the early
`nominal_*` artifacts and the final dump. No resume logic, no `.done` marker. A crash at replication 9
loses ~24 h with no partial results. Final write order:

```
saa_gap.json  →  best_design.npy  →  manifest.json   (manifest is always last)
```

---

## 3. Detecting completion vs failure

**Silence is ambiguous** — a missing `saa_gap.json` means either "still working" or "died two hours ago".
Always check both the file and the process:

```bash
cd /raid/ovb/stochasticTO
test -f output/studies/saa_gap/20260728T192558Z_nogit/saa_gap.json && echo "JSON PRESENT"
pgrep -cf "scripts/saa_gap_study.py"     # ~65 while alive, 0 when gone
tail -5 gap_log.txt
```

| Process | `saa_gap.json` | Meaning | Action |
|---|---|---|---|
| alive | absent | still running | reschedule, do nothing else |
| gone | **present** | ✅ success | go to §5 |
| gone | absent | 💥 crash / OOM / kill | go to §4 |
| alive | absent, log frozen ≥2 h | possible hang | push a warning, keep waiting |

Progress markers in `gap_log.txt`:

| Marker | Meaning |
|---|---|
| `=== replication m/10 (seed=..., N=512, lambda=4) ===` | replication `m` started (10 total) |
| `SAA lambda=4 stage k/5: beta=..., max_iter=30` | beta-continuation stage (5 stages × 30 iters = 150) |
| `[SAA] outer_iter=N: mu_C=... sigma_C=... J=...` | one iteration done |
| `replication m: in-sample J=..., out-of-sample J=... (optimism ...%)` | replication finished |
| `Run manifest written to <path>` | **last line of any successful run** |

Rough progress estimate: `grep -c "=== replication" gap_log.txt` gives the replication count reached.

---

## 4. If it crashed

Do **not** blindly relaunch. A relaunch costs another ~26 h of a shared 64-rank machine, so it is
worth one report-and-wait rather than a reflex.

1. Get the failure: `tail -100 gap_log.txt`, and `grep -nE "Traceback|Error|MPI_ABORT|Killed|OOM|assert" gap_log.txt | tail -30`
2. Check for an OOM kill: `dmesg -T 2>/dev/null | grep -i "killed process" | tail -5`
3. Confirm the container is still healthy: `sudo docker ps --filter name=fenitop_container`
4. Report the diagnosis with the log tail, and push a ping (priority `high`).
5. **Relaunch unattended only if** you have identified a specific blocker and fixed it. Otherwise
   report and wait. If you do relaunch, say so loudly at the top of the report.

Relaunch command (same form as the original):

```bash
sudo docker exec fenitop_container bash -lc \
  'cd /shared && source mpi-env.sh && mpirun -n 64 python scripts/saa_gap_study.py src/config/configStudy.yaml > gap_log.txt 2>&1'
```

---

## 5. Reading `saa_gap.json`

Path: `output/studies/saa_gap/20260728T192558Z_nogit/saa_gap.json`. **All keys are top-level/flat**
(except the two `sigma_C_*` dicts and the `replications` list).

```
n_replications                        int    = 10
saa_n_samples                         int    = 512
n_evaluation                          int    = 5000
lambda                                float  = 4.0   (only lambda_sweep[-1] is studied)
lower_bound / _standard_error         float  mean of in-sample objectives
upper_bound / _standard_error         float  best design re-evaluated out of sample
optimality_gap / _standard_error      float
optimality_gap_relative               float  ← HEADLINE
gap_ci_95                             [lo, hi]
sigma_C_in_sample                     {"mean": float, "std": float}
sigma_C_out_of_sample                 {"mean": float, "std": float}
sigma_optimism_relative               float  ← HEADLINE, see sign warning below
run_to_run_sigma_variability_relative float  ← HEADLINE, the noise floor
replications                          [10 × {replication, seed, in_sample_objective,
                                       in_sample_mu_C, in_sample_sigma_C,
                                       out_of_sample_objective, out_of_sample_mu_C,
                                       out_of_sample_sigma_C, converged}]
interpretation                        str
```

### ⚠️ Sign convention — get this right

`sigma_optimism_relative = (sigma_C_in_sample.mean − sigma_C_out_of_sample.mean) / sigma_C_out_of_sample.mean`

**NEGATIVE = overfitting.** The design looks *less* variable on the samples it was fitted to than on
fresh ones. This matches [src/mainClean.py:827-833](../src/mainClean.py#L827-L833).

The `interpretation` string inside `saa_gap.json` used to claim the opposite ("POSITIVE means
overfitting"). That was a documentation bug; it was corrected in `saa_gap_study.py` on 2026-07-28.
**If you are reading a JSON produced before that fix, its embedded `interpretation` is wrong.**
Always sanity-check against the raw numbers: if `sigma_C_in_sample.mean < sigma_C_out_of_sample.mean`,
the design is overfitting, whatever any string says.

Expected direction, from the completed n-opt study (`n_convergence_reoptimize.json`): optimism ran
−0.96 at N=64 → −0.40 at N=512. Strongly negative, i.e. **severe in-sample optimism that improves
with N but has not vanished by 512.** A gap-study result near −0.4 is consistent with that; a
*positive* value would be surprising and worth flagging rather than glossing.

### What each headline means

- **`optimality_gap_relative`** — how far N=512 SAA lands from the true robust optimum, as a fraction
  of the objective. Compare against `gap_ci_95`: if the interval straddles zero, the gap is not resolved.
- **`sigma_optimism_relative`** — the in-sample/out-of-sample inversion, quantified. This is the number
  the whole study exists to produce.
- **`run_to_run_sigma_variability_relative`** — **the noise floor.** σ_C spread across independent seeds.
  Any claimed difference between designs smaller than this is the seed, not the method, and must not be
  reported as an improvement. The script itself emits a `logger.warning` saying exactly this.

Also check `replications[*].converged`. These are expected to be **`false`** on the study tier
(`max_iter: 150` across 5 beta stages is not enough to converge) — that is normal here and neither
study script aborts on it. Do not treat it as a failure.

---

## 6. After success — the sequence

### Step A: sanity-check

10 replications present, no NaN/Inf in the headline fields, seeds are 1007, 2007, … 10007.

### Step B: report #1

Write the full report in the session (see §8 for required contents) and send a short push ping.
Quote the actual numbers from `saa_gap.json` in the report — do not just point at the file.

### Step C: launch baseline comparison (~3 h)

It needs all 64 ranks, so it **cannot overlap** with anything else. Confirm the gap study's processes
are gone (`pgrep -cf saa_gap_study.py` → 0) before launching.

```bash
sudo docker exec fenitop_container bash -lc \
  'cd /shared && source mpi-env.sh && mpirun -n 64 python scripts/baseline_comparison.py src/config/configStudy.yaml > baseline_log.txt 2>&1'
```

Run it **in the background** so you get a completion notification rather than blocking.

Output lands in `output/studies/baseline_comparison/<new_run_id>/` — this directory has **never
existed**; the first run creates it. Find the run id with:

```bash
ls -1dt output/studies/baseline_comparison/*/ | head -1
```

Watch for `baseline_comparison.json`, then `manifest.json` (written last, same as every other script).

### Step D: read `baseline_comparison.json`

Headline is `saa_vs_erode_dilate`:

- `saa_vs_erode_dilate.verdict` — the plain-language result
- `saa_vs_erode_dilate.std_difference_resolvable` — **if `false`, SAA's extra cost bought nothing
  measurable.** The script logs a warning when this happens. This is a legitimate result, not a
  failure — report it straight.
- `cost.cost_ratio` — how many more FEA solves SAA spent vs erode/dilate

Other keys: `discreteness` (`M_nd_percent` per method), `convergence`, `volume`,
`per_design` (nominal / saa / erode_dilate summaries), `paired_vs_nominal`. Comparisons use common
random numbers, so they are paired — `p_value_mean_difference` and `mean_difference_resolvable`
are meaningful.

### Step E: report #2, then STOP

Include a go/no-go recommendation for production, push a ping, and **wait for a human reply**.

🛑 **Do not launch `src/mainClean.py src/config/config.yaml`.** That is ~142 h / 5.9 days on all 64
ranks and is explicitly the user's decision. Recommending it is your job; starting it is not.

---

## 7. Code-change latitude

Authorized: **fix blockers + tune parameters.**

**You may:**
- Edit source to clear crashes, tracebacks, and anything blocking the next step.
- Adjust config parameters when the results show current settings are inadequate.

**You must:**
- Report every diff in the report that follows — paste the actual `git diff` of source changes, not a
  summary of it.
- Call out at the **top** of the report any change that alters what a number means. Never bury a
  scientific parameter change in a diff.

**Config edits must survive the loader's cross-checks** in
[src/config/loader.py:240-292](../src/config/loader.py#L240-L292), which reject:

- `optimization.saa_seed != mc_validation.seed`
- `mc_validation.beta != optimization.saa_beta_max`
- `len(lambda_sweep) < 2` (warns below 5)
- `eta_min >= eta_max`
- `robust_method` outside `{saa, pce}`
- any unknown key

A config that trips these fails the *next* run, not the current one — so a bad edit costs you a
launch and hours of wall-clock before you find out. Re-read the loader before touching the YAML.

**Do not** `git commit` or `git push` unless explicitly asked. Note that `output/` contains many
modified tracked files (ParaView/VTU artifacts) unrelated to this work — don't sweep them into a commit.

---

## 8. Reporting

**Email does not work and was abandoned.** The NIST relay (`smtp.nist.gov:25`) accepts the message at
the SMTP level but it never arrives at gmail; with local postfix inactive, the bounce is lost too, so a
successful-looking send there proves nothing. Do not try to resurrect it.

Two channels replace it:

**1. Push notification — must carry the actual result**

The user has explicitly asked that the push contain the pertinent details, not just a "come look"
ping. They should be able to read the outcome from a phone without opening a session. A notification
that says only "job finished" is a failure to follow instructions.

```bash
cd /raid/ovb/stochasticTO
scripts/notify_push.sh "SAA gap study done (10/10 reps, 26.3 h)" "$(cat <<'EOF'
sigma optimism: -0.41  -> OVERFITTING (in 0.051 < out 0.086)
noise floor:     8.2%  -> claims below this are seed noise
optimality gap:  3.1% +/- 0.9%
converged:       0/10 (expected on study tier)

code changed: none
next: baseline_comparison.py launched, ~3 h
EOF
)"
```

Posts to `ntfy.sh/stochasticTO-488039e71edb`. Exits non-zero on failure — **check it**. Accepts
`--file PATH` or `--stdin` for longer bodies, and truncates near 4 KB rather than failing.

Every push must include, at minimum:

1. **What finished** + wall-clock duration, in the title
2. **The headline numbers** with their interpretation spelled out (e.g. "-0.41 → OVERFITTING", not a
   bare number — the sign is the whole point and is easy to misread; see §5)
3. **Whether any code changed** — "none" is a valid and useful answer
4. **What happens next**, or what is being waited on
5. **Anything surprising** — a NaN, a positive optimism, an unexpected crash

Priority: use `high` as the third argument for crashes and for the production go/no-go, default otherwise.

Keep raw file contents, paths full of internal hostnames, and anything credential-shaped out of it —
the topic name is the only thing keeping the channel private. Study results and headline statistics
are fine; the user has asked for them explicitly.

**2. The session transcript — the actual report**

The full write-up goes in your reply in the session, and that is the primary deliverable. If the session
is unattended, also write it to `output/studies/_reports/<timestamp>_<step>.md` so it survives.

Each report should contain:

1. **What finished**, wall-clock duration (from `manifest.json`: `started_utc`, `finished_utc`, `total_seconds`)
2. **Headline numbers** with their interpretation — including which sign means what
3. **The noise floor** and what claims it does or doesn't permit
4. **Per-replication table** (gap study) or the head-to-head verdict (baseline)
5. **Code changed**: full `git diff`, or "none"
6. **What I did next** and why — or what I'm waiting on
7. **Anything surprising** — a positive optimism, a NaN, an unexpected convergence flag

---

## 9. Gotchas

- **`sudo docker`, always.** Bare `docker` fails.
- **`cd /shared` before every launch.** Relative output paths.
- **`source mpi-env.sh`** pins thread counts to 1 (`OMP_NUM_THREADS` etc.) so the 64 ranks don't
  oversubscribe. The original run used it; keep doing so.
- **`run_id` is always `..._nogit`** — git fails inside the container (root in an `ovb`-owned tree),
  so `manifest.json`'s `git` block is all `null`. Expected, not a bug.
- **`saa_gap_study.py` never calls `manifest.record_config(...)`**, unlike `baseline_comparison.py`,
  so its manifest lacks `config_declared` / `fem_effective` / `opt_effective`. Not a bug.
- **`replications[*].converged: false` is expected** on the study tier. Not a failure.
- **The KL cache** at `output/cache/kl_expansion/` is shared and warm for the study mesh, so re-runs
  skip the eigensolve.
- **`saa_n_samples: 512` in `configStudy.yaml` is an uncommitted change** from 128. It is intentional
  — the whole point of this study. Don't "fix" it back.
- **Both study scripts only use `lambda_sweep[-1]` = 4.0**, not the full sweep.
- Disk is not a concern: ~6.3 TB free, `output/` is ~4.3 GB.
