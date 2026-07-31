"""Turn a finished study's JSON into a phone-readable summary.

    python scripts/summarize_result.py gap|lc|probe|uniform|baseline|pipeline

Used by scripts/job_watch.sh to build push notifications. Per
docs/AUTONOMOUS_PIPELINE_HANDOFF.md section 8, a push must carry the actual
result, not a "come look" ping: headline numbers WITH their interpretation
spelled out, because several of these signs are easy to misread and the sign is
the whole point.

Prints to stdout; exits 1 if the artifact is missing so the caller can tell
"not finished" from "finished with nothing to say".
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _newest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(ROOT / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return Path(hits[-1]) if hits else None


def _load(pattern: str):
    path = _newest(pattern)
    if path is None:
        return None, None
    try:
        with open(path) as handle:
            return json.load(handle), path
    except (OSError, ValueError):
        return None, path


def _pct(x, digits=1):
    return "n/a" if x is None else f"{100 * float(x):+.{digits}f}%"


# ---------------------------------------------------------------- gap study

def gap() -> str:
    data, path = _load("output/studies/saa_gap/*/saa_gap.json")
    if data is None:
        return ""
    out = []
    sin = data.get("sigma_C_in_sample", {}).get("mean")
    sout = data.get("sigma_C_out_of_sample", {}).get("mean")
    opt = data.get("sigma_optimism_relative")

    # NEGATIVE = overfitting (design looks less variable on its own samples).
    verdict = "n/a"
    if opt is not None:
        verdict = "OVERFITTING" if opt < 0 else "no overfitting (surprising - flag it)"
    out.append(f"sigma optimism: {_pct(opt)} -> {verdict}")
    if sin is not None and sout is not None:
        out.append(f"  in-sample {sin:.4f} vs out-of-sample {sout:.4f}")

    # Noise floor: the raw std/mean is what the JSON reports, but the measured
    # distribution is heavy-tailed, so report the robust spread beside it.
    reps = data.get("replications") or []
    sig = np.array([r.get("out_of_sample_sigma_C", np.nan) for r in reps], dtype=float)
    sig = sig[np.isfinite(sig)]
    raw = data.get("run_to_run_sigma_variability_relative")
    out.append(f"noise floor (raw std/mean): {_pct(raw)}")
    if sig.size >= 3:
        med = float(np.median(sig))
        mad = float(np.median(np.abs(sig - med)))
        robust = 1.4826 * mad / med if med > 0 else float("nan")
        n_out = int(np.sum(np.abs(sig - med) > 3 * 1.4826 * mad)) if mad > 0 else 0
        out.append(f"noise floor (robust MAD):   {100*robust:+.1f}%   <- use this one")
        out.append(f"  worst/median {sig.max()/med:.2f}x, {n_out} outlier(s) of {sig.size}")
        if n_out:
            out.append("  HEAVY-TAILED: single-solve comparisons are underpowered")

    gap_rel = data.get("optimality_gap_relative")
    ci = data.get("gap_ci_95") or [None, None]
    out.append(f"optimality gap: {_pct(gap_rel)}")
    if ci[0] is not None:
        straddles = ci[0] <= 0 <= ci[1]
        out.append(f"  95% CI [{ci[0]:.4g}, {ci[1]:.4g}]"
                   + ("  (straddles 0 -> unresolved)" if straddles else ""))

    conv = sum(1 for r in reps if r.get("converged"))
    out.append(f"converged: {conv}/{len(reps)} (false is EXPECTED on study tier)")
    out.append(f"\nfile: {path.relative_to(ROOT)}")
    return "\n".join(out)


# ------------------------------------------------------------ l_c sweep (E2)

def lc() -> str:
    data, path = _load("output/studies/correlation_length/*/correlation_length_fixed_design.json")
    if data is None:
        return ""
    out = ["cv = sigma_C/mu_C vs correlation length:"]
    for lvl in data.get("levels", []):
        label = lvl.get("label", "?")
        out.append(f"  l_c={label:<8} n_kl={lvl.get('n_kl','?'):<4} cv={lvl.get('cv', float('nan')):.4f}")
    peak = data.get("peak") or {}
    uni = data.get("uniform_limit") or {}
    excess = data.get("peak_excess_over_uniform_relative")
    interior = data.get("peak_is_interior")
    out.append(f"\npeak at l_c={peak.get('length_scale','?')} (cv={peak.get('cv',float('nan')):.4f})")
    out.append(f"uniform limit cv={uni.get('cv', float('nan')):.4f}")
    out.append(f"peak excess over uniform: {_pct(excess)}")
    if interior is False:
        out.append("PEAK IS AT AN ENDPOINT -> curve is monotonic, no interior worst case.")
        out.append("If the max is at the large-l_c end, the UNIFORM case is worst,")
        out.append("which SUPPORTS Schevenels 2011 and undercuts the field premise.")
    elif interior:
        out.append("Interior peak -> there IS a worst correlation length. This is the")
        out.append("new result: it says WHEN spatial correlation matters.")
    out.append(f"\nfile: {path.relative_to(ROOT)}")
    return "\n".join(out)


# --------------------------------------------------------- move-limit probe

def probe() -> str:
    data, path = _load("output/studies/move_limit_probe/probe_factor_*.json")
    if data is None:
        return ""
    base = data.get("baseline", {})
    imp = data.get("stationarity_improvement_relative")
    objch = data.get("objective_change_relative")
    out = [
        f"move continuation factor {data.get('move_reduction')}",
        f"final stat_rel {data.get('final_stationarity_rel', float('nan')):.4g} "
        f"vs baseline {base.get('final_stat_rel', float('nan')):.4g}",
        f"stationarity improvement: {_pct(imp)}  (higher is better)",
        f"objective change: {_pct(objch)}  (near 0 is required)",
        f"M_nd {data.get('M_nd_percent', float('nan')):.3g}% vs {base.get('M_nd_percent', float('nan')):.3g}%",
        f"converged: {data.get('converged')}",
        "",
        "per stage (beta / move / stat_rel / dx):",
    ]
    for s in data.get("per_stage", []):
        out.append(f"  b={s.get('beta'):<6g} m={s.get('move_limit',0):<8.4g} "
                   f"stat={s.get('stationarity_rel',float('nan')):<9.4g} "
                   f"dx={s.get('design_change',float('nan')):.4g}")
    if imp is not None:
        if imp > 0.5 and (objch or 0) < 0.02:
            out.append("\nPASS -> enable for the production run.")
        elif imp > 0.5:
            out.append("\nMIXED -> stationarity better but objective worse; inspect before enabling.")
        else:
            out.append("\nFAIL -> plateau is not the move limit. Likely beta=128 projection")
            out.append("stiffness. Report the achieved residual honestly instead.")
    out.append(f"\nfile: {path.relative_to(ROOT)}")
    return "\n".join(out)


# ------------------------------------------------------------- E1 uniform vs field

def uniform() -> str:
    data, path = _load("output/studies/uniform_eta/uniform_eta_comparison.json")
    if data is None:
        return ""
    sig = data.get("sigma_C_under_correlated_error", {})
    rule = data.get("decision_rule", {})
    out = [
        "Does spatial correlation buy anything? (Schevenels 2011 control)",
        f"  uniform-optimized sigma_C: {sig.get('uniform_optimized', float('nan')):.5g}",
        f"  field-optimized   sigma_C: {sig.get('field_optimized', float('nan')):.5g}",
        f"  reduction from field arm : {_pct(sig.get('relative_reduction_from_field_arm'))}",
        f"  noise floor              : {_pct(rule.get('run_to_run_sigma_variability'))}",
        f"  VERDICT: {rule.get('verdict','n/a')}",
    ]
    v = rule.get("verdict")
    if v == "spatial_correlation_matters":
        out.append("  -> the KL field earns its cost.")
    elif v == "field_arm_significantly_worse":
        out.append("  -> ANOMALY: check the field arm converged before interpreting.")
    elif v == "indistinguishable":
        out.append("  -> reproduces Schevenels 2011 in 3D. Still publishable, but")
        out.append("     the paper's framing changes. Re-scope production premise.")
    out.append(f"\nfile: {path.relative_to(ROOT)}")
    return "\n".join(out)


# ----------------------------------------------------- erode/dilate baseline

def baseline() -> str:
    data, path = _load("output/studies/baseline_comparison/*/baseline_comparison.json")
    if data is None:
        return ""
    h2h = data.get("saa_vs_erode_dilate", {})
    cost = data.get("cost", {})
    disc = data.get("discreteness", {})
    out = [
        f"cost ratio: {cost.get('cost_ratio', float('nan')):.0f}x "
        f"({cost.get('saa_fea_solves')} vs {cost.get('erode_dilate_fea_solves')} FEA solves)",
        f"std difference resolvable: {h2h.get('std_difference_resolvable')}",
        f"M_nd: SAA {disc.get('saa_M_nd_percent', float('nan')):.3g}% / "
        f"E-D {disc.get('erode_dilate_M_nd_percent', float('nan')):.3g}%",
        "",
        f"verdict: {h2h.get('verdict','n/a')}",
    ]
    if h2h.get("std_difference_resolvable") is False:
        out.append("\n-> SAA's extra cost bought NOTHING measurable. Legitimate result,")
        out.append("   report it straight.")
    out.append(f"\nfile: {path.relative_to(ROOT)}")
    return "\n".join(out)


# -------------------------------------------------------- full pipeline run

def pipeline() -> str:
    data, path = _load("output/stage5_optimization/*/pareto_results.json")
    if data is None:
        return ""
    out = ["Pareto front (lambda / mu_C / sigma_C / M_nd / converged):"]
    for p in data.get("points", []):
        out.append(
            f"  L={p.get('lambda'):<5g} mu={p.get('mu_C', float('nan')):<9.5g} "
            f"sig={p.get('sigma_C', float('nan')):<9.5g} "
            f"M_nd={p.get('M_nd_percent', float('nan')):<6.3g}% conv={p.get('converged')}"
        )
    n_conv = sum(1 for p in data.get("points", []) if p.get("converged"))
    out.append(f"\nconverged: {n_conv}/{len(data.get('points', []))}")
    val, vpath = _load("output/stage6_validation/*/validation_summary.json")
    if val is not None:
        out.append(f"stage6 validation present: {vpath.relative_to(ROOT)}")
    out.append(f"\nfile: {path.relative_to(ROOT)}")
    return "\n".join(out)


HANDLERS = {
    "gap": gap, "lc": lc, "probe": probe,
    "uniform": uniform, "baseline": baseline, "pipeline": pipeline,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in HANDLERS:
        sys.exit(f"usage: summarize_result.py {'|'.join(HANDLERS)}")
    text = HANDLERS[sys.argv[1]]()
    if not text:
        sys.exit(1)
    print(text)
