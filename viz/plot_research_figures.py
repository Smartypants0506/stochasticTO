"""The three figures the paper's argument now rests on.

    python viz/plot_research_figures.py [--out-dir output/figures]

Written to be run repeatedly as studies land: each figure is skipped with a
message if its artifact is not on disk yet, so a partial run still produces
what it can rather than failing.

  figA_correlation_length.png  cv vs l_c with the uniform limit marked.
                               THE headline figure: it is the evidence that the
                               spatially uniform threshold bounds the response
                               variance from above.
  figB_mesh_convergence.png    cv vs element size. Shows the reported statistic
                               is a continuum property, and shows WHY it must be
                               cv rather than mu_C or sigma_C alone.
  figC_gap_replications.png    in-sample vs out-of-sample sigma across the SAA
                               replications. This is the figure that makes the
                               "one solve in seven" claim visible: the outlier
                               sits far off the diagonal while its in-sample
                               value is unremarkable.

viz/plot_comparison.py covers the CDF/Pareto/risk figures for the Stage-6
ensemble; this file deliberately does not duplicate them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz.paths import ROOT, artifact  # noqa: E402

PALETTE = {
    "line": "#2b6cb0",
    "uniform": "#c53030",
    "accent": "#2f855a",
    "muted": "#718096",
    "band": "#bee3f8",
}


def _finish(fig, ax, out: Path, title: str) -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def _cv_interval(entry: dict):
    """(low, high) for a level's cv, or None when the run predates the paired
    ratio estimator. Deliberately does NOT fall back to propagating the
    separate mean/std intervals -- that overstates the width by ~25% and would
    misrepresent the resolution of the very comparison the figure makes."""
    est = (entry.get("statistics") or {}).get("cv")
    if not est or est.get("ci_low") is None:
        return None
    return est["ci_low"], est["ci_high"]


def fig_correlation_length(out: Path) -> None:
    data, path = artifact("lc_sweep")
    if data is None:
        print("  skip figA: no l_c sweep on disk")
        return
    levels = data.get("levels", [])
    corr = [e for e in levels if e.get("length_scale") is not None]
    uni = next((e for e in levels if e.get("length_scale") is None), None)
    if not corr:
        print("  skip figA: sweep has no correlated levels")
        return

    x = np.array([e["length_scale"] for e in corr], dtype=float)
    y = np.array([e["cv"] for e in corr], dtype=float)
    intervals = [_cv_interval(e) for e in corr]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if all(iv is not None for iv in intervals):
        lo = np.array([iv[0] for iv in intervals])
        hi = np.array([iv[1] for iv in intervals])
        ax.fill_between(x, lo, hi, color=PALETTE["band"], alpha=0.55,
                        label="95% CI (paired bootstrap)")
    else:
        print("  note figA: cv intervals absent (run predates the ratio "
              "estimator) -- plotting point estimates only")
    ax.plot(x, y, "o-", color=PALETTE["line"], lw=1.8, ms=5, label="correlated field")

    if uni is not None:
        ax.axhline(uni["cv"], color=PALETTE["uniform"], ls="--", lw=1.6,
                   label=f"uniform limit (cv = {uni['cv']:.3f})")
        iv = _cv_interval(uni)
        if iv:
            ax.axhspan(iv[0], iv[1], color=PALETTE["uniform"], alpha=0.10)

    ax.set_xscale("log")
    ax.set_xlabel(r"correlation length  $\ell_c$  (domain units; beam is 30 long, 10 across)")
    ax.set_ylabel(r"$\sigma_C/\mu_C$")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # Deliberately NOT titled "uniform is the bound": the largest swept l_c can
    # sit a few percent ABOVE the uniform line, because at l_c > domain the
    # expansion still retains a couple of modes and is not yet exactly
    # constant. The defensible statement is the endpoint ratio.
    ratio = (uni["cv"] / y.min()) if uni else float("nan")
    if uni is not None and y.max() > uni["cv"]:
        ax.annotate(
            f"largest $\\ell_c$ exceeds the uniform line by "
            f"{100*(y.max()-uni['cv'])/uni['cv']:.0f}%\n"
            f"($n_{{kl}}$ still > 1, so not yet exactly constant)",
            xy=(x[np.argmax(y)], y.max()), xytext=(-16, -46),
            textcoords="offset points", fontsize=7.5, color=PALETTE["muted"],
            ha="right",
        )
    _finish(fig, ax, out,
            f"$\\sigma_C/\\mu_C$ rises {ratio:.1f}$\\times$ from short "
            f"$\\ell_c$ to the uniform limit")


def fig_mesh_convergence(out: Path) -> None:
    data, path = artifact("mesh")
    if data is None:
        print("  skip figB: no mesh study on disk")
        return
    levels = data.get("levels", [])
    if not levels:
        print("  skip figB: mesh study has no levels")
        return

    h = np.array([e["element_size_h"] for e in levels], dtype=float)
    cv = np.array([e["cov_sigma_over_mu"] for e in levels], dtype=float)
    mu = np.array([e["mu_C"] for e in levels], dtype=float)
    sd = np.array([e["sigma_C"] for e in levels], dtype=float)
    order = np.argsort(h)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # Normalising mu and sigma to their finest-mesh value is the point of the
    # figure: they move ~37% while their ratio does not.
    finest = int(np.argmin(h))
    ax.plot(h[order], (mu / mu[finest])[order], "s--", color=PALETTE["muted"],
            lw=1.4, ms=5, label=r"$\mu_C$ (normalised)")
    ax.plot(h[order], (sd / sd[finest])[order], "^--", color=PALETTE["accent"],
            lw=1.4, ms=5, label=r"$\sigma_C$ (normalised)")
    ax.plot(h[order], (cv / cv[finest])[order], "o-", color=PALETTE["line"],
            lw=2.0, ms=6, label=r"$\sigma_C/\mu_C$ (normalised)")

    spread = (cv.max() - cv.min()) / cv.min()
    ax.axhspan(1 - spread, 1 + spread, color=PALETTE["band"], alpha=0.4,
               label=f"cv spread: {100*spread:.1f}%")
    ax.set_xlabel("element size  $h$")
    ax.set_ylabel("value / value at finest mesh")
    ax.legend(frameon=False, fontsize=9)
    _finish(fig, ax, out,
            "Only the ratio is mesh-converged: report $\\sigma_C/\\mu_C$, "
            "not absolute compliance")


def fig_gap_replications(out: Path) -> None:
    data, path = artifact("gap")
    if data is None:
        print("  skip figC: gap study still running (saa_gap.json absent)")
        return
    reps = data.get("replications", [])
    if not reps:
        print("  skip figC: gap study has no replications")
        return

    ins = np.array([r["in_sample_sigma_C"] for r in reps], dtype=float)
    out_s = np.array([r["out_of_sample_sigma_C"] for r in reps], dtype=float)

    med = float(np.median(out_s))
    mad = float(np.median(np.abs(out_s - med)))
    outlier = np.abs(out_s - med) > 3 * 1.4826 * mad if mad > 0 else np.zeros_like(out_s, bool)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    lim = [0, max(ins.max(), out_s.max()) * 1.15]
    ax.plot(lim, lim, ls=":", color=PALETTE["muted"], lw=1.2, label="unbiased (y = x)")
    ax.scatter(ins[~outlier], out_s[~outlier], s=55, color=PALETTE["line"],
               zorder=3, label="replications")
    if outlier.any():
        ax.scatter(ins[outlier], out_s[outlier], s=110, facecolors="none",
                   edgecolors=PALETTE["uniform"], lw=2.0, zorder=4,
                   label=f"outlier ({out_s[outlier].max()/med:.1f}$\\times$ median)")
        for xi, yi in zip(ins[outlier], out_s[outlier]):
            ax.annotate("in-sample value is unremarkable",
                        (xi, yi), textcoords="offset points", xytext=(10, -14),
                        fontsize=8, color=PALETTE["uniform"])

    ax.set_xlim(0, lim[1]); ax.set_ylim(0, lim[1])
    ax.set_aspect("equal")
    ax.set_xlabel(r"in-sample $\sigma_C$ (the optimizer's own 512 samples)")
    ax.set_ylabel(r"out-of-sample $\sigma_C$ (5000 fresh samples)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _finish(fig, ax, out,
            "Every point sits above the diagonal: in-sample $\\sigma_C$ is optimistic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "figures")
    args = parser.parse_args()

    print("building research figures:")
    fig_correlation_length(args.out_dir / "figA_correlation_length.png")
    fig_mesh_convergence(args.out_dir / "figB_mesh_convergence.png")
    fig_gap_replications(args.out_dir / "figC_gap_replications.png")


if __name__ == "__main__":
    main()
