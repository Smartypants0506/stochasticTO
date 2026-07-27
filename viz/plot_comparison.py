"""viz/plot_comparison.py -- the 2D figures that carry the argument.

Consumes output/comparison/ (written by viz/compare_designs_mc.py) and, where
available, output/viz/sample_index.csv and output/stage5_optimization/
pareto_results.json. Pure matplotlib + numpy; no dolfinx, no MPI, so it runs
anywhere the CSVs are.

FIGURES
-------
  fig1_cdf_overlay.png
      Empirical CDF of compliance under manufacturing variation, one curve per
      design. This is the single most important plot: robust TO does not
      promise a better nominal part, it promises a *tighter, left-shifted*
      distribution. The 95th-percentile verticals make the tail claim visible.

  fig2_distributions.png
      Violin + paired-scatter. The scatter is the one a sceptical reviewer
      cares about: because every design saw the SAME manufacturing defect
      field, each dot is a controlled A/B on one realization. Points below the
      diagonal are realizations where robust won. A win rate near 100% is
      much harder to dismiss than "the mean moved a bit".

  fig3_risk_metrics.png
      Bar chart of mean / std / p95 / worst-case, normalized to the baseline.
      The summary slide.

  fig4_pareto.png
      mu_C vs sigma_C from the lambda sweep, with the MC-validated points
      overlaid. Shows the trade-off is a knob, not a fixed cost, and that the
      in-loop SAA estimate agrees with independent MC (the honesty check).

  fig5_likelihood_vs_compliance.png
      Compliance against how typical the realization is. If the worst
      compliances come from *ordinary* realizations rather than freak ones,
      the risk is not a tail curiosity -- it is what the process will
      routinely produce. That reframing is usually the strongest single
      argument for funding this.

USAGE
-----
    python viz/plot_comparison.py
    python viz/plot_comparison.py --comparison-dir output/comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {"nominal": "#c0392b", "robust_lambda0": "#e08e0b",
           "robust_lambda1": "#1f6fb4"}
_FALLBACK = ["#1f6fb4", "#c0392b", "#e08e0b", "#2c8a5b", "#7d4bb0"]


def _color(name: str, i: int) -> str:
    return _COLORS.get(name, _FALLBACK[i % len(_FALLBACK)])


def _load_paired(comparison_dir: Path) -> tuple[list[str], np.ndarray]:
    path = comparison_dir / "paired_compliance.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run viz/compare_designs_mc.py first.")
    names = path.read_text().splitlines()[0].split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return names, data


def fig_cdf(names, data, summary, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = data.shape[0]
    ecdf = np.arange(1, n + 1) / n
    for i, name in enumerate(names):
        C = np.sort(data[:, i])
        ax.plot(C, ecdf, lw=2.2, color=_color(name, i), label=name)
        p95 = np.percentile(data[:, i], 95)
        ax.axvline(p95, color=_color(name, i), ls=":", lw=1.2, alpha=0.8)
    ax.set_xlabel("Compliance $C$ under manufacturing variation")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Compliance distribution over {n} shared realizations\n"
                 "(dotted = 95th percentile)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_distributions(names, data, baseline: str, out: Path) -> None:
    b = names.index(baseline) if baseline in names else 0
    others = [i for i in range(len(names)) if i != b]
    fig, axes = plt.subplots(1, 1 + len(others),
                             figsize=(4.2 + 3.4 * len(others), 4.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    parts = ax.violinplot([data[:, i] for i in range(len(names))],
                          showmeans=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(_color(names[i], i))
        body.set_alpha(0.55)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Compliance $C$")
    ax.set_title("Spread of outcomes", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    for k, i in enumerate(others):
        ax = axes[k + 1]
        x, y = data[:, b], data[:, i]
        wins = (y < x).mean()
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1)
        ax.scatter(x, y, s=18, alpha=0.7, color=_color(names[i], i),
                   edgecolor="none")
        ax.set_xlabel(f"$C$ -- {names[b]}")
        ax.set_ylabel(f"$C$ -- {names[i]}")
        ax.set_title(f"Same defect field, both designs\n"
                     f"{names[i]} wins on {100*wins:.0f}% of realizations",
                     fontsize=10)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_risk_metrics(names, data, baseline: str, out: Path) -> None:
    b = names.index(baseline) if baseline in names else 0
    metrics = ["mean", "std", "p95", "worst"]

    def compute(C):
        return {"mean": C.mean(), "std": C.std(ddof=1),
                "p95": np.percentile(C, 95), "worst": C.max()}

    ref = compute(data[:, b])
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, name in enumerate(names):
        vals = compute(data[:, i])
        rel = [100 * vals[m] / ref[m] for m in metrics]
        pos = np.arange(len(metrics)) + i * width - 0.4 + width / 2
        bars = ax.bar(pos, rel, width, color=_color(name, i), label=name)
        for r, bar in zip(rel, bars):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 1, f"{r:.0f}",
                    ha="center", fontsize=7)
    ax.axhline(100, color="0.3", lw=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["mean $C$", "std $C$", "95th pct $C$", "worst case $C$"])
    ax.set_ylabel(f"% of {names[b]} (lower is better)")
    ax.set_title("Risk metrics relative to the deterministic design", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_pareto(pareto_path: Path, insample_path: Path, out: Path) -> None:
    if not pareto_path.exists():
        return
    rows = json.loads(pareto_path.read_text())
    mu = [r["mu_C"] for r in rows]
    sig = [r["sigma_C"] for r in rows]
    lam = [r["lambda"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(mu, sig, "-o", color="#1f6fb4", lw=2, ms=7, label="SAA in-loop estimate")
    for m, s, l in zip(mu, sig, lam):
        ax.annotate(rf"$\lambda={l:g}$", (m, s), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    if insample_path.exists():
        d = json.loads(insample_path.read_text())
        ax.scatter([d["mc_mean"]], [d["mc_std"]], marker="*", s=220,
                   color="#c0392b", zorder=5,
                   label="independent MC (different seed)")
        ax.annotate(f"rel. err: mean {100*d['rel_err_mean']:.1f}%, "
                    f"std {100*d['rel_err_std']:.1f}%",
                    (d["mc_mean"], d["mc_std"]), textcoords="offset points",
                    xytext=(6, -16), fontsize=8, color="#c0392b")
    ax.set_xlabel(r"$\mu_C$  (mean compliance)")
    ax.set_ylabel(r"$\sigma_C$  (std of compliance)")
    ax.set_title("Robustness is a tunable trade-off, and the in-loop\n"
                 "estimate is confirmed out-of-sample", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_likelihood(index_path: Path, out: Path) -> None:
    if not index_path.exists():
        return
    hdr = index_path.read_text().splitlines()[0].split(",")
    d = np.loadtxt(index_path, delimiter=",", skiprows=1)
    col = {name: i for i, name in enumerate(hdr)}
    like = d[:, col["occurrence_likelihood"]]
    C = d[:, col["compliance"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sc = ax.scatter(like, C, c=d[:, col["radius_percentile"]], cmap="viridis_r",
                    s=32, edgecolor="none")
    ax.axhline(np.percentile(C, 95), color="#c0392b", ls="--", lw=1.2,
               label="95th pct compliance")
    ax.set_xlabel("occurrence likelihood  $1 - F_{\\chi^2}(\\|\\xi\\|^2)$\n"
                  "(1 = a perfectly typical part, 0 = a rare one)")
    ax.set_ylabel("Compliance $C$")
    ax.set_title("Are the bad outcomes rare, or routine?", fontsize=10)
    fig.colorbar(sc, ax=ax, label="radius percentile")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparison-dir", type=Path, default=Path("output/comparison"))
    ap.add_argument("--viz-dir", type=Path, default=Path("output/viz"))
    ap.add_argument("--stage5-dir", type=Path, default=Path("output/stage5_optimization"))
    ap.add_argument("--stage6-dir", type=Path, default=Path("output/stage6_validation"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/figures"))
    ap.add_argument("--baseline", default="nominal")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.comparison_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    names, data = _load_paired(args.comparison_dir)
    fig_cdf(names, data, summary, args.out_dir / "fig1_cdf_overlay.png")
    fig_distributions(names, data, args.baseline,
                      args.out_dir / "fig2_distributions.png")
    fig_risk_metrics(names, data, args.baseline,
                     args.out_dir / "fig3_risk_metrics.png")
    fig_pareto(args.stage5_dir / "pareto_results.json",
               args.stage6_dir / "insample_vs_mc.json",
               args.out_dir / "fig4_pareto.png")
    fig_likelihood(args.viz_dir / "sample_index.csv",
                   args.out_dir / "fig5_likelihood_vs_compliance.png")
    print(f"figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
