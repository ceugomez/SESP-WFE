# plot_theta_baseline.py
# cgf  2026-07-13
#
# R1 head-to-head figure: the Θ-regression baseline (2-parameter forcing fit +
# conditional-mean lookup) vs the reduced-order KF, on full-domain and
# wake-region RMSE, with the Θ-oracle (true forcing through the same map) and
# the subspace-oracle projection floor. Reads the CSV written by
# scripts/run_theta_baseline.jl. Panel (c) shows the Θ̂ trajectory against the
# truth forcing — whether the baseline even identifies the right (U_g, V_g).
#
# Usage:
#   python plot_theta_baseline.py <theta_baseline_truth_<id>.csv> [--out PNG] \
#       [--ug-true X --vg-true Y]

import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="theta_baseline_truth_<id>.csv from run_theta_baseline.jl")
    p.add_argument("--out", default=None, help="output PNG (default: figure/ alongside repo)")
    p.add_argument("--ug-true", type=float, default=None, help="truth U_g for panel (c) reference")
    p.add_argument("--vg-true", type=float, default=None, help="truth V_g for panel (c) reference")
    args = p.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    t = np.asarray(d["time"], dtype=float)

    series = [  # (csv column stem, label, color, linestyle)
        ("theta_rmse",       "Θ-baseline (2-param fit)",   "C1", "-"),
        ("thetaoracle_rmse", "Θ-oracle (true forcing)",    "C1", "--"),
        ("kf_rmse",          "reduced-order KF",           "C0", "-"),
        ("oracle_rmse",      "subspace oracle (floor)",    "0.4", ":"),
    ]
    if "const_rmse_full" in (d.dtype.names or ()):  # added 2026-07-15
        series.insert(0, ("const_rmse", "constant wind (obs mean)", "C4", "-"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    for ax, region, name in ((axes[0], "full", "(a) full analysis domain"),
                             (axes[1], "wake", "(b) wake region")):
        for stem, label, col, ls in series:
            y = d[f"{stem}_{region}"]
            ax.plot(t, y, ls, color=col, lw=1.3,
                    label=f"{label}  [mean {np.mean(y):.3f}]")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("field RMSE (m/s)")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")

    axc = axes[2]
    axc.plot(t, d["Ug_hat"], "-", color="C2", lw=1.2, label=r"$\hat{U}_g$")
    axc.plot(t, d["Vg_hat"], "-", color="C3", lw=1.2, label=r"$\hat{V}_g$")
    if args.ug_true is not None:
        axc.axhline(args.ug_true, color="C2", ls=":", lw=1.0, label=r"$U_g$ truth")
    if args.vg_true is not None:
        axc.axhline(args.vg_true, color="C3", ls=":", lw=1.0, label=r"$V_g$ truth")
    axc.set_xlabel("time (s)")
    axc.set_ylabel("estimated forcing (m/s)")
    axc.set_title("(c) Θ̂ trajectory of the baseline")
    axc.grid(alpha=0.3)
    axc.legend(fontsize=7.5, loc="best")

    out = args.out
    if out is None:
        repo_fig = pathlib.Path(__file__).resolve().parents[2] / "figure"
        repo_fig.mkdir(exist_ok=True)
        out = str(repo_fig / (pathlib.Path(args.csv).stem + ".png"))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
