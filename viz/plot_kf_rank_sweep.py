# plot_kf_rank_sweep.py
# cgf  2026-07-14
#
# Rank-sensitivity figure: full-domain KF field RMSE vs truncation rank r, with the
# rank-r subspace-oracle floor and the (rank-independent) Θ-baseline / Θ-oracle
# reference lines from the theta_baseline CSV. Second panel: mean NIS vs r.
# Wake-region scoring was removed 2026-11-05 (to be redesigned for the new ensemble);
# the region loop is kept so a revised region can be added back as another panel.
#
# Usage:
#   python plot_kf_rank_sweep.py <kf_rank_sweep_truth_<id>.csv> \
#       [--theta theta_baseline_truth_<id>.csv] [--out PNG]

import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sesp_paths import figure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="kf_rank_sweep_truth_<id>.csv from run_kf_rank_sweep.jl")
    p.add_argument("--theta", default=None,
                   help="theta_baseline_truth_<id>.csv (default: inferred by filename swap)")
    p.add_argument("--out", default=None, help="output PNG (default: figure/<tag>/)")
    args = p.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    r = np.asarray(d["r"], dtype=int)

    theta_path = args.theta or args.csv.replace("kf_rank_sweep_", "theta_baseline_")
    theta = None
    if pathlib.Path(theta_path).is_file():
        theta = np.genfromtxt(theta_path, delimiter=",", names=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, region, name in ((axes[0], "full", "(a) full analysis domain"),):
        ax.plot(r, d[f"kf_rmse_{region}"], "o-", color="C0", lw=1.4,
                label="reduced-order KF")
        ax.plot(r, d[f"oracle_rmse_{region}"], "s:", color="0.4", lw=1.2,
                label="subspace oracle (floor)")
        if theta is not None:
            tb = np.mean(theta[f"theta_rmse_{region}"])
            to = np.mean(theta[f"thetaoracle_rmse_{region}"])
            ax.axhline(tb, color="C1", ls="-", lw=1.2,
                       label=f"Θ-baseline [{tb:.3f}]")
            ax.axhline(to, color="C1", ls="--", lw=1.0,
                       label=f"Θ-oracle [{to:.3f}]")
            if f"const_rmse_{region}" in (theta.dtype.names or ()):
                cw = np.mean(theta[f"const_rmse_{region}"])
                ax.axhline(cw, color="C4", ls="--", lw=1.2,
                           label=f"constant wind [{cw:.3f}]")
        ax.set_xscale("log")
        ax.set_xticks(r); ax.set_xticklabels([str(x) for x in r], fontsize=8)
        ax.set_xlabel("truncation rank r")
        ax.set_ylabel("field RMSE (m/s)")
        ax.set_title(name)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="best")

    axc = axes[1]
    axc.plot(r, d["mean_nis"], "o-", color="C2", lw=1.4)
    axc.axhline(1.0, color="0.3", ls=":", lw=1.0)
    axc.set_xscale("log")
    axc.set_xticks(r); axc.set_xticklabels([str(x) for x in r], fontsize=8)
    axc.set_xlabel("truncation rank r")
    axc.set_ylabel("mean NIS")
    axc.set_title("(b) innovation consistency vs rank")
    axc.grid(alpha=0.3, which="both")

    out = args.out
    if out is None:
        out = str(figure_dir() / (pathlib.Path(args.csv).stem + ".png"))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
