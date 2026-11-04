# plot_kf_rank_sweep_dense.py
# cgf  2026-07-15
#
# Rank-sensitivity figure for the dense sensor array (NM=175, 500 m spacing).
# Same content as plot_kf_rank_sweep.py plus the gappy-POD conditioning
# diagnostic cond(H·Pr) vs r, which explains where the KF departs from the
# subspace-oracle floor.
#
# Usage:
#   python plot_kf_rank_sweep_dense.py <kf_rank_sweep_dense_truth_<id>.csv> \
#       [--theta theta_baseline_truth_<id>.csv] [--out PNG]

import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="kf_rank_sweep_dense_truth_<id>.csv")
    p.add_argument("--theta", default=None,
                   help="theta_baseline_truth_<id>.csv (default: inferred by filename swap)")
    p.add_argument("--out", default=None, help="output PNG (default: figure/ alongside repo)")
    args = p.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    r = np.asarray(d["r"], dtype=int)

    # Prefer the dense-array baseline (matched sensor config); the rake file
    # theta_baseline_truth_<id>.csv is a 45-obs-row mismatch — refuse to fall
    # back to it silently.
    theta_path = args.theta or args.csv.replace("kf_rank_sweep_dense_",
                                                "theta_baseline_dense_")
    theta = None
    if pathlib.Path(theta_path).is_file():
        theta = np.genfromtxt(theta_path, delimiter=",", names=True)
    else:
        print(f"WARNING: no dense Θ-baseline at {theta_path}; "
              "omitting Θ lines (pass --theta to override)")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    for ax, region, name in ((axes[0, 0], "full", "(a) full analysis domain"),
                             (axes[0, 1], "wake", "(b) wake region")):
        ax.plot(r, d[f"kf_rmse_{region}"], "o-", color="C0", lw=1.4,
                label="reduced-order KF")
        ax.plot(r, d[f"oracle_rmse_{region}"], "s:", color="0.4", lw=1.2,
                label="subspace oracle (floor)")
        if theta is not None:
            tb = np.nanmean(theta[f"theta_rmse_{region}"])
            to = np.nanmean(theta[f"thetaoracle_rmse_{region}"])
            ax.axhline(tb, color="C1", ls="-", lw=1.2,
                       label=f"Θ-baseline [{tb:.3f}]")
            ax.axhline(to, color="C1", ls="--", lw=1.0,
                       label=f"Θ-oracle [{to:.3f}]")
            if f"const_rmse_{region}" in (theta.dtype.names or ()):
                cw = np.nanmean(theta[f"const_rmse_{region}"])
                ax.axhline(cw, color="C4", ls="--", lw=1.2,
                           label=f"constant wind [{cw:.3f}]")
        i_best = int(np.argmin(d[f"kf_rmse_{region}"]))
        ax.annotate(f"r={r[i_best]}", (r[i_best], d[f"kf_rmse_{region}"][i_best]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=8, color="C0")
        ax.set_xscale("log")
        ax.set_xticks(r); ax.set_xticklabels([str(x) for x in r], fontsize=8)
        ax.set_xlabel("truncation rank r")
        ax.set_ylabel("field RMSE (m/s)")
        ax.set_title(name)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="best")

    axc = axes[1, 0]
    axc.plot(r, d["mean_nis"], "o-", color="C2", lw=1.4)
    axc.axhline(1.0, color="0.3", ls=":", lw=1.0)
    axc.set_xscale("log")
    axc.set_xticks(r); axc.set_xticklabels([str(x) for x in r], fontsize=8)
    axc.set_xlabel("truncation rank r")
    axc.set_ylabel("mean NIS")
    axc.set_title("(c) innovation consistency vs rank")
    axc.grid(alpha=0.3, which="both")

    axd = axes[1, 1]
    axd.plot(r, d["cond_HPr"], "o-", color="C3", lw=1.4)
    axd.set_xscale("log"); axd.set_yscale("log")
    axd.set_xticks(r); axd.set_xticklabels([str(x) for x in r], fontsize=8)
    axd.set_xlabel("truncation rank r")
    axd.set_ylabel(r"cond$(H P_r)$")
    axd.set_title("(d) observability conditioning vs rank")
    axd.grid(alpha=0.3, which="both")

    out = args.out
    if out is None:
        repo_fig = pathlib.Path(__file__).resolve().parents[2] / "figure"
        repo_fig.mkdir(exist_ok=True)
        out = str(repo_fig / (pathlib.Path(args.csv).stem + ".png"))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
