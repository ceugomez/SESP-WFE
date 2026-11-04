# plot_kf_rank_sweep_folds.py
# cgf  2026-07-14
#
# LOO-fold aggregate of the rank sweep (R1 across held-out members): wake and
# full-domain KF RMSE vs truncation rank r, one thin line per fold plus the
# fold mean, with each fold's Θ-baseline shown as the paired reference (thin
# horizontal ticks at the right edge + fold-mean line). Third panel: NIS vs r
# per fold. Reads every kf_rank_sweep_truth_<id>.csv in the results dir
# (stride-tagged files excluded unless --stride is given) and the matching
# theta_baseline_truth_<id>.csv.
#
# Usage:
#   python plot_kf_rank_sweep_folds.py [--dir results/kf] [--stride N] [--out PNG]

import argparse
import pathlib
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=None,
                   help="results/kf directory (default: repo results/kf)")
    p.add_argument("--stride", type=int, default=None,
                   help="use the _strideN sweep files instead of stride-1")
    p.add_argument("--out", default=None, help="output PNG")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[2]
    kfdir = pathlib.Path(args.dir) if args.dir else repo / "results" / "kf"
    tag = f"_stride{args.stride}" if args.stride else ""
    pat = re.compile(rf"kf_rank_sweep_truth_(\d\d|truth){re.escape(tag)}\.csv$")

    folds = {}
    for f in sorted(kfdir.glob(f"kf_rank_sweep_truth_*{tag}.csv")):
        m = pat.match(f.name)
        if not m:
            continue
        fid = m.group(1)
        d = np.genfromtxt(f, delimiter=",", names=True)
        theta_f = kfdir / f"theta_baseline_truth_{fid}.csv"
        theta = np.genfromtxt(theta_f, delimiter=",", names=True) \
            if theta_f.is_file() else None
        folds[fid] = (d, theta)
    if not folds:
        raise SystemExit(f"no kf_rank_sweep_truth_*{tag}.csv found in {kfdir}")
    print(f"folds found: {', '.join(folds)}")

    r = np.asarray(next(iter(folds.values()))[0]["r"], dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    for ax, region, name in ((axes[0], "full", "(a) full analysis domain"),
                             (axes[1], "wake", "(b) wake region")):
        kf_all, tb_all, cw_all = [], [], []
        for fid, (d, theta) in folds.items():
            ax.plot(r, d[f"kf_rmse_{region}"], "-", color="C0", lw=0.7, alpha=0.35)
            kf_all.append(d[f"kf_rmse_{region}"])
            if theta is not None:
                tb = np.mean(theta[f"theta_rmse_{region}"])
                tb_all.append(tb)
                ax.plot([r[-1] * 1.05], [tb], "_", color="C1", ms=7, alpha=0.6)
                if f"const_rmse_{region}" in (theta.dtype.names or ()):
                    cw_all.append(np.mean(theta[f"const_rmse_{region}"]))
        kf_mean = np.mean(kf_all, axis=0)
        ax.plot(r, kf_mean, "o-", color="C0", lw=1.8,
                label=f"KF, fold mean (min {kf_mean.min():.3f} @ r={r[np.argmin(kf_mean)]})")
        if tb_all:
            ax.axhline(np.mean(tb_all), color="C1", ls="-", lw=1.4,
                       label=f"Θ-baseline, fold mean [{np.mean(tb_all):.3f}]")
        if cw_all:
            ax.axhline(np.mean(cw_all), color="C4", ls="--", lw=1.4,
                       label=f"constant wind, fold mean [{np.mean(cw_all):.3f}]")
        ax.set_xscale("log")
        ax.set_xticks(r); ax.set_xticklabels([str(x) for x in r], fontsize=8)
        ax.set_xlabel("truncation rank r")
        ax.set_ylabel("field RMSE (m/s)")
        ax.set_title(name)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="best")

    axc = axes[2]
    nis_all = []
    for fid, (d, _) in folds.items():
        axc.plot(r, d["mean_nis"], "-", color="C2", lw=0.7, alpha=0.35)
        nis_all.append(d["mean_nis"])
    axc.plot(r, np.mean(nis_all, axis=0), "o-", color="C2", lw=1.8,
             label="fold mean")
    axc.axhline(1.0, color="0.3", ls=":", lw=1.0)
    axc.set_xscale("log")
    axc.set_xticks(r); axc.set_xticklabels([str(x) for x in r], fontsize=8)
    axc.set_xlabel("truncation rank r")
    axc.set_ylabel("mean NIS")
    axc.set_title("(c) innovation consistency vs rank")
    axc.grid(alpha=0.3, which="both")
    axc.legend(fontsize=7.5, loc="best")

    out = args.out
    if out is None:
        fig_dir = repo / "figure"
        fig_dir.mkdir(exist_ok=True)
        out = str(fig_dir / f"kf_rank_sweep_folds{tag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
