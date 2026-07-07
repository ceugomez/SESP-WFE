# plot_heldout_residual_curve.py
# cgf 2026-06-27
#
# Leave-one-member-out residual-energy vs. truncation rank (single full-domain curve).
# Reads results/heldout/heldout_residual_curve.csv written by scripts/run_heldout_residual.jl:
#   rank,resid_mean,<member cols...>
# Thin light lines = per held-out member; bold = LOO mean. Threshold markers on the
# mean (90/95/99% captured = 0.10/0.05/0.01 residual). Vertical line at operating rank.
#
# y = fraction of fluctuation energy left UNREPRESENTED at rank r; the endpoint is the
# operating-rank floor (ceiling on any filter built from this basis).
#
# Usage: python plot_heldout_residual_curve.py <csv> [--op-rank 50] [--out PNG] [--linear]

import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLDS = [(0.10, "90%"), (0.05, "95%"), (0.01, "99%")]


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        members = header[2:]
        rank, mean, per = [], [], []
        for line in r:
            rank.append(int(line[0]))
            mean.append(float(line[1]))
            per.append([float(x) for x in line[2:]])
    return np.array(rank), np.array(mean), np.array(per), members


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--op-rank", type=int, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--linear", action="store_true", help="linear y instead of log")
    args = p.parse_args()

    rank, mean, per, members = load(args.csv)
    log = not args.linear

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(per.shape[1]):
        ax.plot(rank, per[:, k], color="C0", lw=0.6, alpha=0.30)
    ax.plot(rank, mean, color="C0", lw=2.4, label=f"LOO mean (n={len(members)})")

    for thr, name in THRESHOLDS:
        below = np.where(mean <= thr)[0]
        if below.size:
            r = rank[below[0]]
            ax.plot([r], [mean[below[0]]], "o", color="C3", ms=6, mec="k", mew=0.5, zorder=5)
            ax.annotate(f"{name}\n@r={r}", (r, mean[below[0]]), textcoords="offset points",
                        xytext=(5, 7), fontsize=8, color="C3")

    if log:
        ax.set_yscale("log")
    ax.axvline(args.op_rank, color="0.3", ls="--", lw=1.2)
    ax.text(args.op_rank, ax.get_ylim()[1], " operating r", rotation=90,
            va="top", ha="right", fontsize=8, color="0.3")
    ax.set_xlabel("truncation rank r")
    ax.set_ylabel("fraction of fluctuation energy unrepresented")
    ax.set_title("Leave-one-member-out residual energy vs. rank")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    ro = min(args.op_rank, rank[-1])
    idx = np.where(rank == ro)[0][0]
    vals = per[idx, :]
    fig.suptitle(f"at operating r={ro}: {100*vals.mean():.1f}% ± {100*vals.std():.1f}% "
                 f"unrepresentable", fontsize=10, y=0.98)
    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(args.csv), "heldout_residual_curve.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
