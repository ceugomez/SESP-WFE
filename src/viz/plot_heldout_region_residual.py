# plot_heldout_region_residual.py
# cgf 2026-06-26
#
# Leave-one-member-out residual energy vs. truncation rank, by region and component.
# Reads results/heldout/heldout_region_residual.csv (region,rank,mean,<member cols...>) written by
# scripts/make_heldout_region_residual.jl. Four panels, shared log-y:
#   (1) Full analysis domain   (headline closure curve)
#   (2) Wake region: all-comp / horizontal (u,v) / w-only overlaid
#   (3) Ambient (outside wake) (easy-case reference)
#   (4) w-only across regions  (the hazard variable, hardest to represent)
# Thin light lines = per held-out member; bold = LOO mean. Threshold markers on the mean
# (90/95/99% captured = 0.10/0.05/0.01 residual). Vertical line at the operating rank.
#
# y = fraction of region FLUCTUATION energy left UNREPRESENTED (block-weighted W-norm,
# w scaled up to u,v). The endpoint = the operating-rank floor: the ceiling on any filter.
#
# Usage: python plot_heldout_region_residual.py <csv> [--op-rank 50] [--out PNG] [--linear]

import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLDS = [(0.10, "90%"), (0.05, "95%"), (0.01, "99%")]


def load(path):
    rows = {}
    members = None
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        members = header[3:]
        for line in r:
            region, rank, mean = line[0], int(line[1]), float(line[2])
            per = [float(x) for x in line[3:]]
            rows.setdefault(region, {"rank": [], "mean": [], "members": []})
            rows[region]["rank"].append(rank)
            rows[region]["mean"].append(mean)
            rows[region]["members"].append(per)
    for region in rows:
        rows[region]["rank"] = np.array(rows[region]["rank"])
        rows[region]["mean"] = np.array(rows[region]["mean"])
        rows[region]["members"] = np.array(rows[region]["members"])  # (L, M)
    return rows, members


def plot_group(ax, data, region, color, label, log, show_members=True):
    rk = data[region]["rank"]
    mu = data[region]["mean"]
    if show_members:
        for k in range(data[region]["members"].shape[1]):
            ax.plot(rk, data[region]["members"][:, k], color=color, lw=0.6, alpha=0.30)
    ax.plot(rk, mu, color=color, lw=2.4, label=label)
    return rk, mu


def mark_thresholds(ax, rk, mu, color):
    for thr, name in THRESHOLDS:
        below = np.where(mu <= thr)[0]
        if below.size:
            r = rk[below[0]]
            ax.plot([r], [mu[below[0]]], "o", color=color, ms=5, mec="k", mew=0.5, zorder=5)
            ax.annotate(f"{name}\n@r={r}", (r, mu[below[0]]), textcoords="offset points",
                        xytext=(4, 6), fontsize=7, color=color)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--op-rank", type=int, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--linear", action="store_true", help="linear y instead of log")
    args = p.parse_args()

    data, members = load(args.csv)
    log = not args.linear
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    # Panel 1: full domain
    ax = axes[0]
    rk, mu = plot_group(ax, data, "full_all", "C0", "full domain (mean)", log)
    mark_thresholds(ax, rk, mu, "C0")
    ax.set_title("Full analysis domain")
    ax.set_ylabel("fraction of fluctuation energy unrepresented\n(block-weighted W-norm)")

    # Panel 2: wake, component split
    ax = axes[1]
    for region, color, lab in [("wake_all", "C1", "wake: all comp"),
                               ("wake_uv", "C2", "wake: horiz (u,v)"),
                               ("wake_w", "C3", "wake: w-only")]:
        rk, mu = plot_group(ax, data, region, color, lab, log)
    mark_thresholds(ax, data["wake_w"]["rank"], data["wake_w"]["mean"], "C3")
    ax.set_title("Wake region (component split)")

    # Panel 3: ambient
    ax = axes[2]
    rk, mu = plot_group(ax, data, "ambient_all", "C4", "ambient (mean)", log)
    mark_thresholds(ax, rk, mu, "C4")
    ax.set_title("Ambient (outside wake)")

    # Panel 4: w-only across regions
    ax = axes[3]
    for region, color, lab in [("full_all", "C0", "full (all comp)"),
                               ("wake_w", "C3", "wake w-only"),
                               ("ambient_all", "C4", "ambient (all comp)")]:
        plot_group(ax, data, region, color, lab, log, show_members=False)
    ax.set_title("w-only / region comparison")

    for ax in axes:
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("truncation rank r")
        ax.axvline(args.op_rank, color="0.3", ls="--", lw=1.2)
        ax.text(args.op_rank, ax.get_ylim()[1], " operating r", rotation=90,
                va="top", ha="right", fontsize=7, color="0.3")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    # headline number in the suptitle
    wkw = data["wake_w"]
    ro = min(args.op_rank, wkw["rank"][-1])
    idx = np.where(wkw["rank"] == ro)[0][0]
    vals = wkw["members"][idx, :]
    fig.suptitle(
        f"LOO residual energy vs rank  —  at operating r={ro}, "
        f"{100*vals.mean():.1f}% ± {100*vals.std():.1f}% of held-out WAKE w-fluctuation "
        f"energy is unrepresentable (n={len(members)} members)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out or os.path.join(os.path.dirname(args.csv), "heldout_region_residual.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
