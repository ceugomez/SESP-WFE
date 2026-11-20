# plot_ensemble_distribution.py
# cgf cego6160@colorado.edu
#
# Plot the 2D Gaussian prior over (wind speed, wind direction) as filled
# contours, and overlay scatter points for the sampled ensemble members
# and the truth member. Reads ensemble_config.json written by
# create_synthetic_ensemble_sim.py.

#
# usage: python plot_ensemble_distribution.py [ensemble-lowvar] [--dir-halfwidth 80]
#   ensemble name defaults to $SESP_ENSEMBLE, then ensemble-highvar.
#   --dir-halfwidth fixes the direction axis to mu +/- that many degrees so
#   ensembles with different sigma_dir are visually comparable (the default
#   +/-4 sigma autoscale makes every same-seed LHS ensemble look identical).

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# AMS figure style: sans-serif (Arial/Helvetica), 7-8 pt text at final size, TrueType in PDF
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.default": "regular",
})

REPO_ROOT = "/home/cego6160/workspace/ensemble_runs_paper"
FIG_DIR   = os.path.join(REPO_ROOT, "SESP-WFE", "preprocessing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble", nargs="?",
                    default=os.environ.get("SESP_ENSEMBLE", "ensemble-highvar"))
    ap.add_argument("--dir-halfwidth", type=float, default=None,
                    help="fixed direction-axis half-width [deg] about the prior mean")
    args = ap.parse_args()

    config_in = os.path.join(REPO_ROOT, args.ensemble, "ensemble_config.json")
    tag = args.ensemble.removeprefix("ensemble-")
    fig_out = os.path.join(FIG_DIR, f"ensemble_distribution_{tag}.png")
    print(f"reading {config_in}")
    with open(config_in) as f:
        cfg = json.load(f)

    ens   = cfg["ensemble"]
    mu    = np.array([ens["mu_spd_ms"],   ens["mu_dir_deg"]])
    sigma = np.array([ens["sigma_spd_ms"], ens["sigma_dir_deg"]])

    members = cfg["prior_members"]
    truth   = cfg["truth_member"]

    m_spd = np.array([m["spd_ms"]      for m in members])
    m_dir = np.array([m["met_dir_deg"] for m in members])

    # --- Gaussian grid (independent dims -> product of 1D Gaussians) ---
    spd = np.linspace(mu[0] - 4 * sigma[0], mu[0] + 4 * sigma[0], 400)
    hw = args.dir_halfwidth if args.dir_halfwidth else 4 * sigma[1]
    dir_ = np.linspace(mu[1] - hw, mu[1] + hw, 400)
    SPD, DIR = np.meshgrid(spd, dir_)
    # Mahalanobis radius in units of sigma (prior is diagonal)
    R = np.hypot((SPD - mu[0]) / sigma[0], (DIR - mu[1]) / sigma[1])

    # --- Plot (AMS: single column = 19 pc = 3.2 in; no title, caption goes in LaTeX) ---
    fig, ax = plt.subplots(figsize=(3.2, 3.0), constrained_layout=True)

    # Density is a monotone function of R, so discrete 1/2/3-sigma bands carry the same
    # information as a colorbar of unnormalized density, and read in grayscale.
    bands = ["#c6dbef", "#9ecae1", "#6baed6"]   # outside 3, 2-3, 1-2 sigma ... inner below
    ax.contourf(DIR, SPD, R, levels=[0, 1, 2, 3], colors=bands[::-1], zorder=0)
    ax.contour(DIR, SPD, R, levels=[1, 2, 3], colors="0.35", linewidths=0.5,
               linestyles="--", zorder=1)
    # label the rings along the +speed axis
    for k in (1, 2, 3):
        ax.annotate(f"{k}$\\sigma$", (mu[1], mu[0] + k * sigma[0]), xytext=(0, 1.5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, color="0.3", zorder=2)

    ax.scatter(m_dir, m_spd, s=14, c="k", zorder=5, linewidths=0, label="Ensemble members")
    ax.scatter(truth["met_dir_deg"], truth["spd_ms"], marker="*", s=70, c="white",
               edgecolor="k", linewidths=0.7, zorder=6, label="Truth")
    ax.scatter(mu[1], mu[0], marker="+", s=36, c="k", linewidths=0.9, zorder=7,
               label="Prior mean")

    ax.set_xlim(dir_[0], dir_[-1])
    ax.set_ylim(spd[0], spd[-1])
    ax.set_box_aspect(1)   # equal sigma-units per axis -> rings render as circles
    ax.set_xlabel("Wind direction (\u00b0)")
    ax.set_ylabel("Wind speed (m s$^{-1}$)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.tick_params(direction="out", length=3, width=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.7",
              fancybox=False, borderpad=0.4, handletextpad=0.3, labelspacing=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.splitext(fig_out)[0] + "." + ext, dpi=300)
    print(f"wrote {os.path.splitext(fig_out)[0]}.pdf|png")


if __name__ == "__main__":
    main()
