# plot_ensemble_distribution.py
# cgf cego6160@colorado.edu
#
# Plot the 2D Gaussian prior over (wind speed, wind direction) as filled
# contours, and overlay scatter points for the sampled ensemble members
# and the truth member. Reads ensemble_config.json written by
# create_synthetic_ensemble_sim.py.

import json
import os
import numpy as np
import matplotlib.pyplot as plt

CONFIG_IN = "/home/cego6160/workspace/ensemble_runs_paper/ensemble/ensemble_config.json"
FIG_OUT   = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/preprocessing/ensemble_distribution.png"


def main():
    with open(CONFIG_IN) as f:
        cfg = json.load(f)

    ens   = cfg["ensemble"]
    mu    = np.array([ens["mu_spd_ms"],   ens["mu_dir_deg"]])
    sigma = np.array([ens["sigma_spd_ms"], ens["sigma_dir_deg"]])

    members = cfg["prior_members"]
    truth   = cfg["truth_member"]

    m_spd = np.array([m["spd_ms"]      for m in members])
    m_dir = np.array([m["met_dir_deg"] for m in members])

    # --- Gaussian grid (independent dims → product of 1D Gaussians) ---
    spd = np.linspace(mu[0] - 4 * sigma[0], mu[0] + 4 * sigma[0], 300)
    dir_ = np.linspace(mu[1] - 4 * sigma[1], mu[1] + 4 * sigma[1], 300)
    SPD, DIR = np.meshgrid(spd, dir_)
    Z = (np.exp(-0.5 * ((SPD - mu[0]) / sigma[0]) ** 2) *
         np.exp(-0.5 * ((DIR - mu[1]) / sigma[1]) ** 2))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(DIR, SPD, Z, levels=20, cmap="Blues")
    fig.colorbar(cf, ax=ax, label="prior density (unnormalized)")

    # 1/2/3-sigma rings as guide (in std units, plotted as ellipses via contour)
    Zsig = np.sqrt(((SPD - mu[0]) / sigma[0]) ** 2 + ((DIR - mu[1]) / sigma[1]) ** 2)
    ax.contour(DIR, SPD, Zsig, levels=[1, 2, 3], colors="0.4",
               linewidths=0.8, linestyles="--")

    ax.scatter(m_dir, m_spd, c="k", s=40, zorder=5, label="ensemble members")
    ax.scatter(truth["met_dir_deg"], truth["spd_ms"], c="red", marker="*",
               s=300, edgecolor="k", zorder=6, label="truth")
    ax.scatter(mu[1], mu[0], c="white", marker="+", s=120, zorder=6, label="prior mean")

    ax.set_xlabel("wind direction [deg]")
    ax.set_ylabel("wind speed [m/s]")
    ax.set_title("Prior Gaussian over wind speed & direction")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
