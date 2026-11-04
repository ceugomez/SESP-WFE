# plot_filter_covariance.py
# cgf 2026-07-20
# Per-mode posterior/prior variance ratio P_inf/Sigma0 at r=50 for the four configs
# (dense/rake x twin/loo), from scripts/analyze_filter_covariance.jl. Shows the observed
# subspace (ratio << 1), the prior line (ratio = 1), and the Q-inflated tail (ratio > 1),
# with the meander mode (13) marked. Answers "does the filter reduce covariance from prior,
# and for which modes" directly.

import pathlib
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parents[2]
cov = here / "results" / "kf" / "cov_analysis"
R = 50
CONFIGS = [("dense", "twin", "C0", "-"), ("dense", "loo", "C0", "--"),
           ("rake", "twin", "C3", "-"), ("rake", "loo", "C3", "--")]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

for sensor, basis, color, ls in CONFIGS:
    f = cov / f"cov_fold03_{sensor}_{basis}.jld2"
    if not f.is_file():
        print(f"missing {f}"); continue
    with h5py.File(f, "r") as h:
        ratio = np.asarray(h[f"r{R}/ratio"])
    modes = np.arange(1, len(ratio) + 1)
    lab = f"{sensor} / {basis}"
    axL.semilogy(modes, ratio, ls, color=color, marker="o", ms=3, lw=1.4, label=lab)
    axR.semilogy(modes, ratio, ls, color=color, marker="o", ms=3, lw=1.4, label=lab)

for ax in (axL, axR):
    ax.axhline(1.0, color="0.3", ls=":", lw=1.2)
    ax.axhspan(1.0, ax.get_ylim()[1], color="0.85", alpha=0.4, lw=0)  # provisional; reset below
    ax.axvline(13, color="0.5", ls="-", lw=0.8)
    ax.set_xlabel("POD mode index")
    ax.grid(alpha=0.3, which="both")

# redo shading with final ylims
for ax, ttl in ((axL, "(a) all four configs (r=50)"), (axR, "(b) dense only, zoom")):
    ax.set_title(ttl)
axL.set_ylabel(r"posterior/prior variance  $P_\infty/\Sigma_0$")
axL.legend(fontsize=8, loc="upper left")
axL.text(13.5, axL.get_ylim()[0]*2, "mode 13\n(meander)", fontsize=7, color="0.4")
axL.text(2, 1.6, "prior (no info) →", fontsize=7, color="0.3")
axL.text(2, 3e-6, "← observed\n(variance crushed)", fontsize=7, color="0.3")

# right panel: dense-only, so the ~2x meander inflation is visible without the rake 100x swamping it
axR.set_ylim(1e-6, 5)
axR.legend(fontsize=8, loc="upper left")
axR.set_ylabel(r"$P_\infty/\Sigma_0$")

out = here / "figure" / "filter_covariance_reduction.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
