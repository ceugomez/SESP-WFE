# plot_rep_calibration_riccati.py
# cgf 2026-07-22
# Rep-error calibration study, Riccati side: per-mode posterior/prior variance ratio
# P_inf/Sigma0 at r=50 for the three R-model arms (none/diag/full) on the fixed config
# fold 03 / LOO basis, from run_cov_rep_arms.sh. The claim the figure carries: ignoring
# rep error (none) makes the filter claim to observe modes past the observability edge
# that the proper model (full) correctly refuses to; diag isolates the correlation
# structure from the variance level. Panels: (a) dense 500 m lattice, (b) 15-sensor rake.

import pathlib
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sesp_paths import COV_DIR, figure_dir

cov = COV_DIR
R = 50
ARMS = [("none", "C3", "-", r"$R_{\rm sensor}$ only (rep ignored)"),
        ("diag", "C1", "--", r"$+\,{\rm diag}(R_{\rm rep})$"),
        ("full", "C0", "-", r"$+\,R_{\rm rep}$ (full)")]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

for ax, sensor, ttl in ((axes[0], "dense", "(a) dense lattice (500 m)"),
                        (axes[1], "rake", "(b) tower rake (15 sensors)")):
    for arm, color, ls, lab in ARMS:
        f = cov / f"cov_fold03_{sensor}_loo_rep{arm}.jld2"
        if not f.is_file():
            print(f"missing {f}"); continue
        with h5py.File(f, "r") as h:
            ratio = np.asarray(h[f"r{R}/ratio"])
            eigmin = float(np.asarray(h["eigmin_Rtot"]))
        if arm == "full" and np.isfinite(eigmin):
            print(f"{sensor}: eigmin(R_sensor + R_rep) = {eigmin:+.3f}"
                  + ("  ** NOT PD — do not quote **" if eigmin < 0 else ""))
        modes = np.arange(1, len(ratio) + 1)
        ax.semilogy(modes, ratio, ls, color=color, marker="o", ms=3, lw=1.4, label=lab)
    ax.axhline(1.0, color="0.3", ls=":", lw=1.2)
    ax.axvline(13, color="0.5", ls="-", lw=0.8)
    ax.set_title(ttl)
    ax.set_xlabel("POD mode index")
    ax.grid(alpha=0.3, which="both")

axes[0].set_ylabel(r"posterior/prior variance  $P_\infty/\Sigma_0$")
axes[0].legend(fontsize=8, loc="upper left")
axes[0].text(13.5, axes[0].get_ylim()[0] * 2, "mode 13\n(meander)", fontsize=7, color="0.4")
axes[0].text(2, 1.6, "prior (no info) →", fontsize=7, color="0.3")
axes[1].legend(fontsize=8, loc="upper left")
axes[1].set_ylabel(r"$P_\infty/\Sigma_0$")

out = figure_dir() / "rep_calibration_riccati.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
