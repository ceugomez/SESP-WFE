#!/usr/bin/env python3
"""One figure per frame: truth | KF estimate | estimate-truth (xy speed) for the scratchpad
lowvar-truth / highvar-3-member-basis run. Reads results/scratch/lowvar_truth_highvar_basis.nc."""
import numpy as np, netCDF4 as nc, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sesp_paths import PKG_ROOT
d = nc.Dataset(PKG_ROOT / "results/scratch/lowvar_truth_highvar_basis.nc")
out = PKG_ROOT / "figure/scratch"; out.mkdir(parents=True, exist_ok=True)
X = d["xPos"][:][:, 0] / 1e3; Y = d["yPos"][:][0, :] / 1e3
m = d["mask"][:].astype(bool)[..., None]; sx = d["sensor_xyz"][:] / 1e3
TR = np.where(m, np.array(d["truth"][:], float), np.nan); ES = np.where(m, np.array(d["est"][:], float), np.nan)
lo, hi = np.nanpercentile(TR, 1), np.nanpercentile(TR, 99); em = np.nanpercentile(np.abs(ES - TR), 99)
ttl = (f"lowvar truth ({d.truth_dir_deg:.0f}°), basis = highvar {d.basis_members}, r={int(d.rank)}, z={d.z_agl_m:.0f} m AGL\n"
       f"total RMSE: estimate {d.rmse_est:.3f} | oracle (in-subspace floor) {d.rmse_oracle:.3f} | prior mean {d.rmse_prior:.3f} m/s")
for k in range(TR.shape[2]):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    for a, t, F, cm, v0, v1 in [(ax[0], "truth", TR[..., k], "viridis", lo, hi), (ax[1], "ROKF estimate", ES[..., k], "viridis", lo, hi),
                                (ax[2], "estimate − truth", ES[..., k] - TR[..., k], "RdBu_r", -em, em)]:
        pc = a.pcolormesh(X, Y, F.T, cmap=cm, vmin=v0, vmax=v1, shading="auto"); fig.colorbar(pc, ax=a, label="m/s")
        a.scatter(sx[:, 0], sx[:, 1], c="k", marker="*", s=40, edgecolors="w", linewidths=0.4)
        a.set_aspect("equal"); a.set_title(t); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
    fig.suptitle(f"frame {k+1}/{TR.shape[2]}, t={d['t'][k]:.0f} s\n{ttl}", fontsize=11)
    fn = out / f"lowvar_truth_frame{k+1}.png"; fig.savefig(fn, dpi=120); plt.close(fig); print("wrote", fn)
