#!/usr/bin/env python3
"""Plot the filter's actual reconstructed wind field at one snapshot, vs truth, oracle
(truth-projected), and the ensemble-mean prior. Answers: does the estimate carry the wake
deficit, or just the bulk wind? Reads results/fields/kf_field_snapshot_truth_<ID>.nc."""
import argparse, os, numpy as np, netCDF4 as nc, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sesp_paths import FIELDS_DIR, figure_dir
FIELDS = str(FIELDS_DIR)
FIGURE = str(figure_dir())

p = argparse.ArgumentParser()
p.add_argument("id", nargs="?", default="truth")
p.add_argument("--level-step", type=float, default=0.5, help="contour level spacing for speed panels (m/s)")
p.add_argument("--level-step-diff", type=float, default=0.25, help="contour level spacing for increment/error panels (m/s)")
args = p.parse_args()

ID = args.id
src = os.path.join(FIELDS, f"kf_field_snapshot_truth_{ID}.nc")
d   = nc.Dataset(src)
x = d["xPos"][:] / 1000.0; y = d["yPos"][:] / 1000.0          # km
mask = d["mask"][:].astype(bool)
zagl = float(d.z_agl_m); kidx = int(d.snapshot_k)
sxyz = d["sensor_xyz"][:] / 1000.0 if "sensor_xyz" in d.variables else None

def fld(name):
    a = np.array(d[f"{name}_speed"][:], float)
    a[~mask] = np.nan
    return a

truth, est, oracle, prior = (fld(n) for n in ("truth", "est", "oracle", "prior"))
incr = est - prior   # what assimilation added on top of the static prior

# common scale for the speed panels from the truth field
vmin = np.nanpercentile(truth, 1); vmax = np.nanpercentile(truth, 99)

fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
panels = [("truth  x*", truth, vmin, vmax, "viridis"),
          ("filter estimate  Prα", est, vmin, vmax, "viridis"),
          ("oracle  Prα*  (truth projected)", oracle, vmin, vmax, "viridis"),
          ("ensemble-mean prior  Prα₀", prior, vmin, vmax, "viridis"),
          ("estimate − prior  (assimilation increment)", incr, None, None, "RdBu_r"),
          ("estimate − truth  (filter error)", est - truth, None, None, "RdBu_r")]

for a, (title, F, lo, hi, cmap) in zip(ax.flat, panels):
    if lo is None:
        step = args.level_step_diff
        m = np.ceil(np.nanpercentile(np.abs(F), 99) / step) * step
        lo, hi = -m, m
        levels = np.arange(lo, hi + step, step)
    else:
        step = args.level_step
        lo = np.floor(lo / step) * step; hi = np.ceil(hi / step) * step
        levels = np.arange(lo, hi + step, step)
    pc = a.contourf(x, y, F.T, levels=levels, vmin=lo, vmax=hi, cmap=cmap, extend="both")
    a.contour(x, y, F.T, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.5)
    fig.colorbar(pc, ax=a, shrink=0.85, label="speed (m/s)")
    if sxyz is not None:
        a.scatter(sxyz[:, 0], sxyz[:, 1], c="k", marker="*", s=180,
                  edgecolors="w", linewidths=0.8, zorder=5)
    a.set_title(title, fontsize=11); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
    a.set_aspect("equal")

fig.suptitle(f"KF wind-field estimate — fold {ID}, snapshot k={kidx}, z={zagl:.0f} m AGL "
             f"(★ = sensors)", fontsize=13)
out = os.path.join(FIGURE, f"kf_field_snapshot_{ID}.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
