#!/usr/bin/env python3
"""Montage across LOO folds: one row per fold, columns = truth / estimate Prα / oracle Prα* /
estimate-truth error. Lets you scan whether the deficit is recovered and where the spurious
fast-mode artifacts appear, across all folds at once. Reads kf_field_snapshot_truth_<ID>.nc."""
import glob, os, re, numpy as np, netCDF4 as nc, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sesp_paths import FIELDS_DIR, figure_dir
FIELDS = str(FIELDS_DIR)
FIGURE = str(figure_dir())
files = sorted(glob.glob(f"{FIELDS}/kf_field_snapshot_truth_*.nc"))
# order: truth first, then 00,01,...
def key(f):
    t = re.search(r"truth_(\w+)\.nc", f).group(1)
    return (0, "") if t == "truth" else (1, t)
files.sort(key=key)

cols = ["truth", "est", "oracle", "err"]
titles = ["truth  x*", "estimate  Prα", "oracle  Prα*", "estimate − truth"]
nrow = len(files)
fig, ax = plt.subplots(nrow, 4, figsize=(15, 3.2 * nrow), squeeze=False)

for r, f in enumerate(files):
    d = nc.Dataset(f)
    tid = d.truth_id
    x = d["xPos"][:] / 1000.0; y = d["yPos"][:] / 1000.0
    mask = d["mask"][:].astype(bool)
    sxyz = d["sensor_xyz"][:] / 1000.0 if "sensor_xyz" in d.variables else None
    def fld(name):
        a = np.array(d[f"{name}_speed"][:], float); a[~mask] = np.nan; return a
    truth, est, oracle = fld("truth"), fld("est"), fld("oracle")
    err = est - truth
    vmin, vmax = np.nanpercentile(truth, 1), np.nanpercentile(truth, 99)
    em = np.nanpercentile(np.abs(err), 99)
    panels = [(truth, vmin, vmax, "viridis"), (est, vmin, vmax, "viridis"),
              (oracle, vmin, vmax, "viridis"), (err, -em, em, "RdBu_r")]
    for c, (F, lo, hi, cmap) in enumerate(panels):
        a = ax[r][c]
        pc = a.pcolormesh(x, y, F.T, shading="auto", vmin=lo, vmax=hi, cmap=cmap)
        fig.colorbar(pc, ax=a, shrink=0.8)
        if sxyz is not None:
            a.scatter(sxyz[:, 0], sxyz[:, 1], c="k", marker="*", s=90,
                      edgecolors="w", linewidths=0.6, zorder=5)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title(titles[c], fontsize=12)
        if c == 0: a.set_ylabel(f"fold {tid}", fontsize=11)

fig.suptitle("KF wind-field estimate across leave-one-out folds (★ = sensors, 50 m AGL)",
             fontsize=14, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.995])
out = f"{FIGURE}/kf_field_snapshot_grid.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out, "from", len(files), "folds")
