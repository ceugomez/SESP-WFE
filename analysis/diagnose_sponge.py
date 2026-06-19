# diagnose_sponge.py
# cgf / assistant  2026-06-10
#
# Lightweight lateral-sponge diagnostic for FastEddy output.
# Removes the geostrophic base state to expose PERTURBATIONS (wake + any
# boundary reflections), overlays the lateral sponge zones, and plots the
# perturbation-energy decay toward each face so you can see whether the wake
# is absorbed BEFORE it reaches the boundary.
#
# Usage:
#   python diagnose_sponge.py <dir> [--zagl 50] [--width 40] [--Ug 9.961947] [--Vg 0.871557]

import argparse, os, glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# reuse the AGL interpolation from the user's plot_xy_slices.py
from plot_xy_slices import interp_to_zagl_fast


def diagnose(nc_path, z_agl, width, Ug, Vg, dx, out_dir):
    with nc.Dataset(nc_path) as ds:
        zpos = ds["zPos"][0, :, :, :]
        topo = ds["topoPos"][0, :, :]
        u = np.array(ds["u"][0], dtype=np.float32)
        v = np.array(ds["v"][0], dtype=np.float32)
        w = np.array(ds["w"][0], dtype=np.float32)

    # base state: geostrophic (Ug,Vg) for momentum, 0 for w
    up = interp_to_zagl_fast(u, zpos, topo, z_agl) - Ug
    vp = interp_to_zagl_fast(v, zpos, topo, z_agl) - Vg
    wp = interp_to_zagl_fast(w, zpos, topo, z_agl)          # base w = 0
    tke = 0.5 * (up**2 + vp**2 + wp**2)                     # perturbation KE (m2/s2)

    Ny, Nx = tke.shape
    xkm = (np.arange(Nx) * dx) / 1e3
    ykm = (np.arange(Ny) * dx) / 1e3
    wkm = width * dx / 1e3                                  # sponge width in km
    stem = os.path.basename(nc_path)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    # --- Panel A: perturbation KE top-down with sponge zones -----------------
    lim = max(0.05, np.nanpercentile(tke, 99))
    pc = ax[0].pcolormesh(xkm, ykm, tke, cmap="magma_r", vmin=0, vmax=lim, shading="auto")
    fig.colorbar(pc, ax=ax[0], pad=0.02, fraction=0.046).set_label("perturbation KE  (m² s⁻²)")
    ax[0].contour(xkm, ykm, topo, levels=np.arange(10, max(11, topo.max()), 15),
                  colors="c", linewidths=0.5, alpha=0.7)
    # sponge zones (4 faces)
    sp = dict(facecolor="none", edgecolor="lime", lw=1.4, ls="--")
    ax[0].add_patch(Rectangle((0, 0), wkm, ykm[-1], **sp))                 # west
    ax[0].add_patch(Rectangle((xkm[-1]-wkm, 0), wkm, ykm[-1], **sp))       # east
    ax[0].add_patch(Rectangle((0, 0), xkm[-1], wkm, **sp))                 # south
    ax[0].add_patch(Rectangle((0, ykm[-1]-wkm), xkm[-1], wkm, **sp))       # north
    ax[0].set(xlabel="x (km)", ylabel="y (km)", aspect="equal",
              title=f"perturbation KE @ {z_agl:.0f} m AGL\n(green dashed = {wkm:.1f} km sponge)")

    # --- Panel B: perturbation-energy decay toward each face -----------------
    # mean KE along rows/cols vs distance into the domain from each face
    ke_x = np.nanmean(tke, axis=0)     # vs x (averaged over y)
    ke_y = np.nanmean(tke, axis=1)     # vs y (averaged over x)
    ax[1].plot(xkm, ke_x, color="tab:blue", label="⟨KE⟩ vs x (W↔E faces)")
    ax[1].plot(ykm, ke_y, color="tab:orange", label="⟨KE⟩ vs y (S↔N faces)")
    ax[1].axvspan(0, wkm, color="lime", alpha=0.15)
    ax[1].axvspan(xkm[-1]-wkm, xkm[-1], color="lime", alpha=0.15)
    ax[1].set(xlabel="distance (km)", ylabel="⟨perturbation KE⟩ (m² s⁻²)",
              title="face decay — wake should fall to ~0 inside the\nshaded sponge before the boundary")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    fig.suptitle(stem, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = os.path.join(out_dir, f"{stem}_sponge.png")
    fig.savefig(png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    p.add_argument("--zagl",  type=float, default=50.0)
    p.add_argument("--width", type=int,   default=40)      # sponge width in cells
    p.add_argument("--dx",    type=float, default=10.0)    # m
    p.add_argument("--Ug",    type=float, default=9.961947)
    p.add_argument("--Vg",    type=float, default=0.871557)
    p.add_argument("--outdir", default=None)
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.output_dir, "outputFE_*")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"No outputFE_* files in {a.output_dir}")
    out_dir = a.outdir or os.path.join(a.output_dir, "sponge_diag")
    os.makedirs(out_dir, exist_ok=True)
    print(f"sponge diagnostic: {len(files)} files @ {a.zagl} m AGL, width={a.width} cells → {out_dir}")
    for f in files:
        diagnose(f, a.zagl, a.width, a.Ug, a.Vg, a.dx, out_dir)
