# plot_pod_modes.py
# cgf  2026-06-23
#
# Panel plot of the leading POD modes exported by scripts/export_pod_modes.jl
# (FE_mode_*.0 FastEddy-format NetCDFs). Layout: one row per mode (decreasing
# page-downward), columns u | v | w, each interpolated to a constant height AGL.
# Each panel gets its own symmetric colour scale (mode amplitudes vary by mode
# and component, and POD modes are sign-arbitrary).
#
# Usage:
#   python plot_pod_modes.py <modes_dir> [--zagl Z] [--out PNG]

import argparse
import glob
import os
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the terrain-following AGL interpolation from the slice plotter
from plot_xy_slices import interp_to_zagl_fast

COMPS = ["u", "v", "w"]


def load_mode(nc_path, z_agl):
    """Return (xkm, ykm, topo2d, {comp: 2d slice at z_agl}, lambda)."""
    with nc.Dataset(nc_path) as ds:
        xpos = ds["xPos"][0, 0, :, :]            # (Ny, Nx)
        ypos = ds["yPos"][0, 0, :, :]
        zpos = ds["zPos"][0, :, :, :]            # (Nz, Ny, Nx)
        topo = ds["topoPos"][0, :, :]            # (Ny, Nx)
        lam = float(ds["time"][0])               # eigenvalue stashed in time slot
        sl = {c: interp_to_zagl_fast(np.array(ds[c][0, :, :, :], dtype=np.float32),
                                     zpos, topo, z_agl)
              for c in COMPS}
    return xpos / 1e3, ypos / 1e3, topo, sl, lam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modes_dir", help="dir of FE_mode_*.0 files from export_pod_modes.jl")
    p.add_argument("--zagl", type=float, default=50.0, help="height AGL (m), default 50")
    p.add_argument("--out", default=None, help="output PNG (default: modes_dir/pod_modes.png)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.modes_dir, "FE_mode_*")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"No FE_mode_* files in {args.modes_dir}")

    nrows = len(files)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.4 * nrows),
                             constrained_layout=True, squeeze=False)

    for r, f in enumerate(files):
        xkm, ykm, topo, sl, lam = load_mode(f, args.zagl)
        mode_idx = os.path.basename(f).split("_")[-1].split(".")[0]
        for c, comp in enumerate(COMPS):
            ax = axes[r][c]
            fld = sl[comp]
            lim = np.nanpercentile(np.abs(fld), 99.5)
            lim = lim if lim > 0 else 1.0
            cf = ax.pcolormesh(xkm, ykm, fld, cmap="RdBu_r",
                               vmin=-lim, vmax=lim, shading="auto")
            ax.contour(xkm, ykm, topo, levels=np.arange(10, topo.max(), 20),
                       colors="k", linewidths=0.3, alpha=0.4)
            fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(comp, fontsize=13)
            if c == 0:
                ax.set_ylabel(f"mode {mode_idx}\nλ={lam:.3g}\ny (km)", fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel("x (km)")

    fig.suptitle(f"POD modes @ {args.zagl:.0f} m AGL", fontsize=14)
    out = args.out or os.path.join(args.modes_dir, f"pod_modes_z{args.zagl:.0f}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
