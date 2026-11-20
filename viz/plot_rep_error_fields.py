# plot_rep_error_fields.py
# cgf  2026-07-08
#
# Plain x-y and x-z slices of the representability-error (σ_rep) fields, one
# column per truth id, sharing a colourbar so panels are directly comparable.
# Reads rep_error_field_XX.nc written by scripts/make_rep_error_fields.jl —
# standard FastEddy-format NetCDF (u/v/w/xPos/yPos/zPos/topoPos), where u/v/w
# here are the per-component σ_rep(ζ), not a wind field. Plotted quantity is
# the combined magnitude speed = sqrt(u²+v²+w²).
#
# Usage:
#   python plot_rep_error_fields.py [--nc-dir DIR] [--z-agl 100] [--jslice J]
#                                   [--members 00 01 ...] [--out PNG]
#   Defaults: nc-dir = output/fields, z-agl = 100 m, jslice = domain midpoint,
#             all members found.

import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sesp_paths import FIELDS_DIR


def interp_to_zagl(field3d, zpos3d, topo2d, z_agl):
    """(Nz,Ny,Nx) field, terrain-following zpos3d -> (Ny,Nx) slice at z_agl AGL."""
    Nz, Ny, Nx = field3d.shape
    z_target = (topo2d + z_agl).astype(np.float32)
    zT = np.moveaxis(zpos3d, 0, -1)   # (Ny,Nx,Nz)
    fT = np.moveaxis(field3d, 0, -1)  # (Ny,Nx,Nz)
    out = np.full((Ny, Nx), np.nan, dtype=np.float32)
    for j in range(Ny):
        zrow, frow, zt = zT[j], fT[j], z_target[j]
        idx = np.searchsorted(zrow[0], zt)
        valid = (idx > 0) & (idx < Nz)
        i_lo = np.clip(idx - 1, 0, Nz - 2)
        z_lo = zrow[np.arange(Nx), i_lo]; z_hi = zrow[np.arange(Nx), i_lo + 1]
        f_lo = frow[np.arange(Nx), i_lo]; f_hi = frow[np.arange(Nx), i_lo + 1]
        dz = z_hi - z_lo
        w = np.where(dz > 0, (zt - z_lo) / dz, 0.5)
        out[j] = np.where(valid, f_lo + w * (f_hi - f_lo), np.nan)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc-dir", default=None,
                   help="directory containing rep_error_field_XX.nc files")
    p.add_argument("--z-agl", type=float, default=100.0, help="x-y slice height AGL (m)")
    p.add_argument("--jslice", type=int, default=None, help="y-index for x-z slice (default: midpoint)")
    p.add_argument("--members", nargs="*", default=None,
                   help="member/truth IDs to include, e.g. 00 01 02 (default: all found)")
    p.add_argument("--level-step", type=float, default=0.1, help="contour level spacing (m/s)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    nc_dir = args.nc_dir or str(FIELDS_DIR)
    nc_dir = os.path.realpath(nc_dir)

    if args.members:
        nc_files = [(mid, os.path.join(nc_dir, f"rep_error_field_{mid}.nc")) for mid in args.members]
    else:
        found = sorted(glob.glob(os.path.join(nc_dir, "rep_error_field_*.nc")))
        nc_files = [(os.path.basename(f)[len("rep_error_field_"):-3], f) for f in found]

    if not nc_files:
        raise SystemExit(f"No rep_error_field_*.nc files found in {nc_dir}")

    data = []
    for mid, fpath in nc_files:
        with Dataset(fpath) as ds:
            xpos = ds["xPos"][0, :, :, :]   # (Nz,Ny,Nx)
            ypos = ds["yPos"][0, :, :, :]
            zpos = ds["zPos"][0, :, :, :]
            topo = np.asarray(ds["topoPos"][0, :, :])   # (Ny,Nx)
            u = np.asarray(ds["u"][0, :, :, :], dtype=np.float32)
            v = np.asarray(ds["v"][0, :, :, :], dtype=np.float32)
            w = np.asarray(ds["w"][0, :, :, :], dtype=np.float32)
        speed = np.sqrt(u**2 + v**2 + w**2)   # (Nz,Ny,Nx)
        data.append((mid, xpos, ypos, zpos, topo, speed))

    Nz, Ny, Nx = data[0][5].shape
    j_slice = args.jslice if args.jslice is not None else Ny // 2
    vmax = max(np.nanpercentile(d[5], 99) for d in data)

    N = len(data)
    fig, axes = plt.subplots(2, N, figsize=(4.5 * N, 9), constrained_layout=True, squeeze=False)
    fig.suptitle(
        f"Representability error σ_rep — LOO statistical model\n"
        f"x–y @ {args.z_agl:.0f} m AGL  |  x–z @ j={j_slice}",
        fontsize=11)

    level_step = args.level_step
    level_max = np.ceil(vmax / level_step) * level_step
    levels = np.arange(0, level_max + level_step, level_step)

    for col, (mid, xpos, ypos, zpos, topo, speed) in enumerate(data):

        # --- x-y plane, interpolated to z_agl AGL ---
        ax = axes[0, col]
        xy = interp_to_zagl(speed, zpos, topo, args.z_agl)
        xkm = xpos[0] / 1e3; ykm = ypos[0] / 1e3
        pc = ax.contourf(xkm, ykm, xy, levels=levels, cmap="magma", vmin=0, vmax=levels[-1], extend="max")
        ax.contour(xkm, ykm, xy, levels=levels[::4], colors="k", linewidths=0.3, alpha=0.5)
        ax.contour(xkm, ykm, topo, 6, colors="cyan", linewidths=0.4)
        ax.set_title(f"member {mid}")
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_aspect("equal")
        if col == N - 1:
            fig.colorbar(pc, ax=ax, label="σ_rep (m/s)")

        # --- x-z slice at fixed j ---
        ax = axes[1, col]
        xkm_xz = xpos[:, j_slice, :] / 1e3
        zxz = zpos[:, j_slice, :]
        zz = speed[:, j_slice, :]
        pc = ax.contourf(xkm_xz, zxz, zz, levels=levels, cmap="magma", vmin=0, vmax=levels[-1], extend="max")
        ax.contour(xkm_xz, zxz, zz, levels=levels[::4], colors="k", linewidths=0.3, alpha=0.5)
        ax.fill_between(xkm_xz[0], 0, topo[j_slice, :], color="0.3")
        ax.set_xlabel("x (km)"); ax.set_ylabel("z (m)")
        ax.set_ylim(0, min(zxz.max(), 1500))
        if col == N - 1:
            fig.colorbar(pc, ax=ax, label="σ_rep (m/s)")

    out = args.out or os.path.join(nc_dir, "rep_error_fields.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
