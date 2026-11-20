# plot_kf_field_rmse.py
# cgf  2026-06-25
#
# Spatial maps of the KF reconstruction error written by scripts/make_kf_field_rmse.jl.
# Two error fields share a colorbar so they're directly comparable:
#   rmse_*  — filter total reconstruction error  Prα − x_truth (carries the floor)
#   proj_*  — irreducible projection floor        Pr·truth_coeffs − x_truth (subspace limit)
# For each, an x–y plane at height --z-agl (m AGL) and a vertical plane along the wake
# azimuth --az (deg from +x; default 5° ≈ the truth Ug/Vg flow direction) through the
# domain centre, extracted by bilinear interpolation in x,y at each (s,z).
#
# Usage:
#   python plot_kf_field_rmse.py results/<tag>/fields/kf_field_rmse_truth_truth.nc [--z-agl 100] [--az 5] [--out PNG]

import argparse, os, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sesp_paths import figure_dir


def load(nc, name):
    # stored (x,y,z,time); NCDatasets writes reversed so Python sees (time,z,y,x)
    return np.asarray(nc[name][0]).transpose(2, 1, 0)  # -> (x,y,z)


def wake_transect(F, X, Y, Z, topo, az_deg, n_s=300):
    """Vertical plane through the domain centre along azimuth az (deg from +x).
    Returns (s, zcol, plane) for pcolormesh: s along-wake distance from centre (m)."""
    nx, ny, nz = F.shape
    x1d, y1d = X[:, 0, 0], Y[0, :, 0]
    x0, y0 = x1d.mean(), y1d.mean()
    half = 0.5 * min(np.ptp(x1d), np.ptp(y1d))
    s = np.linspace(-half, half, n_s)
    ux, uy = np.cos(np.deg2rad(az_deg)), np.sin(np.deg2rad(az_deg))
    xs, ys = x0 + s * ux, y0 + s * uy
    # fractional grid indices for bilinear interp in x,y (column by column in z)
    fx = np.interp(xs, x1d, np.arange(nx)); fy = np.interp(ys, y1d, np.arange(ny))
    i0 = np.clip(fx.astype(int), 0, nx - 2); j0 = np.clip(fy.astype(int), 0, ny - 2)
    tx = fx - i0; ty = fy - j0
    plane = np.empty((nz, n_s))
    for c in range(n_s):
        i, j, a, b = i0[c], j0[c], tx[c], ty[c]
        plane[:, c] = ((1 - a) * (1 - b) * F[i, j, :] + a * (1 - b) * F[i + 1, j, :]
                       + (1 - a) * b * F[i, j + 1, :] + a * b * F[i + 1, j + 1, :])
    zcol = Z[nx // 2, ny // 2, :]
    terr = ((1 - tx) * (1 - ty) * topo[i0, j0] + tx * (1 - ty) * topo[i0 + 1, j0]
            + (1 - tx) * ty * topo[i0, j0 + 1] + tx * ty * topo[i0 + 1, j0 + 1])
    return s, zcol, plane, terr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("nc")
    p.add_argument("--z-agl", type=float, default=100.0, help="height AGL for the x-y plane (m)")
    p.add_argument("--az", type=float, default=5.0, help="wake azimuth, deg from +x")
    p.add_argument("--level-step", type=float, default=0.1, help="contour level spacing (m/s)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    nc = Dataset(args.nc)
    X, Y, Z = load(nc, "xPos"), load(nc, "yPos"), load(nc, "zPos")
    topo = np.asarray(nc["topoPos"][0]).T  # (x,y)

    # static sensor rake positions (m), if present in the NetCDF
    if all(v in nc.variables for v in ("sensor_x", "sensor_y", "sensor_z")):
        sx = np.asarray(nc["sensor_x"][:]); sy = np.asarray(nc["sensor_y"][:]); sz = np.asarray(nc["sensor_z"][:])
    else:
        sx = sy = sz = None
    fields = {"filter total error  Prα − x_truth": load(nc, "rmse_speed"),
              "projection floor  Pr·a_truth − x_truth": load(nc, "proj_speed")}

    # x-y plane at the z-level closest to z_agl above local terrain (use column-mean AGL)
    zc = Z[0, 0, :]
    kz = int(np.argmin(np.abs((zc - zc[0]) - args.z_agl)))
    vmax = max(np.percentile(f[f > 0], 99) for f in fields.values())
    level_step = args.level_step
    level_max = np.ceil(vmax / level_step) * level_step
    levels = np.arange(0, level_max + level_step, level_step)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for col, (title, F) in enumerate(fields.items()):
        # row 0: x-y plane
        ax = axes[0, col]
        xy = np.ma.masked_equal(F[:, :, kz], 0.0)
        pc = ax.contourf(X[:, :, kz] / 1000, Y[:, :, kz] / 1000, xy,
                         levels=levels, cmap="magma", vmin=0, vmax=levels[-1], extend="max")
        ax.contour(X[:, :, kz] / 1000, Y[:, :, kz] / 1000, xy,
                   levels=levels[::4], colors="k", linewidths=0.3, alpha=0.5)
        ax.contour(X[:, :, 0] / 1000, Y[:, :, 0] / 1000, topo, 6, colors="cyan", linewidths=0.4)
        if sx is not None:
            ax.scatter(sx / 1000, sy / 1000, s=28, marker="v", facecolors="none",
                       edgecolors="lime", linewidths=1.1, zorder=5, label="sensors")
        ax.set_title(f"{title}\nx–y @ {args.z_agl:.0f} m AGL")
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="speed-error RMSE (m/s)")
        # row 1: wake-aligned vertical plane
        ax = axes[1, col]
        s, zcol, plane, terr = wake_transect(F, X, Y, Z, topo, args.az)
        plane = np.ma.masked_equal(plane, 0.0)
        pc = ax.contourf(s / 1000, zcol, plane, levels=levels, cmap="magma",
                         vmin=0, vmax=levels[-1], extend="max")
        ax.contour(s / 1000, zcol, plane, levels=levels[::4], colors="k", linewidths=0.3, alpha=0.5)
        ax.fill_between(s / 1000, 0, terr, color="0.3")
        if sx is not None:
            # project each sensor onto the transect azimuth (through domain centre)
            x1d, y1d = X[:, 0, 0], Y[0, :, 0]
            x0, y0 = x1d.mean(), y1d.mean()
            ux, uy = np.cos(np.deg2rad(args.az)), np.sin(np.deg2rad(args.az))
            s_sensor = ((sx - x0) * ux + (sy - y0) * uy) / 1000
            ax.scatter(s_sensor, sz, s=28, marker="v", facecolors="none",
                       edgecolors="lime", linewidths=1.1, zorder=5, label="sensors")
        ax.set_title(f"vertical plane along wake (az={args.az:.0f}° from +x)")
        ax.set_xlabel("along-wake distance from centre (km)"); ax.set_ylabel("height (m)")
        ax.set_ylim(0, min(zcol.max(), 1500))
        fig.colorbar(pc, ax=ax, label="speed-error RMSE (m/s)")

    _fig_dir = figure_dir()
    stem = pathlib.Path(args.nc).stem  # e.g. kf_field_rmse_truth_truth
    out = args.out or str(_fig_dir / f"{stem}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
