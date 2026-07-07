# plot_rep_error_fields.py
# cgf  2026-06-26
#
# Spatial representability-error (projection-floor) contour plots for each LOO basis.
# Reads rep_error_field_XX.nc files written by scripts/make_rep_error_fields.jl and
# produces a 2×N panel figure (x-y plane + wake-aligned vertical slice per member).
#
# A shared colourbar spans all members so the panels are directly comparable.
#
# Usage:
#   python plot_rep_error_fields.py [--nc-dir DIR] [--z-agl 100] [--az 5]
#                                   [--members 00 01 ...] [--out PNG]
#   Defaults: nc-dir = output/, z-agl = 100 m, az = 5°, all members found.

import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def load(nc, name):
    return np.asarray(nc[name][0]).transpose(2, 1, 0)   # (time,z,y,x) -> (x,y,z)


def wake_transect(F, X, Y, Z, topo, az_deg, n_s=300):
    nx, ny, nz = F.shape
    x1d, y1d = X[:, 0, 0], Y[0, :, 0]
    x0, y0 = x1d.mean(), y1d.mean()
    half = 0.5 * min(np.ptp(x1d), np.ptp(y1d))
    s = np.linspace(-half, half, n_s)
    ux, uy = np.cos(np.deg2rad(az_deg)), np.sin(np.deg2rad(az_deg))
    xs, ys = x0 + s * ux, y0 + s * uy
    fx = np.interp(xs, x1d, np.arange(nx))
    fy = np.interp(ys, y1d, np.arange(ny))
    i0 = np.clip(fx.astype(int), 0, nx - 2)
    j0 = np.clip(fy.astype(int), 0, ny - 2)
    tx, ty = fx - i0, fy - j0
    plane = np.empty((nz, n_s))
    for c in range(n_s):
        i, j, a, b = i0[c], j0[c], tx[c], ty[c]
        plane[:, c] = ((1-a)*(1-b)*F[i,j,:] + a*(1-b)*F[i+1,j,:]
                      + (1-a)*b*F[i,j+1,:] + a*b*F[i+1,j+1,:])
    zcol = Z[nx//2, ny//2, :]
    terr = ((1-tx)*(1-ty)*topo[i0,j0] + tx*(1-ty)*topo[i0+1,j0]
           + (1-tx)*ty*topo[i0,j0+1] + tx*ty*topo[i0+1,j0+1])
    return s, zcol, plane, terr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc-dir", default=None,
                   help="directory containing rep_error_field_XX.nc files")
    p.add_argument("--z-agl", type=float, default=100.0)
    p.add_argument("--az", type=float, default=5.0)
    p.add_argument("--members", nargs="*", default=None,
                   help="member IDs to include, e.g. 00 01 02 (default: all found)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    nc_dir = args.nc_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "output")
    nc_dir = os.path.realpath(nc_dir)

    if args.members:
        nc_files = [(mid, os.path.join(nc_dir, f"rep_error_field_{mid}.nc"))
                    for mid in args.members]
    else:
        found = sorted(glob.glob(os.path.join(nc_dir, "rep_error_field_??.nc")))
        nc_files = [(os.path.basename(f)[16:18], f) for f in found]

    if not nc_files:
        raise SystemExit(f"No rep_error_field_XX.nc files found in {nc_dir}")

    # load all fields; build shared colour scale
    data = []
    for mid, fpath in nc_files:
        nc = Dataset(fpath)
        X = load(nc, "xPos"); Y = load(nc, "yPos"); Z = load(nc, "zPos")
        topo = np.asarray(nc["topoPos"][0]).T
        F = load(nc, "proj_speed")
        data.append((mid, X, Y, Z, topo, F))

    zc = data[0][3][0, 0, :]
    kz = int(np.argmin(np.abs((zc - zc[0]) - args.z_agl)))
    vmax = max(np.percentile(d[5][d[5] > 0], 99) for d in data)

    N = len(data)
    fig, axes = plt.subplots(2, N, figsize=(4.5 * N, 9), constrained_layout=True,
                              squeeze=False)
    fig.suptitle(
        f"Representability error (projection floor) — LOO bases\n"
        f"x–y @ {args.z_agl:.0f} m AGL  |  vertical slice az={args.az:.0f}° from +x",
        fontsize=11)

    for col, (mid, X, Y, Z, topo, F) in enumerate(data):
        # --- x-y plane ---
        ax = axes[0, col]
        xy = np.ma.masked_equal(F[:, :, kz], 0.0)
        pc = ax.pcolormesh(X[:, :, kz]/1000, Y[:, :, kz]/1000, xy,
                           cmap="magma", vmin=0, vmax=vmax, shading="auto")
        ax.contour(X[:, :, 0]/1000, Y[:, :, 0]/1000, topo, 6,
                   colors="cyan", linewidths=0.4)
        ax.set_title(f"member {mid}")
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_aspect("equal")
        if col == N - 1:
            fig.colorbar(pc, ax=ax, label="speed RMSE (m/s)")

        # --- wake vertical slice ---
        ax = axes[1, col]
        s, zcol, plane, terr = wake_transect(F, X, Y, Z, topo, args.az)
        plane = np.ma.masked_equal(plane, 0.0)
        pc = ax.pcolormesh(s/1000, zcol, plane,
                           cmap="magma", vmin=0, vmax=vmax, shading="auto")
        ax.fill_between(s/1000, 0, terr, color="0.3")
        ax.set_xlabel("along-wake (km)"); ax.set_ylabel("height (m)")
        ax.set_ylim(0, min(zcol.max(), 1500))
        if col == N - 1:
            fig.colorbar(pc, ax=ax, label="speed RMSE (m/s)")

    out = args.out or os.path.join(nc_dir, "rep_error_fields.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
