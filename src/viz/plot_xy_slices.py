# plot_xy_slices.py
# cgf  2026-05-27
#
# Make top-down (x-y) PNGs of u, v, w from FastEddy output at a target
# height AGL, interpolated from terrain-following coordinates.
# Usage:
#   python plot_xy_slices.py <output_dir> [--zagl Z] [--outdir PNG_DIR]

import argparse
import os
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# field configs: (varname, colormap, symmetric, label, fixed_abs_limit)
# Fixed limits keep the colorbar (and thus frame size) constant across all
# frames so animations don't jitter. Override per-field on the CLI.
FIELDS = [
    ("w", "RdBu_r", True,  "w  (m s⁻¹)", 2.0),
    ("u", "RdBu_r", True,  "u  (m s⁻¹)", 12.0),
    ("v", "RdBu_r", True,  "v  (m s⁻¹)", 2.0),
]

def sim_time_seconds(stem, dt):
    """Parse the trailing integer step index off a frame name -> seconds."""
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) * dt if digits else float("nan")

# ---------------------------------------------------------------------------
def interp_to_zagl(field3d, zpos3d, topo2d, z_agl):
    """
    Interpolate (Nz, Ny, Nx) field to a constant height AGL.
    zpos3d is physical height ASL; z_agl is height above local terrain.
    Returns (Ny, Nx) slice.  Cells where z_agl is below terrain → NaN.
    """
    Nz, Ny, Nx = field3d.shape
    z_target = topo2d + z_agl          # (Ny, Nx)  ASL target height
    out = np.full((Ny, Nx), np.nan, dtype=np.float32)

    for j in range(Ny):
        for i in range(Nx):
            zc = zpos3d[:, j, i]       # (Nz,) height column
            zt = z_target[j, i]
            if zt < zc[0] or zt > zc[-1]:
                continue
            out[j, i] = np.interp(zt, zc, field3d[:, j, i])
    return out

def interp_to_zagl_fast(field3d, zpos3d, topo2d, z_agl):
    """Vectorised version using searchsorted along the z axis."""
    Nz, Ny, Nx = field3d.shape
    z_target = (topo2d + z_agl).astype(np.float32)   # (Ny, Nx)

    # find lower index for each column
    # zpos3d shape (Nz, Ny, Nx) → transpose to (Ny, Nx, Nz) for searchsorted
    zT = np.moveaxis(zpos3d, 0, -1)          # (Ny, Nx, Nz)
    fT = np.moveaxis(field3d, 0, -1)         # (Ny, Nx, Nz)

    out = np.full((Ny, Nx), np.nan, dtype=np.float32)
    for j in range(Ny):
        zrow = zT[j]                          # (Nx, Nz)
        frow = fT[j]                          # (Nx, Nz)
        zt   = z_target[j]                    # (Nx,)
        idx  = np.searchsorted(zrow[0], zt)   # assume z monotone & same for all x
        # clip to valid range
        valid = (idx > 0) & (idx < Nz)
        i_lo = np.clip(idx - 1, 0, Nz - 2)
        z_lo = zrow[np.arange(Nx), i_lo]
        z_hi = zrow[np.arange(Nx), i_lo + 1]
        f_lo = frow[np.arange(Nx), i_lo]
        f_hi = frow[np.arange(Nx), i_lo + 1]
        dz   = z_hi - z_lo
        w    = np.where(dz > 0, (zt - z_lo) / dz, 0.5)
        out[j] = np.where(valid, f_lo + w * (f_hi - f_lo), np.nan)
    return out

# ---------------------------------------------------------------------------
def plot_xy(nc_path, z_agl, out_dir, dt):
    with nc.Dataset(nc_path) as ds:
        xpos = ds["xPos"]   [0, 0, :, :]     # (Ny, Nx) — k=0 row, all same x
        ypos = ds["yPos"]   [0, 0, :, :]     # (Ny, Nx)
        zpos = ds["zPos"]   [0, :, :, :]     # (Nz, Ny, Nx)
        topo = ds["topoPos"][0, :, :]         # (Ny, Nx)
        data = {v: np.array(ds[v][0, :, :, :], dtype=np.float32)
                for v, *_ in FIELDS}

    xkm = xpos / 1e3
    ykm = ypos / 1e3
    stem = os.path.basename(nc_path)
    tsec = sim_time_seconds(stem, dt)

    # interpolate all fields to z_agl once, reuse for quivers
    interp = {v: interp_to_zagl_fast(data[v], zpos, topo, z_agl) for v, *_ in FIELDS}

    for varname, cmap, symmetric, label, lim in FIELDS:
        fld2d = interp[varname]

        if symmetric:
            levels = np.linspace(-lim, lim, 41)
        else:
            levels = np.linspace(0.0, lim, 41)

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        cf = ax.contourf(xkm, ykm, fld2d, levels=levels, cmap=cmap, extend="both")
        cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label(label)

        # terrain contours for reference
        ax.contour(xkm, ykm, topo, levels=np.arange(10, topo.max(), 10),
                   colors="k", linewidths=0.4, alpha=0.4)

        # sparse u-v quiver arrows at the same AGL height
        qs = max(1, xkm.shape[1] // 20)        # ~20 arrows across domain
        sl = (slice(None, None, qs), slice(None, None, qs))
        u2d, v2d = interp["u"], interp["v"]
        spd2d = np.sqrt(u2d ** 2 + v2d ** 2)
        scale = np.nanpercentile(spd2d, 95) * 20
        ax.quiver(xkm[sl], ykm[sl], u2d[sl], v2d[sl],
                  color="0.25", alpha=0.6, scale=scale,
                  width=0.003, headwidth=3, headlength=4)

        ax.set_xlabel("x  (km)")
        ax.set_ylabel("y  (km)")
        ax.set_title(f"{varname} @ {z_agl:.0f} m AGL   |   t = {tsec:5.0f} s")
        ax.set_aspect("equal")

        png = os.path.join(out_dir, f"{stem}_xy{z_agl:.0f}_{varname}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  saved {png}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir",           help="FastEddy output directory")
    parser.add_argument("--zagl",  type=float,  default=50.0,
                        help="height AGL for horizontal slice (default: 50 m)")
    parser.add_argument("--outdir",             default=None,
                        help="where to save PNGs (default: output_dir/xy_slices)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="timestep (s) for converting frame index -> sim time (default 0.02)")
    parser.add_argument("--ulim", type=float, default=None, help="override fixed |u| colorbar limit")
    parser.add_argument("--vlim", type=float, default=None, help="override fixed |v| colorbar limit")
    parser.add_argument("--wlim", type=float, default=None, help="override fixed |w| colorbar limit")
    parser.add_argument("--force", action="store_true",
                        help="re-render frames even if their PNGs already exist")
    args = parser.parse_args()

    _ovr = {"u": args.ulim, "v": args.vlim, "w": args.wlim}
    for k, (name, cmap, sym, label, lim) in enumerate(FIELDS):
        if _ovr.get(name) is not None:
            FIELDS[k] = (name, cmap, sym, label, _ovr[name])

    files = sorted(glob.glob(os.path.join(args.output_dir, "FE_*")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"No FE_* files found in {args.output_dir}")

    out_dir = args.outdir or os.path.join(args.output_dir, "xy_slices")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plotting {len(files)} files at z={args.zagl} m AGL → {out_dir}")
    n_done = 0
    for f in files:
        stem = os.path.basename(f)
        expected = [os.path.join(out_dir, f"{stem}_xy{args.zagl:.0f}_{v}.png")
                    for v, *_ in FIELDS]
        if not args.force and all(os.path.exists(p) for p in expected):
            n_done += 1
            continue
        plot_xy(f, args.zagl, out_dir, args.dt)
    if n_done:
        print(f"Skipped {n_done} already-rendered frames (use --force to redo).")
