# plot_xz_slices.py
# cgf  2026-05-27
#
# Make x-z cross-section PNGs of vertical velocity from FastEddy output,
# correctly plotted in terrain-following physical coordinates (xPos, zPos).
# Usage:
#   python plot_xz_slices.py <output_dir> [--jslice J] [--outdir PNG_DIR]

import argparse
import os
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# field configs: (varname, colormap, symmetric, label, fixed_abs_limit)
# Fixed limits keep the colorbar (and thus frame size) constant across all
# frames so animations don't jitter. Override per-field on the CLI.
FIELDS = [
    ("w", "RdBu_r",  True,  "w  (m s⁻¹)", 3.0),
    ("u", "RdBu_r",  True,  "u  (m s⁻¹)", 12.0),
    ("v", "RdBu_r",  True,  "v  (m s⁻¹)", 3.0),
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
def plot_xz(nc_path, j_slice, out_dir, dt):
    with nc.Dataset(nc_path) as ds:
        xpos = ds["xPos"]   [0, :, j_slice, :]   # (Nz, Nx)
        zpos = ds["zPos"]   [0, :, j_slice, :]   # (Nz, Nx)
        topo = ds["topoPos"][0, j_slice, :]       # (Nx,)
        data = {v: ds[v][0, :, j_slice, :] for v, *_ in FIELDS}

    xkm  = xpos / 1e3
    stem = os.path.basename(nc_path)
    tsec = sim_time_seconds(stem, dt)

    for varname, cmap, symmetric, label, lim in FIELDS:
        fld = data[varname]

        if symmetric:
            levels = np.linspace(-lim, lim, 41)
        else:
            levels = np.linspace(0.0, lim, 41)

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        cf = ax.contourf(xkm, zpos, fld, levels=levels, cmap=cmap, extend="both")
        cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.03)
        cb.set_label(label)

        ax.fill_between(xkm[0], 0, topo, color="saddlebrown", alpha=0.85, zorder=5)
        ax.plot(xkm[0], topo, color="k", lw=0.8, zorder=6)

        ax.set_xlabel("x  (km)")
        ax.set_ylabel("z  (m)")
        ax.set_title(f"{varname} — x-z slice j={j_slice}   |   t = {tsec:5.0f} s")
        ax.set_ylim(0, zpos.max())

        png = os.path.join(out_dir, f"{stem}_xz_{varname}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  saved {png}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir",          help="FastEddy output directory")
    parser.add_argument("--jslice", type=int,  default=None,
                        help="y-index for x-z slice (default: domain midpoint)")
    parser.add_argument("--outdir",            default=None,
                        help="where to save PNGs (default: output_dir/xz_slices)")
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
    # exclude directories, keep only NetCDF files
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"No FE_FE_Member* files found in {args.output_dir}")

    # determine j_slice from first file if not given
    j_slice = args.jslice
    if j_slice is None:
        with nc.Dataset(files[0]) as ds:
            Ny = ds.dimensions["yIndex"].size
        j_slice = Ny // 2

    out_dir = args.outdir or os.path.join(args.output_dir, "xz_slices")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plotting {len(files)} files, j={j_slice} → {out_dir}")
    n_done = 0
    for f in files:
        stem = os.path.basename(f)
        expected = [os.path.join(out_dir, f"{stem}_xz_{v}.png") for v, *_ in FIELDS]
        if not args.force and all(os.path.exists(p) for p in expected):
            n_done += 1
            continue
        plot_xz(f, j_slice, out_dir, args.dt)
    if n_done:
        print(f"Skipped {n_done} already-rendered frames (use --force to redo).")
