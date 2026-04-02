"""
animate_output.py
-----------------
Loads FastEddy output files (or exported POD mode files) and produces either
an animation of the evolving flow field or a single static image.

Four-panel layout per frame (2 × 2):
  Top-left     — horizontal slice at ~80 m AGL: wind speed + quiver (flow mode)
                                               or u-component, diverging (static/pod mode)
  Top-right    — vertical x-z cross-section of u through the ridge (y = Ny/2)
  Bottom-left  — vertical x-z cross-section of v through the ridge (y = Ny/2)
  Bottom-right — vertical x-z cross-section of w through the ridge (y = Ny/2)

The terrain surface is overlaid on all panels.

Usage — animated flow:
  python animate_output.py [--outdir PATH] [--base FE_trialrun]
                            [--out_anim terrain_flow.mp4] [--fps 6]

Usage — single static image (e.g. POD mode):
  python animate_output.py --outdir PATH --base FE_Pod_mode --static
                            --out_image mode_01.png --symmetric_h --no_quiver
                            --title "POD mode 1  (38.4% energy)"

Requires: numpy, matplotlib, netCDF4
Optional: ffmpeg on PATH (falls back to GIF via Pillow if absent)
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import netCDF4 as nc

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.labelsize":  9,
    "axes.titlesize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth":  0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
script_dir = os.path.dirname(os.path.abspath(__file__))

parser.add_argument("--outdir",     default="/home/cego6160/workspace/runs/ensemble_fp/output/",
                    help="Directory containing FE output NetCDF files")
parser.add_argument("--base",       default="FE_trialrun",
                    help="outFileBase from the .in file")
parser.add_argument("--out_anim",   default=os.path.join(script_dir, "terrain_flow.mp4"),
                    help="Output animation file (.mp4 or .gif) — used when not --static")
parser.add_argument("--target_z_agl", type=float, default=80.0,
                    help="Target height AGL (m) for horizontal slice")
parser.add_argument("--fps",        type=int,   default=3)
parser.add_argument("--Nx",  type=int,   default=320,  help="Grid points in x")
parser.add_argument("--Ny",  type=int,   default=322,  help="Grid points in y")
parser.add_argument("--dx",  type=float, default=15.0, help="Grid spacing in x (m)")
parser.add_argument("--dy",  type=float, default=15.0, help="Grid spacing in y (m)")
parser.add_argument("--dz",  type=float, default=20.0, help="Nominal grid spacing in z (m)")
parser.add_argument("--xz_y_km", type=float, default=None,
                    help="y position (km) for x-z cross-section; defaults to domain centre")

# ── Static / POD-mode options ─────────────────────────────────────────────────
parser.add_argument("--static",      action="store_true",
                    help="Produce a single PNG image instead of an animation")
parser.add_argument("--out_image",   default=None,
                    help="Output PNG path when --static (default: <out_anim stem>.png)")
parser.add_argument("--symmetric_h", action="store_true",
                    help="Horizontal panel: show u-component with diverging colormap "
                         "instead of wind-speed with sequential colormap. Use for POD modes.")
parser.add_argument("--no_quiver",   action="store_true",
                    help="Suppress quiver vectors on the horizontal panel")
parser.add_argument("--title",       default=None,
                    help="Override figure suptitle (default: 'FastEddy | t = X s')")
parser.add_argument("--file",        default=None,
                    help="Use a single specific NetCDF file instead of globbing by --base")
args = parser.parse_args()

# Derive default out_image from out_anim stem
if args.out_image is None:
    stem = os.path.splitext(args.out_anim)[0]
    args.out_image = stem + ".png"

# ── Discover output files ─────────────────────────────────────────────────────
if args.file is not None:
    all_files = [args.file]
else:
    pattern  = os.path.join(args.outdir, f"{args.base}.*")
    all_files = sorted(glob.glob(pattern),
                       key=lambda f: int(re.search(r"\.(\d+)$", f).group(1)))

if not all_files:
    raise FileNotFoundError(f"No files matching {pattern if args.file is None else args.file}")

# Drop files that are entirely NaN (e.g. crashed timestep)
valid_files = []
for fp in all_files:
    with nc.Dataset(fp) as ds:
        u = ds["u"][0, 0, :, :]
        if not np.all(np.isnan(u)):
            valid_files.append(fp)
        else:
            print(f"  Skipping {os.path.basename(fp)} (all-NaN)")

if not valid_files:
    raise RuntimeError("No valid (non-NaN) output files found.")

# When producing a static image only the first file is needed
if args.static:
    valid_files = valid_files[:1]

print(f"{'Static image' if args.static else 'Animating'}: "
      f"{len(valid_files)} frame(s) from {args.outdir}")

# ── Load static geometry from first file ─────────────────────────────────────
with nc.Dataset(valid_files[0]) as ds:
    xpos_raw = ds["xPos"][0]   # (Nz, Ny, Nx) metres — may be zeros at t=0
    ypos_raw = ds["yPos"][0]
    zpos_raw = ds["zPos"][0]
    topo     = ds["topoPos"][0]   # (Ny, Nx) metres
    Nz, Ny, Nx = xpos_raw.shape

# Fall back to analytically computed coordinates if the file values are
# degenerate (all-zero or single-valued) — this happens at t=0 before the
# coordinate arrays are populated by FastEddy.
def _is_degenerate(arr1d):
    return float(arr1d.max()) - float(arr1d.min()) < 1.0

x_km = (xpos_raw[0, 0, :] / 1000.0 if not _is_degenerate(xpos_raw[0, 0, :])
        else np.arange(Nx) * args.dx / 1000.0)
y_km = (ypos_raw[0, :, 0] / 1000.0 if not _is_degenerate(ypos_raw[0, :, 0])
        else np.arange(Ny) * args.dy / 1000.0)

# For z: use file values if valid, otherwise approximate from d_zeta
ic = Nx // 2
if args.xz_y_km is not None:
    jc = int(np.argmin(np.abs(y_km - args.xz_y_km)))
    print(f"x-z slice: j={jc}, y = {y_km[jc]:.3f} km (requested {args.xz_y_km} km)")
else:
    jc = Ny // 2
z_col  = zpos_raw[:, jc, ic]
if _is_degenerate(z_col):
    z_col = np.arange(Nz) * args.dz

z_centre_agl = z_col - float(topo[jc, ic])
k_horiz = int(np.argmin(np.abs(z_centre_agl - args.target_z_agl)))
actual_z_agl = float(z_centre_agl[k_horiz])
print(f"Horizontal slice: k={k_horiz}, ~{actual_z_agl:.0f} m AGL at domain centre")

# x-z coordinates for vertical cross-section at y = jc
xz_x_col = xpos_raw[:, jc, :]
xz_z_col = zpos_raw[:, jc, :]
if _is_degenerate(xpos_raw[0, jc, :]):
    xz_x = np.tile(x_km, (Nz, 1))
else:
    xz_x = xz_x_col / 1000.0
if _is_degenerate(zpos_raw[:, jc, ic]):
    xz_z = np.outer(z_col, np.ones(Nx))
else:
    xz_z = xz_z_col
topo_xsec = topo[jc, :]   # (Nx,) m

# ── Helpers ───────────────────────────────────────────────────────────────────
QUIVER_STRIDE = 12

def read_frame(filepath):
    with nc.Dataset(filepath) as ds:
        t_s  = float(ds["time"][0])
        u    = ds["u"][0]    # (Nz, Ny, Nx)
        v    = ds["v"][0]
        w    = ds["w"][0]
        # fricVel is absent in exported POD mode files
        fvel = ds["fricVel"][0] if "fricVel" in ds.variables else None
    return t_s, u, v, w, fvel

def _horiz_field(u, v):
    """Field shown on the horizontal panel: u-component (symmetric) or wind speed."""
    if args.symmetric_h:
        return u[k_horiz]
    return np.sqrt(u[k_horiz]**2 + v[k_horiz]**2)

# ── Pre-compute colour limits across all frames ───────────────────────────────
print("Computing colour limits …")
h_vals, u_vals, v_vals, w_vals = [], [], [], []
for fp in valid_files:
    _, u, v, w, _ = read_frame(fp)
    h_vals.append(_horiz_field(u, v))
    u_vals.append(u[:, jc, :])
    v_vals.append(v[:, jc, :])
    w_vals.append(w[:, jc, :])

u_abs = float(np.nanpercentile(np.abs(np.stack(u_vals)), 99))
v_abs = float(np.nanpercentile(np.abs(np.stack(v_vals)), 99))
w_abs = float(np.nanpercentile(np.abs(np.stack(w_vals)), 99))

if args.symmetric_h:
    h_abs  = float(np.nanpercentile(np.abs(np.stack(h_vals)), 99))
    norm_h = TwoSlopeNorm(vmin=-h_abs, vcenter=0, vmax=h_abs)
    cmap_h = "RdBu_r"
else:
    h_max  = float(np.nanpercentile(np.stack(h_vals), 99))
    norm_h = Normalize(vmin=0, vmax=h_max)
    cmap_h = "YlOrBr"

norm_xz = TwoSlopeNorm(vmin=-u_abs, vcenter=0, vmax=u_abs)
norm_v  = TwoSlopeNorm(vmin=-v_abs, vcenter=0, vmax=v_abs)
norm_w  = TwoSlopeNorm(vmin=-w_abs, vcenter=0, vmax=w_abs)

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9.0))
gs  = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.45)
ax_h   = fig.add_subplot(gs[0, 0])
ax_xz  = fig.add_subplot(gs[0, 1])
ax_v   = fig.add_subplot(gs[1, 0])
ax_w   = fig.add_subplot(gs[1, 1])

# ── Terrain contours (static) ─────────────────────────────────────────────────
topo_levels = np.arange(25, 210, 25)

XX, YY = np.meshgrid(x_km, y_km)
ax_h.contour(XX, YY, topo, levels=topo_levels,
             colors="0.25", linewidths=0.5, alpha=0.6)

for _ax in (ax_xz, ax_v, ax_w):
    _ax.fill_between(x_km, topo_xsec, color="0.75", alpha=1.0, zorder=3)
    _ax.plot(x_km, topo_xsec, color="0.3", lw=0.8, zorder=4)

# ── Initial frame ─────────────────────────────────────────────────────────────
t0, u0, v0, w0, _ = read_frame(valid_files[0])
h0 = _horiz_field(u0, v0)

im_h = ax_h.pcolormesh(XX, YY, h0, cmap=cmap_h,
                        norm=norm_h, shading="nearest", zorder=1)

sl = slice(None, None, QUIVER_STRIDE)
if not args.no_quiver:
    qv = ax_h.quiver(XX[sl, sl], YY[sl, sl],
                     u0[k_horiz][sl, sl], v0[k_horiz][sl, sl],
                     color="0.15", scale=150, width=0.003,
                     alpha=0.8, zorder=2)

im_xz = ax_xz.pcolormesh(xz_x, xz_z, u0[:, jc, :],
                           cmap="RdBu_r", norm=norm_xz,
                           shading="nearest", zorder=2)

im_v = ax_v.pcolormesh(xz_x, xz_z, v0[:, jc, :],
                        cmap="RdBu_r", norm=norm_v,
                        shading="nearest", zorder=2)

im_w = ax_w.pcolormesh(xz_x, xz_z, w0[:, jc, :],
                        cmap="RdBu_r", norm=norm_w,
                        shading="nearest", zorder=2)

# ── Colorbars & labels ────────────────────────────────────────────────────────
cb_h  = fig.colorbar(im_h,  ax=ax_h,  fraction=0.046, pad=0.04)
cb_xz = fig.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04)
cb_v  = fig.colorbar(im_v,  ax=ax_v,  fraction=0.046, pad=0.04)
cb_w  = fig.colorbar(im_w,  ax=ax_w,  fraction=0.046, pad=0.04)

h_label = "u  (m s\u207b\u00b9)" if args.symmetric_h else "Wind speed (m s\u207b\u00b9)"
cb_h.set_label(h_label)
cb_xz.set_label("u  (m s\u207b\u00b9)")
cb_v.set_label("v  (m s\u207b\u00b9)")
cb_w.set_label("w  (m s\u207b\u00b9)")

ax_h.set_xlabel("x  (km)")
ax_h.set_ylabel("y  (km)")
for _ax in (ax_xz, ax_v, ax_w):
    _ax.set_xlabel("x  (km)")
    _ax.set_ylabel("z  (m)")
    _ax.set_xlim(x_km[0], x_km[-1])
    _ax.set_ylim(0, float(z_col.max()))

ax_h.set_xlim(x_km[0],  x_km[-1])
ax_h.set_ylim(y_km[0],  y_km[-1])

h_panel_title = ("(a) Zonal wind u" if args.symmetric_h
                 else "(a) Horiz. wind speed")
ax_h.set_title(f"{h_panel_title},  z \u2248 {actual_z_agl:.0f} m AGL")
ax_xz.set_title(f"(b) Zonal wind u,  y = {y_km[jc]:.2f} km")
ax_v.set_title(f"(c) Meridional wind v,  y = {y_km[jc]:.2f} km")
ax_w.set_title(f"(d) Vertical wind w,  y = {y_km[jc]:.2f} km")

def _default_title(t_s):
    return f"FastEddy  |  t = {t_s:.0f} s  ({t_s/3600:.2f} h)" if not args.static \
           else f"FastEddy  |  t = {t_s:.0f} s"

title = fig.suptitle(args.title if args.title else _default_title(t0),
                     fontsize=10, y=1.01)

# ── Static output — save and exit ─────────────────────────────────────────────
if args.static:
    fig.savefig(args.out_image, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {args.out_image}")
    raise SystemExit(0)

# ── Animation update ──────────────────────────────────────────────────────────
def update(frame_idx):
    t_s, u, v, w, _ = read_frame(valid_files[frame_idx])
    h = _horiz_field(u, v)

    im_h.set_array(h.ravel())
    if not args.no_quiver:
        qv.set_UVC(u[k_horiz][sl, sl], v[k_horiz][sl, sl])
    im_xz.set_array(u[:, jc, :].ravel())
    im_v.set_array(v[:, jc, :].ravel())
    im_w.set_array(w[:, jc, :].ravel())

    title.set_text(args.title if args.title else _default_title(t_s))
    artists = [im_h, im_xz, im_v, im_w, title]
    if not args.no_quiver:
        artists.append(qv)
    return artists

ani = animation.FuncAnimation(fig, update,
                               frames=len(valid_files),
                               interval=1000 // args.fps,
                               blit=False)

# ── Save animation ────────────────────────────────────────────────────────────
ext = os.path.splitext(args.out_anim)[1].lower()
if ext == ".gif":
    writer = animation.PillowWriter(fps=args.fps)
else:
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2000,
                                    extra_args=["-pix_fmt", "yuv420p"])

ani.save(args.out_anim, writer=writer, dpi=150,
         savefig_kwargs={"facecolor": "white"})
print(f"Saved: {args.out_anim}")
