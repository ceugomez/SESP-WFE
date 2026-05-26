# generate_terrain.py
# cgf cego6160@colorado.edu 5.26.26
#
# Generates the idealized 2-Gaussian-hill terrain for the synthetic ensemble
# at 506x506 / 10m resolution, matching the physical hill geometry from the
# prior 242x242 / 20m runs. Writes FastEddy binary + optional PNG.
#
# FastEddy terrain binary format: [int32 Nx][int32 Ny][float32 topo(Ny, Nx)]

import numpy as np
import struct
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

Nx, Ny   = 506, 506
dx = dy  = 10.0          # m

OUT_DIR  = "/home/cego6160/workspace/runs_synthetic/setup"
BIN_FILE = os.path.join(OUT_DIR, "terrain_idealized.bin")
PNG_FILE = os.path.join(OUT_DIR, "terrain_idealized.png")

# Hill geometry — defined in PHYSICAL units (metres) so they are
# resolution-independent. Fractional positions scale with domain extent.
# These match the prior runs exactly in physical space:
#   Main ridge : H=200m, sx=450m (cross-ridge), sy=1200m (along-ridge)
#   Knoll      : H=100m, sx=sy=350m
HILLS = [
    dict(fx=0.50, fy=0.50, H=200.0, sx=260.0, sy=700.0),   # main ridge — centred, ~26° max slope, ~15% domain width
    dict(fx=0.62, fy=0.60, H=120.0, sx=160.0, sy=160.0),   # satellite knoll — compact, downstream/right, ~26° max slope
]

# ---------------------------------------------------------------------------
# TERRAIN GENERATION
# ---------------------------------------------------------------------------

def gaussian_hill(xx, yy, cx, cy, H, sx, sy):
    return H * np.exp(-0.5 * ((xx - cx)**2 / sx**2 +
                               (yy - cy)**2 / sy**2))

def build_terrain(Nx, Ny, dx, dy, hills):
    x  = np.arange(Nx) * dx
    y  = np.arange(Ny) * dy
    Lx = Nx * dx
    Ly = Ny * dy
    xx, yy = np.meshgrid(x, y)   # (Ny, Nx)
    topo = np.zeros((Ny, Nx), dtype=np.float32)
    for h in hills:
        cx = h["fx"] * Lx
        cy = h["fy"] * Ly
        topo += gaussian_hill(xx, yy, cx, cy, h["H"], h["sx"], h["sy"])
    return topo, x, y, Lx, Ly

# ---------------------------------------------------------------------------
# BINARY WRITER
# ---------------------------------------------------------------------------

def write_fasteddy_terrain(path, topo, Nx, Ny):
    """Write FastEddy terrain binary: [int32 Nx][int32 Ny][float32 topo(Ny,Nx)]"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("i", Nx))
        f.write(struct.pack("i", Ny))
        f.write(topo.astype(np.float32).tobytes())
    size_kb = os.path.getsize(path) / 1024
    print(f"Wrote terrain binary: {path}  ({Nx}x{Ny}, {size_kb:.1f} KB)")

# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def plot_terrain(topo, x, y, Lx, Ly, out_path):
    vmin, vmax    = 0.0, float(topo.max())
    norm          = Normalize(vmin=vmin, vmax=vmax)
    cmap          = plt.cm.terrain
    levels_filled = np.linspace(vmin, vmax, 60)
    levels_lines  = np.arange(25, vmax, 25)
    xx, yy        = np.meshgrid(x, y)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
    })

    fig = plt.figure(figsize=(11, 4.8))
    ax_plan = fig.add_subplot(1, 2, 1)
    ax_3d   = fig.add_subplot(1, 2, 2, projection="3d")

    # Plan view
    cf = ax_plan.contourf(xx/1e3, yy/1e3, topo, levels=levels_filled,
                          cmap=cmap, norm=norm)
    cs = ax_plan.contour(xx/1e3, yy/1e3, topo, levels=levels_lines,
                         colors="k", linewidths=0.5, alpha=0.55)
    ax_plan.clabel(cs, levels_lines[::2], fmt="%g m", fontsize=6.5, inline=True)
    cb = fig.colorbar(cf, ax=ax_plan, fraction=0.046, pad=0.04)
    cb.set_label("Elevation (m)")
    cb.set_ticks(np.arange(0, int(vmax)+1, 50))
    ax_plan.annotate("", xy=(3.8, 0.4), xytext=(3.1, 0.4),
                     arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5))
    ax_plan.text(3.85, 0.4, r"$\bar{\alpha}=270°$",
                 color="#c0392b", fontsize=7.5, va="center")
    ax_plan.set_xlim(0, Lx/1e3); ax_plan.set_ylim(0, Ly/1e3)
    ax_plan.set_xlabel("x (km)"); ax_plan.set_ylabel("y (km)")
    ax_plan.set_title("(a) Plan view"); ax_plan.set_aspect("equal")

    # 3D perspective
    stride = 4
    xs  = x[::stride]/1e3; ys = y[::stride]/1e3
    zs  = topo[::stride, ::stride]
    XXs, YYs = np.meshgrid(xs, ys)
    ls  = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(zs, cmap=cmap, norm=norm, vert_exag=8, blend_mode="soft")
    ax_3d.plot_surface(XXs, YYs, zs, facecolors=rgb,
                       rstride=1, cstride=1, linewidth=0,
                       antialiased=True, shade=False)
    ax_3d.contour(xx/1e3, yy/1e3, topo, levels=levels_lines,
                  zdir="z", offset=0, colors="k", linewidths=0.4, alpha=0.35)
    ax_3d.set_xlim(0, Lx/1e3); ax_3d.set_ylim(0, Ly/1e3); ax_3d.set_zlim(0, 300)
    ax_3d.set_xlabel("x (km)", labelpad=5)
    ax_3d.set_ylabel("y (km)", labelpad=5)
    ax_3d.set_zlabel("z (m)",  labelpad=3)
    ax_3d.set_title("(b) Perspective view")
    ax_3d.view_init(elev=30, azim=-50)
    ax_3d.set_box_aspect([1, 1, 0.3])
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb3 = fig.colorbar(sm, ax=ax_3d, fraction=0.028, pad=0.10, shrink=0.75)
    cb3.set_label("Elevation (m)")

    fig.suptitle(f"Idealized ridge — {Nx}×{Ny} at {dx:.0f} m", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved terrain figure: {out_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", help="skip PNG output")
    args = parser.parse_args()

    print(f"Generating terrain: {Nx}×{Ny} at {dx}m  ({Nx*dx/1e3:.2f} km × {Ny*dy/1e3:.2f} km)")
    topo, x, y, Lx, Ly = build_terrain(Nx, Ny, dx, dy, HILLS)
    print(f"  Max elevation: {topo.max():.1f} m  |  Min: {topo.min():.1f} m")

    write_fasteddy_terrain(BIN_FILE, topo, Nx, Ny)

    if not args.no_plot:
        plot_terrain(topo, x, y, Lx, Ly, PNG_FILE)
