# generate_terrain_steep.py
# cgf cego6160@colorado.edu 6.10.26
#
# Steep, asymmetric single-hill terrain for the lateral-sponge validation /
# truth run. Goal: force lee-side flow separation -> unsteady shear layer ->
# oscillatory (von Karman-like) wake, WITHOUT relying on a low-Froude stratified
# regime (which a 75 m hill cannot reach without an unphysical inversion).
#
# Asymmetry: gentle windward slope (west, sx_w) and steep lee slope (east, sx_l).
# Mean flow is ~85 deg (toward +x/east) for the alpha=265 deg truth member, so
# the lee face is the +x side. Cross-stream extent set by sy (isolated hill).
#
# FastEddy terrain binary format: [int32 Nx][int32 Ny][float32 topo(Ny, Nx)]
# grid.c transposes on read, so we write C-order (Ny, Nx).

import numpy as np
import struct, os, argparse, math

# --- fixed grid (must match namelist Nx/Ny/dx) ---
Nx, Ny  = 506, 506
dx = dy = 10.0          # m

DEF_OUT = "/home/cego6160/workspace/ensemble_runs_paper/validation_sponge/setup/terrain_steep.bin"


def asym_hill(xx, yy, cx, cy, H, sx_w, sx_l, sy):
    """Asymmetric Gaussian: cross-stream sigma switches at the crest.
    sx_w applies windward (x<cx), sx_l applies lee (x>=cx)."""
    sx = np.where(xx < cx, sx_w, sx_l)
    return H * np.exp(-0.5 * ((xx - cx)**2 / sx**2 + (yy - cy)**2 / sy**2))


def max_slope_deg(H, sigma):
    # Gaussian max |dh/dx| = H/sigma * exp(-1/2) at x = sigma
    return math.degrees(math.atan(0.6065 * H / sigma))


def build(H, sx_w, sx_l, sy, fx, fy):
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    xx, yy = np.meshgrid(x, y)                 # (Ny, Nx)
    cx, cy = fx * Nx * dx, fy * Ny * dy
    topo = asym_hill(xx, yy, cx, cy, H, sx_w, sx_l, sy).astype(np.float32)
    return topo, x, y, cx, cy


def write_fasteddy_terrain(path, topo):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("i", Nx))
        f.write(struct.pack("i", Ny))
        f.write(topo.astype(np.float32).tobytes())
    print(f"Wrote {path}  ({Nx}x{Ny}, {os.path.getsize(path)/1024:.1f} KB)")


def plot(topo, x, y, cx, cy, png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    xx, yy = np.meshgrid(x/1e3, y/1e3)
    cf = a.contourf(xx, yy, topo, levels=40, cmap="terrain")
    a.contour(xx, yy, topo, levels=np.arange(10, topo.max(), 10),
              colors="k", linewidths=0.4, alpha=0.5)
    a.annotate("", xy=(cx/1e3+0.9, cy/1e3), xytext=(cx/1e3-0.9, cy/1e3),
               arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5))
    a.set_aspect("equal"); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
    a.set_title("(a) plan — flow toward +x"); fig.colorbar(cf, ax=a, fraction=0.046)
    # centerline cross-section through the crest
    jc = int(round(cy/dy))
    b.plot(x/1e3, topo[jc, :], "k-")
    b.axvline(cx/1e3, ls=":", c="0.5"); b.set_xlabel("x (km)")
    b.set_ylabel("elev (m)"); b.set_title("(b) crest cross-section"); b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--H",    type=float, default=100.0, help="peak height (m)")
    p.add_argument("--sx_w", type=float, default=300.0, help="windward (west) cross-stream sigma (m)")
    p.add_argument("--sx_l", type=float, default=110.0, help="lee (east) cross-stream sigma (m)")
    p.add_argument("--sy",   type=float, default=250.0, help="spanwise sigma (m)")
    p.add_argument("--fx",   type=float, default=0.25,  help="crest x fraction of domain")
    p.add_argument("--fy",   type=float, default=0.50,  help="crest y fraction of domain")
    p.add_argument("--out",  type=str,   default=DEF_OUT)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    topo, x, y, cx, cy = build(a.H, a.sx_w, a.sx_l, a.sy, a.fx, a.fy)
    print(f"crest at (x,y) = ({cx:.0f}, {cy:.0f}) m  |  peak = {topo.max():.1f} m")
    print(f"windward max slope = {max_slope_deg(a.H, a.sx_w):.1f} deg "
          f"(sigma {a.sx_w:.0f} m)")
    print(f"lee      max slope = {max_slope_deg(a.H, a.sx_l):.1f} deg "
          f"(sigma {a.sx_l:.0f} m)   <-- separation driver")
    print(f"lee half-width = {a.sx_l*1.177:.0f} m ; spanwise half-width = {a.sy*1.177:.0f} m")
    write_fasteddy_terrain(a.out, topo)
    if not a.no_plot:
        plot(topo, x, y, cx, cy, a.out.replace(".bin", ".png"))
