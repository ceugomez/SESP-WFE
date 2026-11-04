# plot_transect_path.py
# cgf 2026-07-15
#
# Path of the transiting point observer used by scripts/run_kf_transect_sweep.jl: one
# sensor at fixed x and fixed height AGL, translating up and down along y as a triangle
# wave. The 15-sensor tower rake is overlaid for scale.
#
# The path here MIRRORS the Julia get_transect_sequence: triangle wave in y between
# crest_y +/- TRANSECT_HALFY_M at constant ground speed, snapped to the nearest grid cell
# at each output time, height re-resolved per position so it stays at z_agl above ground.
# If the Julia transect constants change, change them here too.
#
# Usage: python plot_transect_path.py [--nc-dir DIR] [--x 3000] [--z-agl 100]
#                                     [--halfy 1000] [--speed 15] [--stride 1] [--out PNG]

import argparse, glob, os, re
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DIR = "/home/cego6160/workspace/ensemble_runs_paper/ensemble/member_truth/output"
DEFAULT_OUT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/figure/transect_path.png"

MARGIN_M = 600.0
T_SPINUP = 900.0
# Scoring box: crest+2500 m downwind, FULL analysis y-width (the +/-750 m strip was
# dropped so the scoring area stays constant across LOO folds), <=250 m AGL.
WAKE_DX_M, WAKE_AGL_M = 2500.0, 250.0
TOWER_X_M = 3000.0
TOWER_Y_OFFSETS_M = [-500.0, -250.0, 0.0, 250.0, 500.0]
TOWER_HEIGHTS_M = [10.0, 50.0, 100.0]


def member_files(d):
    """Output files sorted by numeric model-step suffix (mirrors discover_member_files)."""
    fs = []
    for p in glob.glob(os.path.join(d, "*.*")):
        m = re.match(r"^\d+$", os.path.basename(p).rsplit(".", 1)[-1])
        if m:
            fs.append((int(os.path.basename(p).rsplit(".", 1)[-1]), p))
    fs.sort(key=lambda t: t[0])
    return [p for _, p in fs]


def load_grid_and_times(d):
    files = member_files(d)
    with nc.Dataset(files[0]) as ds:
        x = np.asarray(ds["xPos"][0]).transpose(2, 1, 0)
        y = np.asarray(ds["yPos"][0]).transpose(2, 1, 0)
        z = np.asarray(ds["zPos"][0]).transpose(2, 1, 0)
    t = []
    for f in files:
        with nc.Dataset(f) as ds:
            t.append(float(np.asarray(ds["time"]).ravel()[0]))
    return x, y, z, np.array(t)


def nearest_cell(x, y, z, mask, x_m, y_m, z_agl_m):
    """Mirrors fasteddy_io.nearest_cell: nearest i,j then nearest k above that column."""
    i = int(np.argmin(np.abs(x[:, 0, 0] - x_m)))
    j = int(np.argmin(np.abs(y[0, :, 0] - y_m)))
    k = int(np.argmin(np.abs(z[i, j, :] - (z[i, j, 0] + z_agl_m))))
    if not mask[i, j, k]:
        raise ValueError(f"transect cell ({i},{j},{k}) is inside the sponge margin")
    return i, j, k


def transect(x, y, z, mask, times, x_m, z_agl_m, y_lo, y_hi, speed):
    """Triangle wave in y, snapped to grid cells — mirrors get_transect_sequence."""
    span = y_hi - y_lo
    t0 = times[0]
    out = []
    for t in times:
        s = speed * (t - t0)
        u = np.mod(s, 2 * span)
        yy = y_lo + u if u <= span else y_hi - (u - span)
        out.append(nearest_cell(x, y, z, mask, x_m, yy, z_agl_m))
    return np.array(out)


def rake(x, y, z, crest_y):
    out = []
    for dy in TOWER_Y_OFFSETS_M:
        for h in TOWER_HEIGHTS_M:
            i = int(np.argmin(np.abs(x[:, 0, 0] - TOWER_X_M)))
            j = int(np.argmin(np.abs(y[0, :, 0] - (crest_y + dy))))
            k = int(np.argmin(np.abs(z[i, j, :] - (z[i, j, 0] + h))))
            out.append((i, j, k))
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc-dir", default=DEFAULT_DIR)
    p.add_argument("--x", type=float, default=3000.0)
    p.add_argument("--z-agl", type=float, default=100.0)
    p.add_argument("--halfy", type=float, default=1900.0)
    p.add_argument("--speed", type=float, default=78.0)
    p.add_argument("--stride", type=int, default=1, help="OBS_STRIDE: assimilate every Nth snapshot")
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()

    x, y, z, t_all = load_grid_and_times(a.nc_dir)
    Nx, Ny, Nz = x.shape
    terrain = z[:, :, 0]

    dx = x[1, 0, 0] - x[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    cx, cy = int(round(MARGIN_M / dx)), int(round(MARGIN_M / dy))
    mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    mask[cx:Nx - cx, cy:Ny - cy, :] = True

    ci, cj = np.unravel_index(np.argmax(terrain), terrain.shape)
    crest_x, crest_y = x[ci, cj, 0], y[ci, cj, 0]

    times = t_all[t_all >= T_SPINUP][::a.stride]
    P = transect(x, y, z, mask, times, a.x, a.z_agl, crest_y - a.halfy, crest_y + a.halfy, a.speed)
    px, py = x[P[:, 0], P[:, 1], P[:, 2]], y[P[:, 0], P[:, 1], P[:, 2]]
    p_agl = z[P[:, 0], P[:, 1], P[:, 2]] - terrain[P[:, 0], P[:, 1]]
    Rk = rake(x, y, z, crest_y)
    rx, ry = x[Rk[:, 0], Rk[:, 1], Rk[:, 2]], y[Rk[:, 0], Rk[:, 1], Rk[:, 2]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # --- plan view: where the observer goes ---
    cf = ax1.contourf(x[:, :, 0] / 1000, y[:, :, 0] / 1000, terrain, levels=20, cmap="terrain")
    plt.colorbar(cf, ax=ax1, label="terrain elevation (m)")
    iy_dom = np.where(mask.any(axis=(0, 2)))[0]
    y_s_lo, y_s_hi = y[0, iy_dom[0], 0], y[0, iy_dom[-1], 0]
    ax1.add_patch(plt.Rectangle((crest_x / 1000, y_s_lo / 1000),
                                WAKE_DX_M / 1000, (y_s_hi - y_s_lo) / 1000,
                                fill=False, ec="k", lw=1.5, ls="--", label="wake scoring box"))
    sc = ax1.scatter(px / 1000, py / 1000, c=times, cmap="plasma", s=14, zorder=3,
                     label=f"transect samples ({len(P)})")
    ax1.plot(rx / 1000, ry / 1000, "^", ms=7, mfc="none", mec="k", mew=1.5,
             label=f"tower rake ({len(Rk)})")
    ax1.plot(crest_x / 1000, crest_y / 1000, "*", ms=14, color="yellow", mec="k", label="crest")
    for v in (MARGIN_M, x[-1, 0, 0] - MARGIN_M):
        ax1.axvline(v / 1000, color="0.3", lw=0.8)
    ax1.axhline(MARGIN_M / 1000, color="0.3", lw=0.8)
    ax1.axhline((y[0, -1, 0] - MARGIN_M) / 1000, color="0.3", lw=0.8, label="sponge margin")
    ax1.set_xlabel("x (km)"); ax1.set_ylabel("y (km)")
    ax1.set_title(f"plan view — samples overlap on x = {a.x/1000:.1f} km")
    ax1.set_aspect("equal"); ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- y vs t: the triangle wave, i.e. the actual space-time coverage ---
    ax2.axhspan(y_s_lo / 1000, y_s_hi / 1000, color="0.85", label="scoring y extent")
    ax2.plot(times, py / 1000, "-", color="0.5", lw=0.8, zorder=1)
    ax2.scatter(times, py / 1000, c=times, cmap="plasma", s=14, zorder=2)
    ax2.axhline(crest_y / 1000, color="k", lw=0.8, ls=":", label="crest row")
    cb = plt.colorbar(sc, ax=ax2, label="simulation time (s)")
    step = a.speed * (times[1] - times[0])
    ax2.set_xlabel("simulation time (s)"); ax2.set_ylabel("y (km)")
    ax2.set_title(f"transect y(t): {a.speed:.0f} m/s, {step:.0f} m per update")
    ax2.legend(loc="upper right", fontsize=8)

    fig.savefig(a.out, dpi=150)
    span = 2 * a.halfy
    print(f"{len(P)} updates over {times[0]:.0f}-{times[-1]:.0f} s (dt={times[1]-times[0]:.0f} s)")
    print(f"y range {py.min():.0f}-{py.max():.0f} m, AGL {p_agl.min():.1f}-{p_agl.max():.1f} m")
    print(f"{step:.0f} m along-track per update; {a.speed*(times[-1]-times[0])/span:.1f} traverses")
    print(f"distinct y cells visited: {len(np.unique(P[:,1]))}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
