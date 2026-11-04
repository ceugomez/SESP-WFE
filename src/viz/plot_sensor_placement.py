# plot_sensor_placement.py
# cgf 2026-07-15
#
# Sensor placement for the downwind-array rank sweep (scripts/run_kf_rank_sweep_dense.jl),
# with the 15-sensor tower rake (scripts/run_kf_rank_sweep.jl) overlaid for scale.
#
# The array selection here MIRRORS the Julia: sensors span the downwind half of the
# analysis domain (crest to interior x edge, full interior y), every SENSOR_STRIDE_XY
# cells in x,y and every SENSOR_STRIDE_Z-th level below 250 m AGL, with levels chosen
# per-column because the sigma coordinate rides the terrain. The wake scoring box
# (crest+2500 m, +/-750 m) is drawn for reference — the array is deliberately wider.
# If the Julia span/stride constants change, change them here too.
#
# Usage: python plot_sensor_placement.py [--nc FILE] [--xy 50] [--z 10] [--out PNG]

import argparse
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_NC = ("/home/cego6160/workspace/ensemble_runs_paper/ensemble/member_truth/"
              "output/FE_ensemble_truth.100200")
DEFAULT_OUT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/figure/sensor_placement.png"

MARGIN_M = 600.0                                    # sponge fringe excluded from the state
# Scoring box: crest+2500 m downwind, FULL analysis y-width (the +/-750 m strip was
# dropped so the scoring area stays constant across LOO folds), <=250 m AGL.
WAKE_DX_M, WAKE_AGL_M = 2500.0, 250.0
TOWER_X_M = 3000.0                                  # the rake, for comparison
TOWER_Y_OFFSETS_M = [-500.0, -250.0, 0.0, 250.0, 500.0]
TOWER_HEIGHTS_M = [10.0, 50.0, 100.0]


def load_grid(path):
    with nc.Dataset(path) as d:
        # NetCDF order is (time, z, y, x); transpose to (x, y, z) to match the Julia grid.
        x = np.asarray(d["xPos"][0]).transpose(2, 1, 0)
        y = np.asarray(d["yPos"][0]).transpose(2, 1, 0)
        z = np.asarray(d["zPos"][0]).transpose(2, 1, 0)
    return x, y, z


def dense_lattice(x, y, z, stride_xy, stride_z, crest_x, crest_y, mask):
    """Grid indices (i,j,k) of the downwind array — same rule as the Julia driver."""
    Nx, Ny, Nz = x.shape
    ix_dom = np.where(mask.any(axis=(1, 2)))[0]
    iy_dom = np.where(mask.any(axis=(0, 2)))[0]
    ix_hi, iy_lo, iy_hi = ix_dom[-1], iy_dom[0], iy_dom[-1]
    ix = max(ix_dom[0], np.where(x[:, 0, 0] >= crest_x)[0][0])
    out = []
    for i in range(ix, ix_hi + 1, stride_xy):
        for j in range(iy_lo, iy_hi + 1, stride_xy):
            terr = z[i, j, 0]
            for k in range(1, Nz, stride_z):           # k=1 -> Julia's level 2 (skip surface)
                if z[i, j, k] - terr > WAKE_AGL_M:
                    break
                if not mask[i, j, k]:
                    continue
                out.append((i, j, k))
    return np.array(out)


def rake(x, y, z, crest_y):
    """The 15-sensor tower rake, nearest-cell as in fasteddy_io.nearest_cell."""
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
    p.add_argument("--nc", default=DEFAULT_NC)
    p.add_argument("--xy", type=int, default=50, help="sensor stride in x,y (grid cells)")
    p.add_argument("--z", type=int, default=10, help="sensor stride in vertical levels")
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()

    x, y, z = load_grid(a.nc)
    Nx, Ny, Nz = x.shape
    terrain = z[:, :, 0]

    # analysis domain (sponge margin removed), matching AnalysisDomain(margin_m=600)
    dx = x[1, 0, 0] - x[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    cx, cy = int(round(MARGIN_M / dx)), int(round(MARGIN_M / dy))
    mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    mask[cx:Nx - cx, cy:Ny - cy, :] = True

    ci, cj = np.unravel_index(np.argmax(terrain), terrain.shape)
    crest_x, crest_y = x[ci, cj, 0], y[ci, cj, 0]
    iy_dom = np.where(mask.any(axis=(0, 2)))[0]

    L = dense_lattice(x, y, z, a.xy, a.z, crest_x, crest_y, mask)
    Rk = rake(x, y, z, crest_y)
    lx, ly, lz = x[L[:, 0], L[:, 1], L[:, 2]], y[L[:, 0], L[:, 1], L[:, 2]], z[L[:, 0], L[:, 1], L[:, 2]]
    l_agl = lz - terrain[L[:, 0], L[:, 1]]
    rx, ry, rz = x[Rk[:, 0], Rk[:, 1], Rk[:, 2]], y[Rk[:, 0], Rk[:, 1], Rk[:, 2]], z[Rk[:, 0], Rk[:, 1], Rk[:, 2]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # --- plan view: terrain + lattice columns ---
    cf = ax1.contourf(x[:, :, 0] / 1000, y[:, :, 0] / 1000, terrain, levels=20, cmap="terrain")
    plt.colorbar(cf, ax=ax1, label="terrain elevation (m)")
    y_s_lo, y_s_hi = y[0, iy_dom[0], 0], y[0, iy_dom[-1], 0]
    ax1.add_patch(plt.Rectangle((crest_x / 1000, y_s_lo / 1000),
                                WAKE_DX_M / 1000, (y_s_hi - y_s_lo) / 1000,
                                fill=False, ec="k", lw=1.5, ls="--", label="wake scoring box"))
    ax1.plot(lx / 1000, ly / 1000, ".", ms=5, color="crimson", label=f"downwind array ({len(L)})")
    ax1.plot(rx / 1000, ry / 1000, "^", ms=7, mfc="none", mec="k", mew=1.5, label=f"tower rake ({len(Rk)})")
    ax1.plot(crest_x / 1000, crest_y / 1000, "*", ms=14, color="yellow", mec="k", label="crest")
    ax1.axvline(MARGIN_M / 1000, color="0.3", lw=0.8)
    ax1.axvline((x[-1, 0, 0] - MARGIN_M) / 1000, color="0.3", lw=0.8)
    ax1.axhline(MARGIN_M / 1000, color="0.3", lw=0.8)
    ax1.axhline((y[0, -1, 0] - MARGIN_M) / 1000, color="0.3", lw=0.8, label="sponge margin")
    ax1.set_xlabel("x (km)"); ax1.set_ylabel("y (km)")
    ax1.set_title("plan view (columns overlap in z)")
    ax1.set_aspect("equal"); ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- x-z section through the lattice row nearest the crest (the strided rows do not
    #     generally land on the crest row itself) ---
    j_sec = int(np.unique(L[:, 1])[np.argmin(np.abs(y[0, np.unique(L[:, 1]), 0] - crest_y))])
    y_sec = y[0, j_sec, 0]
    sel = L[:, 1] == j_sec
    ax2.fill_between(x[:, j_sec, 0] / 1000, terrain[:, j_sec], color="0.75", label="terrain")
    ax2.plot(lx[sel] / 1000, lz[sel], ".", ms=5, color="crimson", label="downwind array")
    sel_r = np.abs(ry - crest_y) < 2 * dy
    ax2.plot(rx[sel_r] / 1000, rz[sel_r], "^", ms=8, mfc="none", mec="k", mew=1.5, label="tower rake")
    ax2.plot(x[:, j_sec, 0] / 1000, terrain[:, j_sec] + WAKE_AGL_M, "k--", lw=1,
             label=f"{WAKE_AGL_M:.0f} m AGL")
    ax2.set_xlim(0, x[-1, 0, 0] / 1000); ax2.set_ylim(0, 320)
    ax2.set_xlabel("x (km)"); ax2.set_ylabel("z (m)")
    ax2.set_title(f"x–z section at lattice row y = {y_sec/1000:.2f} km")
    ax2.legend(loc="upper right", fontsize=8)

    fig.savefig(a.out, dpi=150)
    nz_lv = len(np.unique(L[:, 2]))
    print(f"{len(L)} sensors ({3*len(L)} obs rows): "
          f"{len(np.unique(L[:,0]))} x {len(np.unique(L[:,1]))} columns x ~{nz_lv} levels")
    print(f"AGL levels used: {np.unique(np.round(l_agl))} m")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
