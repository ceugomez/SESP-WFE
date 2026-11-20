# plot_wake_ensemble.py
# cgf  2026-06-30
#
# Overlaid time-averaged wake-deficit contours for all ensemble members.
# For each member, reads all post-spin-up frames, averages u and v at a fixed
# height AGL, normalises speed by the member's geostrophic freestream speed,
# and draws contour lines of constant speed-fraction (e.g. 0.70, 0.80, 0.90).
# Members are coloured by wind direction.  Terrain contours and the truth member
# are drawn for reference.
#
# Usage:
#   python plot_wake_ensemble.py [--ensemble-dir DIR] [--zagl Z]
#                                [--levels 0.70 0.80 0.90] [--spinup-s 1000]
#                                [--dt 0.02] [--out PNG]

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import netCDF4 as nc
from sesp_paths import ENSEMBLE_DIR, figure_dir


# ---------------------------------------------------------------------------
# Helpers

def step_of(path):
    """Return the integer step index from a FastEddy output filename."""
    m = re.search(r"\.(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def time_average_speed(output_dir, member_prefix, min_step, k_level):
    """Return (xkm, ykm, topo, speed_mean) for one member.

    Uses a fixed model-level index k_level rather than AGL interpolation —
    fine for a qualitative wake-envelope comparison; saves substantial I/O.
    """
    pattern = os.path.join(output_dir, f"{member_prefix}.*")
    files = sorted(
        [f for f in glob.glob(pattern)
         if os.path.isfile(f) and step_of(f) >= min_step],
        key=step_of
    )
    if not files:
        raise RuntimeError(f"No post-spin-up files in {output_dir}")

    with nc.Dataset(files[0]) as ds:
        xkm  = ds["xPos"][0, 0, :, :] / 1e3    # (Ny, Nx)
        ykm  = ds["yPos"][0, 0, :, :] / 1e3
        topo = np.array(ds["topoPos"][0], dtype=np.float32)

    acc_u = acc_v = None
    count = 0

    for path in files:
        with nc.Dataset(path) as ds:
            u2d = np.array(ds["u"][0, k_level, :, :], dtype=np.float32)
            v2d = np.array(ds["v"][0, k_level, :, :], dtype=np.float32)
        if acc_u is None:
            acc_u = np.zeros_like(u2d)
            acc_v = np.zeros_like(v2d)
        acc_u += u2d
        acc_v += v2d
        count += 1

    speed = np.sqrt((acc_u / count) ** 2 + (acc_v / count) ** 2)
    return xkm, ykm, topo, speed


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-dir", default=None,
                   help="root ensemble dir (default: $SESP_ENSEMBLE, else ensemble-highvar)")
    p.add_argument("--klevel", type=int, default=15,
                   help="model-level index k for the horizontal slice (default 15, "
                        "≈120 m ASL, safely above the ~100 m ridge crest)")
    p.add_argument("--levels", nargs="+", type=float,
                   default=[0.70, 0.80, 0.90],
                   help="speed-fraction contour levels (default: 0.70 0.80 0.90)")
    p.add_argument("--spinup-s", type=float, default=1000.0,
                   help="spin-up duration to discard (s, default 1000)")
    p.add_argument("--dt", type=float, default=0.02,
                   help="model timestep (s, default 0.02)")
    p.add_argument("--no-truth", action="store_true",
                   help="skip the truth member")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ens_dir = args.ensemble_dir or str(ENSEMBLE_DIR)
    config_path = os.path.join(ens_dir, "ensemble_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    min_step = int(args.spinup_s / args.dt)
    levels = sorted(args.levels)

    # -----------------------------------------------------------------------
    # Colour map: members coloured by met wind direction
    prior_members = cfg["prior_members"]
    dirs = [m["met_dir_deg"] for m in prior_members]
    dir_min, dir_max = min(dirs), max(dirs)

    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    topo_ref = None

    print(f"Processing {len(prior_members)} members at k={args.klevel} "
          f"(min_step={min_step}) ...")

    norm_dir = plt.Normalize(dir_min, dir_max)
    sm = cm.ScalarMappable(cmap="plasma", norm=norm_dir)
    sm.set_array([])

    for mem in prior_members:
        mid = mem["id"]
        out_dir = mem["output_dir"]
        prefix = f"FE_ensemble_{mid}"
        spd_g = mem["spd_ms"]
        met_dir = mem["met_dir_deg"]

        print(f"  member {mid}  dir={met_dir:.1f}°  spd={spd_g:.2f} m/s ...", end=" ", flush=True)
        try:
            xkm, ykm, topo, spd_mean = time_average_speed(
                out_dir, prefix, min_step, args.klevel)
        except RuntimeError as e:
            print(f"SKIPPED ({e})")
            continue

        if topo_ref is None:
            topo_ref = (xkm, ykm, topo)

        frac = spd_mean / spd_g
        color = sm.cmap(norm_dir(met_dir))
        ax.contour(xkm, ykm, frac, levels=levels,
                   colors=[color], linewidths=0.9, alpha=0.85,
                   linestyles=["-", "--", ":"])
        ax.plot([], [], color=color,
                label=f"{mid}: {met_dir:.0f}° / {spd_g:.1f} m/s")
        print("done")

    # --- truth member ---
    if not args.no_truth:
        truth_cfg = cfg["truth_member"]
        out_dir = truth_cfg["output_dir"]
        prefix = "FE_ensemble_truth"
        spd_g = truth_cfg["spd_ms"]
        met_dir = truth_cfg["met_dir_deg"]
        print(f"  truth  dir={met_dir:.1f}°  spd={spd_g:.2f} m/s ...", end=" ", flush=True)
        try:
            xkm, ykm, topo, spd_mean = time_average_speed(
                out_dir, prefix, min_step, args.klevel)
            if topo_ref is None:
                topo_ref = (xkm, ykm, topo)
            frac = spd_mean / spd_g
            ax.contour(xkm, ykm, frac, levels=levels,
                       colors=["k"], linewidths=1.5, alpha=1.0,
                       linestyles=["-", "--", ":"])
            ax.plot([], [], color="k", lw=1.5, label=f"truth: {met_dir:.0f}° / {spd_g:.1f} m/s")
            print("done")
        except RuntimeError as e:
            print(f"SKIPPED ({e})")

    # --- terrain contours ---
    if topo_ref is not None:
        xkm, ykm, topo = topo_ref
        ax.contour(xkm, ykm, topo,
                   levels=np.arange(20, topo.max(), 20),
                   colors="0.5", linewidths=0.35, alpha=0.5, zorder=0)

    # --- contour-level legend proxies (line style) ---
    ls_map = {"-": "solid", "--": "dashed", ":": "dotted"}
    lstyles = ["-", "--", ":"]
    for lev, ls in zip(levels, lstyles):
        ax.plot([], [], color="0.4", lw=1.0, linestyle=ls,
                label=f"U/U_g = {lev:.0%}")

    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("met wind direction (°)", fontsize=9)

    ax.legend(fontsize=6.5, loc="upper left", ncol=2,
              framealpha=0.85, handlelength=1.6)
    ax.set_xlabel("x  (km)")
    ax.set_ylabel("y  (km)")
    ax.set_aspect("equal")
    ax.set_title(
        f"Time-averaged wake deficit — ensemble members\n"
        f"(normalised speed fraction U/U_g, k={args.klevel}, "
        f"t > {args.spinup_s:.0f} s)", fontsize=10)

    out = args.out or str(figure_dir() / "wake_ensemble.png")
    os.makedirs(os.path.dirname(os.path.realpath(out)), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
