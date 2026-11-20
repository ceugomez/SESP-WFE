#!/usr/bin/env python3
"""plot_members.py — ONE-OFF (2026-11-10): what each finished lowvar ensemble member looks like.
Reads the raw FastEddy output directly (no Julia products needed).

  members_z<zzz>.png   one row per member (truth + finished prior members, sorted by forcing
                       speed); columns: horizontal speed at the first and last snapshot of the
                       window, time-mean speed over the window, and time-mean minus the prior
                       pool's time-mean (where the members differ). Dashed box = analysis domain
                       (600 m sponge margin excluded).
  members_forcing.png  forcing samples (speed, direction): finished, pending, truth, with the
                       sampling distribution's 1σ/2σ ellipses.

Window = the first --nsnap post-spinup snapshots (default 67, matching out_n67/).
  python plot_members.py [--z 10,50,100] [--nsnap 67] [--out out_n67] [--refresh]
Slices are cached in <out>/fields/members_slices_n<N>.npz, so re-styling is instant.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import netCDF4 as nc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENS = "/home/cego6160/workspace/ensemble_runs_paper/ensemble-lowvar"
POOL = ["03", "10", "11"]            # finished prior members (hardcoded, as in run_partial_kf.jl)
T_SPINUP, MARGIN_M = 900.0, 600.0

p = argparse.ArgumentParser()
p.add_argument("--z", default="10,50,100", help="slice heights, m AGL")
p.add_argument("--nsnap", type=int, default=67, help="post-spinup snapshots in the window (0 = all)")
p.add_argument("--out", default=os.path.join(HERE, "out_n67"))
p.add_argument("--refresh", action="store_true", help="ignore the slice cache and re-read raw output")
args = p.parse_args()
FIG = os.path.join(args.out, "figures"); os.makedirs(FIG, exist_ok=True)
zs = [float(z) for z in args.z.split(",")]

cfg = json.load(open(os.path.join(ENS, "ensemble_config.json")))
info = {m["id"]: m for m in cfg["prior_members"]}
info["truth"] = cfg["truth_member"]


def window_files(mid):
    fs = glob.glob(os.path.join(ENS, f"member_{mid}", "output", f"FE_ensemble_{mid}.*"))
    fs = sorted(fs, key=lambda f: int(f.rsplit(".", 1)[1]))
    keep = []
    for f in fs:
        with nc.Dataset(f) as d:
            if float(d["time"][0]) >= T_SPINUP:
                keep.append(f)
    return keep[: args.nsnap] if args.nsnap > 0 else keep


# grid + per-column level index for each height (terrain-following; same convention as the
# Julia products: level nearest z_agl above the first-level height)
g0 = nc.Dataset(window_files("truth")[0])
zPos = np.array(g0["zPos"][0])                       # (z, y, x)
x = np.array(g0["xPos"][0, 0, 0, :]) / 1e3; y = np.array(g0["yPos"][0, 0, :, 0]) / 1e3   # 1-D axes (km)
topo = zPos[0]
klev = {z: np.argmin(np.abs(zPos - (topo + z)[None]), axis=0) for z in zs}   # (y, x)
kmin = min(int(k.min()) for k in klev.values()); kmax = max(int(k.max()) for k in klev.values())
yy, xx = np.indices(topo.shape)


def slices(mid):
    """{z: (first, last, mean)} horizontal speed slices over the window."""
    fs = window_files(mid)
    acc = {z: None for z in zs}; first = {}; last = {}
    for i, f in enumerate(fs):
        d = nc.Dataset(f)
        u = np.array(d["u"][0, kmin:kmax + 1]); v = np.array(d["v"][0, kmin:kmax + 1])
        d.close()
        for z in zs:
            k = klev[z] - kmin
            s = np.hypot(u[k, yy, xx], v[k, yy, xx])
            acc[z] = s if acc[z] is None else acc[z] + s
            if i == 0: first[z] = s
            last[z] = s
    print(f"  member {mid}: {len(fs)} snapshots")
    return {z: (first[z], last[z], acc[z] / len(fs)) for z in zs}, len(fs)


ids = ["truth"] + sorted(POOL, key=lambda m: info[m]["spd_ms"])
# Slice cache: the raw read is the slow part, so the (first, last, mean) slices are saved once
# to fields/members_slices_n<N>.npz and reused; --refresh re-reads the raw output.
CACHE = os.path.join(args.out, "fields", f"members_slices_n{args.nsnap}.npz")
S = {}
cached = {} if args.refresh or not os.path.isfile(CACHE) else dict(np.load(CACHE))
for mid in ids:
    if all(f"{mid}_z{z:g}" in cached for z in zs):
        S[mid] = {z: tuple(cached[f"{mid}_z{z:g}"]) for z in zs}; nwin = int(cached["nwin"])
    else:
        S[mid], nwin = slices(mid)
        cached.update({f"{mid}_z{z:g}": np.stack(S[mid][z]) for z in zs}); cached["nwin"] = nwin
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
np.savez_compressed(CACHE, **cached)

dx = (x[1] - x[0]); mlo = x[0] + MARGIN_M / 1e3 - dx / 2
box = dict(xy=(mlo, y[0] + MARGIN_M / 1e3 - dx / 2), width=x[-1] - x[0] - 2 * MARGIN_M / 1e3 + dx,
           height=y[-1] - y[0] - 2 * MARGIN_M / 1e3 + dx)

for z in zs:
    pool_mean = np.mean([S[m][z][2] for m in POOL], axis=0)
    allv = np.stack([S[m][z][c] for m in ids for c in range(3)])
    lo, hi = np.nanpercentile(allv, 1), np.nanpercentile(allv, 99)
    dev = np.stack([S[m][z][2] - pool_mean for m in ids]); dmax = np.nanpercentile(np.abs(dev), 99.5)

    fig, ax = plt.subplots(len(ids), 4, figsize=(19, 4.1 * len(ids)), constrained_layout=True, squeeze=False)
    for ri, mid in enumerate(ids):
        f0, f1, fm = S[mid][z]
        cols = [("first snapshot", f0, "viridis", lo, hi), ("last snapshot", f1, "viridis", lo, hi),
                (f"time mean ({nwin} snapshots)", fm, "viridis", lo, hi),
                ("time mean − pool {" + ",".join(POOL) + "} mean", fm - pool_mean, "RdBu_r", -dmax, dmax)]
        for ci, (title, F, cmap, vlo, vhi) in enumerate(cols):
            a = ax[ri, ci]
            pc = a.pcolormesh(x, y, F, cmap=cmap, vmin=vlo, vmax=vhi, shading="auto")
            a.contour(x, y, topo, levels=6, colors="k" if cmap == "RdBu_r" else "w", linewidths=0.4, alpha=0.6)
            a.add_patch(Rectangle(fill=False, ls="--", lw=0.8, ec="k", **box))
            fig.colorbar(pc, ax=a, shrink=0.85, label="m/s")
            a.set_aspect("equal")
            if ri == 0: a.set_title(title, fontsize=11)
            if ri == len(ids) - 1: a.set_xlabel("x (km)")
        m = info[mid]
        role = "TRUTH (nature run)" if mid == "truth" else "prior member"
        ax[ri, 0].set_ylabel(f"{mid} — {role}\n{m['spd_ms']:.2f} m/s @ {m['met_dir_deg']:.1f}°\ny (km)")
    fig.suptitle(f"lowvar ensemble members: horizontal wind speed at {z:.0f} m AGL "
                 f"(window: first {nwin} post-spinup snapshots, 6 s apart)", fontsize=13)
    out = os.path.join(FIG, f"members_z{int(z):03d}.png")
    fig.savefig(out, dpi=105); plt.close(fig); print("wrote", out)

# forcing-space figure
e = cfg["ensemble"]
fig, a = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
for k, al in ((1, 0.25), (2, 0.12)):
    a.add_patch(Ellipse((e["mu_dir_deg"], e["mu_spd_ms"]), 2 * k * e["sigma_dir_deg"],
                        2 * k * e["sigma_spd_ms"], color="C0", alpha=al, lw=0))
for m in cfg["prior_members"]:
    done = m["id"] in POOL
    a.plot(m["met_dir_deg"], m["spd_ms"], "o", ms=9, mfc="C0" if done else "none", mec="C0")
    a.annotate(m["id"], (m["met_dir_deg"], m["spd_ms"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
t = cfg["truth_member"]
a.plot(t["met_dir_deg"], t["spd_ms"], "r*", ms=15, label="truth")
a.plot([], [], "o", mfc="C0", mec="C0", label="finished (in basis)")
a.plot([], [], "o", mfc="none", mec="C0", label="pending")
a.set_xlabel("forcing direction (° met)"); a.set_ylabel("forcing speed (m/s)")
a.set_title("lowvar forcing samples (ellipses: 1σ, 2σ)"); a.legend(fontsize=8)
out = os.path.join(FIG, "members_forcing.png")
fig.savefig(out, dpi=130); print("wrote", out)
