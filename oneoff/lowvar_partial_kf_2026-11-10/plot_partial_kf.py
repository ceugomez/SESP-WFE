#!/usr/bin/env python3
"""plot_partial_kf.py — ONE-OFF (2026-11-10) figures for run_partial_kf.jl.

Reads ./out/kf/{summary.csv, <exp>_steps.csv, <exp>_coeffs.csv} and ./out/fields/<exp>_snap.nc;
writes ./out/figures/:
  perf_<exp>.png              step-by-step performance: full-domain RMSE (estimate vs oracle
                              floor vs prior vs the filter's self-assessed error), per-height
                              horizontal-wind RMSE, NIS with the 95% chi^2 band, leading-mode
                              coefficients alpha +- 2 sigma vs truth projection
  compare_experiments.png     full-domain and 50 m RMSE vs time for every experiment
  snapshot_<exp>_z<zzz>.png   top-down horizontal wind speed: truth | estimate | oracle |
                              estimate - truth | posterior sigma_h, one row per selected step

  conda activate fasteddy; python plot_partial_kf.py [--out out] [--steps 1,5,30,268]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(HERE, "out"), help="run_partial_kf.jl output dir")
p.add_argument("--steps", default="1,5,30", help="snapshot rows (assimilation steps; must be in SNAP_STEPS)")
p.add_argument("--z", default="", help="slice heights to plot, e.g. 50,100 (default all)")
args = p.parse_args()

KF, FLD = os.path.join(args.out, "kf"), os.path.join(args.out, "fields")
FIG = os.path.join(args.out, "figures")
os.makedirs(FIG, exist_ok=True)
summ = pd.read_csv(os.path.join(KF, "summary.csv"))


def zcols(df):
    return sorted({int(c[1:4]) for c in df.columns if c.startswith("z") and c.endswith("_est")})


# ---------------------------------------------------------------------------
# 1. per-experiment performance
# ---------------------------------------------------------------------------
for _, ex in summ.iterrows():
    name = ex["name"]
    st = pd.read_csv(os.path.join(KF, f"{name}_steps.csv"))
    co = pd.read_csv(os.path.join(KF, f"{name}_coeffs.csv"))
    t = st["time"] - st["time"].iloc[0]

    fig, ax = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    a = ax[0, 0]
    a.plot(t, st.field_est, "C0", lw=1.6, label="filter estimate  $P_r\\alpha_k$")
    a.plot(t, st.field_oracle, "k--", lw=1.2, label="oracle floor  $P_r P_r^T x_k$")
    a.plot(t, st.field_prior, "C3:", lw=1.4, label="prior mean  $P_r\\mu_0$")
    a.plot(t, np.sqrt(st.field_oracle**2 + st.field_spread**2), "C0", lw=0.9, alpha=0.5,
           label="self-assessed  $\\sqrt{\\mathrm{oracle}^2 + \\mathrm{tr}P/d}$")
    a.set_title("full analysis domain, [u,v,w] RMSE"); a.set_ylabel("RMSE (m/s)")
    a.set_ylim(bottom=0); a.legend(fontsize=8)

    a = ax[0, 1]
    for i, z in enumerate(zcols(st)):
        a.plot(t, st[f"z{z:03d}_est"], f"C{i}", lw=1.5, label=f"{z} m AGL estimate")
        a.plot(t, st[f"z{z:03d}_oracle"], f"C{i}--", lw=1.0)
        a.plot(t, st[f"z{z:03d}_prior"], f"C{i}:", lw=1.0)
    a.set_title("horizontal wind RMSE by height  (— est, -- oracle, ··· prior)")
    a.set_ylabel("RMSE (m/s)"); a.set_ylim(bottom=0); a.legend(fontsize=8)

    a = ax[1, 0]
    m = int(ex["n_obs"])
    a.semilogy(t, st.nis, "C2.", ms=3)
    a.axhline(1, color="k", lw=0.8)
    a.axhspan(chi2.ppf(0.025, m) / m, chi2.ppf(0.975, m) / m, color="k", alpha=0.12,
              label=f"95% $\\chi^2_{{{m}}}/{m}$")
    a.set_title(f"NIS  (mean {ex['mean_nis']:.2f})"); a.set_ylabel("$\\nu^T S^{-1}\\nu / m$")
    a.legend(fontsize=8)

    for a, mode in zip([ax[1, 1], ax[2, 0], ax[2, 1]], [1, 2, 3]):
        c = co[co["mode"] == mode]
        tc = c["time"] - st["time"].iloc[0]
        a.fill_between(tc, c.alpha - 2 * c.sigma, c.alpha + 2 * c.sigma, color="C0", alpha=0.25)
        a.plot(tc, c.alpha, "C0", lw=1.3, label="$\\alpha_k \\pm 2\\sigma$")
        a.plot(tc, c.truth, "k", lw=1.0, label="truth projection")
        a.set_title(f"POD coefficient, mode {mode}"); a.legend(fontsize=8)
    for a in ax[-1]:
        a.set_xlabel("time since first assimilation (s)")
    ax[1, 0].set_xlabel("time since first assimilation (s)")

    fig.suptitle(f"{name}: truth={ex['truth']}, basis={{{ex['basis_members']}}}, r={ex['rank']}, "
                 f"{ex['sensors']} ({m} obs), R_rep={'on' if ex['rep'] else 'off'}", fontsize=12)
    out = os.path.join(FIG, f"perf_{name}.png")
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)

# ---------------------------------------------------------------------------
# 2. cross-experiment comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for i, (_, ex) in enumerate(summ.iterrows()):
    st = pd.read_csv(os.path.join(KF, f"{ex['name']}_steps.csv"))
    t = st["time"] - st["time"].iloc[0]
    ax[0].plot(t, st.field_est, f"C{i}", lw=1.4, label=ex["name"])
    ax[0].plot(t, st.field_oracle, f"C{i}--", lw=0.8)
    if "z050_est" in st:
        ax[1].plot(t, st.z050_est, f"C{i}", lw=1.4, label=ex["name"])
        ax[1].plot(t, st.z050_oracle, f"C{i}--", lw=0.8)
ax[0].set_title("full-domain RMSE  (— estimate, -- oracle floor)")
ax[1].set_title("50 m AGL horizontal-wind RMSE  (— estimate, -- oracle floor)")
for a in ax:
    a.set_xlabel("time since first assimilation (s)"); a.set_ylabel("RMSE (m/s)")
    a.set_ylim(bottom=0); a.legend(fontsize=8)
out = os.path.join(FIG, "compare_experiments.png")
fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)

# ---------------------------------------------------------------------------
# 3. top-down snapshots
# ---------------------------------------------------------------------------
want_steps = [int(s) for s in args.steps.split(",")]
for _, ex in summ.iterrows():
    name = ex["name"]
    d = nc.Dataset(os.path.join(FLD, f"{name}_snap.nc"))
    x, y = d["xPos"][:] / 1e3, d["yPos"][:] / 1e3
    topo = np.array(d["topo"][:])   # netCDF4 reverses Julia dim order: 2-D (y,x), 4-D (snap,zlev,y,x)
    zagl = np.array(d["z_agl"][:]); steps = list(np.array(d["step"][:]))
    times = np.array(d["time"][:])
    sxyz = np.array(d["sensor_xyz"][:]).T   # -> (sensor, xyz)
    rows = [steps.index(s) for s in want_steps if s in steps]
    if steps[-1] not in [steps[r] for r in rows]:
        rows.append(len(steps) - 1)            # always show the final step
    zsel = [float(z) for z in args.z.split(",")] if args.z else list(zagl)

    def spd(var, iz, isn):
        u = np.array(d[f"{var}_u"][isn, iz, :, :], float)
        v = np.array(d[f"{var}_v"][isn, iz, :, :], float)
        return np.hypot(u, v)

    for zv in zsel:
        iz = int(np.argmin(np.abs(zagl - zv)))
        truth_all = np.stack([spd("truth", iz, r) for r in rows])
        lo, hi = np.nanpercentile(truth_all, 1), np.nanpercentile(truth_all, 99)
        errs = np.stack([spd("est", iz, r) - spd("truth", iz, r) for r in rows])
        emax = np.nanpercentile(np.abs(errs), 99)
        sig = np.stack([np.hypot(np.array(d["std_u"][r, iz, :, :], float),
                                 np.array(d["std_v"][r, iz, :, :], float)) for r in rows])
        smax = np.nanpercentile(sig, 99)

        nr = len(rows)
        fig, ax = plt.subplots(nr, 5, figsize=(22, 4.0 * nr), constrained_layout=True, squeeze=False)
        for ri, r in enumerate(rows):
            tr, es, orc = spd("truth", iz, r), spd("est", iz, r), spd("oracle", iz, r)
            panels = [("truth  $|u_h|$", tr, "viridis", lo, hi),
                      ("estimate  $|P_r\\alpha_k|_h$", es, "viridis", lo, hi),
                      ("oracle  $|P_r P_r^T x|_h$", orc, "viridis", lo, hi),
                      (f"estimate − truth  (RMSE {np.sqrt(np.nanmean((es - tr)**2)):.2f})",
                       es - tr, "RdBu_r", -emax, emax),
                      ("posterior $\\sigma_h$", sig[ri], "magma", 0, smax)]
            for ci, (title, F, cmap, vlo, vhi) in enumerate(panels):
                a = ax[ri, ci]
                pc = a.pcolormesh(x, y, F, cmap=cmap, vmin=vlo, vmax=vhi, shading="auto")
                a.contour(x, y, topo, levels=6, colors="w" if cmap != "RdBu_r" else "k",
                          linewidths=0.4, alpha=0.6)
                a.scatter(sxyz[:, 0] / 1e3, sxyz[:, 1] / 1e3, s=10, c="r", marker="^",
                          edgecolors="k", linewidths=0.3)
                fig.colorbar(pc, ax=a, shrink=0.85, label="m/s")
                a.set_aspect("equal")
                a.set_title(title if ri == 0 or ci == 3 else "", fontsize=10)
                if ci == 0:
                    a.set_ylabel(f"step {steps[r]}  (t+{times[r] - d['time'][0]:.0f} s)\ny (km)")
                if ri == nr - 1:
                    a.set_xlabel("x (km)")
        fig.suptitle(f"{name}: horizontal wind speed at {zagl[iz]:.0f} m AGL  (▲ = sensor columns)",
                     fontsize=13)
        out = os.path.join(FIG, f"snapshot_{name}_z{int(zagl[iz]):03d}.png")
        fig.savefig(out, dpi=110); plt.close(fig); print("wrote", out)
    d.close()
