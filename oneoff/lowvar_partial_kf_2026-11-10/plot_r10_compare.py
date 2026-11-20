"""plot_r10_compare.py — ONE-OFF (2026-11-20): figures for run_r10_compare.jl.

POD-KF (r=10) vs unlocalized EnOI vs unlocalized EnKF (Ne=50), 11-member lowvar pool,
truth = nature run, rake sensors, one row/column per sensor noise level.

  r10_compare_skill.png   rows = σ; cols = field RMSE, spread/RMSE (1 = calibrated), NIS
  r10_compare_coeffs.png  cols = σ; rows = modes 1-4: (α_est − α_truth)/√d with ±2σ band,
                          last row = mean z² = ⟨((α−a)/σ_α)²⟩ per mode (1 = calibrated)
Coefficients are divided by √d so each is that mode's RMS contribution to the field (m/s).

  python plot_r10_compare.py --out out_n67_all11
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

METHODS = [("podkf", "POD-KF r=10", "C0"), ("enoi", "EnOI (unloc.)", "C1"),
           ("enkf", "EnKF Ne=50 (unloc.)", "C2")]
N_MODES_SHOWN = 4

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()
base = os.path.join(args.out, "r10_compare")
summ = pd.read_csv(os.path.join(base, "summary.csv"))
sigmas = sorted(summ.sigma.unique(), reverse=True)
dstate = float(summ.dstate.iloc[0])
figdir = os.path.join(args.out, "figures")
os.makedirs(figdir, exist_ok=True)

data = {}
for s in sigmas:
    d = os.path.join(base, f"s{int(round(100 * s)):03d}")
    data[s] = (pd.read_csv(os.path.join(d, "steps.csv")), pd.read_csv(os.path.join(d, "coeffs.csv")))

# ---------------------------------------------------------------- skill / calibration
fig, axs = plt.subplots(len(sigmas), 3, figsize=(15, 3.8 * len(sigmas)), squeeze=False)
for i, s in enumerate(sigmas):
    st, _ = data[s]
    m_obs = 45
    lo, hi = chi2.ppf([0.025, 0.975], m_obs) / m_obs
    for key, lab, c in METHODS:
        g = st[st.method == key]
        if g.empty:
            continue
        axs[i, 0].plot(g.time, g.field_rmse, color=c, label=lab)
        axs[i, 1].plot(g.time, g.field_spread / g.field_rmse, color=c, label=lab)
        axs[i, 2].plot(g.time, g.nis, color=c, label=lab, lw=1)
    g0 = st[st.method == METHODS[0][0]]
    axs[i, 0].plot(g0.time, g0.prior_rmse, "k--", lw=1, label="prior mean (pool x̄)")
    axs[i, 0].plot(g0.time, g0.oracle_rmse, "k:", lw=1, label="r=10 projection floor")
    axs[i, 1].axhline(1, color="k", lw=0.8)
    axs[i, 2].axhspan(lo, hi, color="0.85", label="95% χ²(45)/45")
    axs[i, 2].axhline(1, color="k", lw=0.8)
    axs[i, 0].set_ylabel(f"σ_sensor = {s:g} m/s\n\nfield RMSE (m/s)")
    axs[i, 0].set_ylim(bottom=0)
    axs[i, 1].set_ylabel("field spread / RMSE")
    axs[i, 1].set_yscale("log")
    axs[i, 2].set_ylabel("NIS")
    axs[i, 2].set_yscale("log")
    for a in axs[i]:
        a.set_xlabel("time (s)")
axs[0, 0].set_title("field RMSE")
axs[0, 1].set_title("calibration: spread / RMSE  (1 = calibrated)")
axs[0, 2].set_title("normalized innovation squared")
axs[0, 0].legend(fontsize=8)
axs[0, 2].legend(fontsize=8)
fig.suptitle("11-member lowvar pool, truth = nature run, 15-sensor rake (45 obs)")
fig.tight_layout()
p = os.path.join(figdir, "r10_compare_skill.png")
fig.savefig(p, dpi=150)
print("wrote", p)

# ---------------------------------------------------------------- coefficients
fig, axs = plt.subplots(N_MODES_SHOWN + 1, len(sigmas), figsize=(7 * len(sigmas), 2.6 * (N_MODES_SHOWN + 1)),
                        squeeze=False)
nrm = np.sqrt(dstate)
for j, s in enumerate(sigmas):
    _, cf = data[s]
    for mode in range(1, N_MODES_SHOWN + 1):
        a = axs[mode - 1, j]
        for key, lab, c in METHODS:
            g = cf[(cf.method == key) & (cf["mode"] == mode)]
            if g.empty:
                continue
            err = (g.alpha - g.truth) / nrm
            band = 2 * g.sigma / nrm
            a.plot(g.time, err, color=c, label=lab)
            a.fill_between(g.time, err - band, err + band, color=c, alpha=0.15, lw=0)
        a.axhline(0, color="k", lw=0.8)
        a.set_ylabel(f"mode {mode}\n(α−a)/√d (m/s)")
    a = axs[N_MODES_SHOWN, j]
    width = 0.27
    modes = np.arange(1, cf["mode"].max() + 1)
    for q, (key, lab, c) in enumerate(METHODS):
        g = cf[cf.method == key]
        if g.empty:
            continue
        z2 = g.assign(z2=((g.alpha - g.truth) / g.sigma) ** 2).groupby("mode").z2.mean()
        a.bar(modes + (q - 1) * width, z2.reindex(modes).values, width, color=c, label=lab)
    a.axhline(1, color="k", lw=0.8)
    a.set_yscale("log")
    a.set_xticks(modes)
    a.set_xlabel("mode")
    a.set_ylabel("mean z²  (1 = calibrated)")
    axs[0, j].set_title(f"σ_sensor = {s:g} m/s — coefficient error ± 2σ")
axs[0, 0].legend(fontsize=8)
fig.tight_layout()
p = os.path.join(figdir, "r10_compare_coeffs.png")
fig.savefig(p, dpi=150)
print("wrote", p)

print(summ.drop(columns=["dstate"]).to_string(index=False))
