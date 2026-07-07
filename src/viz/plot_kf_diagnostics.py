# plot_kf_diagnostics.py
# cgf  2026-06-25
#
# Performance + consistency figures for a reduced-order KF assimilation run
# (output of scripts/run_simple_KF_assimilation.jl -> save_kf_result). Reads the two
# CSVs written there; no JLD2 needed. Three panels:
#   (a) leading-mode coefficient trajectories: filter α ± σ vs the oracle (truth
#       projected onto the subspace). Shows tracking + whether the posterior band
#       is calibrated (truth should sit inside ±σ ~most of the time).
#   (b) physical-field RMSE vs time with the ORACLE PROJECTION FLOOR overlaid. The
#       floor is the irreducible rank-r truncation error; the gap above it is the
#       online filter's own error (the error-budget split, design notes §10).
#   (c) innovation consistency: NIS per step with its E[NIS]=1 reference and the
#       95% χ² band — the calibration check (NIS ~1 ⇒ R_total is right).
#
# Usage:
#   python plot_kf_diagnostics.py <kf_diagnostics_truth_<id>.csv> \
#       [--coeffs kf_coeffs_truth_<id>.csv] [--out PNG]
# If --coeffs is omitted it is inferred by swapping the filename prefix.

import argparse
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="kf_diagnostics_truth_<id>.csv (step,time,coeff_rmse,field_rmse,oracle_rmse,nis)")
    p.add_argument("--coeffs", default=None,
                   help="kf_coeffs_truth_<id>.csv (step,time,mode,alpha,sigma,truth); "
                        "default: inferred from the diagnostics path")
    p.add_argument("--nmodes", type=int, default=4, help="leading modes to draw in panel (a)")
    p.add_argument("--out", default=None, help="output PNG (default: alongside csv)")
    args = p.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    t = np.asarray(d["time"], dtype=float)

    coeffs_path = args.coeffs or args.csv.replace("kf_diagnostics_", "kf_coeffs_")
    have_coeffs = os.path.isfile(coeffs_path)
    if have_coeffs:
        c = np.genfromtxt(coeffs_path, delimiter=",", names=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    # --- (a) coefficient trajectories α ± σ vs oracle ---
    axa = axes[0]
    if have_coeffs:
        modes = np.unique(c["mode"].astype(int))[: args.nmodes]
        for i, mi in enumerate(modes):
            sel = c["mode"].astype(int) == mi
            tt = c["time"][sel]; a = c["alpha"][sel]; sig = c["sigma"][sel]; tr = c["truth"][sel]
            col = f"C{i}"
            axa.fill_between(tt, a - sig, a + sig, color=col, alpha=0.20, lw=0)
            axa.plot(tt, a, "-", color=col, lw=1.2, label=f"mode {mi}: filter ±σ")
            axa.plot(tt, tr, "--", color=col, lw=1.0)
        axa.plot([], [], "k--", lw=1.0, label="oracle (truth-projected)")
        axa.legend(fontsize=7, ncol=1)
    else:
        axa.text(0.5, 0.5, "no coeffs CSV", ha="center", va="center", transform=axa.transAxes)
    axa.set_xlabel("time (s)"); axa.set_ylabel("POD coefficient")
    axa.set_title("(a) coefficient tracking  (filter ±σ vs oracle)")
    axa.grid(True, alpha=0.3)

    # --- (b) field RMSE vs time + oracle floor ---
    axb = axes[1]
    axb.plot(t, d["field_rmse"], "-", color="C0", lw=1.3, label="filter field RMSE")
    axb.plot(t, d["oracle_rmse"], "-", color="C3", lw=1.2, label="oracle projection floor")
    axb.fill_between(t, d["oracle_rmse"], d["field_rmse"],
                     where=d["field_rmse"] >= d["oracle_rmse"],
                     color="C0", alpha=0.15, lw=0, label="online filter error")
    axb.set_xlabel("time (s)"); axb.set_ylabel("RMSE (m/s)")
    axb.set_title("(b) reconstruction error vs truncation floor")
    axb.grid(True, alpha=0.3); axb.legend(fontsize=8)

    # --- (c) innovation consistency (NIS) ---
    axc = axes[2]
    nis = np.asarray(d["nis"], dtype=float)
    axc.plot(t, nis, "o-", ms=2.5, lw=0.8, color="C0", label="NIS")
    axc.axhline(1.0, color="k", lw=1.0, ls="-", label="E[NIS] = 1 (consistent)")
    # 95% band: NIS is mean of a normalized χ²; per-step dim varies, so use a nominal
    # dim from the mean NIS scaling only as a light guide (dim ~ 3·n_sensors).
    axc.axhspan(0.5, 1.5, color="gray", alpha=0.12, label="~consistent band")
    axc.set_xlabel("time (s)"); axc.set_ylabel("normalized innovation squared")
    axc.set_title(f"(c) innovation consistency  (mean NIS = {nis.mean():.2f})")
    axc.grid(True, alpha=0.3); axc.legend(fontsize=8)

    fig.suptitle(f"KF assimilation diagnostics — {os.path.basename(args.csv)}")
    _fig_dir = pathlib.Path(__file__).resolve().parents[2] / "figure"
    _fig_dir.mkdir(exist_ok=True)
    out = args.out or str(_fig_dir / "kf_diagnostics.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
