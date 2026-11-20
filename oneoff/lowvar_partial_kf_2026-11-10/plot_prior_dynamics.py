#!/usr/bin/env python3
"""plot_prior_dynamics.py — ONE-OFF (2026-11-11) figures for ar1_test.jl / rep_corr_test.jl /
prior_test.jl on the out_n67 run (pool {03,10,11}, 67 snapshots).

Reads <out>/repcorr/{steps,sweep,acf_<exp>,channel_<exp>}.csv and
<out>/priortest/{steps,sweep,coeffs,cost,sigma0_eig_<exp>}.csv; writes <out>/figures/:
  pd_timeseries.png    field RMSE and NIS vs time: RW (production) | AR1 | AR1 + diagonal Σ0,
                       with oracle floor and prior (no assimilation) baselines
  pd_mode2.png         leading-mode coefficients α ± 2σ vs truth, the three configurations
  pd_prior_spectrum.png  Σ0 eigen-spectrum (prior sd per eigen-direction) vs the truth's
                       deviation from μ0 along each direction: where the prior rules the truth out
  pd_repcorr.png       representation-residual uncentered ACF per sensor channel + calibration
                       of R_rep (rep-part mean square vs diag R_rep)
  pd_sweeps.png        mean field RMSE vs R_rep inflation c, prior inflation s, diagonal shrinkage γ

  conda activate fasteddy; python plot_prior_dynamics.py [--out out_n67]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(HERE, "out_n67"), help="run output dir")
args = p.parse_args()
RC, PT = os.path.join(args.out, "repcorr"), os.path.join(args.out, "priortest")
FIG = os.path.join(args.out, "figures")
os.makedirs(FIG, exist_ok=True)

EXPS = [("truth_rake_full_rep", "truth (nature run), basis {03,10,11}"),
        ("loo11_rake_full_rep", "LOO member 11, basis {03,10}")]
DT = 6.0          # s per assimilation step
M_OBS = 45        # 15 rake sensors × (u,v,w)
# (label, dynamics, s/γ column value, colour, style)
CONFIGS = [("RW (production)", "RW", 1.0, "C3", "-"),
           ("AR(1)", "AR1", 1.0, "C0", "-"),
           ("AR(1) + diagonal $\\Sigma_0$", "AR1_shrink", 1.0, "C2", "-")]

pst = pd.read_csv(os.path.join(PT, "steps.csv"))
rst = pd.read_csv(os.path.join(RC, "steps.csv"))
pco = pd.read_csv(os.path.join(PT, "coeffs.csv"))
psw = pd.read_csv(os.path.join(PT, "sweep.csv"))
rsw = pd.read_csv(os.path.join(RC, "sweep.csv"))


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# 1. time series
# ---------------------------------------------------------------------------
lo, hi = chi2.ppf([0.025, 0.975], M_OBS) / M_OBS
fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, constrained_layout=True)
for j, (exp, title) in enumerate(EXPS):
    base = rst[(rst.experiment == exp) & (rst.dynamics == "RW") & (rst.c == 1)].sort_values("step")
    t = base.step.values * DT / 60
    for lab, dyn, sv, col, ls in CONFIGS:
        d = pst[(pst.experiment == exp) & (pst.dynamics == dyn) & (pst.s == sv)].sort_values("step")
        ax[0, j].plot(t, d.field, color=col, ls=ls, lw=1.6, label=lab)
        nis_s = pd.Series(d.nis.values).rolling(5, center=True, min_periods=1).mean()
        ax[1, j].plot(t, nis_s, color=col, lw=1.4, label=lab)
    ax[0, j].plot(t, base.oracle, "k--", lw=1.2, label="oracle floor")
    ax[0, j].plot(t, base.prior, color="0.5", ls=":", lw=1.6, label="prior mean (no assimilation)")
    ax[0, j].set_title(title)
    ax[0, j].set_ylabel("full-domain RMSE (m/s)")
    ax[1, j].axhspan(lo, hi, color="0.85", label="95% band, single step")
    ax[1, j].axhline(1, color="k", lw=0.8)
    ax[1, j].set_ylabel("NIS per obs (5-step running mean)")
    ax[1, j].set_xlabel("time since first assimilation (min)")
ax[0, 0].legend(fontsize=8, loc="upper left")
ax[1, 0].legend(fontsize=8, loc="upper left")
save(fig, "pd_timeseries.png")

# ---------------------------------------------------------------------------
# 2. leading-mode coefficients
# ---------------------------------------------------------------------------
modes = [1, 2, 3, 4]
fig, ax = plt.subplots(len(modes), 2, figsize=(13, 10), sharex=True, constrained_layout=True)
for j, (exp, title) in enumerate(EXPS):
    for i, mo in enumerate(modes):
        a = ax[i, j]
        tr = None
        for lab, dyn, sv, col, _ in CONFIGS:
            d = pco[(pco.experiment == exp) & (pco.dynamics == dyn) & (pco.s == sv) & (pco["mode"] == mo)].sort_values("step")
            t = d.step.values * DT / 60
            a.plot(t, d.alpha, color=col, lw=1.4, label=lab)
            a.fill_between(t, d.alpha - 2 * d.sigma, d.alpha + 2 * d.sigma, color=col, alpha=0.15, lw=0)
            tr = (t, d.truth.values)
        a.plot(*tr, "k", lw=1.8, label="truth projection $a_k$")
        a.set_ylabel(f"$\\alpha_{mo}$")
        if i == 0:
            a.set_title(title)
    ax[-1, j].set_xlabel("time since first assimilation (min)")
ax[0, 0].legend(fontsize=8)
fig.suptitle("Leading POD coefficients: estimate ± 2σ vs truth")
save(fig, "pd_mode2.png")

# ---------------------------------------------------------------------------
# 3. prior spectrum vs truth deviation
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
for j, (exp, title) in enumerate(EXPS):
    e = pd.read_csv(os.path.join(PT, f"sigma0_eig_{exp}.csv"))
    a = ax[j]
    a.semilogy(e.eig_rank, np.sqrt(e["lambda"]), "o-", ms=3, color="C0", label="prior sd $\\sqrt{\\lambda_j}$")
    a.semilogy(e.eig_rank, np.abs(e.truth_dev), "s", ms=4, color="C3", label="|truth − $\\mu_0$| along $v_j$ (step 1)")
    worst = e.loc[e.mahal_contrib.idxmax()]
    a.annotate(f"{abs(worst.truth_dev) / np.sqrt(worst['lambda']):.0f} sd\n(mode-2 loading {worst.mode2_loading:+.2f})",
               (worst.eig_rank, abs(worst.truth_dev)), xytext=(-110, -10), textcoords="offset points",
               arrowprops=dict(arrowstyle="->"), fontsize=9)
    a.set_xlabel("eigen-direction of $\\Sigma_0$ (rank)")
    a.set_title(title)
ax[0].set_ylabel("coefficient units")
ax[0].legend(fontsize=8)
save(fig, "pd_prior_spectrum.png")

# ---------------------------------------------------------------------------
# 4. representation residual
#    Lag-0 power has the known sensor-noise variance removed, so channels whose rep part is
#    small vs the noise give a near-zero denominator and meaningless ACFs. Show only channels
#    with rep mean square > sensor noise variance (coloured by that ratio); τ_int is reported
#    for those channels AND for all channels (the "all" median is noise-dominated).
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
cmap = plt.get_cmap("viridis")
for j, (exp, title) in enumerate(EXPS):
    acf = pd.read_csv(os.path.join(RC, f"acf_{exp}.csv"))
    ch = pd.read_csv(os.path.join(RC, f"channel_{exp}.csv"))
    snr = (ch.rep_ms / ch.Rsensor_diag).values
    keep = snr > 1
    lags = np.arange(acf.shape[1] - 2)
    R = acf.iloc[:, 2:].values
    for row, q in zip(R[keep], snr[keep]):
        ax[j].plot(lags * DT, row, color=cmap(min(np.log10(q) / 1.0, 1.0)), lw=1.0)
    ax[j].plot(lags * DT, np.median(R[keep], axis=0), "k", lw=2, label="median (shown channels)")
    ax[j].axhline(0, color="k", lw=0.6)
    ax[j].set_ylim(-1.2, 1.6)
    ax[j].set_title(f"{title}\n{keep.sum()}/{len(keep)} channels with rep > sensor noise: "
                    f"median $\\tau_{{int}}$ = {np.median(ch.tau[keep]):.1f} steps\n"
                    f"(rep > 3× noise: {np.median(ch.tau[snr > 3]):.0f} steps, n={int((snr > 3).sum())}; "
                    f"all channels: {ch.tau.median():.1f})", fontsize=9)
    ax[j].set_xlabel("lag (s)")
    col = "C0" if j == 0 else "C3"
    ax[2].loglog(ch.Rrep_diag, ch.rep_ms, "o", ms=4, color=col,
                 label=f"{exp.split('_')[0]}: median ratio {np.median(ch.rep_ms / ch.Rrep_diag):.2f}")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(1, 10))
fig.colorbar(sm, ax=ax[1], label="rep / sensor-noise variance", shrink=0.8)
ax[0].set_ylabel("uncentered ACF of rep. residual $y - Ha$")
ax[0].legend(fontsize=8)
noise = pd.read_csv(os.path.join(RC, f"channel_{EXPS[0][0]}.csv")).Rsensor_diag.iloc[0]
ax[2].axhline(noise, color="0.4", ls=":", lw=1, label="sensor noise variance")
lim = [min(ax[2].get_xlim()[0], ax[2].get_ylim()[0]), max(ax[2].get_xlim()[1], ax[2].get_ylim()[1])]
ax[2].plot(lim, lim, "k--", lw=0.8)
ax[2].set_xlabel("diag $R_{rep}$ (m²/s²)")
ax[2].set_ylabel("measured rep. mean square (m²/s²)")
ax[2].set_title("R_rep calibration per channel")
ax[2].legend(fontsize=8)
save(fig, "pd_repcorr.png")

# ---------------------------------------------------------------------------
# 5. sweeps
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for j, (exp, title) in enumerate(EXPS):
    r = rsw[rsw.experiment == exp]
    q = psw[psw.experiment == exp]
    panels = [
        (r[r.dynamics == "RW"], r[r.dynamics == "AR1"], "c", "R_rep inflation $c$", True),
        (q[q.dynamics == "RW"], q[q.dynamics == "AR1"], "s", "prior inflation $s$ ($\\Sigma_0 \\to s\\Sigma_0$)", True),
        (q[q.dynamics == "RW_shrink"], q[q.dynamics == "AR1_shrink"], "s", "shrinkage to diagonal $\\gamma$", False),
    ]
    for k, (rw, ar, xc, xl, logx) in enumerate(panels):
        a = ax[j, k]
        if logx:
            rw, ar = rw[rw[xc] > 0], ar[ar[xc] > 0]
        rw, ar = rw.sort_values(xc), ar.sort_values(xc)
        a.plot(rw[xc], rw.field_mean, "o-", color="C3", label="RW")
        a.plot(ar[xc], ar.field_mean, "o-", color="C0", label="AR(1)")
        a.axhline(q.prior_mean.iloc[0], color="0.5", ls=":", lw=1.6, label="prior mean")
        a.axhline(q.oracle_mean.iloc[0], color="k", ls="--", lw=1.0, label="oracle floor")
        if logx:
            a.set_xscale("log")
        else:
            a.set_xscale("symlog", linthresh=0.01, linscale=0.5)
            a.set_xlim(-0.002, 1.1)
            a.set_xticks([0, 0.01, 0.1, 0.5, 1])
            a.set_xticklabels(["0", "0.01", "0.1", "0.5", "1"])
        a.set_xlabel(xl)
        a.set_ylim(0, min(0.45, a.get_ylim()[1]))
        if k == 0:
            a.set_ylabel(f"{title}\nmean full-domain RMSE (m/s)")
ax[0, 0].legend(fontsize=8)
save(fig, "pd_sweeps.png")
