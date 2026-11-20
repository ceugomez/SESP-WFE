#!/usr/bin/env python3
"""plot_rank_sweep.py — ONE-OFF (2026-11-10): full-field RMSE vs truncation rank, from
out_n<N>/kf/rank_sweep.csv (run_rank_sweep.jl). Writes out_n<N>/figures/rank_sweep.png.

(a) full-domain [u,v,w] RMSE vs rank: filter estimate (solid, per sweep), oracle projection
    floor (dashed, per truth), prior mean (dotted, per truth); RMS over all K steps and cells.
(b) the filter's excess error over its floor, est − oracle (2nd half of the record, i.e.
    after the initial transient).
(c) mean NIS vs rank (1 = consistent).

  python plot_rank_sweep.py [--out out_n67]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(HERE, "out_n67"))
args = p.parse_args()

df = pd.read_csv(os.path.join(args.out, "kf", "rank_sweep.csv"))
FIG = os.path.join(args.out, "figures"); os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
truth_style = {}
for i, (name, g) in enumerate(df.groupby("sweep", sort=False)):
    g = g.sort_values("rank")
    lab = f"{name}  ({g.n_obs.iloc[0]} obs)"
    ax[0].plot(g["rank"], g.field_est, f"C{i}o-", ms=4, lw=1.5, label=lab)
    ax[1].plot(g["rank"], g.field_est_2ndhalf - g.field_oracle, f"C{i}o-", ms=4, lw=1.5, label=name)
    ax[2].plot(g["rank"], g.mean_nis, f"C{i}o-", ms=4, lw=1.5, label=name)
    truth_style.setdefault(g.truth.iloc[0], g)

for j, (tid, g) in enumerate(truth_style.items()):
    c = "k" if j == 0 else "0.5"
    ax[0].plot(g["rank"], g.field_oracle, "--", color=c, lw=1.2, label=f"oracle floor (truth={tid})")
    ax[0].plot(g["rank"], g.field_prior, ":", color=c, lw=1.4, label=f"prior mean (truth={tid})")

ax[0].set_title("full-domain RMSE vs rank  (RMS over all steps)")
ax[0].set_ylabel("RMSE (m/s)"); ax[0].set_ylim(bottom=0)
ax[1].set_title("filter excess over oracle floor  (2nd half of record)")
ax[1].set_ylabel("RMSE$_{est}$ − RMSE$_{oracle}$ (m/s)"); ax[1].axhline(0, color="k", lw=0.6)
ax[2].set_title("mean NIS"); ax[2].set_yscale("log"); ax[2].axhline(1, color="k", lw=0.8)
for a in ax:
    a.set_xscale("log"); a.set_xlabel("rank r")
    a.set_xticks([2, 3, 5, 10, 20, 50]); a.set_xticklabels(["2", "3", "5", "10", "20", "50"])
    a.legend(fontsize=7.5)
fig.suptitle("Rank sweep — lowvar partial ensemble (basis {03, 10, 11}; LOO-11 basis {03, 10})", fontsize=12)
out = os.path.join(FIG, "rank_sweep.png")
fig.savefig(out, dpi=130); print("wrote", out)
