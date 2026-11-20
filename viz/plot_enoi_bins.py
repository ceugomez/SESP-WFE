# plot_enoi_bins.py
# cgf  2026-11-18
#
# POD-KF vs EnOI / localized EnOI / cycled EnKF, binned by 3-D distance to the nearest rake
# sensor. Reads the bins.csv written by oneoff/lowvar_partial_kf_2026-11-10/run_enoi.jl.
# One column per truth (nature run, LOO folds); rows:
#   1. RMSE per distance bin (log y), climatological prior (pool mean) in black dashed
#   2. RMSE / prior RMSE  (<1 = skill over climatology; log y, 1 marked)
#   3. spread / RMSE       (calibration; 1 = consistent, <1 overconfident). POD spread is the
#      in-subspace posterior only, so its ratio excludes the truncation error by construction.
# Tolerates a partial bins.csv (job still running): missing methods/truths are just absent.
#
# Usage:
#   python plot_enoi_bins.py <OUT>/enoi/bins.csv [--out PNG]

import argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

TRUTH_TITLE = {"truth": "nature run", "11": "LOO member 11", "04": "LOO member 04"}


def style(method, exp, c):
    """(label, color, linestyle, marker) for one curve."""
    if method == "pod_kf":
        rep = not exp.endswith("norep")
        return (f"POD-KF ({'R_s+R_rep' if rep else 'R_s only'})",
                "C3", "-" if rep else "--", "o")
    if method == "pod_oracle":
        return "POD projection floor", "C3", ":", "^"
    if method == "enoi":
        if np.isinf(c):
            return "EnOI, no localization", "C0", "-", "s"
        cm = plt.get_cmap("Blues")
        shade = {250: 0.35, 500: 0.5, 1000: 0.7, 2000: 0.9}.get(int(c), 0.6)
        return f"EnOI, c={int(c)} m", cm(shade), "--", "s"
    if method == "enkf":
        cm = plt.get_cmap("Greens")
        shade = {500: 0.45, 1000: 0.7, 2000: 0.9}.get(int(c), 0.6)
        return f"EnKF Ne=50, c={int(c)} m", cm(shade), "-.", "D"
    return exp, "gray", "-", "."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bins")
    p.add_argument("--out", default=None, help="PNG (default <OUT>/figures/enoi_bins.png)")
    a = p.parse_args()

    bins_path = pathlib.Path(a.bins)
    df = pd.read_csv(bins_path)
    df["c_m"] = df["c_m"].astype(float)
    edges = sorted(set(df.bin_lo_m) | set(df.bin_hi_m))
    labels = [f"{int(lo)}–{int(hi)}" if np.isfinite(hi) else f">{int(lo)}"
              for lo, hi in zip(edges[:-1], edges[1:])]
    xi = {lo: i for i, lo in enumerate(edges[:-1])}

    truths = [t for t in ["truth", "11", "04"] if t in set(df.truth.astype(str))]
    df["truth"] = df.truth.astype(str)
    order = {"pod_kf": 0, "pod_oracle": 1, "enoi": 2, "enkf": 3}

    fig, axs = plt.subplots(3, len(truths), figsize=(4.6 * len(truths) + 1.2, 10),
                            sharex=True, sharey="row", squeeze=False)
    handles = {}
    for j, tid in enumerate(truths):
        d = df[df.truth == tid]
        prior = d.drop_duplicates("bin_lo_m").set_index("bin_lo_m").rmse_prior
        px = [xi[b] for b in prior.index]
        h, = axs[0, j].plot(px, prior.values, "k--", lw=1.5, label="prior (pool mean)")
        handles.setdefault(h.get_label(), h)
        for (method, exp), g in sorted(d.groupby(["method", "exp"]),
                                       key=lambda kv: (order.get(kv[0][0], 9),
                                                       -1 if np.isinf(kv[1].c_m.iloc[0]) else kv[1].c_m.iloc[0])):
            g = g.sort_values("bin_lo_m")
            x = [xi[b] for b in g.bin_lo_m]
            lab, col, ls, mk = style(method, exp, g.c_m.iloc[0])
            kw = dict(color=col, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
            if method in ("pod_kf", "pod_oracle"):
                kw.update(lw=2.2, ms=5, zorder=5)
            h, = axs[0, j].plot(x, g.rmse_est, **kw)
            axs[1, j].plot(x, g.rmse_est / g.rmse_prior, **kw)
            if method != "pod_oracle":
                axs[2, j].plot(x, g.spread / g.rmse_est, **kw)
            handles.setdefault(lab, h)
        nrow = d.drop_duplicates("bin_lo_m").sort_values("bin_lo_m").n_rows.values
        axs[0, j].set_title(f"{TRUTH_TITLE.get(tid, tid)}")
        if j == 0:
            frac = 100 * nrow / nrow.sum()
            for k, f in enumerate(frac):
                axs[2, j].annotate(f"{f:.1f}%", (k, 0.0), xycoords=("data", "axes fraction"),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha="center", fontsize=7, color="0.4")

    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for ax in axs.flat:
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.25)
        for axis in (ax.yaxis,):
            axis.set_major_formatter(fmt); axis.set_minor_formatter(fmt)
        ax.tick_params(axis="y", which="minor", labelsize=7)
    for ax in axs[1:, :].flat:
        ax.axhline(1, color="k", lw=0.8)
    axs[0, 0].set_ylabel("RMSE (m/s)")
    axs[1, 0].set_ylabel("RMSE / prior RMSE")
    axs[2, 0].set_ylabel("spread / RMSE")
    for ax in axs[2, :]:
        ax.set_xticks(range(len(labels)), labels, rotation=30)
        ax.set_xlabel("distance to nearest sensor (m)")
    axs[2, 0].text(0.99, 0.10, "% of state rows per bin", transform=axs[2, 0].transAxes,
                   ha="right", fontsize=7, color="0.4")
    fig.legend(handles.values(), handles.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, frameon=False)
    fig.suptitle("Field reconstruction error vs distance from the rake (time-mean over steps)")
    fig.tight_layout()

    out = pathlib.Path(a.out) if a.out else bins_path.parent.parent / "figures" / "enoi_bins.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
