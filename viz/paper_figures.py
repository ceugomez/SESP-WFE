#!/usr/bin/env python
"""paper_figures.py — generates EVERY figure used in the manuscript.

One function per figure, registered with @figure. Each figure reads only result files
written by a script under scripts/ (its docstring names them) and saves PDF + PNG to
figure/<tag>/paper/. Nothing here computes estimates; it only reduces and plots stored results.

    conda activate fasteddy
    python viz/paper_figures.py            # all figures
    python viz/paper_figures.py repr_error_time
    python viz/paper_figures.py --list
"""
import argparse
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sesp_paths import CONFIG_PATH, FIGURE_DIR, REPR_DIR

OUT_DIR = FIGURE_DIR / "paper"
CONFIG = CONFIG_PATH

# ---------------------------------------------------------------------------
# Style (one place). Ink/chrome from the dataviz reference palette; magnitude is
# encoded with a single-hue blue sequential ramp, steps 250 -> 700.
# ---------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

LOO_FOLDS = [f"{i:02d}" for i in range(12)]
TRUTH_FOLD = "truth"

FIGURES = {}


def figure(name):
    """Register a figure function under `name`."""
    def deco(fn):
        FIGURES[name] = fn
        return fn
    return deco


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{name}.pdf|png")


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------
def forcing_deviation():
    """d_Θ per fold id: Mahalanobis distance of the member's (dir, speed) from p(Θ)'s mean."""
    cfg = json.loads(CONFIG.read_text())
    e = cfg["ensemble"]
    members = cfg["prior_members"] + [cfg["truth_member"]]
    d = {}
    for m in members:
        ddir = (m["met_dir_deg"] - e["mu_dir_deg"] + 180) % 360 - 180
        d[m["id"]] = float(np.hypot(ddir / e["sigma_dir_deg"],
                                    (m["spd_ms"] - e["mu_spd_ms"]) / e["sigma_spd_ms"]))
    return d


def load_repr_error(fold):
    """results/repr/repr_error_<fold>.nc (scripts/run_repr_error.jl)."""
    path = REPR_DIR / f"repr_error_{fold}.nc"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing — run scripts/run_repr_error.jl (run_repr_error.sh)")
    return xr.open_dataset(path)


def repr_rmse(ds, r):
    """Component-RMSE time series at rank r: sqrt(Σ_c sse / (3 n_cells)), m/s."""
    sse = ds["sse"].sel(rank=r).sum("component")
    return np.sqrt(sse.values / (3 * int(ds.attrs["n_cells"])))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
@figure("repr_error_time")
def fig_repr_error_time(ranks=(4, 50)):
    """Representation-error RMSE over time, every LOO fold, at ranks 4 and 50.

    Input: results/repr/repr_error_<fold>.nc for folds 00–11 and truth
           (scripts/run_repr_error.jl, region attr must agree across folds).
    Lines: one per LOO fold, colored by forcing deviation d_Θ (single-hue ramp).
    Band:  mean ± 1σ across the 12 LOO folds at each time (11-member bases).
    Dashed: the 'truth' run (13th member, 12-member basis — not a LOO fold, so it is
            shown but excluded from the band).
    """
    dtheta = forcing_deviation()
    data = {f: load_repr_error(f) for f in LOO_FOLDS + [TRUTH_FOLD]}
    regions = {ds.attrs["region"] for ds in data.values()}
    if len(regions) != 1:
        raise ValueError(f"folds scored on different regions: {regions}")
    region = regions.pop()

    norm = Normalize(vmin=0.0, vmax=max(dtheta[f] for f in LOO_FOLDS))
    fig, axes = plt.subplots(len(ranks), 1, figsize=(6.5, 2.4 * len(ranks)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, ranks):
        series = {}
        for f, ds in data.items():
            t = ds["time"].values - ds.attrs["t_spinup"]
            series[f] = (t, repr_rmse(ds, r))
        t0 = series[LOO_FOLDS[0]][0]
        if not all(np.array_equal(series[f][0], t0) for f in LOO_FOLDS):
            raise ValueError("LOO folds have different time axes — cannot form a band")
        Y = np.vstack([series[f][1] for f in LOO_FOLDS])
        mu, sd = Y.mean(axis=0), Y.std(axis=0, ddof=1)

        ax.fill_between(t0, mu - sd, mu + sd, color=MUTED, alpha=0.22, lw=0,
                        label="LOO mean ± 1σ (12 folds)")
        for f in sorted(LOO_FOLDS, key=lambda f: dtheta[f]):
            ax.plot(*series[f], color=SEQ_BLUE(norm(dtheta[f])), lw=0.9)
        ax.plot(t0, mu, color=INK, lw=1.8, label="LOO mean")
        ax.plot(*series[TRUTH_FOLD], color=INK2, lw=1.2, ls=(0, (4, 2)),
                label="truth run (12-member basis)")
        ax.set_ylabel("representation error\nRMSE (m s$^{-1}$)")
        ax.set_title(f"rank r = {r}", loc="left")
        ax.set_ylim(bottom=0)
        print(f"  r={r}: LOO time-mean RMSE {mu.mean():.4f} m/s; across-fold σ (time-mean) "
              f"{sd.mean():.4f}; per-fold time-mean "
              + ", ".join(f"{f}:{series[f][1].mean():.3f}" for f in LOO_FOLDS + [TRUTH_FOLD]))
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("time after spin-up (s)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ_BLUE), ax=axes,
                      shrink=0.8, pad=0.02)
    cb.set_label(r"fold forcing deviation $d_\Theta$ ($\sigma$)")
    cb.outline.set_visible(False)
    fig.suptitle(f"Held-out representation error ({region} region)", x=0.02, ha="left",
                 fontsize=9, color=INK2)
    save(fig, "repr_error_time")


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("names", nargs="*", help="figures to make (default: all)")
    p.add_argument("--list", action="store_true", help="list figures and exit")
    a = p.parse_args(argv)
    if a.list:
        for n, fn in FIGURES.items():
            print(f"{n:24s} {fn.__doc__.strip().splitlines()[0]}")
        return 0
    names = a.names or list(FIGURES)
    bad = [n for n in names if n not in FIGURES]
    if bad:
        p.error(f"unknown figure(s) {bad}; known: {list(FIGURES)}")
    failed = []
    for n in names:
        print(f"[{n}]")
        try:
            FIGURES[n]()
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            failed.append(n)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
