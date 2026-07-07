# plot_eigenvalue_spectrum.py
# cgf  2026-06-25
#
# Eigenvalue-spectrum figure for the UNCENTERED ensemble POD (no mean removed;
# see paper_design_notes.md §4). Two panels:
#   (left)  log eigenvalue vs mode index — shows the two-regime decay predicted
#           in §3(b): a steep head (pinned/parametric "forcing" modes) and a
#           slower tail (advecting meander band + cascade). The leading
#           --n-forcing modes (the uniform-geostrophic subspace, default 2) are
#           drawn in a separate colour and labelled, since uncentered POD puts
#           the mean/forcing flow there.
#   (right) cumulative energy fraction normalized by FLUCTUATION energy
#           (Σλ over modes beyond the forcing subspace), NOT total energy —
#           otherwise the mean-flow modes make the curve deceptively flat
#           (§4 bookkeeping). Rank-at-threshold markers count modes beyond the
#           forcing head.
#
# Usage:
#   python plot_eigenvalue_spectrum.py <eigenvalue_spectrum.csv> \
#       [--n-forcing 2] [--nmodes N] [--out PNG]

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLDS = [0.90, 0.95, 0.99]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="eigenvalue_spectrum.csv (mode,eigenvalue,energy_frac,cumulative_frac)")
    p.add_argument("--n-forcing", type=int, default=2,
                   help="leading modes treated as the mean/forcing subspace, excluded "
                        "from the fluctuation-energy normalization (default 2)")
    p.add_argument("--nmodes", type=int, default=None, help="truncate x-axis to first N modes")
    p.add_argument("--out", default=None, help="output PNG (default: alongside csv)")
    args = p.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    mode = data["mode"].astype(int)
    eigval = np.asarray(data["eigenvalue"], dtype=float)

    nf = args.n_forcing
    # Fluctuation energy = total snapshot energy minus the forcing-subspace head.
    fluct_total = eigval[nf:].sum()
    fluct_total = fluct_total if fluct_total > 0 else 1.0
    cum_fluct = np.concatenate([np.zeros(nf), np.cumsum(eigval[nf:]) / fluct_total])

    show = slice(0, args.nmodes) if args.nmodes else slice(None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # --- left: log eigenvalue spectrum, forcing head highlighted ---
    m, ev = mode[show], eigval[show]
    ax1.semilogy(m, ev, "-", lw=0.8, color="0.6", zorder=1)
    head = m <= nf
    ax1.semilogy(m[~head], ev[~head], "o", ms=3.5, color="C0",
                 label="fluctuation modes", zorder=3)
    ax1.semilogy(m[head], ev[head], "s", ms=6, color="C3",
                 label=f"forcing head (modes 1–{nf})", zorder=4)
    ax1.set_xlabel("mode index")
    ax1.set_ylabel("eigenvalue  (snapshot energy)")
    ax1.set_title("POD eigenvalue spectrum")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    # --- right: cumulative fluctuation energy ---
    mf, cf = mode[show], cum_fluct[show]
    ax2.plot(mf[mf > nf], cf[mf > nf], "o-", ms=3, lw=1, color="C0")
    ax2.set_xlabel("mode index")
    ax2.set_ylabel("cumulative fraction of fluctuation energy")
    ax2.set_title(f"fluctuation energy (modes > {nf})")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)

    for thresh in THRESHOLDS:
        idx = np.searchsorted(cum_fluct, thresh)
        if idx < len(mode):
            r = int(mode[idx])
            ax2.axhline(thresh, color="gray", lw=0.6, ls="--")
            ax2.axvline(r, color="gray", lw=0.6, ls="--")
            ax2.annotate(f"{int(thresh*100)}% @ r={r}", (r, thresh),
                         textcoords="offset points", xytext=(4, -11),
                         fontsize=8, color="dimgray")

    fig.suptitle("Ensemble POD spectrum (uncentered; fluctuation-normalized cumulative)")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                    "eigenvalue_spectrum.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
