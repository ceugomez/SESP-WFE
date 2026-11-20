# plot_kf_twin_vs_loo.py
# cgf  2026-07-17
#
# Rung 1 vs rung 2 of the experiment ladder: identical-twin (in-basis truth) dense-array
# KF against the matched leave-one-out fold, aggregated over the 12 prior members. Both
# rungs share the sensor lattice, rank list, Q, and RNG seed, so they are differenced at
# fixed rank r. The twin is a deliberate inverse crime (truth is in the basis, so its
# projection residual is ~zero); its value is NOT a standalone skill number but the GAP to
# the LOO fold — the measured closure price, i.e. the cost of a held-out realization not
# being spanned by the ensemble. The headline claim this figure supports: when the ensemble
# is close to the truth (the twin limit), time-varying wake structure IS tracked down to
# the subspace-oracle floor; the LOO gap shows how that degrades once the truth is unseen.
#
# Panels:
#   (a) field RMSE (--region) vs r: twin KF, twin oracle, LOO KF, LOO oracle (mean over
#       folds, band = min-max across folds). The twin KF hugging the twin oracle is the result.
#   (b) closure price: LOO KF RMSE - twin KF RMSE vs r (mean over folds, band = min-max).
#   (c) innovation consistency (mean NIS) vs r for both rungs; 1.0 is the calibrated target.
#
# Aggregates over exactly the folds present in BOTH rungs (00..11). The "truth" member is in
# no basis and has no twin, so it is excluded here; it lives in the LOO-only rank-sweep figure.
#
# Usage:
#   python plot_kf_twin_vs_loo.py [--kf-dir results/<tag>/kf] [--out figure/<tag>/kf_twin_vs_loo.png]
#     [--basis truth]                # twin basis id -> kf_twin_dense_basis<basis>_truth_<id>.csv
#     [--region full]                # scoring region for panels (a),(b). Wake-region scoring
#                                    # was removed 2026-11-05 pending a redesign.

import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sesp_paths import KF_DIR, FIGURE_DIR

FOLDS = [f"{i:02d}" for i in range(12)]   # 00..11; "truth" has no twin


def load_fold(path):
    if not pathlib.Path(path).is_file():
        return None
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(d)


def stack(paths, folds, field, ranks_ref):
    """Align `field` over folds onto ranks_ref, NaN-filling ranks a fold has not reached yet
    (twin runs are written incrementally, one rank row at a time). Returns (matrix, fold_ids)
    with one row per present fold, rows in fold order so twin and LOO stacks pair positionally."""
    rows, ids = [], []
    for p, fid in zip(paths, folds):
        d = load_fold(p)
        if d is None:
            continue
        rr = np.asarray(d["r"], dtype=int)
        vals = np.asarray(d[field], dtype=float)
        row = np.full(len(ranks_ref), np.nan)
        idx = {int(rv): k for k, rv in enumerate(ranks_ref)}
        for rv, v in zip(rr, vals):
            if int(rv) in idx:
                row[idx[int(rv)]] = v
        rows.append(row); ids.append(fid)
    if not rows:
        return None, []
    return np.vstack(rows), ids


def band(ax, r, M, color, label, ls="-", marker="o", lw=1.5, fill=True):
    if M is None:
        return 0
    mu = np.nanmean(M, axis=0)
    ax.plot(r, mu, marker + ls, color=color, lw=lw, label=f"{label} (n={M.shape[0]})")
    if fill and M.shape[0] > 1:
        ax.fill_between(r, np.nanmin(M, axis=0), np.nanmax(M, axis=0),
                        color=color, alpha=0.15, lw=0)
    return M.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kf-dir", default=str(KF_DIR))
    p.add_argument("--basis", default="truth", help="twin basis id (default: truth)")
    p.add_argument("--region", default="full", choices=("full",))
    p.add_argument("--out", default=str(FIGURE_DIR / "kf_twin_vs_loo.png"))
    args = p.parse_args()

    kf = pathlib.Path(args.kf_dir)
    twin_paths = [kf / f"kf_twin_dense_basis{args.basis}_truth_{f}.csv" for f in FOLDS]
    loo_paths = [kf / f"kf_rank_sweep_dense_truth_{f}.csv" for f in FOLDS]

    # Reference rank grid: the LONGEST rank list among all present files, so a fold still
    # being written (fewer ranks) is NaN-filled onto the full grid rather than truncating it.
    loaded = [load_fold(x) for x in twin_paths + loo_paths]
    loaded = [d for d in loaded if d is not None]
    if not loaded:
        raise SystemExit(f"No twin or LOO dense CSVs found under {kf}")
    r = max((np.asarray(d["r"], dtype=int) for d in loaded), key=len)
    reg = args.region

    twin_kf, twin_ids = stack(twin_paths, FOLDS, f"kf_rmse_{reg}", r)
    twin_or, _ = stack(twin_paths, FOLDS, f"oracle_rmse_{reg}", r)
    loo_kf, loo_ids = stack(loo_paths, FOLDS, f"kf_rmse_{reg}", r)
    loo_or, _ = stack(loo_paths, FOLDS, f"oracle_rmse_{reg}", r)
    twin_nis, _ = stack(twin_paths, FOLDS, "mean_nis", r)
    loo_nis, _ = stack(loo_paths, FOLDS, "mean_nis", r)

    n_twin = 0 if twin_kf is None else twin_kf.shape[0]
    if n_twin == 0:
        print("NOTE: no twin CSVs present yet — rendering LOO side only. Re-run after the "
              "twin array (run_twin_sweep_dense.sh) completes to draw the closure price.")
    elif n_twin < len(FOLDS):
        print(f"NOTE: {n_twin}/{len(FOLDS)} twin folds present ({', '.join(twin_ids)}); "
              "closure price uses the folds available in both rungs. Re-run as more finish.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    ax = axes[0]
    band(ax, r, twin_or, "0.55", "twin oracle (floor)", ls=":", marker="s", lw=1.1, fill=False)
    band(ax, r, loo_or, "0.35", "LOO oracle (floor)", ls="--", marker="s", lw=1.1, fill=False)
    band(ax, r, twin_kf, "C0", "twin KF (in-basis)")
    band(ax, r, loo_kf, "C3", "LOO KF (held-out)")
    ax.set_xscale("log")
    ax.set_xticks(r); ax.set_xticklabels([str(x) for x in r], fontsize=8)
    ax.set_xlabel("truncation rank r")
    ax.set_ylabel(f"{reg}-region field RMSE (m/s)")
    ax.set_title(f"(a) {reg} RMSE vs rank: twin vs LOO")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="best")

    ax = axes[1]
    if twin_kf is not None and loo_kf is not None:
        # Match folds present in BOTH rungs and align row-wise by fold id before differencing.
        both = [f for f in twin_ids if f in loo_ids]
        ti = {f: k for k, f in enumerate(twin_ids)}
        li = {f: k for k, f in enumerate(loo_ids)}
        price = np.vstack([loo_kf[li[f]] - twin_kf[ti[f]] for f in both]) if both else None
        band(ax, r, price, "C2", "closure price = LOO KF − twin KF")
        ax.axhline(0.0, color="0.3", ls=":", lw=1.0)
    else:
        ax.text(0.5, 0.5, "closure price\nneeds twin data", transform=ax.transAxes,
                ha="center", va="center", color="0.5", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(r); ax.set_xticklabels([str(x) for x in r], fontsize=8)
    ax.set_xlabel("truncation rank r")
    ax.set_ylabel(f"Δ {reg} RMSE (m/s)")
    ax.set_title("(b) closure price vs rank")
    ax.grid(alpha=0.3, which="both")
    if twin_kf is not None and loo_kf is not None:
        ax.legend(fontsize=7.5, loc="best")

    ax = axes[2]
    band(ax, r, twin_nis, "C0", "twin", fill=False)
    band(ax, r, loo_nis, "C3", "LOO", fill=False)
    ax.axhline(1.0, color="0.3", ls=":", lw=1.0, label="calibrated (1.0)")
    ax.set_xscale("log")
    ax.set_xticks(r); ax.set_xticklabels([str(x) for x in r], fontsize=8)
    ax.set_xlabel("truncation rank r")
    ax.set_ylabel("mean NIS")
    ax.set_title("(c) innovation consistency vs rank")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="best")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}  (twin folds: {n_twin}, LOO folds: "
          f"{0 if loo_kf is None else loo_kf.shape[0]})")


if __name__ == "__main__":
    main()
