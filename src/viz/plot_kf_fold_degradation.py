# plot_kf_fold_degradation.py
# cgf  2026-07-15
#
# Degradation of the reduced-order KF with deviation of the (held-out) truth
# forcing from the ensemble mean, using the LOO folds as the sample. Each fold
# f has a known forcing draw (dir_f, spd_f) from ensemble_config.json; the
# x-axis is its Mahalanobis distance from the prior mean in sigma units,
#   d_f = sqrt( (wrap(dir_f - mu_dir)/sigma_dir)^2 + ((spd_f - mu_spd)/sigma_spd)^2 ).
# Panel (a): wake RMSE at a fixed rank r* (fold-mean-optimal unless --rank)
# for the KF, its subspace oracle (projection of truth = representation
# floor), and the paired Theta-baseline. Oracle-vs-KF separates representation
# degradation (basis loses the far member) from filtering degradation.
# Panel (b): KF/Theta-baseline RMSE ratio (skill relative to the 2-parameter
# forcing regression); ratio > 1 means the subspace filter has lost its edge.
#
# Reads kf_rank_sweep_truth_<id>.csv + theta_baseline_truth_<id>.csv from
# results/kf and the forcing draws from ensemble/ensemble_config.json.
# With --dense, reads the dense-array pair instead
# (kf_rank_sweep_dense_truth_<id>.csv + theta_baseline_dense_truth_<id>.csv),
# which are matched on the same 175-sensor lattice.
#
# Usage:
#   python plot_kf_fold_degradation.py [--dense] [--rank R] [--region wake|full]
#                                      [--stride N] [--out PNG]

import argparse
import json
import pathlib
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wrap_deg(a):
    """wrap angle difference to [-180, 180)"""
    return (a + 180.0) % 360.0 - 180.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=None,
                   help="results/kf directory (default: repo results/kf)")
    p.add_argument("--config", default=None,
                   help="ensemble_config.json (default: ../ensemble/)")
    p.add_argument("--rank", type=int, default=None,
                   help="truncation rank (default: fold-mean-optimal in region)")
    p.add_argument("--region", default="wake", choices=("wake", "full"))
    p.add_argument("--stride", type=int, default=None,
                   help="use the _strideN sweep files instead of stride-1")
    p.add_argument("--dense", action="store_true",
                   help="use the dense-array files (kf_rank_sweep_dense_ / "
                        "theta_baseline_dense_) instead of the sparse rake")
    p.add_argument("--normalize", action="store_true",
                   help="divide panel-(a) RMSEs by the fold's forcing speed "
                        "(removes the amplitude-scales-with-U confound)")
    p.add_argument("--out", default=None, help="output PNG")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[2]
    kfdir = pathlib.Path(args.dir) if args.dir else repo / "results" / "kf"
    cfg_path = pathlib.Path(args.config) if args.config \
        else repo.parent / "ensemble" / "ensemble_config.json"
    tag = f"_stride{args.stride}" if args.stride else ""
    kf_prefix = "kf_rank_sweep_dense_truth_" if args.dense else "kf_rank_sweep_truth_"
    theta_prefix = "theta_baseline_dense_truth_" if args.dense else "theta_baseline_truth_"
    pat = re.compile(rf"{re.escape(kf_prefix)}(\d\d|truth){re.escape(tag)}\.csv$")

    cfg = json.loads(cfg_path.read_text())
    ens = cfg["ensemble"]
    mu_dir, sg_dir = ens["mu_dir_deg"], ens["sigma_dir_deg"]
    mu_spd, sg_spd = ens["mu_spd_ms"], ens["sigma_spd_ms"]
    forcing = {m["id"]: (m["met_dir_deg"], m["spd_ms"])
               for m in cfg["prior_members"]}
    forcing["truth"] = (cfg["truth_member"]["met_dir_deg"],
                        cfg["truth_member"]["spd_ms"])

    folds = {}
    for f in sorted(kfdir.glob(f"{kf_prefix}*{tag}.csv")):
        m = pat.match(f.name)
        if not m or m.group(1) not in forcing:
            continue
        fid = m.group(1)
        d = np.genfromtxt(f, delimiter=",", names=True)
        theta_f = kfdir / f"{theta_prefix}{fid}.csv"
        theta = np.genfromtxt(theta_f, delimiter=",", names=True) \
            if theta_f.is_file() else None
        if theta is None:
            print(f"WARNING: no {theta_f.name} — fold {fid} Θ columns will be NaN")
        folds[fid] = (d, theta)
    if not folds:
        raise SystemExit(f"no {kf_prefix}*{tag}.csv found in {kfdir}")
    print(f"folds found: {', '.join(folds)}")

    reg = args.region
    r = np.asarray(next(iter(folds.values()))[0]["r"], dtype=int)
    if args.rank is None:
        kf_mean = np.mean([d[f"kf_rmse_{reg}"] for d, _ in folds.values()], axis=0)
        rstar = int(r[np.argmin(kf_mean)])
    else:
        rstar = args.rank
        if rstar not in r:
            raise SystemExit(f"rank {rstar} not in sweep ranks {list(r)}")
    ir = int(np.where(r == rstar)[0][0])
    print(f"rank r* = {rstar}")

    rows = []  # (fid, dist, kf, oracle, theta, constwind)
    for fid, (d, theta) in folds.items():
        dd, ss = forcing[fid]
        dist = np.hypot(wrap_deg(dd - mu_dir) / sg_dir, (ss - mu_spd) / sg_spd)
        tb = np.mean(theta[f"theta_rmse_{reg}"]) if theta is not None else np.nan
        cw = np.mean(theta[f"const_rmse_{reg}"]) if theta is not None and \
            f"const_rmse_{reg}" in (theta.dtype.names or ()) else np.nan
        norm = ss if args.normalize else 1.0
        rows.append((fid, dist, d[f"kf_rmse_{reg}"][ir] / norm,
                     d[f"oracle_rmse_{reg}"][ir] / norm, tb / norm, cw / norm))
    rows.sort(key=lambda t: t[1])
    fid_a = [t[0] for t in rows]
    dist = np.array([t[1] for t in rows])
    kf = np.array([t[2] for t in rows])
    orc = np.array([t[3] for t in rows])
    tb = np.array([t[4] for t in rows])
    cw = np.array([t[5] for t in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax = axes[0]
    if np.isfinite(cw).any():
        ax.plot(dist, cw, "^", color="C4", ms=6, label="constant wind (obs mean)")
    ax.plot(dist, tb, "s", color="C1", ms=6, label="Θ-baseline")
    ax.plot(dist, kf, "o", color="C0", ms=6, label=f"KF (r={rstar})")
    ax.plot(dist, orc, "o", color="C0", ms=6, mfc="none",
            label="subspace oracle (representation floor)")
    ax.vlines(dist, orc, kf, color="C0", lw=0.8, alpha=0.5)
    for x, y, fid in zip(dist, kf, fid_a):
        ax.annotate(fid, (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="0.35")
    ax.set_xlabel(r"forcing deviation from ensemble mean, $d_\Theta$ ($\sigma$ units)")
    ax.set_ylabel(f"{reg} RMSE / $U_\\infty$ (–)" if args.normalize
                  else f"{reg} RMSE (m/s)")
    ax.set_title(f"(a) {reg}-region RMSE vs forcing deviation, LOO folds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ok = np.isfinite(tb)
    ratio = kf[ok] / tb[ok]
    ax.plot(dist[ok], ratio, "o", color="C0", ms=6)
    for x, y, fid in zip(dist[ok], ratio, np.array(fid_a)[ok]):
        ax.annotate(fid, (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="0.35")
    ax.axhline(1.0, color="0.3", ls=":", lw=1.0)
    if ok.sum() >= 3:
        c = np.polyfit(dist[ok], ratio, 1)
        xs = np.linspace(dist.min(), dist.max(), 50)
        rho = np.corrcoef(dist[ok], ratio)[0, 1]
        ax.plot(xs, np.polyval(c, xs), "-", color="0.5", lw=1.0,
                label=f"linear fit (Pearson ρ={rho:.2f})")
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel(r"forcing deviation from ensemble mean, $d_\Theta$ ($\sigma$ units)")
    ax.set_ylabel("KF / Θ-baseline RMSE")
    ax.set_title("(b) skill relative to Θ-baseline")
    ax.grid(alpha=0.3)

    out = args.out
    if out is None:
        fig_dir = repo / "figure"
        fig_dir.mkdir(exist_ok=True)
        ntag = "_norm" if args.normalize else ""
        dtag = "_dense" if args.dense else ""
        out = str(fig_dir / f"kf_fold_degradation_{reg}_r{rstar}{ntag}{tag}{dtag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
