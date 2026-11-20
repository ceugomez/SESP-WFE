# check_member_output.py
# cgf / 2026-11-05
#
# Health check for a running (or finished) FastEddy ensemble member. For each output file
# FE_ensemble_<id>.<step> it checks, without loading whole 3-D fields into memory twice:
#   - readable NetCDF with u, v, w, theta, time
#   - all finite (no NaN / Inf)  -> a blow-up is caught at the first bad frame
#   - time == step * dt, consecutive frames frqOutput*dt apart
#   - |u|,|v| < 40 m/s, |w| < 20 m/s  (physical-bounds sanity for a ~10 m/s neutral flow)
#   - domain-mean upper-level (u, v) within 30% of the member's forcing (U_g, V_g) — catches
#     a member running with the wrong forcing or a broken sponge
#   - file size identical across frames
# and reports the wall-clock write cadence (s per output) and a projected finish time
# against the 24 h job limit.
#
#   conda activate fasteddy
#   python check_member_output.py <member_id> [--ensemble ensemble-lowvar] [--all] [--json]
# By default only files not yet recorded in <output_dir>/../log/health.json are checked
# (incremental); --all rechecks everything.
import argparse
import json
import os
import re
import sys
import time

import netCDF4 as nc
import numpy as np

DT, FRQ, NT = 0.02, 300, 125000          # from the FastEddy namelist template
LIMIT_S = 24 * 3600                      # SLURM per-job wall-clock limit


def check_file(path, step, ug, vg):
    issues = []
    with nc.Dataset(path) as d:
        t = float(d["time"][0])
        if abs(t - step * DT) > 1e-3:
            issues.append(f"time {t} != step*dt {step*DT}")
        stats = {}
        for name in ("u", "v", "w", "theta"):
            a = np.asarray(d[name][0], dtype=np.float32)     # (z, y, x)
            if not np.isfinite(a).all():
                issues.append(f"{name}: {np.size(a) - np.isfinite(a).sum()} non-finite values")
                stats[name] = None
                continue
            stats[name] = (float(a.min()), float(a.max()), float(a.mean()))
            if name in ("u", "v") and np.abs(a).max() > 40:
                issues.append(f"{name}: |max| {np.abs(a).max():.1f} > 40 m/s")
            if name == "w" and np.abs(a).max() > 20:
                issues.append(f"w: |max| {np.abs(a).max():.1f} > 20 m/s")
            if name in ("u", "v"):
                top = a[-20:-10].mean()                       # below the top damping layer
                stats[name + "_upper_mean"] = float(top)
        if stats.get("u") and stats.get("v"):
            mag_f = np.hypot(ug, vg)
            du = np.hypot(stats["u_upper_mean"] - ug, stats["v_upper_mean"] - vg)
            if du > 0.3 * mag_f:
                issues.append(f"upper-level mean wind ({stats['u_upper_mean']:.2f}, "
                              f"{stats['v_upper_mean']:.2f}) far from forcing ({ug:.2f}, {vg:.2f})")
    return t, stats, issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("member")
    p.add_argument("--ensemble", default=os.environ.get("SESP_ENSEMBLE", "ensemble-lowvar"))
    p.add_argument("--all", action="store_true")
    p.add_argument("--json", action="store_true", help="print a one-line JSON summary only")
    p.add_argument("--brief", action="store_true", help="print one human-readable status line")
    a = p.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", a.ensemble)
    cfg = json.load(open(os.path.join(root, "ensemble_config.json")))
    m = next(x for x in [cfg["truth_member"]] + cfg["prior_members"] if x["id"] == a.member)
    outdir, logdir = m["output_dir"], m["log_dir"]
    state_path = os.path.join(logdir, "health.json")
    state = {} if a.all or not os.path.isfile(state_path) else json.load(open(state_path))
    checked = state.get("checked", {})

    pat = re.compile(rf"^FE_ensemble_{re.escape(a.member)}\.(\d+)$")
    files = sorted(((int(mm.group(1)), f) for f in os.listdir(outdir) if (mm := pat.match(f))))
    # skip the newest file if it may still be being written (mtime < 20 s ago)
    now = time.time()
    if files and now - os.path.getmtime(os.path.join(outdir, files[-1][1])) < 20:
        files = files[:-1]

    sizes = {os.path.getsize(os.path.join(outdir, f)) for _, f in files}
    new_issues = []
    for step, f in files:
        if f in checked:
            continue
        path = os.path.join(outdir, f)
        try:
            t, stats, issues = check_file(path, step, m["U_g_ms"], m["V_g_ms"])
        except Exception as e:                                  # unreadable / truncated file
            if now - os.path.getmtime(path) < 300:              # maybe still being written:
                continue                                        # retry on the next pass
            t, stats, issues = None, {}, [f"unreadable: {e!r}"]
        checked[f] = {"step": step, "t": t, "mtime": os.path.getmtime(path), "issues": issues,
                      "u": stats.get("u"), "w": stats.get("w")}
        new_issues += [f"{f}: {i}" for i in issues]

    steps = sorted(v["step"] for v in checked.values())
    gaps = [b - a_ for a_, b in zip(steps, steps[1:]) if b - a_ != FRQ]
    mt = sorted(v["mtime"] for v in checked.values())
    cadence = (mt[-1] - mt[0]) / (len(mt) - 1) if len(mt) > 2 else None
    n_total = NT // FRQ + 1
    eta = None
    if cadence and len(mt) > 2:
        start = mt[0] - 10                                       # first frame ~ job start
        eta = start + cadence * (n_total - 1)
    proj = None if eta is None else eta - mt[0]
    summary = {
        "member": a.member, "n_checked": len(checked), "n_expected": n_total,
        "last_step": steps[-1] if steps else None, "step_gaps": gaps[:5],
        "sizes_consistent": len(sizes) <= 1,
        "cadence_s": None if cadence is None else round(cadence, 1),
        "projected_runtime_h": None if proj is None else round(proj / 3600, 2),
        "fits_limit": None if proj is None else proj < LIMIT_S,
        "n_issue_files": sum(1 for v in checked.values() if v["issues"]),
        "new_issues": new_issues[:10],
        "last_u_range": checked[max(checked, key=lambda k: checked[k]["step"])]["u"] if checked else None,
    }
    os.makedirs(logdir, exist_ok=True)
    json.dump({"checked": checked, "summary": summary}, open(state_path, "w"))
    if a.brief:
        u = summary["last_u_range"]
        print(f"{summary['n_checked']}/{summary['n_expected']} frames clean-checked, "
              f"cadence {summary['cadence_s']} s/frame, projected {summary['projected_runtime_h']} h "
              f"(fits={summary['fits_limit']}), last u min/max/mean "
              f"{'n/a' if not u else '/'.join(f'{x:.2f}' for x in u)}"
              + (f"; ISSUES: {summary['new_issues']}" if summary["new_issues"] else "")
              + (f"; STEP GAPS: {summary['step_gaps']}" if summary["step_gaps"] else "")
              + ("" if summary["sizes_consistent"] else "; FILE SIZES DIFFER"))
    else:
        print(json.dumps(summary) if a.json else json.dumps(summary, indent=2))
    sys.exit(1 if (new_issues or gaps or len(sizes) > 1) else 0)


if __name__ == "__main__":
    main()
