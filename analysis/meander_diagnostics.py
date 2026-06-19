# meander_diagnostics.py
# cgf / assistant  2026-06-12
#
# Meander-energy gate diagnostic for the sponge truth run.
# Extracts u at a fixed height AGL for every output frame, computes the
# wake-center line y_c(x,t) as the deficit centroid per x column, and reports:
#   - time-mean centerline and lateral RMS meander amplitude vs fetch
#   - temporal PSD of the centerline anomaly at downstream stations
#   - centerline autocorrelation / decorrelation time at those stations
#   - point-probe u spectra at a wake-center vs an ambient location
#
# Slices are cached to analysis/slices_u<zagl>.npz so reruns are instant.
# Usage:
#   python meander_diagnostics.py [--zagl 50] [--tspin 700]

import argparse
import glob
import os
import re
import sys

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "output")
ANA_DIR  = os.path.join(HERE, "analysis")

DT          = 0.02          # model timestep (s); frame index * DT = sim time
SPONGE_M    = 400.0         # lateral sponge width (m), 40 cells @ 10 m
BUF_M       = 100.0         # extra buffer inside the sponge
X_WAKE_MIN  = 1800.0        # first analysis column (m), past the lee slope
DEF_THRESH  = 1.0           # min column deficit (m/s) to trust a centroid
W_FLOOR     = 0.30          # centroid weight floor: (D - W_FLOOR*Dmax)_+
STATIONS_KM = (2.0, 2.5, 3.0, 3.5, 4.0)


def frame_step(path):
    m = re.search(r"\.(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_slices(zagl, kmax=14):
    """u at zagl (m AGL) for every frame -> (t[s], u(t,y,x), x, y). Cached."""
    cache = os.path.join(ANA_DIR, f"slices_u{zagl:.0f}.npz")
    files = sorted((f for f in glob.glob(os.path.join(OUT_DIR, "FE_*"))
                    if os.path.isfile(f) and frame_step(f) is not None),
                   key=frame_step)
    if not files:
        sys.exit(f"no FE_* frames in {OUT_DIR}")

    if os.path.exists(cache):
        z = np.load(cache)
        if z["t"].size == len(files):
            print(f"using cache {cache} ({z['t'].size} frames)")
            return z["t"], z["u"], z["x"], z["y"]
        print("cache stale (frame count changed) -- rebuilding")

    with nc.Dataset(files[0]) as ds:
        x    = np.array(ds["xPos"][0, 0, 0, :])
        y    = np.array(ds["yPos"][0, 0, :, 0])
        topo = np.array(ds["topoPos"][0, :, :])
        zagl3 = np.array(ds["zPos"][0, :kmax, :, :]) - topo[None, :, :]
    if zagl3[-1].min() < zagl:
        sys.exit(f"kmax={kmax} too shallow for zagl={zagl}")

    # static interpolation weights: zPos is time-invariant (terrain-following)
    hi = (zagl3 > zagl).argmax(axis=0)                  # first level above
    hi = np.clip(hi, 1, kmax - 1)
    lo = hi - 1
    jj, ii = np.meshgrid(np.arange(y.size), np.arange(x.size), indexing="ij")
    z_lo, z_hi = zagl3[lo, jj, ii], zagl3[hi, jj, ii]
    w_hi = np.clip((zagl - z_lo) / np.maximum(z_hi - z_lo, 1e-6), 0.0, 1.0)

    t = np.empty(len(files))
    u = np.empty((len(files), y.size, x.size), dtype=np.float32)
    for n, f in enumerate(files):
        with nc.Dataset(f) as ds:
            uk = np.array(ds["u"][0, :kmax, :, :], dtype=np.float32)
        u[n] = uk[lo, jj, ii] * (1 - w_hi) + uk[hi, jj, ii] * w_hi
        t[n] = frame_step(f) * DT
        if n % 50 == 0:
            print(f"  {n}/{len(files)} frames", flush=True)

    os.makedirs(ANA_DIR, exist_ok=True)
    np.savez_compressed(cache, t=t, u=u, x=x, y=y)
    print(f"cached -> {cache}")
    return t, u, x, y


def wake_centerline(u, x, y):
    """Deficit centroid y_c(x,t) and column deficit amplitude A(x,t)."""
    interior = slice(None)
    yin = (y > SPONGE_M + BUF_M) & (y < y[-1] - SPONGE_M - BUF_M)
    # ambient per column: upper quartile over interior y (wake is a deficit)
    amb = np.percentile(u[:, yin, :], 75, axis=1)        # (Nt, Nx)
    D = np.maximum(amb[:, None, :] - u, 0.0)             # deficit, >=0
    D[:, ~yin, :] = 0.0

    A = D.max(axis=1)                                    # (Nt, Nx)
    W = np.maximum(D - W_FLOOR * A[:, None, :], 0.0)
    sw = W.sum(axis=1)
    with np.errstate(invalid="ignore"):
        yc = (W * y[None, :, None]).sum(axis=1) / sw
    yc[A < DEF_THRESH] = np.nan
    return yc, A


def welch_psd(sig, fs, nseg=5):
    """Simple Welch PSD (Hann, 50% overlap) without scipy dependency."""
    sig = sig - sig.mean()
    nper = int(2 * len(sig) / (nseg + 1))
    nper -= nper % 2
    step = nper // 2
    win = np.hanning(nper)
    norm = fs * (win ** 2).sum()
    specs = []
    for i0 in range(0, len(sig) - nper + 1, step):
        seg = (sig[i0:i0 + nper] - sig[i0:i0 + nper].mean()) * win
        specs.append(np.abs(np.fft.rfft(seg)) ** 2 / norm)
    f = np.fft.rfftfreq(nper, d=1 / fs)
    return f[1:], np.mean(specs, axis=0)[1:]


def autocorr_time(sig, dt_s):
    """Integral-free e-folding decorrelation time of a 1-D series."""
    a = sig - sig.mean()
    ac = np.correlate(a, a, "full")[a.size - 1:]
    ac /= ac[0]
    below = np.where(ac < 1 / np.e)[0]
    return (below[0] * dt_s if below.size else np.nan), ac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zagl",  type=float, default=50.0)
    p.add_argument("--tspin", type=float, default=700.0)
    args = p.parse_args()

    os.makedirs(ANA_DIR, exist_ok=True)
    t, u, x, y = load_slices(args.zagl)

    keep = t > args.tspin
    t, u = t[keep], u[keep]
    dt_s = float(np.median(np.diff(t)))
    fs = 1.0 / dt_s
    print(f"{t.size} frames after spin-up cut at {args.tspin:.0f} s "
          f"(t = {t[0]:.0f}..{t[-1]:.0f} s, cadence {dt_s:.0f} s)")

    yc, A = wake_centerline(u, x, y)

    xmax = x[-1] - SPONGE_M - BUF_M
    cols = (x >= X_WAKE_MIN) & (x <= xmax)
    xc = x[cols]
    yc_c, A_c = yc[:, cols], A[:, cols]

    ybar  = np.nanmean(yc_c, axis=0)
    yanom = yc_c - ybar[None, :]
    sigma = np.nanstd(yc_c, axis=0)
    valid_frac = np.isfinite(yc_c).mean(axis=0)

    # ---- figures ---------------------------------------------------------
    km = 1e-3
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    pc = ax.pcolormesh(xc * km, t, yanom, cmap="RdBu_r",
                       vmin=-np.nanmax(np.abs(yanom)),
                       vmax=np.nanmax(np.abs(yanom)))
    fig.colorbar(pc, ax=ax, label="y$_c$ anomaly (m)")
    ax.set_xlabel("x (km)"); ax.set_ylabel("t (s)")
    ax.set_title(f"wake-centerline anomaly @ {args.zagl:.0f} m AGL")
    fig.savefig(os.path.join(ANA_DIR, "centerline_anomaly_hovmoller.png"), dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax1.plot(xc * km, sigma, "k-", label=r"$\sigma_{y_c}$")
    ax1.set_xlabel("x (km)"); ax1.set_ylabel(r"$\sigma_{y_c}$ (m)")
    ax2 = ax1.twinx()
    ax2.plot(xc * km, np.nanmean(A_c, axis=0), "r--", label="mean deficit")
    ax2.set_ylabel("mean max deficit (m s$^{-1}$)", color="r")
    ax1.set_title("meander amplitude and wake strength vs fetch")
    fig.savefig(os.path.join(ANA_DIR, "meander_amplitude_vs_fetch.png"), dpi=150)
    plt.close(fig)

    fig, (axp, axa) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    summary = []
    for xs_km in STATIONS_KM:
        i = np.argmin(np.abs(xc - xs_km * 1e3))
        sig = yanom[:, i]
        if np.isnan(sig).any():           # fill rare gaps by interpolation
            ok = np.isfinite(sig)
            if ok.mean() < 0.8:
                continue
            sig = np.interp(t, t[ok], sig[ok])
        f, P = welch_psd(sig, fs)
        tau, ac = autocorr_time(sig, dt_s)
        fpk = f[np.argmax(P)]
        axp.loglog(f, P, label=f"x={xs_km:.1f} km")
        axa.plot(np.arange(ac.size) * dt_s, ac, label=f"x={xs_km:.1f} km")
        summary.append((xs_km, sig.std(), fpk, 1 / fpk, tau))
    axp.set_xlabel("f (Hz)"); axp.set_ylabel("PSD of y$_c$ (m$^2$ Hz$^{-1}$)")
    axp.legend(); axp.set_title("centerline meander PSD")
    axa.set_xlim(0, 600); axa.axhline(1 / np.e, color="0.6", ls=":")
    axa.set_xlabel("lag (s)"); axa.set_ylabel("autocorr"); axa.legend()
    axa.set_title("centerline autocorrelation")
    fig.savefig(os.path.join(ANA_DIR, "meander_psd_autocorr.png"), dpi=150)
    plt.close(fig)

    # point probes: wake center vs ambient at the same fetch
    i3 = np.argmin(np.abs(x - 3000.0))
    jw = np.argmin(np.abs(y - np.nanmean(yc[:, np.argmin(np.abs(xc - 3000.0))])))
    ja = np.argmin(np.abs(y - 1200.0))
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for j, lab in ((jw, "wake center"), (ja, "ambient")):
        f, P = welch_psd(u[:, j, i3].astype(float), fs)
        ax.loglog(f, P, label=f"{lab} (y={y[j]*km:.1f} km)")
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("PSD of u (m$^2$ s$^{-2}$ Hz$^{-1}$)")
    ax.legend(); ax.set_title(f"point u spectra at x=3 km, {args.zagl:.0f} m AGL")
    fig.savefig(os.path.join(ANA_DIR, "probe_u_spectra.png"), dpi=150)
    plt.close(fig)

    # ---- text summary ----------------------------------------------------
    lines = [
        f"meander gate diagnostic  (zagl={args.zagl:.0f} m, t>{args.tspin:.0f} s, "
        f"{t.size} frames @ {dt_s:.0f} s)",
        f"analysis columns: x = {xc[0]*km:.2f}..{xc[-1]*km:.2f} km "
        f"(sponge+buffer excluded); centroid valid fraction "
        f"min/mean = {valid_frac.min():.2f}/{valid_frac.mean():.2f}",
        "",
        " station   sigma_yc     peak f      peak T     tau_e",
        "   (km)        (m)        (Hz)        (s)       (s)",
    ]
    for xs_km, s, fpk, Tpk, tau in summary:
        lines.append(f"   {xs_km:4.1f}    {s:7.1f}    {fpk:9.5f}   {Tpk:7.0f}   {tau:7.0f}")
    lines += [
        "",
        f"grid resolution dy = {y[1]-y[0]:.0f} m  "
        "(sigma_yc must sit well above this to be a trackable signal)",
        f"record length = {t[-1]-t[0]:.0f} s "
        "(periods longer than ~1/3 of this are not resolved)",
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(ANA_DIR, "meander_summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(f"\nfigures + summary -> {ANA_DIR}")


if __name__ == "__main__":
    main()
