"""
generate_boundary_planes.py
---------------------------
Write FastEddy LAD (hydroBCs=1) boundary-plane NetCDF4 files for a single
ensemble member using an analytic base-state profile.

Profile choice
--------------
Velocity : geostrophic wind (Ug, Vg) at all heights, w=0 everywhere.
           This is consistent with FastEddy's initialisation (no Ekman
           spiral in the initial condition) and eliminates the periodic
           wraparound without introducing a mismatch with the model
           base state.  Internal turbulence is seeded by the potential-
           temperature perturbation already in the namelist
           (thetaPerturbationSwitch=1).

Thermodynamics : dry hydrostatic integration of the base state defined by
                 stabilityScheme=2 (two-layer stable BL with capping
                 inversion):
                   theta(z) = temp_grnd                         z <= z1
                              temp_grnd + gamma1*(z-z1)         z1 < z <= z2
                              ...+ gamma2*(z-z2)                z2 < z <= z3
                   rho  from Exner-pressure hydrostatic balance

File format (as expected by FastEddy)
--------------------------------------
* NetCDF4
* Dimensions: time=1, zIndex=Nz, yIndex=Ny, xIndex=Nx
* Variables per field (rho, u, v, w, theta):
    {var}_YZL  (time, zIndex, yIndex)  – west  face  (x-low)
    {var}_YZH  (time, zIndex, yIndex)  – east  face  (x-high)
    {var}_XZL  (time, zIndex, xIndex)  – south face  (y-low)
    {var}_XZH  (time, zIndex, xIndex)  – north face  (y-high)
    {var}_XYL  (time, yIndex, xIndex)  – floor face  (z-low)
    {var}_XYH  (time, yIndex, xIndex)  – ceiling face (z-high)
* File naming: {base}.0  and  {base}.1  (two identical files for time-
  invariant BCs; FastEddy loads both at startup and interpolates between
  them — with dtBdyPlaneBCs=600000 s >> 7200 s sim time the values
  never change).
* moistureSelector=0 → only 5 hydrodynamic fields are needed.
* surflayerSelector=1 → no surface-skin (tskin) planes needed.

Usage
-----
  python generate_boundary_planes.py --ug 10.0 --vg 0.0 \\
      --outdir /path/to/member/ICBC --base FE_Bndys
"""

import argparse
import os

import netCDF4 as nc
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
Rd    = 287.05      # J kg^-1 K^-1
cp    = 1004.0      # J kg^-1 K^-1
g     = 9.81        # m s^-2
p00   = 1.0e5       # Pa  (reference pressure for Exner)
kappa = Rd / cp     # ≈ 0.2859

# ---------------------------------------------------------------------------
# Domain parameters (must match the ensemble namelist)
# ---------------------------------------------------------------------------
NX = 242   # grid points in x
NY = 242   # grid points in y
NZ = 82    # grid points in z
DZ = 20.0  # m, nominal vertical spacing (non-deformed)

# ---------------------------------------------------------------------------
# Base-state thermodynamics (stabilityScheme=2 from ensemble namelist)
# ---------------------------------------------------------------------------
TEMP_GRND   = 300.0       # K   potential temperature at surface
PRES_GRND   = 100000.0    # Pa  pressure at surface
Z_STABLE_1  = 500.0;  GAMMA_1 = 0.08   # K m^-1  capping inversion
Z_STABLE_2  = 650.0;  GAMMA_2 = 0.003  # K m^-1  free atmosphere
Z_STABLE_3  = 50000.0; GAMMA_3 = 0.003 # K m^-1  (continuation)


def _cell_centers(Nz: int, dz: float) -> np.ndarray:
    """Return cell-centre heights z_k = (k + 0.5)*dz for k = 0..Nz-1."""
    return (np.arange(Nz) + 0.5) * dz


def _theta_profile(z: np.ndarray) -> np.ndarray:
    """Potential temperature profile matching stabilityScheme=2."""
    theta = np.full_like(z, TEMP_GRND)
    m1 = z > Z_STABLE_1
    theta[m1] = TEMP_GRND + GAMMA_1 * (z[m1] - Z_STABLE_1)
    m2 = z > Z_STABLE_2
    theta[m2] = (TEMP_GRND
                 + GAMMA_1 * (Z_STABLE_2 - Z_STABLE_1)
                 + GAMMA_2 * (z[m2] - Z_STABLE_2))
    m3 = z > Z_STABLE_3
    theta[m3] = (TEMP_GRND
                 + GAMMA_1 * (Z_STABLE_2 - Z_STABLE_1)
                 + GAMMA_2 * (Z_STABLE_3 - Z_STABLE_2)
                 + GAMMA_3 * (z[m3] - Z_STABLE_3))
    return theta


def _base_state(z: np.ndarray):
    """
    Compute hydrostatic base-state rho(z) and theta(z).

    Uses the Exner pressure formulation:
        d(pi)/dz = -g / (cp * theta(z))
        pi = (p/p00)^(Rd/cp)
        rho = p / (Rd * T) = p00 * pi^(cp/Rd) / (Rd * theta * pi)
            = p00 * pi^(cp/Rd - 1) / (Rd * theta)
    Integration is performed by the trapezoidal rule from z=0 upward.
    """
    theta = _theta_profile(z)

    # Exner pressure at each cell centre via cumulative trapezoid integration
    # from the surface (z=0) where pi_0 = (pres_grnd/p00)^kappa
    pi = np.empty_like(z)
    pi_surf = (PRES_GRND / p00) ** kappa

    # First level: integrate from 0 to z[0] using half-step approximation
    # (average theta between surface and first cell centre)
    theta_half = 0.5 * (TEMP_GRND + theta[0])
    pi[0] = pi_surf - (g / (cp * theta_half)) * z[0]

    for k in range(1, len(z)):
        dz_k = z[k] - z[k - 1]
        theta_avg = 0.5 * (theta[k - 1] + theta[k])
        pi[k] = pi[k - 1] - (g / (cp * theta_avg)) * dz_k

    rho = p00 * pi ** (cp / Rd - 1.0) / (Rd * theta)
    return rho, theta


def _write_one_file(filepath: str, Ug: float, Vg: float) -> None:
    """Write a single boundary-plane NetCDF4 file for geostrophic inflow."""
    z   = _cell_centers(NZ, DZ)
    rho, theta = _base_state(z)

    # Vertical profiles for each field (shape: NZ)
    profiles = {
        "rho":   rho,
        "u":     np.full(NZ, Ug),
        "v":     np.full(NZ, Vg),
        "w":     np.zeros(NZ),
        "theta": theta,
    }

    with nc.Dataset(filepath, "w", format="NETCDF4") as ds:
        ds.createDimension("time",   1)
        ds.createDimension("zIndex", NZ)
        ds.createDimension("yIndex", NY)
        ds.createDimension("xIndex", NX)

        for varname, prof in profiles.items():
            # YZ planes (west / east): dims (time, zIndex, yIndex)
            for suffix in ("_YZL", "_YZH"):
                v = ds.createVariable(varname + suffix, "f4",
                                      ("time", "zIndex", "yIndex"))
                arr = np.empty((1, NZ, NY), dtype=np.float32)
                arr[0, :, :] = prof[:, np.newaxis]   # broadcast z-profile
                v[:] = arr

            # XZ planes (south / north): dims (time, zIndex, xIndex)
            for suffix in ("_XZL", "_XZH"):
                v = ds.createVariable(varname + suffix, "f4",
                                      ("time", "zIndex", "xIndex"))
                arr = np.empty((1, NZ, NX), dtype=np.float32)
                arr[0, :, :] = prof[:, np.newaxis]
                v[:] = arr

            # XY planes (floor / ceiling): dims (time, yIndex, xIndex)
            for suffix, k_idx in (("_XYL", 0), ("_XYH", NZ - 1)):
                v = ds.createVariable(varname + suffix, "f4",
                                      ("time", "yIndex", "xIndex"))
                v[:] = np.full((1, NY, NX), float(prof[k_idx]),
                               dtype=np.float32)


def generate_boundary_planes(outdir: str, base: str,
                              Ug: float, Vg: float) -> str:
    """
    Generate the two boundary-plane files required for time-invariant LAD BCs.

    Parameters
    ----------
    outdir : directory to write files into (created if absent)
    base   : file basename (e.g. "FE_Bndys")
    Ug     : zonal geostrophic wind component (m/s)
    Vg     : meridional geostrophic wind component (m/s)

    Returns
    -------
    hydroBndysFileBase string to use in the namelist
    """
    os.makedirs(outdir, exist_ok=True)
    for idx in range(2):           # .0 and .1 — identical time-invariant pair
        filepath = os.path.join(outdir, f"{base}.{idx}")
        _write_one_file(filepath, Ug, Vg)
    return os.path.join(outdir, base)


def generate_ic(template_path: str, outdir: str, ic_name: str,
                Ug: float, Vg: float) -> str:
    """
    Write a FastEddy initial-condition file for a single ensemble member.

    Copies grid coordinates, terrain, density profile, theta (including any
    random perturbations), TKE, and surface fields from an existing FastEddy
    output file (the *template*), then replaces u and v with the member's
    geostrophic wind components.  rho is identical across members because the
    thermodynamic base state does not depend on wind direction.

    Parameters
    ----------
    template_path : path to an existing FE output .nc file to use as the
                    source of grid coordinates and shared state fields.
    outdir        : directory to write the IC file into (created if absent)
    ic_name       : filename for the IC file (e.g. "FE_Member00.0")
    Ug, Vg        : zonal / meridional geostrophic wind (m/s) for this member

    Returns
    -------
    Full path to the written IC file.
    """
    import shutil

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, ic_name)
    if not os.path.samefile(template_path, out_path) if os.path.exists(out_path) else True:
        shutil.copy2(template_path, out_path)

    # Overwrite only u and v; every other field is shared across members.
    with nc.Dataset(out_path, "r+") as ds:
        ds["u"][:] = np.float32(Ug)
        ds["v"][:] = np.float32(Vg)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ug",     type=float, required=True,
                        help="Zonal geostrophic wind (m/s)")
    parser.add_argument("--vg",     type=float, required=True,
                        help="Meridional geostrophic wind (m/s)")
    parser.add_argument("--outdir", type=str,   required=True,
                        help="Output directory for boundary-plane files")
    parser.add_argument("--base",   type=str,   default="FE_Bndys",
                        help="Base filename (default: FE_Bndys)")
    args = parser.parse_args()

    bndys_base = generate_boundary_planes(args.outdir, args.base,
                                          args.ug, args.vg)
    print(f"Wrote {args.base}.0 and {args.base}.1 → {args.outdir}")
    print(f"  hydroBndysFileBase = {bndys_base}")


if __name__ == "__main__":
    main()
