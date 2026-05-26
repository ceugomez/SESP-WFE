## cgf cego6160 5.18.26
## Gets RRFS ensemble member data for prevailing conditions at location specified

from herbie import Herbie
from herbie.toolbox import EasyMap, pc

import matplotlib.pyplot as plt
import numpy as np

# NCAR site bounding box from location.txt
LON_MIN, LON_MAX = -105.205, -105.175
LAT_MIN, LAT_MAX = 39.934, 39.952
PAD = 0.5  # degrees padding around site for context

DATE = "2026-05-18 12:00"
FXX = 6
LEVEL = 850  # hPa
MEMBERS = ["control", 1, 2, 3]


def get_wind_ds(member):
    H = Herbie(DATE, model="rrfs", fxx=FXX, member=member, product="prslev", domain="conus")
    H.download()
    ds_all = H.xarray()
    ds = next(d for d in ds_all if "u" in d and "v" in d and "t" in d)
    ds = ds.where(
        (ds.latitude  >= LAT_MIN - PAD) & (ds.latitude  <= LAT_MAX + PAD) &
        (ds.longitude >= LON_MIN - PAD) & (ds.longitude <= LON_MAX + PAD),
        drop=True,
    )
    return H, ds


fig, axes = plt.subplots(
    2, 2, figsize=(16, 10),
    subplot_kw={"projection": EasyMap("50m").ax.projection},
)

for ax, member in zip(axes.flat, MEMBERS):
    try:
        H, ds = get_wind_ds(member)
    except Exception as e:
        print(f"member={member} failed: {e}")
        try:
            H2 = Herbie(DATE, model="rrfs", fxx=FXX, member=member, domain="conus")
            ds_all2 = H2.xarray()
            for i, d in enumerate(ds_all2):
                print(f"  [{i}] {list(d.data_vars)}")
        except Exception:
            pass
        continue

    u = ds.u.sel(isobaricInhPa=LEVEL).values
    v = ds.v.sel(isobaricInhPa=LEVEL).values
    spd = np.sqrt(u**2 + v**2)

    lons = ds.longitude.values
    lats = ds.latitude.values

    ax.add_feature(__import__("cartopy.feature", fromlist=["STATES"]).STATES, linewidth=0.5)
    p = ax.pcolormesh(lons, lats, spd, transform=pc, cmap="Blues", vmin=0, vmax=40)

    skip = 20
    ax.quiver(
        lons[::skip, ::skip], lats[::skip, ::skip],
        u[::skip, ::skip], v[::skip, ::skip],
        transform=pc, scale=500, width=0.002, color="k",
    )

    vt = ds.valid_time.dt.strftime("%H:%M UTC %d %b %Y").item()
    ax.set_title(f"RRFS member={member}  {LEVEL} hPa wind speed [m/s]\n{vt} F{FXX:02d}")
    plt.colorbar(p, ax=ax, orientation="horizontal", pad=0.04, label="m/s")

plt.tight_layout()
plt.savefig("rrfs_wind_ensemble.png", dpi=150, bbox_inches="tight")
print("Saved rrfs_wind_ensemble.png")
