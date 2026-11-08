"""Download the FourCastNet v0 weights and build the ERA5 input file. Run on the login node."""
import hashlib
import os
import urllib.request
from datetime import datetime, timedelta

import h5py
import numpy as np
import xarray as xr

ERA5_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
WEIGHT_FILES = ["backbone.ckpt", "stats_v0/global_means.npy", "stats_v0/global_stds.npy", "stats_v0/time_means.npy"]

# 20 input channels in the order the model expects
CHANNELS = [
    ("u10", "10m_u_component_of_wind", None),
    ("v10", "10m_v_component_of_wind", None),
    ("t2m", "2m_temperature", None),
    ("sp", "surface_pressure", None),
    ("msl", "mean_sea_level_pressure", None),
    ("t850", "temperature", 850),
    ("u1000", "u_component_of_wind", 1000),
    ("v1000", "v_component_of_wind", 1000),
    ("z1000", "geopotential", 1000),
    ("u850", "u_component_of_wind", 850),
    ("v850", "v_component_of_wind", 850),
    ("z850", "geopotential", 850),
    ("u500", "u_component_of_wind", 500),
    ("v500", "v_component_of_wind", 500),
    ("z500", "geopotential", 500),
    ("t500", "temperature", 500),
    ("z50", "geopotential", 50),
    ("r500", "relative_humidity", 500),
    ("r850", "relative_humidity", 850),
    ("tcwv", "total_column_water_vapour", None),
]


def download(url, dest):
    if os.path.exists(dest):
        return
    print("downloading", url)
    urllib.request.urlretrieve(url, dest + ".part")
    os.rename(dest + ".part", dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(ds, times):
    """Return ERA5 as an array of shape (time, 20, 721, 1440)."""
    out = np.empty((len(times), len(CHANNELS), 721, 1440), dtype=np.float32)
    for var in dict.fromkeys(v for _, v, _ in CHANNELS):
        chans = [(c, lvl) for c, (_, v, lvl) in enumerate(CHANNELS) if v == var]
        levels = [lvl for _, lvl in chans if lvl is not None]
        da = ds[var].sel(time=times)
        if levels:
            da = da.sel(level=levels)
        arr = da.values
        if var == "relative_humidity" and np.nanmax(arr) <= 2:  # model expects percent, not a fraction
            arr = arr * 100
        for i, (c, _) in enumerate(chans):
            out[:, c] = arr[:, i] if levels else arr
        print("  fetched", var)
    return out


def check(snapshot, means, stds):
    """Compare each channel's global mean to the model's training stats to catch wrong order or units."""
    for c, (name, _, _) in enumerate(CHANNELS):
        m = float(snapshot[c].mean())
        ok = abs(m - means[c]) < stds[c]
        print(f"  {name:6s} mean {m:12.2f}   training mean {means[c]:12.2f}   std {stds[c]:10.2f}   {'ok' if ok else 'MISMATCH'}")


def main():
    weights_dir = os.environ["WEIGHTS_DIR"]
    data_dir = os.environ["DATA_DIR"]
    init_times = [datetime.fromisoformat(t) for t in os.environ["INIT_TIMES"].split()]
    nsteps = int(os.environ["NSTEPS"])

    os.makedirs(os.path.join(weights_dir, "stats_v0"), exist_ok=True)
    for name in WEIGHT_FILES:
        download(f"{os.environ['FCN_WEIGHTS_URL']}/{name}", os.path.join(weights_dir, name))
    print("sha256:")
    for name in WEIGHT_FILES:
        print(f"  {sha256(os.path.join(weights_dir, name))}  {name}")

    means = np.load(os.path.join(weights_dir, "stats_v0/global_means.npy")).reshape(-1)[:20]
    stds = np.load(os.path.join(weights_dir, "stats_v0/global_stds.npy")).reshape(-1)[:20]

    # NVlabs inference reads one file per year, indexed in 6 h steps from Jan 1
    year = init_times[0].year
    n_steps_year = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days * 4
    out_path = os.path.join(data_dir, "out_of_sample", f"{year}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds = xr.open_zarr(ERA5_URL, storage_options={"token": "anon"}, chunks=None)
    ds = ds.sortby("latitude", ascending=False)  # model expects 90N first
    print("ERA5 source:", ERA5_URL)

    with h5py.File(out_path, "w") as f:  # recreate each run, so a file left by a killed run can't break this
        # Full-year shape, but only the forecast windows are written; the rest takes no disk space
        fields = f.create_dataset("fields", shape=(n_steps_year, 20, 721, 1440), dtype="f4", chunks=(1, 1, 721, 1440))
        for t0 in init_times:
            times = [t0 + timedelta(hours=6 * k) for k in range(nsteps + 1)]
            i0 = int((t0 - datetime(year, 1, 1)).total_seconds() // 21600)
            print(f"init {t0} -> rows {i0}..{i0 + nsteps}")
            data = fetch(ds, times)
            check(data[0], means, stds)
            fields[i0:i0 + len(times)] = data
    print("wrote", out_path)


if __name__ == "__main__":
    main()
